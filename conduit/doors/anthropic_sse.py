"""Anthropic SSE emitter — Lane C.

Translates a stream of Lane A typed events (KiroEvent / UnknownEvent) into
Anthropic Messages API SSE wire format.

Event sequence emitted per turn:
  message_start
  [thinking: content_block_start / content_block_delta+ / signature_delta / content_block_stop]
  [text:     content_block_start / content_block_delta+ / content_block_stop]
  [tool_use: content_block_start / content_block_delta  / content_block_stop]  (one per tool)
  message_delta   (stop_reason, usage from TurnMetrics)
  message_stop

Thinking signatures: reasoningContentEvent carries a real signature from Kiro.
It is forwarded verbatim as a signature_delta. Nothing is generated here.
Upstream's generate_thinking_signature() is explicitly retired by this module.

Usage: all counters come from TurnMetrics (populated by metadataEvent in Lane A).
  metrics.cache_read_tokens  → cache_read_input_tokens   (Anthropic field name)
  metrics.cache_write_tokens → cache_creation_input_tokens
"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncGenerator, AsyncIterable, Dict, List, Optional

from loguru import logger

from conduit.contracts.events import DispatchedEvent, EventName, KiroEvent, UnknownEvent
from conduit.contracts.metrics import ReasoningBlockMeta, StopReason, TurnMetrics
from conduit.kiro.derive import derive_token_accounting
from conduit.kiro.usage import (
    apply_context_usage_event,
    apply_metadata_event,
    apply_metering_event,
)


# ---------------------------------------------------------------------------
# SSE wire helpers
# ---------------------------------------------------------------------------

def _sse(event_type: str, data: Dict[str, Any]) -> str:
    """Format one SSE frame: ``event: …\\ndata: …\\n\\n``."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _gen_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# Stop-reason translation
# ---------------------------------------------------------------------------

_STOP_REASON_MAP: Dict[StopReason, str] = {
    StopReason.END_TURN:                       "end_turn",
    StopReason.TOOL_USE:                       "tool_use",
    StopReason.MAX_TOKENS:                     "max_tokens",
    StopReason.MODEL_CONTEXT_WINDOW_EXCEEDED:  "max_tokens",
    StopReason.CONTENT_FILTERED:               "end_turn",
    StopReason.PAUSE_TURN:                     "end_turn",
    StopReason.MONTHLY_REQUEST_COUNT:          "end_turn",
    StopReason.DAILY_REQUEST_COUNT:            "end_turn",
    StopReason.INSUFFICIENT_MODEL_CAPACITY:    "end_turn",
}


def _map_stop_reason(kiro_reason: Optional[str]) -> str:
    if kiro_reason is None:
        return "end_turn"
    try:
        sr = StopReason(kiro_reason)
        return _STOP_REASON_MAP.get(sr, "end_turn")
    except ValueError:
        logger.debug(f"Unknown Kiro stopReason: {kiro_reason!r}, defaulting to end_turn")
        return "end_turn"


# ---------------------------------------------------------------------------
# Usage helper
# ---------------------------------------------------------------------------

def _saw_token_accounting(metrics: TurnMetrics) -> bool:
    """True when a metadataEvent actually delivered token counts.

    Distinguishes *measured* from *missing*. Every counter reading zero means no
    metadataEvent (under either wire spelling) carried a usable ``tokenUsage``
    block -- a real turn that produced output cannot legitimately report zero
    output tokens. A genuine zero from the wire would still pass this check via
    its siblings, so a measured count is never suppressed.
    """
    return any((
        metrics.input_tokens,
        metrics.output_tokens,
        metrics.thought_tokens,
        metrics.cache_read_tokens,
        metrics.cache_write_tokens,
        metrics.total_tokens,
    ))


def _build_message_delta_usage(metrics: TurnMetrics) -> Dict[str, Any]:
    """Build the usage dict for the message_delta event from real TurnMetrics.

    **Absent, not zero.** When no tokenUsage arrived, the token fields are left
    out rather than sent as ``0``. A zero is a claim -- "this turn consumed no
    input" -- and reporting one for an unmeasured turn is the same defect as the
    upstream gateway back-calculating tokens from a percentage: a fabricated
    number wearing the costume of accounting. An absent field says "unknown",
    which is what CONDUIT actually knows. Nothing downstream may infer a count
    from silence.

    ``input_tokens`` rides in message_delta rather than message_start because
    Kiro only reveals it near end-of-stream; at message_start time no honest
    number exists yet, and buffering the whole turn to get one would cost the
    client its first byte.

    ``output_tokens`` stays unconditional: it is structurally expected in
    message_delta, and dropping it risks a client-side schema error. When
    accounting is missing it is therefore still emitted as 0 -- the one
    fabricated figure left in this path, kept only because the wire format
    demands the key. The zero-token warning at end-of-turn is what surfaces it.
    """
    usage: Dict[str, Any] = {"output_tokens": metrics.output_tokens}
    if _saw_token_accounting(metrics):
        usage["input_tokens"] = metrics.input_tokens
        usage["cache_read_input_tokens"] = metrics.cache_read_tokens
        usage["cache_creation_input_tokens"] = metrics.cache_write_tokens
    return usage


# ---------------------------------------------------------------------------
# Core emitter
# ---------------------------------------------------------------------------

async def stream_to_anthropic_sse(
    events: AsyncIterable[DispatchedEvent],
    model: str,
    message_id: str,
    metrics: TurnMetrics,
    start_index: int = 0,
    emit_message_start: bool = True,
) -> AsyncGenerator[str, None]:
    """Translate a Lane A event stream into Anthropic SSE wire format.

    Args:
        events:     Async iterable of DispatchedEvent (KiroEvent | UnknownEvent).
        model:      Resolved model string (echoed in message_start).
        message_id: Stable message ID for this turn (caller supplies).
        metrics:    TurnMetrics instance — mutated in-place as metadataEvent
                    and meteringEvent arrive; caller reads it after iteration.
        start_index: First content-block index to use. Non-zero when the caller
                    has already emitted blocks into this message (server-side
                    web search emits two before the model speaks), since
                    Anthropic block indices must not collide within one message.
        emit_message_start: Whether to emit the ``message_start`` frame. False
                    for a continuation stream: exactly one ``message_start`` is
                    allowed per message, and the caller already sent it.

    Yields:
        SSE-formatted strings ready to write to the response stream.
    """
    block_index = start_index
    thinking_open = False
    thinking_idx: Optional[int] = None
    text_open = False
    text_idx: Optional[int] = None
    tool_count = 0
    # One tool call arrives as MANY toolUseEvent frames -- see _flush_tool.
    tool_open = False
    tool_idx: Optional[int] = None
    tool_id: Optional[str] = None
    tool_json: List[str] = []
    # Characters CONDUIT streamed out this turn -- assistant text plus tool-call
    # JSON. The only handle on output size, since Kiro reports no output count
    # and the context percentage folds input and output into one figure.
    output_chars = 0

    # Emit message_start immediately with zeroed usage.
    # Accurate usage is in message_delta (populated from metadataEvent which
    # arrives near end-of-stream).
    if emit_message_start:
        yield _sse("message_start", {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })

        # Ping to keep the connection alive while Kiro thinks.
        yield _sse("ping", {"type": "ping"})

    # --- helpers defined inline so they can access and mutate block_index ---

    async def _open_thinking() -> AsyncGenerator[str, None]:
        nonlocal thinking_open, thinking_idx, block_index
        thinking_idx = block_index
        block_index += 1
        thinking_open = True
        yield _sse("content_block_start", {
            "type": "content_block_start",
            "index": thinking_idx,
            "content_block": {"type": "thinking", "thinking": ""},
        })

    async def _close_thinking() -> AsyncGenerator[str, None]:
        nonlocal thinking_open
        if not thinking_open:
            return
        thinking_open = False
        yield _sse("content_block_stop", {
            "type": "content_block_stop",
            "index": thinking_idx,
        })

    async def _open_text() -> AsyncGenerator[str, None]:
        nonlocal text_open, text_idx, block_index
        text_idx = block_index
        block_index += 1
        text_open = True
        yield _sse("content_block_start", {
            "type": "content_block_start",
            "index": text_idx,
            "content_block": {"type": "text", "text": ""},
        })

    async def _close_text() -> AsyncGenerator[str, None]:
        nonlocal text_open
        if not text_open:
            return
        text_open = False
        yield _sse("content_block_stop", {
            "type": "content_block_stop",
            "index": text_idx,
        })

    async def _flush_tool() -> AsyncGenerator[str, None]:
        """Close the buffered tool call as ONE tool_use block.

        Kiro streams a single tool call as many toolUseEvent frames sharing one
        toolUseId: the first carries `name` with an empty `input`, the middle
        ones carry `input` as raw JSON *string fragments*, and the last sets
        `stop` (PLATFORM-FINDINGS 5a-ter: toolUseEvent = name/input/stop).
        Emitting a block per frame turned one call into N sibling tool_use
        blocks whose first `input` was `{}`, which is exactly what made every
        Bash call fail with "required parameter 'command' is missing".

        Fragments are joined and sent as one input_json_delta rather than
        forwarded one at a time: the reassembled value is then unambiguous for
        both the client and the seam, at the cost of streaming granularity no
        client renders anyway.
        """
        nonlocal tool_open, tool_idx, tool_id, tool_json
        if not tool_open:
            return
        nonlocal output_chars
        partial = "".join(tool_json).strip() or "{}"
        output_chars += len(partial)
        # Validated, not repaired: a fragment run that does not reassemble into
        # JSON is a wire or parse problem worth seeing, not something to paper
        # over with a fabricated input.
        try:
            json.loads(partial)
        except json.JSONDecodeError:
            logger.warning(
                f"[anthropic_sse] tool {tool_id!r} input did not reassemble into "
                f"valid JSON from {len(tool_json)} fragment(s); forwarding as-is"
            )
        yield _sse("content_block_delta", {
            "type": "content_block_delta",
            "index": tool_idx,
            "delta": {"type": "input_json_delta", "partial_json": partial},
        })
        yield _sse("content_block_stop", {
            "type": "content_block_stop",
            "index": tool_idx,
        })
        tool_open = False
        tool_idx = None
        tool_id = None
        tool_json = []

    try:
        async for ev in events:
            if isinstance(ev, UnknownEvent):
                logger.debug(f"[anthropic_sse] skipping unknown event: {ev.name!r}")
                continue

            assert isinstance(ev, KiroEvent)
            name = ev.name
            payload = ev.payload

            # ------------------------------------------------------------------
            # assistantResponseEvent → text delta
            # ------------------------------------------------------------------
            if name == EventName.ASSISTANT_RESPONSE:
                text = payload.get("content", "")
                if not text:
                    continue

                # Close thinking block if still open
                if thinking_open:
                    async for chunk in _close_thinking():
                        yield chunk

                # Open text block on first text chunk
                if not text_open:
                    async for chunk in _open_text():
                        yield chunk

                output_chars += len(text)
                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": text_idx,
                    "delta": {"type": "text_delta", "text": text},
                })

            # ------------------------------------------------------------------
            # reasoningContentEvent → thinking delta + verbatim signature
            # ------------------------------------------------------------------
            elif name == EventName.REASONING_CONTENT:
                thinking_text = payload.get("content", "")
                # Some Kiro revisions carry the reasoning text under "text"
                # instead of "content". The codex door already tolerates both
                # (codex_sse.py, same event); without the fallback here a turn
                # whose payload uses "text" opened a thinking block and streamed
                # nothing into it. "content" stays authoritative — every fixture
                # and Lane A's own parser use it.
                if not thinking_text:
                    thinking_text = payload.get("text", "")
                signature = payload.get("signature")  # may be None

                # Open thinking block if first chunk
                if not thinking_open:
                    async for chunk in _open_thinking():
                        yield chunk
                    # One ReasoningBlockMeta per BLOCK, appended as the block
                    # opens. reasoningContentEvent arrives as MANY deltas per
                    # block, so appending per delta would inflate the count the
                    # seam reports; appending only when a signature arrives
                    # would make an unsigned block invisible to the seam
                    # entirely. Lane A's apply_reasoning_content_event() is
                    # per-event by design and is deliberately not used here —
                    # block boundaries are known only at this layer.
                    metrics.reasoning_blocks.append(ReasoningBlockMeta())

                if thinking_text:
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": thinking_idx,
                        "delta": {"type": "thinking_delta", "thinking": thinking_text},
                    })

                # Enrich the open block's metadata as deltas carry it.
                if metrics.reasoning_blocks:
                    _blk = metrics.reasoning_blocks[-1]
                    _model_id = payload.get("modelId") or payload.get("model_id")
                    if _model_id and not _blk.model_id:
                        _blk.model_id = str(_model_id)
                    if signature:
                        _blk.signature_present = True

                # Forward real signature verbatim — never fabricate
                if signature:
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": thinking_idx,
                        "delta": {"type": "signature_delta", "signature": signature},
                    })

            # ------------------------------------------------------------------
            # toolUseEvent → complete tool_use block
            # ------------------------------------------------------------------
            elif name == EventName.TOOL_USE:
                # Close any open thinking/text blocks before tool block
                if thinking_open:
                    async for chunk in _close_thinking():
                        yield chunk
                if text_open:
                    async for chunk in _close_text():
                        yield chunk

                ev_tool_id = payload.get("toolUseId")
                ev_tool_name = payload.get("name") or ""
                ev_tool_input = payload.get("input")
                ev_stop = bool(payload.get("stop"))

                # A new toolUseId means the previous call ended without its
                # `stop` frame; close it rather than interleaving two calls.
                if tool_open and ev_tool_id and ev_tool_id != tool_id:
                    async for chunk in _flush_tool():
                        yield chunk

                if not tool_open:
                    if thinking_open:
                        async for chunk in _close_thinking():
                            yield chunk
                    if text_open:
                        async for chunk in _close_text():
                            yield chunk

                    tool_id = ev_tool_id or f"toolu_{uuid.uuid4().hex[:24]}"
                    tool_idx = block_index
                    block_index += 1
                    tool_count += 1
                    tool_json = []
                    tool_open = True

                    yield _sse("content_block_start", {
                        "type": "content_block_start",
                        "index": tool_idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": ev_tool_name,
                            "input": {},
                        },
                    })

                # `input` is either a raw JSON string fragment (the streaming
                # case) or a complete dict (single-frame case). Calling
                # json.dumps() on a fragment is what re-encoded it into a
                # quoted JSON string on the wire, escaped quotes and all, so
                # string fragments are appended verbatim. The empty dict the
                # opening and closing frames carry contributes nothing.
                if isinstance(ev_tool_input, str):
                    if ev_tool_input:
                        tool_json.append(ev_tool_input)
                elif isinstance(ev_tool_input, dict):
                    if ev_tool_input:
                        tool_json.append(json.dumps(ev_tool_input, ensure_ascii=False))
                elif ev_tool_input is not None:
                    tool_json.append(json.dumps(ev_tool_input, ensure_ascii=False))

                if ev_stop:
                    async for chunk in _flush_tool():
                        yield chunk

            # ------------------------------------------------------------------
            # metadataEvent → populate TurnMetrics counters
            # ------------------------------------------------------------------
            # Lane A owns every wire field name. Hand-rolling the mapping here
            # is what made this path report zeros while exact counts were on
            # the wire: it read cacheCreationInputTokens (an Anthropic name,
            # not Kiro's), summed total_tokens instead of reading totalTokens,
            # and never read thoughtTokens at all. Lane A also handles BOTH
            # cache-field spellings the wire may use.
            elif name in (EventName.METADATA, EventName.MESSAGE_METADATA):
                # messageMetadataEvent is a second wire spelling that can carry
                # the same tokenUsage block (PLATFORM-FINDINGS 5a-ter lists both
                # MetadataEvent and MessageMetadataEvent). It was registered in
                # dispatch but handled by no emitter, so it fell through to the
                # skip branch and every counter it carried was lost. Routing it
                # to the same parser is safe: apply_metadata_event only assigns
                # fields the payload actually contains.
                apply_metadata_event(metrics, payload)
                logger.debug(
                    f"[anthropic_sse] metadataEvent: in={metrics.input_tokens} "
                    f"out={metrics.output_tokens} thought={metrics.thought_tokens} "
                    f"cr={metrics.cache_read_tokens} cw={metrics.cache_write_tokens} "
                    f"total={metrics.total_tokens}"
                )

            # ------------------------------------------------------------------
            # meteringEvent → credits
            # ------------------------------------------------------------------
            elif name == EventName.METERING:
                apply_metering_event(metrics, payload)

            # ------------------------------------------------------------------
            # contextUsageEvent → context_usage_percentage
            # ------------------------------------------------------------------
            elif name == EventName.CONTEXT_USAGE:
                apply_context_usage_event(metrics, payload)

            # ------------------------------------------------------------------
            # messageStopEvent → stop reason
            # ------------------------------------------------------------------
            elif name == EventName.MESSAGE_STOP:
                kiro_reason = payload.get("stopReason")
                try:
                    metrics.stop_reason = StopReason(kiro_reason) if kiro_reason else None
                except ValueError:
                    logger.debug(f"[anthropic_sse] unrecognised stopReason: {kiro_reason!r}")
                    metrics.stop_reason = None

            # Everything else (intentsEvent, codeEvent, etc.) → debug-log, skip
            else:
                logger.debug(f"[anthropic_sse] skipping event: {name!r}")

    except GeneratorExit:
        logger.debug("[anthropic_sse] client disconnected (GeneratorExit)")
        raise
    except Exception as exc:
        logger.error(f"[anthropic_sse] error during translation: {exc}", exc_info=True)
        yield _sse("error", {
            "type": "error",
            "error": {"type": "api_error", "message": str(exc)},
        })
        raise

    # -----------------------------------------------------------------------
    # Close any still-open blocks
    # -----------------------------------------------------------------------
    # A stream that ends without the final `stop` frame still owes the client a
    # complete tool block; without this the call would never be dispatched.
    if tool_open:
        async for chunk in _flush_tool():
            yield chunk
    if thinking_open:
        async for chunk in _close_thinking():
            yield chunk
    if text_open:
        async for chunk in _close_text():
            yield chunk

    # -----------------------------------------------------------------------
    # Derive token accounting when upstream reported none
    # -----------------------------------------------------------------------
    # Runs after the stream closes: contextUsagePercentage is sampled post-
    # generation, so this is the first point the whole-turn figure exists.
    derive_token_accounting(metrics, model, output_chars)

    # -----------------------------------------------------------------------
    # Determine stop reason for Anthropic wire format
    # -----------------------------------------------------------------------
    if tool_count > 0 and metrics.stop_reason not in (StopReason.TOOL_USE, None):
        # Tool was emitted — override to tool_use unless metrics disagree
        anthropic_stop = "tool_use"
    elif metrics.stop_reason is not None:
        anthropic_stop = _map_stop_reason(metrics.stop_reason.value)
    else:
        anthropic_stop = "tool_use" if tool_count > 0 else "end_turn"

    # -----------------------------------------------------------------------
    # message_delta — all real usage lands here
    # -----------------------------------------------------------------------
    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {
            "stop_reason": anthropic_stop,
            "stop_sequence": None,
        },
        "usage": _build_message_delta_usage(metrics),
    })

    # -----------------------------------------------------------------------
    # message_stop
    # -----------------------------------------------------------------------
    yield _sse("message_stop", {"type": "message_stop"})

    logger.debug(
        f"[anthropic_sse] turn complete: stop={anthropic_stop} "
        f"in={metrics.input_tokens} out={metrics.output_tokens} "
        f"tools={tool_count}"
    )

    # A completed turn with no token counts means no metadataEvent (under either
    # spelling) carried a usable tokenUsage block. Credits and context usage can
    # still be present, which is what made this look healthy for 110 requests
    # while every token field sat at zero. Warn rather than shrug: the whole
    # point of this door is real accounting, and silence read as success.
    if metrics.token_accounting == "none":
        logger.warning(
            "[anthropic_sse] turn completed with NO token accounting "
            f"(credits={metrics.credits} ctx={metrics.context_usage_percentage}%) "
            "- no tokenUsage on the wire and no contextUsagePercentage to derive from"
        )
