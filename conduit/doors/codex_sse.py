"""OpenAI Responses SSE emitter for the Codex door.

Translates Kiro typed events (conduit.contracts.events.KiroEvent /
UnknownEvent) into the OpenAI Responses SSE wire format required by Codex CLI.

HAZARD 1 — Reasoning blobs are NOT portable
    Kiro ``reasoningContentEvent`` payloads may include a ``signature`` field
    that is an opaque, provider-specific blob.  Codex ``encrypted_content``
    blobs are similarly opaque.  Forwarding either blob to the other backend
    would silently corrupt the reasoning chain.  This module NEVER forwards a
    Kiro signature in a Codex response.  Signature bytes are dropped here;
    only the raw thinking text is relayed.

HAZARD 2 — Tool status cannot come from Codex's wire
    ``FunctionCallOutput`` on Codex's wire carries no per-item success/error
    flag.  The tool status that Lane B's tool-result builder needs must come
    from CONDUIT's own tool runner.  This module does not default to
    "success".  It is the caller's responsibility to supply the status from
    the tool runner before handing results to Lane B.

Event catalogue
    response.created                 — lifecycle, emitted once at stream open
    response.in_progress             — lifecycle, emitted once after created
    response.output_item.added       — per output item (text block, tool call)
    response.output_item.done        — per output item, at close
    response.output_text.delta       — text delta inside a text output item
    response.output_text.done        — text output item finalised
    response.content_part.added      — content-part lifecycle (mirrors output_item)
    response.content_part.done       — content-part lifecycle done
    response.function_call_arguments.delta  — streaming tool-call arguments
    response.function_call_arguments.done   — tool-call arguments finalised
    response.reasoning_text.delta           — streaming reasoning text
    response.reasoning_summary_text.delta   — streaming reasoning summary delta
    response.reasoning_summary_text.done    — reasoning summary done
    response.completed               — terminal, carries usage (six-field map)
    response.failed                  — terminal, typed error code
    response.incomplete              — terminal, stream ended without completion

Token usage — six-field map (no estimation)
    Codex field                               TurnMetrics field
    ─────────────────────────────────────────────────────────────
    input_tokens                              input_tokens
    input_tokens_details.cached_tokens        cache_read_tokens
    input_tokens_details.cache_write_tokens   cache_write_tokens
    output_tokens                             output_tokens
    output_tokens_details.reasoning_tokens    thought_tokens
    total_tokens                              total_tokens
"""

from __future__ import annotations

import json
import time
import uuid
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

from loguru import logger

from conduit.contracts.events import EventName, KiroEvent, UnknownEvent, DispatchedEvent
from conduit.contracts.metrics import StopReason, TurnMetrics
from conduit.doors.codex_dialect import freeform_input_from_arguments
from conduit.kiro.usage import (
    apply_context_usage_event,
    apply_metadata_event,
    apply_metering_event,
)


# ---------------------------------------------------------------------------
# Typed error codes for response.failed
# ---------------------------------------------------------------------------

class ResponsesErrorCode(str, Enum):
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INSUFFICIENT_QUOTA = "insufficient_quota"
    INVALID_REQUEST_ERROR = "invalid_request_error"
    SERVER_ERROR = "server_error"


_STOP_REASON_TO_ERROR_CODE: Dict[StopReason, ResponsesErrorCode] = {
    StopReason.MODEL_CONTEXT_WINDOW_EXCEEDED: ResponsesErrorCode.CONTEXT_LENGTH_EXCEEDED,
    StopReason.MONTHLY_REQUEST_COUNT: ResponsesErrorCode.INSUFFICIENT_QUOTA,
    StopReason.DAILY_REQUEST_COUNT: ResponsesErrorCode.INSUFFICIENT_QUOTA,
    StopReason.INSUFFICIENT_MODEL_CAPACITY: ResponsesErrorCode.RATE_LIMIT_EXCEEDED,
    StopReason.CONTENT_FILTERED: ResponsesErrorCode.INVALID_REQUEST_ERROR,
}


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse(event_type: str, data: Dict[str, Any]) -> str:
    """Format one SSE frame: ``event: …\\ndata: …\\n\\n``."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def _item_id() -> str:
    return f"item_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Usage builder
# ---------------------------------------------------------------------------

def build_usage(metrics: TurnMetrics) -> Dict[str, Any]:
    """Construct the six-field usage map from TurnMetrics.

    Field mapping (no estimation — every number is parsed off the wire):

        input_tokens                       <- metrics.input_tokens
        input_tokens_details.cached_tokens <- metrics.cache_read_tokens
        input_tokens_details.cache_write_tokens <- metrics.cache_write_tokens
        output_tokens                      <- metrics.output_tokens
        output_tokens_details.reasoning_tokens  <- metrics.thought_tokens
        total_tokens                       <- metrics.total_tokens
    """
    return {
        "input_tokens": metrics.input_tokens,
        "input_tokens_details": {
            "cached_tokens": metrics.cache_read_tokens,
            "cache_write_tokens": metrics.cache_write_tokens,
        },
        "output_tokens": metrics.output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": metrics.thought_tokens,
        },
        "total_tokens": metrics.total_tokens,
    }


# ---------------------------------------------------------------------------
# Lifecycle frames
# ---------------------------------------------------------------------------

def frame_response_created(response_id: str, model: str) -> str:
    payload: Dict[str, Any] = {
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "realtime.response",
            "status": "in_progress",
            "model": model,
            "created_at": int(time.time()),
            "output": [],
        },
    }
    return _sse("response.created", payload)


def frame_response_in_progress(response_id: str, model: str) -> str:
    payload: Dict[str, Any] = {
        "type": "response.in_progress",
        "response": {
            "id": response_id,
            "object": "realtime.response",
            "status": "in_progress",
            "model": model,
        },
    }
    return _sse("response.in_progress", payload)


# ---------------------------------------------------------------------------
# Output-item lifecycle frames
# ---------------------------------------------------------------------------

def frame_output_item_added(
    response_id: str,
    item_id: str,
    item_type: str,
    index: int,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    item: Dict[str, Any] = {
        "id": item_id,
        "object": "realtime.item",
        "type": item_type,
        "status": "in_progress",
    }
    if extra:
        item.update(extra)
    payload: Dict[str, Any] = {
        "type": "response.output_item.added",
        "response_id": response_id,
        "output_index": index,
        "item": item,
    }
    return _sse("response.output_item.added", payload)


def frame_output_item_done(
    response_id: str,
    item_id: str,
    item_type: str,
    index: int,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    item: Dict[str, Any] = {
        "id": item_id,
        "object": "realtime.item",
        "type": item_type,
        "status": "completed",
    }
    if extra:
        item.update(extra)
    payload: Dict[str, Any] = {
        "type": "response.output_item.done",
        "response_id": response_id,
        "output_index": index,
        "item": item,
    }
    return _sse("response.output_item.done", payload)


# ---------------------------------------------------------------------------
# Text frames
# ---------------------------------------------------------------------------

def frame_content_part_added(
    response_id: str, item_id: str, output_index: int, part_index: int
) -> str:
    payload: Dict[str, Any] = {
        "type": "response.content_part.added",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": output_index,
        "content_index": part_index,
        "part": {"type": "text", "text": ""},
    }
    return _sse("response.content_part.added", payload)


def frame_output_text_delta(
    response_id: str,
    item_id: str,
    output_index: int,
    content_index: int,
    delta: str,
) -> str:
    payload: Dict[str, Any] = {
        "type": "response.output_text.delta",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "delta": delta,
    }
    return _sse("response.output_text.delta", payload)


def frame_output_text_done(
    response_id: str,
    item_id: str,
    output_index: int,
    content_index: int,
    text: str,
) -> str:
    payload: Dict[str, Any] = {
        "type": "response.output_text.done",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "text": text,
    }
    return _sse("response.output_text.done", payload)


def frame_content_part_done(
    response_id: str,
    item_id: str,
    output_index: int,
    content_index: int,
    text: str,
) -> str:
    payload: Dict[str, Any] = {
        "type": "response.content_part.done",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "part": {"type": "text", "text": text},
    }
    return _sse("response.content_part.done", payload)


# ---------------------------------------------------------------------------
# Reasoning frames
# ---------------------------------------------------------------------------

def frame_reasoning_text_delta(
    response_id: str,
    item_id: str,
    output_index: int,
    content_index: int,
    delta: str,
) -> str:
    """Emit a reasoning_text delta.

    HAZARD: the Kiro ``signature`` field on reasoningContentEvent is an
    opaque provider-specific blob and MUST NOT be forwarded.  Only the plain
    thinking text is relayed here.
    """
    payload: Dict[str, Any] = {
        "type": "response.reasoning_text.delta",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "delta": delta,
    }
    return _sse("response.reasoning_text.delta", payload)


def frame_reasoning_summary_text_delta(
    response_id: str,
    item_id: str,
    output_index: int,
    summary_index: int,
    delta: str,
) -> str:
    payload: Dict[str, Any] = {
        "type": "response.reasoning_summary_text.delta",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": output_index,
        "summary_index": summary_index,
        "delta": delta,
    }
    return _sse("response.reasoning_summary_text.delta", payload)


def frame_reasoning_summary_text_done(
    response_id: str,
    item_id: str,
    output_index: int,
    summary_index: int,
    text: str,
) -> str:
    payload: Dict[str, Any] = {
        "type": "response.reasoning_summary_text.done",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": output_index,
        "summary_index": summary_index,
        "text": text,
    }
    return _sse("response.reasoning_summary_text.done", payload)


# ---------------------------------------------------------------------------
# Tool-call frames
# ---------------------------------------------------------------------------

def frame_function_call_arguments_delta(
    response_id: str,
    item_id: str,
    output_index: int,
    call_id: str,
    delta: str,
) -> str:
    payload: Dict[str, Any] = {
        "type": "response.function_call_arguments.delta",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": output_index,
        "call_id": call_id,
        "delta": delta,
    }
    return _sse("response.function_call_arguments.delta", payload)


def frame_function_call_arguments_done(
    response_id: str,
    item_id: str,
    output_index: int,
    call_id: str,
    arguments: str,
) -> str:
    payload: Dict[str, Any] = {
        "type": "response.function_call_arguments.done",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": output_index,
        "call_id": call_id,
        "arguments": arguments,
    }
    return _sse("response.function_call_arguments.done", payload)


# ---------------------------------------------------------------------------
# Freeform ("custom") tool-call frames
#
# A freeform tool takes raw source text, not a JSON arguments object, and it
# travels on its own event names and its own output-item type. Codex's handler
# accepts a freeform tool's result only via ToolPayload::Custom, so framing one
# of these as function_call/arguments makes Codex discard the call as
# unsupported — and on the next request it back-fills the missing result with
# the literal string "aborted".
# ---------------------------------------------------------------------------

def frame_custom_tool_call_input_delta(
    response_id: str,
    item_id: str,
    output_index: int,
    call_id: str,
    delta: str,
) -> str:
    payload: Dict[str, Any] = {
        "type": "response.custom_tool_call_input.delta",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": output_index,
        "call_id": call_id,
        "delta": delta,
    }
    return _sse("response.custom_tool_call_input.delta", payload)


def frame_custom_tool_call_input_done(
    response_id: str,
    item_id: str,
    output_index: int,
    call_id: str,
    input_text: str,
) -> str:
    payload: Dict[str, Any] = {
        "type": "response.custom_tool_call_input.done",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": output_index,
        "call_id": call_id,
        "input": input_text,
    }
    return _sse("response.custom_tool_call_input.done", payload)


# ---------------------------------------------------------------------------
# Terminal frames
# ---------------------------------------------------------------------------

def frame_response_completed(
    response_id: str,
    model: str,
    metrics: TurnMetrics,
    output_items: List[Dict[str, Any]],
) -> str:
    """Emit response.completed with the six-field usage map.

    This frame signals the end of the stream.  Codex treats its receipt as
    the definitive end-of-turn signal.
    """
    usage = build_usage(metrics)
    payload: Dict[str, Any] = {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "object": "realtime.response",
            "status": "completed",
            "model": model,
            "created_at": int(time.time()),
            "output": output_items,
            "usage": usage,
        },
    }
    return _sse("response.completed", payload)


def frame_response_failed(
    response_id: str,
    model: str,
    code: ResponsesErrorCode,
    message: str,
) -> str:
    """Emit response.failed with a typed error code.

    Typed codes:
        context_length_exceeded   — MODEL_CONTEXT_WINDOW_EXCEEDED
        rate_limit_exceeded       — INSUFFICIENT_MODEL_CAPACITY
        insufficient_quota        — MONTHLY_ / DAILY_REQUEST_COUNT
        invalid_request_error     — CONTENT_FILTERED, bad input
        server_error              — fallback for unexpected states
    """
    payload: Dict[str, Any] = {
        "type": "response.failed",
        "response": {
            "id": response_id,
            "object": "realtime.response",
            "status": "failed",
            "model": model,
            "error": {
                "type": code.value,
                "code": code.value,
                "message": message,
            },
        },
    }
    return _sse("response.failed", payload)


def frame_response_incomplete(
    response_id: str,
    model: str,
    reason: str = "interrupted",
) -> str:
    payload: Dict[str, Any] = {
        "type": "response.incomplete",
        "response": {
            "id": response_id,
            "object": "realtime.response",
            "status": "incomplete",
            "model": model,
            "incomplete_details": {"reason": reason},
        },
    }
    return _sse("response.incomplete", payload)


# ---------------------------------------------------------------------------
# Stop-reason helpers
# ---------------------------------------------------------------------------

def stop_reason_to_error_code(reason: Optional[StopReason]) -> ResponsesErrorCode:
    """Map a Kiro StopReason to an OpenAI Responses error code."""
    if reason is None:
        return ResponsesErrorCode.SERVER_ERROR
    return _STOP_REASON_TO_ERROR_CODE.get(reason, ResponsesErrorCode.SERVER_ERROR)


def is_error_stop(reason: Optional[StopReason]) -> bool:
    """Return True when the stop reason represents a hard error."""
    return reason in _STOP_REASON_TO_ERROR_CODE


def is_incomplete_stop(reason: Optional[StopReason]) -> bool:
    """Return True when the stop reason means the turn is incomplete but not
    a hard error (e.g. PAUSE_TURN, MAX_TOKENS)."""
    return reason in (StopReason.PAUSE_TURN, StopReason.MAX_TOKENS)


# ---------------------------------------------------------------------------
# Main stream translator
# ---------------------------------------------------------------------------

async def translate_kiro_to_responses_sse(
    kiro_events: AsyncGenerator[DispatchedEvent, None],
    *,
    model: str,
    response_id: Optional[str] = None,
    metrics: Optional[TurnMetrics] = None,
    custom_tool_names: Optional[set] = None,
) -> AsyncGenerator[str, None]:
    """Translate a stream of Kiro dispatched events into Responses SSE frames.

    This is the core translator.  It maintains just enough state to emit
    properly-sequenced output-item and content-part lifecycle events.

    Signature / encrypted_content fields are silently dropped to honour
    HAZARD 1 (reasoning blobs are not portable).

    Args:
        kiro_events:  Async generator of KiroEvent / UnknownEvent objects
                      (produced by Lane A's parser).
        model:        The Kiro model id that was used for this turn.
        response_id:  Caller-supplied response id.  A fresh one is generated
                      when not provided.
        metrics:      Optional TurnMetrics to populate IN-PLACE, so the caller
                      can read real counts after iteration and hand them to the
                      Lane F seam. A fresh one is used when not supplied.
        custom_tool_names:
                      Names of tools the client declared as freeform
                      ("custom"), from adapt_freeform_tools(). Calls to these
                      are framed as ``custom_tool_call`` carrying raw source
                      text; Codex accepts a freeform tool's result only via
                      ToolPayload::Custom and discards a function_call framing.

    Yields:
        SSE-formatted strings, one per event frame.
    """
    if response_id is None:
        response_id = _response_id()

    # Per-stream state. Reuse the caller's TurnMetrics when given so the door
    # can read real counts after the stream ends (Lane C does the same).
    if metrics is None:
        metrics = TurnMetrics()
    output_index: int = 0        # current output item index
    text_item_id: Optional[str] = None
    text_item_open: bool = False
    accumulated_text: str = ""

    reasoning_item_id: Optional[str] = None
    reasoning_item_open: bool = False
    accumulated_reasoning: str = ""
    reasoning_output_index: int = -1

    tool_item_id: Optional[str] = None
    tool_call_id: Optional[str] = None
    accumulated_args: str = ""
    tool_output_index: int = -1
    _custom_names: set = custom_tool_names or set()

    completed_items: List[Dict[str, Any]] = []
    stream_terminated: bool = False

    # --- lifecycle open ---
    yield frame_response_created(response_id, model)
    yield frame_response_in_progress(response_id, model)

    try:
        async for event in kiro_events:
            if isinstance(event, UnknownEvent):
                logger.debug("codex_sse: unknown event %s, skipping", event.name)
                continue

            assert isinstance(event, KiroEvent)
            name = event.name
            payload = event.payload

            # ------------------------------------------------------------------
            # Text delta
            # ------------------------------------------------------------------
            if name == EventName.ASSISTANT_RESPONSE:
                content = payload.get("content", "")
                if not content:
                    continue

                # Open a text output item on first delta
                if not text_item_open:
                    text_item_id = _item_id()
                    yield frame_output_item_added(
                        response_id, text_item_id, "message", output_index
                    )
                    yield frame_content_part_added(
                        response_id, text_item_id, output_index, 0
                    )
                    text_item_open = True

                accumulated_text += content
                yield frame_output_text_delta(
                    response_id, text_item_id, output_index, 0, content
                )

            # ------------------------------------------------------------------
            # Reasoning / thinking delta
            # NOTE: drop ``signature`` — it is a provider-specific opaque blob
            # ------------------------------------------------------------------
            elif name == EventName.REASONING_CONTENT:
                # Drop the signature field unconditionally.
                text = payload.get("content", "")
                # Some implementations put reasoning text under "text" key
                if not text:
                    text = payload.get("text", "")
                if not text:
                    continue

                if not reasoning_item_open:
                    reasoning_item_id = _item_id()
                    reasoning_output_index = output_index
                    # Close any open text item first
                    if text_item_open:
                        _close_text_item(
                            completed_items,
                            text_item_id,
                            output_index,
                            accumulated_text,
                        )
                        output_index += 1
                        text_item_open = False
                        accumulated_text = ""

                    yield frame_output_item_added(
                        response_id, reasoning_item_id, "reasoning", output_index
                    )
                    reasoning_item_open = True
                    reasoning_output_index = output_index

                accumulated_reasoning += text
                yield frame_reasoning_text_delta(
                    response_id, reasoning_item_id, reasoning_output_index, 0, text
                )
                yield frame_reasoning_summary_text_delta(
                    response_id, reasoning_item_id, reasoning_output_index, 0, text
                )

            # ------------------------------------------------------------------
            # Tool use
            # ------------------------------------------------------------------
            elif name == EventName.TOOL_USE:
                tool_id = payload.get("toolUseId", payload.get("id", _item_id()))
                tool_name = payload.get("name", "")
                tool_input = payload.get("input", "")
                if isinstance(tool_input, dict):
                    tool_input = json.dumps(tool_input, ensure_ascii=False)

                # Close any open text item
                if text_item_open:
                    _close_text_item(
                        completed_items,
                        text_item_id,
                        output_index,
                        accumulated_text,
                    )
                    output_index += 1
                    text_item_open = False
                    accumulated_text = ""

                # Close any open reasoning item
                if reasoning_item_open:
                    _close_reasoning_item(
                        completed_items,
                        reasoning_item_id,
                        reasoning_output_index,
                        accumulated_reasoning,
                    )
                    output_index = reasoning_output_index + 1
                    reasoning_item_open = False
                    accumulated_reasoning = ""

                tool_item_id = _item_id()
                tool_call_id = tool_id
                tool_output_index = output_index
                accumulated_args = tool_input

                if tool_name in _custom_names:
                    # Freeform tool. Kiro replies with a JSON arguments object
                    # because that is how the tool had to be declared to it, so
                    # the raw source is recovered before framing. Everything
                    # here — item type and event names — is the custom variant;
                    # Codex's handler matches only ToolPayload::Custom.
                    source = freeform_input_from_arguments(tool_input)
                    logger.debug(
                        "codex_sse: freeform tool {} framed as custom_tool_call, "
                        "source_len={}",
                        tool_name,
                        len(source),
                    )
                    yield frame_output_item_added(
                        response_id,
                        tool_item_id,
                        "custom_tool_call",
                        tool_output_index,
                        {"name": tool_name, "call_id": tool_call_id},
                    )
                    yield frame_custom_tool_call_input_delta(
                        response_id,
                        tool_item_id,
                        tool_output_index,
                        tool_call_id,
                        source,
                    )
                    yield frame_custom_tool_call_input_done(
                        response_id,
                        tool_item_id,
                        tool_output_index,
                        tool_call_id,
                        source,
                    )
                    completed_items.append(
                        {
                            "id": tool_item_id,
                            "type": "custom_tool_call",
                            "call_id": tool_call_id,
                            "name": tool_name,
                            "input": source,
                        }
                    )
                    yield frame_output_item_done(
                        response_id,
                        tool_item_id,
                        "custom_tool_call",
                        tool_output_index,
                        {
                            "name": tool_name,
                            "call_id": tool_call_id,
                            "input": source,
                        },
                    )
                    output_index += 1
                else:
                    yield frame_output_item_added(
                        response_id,
                        tool_item_id,
                        "function_call",
                        tool_output_index,
                        {"name": tool_name, "call_id": tool_call_id},
                    )
                    yield frame_function_call_arguments_delta(
                        response_id,
                        tool_item_id,
                        tool_output_index,
                        tool_call_id,
                        tool_input,
                    )
                    yield frame_function_call_arguments_done(
                        response_id,
                        tool_item_id,
                        tool_output_index,
                        tool_call_id,
                        tool_input,
                    )
                    completed_items.append(
                        {
                            "id": tool_item_id,
                            "type": "function_call",
                            "call_id": tool_call_id,
                            "name": tool_name,
                            "arguments": tool_input,
                        }
                    )
                    yield frame_output_item_done(
                        response_id,
                        tool_item_id,
                        "function_call",
                        tool_output_index,
                        {
                            "name": tool_name,
                            "call_id": tool_call_id,
                            "arguments": tool_input,
                        },
                    )
                    output_index += 1

            # ------------------------------------------------------------------
            # Usage events — parsed by Lane A, never hand-rolled here.
            #
            # Lane A owns every wire field name. Hand-rolling them here is what
            # made this translator report wrong numbers while exact counts were
            # on the wire:
            #   - it read `thinkingTokens`, which Kiro never sends (the field is
            #     `thoughtTokens`), so reasoning_tokens was always 0;
            #   - it SUMMED total_tokens instead of reading `totalTokens`, and
            #     the sum omitted thought tokens entirely;
            #   - it took cache counts off contextUsageEvent, but they arrive in
            #     metadataEvent.tokenUsage (both spellings), so contextUsage's
            #     real payload — contextUsagePercentage — was never read;
            #   - it dropped meteringEvent credits on the floor.
            # ------------------------------------------------------------------
            elif name == EventName.METERING:
                apply_metering_event(metrics, payload)

            elif name in (EventName.METADATA, EventName.MESSAGE_METADATA):
                # Both wire spellings carry tokenUsage; see the Anthropic door.
                apply_metadata_event(metrics, payload)

            elif name == EventName.CONTEXT_USAGE:
                apply_context_usage_event(metrics, payload)

            # ------------------------------------------------------------------
            # Message stop — primary terminal event
            # ------------------------------------------------------------------
            elif name == EventName.MESSAGE_STOP:
                stop_reason_raw = payload.get("stopReason")
                try:
                    stop_reason = StopReason(stop_reason_raw) if stop_reason_raw else None
                except ValueError:
                    logger.debug(
                        "codex_sse: unrecognised stopReason %r", stop_reason_raw
                    )
                    stop_reason = None

                metrics.stop_reason = stop_reason

                # Close open text item
                if text_item_open:
                    _close_text_item(
                        completed_items, text_item_id, output_index, accumulated_text
                    )
                    output_index += 1
                    text_item_open = False

                # Close open reasoning item
                if reasoning_item_open:
                    yield frame_reasoning_summary_text_done(
                        response_id,
                        reasoning_item_id,
                        reasoning_output_index,
                        0,
                        accumulated_reasoning,
                    )
                    yield frame_output_item_done(
                        response_id,
                        reasoning_item_id,
                        "reasoning",
                        reasoning_output_index,
                        {"summary": [{"type": "text", "text": accumulated_reasoning}]},
                    )
                    completed_items.append(
                        {
                            "id": reasoning_item_id,
                            "type": "reasoning",
                            "summary": [{"type": "summary_text", "text": accumulated_reasoning}],
                        }
                    )
                    reasoning_item_open = False

                # Emit terminal event
                if is_error_stop(stop_reason):
                    code = stop_reason_to_error_code(stop_reason)
                    yield frame_response_failed(
                        response_id,
                        model,
                        code,
                        f"Turn ended with stop reason: {stop_reason_raw}",
                    )
                elif is_incomplete_stop(stop_reason):
                    yield frame_response_incomplete(response_id, model, stop_reason_raw or "interrupted")
                else:
                    yield frame_response_completed(
                        response_id, model, metrics, completed_items
                    )

                stream_terminated = True

            # ------------------------------------------------------------------
            # Invalid state from Kiro
            # ------------------------------------------------------------------
            elif name == EventName.INVALID_STATE:
                message = payload.get("message", "Kiro returned an invalid state")
                yield frame_response_failed(
                    response_id,
                    model,
                    ResponsesErrorCode.SERVER_ERROR,
                    message,
                )
                stream_terminated = True
                return

    except Exception as exc:
        logger.error("codex_sse: stream translation error: %s", exc, exc_info=True)
        if not stream_terminated:
            yield frame_response_failed(
                response_id,
                model,
                ResponsesErrorCode.SERVER_ERROR,
                f"Internal stream error: {type(exc).__name__}",
            )
        return

    # Guard: if the stream ended without a terminal event, emit incomplete
    if not stream_terminated:
        logger.warning(
            "codex_sse: stream ended without messageStopEvent, emitting incomplete"
        )
        # Close any dangling items
        if text_item_open:
            _close_text_item(
                completed_items, text_item_id, output_index, accumulated_text
            )
        yield frame_response_incomplete(response_id, model, "stream_ended_without_stop")


# ---------------------------------------------------------------------------
# Internal helpers — close open items and accumulate to completed list
# ---------------------------------------------------------------------------

def _close_text_item(
    completed_items: List[Dict[str, Any]],
    item_id: Optional[str],
    output_index: int,
    text: str,
) -> None:
    """Emit the done frames for an open text output item (side-effect free
    version — callers yield the return values themselves via the generator).

    Because generators can't call ``yield`` inside a regular function, this
    helper only mutates ``completed_items``.  The caller is responsible for
    yielding the frames it wants before calling this.
    """
    if item_id is None:
        return
    completed_items.append(
        {
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }
    )


def _close_reasoning_item(
    completed_items: List[Dict[str, Any]],
    item_id: Optional[str],
    output_index: int,
    text: str,
) -> None:
    if item_id is None:
        return
    completed_items.append(
        {
            "id": item_id,
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": text}],
        }
    )
