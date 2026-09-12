"""Anthropic door — POST /v1/messages, for Claude Code.

OWNER: Lane C. Registered by conduit/app.py; do not edit app.py to wire it.

Lane C implements:
  - POST /v1/messages, auth via PROXY_API_KEY, transport via KiroHttpClient
    and AccountManager (both in conduit.foundation).
  - POST /v1/messages/count_tokens, lightweight token-count stub.
  - Anthropic SSE out: message_start, content_block_start/delta/stop,
    message_delta, message_stop.
  - reasoningContentEvent -> thinking_delta carrying the REAL upstream
    signature. Upstream scraped <thinking> tags out of plain text and
    fabricated the signature; both are retired.
  - usage block populated from Lane A's TurnMetrics — measured, not estimated.
  - On turn completion, hand TurnMetrics + RewrittenOut to Lane F's emit.
    Call it and move on; never await it.

Lane B interface assumed (stub):
  conduit.kiro.request.build_kiro_payload(
      request_data: AnthropicMessagesRequest,
      conversation_id: str,
      profile_arn: Optional[str],
  ) -> dict

  When Lane B is not yet available, a minimal stub is used that forwards
  the message content verbatim inside a Kiro conversationState envelope.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from loguru import logger

from conduit.config import (
    PROFILE_ARN,
    PROXY_API_KEY,
)
from conduit.contracts.events import DispatchedEvent, EventName, KiroEvent, UnknownEvent
from conduit.contracts.metrics import ReasoningBlockMeta, TurnMetrics
from conduit.contracts.rewritten import RewrittenOut
from conduit.contracts.seam import SeamEnvelope
from conduit.doors.anthropic_sse import stream_to_anthropic_sse
from conduit.kiro.dispatch import iter_events
from conduit.kiro.history import HistoryMessage
from conduit.kiro.request import build_conversation_state
from conduit.kiro.tools import build_tool_results_from_anthropic
from conduit.kiro.usage import (
    apply_context_usage_event,
    apply_metadata_event,
    apply_metering_event,
)
from conduit.seam.emit import emit
from conduit.foundation.auth import resolve_profile_arn
from conduit.foundation.http_client import KiroHttpClient


# ---------------------------------------------------------------------------
# Pydantic models for the Anthropic wire format
# (minimal — only what Claude Code sends; forward-compatible via extra="allow")
# ---------------------------------------------------------------------------

from typing import Literal, Union
from pydantic import BaseModel, Field


class _TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str
    model_config = {"extra": "allow"}


class _ThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str = ""
    model_config = {"extra": "allow"}


class _ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any] = Field(default_factory=dict)
    model_config = {"extra": "allow"}


class _ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Optional[Any] = None
    is_error: Optional[bool] = None
    model_config = {"extra": "allow"}


class _ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    source: Dict[str, Any] = Field(default_factory=dict)
    model_config = {"extra": "allow"}


_ContentBlock = Union[
    _TextBlock, _ThinkingBlock, _ToolUseBlock, _ToolResultBlock, _ImageBlock
]


class _Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[Any]]
    model_config = {"extra": "allow"}


class _SystemBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str
    model_config = {"extra": "allow"}


class _Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    type: Optional[str] = None
    model_config = {"extra": "allow"}


class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: List[_Message] = Field(min_length=1)
    max_tokens: int = 4096
    system: Optional[Union[str, List[Any]]] = None
    stream: bool = False
    thinking: Optional[Dict[str, Any]] = None
    tools: Optional[List[_Tool]] = None
    tool_choice: Optional[Any] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    model_config = {"extra": "allow"}


class _CountTokensRequest(BaseModel):
    model: str
    messages: List[_Message] = Field(min_length=1)
    system: Optional[Union[str, List[Any]]] = None
    tools: Optional[List[_Tool]] = None
    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_x_api_key_hdr = APIKeyHeader(name="x-api-key", auto_error=False)
_auth_hdr       = APIKeyHeader(name="Authorization", auto_error=False)


async def _verify_key(
    x_api_key: Optional[str] = Security(_x_api_key_hdr),
    authorization: Optional[str] = Security(_auth_hdr),
) -> bool:
    if x_api_key and x_api_key == PROXY_API_KEY:
        return True
    if authorization and authorization == f"Bearer {PROXY_API_KEY}":
        return True
    raise HTTPException(
        status_code=401,
        detail={
            "type": "error",
            "error": {
                "type": "authentication_error",
                "message": "Invalid or missing API key. Use x-api-key or Authorization: Bearer.",
            },
        },
    )


# ---------------------------------------------------------------------------
# Lane B adapter — Anthropic wire format -> Lane B's HistoryMessage pipeline
# ---------------------------------------------------------------------------

def _extract_system_prompt(system: Optional[Union[str, List[Any]]]) -> Optional[str]:
    """Flatten Anthropic's `system` field (string or block list) to one string."""
    if system is None:
        return None
    if isinstance(system, str):
        return system or None
    parts: List[str] = []
    for blk in system:
        if isinstance(blk, dict):
            if blk.get("type", "text") == "text":
                parts.append(blk.get("text", ""))
        elif hasattr(blk, "text"):
            parts.append(getattr(blk, "text") or "")
    joined = "\n".join(t for t in parts if t)
    return joined or None


def _blocks_of(content: Union[str, List[Any]]) -> List[Dict[str, Any]]:
    """Normalize message content to a list of plain block dicts."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    out: List[Dict[str, Any]] = []
    for blk in content:
        if isinstance(blk, dict):
            out.append(blk)
        elif hasattr(blk, "model_dump"):
            out.append(blk.model_dump())
        elif hasattr(blk, "type"):
            out.append({"type": getattr(blk, "type"), "text": getattr(blk, "text", "")})
    return out


def _effort_from_thinking(thinking: Optional[Dict[str, Any]]) -> Optional[str]:
    """Map Anthropic's thinking budget onto Kiro's effort enum.

    Anthropic expresses reasoning as a TOKEN BUDGET; Kiro takes an EFFORT ENUM
    (low/medium/high/xhigh/max). There is no lossless mapping between them, so
    this is an explicit, documented heuristic — not a protocol fact.

    Note the direction: upstream's defect was the reverse (converting an effort
    request into an Anthropic-style budget to send to Kiro, which takes no
    budget). Here the budget is what the harness sent us and the enum is what
    the wire needs.

    Returns None when thinking is absent or disabled, letting Lane B apply its
    own default.
    """
    if not thinking:
        return None
    if thinking.get("type") not in ("enabled", "adaptive", None):
        return None
    budget = thinking.get("budget_tokens")
    if not isinstance(budget, int):
        return None
    if budget < 4096:
        return "low"
    if budget < 8192:
        return "medium"
    if budget < 16384:
        return "high"
    if budget < 32768:
        return "xhigh"
    return "max"


def _to_history_messages(
    messages: List[_Message],
) -> List[HistoryMessage]:
    """Convert Anthropic messages to Lane B's HistoryMessage list.

    Per-role handling:
      - user turns: text is joined; `tool_result` blocks go through Lane B's
        `build_tool_results_from_anthropic`, which reads `is_error` so the Kiro
        `status` field is truthful (the §5b-bis fix). Base64 images are carried;
        URL images are reported, not silently dropped.
      - assistant turns: text is joined; `tool_use` blocks become Kiro
        `toolUses`. `thinking` blocks are intentionally dropped — Kiro's
        assistantResponseMessage has no slot for them, and a reasoning
        signature is a provider-specific blob that must not be relayed.

    System-role messages are skipped: the caller extracts the system prompt
    separately, since Kiro carries it in its own `systemPrompt` slot.
    """
    out: List[HistoryMessage] = []

    for msg in messages:
        if msg.role == "system":
            continue

        blocks = _blocks_of(msg.content)
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        images: List[Dict[str, Any]] = []

        for blk in blocks:
            btype = blk.get("type", "text")

            if btype == "text":
                if blk.get("text"):
                    text_parts.append(blk["text"])

            elif btype == "tool_use":
                tool_calls.append({
                    "toolUseId": blk.get("id", ""),
                    "name": blk.get("name", ""),
                    "input": blk.get("input", {}) or {},
                })

            elif btype == "image":
                source = blk.get("source") or {}
                if source.get("type") == "base64":
                    images.append({
                        "media_type": source.get("media_type", "image/png"),
                        "data": source.get("data", ""),
                    })
                else:
                    logger.warning(
                        "[anthropic] URL-based image dropped (Kiro cannot fetch "
                        "remote images); the model is not shown this image"
                    )

            # tool_result blocks are handled below, via Lane B's converter.
            # thinking / redacted_thinking are dropped: no Kiro slot, and the
            # signature is a provider-specific blob that must not be relayed.

        tool_results = build_tool_results_from_anthropic(msg.content) or None

        out.append(HistoryMessage(
            role=msg.role,
            content="\n".join(text_parts),
            tool_calls=tool_calls or None,
            tool_results=tool_results,
            images=images or None,
        ))

    return out


def _build_kiro_payload(
    request_data: AnthropicMessagesRequest,
    conversation_id: str,
    profile_arn: Optional[str],
) -> Tuple[dict, RewrittenOut]:
    """Build the Kiro payload via Lane B, returning it with its RewrittenOut.

    This is the Lane C -> Lane B seam. It calls Lane B directly and does NOT
    swallow failures: a broken seam must surface as an error, not degrade to a
    silently-wrong payload.

    Returns:
        (wire_payload, rewritten_out) — rewritten_out declares every injection
        Lane B made and is handed to the seam emit verbatim.
    """
    system_prompt = _extract_system_prompt(request_data.system)
    history_messages = _to_history_messages(request_data.messages)

    tools: Optional[List[Dict[str, Any]]] = None
    if request_data.tools:
        tools = [
            t if isinstance(t, dict) else t.model_dump(exclude_none=True)
            for t in request_data.tools
        ]

    return build_conversation_state(
        messages=history_messages,
        system_prompt=system_prompt,
        model_id=request_data.model,
        tools=tools,
        conversation_id=conversation_id,
        profile_arn=profile_arn,
        reasoning_effort=_effort_from_thinking(request_data.thinking),
    )


# ---------------------------------------------------------------------------
# Kiro event-stream parser
# ---------------------------------------------------------------------------

async def _iter_kiro_events(
    response: httpx.Response,
) -> AsyncGenerator[DispatchedEvent, None]:
    """Parse a Kiro AWS event-stream response into typed events.

    Thin delegation to Lane A, which owns frame decoding and `:event-type`
    dispatch. Deliberately has no fallback parser: if Lane A's contract breaks,
    that must surface loudly rather than degrade to a second, divergent
    implementation of the same protocol.
    """
    async for event in iter_events(response):
        yield event


# ---------------------------------------------------------------------------
# Seam emit (Lane F)
# ---------------------------------------------------------------------------

def _fire_seam_emit(envelope: SeamEnvelope) -> None:
    """Hand the turn to Lane F's seam emitter.

    `emit()` is itself fire-and-forget: it enqueues and returns, never blocks,
    never raises. So this is a direct call, not a guarded one — the seam
    contract's non-blocking invariant lives in Lane F, and wrapping it here in
    a broad `except` would only hide a genuinely broken seam.
    """
    emit(envelope)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter()


@router.post("/v1/messages", dependencies=[Depends(_verify_key)])
async def messages(
    request: Request,
    request_data: AnthropicMessagesRequest,
    anthropic_version: Optional[str] = Header(None, alias="anthropic-version"),
    x_saga_request_id: Optional[str] = Header(None, alias="x-saga-request-id"),
) -> Any:
    """Anthropic Messages API — POST /v1/messages.

    Accepts Claude Code's Anthropic-format requests, translates them to Kiro's
    protocol, and streams back Anthropic SSE events.
    """
    logger.info(
        f"[anthropic] POST /v1/messages model={request_data.model!r} "
        f"stream={request_data.stream}"
    )

    request_id = x_saga_request_id or f"req_{uuid.uuid4().hex[:16]}"
    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # ------------------------------------------------------------------
    # Resolve account / auth
    # ------------------------------------------------------------------
    account_manager = request.app.state.account_manager

    # Support both account-system mode and legacy single-account mode
    if getattr(request.app.state, "account_system", False):
        account = await account_manager.get_next_account(request_data.model)
        if account is None:
            return JSONResponse(
                status_code=503,
                content={
                    "type": "error",
                    "error": {"type": "api_error", "message": "No available accounts"},
                },
            )
    else:
        # Legacy single-account mode — and note ACCOUNT_SYSTEM defaults to false,
        # so this is the DEFAULT path. Unlike get_next_account(), this one does
        # not lazily initialize; it raises RuntimeError until something has
        # initialized an account. That is an expected operational state (no
        # credentials configured yet), so it becomes a 503 rather than an
        # uncaught RuntimeError surfacing as a 500.
        try:
            account = account_manager.get_first_account()
        except RuntimeError as exc:
            logger.error(f"[anthropic] no initialized account: {exc}")
            return JSONResponse(
                status_code=503,
                content={
                    "type": "error",
                    "error": {"type": "api_error", "message": str(exc)},
                },
            )

    auth_manager = account.auth_manager
    if auth_manager is None:
        return JSONResponse(
            status_code=503,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": "Account not initialized"},
            },
        )

    # ------------------------------------------------------------------
    # Build Kiro payload (Lane B or stub)
    # ------------------------------------------------------------------
    conversation_id = str(uuid.uuid4())
    profile_arn = await resolve_profile_arn(auth_manager, PROFILE_ARN)

    try:
        kiro_payload, rewritten_out = _build_kiro_payload(
            request_data, conversation_id, profile_arn
        )
    except (ValueError, TypeError) as exc:
        logger.error(f"[anthropic] payload build error: {exc}")
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": str(exc)},
            },
        )

    # ------------------------------------------------------------------
    # Send request to Kiro
    # ------------------------------------------------------------------
    url = f"{auth_manager.api_host}/generateAssistantResponse"
    http_client = KiroHttpClient(auth_manager, shared_client=None)

    try:
        kiro_response = await http_client.request_with_retry(
            "POST", url, kiro_payload, stream=True
        )
    except HTTPException as exc:
        await http_client.close()
        raise
    except Exception as exc:
        await http_client.close()
        logger.error(f"[anthropic] upstream request failed: {exc}", exc_info=True)
        return JSONResponse(
            status_code=502,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": str(exc)},
            },
        )

    if kiro_response.status_code != 200:
        try:
            err_body = await kiro_response.aread()
        except Exception:
            err_body = b"(unreadable)"
        await http_client.close()
        err_text = err_body.decode("utf-8", errors="replace")
        logger.warning(
            f"[anthropic] Kiro returned {kiro_response.status_code}: {err_text[:200]}"
        )
        return JSONResponse(
            status_code=kiro_response.status_code,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": err_text},
            },
        )

    # ------------------------------------------------------------------
    # Report success to account manager
    # ------------------------------------------------------------------
    if getattr(request.app.state, "account_system", False):
        await account_manager.report_success(account.id, request_data.model)

    # ------------------------------------------------------------------
    # Streaming path
    # ------------------------------------------------------------------
    if request_data.stream:

        async def _stream_wrapper() -> asyncio.AsyncGenerator[str, None]:
            metrics = TurnMetrics()
            ts_start = int(time.time() * 1000)
            streaming_error: Optional[Exception] = None
            client_disconnected = False

            try:
                async for chunk in stream_to_anthropic_sse(
                    events=_iter_kiro_events(kiro_response),
                    model=request_data.model,
                    message_id=message_id,
                    metrics=metrics,
                ):
                    yield chunk

            except GeneratorExit:
                client_disconnected = True
                logger.debug("[anthropic] client disconnected during streaming")
            except Exception as exc:
                streaming_error = exc
                logger.error(f"[anthropic] streaming error: {exc}", exc_info=True)
                try:
                    err_event = (
                        f'event: error\ndata: {json.dumps({"type": "error", "error": {"type": "api_error", "message": str(exc)}})}\n\n'
                    )
                    yield err_event
                except Exception:
                    pass
            finally:
                await http_client.close()
                if streaming_error:
                    logger.error(
                        f"[anthropic] HTTP 500 - POST /v1/messages (streaming) - {streaming_error}"
                    )
                elif client_disconnected:
                    logger.info("[anthropic] HTTP 200 - POST /v1/messages - client disconnected")
                else:
                    logger.info("[anthropic] HTTP 200 - POST /v1/messages - streaming completed")
                    # Fire-and-forget seam emit
                    _fire_seam_emit(SeamEnvelope(
                        request_id=request_id,
                        ts=int(time.time() * 1000),
                        model_id=request_data.model,
                        metrics=metrics,
                        rewritten_out=rewritten_out,
                    ))

        return StreamingResponse(
            _stream_wrapper(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------
    # Non-streaming path — collect entire response
    # ------------------------------------------------------------------
    metrics = TurnMetrics()
    content_blocks: List[Dict[str, Any]] = []
    thinking_text_buf = ""
    thinking_sig_buf  = ""
    text_buf          = ""
    tool_blocks_out: List[Dict[str, Any]] = []

    # Track block structure
    _curr_thinking = False
    _curr_text     = False

    async for ev in _iter_kiro_events(kiro_response):
        if isinstance(ev, UnknownEvent):
            continue
        name = ev.name
        payload = ev.payload

        if name == EventName.REASONING_CONTENT:
            thinking_text_buf += payload.get("content", "")
            sig = payload.get("signature")
            if sig:
                thinking_sig_buf = sig
            # reasoningContentEvent arrives as MANY deltas per thinking
            # block, so record ONE ReasoningBlockMeta for the block and enrich
            # it as later deltas carry modelId / signature. Lane A's
            # apply_reasoning_content_event() appends per event by design,
            # which would inflate the count the seam reports.
            if not metrics.reasoning_blocks:
                metrics.reasoning_blocks.append(ReasoningBlockMeta())
            _blk = metrics.reasoning_blocks[-1]
            _model_id = payload.get("modelId") or payload.get("model_id")
            if _model_id and not _blk.model_id:
                _blk.model_id = str(_model_id)
            if sig:
                _blk.signature_present = True

        elif name == EventName.ASSISTANT_RESPONSE:
            text_buf += payload.get("content", "")

        elif name == EventName.TOOL_USE:
            tool_blocks_out.append({
                "type": "tool_use",
                "id": payload.get("toolUseId") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": payload.get("name", ""),
                "input": payload.get("input", {}),
            })

        # Metrics are parsed by Lane A, never hand-rolled here. Lane A owns
        # every wire field name — including thoughtTokens and BOTH cache-field
        # spellings (cachedReadTokens / cacheReadInputTokens), which the wire
        # may use interchangeably. Duplicating that mapping here is how the
        # non-streaming path silently reported zeros while exact counts were on
        # the wire: it read cacheCreationInputTokens (an Anthropic name, not
        # Kiro's), took credits from the wrong key, and never read
        # thoughtTokens at all.
        elif name == EventName.METADATA:
            apply_metadata_event(metrics, payload)

        elif name == EventName.METERING:
            apply_metering_event(metrics, payload)

        elif name == EventName.CONTEXT_USAGE:
            apply_context_usage_event(metrics, payload)

        elif name == EventName.MESSAGE_STOP:
            kiro_reason = payload.get("stopReason")
            if kiro_reason:
                from conduit.contracts.metrics import StopReason as SR
                try:
                    metrics.stop_reason = SR(kiro_reason)
                except ValueError:
                    pass

    await http_client.close()

    # Assemble content blocks
    if thinking_text_buf:
        blk: Dict[str, Any] = {"type": "thinking", "thinking": thinking_text_buf}
        if thinking_sig_buf:
            blk["signature"] = thinking_sig_buf
        content_blocks.append(blk)
    if text_buf:
        content_blocks.append({"type": "text", "text": text_buf})
    content_blocks.extend(tool_blocks_out)

    # Determine stop_reason
    if tool_blocks_out:
        stop_reason = "tool_use"
    elif metrics.stop_reason:
        from conduit.doors.anthropic_sse import _map_stop_reason
        stop_reason = _map_stop_reason(metrics.stop_reason.value)
    else:
        stop_reason = "end_turn"

    # Seam emit
    _fire_seam_emit(SeamEnvelope(
        request_id=request_id,
        ts=int(time.time() * 1000),
        model_id=request_data.model,
        metrics=metrics,
        rewritten_out=rewritten_out,
    ))

    usage_out: Dict[str, Any] = {
        "input_tokens":  metrics.input_tokens,
        "output_tokens": metrics.output_tokens,
        "cache_read_input_tokens":    metrics.cache_read_tokens,
        "cache_creation_input_tokens": metrics.cache_write_tokens,
    }

    logger.info("[anthropic] HTTP 200 - POST /v1/messages (non-streaming) - completed")
    return JSONResponse(content={
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": request_data.model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage_out,
    })


@router.post("/v1/messages/count_tokens", dependencies=[Depends(_verify_key)])
async def count_tokens(
    request: Request,
    request_data: _CountTokensRequest,
) -> Any:
    """Token-count stub.

    Returns a rough estimate based on message lengths.  Claude Code calls this
    before deciding to compact; the number only needs to be in the right
    ballpark.
    """
    logger.info(
        f"[anthropic] POST /v1/messages/count_tokens model={request_data.model!r}"
    )

    def _approx_tokens(text: str) -> int:
        # ~4 chars per token, very rough
        return max(1, len(text) // 4)

    total = 0
    for msg in request_data.messages:
        if isinstance(msg.content, str):
            total += _approx_tokens(msg.content)
        elif isinstance(msg.content, list):
            for blk in msg.content:
                if isinstance(blk, dict):
                    total += _approx_tokens(blk.get("text", ""))
                elif hasattr(blk, "text"):
                    total += _approx_tokens(blk.text)

    if isinstance(request_data.system, str):
        total += _approx_tokens(request_data.system)
    elif isinstance(request_data.system, list):
        for blk in request_data.system:
            if isinstance(blk, dict):
                total += _approx_tokens(blk.get("text", ""))

    if request_data.tools:
        for tool in request_data.tools:
            total += _approx_tokens(tool.description or "")
            total += _approx_tokens(
                json.dumps(tool.input_schema) if tool.input_schema else ""
            )

    return JSONResponse(content={"input_tokens": total})
