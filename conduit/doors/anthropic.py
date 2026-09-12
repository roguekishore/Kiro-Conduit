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
from typing import Any, Dict, List, Optional

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
from conduit.contracts.metrics import TurnMetrics
from conduit.contracts.rewritten import RewrittenOut
from conduit.contracts.seam import SeamEnvelope
from conduit.doors.anthropic_sse import stream_to_anthropic_sse
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
# Lane B interface stub
# ---------------------------------------------------------------------------

def _build_kiro_payload(
    request_data: AnthropicMessagesRequest,
    conversation_id: str,
    profile_arn: Optional[str],
) -> dict:
    """Lane B stub: builds a minimal Kiro conversationState payload.

    TODO: replace with ``from conduit.kiro.request import build_kiro_payload``
          once Lane B ships.  The interface is:
            build_kiro_payload(
                request_data: AnthropicMessagesRequest,
                conversation_id: str,
                profile_arn: Optional[str],
            ) -> dict
    """
    try:
        from conduit.kiro.request import build_kiro_payload  # Lane B
        return build_kiro_payload(request_data, conversation_id, profile_arn)
    except ImportError:
        pass

    # Minimal stub — produces a valid Kiro generateAssistantResponse body
    # so the door can be wired up and tested without Lane B.
    history: List[Dict[str, Any]] = []
    for msg in request_data.messages[:-1]:
        if isinstance(msg.content, str):
            content_str = msg.content
        else:
            content_str = " ".join(
                b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
                for b in msg.content
                if (isinstance(b, dict) and b.get("type") == "text")
                or (hasattr(b, "type") and getattr(b, "type") == "text")
            )
        history.append({"role": msg.role, "content": content_str})

    last_msg = request_data.messages[-1]
    if isinstance(last_msg.content, str):
        current_content = last_msg.content
    else:
        current_content = " ".join(
            b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
            for b in last_msg.content
            if (isinstance(b, dict) and b.get("type") == "text")
            or (hasattr(b, "type") and getattr(b, "type") == "text")
        )

    system_prompt = None
    if isinstance(request_data.system, str):
        system_prompt = request_data.system
    elif isinstance(request_data.system, list):
        system_prompt = " ".join(
            b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
            for b in request_data.system
        )

    payload: Dict[str, Any] = {
        "conversationState": {
            "conversationId": conversation_id,
            "history": history,
            "currentMessage": {
                "userInputMessage": {
                    "content": current_content,
                    "modelId": request_data.model,
                }
            },
        }
    }
    if profile_arn:
        payload["profileArn"] = profile_arn
    if system_prompt:
        payload["conversationState"]["systemPrompt"] = {"content": system_prompt}

    return payload


# ---------------------------------------------------------------------------
# Kiro event-stream parser
# ---------------------------------------------------------------------------

async def _parse_kiro_events(
    response: httpx.Response,
) -> asyncio.Queue:
    """Not used directly; events are yielded inline in the route handler."""
    raise NotImplementedError


async def _iter_kiro_events(
    response: httpx.Response,
) -> asyncio.AsyncGenerator[DispatchedEvent, None]:
    """Parse a Kiro AWS event-stream response into DispatchedEvent objects.

    This is the Lane A boundary — in production, Lane A provides
    ``conduit.kiro.dispatch.iter_events(response)``.  Until Lane A is wired,
    this minimal inline parser keeps the door functional.
    """
    try:
        from conduit.kiro.dispatch import iter_events  # Lane A
        async for ev in iter_events(response):
            yield ev
        return
    except ImportError:
        pass

    # -----------------------------------------------------------------------
    # Minimal inline AWS event-stream parser (Lane A stub)
    # Handles the binary framing used by Kiro's generateAssistantResponse.
    # -----------------------------------------------------------------------
    import struct

    buffer = b""

    async for chunk in response.aiter_bytes():
        buffer += chunk

        while len(buffer) >= 12:
            # Prelude: total_length(4) + headers_length(4) + prelude_crc(4)
            total_len = struct.unpack(">I", buffer[:4])[0]
            if len(buffer) < total_len:
                break

            frame = buffer[:total_len]
            buffer = buffer[total_len:]

            # headers_length tells us where headers end
            headers_len = struct.unpack(">I", frame[4:8])[0]
            headers_raw = frame[12 : 12 + headers_len]
            payload_raw = frame[12 + headers_len : total_len - 4]  # strip trailing CRC

            # Parse headers
            headers: Dict[str, str] = {}
            pos = 0
            while pos < len(headers_raw):
                if pos >= len(headers_raw):
                    break
                name_len = headers_raw[pos]
                pos += 1
                name = headers_raw[pos : pos + name_len].decode("utf-8", errors="replace")
                pos += name_len
                value_type = headers_raw[pos]
                pos += 1
                if value_type == 7:  # string
                    val_len = struct.unpack(">H", headers_raw[pos : pos + 2])[0]
                    pos += 2
                    val = headers_raw[pos : pos + val_len].decode("utf-8", errors="replace")
                    pos += val_len
                    headers[name] = val
                else:
                    # Skip unknown header value types
                    break

            event_type = headers.get(":event-type", "")
            if not event_type:
                continue

            # Decode JSON payload
            payload: Dict[str, Any] = {}
            if payload_raw:
                try:
                    payload = json.loads(payload_raw.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    pass

            # Map to typed event
            try:
                ename = EventName(event_type)
                yield KiroEvent(name=ename, payload=payload)
            except ValueError:
                yield UnknownEvent(name=event_type, payload=payload)


# ---------------------------------------------------------------------------
# Seam emit (Lane F)
# ---------------------------------------------------------------------------

def _fire_seam_emit(envelope: SeamEnvelope) -> None:
    """Fire-and-forget emit to Lane F's seam.

    TODO: replace stub with real ``from conduit.seam import emit; emit(envelope)``
          once Lane F ships.
    """
    try:
        from conduit.seam import emit  # Lane F
        # emit is sync fire-and-forget per contract
        asyncio.get_event_loop().call_soon(lambda: None)  # placeholder
        emit(envelope)
    except (ImportError, Exception) as exc:
        logger.debug(f"[anthropic] seam emit skipped: {exc}")


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
        account = account_manager.get_first_account()

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
        kiro_payload = _build_kiro_payload(request_data, conversation_id, profile_arn)
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
                        rewritten_out=RewrittenOut(),  # Lane B populates; stub for now
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

        elif name == EventName.ASSISTANT_RESPONSE:
            text_buf += payload.get("content", "")

        elif name == EventName.TOOL_USE:
            tool_blocks_out.append({
                "type": "tool_use",
                "id": payload.get("toolUseId") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": payload.get("name", ""),
                "input": payload.get("input", {}),
            })

        elif name == EventName.METADATA:
            usage = payload.get("tokenUsage") or {}
            metrics.input_tokens  = int(usage.get("inputTokens", 0))
            metrics.output_tokens = int(usage.get("outputTokens", 0))
            metrics.cache_read_tokens  = int(usage.get("cacheReadInputTokens", 0))
            metrics.cache_write_tokens = int(usage.get("cacheCreationInputTokens", 0))
            metrics.total_tokens = metrics.input_tokens + metrics.output_tokens

        elif name == EventName.METERING:
            metrics.credits = float(payload.get("credits", 0.0))

        elif name == EventName.CONTEXT_USAGE:
            pct = payload.get("contextWindowPercentage")
            if pct is not None:
                metrics.context_usage_percentage = float(pct)

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
        rewritten_out=RewrittenOut(),
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
