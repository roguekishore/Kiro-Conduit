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
    WEB_SEARCH_ENABLED,
)
from conduit.contracts.events import DispatchedEvent, EventName, KiroEvent, UnknownEvent
from conduit.contracts.metrics import ReasoningBlockMeta, TurnMetrics
from conduit.contracts.rewritten import RewrittenOut
from conduit.contracts.seam import SeamEnvelope
from conduit.doors.anthropic_sse import stream_to_anthropic_sse
from conduit.kiro.dispatch import iter_events
from conduit.kiro.history import HistoryMessage
from conduit.kiro.mcp import render_replayed_search
from conduit.kiro.request import build_conversation_state
from conduit.kiro.tools import build_tool_results_from_anthropic
from conduit.kiro.derive import derive_token_accounting
from conduit.kiro.usage import (
    apply_context_usage_event,
    apply_metadata_event,
    apply_metering_event,
)
from conduit.seam.emit import emit
from conduit.foundation.auth import resolve_profile_arn
from conduit.foundation.http_client import KiroHttpClient
from conduit.foundation.kiro_errors import enhance_kiro_error
from conduit.foundation.model_resolver import normalize_model_name


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
    # Claude Code's actual reasoning knob. Declared rather than left to
    # `extra: allow` because the door reads it — see _effort_from_request.
    output_config: Optional[Dict[str, Any]] = None
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


def _effort_from_request(
    thinking: Optional[Dict[str, Any]],
    output_config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Recover the client's requested reasoning effort as Kiro's effort enum.

    Two sources, in priority order.

    **1. ``output_config.effort`` — the real one.** Claude Code sends its session
    effort here, and it is a pass-through, not a translation: the field name is
    the same on both sides and the enum is the same five strings
    (low/medium/high/xhigh/max). Measured over the captured corpus (695
    `claude-code` requests): 334 carried ``output_config: {"effort": "high"}``,
    always paired with ``thinking: {"type": "adaptive"}``.

    **2. ``thinking.budget_tokens`` — a fallback for other clients.** An
    Anthropic-shaped caller may express reasoning as a token budget instead.
    There is no lossless mapping from a budget to an enum, so the thresholds
    below are an explicit heuristic, not a protocol fact. Claude Code never
    sends this: ``budget_tokens`` appears **zero** times in the corpus, which is
    why relying on it alone sent no effort at all on 334 of 448 main requests.

    ``thinking.type == "disabled"`` short-circuits to None deliberately. Those
    are Claude Code's auto-mode permission classifier calls (365 in the corpus,
    a ~122k-character prompt returning a one-token verdict, run *before* each
    gated tool call). Handing them an effort level would make every tool call
    slower for no benefit. Note that 18 of them do carry an ``output_config`` —
    a ``format``/``json_schema`` block, not an effort — so reading the key
    specifically, rather than testing for the object, matters.

    Returns None when no effort was requested, letting Lane B decide.
    """
    if thinking and thinking.get("type") == "disabled":
        return None

    if isinstance(output_config, dict):
        effort = output_config.get("effort")
        # Validation is reasoning.py's job (normalize_effort); the door only
        # decides whether a value was requested at all.
        if isinstance(effort, str) and effort.strip():
            return effort.strip().lower()

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

            # Server-side web search, replayed by the client on a later
            # request. Kiro has no slot for either block type, so before this
            # they fell through every branch and vanished: the model saw an
            # assistant turn that searched for nothing and found nothing, and
            # any follow-up question about the results was answered blind.
            # Rendered into text, they survive the round trip.
            elif btype == "server_tool_use":
                if blk.get("name") == "web_search":
                    args = blk.get("input") or {}
                    query = str(args.get("query", "")) if isinstance(args, dict) else ""
                    if query:
                        text_parts.append(f'\n[web search: "{query}"]\n')

            elif btype == "web_search_tool_result":
                rendered = render_replayed_search(blk.get("content"))
                if rendered:
                    text_parts.append(rendered)

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

    # NORMALIZE before the id goes on the wire. Claude Code spells minor versions
    # with a dash (`claude-sonnet-4-6`); Kiro accepts only the dotted form
    # (`claude-sonnet-4.6`) and answers anything else with
    #   400 {"reason":"INVALID_MODEL_ID"}
    #
    # This door used to hand `request_data.model` straight through, which is why
    # only minor-version models failed: `claude-opus-5` has nothing to convert and
    # worked, while `claude-sonnet-4-6`, `claude-haiku-4-5` and `claude-opus-4-8`
    # were all rejected. The same models work on the upstream gateway precisely
    # because it normalizes here and CONDUIT did not — so this was never a
    # subscription-tier limit, which the error text below wrongly suggested.
    #
    # One conversion, at the single point where the Kiro payload is built:
    # build_conversation_state() propagates model_id into every userInputMessage,
    # the history entries, and the reasoning fields, so normalizing at the call
    # site keeps all of them consistent.
    return build_conversation_state(
        messages=history_messages,
        system_prompt=system_prompt,
        model_id=normalize_model_name(request_data.model),
        tools=tools,
        conversation_id=conversation_id,
        profile_arn=profile_arn,
        reasoning_effort=_effort_from_request(
            request_data.thinking, request_data.output_config
        ),
    )


# ---------------------------------------------------------------------------
# Web search continuation (Paths A and B)
# ---------------------------------------------------------------------------

def _strip_web_search_tools(request_data: AnthropicMessagesRequest) -> None:
    """Remove web_search tool definitions from a request, in place.

    Called before a continuation call. The search has already run; leaving the
    tool advertised invites the model to call it again on a code path with no
    interceptor, producing a tool_use nothing answers and a stalled turn.
    """
    if not request_data.tools:
        return
    kept = []
    for t in request_data.tools:
        name = getattr(t, "name", None) or (
            t.get("name") if isinstance(t, dict) else None
        )
        ttype = getattr(t, "type", None) or (
            t.get("type") if isinstance(t, dict) else None
        )
        if name == "web_search" or (ttype and str(ttype).startswith("web_search")):
            continue
        kept.append(t)
    request_data.tools = kept or None


def _search_block_frames(
    index: int,
    tool_use_id: str,
    query: str,
    results: Dict[str, Any],
) -> List[str]:
    """Build the SSE frames for one server-side search.

    Emits ``server_tool_use`` at ``index`` and ``web_search_tool_result`` at
    ``index + 1`` - the shape Anthropic clients render as a citation.
    """
    search_content = [
        {
            "type": "web_search_result",
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "encrypted_content": r.get("snippet", ""),
            "page_age": None,
        }
        for r in results.get("results", [])
    ]
    return [
        _sse_frame("content_block_start", {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "id": tool_use_id,
                "type": "server_tool_use",
                "name": "web_search",
                "input": {},
            },
        }),
        _sse_frame("content_block_delta", {
            "type": "content_block_delta",
            "index": index,
            "delta": {
                "type": "input_json_delta",
                "partial_json": json.dumps({"query": query}),
            },
        }),
        _sse_frame("content_block_stop", {
            "type": "content_block_stop",
            "index": index,
        }),
        _sse_frame("content_block_start", {
            "type": "content_block_start",
            "index": index + 1,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": tool_use_id,
                "content": search_content,
            },
        }),
        _sse_frame("content_block_stop", {
            "type": "content_block_stop",
            "index": index + 1,
        }),
    ]



def _sse_frame(event_type: str, data: dict) -> str:
    """Format one Anthropic SSE frame."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _augment_messages_for_search(
    request_data: AnthropicMessagesRequest,
    query: str,
    summary: str,
    tool_id: Optional[str] = None,
) -> None:
    """Fold a completed web search into ``request_data.messages``, in place.

    This is what makes search actually inform the answer. Before this existed,
    both search paths returned the raw results and stopped, so the model never
    saw them: the turn ended with results on screen and no answer, and the
    client had nothing to continue from.

    Two shapes, because the two paths differ in what the model asked for:

    - ``tool_id`` given (Path B): the model itself called the injected
      ``web_search`` tool, so the results go back the native way — an assistant
      turn carrying the ``tool_use`` followed by a user turn carrying the
      matching ``tool_result``. Kiro has real slots for both.
    - ``tool_id`` omitted (Path A): the client asked for Anthropic's
      server-side tool, so there is no client tool call to answer. The summary
      is appended to the last user turn as text instead, which keeps the
      user/assistant alternation Kiro requires.

    Args:
        request_data: The request being rewritten; mutated in place.
        query:        The executed search query.
        summary:      Human-readable search results (build_search_summary).
        tool_id:      The model's tool_use id, for Path B only.
    """
    if tool_id:
        request_data.messages = list(request_data.messages) + [
            _Message(
                role="assistant",
                content=[{
                    "type": "tool_use",
                    "id": tool_id,
                    "name": "web_search",
                    "input": {"query": query},
                }],
            ),
            _Message(
                role="user",
                content=[{
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": summary,
                }],
            ),
        ]
        return

    msgs = list(request_data.messages)
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].role == "user":
            msg = msgs[i]
            if isinstance(msg.content, str):
                msg.content = f"{msg.content}\n\n{summary}"
            elif isinstance(msg.content, list):
                msg.content = list(msg.content) + [
                    {"type": "text", "text": summary}
                ]
            break
    else:
        msgs.append(
            _Message(role="user", content=[{"type": "text", "text": summary}])
        )
    request_data.messages = msgs


async def _search_continuation_stream(
    request_data: AnthropicMessagesRequest,
    auth_manager: Any,
    profile_arn: Optional[str],
    message_id: str,
    metrics: TurnMetrics,
    start_index: int,
) -> AsyncGenerator[str, None]:
    """Call Kiro a second time, with search results in context, and stream it.

    Emits no ``message_start``: the caller already opened the message and wrote
    the search blocks into it, and Anthropic allows exactly one per message.
    ``start_index`` continues that block numbering so indices never collide.

    A failure here must not leave the client hanging on an unterminated
    message, so every error path still closes the message with a text block
    plus ``message_delta`` / ``message_stop``.
    """
    conversation_id = str(uuid.uuid4())
    http_client = KiroHttpClient(auth_manager, shared_client=None)
    try:
        try:
            kiro_payload, _rewritten = _build_kiro_payload(
                request_data, conversation_id, profile_arn
            )
            url = f"{auth_manager.api_host}/generateAssistantResponse"
            kiro_response = await http_client.request_with_retry(
                "POST", url, kiro_payload, stream=True
            )
        except Exception as exc:
            logger.error(
                f"[anthropic] web_search continuation request failed: {exc}",
                exc_info=True,
            )
            async for frame in _close_message_with_text(
                start_index,
                "Web search completed, but the follow-up request to the model "
                f"failed: {exc}",
            ):
                yield frame
            return

        if kiro_response.status_code != 200:
            try:
                err_body = await kiro_response.aread()
            except Exception:
                err_body = b"(unreadable)"
            err_text = err_body.decode("utf-8", errors="replace")
            logger.warning(
                f"[anthropic] web_search continuation got "
                f"{kiro_response.status_code}: {err_text[:200]}"
            )
            async for frame in _close_message_with_text(
                start_index,
                "Web search completed, but the model returned "
                f"{kiro_response.status_code}: {err_text[:300]}",
            ):
                yield frame
            return

        async for chunk in stream_to_anthropic_sse(
            events=_iter_kiro_events(kiro_response),
            model=request_data.model,
            message_id=message_id,
            metrics=metrics,
            start_index=start_index,
            emit_message_start=False,
        ):
            yield chunk
    finally:
        await http_client.close()


async def _close_message_with_text(
    index: int,
    text: str,
) -> AsyncGenerator[str, None]:
    """Emit one text block and terminate the message.

    Used when search succeeded but the continuation call did not: the results
    are already on the wire, so the turn must end as a valid message rather
    than an error event the client cannot attach to anything.
    """
    yield _sse_frame("content_block_start", {
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "text", "text": ""},
    })
    yield _sse_frame("content_block_delta", {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    })
    yield _sse_frame("content_block_stop", {
        "type": "content_block_stop",
        "index": index,
    })
    yield _sse_frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 0},
    })
    yield _sse_frame("message_stop", {"type": "message_stop"})


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
    # Web Search - Path A: native Anthropic server-side tool
    # Client asked for Anthropic's server-side web_search. Run the search via
    # MCP, emit the server-side blocks, then CALL THE MODEL AGAIN with the
    # results in context and stream its real answer.
    #
    # This used to early-return the results and stop, which meant the model
    # never saw them: the turn ended with search output and no answer, so the
    # results became the response instead of informing it.
    # ------------------------------------------------------------------
    if request_data.tools:
        for tool in request_data.tools:
            tool_type = getattr(tool, "type", None) or (
                tool.get("type") if isinstance(tool, dict) else None
            )
            if tool_type and str(tool_type).startswith("web_search"):
                from conduit.kiro.mcp import (
                    call_mcp_web_search,
                    build_search_summary,
                )

                # LAST user message, not the first. The query was previously
                # scraped from messages[0], so every search in an ongoing
                # conversation searched the opening line of the session
                # instead of what was just asked.
                query: Optional[str] = None
                for msg in reversed(list(request_data.messages)):
                    if msg.role != "user":
                        continue
                    content = msg.content
                    if isinstance(content, str):
                        query = content.strip()
                    elif isinstance(content, list):
                        parts = []
                        for blk in content:
                            if isinstance(blk, dict) and blk.get("type") == "text":
                                parts.append(blk.get("text", ""))
                            elif getattr(blk, "type", None) == "text":
                                parts.append(getattr(blk, "text", ""))
                        query = "".join(parts).strip()
                    if query:
                        break

                if not query:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "type": "error",
                            "error": {
                                "type": "invalid_request_error",
                                "message": "Cannot extract search query from messages",
                            },
                        },
                    )

                logger.info(f"[anthropic] web_search Path A - query: {query!r}")
                ws_tool_use_id, ws_results = await call_mcp_web_search(
                    query, auth_manager
                )

                if ws_results is None:
                    return JSONResponse(
                        status_code=500,
                        content={
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": "Web search failed. Please try again.",
                            },
                        },
                    )

                summary = build_search_summary(query, ws_results)
                # Fold results into the conversation, then drop the search tool:
                # the continuation call must not offer a tool that nothing on
                # this path is listening to service.
                _augment_messages_for_search(request_data, query, summary)
                _strip_web_search_tools(request_data)
                profile_arn_a = await resolve_profile_arn(auth_manager, PROFILE_ARN)

                if request_data.stream:
                    async def _ws_stream_a() -> AsyncGenerator[str, None]:
                        metrics_a = TurnMetrics()
                        yield _sse_frame("message_start", {
                            "type": "message_start",
                            "message": {
                                "id": message_id,
                                "type": "message",
                                "role": "assistant",
                                "content": [],
                                "model": request_data.model,
                                "stop_reason": None,
                                "stop_sequence": None,
                                "usage": {"input_tokens": 0, "output_tokens": 0},
                            },
                        })
                        yield _sse_frame("ping", {"type": "ping"})
                        for frame in _search_block_frames(
                            0, ws_tool_use_id, query, ws_results
                        ):
                            yield frame
                        # Model speaks from index 2: the two search blocks own
                        # 0 and 1, and indices must not collide in one message.
                        async for frame in _search_continuation_stream(
                            request_data,
                            auth_manager,
                            profile_arn_a,
                            message_id,
                            metrics_a,
                            2,
                        ):
                            yield frame

                    return StreamingResponse(
                        _ws_stream_a(),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "X-Accel-Buffering": "no",
                        },
                    )

                # Non-streaming: fall through to the normal request path below.
                # request_data now carries the results as context, so the model
                # answers with them. The server-side blocks are not replayed in
                # the JSON body - a non-streaming client gets the answer, not
                # the citation scaffolding.
                logger.debug(
                    "[anthropic] web_search Path A (non-streaming) - results "
                    "folded into context; continuing through the normal path"
                )
                break

    # ------------------------------------------------------------------
    # Web Search — Path B: auto-inject web_search tool (MCP emulation)
    # The model decides whether to call it; the streaming wrapper intercepts
    # the tool-use block and replaces it with the MCP result.
    # ------------------------------------------------------------------
    if WEB_SEARCH_ENABLED:
        existing_names = set()
        if request_data.tools:
            for t in request_data.tools:
                n = getattr(t, "name", None) or (
                    t.get("name") if isinstance(t, dict) else None
                )
                if n:
                    existing_names.add(n)

        if "web_search" not in existing_names:
            ws_tool = _Tool(
                name="web_search",
                description=(
                    "Search the web for current information. "
                    "Use when you need up-to-date data from the internet."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        }
                    },
                    "required": ["query"],
                },
            )
            if request_data.tools is None:
                request_data.tools = [ws_tool]
            else:
                request_data.tools = list(request_data.tools) + [ws_tool]
            logger.debug("[anthropic] auto-injected web_search tool (Path B)")

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

        # Kiro's own message is frequently useless on its own. Measured: a
        # request for claude-opus-4.8 came back
        #   {"message":"Invalid model ID. Please select a different model to
        #    continue.","reason":"INVALID_MODEL_ID"}
        # while that exact id IS in FALLBACK_MODELS and normalizes cleanly
        # (claude-opus-4-8 → claude-opus-4.8). So the rejection is Kiro's, not a
        # mapping bug here — and per kiro_errors, INVALID_MODEL_ID means the name
        # is unknown *or the account's subscription does not cover it*. Without
        # that distinction the message sends you looking for a gateway bug that
        # does not exist.
        #
        # enhance_kiro_error() and get_suggestions_for_model() both already
        # existed for exactly this and were never wired to a caller. Wire them.
        message = err_text
        try:
            error_json = json.loads(err_text)
        except (json.JSONDecodeError, TypeError):
            error_json = None
        if isinstance(error_json, dict):
            info = enhance_kiro_error(error_json)
            message = info.user_message
            if info.reason == "INVALID_MODEL_ID":
                requested = request_data.model
                normalized = normalize_model_name(requested)
                message = (
                    f"{info.user_message} Requested {requested!r} "
                    f"(sent to Kiro as {normalized!r})."
                )
                # Same-family suggestions only: get_suggestions_for_model() never
                # crosses families, so an Opus request is never answered with
                # "try Haiku". The resolver hangs off the account and is Optional,
                # so a missing one costs the suggestion list, not the response.
                resolver = getattr(account, "model_resolver", None)
                if resolver is not None:
                    try:
                        alternatives = [
                            m
                            for m in resolver.get_suggestions_for_model(requested)
                            if m != normalized
                        ]
                    except Exception as exc:  # never let a hint break the error path
                        logger.debug(f"[anthropic] model suggestions unavailable: {exc}")
                        alternatives = []
                    if alternatives:
                        message += f" Available in this family: {', '.join(alternatives)}."
                logger.warning(
                    f"[anthropic] INVALID_MODEL_ID for {requested!r} "
                    f"(sent to Kiro as {normalized!r}). Either Kiro does not know that id "
                    f"or this account's plan does not cover it — check it against "
                    f"GET /v1/models before suspecting the mapping."
                )

        return JSONResponse(
            status_code=kiro_response.status_code,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": message,
                    # The raw upstream body stays available for debugging: the
                    # enhanced message is for humans, not a replacement for
                    # evidence.
                    "upstream_message": err_text,
                },
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

            # Path B streaming intercept: buffer SSE chunks to detect a
            # web_search tool_use block, then substitute MCP results for it.
            # We parse the already-formatted SSE text back to check for
            # tool_use blocks whose name is "web_search".
            _ws_intercepting = False  # True once we see a web_search tool_use start
            _ws_tool_idx: Optional[int] = None  # SSE block index being intercepted
            _ws_query_buf: list = []             # accumulate input_json_delta fragments
            # Set once the search results have been handed back to the model and
            # its continuation stream has been forwarded. The original stream is
            # finished at that point (the model stopped to call the tool) and it
            # still owes a message_delta/message_stop pair, which would be a
            # SECOND terminator for a message the continuation already closed.
            # So the outer loop stops reading once this is set.
            _ws_continued = False

            try:
                async for chunk in stream_to_anthropic_sse(
                    events=_iter_kiro_events(kiro_response),
                    model=request_data.model,
                    message_id=message_id,
                    metrics=metrics,
                ):
                    # --- Path B intercept ---
                    # We look for content_block_start with type==tool_use and
                    # name==web_search, then absorb its deltas / stop, call the
                    # MCP API, and emit the synthetic server-side response frames.
                    if "\ndata: " in chunk:
                        # Extract JSON payload from SSE frame
                        for line in chunk.split("\n"):
                            if not line.startswith("data: "):
                                continue
                            try:
                                evt_data = json.loads(line[6:])
                            except (json.JSONDecodeError, ValueError):
                                continue

                            evt_type = evt_data.get("type")

                            if evt_type == "content_block_start":
                                blk = evt_data.get("content_block", {})
                                if (
                                    blk.get("type") == "tool_use"
                                    and blk.get("name") == "web_search"
                                ):
                                    _ws_intercepting = True
                                    _ws_tool_idx = evt_data.get("index")
                                    _ws_query_buf = []
                                    # Suppress: don't yield this frame
                                    break

                            if _ws_intercepting:
                                if evt_type == "content_block_delta":
                                    delta = evt_data.get("delta", {})
                                    if delta.get("type") == "input_json_delta":
                                        _ws_query_buf.append(
                                            delta.get("partial_json", "")
                                        )
                                    break  # suppress

                                if evt_type == "content_block_stop":
                                    if evt_data.get("index") == _ws_tool_idx:
                                        # We have all fragments — call MCP API
                                        _ws_intercepting = False
                                        try:
                                            raw = "".join(_ws_query_buf).strip() or "{}"
                                            ws_args = json.loads(raw)
                                            ws_query = ws_args.get("query", "")
                                        except (json.JSONDecodeError, AttributeError):
                                            ws_query = ""

                                        if ws_query:
                                            logger.info(
                                                f"[anthropic] web_search Path B — "
                                                f"query: {ws_query!r}"
                                            )
                                            from conduit.kiro.mcp import (
                                                call_mcp_web_search,
                                                build_search_summary,
                                            )
                                            ws_tid, ws_res = await call_mcp_web_search(
                                                ws_query, auth_manager
                                            )
                                            if ws_res is not None:
                                                # The intercepted tool_use owns
                                                # idx; the two search blocks take
                                                # idx and idx+1, so the model's
                                                # continuation starts at idx+2.
                                                # Anthropic block indices must be
                                                # unique within one message.
                                                idx = _ws_tool_idx or 0
                                                for _frame in _search_block_frames(
                                                    idx, ws_tid, ws_query, ws_res
                                                ):
                                                    yield _frame

                                                # Hand the results back as a real
                                                # tool_result: on this path the
                                                # MODEL called web_search, so the
                                                # native round trip is what it is
                                                # waiting for. Without this the
                                                # turn ended here - results on
                                                # screen, question unanswered.
                                                _augment_messages_for_search(
                                                    request_data,
                                                    ws_query,
                                                    build_search_summary(
                                                        ws_query, ws_res
                                                    ),
                                                    tool_id=ws_tid,
                                                )
                                                _strip_web_search_tools(request_data)

                                                async for _frame in _search_continuation_stream(
                                                    request_data,
                                                    auth_manager,
                                                    profile_arn,
                                                    message_id,
                                                    metrics,
                                                    idx + 2,
                                                ):
                                                    yield _frame

                                                _ws_continued = True
                                        break  # suppress original stop frame
                                break  # suppress all other frames while intercepting
                        else:
                            # No web_search interception — pass through unchanged
                            if not _ws_intercepting:
                                yield chunk
                        # If we parsed a break above and didn't yield, don't fall through
                        if _ws_continued:
                            # Continuation already closed the message.
                            break
                        continue
                    # Not an SSE data frame (e.g. blank line) — pass through
                    if not _ws_intercepting:
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
    output_chars      = 0

    # Tool fragment state: one entry per in-flight tool call, keyed by toolUseId.
    # Kiro sends opener (name, empty input) + N fragment frames (input string) +
    # closer (stop: true). We accumulate and emit one complete block per call.
    _tool_pending: Dict[str, Dict[str, Any]] = {}

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
            chunk = payload.get("content", "")
            text_buf += chunk
            output_chars += len(chunk)

        elif name == EventName.TOOL_USE:
            tid = payload.get("toolUseId") or f"toolu_{uuid.uuid4().hex[:24]}"
            if tid not in _tool_pending:
                # Opener frame: name present, input empty/absent.
                _tool_pending[tid] = {
                    "id": tid,
                    "name": payload.get("name", ""),
                    "fragments": [],
                }
            raw_input = payload.get("input", "")
            if isinstance(raw_input, str):
                _tool_pending[tid]["fragments"].append(raw_input)
            if payload.get("stop"):
                entry = _tool_pending.pop(tid)
                partial = "".join(entry["fragments"]).strip() or "{}"
                output_chars += len(partial)
                try:
                    parsed = json.loads(partial)
                except (json.JSONDecodeError, ValueError):
                    parsed = {"_raw": partial}
                tool_blocks_out.append({
                    "type": "tool_use",
                    "id": entry["id"],
                    "name": entry["name"],
                    "input": parsed,
                })

        # Metrics are parsed by Lane A, never hand-rolled here. Lane A owns
        # every wire field name — including thoughtTokens and BOTH cache-field
        # spellings (cachedReadTokens / cacheReadInputTokens), which the wire
        # may use interchangeably. Duplicating that mapping here is how the
        # non-streaming path silently reported zeros while exact counts were on
        # the wire: it read cacheCreationInputTokens (an Anthropic name, not
        # Kiro's), took credits from the wrong key, and never read
        # thoughtTokens at all.
        elif name in (EventName.METADATA, EventName.MESSAGE_METADATA):
            # Both wire spellings carry tokenUsage; see the streaming path.
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

    # Determine stop_reason — mirror streaming path logic in anthropic_sse.py:386.
    # CONTENT_FILTERED and other non-TOOL_USE stops must not be clobbered.
    from conduit.contracts.metrics import StopReason as _SR
    from conduit.doors.anthropic_sse import _map_stop_reason
    if tool_blocks_out and metrics.stop_reason in (_SR.TOOL_USE, None):
        stop_reason = "tool_use"
    elif metrics.stop_reason is not None:
        stop_reason = _map_stop_reason(metrics.stop_reason.value)
    else:
        stop_reason = "tool_use" if tool_blocks_out else "end_turn"

    # Derive token accounting when upstream reported none (same logic as SSE path).
    derive_token_accounting(metrics, request_data.model, output_chars)

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
