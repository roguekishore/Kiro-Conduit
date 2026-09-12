"""Codex door — POST /v1/responses, for Codex CLI -> Luna.

OWNER: Lane D. Registered by conduit/app.py; do not edit app.py to wire it.

Lane D implements the OpenAI Responses contract (FINDINGS-codex 1.2/1.5/1.7/1.8).
Load-bearing points, all verified in Codex source:

  - store:false / stream:true / tool_choice:"auto" are hardcoded client-side.
  - System prompt is the top-level `instructions` string -> Kiro systemPrompt.
  - BOTH wire dialects are required. Under responses-lite the prompt and tool
    list move OUT of instructions/tools and INTO input[] as prefix items,
    leaving instructions:"" and tools:null. A gateway reading only
    `instructions` sees an empty prompt for any lite model.
  - Dispatch the dialect on the header
    `x-openai-internal-codex-responses-lite: true`. Do not sniff the body.
    The flag is server-driven and can change with no CLI release.
  - Reasoning blobs are NOT portable: Codex encrypted_content and Kiro
    signature are provider-specific. Drop or store locally, never relay.
  - Tool status cannot come from Codex's wire — FunctionCallOutput has no
    item-level success/error flag. Supply it from CONDUIT's tool runner.
  - The authoritative session id is client_metadata.session_id in the BODY,
    not the session-id header (which carries the prompt-cache key for root
    agents by design). Recorded here for WS-C's benefit.

INTERCEPTION WIRING (document only, not built here):
  There is NO OPENAI_BASE_URL env var in Codex (verified). Two wiring options:
    (a) openai_base_url = "http://localhost:PORT/v1" in config.toml (smallest)
    (b) model_providers entry with wire_api = "responses"             (preferred)
  Under ChatGPT auth the attached token is a ChatGPT credential, not an API key.

LANE A/B INTEGRATION:
  This router is ready to accept the parsed-event stream from Lane A and the
  payload built by Lane B.  When those lanes are available, replace the
  _STUB comments in _make_kiro_request and _parse_kiro_stream_stub with the
  real calls to:
    - conduit.foundation.http_client.KiroHttpClient for the upstream request
    - Lane A's event-stream parser (yields KiroEvent / UnknownEvent)
    - Lane B's payload builder for the Kiro request body
  Until then the route returns a synthetic stream for demonstration only.

See SPEC-WORKSTREAMS.md, Lane D, for the full brief and verification gate.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Security
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from loguru import logger
from pydantic import BaseModel, Field

from conduit.config import PROXY_API_KEY
from conduit.contracts.events import EventName, KiroEvent, UnknownEvent
from conduit.contracts.metrics import StopReason, TurnMetrics
from conduit.doors.codex_dialect import LITE_HEADER, extract_dialect
from conduit.doors.codex_sse import (
    ResponsesErrorCode,
    translate_kiro_to_responses_sse,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def _verify_api_key(auth_header: str = Security(_api_key_header)) -> bool:
    """Require Bearer PROXY_API_KEY on every Codex door request."""
    if not auth_header or auth_header != f"Bearer {PROXY_API_KEY}":
        logger.warning("codex door: invalid or missing API key")
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return True


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class _InputItem(BaseModel):
    """One item in the Responses API ``input[]`` array."""

    model_config = {"extra": "allow"}

    type: Optional[str] = None
    role: Optional[str] = None
    content: Optional[Any] = None
    tools: Optional[List[Any]] = None


class _ClientMetadata(BaseModel):
    """Optional client_metadata block.

    The authoritative session id is ``client_metadata.session_id`` in the
    body, NOT the ``session-id`` header (which carries the prompt-cache key
    for root agents).
    """

    model_config = {"extra": "allow"}

    session_id: Optional[str] = None


class ResponsesRequest(BaseModel):
    """Slim Pydantic model for POST /v1/responses.

    Only the fields Lane D cares about are declared; everything else is
    captured by ``extra = "allow"`` so unknown fields round-trip silently.

    Codex hardcodes these on the wire (no need to re-validate here):
        store = false
        stream = true
        tool_choice = "auto"
    """

    model_config = {"extra": "allow"}

    model: str
    instructions: Optional[str] = None
    tools: Optional[List[Any]] = None
    input: List[_InputItem] = Field(default_factory=list)

    # service_tier: "priority" | "flex" only — "default" is filtered client-side
    service_tier: Optional[str] = None

    # client_metadata carries the authoritative session_id
    client_metadata: Optional[_ClientMetadata] = None

    # Always true from Codex; kept for explicitness
    stream: bool = True
    store: bool = False
    tool_choice: Optional[Any] = "auto"


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/v1/responses", dependencies=[Depends(_verify_api_key)])
async def post_responses(
    request: Request,
    body: ResponsesRequest,
) -> StreamingResponse:
    """POST /v1/responses — Codex CLI entry point.

    Dialect dispatch:
        x-openai-internal-codex-responses-lite: true  -> lite dialect
        (absent)                                       -> standard dialect

    Both dialects are handled unconditionally on every request.
    """
    is_lite = (
        request.headers.get(LITE_HEADER, "").strip().lower() == "true"
    )
    session_id = (
        body.client_metadata.session_id
        if body.client_metadata
        else None
    )

    logger.info(
        "codex door: model=%s lite=%s session=%s",
        body.model,
        is_lite,
        session_id,
    )

    # --- Extract system prompt and tool list (both dialects) ---
    input_items = [item.model_dump(exclude_unset=False) for item in body.input]
    system_prompt, tools = extract_dialect(
        instructions=body.instructions,
        tools=body.tools,
        input_items=input_items,
        is_lite=is_lite,
    )

    logger.debug(
        "codex door: prompt_len=%d tools=%d",
        len(system_prompt),
        len(tools),
    )

    # --- Response id for the whole turn ---
    response_id = f"resp_{uuid.uuid4().hex}"

    # --- Build event stream (Lane A/B stub; real integration noted below) ---
    # _STUB: replace _stub_kiro_event_stream with:
    #   1. Build Kiro payload via Lane B's builder, passing system_prompt and tools
    #   2. Send to Kiro via KiroHttpClient (from conduit.foundation.http_client)
    #   3. Parse the AWS event-stream response via Lane A's parser which yields
    #      KiroEvent / UnknownEvent objects (conduit.contracts.events)
    kiro_events = _stub_kiro_event_stream()

    # --- Translate to Responses SSE ---
    async def _sse_generator() -> AsyncGenerator[bytes, None]:
        try:
            async for frame in translate_kiro_to_responses_sse(
                kiro_events,
                model=body.model,
                response_id=response_id,
            ):
                yield frame.encode("utf-8")
        except Exception as exc:
            logger.error("codex door: SSE generation error: %s", exc, exc_info=True)

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Stub: synthetic event stream used until Lane A/B are ready
# ---------------------------------------------------------------------------

async def _stub_kiro_event_stream() -> AsyncGenerator[Any, None]:
    """Placeholder event stream.

    Yields a minimal valid sequence so the SSE layer can be exercised without
    a live Kiro connection.  Replace with the real Lane A parser in production.

    Emits:
        assistantResponseEvent  "Hello from CONDUIT."
        metadataEvent           with synthetic token counts
        messageStopEvent        END_TURN
    """
    yield KiroEvent(
        name=EventName.ASSISTANT_RESPONSE,
        payload={"content": "Hello from CONDUIT."},
    )
    yield KiroEvent(
        name=EventName.METADATA,
        payload={
            "tokenUsage": {
                "inputTokens": 10,
                "outputTokens": 5,
                "thinkingTokens": 0,
            }
        },
    )
    yield KiroEvent(
        name=EventName.CONTEXT_USAGE,
        payload={
            "cacheReadTokens": 0,
            "cacheWriteTokens": 0,
        },
    )
    yield KiroEvent(
        name=EventName.MESSAGE_STOP,
        payload={"stopReason": "END_TURN"},
    )
