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

LANE A/B/F INTEGRATION (live):
  The route runs the real chain, mirroring Lane C so there is one transport
  pattern in the codebase rather than two:
    - Lane B  conduit.kiro.request.build_conversation_state  builds the payload
              and returns the RewrittenOut declaring every injection it made
    - transport  conduit.foundation.http_client.KiroHttpClient
    - Lane A  conduit.kiro.dispatch.iter_events  parses the AWS event stream
    - Lane F  conduit.seam.emit.emit  receives the turn, fire-and-forget

  There is deliberately no stub fallback. An earlier revision fell back to a
  synthetic "Hello from CONDUIT." stream, which made the door look healthy in
  tests while it never contacted Kiro at all. A broken seam must fail loudly.

See SPEC-WORKSTREAMS.md, Lane D, for the full brief and verification gate.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from loguru import logger
from pydantic import BaseModel, Field

from conduit.config import PROFILE_ARN, PROXY_API_KEY
from conduit.contracts.metrics import TurnMetrics
from conduit.contracts.rewritten import RewrittenOut
from conduit.contracts.seam import SeamEnvelope
from conduit.doors.codex_dialect import LITE_HEADER, extract_dialect
from conduit.doors.codex_sse import translate_kiro_to_responses_sse
from conduit.foundation.auth import resolve_profile_arn
from conduit.foundation.http_client import KiroHttpClient
from conduit.kiro.dispatch import iter_events
from conduit.kiro.history import HistoryMessage
from conduit.kiro.request import build_conversation_state
from conduit.kiro.tools import build_tool_result
from conduit.seam.emit import emit

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

    # reasoning: {"effort": "low"|"medium"|"high", ...} when Codex asks for it
    reasoning: Optional[Dict[str, Any]] = None

    # Always true from Codex; kept for explicitness
    stream: bool = True
    store: bool = False
    tool_choice: Optional[Any] = "auto"


# ---------------------------------------------------------------------------
# Lane B adapter - Responses input[] -> Lane B's HistoryMessage pipeline
# ---------------------------------------------------------------------------

_KIRO_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


def _effort_from_reasoning(reasoning: Optional[Dict[str, Any]]) -> Optional[str]:
    """Map the Responses ``reasoning.effort`` field onto Kiro's effort enum.

    Unlike the Anthropic door - which receives a TOKEN BUDGET and has to bucket
    it heuristically - Codex sends an effort enum already, so this is a
    validated pass-through rather than a guess. Codex emits low/medium/high;
    Kiro also accepts xhigh/max, so those survive untouched if they appear.

    Returns None for an absent or unrecognised value, letting Lane B apply its
    own default instead of forcing a wrong one.
    """
    if not reasoning:
        return None
    effort = reasoning.get("effort")
    if not isinstance(effort, str):
        return None
    normalised = effort.strip().lower()
    if normalised in _KIRO_EFFORTS:
        return normalised
    logger.debug(
        f"codex door: unrecognised reasoning.effort {effort!r}; using Lane B default"
    )
    return None


def _text_of(content: Any) -> str:
    """Flatten a Responses content value (string or block list) to plain text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            # input_text on inbound turns, output_text on assistant turns
            if block.get("type") in ("text", "input_text", "output_text", None, ""):
                value = block.get("text") or block.get("value") or ""
                if isinstance(value, str):
                    parts.append(value)
    return "\n".join(part for part in parts if part)


def _images_of(content: Any) -> List[Dict[str, Any]]:
    """Extract base64 images from a Responses content list.

    Remote URLs are reported rather than silently dropped: Kiro cannot fetch
    them, so the model would simply never see the image.
    """
    if not isinstance(content, list):
        return []
    images: List[Dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "input_image":
            continue
        url = block.get("image_url") or ""
        if isinstance(url, dict):
            url = url.get("url", "")
        if isinstance(url, str) and url.startswith("data:"):
            # data:<media_type>;base64,<payload>
            try:
                header, data = url.split(",", 1)
            except ValueError:
                logger.warning("codex door: malformed data: image URI dropped")
                continue
            media_type = header[5:].split(";", 1)[0] or "image/png"
            images.append({"media_type": media_type, "data": data})
        elif url:
            logger.warning(
                "codex door: URL-based image dropped (Kiro cannot fetch remote "
                "images); the model is not shown this image"
            )
    return images


def _to_history_messages(input_items: List[Dict[str, Any]]) -> List[HistoryMessage]:
    """Convert Responses ``input[]`` items to Lane B's HistoryMessage list.

    One HistoryMessage per item. Lane B's ``merge_turns`` collapses consecutive
    same-role items and declares a single TURN_MERGE injection, so grouping is
    deliberately not re-implemented here.

    Per item type:
      - message (user / assistant): text, plus base64 images on user turns.
      - function_call: becomes a Kiro toolUse. ``arguments`` is a JSON *string*
        on this wire and is parsed into the object Kiro expects.
      - function_call_output: becomes a Kiro toolResult via Lane B's builder.
      - reasoning: dropped. Codex ``encrypted_content`` is provider-specific and
        must never be relayed, and Kiro has no slot for it.
      - role "developer": skipped. Those prefix items ARE the system prompt and
        tool list, already pulled out by ``extract_dialect``; including them
        here would duplicate the entire system prompt into the conversation.
    """
    out: List[HistoryMessage] = []

    for item in input_items:
        item_type = (item.get("type") or "message").strip()
        role = (item.get("role") or "").strip()

        if role == "developer" or item_type == "reasoning":
            continue

        if item_type == "function_call":
            raw_args = item.get("arguments")
            if isinstance(raw_args, dict):
                parsed_args: Dict[str, Any] = raw_args
            elif isinstance(raw_args, str) and raw_args.strip():
                try:
                    decoded = json.loads(raw_args)
                    parsed_args = decoded if isinstance(decoded, dict) else {}
                except json.JSONDecodeError:
                    logger.warning(
                        "codex door: function_call.arguments was not valid JSON; "
                        "forwarding an empty input object"
                    )
                    parsed_args = {}
            else:
                parsed_args = {}

            out.append(HistoryMessage(
                role="assistant",
                content="",
                tool_calls=[{
                    "toolUseId": item.get("call_id") or item.get("id") or "",
                    "name": item.get("name", ""),
                    "input": parsed_args,
                }],
            ))
            continue

        if item_type == "function_call_output":
            # Codex's FunctionCallOutput carries NO item-level success/error
            # flag (verified in Codex source), so a truthful status cannot be
            # read off this wire. Lane B's documented fallback is
            # status="success"; a real value has to come from CONDUIT's own tool
            # runner, which does not exist yet. Known, recorded limitation.
            out.append(HistoryMessage(
                role="user",
                content="",
                tool_results=[build_tool_result(
                    tool_use_id=item.get("call_id") or "",
                    content=item.get("output"),
                )],
            ))
            continue

        content = item.get("content")
        text = _text_of(content)
        images = _images_of(content) if role != "assistant" else []
        if not text and not images:
            continue
        out.append(HistoryMessage(
            role="assistant" if role == "assistant" else "user",
            content=text,
            images=images or None,
        ))

    return out


def _build_kiro_payload(
    body: "ResponsesRequest",
    system_prompt: str,
    tools: List[Dict[str, Any]],
    conversation_id: str,
    profile_arn: Optional[str],
) -> Tuple[dict, RewrittenOut]:
    """Build the Kiro payload via Lane B, returning it with its RewrittenOut.

    The Lane D -> Lane B seam. Calls Lane B directly and does not swallow
    failures: a broken seam must surface as an error rather than degrade to a
    silently-wrong payload.
    """
    history_messages = _to_history_messages(
        [item.model_dump(exclude_unset=False) for item in body.input]
    )
    if not history_messages:
        raise ValueError("input[] contained no conversational content")

    return build_conversation_state(
        messages=history_messages,
        system_prompt=system_prompt or None,
        model_id=body.model,
        tools=tools or None,
        conversation_id=conversation_id,
        profile_arn=profile_arn,
        reasoning_effort=_effort_from_reasoning(body.reasoning),
    )


def _error_response(
    status: int, message: str, err_type: str = "api_error"
) -> JSONResponse:
    """OpenAI-shaped error body, for failures that occur before streaming."""
    return JSONResponse(
        status_code=status,
        content={"error": {"type": err_type, "message": message}},
    )


def _fire_seam_emit(envelope: SeamEnvelope) -> None:
    """Hand the turn to Lane F's seam emitter.

    ``emit()`` is itself fire-and-forget: it enqueues and returns, never blocks,
    never raises. So this is a direct call - wrapping it in a broad ``except``
    would only hide a genuinely broken seam.
    """
    emit(envelope)


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
        f"codex door: model={body.model!r} lite={is_lite} session={session_id!r}"
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
        f"codex door: prompt_len={len(system_prompt)} tools={len(tools)}"
    )

    # --- Ids for the whole turn ---
    response_id = f"resp_{uuid.uuid4().hex}"
    request_id = request.headers.get("x-saga-request-id") or f"req_{uuid.uuid4().hex[:16]}"

    # ------------------------------------------------------------------
    # Resolve account / auth (same shape as Lane C)
    # ------------------------------------------------------------------
    account_manager = getattr(request.app.state, "account_manager", None)
    if account_manager is None:
        logger.error("codex door: no account_manager on app.state")
        return _error_response(503, "Account manager not initialised")

    if getattr(request.app.state, "account_system", False):
        account = await account_manager.get_next_account(body.model)
        if account is None:
            return _error_response(503, "No available accounts")
    else:
        # Legacy single-account mode, and ACCOUNT_SYSTEM defaults to false, so
        # this is the DEFAULT path. get_first_account() does not lazily
        # initialize (unlike get_next_account) — it raises RuntimeError until an
        # account has been initialized. No credentials configured is an expected
        # operational state, so it becomes a 503, not an uncaught 500.
        try:
            account = account_manager.get_first_account()
        except RuntimeError as exc:
            logger.error(f"codex door: no initialized account: {exc}")
            return _error_response(503, str(exc))

    auth_manager = getattr(account, "auth_manager", None) if account else None
    if auth_manager is None:
        return _error_response(503, "Account not initialized")

    # ------------------------------------------------------------------
    # Build the Kiro payload via Lane B
    # ------------------------------------------------------------------
    conversation_id = str(uuid.uuid4())
    profile_arn = await resolve_profile_arn(auth_manager, PROFILE_ARN)

    try:
        kiro_payload, rewritten_out = _build_kiro_payload(
            body, system_prompt, tools, conversation_id, profile_arn
        )
    except (ValueError, TypeError) as exc:
        logger.error(f"codex door: payload build error: {exc}")
        return _error_response(400, str(exc), err_type="invalid_request_error")

    # ------------------------------------------------------------------
    # Send to Kiro
    # ------------------------------------------------------------------
    url = f"{auth_manager.api_host}/generateAssistantResponse"
    http_client = KiroHttpClient(auth_manager, shared_client=None)

    try:
        kiro_response = await http_client.request_with_retry(
            "POST", url, kiro_payload, stream=True
        )
    except HTTPException:
        await http_client.close()
        raise
    except Exception as exc:
        await http_client.close()
        logger.error(f"codex door: upstream request failed: {exc}", exc_info=True)
        return _error_response(502, str(exc))

    if kiro_response.status_code != 200:
        try:
            err_body = await kiro_response.aread()
        except Exception:
            err_body = b"(unreadable)"
        await http_client.close()
        err_text = err_body.decode("utf-8", errors="replace")
        logger.warning(
            f"codex door: Kiro returned {kiro_response.status_code}: {err_text[:200]}"
        )
        return _error_response(kiro_response.status_code, err_text)

    if getattr(request.app.state, "account_system", False):
        await account_manager.report_success(account.id, body.model)

    # ------------------------------------------------------------------
    # Translate Kiro's event stream to Responses SSE
    #
    # Codex hardcodes stream:true client-side (verified in Codex source), so
    # this door is streaming-only by contract; there is no non-streaming path
    # to fall back to.
    # ------------------------------------------------------------------
    async def _sse_generator() -> AsyncGenerator[bytes, None]:
        metrics = TurnMetrics()
        streaming_error: Optional[Exception] = None
        client_disconnected = False

        try:
            async for frame in translate_kiro_to_responses_sse(
                iter_events(kiro_response),
                model=body.model,
                response_id=response_id,
                metrics=metrics,
            ):
                yield frame.encode("utf-8")
        except GeneratorExit:
            client_disconnected = True
            logger.debug("codex door: client disconnected during streaming")
        except Exception as exc:
            streaming_error = exc
            logger.error(f"codex door: SSE generation error: {exc}", exc_info=True)
        finally:
            await http_client.close()
            if streaming_error is not None:
                logger.error(f"codex door: turn failed - {streaming_error}")
            elif client_disconnected:
                logger.info("codex door: client disconnected")
            else:
                logger.info(
                    f"codex door: turn complete in={metrics.input_tokens} "
                    f"out={metrics.output_tokens} thought={metrics.thought_tokens} "
                    f"total={metrics.total_tokens}"
                )
                _fire_seam_emit(SeamEnvelope(
                    request_id=request_id,
                    ts=int(time.time() * 1000),
                    model_id=body.model,
                    metrics=metrics,
                    rewritten_out=rewritten_out,
                ))

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
