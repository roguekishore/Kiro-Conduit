"""Header-driven dispatch: `:event-type` → typed event.

Switches on the `:event-type` frame header to emit one of the typed events
declared in ``conduit.contracts.events``.  Payload-shape fallback is used
**only** when the header is absent; unknown event names are carried through as
:class:`~conduit.contracts.events.UnknownEvent` and logged at DEBUG.

Never guesses the event type from payload shape — that is the upstream defect
being replaced (``_reference_upstream/parsers.py`` substring-matched 7 patterns
that resolved to only 4 of the 17+ known event names).
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, AsyncIterator, Dict

from loguru import logger

from conduit.contracts.events import DispatchedEvent, EventName, KiroEvent, UnknownEvent
from conduit.kiro.frame import FrameDecoder, FrameError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EVENT_NAME_SET: frozenset[str] = frozenset(e.value for e in EventName)


def _decode_payload(payload: bytes) -> Dict[str, Any]:
    """Decode a JSON payload, returning an empty dict on failure."""
    if not payload:
        return {}
    try:
        result = json.loads(payload)
        if isinstance(result, dict):
            return result
        # Scalar / list payloads — wrap so callers always get a dict
        return {"_raw": result}
    except json.JSONDecodeError as exc:
        logger.debug(f"dispatch: payload JSON decode failed ({exc}); carrying raw bytes")
        return {}


def _infer_from_payload(payload_dict: Dict[str, Any]) -> str | None:
    """Last-resort event-name inference from payload keys.

    Used **only** when the `:event-type` header is absent.  Returns the first
    matching EventName value, or ``None`` if no match is found.

    This is intentionally conservative: it only checks for unambiguous top-level
    keys that are unique to a single event type.
    """
    # Mapping of a distinctive top-level payload key → EventName value
    _KEY_HINTS: Dict[str, str] = {
        "assistantResponseEvent": EventName.ASSISTANT_RESPONSE.value,
        "reasoningContentEvent": EventName.REASONING_CONTENT.value,
        "toolUseEvent": EventName.TOOL_USE.value,
        "toolResultEvent": EventName.TOOL_RESULT.value,
        "meteringEvent": EventName.METERING.value,
        "metadataEvent": EventName.METADATA.value,
        "messageMetadataEvent": EventName.MESSAGE_METADATA.value,
        "contextUsageEvent": EventName.CONTEXT_USAGE.value,
        "codeEvent": EventName.CODE.value,
        "codeReferenceEvent": EventName.CODE_REFERENCE.value,
        "followupPromptEvent": EventName.FOLLOWUP_PROMPT.value,
        "intentsEvent": EventName.INTENTS.value,
        "invalidStateEvent": EventName.INVALID_STATE.value,
        "supplementaryWebLinksEvent": EventName.SUPPLEMENTARY_LINKS.value,
        "citationEvent": EventName.CITATION.value,
        "documentCitationEvent": EventName.DOCUMENT_CITATION.value,
        "messageStopEvent": EventName.MESSAGE_STOP.value,
    }
    for key, name in _KEY_HINTS.items():
        if key in payload_dict:
            return name
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def iter_events_from_chunks(
    chunks: AsyncIterator[bytes],
) -> AsyncGenerator[DispatchedEvent, None]:
    """Decode an async byte stream into typed events, in order.

    This is the streaming counterpart to :func:`dispatch_event`: it owns the
    :class:`~conduit.kiro.frame.FrameDecoder` buffer so frames split across
    HTTP chunk boundaries are reassembled before dispatch.

    Takes a bare byte iterator rather than an HTTP response so it can be
    tested without a transport. See :func:`iter_events` for the response-shaped
    wrapper the doors call.

    Args:
        chunks: Async iterator of raw body bytes.

    Yields:
        One :class:`~conduit.contracts.events.KiroEvent` or
        :class:`~conduit.contracts.events.UnknownEvent` per decoded frame.

    Raises:
        FrameError: On a CRC mismatch or structurally invalid frame. Propagated
            deliberately — a corrupt stream is reported, never silently
            truncated into a short-but-plausible response.
    """
    decoder = FrameDecoder()
    frame_count = 0

    async for chunk in chunks:
        try:
            frames = decoder.feed(chunk)
        except FrameError:
            logger.error(
                f"dispatch: frame decode failed after {frame_count} good frame(s); "
                f"stream is corrupt"
            )
            raise
        for msg in frames:
            frame_count += 1
            yield dispatch_event(msg["headers"], msg["payload"])

    logger.debug(f"dispatch: stream ended after {frame_count} frame(s)")


async def iter_events(response: Any) -> AsyncGenerator[DispatchedEvent, None]:
    """Decode a streaming Kiro HTTP response into typed events.

    The door-facing entry point for Lane A. Wraps
    :func:`iter_events_from_chunks` around ``response.aiter_bytes()``.

    Args:
        response: Any object exposing ``aiter_bytes()`` — in production an
            ``httpx.Response`` opened with ``stream=True``.

    Yields:
        One typed event per decoded frame.
    """
    async for event in iter_events_from_chunks(response.aiter_bytes()):
        yield event


def dispatch_event(headers: Dict[str, str], payload: bytes) -> DispatchedEvent:
    """Decode *payload* and route it to the appropriate typed event.

    Args:
        headers: All decoded frame headers (``str`` → ``str``).  The
            ``:event-type`` header drives dispatch.
        payload: Raw payload bytes from the frame (typically UTF-8 JSON).

    Returns:
        A :class:`~conduit.contracts.events.KiroEvent` for every known
        ``EventName``, or an :class:`~conduit.contracts.events.UnknownEvent`
        for anything else.  Unknown events are logged at ``DEBUG`` and carried
        through — never silently dropped.
    """
    payload_dict = _decode_payload(payload)

    # --- Primary: header-driven dispatch ------------------------------------
    event_type: str | None = headers.get(":event-type")

    if event_type is None:
        # Secondary: payload-shape fallback (last resort, header absent)
        event_type = _infer_from_payload(payload_dict)
        if event_type is not None:
            logger.debug(
                f"dispatch: :event-type header absent; inferred '{event_type}' "
                f"from payload key (fallback path)"
            )

    if event_type is None:
        logger.debug(
            f"dispatch: :event-type absent and payload inference failed; "
            f"carrying as UnknownEvent(name='')"
        )
        return UnknownEvent(name="", payload=payload_dict)

    # --- Route known event names -------------------------------------------
    if event_type in _EVENT_NAME_SET:
        return KiroEvent(name=EventName(event_type), payload=payload_dict)

    # --- Unknown event name: carry through, log at DEBUG -------------------
    logger.debug(
        f"dispatch: unknown :event-type '{event_type}'; "
        f"carrying as UnknownEvent (expected for protocol extensions)"
    )
    return UnknownEvent(name=event_type, payload=payload_dict)
