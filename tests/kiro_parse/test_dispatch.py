"""Tests for conduit.kiro.dispatch — header-driven event routing.

All 17 EventName values are covered.  Unknown event names become UnknownEvent.
No guessing from payload shape when a header is present.
"""

from __future__ import annotations

import json

import pytest

from conduit.contracts.events import EventName, KiroEvent, UnknownEvent
from conduit.kiro.dispatch import dispatch_event

from .conftest import json_payload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _headers(event_type: str) -> dict:
    return {":event-type": event_type}


def _dispatch(event_type: str, payload_obj=None) -> KiroEvent | UnknownEvent:
    payload = json_payload(payload_obj or {})
    return dispatch_event(_headers(event_type), payload)


# ---------------------------------------------------------------------------
# One test per EventName — all 17 members
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event_name",
    [
        EventName.ASSISTANT_RESPONSE,
        EventName.REASONING_CONTENT,
        EventName.TOOL_USE,
        EventName.TOOL_RESULT,
        EventName.METERING,
        EventName.METADATA,
        EventName.MESSAGE_METADATA,
        EventName.CONTEXT_USAGE,
        EventName.CODE,
        EventName.CODE_REFERENCE,
        EventName.FOLLOWUP_PROMPT,
        EventName.INTENTS,
        EventName.INVALID_STATE,
        EventName.SUPPLEMENTARY_LINKS,
        EventName.CITATION,
        EventName.DOCUMENT_CITATION,
        EventName.MESSAGE_STOP,
    ],
)
def test_known_event_type_becomes_kiro_event(event_name: EventName) -> None:
    """Every known EventName dispatches to KiroEvent with the correct name."""
    result = _dispatch(event_name.value, {"data": "test"})
    assert isinstance(result, KiroEvent)
    assert result.name == event_name
    assert result.payload == {"data": "test"}


# ---------------------------------------------------------------------------
# Unknown event type → UnknownEvent
# ---------------------------------------------------------------------------


def test_unknown_event_type_becomes_unknown_event() -> None:
    """An unrecognised :event-type is carried as UnknownEvent, not silently dropped."""
    result = _dispatch("someFutureKiroEvent", {"x": 1})
    assert isinstance(result, UnknownEvent)
    assert result.name == "someFutureKiroEvent"
    assert result.payload == {"x": 1}


def test_another_unknown_event_carries_payload() -> None:
    """A second unknown-type case verifies name AND payload are both preserved."""
    payload = {"nested": {"deep": True}}
    result = dispatch_event(
        {":event-type": "experimentalDraftEvent"},
        json_payload(payload),
    )
    assert isinstance(result, UnknownEvent)
    assert result.name == "experimentalDraftEvent"
    assert result.payload == payload


# ---------------------------------------------------------------------------
# Payload is preserved verbatim for known events
# ---------------------------------------------------------------------------


def test_payload_dict_preserved_on_known_event() -> None:
    """The decoded payload dict is attached to the KiroEvent unchanged."""
    obj = {"content": "hello world", "tokens": 42}
    result = dispatch_event(_headers("assistantResponseEvent"), json_payload(obj))
    assert isinstance(result, KiroEvent)
    assert result.payload == obj


# ---------------------------------------------------------------------------
# Empty payload decodes cleanly
# ---------------------------------------------------------------------------


def test_empty_payload_gives_empty_dict() -> None:
    """An empty payload yields an empty payload dict, not an error."""
    result = dispatch_event(_headers("messageStopEvent"), b"")
    assert isinstance(result, KiroEvent)
    assert result.payload == {}


# ---------------------------------------------------------------------------
# Header absent → payload-shape fallback
# ---------------------------------------------------------------------------


def test_no_event_type_header_falls_back_to_payload_inference() -> None:
    """When :event-type is absent, inference from payload keys is used."""
    payload = json_payload({"assistantResponseEvent": {"content": "hi"}})
    result = dispatch_event({}, payload)
    assert isinstance(result, KiroEvent)
    assert result.name == EventName.ASSISTANT_RESPONSE


def test_no_event_type_header_no_match_is_unknown() -> None:
    """When :event-type is absent and inference fails, an empty-name UnknownEvent is returned."""
    payload = json_payload({"someRandomKey": "value"})
    result = dispatch_event({}, payload)
    assert isinstance(result, UnknownEvent)
    assert result.name == ""


# ---------------------------------------------------------------------------
# Header wins over payload shape (the key anti-regression)
# ---------------------------------------------------------------------------


def test_header_takes_precedence_over_payload_shape() -> None:
    """The :event-type header always wins; the payload is never inspected for routing."""
    # Payload looks like a metadataEvent but the header says something else
    payload = json_payload({"metadataEvent": {"tokenUsage": {"inputTokens": 100}}})
    result = dispatch_event(_headers("messageStopEvent"), payload)
    assert isinstance(result, KiroEvent)
    assert result.name == EventName.MESSAGE_STOP


# ---------------------------------------------------------------------------
# Invalid JSON payload does not raise; carries empty dict
# ---------------------------------------------------------------------------


def test_invalid_json_payload_carries_empty_dict() -> None:
    """Malformed payload bytes don't crash dispatch; the event is still routed."""
    result = dispatch_event(_headers("assistantResponseEvent"), b"not json {{{{")
    assert isinstance(result, KiroEvent)
    assert result.name == EventName.ASSISTANT_RESPONSE
    assert result.payload == {}
