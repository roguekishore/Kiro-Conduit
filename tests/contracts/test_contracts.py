"""Lane 0 gate: the frozen contracts import and round-trip.

These tests exist to catch a lane accidentally redefining or reshaping a
shared type. They assert structure and wire-field names, not behaviour.
"""

from dataclasses import asdict

import pytest

from conduit.contracts import (
    DispatchedEvent,
    EventName,
    Injection,
    InjectionType,
    KiroEvent,
    ReasoningBlockMeta,
    RewrittenOut,
    SeamEnvelope,
    StopReason,
    TurnMetrics,
    UnknownEvent,
)


# --- events ---------------------------------------------------------------

def test_event_names_are_wire_verbatim():
    """Names must match Amazon's `:event-type` header values exactly."""
    assert EventName.ASSISTANT_RESPONSE == "assistantResponseEvent"
    assert EventName.REASONING_CONTENT == "reasoningContentEvent"
    assert EventName.METADATA == "metadataEvent"
    assert EventName.METERING == "meteringEvent"
    assert EventName.CONTEXT_USAGE == "contextUsageEvent"
    assert EventName.MESSAGE_STOP == "messageStopEvent"


def test_event_union_covers_the_known_protocol():
    """~17 assistant-path events. Upstream saw 4; regression guard."""
    assert len(EventName) == 17
    # every value is lowerCamel, as the header carries it
    for name in EventName:
        assert name.value[0].islower(), name
        assert name.value.endswith("Event"), name


def test_unknown_event_is_carried_not_dropped():
    """An unrecognized :event-type must survive as data."""
    ev = UnknownEvent(name="brandNewEvent", payload={"a": 1})
    assert ev.name == "brandNewEvent"
    assert ev.payload == {"a": 1}


def test_kiro_event_defaults_to_empty_payload():
    ev = KiroEvent(name=EventName.MESSAGE_STOP)
    assert ev.payload == {}


# --- metrics --------------------------------------------------------------

def test_turn_metrics_has_all_six_measured_counters():
    """Kiro's Usage struct carries six counters; all must be representable."""
    m = TurnMetrics()
    for field_name in (
        "input_tokens",
        "output_tokens",
        "thought_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "total_tokens",
    ):
        assert hasattr(m, field_name), field_name
        assert getattr(m, field_name) == 0


def test_turn_metrics_defaults_are_zero_not_estimated():
    m = TurnMetrics()
    assert m.credits == 0.0
    assert m.context_usage_percentage == 0.0
    assert m.stop_reason is None
    assert m.reasoning_blocks == []


def test_stop_reason_includes_daily_limit():
    """A DAILY limit exists, not only the monthly one."""
    assert StopReason.DAILY_REQUEST_COUNT == "DAILY_REQUEST_COUNT"
    assert StopReason.MONTHLY_REQUEST_COUNT == "MONTHLY_REQUEST_COUNT"


def test_stop_reason_distinguishes_catchable_states():
    assert StopReason.PAUSE_TURN == "PAUSE_TURN"
    assert StopReason.MODEL_CONTEXT_WINDOW_EXCEEDED == "MODEL_CONTEXT_WINDOW_EXCEEDED"


def test_reasoning_block_signature_defaults_absent():
    """Signatures are never fabricated: absent until a real one arrives."""
    b = ReasoningBlockMeta()
    assert b.signature_present is False
    assert b.model_id is None


def test_turn_metrics_round_trips():
    m = TurnMetrics(
        input_tokens=100,
        output_tokens=50,
        thought_tokens=25,
        cache_read_tokens=10,
        cache_write_tokens=5,
        total_tokens=190,
        credits=1.5,
        context_usage_percentage=12.5,
        stop_reason=StopReason.END_TURN,
        reasoning_blocks=[ReasoningBlockMeta(model_id="claude-opus-4.7", signature_present=True)],
    )
    d = asdict(m)
    assert d["thought_tokens"] == 25
    assert d["stop_reason"] == "END_TURN"
    assert d["reasoning_blocks"][0]["signature_present"] is True


# --- rewritten / injections ----------------------------------------------

def test_injection_types_match_the_seam_contract():
    """These four strings are the seam contract's vocabulary."""
    assert InjectionType.THINKING_TAGS == "thinking_tags"
    assert InjectionType.SYSTEM_PROMPT_SLOT == "system_prompt_slot"
    assert InjectionType.TURN_MERGE == "turn_merge"
    assert InjectionType.RELOCATED_TOOL_DOCS == "relocated_tool_docs"


def test_rewritten_out_round_trips():
    r = RewrittenOut(
        system_prompt="you are a careful assistant",
        current_message={"role": "user", "content": "hi"},
        history=[{"role": "user", "content": "prior"}],
        tools=[{"toolSpecification": {"name": "read"}}],
        injections=[Injection(type=InjectionType.SYSTEM_PROMPT_SLOT, location="systemPrompt")],
    )
    d = asdict(r)
    assert d["system_prompt"] == "you are a careful assistant"
    assert d["injections"][0]["type"] == "system_prompt_slot"
    assert d["injections"][0]["location"] == "systemPrompt"


def test_rewritten_out_defaults_are_empty():
    r = RewrittenOut()
    assert r.system_prompt is None
    assert r.current_message == {}
    assert r.history == []
    assert r.tools == []
    assert r.injections == []


# --- seam envelope --------------------------------------------------------

def test_seam_envelope_matches_contract_v1_keys():
    """Top-level keys must match CONTRACT-conduit-saga-seam.md v1 exactly."""
    env = SeamEnvelope(
        request_id="req_abc",
        ts=1234567890,
        model_id="claude-opus-4.7",
        metrics=TurnMetrics(),
        rewritten_out=RewrittenOut(),
    )
    d = asdict(env)
    assert set(d) == {"request_id", "ts", "model_id", "metrics", "rewritten_out"}


def test_seam_envelope_nests_metrics_and_rewritten():
    env = SeamEnvelope(
        request_id="req_abc",
        ts=1,
        model_id="claude-sonnet-4.6",
        metrics=TurnMetrics(input_tokens=7, thought_tokens=3),
        rewritten_out=RewrittenOut(system_prompt="s"),
    )
    d = asdict(env)
    assert d["metrics"]["input_tokens"] == 7
    assert d["metrics"]["thought_tokens"] == 3
    assert d["rewritten_out"]["system_prompt"] == "s"


def test_dispatched_event_union_accepts_both_members():
    known: DispatchedEvent = KiroEvent(name=EventName.CODE)
    unknown: DispatchedEvent = UnknownEvent(name="mysteryEvent")
    assert isinstance(known, KiroEvent)
    assert isinstance(unknown, UnknownEvent)
