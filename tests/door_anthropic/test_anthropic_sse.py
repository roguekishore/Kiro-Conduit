"""Tests for conduit.doors.anthropic_sse — Lane C SSE emitter.

All tests use synthetic Lane A event sequences (no network, no real Kiro calls).

Verification gate:
  - Well-formed Anthropic SSE sequence (correct event types in order)
  - Thinking blocks carry the upstream signature VERBATIM — nothing generated
  - Usage numbers equal the fixture's metadataEvent exactly
  - A tool-use turn round-trips (tool_use block with correct id/name/input)
  - cache_read_tokens and cache_write_tokens present in usage block
"""

from __future__ import annotations

import json
from typing import List, Tuple

import pytest

from conduit.contracts.events import EventName, KiroEvent, UnknownEvent
from conduit.contracts.metrics import StopReason, TurnMetrics
from tests.door_anthropic.conftest import (
    collect_sse,
    make_text_events,
    make_thinking_events,
    make_tool_use_events,
    parse_sse_stream,
)


# ===========================================================================
# Helper: extract event types in order
# ===========================================================================

def event_types(events: List[Tuple[str, dict]]) -> List[str]:
    return [et for et, _ in events]


def events_of_type(events: List[Tuple[str, dict]], name: str) -> List[dict]:
    return [data for et, data in events if et == name]


# ===========================================================================
# 1. Well-formed SSE sequence — simple text turn
# ===========================================================================

@pytest.mark.asyncio
async def test_text_turn_event_order():
    """message_start → ping → content_block_start → delta(s) → stop → message_delta → message_stop."""
    evts, _ = await collect_sse(make_text_events(["Hello", ", world!"]))
    types = event_types(evts)

    assert types[0] == "message_start"
    assert types[1] == "ping"
    # content_block_start for text block must come before any delta
    cbs_idx = types.index("content_block_start")
    cbd_idx = types.index("content_block_delta")
    cbstop_idx = types.index("content_block_stop")
    md_idx   = types.index("message_delta")
    ms_idx   = types.index("message_stop")

    assert cbs_idx < cbd_idx < cbstop_idx < md_idx < ms_idx, (
        f"Order violated: {types}"
    )
    # message_stop must be the last event
    assert ms_idx == len(types) - 1


@pytest.mark.asyncio
async def test_text_turn_content_block_start_type():
    evts, _ = await collect_sse(make_text_events(["hi"]))
    cbs_events = events_of_type(evts, "content_block_start")
    # Only one block for a plain text turn
    assert len(cbs_events) == 1
    assert cbs_events[0]["content_block"]["type"] == "text"
    assert cbs_events[0]["index"] == 0


@pytest.mark.asyncio
async def test_text_deltas_carry_text_delta_type():
    evts, _ = await collect_sse(make_text_events(["Hello", " world"]))
    deltas = events_of_type(evts, "content_block_delta")
    for d in deltas:
        assert d["delta"]["type"] == "text_delta"
        assert "text" in d["delta"]


@pytest.mark.asyncio
async def test_text_deltas_text_matches_input():
    chunks = ["Hello", ", world!"]
    evts, _ = await collect_sse(make_text_events(chunks))
    deltas = events_of_type(evts, "content_block_delta")
    combined = "".join(d["delta"]["text"] for d in deltas)
    assert combined == "".join(chunks)


# ===========================================================================
# 2. Usage numbers from metadataEvent
# ===========================================================================

@pytest.mark.asyncio
async def test_usage_equals_metadata_event():
    """message_delta usage must reflect the metadataEvent fixture exactly."""
    evts, metrics = await collect_sse(
        make_text_events(["text"], stop_reason="END_TURN")
    )
    md_events = events_of_type(evts, "message_delta")
    assert len(md_events) == 1
    usage = md_events[0]["usage"]

    # From make_text_events fixture: input=100, output=50, cr=10, cw=5
    assert usage["output_tokens"] == 50
    assert metrics.input_tokens  == 100
    assert metrics.output_tokens == 50


@pytest.mark.asyncio
async def test_cache_read_tokens_present_in_usage():
    """cache_read_input_tokens must be present in the message_delta usage block."""
    evts, _ = await collect_sse(make_text_events(["hi"]))
    usage = events_of_type(evts, "message_delta")[0]["usage"]
    assert "cache_read_input_tokens" in usage
    assert usage["cache_read_input_tokens"] == 10  # fixture value


@pytest.mark.asyncio
async def test_cache_write_tokens_present_in_usage():
    """cache_creation_input_tokens must be present in the message_delta usage block."""
    evts, _ = await collect_sse(make_text_events(["hi"]))
    usage = events_of_type(evts, "message_delta")[0]["usage"]
    assert "cache_creation_input_tokens" in usage
    assert usage["cache_creation_input_tokens"] == 5  # fixture value


@pytest.mark.asyncio
async def test_metrics_object_populated_from_metadata():
    """TurnMetrics is mutated as metadataEvent arrives — callers can read after iteration."""
    _, metrics = await collect_sse(make_text_events(["data"]))
    assert metrics.input_tokens        == 100
    assert metrics.output_tokens       == 50
    assert metrics.cache_read_tokens   == 10
    assert metrics.cache_write_tokens  == 5
    assert metrics.total_tokens        == 150


# ===========================================================================
# 3. Thinking blocks — verbatim signature forwarding
# ===========================================================================

VERBATIM_SIGNATURE = "kiro_real_sig_abc123xyz"


@pytest.mark.asyncio
async def test_thinking_block_start_type():
    evts, _ = await collect_sse(
        make_thinking_events(["Let me think..."], VERBATIM_SIGNATURE, ["Answer."])
    )
    cbs_events = events_of_type(evts, "content_block_start")
    thinking_blocks = [e for e in cbs_events if e["content_block"]["type"] == "thinking"]
    assert len(thinking_blocks) == 1
    assert thinking_blocks[0]["index"] == 0  # thinking block is first


@pytest.mark.asyncio
async def test_thinking_deltas_carry_thinking_delta_type():
    evts, _ = await collect_sse(
        make_thinking_events(["Let me think..."], VERBATIM_SIGNATURE, ["Answer."])
    )
    deltas = events_of_type(evts, "content_block_delta")
    thinking_deltas = [d for d in deltas if d["delta"]["type"] == "thinking_delta"]
    assert len(thinking_deltas) >= 1
    assert all("thinking" in d["delta"] for d in thinking_deltas)


@pytest.mark.asyncio
async def test_thinking_signature_forwarded_verbatim():
    """The signature from reasoningContentEvent is forwarded as-is — nothing generated."""
    evts, _ = await collect_sse(
        make_thinking_events(["thought"], VERBATIM_SIGNATURE, ["answer"])
    )
    deltas = events_of_type(evts, "content_block_delta")
    sig_deltas = [d for d in deltas if d["delta"]["type"] == "signature_delta"]
    assert len(sig_deltas) == 1
    assert sig_deltas[0]["delta"]["signature"] == VERBATIM_SIGNATURE


@pytest.mark.asyncio
async def test_no_generated_signature():
    """No synthetic/generated signature appears — only the real one from the event."""
    fake_sigs_seen = []
    evts, _ = await collect_sse(
        make_thinking_events(["thought"], VERBATIM_SIGNATURE, ["answer"])
    )
    deltas = events_of_type(evts, "content_block_delta")
    for d in deltas:
        if d["delta"]["type"] == "signature_delta":
            sig = d["delta"]["signature"]
            # The real signature is forwarded verbatim; it must not look like uuid-generated
            assert sig == VERBATIM_SIGNATURE, (
                f"Expected verbatim sig {VERBATIM_SIGNATURE!r}, got {sig!r}"
            )


@pytest.mark.asyncio
async def test_thinking_before_text_block():
    """Thinking content_block_start (index 0) comes before text content_block_start (index 1)."""
    evts, _ = await collect_sse(
        make_thinking_events(["reason"], VERBATIM_SIGNATURE, ["answer"])
    )
    cbs_events = events_of_type(evts, "content_block_start")
    assert len(cbs_events) == 2
    assert cbs_events[0]["content_block"]["type"] == "thinking"
    assert cbs_events[1]["content_block"]["type"] == "text"
    assert cbs_events[0]["index"] < cbs_events[1]["index"]


@pytest.mark.asyncio
async def test_thinking_block_closed_before_text():
    """The thinking content_block_stop must appear before the text content_block_start."""
    evts, _ = await collect_sse(
        make_thinking_events(["thought"], VERBATIM_SIGNATURE, ["answer"])
    )
    types = event_types(evts)
    # Find the content_block_stop for index 0 (thinking) and cbs for index 1 (text)
    # Simplify: check positional order of first stop vs second start
    starts = [(i, et) for i, (et, _) in enumerate(evts) if et == "content_block_start"]
    stops  = [(i, et) for i, (et, _) in enumerate(evts) if et == "content_block_stop"]
    assert starts[0][0] < stops[0][0] < starts[1][0], (
        "thinking must be closed before text opens"
    )


@pytest.mark.asyncio
async def test_thinking_without_signature_no_signature_delta():
    """If no signature arrives, no signature_delta is emitted."""
    async def _no_sig_events():
        yield KiroEvent(EventName.REASONING_CONTENT, {"content": "some thought"})
        yield KiroEvent(EventName.ASSISTANT_RESPONSE, {"content": "the answer"})
        yield KiroEvent(EventName.METADATA, {
            "tokenUsage": {"inputTokens": 10, "outputTokens": 5,
                           "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0}
        })
        yield KiroEvent(EventName.MESSAGE_STOP, {"stopReason": "END_TURN"})

    evts, _ = await collect_sse(_no_sig_events())
    deltas = events_of_type(evts, "content_block_delta")
    sig_deltas = [d for d in deltas if d["delta"]["type"] == "signature_delta"]
    assert sig_deltas == [], "no signature_delta should be emitted when none arrived"


@pytest.mark.asyncio
async def test_thinking_usage_from_metadata():
    """Usage in thinking turn comes from metadataEvent, not estimated."""
    evts, metrics = await collect_sse(
        make_thinking_events(["think"], VERBATIM_SIGNATURE, ["text"])
    )
    # Fixture: input=200, output=80, cr=20, cw=15
    usage = events_of_type(evts, "message_delta")[0]["usage"]
    assert usage["output_tokens"] == 80
    assert usage["cache_read_input_tokens"] == 20
    assert usage["cache_creation_input_tokens"] == 15
    assert metrics.input_tokens  == 200
    assert metrics.output_tokens == 80


# ===========================================================================
# 4. Tool-use turn round-trip
# ===========================================================================

TOOL_ID    = "toolu_abc123def456789012345678"
TOOL_NAME  = "read_file"
TOOL_INPUT = {"path": "/etc/hosts", "max_lines": 100}


@pytest.mark.asyncio
async def test_tool_use_block_start_type():
    evts, _ = await collect_sse(
        make_tool_use_events(TOOL_ID, TOOL_NAME, TOOL_INPUT)
    )
    cbs_events = events_of_type(evts, "content_block_start")
    assert len(cbs_events) == 1
    cb = cbs_events[0]["content_block"]
    assert cb["type"] == "tool_use"


@pytest.mark.asyncio
async def test_tool_use_block_id_round_trips():
    evts, _ = await collect_sse(
        make_tool_use_events(TOOL_ID, TOOL_NAME, TOOL_INPUT)
    )
    cbs_events = events_of_type(evts, "content_block_start")
    cb = cbs_events[0]["content_block"]
    assert cb["id"] == TOOL_ID


@pytest.mark.asyncio
async def test_tool_use_block_name_round_trips():
    evts, _ = await collect_sse(
        make_tool_use_events(TOOL_ID, TOOL_NAME, TOOL_INPUT)
    )
    cbs_events = events_of_type(evts, "content_block_start")
    cb = cbs_events[0]["content_block"]
    assert cb["name"] == TOOL_NAME


@pytest.mark.asyncio
async def test_tool_use_input_json_delta_round_trips():
    evts, _ = await collect_sse(
        make_tool_use_events(TOOL_ID, TOOL_NAME, TOOL_INPUT)
    )
    deltas = events_of_type(evts, "content_block_delta")
    assert len(deltas) == 1
    d = deltas[0]["delta"]
    assert d["type"] == "input_json_delta"
    parsed = json.loads(d["partial_json"])
    assert parsed == TOOL_INPUT


@pytest.mark.asyncio
async def test_tool_use_sequence_order():
    """start → delta → stop → message_delta → message_stop."""
    evts, _ = await collect_sse(
        make_tool_use_events(TOOL_ID, TOOL_NAME, TOOL_INPUT)
    )
    types = event_types(evts)
    cbs_idx   = types.index("content_block_start")
    cbd_idx   = types.index("content_block_delta")
    cbstop_idx = types.index("content_block_stop")
    md_idx    = types.index("message_delta")
    ms_idx    = types.index("message_stop")
    assert cbs_idx < cbd_idx < cbstop_idx < md_idx < ms_idx


@pytest.mark.asyncio
async def test_tool_use_stop_reason_is_tool_use():
    evts, metrics = await collect_sse(
        make_tool_use_events(TOOL_ID, TOOL_NAME, TOOL_INPUT)
    )
    md = events_of_type(evts, "message_delta")[0]
    assert md["delta"]["stop_reason"] == "tool_use"
    assert metrics.stop_reason == StopReason.TOOL_USE


@pytest.mark.asyncio
async def test_tool_use_cache_fields_present():
    """cache_* fields are present even when zero for tool-use turns."""
    evts, _ = await collect_sse(
        make_tool_use_events(TOOL_ID, TOOL_NAME, TOOL_INPUT)
    )
    usage = events_of_type(evts, "message_delta")[0]["usage"]
    assert "cache_read_input_tokens" in usage
    assert "cache_creation_input_tokens" in usage


# ===========================================================================
# 5. Multiple tool calls
# ===========================================================================

@pytest.mark.asyncio
async def test_multiple_tool_calls_emit_multiple_blocks():
    async def _multi_tool():
        yield KiroEvent(EventName.TOOL_USE, {
            "toolUseId": "toolu_001", "name": "list_dir", "input": {"path": "/"}
        })
        yield KiroEvent(EventName.TOOL_USE, {
            "toolUseId": "toolu_002", "name": "read_file", "input": {"path": "/etc/passwd"}
        })
        yield KiroEvent(EventName.METADATA, {
            "tokenUsage": {"inputTokens": 50, "outputTokens": 20,
                           "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0}
        })
        yield KiroEvent(EventName.MESSAGE_STOP, {"stopReason": "TOOL_USE"})

    evts, _ = await collect_sse(_multi_tool())
    cbs_events = events_of_type(evts, "content_block_start")
    assert len(cbs_events) == 2
    assert cbs_events[0]["content_block"]["id"] == "toolu_001"
    assert cbs_events[1]["content_block"]["id"] == "toolu_002"
    # indices must be sequential
    assert cbs_events[0]["index"] == 0
    assert cbs_events[1]["index"] == 1


# ===========================================================================
# 6. Unknown events are silently skipped
# ===========================================================================

@pytest.mark.asyncio
async def test_unknown_events_skipped():
    """UnknownEvent objects must not break the stream."""
    async def _with_unknowns():
        yield UnknownEvent(name="intentsEvent", payload={"x": 1})
        yield KiroEvent(EventName.ASSISTANT_RESPONSE, {"content": "hello"})
        yield UnknownEvent(name="supplementaryWebLinksEvent", payload={})
        yield KiroEvent(EventName.METADATA, {
            "tokenUsage": {"inputTokens": 5, "outputTokens": 5,
                           "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0}
        })
        yield KiroEvent(EventName.MESSAGE_STOP, {"stopReason": "END_TURN"})

    evts, _ = await collect_sse(_with_unknowns())
    types = event_types(evts)
    assert "message_start" in types
    assert "message_stop" in types
    # intentsEvent / supplementaryWebLinksEvent should not appear
    for et, _ in evts:
        assert et not in ("intentsEvent", "supplementaryWebLinksEvent")


# ===========================================================================
# 7. Stop reason translation
# ===========================================================================

@pytest.mark.asyncio
async def test_end_turn_stop_reason():
    evts, _ = await collect_sse(make_text_events(["text"], stop_reason="END_TURN"))
    md = events_of_type(evts, "message_delta")[0]
    assert md["delta"]["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_max_tokens_stop_reason():
    evts, _ = await collect_sse(make_text_events(["text"], stop_reason="MAX_TOKENS"))
    md = events_of_type(evts, "message_delta")[0]
    assert md["delta"]["stop_reason"] == "max_tokens"


@pytest.mark.asyncio
async def test_model_context_window_exceeded_maps_to_max_tokens():
    evts, _ = await collect_sse(
        make_text_events(["text"], stop_reason="MODEL_CONTEXT_WINDOW_EXCEEDED")
    )
    md = events_of_type(evts, "message_delta")[0]
    assert md["delta"]["stop_reason"] == "max_tokens"


# ===========================================================================
# 8. message_start structure
# ===========================================================================

@pytest.mark.asyncio
async def test_message_start_structure():
    evts, _ = await collect_sse(
        make_text_events(["hello"]),
        model="claude-opus-4.7",
        message_id="msg_abc123",
    )
    ms = events_of_type(evts, "message_start")[0]
    msg = ms["message"]
    assert msg["type"] == "message"
    assert msg["role"] == "assistant"
    assert msg["model"] == "claude-opus-4.7"
    assert msg["id"] == "msg_abc123"
    assert msg["content"] == []
    assert msg["stop_reason"] is None


@pytest.mark.asyncio
async def test_message_stop_is_last():
    evts, _ = await collect_sse(make_text_events(["hi"]))
    assert evts[-1][0] == "message_stop"


@pytest.mark.asyncio
async def test_message_stop_has_correct_type_field():
    evts, _ = await collect_sse(make_text_events(["hi"]))
    ms_data = events_of_type(evts, "message_stop")[0]
    assert ms_data["type"] == "message_stop"


# ===========================================================================
# 9. Empty text chunks are skipped (no spurious deltas)
# ===========================================================================

@pytest.mark.asyncio
async def test_empty_text_chunks_not_emitted():
    async def _empty_chunks():
        yield KiroEvent(EventName.ASSISTANT_RESPONSE, {"content": ""})
        yield KiroEvent(EventName.ASSISTANT_RESPONSE, {"content": "real"})
        yield KiroEvent(EventName.ASSISTANT_RESPONSE, {"content": ""})
        yield KiroEvent(EventName.METADATA, {
            "tokenUsage": {"inputTokens": 5, "outputTokens": 5,
                           "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0}
        })
        yield KiroEvent(EventName.MESSAGE_STOP, {"stopReason": "END_TURN"})

    evts, _ = await collect_sse(_empty_chunks())
    deltas = events_of_type(evts, "content_block_delta")
    # Only the "real" chunk should produce a delta
    assert len(deltas) == 1
    assert deltas[0]["delta"]["text"] == "real"


# ===========================================================================
# 10. Metering and context-usage events update metrics
# ===========================================================================

@pytest.mark.asyncio
async def test_metering_event_updates_credits():
    async def _with_metering():
        yield KiroEvent(EventName.ASSISTANT_RESPONSE, {"content": "ok"})
        yield KiroEvent(EventName.METERING, {"usage": 3.14})
        yield KiroEvent(EventName.METADATA, {
            "tokenUsage": {"inputTokens": 10, "outputTokens": 5,
                           "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0}
        })
        yield KiroEvent(EventName.MESSAGE_STOP, {"stopReason": "END_TURN"})

    _, metrics = await collect_sse(_with_metering())
    assert abs(metrics.credits - 3.14) < 1e-6


@pytest.mark.asyncio
async def test_context_usage_event_updates_percentage():
    async def _with_ctx():
        yield KiroEvent(EventName.ASSISTANT_RESPONSE, {"content": "ok"})
        yield KiroEvent(EventName.CONTEXT_USAGE, {"contextUsagePercentage": 42.5})
        yield KiroEvent(EventName.METADATA, {
            "tokenUsage": {"inputTokens": 10, "outputTokens": 5,
                           "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0}
        })
        yield KiroEvent(EventName.MESSAGE_STOP, {"stopReason": "END_TURN"})

    _, metrics = await collect_sse(_with_ctx())
    assert abs(metrics.context_usage_percentage - 42.5) < 1e-6


# ===========================================================================
# 11. SSE wire format integrity
# ===========================================================================

@pytest.mark.asyncio
async def test_every_event_has_event_line_and_data_line():
    """Each SSE block must have both an event: line and a data: line."""
    from conduit.doors.anthropic_sse import stream_to_anthropic_sse

    metrics = TurnMetrics()
    raw_parts = []
    async for chunk in stream_to_anthropic_sse(
        events=make_text_events(["hello"]),
        model="claude-sonnet-4.6",
        message_id="msg_test",
        metrics=metrics,
    ):
        raw_parts.append(chunk)

    raw = "".join(raw_parts)
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        has_event = any(l.startswith("event: ") for l in lines)
        has_data  = any(l.startswith("data: ")  for l in lines)
        assert has_event, f"Missing event: line in block: {block!r}"
        assert has_data,  f"Missing data: line in block: {block!r}"


@pytest.mark.asyncio
async def test_data_line_is_valid_json():
    """Every data: line must be valid JSON."""
    from conduit.doors.anthropic_sse import stream_to_anthropic_sse

    metrics = TurnMetrics()
    raw_parts = []
    async for chunk in stream_to_anthropic_sse(
        events=make_thinking_events(["think"], VERBATIM_SIGNATURE, ["text"]),
        model="claude-opus-4.7",
        message_id="msg_test2",
        metrics=metrics,
    ):
        raw_parts.append(chunk)

    raw = "".join(raw_parts)
    for line in raw.splitlines():
        if line.startswith("data: "):
            data_str = line[len("data: "):]
            try:
                json.loads(data_str)
            except json.JSONDecodeError as exc:
                pytest.fail(f"Invalid JSON in data line: {data_str!r} — {exc}")


# ===========================================================================
# 12. Thinking + tool combined turn
# ===========================================================================

@pytest.mark.asyncio
async def test_thinking_then_tool_sequence():
    """thinking block → tool_use block — thinking must close before tool opens."""
    async def _think_then_tool():
        yield KiroEvent(EventName.REASONING_CONTENT, {
            "content": "I'll use the tool.", "signature": "real_sig_xyz"
        })
        yield KiroEvent(EventName.TOOL_USE, {
            "toolUseId": "toolu_999", "name": "bash", "input": {"cmd": "ls"}
        })
        yield KiroEvent(EventName.METADATA, {
            "tokenUsage": {"inputTokens": 30, "outputTokens": 10,
                           "cacheReadInputTokens": 2, "cacheWriteInputTokens": 1}
        })
        yield KiroEvent(EventName.MESSAGE_STOP, {"stopReason": "TOOL_USE"})

    evts, _ = await collect_sse(_think_then_tool())
    types = event_types(evts)
    cbs_events = events_of_type(evts, "content_block_start")
    assert cbs_events[0]["content_block"]["type"] == "thinking"
    assert cbs_events[1]["content_block"]["type"] == "tool_use"
    # thinking stop must precede tool_use start
    stops   = [(i, data) for i, (et, data) in enumerate(evts) if et == "content_block_stop"]
    starts  = [(i, data) for i, (et, data) in enumerate(evts) if et == "content_block_start"]
    first_stop_pos  = stops[0][0]
    second_start_pos = starts[1][0]
    assert first_stop_pos < second_start_pos
    # Stop reason should be tool_use
    md = events_of_type(evts, "message_delta")[0]
    assert md["delta"]["stop_reason"] == "tool_use"
