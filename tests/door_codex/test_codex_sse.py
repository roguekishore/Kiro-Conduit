"""Tests for codex_sse.py — OpenAI Responses SSE emitter.

Verifies:
  - response.created / response.in_progress lifecycle frames are emitted first
  - Text deltas arrive as response.output_text.delta frames
  - response.completed is emitted as the terminal frame on END_TURN
  - response.completed carries the six-field usage map (no estimation)
  - response.failed is emitted with correct typed code on error stop reasons
  - response.incomplete is emitted on PAUSE_TURN / MAX_TOKENS
  - Kiro ``signature`` field (reasoning blob) is NEVER forwarded
  - reasoning_text.delta emitted for reasoningContentEvent
  - stream ending without messageStopEvent emits response.incomplete
"""

from __future__ import annotations

import json
from typing import AsyncGenerator, List

import pytest
import pytest_asyncio

from conduit.contracts.events import EventName, KiroEvent, UnknownEvent
from conduit.contracts.metrics import StopReason, TurnMetrics
from conduit.doors.codex_sse import (
    ResponsesErrorCode,
    build_usage,
    frame_response_completed,
    frame_response_failed,
    stop_reason_to_error_code,
    translate_kiro_to_responses_sse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _collect_frames(
    events: List[KiroEvent | UnknownEvent],
    model: str = "luna",
    response_id: str = "resp_test",
) -> List[dict]:
    """Run the translator over a fixed event list and collect parsed SSE frames."""

    async def _gen() -> AsyncGenerator:
        for e in events:
            yield e

    frames = []
    async for raw_frame in translate_kiro_to_responses_sse(
        _gen(), model=model, response_id=response_id
    ):
        # Parse "event: TYPE\ndata: JSON\n\n"
        event_type = None
        data_obj = None
        for line in raw_frame.splitlines():
            if line.startswith("event: "):
                event_type = line[len("event: "):]
            elif line.startswith("data: "):
                data_obj = json.loads(line[len("data: "):])
        if event_type and data_obj is not None:
            frames.append({"event": event_type, "data": data_obj})
    return frames


def _events_of(frames: List[dict], event_type: str) -> List[dict]:
    return [f for f in frames if f["event"] == event_type]


def _end_turn_sequence(extra_text: str = "Hello.") -> List[KiroEvent]:
    return [
        KiroEvent(EventName.ASSISTANT_RESPONSE, {"content": extra_text}),
        KiroEvent(
            EventName.METADATA,
            # Kiro's real shape: cache counts ride in metadataEvent.tokenUsage,
            # reasoning is `thoughtTokens` (not `thinkingTokens`), and
            # totalTokens is sent explicitly — it is read, never summed, because
            # a sum of input+output silently omits thought tokens.
            {"tokenUsage": {
                "inputTokens": 20,
                "outputTokens": 8,
                "thoughtTokens": 0,
                "cacheReadInputTokens": 4,
                "cacheWriteInputTokens": 2,
                "totalTokens": 28,
            }},
        ),
        KiroEvent(
            EventName.CONTEXT_USAGE,
            # contextUsageEvent carries the window percentage, not cache counts.
            {"contextUsagePercentage": 12.5},
        ),
        KiroEvent(EventName.MESSAGE_STOP, {"stopReason": "END_TURN"}),
    ]


# ---------------------------------------------------------------------------
# Lifecycle frames
# ---------------------------------------------------------------------------

class TestLifecycleFrames:
    @pytest.mark.asyncio
    async def test_created_and_in_progress_emitted_first(self):
        frames = await _collect_frames(_end_turn_sequence())
        assert frames[0]["event"] == "response.created"
        assert frames[1]["event"] == "response.in_progress"

    @pytest.mark.asyncio
    async def test_response_id_consistent_across_frames(self):
        frames = await _collect_frames(_end_turn_sequence(), response_id="resp_abc")
        for f in frames:
            rid = (
                f["data"].get("response_id")
                or (f["data"].get("response") or {}).get("id")
            )
            if rid:
                assert rid == "resp_abc", f"response id mismatch in {f['event']}"


# ---------------------------------------------------------------------------
# Text content
# ---------------------------------------------------------------------------

class TestTextContent:
    @pytest.mark.asyncio
    async def test_text_delta_emitted(self):
        frames = await _collect_frames(_end_turn_sequence("Hi there."))
        deltas = _events_of(frames, "response.output_text.delta")
        assert any("Hi there." in d["data"].get("delta", "") for d in deltas)

    @pytest.mark.asyncio
    async def test_output_item_added_before_first_delta(self):
        frames = await _collect_frames(_end_turn_sequence("X"))
        event_names = [f["event"] for f in frames]
        added_idx = event_names.index("response.output_item.added")
        delta_idx = next(
            i for i, n in enumerate(event_names) if n == "response.output_text.delta"
        )
        assert added_idx < delta_idx

    @pytest.mark.asyncio
    async def test_content_part_added_before_first_delta(self):
        frames = await _collect_frames(_end_turn_sequence("Y"))
        event_names = [f["event"] for f in frames]
        part_idx = event_names.index("response.content_part.added")
        delta_idx = next(
            i for i, n in enumerate(event_names) if n == "response.output_text.delta"
        )
        assert part_idx < delta_idx


# ---------------------------------------------------------------------------
# Terminal: response.completed
# ---------------------------------------------------------------------------

class TestResponseCompleted:
    @pytest.mark.asyncio
    async def test_completed_is_last_frame(self):
        frames = await _collect_frames(_end_turn_sequence())
        assert frames[-1]["event"] == "response.completed"

    @pytest.mark.asyncio
    async def test_completed_has_status_completed(self):
        frames = await _collect_frames(_end_turn_sequence())
        completed = frames[-1]["data"]
        assert completed["response"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_usage_six_fields_all_present(self):
        """The six-field usage map must be fully populated — no estimation."""
        frames = await _collect_frames(_end_turn_sequence())
        completed = frames[-1]["data"]
        usage = completed["response"]["usage"]

        assert "input_tokens" in usage
        assert "output_tokens" in usage
        assert "total_tokens" in usage
        assert "input_tokens_details" in usage
        assert "cached_tokens" in usage["input_tokens_details"]
        assert "cache_write_tokens" in usage["input_tokens_details"]
        assert "output_tokens_details" in usage
        assert "reasoning_tokens" in usage["output_tokens_details"]

    @pytest.mark.asyncio
    async def test_usage_values_match_events(self):
        """Usage values come from metadata/context_usage events, not estimates."""
        frames = await _collect_frames(_end_turn_sequence())
        usage = frames[-1]["data"]["response"]["usage"]

        assert usage["input_tokens"] == 20
        assert usage["output_tokens"] == 8
        assert usage["input_tokens_details"]["cached_tokens"] == 4
        assert usage["input_tokens_details"]["cache_write_tokens"] == 2
        assert usage["output_tokens_details"]["reasoning_tokens"] == 0


# ---------------------------------------------------------------------------
# Terminal: response.failed
# ---------------------------------------------------------------------------

class TestResponseFailed:
    def _failed_events(self, stop_reason: str) -> List[KiroEvent]:
        return [
            KiroEvent(EventName.ASSISTANT_RESPONSE, {"content": "partial"}),
            KiroEvent(EventName.MESSAGE_STOP, {"stopReason": stop_reason}),
        ]

    @pytest.mark.asyncio
    async def test_context_window_exceeded_gives_context_length_code(self):
        frames = await _collect_frames(
            self._failed_events("MODEL_CONTEXT_WINDOW_EXCEEDED")
        )
        failed = _events_of(frames, "response.failed")
        assert failed, "response.failed not emitted"
        code = failed[0]["data"]["response"]["error"]["code"]
        assert code == ResponsesErrorCode.CONTEXT_LENGTH_EXCEEDED.value

    @pytest.mark.asyncio
    async def test_monthly_quota_gives_insufficient_quota_code(self):
        frames = await _collect_frames(
            self._failed_events("MONTHLY_REQUEST_COUNT")
        )
        failed = _events_of(frames, "response.failed")
        assert failed
        assert (
            failed[0]["data"]["response"]["error"]["code"]
            == ResponsesErrorCode.INSUFFICIENT_QUOTA.value
        )

    @pytest.mark.asyncio
    async def test_daily_quota_gives_insufficient_quota_code(self):
        frames = await _collect_frames(self._failed_events("DAILY_REQUEST_COUNT"))
        failed = _events_of(frames, "response.failed")
        assert failed
        assert (
            failed[0]["data"]["response"]["error"]["code"]
            == ResponsesErrorCode.INSUFFICIENT_QUOTA.value
        )

    @pytest.mark.asyncio
    async def test_insufficient_capacity_gives_rate_limit_code(self):
        frames = await _collect_frames(
            self._failed_events("INSUFFICIENT_MODEL_CAPACITY")
        )
        failed = _events_of(frames, "response.failed")
        assert failed
        assert (
            failed[0]["data"]["response"]["error"]["code"]
            == ResponsesErrorCode.RATE_LIMIT_EXCEEDED.value
        )

    @pytest.mark.asyncio
    async def test_content_filtered_gives_invalid_request_code(self):
        frames = await _collect_frames(self._failed_events("CONTENT_FILTERED"))
        failed = _events_of(frames, "response.failed")
        assert failed
        assert (
            failed[0]["data"]["response"]["error"]["code"]
            == ResponsesErrorCode.INVALID_REQUEST_ERROR.value
        )

    @pytest.mark.asyncio
    async def test_failed_status_in_response_object(self):
        frames = await _collect_frames(
            self._failed_events("MODEL_CONTEXT_WINDOW_EXCEEDED")
        )
        failed = _events_of(frames, "response.failed")
        assert failed[0]["data"]["response"]["status"] == "failed"


# ---------------------------------------------------------------------------
# Terminal: response.incomplete
# ---------------------------------------------------------------------------

class TestResponseIncomplete:
    @pytest.mark.asyncio
    async def test_pause_turn_gives_incomplete(self):
        events = [
            KiroEvent(EventName.ASSISTANT_RESPONSE, {"content": "partial"}),
            KiroEvent(EventName.MESSAGE_STOP, {"stopReason": "PAUSE_TURN"}),
        ]
        frames = await _collect_frames(events)
        incomplete = _events_of(frames, "response.incomplete")
        assert incomplete, "response.incomplete not emitted for PAUSE_TURN"
        assert incomplete[0]["data"]["response"]["status"] == "incomplete"

    @pytest.mark.asyncio
    async def test_max_tokens_gives_incomplete(self):
        events = [
            KiroEvent(EventName.ASSISTANT_RESPONSE, {"content": "cut"}),
            KiroEvent(EventName.MESSAGE_STOP, {"stopReason": "MAX_TOKENS"}),
        ]
        frames = await _collect_frames(events)
        incomplete = _events_of(frames, "response.incomplete")
        assert incomplete

    @pytest.mark.asyncio
    async def test_stream_end_without_stop_event_gives_incomplete(self):
        """Stream ending without messageStopEvent must yield response.incomplete."""
        events = [
            KiroEvent(EventName.ASSISTANT_RESPONSE, {"content": "truncated"}),
            # No MESSAGE_STOP
        ]
        frames = await _collect_frames(events)
        incomplete = _events_of(frames, "response.incomplete")
        assert incomplete, "Expected response.incomplete when stream ends without stop"


# ---------------------------------------------------------------------------
# Reasoning: signature must NOT be forwarded
# ---------------------------------------------------------------------------

class TestReasoningBlobs:
    @pytest.mark.asyncio
    async def test_signature_field_never_forwarded(self):
        """Kiro ``signature`` is provider-specific — must be dropped, not relayed."""
        events = [
            KiroEvent(
                EventName.REASONING_CONTENT,
                {
                    "content": "I need to think about this.",
                    "signature": "OPAQUE_KIRO_BLOB_DO_NOT_FORWARD",
                },
            ),
            KiroEvent(EventName.MESSAGE_STOP, {"stopReason": "END_TURN"}),
        ]
        frames = await _collect_frames(events)
        all_data = json.dumps(frames)
        assert "OPAQUE_KIRO_BLOB_DO_NOT_FORWARD" not in all_data, (
            "Kiro signature blob leaked into Responses SSE output!"
        )

    @pytest.mark.asyncio
    async def test_reasoning_text_delta_emitted(self):
        """Reasoning text must still be emitted even though signature is dropped."""
        events = [
            KiroEvent(
                EventName.REASONING_CONTENT,
                {
                    "content": "thinking text here",
                    "signature": "SHOULD_BE_DROPPED",
                },
            ),
            KiroEvent(EventName.MESSAGE_STOP, {"stopReason": "END_TURN"}),
        ]
        frames = await _collect_frames(events)
        deltas = _events_of(frames, "response.reasoning_text.delta")
        assert any(
            "thinking text here" in d["data"].get("delta", "") for d in deltas
        ), "Reasoning text delta not found"


# ---------------------------------------------------------------------------
# Unknown events are silently skipped
# ---------------------------------------------------------------------------

class TestUnknownEvents:
    @pytest.mark.asyncio
    async def test_unknown_events_skipped_gracefully(self):
        events = [
            UnknownEvent(name="someNewKiroEvent", payload={"x": 1}),
            KiroEvent(EventName.ASSISTANT_RESPONSE, {"content": "ok"}),
            KiroEvent(EventName.MESSAGE_STOP, {"stopReason": "END_TURN"}),
        ]
        frames = await _collect_frames(events)
        assert _events_of(frames, "response.completed"), "Should still complete normally"


# ---------------------------------------------------------------------------
# build_usage unit tests
# ---------------------------------------------------------------------------

class TestBuildUsage:
    def _metrics(self, **kwargs) -> TurnMetrics:
        m = TurnMetrics()
        for k, v in kwargs.items():
            setattr(m, k, v)
        return m

    def test_all_six_fields_present(self):
        m = self._metrics(
            input_tokens=100,
            output_tokens=50,
            thought_tokens=20,
            cache_read_tokens=10,
            cache_write_tokens=5,
            total_tokens=150,
        )
        usage = build_usage(m)
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50
        assert usage["total_tokens"] == 150
        assert usage["input_tokens_details"]["cached_tokens"] == 10
        assert usage["input_tokens_details"]["cache_write_tokens"] == 5
        assert usage["output_tokens_details"]["reasoning_tokens"] == 20

    def test_zeros_are_valid(self):
        usage = build_usage(TurnMetrics())
        assert usage["input_tokens"] == 0
        assert usage["total_tokens"] == 0
        assert usage["output_tokens_details"]["reasoning_tokens"] == 0


# ---------------------------------------------------------------------------
# stop_reason_to_error_code unit tests
# ---------------------------------------------------------------------------

class TestStopReasonMapping:
    def test_context_window(self):
        assert (
            stop_reason_to_error_code(StopReason.MODEL_CONTEXT_WINDOW_EXCEEDED)
            == ResponsesErrorCode.CONTEXT_LENGTH_EXCEEDED
        )

    def test_monthly_quota(self):
        assert (
            stop_reason_to_error_code(StopReason.MONTHLY_REQUEST_COUNT)
            == ResponsesErrorCode.INSUFFICIENT_QUOTA
        )

    def test_daily_quota(self):
        assert (
            stop_reason_to_error_code(StopReason.DAILY_REQUEST_COUNT)
            == ResponsesErrorCode.INSUFFICIENT_QUOTA
        )

    def test_capacity(self):
        assert (
            stop_reason_to_error_code(StopReason.INSUFFICIENT_MODEL_CAPACITY)
            == ResponsesErrorCode.RATE_LIMIT_EXCEEDED
        )

    def test_content_filtered(self):
        assert (
            stop_reason_to_error_code(StopReason.CONTENT_FILTERED)
            == ResponsesErrorCode.INVALID_REQUEST_ERROR
        )

    def test_none_gives_server_error(self):
        assert (
            stop_reason_to_error_code(None) == ResponsesErrorCode.SERVER_ERROR
        )

    def test_unknown_reason_gives_server_error(self):
        # TOOL_USE is not an error stop — unmapped → SERVER_ERROR
        assert (
            stop_reason_to_error_code(StopReason.TOOL_USE)
            == ResponsesErrorCode.SERVER_ERROR
        )
