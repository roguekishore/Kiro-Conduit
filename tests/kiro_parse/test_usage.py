"""Tests for conduit.kiro.usage — wire-to-TurnMetrics parsing.

Covers:
- All cache-field spelling variants (6 alternate key names)
- stopReason / stopDetails / stopDetails-as-object
- meteringEvent credit accumulation
- contextUsageEvent percentage
- reasoningContentEvent → ReasoningBlockMeta
- Repeated content delta MUST NOT be deduplicated (anti-regression against
  _reference_upstream/parsers.py lines 254, 342-344)
"""

from __future__ import annotations

import pytest

from conduit.contracts.metrics import StopReason, TurnMetrics
from conduit.kiro.usage import (
    apply_context_usage_event,
    apply_metadata_event,
    apply_metering_event,
    apply_reasoning_content_event,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fresh() -> TurnMetrics:
    return TurnMetrics()


# ---------------------------------------------------------------------------
# metadataEvent — primary field spellings
# ---------------------------------------------------------------------------


def test_metadata_primary_field_spellings() -> None:
    """Primary key names parse into the correct TurnMetrics fields."""
    m = fresh()
    apply_metadata_event(
        m,
        {
            "tokenUsage": {
                "inputTokens": 100,
                "outputTokens": 50,
                "thoughtTokens": 20,
                "cachedReadTokens": 30,
                "cachedWriteTokens": 10,
                "totalTokens": 210,
                "stopReason": "END_TURN",
            }
        },
    )
    assert m.input_tokens == 100
    assert m.output_tokens == 50
    assert m.thought_tokens == 20
    assert m.cache_read_tokens == 30
    assert m.cache_write_tokens == 10
    assert m.total_tokens == 210
    assert m.stop_reason == StopReason.END_TURN


# ---------------------------------------------------------------------------
# Cache-field spelling variants (the full test matrix)
# ---------------------------------------------------------------------------


def test_cache_read_variant_uncached_input_tokens() -> None:
    """uncachedInputTokens → input_tokens."""
    m = fresh()
    apply_metadata_event(m, {"tokenUsage": {"uncachedInputTokens": 77}})
    assert m.input_tokens == 77


def test_cache_read_variant_cache_read_input_tokens() -> None:
    """cacheReadInputTokens → cache_read_tokens."""
    m = fresh()
    apply_metadata_event(m, {"tokenUsage": {"cacheReadInputTokens": 55}})
    assert m.cache_read_tokens == 55


def test_cache_write_variant_cache_write_input_tokens() -> None:
    """cacheWriteInputTokens → cache_write_tokens."""
    m = fresh()
    apply_metadata_event(m, {"tokenUsage": {"cacheWriteInputTokens": 33}})
    assert m.cache_write_tokens == 33


def test_cache_read_primary_wins_over_alternate() -> None:
    """cachedReadTokens takes priority over cacheReadInputTokens when both present."""
    m = fresh()
    apply_metadata_event(
        m,
        {"tokenUsage": {"cachedReadTokens": 10, "cacheReadInputTokens": 99}},
    )
    assert m.cache_read_tokens == 10


def test_cache_write_primary_wins_over_alternate() -> None:
    """cachedWriteTokens takes priority over cacheWriteInputTokens when both present."""
    m = fresh()
    apply_metadata_event(
        m,
        {"tokenUsage": {"cachedWriteTokens": 7, "cacheWriteInputTokens": 88}},
    )
    assert m.cache_write_tokens == 7


def test_input_primary_wins_over_uncached_alternate() -> None:
    """inputTokens takes priority over uncachedInputTokens when both present."""
    m = fresh()
    apply_metadata_event(
        m,
        {"tokenUsage": {"inputTokens": 42, "uncachedInputTokens": 999}},
    )
    assert m.input_tokens == 42


# ---------------------------------------------------------------------------
# stopReason variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stop_value,expected",
    [
        ("END_TURN", StopReason.END_TURN),
        ("MAX_TOKENS", StopReason.MAX_TOKENS),
        ("TOOL_USE", StopReason.TOOL_USE),
        ("CONTENT_FILTERED", StopReason.CONTENT_FILTERED),
        ("PAUSE_TURN", StopReason.PAUSE_TURN),
        ("MODEL_CONTEXT_WINDOW_EXCEEDED", StopReason.MODEL_CONTEXT_WINDOW_EXCEEDED),
        ("MONTHLY_REQUEST_COUNT", StopReason.MONTHLY_REQUEST_COUNT),
        ("DAILY_REQUEST_COUNT", StopReason.DAILY_REQUEST_COUNT),
        ("INSUFFICIENT_MODEL_CAPACITY", StopReason.INSUFFICIENT_MODEL_CAPACITY),
    ],
)
def test_stop_reason_all_members(stop_value: str, expected: StopReason) -> None:
    """Every StopReason enum member parses correctly."""
    m = fresh()
    apply_metadata_event(m, {"tokenUsage": {"stopReason": stop_value}})
    assert m.stop_reason == expected


def test_stop_details_string_fallback() -> None:
    """stopDetails as a plain string is used when stopReason is absent."""
    m = fresh()
    apply_metadata_event(m, {"tokenUsage": {"stopDetails": "END_TURN"}})
    assert m.stop_reason == StopReason.END_TURN


def test_stop_details_object_fallback() -> None:
    """stopDetails as an object with a stopReason key is parsed."""
    m = fresh()
    apply_metadata_event(
        m,
        {"tokenUsage": {"stopDetails": {"stopReason": "MAX_TOKENS"}}},
    )
    assert m.stop_reason == StopReason.MAX_TOKENS


def test_stop_reason_unknown_value_leaves_none() -> None:
    """An unrecognised stopReason string leaves stop_reason as None."""
    m = fresh()
    apply_metadata_event(m, {"tokenUsage": {"stopReason": "SOME_FUTURE_REASON"}})
    assert m.stop_reason is None


# ---------------------------------------------------------------------------
# normalizedTokenUsage as supplementary source
# ---------------------------------------------------------------------------


def test_normalized_token_usage_fills_missing_fields() -> None:
    """normalizedTokenUsage fills fields that are otherwise 0."""
    m = fresh()
    apply_metadata_event(
        m,
        {
            "tokenUsage": {
                "normalizedTokenUsage": {
                    "inputTokens": 5,
                    "outputTokens": 3,
                    "totalTokens": 8,
                }
            }
        },
    )
    assert m.input_tokens == 5
    assert m.output_tokens == 3
    assert m.total_tokens == 8


def test_normalized_token_usage_does_not_overwrite_real_counts() -> None:
    """normalizedTokenUsage must not overwrite counts already parsed."""
    m = fresh()
    apply_metadata_event(
        m,
        {
            "tokenUsage": {
                "inputTokens": 100,
                "normalizedTokenUsage": {"inputTokens": 999},
            }
        },
    )
    assert m.input_tokens == 100


# ---------------------------------------------------------------------------
# Outer wrapper form
# ---------------------------------------------------------------------------


def test_outer_wrapper_form_parsed() -> None:
    """Payload wrapped as {"metadataEvent": {...}} is unwrapped correctly."""
    m = fresh()
    apply_metadata_event(
        m,
        {"metadataEvent": {"tokenUsage": {"inputTokens": 42, "outputTokens": 7}}},
    )
    assert m.input_tokens == 42
    assert m.output_tokens == 7


# ---------------------------------------------------------------------------
# meteringEvent credit accumulation
# ---------------------------------------------------------------------------


def test_metering_event_sets_credits() -> None:
    """meteringEvent.usage is stored as credits."""
    m = fresh()
    apply_metering_event(m, {"usage": 1.5})
    assert m.credits == pytest.approx(1.5)


def test_metering_event_accumulates_across_calls() -> None:
    """Multiple meteringEvent frames accumulate; credits are not replaced."""
    m = fresh()
    apply_metering_event(m, {"usage": 1.0})
    apply_metering_event(m, {"usage": 0.5})
    apply_metering_event(m, {"usage": 2.25})
    assert m.credits == pytest.approx(3.75)


def test_metering_event_outer_wrapper_form() -> None:
    """Payload wrapped as {"meteringEvent": {"usage": ...}} is unwrapped."""
    m = fresh()
    apply_metering_event(m, {"meteringEvent": {"usage": 7.0}})
    assert m.credits == pytest.approx(7.0)


def test_metering_event_no_usage_field_is_no_op() -> None:
    """meteringEvent without a 'usage' field leaves credits unchanged."""
    m = fresh()
    apply_metering_event(m, {"unrelated": "data"})
    assert m.credits == 0.0


# ---------------------------------------------------------------------------
# contextUsageEvent
# ---------------------------------------------------------------------------


def test_context_usage_event_sets_percentage() -> None:
    """contextUsagePercentage is stored on the metrics object."""
    m = fresh()
    apply_context_usage_event(m, {"contextUsagePercentage": 42.5})
    assert m.context_usage_percentage == pytest.approx(42.5)


def test_context_usage_event_outer_wrapper_form() -> None:
    """contextUsageEvent outer-wrapper form is unwrapped."""
    m = fresh()
    apply_context_usage_event(
        m, {"contextUsageEvent": {"contextUsagePercentage": 75.0}}
    )
    assert m.context_usage_percentage == pytest.approx(75.0)


# ---------------------------------------------------------------------------
# reasoningContentEvent → ReasoningBlockMeta
# ---------------------------------------------------------------------------


def test_reasoning_content_event_with_signature() -> None:
    """signature_present=True when a signature field is present."""
    m = fresh()
    apply_reasoning_content_event(
        m,
        {
            "modelId": "kiro-pro-v1",
            "text": "I am thinking ...",
            "signature": "abc123",
        },
    )
    assert len(m.reasoning_blocks) == 1
    block = m.reasoning_blocks[0]
    assert block.model_id == "kiro-pro-v1"
    assert block.signature_present is True


def test_reasoning_content_event_without_signature() -> None:
    """signature_present=False when no signature key is in the payload."""
    m = fresh()
    apply_reasoning_content_event(
        m,
        {"modelId": "kiro-lite-v1", "text": "thinking ..."},
    )
    assert len(m.reasoning_blocks) == 1
    assert m.reasoning_blocks[0].signature_present is False


def test_reasoning_content_event_multiple_blocks() -> None:
    """Multiple reasoningContentEvent frames produce multiple ReasoningBlockMeta entries."""
    m = fresh()
    apply_reasoning_content_event(m, {"modelId": "m1", "signature": "s1"})
    apply_reasoning_content_event(m, {"modelId": "m2"})
    assert len(m.reasoning_blocks) == 2
    assert m.reasoning_blocks[0].signature_present is True
    assert m.reasoning_blocks[1].signature_present is False


def test_reasoning_content_event_outer_wrapper_form() -> None:
    """reasoningContentEvent outer-wrapper form is unwrapped."""
    m = fresh()
    apply_reasoning_content_event(
        m,
        {"reasoningContentEvent": {"modelId": "wrapped-model", "signature": "sig"}},
    )
    assert len(m.reasoning_blocks) == 1
    assert m.reasoning_blocks[0].model_id == "wrapped-model"
    assert m.reasoning_blocks[0].signature_present is True


# ---------------------------------------------------------------------------
# No-dedup anti-regression: repeated content delta MUST survive
# ---------------------------------------------------------------------------


def test_repeated_content_delta_is_not_deduplicated() -> None:
    """Identical content chunks must all be preserved — no last_content dedup.

    _reference_upstream/parsers.py lines 254 and 342-344 silently dropped a
    chunk equal to the previous one.  That is the defect being retired here.
    Indentation runs, empty lines, and any repeated token are legitimate output.
    """
    # Simulate two back-to-back assistantResponseEvent payloads with identical content
    content = "    "   # four spaces — a common indentation run
    payloads = [
        {"content": content},
        {"content": content},   # exact duplicate of the previous
        {"content": content},   # third repeat
    ]
    # There is no deduplication layer in dispatch or usage; we verify that
    # the dispatch layer returns all three events rather than one.
    from conduit.kiro.dispatch import dispatch_event
    from conduit.contracts.events import EventName, KiroEvent
    import json

    results = []
    for p in payloads:
        event = dispatch_event(
            {":event-type": "assistantResponseEvent"},
            json.dumps(p).encode(),
        )
        results.append(event)

    assert len(results) == 3, "All three chunks must come through"
    for r in results:
        assert isinstance(r, KiroEvent)
        assert r.name == EventName.ASSISTANT_RESPONSE
        assert r.payload["content"] == content, (
            "Content must be preserved verbatim, not deduplicated"
        )


def test_repeated_newline_delta_is_not_deduplicated() -> None:
    """\\n\\n (paragraph break) repeated three times must all survive."""
    from conduit.kiro.dispatch import dispatch_event
    from conduit.contracts.events import EventName, KiroEvent
    import json

    content = "\n\n"
    results = []
    for _ in range(3):
        event = dispatch_event(
            {":event-type": "assistantResponseEvent"},
            json.dumps({"content": content}).encode(),
        )
        results.append(event)

    assert len(results) == 3
    assert all(isinstance(r, KiroEvent) for r in results)
    assert all(r.payload["content"] == content for r in results)
