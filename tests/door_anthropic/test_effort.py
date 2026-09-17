"""Effort recovery at the Anthropic door — `_effort_from_request`.

The door's job is narrow: decide whether the client asked for reasoning effort,
and if so hand the raw string to Lane B. Validation belongs to
`reasoning.normalize_effort`, not here.

Every "corpus" figure quoted below is from the 695 captured `claude-code`
requests in SAGA (`saga.db`, 2026-09-17), which is what motivated the fix:
`output_config.effort` appeared 334 times and `thinking.budget_tokens` **zero**
times, so a budget-only reader sent no effort at all on the majority of real
main-loop requests.
"""

from __future__ import annotations

import pytest

from conduit.doors.anthropic import _effort_from_request
from conduit.kiro.reasoning import EFFORT_VALUES


class TestOutputConfigEffort:
    """`output_config.effort` — the field Claude Code actually sends."""

    @pytest.mark.parametrize("effort", sorted(EFFORT_VALUES))
    def test_each_enum_value_passes_through(self, effort):
        """Pass-through, not translation: same field name, same five values."""
        assert (
            _effort_from_request({"type": "adaptive"}, {"effort": effort}) == effort
        )

    def test_corpus_shape(self):
        """The exact pairing seen 334 times in the corpus."""
        assert (
            _effort_from_request({"type": "adaptive"}, {"effort": "high"}) == "high"
        )

    def test_effort_without_thinking_block(self):
        """`output_config` alone is enough — thinking may be absent."""
        assert _effort_from_request(None, {"effort": "max"}) == "max"

    def test_case_and_whitespace_normalized(self):
        assert _effort_from_request(None, {"effort": "  HIGH  "}) == "high"

    def test_takes_priority_over_budget(self):
        """When both are present, the explicit enum wins over the heuristic."""
        assert (
            _effort_from_request(
                {"type": "enabled", "budget_tokens": 1024}, {"effort": "max"}
            )
            == "max"
        )


class TestClassifierIsLeftAlone:
    """`thinking.type == "disabled"` must never acquire an effort level.

    These are auto-mode permission-classifier calls: 365 in the corpus, a ~122k
    character prompt returning a one-token verdict, run *before* each gated tool
    call. Giving them effort would slow every tool call for no benefit.
    """

    def test_disabled_returns_none(self):
        assert _effort_from_request({"type": "disabled"}, None) is None

    def test_disabled_wins_even_with_explicit_effort(self):
        assert _effort_from_request({"type": "disabled"}, {"effort": "max"}) is None

    def test_disabled_with_json_schema_output_config(self):
        """The 18-request corpus case: `output_config` carrying `format`, not effort.

        Reading the `effort` key specifically — rather than testing whether an
        `output_config` object exists — is what keeps these at None.
        """
        oc = {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                    "additionalProperties": False,
                },
            }
        }
        assert _effort_from_request({"type": "disabled"}, oc) is None

    def test_json_schema_alone_is_not_effort(self):
        """Same shape without `disabled` still yields no effort."""
        assert _effort_from_request(None, {"format": {"type": "json_schema"}}) is None


class TestBudgetFallback:
    """`thinking.budget_tokens` — retained for non-Claude-Code callers.

    Never exercised by Claude Code (zero occurrences in the corpus), but an
    Anthropic-shaped client may express reasoning as a token budget.
    """

    @pytest.mark.parametrize(
        "budget,expected",
        [
            (1024, "low"),
            (4095, "low"),
            (4096, "medium"),
            (8191, "medium"),
            (8192, "high"),
            (16383, "high"),
            (16384, "xhigh"),
            (32767, "xhigh"),
            (32768, "max"),
            (200000, "max"),
        ],
    )
    def test_budget_thresholds(self, budget, expected):
        result = _effort_from_request({"type": "enabled", "budget_tokens": budget})
        assert result == expected
        assert result in EFFORT_VALUES

    def test_adaptive_without_budget_or_config_is_none(self):
        """The original defect, kept as a regression guard.

        Claude Code sends `adaptive` with no budget. Before the fix this fell
        through to None and no effort reached Kiro at all — on 334 of 448 main
        requests, carrying 77% of main credit spend.
        """
        assert _effort_from_request({"type": "adaptive"}, None) is None


class TestDegenerateInput:
    """Nothing here may raise: a malformed knob must not fail the request."""

    @pytest.mark.parametrize(
        "thinking,output_config",
        [
            (None, None),
            ({}, {}),
            ({"type": "adaptive"}, {"effort": None}),
            ({"type": "adaptive"}, {"effort": ""}),
            ({"type": "adaptive"}, {"effort": "   "}),
            ({"type": "adaptive"}, {"effort": 8192}),  # int, not enum
            ({"type": "adaptive"}, {"effort": ["high"]}),
            ({"type": "adaptive"}, "not-a-dict"),
            ({"type": "unrecognized"}, None),
            ({"budget_tokens": "8192"}, None),  # string, not int
            ({"type": "enabled", "budget_tokens": None}, None),
        ],
    )
    def test_returns_none_without_raising(self, thinking, output_config):
        assert _effort_from_request(thinking, output_config) is None

    def test_unknown_effort_string_is_passed_through_for_lane_b(self):
        """The door does not validate; `normalize_effort` owns that.

        A bogus value must reach Lane B rather than being silently dropped,
        because dropping it is indistinguishable from "no effort requested".
        """
        assert _effort_from_request(None, {"effort": "turbo"}) == "turbo"
