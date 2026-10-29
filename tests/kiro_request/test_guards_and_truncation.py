"""
Ported-protection tests — payload_guards.py, truncation.py, tool-doc relocation.

These cover the three gaps CONDUIT had against the upstream gateway, plus the
one place CONDUIT deliberately goes further than upstream:

1. Orphaned tool results are repaired UNCONDITIONALLY (upstream reaches its own
   repair only with AUTO_TRIM_PAYLOAD on, so it never runs by default).
2. Truncation is distinguished from malformed JSON, recorded, and surfaced to
   the model exactly once on the next request.
3. Oversized tool descriptions are relocated into the system prompt rather than
   sent as-is, and the relocation is declared in RewrittenOut.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from conduit.contracts.rewritten import InjectionType
from conduit.kiro.history import HistoryMessage
from conduit.kiro.payload_guards import (
    apply_payload_guards,
    check_payload_size,
    enforce_tool_result_integrity,
    trim_payload_to_limit,
)
from conduit.kiro.request import build_conversation_state
from conduit.kiro.tools import build_tool_specs
from conduit.kiro.truncation import (
    diagnose_json_truncation,
    notice_for_content,
    notice_for_tool,
    pending_counts,
    record_content_truncation,
    record_tool_truncation,
    reset as truncation_reset,
    take_content_truncation,
    take_tool_truncation,
)


@pytest.fixture(autouse=True)
def _clean_truncation_store():
    """Truncation state is process-global; isolate every test."""
    truncation_reset()
    yield
    truncation_reset()


# ===========================================================================
# Helpers
# ===========================================================================

def _user_entry(content: str, tool_results: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    msg: Dict[str, Any] = {"content": content, "modelId": "claude-sonnet-4.6", "origin": "AI_EDITOR"}
    if tool_results is not None:
        msg["userInputMessageContext"] = {"toolResults": tool_results}
    return {"userInputMessage": msg}


def _assistant_entry(content: str, tool_uses: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    msg: Dict[str, Any] = {"content": content}
    if tool_uses is not None:
        msg["toolUses"] = tool_uses
    return {"assistantResponseMessage": msg}


def _result(tool_use_id: str, text: str = "ok") -> Dict[str, Any]:
    return {"toolUseId": tool_use_id, "content": [{"text": text}], "status": "success"}


def _payload(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "conversationState": {
            "chatTriggerType": "MANUAL",
            "conversationId": "c1",
            "currentMessage": _user_entry("now"),
            "history": history,
        }
    }


# ===========================================================================
# 1 — Orphaned tool-result repair
# ===========================================================================

class TestOrphanRepair:

    def test_orphan_is_removed(self):
        """A result with no matching toolUse must not reach Kiro."""
        payload = _payload([
            _assistant_entry("calling", [{"toolUseId": "known", "name": "Read", "input": {}}]),
            _user_entry("result", [_result("known"), _result("ghost")]),
        ])
        stats = enforce_tool_result_integrity(payload)
        assert stats.orphans_removed == 1
        kept = payload["conversationState"]["history"][1]["userInputMessage"]
        ids = [r["toolUseId"] for r in kept["userInputMessageContext"]["toolResults"]]
        assert ids == ["known"]

    def test_orphan_text_is_preserved_not_discarded(self):
        """The point of repair is protocol validity, not data loss."""
        payload = _payload([
            _assistant_entry("calling", [{"toolUseId": "known", "name": "Read", "input": {}}]),
            _user_entry("here", [_result("ghost", "IMPORTANT OUTPUT")]),
        ])
        enforce_tool_result_integrity(payload)
        user_msg = payload["conversationState"]["history"][1]["userInputMessage"]
        assert "IMPORTANT OUTPUT" in user_msg["content"]
        assert "[trimmed tool result]" in user_msg["content"]

    def test_context_removed_when_all_results_orphaned(self):
        payload = _payload([
            _assistant_entry("no tools at all"),
            _user_entry("here", [_result("ghost")]),
        ])
        enforce_tool_result_integrity(payload)
        user_msg = payload["conversationState"]["history"][1]["userInputMessage"]
        assert "userInputMessageContext" not in user_msg

    def test_content_never_left_empty(self):
        """Kiro rejects empty content, so repair must not create it."""
        payload = _payload([
            _assistant_entry("nothing"),
            _user_entry("", [{"toolUseId": "ghost", "content": [], "status": "success"}]),
        ])
        enforce_tool_result_integrity(payload)
        user_msg = payload["conversationState"]["history"][1]["userInputMessage"]
        assert user_msg["content"]

    def test_valid_history_is_untouched(self):
        """No false positives: a well-formed pairing must survive intact."""
        history = [
            _assistant_entry("calling", [{"toolUseId": "t1", "name": "Read", "input": {}}]),
            _user_entry("result", [_result("t1")]),
        ]
        payload = _payload(history)
        before = check_payload_size(payload)
        stats = enforce_tool_result_integrity(payload)
        assert stats.orphans_removed == 0
        assert check_payload_size(payload) == before

    def test_repair_runs_with_auto_trim_off(self):
        """The upstream defect: repair must not depend on AUTO_TRIM_PAYLOAD."""
        payload = _payload([
            _assistant_entry("nothing"),
            _user_entry("here", [_result("ghost")]),
        ])
        apply_payload_guards(payload, max_bytes=10_000_000, auto_trim=False)
        user_msg = payload["conversationState"]["history"][1]["userInputMessage"]
        assert "userInputMessageContext" not in user_msg

    def test_empty_tool_uses_stripped(self):
        payload = _payload([_assistant_entry("hi", [])])
        apply_payload_guards(payload, max_bytes=10_000_000, auto_trim=False)
        assert "toolUses" not in payload["conversationState"]["history"][0]["assistantResponseMessage"]


# ===========================================================================
# 2 — Trimming
# ===========================================================================

class TestTrimming:

    def test_trim_reduces_size(self):
        history = []
        for i in range(40):
            history.append(_user_entry("x" * 2000))
            history.append(_assistant_entry("y" * 2000))
        payload = _payload(history)
        assert check_payload_size(payload) > 50_000
        stats = trim_payload_to_limit(payload, 50_000)
        assert stats.trimmed
        assert stats.final_bytes <= 50_000
        assert stats.final_entries < stats.original_entries

    def test_trim_leaves_history_starting_at_user(self):
        history = []
        for i in range(20):
            history.append(_user_entry("x" * 3000))
            history.append(_assistant_entry("y" * 3000))
        payload = _payload(history)
        trim_payload_to_limit(payload, 20_000)
        remaining = payload["conversationState"].get("history", [])
        if remaining:
            assert "userInputMessage" in remaining[0]

    def test_trim_repairs_orphans_it_creates(self):
        """Dropping an assistant turn orphans the result that followed it."""
        history = [_user_entry("x" * 8000), _assistant_entry("y" * 8000)]
        history.append(_assistant_entry("call", [{"toolUseId": "t9", "name": "Read", "input": {}}]))
        history.append(_user_entry("res", [_result("t9")]))
        payload = _payload(history)
        trim_payload_to_limit(payload, 5_000)
        for i, entry in enumerate(payload["conversationState"].get("history", [])):
            user_msg = entry.get("userInputMessage")
            if not user_msg:
                continue
            ctx = user_msg.get("userInputMessageContext") or {}
            for tr in ctx.get("toolResults", []):
                prev = payload["conversationState"]["history"][i - 1] if i > 0 else {}
                declared = {
                    tu["toolUseId"]
                    for tu in (prev.get("assistantResponseMessage", {}).get("toolUses") or [])
                }
                assert tr["toolUseId"] in declared

    def test_no_history_is_not_an_error(self):
        payload = {"conversationState": {"currentMessage": _user_entry("hi")}}
        stats = trim_payload_to_limit(payload, 100)
        assert not stats.trimmed


# ===========================================================================
# 3 — Truncation diagnosis
# ===========================================================================

class TestTruncationDiagnosis:

    @pytest.mark.parametrize("blob,reason_fragment", [
        # Cut mid-string. The "opened { but never closed it" check fires before
        # quote parity, so this is reported as a missing brace -- the reason
        # differs but the verdict is what the caller acts on.
        ('{"command": "ls -la', "missing 1 closing brace(s)"),
        # Ends with } but an inner object is still open.
        ('{"a": 1, "b": {"c": 2}', "unbalanced braces (2 open, 1 close)"),
        ('[{"a": 1}, {"b": 2}', "missing 1 closing bracket(s)"),
        # Balanced delimiters, odd quote count -- the only route to the
        # quote-parity branch.
        ('{"a: 1}', "unclosed string literal"),
    ])
    def test_truncated_shapes_detected(self, blob, reason_fragment):
        diag = diagnose_json_truncation(blob)
        assert diag["is_truncated"]
        assert reason_fragment in diag["reason"]

    def test_malformed_is_not_truncated(self):
        """A retry can fix bad JSON; it cannot fix a size limit. Don't conflate."""
        diag = diagnose_json_truncation('{"a": : 1}')
        assert not diag["is_truncated"]
        assert diag["reason"] == "malformed JSON"

    def test_escaped_quotes_do_not_read_as_unterminated(self):
        diag = diagnose_json_truncation('{"path": "C:\\\\dir\\\\file.txt"}')
        assert not diag["is_truncated"]

    def test_empty_is_not_truncated(self):
        assert not diagnose_json_truncation("")["is_truncated"]

    def test_size_is_reported_in_bytes_not_chars(self):
        diag = diagnose_json_truncation('{"x": "é')
        assert diag["size_bytes"] == len('{"x": "é'.encode("utf-8"))


# ===========================================================================
# 4 — Truncation record lifecycle
# ===========================================================================

class TestTruncationRecords:

    def test_tool_record_round_trips(self):
        record_tool_truncation("tu_1", "Write", 5000, "missing 2 closing brace(s)")
        got = take_tool_truncation("tu_1")
        assert got is not None
        assert got.tool_name == "Write"
        assert got.size_bytes == 5000

    def test_retrieval_is_destructive(self):
        """A notice must be delivered once, not on every subsequent request."""
        record_tool_truncation("tu_2", "Bash", 100, "unclosed string literal")
        assert take_tool_truncation("tu_2") is not None
        assert take_tool_truncation("tu_2") is None

    def test_content_keyed_by_hash_not_identity(self):
        text = "a partial sentence that stops mid-wo"
        record_content_truncation(text)
        # A separate but equal string must still match.
        assert take_content_truncation("" + text) is not None

    def test_content_miss_returns_none(self):
        record_content_truncation("one thing")
        assert take_content_truncation("a completely different thing") is None

    def test_unknown_id_is_a_miss_not_a_crash(self):
        assert take_tool_truncation("never-seen") is None
        assert take_tool_truncation("") is None

    def test_store_is_bounded(self):
        for i in range(400):
            record_tool_truncation(f"tu_{i}", "Read", 10, "unclosed string literal")
        assert pending_counts()["tool"] <= 256

    def test_notices_are_self_labelling(self):
        """Upstream needed a standing system-prompt paragraph to legitimize
        these; a marker in the text itself costs nothing on a healthy turn."""
        record_tool_truncation("tu_3", "Grep", 42, "unbalanced braces (3 open, 1 close)")
        rec = take_tool_truncation("tu_3")
        tool_notice = notice_for_tool(rec)
        assert "[CONDUIT]" in tool_notice
        assert "Grep" in tool_notice
        assert "[CONDUIT]" in notice_for_content()


# ===========================================================================
# 5 — Tool-doc relocation
# ===========================================================================

class TestToolDocRelocation:

    def test_long_description_is_relocated_not_truncated(self):
        long_desc = "D" * 500
        tools = [{"name": "bigtool", "description": long_desc, "input_schema": {"type": "object"}}]
        specs, docs = build_tool_specs(tools, max_description_length=100)
        spec_desc = specs[0]["toolSpecification"]["description"]
        assert "Full documentation in system prompt" in spec_desc
        assert "bigtool" in spec_desc
        # Nothing lost: the whole original text is in the relocated block.
        assert long_desc in docs
        assert "## Tool: bigtool" in docs

    def test_short_description_untouched(self):
        tools = [{"name": "small", "description": "Reads a file", "input_schema": {}}]
        specs, docs = build_tool_specs(tools, max_description_length=100)
        assert specs[0]["toolSpecification"]["description"] == "Reads a file"
        assert docs == ""

    def test_zero_disables_relocation(self):
        long_desc = "D" * 5000
        tools = [{"name": "t", "description": long_desc, "input_schema": {}}]
        specs, docs = build_tool_specs(tools, max_description_length=0)
        assert specs[0]["toolSpecification"]["description"] == long_desc
        assert docs == ""

    def test_relocated_docs_reach_the_system_prompt(self):
        long_desc = "Z" * 400
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True), \
             patch("conduit.kiro.request.cfg.TOOL_DESCRIPTION_MAX_LENGTH", 100):
            payload, rewritten = build_conversation_state(
                messages=[HistoryMessage(role="user", content="hi")],
                system_prompt="BASE PROMPT",
                model_id="claude-sonnet-4.6",
                tools=[{"name": "big", "description": long_desc, "input_schema": {}}],
            )
        sys_prompt = payload["conversationState"]["systemPrompt"]
        assert "BASE PROMPT" in sys_prompt
        assert long_desc in sys_prompt
        assert any(
            i.type == InjectionType.RELOCATED_TOOL_DOCS for i in rewritten.injections
        )

    def test_relocation_declared_only_when_it_happens(self):
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True), \
             patch("conduit.kiro.request.cfg.TOOL_DESCRIPTION_MAX_LENGTH", 10000):
            _, rewritten = build_conversation_state(
                messages=[HistoryMessage(role="user", content="hi")],
                system_prompt="BASE",
                model_id="claude-sonnet-4.6",
                tools=[{"name": "small", "description": "short", "input_schema": {}}],
            )
        assert not any(
            i.type == InjectionType.RELOCATED_TOOL_DOCS for i in rewritten.injections
        )

    def test_relocation_works_without_a_system_prompt(self):
        """Tool docs must survive even when the request has no system prompt."""
        long_desc = "Q" * 400
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True), \
             patch("conduit.kiro.request.cfg.TOOL_DESCRIPTION_MAX_LENGTH", 100):
            payload, _ = build_conversation_state(
                messages=[HistoryMessage(role="user", content="hi")],
                system_prompt=None,
                model_id="claude-sonnet-4.6",
                tools=[{"name": "big", "description": long_desc, "input_schema": {}}],
            )
        assert long_desc in payload["conversationState"]["systemPrompt"]


# ===========================================================================
# 6 — Guards are actually wired into the build
# ===========================================================================

class TestGuardsWiredIntoBuild:

    def test_orphan_repaired_end_to_end(self):
        """The regression that matters: a resumed conversation must not 400."""
        messages = [
            HistoryMessage(role="user", content="do it"),
            HistoryMessage(
                role="assistant",
                content="calling",
                tool_calls=[{"toolUseId": "real", "name": "Read", "input": {}}],
            ),
            # Result for a call that is NOT in the assistant turn above --
            # exactly what compaction leaves behind.
            HistoryMessage(
                role="user",
                content="result",
                tool_results=[_result("vanished")],
            ),
            HistoryMessage(role="assistant", content="ok"),
            HistoryMessage(role="user", content="continue"),
        ]
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True):
            payload, rewritten = build_conversation_state(
                messages=messages,
                system_prompt="sys",
                model_id="claude-sonnet-4.6",
                tools=None,
            )

        for i, entry in enumerate(payload["conversationState"]["history"]):
            user_msg = entry.get("userInputMessage")
            if not user_msg:
                continue
            ctx = user_msg.get("userInputMessageContext") or {}
            for tr in ctx.get("toolResults", []):
                prev = payload["conversationState"]["history"][i - 1] if i > 0 else {}
                declared = {
                    tu["toolUseId"]
                    for tu in (prev.get("assistantResponseMessage", {}).get("toolUses") or [])
                }
                assert tr["toolUseId"] in declared, "orphan reached the wire"

    def test_rewritten_reports_post_guard_history(self):
        """The seam must show what Kiro received, not the pre-guard draft."""
        messages = [
            HistoryMessage(role="user", content="a"),
            HistoryMessage(role="assistant", content="b"),
            HistoryMessage(role="user", content="c", tool_results=[_result("ghost")]),
            HistoryMessage(role="assistant", content="d"),
            HistoryMessage(role="user", content="e"),
        ]
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True):
            payload, rewritten = build_conversation_state(
                messages=messages,
                system_prompt="sys",
                model_id="claude-sonnet-4.6",
                tools=None,
            )
        assert rewritten.history == payload["conversationState"]["history"]
        assert rewritten.current_message == payload["conversationState"]["currentMessage"]

    def test_no_placeholder_text_survives_guards(self):
        """CONDUIT's core promise; guards must not reintroduce upstream's sentinel."""
        messages = [
            HistoryMessage(role="user", content="a"),
            HistoryMessage(role="user", content="b"),
            HistoryMessage(role="assistant", content="c"),
            HistoryMessage(role="user", content="d"),
        ]
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True):
            payload, _ = build_conversation_state(
                messages=messages,
                system_prompt="sys",
                model_id="claude-sonnet-4.6",
                tools=None,
            )
        import json
        assert "(empty placeholder)" not in json.dumps(payload)
