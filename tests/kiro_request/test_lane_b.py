"""
Lane B unit tests — conduit/kiro/{request,history,tools,reasoning}.py

Verification gate (per spec):
1. System prompt lands in conversationState.systemPrompt with flag on
2. System prompt prepends into first user message with flag off
3. No "(empty placeholder)" text anywhere in merged output
4. Consecutive same-role turns merge
5. is_error: True → status: "error" in tool result
6. Effort maps to the enum string, never to a token-budget integer
7. injections in RewrittenOut lists exactly what was done
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from conduit.contracts.rewritten import Injection, InjectionType
from conduit.kiro.history import HistoryMessage, merge_turns
from conduit.kiro.request import build_conversation_state
from conduit.kiro.reasoning import (
    EFFORT_VALUES,
    build_reasoning_fields,
    normalize_effort,
)
from conduit.kiro.tools import (
    build_tool_result,
    build_tool_results_from_anthropic,
    build_tool_spec,
    build_tool_specs,
    sanitize_json_schema,
)


# ===========================================================================
# Fixtures
# ===========================================================================

def _msgs(*pairs) -> List[HistoryMessage]:
    """Build a list of HistoryMessage from (role, content) pairs."""
    return [HistoryMessage(role=r, content=c) for r, c in pairs]


def _simple_convo() -> List[HistoryMessage]:
    """A minimal well-formed user/assistant/user conversation."""
    return _msgs(
        ("user", "hello"),
        ("assistant", "hi"),
        ("user", "how are you?"),
    )


# ===========================================================================
# 1 & 2 — System prompt placement (native slot vs. prepend fallback)
# ===========================================================================

class TestSystemPromptPlacement:
    def test_native_slot_flag_on_puts_system_in_conversation_state(self):
        """Flag ON: system prompt must appear in conversationState.systemPrompt."""
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True):
            payload, rewritten = build_conversation_state(
                messages=_simple_convo(),
                system_prompt="Be helpful.",
                model_id="claude-sonnet-4.6",
                tools=None,
            )

        cs = payload["conversationState"]
        assert cs.get("systemPrompt") == "Be helpful.", (
            "conversationState.systemPrompt should contain the system prompt"
        )

    def test_native_slot_flag_on_system_not_in_user_content(self):
        """Flag ON: system prompt must NOT appear prepended in any user message."""
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True):
            payload, _ = build_conversation_state(
                messages=_simple_convo(),
                system_prompt="Be helpful.",
                model_id="claude-sonnet-4.6",
                tools=None,
            )

        cs = payload["conversationState"]
        # Check current message
        cur_content = cs["currentMessage"]["userInputMessage"]["content"]
        assert "Be helpful." not in cur_content, (
            "System prompt should not be prepended into user message when native slot is on"
        )
        # Check history
        for entry in cs.get("history", []):
            user_msg = entry.get("userInputMessage", {})
            if user_msg:
                assert "Be helpful." not in user_msg.get("content", ""), (
                    "System prompt should not appear in history user messages"
                )

    def test_fallback_prepend_flag_off_puts_system_in_first_user_message(self):
        """Flag OFF: system prompt must appear prepended in the first user message."""
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", False):
            payload, rewritten = build_conversation_state(
                messages=_simple_convo(),
                system_prompt="You are an expert.",
                model_id="claude-sonnet-4.6",
                tools=None,
            )

        cs = payload["conversationState"]
        # History has the first user message
        history = cs.get("history", [])
        first_user = next(
            (e["userInputMessage"]["content"] for e in history if "userInputMessage" in e),
            None,
        )
        cur_content = cs["currentMessage"]["userInputMessage"]["content"]

        # It must be in one of them — history first if it exists, else current
        if first_user is not None:
            assert "You are an expert." in first_user, (
                "System prompt should be prepended into the first history user message"
            )
        else:
            assert "You are an expert." in cur_content

    def test_fallback_prepend_flag_off_no_native_system_prompt_field(self):
        """Flag OFF: conversationState must NOT have a systemPrompt key."""
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", False):
            payload, _ = build_conversation_state(
                messages=_simple_convo(),
                system_prompt="You are an expert.",
                model_id="claude-sonnet-4.6",
                tools=None,
            )

        cs = payload["conversationState"]
        assert "systemPrompt" not in cs, (
            "conversationState should not have systemPrompt key in fallback mode"
        )

    def test_no_system_prompt_no_slot_in_payload(self):
        """When system_prompt is None/empty, no systemPrompt key in conversationState."""
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True):
            payload, _ = build_conversation_state(
                messages=_simple_convo(),
                system_prompt=None,
                model_id="claude-sonnet-4.6",
                tools=None,
            )

        assert "systemPrompt" not in payload["conversationState"]


# ===========================================================================
# 3 — No "(empty placeholder)" text in output
# ===========================================================================

class TestNoPlaceholderText:
    PLACEHOLDER = "(empty placeholder)"

    def _find_placeholder_in_payload(self, payload: Dict[str, Any]) -> List[str]:
        """Recursively find all strings equal to PLACEHOLDER in payload."""
        hits = []
        self._recurse(payload, hits)
        return hits

    def _recurse(self, obj: Any, hits: List[str]) -> None:
        if isinstance(obj, str) and obj == self.PLACEHOLDER:
            hits.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                self._recurse(v, hits)
        elif isinstance(obj, list):
            for item in obj:
                self._recurse(item, hits)

    def test_simple_conversation_no_placeholder(self):
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True):
            payload, _ = build_conversation_state(
                messages=_simple_convo(),
                system_prompt="sys",
                model_id="claude-sonnet-4.6",
                tools=None,
            )
        hits = self._find_placeholder_in_payload(payload)
        assert hits == [], f"Found '(empty placeholder)' text in payload: {hits}"

    def test_merged_turns_no_placeholder(self):
        """Merging consecutive turns must not produce placeholder sentinel text."""
        msgs = _msgs(
            ("user", "turn one"),
            ("user", "turn two"),
            ("assistant", "response"),
        )
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True):
            payload, _ = build_conversation_state(
                messages=msgs,
                system_prompt=None,
                model_id="claude-sonnet-4.6",
                tools=None,
            )
        hits = self._find_placeholder_in_payload(payload)
        assert hits == [], f"Found '(empty placeholder)' text in payload: {hits}"


# ===========================================================================
# 4 — Consecutive same-role turns merge
# ===========================================================================

class TestTurnMerge:
    def test_two_consecutive_user_turns_merge(self):
        msgs = _msgs(("user", "part one"), ("user", "part two"))
        merged, injections = merge_turns(msgs)

        assert len(merged) == 1, "Two consecutive user turns should merge into one"
        assert "part one" in merged[0].content
        assert "part two" in merged[0].content

    def test_two_consecutive_assistant_turns_merge(self):
        msgs = _msgs(("assistant", "alpha"), ("assistant", "beta"))
        merged, injections = merge_turns(msgs)

        assert len(merged) == 1
        assert "alpha" in merged[0].content
        assert "beta" in merged[0].content

    def test_alternating_roles_not_merged(self):
        msgs = _msgs(("user", "hi"), ("assistant", "hello"), ("user", "bye"))
        merged, injections = merge_turns(msgs)

        assert len(merged) == 3, "Alternating roles should not be merged"

    def test_merge_injection_declared_when_merge_occurs(self):
        msgs = _msgs(("user", "a"), ("user", "b"))
        _, injections = merge_turns(msgs)

        assert any(
            inj.type == InjectionType.TURN_MERGE for inj in injections
        ), "TURN_MERGE injection must be declared when a merge occurs"

    def test_no_injection_when_no_merge(self):
        msgs = _msgs(("user", "hi"), ("assistant", "hello"))
        _, injections = merge_turns(msgs)

        assert not any(
            inj.type == InjectionType.TURN_MERGE for inj in injections
        ), "No TURN_MERGE injection should be declared when no merge occurs"

    def test_tool_calls_merged(self):
        tc1 = {"name": "bash", "input": {"command": "ls"}, "toolUseId": "id1"}
        tc2 = {"name": "read", "input": {"path": "/tmp"}, "toolUseId": "id2"}
        msgs = [
            HistoryMessage(role="assistant", content="first", tool_calls=[tc1]),
            HistoryMessage(role="assistant", content="second", tool_calls=[tc2]),
        ]
        merged, _ = merge_turns(msgs)

        assert len(merged) == 1
        assert len(merged[0].tool_calls) == 2

    def test_tool_results_merged(self):
        tr1 = {"content": [{"text": "r1"}], "status": "success", "toolUseId": "id1"}
        tr2 = {"content": [{"text": "r2"}], "status": "success", "toolUseId": "id2"}
        msgs = [
            HistoryMessage(role="user", content="first", tool_results=[tr1]),
            HistoryMessage(role="user", content="second", tool_results=[tr2]),
        ]
        merged, _ = merge_turns(msgs)

        assert len(merged) == 1
        assert len(merged[0].tool_results) == 2

    def test_merge_in_full_pipeline_no_placeholder(self):
        """Merged content must not contain the upstream placeholder sentinel."""
        msgs = _msgs(("user", "part one"), ("user", "part two"), ("assistant", "ok"))
        merged, _ = merge_turns(msgs)

        for msg in merged:
            assert "(empty placeholder)" not in msg.content, (
                "Merged content must not contain '(empty placeholder)' text"
            )


# ===========================================================================
# 5 — is_error: True → status: "error"
# ===========================================================================

class TestToolResultStatus:
    def test_is_error_true_produces_error_status(self):
        result = build_tool_result("id1", "something went wrong", is_error=True)
        assert result["status"] == "error", "is_error=True must produce status='error'"

    def test_is_error_false_produces_success_status(self):
        result = build_tool_result("id1", "output data", is_error=False)
        assert result["status"] == "success", "is_error=False must produce status='success'"

    def test_default_is_success(self):
        """Default (no is_error arg) must be success, not error."""
        result = build_tool_result("id1", "data")
        assert result["status"] == "success"

    def test_tool_use_id_preserved(self):
        result = build_tool_result("my-call-id", "content")
        assert result["toolUseId"] == "my-call-id"

    def test_content_in_wire_format(self):
        result = build_tool_result("id1", "hello")
        assert result["content"] == [{"text": "hello"}]

    def test_empty_content_sentinel(self):
        result = build_tool_result("id1", "")
        assert result["content"] == [{"text": "(empty result)"}]

    def test_anthropic_is_error_field_read(self):
        """build_tool_results_from_anthropic must read is_error from block."""
        content = [
            {
                "type": "tool_result",
                "tool_use_id": "call_abc",
                "content": "Error: file not found",
                "is_error": True,
            }
        ]
        results = build_tool_results_from_anthropic(content)
        assert len(results) == 1
        assert results[0]["status"] == "error"

    def test_anthropic_no_is_error_defaults_success(self):
        content = [
            {
                "type": "tool_result",
                "tool_use_id": "call_xyz",
                "content": "file contents here",
            }
        ]
        results = build_tool_results_from_anthropic(content)
        assert len(results) == 1
        assert results[0]["status"] == "success"

    def test_anthropic_is_error_false_is_success(self):
        content = [
            {
                "type": "tool_result",
                "tool_use_id": "call_xyz",
                "content": "ok",
                "is_error": False,
            }
        ]
        results = build_tool_results_from_anthropic(content)
        assert results[0]["status"] == "success"


# ===========================================================================
# 6 — Effort maps to the enum string, never to a token-budget integer
# ===========================================================================

class TestReasoningEffort:
    def test_all_valid_effort_values_pass_through(self):
        for effort in ("low", "medium", "high", "xhigh", "max"):
            assert normalize_effort(effort) == effort

    def test_unknown_effort_returns_default_string(self):
        result = normalize_effort("turbo")
        assert isinstance(result, str), "normalize_effort must always return a string"
        assert result in EFFORT_VALUES, "Fallback must be one of the valid enum strings"

    def test_none_effort_returns_string(self):
        result = normalize_effort(None)
        assert isinstance(result, str)
        assert result in EFFORT_VALUES

    def test_claude_model_effort_is_string_not_int(self):
        """output_config.effort must be a string enum, never an integer."""
        fields = build_reasoning_fields("claude-sonnet-4.6", "high")
        effort_val = fields["output_config"]["effort"]
        assert isinstance(effort_val, str), (
            "output_config.effort must be a string enum, not a token-budget integer"
        )
        assert effort_val in EFFORT_VALUES

    def test_gpt_model_effort_is_string_not_int(self):
        fields = build_reasoning_fields("gpt-5.6-luna", "medium")
        effort_val = fields["reasoning"]["effort"]
        assert isinstance(effort_val, str)
        assert effort_val in EFFORT_VALUES

    def test_claude_model_gets_output_config_and_thinking(self):
        fields = build_reasoning_fields("claude-opus-4.7", "max")
        assert "output_config" in fields, "Claude model must get output_config"
        assert "thinking" in fields, "Claude model must get thinking block"
        assert fields["thinking"]["type"] == "adaptive"
        assert fields["thinking"]["display"] == "summarized"

    def test_gpt_model_gets_reasoning_key(self):
        fields = build_reasoning_fields("gpt-5.6-terra", "low")
        assert "reasoning" in fields
        assert "output_config" not in fields

    def test_auto_model_gets_claude_layout(self):
        """'auto' is a Claude variant and must receive the Claude layout."""
        fields = build_reasoning_fields("auto", "high")
        assert "output_config" in fields
        assert "reasoning" not in fields

    def test_effort_is_never_integer_in_payload(self):
        """End-to-end: reasoning fields in the payload must not contain integers."""
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True):
            payload, _ = build_conversation_state(
                messages=_simple_convo(),
                system_prompt=None,
                model_id="claude-sonnet-4.6",
                tools=None,
                reasoning_effort="xhigh",
            )

        additional = payload["conversationState"].get("additionalModelRequestFields", {})
        # Flatten all values and check no int leaked in
        def _find_ints(obj):
            found = []
            if isinstance(obj, int):
                found.append(obj)
            elif isinstance(obj, dict):
                for v in obj.values():
                    found.extend(_find_ints(v))
            elif isinstance(obj, list):
                for item in obj:
                    found.extend(_find_ints(item))
            return found

        ints_found = _find_ints(additional)
        assert ints_found == [], (
            f"Token-budget integers must not appear in reasoning fields, found: {ints_found}"
        )


# ===========================================================================
# 7 — injections in RewrittenOut lists exactly what was done
# ===========================================================================

class TestInjectionDeclarations:
    def test_system_prompt_slot_injection_when_flag_on(self):
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True):
            _, rewritten = build_conversation_state(
                messages=_simple_convo(),
                system_prompt="sys",
                model_id="claude-sonnet-4.6",
                tools=None,
            )

        types = [inj.type for inj in rewritten.injections]
        assert InjectionType.SYSTEM_PROMPT_SLOT in types, (
            "SYSTEM_PROMPT_SLOT injection must be declared when native slot is used"
        )
        # Must NOT declare THINKING_TAGS (that's the fallback path)
        assert InjectionType.THINKING_TAGS not in types

    def test_thinking_tags_injection_when_flag_off(self):
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", False):
            _, rewritten = build_conversation_state(
                messages=_simple_convo(),
                system_prompt="sys",
                model_id="claude-sonnet-4.6",
                tools=None,
            )

        types = [inj.type for inj in rewritten.injections]
        assert InjectionType.THINKING_TAGS in types, (
            "THINKING_TAGS injection must be declared when prepend fallback is used"
        )
        assert InjectionType.SYSTEM_PROMPT_SLOT not in types

    def test_no_system_prompt_no_placement_injection(self):
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True):
            _, rewritten = build_conversation_state(
                messages=_simple_convo(),
                system_prompt=None,
                model_id="claude-sonnet-4.6",
                tools=None,
            )

        placement_types = {InjectionType.SYSTEM_PROMPT_SLOT, InjectionType.THINKING_TAGS}
        declared = {inj.type for inj in rewritten.injections}
        assert not declared.intersection(placement_types), (
            "No placement injection should be declared when there is no system prompt"
        )

    def test_turn_merge_injection_declared(self):
        msgs = _msgs(("user", "a"), ("user", "b"), ("assistant", "c"))
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True):
            _, rewritten = build_conversation_state(
                messages=msgs,
                system_prompt=None,
                model_id="claude-sonnet-4.6",
                tools=None,
            )

        types = [inj.type for inj in rewritten.injections]
        assert InjectionType.TURN_MERGE in types, (
            "TURN_MERGE injection must be declared when turns were merged"
        )

    def test_no_turn_merge_injection_when_no_merge(self):
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True):
            _, rewritten = build_conversation_state(
                messages=_simple_convo(),
                system_prompt=None,
                model_id="claude-sonnet-4.6",
                tools=None,
            )

        types = [inj.type for inj in rewritten.injections]
        assert InjectionType.TURN_MERGE not in types, (
            "TURN_MERGE injection must not be declared when no merge occurred"
        )

    def test_injection_location_system_prompt_slot(self):
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True):
            _, rewritten = build_conversation_state(
                messages=_simple_convo(),
                system_prompt="sys",
                model_id="claude-sonnet-4.6",
                tools=None,
            )

        slot_injections = [
            inj for inj in rewritten.injections
            if inj.type == InjectionType.SYSTEM_PROMPT_SLOT
        ]
        assert len(slot_injections) == 1
        assert slot_injections[0].location == "systemPrompt"

    def test_injection_location_thinking_tags(self):
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", False):
            _, rewritten = build_conversation_state(
                messages=_simple_convo(),
                system_prompt="sys",
                model_id="claude-sonnet-4.6",
                tools=None,
            )

        tag_injections = [
            inj for inj in rewritten.injections
            if inj.type == InjectionType.THINKING_TAGS
        ]
        assert len(tag_injections) == 1
        assert tag_injections[0].location == "last_user_message"

    def test_injection_location_turn_merge(self):
        msgs = _msgs(("user", "a"), ("user", "b"))
        _, injections = merge_turns(msgs)

        merge_injs = [inj for inj in injections if inj.type == InjectionType.TURN_MERGE]
        assert len(merge_injs) == 1
        assert merge_injs[0].location == "history"


# ===========================================================================
# Supplementary — tools conversion, sanitize_json_schema
# ===========================================================================

class TestToolConversion:
    def test_tool_spec_shape(self):
        spec = build_tool_spec("bash", "Run a command", {"type": "object", "properties": {}})
        assert "toolSpecification" in spec
        ts = spec["toolSpecification"]
        assert ts["name"] == "bash"
        assert ts["description"] == "Run a command"
        assert "inputSchema" in ts
        assert "json" in ts["inputSchema"]

    def test_empty_description_gets_placeholder(self):
        spec = build_tool_spec("mystery", None, {})
        assert spec["toolSpecification"]["description"] == "Tool: mystery"

    def test_build_tool_specs_list(self):
        tools = [
            {"name": "read_file", "description": "Reads a file", "input_schema": {"type": "object"}},
            {"name": "write_file", "description": "Writes a file", "input_schema": {"type": "object"}},
        ]
        specs, docs = build_tool_specs(tools)
        assert len(specs) == 2
        names = [s["toolSpecification"]["name"] for s in specs]
        assert "read_file" in names
        assert "write_file" in names
        # Nothing oversized, so no relocation block.
        assert docs == ""

    def test_sanitize_removes_empty_required(self):
        schema = {"type": "object", "required": [], "properties": {"x": {"type": "string"}}}
        result = sanitize_json_schema(schema)
        assert "required" not in result

    def test_sanitize_keeps_nonempty_required(self):
        schema = {"type": "object", "required": ["x"], "properties": {"x": {"type": "string"}}}
        result = sanitize_json_schema(schema)
        assert result["required"] == ["x"]

    def test_sanitize_removes_additional_properties(self):
        schema = {"type": "object", "additionalProperties": False}
        result = sanitize_json_schema(schema)
        assert "additionalProperties" not in result

    def test_sanitize_nested(self):
        schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "required": [],
                    "additionalProperties": True,
                }
            },
        }
        result = sanitize_json_schema(schema)
        nested = result["properties"]["nested"]
        assert "required" not in nested
        assert "additionalProperties" not in nested


# ===========================================================================
# Supplementary — payload structure smoke tests
# ===========================================================================

class TestPayloadStructure:
    def test_conversation_state_present(self):
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True):
            payload, _ = build_conversation_state(
                messages=_simple_convo(),
                system_prompt=None,
                model_id="claude-sonnet-4.6",
                tools=None,
            )
        assert "conversationState" in payload

    def test_chat_trigger_type_is_manual(self):
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True):
            payload, _ = build_conversation_state(
                messages=_simple_convo(),
                system_prompt=None,
                model_id="claude-sonnet-4.6",
                tools=None,
            )
        assert payload["conversationState"]["chatTriggerType"] == "MANUAL"

    def test_conversation_id_present(self):
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True):
            payload, _ = build_conversation_state(
                messages=_simple_convo(),
                system_prompt=None,
                model_id="claude-sonnet-4.6",
                tools=None,
                conversation_id="test-id-123",
            )
        assert payload["conversationState"]["conversationId"] == "test-id-123"

    def test_profile_arn_added_when_provided(self):
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True):
            payload, _ = build_conversation_state(
                messages=_simple_convo(),
                system_prompt=None,
                model_id="claude-sonnet-4.6",
                tools=None,
                profile_arn="arn:aws:codewhisperer:us-east-1:123:profile/test",
            )
        assert payload.get("profileArn") == "arn:aws:codewhisperer:us-east-1:123:profile/test"

    def test_profile_arn_omitted_when_none(self):
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True):
            payload, _ = build_conversation_state(
                messages=_simple_convo(),
                system_prompt=None,
                model_id="claude-sonnet-4.6",
                tools=None,
                profile_arn=None,
            )
        assert "profileArn" not in payload

    def test_tools_in_user_input_context(self):
        tools = [{"name": "bash", "description": "Run cmd", "input_schema": {"type": "object"}}]
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True):
            payload, _ = build_conversation_state(
                messages=_simple_convo(),
                system_prompt=None,
                model_id="claude-sonnet-4.6",
                tools=tools,
            )
        ctx = (
            payload["conversationState"]["currentMessage"]["userInputMessage"]
            .get("userInputMessageContext", {})
        )
        assert "tools" in ctx, "tools must appear in userInputMessageContext"

    def test_rewritten_out_has_correct_structure(self):
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True):
            _, rewritten = build_conversation_state(
                messages=_simple_convo(),
                system_prompt="s",
                model_id="claude-sonnet-4.6",
                tools=None,
            )
        assert isinstance(rewritten.injections, list)
        assert rewritten.current_message is not None
        assert isinstance(rewritten.history, list)

    def test_single_user_message_works(self):
        """A conversation with only one user message must not raise."""
        msgs = _msgs(("user", "just this"))
        with patch("conduit.kiro.request.cfg.KIRO_NATIVE_SYSTEM_PROMPT", True):
            payload, _ = build_conversation_state(
                messages=msgs,
                system_prompt=None,
                model_id="claude-sonnet-4.6",
                tools=None,
            )
        cs = payload["conversationState"]
        assert "currentMessage" in cs
        assert cs.get("history", []) == []
