"""Regression tests for the two defects that broke Codex tool use entirely.

Both were found by replaying SAGA-captured gpt-5.6-luna traffic through the door.

Defect 1 — only the LAST developer instruction item survived
    Codex sends more than one `role: "developer"` message item and every one is
    load-bearing. On the captured turn input[1] carried the 17730-char base Codex
    prompt and input[2] carried 8307 chars of <skills_instructions> /
    <collaboration_mode>. The extractor assigned `prompt = text` per item, so the
    last item won and the base prompt — the half that tells the model how to
    behave and how to drive its tools — was silently discarded.

Defect 2 — freeform ("custom") tools were framed as ordinary functions
    Codex's `exec` is declared `{"type": "custom", "format": {"syntax": "lark",
    ...}}` with NO `parameters`. Treated as a function it reached Kiro with
    `inputSchema.json == {}`, and the reply went back as
    function_call/arguments. Codex routes freeform tools through
    `ToolPayload::Custom` only, so the call was discarded as unsupported, and on
    the next request Codex back-filled the missing result with the literal string
    "aborted" (codex-rs/core/src/context_manager/normalize.rs:64).

    For a `tool_mode: "code_mode_only"` model such as gpt-5.6-luna, `exec` is the
    ONLY route to the filesystem or a shell — which is why every read, edit, and
    command failed while plain text streamed normally.
"""

from __future__ import annotations

import json

from conduit.doors.codex_dialect import (
    FREEFORM_PARAM,
    adapt_freeform_tools,
    extract_dialect,
    freeform_input_from_arguments,
)
from conduit.kiro.tools import build_tool_specs


# ---------------------------------------------------------------------------
# Defect 1 — multiple developer instruction items
# ---------------------------------------------------------------------------

class TestMultipleDeveloperInstructionItems:

    def _two_items(self):
        return [
            {"type": "message", "role": "developer",
             "content": [{"type": "input_text", "text": "BASE PROMPT: You are Codex."}]},
            {"type": "message", "role": "developer",
             "content": [{"type": "input_text", "text": "<skills_instructions>SKILLS"}]},
        ]

    def test_both_items_survive(self):
        prompt, _ = extract_dialect(
            instructions="", tools=None, input_items=self._two_items(), is_lite=True
        )
        assert "BASE PROMPT: You are Codex." in prompt
        assert "<skills_instructions>SKILLS" in prompt

    def test_base_prompt_comes_first(self):
        """Array order must be preserved: the base prompt leads."""
        prompt, _ = extract_dialect(
            instructions="", tools=None, input_items=self._two_items(), is_lite=True
        )
        assert prompt.startswith("BASE PROMPT")
        assert prompt.index("BASE PROMPT") < prompt.index("SKILLS")

    def test_items_joined_with_blank_line(self):
        prompt, _ = extract_dialect(
            instructions="", tools=None, input_items=self._two_items(), is_lite=True
        )
        assert "You are Codex.\n\n<skills_instructions>" in prompt

    def test_additional_tools_item_does_not_break_the_join(self):
        items = [
            {"type": "additional_tools", "role": "developer", "tools": []},
            {"type": "message", "role": "developer", "content": "FIRST"},
            {"type": "message", "role": "developer", "content": "SECOND"},
        ]
        prompt, _ = extract_dialect(
            instructions="", tools=None, input_items=items, is_lite=True
        )
        assert prompt == "FIRST\n\nSECOND"

    def test_single_item_has_no_trailing_separator(self):
        items = [{"type": "message", "role": "developer", "content": "ONLY"}]
        prompt, _ = extract_dialect(
            instructions="", tools=None, input_items=items, is_lite=True
        )
        assert prompt == "ONLY"


# ---------------------------------------------------------------------------
# Defect 2 — freeform tool declaration
# ---------------------------------------------------------------------------

EXEC = {
    "type": "custom",
    "name": "exec",
    "description": "Run JavaScript code to orchestrate/compose tool calls.",
    "format": {"type": "grammar", "syntax": "lark", "definition": "start: X"},
}
WAIT = {
    "type": "function",
    "name": "wait",
    "description": "Wait on a running exec cell.",
    "parameters": {"type": "object", "properties": {"cell_id": {"type": "string"}}},
}


class TestFreeformToolAdaptation:

    def test_custom_tool_is_reported(self):
        _, names = adapt_freeform_tools([EXEC, WAIT])
        assert names == {"exec"}

    def test_custom_tool_gains_a_real_schema(self):
        adapted, _ = adapt_freeform_tools([EXEC])
        params = adapted[0]["parameters"]
        assert adapted[0]["type"] == "function"
        assert params["properties"][FREEFORM_PARAM]["type"] == "string"
        assert params["required"] == [FREEFORM_PARAM]

    def test_grammar_syntax_not_leaked_into_model_facing_text(self):
        """`format.syntax` is "lark" — the GRAMMAR constraining the input, not the
        language of the input (JavaScript). Telling the model to write "lark
        source" would point it at the wrong language entirely."""
        adapted, _ = adapt_freeform_tools([EXEC])
        assert "lark" not in adapted[0]["description"].lower()
        param_desc = adapted[0]["parameters"]["properties"][FREEFORM_PARAM]["description"]
        assert "lark" not in param_desc.lower()

    def test_original_description_preserved(self):
        """It carries the nested tools' TypeScript declarations, so the model
        still knows the real API surface."""
        adapted, _ = adapt_freeform_tools([EXEC])
        assert "Run JavaScript code to orchestrate/compose tool calls." in adapted[0]["description"]

    def test_function_tool_untouched(self):
        adapted, names = adapt_freeform_tools([WAIT])
        assert adapted == [WAIT]
        assert names == set()

    def test_format_without_parameters_treated_as_freeform(self):
        """Detection must not hinge on type == "custom" alone."""
        _, names = adapt_freeform_tools([
            {"name": "ff", "description": "d", "format": {"syntax": "lark"}}
        ])
        assert names == {"ff"}

    def test_unnamed_freeform_tool_dropped(self):
        adapted, names = adapt_freeform_tools([{"type": "custom", "format": {}}])
        assert adapted == []
        assert names == set()

    def test_kiro_receives_a_usable_schema(self):
        """The regression that mattered: exec used to reach Kiro as {}."""
        adapted, _ = adapt_freeform_tools([EXEC])
        spec = build_tool_specs(adapted)[0]["toolSpecification"]
        assert spec["inputSchema"]["json"] != {}
        assert FREEFORM_PARAM in spec["inputSchema"]["json"]["properties"]

    def test_unadapted_exec_would_reach_kiro_empty(self):
        """Documents the original defect, so it cannot quietly return."""
        spec = build_tool_specs([EXEC])[0]["toolSpecification"]
        assert spec["inputSchema"]["json"] == {}


class TestFreeformInputRecovery:

    def test_declared_shape_unwrapped(self):
        assert freeform_input_from_arguments('{"source":"text(1)"}') == "text(1)"

    def test_alias_keys_unwrapped(self):
        assert freeform_input_from_arguments('{"code":"a"}') == "a"
        assert freeform_input_from_arguments('{"script":"b"}') == "b"

    def test_nested_exec_command_args_wrapped_into_js(self):
        """The shape live traffic actually produced: with an empty schema the
        model imitated the exec_command(args: {cmd, workdir, ...}) declaration
        documented in exec's own prose description."""
        args = json.dumps({"cmd": "ls -la", "workdir": "/tmp",
                           "yield_time_ms": 10000, "max_output_tokens": 4000})
        src = freeform_input_from_arguments(args)
        lines = src.splitlines()
        assert lines[0].startswith("// @exec:")
        assert '"yield_time_ms": 10000' in lines[0]
        assert '"max_output_tokens": 4000' in lines[0]
        assert "await tools.exec_command(" in lines[1]
        assert '"cmd": "ls -la"' in lines[1]
        assert '"workdir": "/tmp"' in lines[1]
        # pragma-only keys must not remain in the call arguments
        assert "yield_time_ms" not in lines[1]
        assert "max_output_tokens" not in lines[1]
        assert src.rstrip().endswith("text(result);")

    def test_nested_args_without_pragma_emit_no_pragma_line(self):
        src = freeform_input_from_arguments('{"cmd":"pwd"}')
        assert not src.startswith("//")
        assert "await tools.exec_command(" in src

    def test_recovered_js_preserves_windows_paths(self):
        args = json.dumps({"cmd": "dir", "workdir": "D:\\PROJECTS\\AI"})
        src = freeform_input_from_arguments(args)
        call = src.splitlines()[0]
        assert json.loads(call[call.index("(") + 1:call.rindex(")")])["workdir"] == "D:\\PROJECTS\\AI"

    def test_bare_source_forwarded_verbatim(self):
        assert freeform_input_from_arguments("text(42)") == "text(42)"

    def test_invalid_json_forwarded_verbatim(self):
        """Inventing substitute source is worse than a real grammar error."""
        broken = '{"cmd":"C:\\P"}'  # invalid \P escape -> not valid JSON
        assert freeform_input_from_arguments(broken) == broken

    def test_json_string_literal_unwrapped(self):
        assert freeform_input_from_arguments('"text(7)"') == "text(7)"

    def test_empty_is_empty(self):
        assert freeform_input_from_arguments("") == ""
        assert freeform_input_from_arguments("   ") == ""

    def test_unknown_object_forwarded_verbatim(self):
        payload = '{"totally":"unexpected"}'
        assert freeform_input_from_arguments(payload) == payload
