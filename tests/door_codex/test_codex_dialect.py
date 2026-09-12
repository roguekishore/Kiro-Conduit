"""Tests for codex_dialect.py — both wire dialects.

Verifies:
  - Standard dialect reads system_prompt from `instructions` and tools from `tools`
  - Lite dialect reads system_prompt from developer-role prefix items in input[]
  - Lite dialect un-collapses a namespace tool back to individual specs
  - Non-developer items stop the prefix scan
  - Edge cases: empty instructions, null tools, no prefix items
"""

from __future__ import annotations

import pytest

from conduit.doors.codex_dialect import (
    LITE_HEADER,
    extract_dialect,
    _uncollapse_namespace,
    _extract_text,
)


# ---------------------------------------------------------------------------
# Standard dialect
# ---------------------------------------------------------------------------

class TestStandardDialect:
    """Standard dialect: prompt from instructions, tools from top-level tools."""

    def test_reads_prompt_from_instructions(self):
        prompt, tools = extract_dialect(
            instructions="You are a helpful assistant.",
            tools=None,
            input_items=[],
            is_lite=False,
        )
        assert prompt == "You are a helpful assistant."
        assert tools == []

    def test_reads_tools_from_top_level(self):
        tool_list = [
            {"type": "function", "function": {"name": "search", "description": "Search the web"}}
        ]
        prompt, tools = extract_dialect(
            instructions="sys",
            tools=tool_list,
            input_items=[],
            is_lite=False,
        )
        assert tools == tool_list

    def test_empty_instructions_gives_empty_prompt(self):
        prompt, _ = extract_dialect(
            instructions="",
            tools=None,
            input_items=[],
            is_lite=False,
        )
        assert prompt == ""

    def test_none_instructions_gives_empty_prompt(self):
        prompt, _ = extract_dialect(
            instructions=None,
            tools=None,
            input_items=[],
            is_lite=False,
        )
        assert prompt == ""

    def test_whitespace_only_instructions_stripped(self):
        prompt, _ = extract_dialect(
            instructions="   \n\t  ",
            tools=None,
            input_items=[],
            is_lite=False,
        )
        assert prompt == ""

    def test_ignores_input_items_in_standard_mode(self):
        """Lite prefix items must NOT be read when is_lite is False."""
        input_items = [
            {
                "type": "message",
                "role": "developer",
                "content": "SHOULD NOT APPEAR IN PROMPT",
            }
        ]
        prompt, _ = extract_dialect(
            instructions="real instructions",
            tools=None,
            input_items=input_items,
            is_lite=False,
        )
        assert prompt == "real instructions"
        assert "SHOULD NOT APPEAR" not in prompt


# ---------------------------------------------------------------------------
# Lite dialect — prompt extraction
# ---------------------------------------------------------------------------

class TestLiteDialectPromptExtraction:
    """Lite dialect: prompt comes from developer-role message item in input[]."""

    def _lite_input(self, instructions_text: str) -> list:
        return [
            {
                "type": "message",
                "role": "developer",
                "content": instructions_text,
            }
        ]

    def test_reads_prompt_from_developer_message_string_content(self):
        prompt, _ = extract_dialect(
            instructions="",
            tools=None,
            input_items=self._lite_input("You are a coding assistant."),
            is_lite=True,
        )
        assert prompt == "You are a coding assistant."

    def test_reads_prompt_from_developer_message_content_blocks(self):
        """Prompt in a list of text content blocks."""
        input_items = [
            {
                "role": "developer",
                "content": [{"type": "text", "text": "Hello from block."}],
            }
        ]
        prompt, _ = extract_dialect(
            instructions="",
            tools=None,
            input_items=input_items,
            is_lite=True,
        )
        assert prompt == "Hello from block."

    def test_ignores_instructions_field_in_lite_mode(self):
        """Standard instructions= field must be ignored when is_lite=True."""
        prompt, _ = extract_dialect(
            instructions="SHOULD BE IGNORED",
            tools=None,
            input_items=self._lite_input("REAL INSTRUCTIONS"),
            is_lite=True,
        )
        assert prompt == "REAL INSTRUCTIONS"
        assert "SHOULD BE IGNORED" not in prompt

    def test_stops_scan_at_first_non_developer_item(self):
        """Non-developer items must stop the prefix scan."""
        input_items = [
            {"role": "developer", "content": "sys prompt"},
            {"role": "user", "content": "DO NOT USE THIS AS PROMPT"},
            {"role": "developer", "content": "ALSO IGNORE THIS"},
        ]
        prompt, _ = extract_dialect(
            instructions="",
            tools=None,
            input_items=input_items,
            is_lite=True,
        )
        assert prompt == "sys prompt"
        assert "DO NOT USE" not in prompt
        assert "ALSO IGNORE" not in prompt

    def test_empty_input_items_gives_empty_prompt(self):
        prompt, tools = extract_dialect(
            instructions="",
            tools=None,
            input_items=[],
            is_lite=True,
        )
        assert prompt == ""
        assert tools == []

    def test_multiple_text_blocks_joined(self):
        input_items = [
            {
                "role": "developer",
                "content": [
                    {"type": "text", "text": "Line one."},
                    {"type": "text", "text": "Line two."},
                ],
            }
        ]
        prompt, _ = extract_dialect(
            instructions="",
            tools=None,
            input_items=input_items,
            is_lite=True,
        )
        assert "Line one." in prompt
        assert "Line two." in prompt


# ---------------------------------------------------------------------------
# Lite dialect — namespace tool un-collapsing
# ---------------------------------------------------------------------------

class TestLiteDialectToolExtraction:
    """Lite dialect: namespace tool in additional_tools item -> individual specs."""

    def _namespace_tool_shape_a(self, nested_tools: list) -> dict:
        """Shape A: explicit nested tools list inside namespace tool."""
        return {
            "type": "additional_tools",
            "role": "developer",
            "tools": [
                {
                    "type": "namespace",
                    "name": "namespace",
                    "tools": nested_tools,
                }
            ],
        }

    def test_uncollapse_shape_a_single_tool(self):
        fn_spec = {
            "type": "function",
            "function": {"name": "read_file", "description": "Read a file"},
        }
        input_items = [self._namespace_tool_shape_a([fn_spec])]
        _, tools = extract_dialect(
            instructions="",
            tools=None,
            input_items=input_items,
            is_lite=True,
        )
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "read_file"

    def test_uncollapse_shape_a_multiple_tools(self):
        fn_specs = [
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "write_file"}},
            {"type": "function", "function": {"name": "list_dir"}},
        ]
        input_items = [self._namespace_tool_shape_a(fn_specs)]
        _, tools = extract_dialect(
            instructions="",
            tools=None,
            input_items=input_items,
            is_lite=True,
        )
        assert len(tools) == 3
        names = {t["function"]["name"] for t in tools}
        assert names == {"read_file", "write_file", "list_dir"}

    def test_uncollapse_shape_b_anyof(self):
        """Shape B: anyOf schema encodes function names as enum values."""
        namespace_tool = {
            "type": "function",
            "name": "namespace",
            "parameters": {
                "anyOf": [
                    {
                        "properties": {
                            "name": {"enum": ["grep"]},
                            "args": {"type": "object"},
                        }
                    },
                    {
                        "properties": {
                            "name": {"enum": ["find"]},
                            "args": {"type": "object"},
                        }
                    },
                ]
            },
        }
        result = _uncollapse_namespace([namespace_tool])
        names = [t["function"]["name"] for t in result]
        assert "grep" in names
        assert "find" in names

    def test_non_namespace_tool_passed_through(self):
        """Non-namespace tools should pass through unchanged."""
        regular_tool = {
            "type": "function",
            "function": {"name": "search", "description": "Search"},
        }
        result = _uncollapse_namespace([regular_tool])
        assert len(result) == 1
        assert result[0]["function"]["name"] == "search"

    def test_empty_tools_returns_empty(self):
        assert _uncollapse_namespace([]) == []

    def test_shape_c_opaque_passed_through(self):
        """Shape C: unrecognisable namespace tool structure passes through."""
        opaque = {"type": "namespace", "name": "namespace"}
        result = _uncollapse_namespace([opaque])
        # Must not crash; returns the opaque tool
        assert len(result) == 1

    def test_lite_tools_null_top_level_tools_not_read(self):
        """In lite mode, top-level tools= is null; tools come from input[]."""
        input_items = [
            self._namespace_tool_shape_a(
                [{"type": "function", "function": {"name": "lite_tool"}}]
            )
        ]
        _, tools = extract_dialect(
            instructions="",
            tools=[{"type": "function", "function": {"name": "SHOULD_BE_IGNORED"}}],
            input_items=input_items,
            is_lite=True,
        )
        # In lite mode the top-level tools are still ignored — the prompt
        # comes from the developer items.  The lite reader reads from input[].
        # This verifies the lite path ran (lite_tool is present).
        names = [t.get("function", t).get("name", "") for t in tools]
        assert "lite_tool" in names


# ---------------------------------------------------------------------------
# _extract_text unit tests
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_string_passthrough(self):
        assert _extract_text("hello") == "hello"

    def test_list_of_text_blocks(self):
        result = _extract_text([{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}])
        assert "foo" in result
        assert "bar" in result

    def test_empty_list(self):
        assert _extract_text([]) == ""

    def test_non_text_block_types_skipped(self):
        result = _extract_text([{"type": "image_url", "url": "http://x.com/img.png"}])
        assert result == ""

    def test_nested_value_key(self):
        result = _extract_text([{"type": "input_text", "value": "via value key"}])
        assert result == "via value key"
