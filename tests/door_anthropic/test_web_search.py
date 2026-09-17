"""Web search: continuation to the model, and replay of search blocks.

Covers the two defects these tests were written against:

1. Both search paths used to return the results and STOP, so the model never
   saw them - the turn ended with search output and no answer.
2. The gateway emitted ``server_tool_use`` / ``web_search_tool_result`` blocks
   that nothing downstream accepted back, so replaying them either 422'd
   (old gateway) or silently dropped the results (CONDUIT).
"""

from __future__ import annotations

import json

import pytest

from conduit.contracts.metrics import TurnMetrics
from conduit.doors.anthropic import (
    AnthropicMessagesRequest,
    _augment_messages_for_search,
    _close_message_with_text,
    _search_block_frames,
    _strip_web_search_tools,
    _to_history_messages,
)
from conduit.kiro.mcp import render_replayed_search

from .conftest import make_text_events, parse_sse_stream

MCP_RESULTS = {
    "results": [
        {
            "title": "GLM-5 model card",
            "url": "https://hf.co/zai-org/GLM-5",
            "snippet": "77.8% SWE-bench Verified",
        },
        {
            "title": "Ollama GLM-5",
            "url": "https://ollama.com/library/glm-5",
            "snippet": "56.2 Terminal-Bench 2.0",
        },
    ]
}


def _req(**kw) -> AnthropicMessagesRequest:
    payload = {
        "model": "claude-sonnet-4.6",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(kw)
    return AnthropicMessagesRequest.model_validate(payload)


# ---------------------------------------------------------------------------
# Folding results back into the conversation
# ---------------------------------------------------------------------------

class TestAugmentMessages:
    """The step that makes search inform the answer instead of replacing it."""

    def test_path_b_uses_native_tool_result_round_trip(self):
        # Path B: the MODEL called the injected tool, so it is waiting for a
        # tool_result. Anything else leaves the call unanswered.
        req = _req()
        _augment_messages_for_search(req, "glm-5", "SUMMARY", tool_id="tu_1")

        assert len(req.messages) == 3
        assistant, user = req.messages[1], req.messages[2]

        assert assistant.role == "assistant"
        block = assistant.content[0]
        assert block["type"] == "tool_use"
        assert block["id"] == "tu_1"
        assert block["name"] == "web_search"
        assert block["input"] == {"query": "glm-5"}

        assert user.role == "user"
        result = user.content[0]
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "tu_1"
        assert result["content"] == "SUMMARY"

    def test_path_a_appends_to_last_user_turn(self):
        # Path A has no client tool call to answer, so the summary rides along
        # on the last user turn. A new trailing user turn would be wrong: Kiro
        # requires user/assistant alternation.
        req = _req(messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "what are GLM-5 scores?"},
        ])
        _augment_messages_for_search(req, "glm-5", "SUMMARY")

        assert len(req.messages) == 3
        assert req.messages[2].role == "user"
        assert "what are GLM-5 scores?" in req.messages[2].content
        assert "SUMMARY" in req.messages[2].content

    def test_path_a_appends_block_to_list_content(self):
        req = _req(messages=[
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
        ])
        _augment_messages_for_search(req, "q", "SUMMARY")

        blocks = req.messages[0].content
        assert blocks[-1] == {"type": "text", "text": "SUMMARY"}

    def test_path_a_with_no_user_turn_adds_one(self):
        req = _req(messages=[{"role": "assistant", "content": "orphan"}])
        _augment_messages_for_search(req, "q", "SUMMARY")

        assert req.messages[-1].role == "user"
        assert "SUMMARY" in req.messages[-1].content[0]["text"]


class TestStripWebSearchTools:
    """The search already ran; re-advertising it strands the next tool call."""

    def test_removes_by_name_and_by_type(self):
        req = _req(tools=[
            {"name": "web_search", "type": "web_search_20250305"},
            {"name": "Bash", "input_schema": {"type": "object"}},
        ])
        _strip_web_search_tools(req)

        assert req.tools is not None
        assert [t.name for t in req.tools] == ["Bash"]

    def test_tools_becomes_none_when_only_search_was_offered(self):
        # None, not [] - an empty tools array is a different statement to Kiro
        # than no tools at all.
        req = _req(tools=[{"name": "web_search", "type": "web_search_20250305"}])
        _strip_web_search_tools(req)
        assert req.tools is None

    def test_no_tools_is_a_noop(self):
        req = _req()
        _strip_web_search_tools(req)
        assert req.tools is None


# ---------------------------------------------------------------------------
# Wire shape
# ---------------------------------------------------------------------------

class TestSearchBlockFrames:
    def test_indices_do_not_collide_and_carry_results(self):
        frames = parse_sse_stream("".join(
            _search_block_frames(3, "srvtoolu_9", "glm-5", MCP_RESULTS)
        ))
        types = [t for t, _ in frames]
        assert types == [
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_stop",
        ]

        # server_tool_use at the given index, result at index+1.
        assert frames[0][1]["index"] == 3
        assert frames[0][1]["content_block"]["type"] == "server_tool_use"
        assert frames[0][1]["content_block"]["id"] == "srvtoolu_9"
        assert json.loads(frames[1][1]["delta"]["partial_json"]) == {"query": "glm-5"}
        assert frames[3][1]["index"] == 4
        result_block = frames[3][1]["content_block"]
        assert result_block["type"] == "web_search_tool_result"
        assert result_block["tool_use_id"] == "srvtoolu_9"

        # MCP calls the text `snippet`; the Anthropic wire field is
        # `encrypted_content`. Getting this mapping wrong loses the results.
        assert result_block["content"][0]["encrypted_content"] == (
            "77.8% SWE-bench Verified"
        )
        assert len(result_block["content"]) == 2


class TestContinuationStreamShape:
    """A message may carry exactly one message_start and one message_stop."""

    @pytest.mark.asyncio
    async def test_continuation_suppresses_message_start_and_offsets_indices(self):
        from conduit.doors.anthropic_sse import stream_to_anthropic_sse

        metrics = TurnMetrics()
        raw = []
        async for chunk in stream_to_anthropic_sse(
            events=make_text_events(["answer"]),
            model="claude-sonnet-4.6",
            message_id="msg_x",
            metrics=metrics,
            start_index=2,
            emit_message_start=False,
        ):
            raw.append(chunk)

        events = parse_sse_stream("".join(raw))
        types = [t for t, _ in events]

        assert "message_start" not in types
        assert types.count("message_stop") == 1

        # Text must not reuse index 0/1 - the search blocks own those.
        starts = [d["index"] for t, d in events if t == "content_block_start"]
        assert starts and min(starts) == 2

    @pytest.mark.asyncio
    async def test_default_still_emits_message_start_at_index_zero(self):
        from conduit.doors.anthropic_sse import stream_to_anthropic_sse

        metrics = TurnMetrics()
        raw = []
        async for chunk in stream_to_anthropic_sse(
            events=make_text_events(["hi"]),
            model="claude-sonnet-4.6",
            message_id="msg_y",
            metrics=metrics,
        ):
            raw.append(chunk)

        events = parse_sse_stream("".join(raw))
        types = [t for t, _ in events]
        assert types[0] == "message_start"
        assert types.count("message_start") == 1
        starts = [d["index"] for t, d in events if t == "content_block_start"]
        assert min(starts) == 0


class TestCloseMessageWithText:
    """Search succeeded, continuation failed: still close a valid message."""

    @pytest.mark.asyncio
    async def test_emits_terminated_text_block(self):
        raw = []
        async for frame in _close_message_with_text(2, "search ok, model failed"):
            raw.append(frame)

        events = parse_sse_stream("".join(raw))
        types = [t for t, _ in events]
        assert types == [
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert all(
            d["index"] == 2
            for t, d in events
            if t.startswith("content_block")
        )
        assert "model failed" in events[1][1]["delta"]["text"]


# ---------------------------------------------------------------------------
# Replay: the 422 / silent-drop defect
# ---------------------------------------------------------------------------

REPLAYED = [
    {"type": "text", "text": "Let me search."},
    {
        "type": "server_tool_use",
        "id": "srvtoolu_1",
        "name": "web_search",
        "input": {"query": "GLM-5 SWE-bench"},
    },
    {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_1",
        "content": [
            {
                "type": "web_search_result",
                "title": "GLM-5 card",
                "url": "https://hf.co/zai-org/GLM-5",
                "encrypted_content": "77.8% SWE-bench Verified",
            }
        ],
    },
]


class TestRenderReplayedSearch:
    def test_renders_title_url_and_snippet(self):
        out = render_replayed_search(REPLAYED[2]["content"])
        assert "GLM-5 card" in out
        assert "https://hf.co/zai-org/GLM-5" in out
        assert "77.8% SWE-bench Verified" in out

    @pytest.mark.parametrize("payload", [None, [], "", {}])
    def test_empty_shapes_render_nothing(self, payload):
        assert render_replayed_search(payload) == ""

    def test_error_object_does_not_raise(self):
        out = render_replayed_search(
            {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}
        )
        assert "max_uses_exceeded" in out

    def test_bare_string_is_wrapped(self):
        assert "raw text" in render_replayed_search("raw text")


class TestHistoryTranslation:
    """Replayed search must survive the trip into the Kiro payload."""

    def test_search_results_reach_history_text(self):
        req = _req(messages=[
            {"role": "user", "content": "what are GLM-5 scores?"},
            {"role": "assistant", "content": REPLAYED},
            {"role": "user", "content": "and Qwen?"},
        ])
        history = _to_history_messages(req.messages)

        assistant = history[1]
        # Before the fix these blocks matched no branch and vanished, so the
        # model saw an assistant turn that searched for nothing.
        assert "GLM-5 SWE-bench" in assistant.content
        assert "77.8% SWE-bench Verified" in assistant.content
        assert "Let me search." in assistant.content

    def test_ordinary_turns_are_unaffected(self):
        req = _req(messages=[
            {"role": "user", "content": "plain"},
            {"role": "assistant", "content": [{"type": "text", "text": "reply"}]},
        ])
        history = _to_history_messages(req.messages)
        assert history[0].content == "plain"
        assert history[1].content == "reply"

    def test_server_tool_use_without_query_adds_no_marker(self):
        req = _req(messages=[
            {"role": "assistant", "content": [
                {"type": "server_tool_use", "id": "s", "name": "web_search",
                 "input": {}},
            ]},
        ])
        history = _to_history_messages(req.messages)
        assert history[0].content == ""
