"""Shared fixtures and helpers for door_anthropic tests."""

from __future__ import annotations

import json
from typing import AsyncIterable, List, Tuple

import pytest

from conduit.contracts.events import EventName, KiroEvent, UnknownEvent
from conduit.contracts.metrics import TurnMetrics


# ---------------------------------------------------------------------------
# pytest-asyncio configuration
# ---------------------------------------------------------------------------
# Each test module may also declare asyncio_mode; this conftest sets the
# session-wide default so individual tests don't need the decorator.

def pytest_configure(config):
    """Register asyncio_mode=auto so async tests run without decorators."""
    try:
        config.addinivalue_line(
            "markers", "asyncio: mark test as async"
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# SSE parsing helper
# ---------------------------------------------------------------------------

def parse_sse_stream(raw: str) -> List[Tuple[str, dict]]:
    """Parse a raw SSE stream string into a list of (event_type, data_dict)."""
    results = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        event_type = None
        data_str = None
        for line in lines:
            if line.startswith("event: "):
                event_type = line[len("event: "):].strip()
            elif line.startswith("data: "):
                data_str = line[len("data: "):].strip()
        if event_type and data_str:
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                data = {"_raw": data_str}
            results.append((event_type, data))
    return results


async def collect_sse(
    events: AsyncIterable,
    model: str = "claude-sonnet-4.6",
    message_id: str = "msg_test0001",
) -> Tuple[List[Tuple[str, dict]], TurnMetrics]:
    """Collect all SSE output from stream_to_anthropic_sse into parsed tuples."""
    from conduit.doors.anthropic_sse import stream_to_anthropic_sse

    metrics = TurnMetrics()
    raw_parts = []
    async for chunk in stream_to_anthropic_sse(
        events=events,
        model=model,
        message_id=message_id,
        metrics=metrics,
    ):
        raw_parts.append(chunk)
    raw = "".join(raw_parts)
    return parse_sse_stream(raw), metrics


# ---------------------------------------------------------------------------
# Synthetic event builders
# ---------------------------------------------------------------------------

async def make_text_events(text_chunks: List[str], stop_reason: str = "END_TURN"):
    """Simple text-only turn."""
    for chunk in text_chunks:
        yield KiroEvent(
            name=EventName.ASSISTANT_RESPONSE,
            payload={"content": chunk},
        )
    yield KiroEvent(
        name=EventName.METADATA,
        payload={
            "tokenUsage": {
                "inputTokens": 100,
                "outputTokens": 50,
                "cacheReadInputTokens": 10,
                "cacheWriteInputTokens": 5,
                "totalTokens": 150,
            }
        },
    )
    yield KiroEvent(
        name=EventName.MESSAGE_STOP,
        payload={"stopReason": stop_reason},
    )


async def make_thinking_events(
    thinking_chunks: List[str],
    signature: str,
    text_chunks: List[str],
):
    """Thinking-then-text turn."""
    for i, chunk in enumerate(thinking_chunks):
        payload: dict = {"content": chunk}
        # Attach signature on the last thinking chunk (as Kiro does)
        if i == len(thinking_chunks) - 1:
            payload["signature"] = signature
        yield KiroEvent(name=EventName.REASONING_CONTENT, payload=payload)
    for chunk in text_chunks:
        yield KiroEvent(
            name=EventName.ASSISTANT_RESPONSE, payload={"content": chunk}
        )
    yield KiroEvent(
        name=EventName.METADATA,
        payload={
            "tokenUsage": {
                "inputTokens": 200,
                "outputTokens": 80,
                "cacheReadInputTokens": 20,
                "cacheWriteInputTokens": 15,
                "totalTokens": 280,
            }
        },
    )
    yield KiroEvent(name=EventName.MESSAGE_STOP, payload={"stopReason": "END_TURN"})


async def make_tool_use_events(tool_id: str, tool_name: str, tool_input: dict):
    """Tool-use turn with one tool call."""
    yield KiroEvent(
        name=EventName.TOOL_USE,
        payload={"toolUseId": tool_id, "name": tool_name, "input": tool_input},
    )
    yield KiroEvent(
        name=EventName.METADATA,
        payload={
            "tokenUsage": {
                "inputTokens": 150,
                "outputTokens": 30,
                "cacheReadInputTokens": 0,
                "cacheWriteInputTokens": 0,
                "totalTokens": 180,
            }
        },
    )
    yield KiroEvent(
        name=EventName.MESSAGE_STOP, payload={"stopReason": "TOOL_USE"}
    )


# ---------------------------------------------------------------------------
# Fragmented tool use -- what the wire actually sends
# ---------------------------------------------------------------------------
# The fragments below are the real ones observed for a single Bash call
# (SAGA request req_01M2G3K0E9BPWP4MNM1KVE2W46): an opening frame carrying
# `name` with an empty input, eleven raw JSON string fragments, then a closing
# frame with `stop`. Kept verbatim rather than tidied so the test exercises the
# awkward split points -- mid-word ("rking dir" / "ec") and mid-token (", \"").

REAL_BASH_FRAGMENTS = [
    '{"command": ',
    '"pwd',
    ' && ls"',
    ', "',
    'description',
    '": "Show wo',
    'rking dir',
    'ec',
    'tory and c',
    'onte',
    'nts"}',
]


async def make_fragmented_tool_use_events(
    tool_id: str,
    tool_name: str,
    fragments: List[str],
    stop_reason: str = "TOOL_USE",
    emit_stop_frame: bool = True,
):
    """One tool call streamed as many toolUseEvent frames sharing one id."""
    yield KiroEvent(
        name=EventName.TOOL_USE,
        payload={"toolUseId": tool_id, "name": tool_name, "input": {}},
    )
    for frag in fragments:
        yield KiroEvent(
            name=EventName.TOOL_USE,
            payload={"toolUseId": tool_id, "name": tool_name, "input": frag},
        )
    if emit_stop_frame:
        yield KiroEvent(
            name=EventName.TOOL_USE,
            payload={
                "toolUseId": tool_id,
                "name": tool_name,
                "input": {},
                "stop": True,
            },
        )
    yield KiroEvent(
        name=EventName.METADATA,
        payload={
            "tokenUsage": {
                "inputTokens": 150,
                "outputTokens": 30,
                "totalTokens": 180,
            }
        },
    )
    yield KiroEvent(name=EventName.MESSAGE_STOP, payload={"stopReason": stop_reason})


async def make_two_fragmented_tool_calls():
    """Two tool calls back to back, each streamed as fragments."""
    for tool_id, frags in (
        ("toolu_first000000000000000000", ['{"command": ', '"pwd"}']),
        ("toolu_second00000000000000000", ['{"command": ', '"ls"}']),
    ):
        yield KiroEvent(
            name=EventName.TOOL_USE,
            payload={"toolUseId": tool_id, "name": "Bash", "input": {}},
        )
        for frag in frags:
            yield KiroEvent(
                name=EventName.TOOL_USE,
                payload={"toolUseId": tool_id, "name": "Bash", "input": frag},
            )
        yield KiroEvent(
            name=EventName.TOOL_USE,
            payload={
                "toolUseId": tool_id,
                "name": "Bash",
                "input": {},
                "stop": True,
            },
        )
    yield KiroEvent(name=EventName.MESSAGE_STOP, payload={"stopReason": "TOOL_USE"})
