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
