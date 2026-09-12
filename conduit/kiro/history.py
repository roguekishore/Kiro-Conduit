"""
Turn-merge for the Kiro history pipeline.

Consecutive same-role messages are merged into a single turn. The upstream
gateway inserts synthetic "(empty placeholder)" assistant turns to satisfy
Kiro's alternation requirement; that padding pollutes history with text the
model never wrote. Merging instead of padding avoids that.

Public API
----------
- HistoryMessage  — internal message dataclass for Lane B
- merge_turns()   — merge consecutive same-role messages, return injections
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from conduit.contracts.rewritten import Injection, InjectionType


@dataclass
class HistoryMessage:
    """
    Internal wire-agnostic message for the Kiro request pipeline.

    ``content`` holds the already-extracted text.  ``tool_calls`` and
    ``tool_results`` are kept as plain dicts so the caller (request.py) can
    pass them to tools.py without an intermediate conversion step.

    ``images`` is a list of ``{"media_type": str, "data": str}`` dicts
    (base64 only; URL images are reported by the caller before reaching here).
    """

    role: str  # "user" | "assistant"
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = field(default=None)
    tool_results: Optional[List[Dict[str, Any]]] = field(default=None)
    images: Optional[List[Dict[str, Any]]] = field(default=None)


def merge_turns(
    messages: List[HistoryMessage],
) -> Tuple[List[HistoryMessage], List[Injection]]:
    """
    Merge consecutive same-role turns into a single ``HistoryMessage``.

    When the immediately preceding merged message has the same role as the
    incoming one, their text contents are joined with a single newline, and
    their tool_calls / tool_results / images lists are concatenated.

    A single ``Injection(TURN_MERGE, "history")`` is appended to the returned
    list if *any* merges occurred; it is omitted when nothing was merged.

    Args:
        messages: Flat list of messages, already in ``HistoryMessage`` form.

    Returns:
        ``(merged_messages, injections)``
    """
    if not messages:
        return [], []

    merged: List[HistoryMessage] = []
    merge_count = 0

    for msg in messages:
        if merged and merged[-1].role == msg.role:
            last = merged[-1]

            # Text content — join with newline; skip empty parts
            parts = [p for p in (last.content, msg.content) if p]
            last.content = "\n".join(parts)

            # tool_calls (assistant)
            if msg.tool_calls:
                if last.tool_calls is None:
                    last.tool_calls = []
                last.tool_calls = list(last.tool_calls) + list(msg.tool_calls)

            # tool_results (user)
            if msg.tool_results:
                if last.tool_results is None:
                    last.tool_results = []
                last.tool_results = list(last.tool_results) + list(msg.tool_results)

            # images
            if msg.images:
                if last.images is None:
                    last.images = []
                last.images = list(last.images) + list(msg.images)

            merge_count += 1
            logger.debug(
                f"Merged consecutive {msg.role!r} turn into previous "
                f"(total merges so far: {merge_count})"
            )
        else:
            # Copy so downstream mutations don't affect the caller's objects
            merged.append(
                HistoryMessage(
                    role=msg.role,
                    content=msg.content,
                    tool_calls=list(msg.tool_calls) if msg.tool_calls else None,
                    tool_results=list(msg.tool_results) if msg.tool_results else None,
                    images=list(msg.images) if msg.images else None,
                )
            )

    injections: List[Injection] = []
    if merge_count > 0:
        injections.append(Injection(type=InjectionType.TURN_MERGE, location="history"))
        logger.debug(f"merge_turns: merged {merge_count} consecutive same-role turn(s)")

    return merged, injections
