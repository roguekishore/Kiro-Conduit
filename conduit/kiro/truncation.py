# -*- coding: utf-8 -*-

# CONDUIT
# Copyright (C) 2025 Jwadow (original kiro-gateway truncation_state.py /
#                            truncation_recovery.py)
# Copyright (C) 2026 CONDUIT contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Truncation detection and cross-request recovery.

Kiro cuts off large tool-call arguments and long assistant content mid-stream.
It signals this with nothing at all: the stream simply ends. The client sees a
half-written tool call or a sentence stopping mid-word, and the model — on the
next request — sees its own truncated output as though it had chosen to stop
there. It then retries the identical operation and is truncated identically.

This module closes that loop in two halves, because detection and recovery
happen on different requests:

- **Detect** (response path): ``diagnose_json_truncation`` decides whether a
  tool-call argument blob was cut off rather than merely malformed;
  ``record_tool_truncation`` / ``record_content_truncation`` remember it.
- **Recover** (next request path): ``take_tool_truncation`` /
  ``take_content_truncation`` retrieve and clear the record, and the
  ``notice_*`` helpers produce the text that tells the model what happened.

Records are keyed by values stable across requests — the tool-use id, and a
hash of the truncated text — because there is no session identifier to hang
them on. Retrieval is destructive: a notice is delivered exactly once.

State is process-local and in-memory. It is lost on restart, and a multi-worker
deployment will only find a record in the worker that wrote it. That is
acceptable: a missed notice degrades to today's behaviour, and the alternative
is a persistence dependency for advisory text.

Divergence from upstream: upstream's recovery notices carry no marker
distinguishing them from model-authored text, and it separately injects a
system-prompt paragraph telling the model these notices are trustworthy. That
paragraph is itself an injection into every request, present whether or not any
truncation ever occurs. Here the notices are self-labelling and no standing
system-prompt text is added — nothing is injected on a healthy turn.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class ToolTruncation:
    """A tool call whose argument JSON arrived cut off."""
    tool_use_id: str
    tool_name: str
    size_bytes: int
    reason: str
    detected_at: float


@dataclass
class ContentTruncation:
    """An assistant text response that ended without completion signals."""
    content_hash: str
    preview: str
    detected_at: float


# ---------------------------------------------------------------------------
# Process-local store
# ---------------------------------------------------------------------------

_tool_truncations: Dict[str, ToolTruncation] = {}
_content_truncations: Dict[str, ContentTruncation] = {}
_lock = Lock()

# Cap the stores so a pathological run cannot grow them without bound. Records
# are normally consumed on the very next request; anything still here after
# this many entries is stale.
_MAX_RECORDS = 256


def _hash_content(content: str) -> str:
    """Stable short hash of a content prefix, used as a cross-request key."""
    return hashlib.sha256(content[:500].encode("utf-8")).hexdigest()[:16]


def _evict_oldest(store: Dict[str, Any]) -> None:
    """Drop the oldest record when a store is at capacity."""
    if len(store) < _MAX_RECORDS:
        return
    oldest = min(store, key=lambda k: store[k].detected_at)
    del store[oldest]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def diagnose_json_truncation(json_str: str) -> Dict[str, Any]:
    """
    Decide whether a JSON string that failed to parse was *truncated*.

    The distinction matters: a truncated blob is an upstream size limit and
    retrying identically will fail identically, while genuinely malformed JSON
    is a model error that a retry may fix. Telling the model the wrong one sends
    it down the wrong path.

    Detection is structural — unbalanced braces/brackets, an unterminated string
    literal, a value that starts an object but never closes it. Quote counting
    skips escaped characters.

    Args:
        json_str: The raw string that failed ``json.loads``.

    Returns:
        ``{"is_truncated": bool, "reason": str, "size_bytes": int}``
    """
    size_bytes = len(json_str.encode("utf-8"))
    stripped = json_str.strip()

    if not stripped:
        return {"is_truncated": False, "reason": "empty string", "size_bytes": size_bytes}

    open_braces = stripped.count("{")
    close_braces = stripped.count("}")
    open_brackets = stripped.count("[")
    close_brackets = stripped.count("]")

    if stripped.startswith("{") and not stripped.endswith("}"):
        return {
            "is_truncated": True,
            "reason": f"missing {open_braces - close_braces} closing brace(s)",
            "size_bytes": size_bytes,
        }

    if stripped.startswith("[") and not stripped.endswith("]"):
        return {
            "is_truncated": True,
            "reason": f"missing {open_brackets - close_brackets} closing bracket(s)",
            "size_bytes": size_bytes,
        }

    if open_braces != close_braces:
        return {
            "is_truncated": True,
            "reason": f"unbalanced braces ({open_braces} open, {close_braces} close)",
            "size_bytes": size_bytes,
        }

    if open_brackets != close_brackets:
        return {
            "is_truncated": True,
            "reason": f"unbalanced brackets ({open_brackets} open, {close_brackets} close)",
            "size_bytes": size_bytes,
        }

    # Unterminated string literal: count quotes, skipping escapes.
    quote_count = 0
    i = 0
    while i < len(stripped):
        ch = stripped[i]
        if ch == "\\":
            i += 2
            continue
        if ch == '"':
            quote_count += 1
        i += 1

    if quote_count % 2 != 0:
        return {
            "is_truncated": True,
            "reason": "unclosed string literal",
            "size_bytes": size_bytes,
        }

    return {"is_truncated": False, "reason": "malformed JSON", "size_bytes": size_bytes}


# ---------------------------------------------------------------------------
# Recording (response path)
# ---------------------------------------------------------------------------

def record_tool_truncation(
    tool_use_id: str,
    tool_name: str,
    size_bytes: int,
    reason: str,
) -> None:
    """Remember a truncated tool call, keyed by its tool-use id."""
    if not tool_use_id:
        return
    with _lock:
        _evict_oldest(_tool_truncations)
        _tool_truncations[tool_use_id] = ToolTruncation(
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            size_bytes=size_bytes,
            reason=reason,
            detected_at=time.time(),
        )
    logger.warning(
        f"[truncation] tool {tool_name!r} (id={tool_use_id}) was truncated by Kiro: "
        f"{reason}, {size_bytes} bytes. The model will be told on its next request."
    )


def record_content_truncation(content: str) -> str:
    """
    Remember a truncated assistant text response, keyed by a hash of its text.

    Returns:
        The content hash used as the key.
    """
    content_hash = _hash_content(content)
    with _lock:
        _evict_oldest(_content_truncations)
        _content_truncations[content_hash] = ContentTruncation(
            content_hash=content_hash,
            preview=content[:200],
            detected_at=time.time(),
        )
    logger.warning(
        f"[truncation] assistant content truncated by Kiro "
        f"({len(content)} chars, hash={content_hash}). "
        "The model will be told on its next request."
    )
    return content_hash


# ---------------------------------------------------------------------------
# Retrieval (next-request path) — destructive, one notice per truncation
# ---------------------------------------------------------------------------

def take_tool_truncation(tool_use_id: str) -> Optional[ToolTruncation]:
    """Retrieve and clear the record for a tool-use id, if any."""
    if not tool_use_id:
        return None
    with _lock:
        return _tool_truncations.pop(tool_use_id, None)


def take_content_truncation(content: str) -> Optional[ContentTruncation]:
    """Retrieve and clear the record matching this assistant text, if any."""
    if not content:
        return None
    with _lock:
        return _content_truncations.pop(_hash_content(content), None)


# ---------------------------------------------------------------------------
# Notice text
# ---------------------------------------------------------------------------

def notice_for_tool(truncation: ToolTruncation) -> str:
    """
    Text prepended to a tool result whose originating call was truncated.

    Worded to establish three things and nothing more: the cause was an upstream
    limit rather than a mistake, any error in the result below is downstream of
    that, and an identical retry will fail identically. No specific remedy is
    prescribed — upstream's experience was that telling the model to "break this
    into steps" produced pathological micro-stepping.
    """
    return (
        f"[CONDUIT] The call to `{truncation.tool_name}` was cut off in transit by "
        f"an upstream size limit ({truncation.size_bytes} bytes, {truncation.reason}). "
        "Its arguments never arrived complete.\n\n"
        "Any error below is a consequence of that truncation, not its cause. "
        "Repeating this call unchanged will truncate again; a different approach will not."
    )


def notice_for_content() -> str:
    """Text for a synthetic user turn following truncated assistant output."""
    return (
        "[CONDUIT] Your previous response was cut off by an upstream output limit, "
        "not by anything you did. If you were mid-task, continue from where it stopped "
        "rather than restarting the same output."
    )


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

def pending_counts() -> Dict[str, int]:
    """Current record counts, for logging and tests."""
    with _lock:
        return {
            "tool": len(_tool_truncations),
            "content": len(_content_truncations),
            "total": len(_tool_truncations) + len(_content_truncations),
        }


def reset() -> None:
    """Clear all records. For tests."""
    with _lock:
        _tool_truncations.clear()
        _content_truncations.clear()
