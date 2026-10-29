# -*- coding: utf-8 -*-

# CONDUIT
# Copyright (C) 2025 Jwadow (original kiro-gateway payload_guards.py)
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
Protocol and size guards for the assembled Kiro payload.

Two independent concerns, deliberately separated because they have different
default states:

1. **Orphaned tool-result repair** — ALWAYS runs.  A ``toolResults`` entry whose
   ``toolUseId`` is absent from the immediately preceding assistant turn makes
   Kiro reject the whole request with the misleading
   ``"Improperly formed request." (reason: null)``.  This happens naturally after
   a conversation is compacted, resumed, or forked mid-tool-call.

2. **Payload trimming** — runs only when ``AUTO_TRIM_PAYLOAD`` is set.  Kiro
   rejects payloads over ~615KB with the same misleading error.  Dropping
   history is lossy, so it stays opt-in.

Divergence from upstream
------------------------
Upstream (``kiro/payload_guards.py``) calls ``_repair_orphaned_tool_results``
**only** from inside ``trim_payload_to_limit``, which in turn runs only when
``AUTO_TRIM_PAYLOAD`` is true — and that flag defaults to ``false``.  The repair
is therefore dead code in a default upstream deployment: the exact malformed
history it was written to fix still reaches Kiro and still 400s.

Here the repair is a separate, unconditional pass (``enforce_tool_result_integrity``).
Trimming calls it again after dropping entries, because trimming can itself
orphan a tool result by removing the assistant turn that declared it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class PayloadTrimStats:
    """Outcome of a payload trim operation."""
    original_bytes: int
    final_bytes: int
    original_entries: int
    final_entries: int
    trimmed: bool


@dataclass
class OrphanRepairStats:
    """Outcome of an orphaned-tool-result repair pass."""
    orphans_removed: int = 0
    entries_touched: int = 0

    @property
    def repaired(self) -> bool:
        return self.orphans_removed > 0


# ---------------------------------------------------------------------------
# Size measurement
# ---------------------------------------------------------------------------

def check_payload_size(payload: Dict[str, Any]) -> int:
    """Return the serialized byte size of the payload as compact UTF-8 JSON."""
    return len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_empty_tool_uses(history: List[Dict[str, Any]]) -> None:
    """Drop empty ``toolUses`` arrays in-place; Kiro rejects them."""
    for entry in history:
        assistant = entry.get("assistantResponseMessage")
        if assistant and assistant.get("toolUses") == []:
            del assistant["toolUses"]


def _align_to_user_message(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop leading entries until history starts with a ``userInputMessage``."""
    while history and "userInputMessage" not in history[0]:
        history.pop(0)
    return history


def _declared_tool_use_ids(entry: Optional[Dict[str, Any]]) -> set:
    """Collect the ``toolUseId`` values an assistant history entry declares."""
    if not entry:
        return set()
    assistant = entry.get("assistantResponseMessage")
    if not assistant:
        return set()
    ids = set()
    for tu in assistant.get("toolUses") or []:
        tool_use_id = tu.get("toolUseId")
        if tool_use_id:
            ids.add(tool_use_id)
    return ids


def _extract_result_text(content: Any) -> List[str]:
    """Pull human-readable text out of a Kiro toolResult ``content`` value."""
    parts: List[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("text"):
                parts.append(str(part["text"]))
            elif isinstance(part, str) and part:
                parts.append(part)
    elif isinstance(content, str) and content:
        parts.append(content)
    return parts


# ---------------------------------------------------------------------------
# Orphan repair (unconditional)
# ---------------------------------------------------------------------------

def enforce_tool_result_integrity(payload: Dict[str, Any]) -> OrphanRepairStats:
    """
    Remove ``toolResults`` whose ``toolUseId`` no preceding assistant turn declared.

    Kiro pairs a tool result to the tool call that requested it.  An unpaired
    result is a protocol violation and the whole request is rejected.  Rather
    than discarding the result's text along with it, the text is appended to the
    user turn's own ``content`` behind a ``[trimmed tool result]`` marker, so the
    model still sees what the tool returned.

    Operates in-place on ``payload["conversationState"]["history"]``.  The
    ``currentMessage`` is deliberately NOT repaired: its tool results pair
    against the last history entry, and rewriting the live turn would hide a
    genuine caller bug.

    Returns:
        OrphanRepairStats describing what was removed.
    """
    stats = OrphanRepairStats()
    history = payload.get("conversationState", {}).get("history")
    if not history:
        return stats

    for i, entry in enumerate(history):
        user_msg = entry.get("userInputMessage")
        if not user_msg:
            continue

        ctx = user_msg.get("userInputMessageContext")
        if not ctx or "toolResults" not in ctx:
            continue

        valid_ids = _declared_tool_use_ids(history[i - 1] if i > 0 else None)

        kept: List[Dict[str, Any]] = []
        orphaned_text: List[str] = []
        for tr in ctx["toolResults"]:
            if tr.get("toolUseId") in valid_ids:
                kept.append(tr)
            else:
                orphaned_text.extend(_extract_result_text(tr.get("content")))

        removed = len(ctx["toolResults"]) - len(kept)
        if removed == 0:
            continue

        stats.orphans_removed += removed
        stats.entries_touched += 1

        if kept:
            ctx["toolResults"] = kept
        else:
            del ctx["toolResults"]
            if not ctx:
                del user_msg["userInputMessageContext"]

        if orphaned_text:
            marker = "\n[trimmed tool result] " + "; ".join(orphaned_text)
            user_msg["content"] = (user_msg.get("content") or "") + marker

        # Kiro requires non-empty content on every message.
        if not user_msg.get("content"):
            user_msg["content"] = " "

    if stats.repaired:
        logger.info(
            f"[payload_guards] repaired {stats.orphans_removed} orphaned tool "
            f"result(s) across {stats.entries_touched} history entr(ies); "
            "their text was preserved inline"
        )

    return stats


# ---------------------------------------------------------------------------
# Trimming (opt-in)
# ---------------------------------------------------------------------------

def trim_payload_to_limit(payload: Dict[str, Any], max_bytes: int) -> PayloadTrimStats:
    """
    Drop oldest history entries until the serialized payload fits ``max_bytes``.

    Trims two entries at a time (a user/assistant pair), realigns the start to a
    ``userInputMessage``, then re-runs orphan repair — trimming can orphan a tool
    result by removing the assistant turn that declared it.
    """
    original_bytes = check_payload_size(payload)
    history = payload.get("conversationState", {}).get("history")

    if not history:
        return PayloadTrimStats(
            original_bytes=original_bytes,
            final_bytes=original_bytes,
            original_entries=0,
            final_entries=0,
            trimmed=False,
        )

    original_entries = len(history)

    _strip_empty_tool_uses(history)

    while len(history) > 2 and check_payload_size(payload) > max_bytes:
        history.pop(0)
        history.pop(0)

    _align_to_user_message(history)
    enforce_tool_result_integrity(payload)

    final_bytes = check_payload_size(payload)
    return PayloadTrimStats(
        original_bytes=original_bytes,
        final_bytes=final_bytes,
        original_entries=original_entries,
        final_entries=len(history),
        trimmed=original_entries != len(history),
    )


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def apply_payload_guards(
    payload: Dict[str, Any],
    max_bytes: int,
    auto_trim: bool,
) -> Dict[str, Any]:
    """
    Run every guard over an assembled payload, in the correct order.

    1. Strip empty ``toolUses`` (always — Kiro rejects them).
    2. Repair orphaned tool results (always).
    3. Trim to size (only when ``auto_trim``); logs a warning when oversized and
       trimming is off, so the operator sees why Kiro returned a 400.

    Returns the same payload object, mutated in place, for call-site convenience.
    """
    history = payload.get("conversationState", {}).get("history")
    if history:
        _strip_empty_tool_uses(history)

    enforce_tool_result_integrity(payload)

    size = check_payload_size(payload)
    if size > max_bytes:
        if auto_trim:
            stats = trim_payload_to_limit(payload, max_bytes)
            logger.info(
                f"[payload_guards] trimmed history "
                f"{stats.original_entries} → {stats.final_entries} entries "
                f"({stats.original_bytes} → {stats.final_bytes} bytes)"
            )
        else:
            logger.warning(
                f"[payload_guards] payload is {size} bytes, over the "
                f"{max_bytes} limit — Kiro will likely reject this with "
                '"Improperly formed request." Set AUTO_TRIM_PAYLOAD=true to '
                "drop oldest history instead."
            )

    return payload
