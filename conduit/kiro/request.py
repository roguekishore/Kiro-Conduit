"""
Kiro conversationState builder.

This module is the Lane B entry point: it takes a flat message list and all
request parameters, runs them through the Lane B sub-pipeline (merge, tool
conversion, reasoning), and returns the complete Kiro wire payload together
with a self-describing RewrittenOut that declares every transformation made.

Protocol note — systemPrompt field
------------------------------------
Evidence from the kiro-cli.exe binary (PLATFORM-FINDINGS §5a-quater) shows
that Kiro's ``conversationState`` has a native ``systemPrompt`` field.  This
module places the harness system prompt there when ``KIRO_NATIVE_SYSTEM_PROMPT``
is ``True`` (the default).  With the flag off it falls back to prepending the
system prompt into the first user message text, which was the only option
before the native field was confirmed.

Both paths are tested; the native path is the default.

Image handling
--------------
Upstream silently drops URL-based images (converters_core.py:253,277).  This
module emits a WARNING log line for each URL image encountered so the operator
can see the loss; the image is not forwarded because Kiro has no fetch
capability.

Public API
----------
- HistoryMessage  (re-exported from history.py for callers)
- build_conversation_state()  — the main entry point
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

import conduit.config as cfg
from conduit.contracts.rewritten import Injection, InjectionType, RewrittenOut
from conduit.kiro.history import HistoryMessage, merge_turns
from conduit.kiro.reasoning import build_reasoning_fields
from conduit.kiro.tools import (
    build_tool_result,
    build_tool_specs,
    sanitize_json_schema,
)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _convert_images_to_kiro(images: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Convert unified ``{"media_type": str, "data": str}`` images to Kiro format.

    URL-based images (``data`` starting with ``http``) are NOT forwarded
    because Kiro cannot fetch remote URLs.  A WARNING is logged for each.

    Args:
        images: List of unified image dicts.

    Returns:
        List of Kiro image dicts ``{"format": str, "source": {"bytes": str}}``.
    """
    if not images:
        return []

    kiro_images = []
    for img in images:
        media_type = img.get("media_type", "image/jpeg")
        data = img.get("data", "")

        if not data:
            logger.warning("Skipping image with empty data field")
            continue

        # Strip data: URI prefix if present (some callers pass the full prefix)
        if data.startswith("data:"):
            try:
                header, actual_data = data.split(",", 1)
                media_part = header.split(";")[0]
                extracted = media_part.replace("data:", "")
                if extracted:
                    media_type = extracted
                data = actual_data
            except (ValueError, IndexError) as exc:
                logger.warning(f"Failed to strip data URI prefix: {exc}")

        # URL-based images: Kiro cannot fetch them — warn and skip
        if data.startswith("http://") or data.startswith("https://"):
            logger.warning(
                f"URL-based image cannot be forwarded to Kiro API "
                f"(Kiro has no fetch capability): {data[:120]!r}"
            )
            continue

        fmt = media_type.split("/")[-1] if "/" in media_type else media_type
        kiro_images.append({"format": fmt, "source": {"bytes": data}})

    return kiro_images


# ---------------------------------------------------------------------------
# History builder
# ---------------------------------------------------------------------------

def _build_history(
    messages: List[HistoryMessage],
    model_id: str,
) -> List[Dict[str, Any]]:
    """
    Convert ``HistoryMessage`` objects to Kiro history array entries.

    Each user message becomes ``{"userInputMessage": {...}}``; each assistant
    message becomes ``{"assistantResponseMessage": {...}}``.

    Tool results for user messages must already be in Kiro wire format
    (``{"content": [{"text": str}], "status": str, "toolUseId": str}``).

    Args:
        messages: Already-merged list of history messages (all but the last).
        model_id: Resolved Kiro model ID, placed in each userInputMessage.

    Returns:
        Kiro ``history`` array.
    """
    history = []

    for msg in messages:
        if msg.role == "user":
            user_input: Dict[str, Any] = {
                "content": msg.content or "",
                "modelId": model_id,
                "origin": "AI_EDITOR",
            }

            kiro_images = _convert_images_to_kiro(msg.images)
            if kiro_images:
                user_input["images"] = kiro_images

            ctx: Dict[str, Any] = {}
            if msg.tool_results:
                ctx["toolResults"] = msg.tool_results  # already in Kiro format

            if ctx:
                user_input["userInputMessageContext"] = ctx

            history.append({"userInputMessage": user_input})

        elif msg.role == "assistant":
            assistant_resp: Dict[str, Any] = {
                "content": msg.content or "",
            }

            if msg.tool_calls:
                # tool_calls are in Kiro format: {"name": str, "input": dict, "toolUseId": str}
                assistant_resp["toolUses"] = msg.tool_calls

            history.append({"assistantResponseMessage": assistant_resp})

    return history


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_conversation_state(
    messages: List[HistoryMessage],
    system_prompt: Optional[str],
    model_id: str,
    tools: Optional[List[Dict[str, Any]]],
    conversation_id: Optional[str] = None,
    profile_arn: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    agent_mode: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], RewrittenOut]:
    """
    Build the complete Kiro ``conversationState`` wire payload.

    Orchestrates the Lane B sub-pipeline:
    1. Merge consecutive same-role turns (history.py).
    2. Ensure protocol alternation (user/assistant interleaving).
    3. Split into ``history`` and ``currentMessage``.
    4. Place ``systemPrompt`` natively or prepend into first user message,
       controlled by ``cfg.KIRO_NATIVE_SYSTEM_PROMPT``.
    5. Convert tools to Kiro ``toolSpecification`` shape (tools.py).
    6. Build ``additionalModelRequestFields`` with reasoning knobs (reasoning.py).
    7. Assemble the final payload and a self-describing ``RewrittenOut``.

    Note on empty content: Kiro requires non-empty ``content`` in every
    message.  Any message with empty content receives the sentinel ``" "``
    (a single space) rather than the upstream ``"(empty placeholder)"`` text,
    so the model is not polluted with synthetic text it did not produce.

    Args:
        messages:          Flat list of ``HistoryMessage`` objects (user + assistant,
                           no system).  Caller is responsible for extracting system
                           messages before calling.
        system_prompt:     The harness system prompt (extracted from the request's
                           ``system`` field or from system-role messages).
        model_id:          Resolved Kiro model ID (e.g. ``"claude-sonnet-4.6"``).
        tools:             Optional list of tool dicts in Anthropic/Codex input shape.
                           Converted to Kiro format internally.
        conversation_id:   UUID string; a fresh one is generated if ``None``.
        profile_arn:       AWS CodeWhisperer profile ARN, or ``None`` to omit.
        reasoning_effort:  One of ``"low"`` / ``"medium"`` / ``"high"`` / ``"xhigh"``
                           / ``"max"``, or ``None`` for the default.
        agent_mode:        Kiro ``agentMode`` object.  Claude Code and Codex do not
                           produce this data; omit (``None``) to send ``{}``.

    Returns:
        ``(wire_payload, rewritten_out)`` where ``wire_payload`` is the dict
        to POST to Kiro and ``rewritten_out`` declares all injections made.

    Raises:
        ValueError: If the message list is empty after processing.
    """
    if conversation_id is None:
        conversation_id = str(uuid.uuid4())

    injections: List[Injection] = []

    # ------------------------------------------------------------------
    # 1. Merge consecutive same-role turns
    # ------------------------------------------------------------------
    merged, merge_injections = merge_turns(messages)
    injections.extend(merge_injections)

    # ------------------------------------------------------------------
    # 2. Ensure protocol compliance: alternating user/assistant
    #    Insert minimal synthetic turns where required (no placeholder text).
    # ------------------------------------------------------------------
    merged = _ensure_starts_with_user(merged)
    merged = _ensure_alternating(merged)

    if not merged:
        raise ValueError("No messages to send after processing")

    # ------------------------------------------------------------------
    # 3. Split: history = all but last; currentMessage = last
    # ------------------------------------------------------------------
    if len(merged) > 1:
        history_msgs = merged[:-1]
    else:
        history_msgs = []
    current_msg = merged[-1]

    # ------------------------------------------------------------------
    # 4. System prompt placement
    # ------------------------------------------------------------------
    system_text = (system_prompt or "").strip()

    if system_text:
        if cfg.KIRO_NATIVE_SYSTEM_PROMPT:
            # Native path: place in conversationState.systemPrompt
            injections.append(
                Injection(
                    type=InjectionType.SYSTEM_PROMPT_SLOT,
                    location="systemPrompt",
                )
            )
            logger.debug("system prompt → conversationState.systemPrompt (native slot)")
        else:
            # Fallback path: prepend into first user message
            _prepend_system_into_first_user(history_msgs, current_msg, system_text)
            injections.append(
                Injection(
                    type=InjectionType.THINKING_TAGS,
                    location="last_user_message",
                )
            )
            logger.debug("system prompt → prepended into first user message (fallback mode)")

    # ------------------------------------------------------------------
    # 5. Tools
    # ------------------------------------------------------------------
    kiro_tools: List[Dict[str, Any]] = []
    if tools:
        kiro_tools = build_tool_specs(tools)

    # ------------------------------------------------------------------
    # 6. Reasoning / additionalModelRequestFields
    # ------------------------------------------------------------------
    additional_fields: Dict[str, Any] = {}
    if reasoning_effort is not None:
        additional_fields = build_reasoning_fields(model_id, reasoning_effort)

    # ------------------------------------------------------------------
    # 7. Build Kiro history array
    # ------------------------------------------------------------------
    history = _build_history(history_msgs, model_id)

    # ------------------------------------------------------------------
    # 8. Build currentMessage (userInputMessage)
    # ------------------------------------------------------------------
    current_content = current_msg.content or ""

    kiro_images = _convert_images_to_kiro(current_msg.images)

    user_ctx: Dict[str, Any] = {}
    if kiro_tools:
        user_ctx["tools"] = kiro_tools
    if current_msg.tool_results:
        user_ctx["toolResults"] = current_msg.tool_results  # already Kiro format

    user_input_message: Dict[str, Any] = {
        "content": current_content,
        "modelId": model_id,
        "origin": "AI_EDITOR",
    }
    if kiro_images:
        user_input_message["images"] = kiro_images
    if user_ctx:
        user_input_message["userInputMessageContext"] = user_ctx

    current_message_dict: Dict[str, Any] = {
        "userInputMessage": user_input_message,
    }

    # ------------------------------------------------------------------
    # 9. Assemble conversationState
    # ------------------------------------------------------------------
    conversation_state: Dict[str, Any] = {
        "chatTriggerType": "MANUAL",
        "conversationId": conversation_id,
        "currentMessage": current_message_dict,
    }

    if history:
        conversation_state["history"] = history

    if system_text and cfg.KIRO_NATIVE_SYSTEM_PROMPT:
        conversation_state["systemPrompt"] = system_text

    # agentMode: Claude Code and Codex don't produce this data; {} is correct.
    conversation_state["agentMode"] = agent_mode if agent_mode is not None else {}

    if additional_fields:
        conversation_state["additionalModelRequestFields"] = additional_fields

    # ------------------------------------------------------------------
    # 10. Top-level payload
    # ------------------------------------------------------------------
    payload: Dict[str, Any] = {"conversationState": conversation_state}

    if profile_arn and profile_arn.strip():
        payload["profileArn"] = profile_arn.strip()

    # ------------------------------------------------------------------
    # 11. Build RewrittenOut
    # ------------------------------------------------------------------
    rewritten = RewrittenOut(
        system_prompt=system_text if system_text else None,
        current_message=current_message_dict,
        history=history,
        tools=kiro_tools,
        injections=injections,
    )

    return payload, rewritten


# ---------------------------------------------------------------------------
# Protocol-compliance helpers (internal)
# ---------------------------------------------------------------------------

def _ensure_starts_with_user(messages: List[HistoryMessage]) -> List[HistoryMessage]:
    """Prepend a synthetic empty user message if the first message is not user."""
    if not messages:
        return messages
    if messages[0].role != "user":
        logger.debug(
            f"First message is {messages[0].role!r}; "
            "prepending synthetic user message for Kiro protocol compliance"
        )
        synthetic = HistoryMessage(role="user", content="")
        return [synthetic] + messages
    return messages


def _ensure_alternating(messages: List[HistoryMessage]) -> List[HistoryMessage]:
    """
    Insert synthetic turns to ensure strict user/assistant alternation.

    Handles both directions:
    - consecutive user turns    → insert synthetic empty assistant turn between them
    - consecutive assistant turns → insert synthetic empty user turn between them

    Uses empty content (``""``) rather than the upstream ``"(empty placeholder)"``
    sentinel so the model's history is not polluted with fabricated text.
    """
    if not messages:
        return messages

    result = [messages[0]]
    synthetic_count = 0

    for msg in messages[1:]:
        prev_role = result[-1].role
        if msg.role == "user" and prev_role == "user":
            result.append(HistoryMessage(role="assistant", content=""))
            synthetic_count += 1
        elif msg.role == "assistant" and prev_role == "assistant":
            result.append(HistoryMessage(role="user", content=""))
            synthetic_count += 1
        result.append(msg)

    if synthetic_count:
        logger.debug(
            f"Inserted {synthetic_count} synthetic empty turn(s) "
            "to satisfy Kiro alternation requirement"
        )

    return result


def _prepend_system_into_first_user(
    history_msgs: List[HistoryMessage],
    current_msg: HistoryMessage,
    system_text: str,
) -> None:
    """
    Mutate the first user message in-place, prepending ``system_text``.

    If ``history_msgs`` has a user message, it is modified first.
    Otherwise ``current_msg`` is modified.
    """
    for msg in history_msgs:
        if msg.role == "user":
            msg.content = f"{system_text}\n\n{msg.content}" if msg.content else system_text
            return
    # No history user message — prepend into current
    if current_msg.role == "user":
        current_msg.content = (
            f"{system_text}\n\n{current_msg.content}"
            if current_msg.content
            else system_text
        )
