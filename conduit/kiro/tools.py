"""
Tool-spec conversion and tool-result building for the Kiro request pipeline.

Design decision — tool-result status source
--------------------------------------------
Amazon's ``generateAssistantResponse`` schema accepts a real ``status`` field
on each tool result: ``"success"`` or ``"error"``.  The upstream gateway
hardcodes ``"success"`` unconditionally, which silently lies to the model when
a tool actually failed.

This module reads ``is_error`` from the Anthropic tool-result block and emits
the truthful status.  For the Codex / OpenAI path (Lane D), Codex's
``FunctionCallOutput`` carries no item-level success/error flag, so
``build_tool_result`` accepts an explicit ``is_error`` parameter.  Lane D
supplies the value; this module does NOT default to ``"success"`` blindly.

The chosen default for ``is_error`` in ``build_tool_result`` is ``False``
(i.e. ``"success"``), documented here so Lane D can choose to override it.
Defaulting to ``"success"`` is intentional for the common case where the
caller has no error information — it degrades gracefully rather than
mis-labelling every Codex tool result as an error.  Lane D SHOULD inspect
its own result objects and pass the correct flag when the information is
available.

Public API
----------
- build_tool_spec()    — single Anthropic/Codex tool → Kiro toolSpecification
- build_tool_specs()   — list conversion
- build_tool_result()  — single tool-result dict in Kiro wire format
- sanitize_json_schema() — strip fields the Kiro API rejects
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# JSON-Schema sanitization
# ---------------------------------------------------------------------------

def sanitize_json_schema(schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Remove fields from a JSON Schema that the Kiro API rejects, and ensure
    Bedrock-compatible structure.

    Kiro / Bedrock returns ``400`` when:
    - ``required`` is an empty list ``[]``
    - ``additionalProperties`` is present anywhere in the schema
    - The top-level ``type`` field is missing or is not ``"object"``
      (Bedrock: ``"The value at toolConfig.tools.0.toolSpec.inputSchema.json.type
      must be one of the following: object"``)

    The function recurses into ``properties``, plain object values, and list
    items (e.g. ``anyOf`` / ``oneOf``).

    Args:
        schema: A JSON Schema dict, possibly nested.

    Returns:
        A sanitized copy (original is not mutated).
    """
    if not schema:
        return {"type": "object"}

    result: Dict[str, Any] = {}
    for key, value in schema.items():
        if key == "required" and isinstance(value, list) and len(value) == 0:
            continue
        if key == "additionalProperties":
            continue
        if key == "properties" and isinstance(value, dict):
            result[key] = {
                k: sanitize_json_schema(v) if isinstance(v, dict) else v
                for k, v in value.items()
            }
        elif isinstance(value, dict):
            result[key] = sanitize_json_schema(value)
        elif isinstance(value, list):
            result[key] = [
                sanitize_json_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value

    # Bedrock requires the top-level type to be "object".  When it is absent or
    # set to something else (e.g. the WebSearch tool schema that has no "type"
    # at all), inject "object" so the payload is accepted.
    if result.get("type") != "object":
        result["type"] = "object"

    return result


# ---------------------------------------------------------------------------
# Tool-spec conversion
# ---------------------------------------------------------------------------

def build_tool_spec(
    name: str,
    description: Optional[str],
    input_schema: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Convert one tool definition to Kiro's ``tools[].toolSpecification`` shape.

    Kiro requires a non-empty ``description``; a placeholder is used when the
    caller supplies ``None`` or an empty string.

    Args:
        name:         Tool name (max 64 chars; validation is the caller's job).
        description:  Human-readable description of what the tool does.
        input_schema: JSON Schema for the tool's input parameters.

    Returns:
        ``{"toolSpecification": {"name": ..., "description": ..., "inputSchema": {"json": ...}}}``
    """
    if not description or not description.strip():
        description = f"Tool: {name}"
        logger.debug(f"Tool {name!r} has empty description; using placeholder")

    sanitized = sanitize_json_schema(input_schema)

    return {
        "toolSpecification": {
            "name": name,
            "description": description,
            "inputSchema": {"json": sanitized},
        }
    }


def build_tool_specs(
    tools: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert a list of tool definitions (Anthropic or Codex shape) to Kiro format.

    Each tool dict is expected to have at least ``name``.  ``description`` and
    ``input_schema`` / ``parameters`` are optional.

    Anthropic shape:  ``{"name": str, "description": str, "input_schema": {...}}``
    Codex shape:      ``{"name": str, "description": str, "parameters": {...}}``

    Args:
        tools: List of tool dicts.

    Returns:
        List of Kiro ``toolSpecification`` wrappers.
    """
    kiro_tools = []
    for t in tools:
        name = t.get("name", "")
        description = t.get("description")
        # Anthropic uses input_schema; Codex / OpenAI uses parameters
        input_schema = t.get("input_schema") or t.get("parameters") or {}
        kiro_tools.append(build_tool_spec(name, description, input_schema))
    return kiro_tools


# ---------------------------------------------------------------------------
# Tool-result conversion
# ---------------------------------------------------------------------------

def build_tool_result(
    tool_use_id: str,
    content: Any,
    is_error: bool = False,
    images: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Build a single Kiro tool-result dict.

    The ``status`` field reflects the actual outcome: ``"error"`` when
    ``is_error`` is ``True``, ``"success"`` otherwise.  This is intentionally
    NOT hardcoded to ``"success"``; the caller must supply the correct flag.

    The default ``is_error=False`` (→ ``"success"``) is the chosen fallback
    for callers that have no error information (e.g. Lane D / Codex path).
    Lane D SHOULD pass the real value when its result objects expose one.

    Content normalisation:
    - ``str``  → used as-is
    - ``list`` → text blocks extracted and joined
    - other    → ``str(content)``
    Empty content is replaced with the sentinel ``"(empty result)"`` because
    Kiro rejects empty content arrays.

    Args:
        tool_use_id: The ``tool_use_id`` / ``toolUseId`` from the tool call.
        content:     Tool output; any type accepted.
        is_error:    Whether the tool call resulted in an error.
        images:      Optional list of Kiro-format image dicts
                     ``{"format": str, "source": {"bytes": str}}`` to append
                     to the content array (e.g. screenshots from computer-use).

    Returns:
        Kiro wire-format tool result dict.
    """
    # Normalise content to plain text
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = _extract_text_from_list(content)
    else:
        text = str(content) if content is not None else ""

    if not text:
        text = "(empty result)"

    status = "error" if is_error else "success"

    content_array: List[Dict[str, Any]] = [{"text": text}]
    if images:
        content_array.extend(images)

    return {
        "content": content_array,
        "status": status,
        "toolUseId": tool_use_id,
    }


def build_tool_results_from_anthropic(
    content: Any,
) -> List[Dict[str, Any]]:
    """
    Extract and convert all ``tool_result`` blocks from an Anthropic user-turn
    content list into Kiro tool-result dicts.

    Reads ``is_error`` from each block so the ``status`` field is truthful.

    Args:
        content: Anthropic message content (list of content blocks or string).

    Returns:
        List of Kiro tool-result dicts.
    """
    if not isinstance(content, list):
        return []

    results = []
    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type")
            tool_use_id = block.get("tool_use_id", "")
            raw_content = block.get("content", "")
            is_error = bool(block.get("is_error", False))
        elif hasattr(block, "type"):
            block_type = block.type
            tool_use_id = getattr(block, "tool_use_id", "")
            raw_content = getattr(block, "content", "")
            is_error = bool(getattr(block, "is_error", False))
        else:
            continue

        if block_type != "tool_result" or not tool_use_id:
            continue

        # Extract images from list content before flattening text
        result_images: Optional[List[Dict[str, Any]]] = None
        if isinstance(raw_content, list):
            result_images = _extract_images_from_list(raw_content) or None
            raw_content = _extract_text_from_list(raw_content)
        elif not isinstance(raw_content, str):
            raw_content = str(raw_content) if raw_content is not None else ""

        results.append(
            build_tool_result(
                tool_use_id=tool_use_id,
                content=raw_content,
                is_error=is_error,
                images=result_images,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_text_from_list(blocks: List[Any]) -> str:
    """Extract and join text from a list of content blocks."""
    parts = []
    for item in blocks:
        if isinstance(item, dict):
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif "text" in item:
                parts.append(item["text"])
        elif hasattr(item, "text"):
            parts.append(str(item.text))
        elif isinstance(item, str):
            parts.append(item)
    return "".join(parts)


def _extract_images_from_list(blocks: List[Any]) -> List[Dict[str, Any]]:
    """
    Extract base64 image blocks from a list of Anthropic content blocks and
    convert them to Kiro wire format.

    Anthropic image shape:
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}

    Kiro image shape:
        {"format": "jpeg", "source": {"bytes": "..."}}

    URL-based images (``source.type == "url"``) are logged as warnings and
    skipped — Kiro has no fetch capability.

    Args:
        blocks: List of Anthropic content block dicts.

    Returns:
        List of Kiro-format image dicts; empty if no images found.
    """
    kiro_images: List[Dict[str, Any]] = []
    for item in blocks:
        if isinstance(item, dict):
            item_type = item.get("type")
        elif hasattr(item, "type"):
            item_type = item.type
        else:
            continue

        if item_type != "image":
            continue

        source = item.get("source", {}) if isinstance(item, dict) else getattr(item, "source", {})
        if isinstance(source, dict):
            src_type = source.get("type")
            if src_type == "base64":
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
                if data:
                    fmt = media_type.split("/")[-1] if "/" in media_type else media_type
                    kiro_images.append({"format": fmt, "source": {"bytes": data}})
            elif src_type == "url":
                logger.warning(
                    f"URL-based image in tool_result cannot be forwarded to Kiro "
                    f"(Kiro has no fetch capability): {source.get('url', '')[:120]!r}"
                )
        elif hasattr(source, "type"):
            if source.type == "base64":
                media_type = getattr(source, "media_type", "image/jpeg")
                data = getattr(source, "data", "")
                if data:
                    fmt = media_type.split("/")[-1] if "/" in media_type else media_type
                    kiro_images.append({"format": fmt, "source": {"bytes": data}})
            elif source.type == "url":
                logger.warning(
                    f"URL-based image in tool_result cannot be forwarded to Kiro "
                    f"(Kiro has no fetch capability): {getattr(source, 'url', '')[:120]!r}"
                )
    return kiro_images
