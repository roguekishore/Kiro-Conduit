"""Extract system prompt and tool list from Codex CLI /v1/responses requests.

Handles two wire dialects that Codex CLI can send to a Responses-compatible
gateway:

Standard dialect
    system_prompt  <- top-level `instructions` string
    tools          <- top-level `tools` list (pre-serialized, pass-through)

Responses-lite dialect  (header x-openai-internal-codex-responses-lite: true)
    Codex moves the prompt OUT of `instructions` and INTO `input[]` as prefix
    items, leaving `instructions` as "" and `tools` as null.

    Prefix structure in input[]:
      1. An item with role "developer" whose content carries the instructions
         text (either a string or a content-block list with a text block).
      2. An AdditionalTools item with role "developer" whose "tools" list
         contains a single "namespace" tool that wraps all the individual
         function specs inside it.

    This module un-collapses the namespace tool back to a flat list of
    individual specs so the rest of the pipeline sees the same shape in
    both dialects.

Dispatch is on the header ONLY.  Body sniffing is explicitly prohibited:
the flag is server-driven (comes from /models) and can flip without any
CLI release.

INTERCEPTION NOTE (document only, not built here)
    There is no OPENAI_BASE_URL env var in Codex.  Two wiring options exist:
      (a) openai_base_url = "http://localhost:PORT/v1" in config.toml  (smallest change)
      (b) model_providers entry with wire_api = "responses"             (preferred)
    Under ChatGPT auth the attached token is a ChatGPT credential, not an API key.

IDENTITY NOTE (document only, not built here)
    The authoritative session id is client_metadata.session_id in the request
    body, NOT the session-id header (which carries the prompt-cache key for
    root agents).  WS-C should read client_metadata.session_id for session
    tracking.

TOOL STATUS NOTE (document only, not built here)
    FunctionCallOutput has no item-level success/error flag on Codex's wire.
    Tool status must come from CONDUIT's own tool runner, not from what Codex
    sends back.  Lane B's tool-result builder must receive status from the
    tool runner; do not default to "success".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# Sentinel header that selects the lite dialect.
LITE_HEADER = "x-openai-internal-codex-responses-lite"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_dialect(
    *,
    instructions: Optional[str],
    tools: Optional[List[Any]],
    input_items: List[Dict[str, Any]],
    is_lite: bool,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Return (system_prompt, tool_list) for the active dialect.

    Args:
        instructions:  Top-level ``instructions`` field from the request body.
        tools:         Top-level ``tools`` field from the request body.
        input_items:   ``input`` array from the request body.
        is_lite:       True when the request header
                       ``x-openai-internal-codex-responses-lite: true`` is present.

    Returns:
        A tuple ``(system_prompt, tools)`` where:
        - ``system_prompt`` is the extracted instruction text (may be empty).
        - ``tools`` is a flat list of individual tool dicts (may be empty).
          In both dialects these are opaque objects — callers should treat them
          as pass-through JSON and not attempt to interpret the schemas here.
    """
    if is_lite:
        return _extract_lite(input_items)
    return _extract_standard(instructions, tools)


# ---------------------------------------------------------------------------
# Standard dialect
# ---------------------------------------------------------------------------

def _extract_standard(
    instructions: Optional[str],
    tools: Optional[List[Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Read prompt and tools directly from the top-level fields."""
    prompt = (instructions or "").strip()
    tool_list = list(tools) if tools else []
    logger.debug(
        "codex standard dialect: prompt_len=%d tools=%d",
        len(prompt),
        len(tool_list),
    )
    return prompt, tool_list


# ---------------------------------------------------------------------------
# Lite dialect
# ---------------------------------------------------------------------------

def _extract_lite(
    input_items: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract prompt and tools from the input[] prefix items.

    Scans the leading ``role: "developer"`` items in input[]:

    * A plain-message item (no explicit ``type``, or ``type == "message"``)
      with ``role == "developer"`` carries the instruction text.
    * An ``AdditionalTools`` item (``type == "additional_tools"``,
      ``role == "developer"``) carries the namespace-collapsed tool list.

    Scanning stops at the first non-developer item — everything after that is
    conversation history and must not be touched.
    """
    prompt = ""
    tool_list: List[Dict[str, Any]] = []

    for item in input_items:
        role = item.get("role", "")
        item_type = item.get("type", "")

        # Stop at the first non-developer item; the prefix is over.
        if role != "developer":
            break

        if item_type in ("", "message"):
            # Instructions item — extract text from content
            content = item.get("content", "")
            text = _extract_text(content)
            if text:
                prompt = text
                logger.debug(
                    "codex lite dialect: instructions extracted, len=%d", len(prompt)
                )

        elif item_type == "additional_tools":
            raw_tools = item.get("tools") or []
            tool_list = _uncollapse_namespace(raw_tools)
            logger.debug(
                "codex lite dialect: %d tools extracted (uncollapsed)", len(tool_list)
            )

    return prompt.strip(), tool_list


def _extract_text(content: Any) -> str:
    """Pull plain text from a content value that may be a string or a list of
    content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type in ("text", "input_text", ""):
                    text = block.get("text") or block.get("value") or ""
                    if isinstance(text, str):
                        parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return ""


# ---------------------------------------------------------------------------
# Namespace-tool un-collapsing
# ---------------------------------------------------------------------------

def _uncollapse_namespace(raw_tools: List[Any]) -> List[Dict[str, Any]]:
    """Expand namespace-collapsed tools back to individual function specs.

    In the lite dialect all per-turn tools are folded into a single
    ``namespace`` tool entry.  This function restores them to a flat list so
    the pipeline sees the same shape as the standard dialect.

    Several namespace-tool wire shapes have been observed:

    Shape A — the namespace object carries an explicit ``tools`` array:
        {"type": "namespace", "name": "namespace", "tools": [<individual tools>]}

    Shape B — the namespace ``parameters`` schema encodes tools as ``anyOf``
    sub-schemas, each tagged with a function name:
        {"type": "function", "name": "namespace",
         "parameters": {"anyOf": [{"properties": {"name": ..., "args": ...}}]}}

    Shape C — no structure is recognisable (opaque blob from an unknown server
    revision).  In that case the namespace tool is returned as-is so callers
    at least receive the raw object.

    Returns an empty list when ``raw_tools`` is falsy.
    """
    if not raw_tools:
        return []

    result: List[Dict[str, Any]] = []

    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue

        name = tool.get("name", "")
        tool_type = tool.get("type", "")

        is_namespace = name == "namespace" or tool_type == "namespace"
        if not is_namespace:
            # Non-namespace tool — keep as-is
            result.append(tool)
            continue

        # --- Shape A: explicit nested tools list ---
        nested = tool.get("tools")
        if isinstance(nested, list) and nested:
            logger.debug(
                "namespace un-collapse Shape A: %d nested tools", len(nested)
            )
            result.extend(t for t in nested if isinstance(t, dict))
            continue

        # --- Shape B: anyOf parameters schema ---
        params = tool.get("parameters") or {}
        any_of = params.get("anyOf") or params.get("oneOf") or []
        if any_of:
            extracted = _extract_tools_from_anyof(any_of)
            if extracted:
                logger.debug(
                    "namespace un-collapse Shape B: %d tools from anyOf",
                    len(extracted),
                )
                result.extend(extracted)
                continue

        # --- Shape C: opaque, pass through ---
        logger.debug(
            "namespace un-collapse Shape C: "
            "unrecognised namespace tool structure, passing through"
        )
        result.append(tool)

    return result


def _extract_tools_from_anyof(any_of: List[Any]) -> List[Dict[str, Any]]:
    """Best-effort extraction of individual function specs from an anyOf list."""
    tools: List[Dict[str, Any]] = []
    for schema in any_of:
        if not isinstance(schema, dict):
            continue
        props = schema.get("properties") or {}
        name_prop = props.get("name") or {}
        # Each anyOf entry may encode the function name in an enum
        enum_vals = name_prop.get("enum") or []
        if enum_vals:
            for fn_name in enum_vals:
                if isinstance(fn_name, str):
                    # Build a minimal function spec from what we can infer
                    args_prop = props.get("args") or props.get("arguments") or {}
                    tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "description": schema.get("description", ""),
                                "parameters": args_prop,
                            },
                        }
                    )
        else:
            # The schema might itself describe a named function directly
            fn_name = schema.get("name") or schema.get("title") or ""
            if fn_name:
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "description": schema.get("description", ""),
                            "parameters": schema.get("parameters", {}),
                        },
                    }
                )
    return tools
