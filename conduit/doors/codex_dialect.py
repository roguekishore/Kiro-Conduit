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

import json
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

# Sentinel header that selects the lite dialect.
LITE_HEADER = "x-openai-internal-codex-responses-lite"

# The single synthetic parameter a Codex freeform ("custom") tool is rewritten
# to carry. Kiro's toolSpecification cannot express a grammar-constrained
# freeform tool, so its raw source text travels as one string property under
# this name. See adapt_freeform_tools() for the full rationale.
FREEFORM_PARAM = "source"


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
    prompt_parts: List[str] = []
    tool_list: List[Dict[str, Any]] = []

    for item in input_items:
        role = item.get("role", "")
        item_type = item.get("type", "")

        # Stop at the first non-developer item; the prefix is over.
        if role != "developer":
            break

        if item_type in ("", "message"):
            # Instructions item — extract text from content.
            #
            # Codex sends MORE THAN ONE developer message item, and every one of
            # them is load-bearing. Measured on live traffic (gpt-5.6-luna):
            #   input[1]  17730 chars  the base Codex prompt ("You are Codex...")
            #   input[2]   8307 chars  <skills_instructions> + <collaboration_mode>
            #
            # This used to assign ``prompt = text`` per item, so the LAST item
            # won and the base prompt — the half that tells the model how to
            # behave and how to drive its tools — was silently dropped. The
            # model was left with skills boilerplate and no operating
            # instructions.
            #
            # Codex marks both with requires_separate_message() precisely so
            # they arrive as distinct items; it defines no separator between
            # them, so they are joined in array order with a blank line.
            content = item.get("content", "")
            text = _extract_text(content)
            if text:
                prompt_parts.append(text)
                logger.debug(
                    "codex lite dialect: instructions item {} extracted, len={}",
                    len(prompt_parts),
                    len(text),
                )

        elif item_type == "additional_tools":
            raw_tools = item.get("tools") or []
            tool_list = _uncollapse_namespace(raw_tools)
            logger.debug(
                "codex lite dialect: {} tools extracted (uncollapsed)", len(tool_list)
            )

    prompt = "\n\n".join(prompt_parts)
    if len(prompt_parts) > 1:
        logger.debug(
            "codex lite dialect: joined {} developer instruction items, total len={}",
            len(prompt_parts),
            len(prompt),
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


# ---------------------------------------------------------------------------
# Freeform ("custom") tools
# ---------------------------------------------------------------------------

# Guidance appended to a rewritten freeform tool's description. It must describe
# the language of the SOURCE (JavaScript, per exec's own prose), never the
# grammar that constrains it — see adapt_freeform_tools().
_FREEFORM_GUIDANCE = (
    "Call this tool with a single argument named "
    f'"{FREEFORM_PARAM}" containing the raw source text to run, exactly as you '
    "would write it for this tool. Do not wrap it in JSON, quotes, or markdown "
    "fences, and do not split it across other parameters."
)

# Keys the model is prone to use when it wraps source in an object of its own.
_SOURCE_ALIASES = ("code", "input", "script", "js", "source_code", "text")

# exec_command options that are NOT part of the command itself. They configure
# the harness around it, so they are hoisted into a leading pragma comment
# rather than passed through as call arguments.
_PRAGMA_KEYS = ("yield_time_ms", "max_output_tokens")


def adapt_freeform_tools(
    tools: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Rewrite freeform tools into schema-bearing functions.

    Codex declares its ``exec`` tool as a *freeform* (a.k.a. "custom") tool::

        {"type": "custom", "name": "exec",
         "format": {"type": "grammar", "syntax": "lark", "definition": ...}}

    It has no ``parameters`` at all: the model emits raw source text, not a JSON
    object. Kiro's ``toolSpecification`` has no way to express that — it only
    carries a JSON Schema — so such a tool reached Kiro with
    ``inputSchema.json == {}`` and the model had nothing to aim at.

    The fix is to give the tool exactly one string property, ``FREEFORM_PARAM``,
    and carry the source text in it. The original description is preserved
    because it holds the nested tool declarations that tell the model what API
    is actually available inside the source.

    Note on ``format.syntax``: it names the GRAMMAR constraining the input
    ("lark"), not the language of the input (JavaScript). It is deliberately
    kept out of every model-facing string and used only in the log line — a
    model told to write "lark source" would aim at the wrong language entirely.

    Returns:
        ``(adapted_tools, custom_names)`` — the rewritten list, and the set of
        tool names that were freeform. The caller must pass that set to the SSE
        translator so replies are framed as ``custom_tool_call`` rather than
        ``function_call``; Codex accepts a freeform tool's result only via
        ``ToolPayload::Custom``.
    """
    adapted: List[Dict[str, Any]] = []
    custom_names: Set[str] = set()

    for tool in tools or []:
        if not isinstance(tool, dict):
            adapted.append(tool)
            continue

        # Detection must not hinge on type == "custom" alone: the wire has also
        # carried freeform tools distinguished only by "format" with no
        # "parameters".
        is_freeform = tool.get("type") == "custom" or (
            "format" in tool and "parameters" not in tool
        )
        if not is_freeform:
            adapted.append(tool)
            continue

        name = tool.get("name") or ""
        if not name:
            # Unnamed tool cannot be called or matched to a reply. Dropping it
            # is safer than forwarding something the model might try to invoke.
            logger.warning("freeform tool with no name dropped: {}", tool)
            continue

        description = (tool.get("description") or "").strip()
        # Log only — never model-facing. See the docstring.
        syntax = (tool.get("format") or {}).get("syntax") or "text"

        adapted.append(
            {
                "type": "function",
                "name": name,
                "description": (
                    f"{description}\n\n{_FREEFORM_GUIDANCE}"
                    if description
                    else _FREEFORM_GUIDANCE
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        FREEFORM_PARAM: {
                            "type": "string",
                            "description": (
                                "Raw source text to execute, in the language "
                                "this tool's description specifies."
                            ),
                        }
                    },
                    "required": [FREEFORM_PARAM],
                    "additionalProperties": False,
                },
            }
        )
        custom_names.add(name)
        logger.debug(
            "freeform tool adapted: name={} grammar={} -> single '{}' string param",
            name,
            syntax,
            FREEFORM_PARAM,
        )

    return adapted, custom_names


def freeform_input_from_arguments(raw_arguments: str) -> str:
    """Recover freeform source text from whatever the model actually produced.

    A freeform tool wants one blob of source. Having been declared as a function
    (see adapt_freeform_tools) the model replies with a JSON arguments string,
    and in practice it picks one of several shapes. Each is unwrapped back to
    source here.

    1. The declared shape — ``{"source": "<src>"}``.
    2. An alias the model chose instead — ``{"code": ...}``, ``{"script": ...}``.
    3. A bare JSON string literal — ``"<src>"``.
    4. Nested ``exec_command`` arguments — ``{"cmd": ..., "workdir": ...}``.
       This is what live traffic produced while the schema was empty: the model
       imitated the ``exec_command(args)`` signature documented in ``exec``'s own
       prose instead of writing a program. It is reconstructed into the
       equivalent JavaScript, with harness-only options hoisted to a pragma.
    5. Anything else — returned verbatim.

    Returning input verbatim is deliberate for the unrecognised cases. A real
    grammar error from the model's own text is diagnosable; substitute source
    invented here would execute something the model never asked for.
    """
    if not raw_arguments or not raw_arguments.strip():
        return ""

    text = raw_arguments.strip()

    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        # Not JSON — assume it is already the source. Notably this is the
        # correct branch for source containing Windows paths, whose backslashes
        # make it invalid JSON.
        return raw_arguments

    # Case 3 — a JSON string literal wrapping the source.
    if isinstance(parsed, str):
        return parsed

    if not isinstance(parsed, dict):
        return raw_arguments

    # Case 1 — the declared parameter.
    declared = parsed.get(FREEFORM_PARAM)
    if isinstance(declared, str):
        return declared

    # Case 2 — an alias the model chose on its own.
    for alias in _SOURCE_ALIASES:
        value = parsed.get(alias)
        if isinstance(value, str):
            logger.debug("freeform source recovered from alias key '{}'", alias)
            return value

    # Case 4 — nested exec_command arguments.
    if "cmd" in parsed:
        return _js_from_exec_command_args(parsed)

    # Case 5 — unrecognised. Forward untouched.
    logger.debug(
        "freeform arguments in an unrecognised shape, forwarded verbatim: keys={}",
        sorted(parsed.keys()),
    )
    return raw_arguments


def _js_from_exec_command_args(args: Dict[str, Any]) -> str:
    """Rebuild the JavaScript the model should have written for exec_command."""
    pragma = {k: args[k] for k in _PRAGMA_KEYS if k in args}
    call_args = {k: v for k, v in args.items() if k not in _PRAGMA_KEYS}

    lines: List[str] = []
    if pragma:
        # Hoisted so they configure the cell rather than the command.
        lines.append(f"// @exec: {json.dumps(pragma, ensure_ascii=False)}")
    lines.append(
        "const result = await tools.exec_command("
        f"{json.dumps(call_args, ensure_ascii=False)});"
    )
    lines.append("text(result);")

    logger.debug(
        "freeform source reconstructed from nested exec_command args: keys={}",
        sorted(args.keys()),
    )
    return "\n".join(lines)


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
