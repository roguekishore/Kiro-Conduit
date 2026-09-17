"""
MCP web_search support — Lane C (Anthropic door).

Calls the Kiro MCP API (``/mcp``) to execute a web_search and returns the
results.  Also provides SSE and JSON response generators so the Anthropic door
can synthesise a complete ``web_search`` response without hitting
``/generateAssistantResponse`` at all.

Two paths (mirroring kiro-gateway mcp_tools.py):

Path A — native Anthropic server-side tool
  Claude Code sends a tool with ``type`` starting with ``"web_search"``.
  The door intercepts the request before building a Kiro payload, calls
  the MCP API directly, and returns a fully-formed Anthropic response.

Path B — MCP emulation via auto-injected tool
  The door auto-injects a ``web_search`` tool into every request (when
  ``WEB_SEARCH_ENABLED`` is True).  When the model decides to call it, the
  streaming wrapper intercepts the tool-use block, calls the MCP API, and
  yields the synthetic SSE frames that complete the turn.

Public API
----------
- call_mcp_web_search()      — async MCP API call, returns (tool_use_id, results)
- build_search_summary()     — format results as human-readable text
- anthropic_web_search_sse() — async generator: Anthropic SSE for Path A
- anthropic_web_search_json()— dict: non-streaming JSON for Path A
"""

from __future__ import annotations

import json
import random
import string
import time
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

import httpx
from loguru import logger

from conduit.config import PROFILE_ARN, get_kiro_mcp_url
from conduit.foundation.auth import resolve_profile_arn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_id(n: int) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


# ---------------------------------------------------------------------------
# MCP API call
# ---------------------------------------------------------------------------

async def call_mcp_web_search(
    query: str,
    auth_manager: Any,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Call the Kiro MCP API and return ``(tool_use_id, results_dict)``.

    Returns ``(None, None)`` on any error so callers can degrade gracefully.

    The MCP endpoint lives on the CodeWhisperer host, not on the runtime host.
    The runtime host returns 403 even with a valid profileArn.  The endpoint
    is not present in every region, so the account's API region is tried first
    with us-east-1 as a fallback.

    Args:
        query:        Web search query string.
        auth_manager: An initialised KiroAuthManager instance.

    Returns:
        ``(tool_use_id, results)`` on success, ``(None, None)`` on failure.
    """
    random_22 = _rand_id(22)
    timestamp = int(time.time() * 1000)
    random_8 = _rand_id(8)
    request_id = f"web_search_tooluse_{random_22}_{timestamp}_{random_8}"
    tool_use_id = f"srvtoolu_{uuid.uuid4().hex[:32]}"

    mcp_request: Dict[str, Any] = {
        "id": request_id,
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "web_search",
            "arguments": {"query": query},
        },
    }

    profile_arn = await resolve_profile_arn(auth_manager, PROFILE_ARN)
    if profile_arn:
        mcp_request["profileArn"] = profile_arn
    else:
        logger.warning(
            "[mcp] No profileArn for web_search — MCP API will reject the call. "
            "Set PROFILE_ARN in .env if auto-discovery fails."
        )

    try:
        token = await auth_manager.get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "x-amzn-codewhisperer-optout": "false",
            "Content-Type": "application/json",
        }

        # Try the account's own region, fall back to us-east-1
        regions = [auth_manager.api_region]
        if "us-east-1" not in regions:
            regions.append("us-east-1")

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = None
            for region in regions:
                url = get_kiro_mcp_url(region)
                logger.debug(f"[mcp] POST {url}")
                try:
                    response = await client.post(url, json=mcp_request, headers=headers)
                except httpx.RequestError as exc:
                    logger.warning(f"[mcp] request to {url} failed: {exc}")
                    response = None
                    continue

                if response.status_code == 200:
                    break

                logger.error(
                    f"[mcp] {url} → {response.status_code}: {response.text[:200]}"
                )
                response = None

            if response is None:
                return None, None

            body = response.json()

            if body.get("error"):
                logger.error(f"[mcp] JSON-RPC error: {body['error']}")
                return None, None

            # result.content[0].text is a JSON STRING (not an object)
            result_text = (
                body.get("result", {})
                .get("content", [{}])[0]
                .get("text", "{}")
            )
            results = json.loads(result_text)
            logger.debug(
                f"[mcp] web_search returned {results.get('totalResults', 0)} results"
            )
            return tool_use_id, results

    except httpx.TimeoutException as exc:
        logger.error(f"[mcp] timeout: {exc}")
    except httpx.RequestError as exc:
        logger.error(f"[mcp] request error: {exc}")
    except json.JSONDecodeError as exc:
        logger.error(f"[mcp] JSON parse error: {exc}")
    except Exception as exc:
        logger.error(f"[mcp] unexpected error: {exc}", exc_info=True)

    return None, None


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def build_search_summary(query: str, results: Dict[str, Any]) -> str:
    """Format MCP search results as human-readable text wrapped in XML tags.

    Returns the full snippet for each result without truncation so the model
    has complete information.

    Args:
        query:   Original search query.
        results: Parsed MCP response dict (has ``"results"`` key).

    Returns:
        Formatted summary string.
    """
    lines = [f'\n<web_search>\nSearch results for "{query}":\n']

    for i, result in enumerate(results.get("results", []), 1):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        snippet = result.get("snippet", "")
        published_ms = result.get("publishedDate")

        lines.append(f"{i}. Title: **{title}**")

        if published_ms:
            try:
                dt = datetime.fromtimestamp(published_ms / 1000)
                lines.append(f"   Published: {dt.strftime('%d %b %Y %H:%M:%S')}")
            except (ValueError, OSError):
                pass

        if url:
            lines.append(f"   URL: {url}")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")

    if not results.get("results"):
        lines.append("No results found.")

    lines.append("</web_search>")
    return "\n".join(lines) + "\n"


def render_replayed_search(results: Any) -> str:
    """Render a replayed ``web_search_tool_result`` payload back into text.

    Distinct from :func:`build_search_summary`, which formats a *fresh* MCP
    response (``snippet`` keys). This one formats what a CLIENT replays back on
    a later request: Anthropic-shaped ``web_search_result`` blocks whose text
    lives in ``encrypted_content``.

    Without this, replayed search blocks reach the Kiro payload builder, match
    no known block type, and are dropped - so a follow-up question about search
    results is answered by a model that can no longer see them.

    Args:
        results: The ``content`` of a ``web_search_tool_result`` block. Normally
                 a list of ``web_search_result`` blocks; an error dict or bare
                 string is also possible.

    Returns:
        Formatted text, or "" when there is nothing renderable.
    """
    if not results:
        return ""
    if isinstance(results, str):
        return f"\n<web_search>\n{results}\n</web_search>\n"
    if isinstance(results, dict):
        code = results.get("error_code") or results
        return f"\n<web_search>\n{code}\n</web_search>\n"
    if not isinstance(results, list):
        return ""

    lines = [f"\n<web_search>\nReplayed search results:\n"]
    for i, item in enumerate(results, 1):
        if isinstance(item, dict):
            title = item.get("title", "")
            url = item.get("url", "")
            snippet = item.get("encrypted_content", "")
        else:
            title = getattr(item, "title", "")
            url = getattr(item, "url", "")
            snippet = getattr(item, "encrypted_content", "")
        lines.append(f"{i}. Title: {title}")
        if url:
            lines.append(f"   URL: {url}")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")
    lines.append("</web_search>")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# SSE emitter (Anthropic format, Path A)
# ---------------------------------------------------------------------------

async def anthropic_web_search_sse(
    model: str,
    query: str,
    tool_use_id: str,
    results: Dict[str, Any],
    message_id: str,
) -> AsyncGenerator[str, None]:
    """Yield Anthropic SSE frames for a completed web_search turn (Path A).

    Emits the standard 11-event sequence:
      message_start → server_tool_use start/delta/stop →
      web_search_tool_result start/stop → text start/delta+/stop →
      message_delta → message_stop

    Args:
        model:        Model name to echo in message_start.
        query:        The search query that was executed.
        tool_use_id:  ID from the MCP response.
        results:      Parsed MCP results dict.
        message_id:   Stable message ID for this response.

    Yields:
        SSE-formatted strings (``event: …\\ndata: …\\n\\n``).
    """
    def _sse(evt: str, data: Dict[str, Any]) -> str:
        return f"event: {evt}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    summary = build_search_summary(query, results)

    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    # server_tool_use block
    yield _sse("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "id": tool_use_id,
            "type": "server_tool_use",
            "name": "web_search",
            "input": {},
        },
    })
    yield _sse("content_block_delta", {
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "input_json_delta",
            "partial_json": json.dumps({"query": query}),
        },
    })
    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})

    # web_search_tool_result block
    search_content = [
        {
            "type": "web_search_result",
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "encrypted_content": r.get("snippet", ""),
            "page_age": None,
        }
        for r in results.get("results", [])
    ]
    yield _sse("content_block_start", {
        "type": "content_block_start",
        "index": 1,
        "content_block": {
            "type": "web_search_tool_result",
            "tool_use_id": tool_use_id,
            "content": search_content,
        },
    })
    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 1})

    # text block — stream summary in chunks
    yield _sse("content_block_start", {
        "type": "content_block_start",
        "index": 2,
        "content_block": {"type": "text", "text": ""},
    })
    chunk_size = 100
    for i in range(0, len(summary), chunk_size):
        yield _sse("content_block_delta", {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "text_delta", "text": summary[i: i + chunk_size]},
        })
    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 2})

    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": max(1, len(summary) // 4)},
    })
    yield _sse("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# Non-streaming JSON response (Anthropic format, Path A)
# ---------------------------------------------------------------------------

def anthropic_web_search_json(
    model: str,
    query: str,
    tool_use_id: str,
    results: Dict[str, Any],
    message_id: str,
) -> Dict[str, Any]:
    """Build a complete non-streaming Anthropic message for a web_search turn.

    Args:
        model:        Model name to echo.
        query:        The search query that was executed.
        tool_use_id:  ID from the MCP response.
        results:      Parsed MCP results dict.
        message_id:   Stable message ID for this response.

    Returns:
        JSON-serialisable dict matching the Anthropic Messages API response shape.
    """
    summary = build_search_summary(query, results)
    search_content = [
        {
            "type": "web_search_result",
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "encrypted_content": r.get("snippet", ""),
            "page_age": None,
        }
        for r in results.get("results", [])
    ]
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {
                "type": "server_tool_use",
                "id": tool_use_id,
                "name": "web_search",
                "input": {"query": query},
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": tool_use_id,
                "content": search_content,
            },
            {"type": "text", "text": summary},
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 0,
            "output_tokens": max(1, len(summary) // 4),
        },
    }
