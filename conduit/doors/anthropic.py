"""Anthropic door — POST /v1/messages, for Claude Code.

OWNER: Lane C. Registered by conduit/app.py; do not edit app.py to wire it.

Lane C implements:
  - POST /v1/messages, auth via PROXY_API_KEY, transport via KiroHttpClient
    and AccountManager (both in conduit.foundation).
  - Anthropic SSE out: message_start, content_block_start/delta/stop,
    message_delta, message_stop.
  - reasoningContentEvent -> thinking_delta carrying the REAL upstream
    signature. Upstream scraped <thinking> tags out of plain text and
    fabricated the signature; both are retired.
  - usage block populated from Lane A's TurnMetrics — measured, not estimated.
  - On turn completion, hand TurnMetrics + RewrittenOut to Lane F's emit.
    Call it and move on; never await it.

See SPEC-WORKSTREAMS.md, Lane C, for the full brief and verification gate.
"""

from fastapi import APIRouter

router = APIRouter()
