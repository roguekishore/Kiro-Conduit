"""Codex door — POST /v1/responses, for Codex CLI -> Luna.

OWNER: Lane D. Registered by conduit/app.py; do not edit app.py to wire it.

Lane D implements the OpenAI Responses contract (FINDINGS-codex 1.2/1.5/1.7/1.8).
Load-bearing points, all verified in Codex source:

  - store:false / stream:true / tool_choice:"auto" are hardcoded client-side.
  - System prompt is the top-level `instructions` string -> Kiro systemPrompt.
  - BOTH wire dialects are required. Under responses-lite the prompt and tool
    list move OUT of instructions/tools and INTO input[] as prefix items,
    leaving instructions:"" and tools:null. A gateway reading only
    `instructions` sees an empty prompt for any lite model.
  - Dispatch the dialect on the header
    `x-openai-internal-codex-responses-lite: true`. Do not sniff the body.
    The flag is server-driven and can change with no CLI release.
  - Reasoning blobs are NOT portable: Codex encrypted_content and Kiro
    signature are provider-specific. Drop or store locally, never relay.
  - Tool status cannot come from Codex's wire — FunctionCallOutput has no
    item-level success/error flag. Supply it from CONDUIT's tool runner.
  - The authoritative session id is client_metadata.session_id in the BODY,
    not the session-id header (which carries the prompt-cache key for root
    agents by design). Recorded here for WS-C's benefit.

See SPEC-WORKSTREAMS.md, Lane D, for the full brief and verification gate.
"""

from fastapi import APIRouter

router = APIRouter()
