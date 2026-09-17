"""Per-turn metrics, each labelled by how it was obtained.

Most numbers here are parsed straight off the wire — `meteringEvent.usage` and
`contextUsageEvent` — and where `metadataEvent.tokenUsage` is populated, the
token counters are too.

**Revised 2026-09-14.** An earlier revision of this docstring said "Nothing is
estimated", which was true when written and is no longer. Probing the live
backend showed its `metadataEvent` carries only `{"stopReason": ...}` — no
`tokenUsage` at all — so token counters arrived at zero on every request. Token
accounting is therefore *derived* from `contextUsagePercentage`
(`conduit.kiro.derive`), which measured exactly against controlled probes.

The upstream gateway did something superficially similar and was rightly
criticised for it. The difference is disclosure, not technique:
`token_accounting` states `"upstream"`, `"derived"`, or `"none"` on every turn,
so no reader can mistake a computed figure for a measured one. Upstream reported
its estimates as though the provider had stated them.

FROZEN in shape — Lane A populates; Lanes C/D/F read. Adding a field means
editing CONTRACT-conduit-saga-seam.md, bumping its version, and telling WS-C.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class StopReason(str, Enum):
    """Full stop-reason vocabulary from `kiro-cli.exe` (PLATFORM-FINDINGS 5a-ter).

    PAUSE_TURN and MODEL_CONTEXT_WINDOW_EXCEEDED are distinct, catchable
    states rather than generic failures. Note DAILY_REQUEST_COUNT: there is a
    daily limit, not only the monthly one.
    """

    CONTENT_FILTERED = "CONTENT_FILTERED"
    END_TURN = "END_TURN"
    MAX_TOKENS = "MAX_TOKENS"
    MODEL_CONTEXT_WINDOW_EXCEEDED = "MODEL_CONTEXT_WINDOW_EXCEEDED"
    PAUSE_TURN = "PAUSE_TURN"
    TOOL_USE = "TOOL_USE"
    MONTHLY_REQUEST_COUNT = "MONTHLY_REQUEST_COUNT"
    DAILY_REQUEST_COUNT = "DAILY_REQUEST_COUNT"
    INSUFFICIENT_MODEL_CAPACITY = "INSUFFICIENT_MODEL_CAPACITY"


@dataclass
class ReasoningBlockMeta:
    """Per-reasoning-block metadata from `reasoningContentEvent`.

    `signature_present` records whether a REAL upstream signature arrived.
    Signatures are never fabricated here — upstream's
    `generate_thinking_signature` ("Generate a placeholder signature") is
    precisely what CONDUIT retires.
    """

    model_id: Optional[str] = None
    signature_present: bool = False


@dataclass
class TurnMetrics:
    """Measured token, credit, and stop accounting for one completed turn."""

    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0          # reasoning, metered separately (5a-ter)
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    credits: float = 0.0             # meteringEvent raw credit count
    context_usage_percentage: float = 0.0
    stop_reason: Optional[StopReason] = None
    reasoning_blocks: List[ReasoningBlockMeta] = field(default_factory=list)

    # Whole-turn context footprint: input + output + Kiro's own server-side
    # system prompt. Inverted from contextUsagePercentage and EXACT (float32
    # round-trip noise only). Distinct from total_tokens, which mirrors it today
    # but is the field upstream would populate if it ever reported real counts.
    context_tokens: int = 0

    # How the token counters above were obtained. The one field that keeps a
    # derived figure from being read as a measured one:
    #   "upstream" — metadataEvent.tokenUsage carried real counts
    #   "derived"  — computed from contextUsagePercentage (see conduit.kiro.derive)
    #   "none"     — no token accounting available; counters are meaningless
    token_accounting: str = "none"
