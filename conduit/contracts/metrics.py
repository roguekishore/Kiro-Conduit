"""Real, measured per-turn metrics.

Every number here is parsed off the wire — from `metadataEvent.tokenUsage`,
`meteringEvent.usage`, and `contextUsageEvent`. Nothing is estimated. The
upstream gateway back-calculated input tokens from `contextUsagePercentage`
and estimated output with `len/4` while exact upstream counts were on the
wire and being dropped.

FROZEN. Lane A populates; Lanes C/D/F read.
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
