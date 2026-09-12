"""The CONDUIT -> SAGA emit envelope.

Mirrors CONTRACT-conduit-saga-seam.md v1 exactly. Changing this file requires
editing that contract, bumping its version, and telling WS-C. No lane may
change it unilaterally.

Wire form is the contract's snake_case JSON. The emit is fire-and-forget: it
MUST NOT block, delay, or fail the client response.
"""

from __future__ import annotations

from dataclasses import dataclass

from conduit.contracts.metrics import TurnMetrics
from conduit.contracts.rewritten import RewrittenOut


@dataclass
class SeamEnvelope:
    request_id: str          # echoed from SAGA's x-saga-request-id
    ts: int                  # emit time, ms
    model_id: str            # resolved model actually sent to Kiro
    metrics: TurnMetrics
    rewritten_out: RewrittenOut
