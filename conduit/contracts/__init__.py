"""Frozen shared contracts.

EVERY lane reads these; only Lane 0 writes them. They exist so that six
parallel lanes cannot each invent an incompatible TurnMetrics.

Lanes C and D consume Lane A's and Lane B's output THROUGH these types, never
through each other's implementation files — which is exactly what lets the
lanes run in parallel.
"""

from conduit.contracts.events import (
    DispatchedEvent,
    EventName,
    KiroEvent,
    UnknownEvent,
)
from conduit.contracts.metrics import (
    ReasoningBlockMeta,
    StopReason,
    TurnMetrics,
)
from conduit.contracts.rewritten import (
    Injection,
    InjectionType,
    RewrittenOut,
)
from conduit.contracts.seam import SeamEnvelope

__all__ = [
    "DispatchedEvent",
    "EventName",
    "KiroEvent",
    "UnknownEvent",
    "ReasoningBlockMeta",
    "StopReason",
    "TurnMetrics",
    "Injection",
    "InjectionType",
    "RewrittenOut",
    "SeamEnvelope",
]
