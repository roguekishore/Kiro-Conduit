"""Parse metadataEvent, meteringEvent, contextUsageEvent, and reasoningContentEvent
into :class:`~conduit.contracts.metrics.TurnMetrics`.

All numbers are parsed off the wire — nothing is estimated.  Every known field
spelling variant is handled so callers never receive a zero when the real count
was present under an alternate key.

Supported wire spellings
------------------------
``metadataEvent.tokenUsage``
    * ``inputTokens`` / ``uncachedInputTokens``   → ``input_tokens``
    * ``outputTokens``                             → ``output_tokens``
    * ``thoughtTokens``                            → ``thought_tokens``
    * ``cachedReadTokens`` / ``cacheReadInputTokens``  → ``cache_read_tokens``
    * ``cachedWriteTokens`` / ``cacheWriteInputTokens`` → ``cache_write_tokens``
    * ``totalTokens``                              → ``total_tokens``
    * ``stopReason``                               → ``stop_reason``
    * ``stopDetails``                              → ``stop_reason`` (fallback)
    * ``normalizedTokenUsage``                     → used only when primary counts absent

``meteringEvent.usage``
    Accumulated as ``credits`` (float).

``contextUsageEvent``
    ``contextUsagePercentage`` → ``context_usage_percentage``.

``reasoningContentEvent``
    ``modelId`` + ``signature`` → a :class:`~conduit.contracts.metrics.ReasoningBlockMeta`
    appended to ``reasoning_blocks``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger

from conduit.contracts.metrics import ReasoningBlockMeta, StopReason, TurnMetrics

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STOP_REASON_VALUES: frozenset[str] = frozenset(r.value for r in StopReason)


def _coerce_stop_reason(raw: Any) -> Optional[StopReason]:
    """Map a raw wire value to a :class:`StopReason`, or ``None`` on failure."""
    if raw is None:
        return None
    candidate = str(raw).strip().upper()
    # Try exact match first (most values are already upper-case on wire)
    if candidate in _STOP_REASON_VALUES:
        return StopReason(candidate)
    # Try case-insensitive scan
    for member in StopReason:
        if member.value.upper() == candidate:
            return member
    logger.debug(f"usage: unknown stopReason value '{raw}'; ignoring")
    return None


def _int(value: Any, default: int = 0) -> int:
    """Safely coerce *value* to int."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    """Safely coerce *value* to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Per-event parsers
# ---------------------------------------------------------------------------


def apply_metadata_event(metrics: TurnMetrics, payload: Dict[str, Any]) -> None:
    """Merge ``metadataEvent`` fields into *metrics* **in-place**.

    Handles both the outer ``metadataEvent`` wrapper that Kiro sometimes
    includes and the flat ``tokenUsage`` sub-object.

    Args:
        metrics: The :class:`TurnMetrics` instance to update.
        payload: The already-decoded JSON dict from the frame.
    """
    # The payload may arrive as {"metadataEvent": {"tokenUsage": {...}}}
    # or already unwrapped as {"tokenUsage": {...}}
    meta = payload.get("metadataEvent", payload)

    token_usage: Dict[str, Any] = meta.get("tokenUsage", {})
    if not isinstance(token_usage, dict):
        logger.debug(f"usage: metadataEvent.tokenUsage is not a dict ({type(token_usage)}); skipping")
        token_usage = {}

    # --- input_tokens -------------------------------------------------------
    # Primary: inputTokens; alternate: uncachedInputTokens
    if "inputTokens" in token_usage:
        metrics.input_tokens = _int(token_usage["inputTokens"])
    elif "uncachedInputTokens" in token_usage:
        metrics.input_tokens = _int(token_usage["uncachedInputTokens"])

    # --- output_tokens -------------------------------------------------------
    if "outputTokens" in token_usage:
        metrics.output_tokens = _int(token_usage["outputTokens"])

    # --- thought_tokens (reasoning, metered separately) ---------------------
    if "thoughtTokens" in token_usage:
        metrics.thought_tokens = _int(token_usage["thoughtTokens"])

    # --- cache_read_tokens --------------------------------------------------
    if "cachedReadTokens" in token_usage:
        metrics.cache_read_tokens = _int(token_usage["cachedReadTokens"])
    elif "cacheReadInputTokens" in token_usage:
        metrics.cache_read_tokens = _int(token_usage["cacheReadInputTokens"])

    # --- cache_write_tokens -------------------------------------------------
    if "cachedWriteTokens" in token_usage:
        metrics.cache_write_tokens = _int(token_usage["cachedWriteTokens"])
    elif "cacheWriteInputTokens" in token_usage:
        metrics.cache_write_tokens = _int(token_usage["cacheWriteInputTokens"])

    # --- total_tokens -------------------------------------------------------
    if "totalTokens" in token_usage:
        metrics.total_tokens = _int(token_usage["totalTokens"])

    # --- stop_reason --------------------------------------------------------
    # Prefer stopReason; fall back to stopDetails
    raw_stop = token_usage.get("stopReason") or meta.get("stopReason")
    if raw_stop is None:
        # stopDetails may be a nested object {"stopReason": "..."} or a plain string
        stop_details = token_usage.get("stopDetails") or meta.get("stopDetails")
        if isinstance(stop_details, dict):
            raw_stop = stop_details.get("stopReason")
        elif stop_details is not None:
            raw_stop = stop_details

    if raw_stop is not None:
        resolved = _coerce_stop_reason(raw_stop)
        if resolved is not None:
            metrics.stop_reason = resolved

    # --- normalizedTokenUsage (supplementary; use only for fields still 0) --
    normalized: Dict[str, Any] = token_usage.get("normalizedTokenUsage", {})
    if isinstance(normalized, dict) and normalized:
        if metrics.input_tokens == 0 and "inputTokens" in normalized:
            metrics.input_tokens = _int(normalized["inputTokens"])
        if metrics.output_tokens == 0 and "outputTokens" in normalized:
            metrics.output_tokens = _int(normalized["outputTokens"])
        if metrics.total_tokens == 0 and "totalTokens" in normalized:
            metrics.total_tokens = _int(normalized["totalTokens"])


def apply_metering_event(metrics: TurnMetrics, payload: Dict[str, Any]) -> None:
    """Accumulate credits from a ``meteringEvent`` **in-place**.

    Credits are accumulated (not replaced) because multiple ``meteringEvent``
    frames can arrive in a single turn.

    Args:
        metrics: The :class:`TurnMetrics` instance to update.
        payload: The already-decoded JSON dict from the frame.
    """
    # May arrive as {"meteringEvent": {"usage": 1.5}} or flat {"usage": 1.5}
    metering = payload.get("meteringEvent", payload)
    raw_usage = metering.get("usage")
    if raw_usage is not None:
        metrics.credits += _float(raw_usage)
    else:
        logger.debug("usage: meteringEvent arrived with no 'usage' field; ignoring")


def apply_context_usage_event(metrics: TurnMetrics, payload: Dict[str, Any]) -> None:
    """Parse ``contextUsageEvent`` into ``context_usage_percentage`` **in-place**.

    Args:
        metrics: The :class:`TurnMetrics` instance to update.
        payload: The already-decoded JSON dict from the frame.
    """
    # May arrive as {"contextUsageEvent": {"contextUsagePercentage": 42.0}} or flat
    ctx = payload.get("contextUsageEvent", payload)
    raw_pct = ctx.get("contextUsagePercentage")
    if raw_pct is not None:
        metrics.context_usage_percentage = _float(raw_pct)
    else:
        logger.debug("usage: contextUsageEvent arrived without 'contextUsagePercentage'; ignoring")


def apply_reasoning_content_event(metrics: TurnMetrics, payload: Dict[str, Any]) -> None:
    """Parse ``reasoningContentEvent`` into a :class:`ReasoningBlockMeta` **in-place**.

    Records whether a real signature was present.  Signatures are never
    fabricated here.

    Args:
        metrics: The :class:`TurnMetrics` instance to update.
        payload: The already-decoded JSON dict from the frame.
    """
    # May arrive as {"reasoningContentEvent": {...}} or flat
    rc = payload.get("reasoningContentEvent", payload)

    model_id: Optional[str] = rc.get("modelId") or rc.get("model_id")
    signature: Any = rc.get("signature")
    signature_present: bool = bool(signature)

    block = ReasoningBlockMeta(
        model_id=str(model_id) if model_id is not None else None,
        signature_present=signature_present,
    )
    metrics.reasoning_blocks.append(block)
    logger.debug(
        f"usage: reasoningContentEvent parsed — model_id={block.model_id!r}, "
        f"signature_present={block.signature_present}"
    )
