"""
Reasoning / extended-thinking knobs for the Kiro request pipeline.

Kiro takes an *effort enum*, not a token-budget integer.  The upstream gateway
converts reasoning_effort to an Anthropic-style token budget — that is wrong.
This module maps the effort string to the enum Kiro expects and builds the
correct ``output_config`` / ``reasoning`` blocks per model family.

Model families (authoritative from kiro-cli.exe)
-------------------------------------------------
Claude models : ``claude-*``  and ``auto``
  → ``output_config: {"effort": <enum>}``
  → ``thinking:      {"type": "adaptive", "display": "summarized"}``

GPT models    : ``gpt-*``
  → ``reasoning: {"effort": <enum>}``

Effort enum values: ``low``, ``medium``, ``high``, ``xhigh``, ``max``

Public API
----------
- EFFORT_VALUES    — frozenset of valid effort strings
- normalize_effort() — coerce / validate an effort string
- build_reasoning_fields() — return the correct wire fields for a model+effort
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Effort enum
# ---------------------------------------------------------------------------

EFFORT_VALUES: frozenset = frozenset({"low", "medium", "high", "xhigh", "max"})

# Sensible default when the caller provides no explicit effort
_DEFAULT_EFFORT = "high"


def normalize_effort(effort: Optional[str]) -> str:
    """
    Coerce an effort string to one of the five valid enum values.

    If the value is ``None`` or unrecognised, the default ``"high"`` is
    returned and a debug message is logged.

    Args:
        effort: Raw effort string from the incoming request.

    Returns:
        One of ``"low"``, ``"medium"``, ``"high"``, ``"xhigh"``, ``"max"``.
    """
    if effort is None:
        return _DEFAULT_EFFORT
    low = effort.strip().lower()
    if low in EFFORT_VALUES:
        return low
    logger.debug(
        f"Unrecognised reasoning_effort {effort!r}; "
        f"falling back to {_DEFAULT_EFFORT!r}"
    )
    return _DEFAULT_EFFORT


def build_reasoning_fields(
    model_id: str,
    effort: Optional[str],
) -> Dict[str, Any]:
    """
    Build the ``additionalModelRequestFields`` sub-dict that encodes reasoning
    knobs for the given model and effort level.

    Claude models (``claude-*`` / ``auto``) get:

    .. code-block:: json

        {
            "output_config": {"effort": "<effort>"},
            "thinking":      {"type": "adaptive", "display": "summarized"}
        }

    GPT models (``gpt-*``) get:

    .. code-block:: json

        {
            "reasoning": {"effort": "<effort>"}
        }

    Other / unknown model IDs receive the Claude layout as a safe default.

    The effort value is always one of the five enum strings; it is NEVER
    converted to a token-budget integer.

    Args:
        model_id: Resolved Kiro model ID (e.g. ``"claude-sonnet-4.6"``).
        effort:   Raw effort string from the client, or ``None`` for the default.

    Returns:
        Dict ready for placement in ``additionalModelRequestFields``.
    """
    normed = normalize_effort(effort)
    model_lower = (model_id or "").lower()

    if model_lower.startswith("gpt-"):
        fields: Dict[str, Any] = {
            "reasoning": {"effort": normed},
        }
        logger.debug(
            f"build_reasoning_fields: GPT model {model_id!r} → "
            f"reasoning.effort={normed!r}"
        )
    else:
        # Claude (claude-*, auto) and unknown models
        fields = {
            "output_config": {"effort": normed},
            "thinking": {"type": "adaptive", "display": "summarized"},
        }
        logger.debug(
            f"build_reasoning_fields: Claude/auto model {model_id!r} → "
            f"output_config.effort={normed!r}"
        )

    return fields
