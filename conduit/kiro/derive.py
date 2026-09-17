"""Token accounting recovered from `contextUsagePercentage` — Lane A.

Kiro's `metadataEvent` on this backend carries **only** `{"stopReason": ...}`:
no `tokenUsage`, no counters. Verified by dumping the raw payload off the wire
(2026-09-14). The six-counter `Usage` struct in `kiro-cli.exe` is defined by the
protocol and not populated by this endpoint, so nothing can be read directly.

What *is* on the wire is `contextUsageEvent.contextUsagePercentage`, and it is a
far better signal than its name suggests.

## What was measured (controlled probes, 2026-09-14, claude-sonnet-4-6)

**It inverts to exact integers.** ``pct / 100 * window`` lands on a whole number
every time — deviations of 0.0002–0.01 of a token, which is float32 round-trip
noise and nothing else:

    0.41120001673698425 -> 4112.0002
    0.7120999693870544  -> 7120.9997
    1.0120999813079834  -> 10120.9998

**It is linear in input.** Three requests differing only by a filler block of F
and 2F characters moved the percentage by +0.3009 then +0.3000 — 3009 then 3000
tokens for 12831 then 12800 characters.

**It is deterministic.** Byte-identical requests produce byte-identical
percentages (``0.41120001673698425`` twice, ``0.4745999872684479`` twice).

**It counts input AND output.** Holding input constant while output grew from 2
to 964 to 3009 characters moved the inverted total 4112 -> 4319 -> 4757. So the
percentage is sampled *after* generation and the figure it encodes is the whole
turn's context footprint, not the prompt alone.

## What this module can and cannot know

``context_tokens`` — the sum — is **measured**. It is exact.

Splitting that sum is **not** measured. Two obstacles: Kiro prepends a
server-side system prompt worth ~4100 tokens on a minimal request, invisible to
CONDUIT (48 characters of user text cannot produce 4112 tokens); and no
tokenizer for this backend exists locally. So the split is estimated from the
output text CONDUIT streamed, using a chars-per-token ratio measured against the
probes above, and the input is what remains.

The error therefore lands on `output_tokens` in relative terms (a few percent)
and on `input_tokens` in absolute terms only (a few tokens against thousands,
well under one percent). That asymmetry is deliberate: estimating the *large*
side and subtracting would put a 5% error on input straight onto the small
output figure and inflate it by a third.

**Nothing here is labelled as upstream-reported.** `TurnMetrics.token_accounting`
carries `"derived"` so the seam, SAGA, and any later reader can tell a computed
figure from a measured one. That distinction is the whole point of this door.
"""

from __future__ import annotations

from typing import Dict, Optional

from loguru import logger

from conduit.contracts.metrics import TurnMetrics

# ---------------------------------------------------------------------------
# Context windows
# ---------------------------------------------------------------------------
# A wrong window silently scales every number, so an id is listed only where two
# independent sources agree: the window fitted from captured percentages (which
# must land on integers) and the window the model picker in
# `~/.claude/kiro.settings.json` declares.
#
# Deliberately absent: `claude-sonnet-5`. Its picker entry says 1M while the
# fitted window across the captured rows says 500K, and n=3 is too thin to break
# the tie. An unknown id derives nothing rather than reporting a scaled guess.

CONTEXT_WINDOWS: Dict[str, int] = {
    "claude-opus-5": 1_000_000,
    "claude-opus-4-8": 1_000_000,
    "claude-opus-4-7": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,   # directly calibrated
    "claude-sonnet-4-5": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-haiku-4-5": 200_000,
    "gpt-5.6-luna": 272_000,
    "gpt-5.6-sol": 272_000,
    "gpt-5.6-terra": 272_000,
}

# Characters per output token. Measured on the probes above: 3009 characters of
# prose resolved to 641 tokens (4.69) and 964 characters to 203 (4.75). Prose
# only -- code and JSON tokenize denser, so a tool-heavy turn will skew low.
_OUTPUT_CHARS_PER_TOKEN = 4.7

# Below this the percentage cannot be trusted to carry a turn: 0.0 is also what
# an absent contextUsageEvent leaves behind, and the two are indistinguishable.
_MIN_PCT = 1e-9


def normalize_model_id(model: str) -> str:
    """Reduce a client-facing model string to a bare id for window lookup.

    Strips a context-window suffix (``claude-sonnet-5[1m]``) the way
    `model_resolver` does, and normalizes dotted minors to hyphens so
    ``claude-sonnet-4.6`` and ``claude-sonnet-4-6`` resolve alike. GPT tier ids
    keep their dots -- they are spelled that way on the wire.
    """
    mid = model.strip()
    if "[" in mid:
        mid = mid.split("[", 1)[0]
    if mid.startswith("claude-"):
        mid = mid.replace(".", "-")
    return mid


def context_window_for(model: str) -> Optional[int]:
    """Return the context window for *model*, or None when it is not known."""
    return CONTEXT_WINDOWS.get(normalize_model_id(model))


def tokens_from_context_percentage(pct: float, window: int) -> int:
    """Invert ``contextUsagePercentage`` into whole context tokens.

    Rounds because the wire value is a float32 round-trip of an integer; the
    residual is ~1e-4 of a token, never a real fraction.
    """
    return int(round(pct / 100.0 * window))


def derive_token_accounting(
    metrics: TurnMetrics,
    model: str,
    output_chars: int,
) -> None:
    """Fill in token counts from `contextUsagePercentage`, **in-place**.

    A no-op when upstream already reported real counts -- a measured figure is
    never overwritten by a derived one -- and a no-op when the percentage is
    missing or the model's window is unknown, leaving the counters at zero so a
    caller can tell "not reported" from "reported as zero".

    Args:
        metrics: The turn's metrics; mutated.
        model: Model id as sent to Kiro, for the window lookup.
        output_chars: Characters CONDUIT streamed to the client this turn --
            assistant text plus tool-call JSON.
    """
    if metrics.token_accounting == "upstream":
        return

    if metrics.context_usage_percentage <= _MIN_PCT:
        logger.debug(
            "derive: no contextUsagePercentage on this turn; token counts stay unmeasured"
        )
        return

    window = context_window_for(model)
    if window is None:
        logger.warning(
            f"derive: no known context window for model {model!r}; "
            "token counts stay unmeasured (add it to CONTEXT_WINDOWS once "
            "the window is confirmed from two independent sources)"
        )
        return

    total = tokens_from_context_percentage(metrics.context_usage_percentage, window)
    if total <= 0:
        return

    est_output = int(round(output_chars / _OUTPUT_CHARS_PER_TOKEN))
    # The sum is measured, so the split may never exceed it.
    est_output = max(0, min(est_output, total))

    metrics.context_tokens = total
    metrics.output_tokens = est_output
    metrics.input_tokens = total - est_output
    metrics.total_tokens = total
    metrics.token_accounting = "derived"

    logger.debug(
        f"derive: ctx={metrics.context_usage_percentage}% window={window} "
        f"-> total={total} (input={metrics.input_tokens} "
        f"output~{est_output} from {output_chars} chars)"
    )
