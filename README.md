# CONDUIT

A faithful translating gateway between model-native harnesses (Claude Code,
Codex CLI) and the Kiro / AWS CodeWhisperer backend — built to carry each model
its **own harness's** context untouched, capture **real** metrics, and tee a copy
of both sides of the rewrite to SAGA for observability.

Standalone repo. **AGPL-3.0** (see `LICENSE`) — inherited because foundation code
derives from the upstream `jwadow/kiro-gateway`.

## Why this exists (not a fork, a rebuild)

The upstream gateway works but has the wrong core: it **substring-matches** the
response stream instead of dispatching on the AWS event-stream `:event-type`
header. That single choice cascades into every gap CONDUIT fixes. Full evidence:
`../FORGE/PLATFORM-FINDINGS.md` §5a-bis … §5a-quinquies.

CONDUIT keeps the hard, correct part (auth) and rewrites the rest.

## Repo layout

```
foundation/          # KEPT from upstream — correct, hard-won, reused as-is
                     #   auth (IdC two-file + profileArn discovery + refresh),
                     #   account manager/rotation, config, http transport,
                     #   cache, error taxonomies, utils
_reference_upstream/ # NOT imported. Read-only reference for the REWRITE.
                     #   The wrong-architecture stream parser, converters,
                     #   thinking scraper, tokenizer, routes. Delete once the
                     #   fresh implementations land.
LICENSE              # AGPL-3.0
```

Nothing in `_reference_upstream/` is wired into the running gateway. It is there
to read while writing the header-dispatch replacements, then to remove.

## What gets rewritten (and why) — the build list

Authoritative field/event names came from Amazon's own `kiro-cli.exe`
(`PLATFORM-FINDINGS.md` §5a-ter/quater). Each item is a known gap in the upstream:

| New subsystem | Replaces | Gain |
|---|---|---|
| **Event dispatch by `:event-type` header** | `parsers.py` substring match | Sees all ~17 events, not ~4. Root fix; unlocks the rest. |
| **Real token capture** (`metadataEvent.tokenUsage`) | back-calc from `contextUsagePercentage` | true `inputTokens`/`outputTokens`/`thoughtTokens`/cacheRead/cacheWrite |
| **Credit ledger** (`meteringEvent`) | dropped today | real per-request credit count |
| **Native reasoning** (`reasoningContentEvent` + `output_config`/`adaptive` request knob) | fake `<thinking>` tag scraping + placeholder signature | genuine reasoning + real signature |
| **Native `systemPrompt` slot** | system prompt pasted into first user msg | model receives a real system prompt |
| **Merge consecutive turns** | `(empty placeholder)` synthetic padding | no self-pollution of history |
| **Real tool-result status** | hardcoded `"status":"success"` | Kiro told the truth on tool failure |
| **`/v1/responses` endpoint** | absent | Codex CLI → Luna can connect |
| **`GetUsageLimits` client** | absent | proactive "credits remaining", not reactive 402 |
| **SAGA dual-capture hook** | absent | emits rewritten-out payload so SAGA can diff injections |

## Seam with SAGA (pin before parallel build)

CONDUIT and SAGA meet at one contract, defined in
`../PARALLEL-WORKSTREAMS.md`: CONDUIT emits (a) real metrics and (b) its final
rewritten-to-Kiro payload over a side channel; SAGA consumes both. Build against
that contract, not against each other's internals.

## Status

**Lane 0 done; Lanes A–F open.** The package is importable, `create_app()` builds,
and `conduit/contracts/` is frozen — so the six build lanes can run in parallel.
Lane briefs, file ownership, and per-lane verification gates are in
`SPEC-WORKSTREAMS.md`.

No Kiro protocol code is implemented yet, and nothing has run against real
credentials — that step is maintainer-gated. Don't point a harness at this yet.
