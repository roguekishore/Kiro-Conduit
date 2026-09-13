# CONDUIT

A faithful translating gateway between model-native harnesses (Claude Code,
Codex CLI) and the Kiro / AWS CodeWhisperer backend — built to carry each model
its **own harness's** context untouched, capture **real** metrics, and tee a copy
of both sides of the rewrite to SAGA for observability.

**AGPL-3.0** (see `LICENSE`) — inherited because foundation code derives from the
upstream [`jwadow/kiro-gateway`](https://github.com/jwadow/kiro-gateway).

## Why this exists (not a fork, a rebuild)

The upstream gateway works but has the wrong core: it **substring-matches** the
response stream instead of dispatching on the AWS event-stream `:event-type`
header. That single choice cascades into every gap CONDUIT fixes.

CONDUIT keeps the hard, correct part (auth) and rewrites the rest.

## Repo layout

```
conduit/
  foundation/          # KEPT from upstream — correct, hard-won, reused as-is
                       #   auth (IdC two-file + profileArn discovery + refresh),
                       #   account manager/rotation, config, http transport,
                       #   cache, error taxonomies, utils
  contracts/           # Frozen shared types — events, metrics, rewritten, seam
  kiro/                # AWS event-stream framing, dispatch, usage parsing,
                       #   request/history/tools/reasoning builders
  doors/               # anthropic.py  — POST /v1/messages
                       # codex.py      — POST /v1/responses
  seam/                # fire-and-forget SAGA emitter
  control/             # usage-limits control plane client
_reference_upstream/   # NOT imported. Read-only reference; remove once replaced.
LICENSE                # AGPL-3.0
```

## What gets rewritten (and why)

Authoritative field/event names come from Amazon's own `kiro-cli.exe`. Each item
is a known gap in the upstream:

| New subsystem | Replaces | Gain |
|---|---|---|
| **Event dispatch by `:event-type` header** | substring match | Sees all ~17 events, not ~4 |
| **Real token capture** (`metadataEvent.tokenUsage`) | back-calc from percentage | true input/output/thought/cache tokens |
| **Credit ledger** (`meteringEvent`) | dropped | real per-request credit count |
| **Native reasoning** (`reasoningContentEvent` + request knob) | tag scraping + placeholder | genuine reasoning + real signature |
| **Native `systemPrompt` slot** | pasted into first user msg | model gets a real system prompt |
| **Merge consecutive turns** | synthetic padding | no history self-pollution |
| **Real tool-result status** | hardcoded `"success"` | truthful on tool failure |
| **`/v1/responses` endpoint** | absent | Codex CLI support |
| **`GetUsageLimits` client** | absent | proactive credit visibility |
| **SAGA dual-capture hook** | absent | emits rewritten payload for diffing |

## Status

Package is importable, `create_app()` builds, and `conduit/contracts/` is frozen.
Nothing has run against real credentials — that step is maintainer-gated.
