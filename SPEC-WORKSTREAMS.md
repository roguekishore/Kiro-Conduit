# CONDUIT — Workstream Spec (parallel build)

*Written 2026-09-12. Derived from `HANDOFF-conduit-build.md`, `FORGE/PLATFORM-FINDINGS.md`,
`analysis/FINDINGS-codex.md`, `CONTRACT-conduit-saga-seam.md`. Supersedes the handoff's
build-order list, which assumed a scaffold state that does not match disk.*

**Audience: an orchestrator agent that will fan out lanes to subagents.**
Read §0 → §1 → §2, then dispatch. Each lane brief in §5 is self-contained and
copy-pasteable as a subagent prompt. Facts are inlined with citations so no lane
needs to re-derive the protocol.

---

## 0. Orchestrator instructions — read this first

### The fan-out is NOT flat. There is one blocking lane.

```
LANE 0 (blocker, runs ALONE to completion)
   │  repo is not importable today; 4 hard blockers, see §1
   ▼
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│  LANE A  │  LANE B  │  LANE C  │  LANE D  │  LANE E  │  LANE F  │
│ response │ request  │Anthropic │  Codex   │ control  │   SAGA   │
│  parse   │  build   │   door   │   door   │  plane   │   seam   │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
        all six run fully in parallel after Lane 0 lands
```

**Do not dispatch A–F until Lane 0 is merged and `python -c "import conduit"`
succeeds.** Lane 0 creates the type contracts every other lane imports. Dispatch
A–F before it exists and six agents will each invent an incompatible `TurnMetrics`.

Lanes A–F are parallel because Lane 0 freezes their shared types as real,
importable dataclasses. C and D consume A's and B's output *through those types*,
never through each other's implementation files — so they never need to wait.

### Per-lane contract

Every lane brief in §5 carries: files it OWNS (exclusive write), files it may READ,
the frozen types it must not redefine, its verification gate, and its reference
implementations. **No two lanes write the same file.** If a lane believes it must
edit a file it does not own, it stops and reports instead of editing.

### Global rules (bind every lane)

1. **Do not run against real credentials.** No live Kiro calls, no real auth. All
   verification is against synthetic fixtures the lane constructs itself. This is a
   maintainer-gated step, not a lane's call.
2. **AGPL hygiene.** `_reference_upstream/` is READ-ONLY reference. Read it to
   understand shape, then write your own implementation. Do not copy lines verbatim
   into new files. `foundation/` already carries AGPL intentionally (it is upstream
   code, kept).
3. **Python 3.14.3**, FastAPI + httpx + pydantic + loguru (all already importable
   on this machine). Match `foundation/`'s existing style: module docstrings,
   type-annotated signatures, `loguru` for logging.
4. **Delete as you replace.** When a lane's implementation makes a
   `_reference_upstream/` module obsolete, that lane deletes it in its final commit
   and says so. Nothing in `_reference_upstream/` may be imported by shipped code.
5. **Commit as you work**, small and scoped. Repo is at `D:/PROJECTS/AI/CONDUIT`,
   branch `master`, no remote — do not add one.
6. Report at the end: what is done and verified, what is stubbed, and any place the
   real protocol contradicted the evidence base.

---

## 1. Actual repo state — CORRECTED against disk

The handoff says "already scaffolded" and "core has NO external deps — start
immediately." **Both are wrong.** Verified on disk 2026-09-12:

| Handoff claim | Reality |
|---|---|
| Scaffold at `CONDUIT/` | Was at `kiro-conduit/`; `CONDUIT/` held only a duplicate README. **Resolved** — consolidated into `CONDUIT/`, 27 files, no loss. |
| `foundation/` reusable as-is | Imports itself as `kiro.*` (14 import sites). Package name does not exist. |
| Core has no blockers | **4 blockers**, below. Nothing can even be imported. |

### The four blockers (all owned by Lane 0)

1. **Broken import prefix.** `foundation/` does `from kiro.config import …` at 14
   sites across 7 modules (`account_manager.py:48-65`, `auth.py:44,55`,
   `cache.py:33`, `http_client.py:41-44`, plus `config.py`, `exceptions.py`,
   `utils.py`). There is no `kiro` package.
2. **No `__init__.py` anywhere.** Neither `foundation/` nor the repo root is an
   importable package.
3. **`foundation/` depends on a file marked for deletion.**
   `foundation/account_manager.py:50` does
   `from kiro.model_resolver import ModelResolver, normalize_model_name`, but
   `model_resolver.py` (446 lines) lives in `_reference_upstream/` — the directory
   the handoff says to delete. Used at `account_manager.py:160,559,708,797`.
   **This module must be rescued into the real package, not deleted.** It is model
   name/ID resolution, which is foundation-grade, not wrong-architecture.
4. **No app entry point exists.** `grep 'FastAPI('` returns nothing across the
   whole tree. There is no `main.py`, no app factory, no router registration, no
   `pyproject.toml`, no `requirements.txt`. The server was never copied.

### What IS correct and reusable

`foundation/` (10 modules, 4,273 lines) — auth (IdC two-file, `profileArn`
discovery, refresh), account rotation, config, http transport, cache, error
taxonomies, utils. Public surface:

| Module | Key exports |
|---|---|
| `auth.py` | `KiroAuthManager`, `AuthType`, `resolve_profile_arn()` |
| `account_manager.py` | `AccountManager`, `Account`, `AccountStats`, `ModelAccountList` |
| `http_client.py` | `KiroHttpClient` |
| `config.py` | env-driven settings + `get_kiro_api_host()`, `get_kiro_q_host()`, `get_kiro_refresh_url()`, `get_kiro_profile_url()`, `get_kiro_mcp_url()`, `get_aws_sso_oidc_url()` |
| `cache.py` | `ModelInfoCache` |
| `utils.py` | `get_kiro_headers()`, `get_machine_fingerprint()`, `generate_conversation_id()`, `generate_tool_call_id()`, `generate_completion_id()` |
| `network_errors.py` | `classify_network_error()`, `NetworkErrorInfo`, `ErrorCategory` |
| `account_errors.py` | `classify_error()`, `ErrorType` |
| `kiro_errors.py` | `enhance_kiro_error()`, `KiroErrorInfo` |
| `exceptions.py` | `validation_exception_handler()`, `sanitize_validation_errors()` |

---

## 2. ⚠ Evidence-base hazard — read before any lane touches the system prompt

`FORGE/PLATFORM-FINDINGS.md` **contradicts itself**, and the wrong half reads as
more authoritative because it is longer and cites triangulation.

- **§2c (lines 222-272) says: "CodeWhisperer has no system-prompt field."** It
  concludes the prompt must be prepended into the first user message and recommends
  *not* building a new gateway.
- **§5a-quater (lines 630+) says: "CORRECTION — Kiro HAS a native system-prompt
  field."** `conversationState` carries `systemPrompt` alongside `agentMode` and
  `additionalModelRequestFields`, extracted from Amazon's own `kiro-cli.exe`.

**§5a-quater supersedes §2c.** §2c was written from third-party gateway
triangulation (all of which prepend); §5a-quater is from Amazon's own binary. §2c's
text was never retracted in place.

A lane that reads §2c and stops will build the prepend hack this project exists to
remove. **Lane B builds to §5a-quater.** Caveat that stays honest: the field *name*
is certain, its exact placement in the envelope is inferred from string adjacency,
and `kiro-cli` may speak a newer protocol version than the gateway targets — so
Lane B ships it behind a config flag with the prepend path as fallback (see §5,
Lane B).

Same pattern applies to §2b vs the alternation-padding claim: what looked like
Kiro's limits are mostly the specific request shape the upstream gateway built.

---

## 3. Target tree and file ownership map

Exclusive write ownership. This table is the collision-avoidance contract.

```
CONDUIT/
  pyproject.toml                    L0
  .gitignore                        L0
  README.md                         L0 (status line only)
  SPEC-WORKSTREAMS.md               — this file, nobody edits
  LICENSE                           — frozen
  conduit/
    __init__.py                     L0
    app.py                          L0   app factory + ALL router registration
    config.py                       L0   (moved from foundation/, rebranded)
    contracts/                      L0   ← EVERY OTHER LANE READS, NONE WRITES
      __init__.py
      events.py                       KiroEvent union + EventName
      metrics.py                      TurnMetrics, ReasoningBlockMeta, StopReason
      rewritten.py                    RewrittenOut, Injection, InjectionType
      seam.py                         SeamEnvelope
    foundation/                     L0 renames; FROZEN for A-F
      model_resolver.py             L0 (rescued from _reference_upstream/)
    kiro/
      frame.py                      LA   AWS event-stream frame decode
      dispatch.py                   LA   :event-type → typed event
      usage.py                      LA   metadataEvent + meteringEvent → TurnMetrics
      request.py                    LB   conversationState builder
      history.py                    LB   turn merge
      tools.py                      LB   tool specs + real result status
      reasoning.py                  LB   output_config/thinking request knobs
    doors/
      __init__.py                   L0
      anthropic.py                  LC   /v1/messages
      anthropic_sse.py              LC
      codex.py                      LD   /v1/responses
      codex_dialect.py              LD   standard vs responses-lite
      codex_sse.py                  LD
    control/
      usage_limits.py               LE   GetUsageLimits
    seam/
      emit.py                       LF   fire-and-forget POST to SAGA
  tests/
    contracts/                      L0
    kiro_parse/                     LA
    kiro_request/                   LB
    door_anthropic/                 LC
    door_codex/                     LD
    control/                        LE
    seam/                           LF
```

`app.py` is written **once by Lane 0** with every router already imported and
registered. Door lanes fill in routers Lane 0 stubs — they never edit `app.py`.
That removes the only real collision point.

---

## 4. Frozen contracts — Lane 0 writes these VERBATIM

These are the interfaces. Lane 0 has no design freedom here; A–F have no right to
change them. Field names come from the evidence base, cited.

### `conduit/contracts/events.py`

Wire event names are lowerCamel on the `:event-type` header. The ~17 names below are
verbatim from Amazon's `kiro-cli.exe` (`PLATFORM-FINDINGS` §5a-bis). The upstream
gateway saw only 4.

```python
class EventName(str, Enum):
    ASSISTANT_RESPONSE   = "assistantResponseEvent"
    REASONING_CONTENT    = "reasoningContentEvent"
    TOOL_USE             = "toolUseEvent"
    TOOL_RESULT          = "toolResultEvent"
    METERING             = "meteringEvent"
    METADATA             = "metadataEvent"
    MESSAGE_METADATA     = "messageMetadataEvent"
    CONTEXT_USAGE        = "contextUsageEvent"
    CODE                 = "codeEvent"
    CODE_REFERENCE       = "codeReferenceEvent"
    FOLLOWUP_PROMPT      = "followupPromptEvent"
    INTENTS              = "intentsEvent"
    INVALID_STATE        = "invalidStateEvent"
    SUPPLEMENTARY_LINKS  = "supplementaryWebLinksEvent"
    CITATION             = "citationEvent"
    DOCUMENT_CITATION    = "documentCitationEvent"
    MESSAGE_STOP         = "messageStopEvent"
```

Plus `UnknownEvent(name: str, payload: dict)` — an unrecognized `:event-type` is
logged and carried, never dropped silently and never guessed at by payload shape.

### `conduit/contracts/metrics.py`

```python
class StopReason(str, Enum):
    CONTENT_FILTERED = "CONTENT_FILTERED"
    END_TURN = "END_TURN"
    MAX_TOKENS = "MAX_TOKENS"
    MODEL_CONTEXT_WINDOW_EXCEEDED = "MODEL_CONTEXT_WINDOW_EXCEEDED"
    PAUSE_TURN = "PAUSE_TURN"
    TOOL_USE = "TOOL_USE"
    MONTHLY_REQUEST_COUNT = "MONTHLY_REQUEST_COUNT"
    DAILY_REQUEST_COUNT = "DAILY_REQUEST_COUNT"          # a DAILY limit exists
    INSUFFICIENT_MODEL_CAPACITY = "INSUFFICIENT_MODEL_CAPACITY"

@dataclass
class ReasoningBlockMeta:
    model_id: str | None
    signature_present: bool

@dataclass
class TurnMetrics:
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0          # reasoning, metered separately (§5a-ter)
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    credits: float = 0.0             # meteringEvent raw credit count
    context_usage_percentage: float = 0.0
    stop_reason: StopReason | None = None
    reasoning_blocks: list[ReasoningBlockMeta] = field(default_factory=list)
```

Every number here is **measured, never estimated**. No `len/4`, no back-calculation
from `contextUsagePercentage`.

### `conduit/contracts/rewritten.py`

```python
class InjectionType(str, Enum):
    THINKING_TAGS       = "thinking_tags"
    SYSTEM_PROMPT_SLOT  = "system_prompt_slot"
    TURN_MERGE          = "turn_merge"
    RELOCATED_TOOL_DOCS = "relocated_tool_docs"

@dataclass
class Injection:
    type: InjectionType
    location: str        # e.g. "systemPrompt", "history", "last_user_message"

@dataclass
class RewrittenOut:
    system_prompt: str | None
    current_message: dict
    history: list[dict]
    tools: list[dict]
    injections: list[Injection]
```

### `conduit/contracts/seam.py`

Mirrors `CONTRACT-conduit-saga-seam.md` v1 exactly. Changing this file requires
editing that contract and telling WS-C — no lane may change it unilaterally.

```python
@dataclass
class SeamEnvelope:
    request_id: str          # echoed from x-saga-request-id
    ts: int                  # emit time, ms
    model_id: str            # resolved model actually sent to Kiro
    metrics: TurnMetrics
    rewritten_out: RewrittenOut
```

---

## 5. Lane briefs

### LANE 0 — Package rehab + contracts + app skeleton  ⛔ BLOCKER, RUNS ALONE

**Why first:** the repo cannot be imported. Six parallel lanes would each invent
their own shared types.

**Owns:** `pyproject.toml`, `.gitignore`, `conduit/__init__.py`, `conduit/app.py`,
`conduit/config.py`, all of `conduit/contracts/`, the `foundation/` rename,
`foundation/model_resolver.py`, `conduit/doors/__init__.py`, `tests/contracts/`,
README status line.

**Tasks:**
1. Create `conduit/` package. Move `foundation/` → `conduit/foundation/`, add
   `__init__.py` at every level.
2. **Rewrite all 14 `kiro.*` imports to `conduit.foundation.*`** (sites listed in
   §1.1). Move `config.py` to `conduit/config.py` and point imports there.
3. **Rescue `_reference_upstream/model_resolver.py` → `conduit/foundation/model_resolver.py`.**
   It is a hard dependency of `account_manager.py:50` (used at `:160,559,708,797`).
   Keep `ModelResolver`, `normalize_model_name`, `to_runtime_model_id`,
   `get_model_id_for_kiro`, `extract_model_family`, `ModelResolution`. This file is
   upstream code being *kept*, so leave its AGPL header intact — same status as
   `foundation/`. Do not rewrite its logic; only fix imports.
4. Write `conduit/contracts/` **verbatim from §4**. No additions, no renames.
5. Write `conduit/app.py`: FastAPI factory, `/health`, `/v1/models`, and
   registration for four routers that Lane C/D/E fill in. Ship each as an empty
   `APIRouter` in its own lane-owned file so registration never changes:
   `doors/anthropic.py:router`, `doors/codex.py:router`,
   `control/usage_limits.py:router`. Create those files containing only
   `router = APIRouter()` plus a module docstring naming the owning lane.
6. `pyproject.toml`: name `conduit`, requires-python `>=3.13`, deps
   `fastapi`, `httpx`, `pydantic`, `loguru`, `python-dotenv`; dev dep `pytest`.
7. `.gitignore`: `__pycache__/`, `*.pyc`, `.env`, `.venv/`, `*.db`, `debug/`, and
   **credential/state files** — `accounts.json`, `accounts_state.json`,
   `kiro-creds*.json`. Auth material must never be committable.
8. Rebrand `config.py` `APP_TITLE`/`APP_DESCRIPTION` to CONDUIT. Keep every env var
   name unchanged — `foundation/` reads them and existing `.env` files depend on
   them.

**Gate:** `python -c "import conduit; from conduit.app import create_app; create_app()"`
succeeds. `pytest tests/contracts` passes (assert every contract type imports and
round-trips through `dataclasses.asdict`). `grep -rn "^from kiro\|^import kiro" conduit/`
returns nothing.

**Do NOT:** implement any parsing, translation, door, or emit logic. Lane 0 is
plumbing and contracts only. Do not delete anything from `_reference_upstream/`
except `model_resolver.py`, which you moved.

---

### LANE A — Kiro response path: event dispatch + real metrics

**The root fix.** Handoff steps 1, 2, 3, and the parse half of 4.

**Owns:** `conduit/kiro/frame.py`, `dispatch.py`, `usage.py`, `tests/kiro_parse/`.
**Reads:** `conduit/contracts/*`, `_reference_upstream/parsers.py` (to see what NOT
to do), `FORGE/reference/pi-provider-kiro/src/event-parser.ts`,
`FORGE/reference/Kiro-GoPlus/proxy/kiro.go`.

**The defect being replaced:** upstream substring-matches raw decoded text for 7
patterns (`_reference_upstream/parsers.py:241-249`) — `{"content":`, `{"name":`,
`{"input":`, `{"stop":`, `{"followupPrompt":`, `{"usage":`,
`{"contextUsagePercentage":`. Those 7 patterns resolve to only **4** union members.
`{"name":` and `{"input":` are generic JSON shapes, so tool arguments and MCP
payloads can false-trigger tool events. ~13 members are invisible.

**Tasks:**
1. `frame.py` — decode the AWS event-stream binary framing: total length, headers
   length, prelude CRC, header key/value pairs, payload, message CRC. Extract the
   `:event-type` header. `AIClient2API`'s `_readEventTypeHeader`
   (`claude-kiro.js:2553+`) is a complete reference for the header parse. Handle
   partial frames across chunk boundaries.
2. `dispatch.py` — `switch` on `:event-type`, emit the typed events from
   `contracts/events.py`. Header first; payload-shape fallback **only** when the
   header is absent. Unknown event name → `UnknownEvent`, logged at debug, carried
   through. Never guess.
3. `usage.py` — parse `metadataEvent.tokenUsage` into `TurnMetrics`. **Parse every
   known spelling** and normalize; §5a-ter warns the wire may use either pair:
   - `inputTokens` / `uncachedInputTokens` → `input_tokens`
   - `outputTokens` → `output_tokens`
   - `thoughtTokens` → `thought_tokens`
   - `cachedReadTokens` / `cacheReadInputTokens` → `cache_read_tokens`
   - `cachedWriteTokens` / `cacheWriteInputTokens` → `cache_write_tokens`
   - `totalTokens` → `total_tokens`
   - `stopReason` → `StopReason`; also read `stopDetails`, `normalizedTokenUsage`
4. Same module: `meteringEvent.usage` → `credits` (raw credit count, accumulated
   across the turn), `contextUsageEvent` → `context_usage_percentage`.
5. `reasoningContentEvent` → dispatch it with `text` + `signature` and record
   `ReasoningBlockMeta`. Lane B owns *requesting* reasoning; Lane C owns *rendering*
   it. Lane A only parses.

**MUST NOT:**
- **No consecutive-duplicate dedup.** Upstream keeps `last_content` and silently
  drops any chunk equal to the previous one (`parsers.py:254,342-344`). Legitimate
  output repeats constantly — indentation runs, `\n\n`, repeated punctuation. This
  silently deletes model output. Do not reimplement it.
- No token estimation, no back-calculation from `contextUsagePercentage`.

**Gate:** `pytest tests/kiro_parse`. Build synthetic frames yourself — one fixture
per event name in `EventName`, plus: a frame split across two chunks, a frame with
an unknown `:event-type`, a `metadataEvent` in each cache-field spelling, and a
content delta that repeats the previous delta verbatim (assert it survives).

**Delete on completion:** `_reference_upstream/parsers.py`.

---

### LANE B — Kiro request path: systemPrompt, turn merge, tool status, reasoning knob

Handoff steps 5, 6, 7, and the request half of 4.

**Owns:** `conduit/kiro/request.py`, `history.py`, `tools.py`, `reasoning.py`,
`tests/kiro_request/`.
**Reads:** `conduit/contracts/*`, `_reference_upstream/converters_core.py`,
`converters_anthropic.py`, `payload_guards.py`.

**⚠ Read §2 of this spec before starting.** `PLATFORM-FINDINGS` §2c says Kiro has no
system-prompt field; §5a-quater corrects it from Amazon's own binary. Build to
§5a-quater.

**Tasks:**
1. `request.py` — build `conversationState`: `currentMessage` + `history` +
   `conversationId` + `chatTriggerType`, plus `profileArn`, `agentMode`,
   `additionalModelRequestFields`, `systemPrompt`. The flat-list → currentMessage +
   history restructure is genuinely protocol-forced; keep it. Return a
   `RewrittenOut` alongside the wire payload, with `injections` self-declared.
2. **`systemPrompt` slot.** Send the harness system prompt to
   `conversationState.systemPrompt` instead of prepending into the first user
   message (upstream does the latter at `converters_core.py:1491-1506`).
   **Ship behind a config flag** — `KIRO_NATIVE_SYSTEM_PROMPT`, default on — with
   the prepend path retained as fallback. Reason: the field *name* is certain but
   its placement is inferred from string adjacency, and `kiro-cli` may speak a newer
   protocol version. Declare `SYSTEM_PROMPT_SLOT` at `"systemPrompt"` when the slot
   is used, `THINKING_TAGS` at `"last_user_message"` if the fallback runs.
3. `history.py` — **merge consecutive same-role turns.** Upstream pads with
   `(empty placeholder)` synthetic assistant turns; §2b verified that padding is
   *not* forced. Merging avoids polluting history with text the model didn't write.
   Declare `TURN_MERGE` at `"history"`.
4. `tools.py` — tool specs to `tools[].toolSpecification.inputSchema`. Tool calling
   is first-class native, and upstream maps it correctly, so keep the shape. Note
   `outputSchema`, `annotations`, `icons` exist and are unused — leave them unused.
5. **Tool-result status fix.** Upstream hardcodes `"status": "success"` at
   `converters_core.py:738` and `:765`; the Anthropic `is_error` flag is parsed
   (`models_anthropic.py:108`) but only ever read for token counting
   (`tokenizer.py:161-162`). Amazon's schema has a real `status` field taking
   `error`/`success`, plus `ToolResultContentBlock{toolUseId, structuredContent,
   isError}`. Read `is_error` and emit the truth.
   **Design decision you must make and record:** Codex's `FunctionCallOutput` carries
   no item-level success/error flag (`FINDINGS-codex` §1.5), so on the Codex door the
   status cannot come from the wire. Define the status source as an explicit
   parameter of your tool-result builder so Lane D can supply it from the tool
   runner rather than defaulting to a lie. Document the chosen default in the
   module docstring.
6. `reasoning.py` — request knobs. Claude models: `output_config: {effort}` plus
   `thinking: {type: "adaptive", display: "summarized"}`. GPT models:
   `reasoning: {effort}`. Effort enum: `low`, `medium`, `high`, `xhigh`, `max`.
   Upstream converts `reasoning_effort` into an Anthropic-style token budget
   (`converters_openai.py:300-328`) — **that is the wrong knob**; Kiro takes an
   effort enum, not a budget.

**Model IDs (authoritative, from `kiro-cli.exe`):** `claude-opus-4.7`,
`claude-opus-4.6`, `claude-opus-4.6-1m`, `claude-opus-4.5`, `claude-sonnet-4.6`,
`claude-sonnet-4.5`, `claude-sonnet-4.5-1m`, `claude-sonnet-4`, `claude-haiku-4.5`,
`auto`. `-1m` = 1M context variant. Marketing tier names (Luna/Sol/Terra) are **not**
wire IDs.

**MUST NOT:** fill `editorState`, `shellState`, `gitState`, `envState`,
`userIntent`. Claude Code and Codex do not produce that data; inventing it delivers
the same facts twice in two dialects, which is noise for a model tuned to its own
harness (§5a-quinquies). Leaving them empty is **correct**, not a gap. Do not drop
URL-based images silently (upstream does, `converters_core.py:253,277`) — carry or
report.

**Gate:** `pytest tests/kiro_request`. Assert: system prompt lands in `systemPrompt`
with flag on and in the first user message with it off; consecutive same-role turns
merge with no placeholder text anywhere in output; `is_error: true` produces
`status: "error"`; effort maps to the enum and never to a token budget; `injections`
lists exactly what was done.

**Delete on completion:** `_reference_upstream/converters_core.py`,
`converters_anthropic.py`, `thinking_parser.py`, `tokenizer.py`.

---

### LANE C — Anthropic door (`/v1/messages`) for Claude Code

**Owns:** `conduit/doors/anthropic.py`, `anthropic_sse.py`, `tests/door_anthropic/`.
**Reads:** `contracts/*`, `_reference_upstream/routes_anthropic.py`,
`streaming_anthropic.py`, `models_anthropic.py`.

**Tasks:**
1. `POST /v1/messages` accepting the Anthropic Messages request; auth via the
   existing `PROXY_API_KEY` pattern; `foundation.KiroHttpClient` for transport and
   `AccountManager` for rotation.
2. Consume `contracts.events` from Lane A and emit Anthropic SSE:
   `message_start`, `content_block_start/delta/stop`, `message_delta`,
   `message_stop`. Map `assistantResponseEvent` → `text_delta`.
3. **`reasoningContentEvent` → `thinking_delta` carrying the REAL signature.**
   Upstream scrapes `<thinking>`/`<think>`/`<reasoning>`/`<thought>` tags out of
   plain assistant text (`thinking_parser.py`, an FSM matching only at response
   start) and **fabricates** the signature —
   `streaming_anthropic.py:88-96` is literally `generate_thinking_signature` with
   docstring *"Generate a placeholder signature"*, and `models_anthropic.py:60`
   annotates the field *"placeholder in our case."* Retire both. Native events carry
   no closing `</thinking>` tag to detect on — do not look for one.
4. Populate the Anthropic `usage` block from Lane A's `TurnMetrics` — real
   `input_tokens`, `output_tokens`, and cache read/write. Upstream cannot forward
   cache fields at all because they arrive on the unparsed `metadataEvent`.
5. On turn completion, hand `TurnMetrics` + `RewrittenOut` to Lane F's emit
   function. Call it and move on — never await it.

**Gate:** `pytest tests/door_anthropic` against synthetic Lane A event sequences
(no network). Assert: a well-formed Anthropic SSE sequence; thinking blocks carry
the upstream signature verbatim and nothing generated; usage numbers equal the
fixture's `metadataEvent` exactly; a tool-use turn round-trips.

**Delete on completion:** `_reference_upstream/routes_anthropic.py`,
`streaming_anthropic.py`, `models_anthropic.py`, `streaming_core.py`.

---

### LANE D — Codex door (`/v1/responses`) for Codex CLI → Luna

Handoff steps 10, 11, 12. Fully specified by `FINDINGS-codex` §1.2/§1.5/§1.7/§1.8 —
this is **not** a stub any more.

**Owns:** `conduit/doors/codex.py`, `codex_dialect.py`, `codex_sse.py`,
`tests/door_codex/`.
**Reads:** `contracts/*`, `analysis/FINDINGS-codex.md` Part 1,
`_reference_upstream/routes_openai.py`, `models_openai.py`, `streaming_openai.py`.

**Request contract (all verified in Codex source):**
- `POST /v1/responses`, `Accept: text/event-stream`. Body is `ResponsesApiRequest`.
- `store: false`, `stream: true`, `tool_choice: "auto"` are **hardcoded
  client-side** — rely on always-streaming, never-persisted.
- **System prompt = top-level `instructions` string**, dropped from JSON when empty.
  Maps clean to Kiro `systemPrompt` — the best fidelity win on this door.
- `input[]` is the whole conversation, re-shipped each turn → Kiro
  `currentMessage` + `history[]` via Lane B.
- `tools` is pre-serialized opaque JSON — treat as pass-through.
- `service_tier`, when present, is only `"priority"` or `"flex"`. The sentinel
  `"default"` is filtered out client-side — read absent-or-one-of-two, never the
  literal `"default"`.

**Responses-lite is a REQUIRED second code path, not optional:**
- When active, Codex moves the system prompt and tool list **out of**
  `instructions`/`tools` and **into** `input[]` as prefix items — an
  `AdditionalTools` item with `role: "developer"` plus a `Message` carrying the
  instructions — leaving `instructions: ""` and `tools: null`. Function/freeform
  tools collapse into a single `namespace` tool.
- **A gateway reading the prompt only from `instructions` sees empty for any lite
  model.** This is the single most likely bug on this door.
- **Dispatch on the header `x-openai-internal-codex-responses-lite: true`.** Do not
  sniff the body.
- The flag is **server-driven** from OpenAI's `/models` and can change with no CLI
  release. **Handle both dialects unconditionally** — this is not scope-once-and-forget.

**Response SSE to emit** (OpenAI Responses idiom): `response.created`,
`response.in_progress`, `response.output_item.added/.done`,
`response.output_text.delta/.done`, `response.content_part.added/.done`,
`response.function_call_arguments.delta/.done`, `response.reasoning_text.delta`,
`response.reasoning_summary_text.delta/.done`, `response.completed`,
`response.failed`, `response.incomplete`.
**`response.completed` signals completion and carries the usage.** Failure arrives
as `response.failed`/`.incomplete` with typed codes — `context_length_exceeded`,
`rate_limit_exceeded`, `insufficient_quota`, `invalid_request_error`.

**Token usage — clean six-field map, no estimation on this door:**

| Codex field | Kiro / `TurnMetrics` |
|---|---|
| `input_tokens` | `input_tokens` |
| `input_tokens_details.cached_tokens` | `cache_read_tokens` |
| `input_tokens_details.cache_write_tokens` | `cache_write_tokens` |
| `output_tokens` | `output_tokens` |
| `output_tokens_details.reasoning_tokens` | `thought_tokens` |
| `total_tokens` | `total_tokens` |

**Two hazards to encode now:**
1. **Reasoning blobs are not portable.** Codex `encrypted_content` and Kiro
   `signature` are provider-specific opaque blobs. Forwarding one to the other
   backend is at best ignored, at worst rejected. **Drop or store locally — never
   relay across backends.**
2. **Tool status can't come from Codex's wire.** `FunctionCallOutput` has no
   item-level success/error flag. Supply status to Lane B's builder from CONDUIT's
   tool runner. Do not default to `success`.

**Also record (do not build):** identity arrives in three places at once. The
authoritative session id is **`client_metadata.session_id` in the body, not the
`session-id` header** — the header carries the prompt-cache key for root agents by
design. Note this in your module docstring for WS-C's benefit.

**Interception wiring — there is NO `OPENAI_BASE_URL` env var** (verified including
negative grep). Document both routes in the module docstring: (a)
`openai_base_url = "http://localhost:PORT/v1"` in `config.toml` — smallest change;
(b) a custom `model_providers` entry with `wire_api = "responses"` — preferred, own
provider id/headers/retry. Caveat: under ChatGPT auth the attached token is a
**ChatGPT credential, not an API key** — do not assume Bearer-API-key semantics.

**Gate:** `pytest tests/door_codex`. Assert **both dialects**: standard reads the
prompt from `instructions`; lite (header set) reads it from the `input[]` prefix
items and un-collapses the `namespace` tool. Plus: usage maps six-for-six;
`encrypted_content` never reaches the Kiro payload; `response.completed` closes the
stream.

**Delete on completion:** `_reference_upstream/routes_openai.py`,
`streaming_openai.py`, `models_openai.py`, `converters_openai.py`.

---

### LANE E — Control plane: `GetUsageLimits`

Handoff step 8. Fully independent of every other lane — different host, different
call. Smallest lane; give it to one agent.

**Owns:** `conduit/control/usage_limits.py`, `tests/control/`.
**Reads:** `conduit/foundation/auth.py`, `config.py`, `http_client.py`.

**What's confirmed:** a targeted grep of `kiro-cli.exe` found
`AmazonCodeWhispererService.GetUsageLimits` with `GetUsageLimitsInput` (carries a
`resource` field), `GetUsageLimitsOutput`, and both serializer/deserializer present.
It sits on the **control-plane** host `codewhisperer.{region}.amazonaws.com` (see
`config.py:193` and `get_kiro_profile_url()`), **not** the runtime streaming host.
Returns `UsageBreakdown{currency, current, overage, usage}`, alongside
`MeteringUsageInfo{value, unit, unitPlural, form, url}` and `MonthlyRequestCount`.

**Why it matters:** it flips credit exhaustion from reactive-402 to predictive. Note
there is a **DAILY** limit (`DAILY_REQUEST_COUNT`) as well as the monthly one the
project knew about.

**⚠ Regional hazard — flag it, don't silently ignore it.** `conduit/config.py:179-182`
carries a comment from upstream issue #58 stating that
`codewhisperer.{region}.amazonaws.com` **does not exist for non-`us-east-1`
regions**, which is why the runtime host was moved to `runtime.{region}.kiro.dev`.
Yet the control-plane template is still used for `ListAvailableProfiles` and `/mcp`
(`config.py:187-193`, `get_kiro_profile_url()`, `get_kiro_mcp_url()`). So
`GetUsageLimits` on the control-plane host may only resolve in `us-east-1`. Do not
try to solve this — implement against the control-plane host (that is where Amazon's
own client puts the op), and **record the constraint in your module docstring plus
your final report** so a maintainer can confirm on a non-`us-east-1` profile.

**Tasks:**
1. Client function calling `GetUsageLimits` with the profile's `resource`, via
   `KiroAuthManager` for the token and `get_kiro_headers()` for the AMZ target
   header. **The in-repo precedent is exact:** `config.py:196` defines
   `PROFILE_DISCOVERY_AMZ_TARGET = "AmazonCodeWhispererService.ListAvailableProfiles"`
   and `foundation/auth.py:95 resolve_profile_arn()` is the working control-plane
   call. Add a sibling constant `USAGE_LIMITS_AMZ_TARGET =
   "AmazonCodeWhispererService.GetUsageLimits"` next to it and follow that call's
   shape. Note `config.py` is Lane 0's file — a single constant addition is the one
   sanctioned exception; add it in your own commit and say so in your report.
2. Typed `UsageBreakdown` result. Cache with `ModelInfoCache`-style TTL — this is a
   balance check, not a per-request call.
3. Fill the `router` Lane 0 stubbed: a read-only endpoint exposing remaining credits.
4. Sibling ops found in the same pass, worth a docstring note but **not** this
   lane's build: `GetProfile`, `ListAvailableProfiles`, `ListAvailableModels`
   (the model catalog is queryable per-account rather than hardcoded),
   `CreateSubscriptionToken`, `SendTelemetryEvent`.

**Gate:** `pytest tests/control` against a mocked httpx transport — request targets
the control-plane host and not the runtime host, the AMZ target header is exactly
`AmazonCodeWhispererService.GetUsageLimits`, a fixture response parses into
`UsageBreakdown`, and TTL caching suppresses a second call. **No live call.**

---

### LANE F — SAGA seam emit

Handoff step 9. Build strictly to `CONTRACT-conduit-saga-seam.md` v1 — restated
below so you don't drift. **The contract is frozen; you may not change the schema.**

**Owns:** `conduit/seam/emit.py`, `tests/seam/`.
**Reads:** `conduit/contracts/*`, `CONTRACT-conduit-saga-seam.md`.

**Mechanism:** fire-and-forget HTTP POST to
`http://127.0.0.1:<saga-ingest-port>/ingest/conduit`, loopback only, one JSON object
per completed request.

**The invariant that outranks everything else in this lane: the emit MUST NOT block
or fail the client response.** Send, don't await. If SAGA is down, drop and log; the
client is unaffected. Same rule SAGA's own proxy holds — capture failures never
propagate. A test must prove this.

**Correlation:** read SAGA's `x-saga-request-id` off the inbound request and echo it
as `request_id`. That is how SAGA joins metrics + rewritten-out to the clean-in it
already stored. If the header is absent, generate an id and log at debug — never
drop the emit for a missing header. (WS-C adds the header on their side if it isn't
being forwarded yet; noted so neither side assumes it exists.)

**Self-declared injections:** CONDUIT declares what it did —
`thinking_tags` / `system_prompt_slot` / `turn_merge` / `relocated_tool_docs` —
rather than making SAGA reverse-engineer the clean-in ⊖ rewritten-out diff. SAGA may
still diff to verify. Lane B produces these; you serialize them.

**Two-feed rule:** this seam carries **Kiro-routed traffic only** (Claude via Claude
Code, Luna via Codex). **Gemini never passes through CONDUIT** — Gemini CLI talks to
Google directly and SAGA intercepts it natively. Do not build anything Gemini-shaped.

**Tasks:**
1. `emit(envelope: SeamEnvelope) -> None` — serializes `SeamEnvelope` to the v1
   schema and posts without awaiting. Config: ingest URL/port, plus an enable flag
   defaulting off until a maintainer wires SAGA.
2. Bounded queue with drop-oldest, so a wedged SAGA cannot grow memory without limit.
3. Field names on the wire are the contract's **snake_case JSON**, not Python
   attribute names — verify against §"Payload schema (v1)" of the contract file
   field by field.

**Gate:** `pytest tests/seam`. Assert: emitted JSON matches the v1 schema key for
key; `x-saga-request-id` is echoed; **a SAGA that refuses connections, hangs, or
500s does not raise, delay, or alter the client path** (the load-bearing test); a
full queue drops rather than blocks.

---

## 6. Do NOT (all lanes)

- **Do not build Forge anything.** CONDUIT is Forge-agnostic. A correlation header,
  if one arrives later, rides through untouched.
- **Do not design `project_id`.** Pass through whatever the harness sends.
- **Do not run against real credentials** until a maintainer says so.
- **Do not import from `_reference_upstream/`** in shipped code.
- **Do not fill Kiro's typed context slots** (`editorState`/`gitState`/`shellState`/
  `envState`/`userIntent`) — see Lane B.
- **Do not add a git remote.**

## 7. What "done" means

Report per lane: steps done and verified, steps stubbed, `_reference_upstream/`
files deleted, and — explicitly — **any place the real protocol contradicted the
evidence base.** That last one matters: several claims in `PLATFORM-FINDINGS` are
tagged *inferred* from string adjacency in a binary (notably the `systemPrompt`
placement, §2 of this spec). If code proves one wrong, that is a finding worth more
than the code.

Confirm the seam emit matches the contract schema field by field.
