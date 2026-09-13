"""Cross-lane integration tests — the door routes end to end, seams included.

WHY THIS FILE EXISTS
--------------------
288 unit tests passed while FOUR cross-lane seams were broken: Lane C imported
a Lane B function that did not exist under that name, imported an `iter_events`
that had not been written, imported the seam *module* instead of the emit
*function*, and Lane D never contacted Kiro at all — it served a hardcoded
"Hello from CONDUIT." Every one of those was masked, either by a broad
`except (ImportError, Exception)` that fell back to a stub, or by a stub that
was simply never replaced.

Unit tests could not catch any of it, because each lane's tests exercised that
lane in isolation with its own fixtures. Import checks could not catch it
either: a name referenced inside a function body that is never executed imports
cleanly and still raises NameError at runtime (that is exactly how a missing
FrameDecoder import survived an import check here).

So these tests drive the real ASGI routes over synthetic Kiro wire BYTES and
assert on what the harness would actually receive. They are the regression gate
for lane wiring, and they are deliberately written to fail loudly if a door
starts answering from a stub again.

NO LIVE CREDENTIALS. Every byte is synthetic, built by the frame encoder in
tests/kiro_parse/conftest.py. The transport is patched out, and the fake auth
manager points at a `.invalid` host that cannot resolve even if a real request
somehow escaped.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from conduit.app import create_app
from conduit.contracts.seam import SeamEnvelope
from tests.kiro_parse.conftest import build_frame, json_payload


# ---------------------------------------------------------------------------
# Synthetic transport doubles
# ---------------------------------------------------------------------------


class _FakeStreamResponse:
    """Stands in for the streaming httpx.Response Kiro would return."""

    def __init__(self, chunks: List[bytes], status_code: int = 200) -> None:
        self._chunks = chunks
        self.status_code = status_code

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def aread(self) -> bytes:
        return b"".join(self._chunks)


class _CapturingHttpClient:
    """Captures the payload Lane B built instead of sending it anywhere.

    Records the outbound request so a test can assert on the real Kiro payload,
    which is the only way to prove the door actually went through Lane B rather
    than fabricating a response.
    """

    last_payload: Optional[Dict[str, Any]] = None
    last_url: Optional[str] = None
    response_chunks: List[bytes] = []
    status_code: int = 200
    closed: bool = False

    def __init__(self, auth_manager: Any, shared_client: Any = None) -> None:
        self._auth_manager = auth_manager

    async def request_with_retry(
        self, method: str, url: str, json_data: Any = None, **kwargs: Any
    ) -> _FakeStreamResponse:
        type(self).last_url = url
        type(self).last_payload = json_data
        return _FakeStreamResponse(
            type(self).response_chunks, status_code=type(self).status_code
        )

    async def close(self) -> None:
        type(self).closed = True


class _FakeAuthManager:
    # .invalid is reserved and cannot resolve — a real request would fail fast
    # rather than reach any host, even if the transport patch were bypassed.
    api_host = "https://kiro.invalid"


class _FakeAccount:
    id = "test-account"
    auth_manager = _FakeAuthManager()


class _FakeAccountManager:
    async def get_next_account(self, model: str) -> _FakeAccount:
        return _FakeAccount()

    def get_first_account(self) -> _FakeAccount:
        return _FakeAccount()

    async def report_success(self, account_id: str, model: str) -> None:
        return None


# ---------------------------------------------------------------------------
# Wire fixtures
# ---------------------------------------------------------------------------


def _turn_frames() -> List[bytes]:
    """A complete assistant turn as real AWS event-stream frames.

    Uses Kiro's true field spellings — `thoughtTokens`, `totalTokens`, and the
    cache counts inside metadataEvent.tokenUsage. Wrong names here would let a
    door that hand-rolls its own mapping pass, which is the defect this file
    is meant to catch.
    """
    return [
        build_frame("assistantResponseEvent", json_payload({"content": "Real "})),
        build_frame("assistantResponseEvent", json_payload({"content": "answer."})),
        build_frame("meteringEvent", json_payload({"usage": 2.5})),
        build_frame("contextUsageEvent", json_payload({"contextUsagePercentage": 33.5})),
        build_frame("metadataEvent", json_payload({
            "tokenUsage": {
                "inputTokens": 111,
                "outputTokens": 22,
                "thoughtTokens": 33,
                "cacheReadInputTokens": 44,
                "cacheWriteInputTokens": 55,
                "totalTokens": 166,
            }
        })),
        build_frame("messageStopEvent", json_payload({"stopReason": "END_TURN"})),
    ]


def _anthropic_text(body: str) -> str:
    """Reassemble assistant text from Anthropic SSE `text_delta` events.

    Each assistantResponseEvent becomes its OWN content_block_delta, so the raw
    body never contains the full sentence contiguously — a substring check
    against the body would fail on correct output. This joins the deltas the way
    a real harness does.
    """
    out: List[str] = []
    for event in _sse_events(body):
        if event["event"] != "content_block_delta":
            continue
        delta = event["data"].get("delta") or {}
        if delta.get("type") == "text_delta":
            out.append(delta.get("text", ""))
    return "".join(out)


def _sse_events(body: str) -> List[Dict[str, Any]]:
    """Parse an SSE body into [{event, data}] dicts."""
    out: List[Dict[str, Any]] = []
    event_name: Optional[str] = None
    for line in body.splitlines():
        if line.startswith("event: "):
            event_name = line[len("event: "):].strip()
        elif line.startswith("data: "):
            raw = line[len("data: "):].strip()
            if raw and raw != "[DONE]":
                try:
                    out.append({"event": event_name, "data": json.loads(raw)})
                except json.JSONDecodeError:
                    pass
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def seam_envelopes(monkeypatch: pytest.MonkeyPatch) -> List[SeamEnvelope]:
    """Capture Lane F seam emits from both doors.

    Patches each door's `_fire_seam_emit` rather than `emit` itself: the real
    emit() is gated on SAGA_EMIT_ENABLED (false by default), so asserting on
    emit() would pass vacuously whether or not the door ever called it.
    """
    captured: List[SeamEnvelope] = []

    def _capture(envelope: SeamEnvelope) -> None:
        captured.append(envelope)

    import conduit.doors.anthropic as anthropic_door
    import conduit.doors.codex as codex_door

    monkeypatch.setattr(anthropic_door, "_fire_seam_emit", _capture)
    monkeypatch.setattr(codex_door, "_fire_seam_emit", _capture)
    return captured


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A TestClient whose doors reach a captured transport, never the network."""
    import conduit.doors.anthropic as anthropic_door
    import conduit.doors.codex as codex_door

    async def _fake_resolve_profile_arn(auth_manager: Any, configured: Any) -> str:
        return "arn:aws:codewhisperer:us-east-1:000000000000:profile/TEST"

    for module in (anthropic_door, codex_door):
        monkeypatch.setattr(module, "KiroHttpClient", _CapturingHttpClient)
        monkeypatch.setattr(module, "resolve_profile_arn", _fake_resolve_profile_arn)

    _CapturingHttpClient.last_payload = None
    _CapturingHttpClient.last_url = None
    _CapturingHttpClient.status_code = 200
    _CapturingHttpClient.closed = False
    _CapturingHttpClient.response_chunks = _turn_frames()

    app = create_app()
    with TestClient(app) as test_client:
        # Override after startup so the real lifespan does not clobber the fakes.
        app.state.account_manager = _FakeAccountManager()
        app.state.account_system = False
        yield test_client


_AUTH = {"Authorization": "Bearer my-super-secret-password-123"}


# ===========================================================================
# Lane D — the codex door reaches Kiro (the seam that served a stub)
# ===========================================================================


class TestCodexDoorSeam:
    def test_response_is_not_the_stub(self, client: TestClient) -> None:
        """The door must answer from the wire, never from a hardcoded string."""
        resp = client.post(
            "/v1/responses",
            headers=_AUTH,
            json={
                "model": "claude-sonnet-4.5",
                "instructions": "You are a helpful assistant.",
                "input": [
                    {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "hello"}]}
                ],
            },
        )
        assert resp.status_code == 200
        body = resp.text
        assert "Hello from CONDUIT" not in body, (
            "door answered from the synthetic stub — the Lane A/B transport is "
            "not wired"
        )
        assert "Real answer." in body

    def test_lane_b_built_the_payload(self, client: TestClient) -> None:
        """The outbound Kiro payload must come from Lane B, with the real prompt."""
        client.post(
            "/v1/responses",
            headers=_AUTH,
            json={
                "model": "claude-sonnet-4.5",
                "instructions": "SYSTEM-PROMPT-MARKER",
                "input": [
                    {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "hi"}]}
                ],
            },
        )
        payload = _CapturingHttpClient.last_payload
        assert payload is not None, "door never issued an upstream request"
        assert "conversationState" in payload
        # Kiro's native systemPrompt slot — not smuggled into the user turn.
        serialised = json.dumps(payload)
        assert "SYSTEM-PROMPT-MARKER" in serialised
        assert _CapturingHttpClient.last_url.endswith("/generateAssistantResponse")

    def test_metrics_are_real_and_seam_fires(
        self, client: TestClient, seam_envelopes: List[SeamEnvelope]
    ) -> None:
        """Lane A's counts reach the Lane F envelope, thoughtTokens included."""
        resp = client.post(
            "/v1/responses",
            headers=_AUTH,
            json={
                "model": "claude-sonnet-4.5",
                "instructions": "sys",
                "input": [
                    {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "hi"}]}
                ],
            },
        )
        assert resp.status_code == 200

        assert len(seam_envelopes) == 1, "Lane F seam did not fire exactly once"
        metrics = seam_envelopes[0].metrics
        assert metrics.input_tokens == 111
        assert metrics.output_tokens == 22
        # thoughtTokens was previously read as `thinkingTokens` — always 0.
        assert metrics.thought_tokens == 33
        assert metrics.cache_read_tokens == 44
        assert metrics.cache_write_tokens == 55
        # totalTokens is READ, not summed: 111+22 = 133 would omit thought tokens.
        assert metrics.total_tokens == 166
        assert metrics.credits == pytest.approx(2.5)
        assert metrics.context_usage_percentage == pytest.approx(33.5)

    def test_usage_reaches_the_harness(self, client: TestClient) -> None:
        """The Responses `usage` block carries the real counts."""
        resp = client.post(
            "/v1/responses",
            headers=_AUTH,
            json={
                "model": "claude-sonnet-4.5",
                "instructions": "sys",
                "input": [
                    {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "hi"}]}
                ],
            },
        )
        completed = [e for e in _sse_events(resp.text)
                     if e["event"] == "response.completed"]
        assert completed, "no response.completed frame"
        usage = completed[-1]["data"]["response"]["usage"]
        assert usage["input_tokens"] == 111
        assert usage["output_tokens"] == 22
        assert usage["output_tokens_details"]["reasoning_tokens"] == 33
        assert usage["total_tokens"] == 166

    def test_lite_dialect_prompt_is_not_lost(self, client: TestClient) -> None:
        """Under responses-lite the prompt lives in input[], not instructions."""
        client.post(
            "/v1/responses",
            headers={**_AUTH, "x-openai-internal-codex-responses-lite": "true"},
            json={
                "model": "claude-sonnet-4.5",
                "instructions": "",
                "tools": None,
                "input": [
                    {"type": "message", "role": "developer",
                     "content": [{"type": "input_text", "text": "LITE-PROMPT-MARKER"}]},
                    {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "hi"}]},
                ],
            },
        )
        serialised = json.dumps(_CapturingHttpClient.last_payload)
        assert "LITE-PROMPT-MARKER" in serialised, (
            "lite-dialect prompt was dropped — a gateway reading only "
            "`instructions` sees an empty prompt for every lite model"
        )
        # The developer prefix item must not ALSO land in the conversation.
        assert serialised.count("LITE-PROMPT-MARKER") == 1

    def test_tool_call_history_round_trips(self, client: TestClient) -> None:
        """function_call / function_call_output become Kiro toolUses/toolResults."""
        client.post(
            "/v1/responses",
            headers=_AUTH,
            json={
                "model": "claude-sonnet-4.5",
                "instructions": "sys",
                "input": [
                    {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "list files"}]},
                    {"type": "function_call", "call_id": "call_abc",
                     "name": "list_dir", "arguments": '{"path": "/tmp"}'},
                    {"type": "function_call_output", "call_id": "call_abc",
                     "output": "a.txt\nb.txt"},
                ],
            },
        )
        serialised = json.dumps(_CapturingHttpClient.last_payload)
        assert "call_abc" in serialised
        assert "list_dir" in serialised
        # arguments arrive as a JSON *string* and must be parsed to an object
        assert '"path": "/tmp"' in serialised or '"path":"/tmp"' in serialised
        assert "a.txt" in serialised

    def test_reasoning_blob_is_never_relayed(self, client: TestClient) -> None:
        """Codex encrypted_content is provider-specific — it must be dropped."""
        client.post(
            "/v1/responses",
            headers=_AUTH,
            json={
                "model": "claude-sonnet-4.5",
                "instructions": "sys",
                "input": [
                    {"type": "reasoning", "id": "rs_1",
                     "encrypted_content": "ENCRYPTED-BLOB-MARKER",
                     "summary": []},
                    {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "hi"}]},
                ],
            },
        )
        serialised = json.dumps(_CapturingHttpClient.last_payload)
        assert "ENCRYPTED-BLOB-MARKER" not in serialised

    def test_upstream_error_is_not_masked(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Kiro error must surface, never degrade into a fake success."""
        _CapturingHttpClient.status_code = 429
        _CapturingHttpClient.response_chunks = [b'{"message":"throttled"}']
        resp = client.post(
            "/v1/responses",
            headers=_AUTH,
            json={
                "model": "claude-sonnet-4.5",
                "instructions": "sys",
                "input": [
                    {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "hi"}]}
                ],
            },
        )
        assert resp.status_code == 429
        assert "Hello from CONDUIT" not in resp.text

    def test_auth_is_enforced(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/responses",
            json={"model": "claude-sonnet-4.5", "input": []},
        )
        assert resp.status_code == 401


# ===========================================================================
# Lane C — the anthropic door, same seams
# ===========================================================================


class TestAnthropicDoorSeam:
    def test_streaming_turn_reaches_the_harness(
        self, client: TestClient, seam_envelopes: List[SeamEnvelope]
    ) -> None:
        resp = client.post(
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "claude-sonnet-4.5",
                "max_tokens": 1024,
                "system": "SYSTEM-PROMPT-MARKER",
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 200
        assert _anthropic_text(resp.text) == "Real answer."

        payload = _CapturingHttpClient.last_payload
        assert payload is not None, "door never issued an upstream request"
        assert "SYSTEM-PROMPT-MARKER" in json.dumps(payload)

        assert len(seam_envelopes) == 1, "Lane F seam did not fire exactly once"
        metrics = seam_envelopes[0].metrics
        assert metrics.input_tokens == 111
        assert metrics.thought_tokens == 33
        assert metrics.total_tokens == 166
        assert metrics.credits == pytest.approx(2.5)

    def test_rewritten_out_is_real_not_empty(
        self, client: TestClient, seam_envelopes: List[SeamEnvelope]
    ) -> None:
        """The seam must carry Lane B's real RewrittenOut, not a blank stub."""
        client.post(
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "claude-sonnet-4.5",
                "max_tokens": 1024,
                "system": "sys",
                "stream": True,
                "messages": [
                    {"role": "user", "content": "one"},
                    {"role": "user", "content": "two"},
                ],
            },
        )
        assert seam_envelopes
        rewritten = seam_envelopes[0].rewritten_out
        # Two consecutive user turns must be merged, and the merge declared.
        assert rewritten.injections, (
            "RewrittenOut carried no injections — the door is emitting a stub "
            "instead of Lane B's real output"
        )
        assert any(i.type == "turn_merge" for i in rewritten.injections)

    def test_non_streaming_turn(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "claude-sonnet-4.5",
                "max_tokens": 1024,
                "stream": False,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"][0]["text"] == "Real answer."
        assert data["usage"]["input_tokens"] == 111
        assert data["usage"]["cache_creation_input_tokens"] == 55

    def test_tool_result_error_status_is_truthful(self, client: TestClient) -> None:
        """is_error=True must become Kiro status='error', never 'success'."""
        client.post(
            "/v1/messages",
            headers=_AUTH,
            json={
                "model": "claude-sonnet-4.5",
                "max_tokens": 1024,
                "stream": True,
                "messages": [
                    {"role": "user", "content": [{
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "boom",
                        "is_error": True,
                    }]},
                ],
            },
        )
        serialised = json.dumps(_CapturingHttpClient.last_payload)
        assert '"status": "error"' in serialised or '"status":"error"' in serialised


# ===========================================================================
# Frame reassembly through the real route
# ===========================================================================


class TestFrameReassembly:
    def test_frames_split_across_chunks(self, client: TestClient) -> None:
        """A frame split mid-payload must still decode through the live route.

        Kiro's frames do not align with HTTP chunk boundaries. The door used to
        have no streaming decoder at all, so this is the regression gate for it.
        """
        blob = b"".join(_turn_frames())
        # One byte at a time — the most hostile split possible.
        _CapturingHttpClient.response_chunks = [blob[i:i + 1] for i in range(len(blob))]

        resp = client.post(
            "/v1/responses",
            headers=_AUTH,
            json={
                "model": "claude-sonnet-4.5",
                "instructions": "sys",
                "input": [
                    {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "hi"}]}
                ],
            },
        )
        assert resp.status_code == 200
        assert "Real answer." in resp.text
