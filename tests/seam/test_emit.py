"""Tests for conduit/seam/emit.py — Lane F verification gate.

Five assertions required by the spec:
1. Emitted JSON matches the v1 schema key-for-key (field names exact).
2. x-saga-request-id is echoed as request_id.
3. THE LOAD-BEARING TEST — connection refused, 10-second hang, and 500 each
   do NOT raise, delay, or alter the client-side return value of emit().
4. A full queue evicts the oldest entry rather than blocking.
5. SAGA_EMIT_ENABLED=False → no HTTP call is made.
"""
from __future__ import annotations

import dataclasses
import queue
import threading
import time
from unittest.mock import MagicMock, patch, call

import httpx
import pytest

import conduit.config as cfg
from conduit.contracts.metrics import ReasoningBlockMeta, StopReason, TurnMetrics
from conduit.contracts.rewritten import Injection, InjectionType, RewrittenOut
from conduit.contracts.seam import SeamEnvelope
import conduit.seam.emit as emit_mod
from conduit.seam.emit import _reset_for_testing, emit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_envelope(request_id: str = "req_abc123") -> SeamEnvelope:
    """Return a fully-populated SeamEnvelope for use in tests."""
    return SeamEnvelope(
        request_id=request_id,
        ts=1_700_000_000_000,
        model_id="claude-opus-4.7",
        metrics=TurnMetrics(
            input_tokens=100,
            output_tokens=200,
            thought_tokens=50,
            cache_read_tokens=10,
            cache_write_tokens=5,
            total_tokens=365,
            credits=1.5,
            context_usage_percentage=42.0,
            stop_reason=StopReason.END_TURN,
            reasoning_blocks=[
                ReasoningBlockMeta(model_id="claude-opus-4.7", signature_present=True)
            ],
        ),
        rewritten_out=RewrittenOut(
            system_prompt="You are a helpful assistant.",
            current_message={"role": "user", "content": "Hello"},
            history=[{"role": "assistant", "content": "Hi there"}],
            tools=[{"name": "read_file", "description": "reads a file"}],
            injections=[
                Injection(
                    type=InjectionType.THINKING_TAGS,
                    location="last_user_message",
                )
            ],
        ),
    )


def _wait_for_event(event: threading.Event, timeout: float = 2.0) -> bool:
    """Wait for event; return True if fired within timeout, False otherwise."""
    return event.wait(timeout=timeout)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    """Reset module state, enable emit, and use a small default queue each test."""
    monkeypatch.setattr(cfg, "SAGA_EMIT_ENABLED", True)
    monkeypatch.setattr(cfg, "SAGA_EMIT_QUEUE_SIZE", 256)
    monkeypatch.setattr(cfg, "SAGA_INGEST_URL", "http://127.0.0.1:9400/ingest/conduit")
    _reset_for_testing()
    yield
    _reset_for_testing()


# ---------------------------------------------------------------------------
# Test 1 — v1 schema field names
# ---------------------------------------------------------------------------

class TestSchemaFieldNames:
    """Assert emitted JSON has exactly the field names from CONTRACT v1."""

    def _capture_payload(self) -> dict:
        """Emit one envelope and return the payload dict seen by _post_once."""
        received: list[dict] = []
        fired = threading.Event()

        original = emit_mod._post_once

        def capturing_post(client, payload: dict) -> None:
            received.append(payload)
            fired.set()

        emit_mod._post_once = capturing_post
        try:
            emit(make_envelope())
            assert _wait_for_event(fired, timeout=2.0), (
                "Background worker did not process the emit within 2 s"
            )
        finally:
            emit_mod._post_once = original

        assert received, "No payload was captured"
        return received[0]

    def test_top_level_keys(self):
        """Top-level keys must match contract v1 exactly."""
        payload = self._capture_payload()
        expected = {"request_id", "ts", "model_id", "metrics", "rewritten_out"}
        assert set(payload.keys()) == expected, (
            f"Top-level keys mismatch.\n  got:      {set(payload.keys())}\n"
            f"  expected: {expected}"
        )

    def test_metrics_keys(self):
        """metrics sub-object keys must match contract v1 exactly."""
        payload = self._capture_payload()
        expected = {
            "input_tokens",
            "output_tokens",
            "thought_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "total_tokens",
            "credits",
            "context_usage_percentage",
            "stop_reason",
            "reasoning_blocks",
        }
        got = set(payload["metrics"].keys())
        assert got == expected, (
            f"metrics keys mismatch.\n  got:      {got}\n  expected: {expected}"
        )

    def test_rewritten_out_keys(self):
        """rewritten_out sub-object keys must match contract v1 exactly."""
        payload = self._capture_payload()
        expected = {
            "system_prompt",
            "current_message",
            "history",
            "tools",
            "injections",
        }
        got = set(payload["rewritten_out"].keys())
        assert got == expected, (
            f"rewritten_out keys mismatch.\n  got:      {got}\n  expected: {expected}"
        )

    def test_reasoning_blocks_keys(self):
        """Each reasoning_block entry must have model_id and signature_present."""
        payload = self._capture_payload()
        blocks = payload["metrics"]["reasoning_blocks"]
        assert blocks, "Expected at least one reasoning_blocks entry"
        expected = {"model_id", "signature_present"}
        for blk in blocks:
            got = set(blk.keys())
            assert got == expected, (
                f"reasoning_block keys mismatch.\n  got: {got}\n  expected: {expected}"
            )

    def test_injection_keys(self):
        """Each injections entry must have type and location."""
        payload = self._capture_payload()
        injections = payload["rewritten_out"]["injections"]
        assert injections, "Expected at least one injection entry"
        expected = {"type", "location"}
        for inj in injections:
            got = set(inj.keys())
            assert got == expected, (
                f"injection keys mismatch.\n  got: {got}\n  expected: {expected}"
            )

    def test_ts_is_int_milliseconds(self):
        """ts must be an integer, not a float or string."""
        payload = self._capture_payload()
        assert isinstance(payload["ts"], int), (
            f"ts must be int, got {type(payload['ts'])}"
        )
        # Sanity: value is large enough to be ms since epoch (> year 2000 in ms)
        assert payload["ts"] > 946_684_800_000, "ts looks too small to be ms since epoch"

    def test_stop_reason_is_string_value(self):
        """stop_reason must be serialised as its string value, not enum repr."""
        payload = self._capture_payload()
        assert payload["metrics"]["stop_reason"] == "END_TURN", (
            f"stop_reason should be 'END_TURN', got {payload['metrics']['stop_reason']!r}"
        )

    def test_injection_type_is_string_value(self):
        """injection.type must be serialised as its string value."""
        payload = self._capture_payload()
        inj_type = payload["rewritten_out"]["injections"][0]["type"]
        assert inj_type == "thinking_tags", (
            f"injection type should be 'thinking_tags', got {inj_type!r}"
        )


# ---------------------------------------------------------------------------
# Test 2 — request_id correlation
# ---------------------------------------------------------------------------

class TestRequestIdCorrelation:
    """request_id in the emitted payload must equal the envelope's request_id."""

    def _capture_for_id(self, request_id: str) -> dict:
        received: list[dict] = []
        fired = threading.Event()

        def capturing_post(client, payload: dict) -> None:
            received.append(payload)
            fired.set()

        emit_mod._post_once = capturing_post
        try:
            emit(make_envelope(request_id=request_id))
            assert _wait_for_event(fired, timeout=2.0)
        finally:
            emit_mod._post_once = emit_mod._post_once  # noop restore; test tears down

        # Proper restore
        _reset_for_testing()
        return received[0]

    def test_request_id_is_echoed(self):
        """The payload request_id must match the id set on the envelope."""
        expected_id = "req_saga_abc_0042"
        received: list[dict] = []
        fired = threading.Event()

        def capturing_post(client, payload: dict) -> None:
            received.append(payload)
            fired.set()

        emit_mod._post_once = capturing_post
        try:
            emit(make_envelope(request_id=expected_id))
            assert _wait_for_event(fired, timeout=2.0), (
                "Worker did not process emit within 2 s"
            )
        finally:
            # Restore: reset replaces the worker; the capturing function
            # will never run again.
            _reset_for_testing()

        assert received[0]["request_id"] == expected_id, (
            f"request_id not echoed: expected {expected_id!r}, "
            f"got {received[0]['request_id']!r}"
        )

    def test_missing_request_id_generates_one_and_does_not_drop(self):
        """An envelope with an empty request_id must still be emitted (never dropped)."""
        received: list[dict] = []
        fired = threading.Event()

        def capturing_post(client, payload: dict) -> None:
            received.append(payload)
            fired.set()

        emit_mod._post_once = capturing_post
        try:
            emit(make_envelope(request_id=""))
            assert _wait_for_event(fired, timeout=2.0), (
                "emit was dropped for a missing request_id — must never drop"
            )
        finally:
            _reset_for_testing()

        rid = received[0]["request_id"]
        assert rid, "Generated request_id must not be empty"
        assert rid.startswith("gen-"), (
            f"Generated id should start with 'gen-', got {rid!r}"
        )


# ---------------------------------------------------------------------------
# Test 3 — THE LOAD-BEARING TEST: emit() must never block or raise
# ---------------------------------------------------------------------------

class TestNonBlocking:
    """
    THE LOAD-BEARING TEST.

    A SAGA that refuses connections, hangs for 10 seconds, or returns 500
    must NOT raise, delay, or alter the return value of emit().

    Threshold: emit() must return in < 0.5 s regardless of SAGA behaviour.
    """

    # -- helpers --

    @staticmethod
    def _time_emit(envelope: SeamEnvelope) -> float:
        """Return how long emit() took in seconds."""
        t0 = time.monotonic()
        emit(envelope)
        return time.monotonic() - t0

    # -- connection refused --

    def test_connection_refused_does_not_raise(self, monkeypatch):
        """emit() must not raise when SAGA refuses the connection."""
        def refusing_post(client, payload: dict) -> None:
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(emit_mod, "_post_once", refusing_post)
        # Must not raise:
        emit(make_envelope())

    def test_connection_refused_does_not_block(self, monkeypatch):
        """emit() must return in < 0.5 s when SAGA refuses connection."""
        worker_started = threading.Event()

        def refusing_post(client, payload: dict) -> None:
            worker_started.set()
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(emit_mod, "_post_once", refusing_post)

        elapsed = self._time_emit(make_envelope())
        assert elapsed < 0.5, (
            f"emit() took {elapsed:.3f}s with connection refused — must be < 0.5s"
        )

    def test_connection_refused_return_value_is_none(self, monkeypatch):
        """emit() return value must be None (not affected by SAGA failure)."""
        def refusing_post(client, payload: dict) -> None:
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(emit_mod, "_post_once", refusing_post)
        result = emit(make_envelope())
        assert result is None

    # -- 10-second hang --

    def test_saga_hang_does_not_block_caller(self, monkeypatch):
        """emit() must return in < 0.5 s even if the worker hangs for 10 s.

        The hang is simulated by patching _post_once to sleep; the caller
        must already be back before the sleep begins.
        """
        hang_entered = threading.Event()

        def hanging_post(client, payload: dict) -> None:
            hang_entered.set()
            time.sleep(10)  # simulate SAGA that never responds

        monkeypatch.setattr(emit_mod, "_post_once", hanging_post)

        elapsed = self._time_emit(make_envelope())

        assert elapsed < 0.5, (
            f"emit() took {elapsed:.3f}s while SAGA hangs for 10 s — "
            "emit() must hand off to background thread and return immediately"
        )

    def test_saga_hang_does_not_raise(self, monkeypatch):
        """emit() must not raise when SAGA hangs."""
        def hanging_post(client, payload: dict) -> None:
            time.sleep(10)

        monkeypatch.setattr(emit_mod, "_post_once", hanging_post)
        emit(make_envelope())  # must not raise

    # -- HTTP 500 --

    def test_http_500_does_not_raise(self):
        """emit() must not raise when SAGA returns HTTP 500."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch("conduit.seam.emit.httpx.Client", return_value=mock_client):
            _reset_for_testing()
            emit(make_envelope())  # must not raise
            # Give the worker thread a moment to process
            time.sleep(0.3)

    def test_http_500_does_not_block(self):
        """emit() must return in < 0.5 s when SAGA returns HTTP 500."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch("conduit.seam.emit.httpx.Client", return_value=mock_client):
            _reset_for_testing()
            elapsed = self._time_emit(make_envelope())

        assert elapsed < 0.5, (
            f"emit() took {elapsed:.3f}s with SAGA returning 500 — must be < 0.5s"
        )

    def test_http_500_return_value_is_none(self):
        """emit() return value must be None when SAGA returns HTTP 500."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch("conduit.seam.emit.httpx.Client", return_value=mock_client):
            _reset_for_testing()
            result = emit(make_envelope())

        assert result is None

    # -- timeout --

    def test_timeout_exception_does_not_raise(self, monkeypatch):
        """emit() must not raise when httpx raises a timeout."""
        def timeout_post(client, payload: dict) -> None:
            raise httpx.TimeoutException("timed out")

        monkeypatch.setattr(emit_mod, "_post_once", timeout_post)
        emit(make_envelope())  # must not raise

    def test_timeout_does_not_block(self, monkeypatch):
        """emit() must return in < 0.5 s when httpx raises a timeout."""
        def timeout_post(client, payload: dict) -> None:
            raise httpx.TimeoutException("timed out")

        monkeypatch.setattr(emit_mod, "_post_once", timeout_post)
        elapsed = self._time_emit(make_envelope())
        assert elapsed < 0.5, (
            f"emit() took {elapsed:.3f}s with timeout — must be < 0.5s"
        )


# ---------------------------------------------------------------------------
# Test 4 — bounded queue with drop-oldest
# ---------------------------------------------------------------------------

class TestBoundedQueue:
    """A full queue must evict the oldest entry rather than blocking."""

    def test_full_queue_evicts_oldest_not_newest(self, monkeypatch):
        """When queue is full, the OLDEST item is dropped to make room for the newest."""
        # Use queue size of 3 so it is easy to fill.
        monkeypatch.setattr(cfg, "SAGA_EMIT_QUEUE_SIZE", 3)
        _reset_for_testing()

        # Block the worker from consuming items while we fill the queue.
        block_worker = threading.Event()
        first_item_consumed = threading.Event()

        def blocking_post(client, payload: dict) -> None:
            first_item_consumed.set()
            block_worker.wait(timeout=5.0)

        monkeypatch.setattr(emit_mod, "_post_once", blocking_post)

        # Emit 3 items to fill the queue, then emit a 4th (should evict oldest).
        ids = ["id-0", "id-1", "id-2", "id-NEW"]

        for rid in ids[:3]:
            emit(make_envelope(request_id=rid))

        # Wait until the worker has picked up the first item (queue now holds 2).
        # Then emit a new item while the worker is blocked — queue will fill again.
        assert _wait_for_event(first_item_consumed, timeout=2.0), (
            "Worker never started — check _ensure_worker()"
        )

        # Re-fill: emit two more so the 3-slot queue is full again.
        emit(make_envelope(request_id="id-fill-a"))
        emit(make_envelope(request_id="id-fill-b"))

        # Now queue is [id-1, id-fill-a, id-fill-b] (id-0 was consumed by worker).
        # Emitting one more must evict the oldest remaining (id-1).
        emit(make_envelope(request_id="id-NEW"))

        # Collect what is now in the queue (without consuming via worker).
        q = emit_mod._get_queue()
        items = []
        while True:
            try:
                items.append(q.get_nowait())
            except queue.Empty:
                break

        request_ids = [item["request_id"] for item in items]

        # The newest item must be present.
        assert "id-NEW" in request_ids, (
            f"Newest item 'id-NEW' was dropped instead of oldest: {request_ids}"
        )
        # Queue must not exceed the configured max size.
        assert len(items) <= 3, (
            f"Queue grew beyond maxsize=3: contains {len(items)} items"
        )

        block_worker.set()  # unblock worker so it can clean up

    def test_emit_does_not_block_when_queue_is_full(self, monkeypatch):
        """emit() must return quickly even when the queue is completely full."""
        monkeypatch.setattr(cfg, "SAGA_EMIT_QUEUE_SIZE", 2)
        _reset_for_testing()

        # Block the worker so nothing is consumed.
        block_worker = threading.Event()

        def blocking_post(client, payload: dict) -> None:
            block_worker.wait(timeout=5.0)

        monkeypatch.setattr(emit_mod, "_post_once", blocking_post)

        # Fill beyond capacity — all calls must be fast.
        for i in range(10):
            t0 = time.monotonic()
            emit(make_envelope(request_id=f"id-{i}"))
            elapsed = time.monotonic() - t0
            assert elapsed < 0.5, (
                f"emit() call {i} blocked for {elapsed:.3f}s on a full queue"
            )

        block_worker.set()


# ---------------------------------------------------------------------------
# Test 5 — SAGA_EMIT_ENABLED=False → no HTTP traffic
# ---------------------------------------------------------------------------

class TestEmitDisabled:
    """When SAGA_EMIT_ENABLED is False, emit() must be a complete no-op."""

    def test_disabled_makes_no_http_call(self, monkeypatch):
        """No httpx.Client is created when the feature flag is off."""
        monkeypatch.setattr(cfg, "SAGA_EMIT_ENABLED", False)
        _reset_for_testing()

        with patch("conduit.seam.emit.httpx.Client") as mock_client_cls:
            emit(make_envelope())
            # Give any potential background thread a moment to wake.
            time.sleep(0.1)
            mock_client_cls.assert_not_called()

    def test_disabled_returns_none(self, monkeypatch):
        """emit() returns None when disabled."""
        monkeypatch.setattr(cfg, "SAGA_EMIT_ENABLED", False)
        _reset_for_testing()
        result = emit(make_envelope())
        assert result is None

    def test_disabled_does_not_start_worker(self, monkeypatch):
        """No worker thread is started when the feature flag is off."""
        monkeypatch.setattr(cfg, "SAGA_EMIT_ENABLED", False)
        _reset_for_testing()

        emit(make_envelope())
        time.sleep(0.1)

        # _worker_thread should remain None since emit() returned early.
        assert emit_mod._worker_thread is None, (
            "Worker thread was started even though SAGA_EMIT_ENABLED=False"
        )

    def test_re_enabling_after_disable_posts_again(self, monkeypatch):
        """After re-enabling, the next emit() does reach the worker."""
        monkeypatch.setattr(cfg, "SAGA_EMIT_ENABLED", False)
        _reset_for_testing()

        # Disable: no emission.
        emit(make_envelope(request_id="disabled-id"))
        time.sleep(0.1)

        # Re-enable.
        monkeypatch.setattr(cfg, "SAGA_EMIT_ENABLED", True)
        _reset_for_testing()

        received: list[dict] = []
        fired = threading.Event()

        def capturing_post(client, payload: dict) -> None:
            received.append(payload)
            fired.set()

        emit_mod._post_once = capturing_post
        try:
            emit(make_envelope(request_id="enabled-id"))
            assert _wait_for_event(fired, timeout=2.0), (
                "No emit after re-enabling SAGA_EMIT_ENABLED"
            )
        finally:
            _reset_for_testing()

        assert received[0]["request_id"] == "enabled-id"
        for item in received:
            assert item["request_id"] != "disabled-id", (
                "disabled-id was queued and later emitted — should have been dropped"
            )
