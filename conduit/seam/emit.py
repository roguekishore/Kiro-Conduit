"""SAGA seam emitter — Lane F.

Fire-and-forget HTTP POST to the SAGA ingest endpoint.
Contract: CONTRACT-conduit-saga-seam.md v1 (frozen 2026-09-12).

THE invariant: emit() MUST NOT block, delay, or fail the client response.
If SAGA is unreachable, the emit is dropped and logged; the caller is
unaffected in all failure modes.

Architecture
------------
A single module-level :class:`queue.Queue` (bounded, drop-oldest on full)
decouples the caller from the network call. A daemon background thread drains
the queue and POSTs each payload to SAGA. The caller only does a put_nowait —
O(1) and non-blocking — then returns.
"""
from __future__ import annotations

import dataclasses
import json
import queue
import threading
import uuid
from enum import Enum
from typing import Optional

import httpx
from loguru import logger

from conduit.contracts.seam import SeamEnvelope
import conduit.config as cfg


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

class _SeamEncoder(json.JSONEncoder):
    """JSON encoder that serialises Enum instances as their .value."""

    def default(self, obj: object) -> object:
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)  # type: ignore[arg-type]


def _serialize(envelope: SeamEnvelope) -> dict:
    """Convert SeamEnvelope to a plain dict safe for JSON serialisation.

    Uses dataclasses.asdict for recursive conversion; enums are left as
    instances and handled by _SeamEncoder at POST time.
    """
    return dataclasses.asdict(envelope)


# ---------------------------------------------------------------------------
# Module-level queue and worker
# ---------------------------------------------------------------------------

_queue: Optional[queue.Queue] = None
_worker_thread: Optional[threading.Thread] = None
_init_lock = threading.Lock()


def _get_queue() -> queue.Queue:
    """Return the shared emit queue, creating it on first call.

    Queue size is read from config at creation time so tests that monkeypatch
    SAGA_EMIT_QUEUE_SIZE before calling _reset_for_testing() see the right size.
    """
    global _queue
    if _queue is not None:
        return _queue
    with _init_lock:
        if _queue is None:
            _queue = queue.Queue(maxsize=cfg.SAGA_EMIT_QUEUE_SIZE)
    return _queue  # type: ignore[return-value]


def _ensure_worker() -> None:
    """Start the background consumer thread if it is not already alive."""
    global _worker_thread
    if _worker_thread is not None and _worker_thread.is_alive():
        return
    with _init_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            t = threading.Thread(
                target=_run_worker,
                daemon=True,
                name="saga-emit-worker",
            )
            t.start()
            _worker_thread = t


def _run_worker() -> None:
    """Background thread: drain the queue and POST each payload to SAGA.

    Runs forever (daemon=True, killed on process exit). Captures the queue
    reference once at startup so _reset_for_testing() can replace the module
    global without interfering with an already-running worker.
    """
    with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
        q = _get_queue()
        while True:
            try:
                payload: dict = q.get(block=True, timeout=1.0)
            except queue.Empty:
                continue
            _post_once(client, payload)


def _post_once(client: httpx.Client, payload: dict) -> None:
    """POST one payload dict to SAGA; swallow and log all exceptions.

    This function MUST NOT raise. All failure modes — connection refused,
    timeout, 4xx, 5xx — are caught here so they never surface to the caller.
    """
    try:
        body = json.dumps(payload, cls=_SeamEncoder)
        resp = client.post(
            cfg.SAGA_INGEST_URL,
            content=body.encode(),
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code >= 400:
            logger.warning(
                "seam/emit: SAGA returned HTTP {} for request_id={}",
                resp.status_code,
                payload.get("request_id", "?"),
            )
        else:
            logger.debug(
                "seam/emit: OK HTTP {} request_id={}",
                resp.status_code,
                payload.get("request_id", "?"),
            )
    except httpx.ConnectError as exc:
        logger.warning("seam/emit: connection refused — {}", exc)
    except httpx.TimeoutException as exc:
        logger.warning("seam/emit: timeout — {}", exc)
    except Exception as exc:
        logger.warning("seam/emit: unexpected POST error — {}", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def emit(envelope: SeamEnvelope) -> None:
    """Fire-and-forget emit to SAGA ingest endpoint.

    Never blocks. Never raises. Returns None in all circumstances.

    If SAGA_EMIT_ENABLED is False this is a no-op.

    Drop-oldest: when the bounded queue is full the oldest entry is evicted
    to make room. A wedged SAGA therefore cannot grow memory without limit.

    Missing request_id: if the envelope's request_id is empty (x-saga-request-id
    header was absent on the inbound request), a generated id is logged at DEBUG
    and the emit proceeds — it is never suppressed for a missing header.
    """
    if not cfg.SAGA_EMIT_ENABLED:
        return

    try:
        # Defensive: if caller passed an empty request_id, generate one.
        # The normal path is: route handler reads x-saga-request-id and passes
        # it when constructing SeamEnvelope; this branch guards the gap period
        # before WS-C forwards that header.
        if not envelope.request_id:
            generated = f"gen-{uuid.uuid4().hex}"
            logger.debug(
                "seam/emit: x-saga-request-id absent — using generated id {}",
                generated,
            )
            envelope.request_id = generated  # SeamEnvelope is not frozen

        payload = _serialize(envelope)
        q = _get_queue()

        try:
            q.put_nowait(payload)
        except queue.Full:
            # Drop-oldest: evict the front of the queue and retry once.
            try:
                q.get_nowait()
                logger.debug(
                    "seam/emit: queue full — oldest entry evicted to make room"
                )
            except queue.Empty:
                pass
            try:
                q.put_nowait(payload)
            except queue.Full:
                logger.warning(
                    "seam/emit: queue still full after eviction — dropping emit "
                    "for request_id={}",
                    payload.get("request_id", "?"),
                )
                return

        _ensure_worker()

    except Exception as exc:
        # Absolute last-resort catch: emit() must NEVER raise, even if this
        # module has an internal bug.
        logger.warning("seam/emit: internal error, dropping emit — {}", exc)


# ---------------------------------------------------------------------------
# Test utilities — NOT for production use
# ---------------------------------------------------------------------------

def _reset_for_testing() -> None:
    """Reset module-level state for test isolation.

    Drains and discards the existing queue so any still-running worker thread
    (daemon) has nothing left to process. Sets both globals to None so the next
    call to _get_queue() / _ensure_worker() starts fresh with current config
    values (allowing tests to monkeypatch SAGA_EMIT_QUEUE_SIZE etc. before the
    queue is created).
    """
    global _queue, _worker_thread
    with _init_lock:
        if _queue is not None:
            # Drain so an old worker thread processes no leftover items.
            while True:
                try:
                    _queue.get_nowait()
                except queue.Empty:
                    break
        _queue = None
        _worker_thread = None
