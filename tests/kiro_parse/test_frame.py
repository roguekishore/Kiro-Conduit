"""Tests for conduit.kiro.frame — AWS event-stream binary framing.

All fixtures are synthetic (no live Kiro calls).
"""

from __future__ import annotations

import binascii
import struct

import pytest

from conduit.kiro.frame import FrameDecoder, FrameError

from .conftest import build_frame, json_payload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _feed_all(frames: list[bytes]) -> list[dict]:
    decoder = FrameDecoder()
    results = []
    for chunk in frames:
        results.extend(decoder.feed(chunk))
    return results


# ---------------------------------------------------------------------------
# Basic round-trip: one frame per known event type
# ---------------------------------------------------------------------------


EVENT_TYPES = [
    "assistantResponseEvent",
    "reasoningContentEvent",
    "toolUseEvent",
    "toolResultEvent",
    "meteringEvent",
    "metadataEvent",
    "messageMetadataEvent",
    "contextUsageEvent",
    "codeEvent",
    "codeReferenceEvent",
    "followupPromptEvent",
    "intentsEvent",
    "invalidStateEvent",
    "supplementaryWebLinksEvent",
    "citationEvent",
    "documentCitationEvent",
    "messageStopEvent",
]


@pytest.mark.parametrize("event_type", EVENT_TYPES)
def test_round_trip_each_event_type(event_type: str) -> None:
    """Every known EventName value survives a full encode→decode cycle."""
    payload = json_payload({"_type": event_type, "data": "hello"})
    frame = build_frame(event_type, payload)

    decoder = FrameDecoder()
    messages = decoder.feed(frame)

    assert len(messages) == 1
    msg = messages[0]
    assert msg["headers"][":event-type"] == event_type
    assert msg["payload"] == payload


# ---------------------------------------------------------------------------
# Multiple frames in one feed() call
# ---------------------------------------------------------------------------


def test_two_frames_in_one_chunk() -> None:
    """Two back-to-back frames fed as a single chunk both decode."""
    frame1 = build_frame("assistantResponseEvent", json_payload({"content": "hello"}))
    frame2 = build_frame("metadataEvent", json_payload({"tokenUsage": {}}))

    decoder = FrameDecoder()
    messages = decoder.feed(frame1 + frame2)

    assert len(messages) == 2
    assert messages[0]["headers"][":event-type"] == "assistantResponseEvent"
    assert messages[1]["headers"][":event-type"] == "metadataEvent"


# ---------------------------------------------------------------------------
# Partial frame buffering — the critical split-chunk test
# ---------------------------------------------------------------------------


def test_partial_frame_split_across_two_chunks() -> None:
    """A frame split at an arbitrary byte boundary is reassembled correctly."""
    payload = json_payload({"content": "partial buffering works"})
    frame = build_frame("assistantResponseEvent", payload)

    # Split at byte 7 — right inside the prelude, before the CRC
    split_point = 7
    chunk_a = frame[:split_point]
    chunk_b = frame[split_point:]

    decoder = FrameDecoder()
    part1 = decoder.feed(chunk_a)
    assert part1 == [], "No complete frame yet after first chunk"

    part2 = decoder.feed(chunk_b)
    assert len(part2) == 1
    assert part2[0]["headers"][":event-type"] == "assistantResponseEvent"
    assert part2[0]["payload"] == payload


def test_partial_frame_split_mid_payload() -> None:
    """A frame split mid-payload is buffered and completed on the next feed."""
    payload = json_payload({"content": "x" * 200})
    frame = build_frame("assistantResponseEvent", payload)

    # Split exactly in the middle of the payload section
    mid = len(frame) // 2

    decoder = FrameDecoder()
    assert decoder.feed(frame[:mid]) == []
    messages = decoder.feed(frame[mid:])
    assert len(messages) == 1
    assert messages[0]["payload"] == payload


def test_partial_frame_one_byte_at_a_time() -> None:
    """A frame fed byte-by-byte still yields exactly one decoded message."""
    payload = json_payload({"k": "v"})
    frame = build_frame("messageStopEvent", payload)

    decoder = FrameDecoder()
    results = []
    for i in range(len(frame)):
        results.extend(decoder.feed(bytes([frame[i]])))

    assert len(results) == 1
    assert results[0]["headers"][":event-type"] == "messageStopEvent"


# ---------------------------------------------------------------------------
# Unknown `:event-type` passes through at the frame level
# ---------------------------------------------------------------------------


def test_unknown_event_type_passes_through_frame_layer() -> None:
    """frame.py carries unknown event-type headers — dispatch, not frame, filters them."""
    frame = build_frame("someFutureEvent", json_payload({"data": 1}))
    decoder = FrameDecoder()
    messages = decoder.feed(frame)
    assert len(messages) == 1
    assert messages[0]["headers"][":event-type"] == "someFutureEvent"


# ---------------------------------------------------------------------------
# Empty payload
# ---------------------------------------------------------------------------


def test_empty_payload_frame() -> None:
    """A frame with zero payload bytes decodes without error."""
    frame = build_frame("messageStopEvent", b"")
    decoder = FrameDecoder()
    messages = decoder.feed(frame)
    assert len(messages) == 1
    assert messages[0]["payload"] == b""


# ---------------------------------------------------------------------------
# CRC error detection
# ---------------------------------------------------------------------------


def test_prelude_crc_mismatch_raises() -> None:
    """Flipping a byte in the prelude CRC triggers FrameError."""
    frame = bytearray(build_frame("metadataEvent", json_payload({})))
    # Bytes 8-11 are the prelude CRC — flip one bit
    frame[8] ^= 0xFF
    decoder = FrameDecoder()
    with pytest.raises(FrameError, match="Prelude CRC"):
        decoder.feed(bytes(frame))


def test_message_crc_mismatch_raises() -> None:
    """Flipping a payload byte triggers a message CRC error."""
    payload = json_payload({"x": 1})
    frame = bytearray(build_frame("metadataEvent", payload))
    # Flip the first payload byte (after the 12-byte prelude + headers)
    # Locate payload start: prelude(12) + headers
    headers_len = struct.unpack_from(">I", frame, 4)[0]
    payload_start = 12 + headers_len
    frame[payload_start] ^= 0x01
    decoder = FrameDecoder()
    with pytest.raises(FrameError, match="Message CRC"):
        decoder.feed(bytes(frame))


# ---------------------------------------------------------------------------
# Extra headers round-trip
# ---------------------------------------------------------------------------


def test_extra_headers_preserved() -> None:
    """Additional frame headers beyond :event-type are decoded and returned."""
    frame = build_frame(
        "metadataEvent",
        json_payload({}),
        extra_headers={":content-type": "application/json"},
    )
    decoder = FrameDecoder()
    messages = decoder.feed(frame)
    assert len(messages) == 1
    assert messages[0]["headers"][":content-type"] == "application/json"
