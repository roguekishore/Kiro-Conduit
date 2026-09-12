"""Shared helpers for building synthetic AWS event-stream binary frames.

No live Kiro calls.  All fixtures are constructed from first principles so the
tests have no external dependencies.
"""

from __future__ import annotations

import binascii
import json
import struct
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Low-level frame builder
# ---------------------------------------------------------------------------


def _encode_header(name: str, value: str) -> bytes:
    """Encode a single AWS event-stream header (value type 7 = string)."""
    name_bytes = name.encode("utf-8")
    value_bytes = value.encode("utf-8")
    return (
        struct.pack("B", len(name_bytes))
        + name_bytes
        + struct.pack("B", 7)          # value type: string
        + struct.pack(">H", len(value_bytes))
        + value_bytes
    )


def build_frame(
    event_type: str,
    payload: bytes | None = None,
    extra_headers: Dict[str, str] | None = None,
) -> bytes:
    """Construct a valid AWS event-stream binary frame.

    Args:
        event_type: Value for the ``:event-type`` header.
        payload: Raw payload bytes.  Defaults to ``b""``.
        extra_headers: Additional headers beyond ``:event-type``.

    Returns:
        A fully CRC-signed frame as bytes.
    """
    payload = payload or b""
    headers_blob = _encode_header(":event-type", event_type)
    if extra_headers:
        for k, v in extra_headers.items():
            headers_blob += _encode_header(k, v)

    headers_len = len(headers_blob)
    total_len = 12 + headers_len + len(payload) + 4

    prelude = struct.pack(">II", total_len, headers_len)
    prelude_crc = struct.pack(">I", binascii.crc32(prelude) & 0xFFFFFFFF)

    body_without_crc = prelude + prelude_crc + headers_blob + payload
    msg_crc = struct.pack(">I", binascii.crc32(body_without_crc) & 0xFFFFFFFF)
    return body_without_crc + msg_crc


def build_frame_no_event_type_header(payload: bytes) -> bytes:
    """Build a frame with NO ``:event-type`` header (for fallback-inference tests)."""
    headers_blob = _encode_header(":message-type", "event")
    headers_len = len(headers_blob)
    total_len = 12 + headers_len + len(payload) + 4

    prelude = struct.pack(">II", total_len, headers_len)
    prelude_crc = struct.pack(">I", binascii.crc32(prelude) & 0xFFFFFFFF)
    body_without_crc = prelude + prelude_crc + headers_blob + payload
    msg_crc = struct.pack(">I", binascii.crc32(body_without_crc) & 0xFFFFFFFF)
    return body_without_crc + msg_crc


def json_payload(obj: Any) -> bytes:
    """Serialise *obj* to compact UTF-8 JSON bytes."""
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")
