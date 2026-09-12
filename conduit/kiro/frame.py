"""AWS event-stream binary framing decoder.

Implements the 12-byte prelude + headers + payload + CRC32 frame format used
by Amazon's CodeWhisperer streaming service.  A caller feeds raw bytes from an
HTTP chunked body and receives back a list of fully-decoded messages; partial
frames are held in an internal buffer until the rest arrives.

References
----------
- AWS Event Stream Encoding specification (binary framing)
- CONDUIT PLATFORM-FINDINGS 5a-bis (Kiro wire observations)
"""

from __future__ import annotations

import binascii
import struct
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FrameError(Exception):
    """Raised when a decoded frame fails a CRC check or is structurally invalid."""


# ---------------------------------------------------------------------------
# Internal header parser
# ---------------------------------------------------------------------------


def _parse_headers(data: bytes) -> Dict[str, str]:
    """Parse the raw headers section of one AWS event-stream frame.

    Each header is encoded as::

        1  byte  : name length (N)
        N  bytes : name (UTF-8)
        1  byte  : value type  (7 = string; only type used by Kiro)
        2  bytes : value length (M, uint16 big-endian)
        M  bytes : value (UTF-8)

    Non-string value types are stored as their raw bytes decoded as latin-1 so
    callers always receive ``str`` values.  Unknown types never raise; they are
    carried through.

    Args:
        data: The raw headers section bytes (exactly ``headers_len`` bytes).

    Returns:
        A dict mapping each header name to its decoded string value.
    """
    headers: Dict[str, str] = {}
    pos = 0
    while pos < len(data):
        if pos >= len(data):
            break
        name_len: int = data[pos]
        pos += 1
        name: str = data[pos : pos + name_len].decode("utf-8")
        pos += name_len
        value_type: int = data[pos]
        pos += 1
        value_len: int = struct.unpack_from(">H", data, pos)[0]
        pos += 2
        raw_value: bytes = data[pos : pos + value_len]
        pos += value_len
        if value_type == 7:
            # STRING
            headers[name] = raw_value.decode("utf-8")
        else:
            # Carry unknown types as latin-1 so callers always get str
            headers[name] = raw_value.decode("latin-1")
    return headers


# ---------------------------------------------------------------------------
# Public decoder
# ---------------------------------------------------------------------------


class FrameDecoder:
    """Stateful decoder for the AWS event-stream binary framing.

    Usage::

        decoder = FrameDecoder()
        for http_chunk in response.iter_bytes():
            for msg in decoder.feed(http_chunk):
                event_type = msg["headers"].get(":event-type", "")
                payload    = msg["payload"]   # bytes

    Each returned message dict has the shape::

        {
            "headers": dict[str, str],   # all frame headers, e.g. {":event-type": "assistantResponseEvent"}
            "payload": bytes,             # raw payload bytes (usually JSON-encoded)
        }

    Partial frames are buffered; a subsequent ``feed`` call will complete them.

    CRC errors raise :class:`FrameError` immediately.
    """

    def __init__(self) -> None:
        self._buf: bytearray = bytearray()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, chunk: bytes) -> List[Dict]:
        """Append *chunk* to the internal buffer and return all complete frames.

        Args:
            chunk: Raw bytes received from the HTTP stream.

        Returns:
            A list (possibly empty) of decoded message dicts.

        Raises:
            FrameError: If a frame's prelude or message CRC does not match.
        """
        self._buf.extend(chunk)
        results: List[Dict] = []
        while True:
            msg = self._try_decode_one()
            if msg is None:
                break
            results.append(msg)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_decode_one(self) -> Optional[Dict]:
        """Attempt to decode one frame from the head of the buffer.

        Returns:
            A decoded message dict, or ``None`` if the buffer holds fewer bytes
            than the next complete frame.

        Raises:
            FrameError: On any CRC mismatch.
        """
        buf = self._buf

        # Need at least the 12-byte prelude before we can determine frame size
        if len(buf) < 12:
            return None

        total_len: int = struct.unpack_from(">I", buf, 0)[0]
        headers_len: int = struct.unpack_from(">I", buf, 4)[0]
        stated_prelude_crc: int = struct.unpack_from(">I", buf, 8)[0]

        # Guard against unreasonably large total_len before allocating
        if total_len < 16:
            raise FrameError(
                f"Frame total_len {total_len} is below minimum (16 bytes); "
                "stream is corrupt or misaligned."
            )

        # Wait until the full frame is in the buffer
        if len(buf) < total_len:
            return None

        # --- Prelude CRC (covers bytes 0-7) --------------------------------
        computed_prelude_crc: int = binascii.crc32(bytes(buf[0:8])) & 0xFFFFFFFF
        if computed_prelude_crc != stated_prelude_crc:
            raise FrameError(
                f"Prelude CRC mismatch at frame start: "
                f"stated={stated_prelude_crc:#010x}, "
                f"computed={computed_prelude_crc:#010x}"
            )

        # --- Message CRC (covers bytes 0..total_len-5) ---------------------
        stated_msg_crc: int = struct.unpack_from(">I", buf, total_len - 4)[0]
        computed_msg_crc: int = binascii.crc32(bytes(buf[0 : total_len - 4])) & 0xFFFFFFFF
        if computed_msg_crc != stated_msg_crc:
            raise FrameError(
                f"Message CRC mismatch: "
                f"stated={stated_msg_crc:#010x}, "
                f"computed={computed_msg_crc:#010x}"
            )

        # --- Decode headers -------------------------------------------------
        headers_start = 12
        headers_end = 12 + headers_len
        headers: Dict[str, str] = _parse_headers(bytes(buf[headers_start:headers_end]))

        # --- Extract payload ------------------------------------------------
        payload: bytes = bytes(buf[headers_end : total_len - 4])

        # Consume the frame from the buffer
        del buf[:total_len]

        return {"headers": headers, "payload": payload}
