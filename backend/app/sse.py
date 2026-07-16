"""Byte-buffered SSE framing — the UTF-8-safe core of streaming (SPEC §5).

THE invariant: bytes are decoded ONLY after a complete event has been isolated
at a delimiter boundary. Never per-packet — a multibyte character (Devanagari,
Tamil, Bengali, emoji) can split across TCP reads, and decoding a fragment
raises UnicodeDecodeError. This is safe because SSE delimiters are pure ASCII
and UTF-8 continuation bytes can never look like ASCII (high bit set), so a
delimiter can never land *inside* a multibyte character: a complete event is
always a complete UTF-8 sequence.

Framing rules (WHATWG SSE spec, and what providers actually send):
* An event ends at a blank line. OpenAI/Fireworks send LF-LF; CRLF servers
  send CRLF-CRLF. A naive b"\\n\\n" search NEVER matches b"\\r\\n\\r\\n"
  (no two LFs are adjacent in it — the stream would buffer forever), so we
  scan for both and cut at whichever occurs earliest.
* Anything after the last delimiter stays in the buffer — fragmentation
  (one event across many reads) and coalescing (many events in one read)
  both fall out of the same loop: append, split-while-possible, retain.

This module is pure logic: no I/O, no httpx, no knowledge of OpenAI payload
shapes. Parsing `data: {...}` JSON out of an event is the caller's concern.
"""
from __future__ import annotations

from collections.abc import Iterator

# Recognized event delimiters, checked earliest-match-first.
_DELIMITERS: tuple[bytes, ...] = (b"\n\n", b"\r\n\r\n")


class ByteStreamProcessor:
    """Incremental SSE event framer over a raw byte stream.

    Usage:
        proc = ByteStreamProcessor()
        async for chunk in upstream_bytes:
            for event in proc.feed(chunk):
                ...             # event is a complete, decoded str
        tail = proc.flush()     # anything left after the stream closed
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, chunk: bytes) -> Iterator[str]:
        """Absorb a chunk; yield every complete event it completes.

        Yields decoded event text WITHOUT the trailing delimiter. Partial
        trailing data (including a partially-arrived delimiter or a split
        multibyte character) stays buffered for the next feed().
        """
        self._buf += chunk
        while True:
            cut = self._find_earliest_delimiter()
            if cut is None:
                return
            start, end = cut
            event_bytes = bytes(self._buf[:start])
            del self._buf[:end]
            # Complete event ⇒ complete UTF-8 (see module docstring). A
            # decode failure here means the upstream sent genuinely invalid
            # UTF-8 — that should surface, not be masked with replacement
            # characters that would corrupt the reassembled log copy.
            yield event_bytes.decode("utf-8")

    def flush(self) -> str | None:
        """Decode whatever remains after the stream has ended.

        Some servers close without a final delimiter; the tail is still a
        complete event from their point of view. Returns None if nothing is
        buffered. Raises UnicodeDecodeError if the connection died mid-
        character — at true end-of-stream that's real data loss and must be
        loud, not papered over.
        """
        if not self._buf:
            return None
        tail = bytes(self._buf).decode("utf-8")
        self._buf.clear()
        return tail

    @property
    def pending_bytes(self) -> int:
        """Bytes currently buffered awaiting a delimiter (for tests/metrics)."""
        return len(self._buf)

    def _find_earliest_delimiter(self) -> tuple[int, int] | None:
        """Locate the earliest delimiter; return (event_end, delimiter_end).

        Earliest match wins. (The two delimiters begin with different bytes,
        so they can never match at the same index; the length comparison is
        pure belt-and-suspenders should the delimiter set ever grow.)
        """
        best: tuple[int, int] | None = None
        for delim in _DELIMITERS:
            idx = self._buf.find(delim)
            if idx == -1:
                continue
            if best is None or idx < best[0] or (idx == best[0] and len(delim) > best[1] - best[0]):
                best = (idx, idx + len(delim))
        return best
