"""ByteStreamProcessor tests — fragmentation, coalescing, UTF-8 boundaries.

The centerpiece is the SPEC-mandated Devanagari test: a multibyte character
split mid-sequence across two reads must reassemble perfectly. Devanagari
chars are 3 bytes in UTF-8, so every split point inside one is a guaranteed
UnicodeDecodeError for any per-packet decoder — which is the bug class this
processor exists to make impossible.
"""
from __future__ import annotations

import pytest

from app.sse import ByteStreamProcessor


def feed_all(proc: ByteStreamProcessor, chunks: list[bytes]) -> list[str]:
    out: list[str] = []
    for c in chunks:
        out.extend(proc.feed(c))
    return out


# ---------------------------------------------------------------------------
# Basic framing
# ---------------------------------------------------------------------------

class TestFraming:
    def test_single_complete_event(self):
        proc = ByteStreamProcessor()
        assert feed_all(proc, [b'data: {"x":1}\n\n']) == ['data: {"x":1}']
        assert proc.pending_bytes == 0

    def test_no_yield_until_delimiter(self):
        proc = ByteStreamProcessor()
        assert list(proc.feed(b"data: incompl")) == []
        assert proc.pending_bytes == len(b"data: incompl")

    def test_fragmentation_one_event_across_many_reads(self):
        # One event drip-fed byte by byte — yields nothing until the final \n.
        proc = ByteStreamProcessor()
        payload = b'data: {"delta":"hi"}\n\n'
        events: list[str] = []
        for i, b in enumerate(payload):
            got = list(proc.feed(bytes([b])))
            if i < len(payload) - 1:
                assert got == [], f"yielded early at byte {i}"
            events.extend(got)
        assert events == ['data: {"delta":"hi"}']

    def test_coalescing_many_events_in_one_read(self):
        proc = ByteStreamProcessor()
        got = feed_all(proc, [b"data: a\n\ndata: b\n\ndata: c\n\n"])
        assert got == ["data: a", "data: b", "data: c"]
        assert proc.pending_bytes == 0

    def test_coalesced_read_with_partial_trailing_event(self):
        # Two complete events + the head of a third in one read: yield two,
        # retain the tail.
        proc = ByteStreamProcessor()
        got = feed_all(proc, [b"data: a\n\ndata: b\n\ndata: c"])
        assert got == ["data: a", "data: b"]
        assert proc.pending_bytes == len(b"data: c")
        # The tail completes on the next read.
        assert feed_all(proc, [b"\n\n"]) == ["data: c"]

    def test_delimiter_itself_split_across_reads(self):
        # First \n at the end of read 1, second \n at the start of read 2.
        proc = ByteStreamProcessor()
        assert feed_all(proc, [b"data: x\n"]) == []
        assert feed_all(proc, [b"\n"]) == ["data: x"]

    def test_multiline_event_single_newlines_do_not_split(self):
        # SSE events may span lines (data: + data:); only the BLANK line ends one.
        proc = ByteStreamProcessor()
        got = feed_all(proc, [b"data: line1\ndata: line2\n\n"])
        assert got == ["data: line1\ndata: line2"]

    def test_empty_chunks_are_harmless(self):
        proc = ByteStreamProcessor()
        assert feed_all(proc, [b"", b"data: x", b"", b"\n\n", b""]) == ["data: x"]


# ---------------------------------------------------------------------------
# CRLF framing (real servers do this; naive \n\n search hangs forever)
# ---------------------------------------------------------------------------

class TestCRLF:
    def test_crlf_delimited_events(self):
        proc = ByteStreamProcessor()
        got = feed_all(proc, [b"data: a\r\n\r\ndata: b\r\n\r\n"])
        assert got == ["data: a", "data: b"]

    def test_crlf_delimiter_split_across_reads(self):
        proc = ByteStreamProcessor()
        assert feed_all(proc, [b"data: a\r\n"]) == []
        assert feed_all(proc, [b"\r\n"]) == ["data: a"]

    def test_mixed_lf_and_crlf_streams(self):
        proc = ByteStreamProcessor()
        got = feed_all(proc, [b"data: a\n\ndata: b\r\n\r\ndata: c\n\n"])
        assert got == ["data: a", "data: b", "data: c"]


# ---------------------------------------------------------------------------
# UTF-8 boundary safety — the reason this class exists
# ---------------------------------------------------------------------------

class TestUTF8Boundaries:
    def test_devanagari_split_mid_character(self):
        """SPEC-mandated: split नमस्ते's bytes down the middle of a character,
        feed in two chunks, assert perfect reassembly with no decode error."""
        text = "नमस्ते"
        payload = f"data: {text}".encode("utf-8") + b"\n\n"
        # "data: " is 6 bytes; न is bytes 6..9 (3 bytes: e0 a4 a8).
        # Cut at byte 7 — INSIDE न. Decoding either half alone must fail;
        # the processor decodes only the reassembled whole.
        cut = 7
        first, second = payload[:cut], payload[cut:]
        with pytest.raises(UnicodeDecodeError):
            first.decode("utf-8")  # proves the split is genuinely mid-character

        proc = ByteStreamProcessor()
        assert list(proc.feed(first)) == []      # buffered, not decoded
        events = list(proc.feed(second))
        assert events == [f"data: {text}"]       # intact, byte-perfect

    def test_devanagari_every_possible_split_point(self):
        # Not just one lucky cut: EVERY split of the payload must reassemble.
        text = "नमस्ते"
        payload = f"data: {text}".encode("utf-8") + b"\n\n"
        for cut in range(1, len(payload)):
            proc = ByteStreamProcessor()
            events = feed_all(proc, [payload[:cut], payload[cut:]])
            assert events == [f"data: {text}"], f"failed at split {cut}"

    def test_emoji_4byte_split_three_ways(self):
        # 🚀 is 4 bytes (f0 9f 9a 80) — feed it one byte at a time.
        payload = "data: 🚀".encode("utf-8") + b"\n\n"
        proc = ByteStreamProcessor()
        events = feed_all(proc, [bytes([b]) for b in payload])
        assert events == ["data: 🚀"]

    def test_tamil_bengali_mixed_stream_fragmented(self):
        # The India-market scripts from SPEC, streamed in awkward chunks.
        text = "தமிழ் বাংলা हिन्दी"
        payload = f"data: {text}".encode("utf-8") + b"\n\n"
        third = len(payload) // 3
        proc = ByteStreamProcessor()
        events = feed_all(
            proc, [payload[:third], payload[third : 2 * third], payload[2 * third :]]
        )
        assert events == [f"data: {text}"]

    def test_multibyte_spanning_event_boundary_chunks(self):
        # Two events where the chunk boundary lands inside the SECOND event's
        # multibyte char — the first event must still be yielded immediately.
        e1 = "data: ok".encode("utf-8") + b"\n\n"
        e2 = "data: नमस्ते".encode("utf-8") + b"\n\n"
        combined = e1 + e2
        cut = len(e1) + 8  # inside न of event 2
        proc = ByteStreamProcessor()
        got1 = list(proc.feed(combined[:cut]))
        assert got1 == ["data: ok"]  # event 1 released without waiting
        got2 = list(proc.feed(combined[cut:]))
        assert got2 == ["data: नमस्ते"]


# ---------------------------------------------------------------------------
# flush() — end-of-stream semantics
# ---------------------------------------------------------------------------

class TestFlush:
    def test_flush_returns_undelimited_tail(self):
        proc = ByteStreamProcessor()
        list(proc.feed(b"data: [DONE]"))  # server closed without final \n\n
        assert proc.flush() == "data: [DONE]"
        assert proc.pending_bytes == 0

    def test_flush_empty_returns_none(self):
        proc = ByteStreamProcessor()
        assert proc.flush() is None

    def test_flush_after_complete_events_returns_none(self):
        proc = ByteStreamProcessor()
        list(proc.feed(b"data: x\n\n"))
        assert proc.flush() is None

    def test_flush_mid_character_raises_loudly(self):
        # Connection died in the middle of न: that is real data loss and must
        # surface as an error, not be silently replaced with U+FFFD.
        proc = ByteStreamProcessor()
        list(proc.feed("data: न".encode("utf-8")[:-1]))  # truncated 3-byte char
        with pytest.raises(UnicodeDecodeError):
            proc.flush()
