"""StreamUnmasker unit tests — the holdback buffer under hostile chunking."""
from __future__ import annotations

import pytest

from app.pii import StreamUnmasker, mask_text, mask_texts


PII_MAP = {"<EMAIL_1>": "alice@example.com", "<PHONE_1>": "+91 98765 43210"}


def run_chunks(chunks: list[bytes], pii_map: dict = PII_MAP) -> tuple[bytes, list[bytes]]:
    u = StreamUnmasker(pii_map)
    outs = [u.feed(c) for c in chunks]
    outs.append(u.flush())
    return b"".join(outs), [o for o in outs if o]


def test_whole_placeholder_in_one_chunk():
    total, _ = run_chunks([b"mail <EMAIL_1> today"])
    assert total == b"mail alice@example.com today"


def test_split_at_every_position_of_placeholder():
    """The paranoid proof (same style as the Devanagari framer test): for
    EVERY split point inside '<EMAIL_1>', the client output is identical and
    no emitted chunk ever contains a fractured placeholder."""
    text = b"data: contact <EMAIL_1> now\n\n"
    for cut in range(1, len(text)):
        total, emitted = run_chunks([text[:cut], text[cut:]])
        assert total == b"data: contact alice@example.com now\n\n", f"split {cut}"
        for chunk in emitted:
            s = chunk.decode()
            # No emitted chunk may end inside a placeholder signature...
            assert not s.endswith(("<", "<E", "<EM", "<EMA", "<EMAI", "<EMAIL",
                                   "<EMAIL_", "<EMAIL_1")), f"split {cut}: {s!r}"
            # ...and no unreplaced full placeholder may slip through.
            assert "<EMAIL_1>" not in s


def test_the_task_example_chunk_ending_in_EMA():
    total, emitted = run_chunks([b"call <EMA", b"IL_1> soon"])
    assert total == b"call alice@example.com soon"
    # The first yield withheld the partial: it is exactly "call ".
    assert emitted[0] == b"call "


def test_placeholder_split_across_three_chunks():
    total, _ = run_chunks([b"hi <E", b"MAIL", b"_1> bye"])
    assert total == b"hi alice@example.com bye"


def test_ordinary_less_than_is_released_not_swallowed():
    # '<' from real content (math, HTML) must flow through: the holdback
    # releases the moment following text breaks the placeholder shape.
    total, _ = run_chunks([b"3 < 5 is true, and 2<", b"=3 also"])
    assert total == b"3 < 5 is true, and 2<=3 also"


def test_html_tag_not_a_placeholder():
    total, _ = run_chunks([b"use <div> tags"])
    assert total == b"use <div> tags"  # lowercase → not placeholder shape...

def test_trailing_partial_released_at_flush():
    # Stream dies mid-placeholder: flush releases the artifact verbatim
    # rather than withholding bytes forever.
    total, _ = run_chunks([b"ending <EMA"])
    assert total == b"ending <EMA"


def test_multibyte_utf8_split_and_placeholder_split_together():
    # Devanagari char split mid-sequence AND placeholder split mid-signature
    # in the same stream — both boundary guards active at once.
    text = "नमस्ते <EMAIL_1> धन्यवाद".encode("utf-8")
    idx = text.find(b"<EMAIL_1>")
    cuts = [text[: idx - 1], text[idx - 1 : idx + 4], text[idx + 4 :]]
    # also split inside न (first char, 3 bytes):
    chunks = [cuts[0][:2], cuts[0][2:], cuts[1], cuts[2]]
    total, _ = run_chunks(chunks)
    assert total.decode("utf-8") == "नमस्ते alice@example.com धन्यवाद"


def test_multiple_placeholders_interleaved_chunks():
    total, _ = run_chunks([b"mail <EMA", b"IL_1>, call <PH", b"ONE_1> ok"])
    assert total == b"mail alice@example.com, call +91 98765 43210 ok"


def test_empty_map_passthrough_identity():
    u = StreamUnmasker({})
    assert u.feed(b"data: <EMAIL_1> stays\n\n") == b"data: <EMAIL_1> stays\n\n"
    assert u.flush() == b""


def test_mask_texts_shared_numbering_across_messages():
    masked, pii = mask_texts(
        ["contact a@x.com", "and also b@y.org", "again a@x.com"]
    )
    assert masked == ["contact <EMAIL_1>", "and also <EMAIL_2>", "again <EMAIL_1>"]
    assert pii == {"<EMAIL_1>": "a@x.com", "<EMAIL_2>": "b@y.org"}
