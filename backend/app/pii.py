"""Reversible PII masking (Phase 5) — regex + Luhn, no heavyweight NLP.

Approach: regex detection for emails, credit cards, and phone numbers, with
Luhn checksum validation gating card matches (a 16-digit order id that fails
Luhn is NOT a card — checksum beats pattern-matching for false positives).
Presidio was considered and rejected: it pulls spaCy + NLP models (hundreds
of MB) into a 16GB-constrained deployment for marginal gain on these
structured PII types. SPEC §6 explicitly allows the regex path.

Reversible tokenization: detected values become uniquely numbered
placeholders (<EMAIL_1>, <PHONE_1>, <CARD_1>) with a per-request map
{placeholder: raw} kept gateway-side. The SAME raw value reuses its
placeholder (dedupe), so "email me at a@x.com, again: a@x.com" masks to one
<EMAIL_1> twice. The model is expected to echo placeholders back;
`unmask_response` restores them — including gracefully handling the model
dropping a placeholder (nothing to restore) or lightly mangling it
(case/spacing variants are matched leniently).

Priority on overlapping matches: EMAIL > CARD > PHONE. An email containing a
10-digit local part must not be re-detected as a phone; a Luhn-valid card
must not be eaten as a phone number.

Classification of digit runs (after card/email exclusion):
* 13-19 digits + Luhn pass → card
* 10-13 digits → phone (covers bare 10-digit, +91/+1 country codes)
* anything else → left alone (years, prices, short ids)
Known tradeoff: a hypothetical 13-digit Luhn-valid phone number would
classify as a card — masked either way, only the label differs.

SSNs (A-1 audit fix): US Social Security Numbers in the canonical
NNN-NN-NNNN dashed form are detected as <SSN_n>. Undelimited 9-digit runs
are NOT treated as SSNs — 9 digits with no structure is more often an
order/tracking id, and false-positive masking corrupts prompts.

ACCEPTED LIMITATIONS (documented, deliberate — see audit A-1):
* Indian PAN card numbers (alphanumeric, e.g. ABCDE1234F) are NOT detected:
  the pattern is 5 letters + 4 digits + 1 letter, which collides with too
  many ordinary uppercase codes (invoice ids, ticket refs) for a regex-only
  detector to separate safely. Masking them needs contextual NLP — that's
  Presidio territory, rejected for weight (see above).
* Aadhaar numbers (12 digits) are caught INCIDENTALLY by the phone range,
  labeled <PHONE_n> — masked, mislabeled; acceptable for a redactor.
* Names, physical addresses, dates of birth: out of scope for regex; same
  Presidio tradeoff.
"""
from __future__ import annotations

import codecs
import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Domain = dot-separated labels, final label ≥2 chars and NO required TLD dot:
# India-market UPI/VPA addresses ("9876543210@upi", "name@okbank") are PII of
# the same sensitivity as emails and must mask the same way. Over-matching an
# @-address is the safe direction for a redactor.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)*[A-Za-z0-9-]{2,}")
# US SSN, canonical dashed form only (see module docstring for why
# undelimited 9-digit runs are deliberately excluded).
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# A digit run allowing common separators; optional leading + / ( so formats
# like "(555) 123-4567" capture whole. Classified by digit count + Luhn.
_NUMBER_RUN_RE = re.compile(r"\+?\(?\d(?:[\d\s().-]*\d)")

_PLACEHOLDER_LABELS = {"EMAIL", "SSN", "CARD", "PHONE"}


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass(frozen=True)
class _Match:
    start: int
    end: int
    label: str  # EMAIL | CARD | PHONE
    raw: str


def _detect(text: str) -> list[_Match]:
    """All PII spans, non-overlapping, priority EMAIL > SSN > CARD > PHONE."""
    matches: list[_Match] = []

    def overlaps(start: int, end: int) -> bool:
        return any(m.start < end and start < m.end for m in matches)

    for m in _EMAIL_RE.finditer(text):
        matches.append(_Match(m.start(), m.end(), "EMAIL", m.group()))

    for m in _SSN_RE.finditer(text):
        if not overlaps(m.start(), m.end()):
            matches.append(_Match(m.start(), m.end(), "SSN", m.group()))

    for m in _NUMBER_RUN_RE.finditer(text):
        if overlaps(m.start(), m.end()):
            continue
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            matches.append(_Match(m.start(), m.end(), "CARD", m.group()))
        elif 10 <= len(digits) <= 13:
            matches.append(_Match(m.start(), m.end(), "PHONE", m.group()))

    return sorted(matches, key=lambda m: m.start)


def mask_text(text: str) -> tuple[str, dict[str, str]]:
    """Redact PII; return (masked_text, {placeholder: raw_value}).

    Placeholders number per type in order of first appearance; identical raw
    values share one placeholder.
    """
    masked_list, pii_map = mask_texts([text])
    return masked_list[0], pii_map


def mask_texts(texts: list[str]) -> tuple[list[str], dict[str, str]]:
    """Mask several texts with ONE shared numbering/dedupe space.

    This is what makes multi-message requests correct: masking each message
    independently would hand two DIFFERENT emails the same <EMAIL_1> and
    corrupt the map. Here counters and value-dedupe span all texts, so a
    value repeated across messages reuses its placeholder and distinct
    values never collide.
    """
    pii_map: dict[str, str] = {}
    by_value: dict[tuple[str, str], str] = {}  # (label, raw) → placeholder
    counters: dict[str, int] = {}
    masked_out: list[str] = []

    for text in texts:
        matches = _detect(text)
        if not matches:
            masked_out.append(text)
            continue
        out: list[str] = []
        cursor = 0
        for m in matches:
            placeholder = by_value.get((m.label, m.raw))
            if placeholder is None:
                counters[m.label] = counters.get(m.label, 0) + 1
                placeholder = f"<{m.label}_{counters[m.label]}>"
                by_value[(m.label, m.raw)] = placeholder
                pii_map[placeholder] = m.raw
            out.append(text[cursor:m.start])
            out.append(placeholder)
            cursor = m.end
        out.append(text[cursor:])
        masked_out.append("".join(out))

    if pii_map:
        # Count only — never the values themselves — reaches the logs.
        logger.debug("masked %d PII value(s): %s", len(pii_map), sorted(pii_map))
    return masked_out, pii_map


def mask_request(body: dict) -> tuple[dict, dict[str, str]]:
    """Mask every message's content in an OpenAI-shaped request body.

    Returns (masked_body, pii_map). The original body is NOT mutated —
    masked_body is a shallow copy with a fresh messages list. When no PII is
    found the original body object is returned as-is (zero-copy fast path).
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return body, {}

    texts = [
        m["content"]
        for m in messages
        if isinstance(m, dict) and isinstance(m.get("content"), str)
    ]
    masked_texts, pii_map = mask_texts(texts)
    if not pii_map:
        return body, {}

    it = iter(masked_texts)
    new_messages = [
        {**m, "content": next(it)}
        if isinstance(m, dict) and isinstance(m.get("content"), str)
        else m
        for m in messages
    ]
    return {**body, "messages": new_messages}, pii_map


def mask_prompt(prompt: str) -> str:
    """Masked text only — the cache/coalescing pipelines key on this and
    never need reversal (their artifacts must simply be PII-free)."""
    masked, _ = mask_text(prompt)
    return masked


def contains_pii(text: str) -> bool:
    """True if any maskable PII is present. Used by cache/coalescing guards:
    two DIFFERENT users' prompts can mask to the SAME text ("my email is
    <EMAIL_1>"), so PII-bearing requests must never share cached/coalesced
    results — the collision would hand user A's answer to user B."""
    return bool(_detect(text))


def unmask_response(text: str, pii_map: dict[str, str]) -> str:
    """Swap placeholders back to raw values. Safe by construction:

    * exact placeholders are replaced;
    * lightly mangled ones (case changes, stray spaces/hyphens the model
      introduced: "< email_1 >", "<Email-1>") are caught by a lenient
      per-placeholder pattern;
    * DROPPED placeholders need no action — the model simply didn't echo
      that value, and nothing PII-bearing appears in its place;
    * unknown placeholders the model hallucinated (<EMAIL_9> with no map
      entry) are left verbatim — inventing data for them would be worse.
    Never raises on any input shape.
    """
    if not text or not pii_map:
        return text

    for placeholder, raw in pii_map.items():
        # Shape-validate FIRST: a malformed/corrupted map key must never
        # drive a replacement, not even an exact substring one.
        parsed = re.fullmatch(r"<([A-Z]+)_(\d+)>", placeholder)
        if parsed is None:
            continue
        if placeholder in text:
            text = text.replace(placeholder, raw)
            continue
        label, num = parsed.group(1), parsed.group(2)
        lenient = re.compile(
            rf"<\s*{label}[\s_-]*{num}\s*>", re.IGNORECASE
        )
        text, n = lenient.subn(raw, text)
        if n:
            logger.debug("unmask: lenient match restored %s (%d site(s))",
                         placeholder, n)
    return text


def unmask_payload(payload: dict, pii_map: dict[str, str]) -> dict:
    """Unmask a whole JSON response payload in one pass.

    Serialize → textual unmask → parse. SAFE BY CHARSET INVARIANT: every
    value our detectors can capture (emails, phones, cards) is drawn from
    [A-Za-z0-9._%+-@ ()-] — no double quotes, no backslashes, no control
    characters — so substituting raw values into serialized JSON strings
    cannot break JSON structure or escape a string literal. This covers
    placeholders anywhere in the payload (message content, refusals, error
    messages) without hand-walking the schema.
    """
    if not pii_map or not isinstance(payload, dict):
        return payload
    return json.loads(unmask_response(json.dumps(payload, ensure_ascii=False), pii_map))


# A trailing fragment that could be the START of a (possibly model-mangled)
# placeholder: "<", "<EMA", "<EMAIL_", "< email 1"... Anything a later chunk
# might complete into something unmask_response would match.
_PARTIAL_PLACEHOLDER_RE = re.compile(r"<\s?[A-Za-z]{0,12}[\s_-]?\d{0,4}\s?$")
# A placeholder (even mangled) is short; a '<' further back than this from
# the end can never be completed into one — release it.
_MAX_HOLDBACK = 24


class StreamUnmasker:
    """Incremental placeholder→raw substitution over an SSE byte stream.

    Two boundary problems, both handled:

    * UTF-8: a multibyte character can split across chunks. Bytes go through
      an incremental decoder that internally buffers partial sequences —
      never a per-chunk decode (same rule as sse.py).
    * THE HOLDBACK BUFFER: a placeholder can split across chunks too
      ("...call <EMA" | "IL_1> now..."). After substituting all complete
      placeholders, if the emitted text ends in a partial-placeholder
      signature, that tail is withheld until the next chunk completes or
      breaks the pattern. The client NEVER receives a fractured placeholder.
      A lone '<' from ordinary content ("3 < 5, right?") is released the
      moment the following text breaks the placeholder shape.

    SSE framing is untouched: substitution only ever rewrites placeholder
    text inside data lines ('\\n' breaks the partial pattern, so a holdback
    can never straddle an event delimiter). flush() releases any dangling
    tail verbatim at end-of-stream (a stream dying mid-placeholder is an
    upstream artifact; withholding bytes forever would be worse).
    """

    def __init__(self, pii_map: dict[str, str]):
        self._map = pii_map
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._tail = ""  # withheld partial-placeholder suffix

    def feed(self, chunk: bytes) -> bytes:
        text = self._tail + self._decoder.decode(chunk)
        self._tail = ""
        if not text:
            return b""
        text = unmask_response(text, self._map)

        cut = text.rfind("<", max(0, len(text) - _MAX_HOLDBACK))
        if cut != -1 and _PARTIAL_PLACEHOLDER_RE.fullmatch(text, cut):
            self._tail = text[cut:]
            text = text[:cut]
        return text.encode("utf-8")

    def flush(self) -> bytes:
        """End of stream: release whatever is withheld, unmasked if possible."""
        text = self._tail + self._decoder.decode(b"", final=True)
        self._tail = ""
        if not text:
            return b""
        return unmask_response(text, self._map).encode("utf-8")
