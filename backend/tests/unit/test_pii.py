"""PII redactor tests — detection, reversible tokenization, graceful unmask.

Every masking assertion checks BOTH directions: raw value absent from the
masked text, and the map able to restore it exactly.
"""
from __future__ import annotations

import pytest

from app.pii import contains_pii, mask_prompt, mask_text, unmask_response


# ---------------------------------------------------------------------------
# Detection + masking
# ---------------------------------------------------------------------------

class TestEmails:
    def test_single_email(self):
        masked, pii = mask_text("contact alice@example.com for details")
        assert masked == "contact <EMAIL_1> for details"
        assert pii == {"<EMAIL_1>": "alice@example.com"}

    def test_multiple_distinct_emails_numbered_in_order(self):
        masked, pii = mask_text("cc bob@x.co and alice@y.org")
        assert masked == "cc <EMAIL_1> and <EMAIL_2>"
        assert pii == {"<EMAIL_1>": "bob@x.co", "<EMAIL_2>": "alice@y.org"}

    def test_repeated_email_shares_one_placeholder(self):
        masked, pii = mask_text("a@x.com wrote to a@x.com")
        assert masked == "<EMAIL_1> wrote to <EMAIL_1>"
        assert len(pii) == 1

    def test_plus_addressing_and_subdomains(self):
        masked, pii = mask_text("send to dev+test@mail.corp.example.co.in now")
        assert pii == {"<EMAIL_1>": "dev+test@mail.corp.example.co.in"}
        assert "dev+test" not in masked


class TestCards:
    def test_luhn_valid_card_with_spaces(self):
        masked, pii = mask_text("pay with 4111 1111 1111 1111 please")
        assert masked == "pay with <CARD_1> please"
        assert pii == {"<CARD_1>": "4111 1111 1111 1111"}

    def test_luhn_valid_card_with_dashes(self):
        masked, pii = mask_text("card: 5500-0000-0000-0004")
        assert pii == {"<CARD_1>": "5500-0000-0000-0004"}

    def test_luhn_INVALID_16_digits_is_not_a_card(self):
        # An order id that merely LOOKS like a card must not be redacted as
        # one — Luhn is the gate. 16 digits, checksum fails:
        masked, pii = mask_text("order 4111111111111112 shipped")
        assert "<CARD" not in masked
        assert pii == {}  # (16 digits also exceeds the phone range)

    def test_amex_15_digits(self):
        masked, pii = mask_text("amex 3782 822463 10005")
        assert pii == {"<CARD_1>": "3782 822463 10005"}


class TestPhones:
    @pytest.mark.parametrize(
        "phone",
        [
            "+91 98765 43210",      # India, country code
            "9876543210",           # India, bare 10-digit
            "(555) 123-4567",       # US, parens
            "555-123-4567",         # US, dashed
            "+1 555 123 4567",      # US, country code
        ],
    )
    def test_common_formats(self, phone):
        masked, pii = mask_text(f"call me at {phone} tomorrow")
        assert masked == "call me at <PHONE_1> tomorrow"
        assert pii == {"<PHONE_1>": phone}

    @pytest.mark.parametrize(
        "not_phone",
        [
            "meeting in 2026",              # year
            "costs 12,345 rupees",          # short number
            "error code 404",               # tiny number
            "id 12345678",                  # 8 digits — below range
        ],
    )
    def test_non_phones_left_alone(self, not_phone):
        masked, pii = mask_text(not_phone)
        assert masked == not_phone
        assert pii == {}


class TestSSN:
    """A-1 audit fix: dashed US SSNs are detected; undelimited 9-digit runs
    and Indian PAN are documented non-goals."""

    def test_dashed_ssn_masked_and_reversible(self):
        masked, pii = mask_text("my ssn is 123-45-6789, thanks")
        assert masked == "my ssn is <SSN_1>, thanks"
        assert pii == {"<SSN_1>": "123-45-6789"}
        assert unmask_response("confirmed for <SSN_1>", pii) == (
            "confirmed for 123-45-6789"
        )

    def test_undelimited_9_digits_is_not_an_ssn(self):
        # 9 bare digits: order/tracking-id territory, deliberately untouched
        # (also below the 10-digit phone floor).
        masked, pii = mask_text("tracking 123456789 shipped")
        assert masked == "tracking 123456789 shipped"
        assert pii == {}

    def test_ssn_and_phone_coexist(self):
        masked, pii = mask_text("ssn 123-45-6789, cell 9876543210")
        assert masked == "ssn <SSN_1>, cell <PHONE_1>"
        assert pii["<SSN_1>"] == "123-45-6789"
        assert pii["<PHONE_1>"] == "9876543210"

    def test_pan_is_a_documented_limitation_not_detected(self):
        # Indian PAN (alphanumeric) — ACCEPTED limitation per module
        # docstring; this test pins the documented behavior so a future
        # detector change is a conscious decision.
        masked, pii = mask_text("my PAN is ABCDE1234F")
        assert masked == "my PAN is ABCDE1234F"
        assert pii == {}

    def test_contains_pii_sees_ssn(self):
        assert contains_pii("ssn: 123-45-6789")


class TestMixedAndPriority:
    def test_all_three_types_in_one_text(self):
        text = (
            "I'm alice@example.com, phone +91 98765 43210, "
            "card 4111 1111 1111 1111."
        )
        masked, pii = mask_text(text)
        assert masked == "I'm <EMAIL_1>, phone <PHONE_1>, card <CARD_1>."
        assert pii["<EMAIL_1>"] == "alice@example.com"
        assert pii["<PHONE_1>"] == "+91 98765 43210"
        assert pii["<CARD_1>"] == "4111 1111 1111 1111"

    def test_phone_like_digits_inside_email_not_double_masked(self):
        # EMAIL wins overlap: the 10-digit local part must not ALSO become
        # a <PHONE_n> (that would corrupt the placeholder structure).
        masked, pii = mask_text("UPI id 9876543210@upi is mine")
        assert masked == "UPI id <EMAIL_1> is mine"
        assert pii == {"<EMAIL_1>": "9876543210@upi"}

    def test_no_pii_returns_text_unchanged_empty_map(self):
        text = "what is the capital of France?"
        masked, pii = mask_text(text)
        assert masked == text and pii == {}

    def test_mask_prompt_returns_masked_string_only(self):
        out = mask_prompt("mail a@b.com")
        assert isinstance(out, str) and "<EMAIL_1>" in out

    def test_contains_pii(self):
        assert contains_pii("call 9876543210")
        assert contains_pii("mail a@b.io")
        assert not contains_pii("hello world 2026")


# ---------------------------------------------------------------------------
# Unmasking — including hostile/mangled model output
# ---------------------------------------------------------------------------

class TestUnmask:
    def test_perfect_round_trip(self):
        text = "I'm alice@example.com, call +91 98765 43210, card 4111 1111 1111 1111"
        masked, pii = mask_text(text)
        assert unmask_response(masked, pii) == text

    def test_model_echoes_placeholder_multiple_times(self):
        _, pii = mask_text("mail a@x.com")
        reply = "I'll email <EMAIL_1> and CC <EMAIL_1> as requested."
        assert unmask_response(reply, pii) == (
            "I'll email a@x.com and CC a@x.com as requested."
        )

    def test_model_DROPPED_placeholder_is_graceful(self):
        # The model answered without echoing the placeholder: nothing to
        # restore, nothing crashes, text passes through.
        _, pii = mask_text("mail a@x.com and b@y.org")
        reply = "Sure, I'll email <EMAIL_1>. (second contact omitted)"
        assert unmask_response(reply, pii) == (
            "Sure, I'll email a@x.com. (second contact omitted)"
        )

    @pytest.mark.parametrize(
        "mangled,expected",
        [
            ("<email_1>", "a@x.com"),        # lowercased
            ("<Email_1>", "a@x.com"),        # title-cased
            ("< EMAIL_1 >", "a@x.com"),      # padded
            ("<EMAIL 1>", "a@x.com"),        # underscore dropped
            ("<EMAIL-1>", "a@x.com"),        # hyphenated
        ],
    )
    def test_model_MANGLED_placeholder_still_restored(self, mangled, expected):
        _, pii = mask_text("mail a@x.com")
        assert unmask_response(f"ok: {mangled}", pii) == f"ok: {expected}"

    def test_hallucinated_unknown_placeholder_left_verbatim(self):
        # <EMAIL_9> has no map entry — inventing a value would be worse than
        # leaving the model's artifact visible.
        _, pii = mask_text("mail a@x.com")
        reply = "contacts: <EMAIL_1>, <EMAIL_9>"
        assert unmask_response(reply, pii) == "contacts: a@x.com, <EMAIL_9>"

    def test_empty_inputs_never_raise(self):
        assert unmask_response("", {"<EMAIL_1>": "a@x.com"}) == ""
        assert unmask_response("plain text", {}) == "plain text"
        assert unmask_response("", {}) == ""

    def test_malformed_map_entry_skipped_not_crashed(self):
        # Defensive: a corrupted map key that isn't <LABEL_N> shaped.
        assert unmask_response("text <weird>", {"<weird": "x"}) == "text <weird>"


# ---------------------------------------------------------------------------
# The collision guards (why cache/coalescing must skip PII prompts)
# ---------------------------------------------------------------------------

class TestCollisionGuards:
    def test_two_users_distinct_pii_mask_to_identical_text(self):
        # THE hazard the guards exist for, demonstrated:
        a, _ = mask_text("my email is alice@corp.com")
        b, _ = mask_text("my email is bob@rival.io")
        assert a == b == "my email is <EMAIL_1>"

    def test_cache_rejects_pii_bearing_request(self):
        from app.cache import is_cacheable_request

        body = {
            "model": "gpt-4o-mini",
            "temperature": 0,
            "messages": [{"role": "user", "content": "mail alice@corp.com the report"}],
        }
        assert not is_cacheable_request(body)
        body["messages"][0]["content"] = "mail the report to my colleague"
        assert is_cacheable_request(body)

    def test_coalescing_rejects_pii_bearing_request(self):
        from app.coalesce import is_coalesceable

        body = {
            "model": "gpt-4o-mini",
            "temperature": 0,
            "messages": [{"role": "user", "content": "my card is 4111 1111 1111 1111"}],
        }
        assert not is_coalesceable(body)
        body["messages"][0]["content"] = "what are card fraud protections?"
        assert is_coalesceable(body)
