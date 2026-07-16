"""PII masking seam — placeholder until Phase 5 implements real redaction.

Why this exists NOW, two phases early: SPEC §6's cross-cutting rule says
masking runs UPSTREAM of embedding/caching — embed and cache the MASKED text,
or the vector store becomes a PII database and the masking feature is defeated
retroactively. The cache layer is being built in Phase 3, so the seam must
exist in Phase 3: every prompt MUST flow through `mask_prompt()` before it is
embedded, searched, or stored. When Phase 5 lands real redaction inside this
function, the cache pipeline is already correct without touching it.

Current behavior: identity (returns the prompt verbatim). That means until
Phase 5 the cache DOES contain raw prompt text — a known, documented interim
state on the way to the real thing, not a hidden one.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def mask_prompt(prompt: str) -> str:
    """Redact PII from a prompt before it is embedded, searched, or cached.

    Phase 3 placeholder: identity. Phase 5 replaces the body with reversible
    placeholder tokenization (<EMAIL_1>, <PHONE_1>, ...) per SPEC §6 — the
    signature and the call sites stay exactly as they are.
    """
    return prompt
