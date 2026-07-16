"""Local CPU embeddings via sentence-transformers (Phase 3).

Hardware rules (CLAUDE.md, non-negotiable):
* Inference runs on **CPU, explicitly** (`device="cpu"`). Never the GPU — the
  target machine has ~4GB VRAM with everything else fighting for it; GPU
  batching is DOCUMENT-ONLY in SCALING.md. Explicit beats auto-detect here:
  auto would happily grab a GPU that another process needs.
* Inference is synchronous, CPU-bound work → it goes through
  `run_in_executor` (threadpool), NEVER inline in a coroutine and NEVER in a
  BackgroundTask (same event loop! SPEC.md's critical distinction).

Model loading is itself seconds of blocking I/O+CPU, so it also happens inside
the executor, lazily, behind an asyncio.Lock (single-flight: concurrent first
callers wait for one load instead of racing three copies of a ~90MB model into
16GB RAM). Callers who want to pay the load cost at boot call `warmup()`.

PII note: callers embed MASKED text only (pii.mask_prompt) — see the
cross-cutting rule in SPEC §6. This module doesn't enforce it (it can't know);
the cache pipeline does.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from .config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model: "SentenceTransformer | None" = None
_load_lock: asyncio.Lock | None = None


def _get_load_lock() -> asyncio.Lock:
    # Created lazily so the lock binds to the running loop (module import
    # happens before any loop exists; tests create/destroy loops per test).
    global _load_lock
    if _load_lock is None:
        _load_lock = asyncio.Lock()
    return _load_lock


def _load_model_sync() -> "SentenceTransformer":
    """Blocking model load — only ever runs inside the executor."""
    from sentence_transformers import SentenceTransformer  # heavy import, deferred

    settings = get_settings()
    logger.info(
        "loading embedding model %r on cpu (first call pays this once)...",
        settings.embedding_model_id,
    )
    model = SentenceTransformer(settings.embedding_model_id, device="cpu")
    dim = model.get_sentence_embedding_dimension()
    if dim != settings.embedding_dim:
        # Fail LOUDLY at load, not at insert: a mismatched vector would be
        # rejected by the vector(N) column anyway, with a far worse message.
        raise RuntimeError(
            f"embedding model {settings.embedding_model_id!r} produces {dim}-d "
            f"vectors but EMBEDDING_DIM={settings.embedding_dim} — fix the env "
            "or re-dimension the semantic_cache column"
        )
    logger.info("embedding model ready (%d-d, cpu)", dim)
    return model


async def _ensure_model() -> "SentenceTransformer":
    global _model
    if _model is not None:
        return _model
    async with _get_load_lock():
        if _model is None:  # double-check: another waiter may have loaded it
            loop = asyncio.get_running_loop()
            _model = await loop.run_in_executor(None, _load_model_sync)
    return _model


def _encode_sync(model: "SentenceTransformer", text: str) -> list[float]:
    """Blocking inference — only ever runs inside the executor."""
    # normalize_embeddings=True: unit vectors make cosine similarity a plain
    # dot product and keep pgvector distances well-conditioned.
    vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return vec.tolist()


async def get_embedding(text: str) -> list[float]:
    """Embed `text` on the CPU threadpool; returns a 384-d unit vector.

    The event loop is never blocked: both the (lazy, once) model load and
    every inference run in the default executor.
    """
    model = await _ensure_model()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _encode_sync, model, text)


async def warmup() -> None:
    """Optionally pay the model-load cost at startup instead of first request."""
    await _ensure_model()


def reset_for_tests() -> None:
    """Drop the cached model/lock (test isolation only)."""
    global _model, _load_lock
    _model = None
    _load_lock = None
