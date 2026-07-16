"""Embedding layer tests.

These use the REAL all-MiniLM-L6-v2 model (~90MB, auto-downloaded to the HF
cache on first run, CPU inference ~10ms/prompt). Real model, still hermetic:
no service, no API key, no network after the one-time download. Session-scoped
warmup pays the load once for the whole file.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from app import embeddings
from app.config import get_settings


@pytest.fixture(scope="module", autouse=True)
def _module_isolation():
    # One model load for this file; drop it afterwards so other test files
    # never see (or wait on) a warm model they didn't ask for.
    embeddings.reset_for_tests()
    yield
    embeddings.reset_for_tests()


async def test_returns_vector_of_configured_dimension():
    vec = await embeddings.get_embedding("what is the capital of France?")
    assert isinstance(vec, list)
    assert len(vec) == get_settings().embedding_dim == 384
    assert all(isinstance(x, float) for x in vec)


async def test_vectors_are_normalized_unit_length():
    # normalize_embeddings=True is what makes cosine ≡ dot product; pin it.
    vec = await embeddings.get_embedding("hello world")
    norm = sum(x * x for x in vec) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)


async def test_semantically_similar_beats_dissimilar():
    # Sanity-check the model actually embeds MEANING (guards against e.g. a
    # wrong-model env var pointing at a garbage checkpoint).
    a = await embeddings.get_embedding("How do I reset my password?")
    b = await embeddings.get_embedding("I forgot my password, help me log in")
    c = await embeddings.get_embedding("What's a good recipe for biryani?")
    sim_ab = sum(x * y for x, y in zip(a, b))
    sim_ac = sum(x * y for x, y in zip(a, c))
    assert sim_ab > sim_ac
    assert sim_ab > 0.5


async def test_event_loop_is_not_blocked_during_inference():
    """THE concurrency proof: a 5ms heartbeat must keep ticking on the event
    loop while model load + inference run. If inference ran inline, the loop
    would freeze and the heartbeat count would collapse to ~zero."""
    embeddings.reset_for_tests()  # force THIS test to pay the model load too

    ticks = 0
    stop = asyncio.Event()

    async def heartbeat():
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0.005)

    hb = asyncio.create_task(heartbeat())
    t0 = time.perf_counter()
    # Load (seconds) + 20 inferences — all of it must stay off the loop.
    for i in range(20):
        await embeddings.get_embedding(f"prompt number {i} about various topics")
    elapsed = time.perf_counter() - t0
    stop.set()
    await hb

    # The loop must have been free for the overwhelming majority of the
    # elapsed time. Expected ticks at perfect freedom ≈ elapsed/5ms; demand
    # at least a third of that (generous margin for scheduling jitter).
    expected_if_free = elapsed / 0.005
    assert ticks >= expected_if_free / 3, (
        f"event loop starved: {ticks} heartbeats in {elapsed:.2f}s "
        f"(≈{expected_if_free:.0f} expected if free) — inference is "
        "blocking the loop"
    )


async def test_concurrent_first_calls_load_model_exactly_once(monkeypatch):
    """Single-flight load: N concurrent cold callers must trigger ONE load."""
    embeddings.reset_for_tests()
    real_load = embeddings._load_model_sync
    loads = 0

    def counting_load():
        nonlocal loads
        loads += 1
        return real_load()

    monkeypatch.setattr(embeddings, "_load_model_sync", counting_load)

    vecs = await asyncio.gather(*(embeddings.get_embedding(f"p{i}") for i in range(8)))
    assert loads == 1
    assert all(len(v) == 384 for v in vecs)


async def test_dimension_mismatch_fails_loudly(monkeypatch):
    embeddings.reset_for_tests()
    monkeypatch.setenv("EMBEDDING_DIM", "512")  # wrong for MiniLM on purpose
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="384.*EMBEDDING_DIM=512|EMBEDDING_DIM=512.*384"):
        await embeddings.get_embedding("this must not silently mis-dimension")
    get_settings.cache_clear()
    embeddings.reset_for_tests()
