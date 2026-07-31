# SCALING.md — Designed, not built

This file is the home for the **EXCLUDED SCOPE — DOCUMENT ONLY** items from
`SPEC.md`. These are deliberately *not implemented*: half-built versions would be
worse than well-reasoned designs, and each is a strong systems-design interview
talking point precisely because the judgment to *not* build it is the point.

Each section contrasts what the built system actually does (with pointers into
the code) against the scaled-up design, and explains why v1 deliberately stops
short.

---

## 1. GPU dynamic batching (embeddings)

### What v1 builds instead: CPU embeddings feeding a pgvector semantic cache

The cache's job is to make a redundant LLM call impossible: if a semantically
equivalent prompt was answered before, the stored response is served with zero
upstream calls and cost recorded as a literal $0.

The built pipeline (`app/cache.py`, `app/embeddings.py`):

1. The request is serialized (every message, role-tagged — the whole
   conversation is the cache key) and **PII-masked before anything is embedded
   or stored**, so the vector store never becomes a PII database.
2. The masked text is embedded by a local `all-MiniLM-L6-v2` model on **CPU,
   explicitly** (`device="cpu"`), inside `run_in_executor` — inference is
   synchronous CPU-bound work and must never run inline on the event loop, nor
   in a `BackgroundTask` (BackgroundTasks share that same loop). The ~90 MB
   model loads lazily, once, behind a single-flight `asyncio.Lock`, so N
   concurrent cold callers trigger one load, not N copies racing into 16 GB of
   system RAM.
3. Vectors are L2-normalized at encode time; cosine similarity then reduces to
   a plain dot product, and pgvector's `<=>` cosine-distance query stays
   well-conditioned. The nearest same-model row is a hit iff similarity ≥ 0.90
   — tuned empirically: paraphrases score ≳ 0.92, the adversarial near-misses
   that must NOT collide ("delete my account" vs "recover my account") ≈ 0.71.
4. Correctness guards on serving a hit: deterministic requests only (explicit
   `temperature` ≤ 0.2 — a frozen answer must never substitute a creative
   call), exact model match, only complete `finish_reason="stop"` responses are
   stored, and PII-bearing prompts bypass the cache entirely (two users'
   different secrets can mask to identical placeholder text; a shared cache
   entry would hand one user's answer to the other).

### The excluded design: batched GPU inference

The scaled version is ONNX Runtime with the CUDA execution provider plus a
micro-batching window: hold each embedding request for up to ~5–10 ms, batch
whatever accumulated, run one forward pass. Throughput rises roughly linearly
with batch size until VRAM or kernel occupancy saturates; the price is a fixed
latency floor equal to the window on every cache lookup — paid on the HOT
path, where the entire point of the cache is to be faster than the upstream
call.

The VRAM math kills it on the target machine (RTX 3050, 4 GB, shared with the
desktop). MiniLM-L6's weights are trivial (~23 M params, ~45 MB fp16), but
attention activations scale with batch × heads × seq²: at batch 256 and
sequence length 256, the attention score tensors alone are roughly
256 × 12 heads × 256² × 4 B ≈ 800 MB *per layer* (×6 layers) before any other
activations — and dynamic batch shapes push the allocator's high-water mark
higher still. A batching embedder OOMs a 4 GB card almost immediately, and any
hard GPU requirement breaks the "clean `docker compose up` on any grader's
machine" constraint. CPU MiniLM at ~10 ms/prompt is simply fast enough at demo
scale, so the GPU path stays on paper.

---

## 2. Cross-schema failover (OpenAI → Anthropic)

### What v1 builds instead: an OpenAI-compatible provider chain

`app/upstream.py` maintains an ordered chain — the primary, then an optional
secondary of the same schema family (OpenAI → Fireworks → Together class).
Same wire format means failover is literally "swap base URL + API key":

- Each provider gets its **full retry budget** (per-request timeout +
  exponential backoff, 0.5 s doubling to an 8 s cap) before the chain
  advances.
- The chain advances only on **transport exhaustion** (DNS/connect/read
  failures after all retries) or a **final 5xx** — i.e. the provider itself is
  down or broken.
- A **4xx never advances the chain**: it is the client's error (bad model id,
  bad key, malformed body). Replaying it at another provider would just fail
  again and mask the original provider's error message.
- Streaming failover applies to the **connection phase only**. Once the first
  byte has been relayed to the client, neither retry nor provider switch is
  possible — replaying tokens into a half-consumed SSE stream would corrupt
  it. A mid-stream death instead degrades loudly (`StreamResult.error`, trace
  outcome `inconclusive`).

The chaos suite proves the claim live (`tests/chaos/test_chaos.py`): with the
primary pointed at a dead port, the client still receives a clean 200 served
by the secondary — the reroute is an implementation detail, loudly logged and
counted in `gateway_failovers_total`, never a client-visible error.

### The excluded design: a schema translation layer

Anthropic failover requires a bidirectional adapter across every axis of the
API contract: message shape (`system` as a top-level field vs a message role),
tool calling (`tools` + `tool_calls` vs `input_schema` + `tool_use` /
`tool_result` content blocks), stop semantics (`stop` vs `stop_sequences`,
with different finish-reason vocabularies), the streaming event grammar
(OpenAI's uniform delta chunks vs Anthropic's typed event stream —
`message_start`, `content_block_delta`, `message_delta`, ...), and usage
accounting (`prompt_tokens`/`completion_tokens` vs
`input_tokens`/`output_tokens`) — and the translated stream must still satisfy
the gateway's own byte-level SSE reassembly and PII-unmasking invariants.
Every axis is a correctness minefield precisely because clients hold the
OpenAI contract: a 95%-faithful translation is a lying proxy. That adapter is
its own mini-project, so it stays a design.

---

## 3. Distributed Redis locks for coalescing

### What v1 builds instead: in-process atomic compare-and-set

Coalescing (`app/coalesce.py`) collapses N concurrent identical requests into
one upstream flight, shielding the provider from thundering herds. The entire
concurrency argument is one line:

```python
existing = self._pending.setdefault(key, new_future)  # <- the CAS
```

`dict.setdefault` executes atomically on the event loop — there is no `await`
between "is a flight already pending for this key?" and "then I own it", so no
interleaving is possible and no lock is needed. (Precise terminology: this is
an **atomic compare-and-set**, not "ACID" — there is no transaction anywhere
near it.) The CAS winner (leader) makes the upstream call; losers (followers)
await the leader's future and share its result — or its exception: a failed
flight lands every follower in its own clean-502 error path. Details that keep
it correct:

- The key is SHA-256 of (model + **masked** prompt) — raw prompt text never
  feeds derived artifacts, even in-memory keys.
- Only deterministic requests (explicit `temperature` ≤ 0.2) coalesce;
  collapsing N creative calls into one identical answer is a correctness bug.
- PII-bearing requests never coalesce: masking erases the difference between
  two users' distinct secrets, and sharing the flight would leak one user's
  answer to the other.
- The leader's flight runs in its own task behind `asyncio.shield`, so a
  leader whose client disconnects does not cancel the flight out from under
  its followers.

The chaos suite demonstrates the headline number: 500 identical concurrent
requests produce exactly **one** upstream call, with the real token spend
recorded once on the leader's trace and $0 on all 499 followers.

This is single-process by construction — the pending map is process memory.
Run two replicas and each coalesces only its own traffic: still correct, just
less effective (the herd splits across replicas before collapsing).

### The excluded design: cross-replica single-flight via Redis

The standard shape: `SET coalesce:<key> <owner-id> NX PX <lease-ms>` as the
distributed CAS; the winner calls upstream and publishes the result (pub/sub,
or a result key with its own TTL); losers subscribe and wait. Every additional
moving part exists to answer one question the in-memory version never has to
ask: *what if the lock holder dies mid-flight?* The lease TTL bounds how long
a dead leader can block the key. Lease renewal (a watchdog extending `PX`
while the flight is alive) prevents a slow-but-healthy flight from losing its
lease and letting a second leader duplicate the upstream call — but renewal
reintroduces exactly the split-brain risk the TTL was meant to bound, now
coupled to clock behavior and network partitions. Followers additionally need
a timeout-and-promote path for when no result ever arrives. None of that
hazard budget buys anything for a single-process deployment where
`dict.setdefault` is already a race-free CAS with zero failure modes — so the
Redis design stays on paper, and the pending map stays in memory.
