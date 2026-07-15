# LLM Gateway + Eval/Observability Platform — Technical Blueprint (SPEC.md)

> This is the **single source of truth** for scope. If a feature is not in "Locked Scope
> (BUILD)" below, it does not get built — it gets documented in `SCALING.md` or left out.
> The whole point of this project is a **tightly-scoped, fully-functional, well-documented
> core** — not a half-finished monolith with a dozen broken features.

---

## What this is (one sentence)

A transparent, OpenAI-compatible **proxy** that sits between an application and any LLM
provider and, while forwarding every request, **records, measures, caches, coalesces,
and guards** what passes through. Mental model: *Stripe/Cloudflare, but for LLM calls.*
Adopting it must require changing exactly **one line** (the base URL) in the client app —
if adoption needs a rewrite, nobody uses it.

---

## The core request lifecycle (the heart of the system)

```
Client app ──POST /v1/chat/completions──▶ GATEWAY
                                            1. Auth: validate gateway API key
                                            2. Rate limit: per-key counter (in-memory v1)
                                            3. PII mask: redact BEFORE embed/cache/forward
                                            4. Semantic cache lookup (on the MASKED prompt)
                                                 hit  → return cached, $0, DONE
                                            5. Coalescing: exact-match pending lock
                                                 duplicate in-flight → wait, share result
                                            6. Forward upstream (httpx, STREAMING)
                                                 on failure → OpenAI-compatible failover
                                            7. Stream tokens to client in real time
                                                 (byte-buffered SSE, UTF-8 safe)
                                                 AND buffer to reassemble full response
                                            8. HOT PATH ENDS — release client connection
                                            ─────────── cold path (deferred) ───────────
                                            9. BackgroundTask: token count, cost, Postgres
                                                 write, store embedding+response in cache,
                                                 Prometheus counters
                                           10. Threadpool/worker: embedding generation,
                                                 optional LLM-as-judge eval sampling
```

The **hot path / cold path split (step 8)** is the architectural foundation. Everything
else attaches to a lifecycle that must be correct before anything else is worth writing.

---

## LOCKED SCOPE — BUILD

These are implemented, tested, and demoable. Nothing here is optional.

1. **API Proxy** — OpenAI-compatible `/v1/chat/completions`. Acceptance test: the official
   `openai` Python SDK talks to the gateway **unmodified** and cannot tell the difference.
2. **Hot/Cold path decoupling**
   - Hot path (inline): auth, rate-limit, PII mask, cache lookup, coalescing, stream forward.
   - Cold path async I/O → **FastAPI `BackgroundTasks`** (Postgres write, cost calc, metrics).
   - Cold path CPU-bound (embedding gen, eval) → **threadpool executor** (`run_in_executor`).
   - **Critical distinction:** `BackgroundTasks` run on the same event loop. CPU-bound work
     in a BackgroundTask STILL blocks the loop — it must go to the threadpool. Do not conflate.
   - **Bounded queue + backpressure:** the deferred-work queue is bounded. Under sustained
     backpressure (slow DB), **shed/sample** cold-path work rather than growing unboundedly
     and OOMing. Rule: drop 5% of trace logs before you drop the live request path.
3. **Request Coalescing** — **exact-match** (hash of the masked prompt), **in-memory**,
   **atomic compare-and-set** on a pending map. NOT semantic coalescing (too slow — would
   compute cosine similarity over a live queue). NOT distributed. Only coalesce
   low-temperature (deterministic) requests — collapsing `temperature>0` creative calls into
   one identical answer is a correctness bug.
4. **OpenAI-compatible failover** — on upstream failure after N retries, reroute to an
   equivalent **OpenAI-compatible** provider (OpenAI → Fireworks → Together). Same schema =
   swap base URL + key. NOT cross-schema (Anthropic) — that needs a translation layer (DOCUMENT).
5. **Byte-buffered SSE streaming** — forward tokens live while reassembling for logging.
   - Buffer the raw **byte** stream; only yield on the `\n\n` event delimiter.
   - Handle BOTH fragmentation (one event split across packets) AND coalescing (multiple
     events in one read) — buffer, split-on-delimiter in a loop, retain remainder.
   - **UTF-8 safety:** never decode per-packet. A multibyte character (Devanagari, Tamil,
     Bengali, emoji) can split across packets → `UnicodeDecodeError`. Accumulate bytes,
     decode incrementally. This is a REQUIREMENT given the India target market — write a
     deliberate test sending a Devanagari response and asserting intact reassembly.
6. **PII masking** — redact PII (email, phone, credit card, etc.) **before** the prompt
   leaves the system, via regex or Microsoft Presidio.
   - **Reversible tokenization:** replace with uniquely-numbered placeholders (`<EMAIL_1>`),
     keep a per-request map, unmask on the response. The model must echo placeholders back
     (imperfect — handle the miss gracefully).
   - **Cross-cutting rule:** masking runs UPSTREAM of embedding/caching. Embed/cache the
     MASKED text, or the vector store becomes a PII database — defeating the feature.
7. **Chaos Testing matrix** — the proof layer, non-negotiable. Each injected failure proves
   one headline claim (see the matrix below). "Don't claim resilience — inject failure and
   screenshot the survival."

---

## EXCLUDED SCOPE — DOCUMENT ONLY (in SCALING.md, not code)

Design and write these up with real depth (they're strong interview talking points), but
do NOT implement them. Half-built versions are worse than well-documented designs.

- **GPU dynamic batching** — ONNX Runtime + CUDA EP, batched embedding inference. Excluded
  because it's hardware-dependent (breaks "runs on any grader's machine") and OOMs a 4GB
  RTX 3050 instantly under batching. Local embeddings MUST use CPU with graceful fallback.
- **Cross-schema failover** (OpenAI → Anthropic) — requires a full API translation layer
  (different message format, tool-calling schema, stop sequences). Its own mini-project.
- **Distributed Redis locks** for coalescing — TTL + lease expiry + holder-death recovery
  for cross-replica coalescing. v1 is single-process in-memory; document the Redis path.

---

## Tech stack (decided — do not re-litigate)

| Layer | Choice | Why |
|---|---|---|
| Gateway API | Python 3.11 + FastAPI (async) | Proxy is I/O-bound; async handles many in-flight requests |
| HTTP client | httpx (async, streaming) | `client.stream()` for SSE forwarding |
| Data store | Postgres (Neon-compatible) | Structured, aggregatable logs; JSONB for variable bodies |
| Cache / vectors | Redis (exact cache + rate-limit counters) + **pgvector** (semantic) | Keep stack small; pgvector over a separate vector DB for v1 |
| Embeddings | small local model (`all-MiniLM` via sentence-transformers), **CPU**, threadpool | No per-call cost; the cost-saver must not have a cost |
| Metrics | Prometheus (`/metrics`) + Grafana | Ops maturity signal; the dashboards are resume screenshots |
| Dashboard UI | React + Vite + TS + Tailwind | Reuse Shadow QA muscle memory |
| Eval | LLM-as-judge (sampled) + golden-dataset regression runner | Logic + prompts, no new infra |
| Packaging | Docker Compose | One `docker compose up`, runs on any machine |
| Load/chaos | locust or k6 + a Python chaos script | Produces the numbers and the resilience proofs |

**No Kafka, no Kubernetes, no message queue for v1.** Adding them signals over-engineering,
not maturity.

---

## Phased build order — DO NOT skip ahead. Each phase ends green and demoable.

### Phase 0 — The dumb pipe
Forward an OpenAI-format request to a real provider, non-streaming, response unchanged.
- **Done when:** the official `openai` SDK talks to the gateway unmodified and gets a valid completion.

### Phase 1 — Logging + cost (+ hot/cold seam)
Add the Postgres `Trace` model and a price table (model → $/1K in, $/1K out). Design the
hot/cold seam NOW — defer the write to a `BackgroundTask`. Unit-test the cost calculator
against hand-computed values.
- **Done when:** every call leaves an accurate, queryable Trace row; the DB write is off the hot path.
- **Metric:** cost per request, correct to the token.

### Phase 2 — Streaming (the hard one)
Switch to SSE forwarding: tokens to the client live + byte-buffered reassembly. Handle
fragmentation, coalescing, and UTF-8 boundaries.
- **Done when:** streaming feels instant AND a full Trace is logged AND a Devanagari response
  reassembles intact.
- **Metric:** time-to-first-token vs non-streamed.

### Phase 3 — Semantic caching
Embed-on-request (CPU, threadpool), pgvector similarity search, cache-hit short-circuit.
Mask BEFORE embed. Tune the threshold; test adversarial near-misses ("delete account" vs
"recover account" must NOT collide).
- **Done when:** near-duplicate prompts serve instantly at $0, with zero upstream call (assert it).
- **Metric:** cache-hit rate; cost reduction %.

### Phase 4 — Coalescing + rate limit + failover + resilience
Exact-match atomic pending lock; per-key rate limits (429); OpenAI-compatible failover;
timeouts + retries + graceful upstream-down handling.
- **Done when:** 500 identical concurrent requests fire exactly ONE upstream call; over-limit → 429;
  dead primary → automatic failover, no crash.
- **Metric:** upstream calls saved under duplicate load.

### Phase 5 — PII masking
Reversible placeholder tokenization, per-request map, unmask on response, upstream of cache.
- **Done when:** PII never leaves the gateway unmasked; the vector store contains no raw PII.
- **Metric:** PII detection/redaction on a labeled test set.

### Phase 6 — Eval harness
Online: sample X% of traffic, LLM-as-judge scoring. Offline: golden-dataset regression
comparing two prompt/model configs. (LLM-as-judge is noisy — state this honestly.)
- **Done when:** "config B scores 8% higher than config A on the golden set."
- **Metric:** quality score; regression delta between configs.

### Phase 7 — Dashboard + metrics + load/chaos + package
Grafana (cost over time, P50/P95/P99 latency, cache-hit rate, req/model), React trace
browser, load test, chaos suite, clean `docker compose up`.
- **Done when:** screenshots tell the story; `LOAD_TEST.md` has graphs; clean checkout boots.
- **Metric:** "sustained N req/s at P99 X ms with Y% cache hit rate."

---

## Chaos Testing matrix — each failure proves one resume claim

| Injected failure (how) | Proves the claim | The demo artifact |
|---|---|---|
| Kill upstream mid-stream (point at dead port / drop connection) | streaming error handling + failover | client gets clean error / reroutes; no hang, no crash |
| **Slow/freeze Postgres** (`docker pause` the DB container mid-load) | **hot/cold decoupling + bounded-queue backpressure** | **hot-path P99 stays flat while the log queue sheds** — the crown jewel |
| 500 identical requests in 1s (asyncio fan-out script) | request coalescing | exactly **1** upstream call fires, not 500 |
| Over-rate-limit burst | rate limiting | clean 429s, no degradation of within-limit traffic |

The frozen-Postgres test is the single best screenshot in the project: it's the visual
answer to "what happens when your database gets slow?" — the latency graph doesn't move.

---

## Concepts to understand before writing code

OpenAI-compatible API shape · Server-Sent Events (SSE) · async I/O & the event loop
(concurrency vs parallelism) · embeddings & cosine similarity · token-based pricing ·
P50/P95/P99 percentiles · LLM-as-judge and its noise · single-flight/request coalescing ·
atomic compare-and-set (NOT "ACID" — precise wording matters).
