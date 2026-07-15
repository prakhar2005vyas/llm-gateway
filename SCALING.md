# SCALING.md — Designed, not built

This file is the home for the **EXCLUDED SCOPE — DOCUMENT ONLY** items from
`SPEC.md`. These are deliberately *not implemented*: half-built versions would be
worse than well-reasoned designs, and each is a strong systems-design interview
talking point precisely because the judgment to *not* build it is the point.

> Status: **stubs**. Each section is fleshed out with real depth as the core
> phases land (the designs reference mechanisms — coalescing, embedding path,
> failover — that are built in Phases 3–4). Placeholder now so excluded-scope
> ideas have a documented home and never leak into the codebase.

---

## 1. GPU dynamic batching (embeddings)

**What it would be:** ONNX Runtime + CUDA execution provider, batching embedding
inference across concurrent requests for higher throughput.

**Why it's excluded:** hardware-dependent (breaks "runs on any grader's
machine"); batching OOMs a 4 GB RTX 3050 instantly. The built system runs
embeddings on **CPU via a threadpool** with graceful fallback and hardware
auto-detect.

_TODO (Phase 3+): batch-window design, latency/throughput tradeoff, VRAM math._

---

## 2. Cross-schema failover (OpenAI → Anthropic)

**What it would be:** failover across providers with *different* API schemas
(message format, tool-calling, stop sequences), via a translation layer.

**Why it's excluded:** it's its own mini-project. The built system does
**OpenAI-compatible** failover only (OpenAI → Fireworks → Together): same schema,
so failover is just swap base URL + key.

_TODO (Phase 4+): translation-layer sketch, the schema-diff surface, why it's a
correctness minefield._

---

## 3. Distributed Redis locks for coalescing

**What it would be:** cross-replica request coalescing via Redis locks with TTL +
lease expiry + holder-death recovery.

**Why it's excluded:** v1 coalescing is **single-process, in-memory, atomic
compare-and-set** on a pending map. Distributed locking adds correctness hazards
(holder death, clock skew) that aren't worth it for a single-process demo.

_TODO (Phase 4+): lease/TTL design, holder-death recovery, why in-memory is
correct for v1's single-process deployment._
