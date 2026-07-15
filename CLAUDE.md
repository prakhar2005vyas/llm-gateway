# LLM Gateway — System Instructions (CLAUDE.md)

> These are standing instructions for every session on this project. Read `SPEC.md` and
> `CAREER_CONTEXT.md` alongside this file before starting any work.

---

## Prime directive: no feature creep

- **`SPEC.md` is the single source of truth for scope.** If a feature is not in `SPEC.md`'s
  "LOCKED SCOPE — BUILD" list, do not build it. Do not suggest new architecture, new layers,
  or new "wouldn't it be cool if" features.
- The "EXCLUDED SCOPE — DOCUMENT ONLY" items (GPU dynamic batching, cross-schema failover,
  distributed Redis locks) go in `SCALING.md` as design write-ups **only**. Never implement them.
- If I ask for something outside the spec, **push back first**: name that it's out of scope,
  explain the cost, and ask me to confirm a deliberate scope change before writing any code.
- The failure mode we are actively avoiding is a half-finished monolith with a dozen broken
  features. A tightly-scoped, fully-functional, well-documented core beats it every time.
  When in doubt, **cut, don't add.**

## Hardware constraints — never assume infinite resources

- Target machine: **16 GB system RAM, RTX 3050 (~4 GB VRAM)**. An OS, IDE, Docker containers,
  Postgres/Redis, and an embedding model all run simultaneously — margin is near zero.
- **Local embeddings run on CPU via a threadpool executor.** Do not put embedding inference
  on the GPU in the built system. GPU batching would OOM 4 GB VRAM instantly and is
  DOCUMENT-ONLY.
- Any embedding/inference path must have **graceful CPU fallback** and auto-detect hardware —
  never hard-require a GPU. The repo must `docker compose up` cleanly on a GPU-less machine
  (e.g. an M1 Mac or a plain cloud box).
- Respect the **bounded-queue + backpressure** rule from `SPEC.md`: under load, shed/sample
  cold-path work rather than growing memory unboundedly and OOMing. Never assume the deferred
  queue can grow forever.

## Continuity with Shadow QA (my previous project) — carry these practices forward

- **Timeout + limited retries (exponential backoff)** on every external call (LLM providers,
  embedding APIs if any). On repeated failure, degrade gracefully — mark inconclusive / return
  a clean error — never crash the process.
- **Never silently swallow errors.** Log them, and surface a clear degraded state rather than
  pretending a step succeeded.
- **Honest scoping.** State plainly what is BUILT vs DESIGNED-ONLY, what is validated vs
  assumed. Don't overclaim. (This is the same discipline as the Shadow QA MOCK vs real-model
  caveat — it reads as maturity, not weakness.)
- **Phased delivery.** Work through `SPEC.md`'s Phase 0→7 IN ORDER. After each phase: run that
  phase's tests, show results and how to see it running, and wait for explicit go-ahead before
  the next phase. Do not bundle phases.
- **Testing bar is non-negotiable.** Unit tests for pure logic (cost calc, PII redaction, SSE
  buffer parsing, coalescing lock). Integration tests for the real request lifecycle. A chaos
  suite that proves each resilience claim (see `SPEC.md` matrix). Run tests before calling a
  phase done; fix failures before moving on.
- **Never commit secrets.** `.env` is gitignored; `.env.example` ships placeholders only for
  every variable. (Reinforced by the Shadow QA incident where a live key nearly landed in
  `.env.example` — always verify before staging.)
- **Everything config-driven** — provider URLs, model IDs, keys, thresholds, budgets are env
  vars, never hardcoded.
- **Verify, don't assume.** Check the actual installed library behavior (e.g. exception
  hierarchies, httpx streaming APIs) before relying on it.

## Coding standards

- Python 3.11, FastAPI, Pydantic v2, async throughout the hot path. Match the surrounding
  code's style, naming, and comment density.
- Nothing blocking on the event loop — sync/CPU-bound work goes to a threadpool or a
  `BackgroundTask` per the hot/cold split in `SPEC.md`.
- Precise terminology: "atomic compare-and-set," not "ACID," for the coalescing lock.

## Session workflow

1. Read `SPEC.md`, `CLAUDE.md`, `CAREER_CONTEXT.md`.
2. Identify the current phase (check what's built vs the phase list).
3. Do that phase's work, test it, show it running.
4. Tie the result back to a metric or an interview story (see `CAREER_CONTEXT.md`).
5. Stop and wait for go-ahead before the next phase.
