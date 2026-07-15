# Career Context — The "Why" (CAREER_CONTEXT.md)

> This file exists so every technical decision is anchored to the reason this project is
> being built. Read it alongside `SPEC.md` and `CLAUDE.md`.

---

## The goal

I am a **B.Tech student** building this project to **bridge traditional SDE and AI
Engineering** and secure a **top-tier internship** (SDE or AI Engineer) at MNCs and
startups in India and abroad, based on 2026 market conditions.

Why this project specifically: it sits at the intersection that's hiring — **AI-native
systems engineering**. It's not "another RAG chatbot" (oversaturated) and not pure ML
research (not what interns get hired for). Building the *infrastructure around* LLMs —
gateways, observability, caching, resilience — demonstrates real systems fundamentals
*and* AI fluency at once. It's a focused version of a real, venture-backed product category
(Langfuse / Helicone / Portkey).

## The predecessor project (context)

I already shipped **Shadow QA** — an autonomous visual-QA agent (Playwright + a multimodal
model + FastAPI + React + Docker), with an SSRF guard, step/time budgets, retries, a test
suite, and graceful crash handling. This LLM Gateway is the deliberate next step: there I
*consumed* model endpoints with retries and budgets; here I *am* the infrastructure in front
of them. Same reliability instincts, one level deeper into systems. The two projects together
tell one story: **"I build reliable AI systems and the infrastructure they run on."**

## The resume strategy — every decision ties to an outcome or a story

**Rule: no bullet point without a measurable outcome or a systems-design interview story.**

Every technical choice must map to one of:

### A) A measurable outcome (numbers on the resume)
- **Inference cost reduction %** — from semantic caching + coalescing.
- **Cache-hit rate** — the % of requests served at $0.
- **P50 / P95 / P99 latency** — hot-path latency, and how it stays flat under DB stress.
- **Throughput** — sustained req/s under load test (with P99 attached).
- **Quality regression delta** — eval harness comparing prompt/model configs.

Target resume bullet shape:
> *Built an OpenAI-compatible LLM gateway with real-time streaming, semantic caching,
> request coalescing, and an eval harness. Cut inference cost ~X% via similarity caching,
> held hot-path P99 at Y ms while the database was frozen under load, and caught prompt
> regressions before deploy. Sustained N req/s. [live demo] [GitHub]*

### B) A systems-design interview story
Each built feature is rehearsal for a specific question:
- **Hot/cold decoupling + bounded queue** → *"What happens to memory and the event loop if
  1,000 users stream concurrently and Postgres suddenly gets slow?"* → Answer: streaming
  passes through without blocking; logging is deferred to a bounded background queue that
  **sheds** under backpressure rather than OOMing. Proven live by the `docker pause` Postgres
  chaos test — the latency graph doesn't move.
- **Request coalescing** → *"50 users ask the same uncached question at once — do you pay 50×?"*
  → Answer: atomic compare-and-set pending lock; 49 wait and share the one result. Proven by
  the 500-identical-requests chaos test firing exactly one upstream call.
- **Byte-buffered SSE + UTF-8** → *"How do you parse a stream where one JSON event splits
  across TCP packets, or a Devanagari character splits across two packets?"* → Answer: buffer
  bytes, split on delimiter, decode incrementally. Relevant to the India market (multibyte scripts).
- **OpenAI-compatible failover** → *"The primary provider has an outage — does the app crash?"*
  → Answer: retries then reroute to an equivalent provider; fault tolerance + API abstraction.
- **PII masking upstream of cache** → *"How do you keep proprietary/PII data from leaking to a
  third-party model — and from leaking into your own vector store?"* → Answer: reversible
  masking before embed/forward. Ties to DPDP Act (India) / GDPR — a 2026 enterprise concern.

## What "done" looks like for the career goal

- A **live demo** URL + a **clean public GitHub repo** with a README that leads with numbers
  and screenshots (Grafana dashboards, the frozen-DB latency graph).
- A `SCALING.md` that discusses the DOCUMENT-ONLY features (GPU batching, cross-schema
  failover, distributed locks) with real depth — so I can speak to sophistication I chose
  *not* to build, which signals judgment (the thing juniors lack).
- 2–3 rigorous projects total, coherent theme, depth over count. Shadow QA + this gateway
  is the spine of that portfolio.

## The discipline reminder (why the scope is locked)

Recruiters spend ~20 seconds per resume and can smell a project that's 40% done across ten
fronts. A fully-working core with two well-chosen differentiators beats an ambitious
half-built system. **The scope in `SPEC.md` is locked precisely to protect this.** Ship the
core, document the rest, attach numbers, move on.
