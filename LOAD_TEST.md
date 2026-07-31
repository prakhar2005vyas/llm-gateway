# LOAD_TEST.md — driving load and watching the gateway hold

Phase 7's load harness: `scripts/load_test.py` generates a controlled burst of
mixed traffic against a running gateway while the bundled Prometheus + Grafana
stack visualizes what the gateway does under it. This is the runbook.

## Prerequisites

1. **A running stack:**
   ```bash
   docker compose up --build
   ```
2. **A reachable upstream.** Either a real provider key in `.env`
   (`UPSTREAM_API_KEY`), or — for free, offline runs — the local stub:
   ```bash
   python scripts/demo_stub_upstream.py
   # then point UPSTREAM_BASE_URL at it in .env and restart the gateway
   ```
3. **Rate limit raised for the burst.** The default is 60 req/min; a load test
   against defaults measures the rate limiter, not the gateway:
   ```bash
   # .env
   RATE_LIMIT_REQUESTS=100000
   ```
   (429s are counted separately in the report, so a forgotten limit is
   obvious, not silently misleading.)

## Running

```bash
pip install httpx   # the only dependency
python scripts/load_test.py                       # 500 requests, 50 workers
python scripts/load_test.py --requests 2000 --concurrency 100
python scripts/load_test.py --hot-ratio 0.8       # cache-heavy mix
```

## Methodology

Every run fires `--requests` total requests through `--concurrency` async
workers (httpx + asyncio, bounded by a semaphore), split into two lanes:

| Lane | Share | Shape | What it exercises |
|---|---|---|---|
| **hot** | `--hot-ratio` (default 50%) | the SAME prompt, `temperature: 0` | semantic cache (first miss populates pgvector; the rest hit at $0) and coalescing (concurrent duplicates share ONE upstream flight) |
| **unique** | the rest | fresh randomized prompt, no temperature | pure pass-through: deliberately uncacheable and uncoalesceable, every request must reach the upstream |

The lanes are interleaved deterministically and shuffled, so hot requests
overlap in flight (coalescing needs concurrency, not just repetition).

The split is **verified, not assumed**: the gateway's `x-gateway-cache`
response header is tallied per lane. The report prints, per lane and overall:
request count, successes, 429s, transport errors, cache hits, and
client-side P50/P95/P99/max latency, plus wall time, throughput, and success
rate for the whole run.

## Watching it in Grafana — which panel proves which claim

Run the load test while the dashboard is open
(`http://localhost:3000`, login `admin`/`admin`, dashboard **LLM Gateway**).
This is the intended way to verify the claims documented in `SCALING.md`:

| Panel | Claim it verifies |
|---|---|
| Requests/s by model + cache_hit | the hot lane flips to `cache_hit=true` after the first population — served at $0 with zero upstream calls (`SCALING.md` §1) |
| Request latency P95/P99 | cache hits pull the aggregate P95 far below the upstream's own latency; the unique lane sets the ceiling |
| TTFT P95 | streaming responsiveness under load (run streaming traffic to populate) |
| Retries + failovers | flat at 0 in a healthy run; watch it climb if you kill the primary upstream mid-run (chaos matrix row 1) |

For the crown-jewel demo (`SPEC.md` chaos matrix): start a load run, then
`docker pause <postgres-container>` mid-burst. Hot-path latency panels stay
flat while the gateway logs `TRACE WRITE FAILED` sheds — the hot/cold seam
holding under a frozen database. (`docker unpause` recovers.) The same claim
is proven hermetically in CI by `backend/tests/chaos/test_frozen_db.py`.

## Honest caveats

* Client-side latency includes the load generator's own scheduling overhead;
  the server-side truth is the Grafana histogram. Quote the server-side
  numbers, use the client-side ones for sanity.
* Loopback on one machine: generator, gateway, and DB share CPU. Numbers are
  demo-scale evidence of *behavior* (flat P99, cache hits, one coalesced
  flight), not capacity benchmarks.
* Against a real provider the unique lane costs real tokens and its latency
  variance dominates everything — use the stub upstream for repeatable runs.

## Recording results

Paste the report block plus a dashboard screenshot here per notable run:

```
(results go here — date, config, report output, screenshot link)
```
