#!/usr/bin/env python
"""Async load generator for the LLM Gateway (Phase 7).

Fires a configurable burst of chat completions at a running gateway with a
50/50 traffic mix (tunable via --hot-ratio):

* HOT half — the SAME prompt with temperature 0 on every request. The first
  miss populates the semantic cache; concurrent duplicates coalesce into one
  upstream flight; everything after the population is served from pgvector at
  $0 with zero upstream calls. This is the cache/coalescing stress lane.
* UNIQUE half — a fresh randomized prompt per request, no temperature.
  Deliberately uncacheable and uncoalesceable: every one must reach the
  upstream provider. This is the pass-through lane.

The split is verified, not assumed: the gateway's x-gateway-cache response
header (hit/miss) is tallied per lane and printed in the report alongside
client-side P50/P95/P99 latency, success rate, and throughput.

Prerequisites:
* Gateway running (docker compose up) with a reachable upstream — a real
  provider key in .env, or the free local stub:
      python scripts/demo_stub_upstream.py   # then UPSTREAM_BASE_URL at it
* RATE_LIMIT_REQUESTS raised well above the burst size (default is 60/min —
  a load test against defaults measures the rate limiter, not the gateway).
  429s are counted separately so a misconfigured run is obvious.

Usage:
    python scripts/load_test.py [--requests 500] [--concurrency 50]
        [--hot-ratio 0.5] [--gateway-url http://localhost:8000]
        [--model gpt-4o-mini] [--api-key ...]

Watch the run live in Grafana (http://localhost:3000, dashboard
"LLM Gateway") — see LOAD_TEST.md for which panel proves which claim.
"""
from __future__ import annotations

import argparse
import asyncio
import random
import string
import time

import httpx

HOT_PROMPT = "Explain what request coalescing is in one short paragraph."


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    return sorted_vals[int(q * (len(sorted_vals) - 1))]


def make_body(i: int, hot: bool, model: str) -> dict:
    if hot:
        # Explicit temperature 0: the gateway only caches/coalesces
        # deterministic requests (omitted temperature = 1.0 upstream = bypass).
        return {
            "model": model,
            "temperature": 0,
            "messages": [{"role": "user", "content": HOT_PROMPT}],
        }
    salt = "".join(random.choices(string.ascii_lowercase, k=8))
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": f"Unique question #{i} [{salt}]: say 'ok'."}
        ],
    }


async def fire_one(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, i: int, hot: bool, model: str
) -> dict:
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.post("/v1/chat/completions", json=make_body(i, hot, model))
            return {
                "hot": hot,
                "status": r.status_code,
                "ok": r.status_code == 200,
                "latency": time.perf_counter() - t0,
                "cache": r.headers.get("x-gateway-cache", "-"),
            }
        except httpx.HTTPError as exc:
            return {
                "hot": hot,
                "status": None,
                "ok": False,
                "latency": time.perf_counter() - t0,
                "cache": "-",
                "error": f"{type(exc).__name__}",
            }


def report_lane(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    lat = sorted(r["latency"] for r in rows)
    ok = sum(1 for r in rows if r["ok"])
    hits = sum(1 for r in rows if r["cache"] == "hit")
    n429 = sum(1 for r in rows if r["status"] == 429)
    errs = sum(1 for r in rows if r["status"] is None)
    print(
        f"{name:<8} {len(rows):>5} {ok:>5} {n429:>5} {errs:>5} {hits:>5}  "
        f"{percentile(lat, 0.50)*1000:>7.0f} {percentile(lat, 0.95)*1000:>7.0f} "
        f"{percentile(lat, 0.99)*1000:>7.0f} {lat[-1]*1000:>7.0f}"
    )


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gateway-url", default="http://localhost:8000")
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--hot-ratio", type=float, default=0.5,
                    help="fraction of requests using the identical hot prompt")
    ap.add_argument("--model", default="llama-3.1-8b-instant",
                    help="forwarded verbatim — must exist at the configured upstream")
    ap.add_argument("--api-key", default="", help="gateway bearer token if auth is on")
    args = ap.parse_args()

    # Deterministic interleave (not a coin flip): exact lane sizes, and hot
    # requests spread across the whole run so coalescing sees real overlap.
    jobs = [(i, (i % 100) < args.hot_ratio * 100) for i in range(args.requests)]
    random.shuffle(jobs)

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    limits = httpx.Limits(
        max_connections=args.concurrency, max_keepalive_connections=args.concurrency
    )
    sem = asyncio.Semaphore(args.concurrency)

    print(
        f"firing {args.requests} requests at {args.gateway_url} "
        f"(concurrency={args.concurrency}, hot-ratio={args.hot_ratio:.0%}, "
        f"model={args.model})"
    )
    t0 = time.perf_counter()
    async with httpx.AsyncClient(
        base_url=args.gateway_url, headers=headers, limits=limits, timeout=180
    ) as client:
        results = await asyncio.gather(
            *(fire_one(client, sem, i, hot, args.model) for i, hot in jobs)
        )
    wall = time.perf_counter() - t0

    hot_rows = [r for r in results if r["hot"]]
    uniq_rows = [r for r in results if not r["hot"]]
    ok = sum(1 for r in results if r["ok"])

    print(f"\n{'lane':<8} {'n':>5} {'ok':>5} {'429':>5} {'err':>5} {'hit':>5}  "
          f"{'p50ms':>7} {'p95ms':>7} {'p99ms':>7} {'maxms':>7}")
    print("-" * 78)
    report_lane("hot", hot_rows)
    report_lane("unique", uniq_rows)
    report_lane("overall", list(results))
    print("-" * 78)
    print(
        f"wall time {wall:.1f}s  |  throughput {len(results)/wall:.1f} req/s  |  "
        f"success {ok}/{len(results)} ({ok/len(results):.1%})"
    )
    if any(r["status"] == 429 for r in results):
        print("NOTE: 429s present — raise RATE_LIMIT_REQUESTS to load-test the "
              "gateway rather than its rate limiter.")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
