#!/usr/bin/env python
"""Golden-dataset regression runner (Phase 6, offline half).

Fires the golden dataset through a LIVE gateway, forces LLM-as-judge evals
for every case (x-gateway-force-eval header -- bypasses EVAL_SAMPLE_RATE),
waits for the async Eval rows to land, and prints a scored report with a
PASS/FAIL verdict against per-case baselines. Save a run with --out, then
--compare a later run against it: that is the "config B vs config A" number
SPEC's done-when asks for.

Usage (from backend/):
    python scripts/run_evals.py [--gateway-url http://localhost:8000]
        [--model gpt-4o-mini] [--dataset tests/fixtures/golden_dataset.jsonl]
        [--wait 90] [--out results.json] [--compare previous.json]

Requirements:
* The gateway must be reachable at --gateway-url with a real upstream and a
  configured judge (EVAL_BASE_URL/EVAL_API_KEY on the GATEWAY process).
* DATABASE_URL in THIS process must point at the same database the gateway
  writes (compose users: publish 5432 or run inside the container).
* Prompts are sent WITHOUT temperature -- deliberately uncacheable and
  uncoalesceable, so every run hits the real upstream and stays
  eval-eligible (a cached answer is never re-judged).

Honesty note (SPEC): LLM-as-judge is noisy -- single-digit percent deltas
between runs are within judge noise; treat only consistent, sizeable drops
as regressions.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Runnable as `python scripts/run_evals.py` from backend/ -- put backend/ on
# the path so `app.*` imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.db import dispose_db, session  # noqa: E402
from app.models import Eval, Trace  # noqa: E402
from app.pii import mask_texts  # noqa: E402

RAW_PII_MARKERS = ("@example.com", "98765")  # dataset PII the trace must not hold


def load_dataset(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    if not cases:
        sys.exit(f"dataset {path} is empty")
    return cases


async def fire_case(client: httpx.AsyncClient, case: dict, model: str) -> dict:
    t0 = time.perf_counter()
    try:
        r = await client.post(
            "/v1/chat/completions",
            json={"model": model,
                  "messages": [{"role": "user", "content": case["prompt"]}]},
            headers={"x-gateway-force-eval": "1"},
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        ok = r.status_code == 200
        return {"case": case, "ok": ok, "status": r.status_code,
                "latency_ms": latency_ms,
                "error": None if ok else r.text[:200]}
    except httpx.HTTPError as exc:
        return {"case": case, "ok": False, "status": None,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "error": f"{type(exc).__name__}: {exc}"}


async def find_trace_and_eval(masked_prompt: str, since: datetime) -> tuple[Trace | None, Eval | None]:
    async with session() as s:
        recent = (
            await s.execute(
                select(Trace).order_by(Trace.created_at.desc()).limit(200)
            )
        ).scalars().all()
        # Time-filter in Python with normalized (naive-UTC) stamps: SQLite
        # returns naive datetimes, Postgres returns aware -- comparing them in
        # SQL or raw Python raises/mismatches depending on backend.
        match = next(
            (
                t for t in recent
                if t.created_at.replace(tzinfo=None) >= since
                and isinstance(t.request_body, dict)
                and (t.request_body.get("messages") or [{}])[0].get("content") == masked_prompt
            ),
            None,
        )
        if match is None:
            return None, None
        ev = (
            await s.execute(
                select(Eval).where(Eval.trace_id == match.id)
                .order_by(Eval.created_at.desc()).limit(1)
            )
        ).scalars().first()
        return match, ev


async def wait_for_evals(cases: list[dict], since: datetime, timeout: float) -> dict[str, dict]:
    """Poll until every case's Eval reaches a terminal state (or timeout)."""
    masked = {c["id"]: mask_texts([c["prompt"]])[0][0] for c in cases}
    results: dict[str, dict] = {}
    deadline = time.monotonic() + timeout
    pending = set(masked)

    while pending and time.monotonic() < deadline:
        for cid in list(pending):
            trace, ev = await find_trace_and_eval(masked[cid], since)
            if ev is not None and ev.status != "pending":
                results[cid] = {"trace": trace, "eval": ev}
                pending.discard(cid)
        if pending:
            await asyncio.sleep(1.0)

    for cid in pending:
        trace, ev = await find_trace_and_eval(masked[cid], since)
        results[cid] = {"trace": trace, "eval": ev}  # whatever state it's in
    return results


def check_trace_masked(trace: Trace | None) -> tuple[bool, str]:
    if trace is None:
        return False, "trace not found"
    blob = json.dumps(trace.request_body) + json.dumps(trace.response_body or {})
    for marker in RAW_PII_MARKERS:
        if marker in blob:
            return False, f"RAW PII IN TRACE: {marker!r}"
    return True, "trace masked"


def print_report(fired: list[dict], db: dict[str, dict], compare_path: Path | None) -> tuple[bool, dict]:
    rows, help_scores, tone_scores = [], [], []
    all_pass = True

    for f in fired:
        case = f["case"]
        cid = case["id"]
        entry = db.get(cid, {})
        ev: Eval | None = entry.get("eval")
        verdicts: list[str] = []

        h = float(ev.helpfulness_score) if ev and ev.helpfulness_score is not None else None
        t = float(ev.tone_score) if ev and ev.tone_score is not None else None
        if not f["ok"]:
            verdicts.append(f"request failed ({f['error']})")
        if ev is None:
            verdicts.append("no eval row")
        elif ev.status != "completed":
            verdicts.append(f"eval {ev.status}: {ev.error_message or ''}"[:60])
        else:
            help_scores.append(h)
            tone_scores.append(t)
            if h < case.get("min_helpfulness", 0):
                verdicts.append(f"helpfulness {h} < baseline {case['min_helpfulness']}")
            if t < case.get("min_tone", 0):
                verdicts.append(f"tone {t} < baseline {case['min_tone']}")
        if "trace_masked" in case.get("checks", []):
            ok, msg = check_trace_masked(entry.get("trace"))
            if not ok:
                verdicts.append(msg)

        passed = not verdicts
        all_pass &= passed
        rows.append((cid, case["category"], h, t, f["latency_ms"],
                     "PASS" if passed else "FAIL: " + "; ".join(verdicts)))

    print(f"\n{'case':<18} {'category':<14} {'help':>5} {'tone':>5} {'ms':>6}  verdict")
    print("-" * 92)
    for cid, cat, h, t, ms, verdict in rows:
        print(f"{cid:<18} {cat:<14} {h if h is not None else '--':>5} "
              f"{t if t is not None else '--':>5} {ms:>6}  {verdict}")

    avg_h = sum(help_scores) / len(help_scores) if help_scores else 0.0
    avg_t = sum(tone_scores) / len(tone_scores) if tone_scores else 0.0
    print("-" * 92)
    print(f"averages over {len(help_scores)} completed eval(s): "
          f"helpfulness={avg_h:.2f}  tone={avg_t:.2f}")

    summary = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "overall": {"helpfulness": avg_h, "tone": avg_t},
        "cases": {cid: {"helpfulness": h, "tone": t}
                  for cid, _, h, t, _, _ in rows},
    }

    if compare_path is not None:
        base = json.loads(compare_path.read_text(encoding="utf-8"))
        bh, bt = base["overall"]["helpfulness"], base["overall"]["tone"]
        dh = ((avg_h - bh) / bh * 100) if bh else 0.0
        dt = ((avg_t - bt) / bt * 100) if bt else 0.0
        print(f"\nvs baseline {compare_path.name} "
              f"(ran {base.get('ran_at', '?')[:19]}):")
        print(f"  helpfulness: {bh:.2f} -> {avg_h:.2f}  ({dh:+.1f}%)")
        print(f"  tone:        {bt:.2f} -> {avg_t:.2f}  ({dt:+.1f}%)")
        print("  (LLM-as-judge is noisy -- single-digit % deltas are within noise)")

    print(f"\nRESULT: {'PASS' if all_pass else 'FAIL'} "
          f"({sum(1 for r in rows if r[5] == 'PASS')}/{len(rows)} cases)")
    return all_pass, summary


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gateway-url", default="http://localhost:8000")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--dataset", type=Path,
                    default=Path(__file__).parent.parent / "tests/fixtures/golden_dataset.jsonl")
    ap.add_argument("--wait", type=float, default=90.0,
                    help="seconds to wait for async Eval rows")
    ap.add_argument("--pace", type=float, default=0.5,
                    help="seconds between requests: forced evals are still "
                         "subject to the gateway's EVAL_MAX_CONCURRENT shed "
                         "cap; pacing keeps a burst from shedding samples")
    ap.add_argument("--api-key", default="", help="gateway bearer token if auth is on")
    ap.add_argument("--out", type=Path, help="write run summary JSON here")
    ap.add_argument("--compare", type=Path, help="prior --out file to diff against")
    args = ap.parse_args()

    cases = load_dataset(args.dataset)
    print(f"golden dataset: {len(cases)} case(s) -> {args.gateway_url} "
          f"(model={args.model}, forced evals)")

    since = datetime.now(timezone.utc).replace(tzinfo=None)  # naive-UTC, matches storage
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    async with httpx.AsyncClient(base_url=args.gateway_url, timeout=120,
                                 headers=headers) as client:
        fired = []
        for i, case in enumerate(cases):
            if i and args.pace:
                await asyncio.sleep(args.pace)
            fired.append(await fire_case(client, case, args.model))

    ok_count = sum(1 for f in fired if f["ok"])
    print(f"fired {len(fired)} request(s), {ok_count} returned 200; "
          f"waiting up to {args.wait:.0f}s for async evals...")

    db_results = await wait_for_evals(cases, since, args.wait)
    all_pass, summary = print_report(fired, db_results, args.compare)

    if args.out:
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"summary written to {args.out}")

    await dispose_db()
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
