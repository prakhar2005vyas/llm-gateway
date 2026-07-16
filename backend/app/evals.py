"""Online eval harness — sampled LLM-as-judge scoring (Phase 6).

Position in the lifecycle: the LAST cold-path stage. The client responded
long ago; the trace row is committed; THEN a sampled fraction of successful
requests gets a fire-and-forget judge task. Three isolation guarantees:

* Never blocks the loop: the judge call is async httpx I/O in its own task;
  score parsing is trivial. Nothing here runs on the hot path.
* Never delays the trace: enqueue happens strictly AFTER the trace commit,
  and the eval writes its own row in its own session.
* Never piles up: at most EVAL_MAX_CONCURRENT tasks in flight. Beyond that,
  samples are SHED with a log line (SPEC bounded-queue rule — dropping an
  eval is free; letting a slow judge endpoint accumulate tasks is an OOM).

Eligibility: outcome ok, real upstream answers only — cache hits and
coalesced followers re-serve an already-evaluable answer; judging them again
would just multiply noise and cost.

PII: the bodies given to the judge are the MASKED copies (record_trace only
ever holds masked text after Phase 5) — the judge endpoint is an external
model and gets the same PII treatment as the primary provider.

Honesty (SPEC): LLM-as-judge is NOISY. The raw verdict is stored verbatim
alongside the extracted scores; a failed parse is status='failed' with the
reason, never a silently invented score.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import uuid
from datetime import datetime, timezone

import httpx

from .config import get_settings
from .db import session
from .models import Eval

logger = logging.getLogger(__name__)

# Strong refs to in-flight tasks: prevents GC of running tasks AND is the
# concurrency gauge for the shed decision.
_tasks: set[asyncio.Task] = set()

_JUDGE_SYSTEM = (
    "You are a strict response-quality judge. Given a user conversation and "
    "an assistant answer, score the ANSWER on two axes, each an integer 1-10:\n"
    "- helpfulness: does it correctly, completely address the request?\n"
    "- tone: is it clear, appropriately professional, well-calibrated?\n"
    'Reply with ONLY this JSON: {"helpfulness": <1-10>, "tone": <1-10>, '
    '"rationale": "<one sentence>"}'
)


def _eligible(outcome: str, cache_hit: bool, coalesced: bool, content: str | None) -> bool:
    return (
        outcome == "ok"
        and not cache_hit
        and not coalesced
        and isinstance(content, str)
        and bool(content.strip())
    )


def maybe_enqueue_eval(
    *,
    trace_id: uuid.UUID,
    request_body: dict | None,
    response_content: str | None,
    outcome: str,
    cache_hit: bool = False,
    coalesced: bool = False,
    force: bool = False,
) -> bool:
    """Sampling gate + fire-and-forget enqueue. Returns True if enqueued.

    Called from record_trace AFTER the trace commit. Never raises.

    `force=True` (the golden-dataset runner, via the x-gateway-force-eval
    header) bypasses the RATE and the coin flip — never the eligibility
    rules or the shed cap. Single-tenant tool: any client can burn eval
    tokens with the header; acceptable per scope, revisit if multi-tenant.
    """
    try:
        settings = get_settings()
        rate = settings.eval_sample_rate
        if not force and rate <= 0.0:
            return False
        if not _eligible(outcome, cache_hit, coalesced, response_content):
            return False
        if not force and random.random() >= rate:
            return False
        if len(_tasks) >= settings.eval_max_concurrent:
            # SHED, loudly: bounded cold-path work beats unbounded pileup.
            logger.warning(
                "eval SHED for trace %s: %d eval task(s) already in flight",
                trace_id, len(_tasks),
            )
            return False

        task = asyncio.create_task(
            _run_eval(trace_id, request_body, response_content)  # type: ignore[arg-type]
        )
        _tasks.add(task)
        task.add_done_callback(_on_task_done)
        return True
    except Exception as exc:  # noqa: BLE001 — sampling must never hurt tracing
        logger.error("eval enqueue failed (non-fatal): %s: %s", type(exc).__name__, exc)
        return False


def _on_task_done(task: asyncio.Task) -> None:
    _tasks.discard(task)
    if not task.cancelled() and (exc := task.exception()):
        # _run_eval catches everything itself; this is the belt-and-braces
        # backstop so no exception is ever silently swallowed by the Task.
        logger.error("eval task died unexpectedly: %s: %s", type(exc).__name__, exc)


async def _run_eval(
    trace_id: uuid.UUID, request_body: dict | None, response_content: str
) -> None:
    """Create the Eval row, call the judge, persist the verdict. Never raises."""
    eval_id = uuid.uuid4()
    settings = get_settings()
    try:
        async with session() as s:
            s.add(Eval(id=eval_id, trace_id=trace_id, status="pending"))

        if not settings.eval_base_url:
            await _finish(eval_id, status="skipped",
                          error_message="no evaluator configured (EVAL_BASE_URL empty)")
            return

        verdict = await _call_judge(request_body, response_content)
        helpfulness, tone = _extract_scores(verdict)
        await _finish(
            eval_id,
            status="completed",
            helpfulness=helpfulness,
            tone=tone,
            evaluator_model=settings.eval_model_id,
            raw_verdict=verdict,
        )
        logger.info("eval %s completed (trace %s): helpfulness=%s tone=%s",
                    eval_id, trace_id, helpfulness, tone)
    except Exception as exc:  # noqa: BLE001 — cold-path boundary, same as tracing
        logger.error("EVAL FAILED (trace %s, non-fatal): %s: %s",
                     trace_id, type(exc).__name__, exc)
        try:
            await _finish(eval_id, status="failed",
                          error_message=f"{type(exc).__name__}: {exc}"[:500])
        except Exception:  # noqa: BLE001 — DB down too; already logged above
            pass


async def _call_judge(request_body: dict | None, response_content: str) -> dict:
    """One judge completion, timeout + limited retries (CLAUDE.md rule)."""
    settings = get_settings()
    conversation = json.dumps(
        (request_body or {}).get("messages", []), ensure_ascii=False
    )
    judge_request = {
        "model": settings.eval_model_id,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content":
                f"CONVERSATION:\n{conversation}\n\nASSISTANT ANSWER:\n{response_content}"},
        ],
    }
    url = f"{settings.eval_base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {settings.eval_api_key}"}
    max_tries = settings.eval_max_retries + 1
    last_exc: Exception | None = None

    async with httpx.AsyncClient(timeout=settings.eval_timeout_seconds) as client:
        for attempt in range(max_tries):
            try:
                resp = await client.post(url, json=judge_request, headers=headers)
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]
                return _parse_verdict(text)
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                last_exc = exc
                if attempt < max_tries - 1:
                    await asyncio.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"judge call failed after {max_tries} attempt(s): {last_exc}")


def _parse_verdict(text: str) -> dict:
    """Judge output → dict, tolerating markdown fences and prose padding."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", candidate)
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if match is None:
        raise ValueError(f"no JSON object in judge output (len={len(text)})")
    verdict = json.loads(match.group())
    if not isinstance(verdict, dict):
        raise ValueError("judge output is not a JSON object")
    return verdict


def _extract_scores(verdict: dict) -> tuple[float, float]:
    scores = []
    for key in ("helpfulness", "tone"):
        value = verdict.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"judge verdict missing numeric {key!r}")
        scores.append(min(10.0, max(1.0, float(value))))  # clamp to contract
    return scores[0], scores[1]


async def _finish(
    eval_id: uuid.UUID,
    *,
    status: str,
    helpfulness: float | None = None,
    tone: float | None = None,
    evaluator_model: str | None = None,
    raw_verdict: dict | None = None,
    error_message: str | None = None,
) -> None:
    async with session() as s:
        row = await s.get(Eval, eval_id)
        if row is None:
            return
        row.status = status
        row.helpfulness_score = helpfulness
        row.tone_score = tone
        row.evaluator_model = evaluator_model
        row.raw_verdict = raw_verdict
        row.error_message = error_message
        row.completed_at = datetime.now(timezone.utc)


async def drain_for_tests() -> None:
    """Await all in-flight eval tasks (test determinism only)."""
    if _tasks:
        await asyncio.gather(*list(_tasks), return_exceptions=True)
