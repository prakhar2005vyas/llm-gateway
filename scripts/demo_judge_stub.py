"""Demo judge stub — deterministic LLM-as-judge for runner demos.

Scores by answer length (longer = marginally more helpful, capped), fixed
tone: deterministic, so two identical runs produce identical numbers and a
--compare demo shows a clean zero-delta. NOT product code; tests use respx.
Runs on :9098.
"""
from __future__ import annotations

import json

import uvicorn
from fastapi import FastAPI, Request

app = FastAPI(title="demo-judge-stub")


@app.post("/v1/chat/completions")
async def judge(request: Request) -> dict:
    body = await request.json()
    answer = body["messages"][-1]["content"]
    helpfulness = min(9, 6 + len(answer) // 400)
    verdict = {"helpfulness": helpfulness, "tone": 8,
               "rationale": "deterministic demo verdict"}
    return {
        "id": "chatcmpl-judge-stub",
        "object": "chat.completion",
        "created": 1,
        "model": "demo-judge",
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": json.dumps(verdict)},
                     "finish_reason": "stop"}],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9098, log_level="warning")
