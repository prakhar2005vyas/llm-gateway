"""Demo stub upstream — an OpenAI-shaped /v1/chat/completions that always
answers with a fixed completion and usage {prompt:100, completion:50}.

Purpose: let the full gateway → Postgres path run on a machine with no
provider API key (same ethos as Shadow QA's MOCK_VLM: everything real except
the paid call). With gpt-4o-mini's seeded prices the expected trace cost is
exactly 100/1000*0.00015 + 50/1000*0.0006 = $0.00004500 — a known answer the
demo can assert against.

Run:  python scripts/demo_stub_upstream.py   (listens on :9099)
Then in .env:  UPSTREAM_BASE_URL=http://host.docker.internal:9099/v1

NOT part of the product. Never used by unit tests (they use respx).
"""
from __future__ import annotations

import time

import uvicorn
from fastapi import FastAPI, Request

app = FastAPI(title="demo-stub-upstream")


@app.post("/v1/chat/completions")
async def chat(request: Request) -> dict:
    body = await request.json()
    return {
        "id": "chatcmpl-demo-stub",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "gpt-4o-mini"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from the demo stub upstream!",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9099, log_level="warning")
