"""Demo stub upstream that behaves like a placeholder-respecting model: it
echoes any <LABEL_N> placeholders found in the prompt back in its reply.
Used to demo the Phase 5 PII round trip against the live stack. NOT product
code; never used by tests (they use respx)."""
from __future__ import annotations

import re
import time

import uvicorn
from fastapi import FastAPI, Request

app = FastAPI(title="demo-echo-stub")


@app.post("/v1/chat/completions")
async def chat(request: Request) -> dict:
    body = await request.json()
    prompt = " ".join(
        m.get("content", "") for m in body.get("messages", []) if isinstance(m, dict)
    )
    placeholders = re.findall(r"<[A-Z]+_\d+>", prompt)
    reply = (
        "Understood. I will use " + ", ".join(placeholders) + " as provided."
        if placeholders
        else "No placeholders were present in the prompt."
    )
    return {
        "id": "chatcmpl-echo-stub",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "gpt-4o-mini"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 40, "completion_tokens": 15, "total_tokens": 55},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9099, log_level="warning")
