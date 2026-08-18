import httpx
import pytest
from openai import OpenAI
import os
import uuid

# Expected to run against a live stack (e.g. docker compose up --build)
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
API_KEY = os.getenv("GATEWAY_API_KEY", "my_secure_local_password")

@pytest.fixture
def openai_client():
    # We point the OpenAI client to the Gateway's /v1 endpoint
    return OpenAI(base_url=f"{GATEWAY_URL}/v1", api_key=API_KEY)

def test_health_endpoint():
    """Verify the /health endpoint is responsive and returns ok."""
    response = httpx.get(f"{GATEWAY_URL}/health")
    assert response.status_code == 200
    assert response.json().get("status") == "ok"

def test_metrics_endpoint():
    """Verify the /metrics endpoint exposes Prometheus metrics."""
    response = httpx.get(f"{GATEWAY_URL}/metrics")
    assert response.status_code == 200
    assert "process_cpu_seconds_total" in response.text
    assert "gateway_requests_total" in response.text

def test_chat_completion_non_streaming(openai_client):
    """Verify that a basic non-streaming chat completion passes through the gateway."""
    unique_content = f"hello {uuid.uuid4()}"
    response = openai_client.chat.completions.create(
        model="openai/gpt-oss-120b", # The model string might be ignored/mapped by upstream
        messages=[{"role": "user", "content": unique_content}]
    )
    
    # Assert we get a valid response structure
    assert response.choices[0].message.content is not None
    assert len(response.choices[0].message.content) > 0

def test_chat_completion_streaming(openai_client):
    """Verify that a streaming chat completion works correctly."""
    unique_content = f"hello streaming {uuid.uuid4()}"
    response_stream = openai_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": unique_content}],
        stream=True
    )
    
    chunks = []
    raw_chunks = []
    for chunk in response_stream:
        raw_chunks.append(str(chunk))
        if chunk.choices and chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)
            
    full_response = "".join(chunks)
    assert len(full_response) > 0, f"Expected non-empty stream response, but got {len(full_response)} characters. Raw chunks received: {raw_chunks}"
