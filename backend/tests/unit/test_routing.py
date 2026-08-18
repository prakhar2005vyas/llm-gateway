import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.upstream import (
    providers, _rewrite_model, _get_routes, StreamResult,
    forward_chat_completion, forward_stream, UpstreamError,
    Provider
)
from app.config import get_settings

@pytest.fixture(autouse=True)
def clear_route_cache():
    _get_routes.cache_clear()

def test_default_empty_routes():
    # 11a. Default/empty model_routes_json produces identical providers()
    # and _rewrite_model() output to current behavior
    settings = get_settings()
    settings.upstream_model_id = ""
    settings.failover_base_url = ""
    
    # Test rewrite
    body = {"model": "my-model"}
    rewritten = _rewrite_model(body, "my-model")
    assert rewritten["model"] == "my-model"
    
    # Test providers
    chain = providers("my-model")
    assert len(chain) == 1
    assert chain[0].name == "primary"
    
    settings.failover_base_url = "http://failover"
    chain_with_failover = providers("my-model")
    assert len(chain_with_failover) == 2
    assert chain_with_failover[0].name == "primary"
    assert chain_with_failover[1].name == "failover"

def test_populated_route():
    # 11b. A populated route returns only that route's provider, no
    # failover chain, and uses "model_id" when set.
    settings = get_settings()
    settings.failover_base_url = "http://failover"
    settings.upstream_model_id = "default-upstream-model"
    settings.model_routes_json = json.dumps({
        "routed-model": {
            "base_url": "http://routed",
            "api_key": "routed-key",
            "model_id": "actual-model",
            "provider_label": "routed_cloud"
        }
    })
    
    body = {"model": "routed-model"}
    rewritten = _rewrite_model(body, "routed-model")
    assert rewritten["model"] == "actual-model"
    
    chain = providers("routed-model")
    assert len(chain) == 1
    assert chain[0].name == "routed_cloud"
    assert chain[0].base_url == "http://routed"
    assert chain[0].api_key == "routed-key"

def test_malformed_routes():
    # 11e. Malformed MODEL_ROUTES_JSON raises a clear error
    settings = get_settings()
    settings.model_routes_json = "{bad-json"
    with pytest.raises(ValueError, match="malformed MODEL_ROUTES_JSON"):
        _get_routes()

def test_route_missing_provider_label_raises():
    settings = get_settings()
    settings.model_routes_json = json.dumps({
        "routed-model": {
            "base_url": "http://routed"
        }
    })
    with pytest.raises(ValueError, match="missing required 'provider_label'"):
        _get_routes()

@pytest.mark.asyncio
@patch("app.upstream._forward_via")
async def test_non_streaming_provider_propagation(mock_forward):
    # 11c. A successful non-streaming request through a routed provider
    settings = get_settings()
    settings.model_routes_json = json.dumps({
        "routed-model": {
            "base_url": "http://routed",
            "api_key": "routed-key",
            "provider_label": "my_route"
        }
    })
    
    mock_forward.return_value = (200, {"id": "1"})
    
    status, payload, provider_name, _ = await forward_chat_completion({"model": "routed-model"})
    assert status == 200
    assert provider_name == "my_route"
    assert mock_forward.call_count == 1
    passed_provider = mock_forward.call_args[0][0]
    assert passed_provider.name == "my_route"

@pytest.mark.asyncio
@patch("app.upstream.httpx.AsyncClient")
async def test_streaming_provider_propagation(mock_client_cls):
    # 11d. A successful streaming request sets result.provider correctly
    settings = get_settings()
    settings.model_routes_json = json.dumps({
        "routed-model": {
            "base_url": "http://routed",
            "provider_label": "my_stream_route"
        }
    })
    
    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    async def fake_aiter():
        yield b"data: {}\n\n"
    mock_resp.aiter_bytes = fake_aiter
    
    mock_client = MagicMock()
    mock_client.stream.return_value.__aenter__.return_value = mock_resp
    mock_client_cls.return_value.__aenter__.return_value = mock_client
    
    result = StreamResult()
    gen = forward_stream({"model": "routed-model"}, result)
    
    await anext(gen)
    
    assert result.status_code == 200
    assert result.provider == "my_stream_route"

@pytest.mark.asyncio
@patch("app.upstream._forward_via")
async def test_non_streaming_failover_mislabeling(mock_forward):
    # 11f. Non-streaming failover mislabeling test
    settings = get_settings()
    settings.failover_base_url = "http://failover"
    
    # Primary raises UpstreamError, Failover succeeds
    mock_forward.side_effect = [
        UpstreamError("primary down"),
        (200, {"id": "failover-resp"})
    ]
    
    status, payload, provider_name, _ = await forward_chat_completion({"model": "my-model"})
    assert status == 200
    assert provider_name == "failover"

@pytest.mark.asyncio
@patch("app.upstream.httpx.AsyncClient")
async def test_streaming_failover_mislabeling(mock_client_cls):
    # 11g. Streaming failover mislabeling test
    settings = get_settings()
    settings.failover_base_url = "http://failover"
    settings.upstream_max_retries = 0
    
    # Primary gives 500, failover gives 200
    mock_resp_500 = AsyncMock()
    mock_resp_500.status_code = 500
    mock_resp_500.aread.return_value = b""
    
    mock_resp_200 = AsyncMock()
    mock_resp_200.status_code = 200
    async def fake_aiter():
        yield b"data: {}\n\n"
    mock_resp_200.aiter_bytes = fake_aiter
    
    mock_client = MagicMock()
    mock_client.stream.return_value.__aenter__.side_effect = [mock_resp_500, mock_resp_200]
    mock_client_cls.return_value.__aenter__.return_value = mock_client
    
    result = StreamResult()
    gen = forward_stream({"model": "my-model"}, result)
    
    await anext(gen)
    
    assert result.status_code == 200
    assert result.provider == "failover"
