"""Phase 7: Prometheus metrics definitions."""
from prometheus_client import Counter, Histogram

from .seed import BASELINE_PRICES

# Cardinality guard: the `model` label only ever takes values from this fixed
# set (the seeded price table) plus "unknown_model". Model ids arrive from
# CLIENT input — labelling them raw would let any client mint unbounded new
# time series ("model": "garbage-1", "garbage-2", ...) and bloat the registry.
_KNOWN_MODELS = frozenset(BASELINE_PRICES)


def normalize_model_name(model: object) -> str:
    """Collapse a client-supplied model id into a bounded label value.

    Known (seeded-price) models pass through verbatim; anything else —
    unknown ids, garbage, None, non-strings — becomes "unknown_model".
    """
    if isinstance(model, str) and model in _KNOWN_MODELS:
        return model
    return "unknown_model"


REQUEST_COUNT = Counter(
    "gateway_requests_total",
    "Total number of requests processed by the gateway",
    ["model", "status_code", "cache_hit"]
)

# Bucket tuning: prometheus-client's defaults top out at 10s — LLM
# generations routinely exceed that, which would collapse P99 into the +Inf
# bucket and make histogram_quantile lie. Full-request buckets stretch to
# 120s; TTFT buckets are dense below 1s because that's where streaming UX
# is won or lost.
REQUEST_LATENCY = Histogram(
    "gateway_request_latency_seconds",
    "Total time taken to process a request",
    ["model", "status_code"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, float("inf")],
)

TTFT_LATENCY = Histogram(
    "gateway_ttft_seconds",
    "Time to first token for streaming requests",
    ["model"],
    buckets=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0, float("inf")],
)

FAILOVER_COUNT = Counter(
    "gateway_failovers_total",
    "Total number of times the gateway retried or failed over",
    ["model"]
)
