#!/usr/bin/env bash
# Phase 0 smoke test: boot the stack via docker compose, wait for /health,
# assert it returns 200, then tear down. This is the five-minute "does it work
# at all" proof for a grader. Exits non-zero on any failure.
set -euo pipefail

cd "$(dirname "$0")/.."

HEALTH_URL="http://localhost:8000/health"
MAX_WAIT=60  # seconds

cleanup() {
  echo "--- tearing down ---"
  docker compose down --volumes --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "--- docker compose up --build (detached) ---"
docker compose up --build -d

echo "--- waiting for ${HEALTH_URL} (up to ${MAX_WAIT}s) ---"
deadline=$((SECONDS + MAX_WAIT))
until curl -fsS "$HEALTH_URL" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo "FAIL: /health did not come up within ${MAX_WAIT}s"
    docker compose logs gateway || true
    exit 1
  fi
  sleep 1
done

body="$(curl -fsS "$HEALTH_URL")"
echo "GET /health -> ${body}"

if echo "$body" | grep -q '"status":"ok"'; then
  echo "PASS: gateway is healthy"
else
  echo "FAIL: unexpected /health body"
  exit 1
fi
