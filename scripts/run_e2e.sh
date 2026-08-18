#!/usr/bin/env bash
# End-to-End integration test suite runner.
# This boots the gateway with a stub upstream provider, runs the pytest E2E tests,
# and cleans up afterwards.
set -euo pipefail

cd "$(dirname "$0")/.."

# Start the mock upstream provider in the background
echo "--- starting mock upstream provider ---"
python scripts/demo_stub_upstream.py &
UPSTREAM_PID=$!

cleanup() {
  echo "--- tearing down ---"
  kill $UPSTREAM_PID || true
  docker compose -f docker-compose.yml -f compose.demo.yml down --volumes --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "--- docker compose up --build (detached, using demo config) ---"
docker compose -f docker-compose.yml -f compose.demo.yml up --build -d

HEALTH_URL="http://localhost:8000/health"
MAX_WAIT=60  # seconds

echo "--- waiting for gateway to be healthy (up to ${MAX_WAIT}s) ---"
deadline=$((SECONDS + MAX_WAIT))
until curl -fsS "$HEALTH_URL" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo "FAIL: gateway did not come up within ${MAX_WAIT}s"
    docker compose logs gateway || true
    exit 1
  fi
  sleep 1
done

echo "--- gateway is up, running pytest E2E tests ---"
# Use the virtual environment if it exists, otherwise use system pytest
if [ -f "backend/.venv/bin/activate" ]; then
  source backend/.venv/bin/activate
elif [ -f "backend/.venv/Scripts/activate" ]; then
  source backend/.venv/Scripts/activate
fi

# We use the key expected by the .env or demo script
export GATEWAY_API_KEY="my_secure_local_password"
pytest scripts/test_e2e.py -v

echo "--- E2E tests passed successfully ---"
