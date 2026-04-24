#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_NAME="${PORTLESS_API_NAME:-api.magisterka}"
WEB_NAME="${PORTLESS_WEB_NAME:-magisterka}"
PORTLESS_URL_SUFFIX=""

if [[ -n "${PORTLESS_PORT:-}" && "${PORTLESS_PORT}" != "443" ]]; then
  PORTLESS_URL_SUFFIX=":${PORTLESS_PORT}"
fi

WEB_URL="https://${WEB_NAME}.localhost${PORTLESS_URL_SUFFIX}"
API_URL="${VITE_API_PROXY_TARGET:-https://${API_NAME}.localhost${PORTLESS_URL_SUFFIX}}"

run_portless() {
  npx --yes portless@0.13.0 "$@"
}

cleanup() {
  local status=$?
  if [[ -n "${API_PID:-}" ]]; then kill "${API_PID}" 2>/dev/null || true; fi
  if [[ -n "${WEB_PID:-}" ]]; then kill "${WEB_PID}" 2>/dev/null || true; fi
  wait 2>/dev/null || true
  exit "${status}"
}

trap cleanup INT TERM EXIT

cd "${ROOT_DIR}"

run_portless "${API_NAME}" sh -c 'venv/bin/uvicorn src.api.main:app --reload --host 0.0.0.0 --port "${PORT}"' &
API_PID=$!

(
  cd "${ROOT_DIR}/frontend"
  export VITE_API_PROXY_TARGET="${API_URL}"
  run_portless "${WEB_NAME}" sh -c 'npm run dev -- --host 0.0.0.0 --port "${PORT}"'
) &
WEB_PID=$!

echo "Portless frontend: ${WEB_URL}"
echo "Portless API:      ${API_URL}"
echo "Press Ctrl+C to stop both services."

wait "${API_PID}" "${WEB_PID}"
