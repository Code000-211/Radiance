#!/usr/bin/env bash
set -euo pipefail

PORT="8002"
PID_FILE="/home/orca/.gemini/antigravity-cli/scratch/engine_8002.pid"

echo "Stopping engine on port ${PORT}..."

if [ -f "${PID_FILE}" ]; then
  PID="$(cat "${PID_FILE}")"
  echo "Signaling launcher process group ${PID}..."
  kill -TERM -- "-${PID}" 2>/dev/null || kill -TERM "${PID}" 2>/dev/null || true
fi

# Also kill any radiance process bound to port 8002
pkill -TERM -f "radiance.*--port ${PORT}" 2>/dev/null || true

for _ in $(seq 1 30); do
  if ! pgrep -f "radiance.*--port ${PORT}" >/dev/null 2>&1; then
    echo "✓ Port ${PORT} engine stopped cleanly."
    rm -f "${PID_FILE}"
    exit 0
  fi
  sleep 1
done

echo "Force killing any remaining port ${PORT} processes..."
pkill -KILL -f "radiance.*--port ${PORT}" 2>/dev/null || true
rm -f "${PID_FILE}"
echo "✓ Stopped."
