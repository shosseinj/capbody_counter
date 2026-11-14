#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
COUNT_PYTHON="${COUNT_PYTHON:-python3}"
if ! command -v "$COUNT_PYTHON" >/dev/null 2>&1; then
  echo "Existing Python not found: $COUNT_PYTHON. Set COUNT_PYTHON to your global environment." >&2
  exit 1
fi
exec "$COUNT_PYTHON" -m uvicorn local_service:app --host 127.0.0.1 --port "${COUNT_PORT:-8091}" --workers 1
