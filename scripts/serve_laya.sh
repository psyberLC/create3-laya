#!/usr/bin/env bash
# Start Laya, the local decision-model server, for the Create 3 (Linux / WSL).
# Serves `typed-decisions` (task understanding) and, once trained, the `create3-driving`
# fine-tune from models/create3-driving. Binds to 127.0.0.1 unless LAYA_HOST is set:
# laya-serve otherwise listens on 0.0.0.0 with no authentication.
#
# Each checkpoint holds ~2 GB of GPU memory, so only task understanding is preloaded; the
# driving fine-tune loads on first use, and requests naming no model use typed-decisions.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
lazy=""
if [ -f "$root/models/create3-driving/model.safetensors" ]; then
  export LAYA_EXTRA_MODELS="{\"create3-driving\": \"$root/models/create3-driving\"}"
  lazy=" (create3-driving on demand)"
else
  echo "No driving fine-tune in models/create3-driving yet; serving task understanding only."
fi
export LAYA_HOST="${LAYA_HOST:-127.0.0.1}"
export LAYA_PORT="${LAYA_PORT:-8000}"
export LAYA_DEVICE="${LAYA_DEVICE:-cuda}"
export LAYA_PRELOAD=1
export LAYA_MODELS=typed-decisions
export LAYA_MAX_LOADED=2
export LAYA_DEFAULT_MODEL=typed-decisions
echo "Laya on http://$LAYA_HOST:$LAYA_PORT ($LAYA_DEVICE): typed-decisions$lazy"
exec laya-serve
