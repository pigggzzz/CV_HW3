#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-.}"
if [[ ! -d "$ROOT/src/crossenv_act" ]]; then
  echo "[patch] Cannot find src/crossenv_act under $ROOT" >&2
  echo "[patch] Usage: bash apply_patch.sh /home/lama/task2" >&2
  exit 1
fi
cp -f "$(dirname "$0")/files/src/crossenv_act/offline_replay.py" "$ROOT/src/crossenv_act/offline_replay.py"
mkdir -p "$ROOT/scripts"
cp -f "$(dirname "$0")/files/scripts/offline_replay_visualize.sh" "$ROOT/scripts/offline_replay_visualize.sh"
chmod +x "$ROOT/scripts/offline_replay_visualize.sh"
echo "[patch] Applied valid-horizon replay visualization patch to $ROOT"
echo "[patch] Replay no longer scans all safe candidates; it clips invalid/padded action horizons per frame."
echo "[patch] --sequence-ids now selects episode ids directly, e.g. --sequence-ids 2,10,17,21."
