#!/usr/bin/env bash
set -euo pipefail

TARGET_ROOT="${1:-.}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "$TARGET_ROOT/src/crossenv_act" ]]; then
  echo "[patch error] Target root must contain src/crossenv_act: $TARGET_ROOT" >&2
  exit 1
fi

cp "$PATCH_DIR/files/src/crossenv_act/eval.py" "$TARGET_ROOT/src/crossenv_act/eval.py"
cp "$PATCH_DIR/files/src/crossenv_act/cli.py" "$TARGET_ROOT/src/crossenv_act/cli.py"
if [[ -d "$TARGET_ROOT/scripts" ]]; then
  cp "$PATCH_DIR/files/scripts/eval_zero_shot.sh" "$TARGET_ROOT/scripts/eval_zero_shot.sh"
  chmod +x "$TARGET_ROOT/scripts/eval_zero_shot.sh"
fi

python -m py_compile "$TARGET_ROOT/src/crossenv_act/eval.py" "$TARGET_ROOT/src/crossenv_act/cli.py"
if [[ -f "$TARGET_ROOT/scripts/eval_zero_shot.sh" ]]; then
  bash -n "$TARGET_ROOT/scripts/eval_zero_shot.sh"
fi

echo "[patch] Replaced strict zero-shot eval files. Reinstall with: pip install -e $TARGET_ROOT"
