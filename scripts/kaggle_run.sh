#!/usr/bin/env bash
# Push kaggle/train.py to Kaggle, wait, download outputs to out/<stage>/.
# Usage: bash scripts/kaggle_run.sh probe|baseline|all [steps] [corti]
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE="${1:-probe}"; STEPS="${2:-600}"; CORTI="${3:-}"
[[ "$STAGE" =~ ^(probe|baseline|all)$ ]] || { echo "stage must be probe|baseline|all"; exit 1; }

USER_NAME="${KAGGLE_USERNAME:-}"
if [[ -z "$USER_NAME" && -f "$HOME/.kaggle/kaggle.json" ]]; then
  USER_NAME=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
fi
[[ -n "$USER_NAME" ]] || { echo "Set KAGGLE_USERNAME or create ~/.kaggle/kaggle.json"; exit 1; }
SLUG="$USER_NAME/whisper-medasr-lora"
CV=False; [[ "$CORTI" == "corti" ]] && CV=True

python3 - "$STAGE" "$STEPS" "$CV" "$SLUG" <<'PY'
import re, sys
stage, steps, cv, slug = sys.argv[1:]
p = open("kaggle/train.py", encoding="utf8").read()
p = re.sub(r'(?m)^STAGE = "[^"]*"', f'STAGE = "{stage}"', p)
p = re.sub(r'(?m)^MAX_STEPS = \d+', f'MAX_STEPS = {steps}', p)
p = re.sub(r'(?m)^RUN_CORTI = \w+', f'RUN_CORTI = {cv}', p)
open("kaggle/train.py", "w", encoding="utf8").write(p)
m = open("kaggle/kernel-metadata.json", encoding="utf8").read()
open("kaggle/kernel-metadata.json", "w", encoding="utf8").write(re.sub(r'"id": "[^"]*"', f'"id": "{slug}"', m))
PY

echo ">> pushing $SLUG stage=$STAGE steps=$STEPS corti=$CV"
kaggle kernels push -p kaggle

sleep 20
START=$(date +%s)
while true; do
  S=$(kaggle kernels status "$SLUG" 2>&1 || true)
  echo "[$(date +%H:%M:%S)] $S"
  [[ "$S" == *COMPLETE* ]] && break
  [[ "$S" == *ERROR* || "$S" == *CANCEL* ]] && { echo "!! kernel failed - downloading log anyway"; break; }
  (( $(date +%s) - START > 10800 )) && { echo "!! local timeout"; break; }
  sleep 60
done

OUT="out/$STAGE"; mkdir -p "$OUT"
kaggle kernels output "$SLUG" -p "$OUT"
echo ">> outputs in $OUT"; ls -la "$OUT"
