#!/usr/bin/env bash
set -euo pipefail

R="${R:-/workspace/qwen_calligraphy_lora}"
cd "$R"

TRAIN_SESSION="${1:?train tmux session name required}"
NEW_CKPT="${2:?new official LoRA checkpoint required}"
BASE_SOUP="${3:?base official soup checkpoint required}"
OUT_SOUP_DIR="${4:?output soup dir required}"
EVAL_OUT_DIR="${5:?eval output dir required}"

BASE_WEIGHT="${BASE_WEIGHT:-4}"
POLL_SECONDS="${POLL_SECONDS:-600}"
EVAL_GROUPS="${EVAL_GROUPS:-experiments/results/coverage_v10_eval/compound_eval28_groups.json}"
SEED="${SEED:-7000}"

echo "[post] train_session=$TRAIN_SESSION"
echo "[post] new_ckpt=$NEW_CKPT"
echo "[post] base_soup=$BASE_SOUP base_weight=$BASE_WEIGHT"
echo "[post] out_soup_dir=$OUT_SOUP_DIR"
echo "[post] eval_out_dir=$EVAL_OUT_DIR"

while tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; do
  echo "[post] $(date -u +%FT%TZ) training still alive; sleep ${POLL_SECONDS}s"
  sleep "$POLL_SECONDS"
done

if [[ ! -f "$NEW_CKPT" ]]; then
  echo "[post][ERROR] final checkpoint missing: $NEW_CKPT" >&2
  exit 2
fi

mkdir -p "$OUT_SOUP_DIR"

R="$R" NEW_CKPT="$NEW_CKPT" BASE_SOUP="$BASE_SOUP" OUT_SOUP_DIR="$OUT_SOUP_DIR" BASE_WEIGHT="$BASE_WEIGHT" \
python3 - <<'PY'
import os
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

base = Path(os.environ["BASE_SOUP"])
new = Path(os.environ["NEW_CKPT"])
out = Path(os.environ["OUT_SOUP_DIR"]) / "step-soup.safetensors"
base_weight = float(os.environ["BASE_WEIGHT"])
denom = base_weight + 1.0

with safe_open(base, framework="pt") as bf, safe_open(new, framework="pt") as nf:
    bkeys = list(bf.keys())
    nkeys = list(nf.keys())
    if bkeys != nkeys:
        raise SystemExit(f"key mismatch: base={len(bkeys)} new={len(nkeys)}")
    sd = {}
    for k in bkeys:
        sd[k] = ((bf.get_tensor(k).float() * base_weight + nf.get_tensor(k).float()) / denom).contiguous()

save_file(sd, out)
print(f"[post] wrote weighted soup: {out} ({base_weight:g}:1)")
PY

python3 experiments/scripts/v10/convert_official_lora_to_infer.py \
  "$NEW_CKPT" "${NEW_CKPT%.safetensors}_infer.safetensors"

python3 experiments/scripts/v10/convert_official_lora_to_infer.py \
  "$OUT_SOUP_DIR/step-soup.safetensors" "$OUT_SOUP_DIR/step-soup_infer.safetensors"

python3 experiments/scripts/v10/render_compound_eval.py \
  --groups "$EVAL_GROUPS" \
  --lora_ckpt "$OUT_SOUP_DIR/step-soup_infer.safetensors" \
  --out_dir "$EVAL_OUT_DIR" \
  --seed "$SEED"

echo "[post][DONE] soup + eval28 ready: $EVAL_OUT_DIR"
