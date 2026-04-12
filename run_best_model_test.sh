#!/bin/bash
set -e

# One-command cloud test runner:
# 1) installs env via existing installer
# 2) downloads best_model.pth from HF during test
# 3) evaluates on deterministic 95/5 split with seed 42
# 4) preprocesses on-the-fly (no tensor cache files)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HF_REPO="${HF_REPO:-23LebronJames23/MM-UNet}"
HF_FILENAME="${HF_FILENAME:-best_model.pth}"
HF_TOKEN="${HF_TOKEN:-}"
DATA_ROOT="${DATA_ROOT:-./fives_preprocessed}"
OUTPUT_DIR="${OUTPUT_DIR:-./test_results_best_model}"
SPLIT_FILE="${SPLIT_FILE:-${OUTPUT_DIR}/fives_split_seed42_ratio005.json}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
SEED="${SEED:-42}"
TEST_RATIO="${TEST_RATIO:-0.05}"
GPU="${GPU:-0}"
SAVE_PREDS="${SAVE_PREDS:-1}"

echo "============================================================"
echo "MM-UNet Best Model Cloud Test"
echo "============================================================"
echo "Repo: ${HF_REPO}"
echo "Model file: ${HF_FILENAME}"
echo "Data root: ${DATA_ROOT}"
echo "Output: ${OUTPUT_DIR}"
echo "Split: ${SPLIT_FILE}"
echo "Seed/Test ratio: ${SEED}/${TEST_RATIO}"
echo "============================================================"

if [ -f "./INSTALL_PYTHON310.sh" ]; then
  bash ./INSTALL_PYTHON310.sh || true
fi

if command -v python3.10 >/dev/null 2>&1; then
  PY=python3.10
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

$PY -m pip install -q --upgrade huggingface-hub

mkdir -p "$OUTPUT_DIR"

ARGS=(
  test_fives.py
  --hf_repo "$HF_REPO"
  --hf_filename "$HF_FILENAME"
  --data_root "$DATA_ROOT"
  --output_dir "$OUTPUT_DIR"
  --split_file "$SPLIT_FILE"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --image_size "$IMAGE_SIZE"
  --seed "$SEED"
  --test_ratio "$TEST_RATIO"
  --gpu "$GPU"
  --norm_mean 0.485 0.456 0.406
  --norm_std 0.229 0.224 0.225
)

if [ -n "$HF_TOKEN" ]; then
  ARGS+=(--hf_token "$HF_TOKEN")
fi

if [ "$SAVE_PREDS" = "1" ]; then
  ARGS+=(--save_predictions)
fi

$PY "${ARGS[@]}"

echo "Done. Results at: ${OUTPUT_DIR}"
