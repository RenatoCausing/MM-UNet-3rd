#!/bin/bash
# ========================================
# MM-UNet Complete Test Pipeline
# Downloads dataset, installs deps, pulls best_model.pth from HF, and tests
# ========================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration (edit these or set as env vars)
HF_TOKEN="${HF_TOKEN:-}"
HF_REPO="${HF_REPO:-23LebronJames23/MM-UNet}"
HF_FILENAME="${HF_FILENAME:-best_model.pth}"
DATA_ROOT="${DATA_ROOT:-./fives_preprocessed}"
OUTPUT_DIR="${OUTPUT_DIR:-./test_results_best_model}"
GPU="${GPU:-0}"

echo "========================================================================"
echo " MM-UNet Complete Test Pipeline"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  HF Repo: ${HF_REPO}"
echo "  Model: ${HF_FILENAME}"
echo "  Data: ${DATA_ROOT}"
echo "  Output: ${OUTPUT_DIR}"
echo "  GPU: ${GPU}"
echo ""

# Step 1: Install Python dependencies
echo "========================================================================"
echo "[1/4] Installing Dependencies..."
echo "========================================================================"
if [ -f "./INSTALL_PYTHON310.sh" ]; then
    bash ./INSTALL_PYTHON310.sh 2>&1 | tail -20 || echo "⚠ Installation completed with warnings"
else
    echo "⚠ INSTALL_PYTHON310.sh not found, attempting minimal install..."
    if [ -f "./INSTALL_MINIMAL.sh" ]; then
        bash ./INSTALL_MINIMAL.sh
    else
        echo "Error: No installation script found"
        exit 1
    fi
fi
echo "✓ Dependencies installed"
echo ""

# Step 2: Download FIVEs dataset
echo "========================================================================"
echo "[2/4] Downloading & Verifying FIVEs Dataset..."
echo "========================================================================"
if [ -f "./setup_fives_dataset.sh" ]; then
    DATA_ROOT="$DATA_ROOT" bash ./setup_fives_dataset.sh
else
    if [ ! -d "${DATA_ROOT}/Original" ] || [ ! -d "${DATA_ROOT}/Segmented" ]; then
        echo "Error: Dataset setup script not found and dataset missing"
        exit 1
    fi
fi
echo ""

# Step 3: Set Python command
echo "========================================================================"
echo "[3/4] Detecting Python..."
echo "========================================================================"
if command -v python3.10 >/dev/null 2>&1; then
    PY=python3.10
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    PY=python
fi
$PY --version
echo ""

# Step 4: Run test with best_model.pth from HF
echo "========================================================================"
echo "[4/4] Running Test with best_model.pth..."
echo "========================================================================"
mkdir -p "$OUTPUT_DIR"

# Build command
PY_CMD="$PY test_fives.py"
PY_CMD="$PY_CMD --hf_repo '$HF_REPO'"
PY_CMD="$PY_CMD --hf_filename '$HF_FILENAME'"
PY_CMD="$PY_CMD --data_root '$DATA_ROOT'"
PY_CMD="$PY_CMD --output_dir '$OUTPUT_DIR'"
PY_CMD="$PY_CMD --split_file '${OUTPUT_DIR}/fives_split_seed42_ratio005.json'"
PY_CMD="$PY_CMD --batch_size 4"
PY_CMD="$PY_CMD --num_workers 4"
PY_CMD="$PY_CMD --image_size 1024"
PY_CMD="$PY_CMD --seed 42"
PY_CMD="$PY_CMD --test_ratio 0.05"
PY_CMD="$PY_CMD --gpu '$GPU'"
PY_CMD="$PY_CMD --norm_mean 0.485 0.456 0.406"
PY_CMD="$PY_CMD --norm_std 0.229 0.224 0.225"
PY_CMD="$PY_CMD --save_predictions"

if [ -n "$HF_TOKEN" ]; then
    PY_CMD="$PY_CMD --hf_token '$HF_TOKEN'"
fi

echo "Command: $PY_CMD"
echo ""

eval "$PY_CMD"

echo ""
echo "========================================================================"
echo "✓ TEST PIPELINE COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to: ${OUTPUT_DIR}/"
echo "  - test_results.json: Metrics (Accuracy, Precision, Recall, F1, AUC, Dice)"
echo "  - fives_split_seed42_ratio005.json: Train/test split used"
echo "  - predictions/: Segmentation predictions (if enabled)"
echo ""
echo "To view results:"
echo "  cat ${OUTPUT_DIR}/test_results.json"
echo ""
