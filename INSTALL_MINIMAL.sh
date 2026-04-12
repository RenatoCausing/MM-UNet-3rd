#!/bin/bash
# ==========================================
# MM-UNet Minimal Installation
# Installs only what's needed for inference
# ==========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "MM-UNet Minimal Installation"
echo "=========================================="
echo ""

# Detect Python
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""

# Upgrade pip
echo "Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip setuptools wheel -q

# Install minimal PyTorch
echo "Installing PyTorch..."
$PYTHON_CMD -m pip install --quiet \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu118 || \
$PYTHON_CMD -m pip install --quiet \
    torch \
    torchvision \
    torchaudio

# Install basic inference requirements
echo "Installing inference dependencies..."
$PYTHON_CMD -m pip install --quiet \
    numpy \
    opencv-python \
    matplotlib \
    pillow \
    pyyaml \
    easydict \
    objprint \
    huggingface-hub \
    2>/dev/null || true

# Try to install accelerate and transformers
echo "Installing additional dependencies..."
$PYTHON_CMD -m pip install --quiet \
    accelerate \
    transformers \
    monai \
    timm \
    2>/dev/null || true

# Optional: Try to install Mamba if available
if [ -d "requirements/Mamba/causal-conv1d" ] && [ -d "requirements/Mamba/mamba" ]; then
    echo "Installing custom Mamba (optional)..."
    export CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE
    export MAMBA_SKIP_CUDA_BUILD=TRUE
    
    $PYTHON_CMD -m pip install --quiet --no-build-isolation --no-deps "requirements/Mamba/causal-conv1d" 2>/dev/null || true
    $PYTHON_CMD -m pip install --quiet --no-build-isolation --no-deps "requirements/Mamba/mamba" 2>/dev/null || true
fi

echo ""
echo "=========================================="
echo "✓ Minimal installation complete!"
echo "=========================================="
echo ""
echo "You can now run:"
echo "  bash run_inference_test.sh"
echo ""
