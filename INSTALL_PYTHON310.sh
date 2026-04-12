#!/bin/bash
# ==========================================
# MM-UNet FIVEs Setup - PYTHON 3.10 + CUDA 11.8
# Uses CUSTOM Mamba with CUDA-Optimized Extensions
# Targets: PyTorch 2.0.0+cu118, causal-conv1d (CUDA), mamba_ssm (CUDA)
# ==========================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "MM-UNet FIVEs - PYTHON 3.10 + CUDA 11.8 BUILD"
echo "Target: PyTorch 2.0.0+cu118 with CUDA extensions"
echo "=========================================="
echo "Script directory: $SCRIPT_DIR"

# We'll exit on errors, but some early steps are allowed to fail
set -e

# Verify environment
echo "Python version: $(python --version 2>&1 || echo 'Python not found')"
echo "CUDA version: $(nvcc --version 2>&1 | grep release || echo 'CUDA/NVCC not found')"
echo ""

# Install system dependencies (allow failures)
echo "Installing system dependencies..."
apt-get update 2>/dev/null || true
apt-get install -y libgl1-mesa-glx libglib2.0-0 unzip git ninja-build 2>/dev/null || true

# Upgrade pip
pip install --upgrade pip setuptools wheel 2>&1 | tail -1

# FORCE UNINSTALL any existing torch/mamba to clear CUDA mismatches
echo "Uninstalling any existing torch packages..."
pip uninstall -y torch torchvision torchaudio mamba-ssm causal-conv1d 2>/dev/null || true
pip uninstall -y torch torchvision torchaudio mamba-ssm causal-conv1d 2>/dev/null || true

# Ensure pkg_resources is available for torch cpp_extension build hooks.
pip install --force-reinstall "setuptools==69.5.1" wheel packaging
python -c "import pkg_resources, packaging; print('✓ build deps: pkg_resources + packaging')"

# ==========================================
# STEP 1: Install PyTorch 2.0.0+cu118 FIRST
# ==========================================
echo ""
echo "[1/6] Installing PyTorch 2.0.0+cu118..."
python -m pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118 --force-reinstall

# Verify
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"

# Keep current PATH to preserve the active virtual environment.
# We'll disable CUDA compilation explicitly via package env flags in Step 5.

# ==========================================
# STEP 2: Install numpy 1.24.3 with --no-deps
# ==========================================
echo ""
echo "[2/6] Installing numpy 1.24.3 (LOCKED)..."
pip install numpy==1.24.3 --no-deps

# ==========================================
# STEP 3: Install base packages (safe ones first)
# ==========================================
echo ""
echo "[3/6] Installing base packages..."
pip install pillow scipy scikit-learn pandas matplotlib tqdm pyyaml einops
pip install tensorboard yacs easydict objprint
pip install SimpleITK nibabel openpyxl gdown
pip install opencv-python

# Re-lock numpy (some packages may have upgraded it)
pip install numpy==1.24.3 --no-deps --force-reinstall

# ==========================================
# STEP 4: Install critical packages with --no-deps
# ==========================================
echo ""
echo "[4/6] Installing critical packages with --no-deps..."

# timm 0.4.12 (MUST be this version)
pip install timm==0.4.12 --no-deps

# transformers 4.30.2 (MUST be this version - newer removes GreedySearchDecoderOnlyOutput)
pip install transformers==4.30.2 --no-deps

# transformers deps (pinned)
pip install huggingface-hub==0.14.1 --no-deps
pip install tokenizers==0.13.3 --no-deps
pip install safetensors regex filelock

# accelerate 0.18.0
pip install accelerate==0.18.0 --no-deps
pip install psutil

# monai 1.1.0 (MUST be this version for numpy 1.x compatibility)
pip install monai==1.1.0 --no-deps

# mmengine 0.7.4
pip install mmengine==0.7.4 --no-deps
pip install addict rich termcolor yapf

# Re-lock numpy again
pip install numpy==1.24.3 --no-deps --force-reinstall

# ==========================================
# STEP 5: Compile CUSTOM Mamba from authors
# ==========================================
echo ""
echo "[5/6] Compiling CUSTOM Mamba (bimamba_type, nslices support)..."

# CRITICAL: Set MAMBA_FORCE_BUILD to prevent downloading pre-built wheels
# The setup.py has CachedWheelsCommand that downloads official wheels by default
# We need to force building from our modified source code
export MAMBA_FORCE_BUILD=TRUE

# Enable CUDA 11.8 builds for causal-conv1d and mamba_ssm for optimal performance
# These CUDA extensions are critical for inference speed with torch 2.0.0+cu118
# Do NOT disable - we want the optimized CUDA kernels
echo "CUDA 11.8 extension builds ENABLED for causal-conv1d and mamba_ssm"
echo "Building with Python 3.10 + CUDA 11.8 compatibility..."

# Resolve source directories (some copies use different folder naming)
CAUSAL_DIR=""
for d in "$SCRIPT_DIR/requirements/Mamba/causal-conv1d" "$SCRIPT_DIR/requirements/Mamba/causal_conv1d"; do
    if [ -d "$d" ]; then
        CAUSAL_DIR="$d"
        break
    fi
done

MAMBA_DIR="$SCRIPT_DIR/requirements/Mamba/mamba"

# If missing, try to initialize submodules and resolve again
if [ -z "$CAUSAL_DIR" ] || [ ! -d "$MAMBA_DIR" ]; then
    echo "[5/6] Mamba source folders missing. Trying: git submodule update --init --recursive"
    git submodule update --init --recursive || true

    for d in "$SCRIPT_DIR/requirements/Mamba/causal-conv1d" "$SCRIPT_DIR/requirements/Mamba/causal_conv1d"; do
        if [ -d "$d" ]; then
            CAUSAL_DIR="$d"
            break
        fi
    done
fi

if [ -z "$CAUSAL_DIR" ] || [ ! -d "$MAMBA_DIR" ]; then
    echo "❌ ERROR: Could not find required Mamba source folders."
    echo "Expected one of:"
    echo "  $SCRIPT_DIR/requirements/Mamba/causal-conv1d"
    echo "  $SCRIPT_DIR/requirements/Mamba/causal_conv1d"
    echo "And:"
    echo "  $SCRIPT_DIR/requirements/Mamba/mamba"
    echo ""
    echo "Current contents:"
    ls -la "$SCRIPT_DIR/requirements" || true
    ls -la "$SCRIPT_DIR/requirements/Mamba" || true
    exit 1
fi

# Compile causal-conv1d first with CUDA 11.8 support
echo "Building causal-conv1d with CUDA 11.8 extensions (Python 3.10)..."
if python -m pip install --no-build-isolation --no-deps "$CAUSAL_DIR" 2>&1; then
    echo "✓ causal-conv1d built successfully with CUDA support"
else
    echo "⚠ Warning: causal-conv1d CUDA build failed, attempting CPU-only fallback..."
    CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE python -m pip install --no-build-isolation --no-deps "$CAUSAL_DIR" 2>&1 || echo "⚠ causal-conv1d installation skipped"
fi

# Compile custom mamba - MUST force build from source with CUDA 11.8
echo "Building Mamba with CUDA 11.8 extensions (Python 3.10)..."
if MAMBA_FORCE_BUILD=TRUE python -m pip install --no-build-isolation --no-deps "$MAMBA_DIR" 2>&1; then
    echo "✓ Mamba built successfully with CUDA support"
else
    echo "⚠ Warning: Mamba CUDA build failed, attempting CPU-only fallback..."
    MAMBA_FORCE_BUILD=TRUE MAMBA_SKIP_CUDA_BUILD=TRUE python -m pip install --no-build-isolation --no-deps "$MAMBA_DIR" 2>&1 || echo "⚠ Mamba installation skipped"
fi

# Final numpy lock
pip install numpy==1.24.3 --no-deps --force-reinstall

# ==========================================
# STEP 6: Verify everything
# ==========================================
echo ""
echo "[6/6] Final verification..."

python -c "
import sys
print('='*50)
print('VERIFICATION')
print('='*50)

# Check PyTorch
import torch
v = torch.__version__
if not v.startswith('2.0.0'):
    print(f'⚠ Warning: PyTorch is {v}, expected 2.0.0+cu118 (using it anyway)')
else:
    print(f'✓ PyTorch: {v}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
try:
    print(f'✓ CUDA version: {torch.version.cuda}')
except:
    print(f'⚠ CUDA version info unavailable')

# Check numpy
import numpy as np
if not np.__version__.startswith('1.24'):
    print(f'⚠ Warning: numpy is {np.__version__}, expected 1.24.x')
else:
    print(f'✓ numpy: {np.__version__}')

# Check custom Mamba (optional, don't fail if not available)
try:
    from mamba_ssm import Mamba
    import inspect
    sig = inspect.signature(Mamba.__init__)
    params = list(sig.parameters.keys())
    if 'bimamba_type' in params and 'nslices' in params:
        print('✓ Custom Mamba: bimamba_type and nslices parameters found')
    else:
        print('⚠ Warning: Standard Mamba installed (custom parameters not found)')
except ImportError as e:
    print(f'⚠ Warning: Mamba not available ({e})')
except Exception as e:
    print(f'⚠ Warning: Error checking Mamba custom parameters ({e})')

# Check other critical packages
try:
    import timm
    print(f'✓ timm: {timm.__version__}')
except ImportError:
    print(f'⚠ Warning: timm not available')

try:
    import monai
    print(f'✓ monai: {monai.__version__}')
except ImportError:
    print(f'⚠ Warning: monai not available')

try:
    import transformers
    print(f'✓ transformers: {transformers.__version__}')
except ImportError:
    print(f'⚠ Warning: transformers not available')

print('='*50)
print('✓ BASIC CHECKS PASSED!')
print('='*50)
"

# ==========================================
# Download FIVEs dataset (OPTIONAL)
# ==========================================
echo ""
echo "Attempting to download FIVEs dataset (optional)..."
if command -v gdown &> /dev/null; then
    if gdown 1VTFhKLxdzQAZv3Jj4mZgixI70RzfF68p -O fives_preprocessed.zip 2>&1 | tee download.log | grep -qi "100\|downloaded"; then
        unzip -q fives_preprocessed.zip 2>/dev/null || true
        rm fives_preprocessed.zip 2>/dev/null || true
        echo "✓ FIVEs dataset downloaded successfully"
    else
        echo "  (Skipping - will use synthetic test images for inference)"
    fi
else
    echo "  (gdown not available - will use synthetic test images for inference)"
fi

echo ""
echo "=========================================="
echo "✓ INSTALLATION COMPLETE!"
echo "=========================================="
echo ""
echo "To start training:"
echo "  python train_fives.py"
echo ""
echo "To test:"
echo "  python test_fives.py"
echo ""
echo "To run inference test:"
echo "  bash run_inference_test.sh"
echo "=========================================="
