#!/bin/bash
# ==========================================
# MM-UNet FIVEs Setup - AGGRESSIVE VERSION LOCKING
# Uses CUSTOM Mamba from authors (requirements/Mamba/)
# ALL packages installed with --no-deps to prevent upgrades
# ==========================================
set -e  # Exit on any error

echo "=========================================="
echo "MM-UNet FIVEs - AGGRESSIVE INSTALL"
echo "Target: PyTorch 2.0.0+cu118, Custom Mamba"
echo "=========================================="

# Verify environment
echo "Python version: $(python --version)"
echo "CUDA version: $(nvcc --version | grep release)"

# Install system dependencies
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 unzip git ninja-build

# Upgrade pip
pip install --upgrade pip setuptools wheel

# ==========================================
# STEP 1: Install PyTorch 2.0.0+cu118 FIRST
# ==========================================
echo ""
echo "[1/6] Installing PyTorch 2.0.0+cu118..."
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"

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

# Compile causal-conv1d first
cd requirements/Mamba/causal-conv1d
pip install . --no-build-isolation --no-deps
cd ../../..

# Compile custom mamba - MUST force build from source
cd requirements/Mamba/mamba
MAMBA_FORCE_BUILD=TRUE pip install . --no-build-isolation --no-deps
cd ../../..

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
    print(f'❌ ERROR: PyTorch is {v}, expected 2.0.0+cu118')
    sys.exit(1)
print(f'✓ PyTorch: {v}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
print(f'✓ CUDA version: {torch.version.cuda}')

# Check numpy
import numpy as np
if not np.__version__.startswith('1.24'):
    print(f'❌ ERROR: numpy is {np.__version__}, expected 1.24.x')
    sys.exit(1)
print(f'✓ numpy: {np.__version__}')

# Check custom Mamba
from mamba_ssm import Mamba
import inspect
sig = inspect.signature(Mamba.__init__)
params = list(sig.parameters.keys())
if 'bimamba_type' not in params:
    print('❌ ERROR: Mamba does not have bimamba_type parameter!')
    print('   This means standard mamba-ssm was installed instead of custom')
    sys.exit(1)
print('✓ Custom Mamba: bimamba_type parameter found')

if 'nslices' not in params:
    print('❌ ERROR: Mamba does not have nslices parameter!')
    sys.exit(1)
print('✓ Custom Mamba: nslices parameter found')

# Check other critical packages
import timm
print(f'✓ timm: {timm.__version__}')

import monai
print(f'✓ monai: {monai.__version__}')

import transformers
print(f'✓ transformers: {transformers.__version__}')

print('='*50)
print('✓ ALL CHECKS PASSED!')
print('='*50)
"

# ==========================================
# Download FIVEs dataset
# ==========================================
echo ""
echo "Downloading FIVEs dataset..."
gdown --id 1VTFhKLxdzQAZv3Jj4mZgixI70RzfF68p
unzip -q fives_preprocessed.zip
rm fives_preprocessed.zip

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
echo "=========================================="
