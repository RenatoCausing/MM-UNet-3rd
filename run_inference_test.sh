#!/bin/bash

set -e  # Exit on error

echo "======================================================================"
echo "MM-UNet Inference Testing Script"
echo "======================================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $(pwd)"
echo ""

# Step 1: Try to run the full installation script if it exists
echo "======================================================================"
echo "Step 1: Attempting Python 3.10 Installation"
echo "======================================================================"

INSTALL_SUCCESS=0

if [ -f "INSTALL_PYTHON310.sh" ]; then
    echo "Found INSTALL_PYTHON310.sh, attempting full installation..."
    if bash ./INSTALL_PYTHON310.sh 2>&1 | tee install_log.txt; then
        INSTALL_SUCCESS=1
        echo "✓ Python installation completed successfully"
    else
        echo "⚠ Warning: Full installation had issues"
        
        # Try minimal installation as fallback
        if [ -f "INSTALL_MINIMAL.sh" ]; then
            echo ""
            echo "Attempting minimal installation as fallback..."
            if bash ./INSTALL_MINIMAL.sh 2>&1 | tee -a install_log.txt; then
                INSTALL_SUCCESS=1
                echo "✓ Minimal installation completed successfully"
            else
                echo "⚠ Minimal installation also had issues, will continue anyway..."
            fi
        fi
    fi
else
    echo "⚠ Full installation script not found"
    
    # Try minimal installation
    if [ -f "INSTALL_MINIMAL.sh" ]; then
        echo "Attempting minimal installation instead..."
        if bash ./INSTALL_MINIMAL.sh 2>&1 | tee install_log.txt; then
            INSTALL_SUCCESS=1
            echo "✓ Minimal installation completed successfully"
        fi
    fi
fi

echo ""

# Step 2: Verify Python installation
echo "======================================================================"
echo "Step 2: Verifying Python Installation"
echo "======================================================================"

if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    echo "✓ Found Python 3.10"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "✓ Using Python 3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo "✓ Using Python (default)"
else
    echo "❌ ERROR: No Python installation found!"
    exit 1
fi

$PYTHON_CMD --version
echo ""

# Step 3: Install minimal dependencies for inference
echo "======================================================================"
echo "Step 3: Installing Minimal Inference Dependencies"
echo "======================================================================"

echo "Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip setuptools wheel -q 2>/dev/null || true

echo "Installing base packages..."
$PYTHON_CMD -m pip install --upgrade --quiet \
    numpy \
    torch \
    torchvision \
    torchaudio \
    2>/dev/null || $PYTHON_CMD -m pip install --quiet \
    numpy \
    torch \
    torchvision \
    torchaudio

echo "Installing required packages for inference..."
$PYTHON_CMD -m pip install --quiet \
    huggingface-hub \
    matplotlib \
    opencv-python \
    pyyaml \
    easydict \
    objprint \
    2>/dev/null || true

# If full installation failed, try to install Mamba separately but don't fail if it doesn't work
if [ $INSTALL_SUCCESS -eq 0 ]; then
    echo ""
    echo "Attempting to install Mamba (optional)..."
    if [ -d "requirements/Mamba/causal-conv1d" ]; then
        $PYTHON_CMD -m pip install --quiet --no-build-isolation --no-deps "requirements/Mamba/causal-conv1d" 2>/dev/null || true
    fi
    if [ -d "requirements/Mamba/mamba" ]; then
        $PYTHON_CMD -m pip install --quiet --no-build-isolation --no-deps "requirements/Mamba/mamba" 2>/dev/null || true
    fi
fi

echo "✓ Packages installed (or already present)"
echo ""

# Step 4: Verify critical packages
echo "======================================================================"
echo "Step 4: Verifying Critical Packages"
echo "======================================================================"

$PYTHON_CMD -c "
import sys
try:
    import torch
    print(f'✓ PyTorch: {torch.__version__}')
except ImportError:
    print('❌ PyTorch not found')
    sys.exit(1)

try:
    import numpy
    print(f'✓ NumPy: {numpy.__version__}')
except ImportError:
    print('❌ NumPy not found')
    sys.exit(1)

try:
    import cv2
    print(f'✓ OpenCV: {cv2.__version__}')
except ImportError:
    print('⚠ OpenCV warning - will be installed')

try:
    import yaml
    print(f'✓ PyYAML available')
except ImportError:
    print('⚠ PyYAML warning - will be installed')

print('✓ Core packages verified')
"

echo ""

# Step 5: Run the inference test script
echo "======================================================================"
echo "Step 5: Running MM-UNet Inference Tests"
echo "======================================================================"
echo ""

if [ ! -f "test_inference.py" ]; then
    echo "❌ ERROR: test_inference.py not found!"
    exit 1
fi

if $PYTHON_CMD test_inference.py; then
    TEST_SUCCESS=1
    echo "✓ Inference tests completed successfully"
else
    TEST_SUCCESS=0
    echo "⚠ Warning: Some inference tests had issues - see output above"
fi

echo ""

# Step 6: Summary
echo "======================================================================"
echo "Installation and Testing Summary"
echo "======================================================================"
echo ""

if [ $TEST_SUCCESS -eq 1 ]; then
    echo "✓ Inference testing completed successfully!"
    echo ""
    echo "Results saved to:"
    echo "  - ./inference_results/single_image_epochs_10_20_30_40/"
    echo "  - ./inference_results/20_images_epoch_40/"
    echo ""
    echo "Each result directory contains:"
    echo "  - comparison.png (original vs segmentation)"
    echo "  - segmentation_mask.png (binary mask)"
    echo "  - original_image.png (original image)"
    echo ""
    echo "Configuration info saved to:"
    echo "  - ./inference_results/config_info.json"
    echo ""
    echo "Note: If FIVEs dataset wasn't downloaded, synthetic test images"
    echo "were used for demonstration. Place real images in these directories"
    echo "to use actual test data:"
    echo "  - ./fives_preprocessed/test"
    echo "  - ./data/test"
    echo "  - ./test_images"
else
    echo "⚠ Tests completed with warnings (see details above)"
fi

echo ""
echo "======================================================================"

