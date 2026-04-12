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

# Step 1: Install Python 3.10
echo "======================================================================"
echo "Step 1: Installing Python 3.10"
echo "======================================================================"
if [ -f "INSTALL_PYTHON310.sh" ]; then
    bash ./INSTALL_PYTHON310.sh
    echo "Python 3.10 installation completed"
else
    echo "Warning: INSTALL_PYTHON310.sh not found, skipping Python installation"
fi

echo ""

# Step 2: Verify Python installation
echo "======================================================================"
echo "Step 2: Verifying Python Installation"
echo "======================================================================"
python3.10 --version || python3 --version
echo ""

# Step 3: Install required Python packages (if needed)
echo "======================================================================"
echo "Step 3: Installing Required Python Packages"
echo "======================================================================"

# Check if requirements.txt exists and install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing from main requirements.txt..."
    python3.10 -m pip install --upgrade pip -q
    python3.10 -m pip install -r requirements.txt -q
    echo "Main requirements installed"
fi

# Install additional packages needed for inference
echo "Installing inference-specific packages..."
python3.10 -m pip install -q \
    huggingface-hub \
    matplotlib \
    opencv-python \
    pyyaml \
    easydict \
    objprint \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

echo "Packages installed successfully"
echo ""

# Step 4: Run the inference test script
echo "======================================================================"
echo "Step 4: Running MM-UNet Inference Tests"
echo "======================================================================"
echo ""

python3.10 test_inference.py

# Step 5: Summary
echo ""
echo "======================================================================"
echo "Inference Testing Completed"
echo "======================================================================"
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
echo "======================================================================"
