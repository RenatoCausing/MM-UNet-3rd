#!/bin/bash
# ==========================================
# Download FIVEs Dataset (Optional)
# ==========================================
# This script can be run separately to download the FIVEs dataset
# for use with actual test images instead of synthetic ones

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "FIVEs Dataset Downloader"
echo "=========================================="
echo ""

# Check if gdown is available
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    python3 -m pip install gdown -q || python -m pip install gdown -q
fi

echo "Downloading FIVEs dataset..."
echo "(This may take a few minutes)"
echo ""

# Download with better error handling (use newer gdown syntax without --id flag)
if gdown 1VTFhKLxdzQAZv3Jj4mZgixI70RzfF68p -O fives_preprocessed.zip 2>&1 | tee download.log; then
    echo ""
    echo "Download completed. Extracting..."
    
    if unzip -q fives_preprocessed.zip; then
        rm -f fives_preprocessed.zip
        echo "✓ FIVEs dataset extracted successfully!"
        echo ""
        echo "Dataset structure:"
        echo "  fives_preprocessed/"
        echo "  ├── train/"
        echo "  ├── test/"
        echo "  └── metadata.json"
        echo ""
        echo "You can now run inference with real test images:"
        echo "  bash run_inference_test.sh"
    else
        echo "❌ Error: Failed to extract dataset"
        exit 1
    fi
else
    echo "❌ Error: Failed to download dataset"
    echo ""
    echo "Possible causes:"
    echo "  - Network connection issue"
    echo "  - Google Drive quota exceeded"
    echo "  - Invalid file ID"
    echo ""
    echo "You can still use inference testing with synthetic images:"
    echo "  bash run_inference_test.sh"
    exit 1
fi

echo ""
echo "=========================================="
