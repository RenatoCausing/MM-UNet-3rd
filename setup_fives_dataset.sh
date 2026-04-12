#!/bin/bash
# ========================================
# FIVEs Dataset Setup & Verification
# Downloads, extracts, and verifies structure
# ========================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT="${DATA_ROOT:-./fives_preprocessed}"

echo "========================================================================"
echo " FIVEs Dataset Setup"
echo "========================================================================"
echo "Target: ${DATA_ROOT}"
echo ""

# Check if already downloaded and extracted
if [ -d "${DATA_ROOT}/Original" ] && [ -d "${DATA_ROOT}/Segmented" ]; then
    echo "✓ Dataset already verified at ${DATA_ROOT}"
    echo ""
    echo "Structure:"
    echo "  Original/   $(ls ${DATA_ROOT}/Original 2>/dev/null | wc -l) files"
    echo "  Segmented/  $(ls ${DATA_ROOT}/Segmented 2>/dev/null | wc -l) files"
    exit 0
fi

echo "Dataset not found. Downloading..."
echo ""

# Install gdown if needed
if ! command -v gdown >/dev/null 2>&1; then
    echo "Installing gdown..."
    if command -v python3.10 >/dev/null 2>&1; then
        python3.10 -m pip install -q gdown
    elif command -v python3 >/dev/null 2>&1; then
        python3 -m pip install -q gdown
    else
        python -m pip install -q gdown
    fi
    echo ""
fi

# Download
echo "Downloading fives_preprocessed.zip (144 MB)..."
echo "  File ID: 1VTFhKLxdzQAZv3Jj4mZgixI70RzfF68p"
echo ""

if gdown 1VTFhKLxdzQAZv3Jj4mZgixI70RzfF68p -O fives_preprocessed.zip; then
    echo ""
    echo "✓ Download complete. Extracting..."
    
    if [ -f "fives_preprocessed.zip" ]; then
        unzip -q fives_preprocessed.zip && echo "✓ Extracted" || {
            echo "❌ Extraction failed"
            exit 1
        }
        
        rm -f fives_preprocessed.zip
        
        # Verify structure
        if [ -d "${DATA_ROOT}/Original" ] && [ -d "${DATA_ROOT}/Segmented" ]; then
            echo ""
            echo "✓ Dataset verified!"
            echo ""
            echo "Structure:"
            echo "  ${DATA_ROOT}/"
            echo "  ├── Original/   $(ls ${DATA_ROOT}/Original 2>/dev/null | wc -l) images"
            echo "  └── Segmented/  $(ls ${DATA_ROOT}/Segmented 2>/dev/null | wc -l) masks"
            exit 0
        else
            echo "❌ Dataset structure invalid after extraction"
            echo "Expected:"
            echo "  ${DATA_ROOT}/Original/"
            echo "  ${DATA_ROOT}/Segmented/"
            exit 1
        fi
    else
        echo "❌ Download completed but zip not found"
        exit 1
    fi
else
    echo "❌ Download failed"
    echo ""
    echo "Possible causes:"
    echo "  1. Network timeout (try again)"
    echo "  2. Google Drive quota exceeded (try browser download)"
    echo "  3. Invalid file ID"
    echo ""
    echo "Manual fallback:"
    echo "  1. Visit: https://drive.google.com/file/d/1VTFhKLxdzQAZv3Jj4mZgixI70RzfF68p"
    echo "  2. Download fives_preprocessed.zip"
    echo "  3. Extract to $(pwd)/"
    echo "  4. Run this script again to verify"
    exit 1
fi
