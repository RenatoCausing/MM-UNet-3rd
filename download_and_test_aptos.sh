#!/bin/bash

set -e

echo "========================================"
echo "Download & Test APTOS 2019 Dataset"
echo "========================================"

# APTOS 2019 dataset from Kaggle
KAGGLE_DATASET="mariaherrerot/aptos2019"
ZIP_FILE="aptos2019.zip"
EXTRACT_DIR="aptos2019_raw"
PREPROCESS_OUTPUT="aptos_preprocessed"

# Step 1: Install kaggle CLI if needed
echo ""
echo "[1/5] Checking Kaggle CLI..."
if ! command -v kaggle &> /dev/null; then
    echo "Installing kaggle..."
    pip install kaggle -q
fi

# Step 2: Download from Kaggle
echo ""
echo "[2/5] Downloading APTOS 2019 dataset from Kaggle..."
echo "Dataset: $KAGGLE_DATASET"
kaggle datasets download -d "$KAGGLE_DATASET" -p .

if [ ! -f "$ZIP_FILE" ]; then
    echo "❌ Download failed!"
    echo "Make sure:"
    echo "  1. You have a Kaggle account"
    echo "  2. Create API token: https://www.kaggle.com/settings/account"
    echo "  3. Place ~/.kaggle/kaggle.json in your home directory"
    exit 1
fi

echo "✓ Downloaded: $(ls -lh $ZIP_FILE | awk '{print $5}')"

# Step 3: Extract
echo ""
echo "[3/5] Extracting dataset..."
unzip -q "$ZIP_FILE" -d "$EXTRACT_DIR" || unzip "$ZIP_FILE" -d "$EXTRACT_DIR"
echo "✓ Extracted to: $EXTRACT_DIR"

# Step 4: Create preprocessing script
echo ""
echo "[4/5] Creating preprocessing script..."

cat > preprocess_aptos.py << 'EOFPYTHON'
#!/usr/bin/env python3
"""
Preprocess APTOS 2019 dataset for testing
Creates Original folder with 1024x1024 images
"""

import os
import cv2
import numpy as np
from pathlib import Path
import sys

def preprocess_aptos(raw_dir, output_dir, num_images=21):
    """
    Preprocess APTOS dataset
    
    Args:
        raw_dir: Path to APTOS dataset directory
        output_dir: Path to output directory
        num_images: Number of images to preprocess
    """
    
    # Create output directory
    original_dir = Path(output_dir) / "Original"
    original_dir.mkdir(parents=True, exist_ok=True)
    
    raw_path = Path(raw_dir)
    
    # APTOS structure: contains train_images and test_images folders
    image_sources = [
        raw_path / "train_images",
        raw_path / "test_images",
    ]
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    image_files = []
    
    for source in image_sources:
        if source.exists():
            for ext in image_extensions:
                image_files.extend(source.glob(f'*{ext}'))
                image_files.extend(source.glob(f'*{ext.upper()}'))
    
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print(f"❌ No images found in {raw_dir}")
        print(f"  Checked: {[str(s) for s in image_sources if s.exists()]}")
        return False
    
    print(f"Found {len(image_files)} total images")
    print(f"Processing first {num_images} images...\n")
    
    processed = 0
    
    for img_path in image_files:
        if processed >= num_images:
            break
        
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠ Could not read: {img_path.name}")
                continue
            
            # Resize to 1024x1024
            img_resized = cv2.resize(img, (1024, 1024))
            
            # Save as original
            filename = f"image_{processed+1:04d}.png"
            original_path = original_dir / filename
            cv2.imwrite(str(original_path), img_resized)
            
            print(f"[{processed+1:2d}/{num_images}] ✓ {img_path.name} -> {filename}")
            processed += 1
            
        except Exception as e:
            print(f"⚠ Error processing {img_path.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Preprocessing Complete")
    print(f"{'='*60}")
    print(f"Processed: {processed} images")
    print(f"Output: {original_dir}")
    print(f"{'='*60}\n")
    
    return processed > 0

if __name__ == "__main__":
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "aptos2019_raw"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "aptos_preprocessed"
    num_images = int(sys.argv[3]) if len(sys.argv) > 3 else 21
    
    success = preprocess_aptos(raw_dir, output_dir, num_images)
    sys.exit(0 if success else 1)
EOFPYTHON

python preprocess_aptos.py "$EXTRACT_DIR" "$PREPROCESS_OUTPUT" 21

# Step 5: Run test
echo ""
echo "[5/5] Running inference test on APTOS images..."
echo ""

python test_inference_aptos.py

echo ""
echo "========================================"
echo "✓ Pipeline Complete!"
echo "========================================"
echo "Results saved to: ./inference_results/"
