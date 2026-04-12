#!/bin/bash

set -e

echo "========================================"
echo "Download & Preprocess FIVEs Dataset"
echo "========================================"

# Google Drive file ID from the shared link
GOOGLE_DRIVE_FILE_ID="1Z-wcxJuBuMEJwHRPbsdnMWXBulkKPWyx"
ZIP_FILE="fives_raw.zip"
EXTRACT_DIR="fives_raw"
PREPROCESS_OUTPUT="fives_preprocessed_raw"

# Step 1: Download from Google Drive
echo ""
echo "[1/4] Downloading dataset from Google Drive..."
echo "File ID: $GOOGLE_DRIVE_FILE_ID"

if command -v gdown &> /dev/null; then
    gdown "$GOOGLE_DRIVE_FILE_ID" -O "$ZIP_FILE"
else
    echo "gdown not found, installing..."
    pip install gdown -q
    gdown "$GOOGLE_DRIVE_FILE_ID" -O "$ZIP_FILE"
fi

if [ ! -f "$ZIP_FILE" ]; then
    echo "❌ Download failed!"
    exit 1
fi

echo "✓ Downloaded: $(ls -lh $ZIP_FILE | awk '{print $5}')"

# Step 2: Extract zip
echo ""
echo "[2/4] Extracting dataset..."
unzip -q "$ZIP_FILE" -d "$EXTRACT_DIR"
echo "✓ Extracted to: $EXTRACT_DIR"

# Step 3: Create preprocessing script
echo ""
echo "[3/4] Creating preprocessing script for 21 images..."

cat > preprocess_fives_sample.py << 'EOFPYTHON'
#!/usr/bin/env python3
"""
Preprocess FIVEs dataset - sample 21 images only
Converts to 1024x1024 PNG format with Original and Segmented folders
"""

import os
import cv2
import numpy as np
from pathlib import Path
import sys

def preprocess_fives(raw_dir, output_dir, num_images=21):
    """
    Preprocess FIVEs dataset
    
    Args:
        raw_dir: Path to raw dataset directory
        output_dir: Path to output directory
        num_images: Number of images to preprocess (default 21)
    """
    
    # Create output directories
    original_dir = Path(output_dir) / "Original"
    segmented_dir = Path(output_dir) / "Segmented"
    original_dir.mkdir(parents=True, exist_ok=True)
    segmented_dir.mkdir(parents=True, exist_ok=True)
    
    raw_path = Path(raw_dir)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(raw_path.rglob(f'*{ext}'))
        image_files.extend(raw_path.rglob(f'*{ext.upper()}'))
    
    image_files = sorted(list(set(image_files)))  # Remove duplicates
    
    if not image_files:
        print(f"❌ No images found in {raw_dir}")
        return False
    
    print(f"Found {len(image_files)} total images")
    print(f"Processing first {num_images} images...\n")
    
    # Try to pair images with masks
    processed = 0
    
    for img_path in image_files[:num_images]:
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
            
            # Create dummy mask (all black) - replace with actual mask if available
            mask = np.zeros((1024, 1024, 3), dtype=np.uint8)
            segmented_path = segmented_dir / filename
            cv2.imwrite(str(segmented_path), mask)
            
            print(f"[{processed+1:2d}/{num_images}] ✓ {filename}")
            processed += 1
            
        except Exception as e:
            print(f"[{processed+1:2d}/{num_images}] ✗ Error processing {img_path.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Preprocessing Complete")
    print(f"{'='*60}")
    print(f"Processed: {processed} images")
    print(f"Output: {output_dir}")
    print(f"  - Original: {original_dir}")
    print(f"  - Segmented: {segmented_dir}")
    print(f"{'='*60}\n")
    
    return processed > 0

if __name__ == "__main__":
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "fives_raw"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "fives_preprocessed_raw"
    num_images = int(sys.argv[3]) if len(sys.argv) > 3 else 21
    
    success = preprocess_fives(raw_dir, output_dir, num_images)
    sys.exit(0 if success else 1)
EOFPYTHON

python preprocess_fives_sample.py "$EXTRACT_DIR" "$PREPROCESS_OUTPUT" 21

# Step 4: Run test
echo ""
echo "[4/4] Running inference test on 21 preprocessed images..."
echo ""

python test_inference.py

echo ""
echo "========================================"
echo "✓ Pipeline Complete!"
echo "========================================"
echo "Results saved to: ./inference_results/"
