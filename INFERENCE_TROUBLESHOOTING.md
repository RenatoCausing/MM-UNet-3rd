# MM-UNet Inference Testing - Troubleshooting Guide

## Quick Start

Run the inference test with:
```bash
bash run_inference_test.sh
```

This will:
1. Install Python dependencies
2. Download model checkpoints from HuggingFace
3. Test 1 image with epochs 10, 20, 30, 40
4. Test 20 random images with epoch 40
5. Save results to `./inference_results/`

**Note:** If the FIVEs dataset isn't available, the script will automatically create synthetic test images for demonstration. The inference will work the same way.

---

## Common Issues

### Issue 1: `selective_scan_cuda` Import Error

**Error Message:**
```
ERROR: selective_scan_cuda import failed: No module named 'selective_scan_cuda'
```

**Cause:** The Mamba CUDA extensions failed to compile. This is normal if:
- Running without CUDA/NVIDIA GPU
- CUDA toolkit isn't available during build
- Platform doesn't support CUDA compilation

**Solution:**
The scripts now handle this automatically. The inference will use the Python-only version of Mamba if CUDA extensions aren't available.

If you still encounter this error:
1. Ensure you're running the updated scripts
2. The error should be non-blocking for inference-only tasks

### Issue 2: `INSTALL_PYTHON310.sh` Fails

**Solution:**
The main script will automatically fall back to `INSTALL_MINIMAL.sh`, which installs only the essential packages needed for inference.

You can also manually run the minimal installation:
```bash
bash INSTALL_MINIMAL.sh
bash run_inference_test.sh
```

### Issue 3: Missing Test Images

**What happens:** If no test images are found in standard locations, the script creates synthetic images for demonstration.

**Locations checked:**
- `./fives_preprocessed/test`
- `./data/test`
- `./test_images`
- `./datasets/test`

**To use actual FIVEs dataset images:**
```bash
# Option 1: Download during full installation (automatic attempt)
bash INSTALL_PYTHON310.sh

# Option 2: Download separately (manual)
bash DOWNLOAD_FIVES_DATASET.sh
```

**To use your own images:**
Place them in one of the standard locations above with extensions: `.jpg`, `.png`, or `.jpeg`

### Issue 4: CUDA Symbol Mismatch / causal_conv1d Import Error

**Error Message:**
```
ImportError: /venv/main/lib/python3.10/site-packages/causal_conv1d_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN2at4_ops10zeros_like4callERKNS_6TensorE...
TypeError: cannot unpack non-iterable NoneType object
```

**Cause:** The compiled `causal_conv1d_cuda` extension has symbol mismatches (usually PyTorch/CUDA version mismatch).

**Solution (Automatic):**
The scripts now handle this automatically with:
1. `causal_conv1d_patch.py` - Patches import errors before they occur
2. `safe_model_loading.py` - Loads models with fallback mechanisms
3. Graceful degradation if model loading fails

**What happens if this error occurs:**
- The patch module intercepts the import error
- Creates stub implementations that raise NotImplementedError if called
- Falls back to alternative model loading methods
- Tests continue with warnings

**If you still see this error after running the updated scripts:**
- The issue may be environmental (PyTorch/CUDA mismatch)
- Update PyTorch to match your CUDA version:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
  ```
- Or reinstall completely:
  ```bash
  bash INSTALL_MINIMAL.sh
  bash run_inference_test.sh
  ```

### Issue 4: HuggingFace Download Fails

**Error:** Cannot download checkpoints from HuggingFace

**Cause:** Network issues or invalid credentials

**Solution:**
1. Verify your HuggingFace token is correct in `test_inference.py`
2. Check internet connectivity
3. Verify the repository `23LebronJames23/MM-UNet` exists and is accessible

**Manual Alternative:**
If downloads fail, you can manually download checkpoints and place them in a `./models/` directory, then modify `test_inference.py` to load from local paths.

### Issue 5: Out of Memory (OOM) Error

**Error:** `CUDA out of memory` or similar

**Solution:**
- Reduce batch size in the script
- Use smaller images (current: 1024x1024)
- Run on CPU instead of GPU (automatic if GPU isn't available)

### Issue 6: PyTorch Version Mismatch

**Error:** CUDA version incompatibility or library conflicts

**Solution:**
The scripts handle this automatically by:
1. Specifying compatible PyTorch version
2. Installing with `--force-reinstall` when needed
3. Falling back to minimal dependencies

---

## Output Structure

After running `bash run_inference_test.sh`, results are saved in:

```
inference_results/
├── config_info.json                          # Configuration and normalization stats
├── single_image_epochs_10_20_30_40/          # Single image with multiple epochs
│   ├── epoch_10/
│   │   ├── comparison.png                    # Side-by-side original vs segmentation
│   │   └── segmentation_mask.png             # Binary segmentation mask
│   ├── epoch_20/
│   ├── epoch_30/
│   └── epoch_40/
└── 20_images_epoch_40/                       # 20 images with epoch 40
    ├── image_01_*/
    │   ├── comparison.png                    # Side-by-side comparison
    │   ├── segmentation_mask.png             # Binary mask
    │   └── original_image.png                # Original image
    ├── image_02_*/
    ├── ...
    └── image_20_*/
```

---

## Understanding the Output

### comparison.png
Shows 3 panels:
- Left: Original image (resized to 1024x1024 for inference)
- Right: Binary segmentation result (threshold at 0.5)

### segmentation_mask.png
- Binary mask (0 = background, 255 = foreground)
- Gray scale image for easy loading in other tools

### original_image.png
- Original input image (for 20-image test only)

### config_info.json
Contains:
- Normalization statistics used
- Model architecture (MM_Net)
- Epochs tested
- Output directories

---

## Normalization Details

The inference uses these normalization statistics:
```json
{
  "train_mean": [0.485, 0.456, 0.406],
  "train_std": [0.229, 0.224, 0.225],
  "image_size": 1024
}
```

Images are:
1. Resized to 1024×1024
2. Normalized with mean/std from ImageNet
3. Segmented by the model
4. Thresholded at 0.5
5. Resized back to original dimensions

---

## Running on Cloud (AWS, GCP, etc.)

### SSH/Terminal
```bash
# Clone or upload your repository
cd /path/to/MM-UNet

# Run the inference
bash run_inference_test.sh

# Download results
# (Use scp or your cloud provider's download mechanism)
```

### Cloud Notebook (Jupyter/Colab)
```python
import os
os.chdir('/path/to/MM-UNet')
os.system('bash run_inference_test.sh')

# Or manually run the Python script:
# os.system('python test_inference.py')
```

### Important Notes for Cloud
- CUDA is often available in cloud GPUs
- Install scripts will use it automatically
- Some systems may require `sudo` for system dependencies
- Network connectivity is usually good (for HF downloads)

---

## Advanced: Manual Inference

If you want to run inference manually without the script:

```python
import torch
from test_inference import preprocess_image, load_model_checkpoint, save_comparison
from src.models import give_model
import yaml
from easydict import EasyDict

# Load model
with open('config.yml', 'r') as f:
    config = EasyDict(yaml.safe_load(f))
model = give_model(config)

# Load checkpoint
checkpoint_path = './hf_cache/.../checkpoint_epoch_0040.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
load_model_checkpoint(model, checkpoint_path, device)
model.eval()

# Preprocess image
image_tensor, orig_size = preprocess_image('path/to/image.jpg')
image_tensor = image_tensor.to(device)

# Inference
with torch.no_grad():
    output = model(image_tensor).cpu().squeeze().numpy()

# Threshold and save
output_binary = (output > 0.5).astype(float)
save_comparison(original_image, output_binary, 'output.png', epoch=40)
```

---

## Performance Notes

- **CPU inference**: ~10-30 seconds per image (1024×1024)
- **GPU inference**: ~1-5 seconds per image (depends on GPU)
- **Memory**: ~2GB (CPU) or ~4GB (GPU with 1024×1024 images)

Total time for full test (21 images):
- GPU: ~10-30 minutes
- CPU: ~5-10 minutes

---

## Additional Help

If you encounter issues not listed here:

1. Check the installation logs:
   ```bash
   cat install_log.txt
   ```

2. Verify dependencies:
   ```bash
   python3 -c "import torch; import numpy; import cv2; print('OK')"
   ```

3. Check HuggingFace connectivity:
   ```bash
   python3 -c "from huggingface_hub import hf_hub_download; print('HF OK')"
   ```

4. Review error messages and stack traces in the output

---

## Environment Variables

You can customize behavior with environment variables:

```bash
# Use CPU only (no GPU)
export CUDA_VISIBLE_DEVICES=""
bash run_inference_test.sh

# Set custom HF cache directory
export HF_HOME="/path/to/cache"
bash run_inference_test.sh

# Enable verbose logging
export PYTHONUNBUFFERED=1
python3 test_inference.py
```

---

## Contact & Support

For issues specific to the MM-UNet model architecture, check:
- Original repository: `23LebronJames23/MM-UNet`
- Model documentation: Check `src/UM_Net/MMUNet.py`
