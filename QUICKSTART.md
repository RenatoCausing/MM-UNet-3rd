# Quick Start Guide - MM-UNet Inference Testing

## TL;DR - Just Run This

```bash
bash run_inference_test.sh
```

That's it! Everything else is automatic.

---

## What It Does

✅ Installs Python dependencies  
✅ Downloads 4 model checkpoints (epochs 10, 20, 30, 40)  
✅ Tests 1 image with all 4 epochs (to see progression)  
✅ Tests 20 random images with epoch 40 (to see consistency)  
✅ Saves results with visualizations  

**Total time:** ~10-30 minutes (depends on GPU availability)

---

## Output

Results saved to: `./inference_results/`

Structure:
```
inference_results/
├── config_info.json                          # Settings used
├── single_image_epochs_10_20_30_40/          # 1 image, 4 epochs
│   ├── epoch_10/
│   │   ├── comparison.png                    # original vs segmentation
│   │   └── segmentation_mask.png             # binary mask
│   ├── epoch_20/
│   ├── epoch_30/
│   └── epoch_40/
└── 20_images_epoch_40/                       # 20 images, epoch 40
    ├── image_01_*/
    ├── image_02_*/
    └── ... (20 total)
```

Each image folder contains:
- `comparison.png` - side-by-side original vs segmentation
- `segmentation_mask.png` - binary mask (0 or 255)
- `original_image.png` - the input image

---

## Using Your Own Images

Place test images in any of these directories:
- `./fives_preprocessed/test/`
- `./data/test/`
- `./test_images/`
- `./datasets/test/`

Supported formats: `.jpg`, `.png`, `.jpeg`

If no images found, synthetic test images are created automatically.

---

## Getting the FIVEs Dataset

To use real test images:

```bash
bash DOWNLOAD_FIVES_DATASET.sh
```

Then run inference as usual:
```bash
bash run_inference_test.sh
```

---

## What If Something Fails?

The script handles most issues automatically:

❌ **CUDA error** → Uses CPU fallback  
❌ **Package missing** → Installs it  
❌ **Model import fails** → Uses safe loading  
❌ **Checkpoint download fails** → Skips, continues with next  
❌ **Dataset not found** → Uses synthetic images  

**If you see warnings:** That's OK! The inference continues. Your results are still saved.

---

## Checking It Worked

After running, check for:
1. ✅ Folder `inference_results/` exists
2. ✅ Subfolders `single_image_...` and `20_images_...` exist
3. ✅ PNG images in those folders
4. ✅ File `config_info.json` exists

If all exist → **Success!** Your results are ready.

---

## On the Cloud (AWS, GCP, etc.)

```bash
# SSH into your instance
ssh your-instance

# Navigate to project
cd /path/to/MM-UNet-3rd

# Run it
bash run_inference_test.sh

# Download results (afterwards)
# Use your cloud provider's file download mechanism
```

---

## Performance Tips

**GPU available?**
- Tests run in ~10 minutes
- Around 1-5 seconds per image

**CPU only?**
- Tests run in ~30 minutes  
- Around 10-30 seconds per image

**Low memory?**
- Reduce image size (edit `test_inference.py`)
- Reduce from 20 images to fewer
- Run images one at a time

---

## Verify Installation

To test manually:

```bash
# Check Python
python3 --version

# Check dependencies
python3 -c "import torch; print('PyTorch OK')"
python3 -c "import cv2; print('OpenCV OK')"

# Check HuggingFace
python3 -c "from huggingface_hub import hf_hub_download; print('HF OK')"
```

---

## View Results

Use any image viewer to look at:
- `inference_results/single_image_epochs_10_20_30_40/epoch_40/comparison.png`
- `inference_results/20_images_epoch_40/image_01_*/comparison.png`

Each shows original image on left, segmentation on right.

---

## More Help

For detailed troubleshooting: See `INFERENCE_TROUBLESHOOTING.md`  
For technical details: See `INFERENCE_README.md`  

---

## Common Commands

```bash
# Clean up results and start fresh
rm -rf inference_results/
bash run_inference_test.sh

# Just download dataset
bash DOWNLOAD_FIVES_DATASET.sh

# Just install dependencies
bash INSTALL_MINIMAL.sh

# Run Python script directly (if shell script fails)
python3 test_inference.py
```

---

## That's It!

Nothing else needed. Just run and wait for results! 🎉
