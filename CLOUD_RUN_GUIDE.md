# Quick Cloud Run Guide

## Option 1: Automated (Full Pipeline)

**One command that does everything:**

```bash
cd /workspace/MM-UNet-3rd && \
export HF_TOKEN="YOUR_HF_TOKEN" && \
bash ./run_complete_test.sh
```

This:
1. Installs dependencies
2. Downloads dataset (if needed)
3. Pulls best_model.pth from HF
4. Runs tests
5. Saves results

---

## Option 2: Manual Steps (Safer)

If automated fails, do this:

### Step 1: Setup Dataset
```bash
cd /workspace/MM-UNet-3rd
bash ./setup_fives_dataset.sh
```

Verify:
```bash
ls ./fives_preprocessed/
```

Should show: `Original/` and `Segmented/`

### Step 2: Setup Dependencies
```bash
bash ./INSTALL_PYTHON310.sh
```

### Step 3: Run Test
```bash
export HF_TOKEN="YOUR_HF_TOKEN"
python test_fives.py \
  --hf_repo "23LebronJames23/MM-UNet" \
  --hf_filename "best_model.pth" \
  --data_root "./fives_preprocessed" \
  --output_dir "./test_results_best_model" \
  --batch_size 4 \
  --num_workers 4 \
  --image_size 1024 \
  --seed 42 \
  --test_ratio 0.05 \
  --gpu "0" \
  --norm_mean 0.485 0.456 0.406 \
  --norm_std 0.229 0.224 0.225 \
  --save_predictions
```

## What It Does (Step by Step)

```
[1/4] Install Python dependencies
      └─ Runs INSTALL_PYTHON310.sh (or INSTALL_MINIMAL.sh)

[2/4] Download FIVEs dataset
      └─ Runs DOWNLOAD_FIVES_DATASET.sh
      └─ Or manual gdown if script not available

[3/4] Detect Python interpreter
      └─ Finds python3.10, python3, or python

[4/4] Run test with best_model.pth from HF
      └─ Downloads checkpoint from Hugging Face
      └─ Uses deterministic 95/5 split (seed 42)
      └─ Runs on-the-fly preprocessing (no cache)
      └─ Computes metrics: Accuracy, Precision, Recall, F1, AUC, Dice
      └─ Saves predictions (optional)
```

## Output Files

After a successful run, you'll find:

```
./test_results_best_model/
├── test_results.json                      # Metrics
├── fives_split_seed42_ratio005.json       # Training/test split info
└── predictions/                           # Segmentation predictions (if enabled)
    ├── pred_0000.png
    ├── mask_0000.png
    ├── pred_0001.png
    ├── mask_0001.png
    └── ... (one pair per test image)
```

View metrics:
```bash
cat ./test_results_best_model/test_results.json | python -m json.tool
```

## Environment Variables (Optional)

Set before running to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | (empty) | Hugging Face API token |
| `HF_REPO` | `23LebronJames23/MM-UNet` | HF repo with checkpoint |
| `HF_FILENAME` | `best_model.pth` | Checkpoint filename |
| `DATA_ROOT` | `./fives_preprocessed` | Dataset path |
| `OUTPUT_DIR` | `./test_results_best_model` | Results output dir |
| `GPU` | `0` | GPU device ID |

Example:
```bash
export HF_TOKEN="hf_XXX..."
export GPU="1"
bash ./run_complete_test.sh
```

## Troubleshooting

### gdown --id Error
Fixed ✓ - Updated to use newer gdown syntax (no --id flag)

### Dataset Download Fails
- Check internet connectivity
- Verify Google Drive file ID: `1VTFhKLxdzQAZv3Jj4mZgixI70RzfF68p`
- Try manual download and extract

### HF Checkpoint Download Fails
- Verify HF_TOKEN is valid
- Check repo exists: `23LebronJames23/MM-UNet`
- Verify checkpoint file: `best_model.pth`

### Out of Memory
- Reduce batch size: `export BATCH_SIZE="2"`
- Reduce num_workers: `export NUM_WORKERS="2"`
- Use smaller GPU or CPU

## Expected Timing

- Install: ~5-10 minutes
- Dataset download: ~10 minutes (Google Drive)
- Checkpoint download: ~1-2 minutes (HF)
- Testing: ~10-30 minutes (depends on GPU)
- **Total: ~30-60 minutes**

---

**That's it! Just run the command and wait for results.** ✓
