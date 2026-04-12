# Fix: FIVEs Dataset Missing

The test failed because the dataset folder structure is missing:
```
Expected:
  ./fives_preprocessed/
  ├── Original/      (retinal vessel images)
  └── Segmented/     (segmentation masks)

Got: ./fives_preprocessed/ (folder doesn't exist or is empty)
```

## Quick Fix

Run this to download and verify the dataset:

```bash
cd /workspace/MM-UNet-3rd && \
bash ./setup_fives_dataset.sh
```

Then verify it worked:
```bash
ls -la ./fives_preprocessed/
```

You should see:
```
Original/
Segmented/
```

## Then Run the Test

```bash
export HF_TOKEN="YOUR_HF_TOKEN"
bash ./run_complete_test.sh
```

---

## What Happened

The `run_complete_test.sh` tried to run the test before the dataset was fully downloaded/extracted. The separate `setup_fives_dataset.sh` script:

1. ✓ Checks if dataset already exists (skip if yes)
2. ✓ Downloads fives_preprocessed.zip (144 MB) from Google Drive
3. ✓ Extracts it properly
4. ✓ Verifies the folder structure
5. ✓ Reports total files in each folder

---

## Alternative: Manual Download

If the script still fails, you can manually:

1. Visit: https://drive.google.com/file/d/1VTFhKLxdzQAZv3Jj4mZgixI70RzfF68p/view
2. Download `fives_preprocessed.zip`
3. Extract to project root
4. Run test

---

## Verify Dataset Structure

After setup succeeds, check:
```bash
ls ./fives_preprocessed/Original | head
ls ./fives_preprocessed/Segmented | head
```

Should show image files like: `0001.png`, `0002.png`, etc.

---

**Then continue with test:**
```bash
export HF_TOKEN="YOUR_HF_TOKEN"
bash ./run_complete_test.sh
```
