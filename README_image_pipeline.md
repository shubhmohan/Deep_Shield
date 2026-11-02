# DeepShield — Image Detection Pipeline

## Status update (Day 2 of Week 1)

Real-world validation revealed both third-party checkpoints we initially
tried were unreliable:

| Checkpoint | Result | Verdict |
|---|---|---|
| `Yashikaysn29/deepshield` (EfficientNet) | Output exactly `0.000` on all 36 test images, real and fake alike | **Broken/degenerate — do not use** |
| `dima806/deepfake_vs_real_image_detection` (ViT) | Caught only 2/18 known fakes | **Weak — concept drift, as its own model card warned** |

**Current strategy: train both branches ourselves** on the Kaggle
"140k Real and Fake Faces" dataset, which matches the StyleGAN-style faces
in our test set. This is slower than importing a checkpoint but gives you
full control, real metrics for your report, and no dependency on unverified
third-party code.

## What's in this repo

| File | Purpose |
|---|---|
| `preprocessors/image_prep.py` | Loads, validates, resizes (224×224), normalizes images |
| `models/train_efficientnet.py` | **Run this first** — fine-tunes EfficientNet-B4 on your GPU |
| `models/train_vit.py` | **Run this second** — fine-tunes ViT-B/16 on the same dataset |
| `models/efficientnet_model.py` | Loads your trained checkpoint (`./checkpoints/efficientnet_b4_deepfake.pt`) |
| `models/vit_model.py` | Loads your trained checkpoint (`./checkpoints/vit_deepfake.pt`), or falls back to the weak remote checkpoint if you haven't trained yet |
| `models/ensemble.py` | Weighted-average voting → verdict, confidence, risk level |
| `detectors/explainer.py` | Grad-CAM heatmap generation |
| `detectors/image_detector.py` | Orchestrator — the function your FastAPI endpoint calls |
| `tests/validate_pipeline.py` | Runs the full pipeline against `sample_data/real/` and `sample_data/fake/`, reports accuracy with a per-model score breakdown |

## What to do next, in order

### 1. Get the dataset
Download the [Kaggle 140k Real and Fake Faces dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
(~4GB, needs a free Kaggle account). Extract it so you have:
```
data/real_vs_fake/real-vs-fake/train/real/*.jpg
data/real_vs_fake/real-vs-fake/train/fake/*.jpg
data/real_vs_fake/real-vs-fake/valid/real|fake/*.jpg
```

### 2. Train EfficientNet-B4
```bash
python models/train_efficientnet.py
```
Expect a few hours on a consumer GPU for 8 epochs. This saves the best
checkpoint to `./checkpoints/efficientnet_b4_deepfake.pt` automatically —
you don't need to babysit it, just let it run (overnight is fine).

### 3. Train ViT-B/16
```bash
python models/train_vit.py
```
Saves to `./checkpoints/vit_deepfake.pt`.

### 4. Re-run validation
```bash
python tests/validate_pipeline.py
```
With both branches trained on data that matches your test fakes, expect
accuracy well above the 50% baseline we saw with the third-party checkpoints.
If it's still low, the per-model breakdown (`vit:` / `eff:` columns) will
tell you which branch to investigate first.

### 5. Report back before moving to video pipeline
Send the new validation output. Once accuracy looks solid, we move to
Week 1 Day 4-5: frame extraction + MediaPipe + video-level aggregation.
