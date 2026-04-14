# Adaptive Filtering on KPFM Data: A Deep Neural Network Approach

Official implementation for a paper submitted to *Machine Learning: Science and Technology* (IOP Publishing).

## Overview

A self-supervised deep learning framework for automatically optimizing classical filter parameters for Kelvin Probe Force Microscopy (KPFM) surface potential images. The model combines an MLP pathway (global parameter prediction from 22 image-level features) with a CNN pathway (pixel-wise local refinement via an alpha map), trained without clean reference images.

**Supported filter types:** Wiener, Gaussian, Mean, Bilateral, Total Variation (TV)

### Architecture

```
Input Image (256×256)
  ├── Structural Complexity Estimator → 3-dim vector
  ├── Global Feature Extractor (MLP: 22 features + 4 scale + 3 complexity → 64-dim latent)
  ├── Local Feature Extractor (CNN: 2-channel [image + Sobel edge map] → 64-ch feature map)
  ├── Spatial Kernel Weight Predictor (74-dim global context + 64-ch CNN → alpha map)
  ├── Multi-Scale Filter Module (7 Gaussian kernels, σ_k = size/6.0)
  └── Alpha Blending: I_out = α ⊙ I_filtered + (1−α) ⊙ I_original
```

~240K trainable parameters per filter type, trained on 13,218 real-world KPFM measurements.

## Requirements

- Python >= 3.8
- CUDA-capable GPU recommended (tested on NVIDIA A100)

```bash
pip install -r requirements.txt
```

## Setup

After cloning the repository, extract the zip archives to create the required directory structure:

```bash
unzip pretrained.zip -d pretrained/
unzip test_images.zip -d examples/
```

For training from scratch, also extract the training data:

```bash
unzip data.zip -d data/raw/
```

The resulting structure should be:

```
├── phase2_feature_extraction.py   # Feature extraction & PSD analysis
├── phase3_training.py             # Model training
├── test_evaluation.py             # Evaluation with pre-trained models
├── pretrained/                    # (from pretrained.zip)
│   ├── best_model_dynamic_wiener.pt
│   ├── best_model_dynamic_gaussian.pt
│   ├── best_model_dynamic_mean.pt
│   ├── best_model_dynamic_bilateral.pt
│   └── best_model_dynamic_tv.pt
├── examples/                      # (from test_images.zip)
│   ├── highnoise.tiff
│   └── lownoise.tiff
├── data/raw/                      # (from data.zip, for training only)
│   ├── 00001.tiff
│   ├── ...
│   └── 13218.tiff
├── requirements.txt
├── LICENSE
└── README.md
```

## Data

The raw KPFM measurements were acquired using Park Systems instruments, which store data in a proprietary TIFF format. These instrument-specific files have been converted to standard single-channel float32 TIFF for general compatibility, and only images that passed the initial quality classification (Phase 1) are provided. The code in this repository starts from Phase 2 (feature extraction), as the provided dataset has already undergone Phase 1 screening.

- **`data.zip`** — 13,218 training images (256×256, single-channel float32 TIFF), sequentially named (`00001.tiff` to `13218.tiff`).
- **`test_images.zip`** — Two test images (`highnoise.tiff`, `lownoise.tiff`) acquired from the same sample under different measurement conditions.
- **`pretrained.zip`** — Pre-trained model weights for all five filter types.

All image files are standard single-channel float32 TIFF, readable by any image processing library (`tifffile`, PIL, OpenCV, etc.).

## Quick Start: Evaluation with Pre-trained Models

Run the evaluation script directly — no training required:

```bash
python test_evaluation.py
```

This loads the five pre-trained models from `pretrained/`, evaluates them on the example images in `examples/`, and produces:

- `results/quantitative_metrics.csv` — PSNR, SSIM, MAE, EPI for all methods
- `results/fig4_*.png` — Publication-quality comparison figures
- `results/fig_cross_model_*.png` — Cross-model comparison figures
- `results/*_adaptive_*.tiff` — Filtered output images
- `results/*_fixed_*.tiff` — Fixed-parameter baseline outputs

To evaluate on your own images, place TIFF files in the `examples/` directory.

## Training from Scratch

The provided dataset has already passed Phase 1 quality classification. Training starts from Phase 2:

```bash
# Step 1: Extract 22 image-level features and perform PSD analysis
python phase2_feature_extraction.py

# Step 2: Train adaptive filter models (all 5 filter types by default)
python phase3_training.py
```

Each script has a `[USER] Path Configuration` section at the top for data paths.

Key training hyperparameters:

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Scheduler | CosineAnnealingWarmRestarts (T₀=20, T_mult=2) |
| Epochs | 100 (early stopping, patience=15) |
| Batch size | 16 |
| Train / Val / Test | 85% / 10% / 5% |

## Loss Function

The composite loss function comprises five terms with fixed weighting coefficients:

| # | Term | λ | Description |
|---|------|-----|-------------|
| i | L_fq | 0.25 | Frequency-band quality (6 bands with learned per-band weighting) |
| ii | L_ssim | 0.25 | SSIM structural preservation (threshold = 0.85) |
| iii | L_edge | 0.20 | Edge preservation (Sobel mask selecting prominent edge structures) |
| iv | L_min | 0.10 | Dynamic minimum filtering (prevents identity mapping) |
| v | L_smooth | 0.10 | Spatial smoothness (Total Variation of parameter maps) |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| PSNR | Degree of alteration relative to original (dB) |
| SSIM | Structural similarity (11×11 Gaussian-weighted windows) |
| MAE | Mean absolute error (mV on mV-scale images) |
| EPI | Edge Preservation Index (Pearson correlation of Sobel gradient maps) |

## Fixed-Parameter Baselines

| Strength | Noise Variance (nv) | Kernel |
|----------|---------------------|--------|
| Weak | 0.01 | 7×7 Gaussian |
| Medium | 0.1 | 7×7 Gaussian |
| Strong | 1.0 | 7×7 Gaussian |

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
