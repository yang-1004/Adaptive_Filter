#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==========================================================================
Adaptive Filtering on KPFM Data — Test Evaluation
==========================================================================

Description:
    Evaluates pre-trained adaptive filter models on test images and
    compares against fixed-parameter Wiener baselines.

    Outputs per scenario (high-noise / low-noise):
        1. Quantitative metrics: PSNR, SSIM, MAE, EPI (Section S6)
        2. Filtered TIFF images for all models and baselines
        3. Publication figure (5-column comparison + line profile)
        4. Cross-model comparison figure (all 5 adaptive models)
        5. Metrics summary CSV

    Pre-trained model weights are loaded from the pretrained/ directory.
    Fixed baselines use the same Gaussian-kernel Wiener implementation
    as the training model, with noise variance (nv) presets matching
    Table S6: weak=0.01, medium=0.1, strong=1.0.

Usage:
    python test_evaluation.py

    To evaluate on your own images, place them in the examples/ directory
    (filenames containing 'high' or 'low' for automatic scenario assignment).

Reference:
    Anonymous, "Adaptive Filtering on KPFM Data: A Deep Neural Network
    Approach," submitted to Machine Learning: Science and Technology.

Author:  Anonymous
Version: 1.0 (public release)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage, stats as sp_stats
from scipy.fft import fft2, fftshift
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import tifffile

# Import model definition from phase3_training
from phase3_training import AdaptiveFilterModel, Config as ModelConfig, DEVICE


# ==========================================================================
# [USER] Path Configuration
# ==========================================================================

# Directory containing pre-trained model weights (.pt files).
PRETRAINED_DIR = os.path.join(os.path.dirname(__file__), 'pretrained')

# Directory containing test TIFF images.
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'examples')

# Output directory for evaluation results.
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

# Filter types to evaluate.
ALL_FILTER_TYPES = ['wiener', 'gaussian', 'mean', 'bilateral', 'tv']

# Primary filter type for the main comparison figure (Fig. 4 layout).
PRIMARY_FILTER = 'wiener'

# Fixed Wiener baseline noise variance (nv) presets (Table S6).
FIXED_WIENER_PRESETS = {
    'weak':   {'nv': 0.01, 'kernel_size': 7},
    'medium': {'nv': 0.1,  'kernel_size': 7},
    'strong': {'nv': 1.0,  'kernel_size': 7},
}

# Line profile row for comparison figures.
LINE_PROFILE_ROW = 128

config = ModelConfig()


# ==========================================================================
# TIFF I/O
# ==========================================================================

def load_tiff(filepath):
    """Load a single-channel TIFF file and return (image_2d, metadata)."""
    try:
        data = tifffile.imread(filepath)
    except Exception as e:
        print(f"  [ERROR] Failed to load {filepath}: {e}")
        return None, None

    if data.ndim == 3:
        data = data[0]
    elif data.ndim != 2:
        return None, None

    image = data.astype(np.float64)
    metadata = {
        'width': image.shape[1], 'height': image.shape[0],
        'original_min': float(np.min(image)),
        'original_max': float(np.max(image)),
        'original_mean': float(np.mean(image)),
        'original_std': float(np.std(image)),
    }
    return image, metadata


def save_filtered_tiff(filtered_image, output_path):
    """Save filtered image as a standard float32 TIFF."""
    tifffile.imwrite(output_path, filtered_image.astype(np.float32))


# ==========================================================================
# Gaussian-Kernel Wiener Filter (same formulation as training model)
# ==========================================================================

def make_gaussian_kernel(k_size, sigma=None):
    """
    Normalized 2D Gaussian kernel with sigma_k = k_size / 6.0 (Section S2).
    """
    if sigma is None:
        sigma = k_size / 6.0
    ax = np.arange(k_size, dtype=np.float64) - k_size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def apply_gaussian_wiener_filter(image, kernel_size, nv):
    """
    Gaussian-kernel Wiener filter with fixed parameters.

    Uses the same formulation as DifferentiableMultiScaleFilter (Section S2):
        H(f) = |K(f)|^2 / (|K(f)|^2 + nv)

    The image is normalized to [0,1] before filtering (matching training),
    then restored to original scale.
    """
    img_min, img_max = image.min(), image.max()
    if img_max - img_min > 1e-8:
        image_norm = (image - img_min) / (img_max - img_min)
    else:
        return image.copy()

    kernel = make_gaussian_kernel(kernel_size)
    pad = kernel_size // 2
    padded = np.pad(image_norm, pad, mode='reflect')

    local_mean = ndimage.convolve(padded, kernel, mode='constant')[pad:-pad, pad:-pad]
    local_sq = ndimage.convolve(padded**2, kernel, mode='constant')[pad:-pad, pad:-pad]
    local_var = np.clip(local_sq - local_mean**2, 1e-10, None)

    wiener_coef = np.clip((local_var - nv) / (local_var + 1e-10), 0, 1)
    output_norm = local_mean + wiener_coef * (image_norm - local_mean)

    return output_norm * (img_max - img_min) + img_min


# ==========================================================================
# Feature Extraction (22 features, mirrors Phase 2)
# ==========================================================================

def extract_features_for_inference(image):
    """
    Extract 22 image-level features for model inference.
    Feature definitions follow Table S1 in the Supplementary Information.
    """
    data = image.astype(np.float64)
    feats = []

    # Group 1: Basic Statistics (5)
    feats.append(float(np.mean(data)))
    feats.append(float(np.std(data)))
    feats.append(float(np.ptp(data)))
    feats.append(float(sp_stats.skew(data.flatten())))
    feats.append(float(sp_stats.kurtosis(data.flatten())))

    # Group 2: Gradient Magnitude (3)
    gy, gx = np.gradient(data)
    grad_mag = np.sqrt(gx**2 + gy**2)
    feats.extend([float(np.mean(grad_mag)), float(np.std(grad_mag)), float(np.max(grad_mag))])

    # Group 3: Laplacian (2)
    lap = ndimage.laplace(data)
    feats.extend([float(np.mean(np.abs(lap))), float(np.std(lap))])

    # Group 4: Frequency Domain Energy Ratios (3)
    f = fft2(data)
    mag = np.abs(fftshift(f))
    total_e = np.sum(mag**2) + 1e-10
    h, w = data.shape
    cy, cx = h // 2, w // 2
    yg, xg = np.ogrid[:h, :w]
    dist = np.sqrt((xg - cx)**2 + (yg - cy)**2) / min(cy, cx)
    feats.append(float(np.sum(mag[dist < 0.2]**2) / total_e))
    feats.append(float(np.sum(mag[(dist >= 0.2) & (dist < 0.5)]**2) / total_e))
    feats.append(float(np.sum(mag[dist >= 0.5]**2) / total_e))

    # Group 5: SNR (Immerkær, 1996; Ref. 26) (1)
    noise_k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    noise_resp = ndimage.convolve(data, noise_k)
    noise_std = np.std(noise_resp) / np.sqrt(20)
    feats.append(float(20 * np.log10(np.std(data) / (noise_std + 1e-10))))

    # Group 6: Local Variance (2)
    lm = ndimage.uniform_filter(data, size=5)
    lsm = ndimage.uniform_filter(data**2, size=5)
    lv = np.clip(lsm - lm**2, 0, None)
    feats.extend([float(np.mean(lv)), float(np.std(lv))])

    # Group 7: Spatial Correlation (3)
    h_corrs = [np.corrcoef(r[:-1], r[1:])[0, 1] for r in data if len(r) > 1]
    v_corrs = [np.corrcoef(c[:-1], c[1:])[0, 1] for c in data.T if len(c) > 1]
    hc = float(np.nanmean(h_corrs)) if h_corrs else 0.0
    vc = float(np.nanmean(v_corrs)) if v_corrs else 0.0
    feats.extend([hc, vc, (hc + vc) / 2])

    # Group 8: Shannon Entropy (1)
    hist_vals, _ = np.histogram(data, bins=256)
    hn = hist_vals / (hist_vals.sum() + 1e-10)
    hn = hn[hn > 0]
    feats.append(float(-np.sum(hn * np.log2(hn))))

    # Group 9: Edge Density (1)
    sx = ndimage.sobel(data, axis=1)
    sy = ndimage.sobel(data, axis=0)
    em = np.sqrt(sx**2 + sy**2)
    feats.append(float(np.mean(em > (np.mean(em) + np.std(em)))))

    # Group 10: Noise Variance Estimate (1)
    feats.append(float(noise_std**2))

    return np.array(feats, dtype=np.float32)


# ==========================================================================
# Model Loading & Inference
# ==========================================================================

def load_pretrained_model(filter_type, pretrained_dir=None):
    """
    Load a pre-trained AdaptiveFilterModel from the pretrained/ directory.

    Args:
        filter_type (str): One of 'wiener', 'gaussian', 'mean', 'bilateral', 'tv'.
        pretrained_dir (str): Path to the directory containing .pt files.

    Returns:
        AdaptiveFilterModel or None if the weight file is not found.
    """
    if pretrained_dir is None:
        pretrained_dir = PRETRAINED_DIR
    model_path = os.path.join(pretrained_dir, f'best_model_dynamic_{filter_type}.pt')

    if not os.path.exists(model_path):
        print(f"  [SKIP] {filter_type} — weight file not found: {model_path}")
        return None

    model = AdaptiveFilterModel(filter_type, num_features=22, enable_local_params=True)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def run_adaptive_model(model, image):
    """
    Filter a single image using a pre-trained adaptive model.

    The image is normalized to [0,1] for inference, then restored to
    original scale. Features are extracted on-the-fly using the same
    22-feature pipeline as Phase 2.

    Returns:
        tuple: (denoised_image, mean_alpha)
    """
    img_min, img_max = float(image.min()), float(image.max())
    if img_max - img_min > 1e-8:
        image_norm = (image - img_min) / (img_max - img_min)
    else:
        image_norm = np.zeros_like(image)

    image_tensor = torch.tensor(image_norm, dtype=torch.float32) \
        .unsqueeze(0).unsqueeze(0).to(DEVICE)
    scale_info = torch.tensor(
        [img_min, img_max, float(np.mean(image)), float(np.std(image))],
        dtype=torch.float32).unsqueeze(0).to(DEVICE)

    raw_features = extract_features_for_inference(image)
    features_tensor = torch.tensor(raw_features, dtype=torch.float32) \
        .unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor, features_tensor, scale_info)
        denoised_norm = output['denoised'].squeeze().cpu().numpy()
        mean_alpha = output['alpha_map'].mean().item()

    denoised = denoised_norm * (img_max - img_min) + img_min
    return denoised, mean_alpha


# ==========================================================================
# Evaluation Metrics (Section S6)
# ==========================================================================

def compute_metrics(original, filtered):
    """
    Compute PSNR, SSIM, MAE, and EPI (Section S6).
    """
    # PSNR (dB)
    mse = np.mean((original - filtered) ** 2)
    if mse < 1e-10:
        psnr = 100.0
    else:
        max_val = np.ptp(original)
        psnr = 20 * np.log10(max_val / np.sqrt(mse)) if max_val > 1e-10 else 0.0

    # SSIM (11x11 Gaussian-weighted windows, Ref. 22)
    C1, C2, ws = 0.01**2, 0.03**2, 11
    o = (original - original.min()) / (np.ptp(original) + 1e-10)
    f = (filtered - filtered.min()) / (np.ptp(filtered) + 1e-10)
    mu1 = ndimage.uniform_filter(o, ws)
    mu2 = ndimage.uniform_filter(f, ws)
    s1 = ndimage.uniform_filter(o**2, ws) - mu1**2
    s2 = ndimage.uniform_filter(f**2, ws) - mu2**2
    s12 = ndimage.uniform_filter(o * f, ws) - mu1 * mu2
    ssim = float(np.mean(
        ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))))

    # MAE (mV when computed on mV-scale images)
    mae = float(np.mean(np.abs(original - filtered)))

    # EPI: Pearson correlation of Sobel gradient magnitude maps
    e_orig = np.sqrt(ndimage.sobel(original, 0)**2 + ndimage.sobel(original, 1)**2)
    e_filt = np.sqrt(ndimage.sobel(filtered, 0)**2 + ndimage.sobel(filtered, 1)**2)
    if np.std(e_orig) > 1e-10 and np.std(e_filt) > 1e-10:
        epi = float(np.corrcoef(e_orig.flatten(), e_filt.flatten())[0, 1])
        if np.isnan(epi):
            epi = 0.0
    else:
        epi = 1.0

    return {'psnr': psnr, 'ssim': ssim, 'mae': mae, 'epi': epi}


# ==========================================================================
# Visualization
# ==========================================================================

def generate_fig4(original, results_dict, scenario_name, output_path,
                  line_row=128, fig_label='FIG. 4'):
    """
    5-column image comparison (Original / Adaptive / Weak / Medium / Strong)
    with line profile. Matches the layout of Fig. 4 in the paper.
    """
    fig = plt.figure(figsize=(18, 8.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 0.8], hspace=0.28)
    gs_img = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0], wspace=0.08)

    images = [
        ('Original', original),
        ('Adaptive Model', results_dict['adaptive']),
        ('Weak', results_dict['weak']),
        ('Medium', results_dict['medium']),
        ('Strong', results_dict['strong']),
    ]

    vmin = np.percentile(original, 2)
    vmax = np.percentile(original, 98)

    for i, (title, img) in enumerate(images):
        ax = fig.add_subplot(gs_img[i])
        im = ax.imshow(img, cmap='viridis', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axhline(line_row, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 4:
            cax = fig.add_axes([ax.get_position().x1 + 0.005, ax.get_position().y0,
                               0.008, ax.get_position().height])
            plt.colorbar(im, cax=cax)

    ax_line = fig.add_subplot(gs[1])
    colors_map = {
        'Original':       ('black',   '-',  1.8),
        'Adaptive Model': ('#7C3AED', '-',  1.5),
        'Weak':           ('#2563EB', '--', 1.0),
        'Medium':         ('#EA580C', '--', 1.0),
        'Strong':         ('#DC2626', '--', 1.0),
    }
    x_pixels = np.arange(original.shape[1])
    for title, img in images:
        c, ls, lw = colors_map[title]
        ax_line.plot(x_pixels, img[line_row, :], color=c, linestyle=ls,
                    linewidth=lw, label=title, alpha=0.85)

    ax_line.set_xlabel('Pixel Position', fontsize=10)
    ax_line.set_ylabel('Value', fontsize=10)
    ax_line.set_title(f'Line Profile at Row {line_row}', fontsize=11)
    ax_line.legend(fontsize=8, loc='best', ncol=2, framealpha=0.9)
    ax_line.grid(True, alpha=0.2)
    ax_line.set_xlim(0, original.shape[1] - 1)
    ax_line.spines['top'].set_visible(False)
    ax_line.spines['right'].set_visible(False)

    fig.suptitle(f'{fig_label}. {scenario_name}: Adaptive Model vs '
                 f'Fixed-Parameter Wiener Filters',
                 fontsize=13, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_cross_model_figure(original, model_results, scenario_name,
                                output_path, line_row=128):
    """Cross-model comparison: all adaptive models side by side."""
    n_models = len(model_results)
    fig = plt.figure(figsize=(3.5 * (n_models + 1), 8.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 0.8], hspace=0.28)
    gs_img = gridspec.GridSpecFromSubplotSpec(1, n_models + 1, subplot_spec=gs[0], wspace=0.08)

    vmin = np.percentile(original, 2)
    vmax = np.percentile(original, 98)

    ax0 = fig.add_subplot(gs_img[0])
    ax0.imshow(original, cmap='viridis', vmin=vmin, vmax=vmax, aspect='equal')
    ax0.set_title('Original', fontsize=10, fontweight='bold')
    ax0.axhline(line_row, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
    ax0.set_xticks([]); ax0.set_yticks([])

    model_colors = ['#7C3AED', '#2563EB', '#EA580C', '#059669', '#DC2626']
    all_items = [('Original', original, 'black', '-', 1.8)]

    for i, (ftype, img) in enumerate(model_results.items()):
        ax = fig.add_subplot(gs_img[i + 1])
        ax.imshow(img, cmap='viridis', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(f'Adaptive\n{ftype.capitalize()}', fontsize=9, fontweight='bold')
        ax.axhline(line_row, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
        ax.set_xticks([]); ax.set_yticks([])
        all_items.append((ftype.capitalize(), img,
                         model_colors[i % len(model_colors)], '-', 1.2))

    ax_line = fig.add_subplot(gs[1])
    x_pixels = np.arange(original.shape[1])
    for label, img, c, ls, lw in all_items:
        ax_line.plot(x_pixels, img[line_row, :], color=c, linestyle=ls,
                    linewidth=lw, label=label, alpha=0.85)
    ax_line.set_xlabel('Pixel Position', fontsize=10)
    ax_line.set_ylabel('Value', fontsize=10)
    ax_line.set_title(f'Cross-Model Line Profile at Row {line_row}', fontsize=11)
    ax_line.legend(fontsize=7, loc='best', ncol=3, framealpha=0.9)
    ax_line.grid(True, alpha=0.2)
    ax_line.set_xlim(0, original.shape[1] - 1)

    fig.suptitle(f'Cross-Model Comparison: {scenario_name}',
                 fontsize=13, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


# ==========================================================================
# Main
# ==========================================================================

def main():
    print("=" * 70)
    print("  Test Evaluation (Pre-trained Models)")
    print("=" * 70)
    print(f"  Pretrained dir : {PRETRAINED_DIR}")
    print(f"  Test data dir  : {TEST_DATA_DIR}")
    print(f"  Results dir    : {RESULTS_DIR}")
    print(f"  Device         : {DEVICE}")
    print(f"  Timestamp      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Step 1: Scan test images
    print("\n[Step 1] Scanning test images...")
    if not os.path.exists(TEST_DATA_DIR):
        print(f"  [ERROR] TEST_DATA_DIR not found: {TEST_DATA_DIR}")
        return

    test_files = {}
    for fname in sorted(os.listdir(TEST_DATA_DIR)):
        if not fname.lower().endswith(('.tiff', '.tif')):
            continue
        fpath = os.path.join(TEST_DATA_DIR, fname)
        nl = fname.lower()
        if 'high' in nl or 'noisy' in nl:
            test_files['high_noise'] = fpath
        elif 'low' in nl or 'clean' in nl:
            test_files['low_noise'] = fpath
        else:
            if 'high_noise' not in test_files:
                test_files['high_noise'] = fpath
            elif 'low_noise' not in test_files:
                test_files['low_noise'] = fpath

    for k, v in test_files.items():
        print(f"  {k}: {os.path.basename(v)}")
    if not test_files:
        print("  [ERROR] No test TIFF files found.")
        return

    # Step 2: Load pre-trained models
    print("\n[Step 2] Loading pre-trained models...")
    models = {}
    for ftype in ALL_FILTER_TYPES:
        m = load_pretrained_model(ftype)
        if m is not None:
            n_params = sum(p.numel() for p in m.parameters())
            models[ftype] = m
            print(f"  Loaded: {ftype} ({n_params:,} params)")

    if not models:
        print("  [ERROR] No pre-trained models found.")
        return

    # Step 3: Kernel consistency check
    print("\n[Step 3] Kernel consistency check...")
    print("  Training model : Gaussian kernel (sigma_k = kernel_size / 6.0)")
    print("  Fixed baseline : Gaussian kernel (same implementation)")
    k = make_gaussian_kernel(7)
    print(f"  Medium baseline kernel (7x7): sum={k.sum():.6f}, center={k[3,3]:.6f}")

    # Step 4: Evaluate per scenario
    all_metrics = []
    scenarios = {
        'high_noise': ('High Noise', 'FIG. 4-1'),
        'low_noise':  ('Low Noise',  'FIG. 4-2'),
    }

    for scenario_key, (scenario_name, fig_label) in scenarios.items():
        if scenario_key not in test_files:
            continue

        fpath = test_files[scenario_key]
        fname_base = os.path.splitext(os.path.basename(fpath))[0]

        print(f"\n{'=' * 60}")
        print(f"  {scenario_name}: {os.path.basename(fpath)}")
        print(f"{'=' * 60}")

        image, metadata = load_tiff(fpath)
        if image is None:
            continue

        features = extract_features_for_inference(image)
        snr = features[13]
        print(f"  Size: {metadata['width']}x{metadata['height']}")
        print(f"  Range: [{metadata['original_min']:.1f}, {metadata['original_max']:.1f}]")
        print(f"  Estimated SNR: {snr:.1f} dB")

        # A. Fixed Gaussian-Wiener baselines
        print(f"\n  [A] Fixed Gaussian-Wiener baselines (Table S6 nv values)...")
        fixed_results = {}
        for strength_name, params in FIXED_WIENER_PRESETS.items():
            filtered = apply_gaussian_wiener_filter(
                image, params['kernel_size'], params['nv'])
            fixed_results[strength_name] = filtered

            m = compute_metrics(image, filtered)
            m.update({'scenario': scenario_name, 'method': f'Fixed_{strength_name}',
                     'model_type': 'baseline'})
            all_metrics.append(m)
            print(f"    {strength_name:8s} (nv={params['nv']:5.3f}): "
                  f"PSNR={m['psnr']:.1f}  SSIM={m['ssim']:.4f}  "
                  f"MAE={m['mae']:.2f}  EPI={m['epi']:.4f}")

            save_filtered_tiff(filtered, os.path.join(
                RESULTS_DIR, f'{fname_base}_fixed_{strength_name}.tiff'))

        # B. Adaptive models
        print(f"\n  [B] Adaptive models (pre-trained)...")
        adaptive_results = {}
        for ftype, model in models.items():
            denoised, mean_alpha = run_adaptive_model(model, image)
            adaptive_results[ftype] = denoised

            m = compute_metrics(image, denoised)
            m.update({'scenario': scenario_name, 'method': f'Adaptive_{ftype}',
                     'model_type': 'adaptive', 'mean_alpha': mean_alpha})
            all_metrics.append(m)
            print(f"    {ftype:12s}: PSNR={m['psnr']:.1f}  SSIM={m['ssim']:.4f}  "
                  f"MAE={m['mae']:.2f}  EPI={m['epi']:.4f}  "
                  f"alpha={mean_alpha:.4f}")

            save_filtered_tiff(denoised, os.path.join(
                RESULTS_DIR, f'{fname_base}_adaptive_{ftype}.tiff'))

        # C. Primary filter comparison figure (Fig. 4 layout)
        if PRIMARY_FILTER in adaptive_results:
            print(f"\n  [C] Generating {fig_label}...")
            fig4_results = {
                'adaptive': adaptive_results[PRIMARY_FILTER],
                'weak': fixed_results['weak'],
                'medium': fixed_results['medium'],
                'strong': fixed_results['strong'],
            }
            generate_fig4(image, fig4_results, scenario_name,
                os.path.join(RESULTS_DIR, f'fig4_{scenario_key}_{PRIMARY_FILTER}.png'),
                LINE_PROFILE_ROW, fig_label)

        # D. Cross-model comparison
        if len(adaptive_results) > 1:
            print(f"\n  [D] Generating cross-model comparison...")
            generate_cross_model_figure(image, adaptive_results, scenario_name,
                os.path.join(RESULTS_DIR, f'fig_cross_model_{scenario_key}.png'),
                LINE_PROFILE_ROW)

    # Step 5: Save metrics CSV
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        csv_path = os.path.join(RESULTS_DIR, 'quantitative_metrics.csv')
        metrics_df.to_csv(csv_path, index=False)
        print(f"\n  Metrics saved: {csv_path}")

        print(f"\n{'=' * 85}")
        print(f"  QUANTITATIVE METRICS SUMMARY (Section IV.C)")
        print(f"{'=' * 85}")
        print(f"  {'Scenario':12s} {'Method':20s} {'PSNR':>8s} "
              f"{'SSIM':>8s} {'MAE':>8s} {'EPI':>8s} {'Alpha':>8s}")
        print(f"  {'-' * 75}")
        for _, row in metrics_df.iterrows():
            alpha = row.get('mean_alpha', '')
            alpha_str = f"{alpha:8.4f}" if isinstance(alpha, float) else f"{'':>8s}"
            print(f"  {row['scenario']:12s} {row['method']:20s} "
                  f"{row['psnr']:8.1f} {row['ssim']:8.4f} "
                  f"{row['mae']:8.2f} {row['epi']:8.4f} {alpha_str}")

    print(f"\n{'=' * 70}")
    print(f"  Test Evaluation Complete")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
