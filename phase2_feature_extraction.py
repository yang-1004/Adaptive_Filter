#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==========================================================================
Adaptive Filtering on KPFM Data — Phase 2: Feature Extraction & Analysis
==========================================================================

Description:
    Extracts 22 image-level features from normal images identified by
    Phase 1, then performs:
        1. Feature extraction (22 features per image)
        2. PSD analysis with Wiener filter response characterization
        3. Feature-filter response Pearson correlation with bootstrap
           robustness scoring (R in the paper)
        4. IQR-based outlier detection and K-Means clustering
        5. Publication-quality figure generation (Fig. 1a-1c)

    The output CSV files (all_features.csv and feature_columns.json)
    serve as the primary input for Phase 3 training.

Usage:
    1. Run Phase 1 first.
    2. Set CLASSIFICATION_DIR and OUTPUT_DIR below.
    3. Run:  python phase2_feature_extraction.py

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
import seaborn as sns
from scipy import stats, ndimage
from scipy.fft import fft2, fftshift
from scipy.signal import wiener
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import tifffile


# ==========================================================================
# [USER] Path Configuration
# ==========================================================================

# Phase 1 output directory (must contain normal/ and analysis/).
CLASSIFICATION_DIR = './output/01_Classification'

# Phase 2 output directory.
OUTPUT_DIR = './output/02_Feature_Analysis'


# ==========================================================================
# Analysis Configuration
# ==========================================================================

# Wiener filter noise variance (nv) presets for PSD characterization.
# These values match Table S6 in the Supplementary Information:
#   weak: nv = 0.01, medium: nv = 0.1, strong: nv = 1.0
# scipy.signal.wiener mysize parameter controls the local window size.
WIENER_PRESETS = {
    'weak':   {'mysize': 3,  'nv': 0.01},
    'medium': {'mysize': 7,  'nv': 0.1},
    'strong': {'mysize': 13, 'nv': 1.0},
}

# Normalized frequency bands [0, 1], matching Section III.D and Section S4.
# Used in the frequency-band quality loss (L_fq).
FREQUENCY_BANDS = {
    'signal':     (0.00, 0.05),
    'transition': (0.05, 0.15),
    'low_noise':  (0.15, 0.30),
    'mid_noise':  (0.30, 0.50),
    'high_noise': (0.50, 0.80),
    'very_high':  (0.80, 1.00),
}

PSD_SAMPLE_SIZE = 500       # Number of images sampled for PSD analysis
BOOTSTRAP_ITERATIONS = 1000  # Bootstrap iterations for robustness scoring
N_CLUSTERS = 8               # K-Means clusters for diversity verification
RANDOM_SEED = 42


# ==========================================================================
# TIFF I/O
# ==========================================================================

def load_tiff(filepath):
    """
    Load a single-channel TIFF file and return a 2D float64 array with metadata.

    For Park Systems proprietary TIFF, replace with pspylib and return
    the same (image_2d, metadata_dict) interface.
    """
    try:
        data = tifffile.imread(filepath)
    except Exception:
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


# ==========================================================================
# Feature Extraction (22 image-level features)
# ==========================================================================
# Feature definitions correspond to Table S1 in the Supplementary Information.
# Features are grouped as: basic statistics, gradient, Laplacian, frequency
# domain, SNR, local variance, spatial correlation, entropy, edge density,
# and noise variance estimate.

FEATURE_COLUMNS = [
    # Group 1: Basic Statistics (5)
    'mean', 'std', 'data_range', 'skewness', 'kurtosis',
    # Group 2: Gradient (3)
    'mean_gradient', 'std_gradient', 'max_gradient',
    # Group 3: Laplacian (2)
    'mean_abs_laplacian', 'std_laplacian',
    # Group 4: Frequency Domain Energy Ratios (3)
    'low_freq_energy_ratio', 'mid_freq_energy_ratio', 'high_freq_energy_ratio',
    # Group 5: SNR — Laplacian-based noise estimation (Immerkær, 1996) (1)
    'snr_estimated_dB',
    # Group 6: Local Variance (2)
    'mean_local_variance', 'std_local_variance',
    # Group 7: Spatial Correlation (3)
    'horizontal_correlation', 'vertical_correlation', 'mean_correlation',
    # Group 8: Shannon Entropy (1)
    'entropy',
    # Group 9: Edge Density — Sobel-based (1)
    'edge_density',
    # Group 10: Noise Variance Estimate (1)
    'noise_variance_estimate',
]

assert len(FEATURE_COLUMNS) == 22, f"Expected 22 features, got {len(FEATURE_COLUMNS)}"


def extract_features(image, filepath):
    """
    Extract 22 image-level features from a single image.

    These features serve as input to the Global Feature Extractor
    (MLP pathway) described in Section III.C.

    Args:
        image (np.ndarray): 2D image array (float64).
        filepath (str): File path (stored as metadata, not used in computation).

    Returns:
        dict: Feature values keyed by FEATURE_COLUMNS names.
    """
    feats = {'filepath': filepath, 'filename': os.path.basename(filepath)}
    data = image.astype(np.float64)

    # --- Group 1: Basic Statistics ---
    feats['mean']       = float(np.mean(data))
    feats['std']        = float(np.std(data))
    feats['data_range'] = float(np.ptp(data))
    feats['skewness']   = float(stats.skew(data.flatten()))
    feats['kurtosis']   = float(stats.kurtosis(data.flatten()))

    # --- Group 2: Gradient Magnitude ---
    gy, gx = np.gradient(data)
    grad_mag = np.sqrt(gx**2 + gy**2)
    feats['mean_gradient'] = float(np.mean(grad_mag))
    feats['std_gradient']  = float(np.std(grad_mag))
    feats['max_gradient']  = float(np.max(grad_mag))

    # --- Group 3: Laplacian ---
    # High mean |Laplacian| indicates noisy or texturally complex images.
    lap = ndimage.laplace(data)
    feats['mean_abs_laplacian'] = float(np.mean(np.abs(lap)))
    feats['std_laplacian']      = float(np.std(lap))

    # --- Group 4: Frequency Domain Energy Ratios ---
    # 2D FFT followed by radial energy ratio computation.
    f_transform = fft2(data)
    f_shifted   = fftshift(f_transform)
    magnitude   = np.abs(f_shifted)
    total_energy = np.sum(magnitude**2) + 1e-10

    h, w = data.shape
    cy, cx = h // 2, w // 2
    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
    max_radius = min(cy, cx)
    normalized_dist = dist / max_radius

    feats['low_freq_energy_ratio']  = float(np.sum(magnitude[normalized_dist < 0.2]**2) / total_energy)
    feats['mid_freq_energy_ratio']  = float(np.sum(magnitude[(normalized_dist >= 0.2) & (normalized_dist < 0.5)]**2) / total_energy)
    feats['high_freq_energy_ratio'] = float(np.sum(magnitude[normalized_dist >= 0.5]**2) / total_energy)

    # --- Group 5: SNR Estimation (Immerkær, 1996) ---
    # noise_std estimated via Laplacian convolution, Ref. 26 in main text.
    noise_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    noise_response = ndimage.convolve(data, noise_kernel)
    noise_std = np.std(noise_response) / np.sqrt(20)
    signal_std = np.std(data)
    feats['snr_estimated_dB'] = float(20 * np.log10(signal_std / (noise_std + 1e-10)))

    # --- Group 6: Local Variance ---
    # Captures spatial heterogeneity: high local variance in flat regions
    # indicates noise, while near edges it indicates structure.
    local_mean    = ndimage.uniform_filter(data, size=5)
    local_sq_mean = ndimage.uniform_filter(data**2, size=5)
    local_var     = np.clip(local_sq_mean - local_mean**2, 0, None)
    feats['mean_local_variance'] = float(np.mean(local_var))
    feats['std_local_variance']  = float(np.std(local_var))

    # --- Group 7: Spatial Correlation ---
    # Row-wise (horizontal) and column-wise (vertical) adjacent-pixel
    # Pearson correlation. High correlation indicates smooth/structured data.
    h_corrs = []
    for row in data:
        if len(row) > 1:
            c = np.corrcoef(row[:-1], row[1:])[0, 1]
            if not np.isnan(c):
                h_corrs.append(c)
    v_corrs = []
    for col in data.T:
        if len(col) > 1:
            c = np.corrcoef(col[:-1], col[1:])[0, 1]
            if not np.isnan(c):
                v_corrs.append(c)

    feats['horizontal_correlation'] = float(np.mean(h_corrs)) if h_corrs else 0.0
    feats['vertical_correlation']   = float(np.mean(v_corrs)) if v_corrs else 0.0
    feats['mean_correlation']       = (feats['horizontal_correlation'] +
                                       feats['vertical_correlation']) / 2.0

    # --- Group 8: Shannon Entropy ---
    # 256-bin histogram entropy. Higher entropy = more complex or noisy.
    hist, _ = np.histogram(data, bins=256)
    hist_norm = hist / (hist.sum() + 1e-10)
    hist_norm = hist_norm[hist_norm > 0]
    feats['entropy'] = float(-np.sum(hist_norm * np.log2(hist_norm)))

    # --- Group 9: Edge Density ---
    # Fraction of pixels where Sobel magnitude exceeds mean + 1*std,
    # corresponding to approximately the top ~16% of edge pixels
    # (same threshold used in the edge preservation loss, Section III.D).
    sobel_x = ndimage.sobel(data, axis=1)
    sobel_y = ndimage.sobel(data, axis=0)
    edge_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_threshold = np.mean(edge_mag) + np.std(edge_mag)
    feats['edge_density'] = float(np.mean(edge_mag > edge_threshold))

    # --- Group 10: Noise Variance Estimate ---
    # Proxy for the Wiener filter noise variance parameter (nv).
    feats['noise_variance_estimate'] = float(noise_std**2)

    return feats


# ==========================================================================
# PSD Computation Utilities
# ==========================================================================

def compute_radial_psd(image):
    """
    Compute radially averaged Power Spectral Density (PSD).

    2D FFT is computed, then PSD values are averaged over concentric rings
    centered at DC to produce a 1D radial profile.

    Returns:
        tuple: (freqs, radial_psd) — normalized frequency [0,1) and PSD values.
    """
    h, w = image.shape
    f_transform = np.fft.fft2(image)
    f_shifted = np.fft.fftshift(f_transform)
    psd_2d = np.abs(f_shifted) ** 2

    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    max_r = min(cx, cy)

    radial_sum   = np.bincount(r.ravel(), psd_2d.ravel(), minlength=max_r)
    radial_count = np.bincount(r.ravel(), minlength=max_r)
    radial_count[radial_count == 0] = 1

    radial_psd = radial_sum[:max_r] / radial_count[:max_r]
    freqs = np.arange(max_r) / max_r
    return freqs, radial_psd


def compute_band_attenuation(freqs, psd_original, psd_filtered, bands):
    """
    Compute per-band PSD attenuation ratio between original and filtered images.

    Attenuation = (E_original - E_filtered) / E_original
    Positive = energy reduction (desirable in noise bands).
    """
    result = {}
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        if mask.sum() == 0:
            result[band_name] = 0.0
            continue
        e_orig = np.sum(psd_original[mask])
        e_filt = np.sum(psd_filtered[mask])
        result[band_name] = float((e_orig - e_filt) / (e_orig + 1e-10)) if e_orig > 1e-10 else 0.0
    return result


def apply_wiener_filter(image, window_size):
    """Apply scipy Wiener filter with given local window size."""
    return wiener(image, mysize=(window_size, window_size))


# ==========================================================================
# PSD Analysis Pipeline (Section III.B)
# ==========================================================================

def run_psd_analysis(file_list, max_samples=None):
    """
    PSD analysis with three Wiener filter strengths on sampled images.

    For each sampled image:
        1. Compute original radial PSD
        2. Apply weak/medium/strong Wiener filters
        3. Compute filtered radial PSD
        4. Calculate per-band attenuation ratios

    The "filter_response" scalar (mean high-frequency attenuation under
    medium filtering) is used as the target for feature-filter correlation.

    Returns:
        tuple: (band_records, radial_psd_avg, freqs, filter_responses)
    """
    if max_samples is None:
        max_samples = PSD_SAMPLE_SIZE

    np.random.seed(RANDOM_SEED)
    sample_indices = np.random.choice(
        len(file_list), size=min(max_samples, len(file_list)), replace=False)
    sample_files = [file_list[i] for i in sample_indices]

    band_records = []
    filter_responses = {}
    psd_accumulator = {'original': [], 'weak': [], 'medium': [], 'strong': []}
    common_freq_len = None

    print(f"\n  [PSD Analysis] Processing {len(sample_files)} sampled images...")
    for fpath in tqdm(sample_files, desc="  PSD Analysis"):
        image, meta = load_tiff(fpath)
        if image is None:
            continue

        filename = os.path.basename(fpath)
        freqs, psd_orig = compute_radial_psd(image)

        if common_freq_len is None:
            common_freq_len = len(freqs)
        elif len(freqs) != common_freq_len:
            continue

        psd_accumulator['original'].append(psd_orig)
        record = {'filename': filename}

        for strength_name, preset in WIENER_PRESETS.items():
            filtered = apply_wiener_filter(image, preset['mysize'])
            _, psd_filt = compute_radial_psd(filtered)
            psd_accumulator[strength_name].append(psd_filt)

            attenuation = compute_band_attenuation(freqs, psd_orig, psd_filt, FREQUENCY_BANDS)
            for band_name, atten_val in attenuation.items():
                record[f'{strength_name}_{band_name}_attenuation'] = atten_val

        band_records.append(record)

        # filter_response = medium filter's mean high-frequency attenuation
        high_bands = ['mid_noise', 'high_noise', 'very_high']
        response_vals = [record.get(f'medium_{b}_attenuation', 0.0) for b in high_bands]
        filter_responses[filename] = float(np.mean(response_vals))

    radial_psd_avg = {}
    for key, psd_list in psd_accumulator.items():
        if psd_list:
            radial_psd_avg[key] = np.mean(np.array(psd_list), axis=0)

    return band_records, radial_psd_avg, freqs, filter_responses


# ==========================================================================
# Pearson Correlation with Bootstrap Robustness (Section III.B)
# ==========================================================================

def compute_pearson_with_robustness(features_df, filter_responses, n_bootstrap=None):
    """
    Compute Pearson correlation between each feature and filter_response,
    with bootstrap-based robustness score (R).

    Robustness score R = 1 - std(bootstrap_r) / |mean(bootstrap_r)|
    Values close to 1 indicate stable, reproducible correlations.
    """
    if n_bootstrap is None:
        n_bootstrap = BOOTSTRAP_ITERATIONS

    response_series = features_df['filename'].map(filter_responses)
    valid_mask = response_series.notna()
    df_valid = features_df[valid_mask].copy()
    responses = response_series[valid_mask].values.astype(float)

    results = []
    for col in FEATURE_COLUMNS:
        if col not in df_valid.columns:
            results.append({'feature': col, 'pearson_r': np.nan,
                           'p_value': np.nan, 'robustness_score': np.nan})
            continue

        feature_vals = df_valid[col].values.astype(float)
        finite_mask = np.isfinite(feature_vals) & np.isfinite(responses)
        x, y = feature_vals[finite_mask], responses[finite_mask]

        if len(x) < 10:
            results.append({'feature': col, 'pearson_r': np.nan,
                           'p_value': np.nan, 'robustness_score': np.nan})
            continue

        r, p = stats.pearsonr(x, y)

        np.random.seed(RANDOM_SEED)
        bootstrap_rs = []
        for _ in range(n_bootstrap):
            idx = np.random.randint(0, len(x), size=len(x))
            try:
                br, _ = stats.pearsonr(x[idx], y[idx])
                bootstrap_rs.append(br)
            except Exception:
                pass

        if bootstrap_rs and abs(np.mean(bootstrap_rs)) > 1e-6:
            robustness = max(0.0, 1.0 - (np.std(bootstrap_rs) / abs(np.mean(bootstrap_rs))))
        else:
            robustness = 0.0

        results.append({'feature': col, 'pearson_r': float(r),
                       'p_value': float(p), 'robustness_score': float(robustness)})

    return pd.DataFrame(results)


# ==========================================================================
# Outlier Detection & Clustering
# ==========================================================================

def detect_outliers_and_cluster(df):
    """
    IQR-based outlier tagging and K-Means clustering.

    Outlier detection: For features with |skewness| < 1.0 (approximately
    normal), samples outside the 2.5th/97.5th percentile are tagged.
    Outlier samples are excluded from Phase 3 training.

    K-Means: Non-outlier data are clustered into N_CLUSTERS groups to
    verify that training data spans diverse measurement conditions.
    """
    print("\n  [Outlier Detection] Tagging extreme samples...")
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    outlier_mask = np.zeros(len(df), dtype=bool)

    for col in available_features:
        vals = df[col].dropna().values
        if len(vals) < 10:
            continue
        if abs(stats.skew(vals)) < 1.0:
            q1, q3 = np.percentile(vals, [2.5, 97.5])
            col_mask = (df[col] < q1) | (df[col] > q3)
            outlier_mask |= col_mask.values

    df['is_outlier'] = outlier_mask
    n_outliers = int(outlier_mask.sum())
    print(f"    Outliers tagged: {n_outliers} / {len(df)} "
          f"({100 * n_outliers / len(df):.1f}%)")

    print("  [Clustering] K-Means clustering...")
    normal_idx = df[~df['is_outlier']].index
    if len(normal_idx) > N_CLUSTERS:
        scaler = StandardScaler()
        X = scaler.fit_transform(df.loc[normal_idx, available_features].fillna(0))
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        df['cluster_id'] = -1
        df.loc[normal_idx, 'cluster_id'] = cluster_labels
    else:
        df['cluster_id'] = 0

    cluster_counts = df[df['cluster_id'] >= 0]['cluster_id'].value_counts().sort_index()
    for cid, count in cluster_counts.items():
        print(f"    Cluster {cid}: {count} files")

    return df


# ==========================================================================
# Visualization Functions (Fig. 1a, 1b, 1c)
# ==========================================================================

def plot_feature_distributions(df, output_dir):
    """Fig. 1(a): Distributions of representative image features."""
    print("\n  [Visualization] Generating Fig. 1(a)...")
    representative_features = [
        ('snr_estimated_dB', 'SNR (dB)',        'darkorange'),
        ('edge_density',     'Edge Density',    'forestgreen'),
        ('mean_correlation', 'Spatial Corr.',   'steelblue'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (col, label, color) in zip(axes, representative_features):
        if col not in df.columns:
            ax.set_title(f'{label}\n(not available)')
            continue
        vals = df[col].dropna().values
        ax.hist(vals, bins=50, color=color, alpha=0.75, edgecolor='black', linewidth=0.3)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.axvline(np.mean(vals), color='red', linestyle='--', linewidth=1.0,
                   label=f'Mean={np.mean(vals):.2f}')
        ax.axvline(np.median(vals), color='black', linestyle=':', linewidth=1.0,
                   label=f'Median={np.median(vals):.2f}')
        ax.legend(fontsize=8)
    fig.suptitle('Fig. 1(a): Representative Feature Distributions',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1a_feature_distributions.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_psd_attenuation_curves(freqs, radial_psd_avg, output_dir):
    """Fig. 1(b): PSD attenuation curves at three filter strengths."""
    print("  [Visualization] Generating Fig. 1(b)...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    style_map = {
        'original': {'color': 'black', 'linestyle': '-', 'linewidth': 2.0, 'label': 'Original'},
        'weak':     {'color': '#2196F3', 'linestyle': '--', 'linewidth': 1.2,
                     'label': f'Weak (nv={WIENER_PRESETS["weak"]["nv"]})'},
        'medium':   {'color': '#FF9800', 'linestyle': '--', 'linewidth': 1.2,
                     'label': f'Medium (nv={WIENER_PRESETS["medium"]["nv"]})'},
        'strong':   {'color': '#F44336', 'linestyle': '--', 'linewidth': 1.2,
                     'label': f'Strong (nv={WIENER_PRESETS["strong"]["nv"]})'},
    }
    for key, style in style_map.items():
        if key in radial_psd_avg:
            psd = radial_psd_avg[key]
            ax.semilogy(freqs[:len(psd)], psd + 1e-10, **style)

    band_colors = ['#E8F5E9', '#FFF9C4', '#FFE0B2', '#FFCCBC', '#F8BBD0', '#E1BEE7']
    for (band_name, (low, high)), bc in zip(FREQUENCY_BANDS.items(), band_colors):
        ax.axvspan(low, high, alpha=0.15, color=bc)

    ax.set_xlabel('Normalized Frequency', fontsize=11)
    ax.set_ylabel('Power Spectral Density (log)', fontsize=11)
    ax.set_title('Radial PSD: Original vs Filtered', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    band_names = list(FREQUENCY_BANDS.keys())
    x_pos = np.arange(len(band_names))
    bar_width = 0.25
    for i, (strength_name, preset) in enumerate(WIENER_PRESETS.items()):
        if strength_name in radial_psd_avg and 'original' in radial_psd_avg:
            attenuations = []
            for band_name, (low, high) in FREQUENCY_BANDS.items():
                atten = compute_band_attenuation(
                    freqs, radial_psd_avg['original'],
                    radial_psd_avg[strength_name], {band_name: (low, high)})
                attenuations.append(atten[band_name])
            ax2.bar(x_pos + i * bar_width, attenuations, bar_width,
                    label=f'{strength_name.capitalize()} (nv={preset["nv"]})', alpha=0.8)

    ax2.set_xticks(x_pos + bar_width)
    ax2.set_xticklabels(band_names, rotation=30, fontsize=9)
    ax2.set_ylabel('Attenuation Ratio', fontsize=11)
    ax2.set_title('Per-Band PSD Attenuation by Filter Strength', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Fig. 1(b): Wiener Filter PSD Analysis',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1b_psd_attenuation_curves.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_correlation_heatmap_and_robustness(pearson_df, features_df, output_dir):
    """Fig. 1(c): Correlation heatmap + robustness bar chart."""
    print("  [Visualization] Generating Fig. 1(c)...")
    valid_pearson = pearson_df.dropna(subset=['pearson_r'])
    top15 = valid_pearson.reindex(
        valid_pearson['pearson_r'].abs().sort_values(ascending=False).index).head(15)
    available_features = [f for f in top15['feature'].tolist() if f in features_df.columns]
    if not available_features:
        print("    [WARNING] No valid features for heatmap. Skipping.")
        return

    corr_matrix = features_df[available_features].corr()
    fig, axes = plt.subplots(1, 2, figsize=(18, 8),
                             gridspec_kw={'width_ratios': [1.2, 1]})

    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, cbar_kws={'shrink': 0.7}, ax=axes[0],
                linewidths=0.5, linecolor='white')
    axes[0].set_title('Inter-Feature Correlation Matrix\n(Top 15 by |r|)', fontsize=11)

    plot_df = valid_pearson.sort_values('pearson_r', ascending=True)
    colors = ['steelblue' if r > 0 else 'coral' for r in plot_df['pearson_r']]
    axes[1].barh(range(len(plot_df)), plot_df['pearson_r'].values,
                 color=colors, edgecolor='black', linewidth=0.3)
    axes[1].set_yticks(range(len(plot_df)))
    axes[1].set_yticklabels(plot_df['feature'].values, fontsize=8)
    axes[1].set_xlabel('Pearson r (vs Filter Response)', fontsize=10)
    axes[1].set_title('Feature-Filter Response Correlation\nwith Robustness Scores', fontsize=11)
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].grid(True, alpha=0.3, axis='x')

    for idx, (_, row) in enumerate(plot_df.iterrows()):
        r_val, rob = row['pearson_r'], row['robustness_score']
        if not np.isnan(rob):
            x_pos = r_val + (0.02 if r_val >= 0 else -0.02)
            ha = 'left' if r_val >= 0 else 'right'
            axes[1].text(x_pos, idx, f'R={rob:.2f}', fontsize=7,
                        va='center', ha=ha, color='gray')

    fig.suptitle('Fig. 1(c): Feature Validity & Robustness Analysis',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1c_correlation_heatmap.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_outlier_summary(df, output_dir):
    """Outlier distribution and cluster summary."""
    print("  [Visualization] Generating outlier summary...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    outlier_counts = df['is_outlier'].value_counts()
    sizes = [outlier_counts.get(False, 0), outlier_counts.get(True, 0)]
    axes[0].pie(sizes, labels=['Normal', 'Outlier'], colors=['#4CAF50', '#F44336'],
                autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Outlier Distribution', fontsize=12)

    cluster_data = df[df['cluster_id'] >= 0]['cluster_id'].value_counts().sort_index()
    axes[1].bar(cluster_data.index, cluster_data.values,
                color='steelblue', edgecolor='black', linewidth=0.3)
    axes[1].set_xlabel('Cluster ID', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('K-Means Cluster Distribution', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'outlier_summary.png'), dpi=200, bbox_inches='tight')
    plt.close()


# ==========================================================================
# Main Execution
# ==========================================================================

def main():
    print("=" * 70)
    print("  Phase 2: Feature Extraction & Analysis")
    print("=" * 70)
    print(f"  Input  : {CLASSIFICATION_DIR}")
    print(f"  Output : {OUTPUT_DIR}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    features_dir = os.path.join(OUTPUT_DIR, 'features')
    figures_dir  = os.path.join(OUTPUT_DIR, 'figures')
    for d in [features_dir, figures_dir]:
        os.makedirs(d, exist_ok=True)

    # Step 1: Load normal file list from Phase 1
    print("\n[Step 1/7] Loading normal file list from Phase 1...")
    normal_files_dir = os.path.join(CLASSIFICATION_DIR, 'normal')
    normal_files_json = os.path.join(CLASSIFICATION_DIR, 'analysis', 'normal_files.json')
    file_list = []

    if os.path.exists(normal_files_dir):
        for fname in os.listdir(normal_files_dir):
            if fname.lower().endswith(('.tiff', '.tif')):
                file_list.append(os.path.join(normal_files_dir, fname))
        print(f"  Scanned {len(file_list)} files from {normal_files_dir}")
    elif os.path.exists(normal_files_json):
        with open(normal_files_json, 'r') as f:
            file_list = json.load(f)
        print(f"  Loaded {len(file_list)} files from normal_files.json")
    else:
        print(f"  [ERROR] Neither normal/ directory nor normal_files.json found.")
        return

    if not file_list:
        print("[ERROR] No files to process.")
        return

    # Step 2: Extract 22 features
    print(f"\n[Step 2/7] Extracting 22 features from {len(file_list)} images...")
    all_features = []
    for fpath in tqdm(file_list, desc="  Feature Extraction"):
        image, meta = load_tiff(fpath)
        if image is not None:
            feats = extract_features(image, fpath)
            if meta is not None:
                feats.update({k: meta[k] for k in
                    ['original_min', 'original_max', 'original_mean', 'original_std']})
            all_features.append(feats)

    if not all_features:
        print("[ERROR] No features extracted. Check data paths.")
        return

    features_df = pd.DataFrame(all_features)
    print(f"  Successfully extracted features from {len(features_df)} images.")

    # Step 3: PSD analysis
    print(f"\n[Step 3/7] Running PSD analysis (sample size: {PSD_SAMPLE_SIZE})...")
    band_records, radial_psd_avg, freqs, filter_responses = run_psd_analysis(file_list)
    features_df['filter_response'] = features_df['filename'].map(filter_responses)
    n_with_response = features_df['filter_response'].notna().sum()
    print(f"  Filter response computed for {n_with_response} / {len(features_df)} images.")

    # Step 4: Pearson correlation + bootstrap robustness
    print(f"\n[Step 4/7] Computing Pearson correlations (bootstrap={BOOTSTRAP_ITERATIONS})...")
    pearson_df = compute_pearson_with_robustness(features_df, filter_responses)
    print("\n  Top 10 Feature-Filter Response Correlations:")
    sorted_pearson = pearson_df.dropna(subset=['pearson_r']).reindex(
        pearson_df['pearson_r'].abs().sort_values(ascending=False).index).head(10)
    for _, row in sorted_pearson.iterrows():
        print(f"    {row['feature']:30s}  r={row['pearson_r']:+.4f}  "
              f"p={row['p_value']:.2e}  R={row['robustness_score']:.3f}")

    # Step 5: Outlier detection and clustering
    print(f"\n[Step 5/7] Detecting outliers and clustering...")
    features_df = detect_outliers_and_cluster(features_df)

    # Step 6: Save CSV outputs
    print(f"\n[Step 6/7] Saving CSV outputs...")
    features_df.to_csv(os.path.join(features_dir, 'all_features.csv'), index=False)
    with open(os.path.join(features_dir, 'feature_columns.json'), 'w') as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    stats_records = []
    for col in FEATURE_COLUMNS:
        if col in features_df.columns:
            vals = features_df[col].dropna()
            if len(vals) > 0:
                stats_records.append({
                    'feature': col, 'count': len(vals),
                    'mean': float(vals.mean()), 'std': float(vals.std()),
                    'min': float(vals.min()), 'max': float(vals.max()),
                    'skewness': float(stats.skew(vals)),
                    'kurtosis': float(stats.kurtosis(vals)),
                })
    pd.DataFrame(stats_records).to_csv(
        os.path.join(features_dir, 'feature_statistics.csv'), index=False)

    if band_records:
        pd.DataFrame(band_records).to_csv(
            os.path.join(features_dir, 'psd_band_attenuation.csv'), index=False)
    if radial_psd_avg and freqs is not None:
        psd_data = {'normalized_frequency': freqs}
        for key, psd_arr in radial_psd_avg.items():
            psd_data[f'psd_{key}'] = psd_arr[:len(freqs)]
        pd.DataFrame(psd_data).to_csv(
            os.path.join(features_dir, 'psd_radial_profiles.csv'), index=False)

    pearson_df.to_csv(os.path.join(features_dir, 'pearson_correlation.csv'), index=False)
    pearson_df[['feature', 'robustness_score']].to_csv(
        os.path.join(features_dir, 'robustness_scores.csv'), index=False)

    # Step 7: Visualization
    print(f"\n[Step 7/7] Generating figures...")
    plot_feature_distributions(features_df, figures_dir)
    if radial_psd_avg and freqs is not None:
        plot_psd_attenuation_curves(freqs, radial_psd_avg, figures_dir)
    plot_correlation_heatmap_and_robustness(pearson_df, features_df, figures_dir)
    plot_outlier_summary(features_df, figures_dir)

    n_normal = int((~features_df['is_outlier']).sum())
    print("\n" + "=" * 70)
    print("  Phase 2 Complete")
    print(f"  Total images: {len(features_df)} | Features: {len(FEATURE_COLUMNS)}")
    print(f"  PSD samples: {len(band_records)} | Outliers: {int(features_df['is_outlier'].sum())}")
    print(f"  Clean samples for Phase 3: {n_normal}")
    print("=" * 70)


if __name__ == "__main__":
    main()
