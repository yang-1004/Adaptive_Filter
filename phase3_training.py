#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==========================================================================
Adaptive Filtering on KPFM Data — Phase 3: Adaptive Filter Training
==========================================================================

Description:
    Trains the adaptive filter parameter prediction model. The model
    comprises two main feature extraction pathways (Global and Local
    Feature Extractors) and two supporting submodules (Complexity
    Estimator and Multi-Scale Filter Module) as described in Section III.C.

    Architecture (six-stage pipeline):
        1. Structural Complexity Estimator
        2. Global Feature Extractor (MLP pathway -> 64-dim latent)
        3. Local Feature Extractor (CNN pathway, 2-channel input)
        4. Spatial Kernel Weight Predictor (fusion -> alpha map)
        5. Multi-Scale Filter Module (7 fixed Gaussian kernels)
        6. Alpha blending: I_out = alpha * I_filtered + (1-alpha) * I_original

    Supports five filter types:
        Wiener, Gaussian, Mean, Bilateral, Total Variation (TV)

    Self-supervised training — no clean reference images required.
    Composite loss function comprises five terms (Section III.D):
        (i) Frequency-band quality, (ii) SSIM structural preservation,
        (iii) Edge preservation, (iv) Dynamic minimum filtering,
        (v) Spatial smoothness.

Usage:
    1. Run Phase 1 and Phase 2 first.
    2. Set CLASSIFICATION_DIR, FEATURES_DIR, and TRAINING_OUTPUT below.
    3. Run:  python phase3_training.py

Reference:
    Anonymous, "Adaptive Filtering on KPFM Data: A Deep Neural Network
    Approach," submitted to Machine Learning: Science and Technology.

Author:  Anonymous
Version: 1.0 (public release)
"""

# ==========================================================================
# Run Configuration
# ==========================================================================

class RunConfig:
    # 'all' = all 5 filters, 'selected' = SELECTED_FILTERS, 'single' = SINGLE_FILTER
    RUN_MODE = 'all'
    SINGLE_FILTER = 'wiener'
    SELECTED_FILTERS = ['wiener', 'gaussian', 'bilateral']

    # Enable local parameter prediction (CNN pathway).
    ENABLE_LOCAL_PARAMS = True

    # Set True for A100-class GPUs (larger batch, more workers).
    USE_A100_OPTIMIZATION = True

    # Set True for quick smoke test with reduced epochs.
    DEBUG_MODE = False

    RANDOM_SEED = 42
    SKIP_COMPLETED_FILTERS = True
    RESUME_FROM_CHECKPOINT = True


# ==========================================================================
# Library Imports
# ==========================================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import tifffile


# ==========================================================================
# Device Setup
# ==========================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU: {gpu_name}, Memory: {gpu_memory:.1f} GB")


# ==========================================================================
# [USER] Path Configuration
# ==========================================================================

# Phase 1 output directory (normal/ folder must contain TIFF files).
CLASSIFICATION_DIR = './output/01_Classification'
NORMAL_FILES_DIR   = os.path.join(CLASSIFICATION_DIR, 'normal')

# Phase 2 output directory (features/ folder with CSV/JSON files).
FEATURES_DIR      = './output/02_Feature_Analysis/features'
FEATURES_CSV      = os.path.join(FEATURES_DIR, 'all_features.csv')
FEATURE_COLS_JSON = os.path.join(FEATURES_DIR, 'feature_columns.json')
FEATURE_STATS_CSV = os.path.join(FEATURES_DIR, 'feature_statistics.csv')

# Phase 3 output directory.
TRAINING_OUTPUT = './output/03_Training'


# ==========================================================================
# Filter-Specific PSD Attenuation Targets (Section S4)
# ==========================================================================
# Per-band target PSD reduction ratios, derived from Phase 2 PSD analysis.
# Scaled at runtime by the complexity estimator output.

FILTER_SPECIFIC_TARGETS = {
    'wiener':    {'signal': 0.00, 'transition': 0.50, 'low_noise': 0.75,
                  'mid_noise': 0.90, 'high_noise': 0.95, 'very_high': 0.95},
    'gaussian':  {'signal': 0.00, 'transition': 0.65, 'low_noise': 0.90,
                  'mid_noise': 0.95, 'high_noise': 0.98, 'very_high': 0.98},
    'mean':      {'signal': 0.00, 'transition': 0.55, 'low_noise': 0.90,
                  'mid_noise': 0.95, 'high_noise': 0.98, 'very_high': 0.98},
    'bilateral': {'signal': 0.00, 'transition': 0.45, 'low_noise': 0.70,
                  'mid_noise': 0.85, 'high_noise': 0.92, 'very_high': 0.95},
    'tv':        {'signal': 0.00, 'transition': 0.20, 'low_noise': 0.60,
                  'mid_noise': 0.80, 'high_noise': 0.90, 'very_high': 0.90},
}

# Per-band tolerances (error within this range incurs no penalty).
BAND_TOLERANCES = {
    'signal': 0.10, 'transition': 0.15, 'low_noise': 0.10,
    'mid_noise': 0.08, 'high_noise': 0.05, 'very_high': 0.05,
}


# ==========================================================================
# Training Configuration
# ==========================================================================

class Config:
    IMAGE_SIZE = 256

    # 7 fixed Gaussian kernels: sigma_k = size / 6.0 (Section S2)
    KERNEL_SIZES = [3, 5, 7, 9, 11, 13, 15]
    NUM_SCALES = len(KERNEL_SIZES)

    BATCH_SIZE = 16 if RunConfig.USE_A100_OPTIMIZATION else 8
    NUM_WORKERS = 4 if RunConfig.USE_A100_OPTIMIZATION else 0
    NUM_EPOCHS = 5 if RunConfig.DEBUG_MODE else 100
    PATIENCE = 3 if RunConfig.DEBUG_MODE else 15
    LEARNING_RATE = 1e-4     # AdamW (Ref. 24)
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP = 1.0

    # Dataset split ratios: 85/10/5 (Section III.A)
    TRAIN_RATIO = 0.85
    VAL_RATIO = 0.10
    TEST_RATIO = 0.05

    # Alpha map clamping range (Section S2)
    MIN_ALPHA = 0.05
    MAX_ALPHA = 1.0

    # Dynamic minimum filtering parameters (Section III.D, item iv)
    MIN_FILTERING_BASE = 0.01
    MIN_FILTERING_SCALE = 0.04
    DYNAMIC_MIN_WARMUP_EPOCHS = 5

    # Edge-aware local refinement parameters (Section S2)
    MAX_LOCAL_DEVIATION = 0.8
    EDGE_PRESERVATION_WEIGHT = 2.0
    EDGE_AWARE_REDUCTION = 0.85

    # Complexity estimator output dimension: [edge_density, texture_variance, high_freq_ratio]
    COMPLEXITY_DIM = 3

    # TV filter: number of iterations per forward pass (Section S2)
    TV_ITERATIONS = 5

    # Frequency bands matching Section III.D and Section S4
    FREQUENCY_BANDS = {
        'signal': (0.00, 0.05), 'transition': (0.05, 0.15),
        'low_noise': (0.15, 0.30), 'mid_noise': (0.30, 0.50),
        'high_noise': (0.50, 0.80), 'very_high': (0.80, 1.00),
    }
    INITIAL_BAND_WEIGHTS = {
        'signal': 3.0, 'transition': 1.5, 'low_noise': 1.0,
        'mid_noise': 1.0, 'high_noise': 1.0, 'very_high': 0.8,
    }
    DYNAMIC_WEIGHT_WARMUP = 10

    # Loss function weights (Section III.D): five terms
    # Note: The paper specifies five loss terms. There is no "scale_aware" term.
    LOSS_WEIGHTS = {
        'frequency_quality':      0.25,   # (i) L_fq
        'ssim_structure':         0.25,   # (ii) L_ssim
        'edge_preservation':      0.20,   # (iii) L_edge
        'dynamic_min_filtering':  0.10,   # (iv) L_min
        'local_smoothness':       0.10,   # (v) L_smooth
    }

    SSIM_MIN_FILTERING = 0.03
    SSIM_TARGET = 0.85
    DEFAULT_NUM_FEATURES = 22
    DIFFERENTIABLE_FILTERS = ['wiener', 'gaussian', 'mean', 'bilateral', 'tv']
    ALL_FILTERS = DIFFERENTIABLE_FILTERS

config = Config()


# ==========================================================================
# TIFF I/O
# ==========================================================================

def load_tiff(filepath):
    """
    Load a single-channel TIFF file. For proprietary formats (e.g., Park
    Systems), replace with the appropriate loader returning (image_2d, metadata).
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
# Utility Functions
# ==========================================================================

def safe_to_item(tensor_or_value):
    """Safely convert a tensor to a Python scalar."""
    if isinstance(tensor_or_value, torch.Tensor):
        return tensor_or_value.detach().cpu().item()
    return tensor_or_value


def check_batch_valid(batch: Dict) -> bool:
    """Check that a DataLoader batch contains valid data."""
    if 'image' not in batch or batch['image'] is None:
        return False
    valid = batch.get('valid', None)
    if valid is None:
        return True
    try:
        if isinstance(valid, torch.Tensor):
            return bool(valid.any().item()) if valid.numel() > 0 else True
        return True
    except Exception:
        return True


def reflect_conv2d(x, weight, padding_size):
    """
    2D convolution with reflection padding.

    Reflection padding is used throughout the architecture to prevent
    border-flattening artifacts common with zero padding in KPFM data
    (Section S2).
    """
    x_padded = F.pad(x, (padding_size,) * 4, mode='reflect')
    return F.conv2d(x_padded, weight, padding=0)


# ==========================================================================
# Differentiable PSD Computation (for loss function)
# ==========================================================================

def compute_band_powers_differentiable(images, frequency_bands):
    """
    Compute per-band PSD energy ratios using differentiable operations.

    Smooth sigmoid masks at band boundaries maintain gradient flow.
    Used by the frequency-band quality loss L_fq (Section III.D, item i).
    """
    B, C, H, W = images.shape
    device = images.device
    images_squeezed = images.squeeze(1)
    f = torch.fft.fft2(images_squeezed)
    f_shift = torch.fft.fftshift(f, dim=(-2, -1))
    psd = torch.abs(f_shift) ** 2

    center_y, center_x = H // 2, W // 2
    y = torch.arange(H, device=device).float() - center_y
    x = torch.arange(W, device=device).float() - center_x
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    r = torch.sqrt(xx**2 + yy**2)
    max_r = min(center_y, center_x)
    r_normalized = r / max_r

    band_powers = {}
    for band_name, (low, high) in frequency_bands.items():
        margin = 0.02
        low_mask = torch.sigmoid((r_normalized - low) / margin)
        high_mask = torch.sigmoid((high - r_normalized) / margin)
        band_mask = low_mask * high_mask
        masked_psd = psd * band_mask.unsqueeze(0)
        power = masked_psd.sum(dim=(-2, -1))
        total_power = psd.sum(dim=(-2, -1)) + 1e-8
        band_powers[band_name] = power / total_power
    return band_powers


# ==========================================================================
# Evaluation Metrics (Section S6)
# ==========================================================================

class EvaluationMetrics:
    """
    Metrics for evaluating filtering results. These are used during
    validation/testing, not during training (which is self-supervised).
    """
    @staticmethod
    def compute_psnr(original, denoised):
        """
        PSNR measures the degree of alteration relative to the original.
        In this self-supervised context without clean ground truth, higher
        PSNR indicates less modification of the original image.
        """
        mse = np.mean((original - denoised) ** 2)
        if mse < 1e-10:
            return 100.0
        max_val = max(original.max() - original.min(), 1e-10)
        return 20 * np.log10(max_val / np.sqrt(mse))

    @staticmethod
    def compute_ssim(original, denoised, window_size=11):
        """SSIM with 11x11 Gaussian-weighted windows (Ref. 22)."""
        from scipy.ndimage import uniform_filter
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        o, d = original.astype(np.float64), denoised.astype(np.float64)
        mu1 = uniform_filter(o, window_size)
        mu2 = uniform_filter(d, window_size)
        sigma1_sq = uniform_filter(o**2, window_size) - mu1**2
        sigma2_sq = uniform_filter(d**2, window_size) - mu2**2
        sigma12 = uniform_filter(o * d, window_size) - mu1 * mu2
        ssim_map = ((2*mu1*mu2+C1)*(2*sigma12+C2)) / ((mu1**2+mu2**2+C1)*(sigma1_sq+sigma2_sq+C2))
        return float(np.mean(ssim_map))

    @staticmethod
    def compute_mae(original, denoised):
        """MAE: Mean Absolute Error (mV when computed on mV-scale images)."""
        return float(np.mean(np.abs(original - denoised)))

    @staticmethod
    def compute_epi(original, denoised):
        """
        EPI: Edge Preservation Index.
        Pearson correlation of Sobel gradient magnitude maps (Section S6).
        """
        from scipy.ndimage import sobel
        edge_orig = np.sqrt(sobel(original, 0)**2 + sobel(original, 1)**2)
        edge_deno = np.sqrt(sobel(denoised, 0)**2 + sobel(denoised, 1)**2)
        if np.std(edge_orig) < 1e-10 or np.std(edge_deno) < 1e-10:
            return 1.0
        corr = np.corrcoef(edge_orig.flatten(), edge_deno.flatten())[0, 1]
        return corr if not np.isnan(corr) else 0.0


# ==========================================================================
# Model Components (Section III.C, Section S2)
# ==========================================================================

class ImageComplexityEstimator(nn.Module):
    """
    Structural Complexity Estimator (Section S2).

    Computes a 3-dimensional complexity vector:
        [edge_density, texture_variance, high_frequency_energy_ratio]

    Used for adaptive loss scaling and the complexity gate in the
    Global Feature Extractor. All components clamped to [0, 1].
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1,1,3,3))
        self.register_buffer('sobel_y', sobel_y.view(1,1,3,3))
        self.edge_threshold = nn.Parameter(torch.tensor(0.1))  # learnable (init=0.1)

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        # Edge density via reflection-padded Sobel with learnable threshold
        edge_x = reflect_conv2d(x, self.sobel_x.to(device), 1)
        edge_y = reflect_conv2d(x, self.sobel_y.to(device), 1)
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
        edge_density = (edge_magnitude > self.edge_threshold.abs()).float().mean(dim=(1,2,3))

        # Texture variance: std of 5x5 local variance
        local_mean = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        local_var = F.avg_pool2d((x - local_mean)**2, kernel_size=5, stride=1, padding=2)
        texture_variance = local_var.std(dim=(1,2,3))

        # High-frequency energy ratio: FFT magnitude fraction at freq > 0.5
        fft = torch.fft.fft2(x.squeeze(1))
        fft_shift = torch.fft.fftshift(fft, dim=(-2,-1))
        magnitude = torch.abs(fft_shift)
        center_y, center_x = H//2, W//2
        y = torch.arange(H, device=device).float() - center_y
        x_coord = torch.arange(W, device=device).float() - center_x
        yy, xx = torch.meshgrid(y, x_coord, indexing='ij')
        r = torch.sqrt(xx**2 + yy**2)
        max_r = min(center_x, center_y)
        high_freq_mask = (r / max_r) > 0.5
        high_freq_ratio = (magnitude * high_freq_mask.unsqueeze(0)).sum(dim=(1,2)) / \
                          (magnitude.sum(dim=(1,2)) + 1e-8)

        edge_density = torch.clamp(edge_density, 0, 1)
        texture_variance = torch.clamp(texture_variance / (texture_variance.max() + 1e-8), 0, 1)
        high_freq_ratio = torch.clamp(high_freq_ratio, 0, 1)

        return torch.stack([edge_density, texture_variance, high_freq_ratio], dim=1)


class LocalFeatureExtractor(nn.Module):
    """
    CNN pathway (Section III.C): extracts per-pixel local features from
    the original image and its Sobel edge magnitude map.

    Architecture: Conv(2->32) -> Conv(32->48) -> Conv(48->64)
    All layers use ReflectionPad2d. Spatial resolution (256x256) preserved.
    Input channels: [original image, Sobel edge magnitude].
    """
    def __init__(self, embed_dim=64):
        super().__init__()
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1,1,3,3))
        self.register_buffer('sobel_y', sobel_y.view(1,1,3,3))

        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(2, 32, 3, padding=0)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(32, 48, 3, padding=0)
        self.pad3 = nn.ReflectionPad2d(1)
        self.conv3 = nn.Conv2d(48, embed_dim, 3, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        device = x.device
        edge_x = reflect_conv2d(x, self.sobel_x.to(device), 1)
        edge_y = reflect_conv2d(x, self.sobel_y.to(device), 1)
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
        edge_normalized = edge_magnitude / (edge_magnitude.max() + 1e-8)

        x_with_edge = torch.cat([x, edge_normalized], dim=1)
        feat = self.relu(self.conv1(self.pad1(x_with_edge)))
        feat = self.relu(self.conv2(self.pad2(feat)))
        feat = self.relu(self.conv3(self.pad3(feat)))
        return feat


class EdgeAwareLocalRefinement(nn.Module):
    """
    Edge-aware refinement (Section S2): combines global prediction with
    local edge-aware adjustment to generate the per-pixel alpha map.

    Near edges, filter strength is reduced to preserve structure.
    In flat regions, the global prediction is maintained.

    Alpha map output is clamped to [MIN_ALPHA, MAX_ALPHA] = [0.05, 1.0].
    """
    def __init__(self, input_dim, max_deviation=0.8, edge_reduction=0.85):
        super().__init__()
        self.max_deviation = max_deviation
        self.edge_reduction = edge_reduction

        self.edge_pad = nn.ReflectionPad2d(1)
        self.edge_detector = nn.Sequential(
            nn.Conv2d(input_dim, 32, 3, padding=0), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 16, 3, padding=0), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1), nn.Sigmoid())

        self.local_pad = nn.ReflectionPad2d(1)
        self.local_net = nn.Sequential(
            nn.Conv2d(input_dim, 32, 3, padding=0), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 16, 3, padding=0), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1))

    def forward(self, global_param, local_features, original_image=None):
        B, C, H, W = local_features.shape
        edge_mask = self.edge_detector(self.edge_pad(local_features))
        delta = self.local_net(self.local_pad(local_features))

        bounded_delta = torch.tanh(delta) * self.max_deviation
        edge_adjustment = 1.0 - edge_mask * self.edge_reduction

        global_expanded = global_param.view(B, 1, 1, 1).expand(B, 1, H, W)
        alpha_map = global_expanded * (1 + bounded_delta) * edge_adjustment
        alpha_map = torch.clamp(alpha_map, config.MIN_ALPHA, config.MAX_ALPHA)

        # Smoothness regularization for spatial smoothness loss
        diff_h = delta[:, :, 1:, :] - delta[:, :, :-1, :]
        diff_w = delta[:, :, :, 1:] - delta[:, :, :, :-1]
        smoothness_loss = torch.mean(diff_h.abs()) + torch.mean(diff_w.abs())

        return alpha_map, edge_mask, smoothness_loss


class GlobalFeatureExtractor(nn.Module):
    """
    MLP pathway (Section III.C, Section S2).

    Inputs: 22 image features, scale info (min, max, mean, std),
    complexity vector (3-dim).

    Outputs:
        - scale_weights: SoftMax mixing weights for 7 kernel sizes
        - Filter-specific parameters (e.g., nv for Wiener)
        - complexity_factor: Complexity gate output in [0.5, 1.0]

    Architecture (Section S2):
        Feature encoder: Linear(22->64)->ReLU->Dropout(0.2)->Linear(64->64)->ReLU
        Scale encoder:   Linear(4->32)->ReLU->Linear(32->32)->ReLU
        Complexity encoder: Linear(3->32)->ReLU->Linear(32->32)->ReLU
        Fusion MLP: Linear(128->96)->ReLU->Linear(96->64)->ReLU
        Scale head: Linear(64->7)->SoftMax
        Complexity gate: Linear(3->16)->ReLU->Linear(16->1)->Sigmoid -> [0.5, 1.0]
    """
    def __init__(self, num_features, filter_type):
        super().__init__()
        self.filter_type = filter_type

        # Independent embedding layers per feature group (Section S2)
        self.scale_embed = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.feature_encoder = nn.Sequential(
            nn.Linear(num_features, 64), nn.ReLU(), nn.Dropout(0.2),  # Dropout (Ref. 27)
            nn.Linear(64, 64), nn.ReLU())
        self.complexity_encoder = nn.Sequential(
            nn.Linear(config.COMPLEXITY_DIM, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU())

        # Fusion: 64 + 32 + 32 = 128 -> 96 -> 64
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32 + 32, 96), nn.ReLU(),
            nn.Linear(96, 64), nn.ReLU())

        # Scale head: 64 -> 7 -> SoftMax
        self.scale_head = nn.Linear(64, config.NUM_SCALES)

        # Complexity gate: 3 -> 16 -> 1 -> Sigmoid -> output [0.5, 1.0]
        self.complexity_gate = nn.Sequential(
            nn.Linear(config.COMPLEXITY_DIM, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid())

        # Parameter head: filter-type-specific
        if filter_type == 'wiener':
            self.param_head = nn.Linear(64, 1)   # nv (noise variance)
        elif filter_type == 'gaussian':
            self.param_head = nn.Linear(64, 1)   # sigma_G
        elif filter_type == 'bilateral':
            self.param_head = nn.Linear(64, 2)   # sigma_s, sigma_r
        elif filter_type == 'tv':
            self.param_head = nn.Linear(64, 1)   # tv_weight
        else:
            self.param_head = None

    def forward(self, features, scale_info, complexity_info):
        features = torch.nan_to_num(features, nan=0.0)
        scale_info = torch.nan_to_num(scale_info, nan=0.0)
        complexity_info = torch.nan_to_num(complexity_info, nan=0.0)

        scale_max = scale_info.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-6)
        scale_norm = scale_info / scale_max

        scale_embed = self.scale_embed(scale_norm)
        feat_embed = self.feature_encoder(features)
        complexity_embed = self.complexity_encoder(complexity_info)

        fused = self.fusion(torch.cat([feat_embed, scale_embed, complexity_embed], dim=-1))
        scale_weights = F.softmax(self.scale_head(fused), dim=-1)

        params = {'scale_weights': scale_weights, 'latent': fused,
                  'complexity': complexity_info}

        # Complexity gate: output in [0.5, 1.0]
        complexity_factor = 0.5 + 0.5 * self.complexity_gate(complexity_info).squeeze(-1)

        if self.param_head:
            p = torch.sigmoid(self.param_head(fused))
            if self.filter_type == 'wiener':
                # nv: 10^(-4) to 10^(0) on log scale, modulated by complexity
                base_nv = 10 ** (-4.0 + 4.0 * p)
                params['noise_variance'] = (base_nv * complexity_factor.unsqueeze(-1)).squeeze(-1)
            elif self.filter_type == 'gaussian':
                # sigma_G: 0.3 to 5.0
                params['sigma'] = ((0.3 + 4.7 * p) * complexity_factor.unsqueeze(-1)).squeeze(-1)
            elif self.filter_type == 'bilateral':
                # sigma_s: 1.0 to 15.0, sigma_r: 0.005 to 0.5
                params['sigma_spatial'] = ((1.0 + 14.0 * p[:, 0]) * complexity_factor).unsqueeze(-1)
                params['sigma_intensity'] = 0.005 + 0.495 * p[:, 1]
            elif self.filter_type == 'tv':
                # tv_weight: 0.001 to 0.1
                params['tv_weight'] = ((0.001 + 0.1 * p) * complexity_factor.unsqueeze(-1)).squeeze(-1)
        return params


class SpatialKernelWeightPredictor(nn.Module):
    """
    Spatial Kernel Weight Predictor (Section S2).

    Fuses 74-dimensional global context (scale_weights[7] + latent[64] +
    complexity[3]) with 64-channel CNN features to predict per-pixel
    kernel weights and the alpha map.

    Architecture (Section S2):
        Global context: Linear(74->48)->ReLU, spatially expanded to 256x256
        Conv2d(112->64, 3x3)->ReLU -> Conv2d(64->32, 3x3)->ReLU
        Kernel head: Conv2d(32->7, 1x1)->SoftMax
        Strength head: Conv2d(32->1, 1x1)->Sigmoid
        Edge-aware refinement -> final alpha map clamped to [0.05, 1.0]
    """
    def __init__(self, input_dim=64, filter_type='wiener', enable_local=True):
        super().__init__()
        self.filter_type = filter_type
        self.enable_local = enable_local

        # 74 = NUM_SCALES(7) + latent(64) + COMPLEXITY_DIM(3)
        self.global_context = nn.Sequential(
            nn.Linear(config.NUM_SCALES + 64 + config.COMPLEXITY_DIM, 48), nn.ReLU())

        # 64 (CNN) + 48 (global context) = 112 channels
        self.local_pad = nn.ReflectionPad2d(1)
        self.local_net = nn.Sequential(
            nn.Conv2d(input_dim + 48, 64, 3, padding=0), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, 3, padding=0), nn.ReLU(inplace=True))

        self.kernel_head = nn.Conv2d(32, config.NUM_SCALES, 1)  # per-pixel SoftMax mixing
        self.strength_head = nn.Conv2d(32, 1, 1)                # base alpha
        self.local_refinement = EdgeAwareLocalRefinement(
            32, config.MAX_LOCAL_DEVIATION, config.EDGE_AWARE_REDUCTION)

        if enable_local:
            if filter_type == 'wiener':
                self.local_param_head = nn.Conv2d(32, 1, 1)
            elif filter_type == 'gaussian':
                self.local_param_head = nn.Conv2d(32, 1, 1)
            elif filter_type == 'bilateral':
                self.local_param_head = nn.Conv2d(32, 2, 1)
            elif filter_type == 'tv':
                self.local_param_head = nn.Conv2d(32, 1, 1)
            else:
                self.local_param_head = None
        else:
            self.local_param_head = None

    def forward(self, local_feat, global_params, original_image=None):
        B, C, H, W = local_feat.shape
        complexity = global_params.get('complexity',
            torch.zeros(B, config.COMPLEXITY_DIM, device=local_feat.device))

        global_info = torch.cat(
            [global_params['scale_weights'], global_params['latent'], complexity], dim=-1)
        global_ctx = self.global_context(global_info).view(B, 48, 1, 1).expand(-1, -1, H, W)

        combined = torch.cat([local_feat, global_ctx], dim=1)
        local_out = self.local_net(self.local_pad(combined))

        kernel_weights = F.softmax(self.kernel_head(local_out), dim=1)
        base_strength = torch.sigmoid(self.strength_head(local_out))
        global_strength = base_strength.mean(dim=(2, 3), keepdim=True)

        alpha_map, edge_mask, smoothness_loss = \
            self.local_refinement(global_strength, local_out, original_image)

        local_params = {'edge_mask': edge_mask}
        if self.enable_local and self.local_param_head:
            local_raw = torch.sigmoid(self.local_param_head(local_out))
            edge_adjustment = 1.0 - edge_mask * config.EDGE_AWARE_REDUCTION
            if self.filter_type == 'wiener':
                local_params['noise_variance'] = (10 ** (-4.0 + 4.0 * local_raw)) * edge_adjustment
            elif self.filter_type == 'gaussian':
                local_params['sigma'] = (0.3 + 4.7 * local_raw) * edge_adjustment
            elif self.filter_type == 'bilateral':
                local_params['sigma_spatial'] = (1.0 + 14.0 * local_raw[:, 0:1]) * edge_adjustment
                local_params['sigma_intensity'] = 0.005 + 0.495 * local_raw[:, 1:2]
            elif self.filter_type == 'tv':
                local_params['tv_weight'] = (0.001 + 0.1 * local_raw) * edge_adjustment

        return kernel_weights, alpha_map, local_params, smoothness_loss


class DifferentiableMultiScaleFilter(nn.Module):
    """
    Multi-Scale Filter Module (Section III.C, Section S2).

    Uses 7 pre-computed Gaussian kernels with sigma_k = kernel_size / 6.0.
    The composite kernel K is constructed as a weighted sum of per-scale
    outputs using the SoftMax mixing weights predicted by the model.

    Final output:
        I_out = alpha * I_filtered + (1 - alpha) * I_original

    where alpha is the per-pixel alpha map.
    """
    def __init__(self, filter_type):
        super().__init__()
        self.filter_type = filter_type
        self.kernels = nn.ParameterDict()

        for i, k_size in enumerate(config.KERNEL_SIZES):
            sigma = k_size / 6.0  # sigma_k (Section S2)
            ax = torch.arange(k_size).float() - k_size // 2
            xx, yy = torch.meshgrid(ax, ax, indexing='ij')
            kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            kernel = kernel / kernel.sum()
            self.kernels[f'kernel_{i}'] = nn.Parameter(
                kernel.unsqueeze(0).unsqueeze(0), requires_grad=False)

    def apply_filter(self, x, scale_idx, local_params, global_params):
        """Apply filter at a single scale."""
        k_size = config.KERNEL_SIZES[scale_idx]
        kernel = self.kernels[f'kernel_{scale_idx}']
        padding = k_size // 2
        x_pad = F.pad(x, (padding,) * 4, mode='reflect')

        if self.filter_type == 'wiener':
            # Wiener filter transfer function (Section S2, Eq. for Wiener):
            # H(f) = |K(f)|^2 / (|K(f)|^2 + nv)
            nv = local_params.get('noise_variance',
                     global_params.get('noise_variance', 0.01))
            if not isinstance(nv, torch.Tensor):
                nv = torch.tensor(nv, device=x.device).view(1, 1, 1, 1)
            elif nv.dim() < 4:
                nv = nv.view(-1, 1, 1, 1)
            nv = torch.clamp(nv, min=1e-6, max=1.0)

            local_mean = F.conv2d(x_pad, kernel.to(x.device), padding=0)
            local_sq = F.conv2d(x_pad ** 2, kernel.to(x.device), padding=0)
            local_var = torch.clamp(local_sq - local_mean ** 2, min=1e-6)
            wiener_coef = torch.clamp((local_var - nv) / (local_var + 1e-8), 0, 1)
            return local_mean + wiener_coef * (x - local_mean)

        elif self.filter_type == 'gaussian':
            return F.conv2d(x_pad, kernel.to(x.device), padding=0)

        elif self.filter_type == 'mean':
            mean_k = torch.ones(1, 1, k_size, k_size, device=x.device) / (k_size * k_size)
            return F.conv2d(x_pad, mean_k, padding=0)

        elif self.filter_type == 'bilateral':
            # Spatial weighting handled by the Gaussian kernel convolution;
            # intensity-dependent weighting approximated at training scale.
            return F.conv2d(x_pad, kernel.to(x.device), padding=0)

        elif self.filter_type == 'tv':
            # TV denoising: 5 iterations per forward pass (Section S2)
            tv_weight = local_params.get('tv_weight',
                            global_params.get('tv_weight', 0.05))
            if not isinstance(tv_weight, torch.Tensor):
                tv_weight = torch.tensor(tv_weight, device=x.device).view(1, 1, 1, 1)
            elif tv_weight.dim() < 4:
                tv_weight = tv_weight.view(-1, 1, 1, 1)
            # Discrete Laplacian: [[0,1,0],[1,-4,1],[0,1,0]]
            laplacian = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],
                                     dtype=torch.float32, device=x.device).view(1,1,3,3)
            result = x
            for _ in range(config.TV_ITERATIONS):
                grad = reflect_conv2d(result, laplacian, 1)
                result = result + tv_weight * grad
            return result

        return x

    def forward(self, x, kernel_weights, alpha_map, local_params, global_params):
        """
        Apply multi-scale filtering and alpha blending.

        I_out = alpha * I_filtered + (1 - alpha) * I_original
        """
        filtered_scales = [self.apply_filter(x, i, local_params, global_params)
                          for i in range(config.NUM_SCALES)]
        filtered_stack = torch.stack(filtered_scales, dim=1).squeeze(2)
        weighted = (filtered_stack * kernel_weights).sum(dim=1, keepdim=True)
        return alpha_map * weighted + (1 - alpha_map) * x


class AdaptiveFilterModel(nn.Module):
    """
    Complete adaptive filter model (~240K parameters, Section S2).

    Combines the MLP global pathway and CNN local pathway into the
    six-stage pipeline described in Section III.C.
    """
    def __init__(self, filter_type, num_features=22, enable_local_params=True):
        super().__init__()
        self.filter_type = filter_type
        self.enable_local_params = enable_local_params
        self.num_features = num_features

        self.complexity_estimator = ImageComplexityEstimator()
        self.global_predictor = GlobalFeatureExtractor(num_features, filter_type)
        self.local_extractor = LocalFeatureExtractor(embed_dim=64)
        self.kernel_predictor = SpatialKernelWeightPredictor(
            input_dim=64, filter_type=filter_type, enable_local=enable_local_params)
        self.filters = DifferentiableMultiScaleFilter(filter_type)

    def forward(self, image, features, scale_info):
        image = torch.nan_to_num(image, nan=0.0)
        features = torch.nan_to_num(features, nan=0.0)
        scale_info = torch.nan_to_num(scale_info, nan=0.0)

        complexity_info = self.complexity_estimator(image)
        global_params = self.global_predictor(features, scale_info, complexity_info)
        local_features = self.local_extractor(image)
        kernel_weights, alpha_map, local_params, smoothness_loss = \
            self.kernel_predictor(local_features, global_params, image)
        denoised = self.filters(image, kernel_weights, alpha_map,
                               local_params, global_params)

        return {
            'denoised': denoised,
            'kernel_weights': kernel_weights,
            'alpha_map': alpha_map,
            'global_params': global_params,
            'local_params': local_params,
            'local_smoothness_loss': smoothness_loss,
            'complexity_info': complexity_info,
            'edge_mask': local_params.get('edge_mask', None),
        }


# ==========================================================================
# Loss Functions — Five Terms (Section III.D, Section S4)
# ==========================================================================

class DynamicBandWeights(nn.Module):
    """
    Learned per-frequency-band weighting network (Section S4).

    Dynamically adjusts the relative contribution of each frequency band
    within the frequency-band quality loss. Implemented as a lightweight
    MLP with EMA-based weight updates.
    """
    def __init__(self, num_bands=6):
        super().__init__()
        initial = torch.tensor(list(config.INITIAL_BAND_WEIGHTS.values()), dtype=torch.float32)
        self.register_buffer('base_weights', initial)
        self.register_buffer('ema_weights', initial.clone())
        self.adjustment_net = nn.Sequential(
            nn.Linear(num_bands, 32), nn.ReLU(),
            nn.Linear(32, num_bands), nn.Tanh())

    def forward(self, band_errors):
        if band_errors.device != self.base_weights.device:
            band_errors = band_errors.to(self.base_weights.device)
        adjustment = self.adjustment_net(band_errors)
        return torch.clamp(self.ema_weights.unsqueeze(0) * (1.0 + adjustment * 0.5),
                          min=0.5, max=2.0)

    def update_ema(self, new_weights):
        with torch.no_grad():
            self.ema_weights.data.copy_(
                0.95 * self.ema_weights.data +
                0.05 * new_weights.mean(dim=0).to(self.ema_weights.device))

    def step_epoch(self):
        pass


class FrequencyBandQualityLoss(nn.Module):
    """(i) Frequency-band quality loss L_fq (lambda = 0.25)."""
    def __init__(self, filter_type, use_dynamic_weights=True):
        super().__init__()
        self.filter_type = filter_type
        self.use_dynamic_weights = use_dynamic_weights
        targets = FILTER_SPECIFIC_TARGETS.get(filter_type, FILTER_SPECIFIC_TARGETS['wiener'])
        band_names = list(config.FREQUENCY_BANDS.keys())
        self.register_buffer('target_tensor',
            torch.tensor([targets[b] for b in band_names], dtype=torch.float32))
        self.register_buffer('tolerance_tensor',
            torch.tensor([BAND_TOLERANCES[b] for b in band_names], dtype=torch.float32))
        if use_dynamic_weights:
            self.dynamic_weights = DynamicBandWeights(num_bands=len(band_names))
        else:
            self.register_buffer('fixed_weights',
                torch.tensor(list(config.INITIAL_BAND_WEIGHTS.values()), dtype=torch.float32))

    def forward(self, original, filtered, complexity_info=None):
        device = original.device
        band_names = list(config.FREQUENCY_BANDS.keys())
        B = original.shape[0]

        if complexity_info is not None:
            comp_factor = complexity_info[:, 2].detach().unsqueeze(1)
            adaptive_scale = 0.2 + 0.8 * comp_factor
        else:
            adaptive_scale = torch.ones(B, 1, device=device)

        orig_powers = compute_band_powers_differentiable(original, config.FREQUENCY_BANDS)
        filt_powers = compute_band_powers_differentiable(filtered, config.FREQUENCY_BANDS)

        reductions, errors = [], []
        for i, band_name in enumerate(band_names):
            orig_p = orig_powers[band_name]
            filt_p = filt_powers[band_name]
            if orig_p.dim() == 0: orig_p = orig_p.unsqueeze(0).expand(B)
            if filt_p.dim() == 0: filt_p = filt_p.unsqueeze(0).expand(B)
            reduction = torch.clamp((orig_p - filt_p) / (orig_p + 1e-8), -1.0, 1.0)
            base_target = self.target_tensor[i].to(device)
            if base_target.dim() > 0: base_target = base_target[0]
            if base_target > 0.1:
                current_target = base_target * adaptive_scale.squeeze()
            else:
                current_target = base_target.expand(B)
            error = current_target - reduction
            reductions.append(reduction)
            errors.append(error)

        reductions_tensor = torch.stack(reductions, dim=-1)
        errors_tensor = torch.stack(errors, dim=-1)

        if self.use_dynamic_weights:
            weights = self.dynamic_weights(errors_tensor.detach())
        else:
            weights = self.fixed_weights.to(device).unsqueeze(0).expand(B, -1)

        total_loss = None
        loss_details = {}
        for i, band_name in enumerate(band_names):
            reduction = reductions_tensor[:, i]
            base_target_val = self.target_tensor[i].to(device)
            tolerance = self.tolerance_tensor[i].to(device)
            weight = weights[:, i]
            if base_target_val > 0.1:
                current_target = base_target_val * adaptive_scale.squeeze()
                band_loss = ((F.relu(current_target - tolerance - reduction) +
                             F.relu(reduction - current_target - tolerance) * 0.5) * weight).mean()
            else:
                band_loss = ((F.relu(reduction - tolerance) * weight).mean() * 2.0 +
                            (F.relu(-reduction - 0.15) * weight).mean())
            total_loss = band_loss if total_loss is None else total_loss + band_loss
            loss_details[f'{band_name}_reduction'] = safe_to_item(reduction.mean())

        if self.use_dynamic_weights:
            self.dynamic_weights.update_ema(weights.detach())
        return total_loss, loss_details

    def step_epoch(self):
        if self.use_dynamic_weights:
            self.dynamic_weights.step_epoch()


class SSIMStructureLoss(nn.Module):
    """(ii) SSIM structural preservation loss L_ssim (lambda = 0.25)."""
    def __init__(self, window_size=11, min_filtering=0.03, ssim_target=0.85):
        super().__init__()
        self.window_size = window_size
        self.min_filtering = min_filtering
        self.ssim_target = ssim_target
        gauss = torch.exp(torch.tensor(
            [-(x - window_size // 2)**2 / (2 * 1.5**2) for x in range(window_size)]))
        gauss = gauss / gauss.sum()
        self.register_buffer('window',
            (gauss.unsqueeze(1) @ gauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0))

    def forward(self, original, denoised):
        C1, C2 = 0.01**2, 0.03**2
        window = self.window.to(original.device)
        pad = self.window_size // 2
        orig_pad = F.pad(original, (pad,)*4, mode='reflect')
        deno_pad = F.pad(denoised, (pad,)*4, mode='reflect')
        mu1 = F.conv2d(orig_pad, window, padding=0)
        mu2 = F.conv2d(deno_pad, window, padding=0)
        sigma1_sq = F.conv2d(orig_pad*orig_pad, window, padding=0) - mu1**2
        sigma2_sq = F.conv2d(deno_pad*deno_pad, window, padding=0) - mu2**2
        sigma12 = F.conv2d(orig_pad*deno_pad, window, padding=0) - mu1*mu2
        ssim_map = ((2*mu1*mu2+C1)*(2*sigma12+C2)) / ((mu1**2+mu2**2+C1)*(sigma1_sq+sigma2_sq+C2))
        ssim_value = ssim_map.mean()
        # Unidirectional penalty: only when SSIM < target (Section III.D)
        ssim_loss = F.relu(self.ssim_target - ssim_value)
        return ssim_loss, {'ssim_value': ssim_value.item()}


class EdgePreservationLoss(nn.Module):
    """(iii) Edge preservation loss L_edge (lambda = 0.20)."""
    def __init__(self, weight=2.0):
        super().__init__()
        self.weight = weight
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1,1,3,3))
        self.register_buffer('sobel_y', sobel_y.view(1,1,3,3))

    def forward(self, original, denoised, edge_mask=None):
        device = original.device
        # Sobel gradient magnitude (Section III.D)
        orig_edge = torch.sqrt(
            reflect_conv2d(original, self.sobel_x.to(device), 1)**2 +
            reflect_conv2d(original, self.sobel_y.to(device), 1)**2 + 1e-8)
        deno_edge = torch.sqrt(
            reflect_conv2d(denoised, self.sobel_x.to(device), 1)**2 +
            reflect_conv2d(denoised, self.sobel_y.to(device), 1)**2 + 1e-8)
        if edge_mask is None:
            # Binary mask M selecting top ~16% edge pixels:
            # M = {magnitude > mean + std(magnitude)} (Section III.D)
            edge_threshold = orig_edge.mean() + orig_edge.std()
            edge_mask = (orig_edge > edge_threshold).float()
        edge_diff = (orig_edge - deno_edge).abs()
        edge_loss = (edge_diff * edge_mask).sum() / (edge_mask.sum() + 1e-8)
        return edge_loss * self.weight, {'edge_diff_loss': edge_loss.item()}


class DynamicMinFilteringLoss(nn.Module):
    """(iv) Dynamic minimum filtering loss L_min (lambda = 0.10)."""
    def __init__(self, base_min=0.01, scale=0.04, warmup_epochs=5):
        super().__init__()
        self.base_min = base_min
        self.scale = scale
        self.warmup_epochs = warmup_epochs

    def forward(self, original, denoised, alpha_map, complexity_info, current_epoch=None):
        if complexity_info is not None:
            # c_hf: high-frequency energy ratio from complexity vector
            complexity_weight = complexity_info[:, 2].detach()
        else:
            complexity_weight = torch.ones(original.shape[0], device=original.device)

        diff = (original - denoised).abs().mean(dim=(1, 2, 3))
        # Mean filter strength = spatial average of the alpha map
        avg_alpha = alpha_map.mean(dim=(1, 2, 3))

        if current_epoch is not None and current_epoch < self.warmup_epochs:
            # Warmup (epoch < 5): fixed threshold = 0.03 (Section III.D)
            dynamic_min = torch.full_like(avg_alpha, 0.03)
        else:
            # Post-warmup: base + scale * mean_alpha (Section S4)
            dynamic_min = self.base_min + self.scale * avg_alpha.detach()

        adaptive_min = dynamic_min * complexity_weight
        min_filter_loss = F.relu(adaptive_min - diff).mean()

        return min_filter_loss, {
            'avg_alpha': avg_alpha.mean().item(),
            'avg_diff': diff.mean().item(),
        }


class LocalSmoothnessLoss(nn.Module):
    """
    (v) Spatial smoothness loss L_smooth (lambda = 0.10).
    TV of the kernel weight map W (7x256x256) and alpha map (1x256x256).
    Penalizes abrupt spatial transitions (Ref. 23).
    """
    def forward(self, kernel_weights, alpha_map, local_params=None):
        def compute_tv(x):
            return ((x[..., :, 1:] - x[..., :, :-1]).abs().mean() +
                    (x[..., 1:, :] - x[..., :-1, :]).abs().mean())
        return compute_tv(kernel_weights) + compute_tv(alpha_map)


class AdaptiveCompositeLoss(nn.Module):
    """
    Composite loss function with five terms (Section III.D).

    L_total = 0.25*L_fq + 0.25*L_ssim + 0.20*L_edge + 0.10*L_min + 0.10*L_smooth
    """
    def __init__(self, filter_type, use_dynamic_weights=True):
        super().__init__()
        self.filter_quality = FrequencyBandQualityLoss(filter_type, use_dynamic_weights)
        self.ssim_structure = SSIMStructureLoss(11, config.SSIM_MIN_FILTERING, config.SSIM_TARGET)
        self.edge_preservation = EdgePreservationLoss(weight=config.EDGE_PRESERVATION_WEIGHT)
        self.dynamic_min_filtering = DynamicMinFilteringLoss(
            config.MIN_FILTERING_BASE, config.MIN_FILTERING_SCALE,
            config.DYNAMIC_MIN_WARMUP_EPOCHS)
        self.local_smoothness = LocalSmoothnessLoss()
        self.weights = config.LOSS_WEIGHTS
        self.current_epoch = 0

    def forward(self, output, batch):
        denoised = torch.nan_to_num(output['denoised'], nan=0.0)
        original = torch.nan_to_num(batch['image'].to(denoised.device), nan=0.0)
        alpha_map = output['alpha_map']
        edge_mask = output.get('edge_mask', None)
        complexity_info = output.get('complexity_info', None)

        losses = {}

        fq_loss, fq_details = self.filter_quality(original, denoised, complexity_info)
        losses['frequency_quality'] = fq_loss
        losses.update(fq_details)

        ssim_loss, ssim_details = self.ssim_structure(original, denoised)
        losses['ssim_structure'] = ssim_loss
        losses.update(ssim_details)

        edge_loss, edge_details = self.edge_preservation(original, denoised, edge_mask)
        losses['edge_preservation'] = edge_loss
        losses.update(edge_details)

        dmf_loss, dmf_details = self.dynamic_min_filtering(
            original, denoised, alpha_map, complexity_info, self.current_epoch)
        losses['dynamic_min_filtering'] = dmf_loss
        losses.update(dmf_details)

        losses['local_smoothness'] = self.local_smoothness(
            output.get('kernel_weights'), alpha_map, output.get('local_params', {}))

        # Sanitize NaN/Inf losses
        for name in list(self.weights.keys()):
            val = losses.get(name)
            if val is None or (isinstance(val, torch.Tensor) and
                              (torch.isnan(val).any() or torch.isinf(val).any())):
                losses[name] = original.new_zeros(1).squeeze()

        total_loss = sum(self.weights.get(n, 0.1) * losses[n]
                        for n in self.weights.keys() if n in losses)
        loss_dict = {name: safe_to_item(val) for name, val in losses.items()}
        loss_dict['total'] = safe_to_item(total_loss)
        return total_loss, loss_dict

    def step_epoch(self):
        self.current_epoch += 1
        self.filter_quality.step_epoch()


# ==========================================================================
# Dataset
# ==========================================================================

class KPFMDataset(Dataset):
    """
    Dataset loading TIFF images and pre-extracted features.

    Features are z-score normalized using Phase 2 statistics.
    Images are min-max normalized to [0, 1].
    """
    def __init__(self, file_list, features_df, feature_columns, feature_stats=None):
        self.file_list = file_list
        self.features_df = features_df
        self.feature_columns = feature_columns

        self.feature_stats = {}
        if feature_stats is not None:
            self.feature_stats = feature_stats
        else:
            for col in self.feature_columns:
                if col in features_df.columns:
                    vals = features_df[col].dropna()
                    if len(vals) > 0:
                        std = vals.std()
                        self.feature_stats[col] = {
                            'mean': float(vals.mean()),
                            'std': float(std) if std > 0 else 1.0}

        self.filename_to_idx = {}
        if 'filename' in features_df.columns:
            for idx, row in features_df.iterrows():
                self.filename_to_idx[row['filename']] = idx

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        filename = os.path.basename(filepath)
        image, metadata = load_tiff(filepath)

        if image is None:
            return {
                'image': torch.zeros(1, config.IMAGE_SIZE, config.IMAGE_SIZE),
                'features': torch.zeros(len(self.feature_columns)),
                'scale_info': torch.zeros(4),
                'filepath': filepath, 'valid': False}

        scale_info = torch.tensor([
            metadata['original_min'], metadata['original_max'],
            metadata['original_mean'], metadata['original_std'],
        ], dtype=torch.float32)

        if image.shape[0] != config.IMAGE_SIZE or image.shape[1] != config.IMAGE_SIZE:
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            image_tensor = F.interpolate(image_tensor,
                size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
                mode='bilinear', align_corners=False)
            image = image_tensor.squeeze().numpy()

        # Min-max normalization to [0, 1]
        img_min, img_max = image.min(), image.max()
        if img_max - img_min > 1e-8:
            image = (image - img_min) / (img_max - img_min)
        else:
            image = np.zeros_like(image)

        features = self._get_features(filename)

        return {
            'image': torch.tensor(image, dtype=torch.float32).unsqueeze(0),
            'features': features,
            'scale_info': scale_info,
            'filepath': filepath, 'valid': True}

    def _get_features(self, filename):
        """Look up and z-score normalize 22 features by filename."""
        row_idx = self.filename_to_idx.get(filename, None)
        feature_values = []
        for col in self.feature_columns:
            if row_idx is not None and col in self.features_df.columns:
                val = self.features_df.at[row_idx, col]
                st = self.feature_stats.get(col, {'mean': 0.0, 'std': 1.0})
                val = np.clip((val - st['mean']) / st['std'], -3, 3)
                feature_values.append(float(val))
            else:
                feature_values.append(0.0)
        return torch.tensor(feature_values, dtype=torch.float32)


# ==========================================================================
# Trainer (Dual: dynamic weights + fixed weights)
# ==========================================================================

class DualTrainer:
    """
    Trains two models in parallel:
        - model_dynamic: with learned per-frequency-band weighting
        - model_fixed:   with fixed loss weights (baseline comparison)

    Uses validation loss for early stopping and saves the best model.
    Training uses AdamW (Ref. 24) with CosineAnnealingWarmRestarts (Ref. 25).
    """
    def __init__(self, filter_type, num_features, output_dir, enable_local_params=True):
        self.filter_type = filter_type
        self.output_dir = output_dir

        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)

        self.model_dynamic = AdaptiveFilterModel(
            filter_type, num_features, enable_local_params).to(DEVICE)
        self.model_fixed = AdaptiveFilterModel(
            filter_type, num_features, enable_local_params).to(DEVICE)

        self.loss_dynamic = AdaptiveCompositeLoss(filter_type, use_dynamic_weights=True).to(DEVICE)
        self.loss_fixed = AdaptiveCompositeLoss(filter_type, use_dynamic_weights=False).to(DEVICE)

        self.optimizer_dynamic = AdamW(self.model_dynamic.parameters(),
            lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.optimizer_fixed = AdamW(self.model_fixed.parameters(),
            lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

        # CosineAnnealingWarmRestarts: T_0=20, T_mult=2 (Section III.D)
        self.scheduler_dynamic = CosineAnnealingWarmRestarts(
            self.optimizer_dynamic, T_0=20, T_mult=2)
        self.scheduler_fixed = CosineAnnealingWarmRestarts(
            self.optimizer_fixed, T_0=20, T_mult=2)

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': [], 'stability': []}

    def save_checkpoint(self, epoch, train_loss, val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_dynamic': self.model_dynamic.state_dict(),
            'model_fixed': self.model_fixed.state_dict(),
            'optimizer_dynamic': self.optimizer_dynamic.state_dict(),
            'optimizer_fixed': self.optimizer_fixed.state_dict(),
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'train_loss': train_loss, 'val_loss': val_loss,
            'history': self.history,
        }
        torch.save(checkpoint, os.path.join(
            self.output_dir, 'checkpoints', f'latest_checkpoint_{self.filter_type}.pt'))

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            torch.save(self.model_dynamic.state_dict(), os.path.join(
                self.output_dir, f'best_model_dynamic_{self.filter_type}.pt'))
            torch.save(self.model_fixed.state_dict(), os.path.join(
                self.output_dir, f'best_model_fixed_{self.filter_type}.pt'))
            print(f"  * Best model saved (val_loss={val_loss:.4f})")
        else:
            self.patience_counter += 1

    def load_checkpoint(self):
        path = os.path.join(self.output_dir, 'checkpoints',
                           f'latest_checkpoint_{self.filter_type}.pt')
        if os.path.exists(path) and RunConfig.RESUME_FROM_CHECKPOINT:
            print(f"  Resuming {self.filter_type} from checkpoint...")
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            self.model_dynamic.load_state_dict(ckpt['model_dynamic'])
            self.model_fixed.load_state_dict(ckpt['model_fixed'])
            self.optimizer_dynamic.load_state_dict(ckpt['optimizer_dynamic'])
            self.optimizer_fixed.load_state_dict(ckpt['optimizer_fixed'])
            self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
            self.patience_counter = ckpt.get('patience_counter', 0)
            self.history = ckpt.get('history',
                {'train_loss': [], 'val_loss': [], 'stability': []})
            return ckpt['epoch']
        return 0

    def evaluate(self, val_loader):
        self.model_dynamic.eval()
        total_loss, num_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                if not check_batch_valid(batch): continue
                image = batch['image'].to(DEVICE)
                features = batch['features'].to(DEVICE)
                scale_info = batch['scale_info'].to(DEVICE)
                output = self.model_dynamic(image, features, scale_info)
                loss, _ = self.loss_dynamic(output, batch)
                total_loss += loss.item()
                num_batches += 1
        return total_loss / max(num_batches, 1)

    def plot_history(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(self.history['train_loss'], label='Train', color='blue')
        if self.history['val_loss']:
            axes[0].plot(self.history['val_loss'], label='Val', color='red', linestyle='--')
        axes[0].set_title(f'{self.filter_type} - Loss Curve')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3); axes[0].legend()

        if self.history['stability']:
            axes[1].plot(self.history['stability'], label='Stability (Variance)', color='orange')
            axes[1].set_title('Training Stability')
            axes[1].set_xlabel('Epoch')
            axes[1].grid(True, alpha=0.3); axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'logs', f'history_{self.filter_type}.png'))
        plt.close()

    def validate_and_visualize(self, val_loader, num_samples=10):
        print(f"  Generating validation visualizations for {self.filter_type}...")
        self.model_dynamic.eval()
        count = 0
        viz_dir = os.path.join(self.output_dir, 'visualizations', self.filter_type)
        os.makedirs(viz_dir, exist_ok=True)

        with torch.no_grad():
            for batch in val_loader:
                if not check_batch_valid(batch): continue
                image = batch['image'].to(DEVICE)
                output = self.model_dynamic(image, batch['features'].to(DEVICE),
                                           batch['scale_info'].to(DEVICE))
                denoised = output['denoised']
                for i in range(image.shape[0]):
                    if count >= num_samples: break
                    orig_np = image[i, 0].cpu().numpy()
                    deno_np = denoised[i, 0].cpu().numpy()
                    error_map = np.abs(orig_np - deno_np)

                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    im0 = axs[0].imshow(orig_np, cmap='viridis')
                    axs[0].set_title('Original'); plt.colorbar(im0, ax=axs[0])
                    im1 = axs[1].imshow(deno_np, cmap='viridis')
                    axs[1].set_title(f'Filtered ({self.filter_type})'); plt.colorbar(im1, ax=axs[1])
                    im2 = axs[2].imshow(error_map, cmap='inferno')
                    axs[2].set_title('|Original - Filtered|'); plt.colorbar(im2, ax=axs[2])
                    plt.suptitle(f"Sample {count+1} - {self.filter_type.upper()}", fontsize=14)
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, f'val_sample_{count}.png'))
                    plt.close()
                    count += 1
                if count >= num_samples: break

    def train(self, train_loader, val_loader, num_epochs):
        start_epoch = self.load_checkpoint()
        if start_epoch >= num_epochs:
            print(f"  {self.filter_type} already completed.")
            return

        print(f"  Training {self.filter_type} from epoch {start_epoch + 1}")
        loss_buffer = []

        for epoch in range(start_epoch, num_epochs):
            self.model_dynamic.train()
            self.model_fixed.train()
            total_loss, num_batches = 0, 0

            pbar = tqdm(train_loader, desc=f"[{self.filter_type}] Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                if not check_batch_valid(batch): continue
                image = batch['image'].to(DEVICE)
                features = batch['features'].to(DEVICE)
                scale_info = batch['scale_info'].to(DEVICE)

                # Dynamic model update
                self.optimizer_dynamic.zero_grad()
                out_d = self.model_dynamic(image, features, scale_info)
                loss_d, _ = self.loss_dynamic(out_d, batch)
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(self.model_dynamic.parameters(), config.GRADIENT_CLIP)
                self.optimizer_dynamic.step()

                # Fixed model update
                self.optimizer_fixed.zero_grad()
                out_f = self.model_fixed(image, features, scale_info)
                loss_f, _ = self.loss_fixed(out_f, batch)
                loss_f.backward()
                torch.nn.utils.clip_grad_norm_(self.model_fixed.parameters(), config.GRADIENT_CLIP)
                self.optimizer_fixed.step()

                current_loss = loss_d.item()
                total_loss += current_loss
                num_batches += 1
                pbar.set_postfix({'loss': f"{current_loss:.4f}"})

            avg_train_loss = total_loss / max(num_batches, 1)
            avg_val_loss = self.evaluate(val_loader)

            loss_buffer.append(avg_train_loss)
            stability = np.var(loss_buffer[-5:]) if len(loss_buffer) >= 5 else 0.0

            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['stability'].append(stability)

            print(f"  [{self.filter_type}] Epoch {epoch+1} | "
                  f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
                  f"Stability: {stability:.6f} | "
                  f"Patience: {self.patience_counter}/{config.PATIENCE}")

            self.save_checkpoint(epoch + 1, avg_train_loss, avg_val_loss)
            self.plot_history()
            self.scheduler_dynamic.step()
            self.scheduler_fixed.step()
            try:
                self.loss_dynamic.step_epoch()
                self.loss_fixed.step_epoch()
            except Exception as e:
                print(f"  Warning: step_epoch - {e}")

            if self.patience_counter >= config.PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    def save_final(self):
        torch.save(self.model_dynamic.state_dict(), os.path.join(
            self.output_dir, f'model_dynamic_final_{self.filter_type}.pt'))
        torch.save(self.model_fixed.state_dict(), os.path.join(
            self.output_dir, f'model_fixed_final_{self.filter_type}.pt'))


# ==========================================================================
# Main Execution
# ==========================================================================

def main():
    print("=" * 70)
    print("  Phase 3: Adaptive Filter Training")
    print("=" * 70)
    print(f"  Output: {TRAINING_OUTPUT}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    os.makedirs(TRAINING_OUTPUT, exist_ok=True)

    # Step 1: Load feature columns
    print("\n[Step 1] Loading feature columns...")
    if os.path.exists(FEATURE_COLS_JSON):
        with open(FEATURE_COLS_JSON, 'r') as f:
            feature_columns = json.load(f)
        print(f"  Loaded {len(feature_columns)} feature columns")
    else:
        print(f"  [WARNING] feature_columns.json not found. Using defaults.")
        feature_columns = [f'feature_{i}' for i in range(config.DEFAULT_NUM_FEATURES)]
    num_features = len(feature_columns)

    # Step 2: Load features CSV
    print("\n[Step 2] Loading features CSV...")
    if not os.path.exists(FEATURES_CSV):
        print(f"  [ERROR] {FEATURES_CSV} not found.")
        return
    features_df = pd.read_csv(FEATURES_CSV)
    print(f"  Loaded {len(features_df)} rows")
    if 'is_outlier' in features_df.columns:
        n_before = len(features_df)
        features_df = features_df[features_df['is_outlier'] == False].reset_index(drop=True)
        print(f"  Filtered outliers: {n_before} -> {len(features_df)}")

    feature_stats = None
    if os.path.exists(FEATURE_STATS_CSV):
        sdf = pd.read_csv(FEATURE_STATS_CSV)
        feature_stats = {row['feature']: {'mean': row['mean'],
            'std': row['std'] if row['std'] > 0 else 1.0} for _, row in sdf.iterrows()}

    # Step 3: Collect TIFF files
    print("\n[Step 3] Scanning normal file directory...")
    files = []
    if os.path.exists(NORMAL_FILES_DIR):
        for fname in os.listdir(NORMAL_FILES_DIR):
            if fname.lower().endswith(('.tiff', '.tif')):
                files.append(os.path.join(NORMAL_FILES_DIR, fname))
    valid_filenames = set(features_df['filename'].values)
    files = [f for f in files if os.path.basename(f) in valid_filenames]
    print(f"  Found {len(files)} files")
    if not files:
        print("[ERROR] No valid files found.")
        return

    # Step 4: Dataset splits (85/10/5)
    print("\n[Step 4] Preparing dataset splits...")
    full_dataset = KPFMDataset(files, features_df, feature_columns, feature_stats)
    total = len(full_dataset)
    test_size = int(total * config.TEST_RATIO)
    val_size = int(total * config.VAL_RATIO)
    train_size = total - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(RunConfig.RANDOM_SEED))
    print(f"  Train: {train_size} | Val: {val_size} | Test: {test_size}")

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.NUM_WORKERS)

    test_indices = test_dataset.indices
    test_files = [files[i] for i in test_indices]
    with open(os.path.join(TRAINING_OUTPUT, 'test_file_list.json'), 'w') as f:
        json.dump(test_files, f, indent=2)

    # Step 5: Train per filter type
    if RunConfig.RUN_MODE == 'all':
        filters_to_run = config.ALL_FILTERS
    elif RunConfig.RUN_MODE == 'selected':
        filters_to_run = RunConfig.SELECTED_FILTERS
    else:
        filters_to_run = [RunConfig.SINGLE_FILTER]

    print(f"\n[Step 5] Scheduled filters: {filters_to_run}")
    training_summary = {}

    for f_type in filters_to_run:
        print(f"\n{'=' * 60}")
        print(f"  Training Filter: {f_type.upper()}")
        print(f"{'=' * 60}")
        try:
            trainer = DualTrainer(f_type, num_features, TRAINING_OUTPUT)
            trainer.train(train_loader, val_loader, config.NUM_EPOCHS)
            trainer.save_final()
            if val_size > 0:
                trainer.validate_and_visualize(val_loader, num_samples=10)
            training_summary[f_type] = {
                'final_train_loss': (trainer.history['train_loss'][-1]
                                    if trainer.history['train_loss'] else None),
                'best_val_loss': trainer.best_val_loss,
                'epochs_trained': len(trainer.history['train_loss']),
                'status': 'completed'}
        except Exception as e:
            print(f"  Failed: {f_type.upper()} - {e}")
            import traceback; traceback.print_exc()
            training_summary[f_type] = {'status': 'failed', 'error': str(e)}
        torch.cuda.empty_cache()

    with open(os.path.join(TRAINING_OUTPUT, 'training_summary.json'), 'w') as f:
        json.dump(training_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("  Phase 3 Training Complete")
    print(f"  Results: {TRAINING_OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
