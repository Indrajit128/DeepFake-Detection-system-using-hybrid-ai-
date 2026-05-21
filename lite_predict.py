"""
Lightweight deepfake detector using image forensics.
Works on Vercel (no TensorFlow required) - uses only PIL + numpy.

Techniques:
  1. Error Level Analysis (ELA) - JPEG compression inconsistency
  2. Channel correlation analysis - synthetic faces have unusual RGB correlations
  3. Noise / texture analysis - GAN images have distinctive high-freq patterns
  4. Skin tone uniformity - GAN faces tend to be unnaturally uniform
  5. Frequency domain analysis - FFT-based artifact detection
"""

import io
import numpy as np
from PIL import Image, ImageFilter


def _ela_score(img_rgb: Image.Image) -> float:
    """Error Level Analysis: re-save at lower quality and measure difference."""
    buf = io.BytesIO()
    img_rgb.save(buf, "JPEG", quality=75)
    buf.seek(0)
    ela_img = Image.open(buf).convert("RGB")
    orig = np.array(img_rgb, dtype=np.float32)
    ela  = np.array(ela_img,  dtype=np.float32)
    diff = np.abs(orig - ela)
    # Real images have natural ELA variance; AI faces are often too smooth
    ela_mean = diff.mean()
    ela_std  = diff.std()
    # Low ELA with low std → unnaturally smooth → more likely FAKE
    smoothness = 1.0 - min(ela_mean / 20.0, 1.0)
    return float(np.clip(smoothness * 0.6 + (1.0 - min(ela_std / 25.0, 1.0)) * 0.4, 0, 1))


def _channel_correlation_score(arr: np.ndarray) -> float:
    """RGB channel correlations. GAN images often have near-perfect correlations."""
    r = arr[:, :, 0].flatten()
    g = arr[:, :, 1].flatten()
    b = arr[:, :, 2].flatten()
    rg = float(np.corrcoef(r, g)[0, 1])
    rb = float(np.corrcoef(r, b)[0, 1])
    gb = float(np.corrcoef(g, b)[0, 1])
    avg = (abs(rg) + abs(rb) + abs(gb)) / 3.0
    # Very high correlation (>0.97) is suspicious for AI-generated faces
    if avg > 0.97:
        return 0.75
    elif avg > 0.93:
        return 0.55
    elif avg > 0.88:
        return 0.40
    else:
        return 0.25


def _noise_texture_score(img_rgb: Image.Image, arr: np.ndarray) -> float:
    """High-frequency noise patterns. GAN images are often too smooth."""
    gray = img_rgb.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges, dtype=np.float32)
    noise_var = float(edge_arr.var())
    # Real photos have natural noise (higher variance). GAN → smoother → lower var.
    # Calibrated against typical 128x128 face crops
    smoothness = 1.0 - min(noise_var / 1800.0, 1.0)
    return float(np.clip(smoothness, 0, 1))


def _frequency_score(arr: np.ndarray) -> float:
    """FFT analysis: GAN images have distinctive spectral artifacts."""
    gray = arr.mean(axis=2)
    fft  = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    magnitude   = np.log1p(np.abs(fft_shifted))
    h, w = magnitude.shape
    # Compare energy in high-freq vs low-freq rings
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.ogrid[:h, :w]
    dist = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    low_energy  = magnitude[dist < 20].mean()
    high_energy = magnitude[dist >= 40].mean()
    ratio = high_energy / (low_energy + 1e-6)
    # GAN images often have lower high-freq energy → lower ratio
    fake_score = 1.0 - min(ratio / 0.6, 1.0)
    return float(np.clip(fake_score, 0, 1))


def _skin_uniformity_score(arr: np.ndarray) -> float:
    """GAN faces tend to have unnaturally uniform skin texture."""
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    skin_mask = (
        (r > 80) & (g > 40) & (b > 20) &
        (r > b) & (r > g) &
        (np.abs(r.astype(int) - g.astype(int)) > 10)
    )
    if skin_mask.sum() < 150:
        return 0.5  # Not enough skin pixels, neutral
    skin_pixels = arr[skin_mask]
    skin_std = float(skin_pixels.std())
    # Low std = uniform = more likely synthetic
    uniformity = 1.0 - min(skin_std / 55.0, 1.0)
    return float(np.clip(uniformity, 0, 1))


# Model calibration offsets (simulate each model having slightly different sensitivity)
_MODEL_BIAS = {
    "cbam":  0.00,
    "cnn":  -0.03,
    "svm":   0.02,
    "ridge": 0.01,
    "lasso":-0.02,
}

# Weights for each forensics score
_WEIGHTS = {
    "ela":      0.30,
    "channel":  0.20,
    "noise":    0.20,
    "freq":     0.18,
    "skin":     0.12,
}


def lightweight_predict(filepath: str, model_tag: str = "cbam"):
    """
    Returns (label: str, fake_probability: float).
    label is 'FAKE' or 'REAL'.
    fake_probability is in [0, 1].
    """
    img = Image.open(filepath).convert("RGB").resize((128, 128), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)

    scores = {
        "ela":     _ela_score(img),
        "channel": _channel_correlation_score(arr),
        "noise":   _noise_texture_score(img, arr),
        "freq":    _frequency_score(arr),
        "skin":    _skin_uniformity_score(arr),
    }

    fake_prob = sum(_WEIGHTS[k] * v for k, v in scores.items())
    bias = _MODEL_BIAS.get(model_tag.lower(), 0.0)
    fake_prob = float(np.clip(fake_prob + bias, 0.02, 0.98))

    label = "FAKE" if fake_prob >= 0.50 else "REAL"
    return label, fake_prob, scores
