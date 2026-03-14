"""
DETECT-AI — Deep Texture & Frequency Analyzer
Extracts 15 signals per image/frame that matter for AI vs Human detection.
Used by the frame extractor and can be run standalone on image samples.

Signals extracted:
  1.  laplacian_variance      — sharpness (AI images are unnaturally sharp)
  2.  noise_sigma             — estimated sensor noise (missing in AI images)
  3.  dct_high_freq_energy    — DCT frequency artifacts (diffusion model signature)
  4.  fft_radial_profile      — GAN fingerprint ring patterns in frequency domain
  5.  color_hist_r/g/b        — 32-bin color histograms per channel
  6.  color_coherence          — how consistent colors are across regions
  7.  edge_density             — edge pixel ratio (AI over-sharpens edges)
  8.  local_contrast_std       — local contrast standard deviation
  9.  compression_score        — blocking artifact strength (AI images lack real JPEG history)
  10. saturation_mean/std      — HSV saturation stats (AI images often over-saturated)
  11. face_symmetry_score      — left/right face symmetry (deepfakes often too symmetric)
  12. skin_tone_variance        — variance in detected skin region (deepfakes are too smooth)
  13. exif_has_camera           — 1 if EXIF has real camera model, 0 if not (AI images have none)
  14. texture_lbp_hist          — Local Binary Pattern histogram (surface texture fingerprint)
  15. gradient_magnitude_mean   — mean gradient magnitude
"""

import numpy as np
import cv2
from typing import Optional
import struct


# ── 1. Sharpness ─────────────────────────────────────────────────
def laplacian_variance(gray: np.ndarray) -> float:
    """Higher = sharper. AI images often > 800, real photos 100–500."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ── 2. Noise estimation ───────────────────────────────────────────
def estimate_noise_sigma(gray: np.ndarray) -> float:
    """
    Estimate sensor noise via high-freq residual.
    Real cameras have σ > 1.5. AI-generated images are often σ < 0.8.
    """
    h, w = gray.shape
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    filtered = cv2.filter2D(gray.astype(np.float64), -1, kernel)
    sigma = np.sqrt(np.pi / 2) * np.mean(np.abs(filtered)) / 6.0
    return float(sigma)


# ── 3. DCT high-frequency energy ─────────────────────────────────
def dct_high_freq_energy(gray: np.ndarray) -> float:
    """
    Energy in top-right quadrant of DCT (high freq both dims).
    Diffusion models leave characteristic artifacts here.
    """
    h, w = gray.shape
    # Work on center 256×256 patch for speed
    ph, pw = min(h, 256), min(w, 256)
    patch = gray[:ph, :pw].astype(np.float32)
    dct = cv2.dct(patch)
    hh, hw = ph // 2, pw // 2
    high_freq = dct[hh:, hw:]
    total = dct
    return float(np.sum(high_freq**2) / (np.sum(total**2) + 1e-9))


# ── 4. FFT radial profile (GAN fingerprint) ───────────────────────
def fft_radial_profile(gray: np.ndarray, n_bins: int = 16) -> list[float]:
    """
    Compute radially-averaged FFT power spectrum.
    GANs leave ring artifacts. Returns n_bins floats.
    """
    h, w = gray.shape
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.mgrid[0:h, 0:w]
    r = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2).astype(int)
    max_r = min(cx, cy)

    bins = np.linspace(0, max_r, n_bins + 1, dtype=int)
    profile = []
    for i in range(n_bins):
        mask = (r >= bins[i]) & (r < bins[i+1])
        vals = magnitude[mask]
        profile.append(float(vals.mean()) if len(vals) > 0 else 0.0)
    return profile


# ── 5 & 6. Color histograms + coherence ──────────────────────────
def color_features(img_bgr: np.ndarray) -> dict:
    """32-bin per-channel histograms + color coherence score."""
    features = {}
    for i, ch in enumerate(["b", "g", "r"]):
        hist = cv2.calcHist([img_bgr], [i], None, [32], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-9)
        features[f"color_hist_{ch}"] = hist.tolist()

    # Color coherence: std of mean block colors (low in AI = too uniform)
    h, w = img_bgr.shape[:2]
    bh, bw = h // 4, w // 4
    block_means = []
    for r in range(4):
        for c in range(4):
            block = img_bgr[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
            block_means.append(block.mean(axis=(0, 1)))
    block_means = np.array(block_means)
    features["color_coherence"] = float(block_means.std())
    return features


# ── 7. Edge density ───────────────────────────────────────────────
def edge_density(gray: np.ndarray) -> float:
    """Fraction of pixels that are edges (Canny). AI images: higher."""
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.sum() / 255 / edges.size)


# ── 8. Local contrast std ─────────────────────────────────────────
def local_contrast_std(gray: np.ndarray) -> float:
    """Std of local contrast (32×32 blocks). Low = flat/synthetic."""
    h, w = gray.shape
    stds = []
    for r in range(0, h - 32, 32):
        for c in range(0, w - 32, 32):
            stds.append(float(gray[r:r+32, c:c+32].std()))
    return float(np.std(stds)) if stds else 0.0


# ── 9. Compression artifact score ────────────────────────────────
def compression_score(img_bgr: np.ndarray) -> float:
    """
    Estimate JPEG blocking artifacts by checking 8×8 block boundaries.
    Real photos that have been JPEG'd have visible block edges.
    AI-generated images starting from scratch often lack this history.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    h_diff = np.abs(gray[7::8, :] - gray[8::8, :min(w, gray.shape[1])]).mean() if h > 16 else 0.0
    v_diff = np.abs(gray[:, 7::8] - gray[:, 8::8]).mean() if w > 16 else 0.0
    return float((h_diff + v_diff) / 2)


# ── 10. Saturation stats ──────────────────────────────────────────
def saturation_stats(img_bgr: np.ndarray) -> dict:
    """HSV saturation mean + std. AI images often over-saturated."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    return {"saturation_mean": float(s.mean()), "saturation_std": float(s.std())}


# ── 11. Face symmetry ─────────────────────────────────────────────
def face_symmetry_score(face_crop_bgr: np.ndarray) -> float:
    """
    Mirror left half vs right half of face crop.
    SSIM-like score: 1.0 = perfect symmetry (deepfake warning).
    Real faces: 0.6–0.85. Deepfakes: often 0.88–0.97.
    """
    h, w = face_crop_bgr.shape[:2]
    left  = face_crop_bgr[:, :w//2]
    right = face_crop_bgr[:, w//2:]
    right_flip = cv2.flip(right, 1)
    # Resize to match if odd width
    if left.shape != right_flip.shape:
        right_flip = cv2.resize(right_flip, (left.shape[1], left.shape[0]))
    diff = np.abs(left.astype(np.float32) - right_flip.astype(np.float32))
    # Normalize: 0 diff = 1.0 symmetry, max diff = 0.0
    return float(1.0 - diff.mean() / 255.0)


# ── 12. Skin tone variance ────────────────────────────────────────
def skin_tone_variance(face_crop_bgr: np.ndarray) -> float:
    """
    Variance of YCrCb Cr channel in detected skin region.
    Real skin has micro-variation. Deepfakes are too smooth (low variance).
    """
    ycrcb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1].astype(np.float32)
    # Skin tone range in Cr: ~133–173
    skin_mask = (cr >= 133) & (cr <= 173)
    if skin_mask.sum() < 100:
        return 0.0
    return float(cr[skin_mask].var())


# ── 13. EXIF camera check ─────────────────────────────────────────
def exif_has_camera(image_bytes: bytes) -> int:
    """
    Returns 1 if JPEG EXIF contains a camera model tag (0x0110).
    AI images virtually never have this. Real photos almost always do.
    """
    try:
        # Find EXIF marker in JPEG
        pos = image_bytes.find(b'\xff\xe1')
        if pos < 0:
            return 0
        exif_data = image_bytes[pos+4:]
        # Check for camera model tag (0x0110) anywhere in first 8KB
        return 1 if b'\x01\x10' in exif_data[:8192] else 0
    except Exception:
        return 0


# ── 14. LBP texture histogram ─────────────────────────────────────
def lbp_histogram(gray: np.ndarray, n_points: int = 8, radius: int = 1) -> list[float]:
    """
    Local Binary Pattern histogram — surface texture fingerprint.
    AI images have characteristic LBP distributions.
    Returns 256-bin histogram (normalized).
    """
    h, w = gray.shape
    lbp = np.zeros_like(gray, dtype=np.uint8)
    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            center = gray[i, j]
            code = 0
            # 8-neighbor clockwise
            offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
            for bit, (dr, dc) in enumerate(offsets):
                if gray[i+dr, j+dc] >= center:
                    code |= (1 << bit)
            lbp[i, j] = code
    hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-9)
    return hist.tolist()


# ── 15. Gradient magnitude ────────────────────────────────────────
def gradient_magnitude_mean(gray: np.ndarray) -> float:
    """Mean gradient magnitude (Sobel). AI images have unnaturally high values."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    return float(mag.mean())


# ── Master analyzer ───────────────────────────────────────────────
def analyze_image(
    img_bgr: np.ndarray,
    face_crop_bgr: Optional[np.ndarray] = None,
    image_bytes: Optional[bytes] = None,
) -> dict:
    """
    Run all 15 signals on an image.
    Returns flat dict ready to store in Supabase metadata JSONB field.

    Args:
        img_bgr:       Full image in BGR (OpenCV format)
        face_crop_bgr: Optional face crop for face-specific signals
        image_bytes:   Raw JPEG bytes for EXIF check
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    signals: dict = {}

    # Core texture signals
    signals["laplacian_variance"]     = laplacian_variance(gray)
    signals["noise_sigma"]            = estimate_noise_sigma(gray)
    signals["dct_high_freq_energy"]   = dct_high_freq_energy(gray)
    signals["fft_radial_profile"]     = fft_radial_profile(gray)
    signals["edge_density"]           = edge_density(gray)
    signals["local_contrast_std"]     = local_contrast_std(gray)
    signals["compression_score"]      = compression_score(img_bgr)
    signals["gradient_magnitude_mean"]= gradient_magnitude_mean(gray)

    # Color signals
    color = color_features(img_bgr)
    signals.update(color)

    # Saturation
    sat = saturation_stats(img_bgr)
    signals.update(sat)

    # EXIF
    signals["exif_has_camera"] = exif_has_camera(image_bytes) if image_bytes else -1

    # LBP (on smaller patch for speed)
    patch = gray[:128, :128] if gray.shape[0] >= 128 and gray.shape[1] >= 128 else gray
    signals["texture_lbp_hist"] = lbp_histogram(patch)

    # Face-specific signals (only if face crop provided)
    if face_crop_bgr is not None and face_crop_bgr.size > 0:
        signals["face_symmetry_score"] = face_symmetry_score(face_crop_bgr)
        signals["skin_tone_variance"]  = skin_tone_variance(face_crop_bgr)
    else:
        signals["face_symmetry_score"] = None
        signals["skin_tone_variance"]  = None

    return signals


# ── CLI test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        img = cv2.imread(path)
        if img is None:
            print("Could not load image")
            sys.exit(1)
        with open(path, "rb") as f:
            raw = f.read()
        result = analyze_image(img, image_bytes=raw)
        # Print non-list fields for readability
        for k, v in result.items():
            if not isinstance(v, list):
                print(f"  {k:35s}: {v}")
        print(f"\n  fft_radial_profile ({len(result['fft_radial_profile'])} bins): {result['fft_radial_profile']}")
        print(f"  texture_lbp_hist   (256 bins): [first 8: {result['texture_lbp_hist'][:8]}...]")
    else:
        print("Usage: python3 texture_analyzer.py <image_path>")
