from __future__ import annotations

import numpy as np
import scipy.signal
from PySide6.QtGui import QImage, QPixmap

# ── Colormap ───────────────────────────────────────────────────────────────────
# Custom dark-DAW palette: black → deep blue → teal (accent) → amber → white

_CMAP_POINTS = np.array([
    [0.00, ( 6,  6, 18)],
    [0.20, (10, 28, 80)],
    [0.42, ( 0,140,140)],   # teal accent
    [0.65, (80,160, 20)],
    [0.83, (210,150,  0)],
    [1.00, (255,255,255)],
], dtype=object)

_LUT: np.ndarray | None = None  # 256×3 uint8 lookup table


def _build_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    positions = _CMAP_POINTS[:, 0].astype(float)
    colors = np.array(list(_CMAP_POINTS[:, 1]), dtype=float)
    for i in range(256):
        t = i / 255.0
        idx = np.searchsorted(positions, t, side="right") - 1
        idx = int(np.clip(idx, 0, len(positions) - 2))
        t0, t1 = positions[idx], positions[idx + 1]
        alpha = (t - t0) / (t1 - t0 + 1e-12)
        rgb = colors[idx] * (1 - alpha) + colors[idx + 1] * alpha
        lut[i] = np.clip(rgb, 0, 255).astype(np.uint8)
    return lut


def _get_lut() -> np.ndarray:
    global _LUT
    if _LUT is None:
        _LUT = _build_lut()
    return _LUT


# ── Core computation ───────────────────────────────────────────────────────────

def compute_spectrogram_pixmap(
    audio: np.ndarray,
    sample_rate: int,
    width: int,
    height: int,
    db_min: float = -80.0,
    db_max: float = 0.0,
) -> QPixmap:
    """Compute STFT spectrogram and return as a QPixmap of given size."""
    mono = audio.mean(axis=1).astype(np.float32) if audio.ndim == 2 else audio.astype(np.float32)

    nperseg = min(512, len(mono))
    noverlap = nperseg * 3 // 4

    _, _, sxx = scipy.signal.spectrogram(
        mono,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
        scaling="spectrum",
    )

    sxx_db = 10.0 * np.log10(sxx + 1e-12)

    # Normalize to 0–255
    norm = np.clip((sxx_db - db_min) / (db_max - db_min), 0.0, 1.0)

    # Flip frequency axis so low frequencies are at the bottom
    norm = norm[::-1]

    # Resize to (height, width) via nearest-neighbour sampling
    resized = _resize_nn(norm, height, width)

    # Apply colormap LUT → (H, W, 3) uint8
    indices = (resized * 255).astype(np.uint8)
    lut = _get_lut()
    rgb = lut[indices]  # (H, W, 3)

    # Build QPixmap from raw RGB bytes
    h, w, _ = rgb.shape
    img = QImage(
        rgb.tobytes(),
        w, h,
        w * 3,
        QImage.Format.Format_RGB888,
    )
    return QPixmap.fromImage(img)


def _resize_nn(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbour resize of a 2-D float array."""
    in_h, in_w = arr.shape
    y_idx = (np.arange(out_h) * in_h / out_h).astype(np.int32)
    x_idx = (np.arange(out_w) * in_w / out_w).astype(np.int32)
    y_idx = np.clip(y_idx, 0, in_h - 1)
    x_idx = np.clip(x_idx, 0, in_w - 1)
    return arr[np.ix_(y_idx, x_idx)]
