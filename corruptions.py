"""Controlled image corruptions for studying detector robustness.

Each corruption takes a BGR uint8 image and a non-negative ``severity`` and
returns a new BGR uint8 image. ``severity == 0`` is always the identity
(returns an unchanged copy), so a severity sweep starting at 0 gives a clean
baseline for free.

The three corruptions model the three effects we care about:

* ``motion_blur``    -> motion (a moving camera/object smears the image along a
                        direction during the exposure).
* ``gaussian_blur``  -> general blur / loss of focus.
* ``brightness``     -> lighting changes (under- and over-exposure).
"""

from __future__ import annotations

import cv2
import numpy as np


def motion_blur(img: np.ndarray, severity: int, angle: float = 0.0) -> np.ndarray:
    """Linear motion blur.

    ``severity`` is the blur length in pixels (the size of the averaging
    streak). ``angle`` is the motion direction in degrees (0 = horizontal).
    """
    if severity <= 1:
        return img.copy()
    k = int(severity)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0  # horizontal streak
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return cv2.filter2D(img, -1, kernel)


def gaussian_blur(img: np.ndarray, severity: float) -> np.ndarray:
    """Isotropic Gaussian (out-of-focus) blur. ``severity`` is sigma in pixels."""
    if severity <= 0:
        return img.copy()
    return cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=float(severity))


def brightness(img: np.ndarray, severity: float) -> np.ndarray:
    """Multiplicative brightness change.

    ``severity`` is a gain factor: 1.0 is unchanged, <1 darkens, >1 brightens.
    Values are clipped to [0, 255], so large gains realistically blow out
    highlights and small gains crush shadows.
    """
    if severity == 1.0:
        return img.copy()
    out = img.astype(np.float32) * float(severity)
    return np.clip(out, 0, 255).astype(np.uint8)


# Severity sweeps used by the study. The first entry of each is the clean
# baseline (identity) so trends are measured against an unperturbed reference.
SWEEPS = {
    "motion_blur": {
        "fn": motion_blur,
        "severities": [1, 5, 9, 13, 17, 21, 27],  # blur length in px
        "xlabel": "Motion blur length (px)",
        "clean": 1,
    },
    "gaussian_blur": {
        "fn": gaussian_blur,
        "severities": [0, 1, 2, 3, 5, 7, 9],  # gaussian sigma in px
        "xlabel": "Gaussian blur sigma (px)",
        "clean": 0,
    },
    "brightness": {
        "fn": brightness,
        "severities": [0.2, 0.35, 0.5, 1.0, 1.6, 2.2, 3.0],  # gain factor (ascending)
        "xlabel": "Brightness gain (x)",
        "clean": 1.0,
    },
}
