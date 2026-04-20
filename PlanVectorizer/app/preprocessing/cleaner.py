"""Image preprocessing helpers."""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise_image(grayscale_image: np.ndarray) -> np.ndarray:
    """Apply light denoising before edge detection."""
    return cv2.GaussianBlur(grayscale_image, (3, 3), 0)
