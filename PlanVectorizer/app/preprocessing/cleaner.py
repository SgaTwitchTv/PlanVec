"""Image preprocessing helpers."""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def extract_structural_mask(
    image: np.ndarray,
    dark_threshold: int = 160,
    neutral_threshold: int = 40,
) -> np.ndarray:
    """Keep mostly dark neutral pixels that resemble structural drafting lines."""
    grayscale = to_grayscale(image)
    channel_min = image.min(axis=2)
    channel_max = image.max(axis=2)
    channel_spread = channel_max - channel_min

    dark_pixels = grayscale <= dark_threshold
    neutral_pixels = channel_spread <= neutral_threshold

    structural_mask = np.where(
        dark_pixels & neutral_pixels,
        255,
        0,
    ).astype(np.uint8)

    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(structural_mask, cv2.MORPH_CLOSE, kernel)


def apply_structural_mask(
    grayscale_image: np.ndarray,
    structural_mask: np.ndarray,
) -> np.ndarray:
    """Whiten non-structural regions while preserving structural grayscale pixels."""
    masked_image = np.full_like(grayscale_image, 255)
    masked_image[structural_mask > 0] = grayscale_image[structural_mask > 0]
    return masked_image


def denoise_image(grayscale_image: np.ndarray) -> np.ndarray:
    """Apply light denoising before edge detection."""
    return cv2.GaussianBlur(grayscale_image, (3, 3), 0)
