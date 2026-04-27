"""Image preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass

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


def reinforce_outer_wall_continuity(
    structural_mask: np.ndarray,
    band_depth: int = 48,
    gap_span: int = 31,
) -> np.ndarray:
    """Close small breaks only near the outer drawing boundary."""
    bounds = _find_mask_bounds(structural_mask)
    if bounds is None:
        return structural_mask

    min_x, min_y, max_x, max_y = bounds
    healed_mask = structural_mask.copy()

    horizontal_kernel = np.ones((1, max(3, int(gap_span))), dtype=np.uint8)
    vertical_kernel = np.ones((max(3, int(gap_span)), 1), dtype=np.uint8)

    top_y1 = min(structural_mask.shape[0], min_y + band_depth)
    bottom_y0 = max(0, max_y - band_depth + 1)
    left_x1 = min(structural_mask.shape[1], min_x + band_depth)
    right_x0 = max(0, max_x - band_depth + 1)

    _close_band_in_place(healed_mask, 0, top_y1, 0, structural_mask.shape[1], horizontal_kernel)
    _close_band_in_place(
        healed_mask,
        bottom_y0,
        structural_mask.shape[0],
        0,
        structural_mask.shape[1],
        horizontal_kernel,
    )
    _close_band_in_place(healed_mask, 0, structural_mask.shape[0], 0, left_x1, vertical_kernel)
    _close_band_in_place(
        healed_mask,
        0,
        structural_mask.shape[0],
        right_x0,
        structural_mask.shape[1],
        vertical_kernel,
    )

    return healed_mask


def find_structural_bounds(
    structural_mask: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """Return the bounding box of non-zero pixels in the structural mask."""
    return _find_mask_bounds(structural_mask)


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


@dataclass(frozen=True)
class BlackMaskCleanupSettings:
    """Config for architecture-focused black mask cleanup."""

    min_component_area: int = 18
    leader_line_min_length: int = 18
    leader_line_max_thickness: int = 3
    closing_kernel_size: int = 3


def clean_black_mask(
    black_mask_raw: np.ndarray,
    rejected_mask: np.ndarray,
    settings: BlackMaskCleanupSettings,
) -> np.ndarray:
    """Remove non-architectural noise while preserving walls and door arcs."""
    cleaned_mask = black_mask_raw.copy()
    cleaned_mask[rejected_mask > 0] = 0

    kernel_size = max(3, int(settings.closing_kernel_size))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = _remove_small_components(cleaned_mask, settings.min_component_area)
    cleaned_mask = _remove_thin_diagonal_components(cleaned_mask, settings)
    return cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)


def _find_mask_bounds(structural_mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the bounding box of non-zero pixels in the mask."""
    points = cv2.findNonZero(structural_mask)
    if points is None:
        return None

    x, y, width, height = cv2.boundingRect(points)
    return x, y, x + width - 1, y + height - 1


def _close_band_in_place(
    structural_mask: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    kernel: np.ndarray,
) -> None:
    """Apply directional closing inside a selected band of the mask."""
    if y0 >= y1 or x0 >= x1:
        return

    region = structural_mask[y0:y1, x0:x1]
    closed_region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, kernel)
    structural_mask[y0:y1, x0:x1] = np.maximum(region, closed_region)


def _remove_small_components(mask: np.ndarray, min_component_area: int) -> np.ndarray:
    """Remove tiny connected components from a binary mask."""
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered_mask = np.zeros_like(mask)

    for label_index in range(1, component_count):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area >= min_component_area:
            filtered_mask[labels == label_index] = 255

    return filtered_mask


def _remove_thin_diagonal_components(
    mask: np.ndarray,
    settings: BlackMaskCleanupSettings,
) -> np.ndarray:
    """Suppress likely thin diagonal leader lines without touching walls."""
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered_mask = np.zeros_like(mask)

    for label_index in range(1, component_count):
        x = int(stats[label_index, cv2.CC_STAT_LEFT])
        y = int(stats[label_index, cv2.CC_STAT_TOP])
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_index, cv2.CC_STAT_AREA])

        long_enough = max(width, height) >= settings.leader_line_min_length
        thin_enough = min(width, height) <= settings.leader_line_max_thickness
        diagonalish = width > settings.leader_line_max_thickness and height > settings.leader_line_max_thickness
        sparse = area <= max(width, height) * max(2, settings.leader_line_max_thickness)

        if long_enough and thin_enough and diagonalish and sparse:
            continue

        filtered_mask[labels == label_index] = 255

    return filtered_mask
