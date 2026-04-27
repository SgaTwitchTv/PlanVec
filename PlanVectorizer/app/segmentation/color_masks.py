"""Color-based segmentation masks for architecture extraction."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ColorMaskSettings:
    """Thresholds for basic architectural color segmentation."""

    dark_gray_max: int = 165
    dark_value_max: int = 170
    dark_saturation_max: int = 70
    dark_channel_spread_max: int = 55
    white_value_min: int = 215
    white_saturation_max: int = 35
    white_max_component_area: int = 400


@dataclass(frozen=True)
class SegmentationMasks:
    """Binary masks used by the architecture-only pipeline."""

    black_mask_raw: np.ndarray
    cyan_mask: np.ndarray
    colored_outline_mask: np.ndarray
    white_callout_mask: np.ndarray
    rejected_colored_mask: np.ndarray


def build_segmentation_masks(
    image: np.ndarray,
    settings: ColorMaskSettings,
) -> SegmentationMasks:
    """Build binary masks for architecture and non-architectural color groups."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    channel_spread = image.max(axis=2) - image.min(axis=2)

    black_mask_raw = _build_black_mask(
        grayscale=grayscale,
        hsv_image=hsv_image,
        channel_spread=channel_spread,
        settings=settings,
    )
    cyan_mask = _build_cyan_mask(hsv_image)
    colored_outline_mask = _build_colored_outline_mask(hsv_image)
    white_callout_mask = _build_white_callout_mask(hsv_image, grayscale, settings)
    rejected_colored_mask = cv2.bitwise_or(cyan_mask, colored_outline_mask)
    rejected_colored_mask = cv2.bitwise_or(rejected_colored_mask, white_callout_mask)

    return SegmentationMasks(
        black_mask_raw=black_mask_raw,
        cyan_mask=cyan_mask,
        colored_outline_mask=colored_outline_mask,
        white_callout_mask=white_callout_mask,
        rejected_colored_mask=rejected_colored_mask,
    )


def _build_black_mask(
    grayscale: np.ndarray,
    hsv_image: np.ndarray,
    channel_spread: np.ndarray,
    settings: ColorMaskSettings,
) -> np.ndarray:
    dark_value_mask = hsv_image[:, :, 2] <= settings.dark_value_max
    dark_saturation_mask = hsv_image[:, :, 1] <= settings.dark_saturation_max
    dark_gray_mask = grayscale <= settings.dark_gray_max
    low_spread_mask = channel_spread <= settings.dark_channel_spread_max

    black_mask = np.where(
        dark_gray_mask & dark_value_mask & (dark_saturation_mask | low_spread_mask),
        255,
        0,
    ).astype(np.uint8)
    return cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))


def _build_cyan_mask(hsv_image: np.ndarray) -> np.ndarray:
    """Capture bright cyan equipment-like marks."""
    lower = np.array([75, 50, 120], dtype=np.uint8)
    upper = np.array([105, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv_image, lower, upper)


def _build_colored_outline_mask(hsv_image: np.ndarray) -> np.ndarray:
    """Capture bright red, green, and purple annotation outlines."""
    red_1 = cv2.inRange(hsv_image, np.array([0, 70, 80], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8))
    red_2 = cv2.inRange(hsv_image, np.array([170, 70, 80], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8))
    green = cv2.inRange(hsv_image, np.array([35, 60, 70], dtype=np.uint8), np.array([85, 255, 255], dtype=np.uint8))
    purple = cv2.inRange(hsv_image, np.array([120, 50, 70], dtype=np.uint8), np.array([165, 255, 255], dtype=np.uint8))

    combined = cv2.bitwise_or(red_1, red_2)
    combined = cv2.bitwise_or(combined, green)
    combined = cv2.bitwise_or(combined, purple)
    return combined


def _build_white_callout_mask(
    hsv_image: np.ndarray,
    grayscale: np.ndarray,
    settings: ColorMaskSettings,
) -> np.ndarray:
    """Capture small bright white callout dots and similar annotation elements."""
    white_mask = np.where(
        (hsv_image[:, :, 2] >= settings.white_value_min)
        & (hsv_image[:, :, 1] <= settings.white_saturation_max)
        & (grayscale >= settings.white_value_min),
        255,
        0,
    ).astype(np.uint8)
    return _keep_small_components(white_mask, settings.white_max_component_area)


def _keep_small_components(mask: np.ndarray, max_area: int) -> np.ndarray:
    """Keep only connected components up to the configured area."""
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    kept_mask = np.zeros_like(mask)

    for label_index in range(1, component_count):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area <= max_area:
            kept_mask[labels == label_index] = 255

    return kept_mask
