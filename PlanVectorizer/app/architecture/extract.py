"""Architecture-only geometry extraction from a cleaned black mask."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.detection.lines import detect_edges, detect_line_segments
from app.geometry.filtering import deduplicate_segments, filter_short_segments
from app.geometry.models import LineSegment, Polyline


@dataclass(frozen=True)
class ArchitectureExtractionSettings:
    """Config for architecture-only geometry extraction."""

    min_line_length: float = 18.0
    min_curve_component_pixels: int = 10
    min_curve_perimeter: float = 18.0
    max_curve_component_pixels: int = 5000
    min_curve_bbox_dimension: int = 8
    max_curve_aspect_ratio: float = 5.0
    max_curve_fill_ratio: float = 0.58
    contour_approximation_ratio: float = 0.015


@dataclass(frozen=True)
class ArchitectureGeometry:
    """Extracted architecture geometry and debug images."""

    line_segments: list[LineSegment]
    curved_paths: list[Polyline]
    edges: np.ndarray
    preview: np.ndarray


def extract_architecture_geometry(
    black_mask_clean: np.ndarray,
    settings: ArchitectureExtractionSettings,
) -> ArchitectureGeometry:
    """Extract architecture-only lines and curve-like door paths from a binary mask."""
    edges = detect_edges(black_mask_clean)
    line_segments = detect_line_segments(edges)
    line_segments = filter_short_segments(line_segments, settings.min_line_length)
    line_segments = deduplicate_segments(line_segments)

    curved_paths = _extract_curved_paths(black_mask_clean, settings)
    preview = _build_preview(black_mask_clean.shape, line_segments, curved_paths)

    return ArchitectureGeometry(
        line_segments=line_segments,
        curved_paths=curved_paths,
        edges=edges,
        preview=preview,
    )


def _extract_curved_paths(
    black_mask_clean: np.ndarray,
    settings: ArchitectureExtractionSettings,
) -> list[Polyline]:
    """Keep contour-like curved paths that may correspond to door swings."""
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        black_mask_clean,
        connectivity=8,
    )
    curved_paths = []

    for label_index in range(1, component_count):
        component_pixels = int(stats[label_index, cv2.CC_STAT_AREA])
        if component_pixels < settings.min_curve_component_pixels:
            continue
        if component_pixels > settings.max_curve_component_pixels:
            continue

        x = int(stats[label_index, cv2.CC_STAT_LEFT])
        y = int(stats[label_index, cv2.CC_STAT_TOP])
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        max_dimension = max(width, height)
        min_dimension = max(1, min(width, height))
        if max_dimension < settings.min_curve_bbox_dimension:
            continue

        aspect_ratio = max_dimension / float(min_dimension)
        if aspect_ratio > settings.max_curve_aspect_ratio:
            continue

        fill_ratio = component_pixels / max(1.0, float(width * height))
        if fill_ratio > settings.max_curve_fill_ratio:
            continue

        component_mask = np.where(labels == label_index, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        contour = max(contours, key=lambda curve: cv2.arcLength(curve, closed=True))
        perimeter = float(cv2.arcLength(contour, closed=True))
        if perimeter < settings.min_curve_perimeter:
            continue

        epsilon = max(1.5, perimeter * settings.contour_approximation_ratio)
        approximated = cv2.approxPolyDP(contour, epsilon, closed=False)
        points = tuple((int(point[0][0]), int(point[0][1])) for point in approximated)
        if len(points) < 4:
            continue

        if not _looks_curve_like(points, aspect_ratio, fill_ratio):
            continue

        curved_paths.append(Polyline(points=points))

    return _deduplicate_polylines(curved_paths)


def _looks_curve_like(
    points: tuple[tuple[int, int], ...],
    aspect_ratio: float,
    fill_ratio: float,
) -> bool:
    """Reject boxy contour loops while keeping arc-like and mixed door shapes."""
    diagonal_segments = 0
    for left_point, right_point in zip(points, points[1:]):
        dx = right_point[0] - left_point[0]
        dy = right_point[1] - left_point[1]
        if dx == 0 and dy == 0:
            continue

        if abs(dx) > 1 and abs(dy) > 1:
            diagonal_segments += 1

    if diagonal_segments >= 1:
        return True

    if len(points) >= 6 and aspect_ratio <= 3.6 and fill_ratio <= 0.42:
        return True

    return False


def _deduplicate_polylines(polylines: list[Polyline]) -> list[Polyline]:
    """Remove exact duplicate and reversed duplicate polylines."""
    unique_polylines = []
    seen = set()

    for polyline in polylines:
        key = polyline.points
        reverse_key = tuple(reversed(polyline.points))
        if key in seen or reverse_key in seen:
            continue

        seen.add(key)
        unique_polylines.append(polyline)

    return unique_polylines


def _build_preview(
    image_shape: tuple[int, int],
    line_segments: list[LineSegment],
    curved_paths: list[Polyline],
) -> np.ndarray:
    """Render a simple white-background preview of the extracted architecture."""
    height, width = image_shape[:2]
    preview = np.full((height, width, 3), 255, dtype=np.uint8)

    for segment in line_segments:
        cv2.line(
            preview,
            (segment.x1, segment.y1),
            (segment.x2, segment.y2),
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )

    for polyline in curved_paths:
        if len(polyline.points) < 2:
            continue

        contour = np.array(polyline.points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(preview, [contour], isClosed=False, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    return preview
