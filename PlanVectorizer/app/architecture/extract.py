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
    min_curve_area: int = 25
    min_curve_perimeter: float = 18.0
    max_curve_area: int = 5000
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
    contours, _ = cv2.findContours(black_mask_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    curved_paths = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, closed=True))
        if area < settings.min_curve_area or area > settings.max_curve_area:
            continue
        if perimeter < settings.min_curve_perimeter:
            continue

        epsilon = max(1.5, perimeter * settings.contour_approximation_ratio)
        approximated = cv2.approxPolyDP(contour, epsilon, closed=False)
        points = tuple((int(point[0][0]), int(point[0][1])) for point in approximated)
        if len(points) < 4:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        aspect_ratio = max(width, height) / max(1.0, float(min(width, height)))
        fill_ratio = area / max(1.0, float(width * height))

        if 0.15 <= fill_ratio <= 0.7 and aspect_ratio <= 4.5:
            curved_paths.append(Polyline(points=points))

    return _deduplicate_polylines(curved_paths)


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
