"""Line and edge detection routines."""

from __future__ import annotations

import cv2
import numpy as np

from app.geometry.models import LineSegment


def detect_edges(grayscale_image: np.ndarray) -> np.ndarray:
    """Generate an edge map with Canny edge detection."""
    return cv2.Canny(grayscale_image, threshold1=50, threshold2=150, apertureSize=3)


def detect_line_segments(edge_image: np.ndarray) -> list[LineSegment]:
    """Detect line segments from an edge image using probabilistic Hough."""
    detected = cv2.HoughLinesP(
        edge_image,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=25,
        maxLineGap=10,
    )

    if detected is None:
        return []

    segments: list[LineSegment] = []
    for raw_line in detected:
        x1, y1, x2, y2 = raw_line[0]
        segments.append(LineSegment(x1=x1, y1=y1, x2=x2, y2=y2))

    return segments
