"""Filtering helpers for detected geometry."""

from __future__ import annotations

from app.geometry.models import LineSegment


def filter_short_segments(
    line_segments: list[LineSegment],
    min_length: float,
) -> list[LineSegment]:
    """Remove segments shorter than the configured minimum length."""
    return [segment for segment in line_segments if segment.length >= min_length]
