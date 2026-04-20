"""Filtering helpers for detected geometry."""

from __future__ import annotations

from app.geometry.models import LineSegment


def filter_short_segments(
    line_segments: list[LineSegment],
    min_length: float,
) -> list[LineSegment]:
    """Remove segments shorter than the configured minimum length."""
    return [segment for segment in line_segments if segment.length >= min_length]


def deduplicate_segments(line_segments: list[LineSegment]) -> list[LineSegment]:
    """Remove exact duplicate segments, including reversed duplicates."""
    unique_segments = []
    seen_keys = set()

    for segment in line_segments:
        key = segment.canonical_key()
        if key in seen_keys:
            continue

        seen_keys.add(key)
        unique_segments.append(segment)

    return unique_segments
