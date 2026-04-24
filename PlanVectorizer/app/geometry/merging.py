"""Helpers for normalizing and merging orthogonal line segments."""

from __future__ import annotations

from dataclasses import dataclass

from app.geometry.models import LineSegment


@dataclass(frozen=True)
class _ProjectedSegment:
    """Internal orthogonal segment representation used for merging."""

    orientation: str
    fixed: int
    start: int
    end: int


def normalize_orthogonal_segments(
    line_segments: list[LineSegment],
    max_deviation: int = 4,
) -> tuple[list[LineSegment], list[LineSegment]]:
    """Snap nearly horizontal/vertical segments to exact orthogonal lines."""
    orthogonal_segments = []
    remaining_segments = []

    for segment in line_segments:
        dx = segment.x2 - segment.x1
        dy = segment.y2 - segment.y1

        if abs(dy) <= max_deviation and abs(dx) >= abs(dy):
            snapped_y = int(round((segment.y1 + segment.y2) / 2.0))
            start_x = min(segment.x1, segment.x2)
            end_x = max(segment.x1, segment.x2)
            orthogonal_segments.append(
                LineSegment(start_x, snapped_y, end_x, snapped_y)
            )
            continue

        if abs(dx) <= max_deviation and abs(dy) >= abs(dx):
            snapped_x = int(round((segment.x1 + segment.x2) / 2.0))
            start_y = min(segment.y1, segment.y2)
            end_y = max(segment.y1, segment.y2)
            orthogonal_segments.append(
                LineSegment(snapped_x, start_y, snapped_x, end_y)
            )
            continue

        remaining_segments.append(segment)

    return orthogonal_segments, remaining_segments


def merge_collinear_segments(
    line_segments: list[LineSegment],
    axis_tolerance: int = 4,
    gap_tolerance: int = 12,
    orthogonal_deviation: int = 4,
) -> list[LineSegment]:
    """Merge nearby collinear horizontal and vertical segments."""
    orthogonal_segments, remaining_segments = normalize_orthogonal_segments(
        line_segments,
        max_deviation=orthogonal_deviation,
    )

    horizontal_segments = []
    vertical_segments = []

    for segment in orthogonal_segments:
        if segment.y1 == segment.y2:
            horizontal_segments.append(
                _ProjectedSegment(
                    orientation="horizontal",
                    fixed=segment.y1,
                    start=min(segment.x1, segment.x2),
                    end=max(segment.x1, segment.x2),
                )
            )
        elif segment.x1 == segment.x2:
            vertical_segments.append(
                _ProjectedSegment(
                    orientation="vertical",
                    fixed=segment.x1,
                    start=min(segment.y1, segment.y2),
                    end=max(segment.y1, segment.y2),
                )
            )

    merged_segments = []
    merged_segments.extend(
        _merge_projected_segments(horizontal_segments, axis_tolerance, gap_tolerance)
    )
    merged_segments.extend(
        _merge_projected_segments(vertical_segments, axis_tolerance, gap_tolerance)
    )
    merged_segments.extend(remaining_segments)
    return merged_segments


def _merge_projected_segments(
    projected_segments: list[_ProjectedSegment],
    axis_tolerance: int,
    gap_tolerance: int,
) -> list[LineSegment]:
    """Merge projected segments that lie on nearly the same axis."""
    if not projected_segments:
        return []

    sorted_segments = sorted(
        projected_segments,
        key=lambda segment: (segment.fixed, segment.start, segment.end),
    )

    merged = []
    current_group = [sorted_segments[0]]

    for segment in sorted_segments[1:]:
        previous = current_group[-1]
        same_axis = abs(segment.fixed - previous.fixed) <= axis_tolerance
        overlapping = segment.start <= _group_end(current_group) + gap_tolerance

        if same_axis and overlapping:
            current_group.append(segment)
            continue

        merged.append(_build_line_segment(current_group))
        current_group = [segment]

    merged.append(_build_line_segment(current_group))
    return merged


def _group_end(projected_segments: list[_ProjectedSegment]) -> int:
    """Return the furthest extent of a projected segment group."""
    return max(segment.end for segment in projected_segments)


def _build_line_segment(projected_segments: list[_ProjectedSegment]) -> LineSegment:
    """Convert a merged projected segment group back to a line segment."""
    first = projected_segments[0]
    fixed = int(round(sum(segment.fixed for segment in projected_segments) / len(projected_segments)))
    start = min(segment.start for segment in projected_segments)
    end = max(segment.end for segment in projected_segments)

    if first.orientation == "horizontal":
        return LineSegment(start, fixed, end, fixed)

    return LineSegment(fixed, start, fixed, end)
