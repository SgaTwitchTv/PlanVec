"""SVG export helpers."""

from __future__ import annotations

from pathlib import Path

import svgwrite

from app.geometry.models import LineSegment, Polyline


def write_svg(
    output_path: str,
    line_segments: list[LineSegment],
    curved_paths: list[Polyline],
    canvas_width: int,
    canvas_height: int,
) -> None:
    """Write detected line segments to a simple black-stroke SVG."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    normalized_width = int(canvas_width)
    normalized_height = int(canvas_height)

    drawing = svgwrite.Drawing(
        filename=str(output_file),
        size=(normalized_width, normalized_height),
        viewBox=f"0 0 {normalized_width} {normalized_height}",
    )
    drawing.attribs["fill"] = "none"

    for segment in line_segments:
        drawing.add(
            drawing.line(
                start=(int(segment.x1), int(segment.y1)),
                end=(int(segment.x2), int(segment.y2)),
                stroke="black",
                stroke_width=1,
                stroke_linecap="square",
                fill="none",
            )
        )

    for polyline in curved_paths:
        if len(polyline.points) < 2:
            continue

        drawing.add(
            drawing.polyline(
                points=list(polyline.points),
                stroke="black",
                stroke_width=1,
                stroke_linecap="round",
                stroke_linejoin="round",
                fill="none",
            )
        )

    drawing.save()
