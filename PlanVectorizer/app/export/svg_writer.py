"""SVG export helpers."""

from __future__ import annotations

from pathlib import Path

import svgwrite

from app.geometry.models import LineSegment


def write_svg(
    output_path: str,
    line_segments: list[LineSegment],
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

    drawing.save()
