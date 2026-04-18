"""Application pipeline for raster-to-SVG vectorization."""

from __future__ import annotations

from app.detection.lines import detect_edges, detect_line_segments
from app.export.svg_writer import write_svg
from app.geometry.filtering import filter_short_segments
from app.preprocessing.cleaner import denoise_image, to_grayscale
from app.preprocessing.loader import load_image


def run_pipeline(input_path: str, output_path: str) -> None:
    """Run the deterministic MVP vectorization pipeline."""
    print("Pipeline started")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    image = load_image(input_path)
    grayscale = to_grayscale(image)
    denoised = denoise_image(grayscale)
    edges = detect_edges(denoised)
    segments = detect_line_segments(edges)
    filtered_segments = filter_short_segments(segments, min_length=20.0)

    write_svg(
        output_path=output_path,
        line_segments=filtered_segments,
        canvas_width=image.shape[1],
        canvas_height=image.shape[0],
    )

    print(f"Detected {len(segments)} line segments")
    print(f"Exported {len(filtered_segments)} filtered line segments")
    print("Pipeline finished")
