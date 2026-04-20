"""Application pipeline for raster-to-SVG vectorization."""

from __future__ import annotations

from dataclasses import dataclass

from app.detection.lines import detect_edges, detect_line_segments
from app.export.svg_writer import write_svg
from app.geometry.filtering import deduplicate_segments, filter_short_segments
from app.preprocessing.cleaner import denoise_image, to_grayscale
from app.preprocessing.loader import load_image


@dataclass(frozen=True)
class PipelineSettings:
    """Tunable settings for the current deterministic pipeline step."""

    min_line_length: float = 20.0


def run_pipeline(input_path: str, output_path: str) -> None:
    """Run the deterministic MVP vectorization pipeline."""
    settings = PipelineSettings()

    print("Pipeline started")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    image = load_image(input_path)
    grayscale = to_grayscale(image)
    denoised = denoise_image(grayscale)
    edges = detect_edges(denoised)
    segments = detect_line_segments(edges)
    filtered_segments = filter_short_segments(
        segments,
        min_length=settings.min_line_length,
    )
    unique_segments = deduplicate_segments(filtered_segments)

    write_svg(
        output_path=output_path,
        line_segments=unique_segments,
        canvas_width=image.shape[1],
        canvas_height=image.shape[0],
    )

    print(f"Detected {len(segments)} line segments")
    print(f"Retained {len(filtered_segments)} filtered line segments")
    print(f"Exported {len(unique_segments)} unique line segments")
    print("Pipeline finished")
