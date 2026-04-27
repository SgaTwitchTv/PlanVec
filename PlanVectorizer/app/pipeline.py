"""Application pipeline for raster-to-SVG vectorization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.architecture.extract import (
    ArchitectureExtractionSettings,
    extract_architecture_geometry,
)
from app.detection.lines import detect_edges, detect_line_segments
from app.export.svg_writer import write_svg
from app.geometry.filtering import deduplicate_segments, filter_short_segments
from app.preprocessing.cleaner import (
    BlackMaskCleanupSettings,
    clean_black_mask,
)
from app.preprocessing.loader import load_image
from app.segmentation.color_masks import ColorMaskSettings, build_segmentation_masks


@dataclass(frozen=True)
class PipelineSettings:
    """Tunable settings for the current deterministic pipeline step."""

    segmentation: ColorMaskSettings = ColorMaskSettings()
    black_cleanup: BlackMaskCleanupSettings = BlackMaskCleanupSettings()
    architecture: ArchitectureExtractionSettings = ArchitectureExtractionSettings()


def run_pipeline(input_path: str, output_path: str) -> None:
    """Run the architecture-only vectorization pipeline."""
    settings = PipelineSettings()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    debug_dir = output_file.parent / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("Pipeline started")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    image = load_image(input_path)
    segmentation_masks = build_segmentation_masks(
        image,
        settings.segmentation,
    )
    black_mask_clean = clean_black_mask(
        segmentation_masks.black_mask_raw,
        segmentation_masks.rejected_colored_mask,
        settings.black_cleanup,
    )
    architecture_geometry = extract_architecture_geometry(
        black_mask_clean,
        settings.architecture,
    )

    write_svg(
        output_path=output_path,
        line_segments=architecture_geometry.line_segments,
        curved_paths=architecture_geometry.curved_paths,
        canvas_width=image.shape[1],
        canvas_height=image.shape[0],
    )
    _save_debug_outputs(
        image=image,
        black_mask_raw=segmentation_masks.black_mask_raw,
        black_mask_clean=black_mask_clean,
        rejected_colored_mask=segmentation_masks.rejected_colored_mask,
        architecture_edges=architecture_geometry.edges,
        architecture_preview=architecture_geometry.preview,
        debug_dir=debug_dir,
    )

    print(f"Exported {len(architecture_geometry.line_segments)} architecture line segments")
    print(f"Exported {len(architecture_geometry.curved_paths)} curved architecture paths")
    print(f"Debug output: {debug_dir}")
    print("Pipeline finished")


def _save_debug_outputs(
    image: np.ndarray,
    black_mask_raw: np.ndarray,
    black_mask_clean: np.ndarray,
    rejected_colored_mask: np.ndarray,
    architecture_edges: np.ndarray,
    architecture_preview: np.ndarray,
    debug_dir: Path,
) -> None:
    """Write the requested debug images to the output/debug directory."""
    cv2.imwrite(str(debug_dir / "original_image_copy.png"), image)
    cv2.imwrite(str(debug_dir / "black_mask_raw.png"), black_mask_raw)
    cv2.imwrite(str(debug_dir / "black_mask_clean.png"), black_mask_clean)
    cv2.imwrite(str(debug_dir / "rejected_colored_mask.png"), rejected_colored_mask)
    cv2.imwrite(str(debug_dir / "architecture_edges.png"), architecture_edges)
    cv2.imwrite(str(debug_dir / "architecture_preview.png"), architecture_preview)
