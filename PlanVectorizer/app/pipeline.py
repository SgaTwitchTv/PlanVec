"""Application pipeline for raster-to-SVG vectorization."""

from __future__ import annotations

import csv
import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.architecture.annotation_classifier import AnnotationMLPClassifier
from app.architecture.components import (
    ComponentFilterSettings,
    filter_structure_components,
)
from app.architecture.extract import (
    ArchitectureExtractionSettings,
    extract_architecture_geometry,
)
from app.export.svg_writer import write_svg
from app.preprocessing.cleaner import (
    BlackMaskCleanupSettings,
    clean_black_mask,
)
from app.preprocessing.loader import load_image
from app.segmentation.color_masks import ColorMaskSettings, build_segmentation_masks


@dataclass(frozen=True)
class AnnotationClassifierRuntimeSettings:
    """Runtime options for the optional annotation crop classifier."""

    enabled: bool = True
    model_relative_path: str = "models/annotation_classifier.npz"
    export_candidate_crops: bool = True


@dataclass(frozen=True)
class PipelineSettings:
    """Tunable settings for the current deterministic pipeline step."""

    segmentation: ColorMaskSettings = field(default_factory=ColorMaskSettings)
    black_cleanup: BlackMaskCleanupSettings = field(default_factory=BlackMaskCleanupSettings)
    component_filter: ComponentFilterSettings = field(default_factory=ComponentFilterSettings)
    annotation_classifier: AnnotationClassifierRuntimeSettings = field(
        default_factory=AnnotationClassifierRuntimeSettings,
    )
    architecture: ArchitectureExtractionSettings = field(default_factory=ArchitectureExtractionSettings)


def run_pipeline(input_path: str, output_path: str) -> None:
    """Run the architecture-only vectorization pipeline."""
    settings = PipelineSettings()
    input_file = Path(input_path)
    output_file = Path(output_path)
    project_root = Path(__file__).resolve().parent.parent
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
    annotation_classifier = _load_annotation_classifier(project_root, settings)
    black_mask_clean = clean_black_mask(
        segmentation_masks.black_mask_raw,
        segmentation_masks.rejected_colored_mask,
        settings.black_cleanup,
    )
    component_filter_result = filter_structure_components(
        black_mask_clean,
        settings.component_filter,
        classifier=annotation_classifier,
    )
    architecture_geometry = extract_architecture_geometry(
        component_filter_result.structure_mask,
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
        structure_mask=component_filter_result.structure_mask,
        rejected_symbol_mask=component_filter_result.rejected_symbol_mask,
        candidate_crops=component_filter_result.candidate_crops,
        architecture_edges=architecture_geometry.edges,
        architecture_preview=architecture_geometry.preview,
        debug_dir=debug_dir,
        export_candidate_crops=settings.annotation_classifier.export_candidate_crops,
        source_input_path=input_file,
    )

    print(f"Exported {len(architecture_geometry.line_segments)} architecture line segments")
    print(f"Exported {len(architecture_geometry.curved_paths)} curved architecture paths")
    print(f"Prepared {len(component_filter_result.candidate_crops)} candidate crops for AI labeling")
    print(f"Debug output: {debug_dir}")
    print("Pipeline finished")


def _load_annotation_classifier(
    project_root: Path,
    settings: PipelineSettings,
) -> Optional[AnnotationMLPClassifier]:
    """Load a trained annotation classifier when a model file is available."""
    if not settings.annotation_classifier.enabled:
        print("Annotation classifier: disabled")
        return None

    model_path = project_root / settings.annotation_classifier.model_relative_path
    if not model_path.exists():
        print(f"Annotation classifier: model not found at {model_path}, using heuristics only")
        return None

    classifier = AnnotationMLPClassifier.load(str(model_path))
    print(f"Annotation classifier: loaded {model_path}")
    return classifier


def _save_debug_outputs(
    image: np.ndarray,
    black_mask_raw: np.ndarray,
    black_mask_clean: np.ndarray,
    rejected_colored_mask: np.ndarray,
    structure_mask: np.ndarray,
    rejected_symbol_mask: np.ndarray,
    candidate_crops: tuple,
    architecture_edges: np.ndarray,
    architecture_preview: np.ndarray,
    debug_dir: Path,
    export_candidate_crops: bool,
    source_input_path: Path,
) -> None:
    """Write the requested debug images to the output/debug directory."""
    cv2.imwrite(str(debug_dir / "original_image_copy.png"), image)
    cv2.imwrite(str(debug_dir / "black_mask_raw.png"), black_mask_raw)
    cv2.imwrite(str(debug_dir / "black_mask_clean.png"), black_mask_clean)
    cv2.imwrite(str(debug_dir / "rejected_colored_mask.png"), rejected_colored_mask)
    cv2.imwrite(str(debug_dir / "structure_mask.png"), structure_mask)
    cv2.imwrite(str(debug_dir / "rejected_symbol_mask.png"), rejected_symbol_mask)
    cv2.imwrite(str(debug_dir / "architecture_edges.png"), architecture_edges)
    cv2.imwrite(str(debug_dir / "architecture_preview.png"), architecture_preview)
    if export_candidate_crops:
        _save_candidate_crops(
            candidate_crops,
            debug_dir / "annotation_candidates",
            source_input_path,
        )


def _save_candidate_crops(
    candidate_crops: tuple,
    candidate_root_dir: Path,
    source_input_path: Path,
) -> None:
    """Write ambiguous cluster crops and a manifest for future classifier labeling."""
    candidate_root_dir.mkdir(parents=True, exist_ok=True)
    _remove_legacy_candidate_root_files(candidate_root_dir)
    source_id = _build_source_id(source_input_path)
    candidate_dir = candidate_root_dir / source_id
    _reset_directory(candidate_dir)
    manifest_path = candidate_dir / "manifest.csv"

    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(
            [
                "source_id",
                "cluster_id",
                "x",
                "y",
                "width",
                "height",
                "member_count",
                "predicted_label",
                "confidence",
                "image_name",
            ]
        )

        for candidate in candidate_crops:
            safe_label = _sanitize_filename_part(candidate.predicted_label)
            image_name = (
                f"{source_id}_cluster_{candidate.cluster_id:04d}_"
                f"{safe_label}_{candidate.confidence:.2f}.png"
            )
            cv2.imwrite(str(candidate_dir / image_name), candidate.image)
            writer.writerow(
                [
                    source_id,
                    candidate.cluster_id,
                    candidate.x,
                    candidate.y,
                    candidate.width,
                    candidate.height,
                    candidate.member_count,
                    candidate.predicted_label,
                    f"{candidate.confidence:.4f}",
                    image_name,
                ]
            )


def _build_source_id(source_input_path: Path) -> str:
    """Build a deterministic, readable identifier for one source image file."""
    source_hash = hashlib.sha256(source_input_path.read_bytes()).hexdigest()[:10]
    source_stem = _sanitize_filename_part(source_input_path.stem)
    return f"{source_stem}_{source_hash}"


def _reset_directory(target_dir: Path) -> None:
    """Recreate one candidate batch directory so reruns overwrite stale files cleanly."""
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)


def _remove_legacy_candidate_root_files(candidate_root_dir: Path) -> None:
    """Delete old flat candidate files left from the pre-hash export layout."""
    for legacy_file in candidate_root_dir.iterdir():
        if not legacy_file.is_file():
            continue
        if legacy_file.suffix.lower() == ".png" or legacy_file.name.lower() == "manifest.csv":
            legacy_file.unlink()


def _sanitize_filename_part(value: str) -> str:
    """Convert a prediction label into a simple filename-safe token."""
    cleaned = [
        character if character.isalnum() else "_"
        for character in value.strip().lower()
    ]
    safe_value = "".join(cleaned).strip("_")
    return safe_value or "unclassified"
