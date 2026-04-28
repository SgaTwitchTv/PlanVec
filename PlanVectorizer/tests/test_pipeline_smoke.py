"""Small smoke test for the raster-to-SVG pipeline."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app.architecture.annotation_classifier import AnnotationPrediction, normalize_candidate_crop
from app.architecture.components import ComponentFilterSettings, filter_structure_components
from app.pipeline import run_pipeline
from app.preprocessing.cleaner import BlackMaskCleanupSettings, clean_black_mask
from app.segmentation.color_masks import ColorMaskSettings, build_segmentation_masks


class _FakeStructureClassifier:
    """Minimal classifier stub for component-filter tests."""

    def predict_crop(self, crop_mask: np.ndarray) -> AnnotationPrediction:
        _ = crop_mask
        return AnnotationPrediction(
            label="structure_square_pillar",
            confidence=0.99,
            probabilities={"structure_square_pillar": 0.99},
        )


class PipelineSmokeTest(unittest.TestCase):
    """Verify that the architecture-only pipeline produces clean outputs."""

    def test_pipeline_creates_svg_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "source.png"
            output_path = temp_path / "nested" / "result.svg"

            image = Image.new("RGB", (128, 128), "white")
            draw = ImageDraw.Draw(image)
            draw.line((16, 16, 112, 16), fill="black", width=3)
            draw.line((16, 16, 16, 112), fill="black", width=3)
            draw.arc((35, 35, 80, 80), start=0, end=90, fill="black", width=2)
            image.save(input_path)

            run_pipeline(str(input_path), str(output_path))

            self.assertTrue(output_path.exists())
            svg_text = output_path.read_text(encoding="utf-8")
            self.assertIn("<line", svg_text)
            self.assertIn('stroke="black"', svg_text)
            self.assertIn('fill="none"', svg_text)
            self.assertTrue((temp_path / "nested" / "debug" / "black_mask_raw.png").exists())
            self.assertTrue((temp_path / "nested" / "debug" / "black_mask_clean.png").exists())
            self.assertTrue((temp_path / "nested" / "debug" / "structure_mask.png").exists())
            self.assertTrue((temp_path / "nested" / "debug" / "rejected_symbol_mask.png").exists())
            self.assertTrue((temp_path / "nested" / "debug" / "architecture_edges.png").exists())
            self.assertTrue((temp_path / "nested" / "debug" / "architecture_preview.png").exists())
            source_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()[:10]
            candidate_dir = (
                temp_path
                / "nested"
                / "debug"
                / "annotation_candidates"
                / f"{input_path.stem}_{source_hash}"
            )
            self.assertTrue(
                (candidate_dir / "manifest.csv").exists()
            )

    def test_pipeline_raises_for_missing_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "missing.png"
            output_path = temp_path / "result.svg"

            with self.assertRaises(FileNotFoundError):
                run_pipeline(str(input_path), str(output_path))

    def test_segmentation_and_cleanup_remove_colored_annotations(self) -> None:
        image = np.full((48, 48, 3), (200, 180, 150), dtype=np.uint8)
        image[:, 8] = (0, 0, 0)
        image[20, :] = (255, 255, 0)
        image[:, 30] = (0, 0, 255)
        image[10:13, 10:13] = (255, 255, 255)

        segmentation_masks = build_segmentation_masks(image, ColorMaskSettings())
        black_mask_clean = clean_black_mask(
            segmentation_masks.black_mask_raw,
            segmentation_masks.rejected_colored_mask,
            BlackMaskCleanupSettings(min_component_area=4),
        )

        self.assertEqual(int(segmentation_masks.black_mask_raw[8, 8]), 255)
        self.assertEqual(int(segmentation_masks.cyan_mask[20, 2]), 255)
        self.assertEqual(int(segmentation_masks.colored_outline_mask[10, 30]), 255)
        self.assertEqual(int(segmentation_masks.white_callout_mask[11, 11]), 255)
        self.assertEqual(int(black_mask_clean[20, 2]), 0)
        self.assertEqual(int(black_mask_clean[10, 30]), 0)

    def test_component_filter_rejects_text_like_shapes_but_keeps_structure(self) -> None:
        black_mask_clean = np.zeros((80, 80), dtype=np.uint8)
        black_mask_clean[10:14, 8:55] = 255
        pillar_like = Image.new("L", (80, 80), 0)
        pillar_draw = ImageDraw.Draw(pillar_like)
        pillar_draw.rectangle((8, 30, 16, 38), outline=255, width=1)
        black_mask_clean = np.maximum(black_mask_clean, np.array(pillar_like, dtype=np.uint8))

        text_like = Image.new("L", (80, 80), 0)
        text_draw = ImageDraw.Draw(text_like)
        text_draw.text((45, 48), "321", fill=255)
        text_draw.ellipse((58, 50, 64, 56), outline=255, width=1)
        black_mask_clean = np.maximum(black_mask_clean, np.array(text_like, dtype=np.uint8))

        filter_result = filter_structure_components(
            black_mask_clean,
            ComponentFilterSettings(),
        )

        self.assertEqual(int(filter_result.structure_mask[12, 20]), 255)
        self.assertEqual(int(filter_result.structure_mask[34, 8]), 255)
        self.assertGreater(
            int(np.count_nonzero(filter_result.rejected_symbol_mask[48:60, 45:60])),
            0,
        )
        self.assertGreater(
            int(np.count_nonzero(filter_result.rejected_symbol_mask[49:58, 57:66])),
            0,
        )

    def test_component_filter_classifier_can_keep_ambiguous_cluster(self) -> None:
        black_mask_clean = np.zeros((64, 64), dtype=np.uint8)
        drawing = Image.new("L", (64, 64), 0)
        draw = ImageDraw.Draw(drawing)
        draw.ellipse((24, 24, 32, 32), outline=255, width=1)
        black_mask_clean = np.maximum(black_mask_clean, np.array(drawing, dtype=np.uint8))

        filter_result = filter_structure_components(
            black_mask_clean,
            ComponentFilterSettings(),
            classifier=_FakeStructureClassifier(),
        )

        self.assertEqual(int(filter_result.structure_mask[28, 24]), 255)
        self.assertEqual(int(filter_result.rejected_symbol_mask[28, 24]), 0)
        self.assertEqual(len(filter_result.candidate_crops), 1)
        self.assertEqual(filter_result.candidate_crops[0].predicted_label, "structure_square_pillar")

    def test_normalize_candidate_crop_returns_expected_shape(self) -> None:
        candidate_mask = np.zeros((24, 40), dtype=np.uint8)
        candidate_mask[8:16, 12:28] = 255

        features = normalize_candidate_crop(candidate_mask, input_size=32)

        self.assertEqual(features.shape, (1, 32 * 32))
        self.assertGreater(float(features.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
