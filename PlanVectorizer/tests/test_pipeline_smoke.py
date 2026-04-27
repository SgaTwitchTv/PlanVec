"""Small smoke test for the raster-to-SVG pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app.segmentation.color_masks import ColorMaskSettings, build_segmentation_masks
from app.pipeline import run_pipeline
from app.preprocessing.cleaner import BlackMaskCleanupSettings, clean_black_mask


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
            self.assertTrue((temp_path / "nested" / "debug" / "architecture_edges.png").exists())
            self.assertTrue((temp_path / "nested" / "debug" / "architecture_preview.png").exists())

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


if __name__ == "__main__":
    unittest.main()
