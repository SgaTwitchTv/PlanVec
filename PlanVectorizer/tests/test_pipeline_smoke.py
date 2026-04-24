"""Small smoke test for the raster-to-SVG pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app.pipeline import run_pipeline
from app.preprocessing.cleaner import extract_structural_mask


class PipelineSmokeTest(unittest.TestCase):
    """Verify that the MVP pipeline creates an SVG from a simple raster input."""

    def test_pipeline_creates_svg_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "source.png"
            output_path = temp_path / "nested" / "result.svg"

            image = Image.new("RGB", (128, 128), "white")
            draw = ImageDraw.Draw(image)
            draw.line((16, 16, 112, 16), fill="black", width=3)
            draw.line((16, 16, 16, 112), fill="black", width=3)
            image.save(input_path)

            run_pipeline(str(input_path), str(output_path))

            self.assertTrue(output_path.exists())
            svg_text = output_path.read_text(encoding="utf-8")
            self.assertIn("<line", svg_text)
            self.assertIn('stroke="black"', svg_text)
            self.assertIn('fill="none"', svg_text)

    def test_pipeline_raises_for_missing_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "missing.png"
            output_path = temp_path / "result.svg"

            with self.assertRaises(FileNotFoundError):
                run_pipeline(str(input_path), str(output_path))

    def test_structural_mask_rejects_colored_overlays(self) -> None:
        image = np.full((32, 32, 3), 255, dtype=np.uint8)
        image[:, 8] = (0, 0, 0)
        image[16, :] = (255, 255, 0)
        image[:, 24] = (0, 0, 255)

        structural_mask = extract_structural_mask(image)

        self.assertEqual(int(structural_mask[10, 8]), 255)
        self.assertEqual(int(structural_mask[16, 2]), 0)
        self.assertEqual(int(structural_mask[10, 24]), 0)


if __name__ == "__main__":
    unittest.main()
