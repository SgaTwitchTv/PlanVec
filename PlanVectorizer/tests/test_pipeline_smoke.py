"""Small smoke test for the raster-to-SVG pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from app.pipeline import run_pipeline


class PipelineSmokeTest(unittest.TestCase):
    """Verify that the MVP pipeline creates an SVG from a simple raster input."""

    def test_pipeline_creates_svg_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "source.png"
            output_path = temp_path / "result.svg"

            image = Image.new("RGB", (128, 128), "white")
            draw = ImageDraw.Draw(image)
            draw.line((16, 16, 112, 16), fill="black", width=3)
            draw.line((16, 16, 16, 112), fill="black", width=3)
            image.save(input_path)

            run_pipeline(str(input_path), str(output_path))

            self.assertTrue(output_path.exists())
            self.assertIn("<line", output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
