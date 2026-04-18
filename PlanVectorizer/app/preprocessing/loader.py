"""Image loading utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


def load_image(input_path: str) -> np.ndarray:
    """Load an image from disk as a BGR NumPy array."""
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {path}")

    try:
        with Image.open(path) as image:
            rgb_image = image.convert("RGB")
            rgb_array = np.array(rgb_image)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Unable to read image file: {path}") from exc

    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
