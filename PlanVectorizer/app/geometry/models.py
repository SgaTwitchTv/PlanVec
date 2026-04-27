"""Geometry data models."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Tuple


@dataclass(frozen=True)
class LineSegment:
    """A 2D line segment defined by start and end pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        """Normalize NumPy scalar coordinates to plain Python ints."""
        object.__setattr__(self, "x1", int(self.x1))
        object.__setattr__(self, "y1", int(self.y1))
        object.__setattr__(self, "x2", int(self.x2))
        object.__setattr__(self, "y2", int(self.y2))

    @property
    def length(self) -> float:
        """Return the Euclidean length of the segment."""
        return hypot(self.x2 - self.x1, self.y2 - self.y1)

    def canonical_key(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return an order-independent key for exact duplicate removal."""
        start = (self.x1, self.y1)
        end = (self.x2, self.y2)
        return (start, end) if start <= end else (end, start)


@dataclass(frozen=True)
class Polyline:
    """A polyline defined by ordered 2D pixel coordinates."""

    points: Tuple[Tuple[int, int], ...]

    def __post_init__(self) -> None:
        normalized_points = tuple((int(x), int(y)) for x, y in self.points)
        object.__setattr__(self, "points", normalized_points)
