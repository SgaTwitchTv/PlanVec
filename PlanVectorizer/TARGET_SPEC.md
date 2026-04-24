# PlanVectorizer Target Spec

This document captures the current target style inferred from the provided PNG/SVG reference pairs.

## Keep

- Architectural wall geometry
- Room boundary lines
- Long straight partitions
- Window and opening breaks when they affect wall continuity
- Door geometry where it is clearly structural

## Ignore

- Room labels and numbers
- Leader lines and annotation dots
- Legends and measurement callouts
- Colored cyan, red, green, and blue overlays
- UI artifacts, screenshots, and desktop elements
- Most non-structural symbols inside rooms

## Output Style

- Black strokes only
- No fill
- Correct SVG coordinates
- Strong preference for horizontal and vertical geometry
- Fewer, longer segments over many tiny contour fragments
- Minimal duplicates

## Near-Term Milestones

1. Isolate the structural dark drafting layer from the raster image.
2. Merge broken collinear wall segments into longer editable lines.
3. Preserve openings and selected door geometry.
4. Remove small fragments, text leftovers, and annotation remnants.
5. Export cleaner wall primitives with lower node counts.
