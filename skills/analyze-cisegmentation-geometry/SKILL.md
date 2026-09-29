---
name: analyze-cisegmentation-geometry
description: Inspect and query CI Segmentation geometry companion databases in DuckDB or SQLite, including final-mask WKB polygons, holes, disconnected components, per-Z outlines, native Spotiflow subpixel points, calibrated transforms, bounding-box filtering and safe links to measurement objects.
metadata:
  version: "1"
---

# Analyze CI Segmentation geometry

Read [GEOMETRY.md](references/GEOMETRY.md) when this skill activates.
Open the attached geometry database read-only, inspect `geometry_info`, and
verify geometry schema version and `output_store_uuid`. Use bounded scalar
queries before loading any WKB. When a measurement companion is available,
cross-check its output store UUID before joining objects client-side.

Explain that polygons describe final raster masks and preserve holes and
disconnected components. Volumetric masks have per-Z XY outlines, not 3D mesh
surfaces. Native spot points retain subpixel positions. Keep field, label set,
timepoint and label value attached to each geometry; numeric object IDs alone
are insufficient across runs. Convert pixels using the recorded transforms.
Never infer physical units when calibration is missing.

Geometry export does not make polygons appear automatically in an image viewer.
Use portable WKB and scalar bounding boxes; do not require database spatial
extensions or external file attachment in a sandboxed query service. Raster
masks remain authoritative for existing object and intensity measurements.
