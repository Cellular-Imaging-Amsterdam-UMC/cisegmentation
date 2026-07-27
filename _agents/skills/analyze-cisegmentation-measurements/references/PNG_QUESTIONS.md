# PNG question patterns

Use this reference when the user asks to see measured objects, segmentation
quality, image quality, fields, wells, or time/Z context. First execute one
bounded SQL query that returns the ranking metric and complete
`object_navigation` or `field_quality_summary` context. Then send one
`zarr-render-v2` request for a single crop or one `zarr-gallery-v1` request for
up to 25 panels. Cite the successful analysis evidence ID.

For galleries, return the complete evidence contract as
`result = {"store_uuid": store_uuid, "render_panels": panels}` and copy those
panels unchanged into the render call. A shortened table is useful for display
but is not sufficient render evidence.

Do not ask “render now?” or “go?” when the target, channels, crop, and overlays
are already known. Do not claim object tracking unless the database contains
tracking relationships.

## Object galleries

- Show the top 12 cells with most foci.
- Show the largest, smallest, roundest, most eccentric, or least-solid cells.
- Show the brightest or dimmest cells in a named original channel.
- Show representative cells nearest the median and each quartile.
- Show morphological outliers beside a typical size-matched cell.

For ranking, keep ties deterministic with `object_id`. Include the metric in
each panel caption.

## Relationships and spatial context

- Show cells whose foci are mainly nuclear or cytoplasmic.
- Show cells with no nucleus, multiple overlapping nuclei, or unassigned foci.
- Show crowded versus isolated cells and their neighbours.
- Show likely clumps with surrounding cells.

Render the primary cell outline and related focus/nucleus values as separate
overlays. Expand the crop around the cell by a small, bounded margin so the
spatial claim is visible.

## Segmentation QC

- Show probable over-segmentation or under-segmentation candidates.
- Show extreme area, solidity, aspect ratio, holes, or cell–nucleus mismatch.
- Show border-touching, truncated, tiny, or implausibly large labels.
- Show raw image, contour overlay, translucent mask, and a side-by-side
  comparison.
- Compare label contours with a label mask at adjustable opacity.

Call these candidates for review. Morphology alone does not prove a
segmentation error.

## Field and well QC

- Which fields are blurred, saturated, dim, unevenly illuminated, or contain
  bright debris?
- Show fields or wells with zero, unusually low, or unusually high cell counts.
- Well A1 looks wrong: show its most anomalous field beside a typical peer.
- Show all fields from this well as a montage.
- Create a plate heatmap and a gallery of the top review candidates.
- Show one representative image for a row, column, treatment, or control
  group when those annotations exist.

Use schema-v4 `field_quality_summary` and `image_quality_features`. If those
relations are absent, say image-QC metrics are unavailable in the older
database; do not invent scores. A plate heatmap is a chart artifact; use its
ranked cells to request representative PNG panels when useful.

## Time and Z

- Show the sharpest and least-sharp Z planes.
- Show the same field across selected timepoints.
- Show signal or segmentation changes across Z without calling them tracks.

Keep all selected panels on the same display ranges when visual comparison is
the point.

## Render defaults

- Focused object: complete 2-screen-pixel outline.
- Mask-only request: fill at 30% opacity.
- Raw/contour/mask comparison: identical crop and intensity ranges.
- Galleries: one montage with titles, metric captions, legend, and scale bar;
  never one chat artifact per panel.
- Maximum: 25 panels, four intensity channels per panel, eight overlays per
  panel, and the renderer’s aggregate pixel limit.
