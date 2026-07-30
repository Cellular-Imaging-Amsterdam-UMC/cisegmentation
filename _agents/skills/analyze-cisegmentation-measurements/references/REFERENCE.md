# CI Segmentation measurements database reference

## Contents

- [Database output](#database-output)
- [Measurement basis](#measurement-basis)
- [Tables](#tables)
- [Convenience views](#convenience-views)
- [Schema inspection](#schema-inspection)
- [Query examples](#query-examples)

## Database output

CI Segmentation can write one measurements database for every top-level input
OME-Zarr. A regular image produces one database containing that image. An HCS
plate produces one database containing every well and field in the plate.

For HCS inputs, measurements begin after all configured model passes and native
label finalization succeed. Spawned CPU workers read source pixels and final
generated or preserved label groups, then a single parent writer merges bounded
field shards into the final database.

The **Create Measurements Database** selector offers:

- `DuckDB` (default): best for large analytical queries and browser-local Analysis.
- `SQLite`: maximum compatibility with Python's standard library and database
  tools.
- `Skip`: do not calculate or write measurements.

The database is written next to the segmentation OME-Zarr:

```text
sample__cisegmentation.ome.zarr
sample__cisegmentation_measurements.duckdb
```

or:

```text
sample__cisegmentation_measurements.sqlite
```

Database creation is atomic. A completed database is installed only after all
fields, indexes, views, and transactions have finished. Benchmark galleries do
not produce a measurements database because their pixels are rendered model
comparisons rather than instance-label results.

## Measurement basis

Measurements are calculated from:

1. Final label masks after cell/nucleus matching, cell expansion, border
   removal, label offsetting, and other post-processing.
2. Level-0 pixels from every original input channel.
3. Pixel sizes from the input OME-Zarr coordinate transformation.

No intensity normalization, background subtraction, flat-field correction, or
photobleaching correction is applied. Intensity values retain the numeric units
of the source image.

Missing physical pixel sizes result in SQL `NULL` for measurements that require
those sizes; pixel-based measurements remain available.

Centroids use the center of mass of mask pixels or voxels. Pixel coordinates
are zero-based. Physical coordinates are calculated as:

```text
pixel coordinate × axis scale
```

Bounding-box minima are inclusive and maxima are exclusive, matching NumPy
slicing.

Schema v5 includes label origin and calculates bounded image-QC metrics directly from
level-0 pixels already in memory. Expensive calculations use a deterministic
regular grid capped at 1,048,576 samples per plane. Scores compare fields
within the same run, channel index, timepoint, and Z plane. They identify
**review candidates** for inspection; they do not establish that an image is
bad.

## Tables

### `schema_info`

Small key/value metadata describing the database schema, coordinate unit, and
bounding-box convention.

### `measurement_runs`

One row describing the workflow run:

- creation time in UTC;
- CI Segmentation version;
- measurement schema version;
- database format;
- source OME-Zarr and output OME-Zarr;
- `output_store_uuid`, shared with the output OME-Zarr root
  `cisegmentation.output_store_uuid` metadata;
- complete workflow settings as JSON.

### `images`

One row per regular image or HCS field. It records source and output resource
paths, plate row/column/field identifiers, `TCZYX` dimensions, source data type,
and available T/Z/Y/X scales.

### `channels`

One row per original input channel. `channel_index` is one-based, matching the
launcher. Channel name and display color come from OME-Zarr OMERO metadata when
available.

### `label_sets`

One row per native OME-Zarr label group, such as:

- `labels_cells`;
- `labels_nuclei`;
- `labels_cytoplasm`;
- `labels_spots_channel_2`;
- `labels_foci_channel_2`;
- `labels_bacteria_channel_1`.

Duplicate Step 3 selections remain separate label sets through
`label_set_index`, even when their displayed names are equal.

`locations_only` distinguishes Spotiflow point locations from true masks.
`label_origin` is `generated` or `existing`. `output_label_path` identifies the
native OME-Zarr label group, `output_label_kind` is `label-image`, and the
compatibility column `output_channel_index` is `NULL` for new outputs.

### `label_set_sources`

One or more rows link each output label set to the original input channel(s)
used to produce it. Each row records the producing workflow step, model,
channel role (`primary` or `nuclei`), and a `channel_id` link to `channels`.
Derived cytoplasm may reference both cell and nucleus inputs. Use the
`label_sources` view to read the one-based channel index and channel name.

### `objects`

One row per nonzero label value and timepoint. Identifiers are:

- `object_id`: database-wide stable identifier used by measurement and
  relationship tables;
- `label_set_id`: the mask channel containing the object;
- `label_value`: the integer value present in the OME-Zarr mask;
- `image_id` and `timepoint`: source image and zero-based T index.

Location and size columns include:

- voxel or pixel count;
- 2D area in pixels² and µm²;
- 3D volume in voxels and µm³;
- centroid Z/Y/X in pixels and, where calibrated, µm;
- bounding-box Z/Y/X minima and exclusive maxima in pixels and µm.

2D shape columns include:

- convex and filled area;
- equivalent circular diameter;
- major and minor axis lengths;
- aspect ratio;
- maximum Feret diameter;
- perimeter and Crofton perimeter;
- circularity, `4π × area / perimeter²`;
- eccentricity;
- solidity, `area / convex area`;
- extent, `area / bounding-box area`;
- orientation in degrees;
- Euler number.

Lengths converted to µm use the mean XY pixel size. Areas use `Y scale × X
scale`. Areas are exact for anisotropic XY calibration; converted perimeters
and axis lengths are approximations when X and Y scales differ.

True 3D objects additionally receive:

- filled and convex volumes;
- equivalent spherical diameter;
- 3D major/minor axis lengths in voxels and, when calibrated, µm;
- extent, solidity, aspect ratio, and Euler number;
- marching-cubes surface area in µm² when XYZ calibration is available;
- sphericity, `π^(1/3) × (6 × volume)^(2/3) / surface area`.

An object is treated as 2D when all its voxels occupy one Z plane. This also
applies to objects from forced slice-wise segmentation.

Point-only label sets store centroid and sampled intensities, but bounding boxes
and shape measurements are `NULL`. A single-pixel mask produced by local
Spotiflow refinement remains a mask, not a point-only location.

### `intensity_measurements`

There is one row for every object × original image channel. Values are sampled
only where the object's final mask is nonzero:

- sample count;
- sum or integrated intensity;
- arithmetic mean;
- population variance and standard deviation;
- minimum and maximum;
- median;
- median absolute deviation;
- 10th, 25th, 75th, and 90th percentiles;
- coefficient of variation, `standard deviation / mean`, with `NULL` for mean
zero.

### `image_quality_measurements` (schema v4)

One row per image/field × timepoint × Z plane × original channel:

- focus score (variance of the discrete Laplacian) and gradient energy;
- mean, median, population standard deviation, MAD, P1 and P99;
- fractions equal to the sampled minimum and maximum;
- illumination block CV, edge/center ratio and normalized fitted gradient;
- robust bright-area and largest connected bright-component fractions;
- plate-relative component and overall anomaly scores;
- `review_candidate` and JSON `review_reasons`.

The minimum/maximum fractions are acquisition-range proxies. Confirm the
channel data type and display range before calling them clipping. Older
schema-v3 databases do not contain this table; report that image-QC metrics
are unavailable and continue using object measurements.

### `field_quality_measurements` (schema v4)

One row per image/field × timepoint with cell, nucleus, foci and total label
counts, plate-relative cell-count score, zero/low/high count flags, the maximum
image anomaly score, and review-candidate provenance.

For a point-only Spotiflow object, all statistics describe its single sampled
pixel. They are consequently equal except variance, standard deviation, median
absolute deviation, and coefficient of variation.

### `relationships`

Relationships are derived from exact overlap between every pair of label sets
in the same image and timepoint. Each overlap is stored in both directions so
queries can ask either which cell contains a focus or which foci belong to a
cell.

Columns include:

- source and target object/label-set identifiers;
- overlap in voxels, plus µm² for 2D or µm³ for calibrated 3D;
- overlap fraction relative to the source and target;
- whether the source centroid lies inside the target;
- physical centroid-to-centroid distance when calibration is available;
- `is_primary_for_source`, identifying the largest overlap with a particular
  target label set.

`relation` has one of four values:

- `inside`: the complete source mask overlaps the target;
- `contains`: the complete target mask overlaps the source;
- `identical_extent`: both masks have the same extent;
- `overlaps`: partial overlap in both directions.

This supports focus membership in cells, nuclei, and cytoplasm without losing
partial or ambiguous overlaps. A focus may have a primary cell relationship and
separate primary nucleus/cytoplasm relationships.

## Convenience views

- `object_features`: objects joined with image and label-set context.
- `intensity_features`: intensity rows joined with object and channel names.
- `image_quality_features`: image-QC rows joined with image, plate and channel
  context (schema v4).
- `field_quality_summary`: field counts and review-candidate scores joined
  with plate and navigation context (schema v4).
- `label_sources`: output label sets joined with their producing step, model,
  channel role, one-based input channel index, and input channel name.
- `object_navigation`: one row per object with output store UUID, output
  OME-Zarr and HCS field path, label storage location, label value, centroid,
  half-open pixel bounding box, image dimensions, and calibration.
- `mask_relationships`: relationships with source/target types and label names.
- `foci_assignments`: primary spot/foci/bacteria relationships to cells,
  nuclei, and cytoplasm.

## Schema inspection

Inspect DuckDB tables and views:

```python
db.sql("""
    SELECT table_name, table_type
    FROM information_schema.tables
    WHERE table_schema = 'main'
    ORDER BY table_type, table_name
""").df()
```

Inspect a DuckDB relation:

```python
db.sql("DESCRIBE object_features").df()
```

Inspect SQLite tables and views:

```python
import pandas as pd

pd.read_sql_query("""
    SELECT name, type
    FROM sqlite_master
    WHERE type IN ('table', 'view')
    ORDER BY type, name
""", db)
```

Inspect a SQLite relation:

```python
pd.read_sql_query("PRAGMA table_info(object_features)", db)
```

## Query examples

Adapt every example to the columns and values present in the actual database.

### Cells and calibrated area

```sql
SELECT image_name, plate_row, plate_column, field_index,
       object_id, area_um2, centroid_y_um, centroid_x_um
FROM object_features
WHERE object_type = 'cells'
```

### Cell intensity for a named input channel

```sql
SELECT image_id, object_id, channel_name,
       intensity_mean, intensity_median, intensity_sum
FROM intensity_features
WHERE object_type = 'cells'
  AND channel_name = 'Cytoplasm'
```

### Producing channels for label sets

```sql
SELECT label_name, source_step, source_model, channel_role,
       channel_index, channel_name
FROM label_sources
ORDER BY label_set_index, source_step, channel_role
```

### Navigate to a measured object

```sql
SELECT output_store_uuid, output_resource_path, output_label_kind,
       output_label_path, output_channel_index, label_value, timepoint,
       centroid_z_px, centroid_y_px, centroid_x_px,
       bbox_min_y_px, bbox_min_x_px, bbox_max_y_px, bbox_max_x_px
FROM object_navigation
WHERE object_id = ?
```

The database has no environment-specific OMERO Image or Plate ID. Obtain it
from the authenticated active OMERO context and compare the viewer store UUID
with `output_store_uuid` before opening a field or rendering an ROI.

### Cell with most assigned foci, including render navigation

This returns the winning cell and every assigned focus in one query. A
successful result should be cached and cited by the render call; do not issue
separate schema or navigation queries.

```sql
WITH focus_counts AS (
    SELECT target_object_id AS cell_object_id, COUNT(*) AS foci_count
    FROM foci_assignments
    WHERE target_object_type = 'cells'
      AND relation IN ('inside', 'identical_extent')
    GROUP BY target_object_id
),
winner AS (
    SELECT cell_object_id, foci_count
    FROM focus_counts
    ORDER BY foci_count DESC, cell_object_id
    LIMIT 1
)
SELECT w.foci_count,
       cell.object_id AS cell_object_id,
       cell.output_store_uuid, cell.output_resource_path,
       cell.output_label_path AS cell_label_path,
       cell.label_value AS cell_label_value,
       cell.timepoint, cell.centroid_z_px,
       cell.bbox_min_y_px, cell.bbox_min_x_px,
       cell.bbox_max_y_px, cell.bbox_max_x_px,
       focus.object_id AS focus_object_id,
       focus.output_label_path AS focus_label_path,
       focus.label_value AS focus_label_value
FROM winner w
JOIN object_navigation cell ON cell.object_id = w.cell_object_id
LEFT JOIN foci_assignments a
  ON a.target_object_id = w.cell_object_id
 AND a.target_object_type = 'cells'
 AND a.relation IN ('inside', 'identical_extent')
LEFT JOIN object_navigation focus ON focus.object_id = a.source_object_id
ORDER BY focus.object_id
```

### Top-cell gallery render contract

For a ranked gallery, keep each cell's navigation fields and assigned focus
rows together while grouping. Build one bounded panel dictionary per selected
cell with these exact snake-case keys:

```python
panel = {
    "field": output_resource_path,
    "roi": [x0, y0, x1, y1],
    "source_channels": [source_channel],
    "t": timepoint,
    "z": z_index,
    "title": f"Cell {cell_label_value}",
    "caption": f"{foci_count} assigned foci",
    "overlays": [
        {
            "label_path": cell_label_path,
            "values": [cell_label_value],
            "mode": "outline",
            "color": "#FFFF00",
            "opacity": 1,
            "outline_width": 2,
            "name": "cell",
        },
        {
            "label_path": focus_label_path,
            "values": focus_label_values,
            "mode": "outline",
            "color": "#FF00FF",
            "opacity": 1,
            "outline_width": 2,
            "name": "foci",
        },
    ],
}
```

The only Python value returned to the assistant is the global `result`.
Return the complete render contract, not a shortened preview:

```python
result = {
    "store_uuid": store_uuid,
    "render_panels": panels,
}
```

Copy `render_panels` unchanged into the gallery call. Never reconstruct label
values from titles, row positions, object IDs, or a separate summary table.

### Assigned foci by compartment

```sql
SELECT target_object_id AS compartment_id,
       target_object_type AS compartment,
       COUNT(*) AS foci_count
FROM foci_assignments
WHERE relation IN ('inside', 'identical_extent')
GROUP BY target_object_id, target_object_type
```

### Cell counts per well

```sql
SELECT plate_row, plate_column,
       COUNT(*) AS cell_count,
       AVG(area_um2) AS mean_cell_area_um2
FROM object_features
WHERE object_type = 'cells'
GROUP BY plate_row, plate_column
ORDER BY plate_row, plate_column
```

### Field review candidates

```sql
SELECT plate_row, plate_column, field_index, image_name, timepoint,
       cell_count, cell_count_robust_z, image_anomaly_score, anomaly_score,
       review_reasons
FROM field_quality_summary
WHERE review_candidate
ORDER BY anomaly_score DESC, plate_row, plate_column, field_index
LIMIT 25
```

### Segmentation-QC gallery candidates

```sql
SELECT object_id, object_type, image_name, plate_row, plate_column, field_index,
       area_px2, solidity, aspect_ratio, euler_number,
       output_resource_path, output_label_path, label_value,
       bbox_min_y_px, bbox_min_x_px, bbox_max_y_px, bbox_max_x_px
FROM object_navigation
WHERE object_type = 'cells'
ORDER BY solidity ASC, area_px2 DESC
LIMIT 25
```

Run filters and aggregation inside the database before using `.df()` or
`pandas.read_sql_query`. Close the connection when finished.
