# Measurements database

CI Segmentation can write one measurements database for every top-level input
OME-Zarr. A regular image produces one database containing that image. An HCS
plate produces one database containing every well and field in the plate.
For HCS inputs, each field is measured immediately after its segmentation is
written, using the source image and final labels already in memory. The workflow
does not reread the output OME-Zarr or retain all plate fields before starting
measurements.

The advanced **Create Measurements Database** selector offers:

- `DuckDB` (default): best for large analytical queries and JupyterLite.
- `SQLite`: maximum compatibility with Python's standard library and database tools.
- `Skip`: do not calculate or write measurements.

The database is written next to the segmentation OME-Zarr:

```text
sample_multistep.ome.zarr
sample_multistep_measurements.duckdb
```

or, for SQLite:

```text
sample_multistep_measurements.sqlite
```

Database creation is atomic. A completed database is installed only after all
fields, indexes, views, and transactions have finished. Benchmark galleries do
not produce a measurements database because their pixels are rendered model
comparisons rather than instance-label results.

## Measurement basis

Measurements are calculated from:

1. The final label masks after cell/nucleus matching, cell expansion, border
   removal, label offsetting, and other post-processing.
2. Level-0 pixels from every original input channel.
3. Pixel sizes from the input OME-Zarr coordinate transformation.

No intensity normalization, background subtraction, flat-field correction, or
photobleaching correction is applied. Intensity values therefore retain the
numeric units of the source image. Missing physical pixel sizes result in SQL
`NULL` for measurements that require those sizes; pixel-based measurements
remain available.

Centroids use the center-of-mass of mask pixels/voxels. Pixel coordinates are
zero-based. Physical coordinates are calculated as `pixel coordinate × axis
scale`. Bounding-box minima are inclusive and maxima are exclusive, matching
NumPy slicing.

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

One row per output label channel, such as:

- `labels_cells`;
- `labels_nuclei`;
- `labels_cytoplasm`;
- `labels_spots_channel_2`;
- `labels_foci_channel_2`;
- `labels_bacteria_channel_1`.

Duplicate Step 3 selections remain separate label sets through
`label_set_index`, even when their displayed names are equal.

`locations_only` distinguishes Spotiflow point locations from true masks.
`output_label_path` identifies the corresponding OME-Zarr label group or image
channel.

### `objects`

One row per nonzero label value and timepoint. Identifiers are:

- `object_id`: database-wide stable identifier used by measurement and
  relationship tables;
- `label_set_id`: the mask channel containing the object;
- `label_value`: the integer value present in the OME-Zarr mask;
- `image_id` and `timepoint`: source image and zero-based T index.

Location and size columns include:

- voxel/pixel count;
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
scale`. This makes areas exact for anisotropic XY calibration; converted
perimeters and axis lengths are approximations when X and Y scales differ.

True 3D objects additionally receive:

- filled and convex volumes;
- equivalent spherical diameter;
- 3D major/minor axis lengths in voxels and, when calibrated, µm;
- extent, solidity, aspect ratio, and Euler number;
- marching-cubes surface area in µm² when XYZ calibration is available;
- sphericity, `π^(1/3) × (6 × volume)^(2/3) / surface area`.

An object is treated as 2D when all of its voxels occupy one Z plane. This is
also true for objects from forced slice-wise segmentation. Point-only label
sets store centroid and sampled intensities, but bounding boxes and shape
measurements are `NULL`. A single-pixel mask produced by local Spotiflow
refinement remains a mask, not a point-only location.

### `intensity_measurements`

There is one row for every object × original image channel. Values are sampled
only where the object's final mask is nonzero:

- sample count;
- sum/integrated intensity;
- arithmetic mean;
- population variance and standard deviation;
- minimum and maximum;
- median;
- median absolute deviation (MAD);
- 10th, 25th, 75th, and 90th percentiles;
- coefficient of variation, `standard deviation / mean` (`NULL` for mean zero).

For a point-only Spotiflow object, all statistics describe its single sampled
pixel and are consequently equal except variance, standard deviation, MAD, and
coefficient of variation.

### `relationships`

Relationships are derived from exact overlap between every pair of label sets
in the same image and timepoint. Each overlap is stored in both directions so
queries can naturally ask either “which cell contains this focus?” or “which
foci belong to this cell?”.

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
partial or ambiguous overlaps. A focus may have a primary cell relationship
and separate primary nucleus/cytoplasm relationships.

## Convenience views

- `object_features`: objects joined with image and label-set context.
- `intensity_features`: intensity rows joined with object and channel names.
- `mask_relationships`: relationships with source/target types and label names.
- `foci_assignments`: primary spot/foci/bacteria relationships to cells,
  nuclei, and cytoplasm.

## Reading DuckDB

DuckDB performs filtering and aggregation before creating a pandas DataFrame:

```python
%pip install duckdb  # needed once in a new Jupyter/JupyterLite environment

import duckdb

db = duckdb.connect("screen_multistep_measurements.duckdb", read_only=True)

cells = db.sql("""
    SELECT image_name, plate_row, plate_column, field_index,
           object_id, area_um2, centroid_y_um, centroid_x_um
    FROM object_features
    WHERE object_type = 'cells'
""").df()
```

Mean cell intensity for one named image channel:

```python
cell_signal = db.sql("""
    SELECT image_id, object_id, channel_name,
           intensity_mean, intensity_median, intensity_sum
    FROM intensity_features
    WHERE object_type = 'cells' AND channel_name = 'Cytoplasm'
""").df()
```

Count nuclear and cytoplasmic foci per cell:

```python
foci_per_cell = db.sql("""
    SELECT target_object_id AS compartment_id,
           target_object_type AS compartment,
           COUNT(*) AS foci_count
    FROM foci_assignments
    WHERE relation IN ('inside', 'identical_extent')
    GROUP BY target_object_id, target_object_type
""").df()
```

Per-well cell counts without loading all objects into notebook memory:

```python
well_summary = db.sql("""
    SELECT plate_row, plate_column, COUNT(*) AS cell_count,
           AVG(area_um2) AS mean_cell_area_um2
    FROM object_features
    WHERE object_type = 'cells'
    GROUP BY plate_row, plate_column
    ORDER BY plate_row, plate_column
""").df()
```

Close the file when finished:

```python
db.close()
```

In JupyterLite, upload the `.duckdb` file through the file browser. Current
Pyodide-based JupyterLite environments can install DuckDB's WebAssembly build
with `%pip install duckdb`; see DuckDB's
[browser-Python guide](https://duckdb.org/2024/10/02/pyodide). Browser execution
is single-threaded, so keep large queries inside DuckDB and return only filtered
or aggregated results with `.df()`; avoid converting an entire
multi-million-row table to pandas.

## Reading SQLite

SQLite requires only Python's standard library. Pandas can execute the same SQL:

```python
import sqlite3
import pandas as pd

db = sqlite3.connect("screen_multistep_measurements.sqlite")

cells = pd.read_sql_query("""
    SELECT * FROM object_features
    WHERE object_type = 'cells'
""", db)

db.close()
```

For very large screens, add restrictive `WHERE` clauses or SQL aggregation
before calling `read_sql_query`; otherwise pandas must allocate memory for the
complete result.
