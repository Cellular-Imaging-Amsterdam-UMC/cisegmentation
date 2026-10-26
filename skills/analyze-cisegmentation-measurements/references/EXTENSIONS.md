# Optional extension schema 3

Base schema 5 tables and views are unchanged. New features are optional and off
by default. Confirm availability before querying: inspect `measurement_extensions`
and its `name`, `version`, `settings_json`. Only enabled features have registry rows;
empty extension tables alone do not mean a measurement was requested. Revision 3
supports `spatial`, `colocalization`, and `geometry`.

## Identity and calibration

`object_id` identifies one measurement row. `label_value` is its raster value at
that field, label set and timepoint. Numeric IDs need not remain stable across runs.
`object_keys(object_id, object_uuid)` adds a stable UUID derived from output store
identity, resource path, label path, timepoint and label value within one output.
Never join outputs merely because their numeric object IDs happen to match.

`coordinate_frames` has `image_id`, `spatial_unit`, `scale_zyx_json`,
`translation_zyx_json`, `seconds_per_frame`, `time_unit`, and `metadata_json`.
Scale and translation compose supported dataset and global NGFF transforms.
Spatial features require finite, positive, unit-calibrated spatial
axes. Geometry without calibration uses pixels and NULL physical coordinates.
Known time units convert to seconds; missing time calibration is reported as
frames. A time scale with no time unit must not be assumed to mean seconds.

`point_localizations(object_id,z_px,y_px,x_px,coordinate_source)` retains native
Spotiflow subpixel coordinates. It does not replace legacy raster centroids.
The source is `spotiflow_subpixel`; fallback coordinates uses `raster_centroid`.
Tiled points receive the crop origin and final label remapping before insertion.

## Spatial measurements

`spatial_measurements`: `object_id`, `nearest_object_id`, `nearest_distance_um`,
`neighbor_count`, `radius_um`, `touching_neighbor_count`, `shared_boundary`,
`boundary_unit`. Neighbors are in the same field, label set and timepoint; the
object itself is excluded. A sole object has NULL nearest distance and ID.
Distances use physical XYZ positions, with native positions for detected spots.

`object_contacts(source_object_id,target_object_id,shared_boundary,boundary_unit)`
stores each unordered label pair once. Shared raster edges give micrometers in
2D; shared voxel faces give square micrometers in 3D. Chunk/tile boundaries are
included. Corner-only touching has zero shared boundary.

`spot_counts(object_id,spot_label_set_id,spot_count)` uses existing primary
overlap assignments into cells/nuclei. Existing parent masks receive zero when
no spots are assigned. Spot-only runs have no parent rows or invented assignments.
Point counts reflect raster membership, preserving existing assignment semantics.

```sql
SELECT n.image_name, n.label_name, o.timepoint, o.label_value,
       s.nearest_distance_um, s.neighbor_count, s.touching_neighbor_count,
       s.shared_boundary, s.boundary_unit
FROM spatial_measurements s JOIN objects o USING (object_id)
JOIN object_features n USING (object_id)
ORDER BY s.nearest_distance_um DESC LIMIT 100;
```

## Colocalization

`colocalization_thresholds`: `image_id`, `timepoint`, `channel_id`, `threshold`,
`sample_count`, `method`, `sampling_json`. Deterministic regular-grid sampling
uses at most 1,048,576 finite source pixels per field/frame/channel. Otsu uses
4096 histogram bins. Constant signals use the constant threshold. No manual
threshold override exists; channel pairs use one-based channel numbers in settings.

`colocalization_measurements`: `object_id`, `channel_a_id`, `channel_b_id`,
`pearson_r`, `manders_a_in_b`, `manders_b_in_a`, `sample_count`, `reason`.
Pearson uses finite paired pixels inside the final raster object. Manders A-in-B
is A intensity at pixels with B strictly above B's Otsu threshold divided by all
finite-paired A intensity; B-in-A is the reverse. Intensities are original,
not normalized or background-corrected. Negative intensity makes Manders NULL;
zero denominators, constant channels or fewer than two pixels produce explained
NULLs. A native one-voxel spot normally has undefined Pearson correlation.

```sql
SELECT o.object_id, o.timepoint, o.object_type,
       a.channel_index AS channel_a, b.channel_index AS channel_b,
       c.pearson_r, c.manders_a_in_b, c.manders_b_in_a, c.reason
FROM colocalization_measurements c JOIN objects o USING (object_id)
JOIN channels a ON a.channel_id=c.channel_a_id
JOIN channels b ON b.channel_id=c.channel_b_id
ORDER BY o.object_id LIMIT 100;
```


## Geometry companion

Geometry export creates `<source>__cisegmentation_geometry.duckdb` or `.sqlite`,
matching the main database format. Load the geometry analysis skill/reference
when geometry is the attached file. Check `geometry_info.output_store_uuid`
against `measurement_runs.output_store_uuid` before joining client-side.
`geometries` contains shared `object_id`, `object_uuid`, `output_store_uuid`,
`image_id`, `resource_path`, `label_set_id`, `label_name`, `label_value`,
`timepoint`, `z_index`, plus `geometry_id`, `geometry_type`, `geometry_wkb`,
`coordinate_unit`, `min_x_px`, `min_y_px`, `max_x_px`, `max_y_px`, `x_px`,
`y_px`, `z_px`, `x_um`, `y_um`, `z_um`, `mask_area_px2`, `coordinate_source`.

WKB is portable OGC/ISO: Polygon/MultiPolygon preserves final-mask holes and
disconnected components. Volumetric masks have one XY outline per occupied Z,
not a 3D surface. Native spots use Point/Point Z, preserving subpixel coordinates.
Pixel centres are integers; polygon edges are half-integers. Scalar bounding
boxes bound each actual outline. Apply `coordinate_frames` to convert geometry
into physical units; WKB itself is in pixel coordinates. Raster masks remain
authoritative for all legacy measurements. Automatic viewer display is separate.

Use scalar filters and bounded WKB batches. Database spatial extensions and
cross-file attachment may be unavailable in sandboxed consumers; query each file
separately and join verified identities client-side. Do not enable external
database access or install spatial extensions merely to inspect an attachment.

## Resource behavior

Feature export follows final object-ID assignment and shard merging. Object
crops and database inserts are bounded; polygon edges spill into temporary
SQLite storage. Live allocation checks stop a polygon that cannot fit in the remaining
budget. Polygon topology are not silently reduced.
