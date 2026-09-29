# Geometry companion schema 1

File name: `<source>__cisegmentation_geometry.duckdb` or `.sqlite`, matching the
selected measurement database format. `geometry_info(key,value)` records format,
`geometry_schema_version`, `measurement_base_schema_version`,
`output_store_uuid`, pixel conventions and WKB encoding.

`geometries` columns:

| Group | Columns |
| --- | --- |
| Identity | `geometry_id`, `object_id`, `object_uuid`, `output_store_uuid`, `image_id`, `resource_path`, `label_set_id`, `label_name`, `label_value`, `timepoint`, `z_index` |
| Shape | `geometry_type`, `geometry_wkb`, `coordinate_unit`, `mask_area_px2`, `coordinate_source` |
| Bounds | `min_x_px`, `min_y_px`, `max_x_px`, `max_y_px` |
| Positions | `x_px`, `y_px`, `z_px`, `x_um`, `y_um`, `z_um` |

WKB stores Polygon or MultiPolygon for each occupied mask Z plane. Exact
half-integer pixel edges retain holes, disconnected components and area.
Spotiflow point-only output uses Point/Point Z with native subpixel coordinates;
other point-only objects fall back to their raster centroids. Mask refinement
uses final-mask polygons. Positions are centroids or native points, not polygon
vertices. `z_index` is present for mask outlines; XYZ points carry their own Z.

Geometry is in zero-based pixel coordinates: integer pixel centres and
half-integer mask edges. Unlike legacy object crop boxes, geometry maxima are
actual continuous outer bounds. Never round or interpret them as exclusive
integer crop limits. `mask_area_px2` is exact raster area on this Z plane,
not total voxel volume or a biological surface area.

`coordinate_frames(image_id,spatial_unit,scale_zyx_json,translation_zyx_json,
seconds_per_frame,time_unit,metadata_json)` records calibration. Convert with
`position_um = position_px * scale_zyx + translation_zyx` when
`spatial_unit='micrometer'`. When calibration is invalid/missing, geometry
remains in pixels and physical scalar coordinates are NULL. The original NGFF
metadata is retained. Missing physical time calibration gives frame units.

Open either format read-only using its standard database library. Filter by
field, time and scalar bounding boxes before retrieving a bounded WKB batch:

```sql
SELECT geometry_id, object_id, object_uuid, label_name, label_value,
       timepoint, z_index, geometry_type, geometry_wkb
FROM geometries
WHERE image_id=? AND timepoint=?
  AND max_x_px>=? AND min_x_px<=?
  AND max_y_px>=? AND min_y_px<=?
ORDER BY geometry_id LIMIT 100;
```

The ROI parameters are minimum X, maximum X, minimum Y, maximum Y, respectively.
Decode selected WKB client-side, for example with Shapely's `from_wkb`.
Do not select the entire geometry table into memory. Portable query services
may forbid spatial extensions and cross-file database attachment; query each
companion separately and join verified output UUID + object UUID client-side.
Within the same verified output, object IDs also link to `objects`.
Geometry export alone does not provide automatic viewer polygon display.
