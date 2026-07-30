from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable, Iterable

import numpy as np

from . import __version__
from .ome_zarr_io import LabelResult, _label_group_names, new_output_store_uuid
from .settings import OUTPUT_NAME_POSTFIX


SCHEMA_VERSION = 5
DATABASE_FORMATS = {"duckdb": ".duckdb", "sqlite": ".sqlite"}
QUALITY_SAMPLE_PIXELS = 1_048_576


OBJECT_COLUMNS = (
    "object_id", "image_id", "label_set_id", "timepoint", "label_value",
    "object_type", "is_point", "spatial_dimensions", "voxel_count",
    "area_px2", "area_um2", "volume_voxels", "volume_um3",
    "centroid_z_px", "centroid_y_px", "centroid_x_px",
    "centroid_z_um", "centroid_y_um", "centroid_x_um",
    "bbox_min_z_px", "bbox_min_y_px", "bbox_min_x_px",
    "bbox_max_z_px", "bbox_max_y_px", "bbox_max_x_px",
    "bbox_min_z_um", "bbox_min_y_um", "bbox_min_x_um",
    "bbox_max_z_um", "bbox_max_y_um", "bbox_max_x_um",
    "convex_area_px2", "convex_area_um2", "filled_area_px2", "filled_area_um2",
    "convex_volume_voxels", "convex_volume_um3", "filled_volume_voxels",
    "filled_volume_um3", "equivalent_diameter_px", "equivalent_diameter_um",
    "major_axis_length_px", "major_axis_length_um", "minor_axis_length_px",
    "minor_axis_length_um", "feret_diameter_max_px", "feret_diameter_max_um",
    "perimeter_px", "perimeter_um", "perimeter_crofton_px",
    "perimeter_crofton_um", "circularity", "eccentricity", "solidity",
    "extent", "orientation_degrees", "euler_number", "aspect_ratio",
    "surface_area_um2", "sphericity",
)

INTENSITY_COLUMNS = (
    "object_id", "channel_id", "sample_count", "intensity_sum", "intensity_mean",
    "intensity_variance", "intensity_stddev", "intensity_min", "intensity_max",
    "intensity_median", "intensity_mad", "intensity_p10", "intensity_p25",
    "intensity_p75", "intensity_p90", "coefficient_of_variation",
)

RELATIONSHIP_COLUMNS = (
    "relationship_id", "image_id", "timepoint", "source_object_id",
    "target_object_id", "source_label_set_id", "target_label_set_id", "relation",
    "overlap_voxels", "overlap_um2", "overlap_um3", "source_overlap_fraction",
    "target_overlap_fraction", "source_centroid_in_target",
    "centroid_distance_um", "is_primary_for_source",
)

IMAGE_QUALITY_COLUMNS = (
    "image_quality_id", "image_id", "channel_id", "timepoint", "z_index",
    "sample_count", "finite_fraction", "focus_score", "gradient_energy",
    "intensity_mean", "intensity_median", "intensity_stddev", "intensity_mad",
    "intensity_p01", "intensity_p99", "minimal_pixel_fraction",
    "maximal_pixel_fraction", "illumination_block_cv", "edge_center_ratio",
    "illumination_fitted_gradient", "bright_area_fraction",
    "largest_bright_component_fraction", "focus_anomaly_score",
    "intensity_anomaly_score", "saturation_anomaly_score",
    "illumination_anomaly_score", "artifact_anomaly_score", "anomaly_score",
    "review_candidate", "review_reasons",
)

FIELD_QUALITY_COLUMNS = (
    "field_quality_id", "image_id", "timepoint", "cell_count",
    "nucleus_count", "foci_count", "total_label_count",
    "cell_count_robust_z", "zero_cell_count", "unusually_low_cell_count",
    "unusually_high_cell_count", "image_anomaly_score", "anomaly_score",
    "review_candidate", "review_reasons",
)


_SCHEMA = """
CREATE TABLE schema_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE measurement_runs (
    run_id BIGINT PRIMARY KEY,
    created_utc TEXT NOT NULL,
    software_version TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    database_format TEXT NOT NULL,
    source_store TEXT NOT NULL,
    output_ome_zarr TEXT NOT NULL,
    output_store_uuid TEXT NOT NULL,
    settings_json TEXT NOT NULL
);
CREATE TABLE images (
    image_id BIGINT PRIMARY KEY,
    run_id BIGINT NOT NULL,
    source_store TEXT NOT NULL,
    source_resource_path TEXT NOT NULL,
    output_resource_path TEXT NOT NULL,
    image_name TEXT NOT NULL,
    plate_row TEXT,
    plate_column TEXT,
    field_index TEXT,
    size_t BIGINT NOT NULL,
    size_c BIGINT NOT NULL,
    size_z BIGINT NOT NULL,
    size_y BIGINT NOT NULL,
    size_x BIGINT NOT NULL,
    source_dtype TEXT NOT NULL,
    scale_t DOUBLE,
    scale_z_um DOUBLE,
    scale_y_um DOUBLE,
    scale_x_um DOUBLE
);
CREATE TABLE channels (
    channel_id BIGINT PRIMARY KEY,
    image_id BIGINT NOT NULL,
    channel_index INTEGER NOT NULL,
    channel_name TEXT NOT NULL,
    channel_color TEXT
);
CREATE TABLE label_sets (
    label_set_id BIGINT PRIMARY KEY,
    image_id BIGINT NOT NULL,
    label_set_index INTEGER NOT NULL,
    label_name TEXT NOT NULL,
    label_origin TEXT NOT NULL,
    object_type TEXT NOT NULL,
    locations_only BOOLEAN NOT NULL,
    output_label_path TEXT NOT NULL,
    output_label_kind TEXT NOT NULL,
    output_channel_index INTEGER
);
CREATE TABLE label_set_sources (
    label_set_id BIGINT NOT NULL,
    channel_id BIGINT NOT NULL,
    channel_role TEXT NOT NULL,
    source_step TEXT NOT NULL,
    source_model TEXT NOT NULL,
    PRIMARY KEY (label_set_id, channel_id, channel_role, source_step)
);
CREATE TABLE objects (
    object_id BIGINT PRIMARY KEY,
    image_id BIGINT NOT NULL,
    label_set_id BIGINT NOT NULL,
    timepoint INTEGER NOT NULL,
    label_value BIGINT NOT NULL,
    object_type TEXT NOT NULL,
    is_point BOOLEAN NOT NULL,
    spatial_dimensions INTEGER NOT NULL,
    voxel_count BIGINT NOT NULL,
    area_px2 DOUBLE,
    area_um2 DOUBLE,
    volume_voxels DOUBLE,
    volume_um3 DOUBLE,
    centroid_z_px DOUBLE,
    centroid_y_px DOUBLE,
    centroid_x_px DOUBLE,
    centroid_z_um DOUBLE,
    centroid_y_um DOUBLE,
    centroid_x_um DOUBLE,
    bbox_min_z_px BIGINT,
    bbox_min_y_px BIGINT,
    bbox_min_x_px BIGINT,
    bbox_max_z_px BIGINT,
    bbox_max_y_px BIGINT,
    bbox_max_x_px BIGINT,
    bbox_min_z_um DOUBLE,
    bbox_min_y_um DOUBLE,
    bbox_min_x_um DOUBLE,
    bbox_max_z_um DOUBLE,
    bbox_max_y_um DOUBLE,
    bbox_max_x_um DOUBLE,
    convex_area_px2 DOUBLE,
    convex_area_um2 DOUBLE,
    filled_area_px2 DOUBLE,
    filled_area_um2 DOUBLE,
    convex_volume_voxels DOUBLE,
    convex_volume_um3 DOUBLE,
    filled_volume_voxels DOUBLE,
    filled_volume_um3 DOUBLE,
    equivalent_diameter_px DOUBLE,
    equivalent_diameter_um DOUBLE,
    major_axis_length_px DOUBLE,
    major_axis_length_um DOUBLE,
    minor_axis_length_px DOUBLE,
    minor_axis_length_um DOUBLE,
    feret_diameter_max_px DOUBLE,
    feret_diameter_max_um DOUBLE,
    perimeter_px DOUBLE,
    perimeter_um DOUBLE,
    perimeter_crofton_px DOUBLE,
    perimeter_crofton_um DOUBLE,
    circularity DOUBLE,
    eccentricity DOUBLE,
    solidity DOUBLE,
    extent DOUBLE,
    orientation_degrees DOUBLE,
    euler_number INTEGER,
    aspect_ratio DOUBLE,
    surface_area_um2 DOUBLE,
    sphericity DOUBLE
);
CREATE TABLE intensity_measurements (
    object_id BIGINT NOT NULL,
    channel_id BIGINT NOT NULL,
    sample_count BIGINT NOT NULL,
    intensity_sum DOUBLE NOT NULL,
    intensity_mean DOUBLE NOT NULL,
    intensity_variance DOUBLE NOT NULL,
    intensity_stddev DOUBLE NOT NULL,
    intensity_min DOUBLE NOT NULL,
    intensity_max DOUBLE NOT NULL,
    intensity_median DOUBLE NOT NULL,
    intensity_mad DOUBLE NOT NULL,
    intensity_p10 DOUBLE NOT NULL,
    intensity_p25 DOUBLE NOT NULL,
    intensity_p75 DOUBLE NOT NULL,
    intensity_p90 DOUBLE NOT NULL,
    coefficient_of_variation DOUBLE,
    PRIMARY KEY (object_id, channel_id)
);
CREATE TABLE image_quality_measurements (
    image_quality_id BIGINT PRIMARY KEY,
    image_id BIGINT NOT NULL,
    channel_id BIGINT NOT NULL,
    timepoint INTEGER NOT NULL,
    z_index INTEGER NOT NULL,
    sample_count BIGINT NOT NULL,
    finite_fraction DOUBLE NOT NULL,
    focus_score DOUBLE,
    gradient_energy DOUBLE,
    intensity_mean DOUBLE,
    intensity_median DOUBLE,
    intensity_stddev DOUBLE,
    intensity_mad DOUBLE,
    intensity_p01 DOUBLE,
    intensity_p99 DOUBLE,
    minimal_pixel_fraction DOUBLE,
    maximal_pixel_fraction DOUBLE,
    illumination_block_cv DOUBLE,
    edge_center_ratio DOUBLE,
    illumination_fitted_gradient DOUBLE,
    bright_area_fraction DOUBLE,
    largest_bright_component_fraction DOUBLE,
    focus_anomaly_score DOUBLE,
    intensity_anomaly_score DOUBLE,
    saturation_anomaly_score DOUBLE,
    illumination_anomaly_score DOUBLE,
    artifact_anomaly_score DOUBLE,
    anomaly_score DOUBLE,
    review_candidate BOOLEAN NOT NULL,
    review_reasons TEXT NOT NULL,
    UNIQUE (image_id, channel_id, timepoint, z_index)
);
CREATE TABLE field_quality_measurements (
    field_quality_id BIGINT PRIMARY KEY,
    image_id BIGINT NOT NULL,
    timepoint INTEGER NOT NULL,
    cell_count BIGINT NOT NULL,
    nucleus_count BIGINT NOT NULL,
    foci_count BIGINT NOT NULL,
    total_label_count BIGINT NOT NULL,
    cell_count_robust_z DOUBLE,
    zero_cell_count BOOLEAN NOT NULL,
    unusually_low_cell_count BOOLEAN NOT NULL,
    unusually_high_cell_count BOOLEAN NOT NULL,
    image_anomaly_score DOUBLE,
    anomaly_score DOUBLE,
    review_candidate BOOLEAN NOT NULL,
    review_reasons TEXT NOT NULL,
    UNIQUE (image_id, timepoint)
);
CREATE TABLE relationships (
    relationship_id BIGINT PRIMARY KEY,
    image_id BIGINT NOT NULL,
    timepoint INTEGER NOT NULL,
    source_object_id BIGINT NOT NULL,
    target_object_id BIGINT NOT NULL,
    source_label_set_id BIGINT NOT NULL,
    target_label_set_id BIGINT NOT NULL,
    relation TEXT NOT NULL,
    overlap_voxels BIGINT NOT NULL,
    overlap_um2 DOUBLE,
    overlap_um3 DOUBLE,
    source_overlap_fraction DOUBLE NOT NULL,
    target_overlap_fraction DOUBLE NOT NULL,
    source_centroid_in_target BOOLEAN NOT NULL,
    centroid_distance_um DOUBLE,
    is_primary_for_source BOOLEAN NOT NULL
);
"""


_VIEWS = """
CREATE VIEW object_features AS
SELECT o.*, i.image_name, i.plate_row, i.plate_column, i.field_index,
       ls.label_name, ls.locations_only
FROM objects o
JOIN images i ON i.image_id = o.image_id
JOIN label_sets ls ON ls.label_set_id = o.label_set_id;

CREATE VIEW intensity_features AS
SELECT m.*, o.image_id, o.label_set_id, o.timepoint, o.label_value, o.object_type,
       c.channel_index, c.channel_name
FROM intensity_measurements m
JOIN objects o ON o.object_id = m.object_id
JOIN channels c ON c.channel_id = m.channel_id;

CREATE VIEW image_quality_features AS
SELECT q.*, i.image_name, i.plate_row, i.plate_column, i.field_index,
       c.channel_index, c.channel_name
FROM image_quality_measurements q
JOIN images i ON i.image_id = q.image_id
JOIN channels c ON c.channel_id = q.channel_id;

CREATE VIEW field_quality_summary AS
SELECT f.*, i.image_name, i.plate_row, i.plate_column, i.field_index,
       i.source_resource_path, i.output_resource_path,
       CASE WHEN f.review_candidate THEN 'review candidate' ELSE 'within peer range' END
           AS review_status
FROM field_quality_measurements f
JOIN images i ON i.image_id = f.image_id;

CREATE VIEW label_sources AS
SELECT s.label_set_id, ls.image_id, ls.label_set_index, ls.label_name,
       ls.object_type, s.source_step, s.source_model, s.channel_role,
       c.channel_id, c.channel_index, c.channel_name
FROM label_set_sources s
JOIN label_sets ls ON ls.label_set_id = s.label_set_id
JOIN channels c ON c.channel_id = s.channel_id;

CREATE VIEW object_navigation AS
SELECT o.*, mr.output_store_uuid, mr.output_ome_zarr,
       i.source_store, i.source_resource_path, i.output_resource_path,
       i.image_name, i.plate_row, i.plate_column, i.field_index,
       i.size_t, i.size_c, i.size_z, i.size_y, i.size_x,
       i.scale_t, i.scale_z_um, i.scale_y_um, i.scale_x_um,
       ls.label_name, ls.locations_only, ls.output_label_path,
       ls.output_label_kind, ls.output_channel_index
FROM objects o
JOIN images i ON i.image_id = o.image_id
JOIN measurement_runs mr ON mr.run_id = i.run_id
JOIN label_sets ls ON ls.label_set_id = o.label_set_id;

CREATE VIEW mask_relationships AS
SELECT r.*, so.object_type AS source_object_type, sl.label_name AS source_label_name,
       tobj.object_type AS target_object_type, tl.label_name AS target_label_name
FROM relationships r
JOIN objects so ON so.object_id = r.source_object_id
JOIN label_sets sl ON sl.label_set_id = r.source_label_set_id
JOIN objects tobj ON tobj.object_id = r.target_object_id
JOIN label_sets tl ON tl.label_set_id = r.target_label_set_id;

CREATE VIEW foci_assignments AS
SELECT * FROM mask_relationships
WHERE source_object_type IN ('spots', 'foci', 'bacteria')
  AND target_object_type IN ('cells', 'nuclei', 'cytoplasm')
  AND is_primary_for_source;
"""


class _DatabaseWriter:
    def __init__(self, path: Path, database_format: str):
        self.path = path
        self.database_format = database_format
        if database_format == "duckdb":
            try:
                import duckdb
            except ImportError as exc:
                raise RuntimeError(
                    "DuckDB measurements require the pinned duckdb package"
                ) from exc
            self.connection = duckdb.connect(str(path))
        elif database_format == "sqlite":
            self.connection = sqlite3.connect(path)
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.execute("PRAGMA journal_mode=DELETE")
            self.connection.execute("PRAGMA synchronous=NORMAL")
        else:
            raise ValueError(f"Unsupported measurement database: {database_format}")
        self.connection.execute("BEGIN TRANSACTION")
        for statement in _SCHEMA.split(";"):
            if statement.strip():
                self.connection.execute(statement)

    def insert(self, table: str, columns: tuple[str, ...], rows: list[tuple]) -> None:
        if not rows:
            return
        if self.database_format == "duckdb":
            import pandas as pd

            frame = pd.DataFrame.from_records(rows, columns=columns)
            self.connection.register("_measurement_batch", frame)
            names = ", ".join(columns)
            self.connection.execute(
                f"INSERT INTO {table} ({names}) SELECT {names} FROM _measurement_batch"
            )
            self.connection.unregister("_measurement_batch")
        else:
            placeholders = ",".join("?" for _ in columns)
            self.connection.executemany(
                f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
                rows,
            )

    def finish(self) -> None:
        for statement in _VIEWS.split(";"):
            if statement.strip():
                self.connection.execute(statement)
        for statement in (
            "CREATE UNIQUE INDEX idx_images_resource ON images(source_store, source_resource_path)",
            "CREATE UNIQUE INDEX idx_channels_image_channel ON channels(image_id, channel_index)",
            "CREATE UNIQUE INDEX idx_label_sets_image_index ON label_sets(image_id, label_set_index)",
            "CREATE INDEX idx_label_set_sources_channel ON label_set_sources(channel_id, label_set_id)",
            "CREATE UNIQUE INDEX idx_objects_label ON objects(label_set_id, timepoint, label_value)",
            "CREATE INDEX idx_objects_image_type ON objects(image_id, object_type)",
            "CREATE INDEX idx_intensity_channel ON intensity_measurements(channel_id, object_id)",
            "CREATE INDEX idx_image_quality_peer ON image_quality_measurements(channel_id, timepoint, z_index)",
            "CREATE INDEX idx_image_quality_review ON image_quality_measurements(review_candidate, anomaly_score)",
            "CREATE INDEX idx_field_quality_review ON field_quality_measurements(review_candidate, anomaly_score)",
            "CREATE INDEX idx_relationship_source ON relationships(source_object_id, target_label_set_id)",
            "CREATE INDEX idx_relationship_target ON relationships(target_object_id, source_label_set_id)",
        ):
            self.connection.execute(statement)
        self.connection.execute("ANALYZE")
        self.connection.execute("COMMIT")
        self.connection.close()

    def abort(self) -> None:
        try:
            self.connection.execute("ROLLBACK")
        except Exception:
            pass
        self.connection.close()


def measurement_database_path(
    output_dir: str | Path, source_name: str, database_format: str
) -> Path:
    if database_format not in DATABASE_FORMATS:
        raise ValueError(f"Unsupported measurement database: {database_format}")
    return Path(output_dir) / (
        f"{source_name}{OUTPUT_NAME_POSTFIX}_measurements"
        f"{DATABASE_FORMATS[database_format]}"
    )


def _finite_scale(scales: dict[str, float], axis: str) -> float | None:
    value = float(scales.get(axis, float("nan")))
    return value if np.isfinite(value) and value > 0 else None


def _channel_metadata(result: LabelResult) -> list[tuple[str, str | None]]:
    configured = (result.source.attrs.get("omero") or {}).get("channels") or []
    output = []
    for index in range(result.source.data.shape[1]):
        item = configured[index] if index < len(configured) else {}
        output.append(
            (
                str(item.get("label") or f"Channel {index + 1}"),
                str(item["color"]) if item.get("color") is not None else None,
            )
        )
    return output


def _object_type(label_name: str) -> str:
    name = label_name.lower()
    for object_type in ("cytoplasm", "nuclei", "cells", "spots", "foci", "bacteria"):
        if object_type in name:
            return object_type
    return name.removeprefix("labels_").split("_channel_", 1)[0] or "objects"


def _location_flags(result: LabelResult) -> list[bool]:
    channel_count = result.labels.shape[1]
    statistics = list(result.provenance.get("output_statistics") or [])
    if len(statistics) == result.labels.shape[0] * channel_count:
        return [
            any(
                bool(statistics[t * channel_count + channel]["locations_only"])
                for t in range(result.labels.shape[0])
            )
            for channel in range(channel_count)
        ]
    parameters = result.provenance.get("parameters") or {}
    return [
        name.startswith("labels_spots_channel_")
        and not bool(parameters.get("spotiflow_local_refinement"))
        for name in (result.channel_labels or [f"labels_{result.target}"])
    ]


def _label_source_runs(result: LabelResult) -> list[list[dict[str, Any]]]:
    """Map each output label set to the inference run(s) that produced it."""
    label_names = result.channel_labels or [f"labels_{result.target}"]
    first_timepoint_runs = [
        run
        for run in (result.provenance.get("step_runs") or [])
        if int(run.get("timepoint", 0)) == 0
    ]
    cell_run = next(
        (
            run
            for run in first_timepoint_runs
            if run.get("step") in ("Step 1 cells", "Step 1 expansion nuclei")
        ),
        None,
    )
    nucleus_run = next(
        (run for run in first_timepoint_runs if run.get("step") == "Step 2 nuclei"),
        None,
    )
    if nucleus_run is None and cell_run is not None:
        if cell_run.get("step") == "Step 1 expansion nuclei":
            nucleus_run = cell_run
    foci_runs = iter(
        run
        for run in first_timepoint_runs
        if str(run.get("step", "")).startswith("Step 3")
    )

    sources: list[list[dict[str, Any]]] = []
    origins = result.label_origins or ["generated"] * len(label_names)
    for label_name, origin in zip(label_names, origins):
        if origin == "existing":
            sources.append([])
            continue
        object_type = _object_type(label_name)
        if object_type == "cells":
            runs = [cell_run] if cell_run is not None else []
        elif object_type == "nuclei":
            runs = [nucleus_run] if nucleus_run is not None else []
        elif object_type == "cytoplasm":
            runs = [
                run
                for run in (cell_run, nucleus_run)
                if run is not None
            ]
        elif object_type in ("spots", "foci", "bacteria"):
            run = next(foci_runs, None)
            runs = [run] if run is not None else []
        else:
            runs = []
        sources.append(runs)
    return sources


def _label_source_rows(
    result: LabelResult,
    label_set_ids: list[int],
    channel_ids: list[int],
) -> list[tuple]:
    rows: list[tuple] = []
    seen: set[tuple] = set()
    for label_set_id, runs in zip(label_set_ids, _label_source_runs(result)):
        for run in runs:
            channels = [("primary", int(run.get("primary_channel", 0)))]
            nuclei_channel = int(run.get("nuclei_channel", 0))
            if nuclei_channel > 0 and nuclei_channel != channels[0][1]:
                channels.append(("nuclei", nuclei_channel))
            for channel_role, channel_index in channels:
                if channel_index < 1 or channel_index > len(channel_ids):
                    continue
                row = (
                    label_set_id,
                    channel_ids[channel_index - 1],
                    channel_role,
                    str(run.get("step") or "unknown"),
                    str(run.get("model") or "unknown"),
                )
                if row not in seen:
                    seen.add(row)
                    rows.append(row)
    return rows


def _safe_property(region, name: str) -> float | None:
    try:
        value = getattr(region, name)
        array = np.asarray(value)
        if array.ndim or not np.isfinite(float(array)):
            return None
        return float(array)
    except (AttributeError, NotImplementedError, ValueError, ZeroDivisionError):
        return None


def _surface_area(mask: np.ndarray, spacing: tuple[float, float, float]) -> float | None:
    if mask.ndim != 3 or not np.any(mask):
        return None
    try:
        from skimage.measure import marching_cubes, mesh_surface_area

        padded = np.pad(mask.astype(np.uint8), 1)
        vertices, faces, _normals, _values = marching_cubes(
            padded, level=0.5, spacing=spacing
        )
        return float(mesh_surface_area(vertices, faces))
    except (RuntimeError, ValueError):
        return None


def _shape_values(
    region,
    *,
    locations_only: bool,
    scales: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    coords = region.coords
    centroid = tuple(float(value) for value in region.centroid)
    bbox = tuple(int(value) for value in region.bbox)
    voxel_count = int(region.area)
    z_um, y_um, x_um = (
        _finite_scale(scales, axis) for axis in ("z", "y", "x")
    )
    xy_mean = (y_um + x_um) / 2.0 if y_um is not None and x_um is not None else None
    pixel_area = y_um * x_um if y_um is not None and x_um is not None else None
    voxel_volume = (
        z_um * y_um * x_um
        if z_um is not None and y_um is not None and x_um is not None
        else None
    )
    is_2d = int(coords[:, 0].min()) == int(coords[:, 0].max())
    values: dict[str, Any] = {name: None for name in OBJECT_COLUMNS[8:]}
    values.update(
        {
            "voxel_count": voxel_count,
            "centroid_z_px": centroid[0],
            "centroid_y_px": centroid[1],
            "centroid_x_px": centroid[2],
            "centroid_z_um": centroid[0] * z_um if z_um is not None else None,
            "centroid_y_um": centroid[1] * y_um if y_um is not None else None,
            "centroid_x_um": centroid[2] * x_um if x_um is not None else None,
        }
    )
    if not locations_only:
        min_z, min_y, min_x, max_z, max_y, max_x = bbox
        values.update(
            {
                "bbox_min_z_px": min_z,
                "bbox_min_y_px": min_y,
                "bbox_min_x_px": min_x,
                "bbox_max_z_px": max_z,
                "bbox_max_y_px": max_y,
                "bbox_max_x_px": max_x,
                "bbox_min_z_um": min_z * z_um if z_um is not None else None,
                "bbox_min_y_um": min_y * y_um if y_um is not None else None,
                "bbox_min_x_um": min_x * x_um if x_um is not None else None,
                "bbox_max_z_um": max_z * z_um if z_um is not None else None,
                "bbox_max_y_um": max_y * y_um if y_um is not None else None,
                "bbox_max_x_um": max_x * x_um if x_um is not None else None,
            }
        )
        if is_2d:
            from skimage.measure import regionprops

            plane = np.asarray(region.image[0], dtype=np.uint8)
            shape_region = regionprops(plane)[0]
            area = float(shape_region.area)
            convex = _safe_property(shape_region, "area_convex")
            filled = _safe_property(shape_region, "area_filled")
            equivalent = _safe_property(shape_region, "equivalent_diameter_area")
            major = _safe_property(shape_region, "axis_major_length")
            minor = _safe_property(shape_region, "axis_minor_length")
            feret = _safe_property(shape_region, "feret_diameter_max")
            perimeter = _safe_property(shape_region, "perimeter")
            crofton = _safe_property(shape_region, "perimeter_crofton")
            values.update(
                {
                    "area_px2": area,
                    "area_um2": area * pixel_area if pixel_area is not None else None,
                    "convex_area_px2": convex,
                    "convex_area_um2": convex * pixel_area if convex is not None and pixel_area is not None else None,
                    "filled_area_px2": filled,
                    "filled_area_um2": filled * pixel_area if filled is not None and pixel_area is not None else None,
                    "equivalent_diameter_px": equivalent,
                    "equivalent_diameter_um": equivalent * xy_mean if equivalent is not None and xy_mean is not None else None,
                    "major_axis_length_px": major,
                    "major_axis_length_um": major * xy_mean if major is not None and xy_mean is not None else None,
                    "minor_axis_length_px": minor,
                    "minor_axis_length_um": minor * xy_mean if minor is not None and xy_mean is not None else None,
                    "feret_diameter_max_px": feret,
                    "feret_diameter_max_um": feret * xy_mean if feret is not None and xy_mean is not None else None,
                    "perimeter_px": perimeter,
                    "perimeter_um": perimeter * xy_mean if perimeter is not None and xy_mean is not None else None,
                    "perimeter_crofton_px": crofton,
                    "perimeter_crofton_um": crofton * xy_mean if crofton is not None and xy_mean is not None else None,
                    "circularity": 4.0 * math.pi * area / (perimeter * perimeter) if perimeter else None,
                    "eccentricity": _safe_property(shape_region, "eccentricity"),
                    "solidity": _safe_property(shape_region, "solidity"),
                    "extent": _safe_property(shape_region, "extent"),
                    "orientation_degrees": math.degrees(_safe_property(shape_region, "orientation")) if _safe_property(shape_region, "orientation") is not None else None,
                    "euler_number": int(shape_region.euler_number),
                    "aspect_ratio": major / minor if major is not None and minor else None,
                }
            )
        else:
            equivalent = (6.0 * voxel_count / math.pi) ** (1.0 / 3.0)
            major = _safe_property(region, "axis_major_length")
            minor = _safe_property(region, "axis_minor_length")
            convex = _safe_property(region, "area_convex")
            filled = _safe_property(region, "area_filled")
            physical_volume = voxel_count * voxel_volume if voxel_volume is not None else None
            surface = (
                _surface_area(region.image, (z_um, y_um, x_um))
                if z_um is not None and y_um is not None and x_um is not None
                else None
            )
            physical_major = physical_minor = None
            if z_um is not None and y_um is not None and x_um is not None:
                from skimage.measure import regionprops

                physical_region = regionprops(
                    np.asarray(region.image, dtype=np.uint8),
                    spacing=(z_um, y_um, x_um),
                )[0]
                physical_major = _safe_property(
                    physical_region, "axis_major_length"
                )
                physical_minor = _safe_property(
                    physical_region, "axis_minor_length"
                )
            values.update(
                {
                    "volume_voxels": float(voxel_count),
                    "volume_um3": physical_volume,
                    "convex_volume_voxels": convex,
                    "convex_volume_um3": convex * voxel_volume if convex is not None and voxel_volume is not None else None,
                    "filled_volume_voxels": filled,
                    "filled_volume_um3": filled * voxel_volume if filled is not None and voxel_volume is not None else None,
                    "equivalent_diameter_px": equivalent,
                    "equivalent_diameter_um": (6.0 * physical_volume / math.pi) ** (1.0 / 3.0) if physical_volume is not None else None,
                    "major_axis_length_px": major,
                    "major_axis_length_um": physical_major,
                    "minor_axis_length_px": minor,
                    "minor_axis_length_um": physical_minor,
                    "solidity": _safe_property(region, "solidity"),
                    "extent": _safe_property(region, "extent"),
                    "euler_number": int(region.euler_number),
                    "aspect_ratio": major / minor if major is not None and minor else None,
                    "surface_area_um2": surface,
                    "sphericity": (math.pi ** (1.0 / 3.0) * (6.0 * physical_volume) ** (2.0 / 3.0) / surface) if physical_volume is not None and surface else None,
                }
            )
    metadata = {
        "count": voxel_count,
        "centroid": centroid,
        "is_2d": is_2d,
        "region_slice": region.slice,
        "region_mask": region.image,
    }
    return values, metadata


def _intensity_row(object_id: int, channel_id: int, values: np.ndarray) -> tuple:
    samples = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(samples))
    variance = float(np.var(samples))
    stddev = math.sqrt(variance)
    median = float(np.median(samples))
    p10, p25, p75, p90 = (float(value) for value in np.percentile(samples, (10, 25, 75, 90)))
    return (
        object_id, channel_id, int(samples.size), float(np.sum(samples)), mean,
        variance, stddev, float(np.min(samples)), float(np.max(samples)), median,
        float(np.median(np.abs(samples - median))), p10, p25, p75, p90,
        stddev / mean if mean != 0 else None,
    )


def _bounded_plane_sample(
    plane: np.ndarray, maximum_pixels: int = QUALITY_SAMPLE_PIXELS
) -> np.ndarray:
    """Return a deterministic grid sample without copying the level-0 plane."""
    array = np.asarray(plane)
    if array.ndim != 2:
        raise ValueError(f"Image-QC planes must be 2D, got shape {array.shape}")
    stride = max(1, int(math.ceil(math.sqrt(array.size / maximum_pixels))))
    return array[::stride, ::stride]


def _block_means(array: np.ndarray, blocks: int = 4) -> np.ndarray:
    means = []
    for y_part in np.array_split(array, min(blocks, array.shape[0]), axis=0):
        for block in np.array_split(y_part, min(blocks, array.shape[1]), axis=1):
            finite = block[np.isfinite(block)]
            if finite.size:
                means.append(float(np.mean(finite)))
    return np.asarray(means, dtype=np.float64)


def _image_quality_values(plane: np.ndarray) -> dict[str, Any]:
    """Calculate bounded, deterministic QC metrics from a level-0 image plane."""
    sampled = np.asarray(_bounded_plane_sample(plane), dtype=np.float64)
    finite_mask = np.isfinite(sampled)
    finite = sampled[finite_mask]
    if not finite.size:
        return {
            "sample_count": 0,
            "finite_fraction": 0.0,
            **{
                name: None
                for name in IMAGE_QUALITY_COLUMNS[7:22]
            },
        }

    filled = sampled.copy()
    median = float(np.median(finite))
    filled[~finite_mask] = median
    mean = float(np.mean(finite))
    stddev = float(np.std(finite))
    mad = float(np.median(np.abs(finite - median)))
    p01, p99 = (float(value) for value in np.percentile(finite, (1, 99)))
    minimum, maximum = float(np.min(finite)), float(np.max(finite))
    minimal_fraction = float(np.mean(finite == minimum))
    maximal_fraction = float(np.mean(finite == maximum))

    gradient_y, gradient_x = np.gradient(filled)
    gradient_energy = float(np.mean(gradient_x * gradient_x + gradient_y * gradient_y))
    laplacian = (
        -4.0 * filled
        + np.roll(filled, 1, axis=0)
        + np.roll(filled, -1, axis=0)
        + np.roll(filled, 1, axis=1)
        + np.roll(filled, -1, axis=1)
    )
    if filled.shape[0] > 2 and filled.shape[1] > 2:
        laplacian = laplacian[1:-1, 1:-1]
    focus_score = float(np.var(laplacian))

    blocks = _block_means(filled)
    block_mean = float(np.mean(blocks)) if blocks.size else mean
    illumination_block_cv = (
        float(np.std(blocks) / abs(block_mean)) if block_mean != 0 else None
    )
    border_y = max(1, filled.shape[0] // 5)
    border_x = max(1, filled.shape[1] // 5)
    edge_mask = np.ones(filled.shape, dtype=bool)
    if filled.shape[0] > 2 * border_y and filled.shape[1] > 2 * border_x:
        edge_mask[border_y:-border_y, border_x:-border_x] = False
    center_y = max(1, filled.shape[0] // 4)
    center_x = max(1, filled.shape[1] // 4)
    center = filled[center_y:-center_y or None, center_x:-center_x or None]
    edge_mean = float(np.mean(filled[edge_mask]))
    center_mean = float(np.mean(center)) if center.size else mean
    edge_center_ratio = edge_mean / center_mean if center_mean != 0 else None

    fit_stride = max(1, int(math.ceil(max(filled.shape) / 128)))
    fitted = filled[::fit_stride, ::fit_stride]
    yy, xx = np.indices(fitted.shape, dtype=np.float64)
    design = np.column_stack(
        (xx.ravel(), yy.ravel(), np.ones(fitted.size, dtype=np.float64))
    )
    coefficients, *_ = np.linalg.lstsq(design, fitted.ravel(), rcond=None)
    diagonal = math.hypot(max(fitted.shape[1] - 1, 1), max(fitted.shape[0] - 1, 1))
    fitted_gradient = (
        float(math.hypot(coefficients[0], coefficients[1]) * diagonal / abs(mean))
        if mean != 0
        else None
    )

    robust_sigma = 1.4826 * mad
    bright_threshold = median + 6.0 * robust_sigma
    if robust_sigma == 0:
        bright_threshold = (
            median + 0.5 * (p99 - median)
            if p99 > median
            else float("inf")
        )
    bright_mask = finite_mask & (sampled > bright_threshold)
    bright_area_fraction = float(np.mean(bright_mask))
    largest_fraction = 0.0
    if np.any(bright_mask):
        from skimage.measure import label

        components = label(bright_mask, connectivity=1)
        component_counts = np.bincount(components.ravel())[1:]
        if component_counts.size:
            largest_fraction = float(component_counts.max() / sampled.size)

    return {
        "sample_count": int(finite.size),
        "finite_fraction": float(finite.size / sampled.size),
        "focus_score": focus_score,
        "gradient_energy": gradient_energy,
        "intensity_mean": mean,
        "intensity_median": median,
        "intensity_stddev": stddev,
        "intensity_mad": mad,
        "intensity_p01": p01,
        "intensity_p99": p99,
        "minimal_pixel_fraction": minimal_fraction,
        "maximal_pixel_fraction": maximal_fraction,
        "illumination_block_cv": illumination_block_cv,
        "edge_center_ratio": edge_center_ratio,
        "illumination_fitted_gradient": fitted_gradient,
        "bright_area_fraction": bright_area_fraction,
        "largest_bright_component_fraction": largest_fraction,
    }


def _robust_z_scores(values: list[float | None]) -> list[float]:
    available = np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if available.size < 2:
        return [0.0 for _ in values]
    center = float(np.median(available))
    scale = 1.4826 * float(np.median(np.abs(available - center)))
    if scale <= np.finfo(np.float64).eps:
        scale = float(np.std(available))
    if scale <= np.finfo(np.float64).eps:
        scale = 1.0
    return [
        (float(value) - center) / scale
        if value is not None and np.isfinite(value)
        else 0.0
        for value in values
    ]


def _score_image_quality(records: list[dict[str, Any]]) -> None:
    peers: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for record in records:
        peers.setdefault(
            (record["channel_index"], record["timepoint"], record["z_index"]), []
        ).append(record)
    for group in peers.values():
        z_scores = {
            name: _robust_z_scores([record.get(name) for record in group])
            for name in (
                "focus_score",
                "gradient_energy",
                "intensity_median",
                "minimal_pixel_fraction",
                "maximal_pixel_fraction",
                "illumination_block_cv",
                "edge_center_ratio",
                "illumination_fitted_gradient",
                "bright_area_fraction",
                "largest_bright_component_fraction",
            )
        }
        for index, record in enumerate(group):
            focus = max(
                0.0,
                -z_scores["focus_score"][index],
                -z_scores["gradient_energy"][index],
            )
            intensity = abs(z_scores["intensity_median"][index])
            saturation = max(
                0.0,
                z_scores["minimal_pixel_fraction"][index],
                z_scores["maximal_pixel_fraction"][index],
            )
            illumination = max(
                0.0,
                z_scores["illumination_block_cv"][index],
                abs(z_scores["edge_center_ratio"][index]),
                z_scores["illumination_fitted_gradient"][index],
            )
            artifact = max(
                0.0,
                z_scores["bright_area_fraction"][index],
                z_scores["largest_bright_component_fraction"][index],
            )
            score = max(focus, intensity, saturation, illumination, artifact)
            reasons = []
            for component, reason in (
                (focus, "low focus relative to peer fields"),
                (intensity, "unusual intensity relative to peer fields"),
                (saturation, "unusual clipping fraction relative to peer fields"),
                (illumination, "unusual illumination relative to peer fields"),
                (artifact, "unusual bright-area pattern relative to peer fields"),
            ):
                if component >= 3.0:
                    reasons.append(reason)
            if max(
                record["minimal_pixel_fraction"] or 0.0,
                record["maximal_pixel_fraction"] or 0.0,
            ) >= 0.05:
                reasons.append("at least 5% of sampled pixels share an extreme value")
            if (record["largest_bright_component_fraction"] or 0.0) >= 0.20:
                reasons.append("large robust bright-area component")
            record.update(
                {
                    "focus_anomaly_score": focus,
                    "intensity_anomaly_score": intensity,
                    "saturation_anomaly_score": saturation,
                    "illumination_anomaly_score": illumination,
                    "artifact_anomaly_score": artifact,
                    "anomaly_score": score,
                    "review_candidate": bool(score >= 3.0 or reasons),
                    "review_reasons": json.dumps(sorted(set(reasons))),
                }
            )


def _quality_rows(records: list[dict[str, Any]]) -> list[tuple]:
    _score_image_quality(records)
    return [
        tuple(record[column] for column in IMAGE_QUALITY_COLUMNS)
        for record in records
    ]


def _field_quality_rows(
    field_records: list[dict[str, Any]],
    image_quality_records: list[dict[str, Any]],
) -> list[tuple]:
    image_scores: dict[tuple[int, int], float] = {}
    image_candidates: dict[tuple[int, int], bool] = {}
    for quality in image_quality_records:
        key = (quality["image_id"], quality["timepoint"])
        image_scores[key] = max(image_scores.get(key, 0.0), quality["anomaly_score"])
        image_candidates[key] = image_candidates.get(key, False) or bool(
            quality["review_candidate"]
        )
    peers: dict[int, list[dict[str, Any]]] = {}
    for record in field_records:
        peers.setdefault(record["timepoint"], []).append(record)
    next_id = 0
    rows = []
    for group in peers.values():
        count_z = _robust_z_scores([record["cell_count"] for record in group])
        for index, record in enumerate(group):
            next_id += 1
            cell_z = count_z[index]
            zero = record["cell_count"] == 0
            low, high = cell_z <= -3.0, cell_z >= 3.0
            image_score = image_scores.get(
                (record["image_id"], record["timepoint"]), 0.0
            )
            anomaly = max(image_score, abs(cell_z))
            reasons = []
            if zero:
                reasons.append("zero cells")
            elif low:
                reasons.append("unusually low cell count relative to peer fields")
            elif high:
                reasons.append("unusually high cell count relative to peer fields")
            if image_candidates.get((record["image_id"], record["timepoint"]), False):
                reasons.append("one or more image planes are review candidates")
            row = {
                "field_quality_id": next_id,
                **record,
                "cell_count_robust_z": cell_z,
                "zero_cell_count": zero,
                "unusually_low_cell_count": low,
                "unusually_high_cell_count": high,
                "image_anomaly_score": image_score,
                "anomaly_score": anomaly,
                "review_candidate": bool(reasons),
                "review_reasons": json.dumps(reasons),
            }
            rows.append(tuple(row[column] for column in FIELD_QUALITY_COLUMNS))
    return rows


def _relation_name(source_overlap: int, source_count: int, target_count: int) -> str:
    if source_overlap == source_count == target_count:
        return "identical_extent"
    if source_overlap == source_count:
        return "inside"
    if source_overlap == target_count:
        return "contains"
    return "overlaps"


def _centroid_inside(
    centroid: tuple[float, float, float], target: np.ndarray, target_label: int
) -> bool:
    coordinate = tuple(
        min(max(int(round(value)), 0), size - 1)
        for value, size in zip(centroid, target.shape)
    )
    return int(target[coordinate]) == target_label


def _centroid_distance_um(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    scales: dict[str, float],
) -> float | None:
    resolved = [_finite_scale(scales, axis) for axis in ("z", "y", "x")]
    differences = []
    for index, scale in enumerate(resolved):
        if scale is None:
            if index == 0 and first[index] == second[index]:
                continue
            return None
        differences.append((first[index] - second[index]) * scale)
    return float(math.sqrt(sum(value * value for value in differences)))


def _relationship_rows(
    *,
    image_id: int,
    timepoint: int,
    first_labels: np.ndarray,
    second_labels: np.ndarray,
    first_set_id: int,
    second_set_id: int,
    first_objects: dict[int, dict[str, Any]],
    second_objects: dict[int, dict[str, Any]],
    scales: dict[str, float],
    first_relationship_id: int,
) -> tuple[list[tuple], int]:
    overlap = (first_labels > 0) & (second_labels > 0)
    if not np.any(overlap):
        return [], first_relationship_id
    pairs, counts = np.unique(
        np.column_stack((first_labels[overlap], second_labels[overlap])),
        axis=0,
        return_counts=True,
    )
    best_first: dict[int, int] = {}
    best_second: dict[int, int] = {}
    for (first_label, second_label), count in zip(pairs, counts):
        first_label, second_label, count = int(first_label), int(second_label), int(count)
        best_first[first_label] = max(best_first.get(first_label, 0), count)
        best_second[second_label] = max(best_second.get(second_label, 0), count)
    pixel_area = None
    y_um, x_um = _finite_scale(scales, "y"), _finite_scale(scales, "x")
    if y_um is not None and x_um is not None:
        pixel_area = y_um * x_um
    voxel_volume = None
    z_um = _finite_scale(scales, "z")
    if pixel_area is not None and z_um is not None:
        voxel_volume = pixel_area * z_um
    rows = []
    relationship_id = first_relationship_id
    for (first_label, second_label), overlap_count in zip(pairs, counts):
        first_label, second_label, overlap_count = int(first_label), int(second_label), int(overlap_count)
        first = first_objects[first_label]
        second = second_objects[second_label]
        overlap_um2 = overlap_count * pixel_area if first["is_2d"] and second["is_2d"] and pixel_area is not None else None
        overlap_um3 = overlap_count * voxel_volume if not (first["is_2d"] and second["is_2d"]) and voxel_volume is not None else None
        distance = _centroid_distance_um(first["centroid"], second["centroid"], scales)
        for source, target, source_set, target_set, source_label, target_label, source_array, target_array, primary in (
            (first, second, first_set_id, second_set_id, first_label, second_label, first_labels, second_labels, overlap_count == best_first[first_label]),
            (second, first, second_set_id, first_set_id, second_label, first_label, second_labels, first_labels, overlap_count == best_second[second_label]),
        ):
            relationship_id += 1
            rows.append(
                (
                    relationship_id, image_id, timepoint, source["object_id"],
                    target["object_id"], source_set, target_set,
                    _relation_name(overlap_count, source["count"], target["count"]),
                    overlap_count, overlap_um2, overlap_um3,
                    overlap_count / source["count"], overlap_count / target["count"],
                    _centroid_inside(source["centroid"], target_array, target_label),
                    distance, primary,
                )
            )
    return rows, relationship_id


def write_measurements_database(
    results: Iterable[LabelResult],
    output_path: str | Path,
    database_format: str,
    *,
    output_ome_zarr: str | Path,
    output_store_uuid: str | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Measure completed label results and atomically write one relational database."""
    from skimage.measure import regionprops

    result_iterator = iter(results)
    first = next(result_iterator, None)
    if first is None:
        raise ValueError("Cannot measure an empty result collection")
    if database_format not in DATABASE_FORMATS:
        raise ValueError(f"Unsupported measurement database: {database_format}")
    output_store_uuid = output_store_uuid or new_output_store_uuid()
    output_path = Path(output_path)
    temporary = output_path.with_name(output_path.name + ".partial")
    for candidate in (temporary, Path(str(temporary) + ".wal"), Path(str(temporary) + "-journal")):
        candidate.unlink(missing_ok=True)
    started = time.perf_counter()
    writer = None
    counts = {
        "images": 0,
        "label_sets": 0,
        "objects": 0,
        "intensities": 0,
        "image_quality": 0,
        "field_quality": 0,
        "relationships": 0,
    }
    image_quality_records: list[dict[str, Any]] = []
    field_quality_records: list[dict[str, Any]] = []
    next_image_id = next_channel_id = next_label_set_id = 0
    next_object_id = next_relationship_id = 0
    iterator_wait_seconds = 0.0

    def measured_results(first_result):
        nonlocal iterator_wait_seconds
        yield first_result
        while True:
            wait_started = time.perf_counter()
            try:
                result = next(result_iterator)
            except StopIteration:
                iterator_wait_seconds += time.perf_counter() - wait_started
                return
            iterator_wait_seconds += time.perf_counter() - wait_started
            yield result

    try:
        writer = _DatabaseWriter(temporary, database_format)
        settings = first.provenance.get("parameters") or {}
        writer.insert("schema_info", ("key", "value"), [
            ("format", "CI Segmentation measurements"),
            ("schema_version", str(SCHEMA_VERSION)),
            ("coordinate_unit", "micrometer"),
            ("bbox_maximum", "exclusive"),
        ])
        writer.insert(
            "measurement_runs",
            ("run_id", "created_utc", "software_version", "schema_version", "database_format", "source_store", "output_ome_zarr", "output_store_uuid", "settings_json"),
            [(
                1,
                datetime.now(timezone.utc).isoformat(),
                __version__,
                SCHEMA_VERSION,
                database_format,
                first.source.resource.store_path.name,
                str(output_ome_zarr),
                output_store_uuid,
                json.dumps(settings, sort_keys=True),
            )],
        )
        all_results = measured_results(first)
        first = None
        for result in all_results:
            next_image_id += 1
            image_id = next_image_id
            source = result.source
            t, c, z, y, x = source.data.shape
            plate = source.resource.plate_path
            output_resource = "/".join(plate) if plate else ""
            writer.insert(
                "images",
                ("image_id", "run_id", "source_store", "source_resource_path", "output_resource_path", "image_name", "plate_row", "plate_column", "field_index", "size_t", "size_c", "size_z", "size_y", "size_x", "source_dtype", "scale_t", "scale_z_um", "scale_y_um", "scale_x_um"),
                [(
                    image_id, 1, source.resource.store_path.name,
                    source.resource.image_path, output_resource, source.resource.name,
                    plate[0] if plate else None, plate[1] if plate else None,
                    plate[2] if plate else None, t, c, z, y, x, source.source_dtype,
                    _finite_scale(source.scales, "t"), _finite_scale(source.scales, "z"),
                    _finite_scale(source.scales, "y"), _finite_scale(source.scales, "x"),
                )],
            )
            channel_ids = []
            channel_rows = []
            for channel_index, (channel_name, color) in enumerate(_channel_metadata(result), start=1):
                next_channel_id += 1
                channel_ids.append(next_channel_id)
                channel_rows.append((next_channel_id, image_id, channel_index, channel_name, color))
            writer.insert("channels", ("channel_id", "image_id", "channel_index", "channel_name", "channel_color"), channel_rows)
            for timepoint in range(t):
                for channel_offset, channel_id in enumerate(channel_ids):
                    for z_index in range(z):
                        image_quality_id = len(image_quality_records) + 1
                        image_quality_records.append(
                            {
                                "image_quality_id": image_quality_id,
                                "image_id": image_id,
                                "channel_id": channel_id,
                                "channel_index": channel_offset + 1,
                                "timepoint": timepoint,
                                "z_index": z_index,
                                **_image_quality_values(
                                    source.data[timepoint, channel_offset, z_index]
                                ),
                            }
                        )

            label_names = result.channel_labels or [f"labels_{result.target}"]
            label_group_names = _label_group_names(result)
            locations = _location_flags(result)
            label_set_ids = []
            label_set_rows = []
            origins = result.label_origins or ["generated"] * len(label_names)
            for label_index, label_name in enumerate(label_names):
                next_label_set_id += 1
                label_set_ids.append(next_label_set_id)
                prefix = f"{output_resource}/" if output_resource else ""
                label_path = f"{prefix}labels/{label_group_names[label_index]}"
                label_kind = "label-image"
                output_channel = None
                label_set_rows.append((
                    next_label_set_id, image_id, label_index + 1, label_name,
                    origins[label_index], _object_type(label_name),
                    bool(locations[label_index]), label_path, label_kind,
                    output_channel,
                ))
            writer.insert(
                "label_sets",
                (
                    "label_set_id",
                    "image_id",
                    "label_set_index",
                    "label_name",
                    "label_origin",
                    "object_type",
                    "locations_only",
                    "output_label_path",
                    "output_label_kind",
                    "output_channel_index",
                ),
                label_set_rows,
            )
            writer.insert(
                "label_set_sources",
                (
                    "label_set_id",
                    "channel_id",
                    "channel_role",
                    "source_step",
                    "source_model",
                ),
                _label_source_rows(result, label_set_ids, channel_ids),
            )

            for timepoint in range(result.labels.shape[0]):
                arrays = [np.asarray(result.labels[timepoint, index], dtype=np.uint32) for index in range(result.labels.shape[1])]
                object_maps: list[dict[int, dict[str, Any]]] = []
                field_type_counts: dict[str, int] = {}
                for label_index, labels in enumerate(arrays):
                    object_rows = []
                    intensity_rows = []
                    object_map: dict[int, dict[str, Any]] = {}
                    for region in regionprops(labels):
                        next_object_id += 1
                        label_value = int(region.label)
                        shape, metadata = _shape_values(
                            region,
                            locations_only=bool(locations[label_index]),
                            scales=source.scales,
                        )
                        object_type = _object_type(label_names[label_index])
                        field_type_counts[object_type] = (
                            field_type_counts.get(object_type, 0) + 1
                        )
                        prefix = (
                            next_object_id, image_id, label_set_ids[label_index],
                            timepoint, label_value, object_type,
                            bool(locations[label_index]), 2 if metadata["is_2d"] else 3,
                        )
                        object_rows.append(prefix + tuple(shape[name] for name in OBJECT_COLUMNS[8:]))
                        metadata["object_id"] = next_object_id
                        object_map[label_value] = metadata
                        region_mask = metadata["region_mask"]
                        region_slice = metadata["region_slice"]
                        for channel_index, channel_id in enumerate(channel_ids):
                            pixel_values = source.data[timepoint, channel_index][region_slice][region_mask]
                            intensity_rows.append(_intensity_row(next_object_id, channel_id, pixel_values))
                    writer.insert("objects", OBJECT_COLUMNS, object_rows)
                    writer.insert("intensity_measurements", INTENSITY_COLUMNS, intensity_rows)
                    counts["objects"] += len(object_rows)
                    counts["intensities"] += len(intensity_rows)
                    object_maps.append(object_map)

                field_quality_records.append(
                    {
                        "image_id": image_id,
                        "timepoint": timepoint,
                        "cell_count": field_type_counts.get("cells", 0),
                        "nucleus_count": field_type_counts.get("nuclei", 0),
                        "foci_count": sum(
                            field_type_counts.get(name, 0)
                            for name in ("foci", "spots", "bacteria")
                        ),
                        "total_label_count": sum(field_type_counts.values()),
                    }
                )

                for first_index in range(len(arrays)):
                    for second_index in range(first_index + 1, len(arrays)):
                        rows, next_relationship_id = _relationship_rows(
                            image_id=image_id,
                            timepoint=timepoint,
                            first_labels=arrays[first_index],
                            second_labels=arrays[second_index],
                            first_set_id=label_set_ids[first_index],
                            second_set_id=label_set_ids[second_index],
                            first_objects=object_maps[first_index],
                            second_objects=object_maps[second_index],
                            scales=source.scales,
                            first_relationship_id=next_relationship_id,
                        )
                        writer.insert("relationships", RELATIONSHIP_COLUMNS, rows)
                        counts["relationships"] += len(rows)
            counts["images"] += 1
            counts["label_sets"] += len(label_names)
            if log is not None:
                log(
                    f"  measured {source.resource.name}: "
                    f"objects={counts['objects']}, relationships={counts['relationships']}"
                )
        image_quality_rows = _quality_rows(image_quality_records)
        writer.insert(
            "image_quality_measurements",
            IMAGE_QUALITY_COLUMNS,
            image_quality_rows,
        )
        field_quality_rows = _field_quality_rows(
            field_quality_records, image_quality_records
        )
        writer.insert(
            "field_quality_measurements",
            FIELD_QUALITY_COLUMNS,
            field_quality_rows,
        )
        counts["image_quality"] = len(image_quality_rows)
        counts["field_quality"] = len(field_quality_rows)
        writer.finish()
        if output_path.exists():
            output_path.unlink()
        os.replace(temporary, output_path)
    except Exception:
        if writer is not None:
            writer.abort()
        for candidate in (temporary, Path(str(temporary) + ".wal"), Path(str(temporary) + "-journal")):
            candidate.unlink(missing_ok=True)
        raise
    counts["runtime_seconds"] = time.perf_counter() - started - iterator_wait_seconds
    counts["path"] = str(output_path)
    counts["format"] = database_format
    return counts
