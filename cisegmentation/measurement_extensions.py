"""Opt-in schema-v5 extensions, computed after final field ID assignment.

All legacy tables/views are immutable. Spatial distances use validated NGFF or
OME-XML calibration; geometry uses pixel-centre coordinates and retains transforms.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections import defaultdict
from contextlib import suppress
from itertools import combinations, product
from pathlib import Path
from uuid import UUID, uuid5

import numpy as np
from scipy.spatial import cKDTree

from .calibration import LENGTH_FACTORS_UM, effective_transform
from .ome_zarr_io import read_image, read_native_label
from .resources import MIB, ResourceMonitor, snapshot

EXTENSION_VERSION = 1
_SCHEMA = """
CREATE TABLE measurement_extensions (name TEXT PRIMARY KEY, version INTEGER, settings_json TEXT);
CREATE TABLE object_keys (object_id BIGINT PRIMARY KEY, object_uuid TEXT UNIQUE);
CREATE TABLE coordinate_frames (image_id BIGINT PRIMARY KEY, spatial_unit TEXT, scale_zyx_json TEXT, translation_zyx_json TEXT, seconds_per_frame DOUBLE, time_unit TEXT, metadata_json TEXT);
CREATE TABLE point_localizations (object_id BIGINT PRIMARY KEY, z_px DOUBLE, y_px DOUBLE, x_px DOUBLE, coordinate_source TEXT);
CREATE TABLE spatial_measurements (object_id BIGINT PRIMARY KEY, nearest_object_id BIGINT, nearest_distance_um DOUBLE, neighbor_count BIGINT, radius_um DOUBLE, touching_neighbor_count BIGINT, shared_boundary DOUBLE, boundary_unit TEXT);
CREATE TABLE object_contacts (source_object_id BIGINT, target_object_id BIGINT, shared_boundary DOUBLE, boundary_unit TEXT, PRIMARY KEY(source_object_id,target_object_id));
CREATE TABLE spot_counts (object_id BIGINT, spot_label_set_id BIGINT, spot_count BIGINT, PRIMARY KEY(object_id,spot_label_set_id));
CREATE TABLE colocalization_thresholds (image_id BIGINT, timepoint INTEGER, channel_id BIGINT, threshold DOUBLE, sample_count BIGINT, method TEXT, sampling_json TEXT, PRIMARY KEY(image_id,timepoint,channel_id));
CREATE TABLE colocalization_measurements (object_id BIGINT, channel_a_id BIGINT, channel_b_id BIGINT, pearson_r DOUBLE, manders_a_in_b DOUBLE, manders_b_in_a DOUBLE, sample_count BIGINT, reason TEXT, PRIMARY KEY(object_id,channel_a_id,channel_b_id));
CREATE TABLE tracking_parameters (label_set_id BIGINT PRIMARY KEY, distance_um DOUBLE, maximum_missed_frames INTEGER, divisions_enabled BOOLEAN, algorithm TEXT);
CREATE TABLE tracks (track_id BIGINT PRIMARY KEY, image_id BIGINT, label_set_id BIGINT, lineage_id BIGINT, parent_track_id BIGINT, start_frame INTEGER, end_frame INTEGER, observation_count BIGINT, duration DOUBLE, path_length_um DOUBLE, displacement_um DOUBLE, mean_speed DOUBLE, maximum_speed DOUBLE, directionality DOUBLE, time_unit TEXT, speed_unit TEXT);
CREATE TABLE track_observations (object_id BIGINT PRIMARY KEY, track_id BIGINT, z_um DOUBLE, y_um DOUBLE, x_um DOUBLE, coordinate_source TEXT);
CREATE TABLE temporal_links (source_object_id BIGINT, target_object_id BIGINT, kind TEXT, frame_difference INTEGER, elapsed DOUBLE, distance_um DOUBLE, speed DOUBLE, time_unit TEXT, PRIMARY KEY(source_object_id,target_object_id));
CREATE TABLE cell_divisions (parent_object_id BIGINT PRIMARY KEY, daughter_a_object_id BIGINT, daughter_b_object_id BIGINT, parent_track_id BIGINT);
"""
_GEOMETRY_SCHEMA = """
CREATE TABLE geometry_info (key TEXT PRIMARY KEY,value TEXT);
CREATE TABLE coordinate_frames (image_id BIGINT PRIMARY KEY, spatial_unit TEXT, scale_zyx_json TEXT, translation_zyx_json TEXT, seconds_per_frame DOUBLE, time_unit TEXT, metadata_json TEXT);
CREATE TABLE geometries (geometry_id BIGINT PRIMARY KEY, object_id BIGINT, object_uuid TEXT, output_store_uuid TEXT, image_id BIGINT, resource_path TEXT, label_set_id BIGINT, label_name TEXT, label_value BIGINT, timepoint INTEGER, z_index INTEGER, geometry_type TEXT, geometry_wkb BLOB, coordinate_unit TEXT, min_x_px DOUBLE, min_y_px DOUBLE, max_x_px DOUBLE, max_y_px DOUBLE, x_px DOUBLE, y_px DOUBLE, z_px DOUBLE, x_um DOUBLE, y_um DOUBLE, z_um DOUBLE, mask_area_px2 DOUBLE, coordinate_source TEXT);
CREATE INDEX idx_geometry_object ON geometries(object_id,z_index);
CREATE INDEX idx_geometry_location ON geometries(image_id,timepoint,min_x_px,min_y_px);
"""


def connect_database(path, database_format, *, read_only=False):
    if database_format == "sqlite":
        return (
            sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
            if read_only
            else sqlite3.connect(path)
        )
    import duckdb

    db = duckdb.connect(str(path), read_only=read_only)
    from .resources import configure_database

    configure_database(db, memory_cap_mib=1024, memory_divisor=8)
    return db


class Batches:
    def __init__(self, db, database_format):
        self.db, self.format, self.pending = db, database_format, defaultdict(list)
        self.counts = defaultdict(int)

    def add(self, table, row):
        self.pending[table].append(row)
        if len(self.pending[table]) >= 512:
            self.flush(table)

    def flush(self, table=None):
        for name in [table] if table else list(self.pending):
            rows = self.pending[name]
            if not rows:
                continue
            if self.format == "duckdb":
                import pandas as pd

                frame = pd.DataFrame.from_records(rows)
                self.db.register("_extension_batch", frame)
                try:
                    self.db.execute(
                        f"INSERT INTO {name} SELECT * FROM _extension_batch"
                    )
                finally:
                    self.db.unregister("_extension_batch")
            else:
                self.db.executemany(
                    f"INSERT INTO {name} VALUES ({','.join('?' for _ in rows[0])})",
                    rows,
                )
            self.counts[name] += len(rows)
            rows.clear()


def calibration(image):
    multiscale = (image.attrs.get("multiscales") or [{}])[0]
    axes = multiscale.get("axes") or []
    names = [a.get("name") if isinstance(a, dict) else a for a in axes]
    units = {a["name"]: a.get("unit") for a in axes if isinstance(a, dict)}
    factors = LENGTH_FACTORS_UM
    time_factors = {
        "second": 1.0,
        "millisecond": 0.001,
        "microsecond": 1e-6,
        "nanosecond": 1e-9,
        "minute": 60.0,
        "hour": 3600.0,
        "day": 86400.0,
    }
    scale, offset = effective_transform(multiscale, names or image.axes)
    needed = ["y", "x"] + (["z"] if image.data.shape[2] > 1 else [])
    valid = all(
        units.get(a) in factors
        and math.isfinite(scale.get(a, 0))
        and scale.get(a, 0) > 0
        and math.isfinite(offset.get(a, 0))
        for a in needed
    )
    spatial_scale = np.asarray(
        [scale.get(a, 1) * factors.get(units.get(a), 1) for a in "zyx"]
    )
    translation = np.asarray(
        [offset.get(a, 0) * factors.get(units.get(a), 1) for a in "zyx"]
    )
    if image.data.shape[2] == 1 and units.get("z") not in factors:
        spatial_scale[0], translation[0] = 1.0, 0.0
    dt = scale.get("t", 0) * time_factors.get(units.get("t"), 0)
    seconds = dt if math.isfinite(dt) and dt > 0 else None
    return valid, spatial_scale, translation, seconds, multiscale


def require_spatial_calibration(image):
    result = calibration(image)
    if not result[0]:
        needed = "X/Y/Z" if image.data.shape[2] > 1 else "X/Y"
        resource = image.resource
        raise ValueError(
            f"{resource.store_path}/{resource.image_path}: spatial/tracking measurements "
            f"require positive calibrated {needed} pixel sizes with physical units; "
            "no usable calibration was found in NGFF metadata or embedded OME-XML "
            "(OME/METADATA.ome.xml)"
        )
    return result


def channel_pairs(selection, count):
    if selection.strip().lower() == "all":
        return list(combinations(range(count), 2))
    pairs = set()
    for item in selection.split(","):
        try:
            a, b = map(int, item.strip().split(":"))
        except ValueError as exc:
            raise ValueError(
                "Colocalization pairs must be 'all' or one-based pairs such as '1:2,1:3'"
            ) from exc
        if not 1 <= a <= count or not 1 <= b <= count or a == b:
            raise ValueError(
                "Colocalization channels must be distinct and present in the input"
            )
        pairs.add(tuple(sorted((a - 1, b - 1))))
    return sorted(pairs)


def thresholds(image, t, channels):
    from skimage.filters import threshold_otsu

    shape = image.data.shape[2:]
    active = sum(size > 1 for size in shape) or 1
    step = max(1, math.ceil((math.prod(shape) / 1_048_576) ** (1 / active)))
    while math.prod(math.ceil(size / step) for size in shape) > 1_048_576:
        step += 1
    strides = [step if size > 1 else 1 for size in shape]
    key = tuple(slice(0, size, stride) for size, stride in zip(shape, strides))
    result = {}
    for c in channels:
        sample = np.asarray(image.data[(t, c, *key)], dtype=np.float64).ravel()
        sample = sample[np.isfinite(sample)]
        value = (
            None
            if not len(sample)
            else float(sample[0])
            if sample.min() == sample.max()
            else float(threshold_otsu(sample, nbins=4096))
        )
        result[c] = (
            value,
            len(sample),
            json.dumps(
                {
                    "stride_zyx": strides,
                    "maximum_pixels": 1_048_576,
                    "histogram_bins": 4096,
                }
            ),
        )
    return result


def object_blocks(bbox, shape, monitor=None):
    z0, y0, x0, z1, y1, x1 = map(int, bbox)
    monitor = monitor or ResourceMonitor()
    budget = max(1, monitor.get().ram_available // 64)
    step = max(16, min(512, int(math.sqrt(budget / 128))))
    for z, y, x in product(range(z0, z1), range(y0, y1, step), range(x0, x1, step)):
        if monitor.get().ram_available < MIB:
            raise MemoryError("Insufficient live RAM for optional object measurements")
        yield (
            slice(z, z + 1),
            slice(y, min(y1, y + step)),
            slice(x, min(x1, x + step)),
        )


def colocalization_values(image, labels, t, label, bbox, pair, limits, *, monitor=None):
    sums = np.zeros(7, dtype=np.float64)
    n, negative = 0, False
    a, b = pair
    ta, tb = limits[a][0], limits[b][0]
    for key in object_blocks(bbox, labels.shape, monitor):
        mask = np.asarray(labels[key]) == label
        aa = np.asarray(image.data[(t, a, *key)], dtype=np.float64)[mask]
        bb = np.asarray(image.data[(t, b, *key)], dtype=np.float64)[mask]
        finite = np.isfinite(aa) & np.isfinite(bb)
        aa, bb = aa[finite], bb[finite]
        negative |= bool(np.any(aa < 0) or np.any(bb < 0))
        n += len(aa)
        sums += [
            aa.sum(),
            bb.sum(),
            np.dot(aa, aa),
            np.dot(bb, bb),
            np.dot(aa, bb),
            aa[bb > tb].sum() if tb is not None else 0,
            bb[aa > ta].sum() if ta is not None else 0,
        ]
    sa, sb, saa, sbb, sab, mab, mba = sums
    reasons, pearson = [], None
    if n >= 2:
        va, vb = max(0, saa - sa * sa / n), max(0, sbb - sb * sb / n)
        if va > max(1, saa) * 1e-14 and vb > max(1, sbb) * 1e-14:
            pearson = float(np.clip((sab - sa * sb / n) / math.sqrt(va * vb), -1, 1))
        else:
            reasons.append("constant_channel")
    else:
        reasons.append("insufficient_finite_pixels")
    ma = float(mab / sa) if not negative and sa > 0 and tb is not None else None
    mb = float(mba / sb) if not negative and sb > 0 and ta is not None else None
    if negative:
        reasons.append("negative_intensities_manders_undefined")
    if sa <= 0 or sb <= 0:
        reasons.append("zero_signal")
    return pearson, ma, mb, n, ";".join(reasons) or None


def contacts(array, scales, dimensions):
    """Count shared faces once, including faces crossing storage chunks."""
    result = defaultdict(float)
    from .streaming import blocks

    axes = [1, 2] if dimensions == 2 else [0, 1, 2]
    for key in blocks(array.shape):
        base = np.asarray(array[key])
        for axis in axes:
            limit = min(key[axis].stop, array.shape[axis] - 1)
            length = limit - key[axis].start
            if length <= 0:
                continue
            left_key = [slice(None)] * 3
            left_key[axis] = slice(0, length)
            shifted = list(key)
            shifted[axis] = slice(key[axis].start + 1, limit + 1)
            a, b = base[tuple(left_key)], np.asarray(array[tuple(shifted)])
            mask = (a > 0) & (b > 0) & (a != b)
            if not mask.any():
                continue
            pairs, counts = np.unique(
                np.sort(np.column_stack((a[mask], b[mask])), axis=1),
                axis=0,
                return_counts=True,
            )
            face = float(np.prod([scales[i] for i in axes if i != axis]))
            for pair, count in zip(pairs, counts):
                result[tuple(map(int, pair))] += int(count) * face
            if len(result) * 200 > snapshot().ram_available // 8:
                raise MemoryError("Contact graph exceeds live RAM budget")
    return result


def spatial_rows(objects, array, scales, radius, dimensions):
    shared = contacts(array, scales, dimensions)
    ids = {p["label"]: p["id"] for p in objects}
    touching, lengths = defaultdict(set), defaultdict(float)
    contact_rows = []
    unit = "micrometer" if dimensions == 2 else "micrometer_squared"
    for (a, b), length in sorted(shared.items()):
        if a not in ids or b not in ids:
            continue
        touching[a].add(b)
        touching[b].add(a)
        lengths[a] += length
        lengths[b] += length
        contact_rows.append((ids[a], ids[b], length, unit))
    if not objects:
        return [], contact_rows
    positions = np.asarray([p["position"] for p in objects])
    tree = cKDTree(positions)
    counts = tree.query_ball_point(positions, radius, return_length=True) - 1
    distances, indexes = tree.query(positions, k=2, workers=1)
    rows = []
    for i, p in enumerate(objects):
        candidate = 0 if int(indexes[i, 0]) != i else 1
        nearest = int(indexes[i, candidate])
        rows.append(
            (
                p["id"],
                objects[nearest]["id"] if nearest < len(objects) else None,
                float(distances[i, 1]) if nearest < len(objects) else None,
                int(counts[i]),
                radius,
                len(touching[p["label"]]),
                lengths[p["label"]],
                unit,
            )
        )
    return rows, contact_rows


def write_extensions(
    database_path,
    database_format,
    resources,
    label_store,
    settings,
    *,
    stage_dir,
    finalization=None,
    log=None,
):
    """Write optional features to staged measurements and a staged geometry DB."""
    if not settings.measurement_extensions_enabled():
        return {"enabled": False}, None
    from .extension_workers import ExtensionWorkers
    from .geometry import point_wkb
    from .tracking import link_observations

    finalization = finalization or {}
    stage_dir = Path(stage_dir)
    started = time.perf_counter()
    db = connect_database(database_path, database_format)
    geo = None
    geometry_path = stage_dir / (
        "geometry.duckdb" if database_format == "duckdb" else "geometry.sqlite"
    )
    geometry_id, track_offset, lineage_offset = 0, 0, 0
    workers = ExtensionWorkers(settings.max_measurement_workers)
    try:
        from .resources import log_resource_check, snapshot

        log_resource_check(
            log,
            "extension_sizing",
            workers=workers.workers,
            effective=snapshot(include_gpu=False).to_dict(),
            requested_cap=settings.max_measurement_workers,
            estimated_worker_rss_mib=512,
            blas_threads_per_worker=1,
        )
        db.execute("BEGIN TRANSACTION")
        for statement in _SCHEMA.split(";"):
            if statement.strip():
                db.execute(statement)
        sink = Batches(db, database_format)
        store_uuid = str(
            db.execute(
                "SELECT output_store_uuid FROM measurement_runs WHERE run_id=1"
            ).fetchone()[0]
        )
        features = {
            "spatial": settings.spatial_measurements,
            "colocalization": settings.colocalization,
            "tracking": settings.tracking,
            "geometry": settings.export_geometry,
        }
        for name, enabled in features.items():
            if enabled:
                sink.add(
                    "measurement_extensions",
                    (
                        name,
                        EXTENSION_VERSION,
                        json.dumps(settings.to_dict(), sort_keys=True),
                    ),
                )
        if settings.export_geometry:
            geometry_path.unlink(missing_ok=True)
            geo = connect_database(geometry_path, database_format)
            geo.execute("BEGIN TRANSACTION")
            for statement in _GEOMETRY_SCHEMA.split(";"):
                if statement.strip():
                    geo.execute(statement)
            geo_sink = Batches(geo, database_format)
            for key, value in {
                "format": "CISegment geometry",
                "geometry_schema_version": "1",
                "measurement_base_schema_version": "5",
                "output_store_uuid": store_uuid,
                "pixel_coordinates": "integer pixel centres; half-integer pixel edges",
                "encoding": "OGC/ISO WKB; XYZ uses type 1001",
            }.items():
                geo_sink.add("geometry_info", (key, value))
        image_rows = db.execute(
            "SELECT image_id,source_resource_path FROM images ORDER BY image_id"
        ).fetchall()
        resource_map = {r.image_path: r for r in resources}
        for image_id, resource_path in image_rows:
            resource = resource_map[resource_path]
            image = read_image(resource, lazy=True)
            from .streaming import ChunkCache

            image.data.cache = ChunkCache(
                min(64 * MIB, max(MIB, snapshot().ram_available // 32))
            )
            valid, scales, translation, seconds, metadata = (
                require_spatial_calibration(image)
                if settings.spatial_measurements or settings.tracking
                else calibration(image)
            )
            time_unit = "second" if seconds else "frame"
            frame_row = (
                image_id,
                "micrometer" if valid else "pixel",
                json.dumps(scales.tolist()),
                json.dumps(translation.tolist()),
                seconds,
                time_unit,
                json.dumps(metadata, sort_keys=True),
            )
            sink.add("coordinate_frames", frame_row)
            if geo is not None:
                geo_sink.add("coordinate_frames", frame_row)
            channels = db.execute(
                "SELECT channel_id FROM channels WHERE image_id=? ORDER BY channel_index",
                (image_id,),
            ).fetchall()
            pairs = (
                channel_pairs(settings.colocalization_pairs, len(channels))
                if settings.colocalization
                else []
            )
            limits_by_t = {}
            if pairs:
                for t in range(image.data.shape[0]):
                    limits_by_t[t] = thresholds(
                        image, t, sorted({c for p in pairs for c in p})
                    )
                    for c, (value, count, sampling) in limits_by_t[t].items():
                        sink.add(
                            "colocalization_thresholds",
                            (
                                image_id,
                                t,
                                channels[c][0],
                                value,
                                count,
                                "otsu_4096",
                                sampling,
                            ),
                        )
            label_sets = db.execute(
                "SELECT label_set_id,output_label_path,object_type,locations_only FROM label_sets WHERE image_id=? ORDER BY label_set_id",
                (image_id,),
            ).fetchall()
            points_path = Path(
                finalization.get(resource_path, {}).get(
                    "point_localizations", stage_dir / "missing"
                )
            )
            point_db = sqlite3.connect(points_path) if points_path.is_file() else None
            try:
                for label_set_id, label_path, object_type, locations_only in label_sets:
                    name = label_path.split("/")[-1]
                    store = (
                        label_store
                        if (Path(label_store) / label_path / ".zattrs").exists()
                        else resource.store_path
                    )
                    label_array = read_native_label(
                        resource, name, store_path=store, lazy=True
                    )
                    label_array.cache = ChunkCache(
                        min(64 * MIB, max(MIB, snapshot().ram_available // 32))
                    )
                    observations = []
                    for t in range(image.data.shape[0]):
                        count = db.execute(
                            "SELECT count(*) FROM objects WHERE label_set_id=? AND timepoint=?",
                            (label_set_id, t),
                        ).fetchone()[0]
                        if count * 2048 > snapshot().ram_available // 4:
                            raise MemoryError(
                                "Optional measurements object batch exceeds live RAM budget"
                            )
                        data = db.execute(
                            "SELECT object_id,label_value,centroid_z_px,centroid_y_px,centroid_x_px,voxel_count,bbox_min_z_px,bbox_min_y_px,bbox_min_x_px,bbox_max_z_px,bbox_max_y_px,bbox_max_x_px FROM objects WHERE label_set_id=? AND timepoint=? ORDER BY object_id",
                            (label_set_id, t),
                        ).fetchall()
                        native = (
                            {
                                int(v): (float(z), float(y), float(x))
                                for v, z, y, x in point_db.execute(
                                    "SELECT label,z,y,x FROM points WHERE label_name=? AND t=?",
                                    (name, t),
                                )
                            }
                            if point_db
                            else {}
                        )
                        objects = []
                        plane = label_array[t, 0]
                        computed = workers.objects(
                            image,
                            plane,
                            data,
                            resource=resource,
                            store=store,
                            name=name,
                            t=t,
                            pairs=pairs,
                            limits=limits_by_t.get(t, {}),
                            geometry=geo is not None,
                            locations_only=locations_only,
                            spool=stage_dir / "polygon-edges.sqlite",
                        )
                        for row, (computed_id, coloc, polygons) in zip(
                            data, computed, strict=True
                        ):
                            oid, value, z, y, x, size, *bbox = row
                            if computed_id != oid:
                                raise RuntimeError(
                                    "Optional measurement worker changed object order"
                                )
                            pixel = np.asarray(
                                native.get(int(value), (z or 0, y, x)), dtype=float
                            )
                            position = pixel * scales + translation
                            point_source = (
                                "spotiflow_subpixel"
                                if int(value) in native
                                else "raster_centroid"
                            )
                            p = {
                                "id": int(oid),
                                "label": int(value),
                                "t": t,
                                "position": position,
                                "pixel": pixel,
                                "size": int(size or 0),
                                "bbox": bbox,
                                "source": point_source,
                            }
                            objects.append(p)
                            object_uuid = str(
                                uuid5(
                                    UUID(store_uuid),
                                    f"{resource_path}:{label_path}:{t}:{value}",
                                )
                            )
                            sink.add("object_keys", (oid, object_uuid))
                            if int(value) in native:
                                sink.add(
                                    "point_localizations", (oid, *pixel, point_source)
                                )
                            if settings.colocalization:
                                if locations_only:
                                    # Native point labels have NULL legacy boxes.
                                    anchor = (int(z or 0), int(y), int(x))
                                    bbox = [*anchor, *(v + 1 for v in anchor)]
                                for pair, values in zip(pairs, coloc, strict=True):
                                    sink.add(
                                        "colocalization_measurements",
                                        (
                                            oid,
                                            channels[pair[0]][0],
                                            channels[pair[1]][0],
                                            *values,
                                        ),
                                    )
                            if geo is not None:
                                physical = position if valid else [None, None, None]
                                common = (
                                    oid,
                                    object_uuid,
                                    store_uuid,
                                    image_id,
                                    resource_path,
                                    label_set_id,
                                    name,
                                    value,
                                    t,
                                )
                                if locations_only:
                                    geometry_id += 1
                                    zz, yy, xx = map(float, pixel)
                                    geo_sink.add(
                                        "geometries",
                                        (
                                            geometry_id,
                                            *common,
                                            None,
                                            "POINT Z"
                                            if image.data.shape[2] > 1
                                            else "POINT",
                                            point_wkb(
                                                xx,
                                                yy,
                                                zz if image.data.shape[2] > 1 else None,
                                            ),
                                            "pixel",
                                            xx,
                                            yy,
                                            xx,
                                            yy,
                                            xx,
                                            yy,
                                            zz,
                                            physical[2],
                                            physical[1],
                                            physical[0],
                                            None,
                                            point_source,
                                        ),
                                    )
                                else:
                                    for zz, result in polygons:
                                        wkb, kind, area, bounds = result
                                        geometry_id += 1
                                        geo_sink.add(
                                            "geometries",
                                            (
                                                geometry_id,
                                                *common,
                                                zz,
                                                kind,
                                                wkb,
                                                "pixel",
                                                *bounds,
                                                float(pixel[2]),
                                                float(pixel[1]),
                                                float(pixel[0]),
                                                physical[2],
                                                physical[1],
                                                physical[0],
                                                area,
                                                "final_mask",
                                            ),
                                        )
                        if settings.spatial_measurements:
                            rows, contact_rows = spatial_rows(
                                objects,
                                plane,
                                scales,
                                settings.spatial_radius_um,
                                2 if image.data.shape[2] == 1 else 3,
                            )
                            for row in rows:
                                sink.add("spatial_measurements", row)
                            for row in contact_rows:
                                sink.add("object_contacts", row)
                        if settings.tracking:
                            observations.extend(objects)
                            if len(observations) * 2048 > snapshot().ram_available // 4:
                                raise MemoryError(
                                    "Tracking observations exceed live RAM budget"
                                )
                    if settings.tracking:
                        divide = settings.tracking_divisions and object_type in {
                            "cells",
                            "nuclei",
                        }
                        sink.add(
                            "tracking_parameters",
                            (
                                label_set_id,
                                settings.tracking_distance_um,
                                settings.tracking_max_gap,
                                divide,
                                "sparse_two_stage_lap_v1",
                            ),
                        )
                        tracks, assignments, links = link_observations(
                            observations,
                            radius=settings.tracking_distance_um,
                            max_gap=settings.tracking_max_gap,
                            divisions=divide,
                            object_type=object_type,
                            seconds_per_frame=seconds,
                        )
                        assignments = dict(assignments)
                        for track, lineage, parent, *values in tracks:
                            sink.add(
                                "tracks",
                                (
                                    track + track_offset,
                                    image_id,
                                    label_set_id,
                                    lineage + lineage_offset,
                                    parent + track_offset if parent else None,
                                    *values,
                                    time_unit,
                                    f"micrometer_per_{time_unit}",
                                ),
                            )
                        for p in observations:
                            sink.add(
                                "track_observations",
                                (
                                    p["id"],
                                    assignments[p["id"]] + track_offset,
                                    *p["position"],
                                    p["source"],
                                ),
                            )
                        daughters = defaultdict(list)
                        for a, b, kind, frames, elapsed, distance, speed in links:
                            sink.add(
                                "temporal_links",
                                (
                                    a,
                                    b,
                                    kind,
                                    frames,
                                    elapsed,
                                    distance,
                                    speed,
                                    time_unit,
                                ),
                            )
                            if kind == "division":
                                daughters[a].append(b)
                        for parent, children in daughters.items():
                            sink.add(
                                "cell_divisions",
                                (
                                    parent,
                                    *sorted(children),
                                    assignments[parent] + track_offset,
                                ),
                            )
                        track_offset += len(tracks)
                        lineage_offset += len(tracks)
                    if log:
                        log(
                            f"Optional measurements: {resource_path or resource.name}/{name} completed"
                        )
            finally:
                if point_db:
                    point_db.close()
                image.data.array.store.close()
        if settings.spatial_measurements:
            # Count existing mask/point assignments without inventing parents.
            db.execute("""INSERT INTO spot_counts SELECT p.object_id,s.label_set_id,count(r.source_object_id)
                FROM objects p JOIN label_sets s ON s.image_id=p.image_id
                AND s.object_type IN ('spots','foci','bacteria')
                LEFT JOIN relationships r ON r.target_object_id=p.object_id
                AND r.source_label_set_id=s.label_set_id AND r.is_primary_for_source
                WHERE p.object_type IN ('cells','nuclei')
                GROUP BY p.object_id,s.label_set_id""")
            sink.counts["spot_counts"] = db.execute(
                "SELECT count(*) FROM spot_counts"
            ).fetchone()[0]
        sink.flush()
        db.execute("COMMIT")
        if geo is not None:
            geo_sink.flush()
            geo.execute("COMMIT")
        summary = {
            "enabled": True,
            "extension_version": EXTENSION_VERSION,
            "rows": dict(sink.counts),
            "geometry_rows": geometry_id,
            "cpu_workers": workers.used_workers,
            "runtime_seconds": time.perf_counter() - started,
        }
        if log:
            log(
                f"Optional measurements complete: {json.dumps(summary, sort_keys=True)}"
            )
        return summary, geometry_path if geo is not None else None
    except Exception:
        errors = (sqlite3.Error,)
        if database_format == "duckdb":
            import duckdb

            errors += (duckdb.Error,)
        with suppress(*errors):
            db.execute("ROLLBACK")
        raise
    finally:
        workers.close()
        db.close()
        if geo is not None:
            geo.close()
