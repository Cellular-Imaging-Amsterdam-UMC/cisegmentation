"""Scientific and compatibility contracts for opt-in extension schema 1."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from cisegmentation.geometry import mask_polygons, point_wkb
from cisegmentation.measurement_extensions import (
    calibration,
    channel_pairs,
    colocalization_values,
    connect_database,
    contacts,
    spatial_rows,
    thresholds,
)
from cisegmentation.settings import SegmentationSettings
from cisegmentation.tracking import link_observations, sparse_assignment


def observation(oid, t, x, y=0, size=10):
    return {"id": oid, "t": t, "position": np.array([0.0, y, x]), "size": size}


@pytest.mark.parametrize(
    "object_type", ["cells", "nuclei", "spots", "foci", "bacteria", "cytoplasm"]
)
def test_binary_division_restricted_to_cells_and_nuclei(object_type):
    obs = [observation(1, 0, 5, size=20), observation(2, 1, 4), observation(3, 1, 6)]
    tracks, assignments, links = link_observations(
        obs, divisions=True, object_type=object_type
    )
    if object_type in {"cells", "nuclei"}:
        assert len(tracks) == 3
        assert [l[2] for l in links] == ["division", "division"]
        assert tracks[1][2] == tracks[2][2] == 1
        assert tracks[0][1] == tracks[1][1] == tracks[2][1]
    else:
        assert len(tracks) == 2
        assert len(links) == 1 and links[0][2] == "link"
        assert all(t[2] is None for t in tracks)
    assert len(dict(assignments)) == 3


def test_gaps_speed_and_calibrated_motion():
    obs = [observation(1, 0, 0), observation(2, 1, 3), observation(3, 4, 9)]
    tracks, assignments, links = link_observations(
        obs, max_gap=2, seconds_per_frame=0.5
    )
    assert len(tracks) == 1 and len(set(dict(assignments).values())) == 1
    assert tracks[0][6:] == pytest.approx((2.0, 9.0, 9.0, 4.5, 6.0, 1.0))
    assert links[1][2] == "gap" and links[1][3] == 3
    assert len(link_observations(obs, max_gap=1)[0]) == 2


def test_crossing_trajectories_use_observed_velocity():
    obs = [
        observation(1, 0, 0),
        observation(2, 0, 10),
        observation(3, 1, 4),
        observation(4, 1, 6),
        observation(5, 2, 8),
        observation(6, 2, 2),
    ]
    tracks, assignments, links = link_observations(obs, radius=5)
    assigned = dict(assignments)
    assert assigned[1] == assigned[3] == assigned[5]
    assert assigned[2] == assigned[4] == assigned[6]
    assert len(tracks) == 2
    assert all(l[2] == "link" for l in links)


def test_no_merges_and_sparse_gate():
    obs = [observation(1, 0, 0), observation(2, 0, 2), observation(3, 1, 1)]
    assert len(link_observations(obs, divisions=True, object_type="cells")[2]) == 1
    assert sparse_assignment([obs[0]], [observation(4, 1, 100)], 20) == []


@pytest.mark.parametrize("seed", range(10))
def test_polygons_reconstruct_masks_with_holes_and_disconnected_components(
    tmp_path, seed
):
    from shapely import from_wkb
    from shapely.geometry import Point

    rng = np.random.default_rng(seed)
    mask = rng.random((1, 17, 19)) > 0.65
    mask[0, 3:11, 3:11] = True
    mask[0, 5:9, 5:9] = False
    value, _kind, area = mask_polygons(
        mask.astype("u4"), 1, 0, (0, 0, 17, 19), tmp_path / "edges.sqlite", block_size=4
    )
    polygon = from_wkb(value)
    assert polygon.is_valid
    assert polygon.area == area == mask.sum()
    for y in range(17):
        for x in range(19):
            assert polygon.covers(Point(x, y)) == bool(mask[0, y, x])


def test_point_wkb_preserves_subpixel_xyz():
    from shapely import from_wkb

    p = from_wkb(point_wkb(2.25, 3.125, 1.875))
    assert list(p.coords) == [(2.25, 3.125, 1.875)]


def test_contacts_include_chunk_boundaries_and_anisotropic_faces(tmp_path):
    labels = zarr.open(
        str(tmp_path / "labels.zarr"),
        mode="w",
        shape=(1, 4, 1026),
        chunks=(1, 2, 512),
        dtype="u4",
    )
    labels[:, :, 0:512] = 1
    labels[:, :, 512:] = 2
    assert contacts(labels, [1.0, 2.0, 3.0], 2) == {(1, 2): 8.0}
    objects = [
        dict(observation(11, 0, 0), label=1),
        dict(observation(12, 0, 0), label=2),
    ]
    rows, _ = spatial_rows(objects, labels, [1.0, 2.0, 3.0], 5, 2)
    assert rows[0][1] == 12 and rows[1][1] == 11
    assert rows[0][2:] == (0.0, 1, 5, 1, 8.0, "micrometer")


def test_colocalization_and_undefined_edge_cases():
    data = np.array([[[[[0.0, 1.0, 2.0, 3.0]]], [[[0.0, 2.0, 4.0, 6.0]]]]])
    image = SimpleNamespace(data=data)
    labels = np.ones((1, 1, 4), dtype="u4")
    values = colocalization_values(
        image, labels, 0, 1, [0, 0, 0, 1, 1, 4], (0, 1), {0: (1, 4, ""), 1: (2, 4, "")}
    )
    assert values[:3] == pytest.approx((1, 5 / 6, 5 / 6))
    assert values[3:] == (4, None)
    data[0, 1] = 0
    values = colocalization_values(
        image, labels, 0, 1, [0, 0, 0, 1, 1, 4], (0, 1), {0: (1, 4, ""), 1: (0, 4, "")}
    )
    assert values[0] is None and values[2] is None
    assert "zero_signal" in values[-1]
    data[0, 0, 0, 0, 0] = -1
    assert (
        colocalization_values(
            image,
            labels,
            0,
            1,
            [0, 0, 0, 1, 1, 4],
            (0, 1),
            {0: (1, 4, ""), 1: (0, 4, "")},
        )[1]
        is None
    )


def test_40k_threshold_sampling_is_bounded(tmp_path):
    from cisegmentation.ome_zarr_io import read_image

    resource = source_store(
        tmp_path / "large.ome.zarr", shape=(1, 2, 1, 40000, 40000), fill=False
    )
    image = read_image(resource, lazy=True)
    values = thresholds(image, 0, [0])
    assert values[0][1] <= 1_048_576
    assert json.loads(values[0][2])["stride_zyx"][0] == 1


def source_store(path, shape=(3, 2, 1, 96, 128), fill=True, time_units=True):
    from cisegmentation.ome_zarr_io import ImageResource

    root = zarr.open_group(str(path), mode="w")
    data = root.create_dataset("0", shape=shape, chunks=(1, 1, 1, 64, 64), dtype="u2")
    axes = [
        {
            "name": a,
            "type": "channel" if a == "c" else "time" if a == "t" else "space",
            **(
                {"unit": "micrometer"}
                if a in "zyx"
                else {"unit": "millisecond"}
                if a == "t" and time_units
                else {}
            ),
        }
        for a in "tczyx"
    ]
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": axes,
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [500, 1, 2, 0.5, 0.25]}
                    ],
                }
            ],
        }
    ]
    if fill:
        for t in range(shape[0]):
            for z in range(shape[2]):
                data[t, 0, z, 20 + 2 * t, 30 + 3 * t] = 100
                data[t, 1, z, 20 + 2 * t, 30 + 3 * t] = 200
                data[t, 0, z, 65 + t, 80 + 2 * t] = 100
                data[t, 1, z, 65 + t, 80 + 2 * t] = 200
    return ImageResource(path)


def fake_spots(data, spec, settings, scales, **kwargs):
    coordinates = np.argwhere(np.asarray(data[0]) > 0)
    labels = np.zeros(data.shape[1:], dtype="u4")
    native = []
    for value, (z, y, x) in enumerate(coordinates, 1):
        labels[z, y, x] = value
        native.append(
            (value, z + 0.125 if labels.shape[0] > 1 else 0, y + 0.25, x + 0.375)
        )
    info = {
        "device": "cpu",
        "runtime_seconds": 0.01,
        "timings": {"inference_seconds": 0.01},
        "effective_parameters": {},
        "model_cache_hits": 0,
        "model_cache_misses": 1,
    }
    if settings.measurement_extensions_enabled():
        info["_native_points"] = np.asarray(native).reshape(-1, 4)
    return labels, info


@pytest.mark.parametrize("database_format", ["sqlite", "duckdb"])
@pytest.mark.parametrize("z_size", [1, 2])
@pytest.mark.parametrize("streaming", ["auto", "on"])
def test_spot_only_full_workflow_native_points_geometry_and_tracking(
    tmp_path, monkeypatch, database_format, z_size, streaming
):
    from shapely import from_wkb

    import cisegmentation.parallel_pipeline as pipeline
    from cisegmentation.engine import run_workflow

    source = source_store(tmp_path / "source.ome.zarr", shape=(3, 2, z_size, 96, 128))
    monkeypatch.setenv("CISEGMENTATION_INLINE_WORKERS", "1")
    monkeypatch.setattr(pipeline, "segment_czyx", fake_spots)
    settings = SegmentationSettings(
        cell_model="skip",
        nucleus_model="skip",
        foci_model_1="spotiflow:general",
        measurements_database=database_format,
        spatial_measurements=True,
        colocalization=True,
        tracking=True,
        tracking_divisions=True,
        export_geometry=True,
        streaming_mode=streaming,
        tile_size=64,
        tile_overlap=24,
        max_inference_workers=1,
        max_measurement_workers=1,
    )
    outputs = run_workflow(source.store_path, tmp_path / "output", settings)
    db_path = next(p for p in outputs if "_measurements." in p.name)
    geometry_path = next(p for p in outputs if "_geometry." in p.name)
    db = connect_database(db_path, database_format, read_only=True)
    assert db.execute(
        "SELECT value FROM schema_info WHERE key='schema_version'"
    ).fetchone() == ("5",)
    assert db.execute("SELECT count(*) FROM objects").fetchone() == (6 * z_size,)
    assert db.execute("SELECT count(*) FROM spot_counts").fetchone() == (0,)
    assert db.execute("SELECT count(*) FROM cell_divisions").fetchone() == (0,)
    assert db.execute("SELECT count(*) FROM tracks").fetchone() == (2 * z_size,)
    assert db.execute("SELECT DISTINCT time_unit FROM tracks").fetchall() == [
        ("second",)
    ]
    assert db.execute(
        "SELECT min(y_px-centroid_y_px),min(x_px-centroid_x_px) FROM point_localizations JOIN objects USING(object_id)"
    ).fetchone() == pytest.approx((0.25, 0.375))
    assert db.execute("SELECT count(*) FROM object_navigation").fetchone() == (
        6 * z_size,
    )
    store_uuid = db.execute(
        "SELECT output_store_uuid FROM measurement_runs"
    ).fetchone()[0]
    db.close()
    geo = connect_database(geometry_path, database_format, read_only=True)
    assert geo.execute("SELECT count(*) FROM geometries").fetchone() == (6 * z_size,)
    assert geo.execute(
        "SELECT DISTINCT output_store_uuid FROM geometries"
    ).fetchall() == [(store_uuid,)]
    row = geo.execute(
        "SELECT geometry_wkb,x_px,y_px,z_px FROM geometries ORDER BY geometry_id LIMIT 1"
    ).fetchone()
    point = from_wkb(bytes(row[0]))
    assert next(iter(point.coords))[:2] == (row[1], row[2])
    if z_size > 1:
        assert point.z == row[3]
    geo.close()
    root = zarr.open_group(str(outputs[0]), mode="r")
    assert root["0"].shape == source_store_shape(source)


def source_store_shape(resource):
    return zarr.open_group(str(resource.store_path), mode="r")["0"].shape


def test_units_are_validated_and_uncalibrated_time_is_frames(tmp_path):
    from cisegmentation.ome_zarr_io import read_image

    resource = source_store(tmp_path / "source.ome.zarr", time_units=False)
    assert calibration(read_image(resource, lazy=True))[3] is None
    root = zarr.open_group(str(resource.store_path), mode="a")
    metadata = root.attrs["multiscales"]
    metadata[0]["axes"][-1].pop("unit")
    root.attrs["multiscales"] = metadata
    assert calibration(read_image(resource, lazy=True))[0] is False


@pytest.mark.parametrize(
    "name", ["spatial_measurements", "colocalization", "tracking", "export_geometry"]
)
def test_extensions_require_database_and_default_off(name):
    assert not SegmentationSettings().measurement_extensions_enabled()
    with pytest.raises(ValueError, match="require DuckDB or SQLite"):
        SegmentationSettings(
            measurements_database="skip", **{name: True}
        ).validate_steps()


def test_pairs_are_one_based_and_have_no_manual_threshold():
    assert channel_pairs("1:3,2:1", 3) == [(0, 1), (0, 2)]
    with pytest.raises(ValueError):
        channel_pairs("1:4", 3)


@pytest.mark.parametrize("database_format", ["sqlite", "duckdb"])
def test_mask_geometry_spot_zero_counts_and_legacy_measurements_unchanged(
    tmp_path, monkeypatch, database_format
):
    from shapely import from_wkb

    import cisegmentation.parallel_pipeline as pipeline
    from cisegmentation.engine import run_workflow

    source = source_store(tmp_path / "source.ome.zarr", shape=(1, 2, 1, 96, 128))

    def segment(data, spec, settings, scales, **kwargs):
        if spec.family == "spotiflow":
            return fake_spots(data, spec, settings, scales)
        labels = np.zeros(data.shape[1:], dtype="u4")
        labels[:, 5:40, 10:40] = 1
        labels[:, 5:40, 40:60] = 2
        labels[:, 10:13, 13:17] = 0
        labels[:, 50:53, 40:43] = 1
        return labels, {
            "device": "cpu",
            "runtime_seconds": 0.01,
            "timings": {},
            "effective_parameters": {},
        }

    monkeypatch.setenv("CISEGMENTATION_INLINE_WORKERS", "1")
    monkeypatch.setattr(pipeline, "segment_czyx", segment)
    base = {
        "cell_model": "cellpose3:cyto3",
        "foci_model_1": "spotiflow:general",
        "remove_border_cells": False,
        "measurements_database": database_format,
        "max_inference_workers": 1,
        "max_measurement_workers": 1,
    }
    old = run_workflow(
        source.store_path, tmp_path / "old", SegmentationSettings(**base)
    )
    new = run_workflow(
        source.store_path,
        tmp_path / "new",
        SegmentationSettings(
            **base,
            spatial_measurements=True,
            colocalization=True,
            tracking=True,
            export_geometry=True,
        ),
    )
    legacy = connect_database(
        next(p for p in old if "_measurements." in p.name),
        database_format,
        read_only=True,
    )
    extended = connect_database(
        next(p for p in new if "_measurements." in p.name),
        database_format,
        read_only=True,
    )
    for table, ordering in [
        ("objects", "object_id"),
        ("intensity_measurements", "object_id,channel_id"),
        ("relationships", "relationship_id"),
    ]:
        assert (
            legacy.execute(f"SELECT * FROM {table} ORDER BY {ordering}").fetchall()
            == extended.execute(f"SELECT * FROM {table} ORDER BY {ordering}").fetchall()
        )
    assert extended.execute(
        "SELECT spot_count FROM spot_counts ORDER BY object_id"
    ).fetchall() == [(1,), (0,)]
    assert extended.execute(
        "SELECT shared_boundary FROM object_contacts"
    ).fetchall() == [(17.5,)]
    assert extended.execute("SELECT count(*) FROM cell_divisions").fetchone() == (0,)
    assert (
        "measurement_extensions"
        not in [
            r[0]
            for r in legacy.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        if database_format == "sqlite"
        else legacy.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name='measurement_extensions'"
        ).fetchone()
        == (0,)
    )
    legacy.close()
    extended.close()
    geo = connect_database(
        next(p for p in new if "_geometry." in p.name), database_format, read_only=True
    )
    row = geo.execute(
        "SELECT geometry_wkb,mask_area_px2,min_x_px,max_x_px FROM geometries WHERE label_name='labels_cells' ORDER BY object_id LIMIT 1"
    ).fetchone()
    polygon = from_wkb(bytes(row[0]))
    assert polygon.is_valid and polygon.area == row[1] == 1047
    assert row[2:] == (9.5, 42.5)
    geo.close()
