"""Disk tracker equivalence and preservation of outputs after optional failure."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from cisegmentation.tracking import link_observations
from cisegmentation.tracking_store import TrackingStore


@pytest.mark.parametrize("object_type", ["spots", "cells"])
@pytest.mark.parametrize("seed", range(8))
def test_disk_tracking_preserves_lap_results(tmp_path, object_type, seed):
    rng = np.random.default_rng(seed)
    obs = []
    for t in [0, 1, 2, 4, 5]:
        for _ in range(9):
            obs.append(
                dict(
                    id=len(obs) * 3 + 11,
                    t=t,
                    position=rng.uniform(0, 8, 3),
                    size=int(rng.integers(5, 20)),
                    source="raster_centroid",
                )
            )
    options = dict(radius=5, max_gap=2, divisions=True, object_type=object_type)
    expected_tracks, expected_assignments, expected_links = link_observations(
        obs, seconds_per_frame=0.5, **options
    )
    store = TrackingStore(tmp_path / "tracking.sqlite")
    try:
        for p in reversed(obs):
            store.add(p)
        store.build(**options)
        tracks = list(store.tracks(0.5))
        assert len(tracks) == len(expected_tracks)
        for actual, expected in zip(tracks, expected_tracks, strict=True):
            assert actual == pytest.approx(expected)
        assert [(r[0], r[1]) for r in store.observations()] == expected_assignments
        for actual, expected in zip(store.links(0.5), expected_links, strict=True):
            assert actual[:4] == expected[:4]
            assert actual[4:] == pytest.approx(expected[4:])
    finally:
        store.close()


def test_long_series_spools_history_under_small_live_budget(tmp_path, monkeypatch):
    import cisegmentation.tracking_store as module

    monkeypatch.setattr(
        module, "snapshot", lambda **kw: SimpleNamespace(ram_available=1024 * 1024)
    )
    store = TrackingStore(tmp_path / "tracking.sqlite")
    try:
        for t in range(600):
            for x in range(3):
                store.add(
                    dict(
                        id=t * 3 + x + 1,
                        t=t,
                        position=(0, 0, x * 100),
                        size=1,
                        source="raster_centroid",
                    )
                )
                assert len(store.pending) < 512
        # The former all-observations guard fails at this count and budget.
        assert store.count * 2048 > 1024 * 1024 // 4
        store.build(radius=2)
        assert store.track_count == 3
        assert [r[5] for r in store.tracks()] == [600, 600, 600]
    finally:
        store.close()


def test_disk_division_and_lineage_output(tmp_path):
    store = TrackingStore(tmp_path / "division.sqlite")
    try:
        for oid, t, x, size in [(1, 0, 5, 20), (2, 1, 4, 10), (3, 1, 6, 10)]:
            store.add(
                dict(
                    id=oid, t=t, position=(0, 0, x), size=size, source="raster_centroid"
                )
            )
        store.build(divisions=True, object_type="cells")
        assert list(store.divisions()) == [(1, 2, 3, 1)]
        assert [row[:3] for row in store.tracks()] == [
            (1, 1, None),
            (2, 1, 1),
            (3, 1, 1),
        ]
    finally:
        store.close()


@pytest.mark.parametrize("database_format", ["sqlite", "duckdb"])
def test_tracking_geometry_and_native_coordinates_across_object_batches(
    tmp_path, monkeypatch, database_format
):
    from test_measurement_extensions import fake_spots, source_store
    import cisegmentation.parallel_pipeline as pipeline
    from cisegmentation.engine import run_workflow
    from cisegmentation.measurement_extensions import connect_database
    from cisegmentation.settings import SegmentationSettings

    monkeypatch.setenv("CISEGMENTATION_INLINE_WORKERS", "1")
    monkeypatch.setattr(pipeline, "segment_czyx", fake_spots)
    source = source_store(tmp_path / "many.ome.zarr", shape=(2, 2, 1, 96, 128))
    data = zarr.open_group(str(source.store_path), mode="a")["0"]
    data[:] = 0
    data[:, 0, 0, :80:4, :120:4] = 100  # 600 objects per frame, >512 batch size.
    settings = SegmentationSettings(
        cell_model="skip",
        nucleus_model="skip",
        foci_model_1="spotiflow:general",
        measurements_database=database_format,
        tracking=True,
        tracking_distance_um=0.1,
        export_geometry=True,
        max_inference_workers=1,
        max_measurement_workers=1,
    )
    outputs = run_workflow(source.store_path, tmp_path / "output", settings)
    db = connect_database(
        next(p for p in outputs if "_measurements." in p.name),
        database_format,
        read_only=True,
    )
    try:
        for table, count in [
            ("objects", 1200),
            ("point_localizations", 1200),
            ("track_observations", 1200),
            ("tracks", 600),
            ("temporal_links", 600),
        ]:
            assert db.execute(f"SELECT count(*) FROM {table}").fetchone() == (count,)
        assert db.execute("SELECT status FROM tracking_status").fetchone() == (
            "complete",
        )
        assert db.execute(
            "SELECT min(y_px-centroid_y_px),min(x_px-centroid_x_px) FROM point_localizations JOIN objects USING(object_id)"
        ).fetchone() == pytest.approx((0.25, 0.375))
    finally:
        db.close()
    geo = connect_database(
        next(p for p in outputs if "_geometry." in p.name),
        database_format,
        read_only=True,
    )
    assert geo.execute("SELECT count(*) FROM geometries").fetchone() == (1200,)
    geo.close()


@pytest.mark.parametrize("database_format", ["sqlite", "duckdb"])
@pytest.mark.parametrize("failure_phase", ["collect", "build", "write", "nonmemory"])
def test_tracking_failure_preserves_image_measurements_and_other_labels(
    tmp_path, monkeypatch, database_format, failure_phase
):
    from test_measurement_extensions import fake_spots, source_store
    import cisegmentation.parallel_pipeline as pipeline
    from cisegmentation.engine import run_workflow
    from cisegmentation.measurement_extensions import Batches, connect_database
    from cisegmentation.settings import SegmentationSettings

    monkeypatch.setenv("CISEGMENTATION_INLINE_WORKERS", "1")
    monkeypatch.setattr(pipeline, "segment_czyx", fake_spots)
    source = source_store(tmp_path / "source.ome.zarr")
    settings = SegmentationSettings(
        cell_model="skip",
        nucleus_model="skip",
        foci_model_1="spotiflow:general",
        foci_model_2="spotiflow:fluo_live",
        measurements_database=database_format,
        tracking=True,
        export_geometry=True,
        max_inference_workers=1,
        max_measurement_workers=1,
    )
    messages = []
    original_add, original_build, original_write = (
        TrackingStore.add,
        TrackingStore.build,
        Batches.add,
    )
    failed = False

    def fail_once():
        nonlocal failed
        if not failed:
            failed = True
            if failure_phase == "nonmemory":
                raise RuntimeError("simulated tracker problem")
            raise MemoryError("Tracking observations exceed live RAM budget")

    def add(self, p):
        fail_once()
        return original_add(self, p)

    def build(self, **kwargs):
        fail_once()
        return original_build(self, **kwargs)

    def write(self, table, row):
        if table == "tracks" and not failed:
            original_write(self, table, row)
            self.flush()  # Ensure actual database writes also roll back.
            fail_once()
        return original_write(self, table, row)

    if failure_phase == "collect":
        monkeypatch.setattr(TrackingStore, "add", add)
    elif failure_phase in {"build", "nonmemory"}:
        monkeypatch.setattr(TrackingStore, "build", build)
    else:
        monkeypatch.setattr(Batches, "add", write)
    outputs = run_workflow(
        source.store_path, tmp_path / "output", settings, log=messages.append
    )
    assert failed
    output_store = next(p for p in outputs if p.name.endswith(".ome.zarr"))
    root = zarr.open_group(str(output_store), mode="r")
    assert len(list(root["labels"].group_keys())) == 2
    db = connect_database(
        next(p for p in outputs if "_measurements." in p.name),
        database_format,
        read_only=True,
    )
    try:
        assert db.execute("SELECT count(*) FROM objects").fetchone() == (12,)
        statuses = db.execute(
            "SELECT status,error_type,details_json FROM tracking_status ORDER BY label_set_id"
        ).fetchall()
        assert [r[0] for r in statuses] == ["failed", "complete"]
        assert statuses[0][1] == (
            "RuntimeError" if failure_phase == "nonmemory" else "MemoryError"
        )
        assert json.loads(statuses[0][2])["failure_kind"] == (
            "tracking_error" if failure_phase == "nonmemory" else "memory"
        )
        assert db.execute("SELECT count(*) FROM tracks").fetchone() == (2,)
        assert db.execute("SELECT count(*) FROM track_observations").fetchone() == (6,)
        assert db.execute("SELECT count(*) FROM tracking_parameters").fetchone() == (1,)
    finally:
        db.close()
    geo = connect_database(
        next(p for p in outputs if "_geometry." in p.name),
        database_format,
        read_only=True,
    )
    assert geo.execute("SELECT count(*) FROM geometries").fetchone() == (12,)
    geo.close()
    assert '"status": "failed"' in json.dumps(dict(root.attrs))
    assert any(
        "Tracking FAILED" in m and "Segmentation and measurements are retained" in m
        for m in messages
    )
    if failure_phase != "nonmemory":
        assert any("Tracking FAILED due to memory" in m for m in messages)
