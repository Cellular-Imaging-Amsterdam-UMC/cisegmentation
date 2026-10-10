"""Scientific equality through real spawned CPU workers and scratch spill."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from cisegmentation.extension_workers import ExtensionWorkers
from cisegmentation.geometry import mask_polygons
from cisegmentation.ome_zarr_io import ImageResource, read_image, read_native_label
from cisegmentation.resources import ResourceMonitor


def test_initializer_limits_backends_loaded_after_cli_import():
    pytest.importorskip("threadpoolctl")
    # A fresh CLI-style interpreter exposes the import-order regression that
    # pre-importing all numerical modules in pytest would hide.
    code = """
import wrapper
from cisegmentation.extension_workers import _initialize_worker
_initialize_worker()
from cisegmentation.measurement_extensions import colocalization_values
from threadpoolctl import threadpool_info
pools = threadpool_info()
assert pools and all(p['num_threads'] == 1 for p in pools), pools
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_polygons_memory_and_spill_are_identical(tmp_path):
    rng = np.random.default_rng(420)
    labels = (rng.random((1, 31, 35)) > 0.45).astype("u4")
    args = (labels, 1, 0, (0, 0, 31, 35), tmp_path / "edges.sqlite")
    in_memory = mask_polygons(*args, block_size=8, return_bounds=True)
    assert not (tmp_path / "edges.sqlite").exists()
    spilled = mask_polygons(
        *args, block_size=8, return_bounds=True, edge_memory_bytes=0
    )
    assert spilled == in_memory


def test_monitor_reuses_then_refreshes_headroom(monkeypatch):
    from cisegmentation import resources

    times = iter([0.0, 0.1, 0.3])
    samples = []
    monkeypatch.setattr(resources.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        resources, "snapshot", lambda: samples.append(object()) or samples[-1]
    )
    monitor = ResourceMonitor()
    first = monitor.get()
    assert monitor.get() is first
    assert monitor.get() is not first
    assert len(samples) == 2


@pytest.mark.parametrize("locations_only", [False, True])
def test_spawned_object_batches_equal_serial(tmp_path, monkeypatch, locations_only):
    import cisegmentation.parallel_pipeline as pipeline

    monkeypatch.delenv("CISEGMENTATION_INLINE_WORKERS", raising=False)
    monkeypatch.setattr(pipeline, "calculate_cpu_workers", lambda *a, **kw: 2)
    root = zarr.open_group(str(tmp_path / "source.ome.zarr"), mode="w")
    yy, xx = np.indices((40, 40))
    pixels = np.stack([yy + xx + 1, 2 * yy + xx + 1]).astype("u2")[None, :, None]
    root.create_dataset("0", data=pixels, chunks=(1, 1, 1, 16, 16))
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [
                {
                    "name": a,
                    "type": "channel" if a == "c" else "time" if a == "t" else "space",
                }
                for a in "tczyx"
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [{"type": "scale", "scale": [1] * 5}],
                }
            ],
        }
    ]
    label_group = root.create_group("labels/labels_cells")
    masks = np.zeros((1, 1, 1, 40, 40), dtype="u4")
    rows = []
    for i in range(300):
        y, x = divmod(i, 30)
        masks[0, 0, 0, y, x] = i + 1
        rows.append(
            (i + 1, i + 1, 0.0, float(y), float(x), 1, 0, y, x, 1, y + 1, x + 1)
        )
    label_group.create_dataset("0", data=masks, chunks=(1, 1, 1, 16, 16))
    label_group.attrs["multiscales"] = root.attrs["multiscales"]
    resource = ImageResource(tmp_path / "source.ome.zarr")
    image = read_image(resource, lazy=True)
    labels = read_native_label(resource, "labels_cells", lazy=True)
    options = {
        "resource": resource,
        "store": resource.store_path,
        "name": "labels_cells",
        "t": 0,
        "pairs": [(0, 1)],
        "limits": {0: (2.0,), 1: (4.0,)},
        "geometry": True,
        "locations_only": locations_only,
        "spool": tmp_path / "serial.sqlite",
    }
    serial, parallel = ExtensionWorkers(1), ExtensionWorkers(2)
    serial.workers = 1
    try:
        expected = list(serial.objects(image, labels[0, 0], rows, **options))
        actual = list(parallel.objects(image, labels[0, 0], rows, **options))
        assert parallel.used_workers == 2
        assert actual == expected
        assert [row[0] for row in actual] == list(range(1, 301))
    finally:
        serial.close()
        parallel.close()
        image.data.array.store.close()
        labels.array.store.close()


def test_small_frame_does_not_spawn(tmp_path):
    worker = ExtensionWorkers(1)
    try:
        assert (
            list(
                worker.objects(
                    SimpleNamespace(),
                    None,
                    [],
                    resource=None,
                    store=None,
                    name="",
                    t=0,
                    pairs=[],
                    limits={},
                    geometry=False,
                    locations_only=False,
                    spool=tmp_path / "edges",
                )
            )
            == []
        )
        assert worker.executor is None
    finally:
        worker.close()
