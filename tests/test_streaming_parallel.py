import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pytest
import zarr
from test_streaming import _segment, _source


@pytest.mark.parametrize("model", ["cellpose3:nuclei", "spotiflow:synth_3d"])
def test_parallel_volume_preserves_z_order_and_instance_ids(
    tmp_path, monkeypatch, model
):
    resource = _source(tmp_path / "source.zarr", shape=(1, 1, 4, 192, 256))
    # Deliberately different foreground in adjacent planes exercises Z overlap
    # as well as XY overlap without relying on a learned model in a unit test.
    source = zarr.open_group(str(resource.store_path), mode="a")["0"]
    source[0, 0, 0, 20:80, 30:100] = 100
    source[0, 0, 1:3, 40:120, 50:140] = 100
    source[0, 0, 3, 90:150, 150:210] = 100
    image = read_image(resource, lazy=True)
    spec = get_model_spec(model)
    settings = SegmentationSettings(
        tile_size=64, tile_overlap=24, tile_depth=2, device="cpu"
    )
    infer_streamed(
        image, spec, settings, tmp_path / "serial.zarr", _segment, allow_parallel=False
    )
    _thread_backend(monkeypatch)
    infos, _ = infer_streamed(
        image, spec, settings, tmp_path / "parallel.zarr", _segment
    )
    np.testing.assert_array_equal(
        raw_array(tmp_path / "serial.zarr")[:], raw_array(tmp_path / "parallel.zarr")[:]
    )
    assert infos[0]["streaming"]["peak_workers"] == 3


@pytest.mark.parametrize("field_count", [1, 2])
def test_field_scheduler_enables_tile_parallelism_only_for_a_single_field(
    tmp_path, monkeypatch, field_count
):
    from cisegmentation.ome_zarr_io import ImageResource

    resources = [
        ImageResource(tmp_path / "source.zarr", f"field{number}")
        for number in range(field_count)
    ]
    payloads_seen = []
    monkeypatch.setattr(pipeline, "_largest_resource", lambda resources: resources[0])

    def execute(payloads, workers, **kwargs):
        payloads_seen.extend(payloads)
        return [
            {
                "ok": True,
                "resource_path": payload["resource"].image_path,
                "records": {"request_01": [{}]},
                "runtime_seconds": 0.1,
                "zarr_read_seconds": 0.01,
                "peak_cuda_mb": 0,
                "rss_mb": 100,
                "device": "cpu",
            }
            for payload in payloads
        ]

    monkeypatch.setattr(pipeline, "_execute_tasks", execute)
    pipeline.run_inference_passes(
        resources, SegmentationSettings(cell_model="cellpose3:cyto3"), tmp_path
    )
    assert len(payloads_seen) == field_count
    assert payloads_seen[0]["allow_tile_parallel"] is (field_count == 1)
    assert all(payload["allow_tile_parallel"] is False for payload in payloads_seen[1:])


from cisegmentation import parallel_pipeline as pipeline
from cisegmentation import streaming_tiles as tiles
from cisegmentation.ome_zarr_io import read_image
from cisegmentation.registry import get_model_spec
from cisegmentation.resources import GIB, MIB, ResourceSnapshot
from cisegmentation.settings import SegmentationSettings
from cisegmentation.streaming import infer_streamed, raw_array


def _thread_backend(monkeypatch, *, oom=False, fatal=False):
    lock = threading.Lock()
    state = {"active": 0, "peak": 0, "completions": [], "pools": []}
    monkeypatch.setattr(tiles, "_can_spawn", lambda segment: True)
    current = ResourceSnapshot(4, 16 * GIB, 14 * GIB, 12 * GIB, 10 * GIB)
    monkeypatch.setattr(tiles, "snapshot", lambda: current)
    monkeypatch.setattr(pipeline, "snapshot", lambda: current)
    monkeypatch.setattr(pipeline, "allocated_cpus", lambda: 4)
    # Simulate a four-CPU allocation independently of CI runner affinity and
    # physical-core count. Allocation discovery has its own resource tests.
    monkeypatch.setattr(
        pipeline,
        "available_cpu_workers",
        lambda task_count, cap=0: min(task_count, cap or 3),
    )

    def pool(workers):
        state["pools"].append(workers)
        return ThreadPoolExecutor(max_workers=workers)

    def task(payload):
        number = int(Path(payload["input"]).name.split(".")[0])
        with lock:
            state["active"] += 1
            active = state["active"]
            state["peak"] = max(state["peak"], active)
        try:
            time.sleep(0.04 if number % 3 == 2 else 0.008)
            if fatal and number > 1:
                raise ValueError("invalid model configuration")
            if oom and active > 1:
                return {"ok": False, "kind": "memory", "error": "out of memory"}
            data = np.load(payload["input"], allow_pickle=False)
            predicted, info = payload["segment"](
                data,
                payload["spec"],
                payload["settings"],
                payload["scales"],
                bounded=True,
            )
            np.save(payload["output"], predicted, allow_pickle=False)
            state["completions"].append(number)
            return {
                "ok": True,
                "info": info,
                "rss_mb": 512.0,
                "cuda_mb": 0.0,
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
            }
        finally:
            with lock:
                state["active"] -= 1

    monkeypatch.setattr(tiles, "_pool", pool)
    monkeypatch.setattr(tiles, "_tile_task", task)
    return state


@pytest.mark.parametrize(
    "model",
    [
        "cellpose3:nuclei",
        "cellpose-sam:cpsam_v2",
        "stardist:SD_Nuclei_Versatile",
        "instanseg:single_channel_nuclei",
        "spotiflow:general",
    ],
)
def test_parallel_tiles_match_serial_labels_despite_out_of_order_completion(
    tmp_path, monkeypatch, model
):
    image = read_image(_source(tmp_path / "source.zarr"), lazy=True)
    spec = get_model_spec(model)
    serial_settings = SegmentationSettings(
        tile_size=64, tile_overlap=24, device="cpu", max_inference_workers=1
    )
    infer_streamed(image, spec, serial_settings, tmp_path / "serial.zarr", _segment)
    state = _thread_backend(monkeypatch)
    settings = SegmentationSettings(tile_size=64, tile_overlap=24, device="cpu")
    infos, _ = infer_streamed(
        image, spec, settings, tmp_path / "parallel.zarr", _segment
    )
    np.testing.assert_array_equal(
        raw_array(tmp_path / "serial.zarr")[:], raw_array(tmp_path / "parallel.zarr")[:]
    )
    assert state["peak"] == 3
    assert state["completions"] != sorted(state["completions"])
    stats = infos[0]["streaming"]
    assert stats["workers"] == stats["peak_workers"] == 3
    assert stats["tiles"] == 12
    assert not list(tmp_path.glob(".ciseg-tiles-*"))


def test_parallel_memory_error_reduces_workers_and_retries_uncommitted_tiles(
    tmp_path, monkeypatch
):
    image = read_image(_source(tmp_path / "source.zarr"), lazy=True)
    spec = get_model_spec("cellpose3:nuclei")
    infer_streamed(
        image,
        spec,
        SegmentationSettings(
            tile_size=64, tile_overlap=24, device="cpu", max_inference_workers=1
        ),
        tmp_path / "serial.zarr",
        _segment,
    )
    state = _thread_backend(monkeypatch, oom=True)
    infos, _ = infer_streamed(
        image,
        spec,
        SegmentationSettings(tile_size=64, tile_overlap=24, device="cpu"),
        tmp_path / "parallel.zarr",
        _segment,
    )
    np.testing.assert_array_equal(
        raw_array(tmp_path / "serial.zarr")[:], raw_array(tmp_path / "parallel.zarr")[:]
    )
    stats = infos[0]["streaming"]
    assert stats["initial_workers"] == 3 and stats["workers"] == 1
    assert stats["concurrency_reductions"] == stats["worker_restarts"] == 1
    assert stats["oom_retries"] == 1
    assert stats["tiles"] == 12
    assert state["pools"] == [1, 3, 1]
    assert not list(tmp_path.glob(".ciseg-tiles-*"))


def test_parallel_nonmemory_failure_propagates_and_cleans_worker_files(
    tmp_path, monkeypatch
):
    image = read_image(_source(tmp_path / "source.zarr"), lazy=True)
    _thread_backend(monkeypatch, fatal=True)
    with pytest.raises(ValueError, match="invalid model configuration"):
        infer_streamed(
            image,
            get_model_spec("cellpose3:nuclei"),
            SegmentationSettings(tile_size=64, device="cpu"),
            tmp_path / "raw.zarr",
            _segment,
        )
    assert not list(tmp_path.glob(".ciseg-tiles-*"))


def test_tile_worker_sizing_uses_gpu_ram_cpu_limits_and_user_cap(monkeypatch):
    monkeypatch.setattr(pipeline, "allocated_cpus", lambda: 4)
    monkeypatch.setattr(
        pipeline,
        "available_cpu_workers",
        lambda task_count, cap=0: min(task_count, cap or 3),
    )
    current = ResourceSnapshot(4, 16 * GIB, 14 * GIB, 12 * GIB, 10 * GIB)
    monkeypatch.setattr(pipeline, "snapshot", lambda: current)
    monkeypatch.setattr(tiles, "snapshot", lambda: current)
    profile = {"info": {"device": "cuda"}, "rss_mb": GIB / MIB, "cuda_mb": GIB / MIB}
    assert tiles._worker_count(20, profile, SegmentationSettings()) == 3
    assert (
        tiles._worker_count(20, profile, SegmentationSettings(max_inference_workers=2))
        == 2
    )
    profile["cuda_mb"] = 6 * GIB / MIB
    assert tiles._worker_count(20, profile, SegmentationSettings()) == 1
    profile["cuda_mb"] = GIB / MIB
    current = ResourceSnapshot(4, 16 * GIB, GIB, 12 * GIB, 10 * GIB)
    assert tiles._worker_count(20, profile, SegmentationSettings()) == 1


def test_live_pressure_waits_for_active_tiles_without_changing_their_geometry(
    tmp_path, monkeypatch
):
    image = read_image(_source(tmp_path / "source.zarr"), lazy=True)
    state = _thread_backend(monkeypatch)

    def resources():
        return ResourceSnapshot(
            4, 16 * GIB, 3 * MIB if state["active"] >= 2 else 14 * GIB
        )

    monkeypatch.setattr(tiles, "snapshot", resources)
    infos, _ = infer_streamed(
        image,
        get_model_spec("cellpose3:nuclei"),
        SegmentationSettings(tile_size=64, tile_overlap=24, device="cpu"),
        tmp_path / "raw.zarr",
        _segment,
    )
    stats = infos[0]["streaming"]
    assert stats["memory_waits"] > 0
    assert stats["pressure_splits"] == 0
    assert stats["tiles"] == 12
    np.testing.assert_array_equal(
        raw_array(tmp_path / "raw.zarr")[0] > 0, np.asarray(image.data)[0, 0] > 0
    )


def test_lost_worker_reduces_pool_and_recovers_uncommitted_cores(tmp_path, monkeypatch):
    image = read_image(_source(tmp_path / "source.zarr"), lazy=True)
    _thread_backend(monkeypatch)
    original_task = tiles._tile_task
    state = {"failed": False}

    def task(payload):
        if Path(payload["input"]).name == "2.input.npy" and not state["failed"]:
            state["failed"] = True
            raise BrokenProcessPool("worker killed during allocation")
        return original_task(payload)

    monkeypatch.setattr(tiles, "_tile_task", task)
    infos, _ = infer_streamed(
        image,
        get_model_spec("cellpose3:nuclei"),
        SegmentationSettings(tile_size=64, tile_overlap=24, device="cpu"),
        tmp_path / "raw.zarr",
        _segment,
    )
    assert infos[0]["streaming"]["concurrency_reductions"] == 1
    assert infos[0]["streaming"]["workers"] == 1
    np.testing.assert_array_equal(
        raw_array(tmp_path / "raw.zarr")[0] > 0, np.asarray(image.data)[0, 0] > 0
    )
    assert not list(tmp_path.glob(".ciseg-tiles-*"))


def test_parent_spool_allocation_error_reduces_workers_and_recovers(
    tmp_path, monkeypatch
):
    image = read_image(_source(tmp_path / "source.zarr"), lazy=True)
    _thread_backend(monkeypatch)
    original_save = np.save

    def save(path, *args, **kwargs):
        if Path(path).name == "2.input.npy":
            raise MemoryError("cannot allocate tile decode buffer")
        return original_save(path, *args, **kwargs)

    monkeypatch.setattr(np, "save", save)
    infos, _ = infer_streamed(
        image,
        get_model_spec("cellpose3:nuclei"),
        SegmentationSettings(tile_size=64, tile_overlap=24, device="cpu"),
        tmp_path / "raw.zarr",
        _segment,
    )
    assert infos[0]["streaming"]["workers"] == 1
    assert infos[0]["streaming"]["concurrency_reductions"] == 1
    np.testing.assert_array_equal(
        raw_array(tmp_path / "raw.zarr")[0] > 0, np.asarray(image.data)[0, 0] > 0
    )


@pytest.mark.parametrize("kind", ["memory", "size"])
def test_failed_probe_is_split_and_reused_without_missing_foreground(
    tmp_path, monkeypatch, kind
):
    image = read_image(_source(tmp_path / "source.zarr"), lazy=True)
    _thread_backend(monkeypatch)
    original_task = tiles._tile_task

    def task(payload):
        if Path(payload["input"]).name == "1.input.npy":
            return {
                "ok": False,
                "kind": kind,
                "error": "out of memory"
                if kind == "memory"
                else "quantile() input tensor is too large",
            }
        return original_task(payload)

    monkeypatch.setattr(tiles, "_tile_task", task)
    infos, _ = infer_streamed(
        image,
        get_model_spec("cellpose3:nuclei"),
        SegmentationSettings(tile_size=128, tile_overlap=8, device="cpu"),
        tmp_path / "raw.zarr",
        _segment,
    )
    stats = infos[0]["streaming"]
    assert stats["oom_retries" if kind == "memory" else "size_limit_retries"] == 1
    assert stats["worker_restarts"] == 1
    assert stats["tiles"] == 5
    np.testing.assert_array_equal(
        raw_array(tmp_path / "raw.zarr")[0] > 0, np.asarray(image.data)[0, 0] > 0
    )
    assert infos[0]["object_count"] == 3


def test_plate_worker_budget_can_disable_nested_tile_pools(tmp_path, monkeypatch):
    image = read_image(_source(tmp_path / "source.zarr"), lazy=True)
    monkeypatch.setattr(tiles, "_can_spawn", lambda segment: True)
    monkeypatch.setattr(
        tiles, "_pool", lambda workers: pytest.fail("Nested tile pool must not start")
    )
    infos, _ = infer_streamed(
        image,
        get_model_spec("cellpose3:nuclei"),
        SegmentationSettings(tile_size=64, device="cpu"),
        tmp_path / "raw.zarr",
        _segment,
        allow_parallel=False,
    )
    assert infos[0]["streaming"]["workers"] == 1
