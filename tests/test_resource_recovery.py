"""Allocation and pool recovery regressions, without a real CUDA device."""

import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from cisegmentation import parallel_pipeline as pipeline
from cisegmentation import resources
from cisegmentation.ome_zarr_io import ImageResource
from cisegmentation.settings import SegmentationSettings


@pytest.mark.parametrize(
    "message",
    [
        "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)",
        "CUDNN_STATUS_ALLOC_FAILED",
        "CUFFT_ALLOC_FAILED",
        "CUDA out of memory",
    ],
)
def test_cuda_allocation_failure_classification(message):
    assert resources.memory_error(RuntimeError(message))
    assert resources.gpu_memory_error(RuntimeError(message))


@pytest.mark.parametrize(
    "message", ["CUBLAS_STATUS_EXECUTION_FAILED", "CUBLAS_STATUS_INVALID_VALUE"]
)
def test_other_cuda_errors_are_not_allocation_failures(message):
    assert not resources.memory_error(RuntimeError(message))


@pytest.mark.parametrize(
    "allocation,count",
    [
        ({"SLURM_GPUS_ON_NODE": "2"}, 2),
        ({"SLURM_STEP_GPUS": "MIG-a,MIG-b"}, 2),
        ({"CUDA_VISIBLE_DEVICES": "MIG-a"}, 1),
    ],
)
def test_memory_per_gpu_uses_local_allocated_count(monkeypatch, allocation, count):
    for name in ("SLURM_GPUS_ON_NODE", "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_MEM_PER_GPU", "2G")
    monkeypatch.setenv("SLURM_JOB_GPUS", "0,1,2,3,4,5,6,7")
    for name, value in allocation.items():
        monkeypatch.setenv(name, value)
    assert (
        resources.slurm_memory_limits(4)["SLURM_MEM_PER_GPU"]
        == count * 2 * resources.GIB
    )


def test_unknown_gpu_count_fails_early_with_logged_budget(monkeypatch):
    for name in ("SLURM_GPUS_ON_NODE", "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_MEM_PER_GPU", "2048")
    result = resources.resource_diagnostics(inspect_gpu=False)
    assert "local allocated GPU count is unknown" in result["budget_error"]
    assert result["environment"]["SLURM_MEM_PER_GPU"] == "2048"


def test_v1_cpu_quota_and_step_allocation(tmp_path, monkeypatch):
    (tmp_path / "cpu.cfs_quota_us").write_text("200000")
    (tmp_path / "cpu.cfs_period_us").write_text("100000")
    monkeypatch.setattr(resources, "cgroup_directories", lambda: [tmp_path])
    monkeypatch.setenv("SLURM_STEP_CPUS_PER_TASK", "4")
    monkeypatch.setattr(os, "cpu_count", lambda: 48)
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda _: set(range(48)), raising=False
    )
    assert resources.allocated_cpus() == 2


def test_mig_sizing_never_uses_parent_nvidia_fallback(monkeypatch):
    cuda = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(
        pipeline,
        "snapshot",
        lambda: resources.ResourceSnapshot(
            4,
            16 * resources.GIB,
            15 * resources.GIB,
            12 * resources.GIB,
            10 * resources.GIB,
        ),
    )
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *a, **k: pytest.fail("Parent GPU query must not run"),
    )
    assert pipeline._gpu_memory_mb() == (12288, 10240)
    cuda.is_available = lambda: False
    assert pipeline._gpu_memory_mb() is None
    cuda.is_available = lambda: True
    monkeypatch.setattr(
        pipeline,
        "snapshot",
        lambda: (_ for _ in ()).throw(RuntimeError("CUDA query failed")),
    )
    assert pipeline._gpu_memory_mb() is None


def test_failed_gpu_query_preserves_cpu_ram_bounds(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "4G")
    monkeypatch.setattr(
        resources,
        "_gpu_budget",
        lambda: (_ for _ in ()).throw(RuntimeError("CUDA failed")),
    )
    current = resources.snapshot()
    assert current.cpus <= 2
    assert current.ram_limit <= 4 * resources.GIB
    assert current.gpu_total == 0


def test_startup_log_identifies_job_partition_mig_and_effective_limits(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "mig")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "16G")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-test-uuid")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-be-logged")
    cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        device_count=lambda: 1,
        get_device_properties=lambda _: SimpleNamespace(
            name="A100 MIG", uuid="MIG-test-uuid"
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="12.6")),
    )
    monkeypatch.setattr(
        resources, "_gpu_budget", lambda: (12 * resources.GIB, 10 * resources.GIB)
    )
    log = []
    resources.log_resource_check(
        log.append, "startup", **resources.resource_diagnostics()
    )
    record = json.loads(log[0].split(" ", 1)[1])
    assert record["environment"]["SLURM_JOB_ID"] == "12345"
    assert record["environment"]["SLURM_JOB_PARTITION"] == "mig"
    assert record["gpu"]["mig"] == "detected"
    assert record["effective"]["cpus"] <= 4
    assert record["effective"]["ram_limit"] <= 16 * resources.GIB
    assert record["effective"]["gpu_total"] == 12 * resources.GIB
    assert "must-not-be-logged" not in log[0]


def test_cublas_error_returns_to_parent_for_fresh_context_retry(
    inputfolder, tmp_path, monkeypatch
):
    cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED")

    monkeypatch.setattr(pipeline, "segment_czyx", fail)
    settings = SegmentationSettings(cell_model="cellpose3:cyto3", nucleus_model="skip")
    payload = {
        "resource": ImageResource(inputfolder / "nuclei-small.ome.zarr", ""),
        "model_pass": pipeline.build_model_passes(settings)[0],
        "stage_dir": str(tmp_path),
    }
    result = pipeline._inference_task(payload)
    assert result["cuda_oom"] and result["memory_oom"]
    assert result["force_streaming_suggested"]
    assert calls == [1]  # No recursion into a possibly damaged CUDA context.


def test_worker_limits_all_backends_after_cli_import():
    code = """
import wrapper
from cisegmentation.parallel_pipeline import _worker_environment
_worker_environment()
from threadpoolctl import threadpool_info
assert all(p['num_threads'] == 1 for p in threadpool_info()), threadpool_info()
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_parent_backends_respect_slurm_cpu_allocation():
    code = """
import wrapper
from cisegmentation.resources import limit_main_threads
limit_main_threads()
from threadpoolctl import threadpool_info
assert all(p['num_threads'] <= 2 for p in threadpool_info()), threadpool_info()
"""
    subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "SLURM_CPUS_PER_TASK": "2"},
        check=True,
    )


@pytest.mark.parametrize(
    "in_worker,expected_threads,expected_mib", [(True, 1, 256), (False, 4, 2048)]
)
def test_database_settings_respect_parent_or_worker_budget(
    monkeypatch, capsys, in_worker, expected_threads, expected_mib
):
    import duckdb

    monkeypatch.setattr(resources, "_THREAD_LIMIT", object() if in_worker else None)
    monkeypatch.setattr(
        resources,
        "snapshot",
        lambda **kw: resources.ResourceSnapshot(
            4, 16 * resources.GIB, 12 * resources.GIB
        ),
    )
    connection = duckdb.connect(":memory:")
    try:
        resources.configure_database(connection)
        assert (
            connection.execute("SELECT current_setting('threads')").fetchone()[0]
            == expected_threads
        )
        expected_display = f"{expected_mib}.0 MiB" if expected_mib < 1024 else "2.0 GiB"
        assert (
            connection.execute("SELECT current_setting('memory_limit')").fetchone()[0]
            == expected_display
        )
    finally:
        connection.close()
    record = json.loads(capsys.readouterr().out.split(" ", 1)[1])
    assert record["event"] == "database_budget"
    assert record["threads"] == expected_threads
    assert record["memory_limit_mib"] == expected_mib


def test_recovery_wait_requeries_visible_headroom(monkeypatch):
    readings = iter([(12288, 1000), (12288, 10000)])
    monkeypatch.setattr(pipeline, "_gpu_memory_mb", lambda: next(readings))
    monkeypatch.setattr(pipeline.time, "sleep", lambda _: None)
    logs = []
    assert pipeline._wait_gpu_headroom(1000, logs.append) == (12288, 10000)
    event = json.loads(logs[-1].split(" ", 1)[1])
    assert event["status"] == "ready"
    assert event["required_mib"] > 3000


def test_recovery_wait_fails_if_device_remains_full(monkeypatch):
    monkeypatch.setattr(pipeline, "_gpu_memory_mb", lambda: (12288, 100))
    with pytest.raises(RuntimeError, match="insufficient recovery headroom"):
        pipeline._wait_gpu_headroom(1000, lambda _: None, timeout=0)


def _small_pool_task(payload):
    path = payload["resource"].image_path
    if path == "good":
        return {"ok": True, "resource_path": path}
    if path == "oom":
        time.sleep(0.5)
        return {"ok": False, "resource_path": path, "memory_oom": True}
    time.sleep(60)
    return {"ok": True, "resource_path": path}


def test_failed_pool_terminates_unfinished_workers_and_preserves_success(monkeypatch):
    monkeypatch.delenv("CISEGMENTATION_INLINE_WORKERS", raising=False)
    monkeypatch.setattr(pipeline, "_inference_task", _small_pool_task)
    owned = []
    original_pool = pipeline.ProcessPoolExecutor

    class RecordingPool(original_pool):
        def shutdown(self, *args, **kwargs):
            owned.extend((self._processes or {}).values())
            return super().shutdown(*args, **kwargs)

    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", RecordingPool)
    payloads = [
        {"resource": SimpleNamespace(image_path=p)}
        for p in ["good", "oom", "slow1", "slow2", "unsubmitted"]
    ]
    started = time.monotonic()
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        result = pipeline._execute_tasks(payloads, 2)
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=10)
    assert time.monotonic() - started < 30
    assert owned and all(not process.is_alive() for process in owned)
    assert {r["resource_path"] for r in result} == {
        p["resource"].image_path for p in payloads
    }
    assert [r["resource_path"] for r in result if r["ok"]] == ["good"]
    assert all(r.get("memory_oom") for r in result if not r["ok"])


def test_retry_preserves_confirmed_fields_and_shrinks_to_one(tmp_path, monkeypatch):
    fields = [ImageResource(tmp_path / "unused.zarr", str(i)) for i in range(4)]
    monkeypatch.setattr(pipeline, "_largest_resource", lambda _: fields[0])
    monkeypatch.setattr(pipeline, "calculate_cpu_workers", lambda *a, **k: 8)
    calls = []

    def success(path):
        return {
            "ok": True,
            "resource_path": path,
            "records": {"request_01": [{}]},
            "zarr_read_seconds": 0,
            "peak_cuda_mb": 100,
            "rss_mb": 100,
            "device": "cpu",
        }

    def execute(payloads, workers, **kwargs):
        paths = [p["resource"].image_path for p in payloads]
        calls.append((workers, paths))
        if len(calls) == 1:
            return [success(paths[0])]
        result = [
            success(p)
            if p == "1" or workers == 1
            else {
                "ok": False,
                "resource_path": p,
                "memory_oom": True,
                "error": "out of memory",
            }
            for p in paths
        ]
        return result

    monkeypatch.setattr(pipeline, "_execute_tasks", execute)
    logs = []
    _, records, provenance = pipeline.run_inference_passes(
        fields,
        SegmentationSettings(cell_model="cellpose3:cyto3", nucleus_model="skip"),
        tmp_path / "stage",
        log=logs.append,
    )
    assert [c[0] for c in calls] == [1, 8, 4, 2, 1]
    assert calls[2][1] == ["2", "3"]
    assert all("request_01" in records[f.image_path] for f in fields)
    assert provenance["inference_passes"][0]["workers"] == 1
    retry_logs = [
        json.loads(line.split(" ", 1)[1])
        for line in logs
        if line.startswith(resources.RESOURCE_LOG_PREFIX)
    ]
    assert any(r.get("preserved_successes") == 1 for r in retry_logs)
