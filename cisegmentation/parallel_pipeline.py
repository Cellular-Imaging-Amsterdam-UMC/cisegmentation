from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
import errno
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import traceback
from typing import Any, Callable

import numpy as np

from .adapters import segment_czyx
from .ome_zarr_io import (
    ImageResource,
    LabelResult,
    existing_label_names,
    read_native_label,
    read_image,
    write_native_label_groups,
)
from .registry import get_model_spec
from .settings import SKIP, SegmentationSettings


GPU_MIN_RESERVE_MB = 2048.0
GPU_RESERVE_FRACTION = 0.20
GPU_WORKER_SAFETY_FACTOR = 1.50
PROGRESS_HEARTBEAT_SECONDS = 15.0
PROGRESS_PERCENT_STEP = 5.0


def _duration_text(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class PhaseProgress:
    """Throttled progress emitted by the parent process."""

    def __init__(
        self,
        phase: str,
        total: int,
        log: Callable[[str], None] | None,
        *,
        completed: int = 0,
    ) -> None:
        self.phase = phase
        self.total = max(0, int(total))
        self.completed = max(0, min(int(completed), self.total))
        self.log = log
        self.started = time.perf_counter()
        self.last_emit = self.started
        self.last_percent = (
            100.0 * self.completed / self.total if self.total else 100.0
        )
        if self.log:
            self._emit(self.started)

    def _line(self, now: float) -> str:
        elapsed = max(0.0, now - self.started)
        percent = (
            100.0 * self.completed / self.total if self.total else 100.0
        )
        rate = self.completed / elapsed if elapsed > 0 else 0.0
        remaining = max(0, self.total - self.completed)
        eta = remaining / rate if rate > 0 else None
        eta_text = _duration_text(eta) if eta is not None else "calculating"
        return (
            f"{self.phase}: {self.completed}/{self.total} "
            f"({percent:.1f}%) | elapsed={_duration_text(elapsed)} "
            f"| rate={rate:.2f} fields/s | ETA={eta_text}"
        )

    def _emit(self, now: float) -> None:
        if self.log:
            self.log(self._line(now))
        self.last_emit = now
        self.last_percent = (
            100.0 * self.completed / self.total if self.total else 100.0
        )

    def advance(self, count: int = 1) -> None:
        self.set_completed(self.completed + count)

    def set_completed(self, completed: int, *, force: bool = False) -> None:
        self.completed = max(0, min(int(completed), self.total))
        now = time.perf_counter()
        percent = (
            100.0 * self.completed / self.total if self.total else 100.0
        )
        if (
            force
            or self.completed == self.total
            or percent - self.last_percent >= PROGRESS_PERCENT_STEP
            or now - self.last_emit >= PROGRESS_HEARTBEAT_SECONDS
        ):
            self._emit(now)

    def heartbeat(self) -> None:
        now = time.perf_counter()
        if now - self.last_emit >= PROGRESS_HEARTBEAT_SECONDS:
            self._emit(now)


def _iter_future_results(
    futures: dict,
    progress: PhaseProgress,
):
    """Yield worker results while the parent emits periodic heartbeats."""
    pending = set(futures)
    while pending:
        done, pending = wait(
            pending,
            timeout=PROGRESS_HEARTBEAT_SECONDS,
            return_when=FIRST_COMPLETED,
        )
        if not done:
            progress.heartbeat()
            continue
        for future in done:
            try:
                yield future.result()
            except Exception as exc:
                yield {
                    "ok": False,
                    "resource_path": futures[future],
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "cuda_oom": False,
                }


@dataclass(frozen=True)
class InferenceConsumer:
    step: str
    kind: str
    label_type: str = ""


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    model_id: str
    target: str
    primary_channel: int
    nuclei_channel: int
    settings: dict[str, Any]
    consumers: tuple[InferenceConsumer, ...]


@dataclass(frozen=True)
class ModelPass:
    pass_id: str
    model_id: str
    requests: tuple[InferenceRequest, ...]


def _foci_target(model_id: str) -> tuple[str, str]:
    spec = get_model_spec(model_id)
    if spec.family == "spotiflow" and "spots" in spec.targets:
        return "spots", "spots"
    if (
        spec.family == "stardist"
        and spec.checkpoint.startswith("SD_Foci")
        and "foci" in spec.targets
    ):
        return "foci", "foci"
    if (
        spec.family == "cellpose3"
        and "bact" in spec.checkpoint.lower()
        and "cells" in spec.targets
    ):
        return "cells", "bacteria"
    raise ValueError(f"Unsupported repeated foci model: {model_id}")


def _step_settings(
    settings: SegmentationSettings,
    *,
    model: str,
    target: str,
    primary_channel: int,
    nuclei_channel: int = 0,
) -> SegmentationSettings:
    resolved = replace(
        settings,
        model=model,
        target=target,
        primary_channel=primary_channel,
        nuclei_channel=nuclei_channel,
        benchmark=False,
    )
    if target == "cells" and "bact" in model and settings.diameter == 0:
        resolved = replace(resolved, diameter=-1.0)
    return resolved


def build_model_passes(settings: SegmentationSettings) -> list[ModelPass]:
    """Group exact inference requests by model in first-selected order."""
    consumers: list[tuple[SegmentationSettings, InferenceConsumer]] = []
    if settings.cell_model != SKIP:
        expansion_model = settings.cell_expansion_model()
        if expansion_model is not None:
            consumers.append(
                (
                    _step_settings(
                        settings,
                        model=expansion_model,
                        target="nuclei",
                        primary_channel=settings.cell_expansion_channel(),
                    ),
                    InferenceConsumer(
                        "Step 1 expansion nuclei", "expansion"
                    ),
                )
            )
        else:
            consumers.append(
                (
                    _step_settings(
                        settings,
                        model=settings.cell_model,
                        target="cells",
                        primary_channel=settings.cell_channel,
                        nuclei_channel=settings.cell_nuclei_channel,
                    ),
                    InferenceConsumer("Step 1 cells", "cell"),
                )
            )
    if settings.nucleus_model != SKIP:
        consumers.append(
            (
                _step_settings(
                    settings,
                    model=settings.nucleus_model,
                    target="nuclei",
                    primary_channel=settings.nucleus_channel,
                ),
                InferenceConsumer("Step 2 nuclei", "nucleus"),
            )
        )
    for slot, model_id, channel in settings.enabled_foci_steps():
        target, label_type = _foci_target(model_id)
        consumers.append(
            (
                _step_settings(
                    settings,
                    model=model_id,
                    target=target,
                    primary_channel=channel,
                ),
                InferenceConsumer(
                    f"Step 3{chr(96 + slot)} {label_type}",
                    "foci",
                    label_type,
                ),
            )
        )

    model_order: list[str] = []
    grouped: dict[str, list[tuple[SegmentationSettings, InferenceConsumer]]] = {}
    for request_settings, consumer in consumers:
        if request_settings.model not in grouped:
            grouped[request_settings.model] = []
            model_order.append(request_settings.model)
        grouped[request_settings.model].append((request_settings, consumer))

    passes = []
    request_number = 0
    for pass_number, model_id in enumerate(model_order, start=1):
        unique: dict[str, tuple[SegmentationSettings, list[InferenceConsumer]]] = {}
        order: list[str] = []
        for request_settings, consumer in grouped[model_id]:
            values = request_settings.to_dict()
            identity = json.dumps(values, sort_keys=True, separators=(",", ":"))
            if identity not in unique:
                unique[identity] = (request_settings, [])
                order.append(identity)
            unique[identity][1].append(consumer)
        requests = []
        for identity in order:
            request_number += 1
            request_settings, request_consumers = unique[identity]
            requests.append(
                InferenceRequest(
                    request_id=f"request_{request_number:02d}",
                    model_id=model_id,
                    target=request_settings.target,
                    primary_channel=request_settings.primary_channel,
                    nuclei_channel=request_settings.nuclei_channel,
                    settings=request_settings.to_dict(),
                    consumers=tuple(request_consumers),
                )
            )
        passes.append(
            ModelPass(
                pass_id=f"model_{pass_number:02d}",
                model_id=model_id,
                requests=tuple(requests),
            )
        )
    return passes


def expected_channel_labels(settings: SegmentationSettings) -> list[str]:
    labels: list[str] = []
    has_cells = settings.cell_model != SKIP
    has_nuclei = (
        settings.nucleus_model != SKIP
        or settings.cell_expansion_model() is not None
    )
    if has_cells and has_nuclei:
        labels.extend(("labels_cells", "labels_nuclei", "labels_cytoplasm"))
    elif has_cells:
        labels.append("labels_cells")
    elif settings.nucleus_model != SKIP:
        labels.append("labels_nuclei")
    for _slot, model_id, channel in settings.enabled_foci_steps():
        _target, label_type = _foci_target(model_id)
        labels.append(f"labels_{label_type}_channel_{channel}")
    return labels


def unique_group_names(labels: list[str], occupied: set[str] | None = None) -> list[str]:
    used = set(occupied or ())
    names = []
    for index, label in enumerate(labels, start=1):
        base = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label).strip("_.-")
        base = base or f"labels_{index}"
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        names.append(candidate)
    return names


def resolve_label_policy(
    resources: list[ImageResource],
    settings: SegmentationSettings,
) -> tuple[dict[str, list[str]], list[str], dict[str, list[str]]]:
    existing_by_resource = {
        resource.image_path: existing_label_names(resource)
        for resource in resources
    }
    generated_labels = expected_channel_labels(settings)
    base_names = unique_group_names(generated_labels)
    if settings.existing_labels == "append":
        occupied = {
            name for names in existing_by_resource.values() for name in names
        }
        generated_names = unique_group_names(generated_labels, occupied)
    else:
        generated_names = base_names

    final_by_resource = {}
    for resource in resources:
        existing = existing_by_resource[resource.image_path]
        if settings.existing_labels == "remove":
            final = list(generated_names)
        elif settings.existing_labels == "overwrite":
            final = [
                name for name in existing if name not in set(generated_names)
            ] + list(generated_names)
        else:
            final = list(existing) + list(generated_names)
        final_by_resource[resource.image_path] = final
    return existing_by_resource, generated_names, final_by_resource


def _resource_key(resource: ImageResource) -> str:
    identity = resource.image_path or "__root__"
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]


def _raw_path(
    stage_dir: Path, model_pass: ModelPass, request: InferenceRequest, resource: ImageResource
) -> Path:
    return (
        stage_dir
        / "raw"
        / model_pass.pass_id
        / request.request_id
        / f"{_resource_key(resource)}.npy"
    )


def _atomic_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, path)


def _worker_environment() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"


def _process_gpu_memory_mb() -> float:
    """Return NVIDIA's memory accounting for the current worker process."""
    try:
        process = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if process.returncode != 0:
            return 0.0
        current_pid = os.getpid()
        for line in process.stdout.splitlines():
            values = [value.strip() for value in line.split(",")]
            if len(values) < 2 or not values[0].isdigit():
                continue
            if int(values[0]) != current_pid:
                continue
            try:
                return float(values[1])
            except ValueError:
                return 0.0
    except Exception:
        pass
    return 0.0


def _inference_task(payload: dict[str, Any]) -> dict[str, Any]:
    _worker_environment()
    resource: ImageResource = payload["resource"]
    model_pass: ModelPass = payload["model_pass"]
    stage_dir = Path(payload["stage_dir"])
    started = time.perf_counter()
    try:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            torch = None
        read_started = time.perf_counter()
        image = read_image(resource)
        read_seconds = time.perf_counter() - read_started
        records: dict[str, list[dict[str, Any]]] = {}
        for request in model_pass.requests:
            request_settings = SegmentationSettings(**request.settings)
            spec = get_model_spec(request.model_id)
            labels_per_time = []
            infos = []
            for time_index in range(image.data.shape[0]):
                labels, info = segment_czyx(
                    image.data[time_index],
                    spec,
                    request_settings,
                    image.scales,
                )
                labels_per_time.append(np.asarray(labels, dtype=np.uint32))
                infos.append(dict(info))
            _atomic_save(
                _raw_path(stage_dir, model_pass, request, resource),
                np.stack(labels_per_time, axis=0),
            )
            records[request.request_id] = infos
        first_info = next(
            (
                info
                for request_infos in records.values()
                for info in request_infos
            ),
            {},
        )
        device = str(first_info.get("device") or "cpu").lower()
        torch_peak_cuda_mb = 0.0
        process_cuda_mb = 0.0
        if (
            device.startswith("cuda")
            and torch is not None
            and torch.cuda.is_available()
        ):
            torch_peak_cuda_mb = max(
                float(torch.cuda.max_memory_reserved()),
                float(torch.cuda.max_memory_allocated()),
            ) / 1024**2
            process_cuda_mb = _process_gpu_memory_mb()
        peak_cuda_mb = max(torch_peak_cuda_mb, process_cuda_mb)
        try:
            import psutil

            rss_mb = psutil.Process().memory_info().rss / 1024**2
        except Exception:
            rss_mb = 0.0
        return {
            "ok": True,
            "resource_path": resource.image_path,
            "records": records,
            "runtime_seconds": time.perf_counter() - started,
            "zarr_read_seconds": read_seconds,
            "peak_cuda_mb": peak_cuda_mb,
            "torch_peak_cuda_mb": torch_peak_cuda_mb,
            "process_cuda_mb": process_cuda_mb,
            "rss_mb": rss_mb,
            "device": device,
        }
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return {
            "ok": False,
            "resource_path": resource.image_path,
            "error": message,
            "traceback": traceback.format_exc(),
            "cuda_oom": "out of memory" in message.lower()
            and "cuda" in (message + traceback.format_exc()).lower(),
        }


def _gpu_memory_mb() -> tuple[float, float] | None:
    try:
        process = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if process.returncode != 0 or not process.stdout.strip():
            return None
        total, free = (
            float(value.strip())
            for value in process.stdout.splitlines()[0].split(",")[:2]
        )
        return total, free
    except Exception:
        return None


def calculate_gpu_workers(
    *,
    total_mb: float,
    free_mb: float,
    peak_worker_mb: float,
    task_count: int,
    cap: int = 0,
) -> int:
    if task_count <= 0:
        return 0
    if peak_worker_mb <= 0:
        return 1
    reserve = max(GPU_MIN_RESERVE_MB, total_mb * GPU_RESERVE_FRACTION)
    usable = max(0.0, free_mb - reserve)
    safe_peak = peak_worker_mb * GPU_WORKER_SAFETY_FACTOR
    workers = max(1, int(math.floor(usable / safe_peak)))
    if cap:
        workers = min(workers, cap)
    return min(workers, task_count)


def available_cpu_workers(task_count: int, cap: int = 0) -> int:
    if task_count <= 0:
        return 0
    candidates = []
    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(name)
        if value and value.isdigit():
            candidates.append(int(value))
    job_cpus = os.environ.get("SLURM_JOB_CPUS_PER_NODE", "")
    match = re.match(r"\s*(\d+)", job_cpus)
    if match:
        candidates.append(int(match.group(1)))
    if hasattr(os, "sched_getaffinity"):
        try:
            candidates.append(len(os.sched_getaffinity(0)))
        except OSError:
            pass
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        if physical:
            candidates.append(int(physical))
    except Exception:
        pass
    allocated = min(candidates) if candidates else (os.cpu_count() or 1)
    workers = max(1, allocated - 1)
    if cap:
        workers = min(workers, cap)
    return min(workers, task_count)


def calculate_cpu_workers(
    task_count: int,
    *,
    peak_worker_mb: float,
    cap: int = 0,
) -> int:
    workers = available_cpu_workers(task_count, cap)
    if workers <= 1 or peak_worker_mb <= 0:
        return workers
    try:
        import psutil

        available_mb = psutil.virtual_memory().available / 1024**2
    except Exception:
        return workers
    memory_workers = max(
        1, int(math.floor((available_mb * 0.80) / (peak_worker_mb * 1.20)))
    )
    return min(workers, memory_workers)


def _largest_resource(resources: list[ImageResource]) -> ImageResource:
    import zarr

    scored = []
    for resource in resources:
        root = zarr.open_group(str(resource.store_path), mode="r")
        group = root[resource.image_path] if resource.image_path else root
        multiscale = (group.attrs.get("multiscales") or [{}])[0]
        dataset = str((multiscale.get("datasets") or [{"path": "0"}])[0]["path"])
        size = int(np.prod(group[dataset].shape))
        root.store.close()
        scored.append((size, resource.image_path, resource))
    return max(scored, key=lambda item: (item[0], item[1]))[2]


def _execute_tasks(
    payloads: list[dict[str, Any]],
    workers: int,
    *,
    progress: PhaseProgress | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    if not payloads:
        return []
    context = multiprocessing.get_context("spawn")
    if workers == 1 and os.environ.get("CISEGMENTATION_INLINE_WORKERS") == "1":
        results = []
        for payload in payloads:
            result = _inference_task(payload)
            results.append(result)
            if on_result:
                on_result(result)
        return results
    results = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_worker_environment,
    ) as executor:
        futures = {
            executor.submit(_inference_task, payload): payload["resource"].image_path
            for payload in payloads
        }
        reporter = progress or PhaseProgress(
            "Inference worker tasks", len(payloads), None
        )
        for result in _iter_future_results(futures, reporter):
            results.append(result)
            if on_result:
                on_result(result)
    return results


def run_inference_passes(
    resources: list[ImageResource],
    settings: SegmentationSettings,
    stage_dir: Path,
    *,
    log: Callable[[str], None] | None = None,
) -> tuple[list[ModelPass], dict[str, dict[str, list[dict[str, Any]]]], dict]:
    passes = build_model_passes(settings)
    records: dict[str, dict[str, list[dict[str, Any]]]] = {
        resource.image_path: {} for resource in resources
    }
    provenance = {"inference_passes": []}
    probe_resource = _largest_resource(resources)
    for model_pass in passes:
        pass_started = time.perf_counter()
        if log:
            log(
                f"Model pass {model_pass.pass_id}: {model_pass.model_id} "
                f"across {len(resources)} field(s)"
            )
        progress = PhaseProgress(
            f"Inference {model_pass.model_id}",
            len(resources),
            log,
        )
        probe_payload = {
            "resource": probe_resource,
            "model_pass": model_pass,
            "stage_dir": str(stage_dir),
        }
        probe_retries = 0
        while True:
            probe = _execute_tasks(
                [probe_payload], 1, progress=progress
            )[0]
            if probe.get("ok"):
                break
            probe_retries += 1
            if probe_retries > 2:
                raise RuntimeError(
                    f"Model probe failed after two retries for "
                    f"{model_pass.model_id}: {probe.get('error')}\n"
                    f"{probe.get('traceback', '')}"
                )
            if log:
                log(f"  model probe retry {probe_retries}/2")
        progress.advance()
        records[probe_resource.image_path].update(probe["records"])
        records[probe_resource.image_path]["__zarr_read_seconds__"] = [
            {"seconds": float(probe["zarr_read_seconds"])}
        ]
        remaining = [
            resource
            for resource in resources
            if resource.image_path != probe_resource.image_path
        ]
        if probe["device"] == "cuda":
            memory = _gpu_memory_mb()
            workers = (
                calculate_gpu_workers(
                    total_mb=memory[0],
                    free_mb=memory[1],
                    peak_worker_mb=float(probe["peak_cuda_mb"]),
                    task_count=len(remaining),
                    cap=settings.max_inference_workers,
                )
                if memory is not None
                else min(1, len(remaining))
            )
        else:
            workers = calculate_cpu_workers(
                len(remaining),
                peak_worker_mb=float(probe["rss_mb"]),
                cap=settings.max_inference_workers,
            )
        selected_workers = max(1, workers) if remaining else 1
        retries = 0
        pending = list(remaining)
        while pending:
            payloads = [
                {
                    "resource": resource,
                    "model_pass": model_pass,
                    "stage_dir": str(stage_dir),
                }
                for resource in pending
            ]
            batch = _execute_tasks(
                payloads,
                selected_workers,
                progress=progress,
                on_result=lambda result: (
                    progress.advance() if result.get("ok") else None
                ),
            )
            failed_paths = {
                result.get("resource_path", "")
                for result in batch
                if not result.get("ok")
            }
            cuda_oom = any(
                result.get("cuda_oom")
                for result in batch
                if not result.get("ok")
            )
            if cuda_oom:
                successful_batch = sum(
                    bool(result.get("ok")) for result in batch
                )
                progress.set_completed(
                    progress.completed - successful_batch,
                    force=True,
                )
                for resource in pending:
                    for request in model_pass.requests:
                        _raw_path(
                            stage_dir, model_pass, request, resource
                        ).unlink(missing_ok=True)
                        records[resource.image_path].pop(
                            request.request_id, None
                        )
            else:
                for result in batch:
                    if result.get("ok"):
                        records[result["resource_path"]].update(
                            result["records"]
                        )
                        previous = records[result["resource_path"]].get(
                            "__zarr_read_seconds__", [{"seconds": 0.0}]
                        )[0]["seconds"]
                        records[result["resource_path"]][
                            "__zarr_read_seconds__"
                        ] = [
                            {
                                "seconds": float(previous)
                                + float(result["zarr_read_seconds"])
                            }
                        ]
            if not failed_paths:
                break
            retries += 1
            if retries > 2:
                first = next(result for result in batch if not result.get("ok"))
                raise RuntimeError(
                    f"Inference failed after two retries for "
                    f"{model_pass.model_id}: {first.get('error')}\n"
                    f"{first.get('traceback', '')}"
                )
            if cuda_oom:
                selected_workers = max(1, selected_workers // 2)
            if not cuda_oom:
                pending = [
                    resource
                    for resource in pending
                    if resource.image_path in failed_paths
                ]
            if log:
                log(
                    f"  retry {retries}/2: {len(pending)} field(s), "
                    f"{selected_workers} worker(s)"
                )
        pass_provenance = {
            "model": model_pass.model_id,
            "device": probe["device"],
            "probe_peak_cuda_mb": float(probe["peak_cuda_mb"]),
            "probe_torch_peak_cuda_mb": float(
                probe.get("torch_peak_cuda_mb", 0.0)
            ),
            "probe_process_cuda_mb": float(
                probe.get("process_cuda_mb", 0.0)
            ),
            "probe_rss_mb": float(probe["rss_mb"]),
            "workers": selected_workers,
            "tasks": len(resources),
            "retries": retries + probe_retries,
            "probe_retries": probe_retries,
            "runtime_seconds": time.perf_counter() - pass_started,
        }
        provenance["inference_passes"].append(pass_provenance)
        if log:
            log(
                "  model pass complete: "
                f"device={pass_provenance['device']}, "
                f"probe CUDA={pass_provenance['probe_peak_cuda_mb']:.1f} MiB, "
                f"(torch={pass_provenance['probe_torch_peak_cuda_mb']:.1f}, "
                f"process={pass_provenance['probe_process_cuda_mb']:.1f}), "
                f"probe RSS={pass_provenance['probe_rss_mb']:.1f} MiB, "
                f"workers={selected_workers}, "
                f"retries={pass_provenance['retries']}"
            )
    return passes, records, provenance


def prepare_label_overlay(
    source_store: Path,
    overlay_path: Path,
    resources: list[ImageResource],
) -> Path:
    """Create a sparse hierarchy containing metadata but no source pixel arrays."""
    import zarr

    if overlay_path.exists():
        shutil.rmtree(overlay_path)
    source = zarr.open_group(str(source_store), mode="r")
    overlay = zarr.open_group(str(overlay_path), mode="w", zarr_version=2)
    overlay.attrs.update(dict(source.attrs.asdict()))
    copied: set[str] = {""}
    for resource in resources:
        components = resource.image_path.split("/") if resource.image_path else []
        for count in range(1, len(components) + 1):
            path = "/".join(components[:count])
            if path in copied:
                continue
            destination = overlay.require_group(path)
            destination.attrs.update(dict(source[path].attrs.asdict()))
            copied.add(path)
    source.store.close()
    overlay.store.close()
    return overlay_path


def _reuse_info(info: dict[str, Any], source_step: str) -> dict[str, Any]:
    reused = dict(info)
    reused.update(
        {
            "runtime_seconds": 0.0,
            "timings": {},
            "model_cache_hit": False,
            "model_cache_hits": 0,
            "model_cache_misses": 0,
            "result_cache_hit": True,
            "reused_from_step": source_step,
        }
    )
    return reused


def _finalize_task(payload: dict[str, Any]) -> dict[str, Any]:
    _worker_environment()
    from copy import deepcopy

    from . import engine
    from .reporting import label_statistics, step_record

    resource: ImageResource = payload["resource"]
    settings = SegmentationSettings(**payload["settings"])
    passes: list[ModelPass] = payload["passes"]
    records: dict[str, list[dict[str, Any]]] = payload["records"]
    stage_dir = Path(payload["stage_dir"])
    overlay_path = Path(payload["overlay_path"])
    generated_names: list[str] = payload["generated_names"]
    final_names: list[str] = payload["final_names"]
    try:
        image = read_image(resource)
        request_by_kind: dict[str, list[tuple[ModelPass, InferenceRequest, InferenceConsumer]]] = {}
        for model_pass in passes:
            for request in model_pass.requests:
                for consumer in request.consumers:
                    request_by_kind.setdefault(consumer.kind, []).append(
                        (model_pass, request, consumer)
                    )
        raw_cache = {
            request.request_id: np.load(
                _raw_path(stage_dir, model_pass, request, resource),
                allow_pickle=False,
            )
            for model_pass in passes
            for request in model_pass.requests
        }
        used: dict[tuple[int, str], str] = {}
        infos: list[dict] = []
        step_runs: list[dict] = []
        output_statistics: list[dict] = []
        per_time = []
        channel_labels: list[str] = []
        next_id = 0

        def consume(kind: str, time_index: int, index: int = 0):
            model_pass, request, consumer = request_by_kind[kind][index]
            labels = np.asarray(raw_cache[request.request_id][time_index]).copy()
            info = deepcopy(records[request.request_id][time_index])
            reuse_key = (time_index, request.request_id)
            if reuse_key in used:
                info = _reuse_info(info, used[reuse_key])
            else:
                info["result_cache_hit"] = False
                info["reused_from_step"] = None
                used[reuse_key] = consumer.step
            record = step_record(
                step=consumer.step,
                timepoint=time_index,
                model=request.model_id,
                target=request.target,
                primary_channel=request.primary_channel,
                nuclei_channel=request.nuclei_channel,
                labels=labels,
                info=info,
                scales=image.scales,
                include_label_statistics=settings.labels_log_info,
            )
            infos.append(info)
            step_runs.append(record)
            return labels

        for time_index in range(image.data.shape[0]):
            cell_labels = nucleus_labels = cell_step_nuclei = None
            if request_by_kind.get("expansion"):
                cell_step_nuclei = consume("expansion", time_index)
                cell_labels = engine._expand_nuclei_to_cells(
                    cell_step_nuclei,
                    settings.cell_expansion_distance,
                    image.scales,
                )
            elif request_by_kind.get("cell"):
                cell_labels = consume("cell", time_index)
            if request_by_kind.get("nucleus"):
                nucleus_labels = consume("nucleus", time_index)

            time_channels = []
            time_labels = []
            matching_nuclei = (
                nucleus_labels if nucleus_labels is not None else cell_step_nuclei
            )
            if cell_labels is not None and matching_nuclei is not None:
                cells, nuclei, cytoplasm, next_id = engine._match_cells_and_nuclei(
                    cell_labels,
                    matching_nuclei,
                    first_id=next_id + 1,
                    remove_border_cells=settings.remove_border_cells,
                )
                time_channels.extend((cells, nuclei, cytoplasm))
                time_labels.extend(
                    ("labels_cells", "labels_nuclei", "labels_cytoplasm")
                )
            elif cell_labels is not None:
                if settings.remove_border_cells:
                    touching = engine._border_ids(cell_labels)
                    if touching:
                        cell_labels = np.where(
                            np.isin(cell_labels, list(touching)), 0, cell_labels
                        )
                cell_labels, next_id = engine._offset_labels(cell_labels, next_id)
                time_channels.append(cell_labels)
                time_labels.append("labels_cells")
            elif nucleus_labels is not None:
                nucleus_labels, next_id = engine._offset_labels(
                    nucleus_labels, next_id
                )
                time_channels.append(nucleus_labels)
                time_labels.append("labels_nuclei")
            for index, (_model_pass, request, consumer) in enumerate(
                request_by_kind.get("foci", [])
            ):
                labels = consume("foci", time_index, index)
                labels, next_id = engine._offset_labels(labels, next_id)
                time_channels.append(labels)
                time_labels.append(
                    f"labels_{consumer.label_type}_channel_{request.primary_channel}"
                )
            if settings.labels_log_info:
                for name, labels in zip(time_labels, time_channels):
                    output_statistics.append(
                        {
                            "timepoint": time_index,
                            "channel": name,
                            "locations_only": bool(
                                name.startswith("labels_spots_channel_")
                                and not settings.spotiflow_local_refinement
                            ),
                            "label_statistics": label_statistics(labels, image.scales),
                        }
                    )
            if not channel_labels:
                channel_labels = time_labels
            per_time.append(np.stack(time_channels, axis=0))
        tczyx = np.stack(per_time, axis=0)
        timings = engine._aggregate_timings(
            [dict(info.get("timings", {})) for info in infos]
        )
        timings["zarr_read_seconds"] = float(
            records.get("__zarr_read_seconds__", [{"seconds": 0.0}])[0][
                "seconds"
            ]
        )
        provenance = {
            "device": infos[-1].get("device") if infos else None,
            "runtime_seconds": sum(
                float(info.get("runtime_seconds", 0.0)) for info in infos
            ),
            "segmentation_count": sum(
                not bool(info.get("result_cache_hit")) for info in infos
            ),
            "model_cache_hits": sum(
                int(info.get("model_cache_hits", 0))
                for info in infos
                if not info.get("result_cache_hit")
            ),
            "model_cache_misses": sum(
                int(info.get("model_cache_misses", 0))
                for info in infos
                if not info.get("result_cache_hit")
            ),
            "result_cache_hits": sum(
                bool(info.get("result_cache_hit")) for info in infos
            ),
            "timings": timings,
            "step_runs": step_runs,
            "parameters": settings.to_dict(),
            "shared_instance_ids": ["cells", "nuclei", "cytoplasm"]
            if settings.cell_model != SKIP
            and (
                settings.nucleus_model != SKIP
                or settings.cell_expansion_model() is not None
            )
            else [],
        }
        if output_statistics:
            provenance["output_statistics"] = output_statistics
        result = LabelResult(
            tczyx,
            image,
            "multi-step",
            "multi-step",
            provenance=provenance,
            channel_labels=channel_labels,
            label_origins=["generated"] * len(channel_labels),
        )
        write_started = time.perf_counter()
        write_native_label_groups(
            overlay_path,
            resource.image_path,
            result,
            generated_names,
            final_names,
        )
        return {
            "ok": True,
            "resource_path": resource.image_path,
            "provenance": provenance,
            "channel_labels": channel_labels,
            "zarr_write_seconds": time.perf_counter() - write_started,
        }
    except Exception as exc:
        return {
            "ok": False,
            "resource_path": resource.image_path,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def run_label_finalization(
    resources: list[ImageResource],
    settings: SegmentationSettings,
    passes: list[ModelPass],
    records: dict[str, dict[str, list[dict[str, Any]]]],
    stage_dir: Path,
    overlay_path: Path,
    generated_names: list[str],
    final_by_resource: dict[str, list[str]],
    *,
    log: Callable[[str], None] | None = None,
) -> tuple[dict[str, dict], dict]:
    phase_started = time.perf_counter()
    workers = available_cpu_workers(
        len(resources), settings.max_measurement_workers
    )
    progress = PhaseProgress(
        "Label finalization",
        len(resources),
        log,
    )
    pending = list(resources)
    completed: dict[str, dict] = {}
    retries = 0
    context = multiprocessing.get_context("spawn")
    while pending:
        payloads = [
            {
                "resource": resource,
                "settings": settings.to_dict(),
                "passes": passes,
                "records": records[resource.image_path],
                "stage_dir": str(stage_dir),
                "overlay_path": str(overlay_path),
                "generated_names": generated_names,
                "final_names": final_by_resource[resource.image_path],
            }
            for resource in pending
        ]
        results = []
        if workers == 1 and os.environ.get("CISEGMENTATION_INLINE_WORKERS") == "1":
            for payload in payloads:
                result = _finalize_task(payload)
                results.append(result)
                if result["ok"]:
                    progress.advance()
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_worker_environment,
            ) as executor:
                futures = {
                    executor.submit(_finalize_task, item): item[
                        "resource"
                    ].image_path
                    for item in payloads
                }
                for result in _iter_future_results(futures, progress):
                    results.append(result)
                    if result["ok"]:
                        progress.advance()
        failures = {
            result["resource_path"] for result in results if not result["ok"]
        }
        for result in results:
            if result["ok"]:
                completed[result["resource_path"]] = result
        if not failures:
            break
        retries += 1
        if retries > 2:
            first = next(result for result in results if not result["ok"])
            raise RuntimeError(
                f"Label finalization failed after two retries: "
                f"{first['error']}\n{first['traceback']}"
            )
        pending = [
            resource for resource in pending if resource.image_path in failures
        ]
        if log:
            log(f"Label finalization retry {retries}/2 for {len(pending)} field(s)")
    return completed, {
        "workers": workers,
        "retries": retries,
        "runtime_seconds": time.perf_counter() - phase_started,
        "zarr_write_seconds": sum(
            float(item.get("zarr_write_seconds", 0.0))
            for item in completed.values()
        ),
    }


def resolved_label_result(
    resource: ImageResource,
    overlay_path: Path,
    final_names: list[str],
    generated_names: list[str],
    provenance: dict[str, Any],
) -> LabelResult:
    """Load the label view that will exist after applying an overlay."""
    image = read_image(resource)
    generated = set(generated_names)
    arrays = []
    origins = []
    for name in final_names:
        if name in generated:
            arrays.append(
                read_native_label(resource, name, store_path=overlay_path)
            )
            origins.append("generated")
        else:
            arrays.append(read_native_label(resource, name))
            origins.append("existing")
    if not arrays:
        raise ValueError(f"No resolved labels for {resource.name}")
    labels = np.concatenate(arrays, axis=1)
    return LabelResult(
        labels,
        image,
        "multi-step",
        "multi-step",
        provenance=provenance,
        channel_labels=list(final_names),
        label_origins=origins,
    )


def write_overlay_manifest(
    overlay_path: Path,
    source_store: Path,
    settings: SegmentationSettings,
    resources: list[ImageResource],
    existing_by_resource: dict[str, list[str]],
    generated_names: list[str],
    final_by_resource: dict[str, list[str]],
) -> Path:
    generated_labels = expected_channel_labels(settings)
    fields = []
    for resource in resources:
        prefix = f"{resource.image_path}/" if resource.image_path else ""
        if settings.existing_labels == "remove":
            replace_paths = [f"{prefix}labels"]
        elif settings.existing_labels == "overwrite":
            replace_paths = [
                f"{prefix}labels/{name}" for name in generated_names
            ]
        else:
            replace_paths = []
        fields.append(
            {
                "resource_path": resource.image_path,
                "existing_labels": existing_by_resource[resource.image_path],
                "final_labels": final_by_resource[resource.image_path],
                "replace_paths": replace_paths,
                "remove_existing_labels_tree": (
                    settings.existing_labels == "remove"
                ),
            }
        )
    manifest = {
        "format": "cisegmentation-label-overlay",
        "version": 1,
        "source_store": source_store.name,
        "source_identity": {
            "store_name": source_store.name,
            "root_zattrs_sha256": hashlib.sha256(
                (source_store / ".zattrs").read_bytes()
            ).hexdigest(),
            "field_count": len(resources),
        },
        "existing_labels_policy": settings.existing_labels,
        "generated_label_mapping": dict(zip(generated_labels, generated_names)),
        "fields": fields,
    }
    path = overlay_path / "cisegmentation-label-overlay.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _journal_path(store: Path) -> Path:
    return store / ".cisegmentation-label-journal.json"


def _rename_with_retry(origin: Path, target: Path) -> None:
    """Rename a file or directory, tolerating brief Windows handle locks."""
    for attempt in range(50):
        try:
            os.replace(origin, target)
            return
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(0.1)


def _tree_manifest(root: Path) -> dict[str, int]:
    return {
        path.relative_to(root).as_posix(): path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
    }


def _move_or_copy_tree(source: Path, destination: Path) -> str:
    """Move a tree atomically, with a verified cross-filesystem fallback."""
    try:
        _rename_with_retry(source, destination)
        return "rename"
    except OSError as exc:
        if (
            getattr(exc, "errno", None) != errno.EXDEV
            and getattr(exc, "winerror", None) != 17
        ):
            raise
    partial = destination.with_name(
        f".{destination.name}.cisegmentation-install-{os.getpid()}"
    )
    if partial.exists():
        shutil.rmtree(partial)
    try:
        shutil.copytree(source, partial)
        if _tree_manifest(source) != _tree_manifest(partial):
            raise OSError(
                "Cross-filesystem label copy verification failed"
            )
        _rename_with_retry(partial, destination)
        shutil.rmtree(source)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    return "copy"


def recover_label_commit(store: Path) -> bool:
    """Restore label trees left by an interrupted in-place commit."""
    journal_path = _journal_path(store)
    if not journal_path.exists():
        return False
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    for item in reversed(journal.get("fields", [])):
        labels_path = store / item["labels_path"]
        backup_path = store / item["backup_path"]
        if labels_path.exists() and backup_path.exists():
            for name in item.get("preserve", []):
                    current = labels_path / name
                    restored = backup_path / name
                    if current.exists() and not restored.exists():
                        _rename_with_retry(current, restored)
        if labels_path.exists():
            shutil.rmtree(labels_path)
        if item["had_labels"] and backup_path.exists():
            labels_path.parent.mkdir(parents=True, exist_ok=True)
            _rename_with_retry(backup_path, labels_path)
    for item in journal.get("metadata", []):
        metadata_path = store / item["path"]
        backup_path = store / item["backup_path"]
        if item["existed"] and backup_path.exists():
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup_path, metadata_path)
        elif not item["existed"]:
            metadata_path.unlink(missing_ok=True)
    backup_root = store / journal["backup_root"]
    if backup_root.exists():
        shutil.rmtree(backup_root)
    journal_path.unlink(missing_ok=True)
    return True


def commit_overlay_labels(
    source_store: Path,
    overlay_path: Path,
    resources: list[ImageResource],
    settings: SegmentationSettings,
    existing_by_resource: dict[str, list[str]],
    generated_names: list[str],
    *,
    retain_journal: bool = False,
) -> None:
    """Journal and commit generated overlay labels into the source store."""
    run_id = f"{int(time.time())}_{os.getpid()}"
    backup_root_name = f".cisegmentation-label-backup-{run_id}"
    backup_root = source_store / backup_root_name
    fields = []
    for resource in resources:
        prefix = Path(resource.image_path) if resource.image_path else Path()
        labels_rel = prefix / "labels"
        backup_rel = Path(backup_root_name) / _resource_key(resource) / "labels"
        preserve = existing_by_resource[resource.image_path]
        if settings.existing_labels == "remove":
            preserve = []
        elif settings.existing_labels == "overwrite":
            preserve = [
                name for name in preserve if name not in set(generated_names)
            ]
        fields.append(
            {
                "resource_path": resource.image_path,
                "labels_path": labels_rel.as_posix(),
                "backup_path": backup_rel.as_posix(),
                "had_labels": (source_store / labels_rel).exists(),
                "preserve": preserve,
            }
        )
    journal = {
        "version": 1,
        "backup_root": backup_root_name,
        "fields": fields,
        "metadata": [],
    }
    metadata_paths = [Path(".zattrs")]
    metadata_paths.extend(
        (
            Path(resource.image_path) / ".zattrs"
            if resource.image_path
            else Path(".zattrs")
        )
        for resource in resources
    )
    for index, metadata_path in enumerate(
        dict.fromkeys(metadata_paths)
    ):
        journal["metadata"].append(
            {
                "path": metadata_path.as_posix(),
                "backup_path": (
                    Path(backup_root_name)
                    / "metadata"
                    / f"{index:06d}.zattrs"
                ).as_posix(),
                "existed": (source_store / metadata_path).exists(),
            }
        )
    _journal_path(source_store).write_text(
        json.dumps(journal, indent=2), encoding="utf-8"
    )
    try:
        for item in journal["metadata"]:
            metadata_path = source_store / item["path"]
            if not item["existed"]:
                continue
            backup_path = source_store / item["backup_path"]
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(metadata_path, backup_path)
        for resource, item in zip(resources, fields):
            source_labels = source_store / item["labels_path"]
            overlay_labels = overlay_path / item["labels_path"]
            backup_labels = source_store / item["backup_path"]
            if item["had_labels"]:
                backup_labels.parent.mkdir(parents=True, exist_ok=True)
                _rename_with_retry(source_labels, backup_labels)
            source_labels.parent.mkdir(parents=True, exist_ok=True)
            _move_or_copy_tree(overlay_labels, source_labels)
            if item["had_labels"]:
                for name in item["preserve"]:
                    old_group = backup_labels / name
                    if old_group.exists():
                        _rename_with_retry(
                            old_group, source_labels / name
                        )
        if not retain_journal:
            if backup_root.exists():
                shutil.rmtree(backup_root)
            _journal_path(source_store).unlink(missing_ok=True)
    except Exception:
        recover_label_commit(source_store)
        raise


def finalize_label_commit(store: Path) -> None:
    """Discard a retained rollback journal after publication succeeds."""
    journal_path = _journal_path(store)
    if not journal_path.exists():
        return
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    backup_root = store / journal["backup_root"]
    if backup_root.exists():
        shutil.rmtree(backup_root)
    journal_path.unlink(missing_ok=True)


def _replace_destination(source: Path, destination: Path) -> None:
    previous = destination.with_name(destination.name + ".previous")
    if previous.exists():
        shutil.rmtree(previous)
    if destination.exists():
        _rename_with_retry(destination, previous)
    try:
        _rename_with_retry(source, destination)
    except Exception:
        if previous.exists() and not destination.exists():
            _rename_with_retry(previous, destination)
        raise
    if previous.exists():
        shutil.rmtree(previous)


def publish_consumed_store(source: Path, destination: Path) -> str:
    """Move a completed source store, copying safely across filesystems."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        _replace_destination(source, destination)
        return "rename"
    except OSError as exc:
        if getattr(exc, "errno", None) not in {18, 17}:
            raise
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    try:
        shutil.copytree(source, partial)
        source_manifest = _tree_manifest(source)
        copied_manifest = _tree_manifest(partial)
        if source_manifest != copied_manifest:
            raise OSError(
                "Cross-filesystem OME-Zarr copy verification failed"
            )
        _replace_destination(partial, destination)
        try:
            shutil.rmtree(source)
        except Exception:
            recover_label_commit(source)
            return "copy-source-retained"
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    return "copy"


def publish_overlay(overlay_path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _replace_destination(overlay_path, destination)
