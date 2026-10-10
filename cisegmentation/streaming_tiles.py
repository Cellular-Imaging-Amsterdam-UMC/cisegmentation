"""Memory-sized tile inference, with ordered results and one label writer."""

from __future__ import annotations

import multiprocessing
import os
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

from .resources import MIB, memory_error, snapshot, tile_size_error


def _extended(core, halos, shape):
    return tuple(
        slice(max(0, start - halo), min(size, end + halo))
        for (start, end), halo, size in zip(core, halos, shape)
    )


def _resources(stats):
    current = snapshot()
    stats["minimum_ram_available"] = min(
        stats["minimum_ram_available"], current.ram_available
    )
    if current.gpu_total:
        stats["minimum_gpu_available"] = min(
            stats["minimum_gpu_available"] or current.gpu_available,
            current.gpu_available,
        )
    return current


def _budget(current, settings):
    budget = current.ram_available * 0.4
    if current.gpu_total and settings.device != "cpu":
        budget = min(
            budget,
            max(0, current.gpu_available - max(512 * MIB, current.gpu_total * 0.15))
            * 0.65,
        )
    return budget


def _fit(queue, shape, halos, spec, settings, scales, stats, *, wait_for_workers=False):
    from .streaming import _estimate_bytes, _split

    while queue:
        core = queue.popleft()
        extended = _extended(core, halos, shape)
        current = _resources(stats)
        estimate = _estimate_bytes(
            tuple(s.stop - s.start for s in extended), spec, scales
        )
        if estimate <= _budget(current, settings):
            return core, extended, current
        if wait_for_workers:
            # Active inference can temporarily occupy most of the budget. Drain
            # the bounded batch before changing the next tile's geometry.
            queue.appendleft(core)
            stats["memory_waits"] = stats.get("memory_waits", 0) + 1
            return None
        children = _split(core)
        if not children:
            queue.appendleft(core)
            raise MemoryError(
                f"{spec.id}: minimum tile plus overlap needs an estimated "
                f"{estimate / MIB:.1f} MiB; live safe budget is "
                f"{_budget(current, settings) / MIB:.1f} MiB. "
                "Reduce overlap or request more resources."
            )
        queue.extendleft(reversed(children))
        stats["pressure_splits"] += 1
    return None


def _retry(core, queue, spec, stats, kind, error, log):
    from .streaming import _split

    children = _split(core)
    if not children:
        message = f"{spec.id} cannot fit a minimum tile including overlap: {error}"
        if kind == "size":
            raise RuntimeError(message)
        raise MemoryError(message)
    queue.extendleft(reversed(children))
    stats["size_limit_retries" if kind == "size" else "oom_retries"] += 1
    if log:
        log(f"Streaming {spec.id}: {kind} retry at {core}; splitting tile")


def _worker_init():
    from .parallel_pipeline import _worker_environment

    _worker_environment()
    os.environ["NUMBA_NUM_THREADS"] = "1"


def _pool(workers):
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init,
    )


class _PeakRSS:
    def __init__(self, children=False):
        import psutil

        self.process = psutil.Process()
        self.children = children
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        import psutil

        while not self.stop.is_set():
            processes = [self.process]
            if self.children:
                processes.extend(self.process.children(recursive=True))
            rss = 0
            for process in processes:
                try:
                    rss += process.memory_info().rss
                except psutil.Error:
                    pass
            self.peak = max(self.peak, rss)
            self.stop.wait(0.02)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def _tile_task(payload):
    """Workers never touch the shared label store or return large arrays via IPC."""
    import torch

    from .parallel_pipeline import _process_gpu_memory_mb
    from .resources import apply_gpu_limit

    try:
        apply_gpu_limit(torch)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with _PeakRSS() as monitor:
            data = np.load(payload["input"], mmap_mode="c", allow_pickle=False)
            predicted, info = payload["segment"](
                data,
                payload["spec"],
                payload["settings"],
                payload["scales"],
                bounded=True,
            )
            predicted = np.asarray(predicted, dtype=np.uint32)
            if predicted.shape != tuple(payload["shape"]):
                raise ValueError(
                    f"Adapter returned {predicted.shape}, expected {payload['shape']}"
                )
            np.save(payload["output"], predicted, allow_pickle=False)
        gpu_mb = allocated_mb = reserved_mb = 0.0
        if (
            str(info.get("device", "cpu")).startswith("cuda")
            and torch.cuda.is_available()
        ):
            allocated_mb = torch.cuda.max_memory_allocated() / MIB
            reserved_mb = torch.cuda.max_memory_reserved() / MIB
            gpu_mb = max(
                max(reserved_mb, allocated_mb) + 512.0, _process_gpu_memory_mb()
            )
        return {
            "ok": True,
            "info": info,
            "rss_mb": monitor.peak / MIB,
            "cuda_mb": gpu_mb,
            "allocated_mb": allocated_mb,
            "reserved_mb": reserved_mb,
        }
    except Exception as exc:
        if not memory_error(exc) and not tile_size_error(exc):
            raise
        return {
            "ok": False,
            "kind": "size" if tile_size_error(exc) else "memory",
            "error": str(exc),
        }


def _worker_count(task_count, profile, settings):
    from .parallel_pipeline import calculate_cpu_workers, calculate_gpu_workers

    workers = calculate_cpu_workers(
        task_count, peak_worker_mb=profile["rss_mb"], cap=settings.max_inference_workers
    )
    if str(profile["info"].get("device", "cpu")).startswith("cuda"):
        import torch

        from .resources import apply_gpu_limit

        # A direct lazy-IO caller may not have imported Torch in the parent.
        # The successful worker probe tells us that CUDA is actually in use.
        apply_gpu_limit(torch)
        current = snapshot()
        workers = min(
            workers,
            calculate_gpu_workers(
                total_mb=current.gpu_total / MIB,
                free_mb=current.gpu_available / MIB,
                peak_worker_mb=profile["cuda_mb"],
                task_count=task_count,
                cap=settings.max_inference_workers,
            ),
        )
    return max(1, workers)


def _serial(image, t, queue, halos, spec, settings, segment, stats, log):
    from .streaming import _cleanup_memory

    while queue:
        core, extended, current = _fit(
            queue, image.data.shape[2:], halos, spec, settings, image.scales, stats
        )
        started = time.perf_counter()
        try:
            data = np.asarray(image.data[(t, slice(None), *extended)])
            read_seconds = time.perf_counter() - started
            predicted, info = segment(data, spec, settings, image.scales, bounded=True)
            predicted = np.asarray(predicted, dtype=np.uint32)
            shape = tuple(s.stop - s.start for s in extended)
            if predicted.shape != shape:
                raise ValueError(
                    f"Adapter returned {predicted.shape}, expected {shape}"
                )
        except Exception as exc:
            if not memory_error(exc) and not tile_size_error(exc):
                raise
            data = None
            exc.__traceback__ = None
            _cleanup_memory()
            _retry(
                core,
                queue,
                spec,
                stats,
                "size" if tile_size_error(exc) else "memory",
                str(exc),
                log,
            )
            continue
        stats["pending_tiles"] = len(queue)
        yield core, extended, predicted, info, read_seconds, current
        del data, predicted


def _can_spawn(segment):
    return (
        segment.__module__ == "cisegmentation.adapters"
        and segment.__name__ == "segment_czyx"
    )


def iter_tile_predictions(
    image,
    t,
    cores,
    halos,
    spec,
    settings,
    path,
    segment,
    stats,
    *,
    allow_parallel=True,
    log=None,
):
    """Keep only a bounded batch in flight; commit masks in geometric order.

    Custom callback adapters keep the original in-process execution semantics.
    The shared production adapter is importable in spawned CUDA-safe workers.
    """
    queue = deque(cores)
    if (
        not allow_parallel
        or settings.max_inference_workers == 1
        or len(queue) < 2
        or not _can_spawn(segment)
    ):
        yield from _serial(image, t, queue, halos, spec, settings, segment, stats, log)
        return
    output_parent = Path(path).resolve().parent
    sequence = 0
    pool = None
    pending = deque()
    probe = None
    workers = 1
    profile = None
    with (
        _PeakRSS(children=True) as parent_monitor,
        tempfile.TemporaryDirectory(
            prefix=".ciseg-tiles-", dir=output_parent
        ) as scratch,
    ):
        scratch = Path(scratch)
        assert scratch.resolve().is_relative_to(output_parent)

        def submit(core, extended, current):
            nonlocal sequence
            sequence += 1
            input_path, output_path = (
                scratch / f"{sequence}.input.npy",
                scratch / f"{sequence}.labels.npy",
            )
            started = time.perf_counter()
            read_seconds = 0.0
            try:
                data = np.asarray(image.data[(t, slice(None), *extended)])
                read_seconds = time.perf_counter() - started
                started = time.perf_counter()
                np.save(input_path, data, allow_pickle=False)
                stats["spool_seconds"] += time.perf_counter() - started
            except Exception as exc:
                if not memory_error(exc):
                    raise
                future = Future()
                future.set_result({"ok": False, "kind": "memory", "error": str(exc)})
                return {
                    "core": core,
                    "extended": extended,
                    "resources": current,
                    "read_seconds": read_seconds,
                    "input": input_path,
                    "output": output_path,
                    "future": future,
                }
            payload = {
                "input": str(input_path),
                "output": str(output_path),
                "segment": segment,
                "spec": spec,
                "settings": settings,
                "scales": image.scales,
                "shape": tuple(s.stop - s.start for s in extended),
            }
            try:
                future = pool.submit(_tile_task, payload)
            except BrokenProcessPool as exc:
                future = Future()
                future.set_result({"ok": False, "kind": "memory", "error": str(exc)})
            return {
                "core": core,
                "extended": extended,
                "resources": current,
                "read_seconds": read_seconds,
                "input": input_path,
                "output": output_path,
                "future": future,
            }

        def discard(item):
            item["input"].unlink(missing_ok=True)
            item["output"].unlink(missing_ok=True)

        def stop_pool(*, abort=False):
            nonlocal pool
            if pool is not None:
                from .process_pool import close_pool

                close_pool(pool, abort=abort)
                pool = None

        try:
            # Select a largest core, then fit its halo to the current job budget.
            # Its prediction is reused at its normal commit position.
            largest = max(
                queue,
                key=lambda core: math_volume(
                    _extended(core, halos, image.data.shape[2:])
                ),
            )
            largest_index = queue.index(largest)
            original = list(queue)
            probe_queue = deque([largest])
            pool = _pool(1)
            while profile is None:
                core, extended, current = _fit(
                    probe_queue,
                    image.data.shape[2:],
                    halos,
                    spec,
                    settings,
                    image.scales,
                    stats,
                )
                item = submit(core, extended, current)
                try:
                    result = item["future"].result()
                except BrokenProcessPool as exc:
                    result = {"ok": False, "kind": "memory", "error": str(exc)}
                if result["ok"]:
                    profile, probe = result, item
                    break
                discard(item)
                _retry(
                    core, probe_queue, spec, stats, result["kind"], result["error"], log
                )
                stop_pool(abort=result["kind"] == "memory")
                stats["worker_restarts"] += 1
                pool = _pool(1)
            # Insert the fitted probe and any unprocessed probe children at their
            # geometric positions. This preserves stitching order after splitting.
            queue = deque(
                [
                    *original[:largest_index],
                    probe["core"],
                    *probe_queue,
                    *original[largest_index + 1 :],
                ]
            )
            stop_pool()
            workers = _worker_count(max(1, len(queue) - 1), profile, settings)
            stats.update(
                workers=workers,
                initial_workers=workers,
                peak_workers=workers,
                probe_peak_rss_mb=profile["rss_mb"],
                probe_peak_cuda_mb=profile["cuda_mb"],
                probe_resources=snapshot().to_dict(),
            )
            if log:
                log(
                    f"Streaming {spec.id}: tile probe RSS={profile['rss_mb']:.1f} MiB, "
                    f"CUDA={profile['cuda_mb']:.1f} MiB; selected {workers} tile worker(s)"
                )
            from .resources import log_resource_check

            log_resource_check(
                log,
                "tile_sizing",
                model=spec.id,
                workers=workers,
                probe_rss_mib=profile["rss_mb"],
                probe_cuda_mib=profile["cuda_mb"],
                effective=stats["probe_resources"],
                blas_threads_per_worker=1,
            )
            pool = _pool(workers)
            while queue or pending:
                while queue and len(pending) < workers:
                    if probe is not None and queue[0] == probe["core"]:
                        queue.popleft()
                        pending.append(probe)
                        probe = None
                        continue
                    try:
                        item = _fit(
                            queue,
                            image.data.shape[2:],
                            halos,
                            spec,
                            settings,
                            image.scales,
                            stats,
                            wait_for_workers=bool(pending),
                        )
                    except MemoryError:
                        if workers == 1:
                            raise
                        # Idle worker models can retain RAM/VRAM. Release them
                        # and retry with fewer workers before declaring failure.
                        stop_pool()
                        workers = max(1, workers // 2)
                        stats["workers"] = workers
                        stats["concurrency_reductions"] += 1
                        stats["worker_restarts"] += 1
                        pool = _pool(workers)
                        if log:
                            log(
                                f"Streaming {spec.id}: live memory pressure; reducing to {workers} tile worker(s)"
                            )
                        continue
                    if item is None:
                        break
                    pending.append(submit(*item))
                item = pending.popleft()
                try:
                    result = item["future"].result()
                except BrokenProcessPool as exc:
                    result = {
                        "ok": False,
                        "kind": "memory",
                        "error": f"Tile worker exited unexpectedly: {exc}",
                    }
                if not result["ok"]:
                    retry_items = [item, *pending]
                    stop_pool(abort=result["kind"] == "memory")
                    pending.clear()
                    queue.extendleft(
                        reversed([retry_item["core"] for retry_item in retry_items])
                    )
                    for retry_item in retry_items:
                        discard(retry_item)
                    if result["kind"] == "memory" and workers > 1:
                        workers = max(1, workers // 2)
                        stats["concurrency_reductions"] += 1
                        stats["oom_retries"] += 1
                        if log:
                            log(
                                f"Streaming {spec.id}: memory retry; reducing to {workers} tile worker(s)"
                            )
                    else:
                        core = queue.popleft()
                        _retry(
                            core,
                            queue,
                            spec,
                            stats,
                            result["kind"],
                            result["error"],
                            log,
                        )
                    stats["workers"] = workers
                    stats["worker_restarts"] += 1
                    if result["kind"] == "memory":
                        if str(profile["info"].get("device", "cpu")).startswith("cuda"):
                            from .parallel_pipeline import _wait_gpu_headroom

                            _wait_gpu_headroom(profile["cuda_mb"], log)
                        workers = min(
                            workers,
                            _worker_count(max(1, len(queue)), profile, settings),
                        )
                        stats["workers"] = workers
                        log_resource_check(
                            log,
                            "tile_retry",
                            workers=workers,
                            error=result["error"],
                            fresh_pool=True,
                            effective=snapshot().to_dict(),
                        )
                    pool = _pool(workers)
                    continue
                predicted = np.load(item["output"], mmap_mode="r", allow_pickle=False)
                stats["peak_worker_rss_mb"] = max(
                    stats["peak_worker_rss_mb"], result["rss_mb"]
                )
                stats["peak_worker_cuda_mb"] = max(
                    stats["peak_worker_cuda_mb"], result["cuda_mb"]
                )
                stats["peak_worker_cuda_allocated_mb"] = max(
                    stats["peak_worker_cuda_allocated_mb"], result["allocated_mb"]
                )
                stats["peak_worker_cuda_reserved_mb"] = max(
                    stats["peak_worker_cuda_reserved_mb"], result["reserved_mb"]
                )
                try:
                    stats["pending_tiles"] = len(queue) + len(pending)
                    yield (
                        item["core"],
                        item["extended"],
                        predicted,
                        result["info"],
                        item["read_seconds"],
                        item["resources"],
                    )
                finally:
                    predicted._mmap.close()
                    discard(item)
        finally:
            stop_pool()
            stats["peak_parallel_rss_mb"] = parent_monitor.peak / MIB


def math_volume(extended):
    result = 1
    for part in extended:
        result *= part.stop - part.start
    return result
