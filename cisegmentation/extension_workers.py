"""Bounded CPU batches for geometry and colocalization, with one DB writer.

Workers receive immutable legacy object rows and open only read-only Zarr data.
They never open the parent's DuckDB/SQLite databases. Ordered consumption keeps
geometry IDs and all scientific output deterministic.
"""

from __future__ import annotations

import multiprocessing
import os
import tempfile
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .resources import MIB, ResourceMonitor


def _initialize_worker():
    # Each process is already a CPU worker; nested BLAS pools oversubscribe jobs.
    # Load both numerical backends before limiting them. In a spawned CLI worker
    # SciPy may otherwise load a second BLAS library after the initializer.
    from . import measurement_extensions  # noqa: F401
    from .parallel_pipeline import _worker_environment

    _worker_environment()


def compute_objects(
    image, plane, rows, t, pairs, limits, geometry, locations_only, monitor, spool
):
    from .geometry import mask_polygons
    from .measurement_extensions import colocalization_values

    results = []
    for row in rows:
        oid, value, z, y, x, _size, *bbox = row
        if locations_only:
            anchor = (int(z or 0), int(y), int(x))
            bbox = [*anchor, *(v + 1 for v in anchor)]
        coloc = [
            colocalization_values(
                image, plane, t, value, bbox, pair, limits, monitor=monitor
            )
            for pair in pairs
        ]
        polygons = []
        if geometry and not locations_only:
            z0, y0, x0, z1, y1, x1 = map(int, bbox)
            for zz in range(z0, z1):
                result = mask_polygons(
                    plane,
                    value,
                    zz,
                    (y0, x0, y1, x1),
                    spool,
                    return_bounds=True,
                    monitor=monitor,
                )
                if result is not None:
                    polygons.append((zz, result))
        results.append((oid, coloc, polygons))
    return results


def _compute_batch(task):
    from .ome_zarr_io import read_image, read_native_label
    from .streaming import ChunkCache

    resource, store, name, t, rows, pairs, limits, geometry, locations_only = task
    monitor = ResourceMonitor()
    cache_bytes = min(64 * MIB, max(MIB, monitor.get().ram_available // 64))
    image = read_image(resource, lazy=True)
    labels = None
    try:
        image.data.cache = ChunkCache(cache_bytes)
        labels = read_native_label(resource, name, store_path=store, lazy=True)
        labels.cache = ChunkCache(cache_bytes)
        # Private scratch avoids shared SQLite files on Linux and Windows.
        with tempfile.TemporaryDirectory(prefix="cisegmentation-geometry-") as tmp:
            return compute_objects(
                image,
                labels[t, 0],
                rows,
                t,
                pairs,
                limits,
                geometry,
                locations_only,
                monitor,
                Path(tmp) / "edges.sqlite",
            )
    finally:
        image.data.array.store.close()
        if labels is not None:
            labels.array.store.close()


class ExtensionWorkers:
    """Lazily spawn workers; keep at most two small batches queued per worker."""

    def __init__(self, cap=0):
        from .parallel_pipeline import calculate_cpu_workers

        self.workers = calculate_cpu_workers(10**6, peak_worker_mb=512, cap=cap)
        if os.environ.get("CISEGMENTATION_INLINE_WORKERS") == "1":
            self.workers = 1
        self.executor = None
        self.used_workers = 1
        self.monitor = ResourceMonitor()

    def objects(
        self,
        image,
        plane,
        rows,
        *,
        resource,
        store,
        name,
        t,
        pairs,
        limits,
        geometry,
        locations_only,
        spool,
    ):
        # Small frames cannot amortize spawning/imports. Serial has identical math.
        if (
            self.workers <= 1
            or len(rows) < 256
            or not (pairs or (geometry and not locations_only))
        ):
            for start in range(0, len(rows), 64):
                yield from compute_objects(
                    image,
                    plane,
                    rows[start : start + 64],
                    t,
                    pairs,
                    limits,
                    geometry,
                    locations_only,
                    self.monitor,
                    spool,
                )
            return
        if self.executor is None:
            # A frame cannot keep more processes busy than it has batches.
            # Grow on a later, larger frame rather than spawning idle HPC cores.
            self.pool_workers = min(self.workers, (len(rows) + 63) // 64)
            self.executor = ProcessPoolExecutor(
                max_workers=self.pool_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker,
            )
        elif (
            len(rows) + 63
        ) // 64 > self.pool_workers and self.pool_workers < self.workers:
            self.close()
            self.executor = None
            yield from self.objects(
                image,
                plane,
                rows,
                resource=resource,
                store=store,
                name=name,
                t=t,
                pairs=pairs,
                limits=limits,
                geometry=geometry,
                locations_only=locations_only,
                spool=spool,
            )
            return
        self.used_workers = max(self.used_workers, self.pool_workers)
        pending = deque()
        for start in range(0, len(rows), 64):
            task = (
                resource,
                store,
                name,
                t,
                rows[start : start + 64],
                pairs,
                limits,
                geometry,
                locations_only,
            )
            pending.append(self.executor.submit(_compute_batch, task))
            if len(pending) >= self.pool_workers * 2:
                yield from pending.popleft().result()
        while pending:
            yield from pending.popleft().result()

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
