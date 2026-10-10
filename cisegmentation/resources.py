"""Live allocation limits, including Slurm and container memory boundaries."""

from __future__ import annotations

import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

MIB = 1024**2
GIB = 1024**3


def memory_bytes(value: str | None, *, default_unit: int = MIB) -> int | None:
    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?)B?\s*", value or "", re.IGNORECASE
    )
    if not match:
        return None
    multiplier = {"K": 1024, "M": MIB, "G": GIB, "T": 1024**4}.get(
        match[2].upper(), default_unit
    )
    return int(float(match[1]) * multiplier)


def allocated_cpus() -> int:
    candidates = [os.cpu_count() or 1]
    for name in (
        "SLURM_CPUS_PER_TASK",
        "SLURM_CPUS_ON_NODE",
        "SLURM_JOB_CPUS_PER_NODE",
    ):
        match = re.match(r"\s*(\d+)", os.environ.get(name, ""))
        if match and int(match[1]) > 0:
            candidates.append(int(match[1]))
    try:
        candidates.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    for parent in cgroup_directories():
        try:
            quota, period = (parent / "cpu.max").read_text().split()
            if quota != "max":
                candidates.append(max(1, math.floor(int(quota) / int(period))))
        except (OSError, ValueError):
            pass
    return max(1, min(candidates))


def cgroup_directories() -> list[Path]:
    root = Path("/sys/fs/cgroup")
    if not root.exists():
        return []
    paths = [root]
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            hierarchy, controllers, relative = line.split(":", 2)
            if hierarchy == "0" and not controllers:
                candidate = root / relative.lstrip("/")
                if candidate.exists() and candidate.is_relative_to(root):
                    paths.extend([candidate, *candidate.parents])
            elif "memory" in controllers.split(","):
                candidate = root / "memory" / relative.lstrip("/")
                if candidate.exists():
                    paths.extend([candidate, *candidate.parents])
        if (root / "memory").exists():
            paths.append(root / "memory")
    except (OSError, ValueError):
        pass
    return list(dict.fromkeys(path for path in paths if path.is_relative_to(root)))


def _job_rss() -> int:
    try:
        import psutil

        process = psutil.Process()
        job = os.environ.get("SLURM_JOB_ID")
        if job:
            for parent in process.parents():
                try:
                    if parent.environ().get("SLURM_JOB_ID") != job:
                        break
                    process = parent
                except (psutil.Error, OSError):
                    break
        processes = [process, *process.children(recursive=True)]
        total = 0
        for child in processes:
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        return total
    except (ImportError, OSError):
        return 0


@dataclass(frozen=True)
class ResourceSnapshot:
    cpus: int
    ram_limit: int
    ram_available: int
    gpu_total: int = 0
    gpu_available: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class ResourceMonitor:
    """Refresh headroom periodically rather than scanning processes per object.

    Keep this monitor scoped to one measurement run or worker batch. Large
    allocations still use the measured headroom; it is refreshed every 250 ms.
    """

    def __init__(self, interval=0.25):
        self.interval = interval
        self._value = None
        self._checked = float("-inf")

    def get(self):
        now = time.monotonic()
        if self._value is None or now - self._checked >= self.interval:
            self._value = snapshot()
            self._checked = now
        return self._value


def snapshot() -> ResourceSnapshot:
    """Read current headroom, rather than assuming the node equals the job."""
    cpus = allocated_cpus()
    try:
        import psutil

        memory = psutil.virtual_memory()
        limits, available = [int(memory.total)], [int(memory.available)]
    except ImportError:
        limits, available = [4 * GIB], [2 * GIB]
    rss = _job_rss()
    slurm = memory_bytes(os.environ.get("SLURM_MEM_PER_NODE"))
    per_cpu = memory_bytes(os.environ.get("SLURM_MEM_PER_CPU"))
    if per_cpu is not None:
        slurm = min(slurm, per_cpu * cpus) if slurm else per_cpu * cpus
    if slurm:
        limits.append(slurm)
        available.append(max(0, slurm - rss))
    for directory in cgroup_directories():
        for limit_file, usage_file in (
            ("memory.max", "memory.current"),
            ("memory.limit_in_bytes", "memory.usage_in_bytes"),
        ):
            try:
                limit = int((directory / limit_file).read_text())
                usage = int((directory / usage_file).read_text())
                if 0 < limit < 2**60:
                    # File-backed Zarr chunks are reclaimable page cache, not
                    # permanently occupied inference RAM.
                    try:
                        statistics = dict(
                            line.split()
                            for line in (directory / "memory.stat")
                            .read_text()
                            .splitlines()
                        )
                        usage -= int(statistics.get("inactive_file", 0))
                    except (OSError, ValueError):
                        pass
                    limits.append(limit)
                    available.append(max(0, limit - usage))
            except (OSError, ValueError):
                pass
    gpu_total = gpu_available = 0
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        device = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(device)
        fraction = getattr(
            torch.cuda, "get_per_process_memory_fraction", lambda _device: 1.0
        )(device)
        gpu_total = int(total * fraction)
        gpu_available = min(
            int(free), max(0, gpu_total - int(torch.cuda.memory_reserved(device)))
        )
        # An optional smaller test/user budget never widens the physical allocation.
        cap = memory_bytes(os.environ.get("CISEGMENTATION_GPU_MEMORY_LIMIT_MB"))
        if cap:
            gpu_total = min(gpu_total, cap)
            gpu_available = min(
                gpu_available,
                max(0, gpu_total - int(torch.cuda.memory_reserved(device))),
            )
    return ResourceSnapshot(cpus, min(limits), min(available), gpu_total, gpu_available)


def memory_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, MemoryError) or any(
        phrase in text
        for phrase in (
            "out of memory",
            "cannot allocate memory",
            "can't allocate memory",
            "bad alloc",
            "not enough memory",
            "unable to allocate",
        )
    )


def tile_size_error(exc: BaseException) -> bool:
    """Known per-operation shape limits that smaller inference tiles avoid."""
    return isinstance(exc, RuntimeError) and (
        "quantile() input tensor is too large" in str(exc).lower()
    )


def apply_gpu_limit(torch) -> None:
    """Enforce an explicitly requested smaller budget in every spawned worker."""
    cap = memory_bytes(os.environ.get("CISEGMENTATION_GPU_MEMORY_LIMIT_MB"))
    if cap and torch.cuda.is_available():
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory
        current = getattr(
            torch.cuda, "get_per_process_memory_fraction", lambda _device: 1.0
        )(device)
        torch.cuda.set_per_process_memory_fraction(min(current, cap / total), device)
