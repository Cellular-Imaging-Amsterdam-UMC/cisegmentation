"""Live allocation limits, including Slurm and container memory boundaries."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

MIB = 1024**2
GIB = 1024**3
_THREAD_LIMIT = None
_MAIN_THREAD_LIMIT = None
RESOURCE_LOG_PREFIX = "[CISEGMENTATION_RESOURCE_CHECK]"


def local_gpu_count() -> int | None:
    """Count only GPUs assigned on this node, never a physical GPU inventory."""
    value = os.environ.get("SLURM_GPUS_ON_NODE", "")
    if value.isdigit():
        return int(value)
    for name in ("SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES"):
        value = os.environ.get(name, "").strip()
        if value in ("none", "-1", "NoDevFiles"):
            return 0
        if value and value not in ("all", "none", "-1", "NoDevFiles"):
            return len([part for part in value.split(",") if part.strip()])
    return None


def slurm_memory_limits(cpus: int) -> dict[str, int]:
    """System RAM budgets; --mem-per-gpu is not GPU VRAM."""
    limits = {}
    for name, count in [("SLURM_MEM_PER_NODE", 1), ("SLURM_MEM_PER_CPU", cpus)]:
        value = memory_bytes(os.environ.get(name))
        if value:
            limits[name] = value * count
    per_gpu = memory_bytes(os.environ.get("SLURM_MEM_PER_GPU"))
    if per_gpu:
        count = local_gpu_count()
        if count is None or count < 1:
            raise RuntimeError(
                "SLURM_MEM_PER_GPU is set but the local allocated GPU count is unknown; expose SLURM_GPUS_ON_NODE or CUDA_VISIBLE_DEVICES"
            )
        limits["SLURM_MEM_PER_GPU"] = per_gpu * count
    return limits


def _limit_numerical_threads(threads):
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[name] = str(threads)
    # Spawn imports the CLI before calling an initializer. Load both numerical
    # backends before applying the runtime limiter, not only environment flags.
    import numpy  # noqa: F401
    import scipy.linalg  # noqa: F401
    from threadpoolctl import threadpool_limits

    limiter = threadpool_limits(limits=threads)
    torch = sys.modules.get("torch")
    if torch is not None and hasattr(torch, "set_num_threads"):
        torch.set_num_threads(threads)
    return limiter


def limit_worker_threads():
    """Limit backends already imported by spawn, as well as later imports."""
    global _THREAD_LIMIT
    _THREAD_LIMIT = _limit_numerical_threads(1)


def limit_main_threads():
    """Keep serial/benchmark work inside the job CPU allocation too."""
    global _MAIN_THREAD_LIMIT
    _MAIN_THREAD_LIMIT = _limit_numerical_threads(allocated_cpus())


def log_resource_check(log, event, **values):
    line = (
        RESOURCE_LOG_PREFIX
        + " "
        + json.dumps({"event": event, **values}, sort_keys=True)
    )
    if log is not None:
        log(line)
    else:
        print(line, flush=True)


def configure_database(connection, *, memory_cap_mib=2048, memory_divisor=4):
    """Keep database thread pools inside the same numerical worker budget."""
    current = snapshot(include_gpu=False)
    in_worker = _THREAD_LIMIT is not None
    threads = 1 if in_worker else current.cpus
    cap = min(memory_cap_mib, 256) if in_worker else memory_cap_mib
    memory_mib = max(1, min(cap, current.ram_available // (memory_divisor * MIB)))
    connection.execute(f"SET threads={threads}")
    connection.execute(f"SET memory_limit='{memory_mib}MiB'")
    log_resource_check(
        None,
        "database_budget",
        pid=os.getpid(),
        threads=threads,
        memory_limit_mib=memory_mib,
        in_worker=in_worker,
        effective=current.to_dict(),
    )


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
        "SLURM_STEP_CPUS_PER_TASK",
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
        try:
            quota = int((parent / "cpu.cfs_quota_us").read_text())
            period = int((parent / "cpu.cfs_period_us").read_text())
            if quota > 0 and period > 0:
                candidates.append(max(1, math.floor(quota / period)))
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
            else:
                for mount in (root / "memory", root / "cpu", root / "cpu,cpuacct"):
                    if mount.name in (
                        "cpu",
                        "cpu,cpuacct",
                    ) and "cpu" not in controllers.split(","):
                        continue
                    if mount.name == "memory" and "memory" not in controllers.split(
                        ","
                    ):
                        continue
                    candidate = mount / relative.lstrip("/")
                    if candidate.exists():
                        paths.extend([candidate, *candidate.parents])
        paths.extend(
            mount
            for mount in (root / "memory", root / "cpu", root / "cpu,cpuacct")
            if mount.exists()
        )
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


def snapshot(*, include_gpu=True) -> ResourceSnapshot:
    """Read current headroom, rather than assuming the node equals the job."""
    cpus = allocated_cpus()
    try:
        import psutil

        memory = psutil.virtual_memory()
        limits, available = [int(memory.total)], [int(memory.available)]
    except ImportError:
        limits, available = [4 * GIB], [2 * GIB]
    rss = _job_rss()
    for slurm in slurm_memory_limits(cpus).values():
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
                        usage -= int(
                            statistics.get(
                                "inactive_file",
                                statistics.get("total_inactive_file", 0),
                            )
                        )
                    except (OSError, ValueError):
                        pass
                    limits.append(limit)
                    available.append(max(0, limit - usage))
            except (OSError, ValueError):
                pass
    gpu_total = gpu_available = 0
    try:
        gpu_total, gpu_available = _gpu_budget() if include_gpu else (0, 0)
    except (RuntimeError, AssertionError):
        # Keep CPU/RAM limits intact if the CUDA-visible query fails. Unknown
        # VRAM forces one GPU worker; it never enables a parent-GPU fallback.
        gpu_total = gpu_available = 0
    return ResourceSnapshot(cpus, min(limits), min(available), gpu_total, gpu_available)


def _gpu_budget():
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
    return gpu_total, gpu_available


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
            "cublas_status_alloc_failed",
            "cudnn_status_alloc_failed",
            "cufft_alloc_failed",
        )
    )


def gpu_memory_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return memory_error(exc) and any(
        word in text for word in ("cuda", "cublas", "cudnn", "cufft", "gpu", "hip")
    )


def resource_diagnostics(*, inspect_gpu=True):
    """Whitelisted allocation evidence suitable for Slurm stdout and JSON logs."""
    names = (
        "SLURM_JOB_ID",
        "SLURM_STEP_ID",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_NODELIST",
        "SLURM_CPUS_PER_TASK",
        "SLURM_STEP_CPUS_PER_TASK",
        "SLURM_CPUS_ON_NODE",
        "SLURM_JOB_CPUS_PER_NODE",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
        "SLURM_MEM_PER_GPU",
        "SLURM_GPUS_ON_NODE",
        "SLURM_JOB_GPUS",
        "SLURM_STEP_GPUS",
        "SLURM_GPUS_PER_TASK",
        "SLURM_GPUS_PER_NODE",
        "SLURM_TRES_PER_TASK",
        "SLURM_NTASKS",
        "SLURM_THREADS_PER_CORE",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "CISEGMENTATION_GPU_MEMORY_LIMIT_MB",
    )
    record = {
        "environment": {name: os.environ[name] for name in names if name in os.environ},
        "host_logical_cpus": os.cpu_count(),
        "warnings": [],
        "gpu_discovery": "cuda-visible-device-only",
    }
    try:
        import psutil

        host_memory = psutil.virtual_memory()
        record["host_ram_mib"] = host_memory.total / MIB
        record["host_available_ram_mib"] = host_memory.available / MIB
    except ImportError:
        record["host_ram_mib"] = None
    try:
        record["affinity_cpus"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        record["affinity_cpus"] = None
    record["cgroups"] = []
    for directory in cgroup_directories():
        values = {"path": str(directory)}
        for name in (
            "cpu.max",
            "cpu.cfs_quota_us",
            "cpu.cfs_period_us",
            "cpuset.cpus.effective",
            "memory.max",
            "memory.current",
            "memory.limit_in_bytes",
            "memory.usage_in_bytes",
        ):
            try:
                values[name] = (directory / name).read_text().strip()
            except OSError:
                pass
        if len(values) > 1:
            record["cgroups"].append(values)
    if inspect_gpu:
        try:
            import torch

            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device)
                name, uuid = (
                    str(props.name),
                    str(getattr(props, "uuid", "not-reported")),
                )
                visibility = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                record["gpu"] = {
                    "cuda_index": device,
                    "visible_device_count": torch.cuda.device_count(),
                    "torch_version": getattr(torch, "__version__", "not-reported"),
                    "name": name,
                    "uuid": uuid,
                    "mig": "detected"
                    if "MIG" in (name + uuid + visibility).upper()
                    else "not-reported",
                    "cuda_runtime": getattr(
                        getattr(torch, "version", None), "cuda", None
                    ),
                }
                try:
                    total, free = _gpu_budget()
                    record["gpu"].update(
                        memory_status="available",
                        total_mib=total / MIB,
                        free_mib=free / MIB,
                    )
                except (RuntimeError, AssertionError) as exc:
                    record["gpu"].update(
                        memory_status="unavailable", memory_error=str(exc)
                    )
            else:
                record["gpu"] = {"status": "cuda-unavailable"}
        except Exception as exc:
            record["gpu"] = {"status": "discovery-unavailable", "error": str(exc)}
    try:
        current = snapshot(include_gpu=inspect_gpu)
        record["effective"] = current.to_dict()
        record["effective_mib"] = {
            k: v / MIB for k, v in current.to_dict().items() if k != "cpus"
        }
        record["slurm_ram_limits_mib"] = {
            k: v / MIB for k, v in slurm_memory_limits(current.cpus).items()
        }
        if os.environ.get("SLURM_JOB_ID") and not record["slurm_ram_limits_mib"]:
            record["warnings"].append(
                "No positive Slurm RAM environment budget; verify the logged cgroup limits (--mem=0 means whole-node memory)."
            )
        if inspect_gpu and current.gpu_total == 0:
            record["warnings"].append(
                "GPU memory is unknown; automatic CUDA concurrency will be limited to one worker, with no whole-GPU NVIDIA fallback."
            )
    except Exception as exc:
        record["budget_error"] = str(exc)
    try:
        from threadpoolctl import threadpool_info

        record["blas_pools"] = [
            {k: pool[k] for k in ("internal_api", "num_threads") if k in pool}
            for pool in threadpool_info()
        ]
    except ImportError:
        record["blas_pools"] = []
    return record


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
