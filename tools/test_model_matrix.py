"""Run every real checkpoint in isolated processes; no image build is needed.

Execution and detection quality have separate outcomes. The bundled fluorescence
fixture is not biological ground truth for yeast, bacteria or brightfield models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
import traceback
from collections import Counter
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cisegmentation.registry import MODEL_REGISTRY, get_model_spec


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def make_plan(models=None, *, large_size=8192, quick_only=False, native_3d=True):
    if isinstance(models, str):
        models = [models]
    selected = list(MODEL_REGISTRY) if models is None else list(dict.fromkeys(models))
    cases = []
    for stage in ["small"] + ([] if quick_only else ["large"]):
        for model_id in selected:
            spec = get_model_spec(model_id)
            modes = ["slice-2d"]
            if stage == "small" and native_3d and spec.dimensions == "3d":
                modes.append("native-3d")
            if (
                stage == "large"
                and spec.family == "spotiflow"
                and spec.dimensions == "3d"
            ):
                modes = ["native-3d"]
            for target in spec.targets:
                for mode in modes:
                    slug = model_id.replace(":", "--")
                    cases.append(
                        {
                            "id": f"{stage}--{slug}--{target}--{mode}",
                            "stage": stage,
                            "model": model_id,
                            "target": target,
                            "mode": mode,
                            "xy": 512 if stage == "small" else large_size,
                            "z": (8 if spec.family == "spotiflow" else 16)
                            if mode == "native-3d"
                            else 1,
                        }
                    )
    return cases


def checkpoint_path(spec, root):
    name = {
        "nuclei": "nucleitorch_0",
        "cyto": "cytotorch_0",
        "cyto2": "cyto2torch_0",
    }.get(spec.checkpoint, spec.checkpoint)
    if spec.family == "cellpose3":
        return root / spec.family / name
    if spec.family == "cellpose-sam":
        return root / spec.family / spec.checkpoint
    if spec.family == "instanseg":
        return root / spec.family / spec.checkpoint / "instanseg.pt"
    return root / spec.family / spec.checkpoint


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    files = [
        *root.glob("cisegmentation/*.py"),
        *root.glob("tools/*.py"),
        *root.glob("tools/*.ps1"),
        root / "config.yaml",
    ]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
        if path.is_file()
    }


def checkpoint_inventory(models):
    root = Path(os.environ.get("CISEGMENTATION_MODELS", "/opt/cisegmentation/models"))
    inventory = []
    for model_id in models:
        path = checkpoint_path(get_model_spec(model_id), root)
        files = [path] if path.is_file() else list(path.glob("*.pt"))
        inventory.append(
            {
                "model": model_id,
                "path": str(path),
                "cached": bool(files),
                "weights": [
                    {"name": item.name, "bytes": item.stat().st_size} for item in files
                ],
            }
        )
    return inventory


def prepare_image(case, config):
    """Produce lazy 2D/3D views; never materialize the large input."""
    import numpy as np

    from cisegmentation.ome_zarr_io import ImageResource, read_image

    path = config["fixture"] if case["stage"] == "small" else config["input"]
    image = read_image(ImageResource(Path(path)), lazy=True)
    xy = case["xy"]
    base = image.data[:1, :, :1, :xy, :xy]
    if case["stage"] == "large" and base.shape[-2:] != (xy, xy):
        raise ValueError(f"Requested {xy}x{xy} input, available shape is {base.shape}")
    spec = get_model_spec(case["model"])
    channels = max(base.shape[1], spec.min_channels)
    depth = case["z"]

    class TestView:
        # The source is read lazily; channel repetition and the Gaussian 3D
        # fixture happen only for the requested bounded inference region.
        shape = (1, channels, depth, *base.shape[-2:])
        dtype = base.dtype

        def __getitem__(self, key):
            t, c, z, y, x = key
            if not isinstance(t, int):
                raise IndexError("TestView reads one integer timepoint at a time")
            channel_ids = list(range(channels))[c]
            channel_ids = [channel_ids] if isinstance(channel_ids, int) else channel_ids
            planes = [
                np.asarray(base[t, min(i, base.shape[1] - 1), 0, y, x])
                for i in channel_ids
            ]
            plane = np.stack(planes)
            z_ids = list(range(depth))[z]
            z_ids = [z_ids] if isinstance(z_ids, int) else z_ids
            weights = (
                np.exp(
                    -((np.asarray(z_ids) - depth // 2) ** 2)
                    / (2 * max(1, depth / 8) ** 2)
                )
                if depth > 1
                else np.ones(1)
            )
            result = (plane[:, None] * weights[None, :, None, None]).astype(self.dtype)
            if isinstance(c, int):
                result = result[0]
            if isinstance(z, int):
                result = np.take(result, 0, axis=0 if isinstance(c, int) else 1)
            return result

    return replace(image, data=TestView(), axes=tuple("tczyx")), {
        "source": str(path),
        "source_shape": list(image.data.shape),
        "test_shape_tczyx": list(TestView.shape),
        "repeated_channels": channels > base.shape[1],
        "synthetic_z_volume": depth > 1,
        "biological_accuracy_validated": False,
    }


def compare_labels(eager, tiled, *, points=False):
    import numpy as np

    ref_count = len(np.unique(eager[eager > 0]))
    tiled_count = len(np.unique(tiled[tiled > 0]))
    denominator = np.count_nonzero(eager) + np.count_nonzero(tiled)
    result = {
        "untiled_objects": ref_count,
        "tiled_objects": tiled_count,
        "foreground_dice": 2 * np.count_nonzero((eager > 0) & (tiled > 0)) / denominator
        if denominator
        else None,
    }
    if not ref_count:
        result["quality_status"] = "inconclusive_no_reference_detections"
    elif not tiled_count:
        failure = AssertionError(
            "Tiled result lost every detection from a positive untiled reference"
        )
        result["quality_status"] = "lost_reference_detections"
        failure.comparison = result
        raise failure
    else:
        result["quality_status"] = (
            "agreement_pass" if result["foreground_dice"] >= 0.8 else "review_needed"
        )
        if points:
            from scipy.spatial import cKDTree

            reference, prediction = np.argwhere(eager > 0), np.argwhere(tiled > 0)
            recall = float(np.mean(cKDTree(prediction).query(reference)[0] <= 2))
            precision = float(np.mean(cKDTree(reference).query(prediction)[0] <= 2))
            result.update(
                point_recall_within_2px=recall,
                point_precision_within_2px=precision,
                quality_status="agreement_pass"
                if min(recall, precision) >= 0.8
                else "review_needed",
            )
    return result


def worker(case, config, output):
    import numpy as np
    import torch
    import zarr

    from cisegmentation import adapters
    from cisegmentation.ome_zarr_io import LabelResult
    from cisegmentation.resources import apply_gpu_limit, snapshot
    from cisegmentation.settings import SegmentationSettings
    from cisegmentation.streaming import (
        ArrayView,
        infer_streamed,
        raw_array,
        write_streamed_label_groups,
    )
    from tools.tilescan_resource_smoke import Monitor, validate_output

    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        **case,
        "execution_status": "running",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    try:
        if config.get("gpu_memory_mb"):
            os.environ["CISEGMENTATION_GPU_MEMORY_LIMIT_MB"] = str(
                config["gpu_memory_mb"]
            )
        assert torch.cuda.is_available(), (
            "CUDA is unavailable; CPU fallback is forbidden in this GPU test"
        )
        apply_gpu_limit(torch)
        report.update(
            resources=snapshot().to_dict(),
            gpu_name=torch.cuda.get_device_name(),
            allocator_fraction=torch.cuda.get_per_process_memory_fraction(),
        )
        spec = get_model_spec(case["model"])
        checkpoint = checkpoint_path(
            spec,
            Path(os.environ.get("CISEGMENTATION_MODELS", "/opt/cisegmentation/models")),
        )
        assert checkpoint.exists(), (
            f"Cached checkpoint is missing: {checkpoint}; refusing a hidden replacement/download"
        )
        report["checkpoint"] = str(checkpoint)
        image, fixture = prepare_image(case, config)
        report["fixture"] = fixture
        # Bundled fixture: C1 nuclei, C2 puncta, C3 cytoplasm. A single-channel
        # large input tests execution only, not each checkpoint's training domain.
        channel = (
            2
            if spec.family == "spotiflow" or case["target"] == "foci"
            else 3
            if case["target"] == "cells"
            else 1
        )
        channel = min(channel, image.data.shape[1])
        settings = SegmentationSettings(
            model=spec.id,
            target=case["target"],
            primary_channel=channel,
            nuclei_channel=1 if case["target"] == "cells" and channel != 1 else 0,
            device="cuda",
            dimension_mode="auto" if case["mode"] == "native-3d" else "slice-2d",
            streaming_mode="on",
            tile_size=256 if case["stage"] == "small" else 4096,
            tile_overlap=96,
            tile_depth=8 if case["stage"] == "small" else 32,
            tile_overlap_z=8,
        )
        report["settings"] = settings.to_dict()
        torch.cuda.reset_peak_memory_stats()
        with Monitor() as monitor:
            raw_path = output / "raw-labels.zarr"
            infos, read_seconds = infer_streamed(
                image,
                spec,
                settings,
                raw_path,
                adapters.segment_czyx,
                log=lambda message: print(message, flush=True),
            )
            raw = raw_array(raw_path)
            assert raw.shape == (1, *image.data.shape[2:])
            assert (
                raw.dtype == np.dtype("u4")
                and zarr.open_group(str(raw_path), mode="r").attrs["complete"]
            )
            assert all(info["device"] == "cuda" for info in infos)
            report.update(
                infos=infos,
                read_seconds=read_seconds,
                peak_cuda_allocated=int(torch.cuda.max_memory_allocated()),
                peak_cuda_reserved=int(torch.cuda.max_memory_reserved()),
            )
            report["tiled_peak_cuda_allocated"] = report["peak_cuda_allocated"]
            report["tiled_peak_cuda_reserved"] = report["peak_cuda_reserved"]
            assert report["peak_cuda_allocated"] > 0, (
                "Adapter did not allocate any PyTorch GPU memory"
            )
            loaded = adapters._MODEL_CACHE.get((spec.id, "cuda"))
            report["loaded_checkpoint"] = str(
                getattr(loaded, "pretrained_model", checkpoint)
            )
            if spec.family in {"cellpose3", "cellpose-sam"}:
                actual = getattr(loaded, "pretrained_model", None)
                if actual is not None:
                    assert Path(str(actual)).name == checkpoint.name, (
                        f"Requested {checkpoint.name}, loaded replacement {actual}"
                    )
            report["quality_status"] = (
                "not_evaluated" if case["stage"] == "small" else "large_execution_only"
            )
            if case["stage"] == "small":
                tiled = np.asarray(raw)[0]
                eager, eager_info = adapters.segment_czyx(
                    np.asarray(image.data[0, :, :, :, :]), spec, settings, image.scales
                )
                assert eager.shape == tiled.shape
                report["untiled_info"] = eager_info
                np.savez_compressed(
                    output / "comparison.npz", tiled=tiled, untiled=eager
                )
                try:
                    report.update(
                        compare_labels(eager, tiled, points=spec.family == "spotiflow")
                    )
                except AssertionError as exc:
                    report.update(getattr(exc, "comparison", {}))
                    raise
            else:
                view = ArrayView(raw, "tzyx", "tczyx")
                overlay = output / "labels.ome.zarr"
                result = LabelResult(
                    view,
                    image,
                    spec.id,
                    case["target"],
                    provenance={"test_case": case, "infos": infos},
                    channel_labels=[case["target"]],
                )
                name = "labels_" + case["target"]
                write_streamed_label_groups(overlay, "", result, [name], [name])
                report["pyramids"] = validate_output(overlay)
                report["label_overlay"] = str(overlay)
                report["tiled_objects"] = infos[0]["object_count"]
                report["overlay_note"] = (
                    "Labels-only diagnostic overlay; merge with input pixels for standalone viewing."
                )
                if not report["tiled_objects"]:
                    report["quality_status"] = "inconclusive_no_detections"
            report["execution_status"] = "passed"
            report["peak_cuda_allocated"] = int(torch.cuda.max_memory_allocated())
            report["peak_cuda_reserved"] = int(torch.cuda.max_memory_reserved())
        report["monitor"] = monitor.report()
        print(
            f"CASE PASS: {case['id']}; quality={report['quality_status']}", flush=True
        )
    except Exception as exc:  # noqa: BLE001 - persist any model failure so the next case can run
        report.update(
            execution_status="failed", error=repr(exc), traceback=traceback.format_exc()
        )
        if "monitor" in locals():
            report["monitor"] = monitor.report()
        traceback.print_exc()
    finally:
        report["runtime_seconds"] = time.perf_counter() - started
        save_json(output / "report.json", report)
    return 0 if report["execution_status"] == "passed" else 1


def terminate_tree(process):
    # Each case has its own session on Linux: kill only this case, including
    # subprocesses, rather than unrelated Slurm jobs or the coordinator.
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        import psutil

        try:
            parent = psutil.Process(process.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except psutil.Error:
            pass
    process.wait()


def run_isolated(command, log_path, timeout_seconds):
    with Path(log_path).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        try:
            return process.wait(timeout=timeout_seconds), False
        except subprocess.TimeoutExpired:
            terminate_tree(process)
            return process.returncode, True
        except BaseException:
            terminate_tree(process)
            raise


CSV_FIELDS = [
    "id",
    "model",
    "target",
    "mode",
    "stage",
    "execution_status",
    "quality_status",
    "runtime_seconds",
    "tiled_objects",
    "untiled_objects",
    "foreground_dice",
    "point_recall_within_2px",
    "point_precision_within_2px",
    "peak_process_tree_rss",
    "peak_cuda_allocated",
    "peak_cuda_reserved",
    "tiles",
    "oom_retries",
    "size_limit_retries",
    "pressure_splits",
    "error",
    "report",
]


def summary_row(report, path):
    row = {key: report.get(key) for key in CSV_FIELDS}
    row["report"] = str(path)
    row["peak_process_tree_rss"] = report.get("monitor", {}).get(
        "peak_process_tree_rss"
    )
    info = (report.get("infos") or [{}])[0].get("streaming", {})
    row.update(
        {key: info.get(key) for key in ("tiles", "oom_retries", "size_limit_retries", "pressure_splits")}
    )
    return row


def publish_summary(output, rows, cases, *, finished=False):
    counts = Counter(row["execution_status"] for row in rows)
    summary = {
        "status": "completed" if finished else "running",
        "total_cases": len(cases),
        "completed_cases": len(rows),
        "execution_counts": dict(counts),
        "quality_counts": dict(Counter(row["quality_status"] for row in rows)),
        "cases": rows,
    }
    if finished and counts.get("failed", 0) + counts.get("timeout", 0):
        summary["status"] = "completed_with_failures"
    save_json(output / "summary.json", summary)
    temporary = output / "summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "summary.csv")
    return summary


def collect_evidence(output):
    # Do not package multi-gigabyte label stores. Small comparisons, reports and
    # all logs are portable; large overlays remain in the shared Docker volume.
    destination = output / "evidence.tar.gz"
    temporary = output / "evidence.tar.gz.tmp"
    with tarfile.open(temporary, "w:gz") as archive:
        for current, directories, files in os.walk(output):
            directories[:] = [
                name for name in directories if not name.endswith(".zarr")
            ]
            for name in files:
                path = Path(current) / name
                if path.suffix in {".json", ".csv", ".log", ".npz"}:
                    archive.add(path, arcname=str(path.relative_to(output)))
    temporary.replace(destination)


def coordinate(config, config_path, *, plan_only=False, retry_failures=False):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    cases = make_plan(
        config.get("models"),
        large_size=config.get("large_size", 8192),
        quick_only=config.get("quick_only", False),
        native_3d=config.get("native_3d", True),
    )
    models = list(dict.fromkeys(case["model"] for case in cases))
    inventory = checkpoint_inventory(models)
    save_json(
        output / "plan.json",
        {
            "models": models,
            "total_cases": len(cases),
            "cases": cases,
            "checkpoint_inventory": inventory,
            "missing_checkpoints": [
                item["model"] for item in inventory if not item["cached"]
            ],
        },
    )
    if plan_only:
        print(
            f"PLAN: {len({case['model'] for case in cases})} checkpoints, {len(cases)} cases",
            flush=True,
        )
        return 0
    manifest_path = output / "run-manifest.json"
    manifest = {
        "config": config,
        "source_hashes": source_hashes(),
        "python": sys.version,
        "sessions": [],
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            previous["config"] != config
            or previous["source_hashes"] != manifest["source_hashes"]
        ):
            raise ValueError(
                "Existing run has different code/configuration; create a new output directory"
            )
        manifest = previous
    manifest["sessions"].append(
        {"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "started_at": time.time()}
    )
    save_json(manifest_path, manifest)
    rows = []
    publish_summary(output, rows, cases)
    for index, case in enumerate(cases, 1):
        directory = output / "cases" / case["id"]
        attempts = sorted(directory.glob("attempt-*"))
        previous = attempts[-1] / "report.json" if attempts else None
        if previous and previous.exists():
            report = json.loads(previous.read_text(encoding="utf-8"))
            if report["execution_status"] == "passed" or not retry_failures:
                rows.append(summary_row(report, previous))
                publish_summary(output, rows, cases)
                continue
        attempt = directory / f"attempt-{len(attempts) + 1:03}"
        attempt.mkdir(parents=True)
        save_json(attempt / "case.json", case)
        print(f"[{index}/{len(cases)}] {case['id']}", flush=True)
        save_json(
            output / "current-case.json",
            {**case, "index": index, "total": len(cases), "started_at": time.time()},
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(config_path),
            "--worker",
            str(attempt / "case.json"),
            "--worker-output",
            str(attempt),
        ]
        started = time.monotonic()
        exit_code, timeout = run_isolated(
            command, attempt / "case.log", config.get("timeout_minutes", 30) * 60
        )
        path = attempt / "report.json"
        if path.exists() and not timeout:
            report = json.loads(path.read_text(encoding="utf-8"))
            if exit_code != 0:
                report["execution_status"] = "failed"
        else:
            report = {
                **case,
                "execution_status": "timeout" if timeout else "failed",
                "error": "Case timeout"
                if timeout
                else f"Worker exited {exit_code} without a report (see case.log)",
                "runtime_seconds": time.monotonic() - started,
            }
        save_json(path, report)
        rows.append(summary_row(report, path))
        publish_summary(output, rows, cases)
        print(
            f"  {report['execution_status']}: {report.get('quality_status', report.get('error', ''))}",
            flush=True,
        )
    summary = publish_summary(output, rows, cases, finished=True)
    collect_evidence(output)
    print(
        json.dumps({key: value for key, value in summary.items() if key != "cases"}),
        flush=True,
    )
    return int(summary["status"] == "completed_with_failures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if args.worker:
        return worker(
            json.loads(args.worker.read_text(encoding="utf-8")),
            config,
            args.worker_output,
        )
    if args.collect_only:
        collect_evidence(Path(config["output"]))
        return 0
    return coordinate(
        config,
        args.config,
        plan_only=args.plan_only,
        retry_failures=args.retry_failures,
    )


if __name__ == "__main__":
    raise SystemExit(main())
