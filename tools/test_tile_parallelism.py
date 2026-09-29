"""Compare serial and automatically sized tile workers on an allocated GPU.

Run this checkout in the normal CISegment environment, or bind it over /app in
an existing image. Each case has a separate process so models from the serial
reference cannot consume the parallel run's memory budget.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_MODELS = [
    "stardist:SD_Nuclei_Versatile",
    "cellpose3:transformer_cp3",
    "cellpose3:neurips_cellpose_transformer",
    "cellpose-sam:cpsam_v2",
    "instanseg:single_channel_nuclei",
    "spotiflow:general",
]


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def run_case(args):
    import torch
    import zarr
    from tilescan_resource_smoke import Monitor, validate_output

    from cisegmentation.adapters import segment_czyx
    from cisegmentation.ome_zarr_io import ImageResource, LabelResult, read_image
    from cisegmentation.registry import get_model_spec
    from cisegmentation.resources import apply_gpu_limit, snapshot
    from cisegmentation.settings import SegmentationSettings
    from cisegmentation.streaming import (
        ArrayView,
        infer_streamed,
        raw_array,
        write_streamed_label_groups,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    record = {
        "model": args.case,
        "workers_cap": args.workers,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "status": "failed",
    }
    started = time.perf_counter()
    try:
        assert torch.cuda.is_available(), "CUDA is required; no CPU fallback"
        apply_gpu_limit(torch)
        record["resources"] = snapshot().to_dict()
        record["gpu_name"] = torch.cuda.get_device_name()
        image = read_image(ImageResource(args.input), lazy=True)
        # Keep the large source lazy; only the bounded tile regions are decoded.
        image = replace(image, data=image.data[:1, :1, :1, : args.size, : args.size])
        assert max(image.data.shape[-2:]) > 4096, image.data.shape
        spec = get_model_spec(args.case)
        target = "nuclei" if "nuclei" in spec.targets else spec.targets[0]
        settings = SegmentationSettings(
            model=spec.id,
            target=target,
            primary_channel=1,
            device="cuda",
            dimension_mode="slice-2d",
            tile_size=args.tile_size,
            tile_overlap=96,
            max_inference_workers=args.workers,
        )
        record["settings"] = settings.to_dict()
        record["shape"] = list(image.data.shape)
        with Monitor() as monitor:
            raw_path = args.output / "raw.zarr"
            infos, read_seconds = infer_streamed(
                image,
                spec,
                settings,
                raw_path,
                segment_czyx,
                log=lambda message: print(message, flush=True),
            )
            record.update(infos=infos, read_seconds=read_seconds)
            assert all(info["device"] == "cuda" for info in infos)
            assert zarr.open_group(str(raw_path), mode="r").attrs["complete"]
            stats = infos[0]["streaming"]
            gpu_peak = max(
                stats.get("peak_worker_cuda_allocated_mb", 0),
                torch.cuda.max_memory_allocated() / 1024**2,
            )
            assert gpu_peak > 0, "No CUDA inference allocations recorded"
            overlay = args.output / "labels.ome.zarr"
            view = ArrayView(raw_array(raw_path), "tzyx", "tczyx")
            labels = LabelResult(
                view,
                image,
                spec.id,
                target,
                provenance={"parallel_tile_test": record.copy()},
            )
            name = "labels_" + target
            write_streamed_label_groups(overlay, "", labels, [name], [name])
            record["pyramids"] = validate_output(overlay)
            record["objects"] = infos[0]["object_count"]
            record["status"] = "passed"
        record["monitor"] = monitor.report()
    except Exception as exc:  # noqa: BLE001 - save diagnostics and allow subsequent models to run
        record.update(error=repr(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        record["runtime_seconds"] = time.perf_counter() - started
        save(args.output / "report.json", record)
    return 0 if record["status"] == "passed" else 1


def compare(serial_path, parallel_path):
    import numpy as np

    from cisegmentation.streaming import blocks, raw_array

    serial, parallel = raw_array(serial_path), raw_array(parallel_path)
    assert serial.shape == parallel.shape
    different = foreground_different = intersection = total = 0
    for key in blocks(serial.shape):
        a, b = np.asarray(serial[key]), np.asarray(parallel[key])
        different += int(np.count_nonzero(a != b))
        foreground_different += int(np.count_nonzero((a > 0) != (b > 0)))
        intersection += int(np.count_nonzero((a > 0) & (b > 0)))
        total += int(np.count_nonzero(a) + np.count_nonzero(b))
    return {
        "different_label_pixels": different,
        "different_foreground_pixels": foreground_different,
        "foreground_dice": 2 * intersection / max(1, total),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--size", type=int, default=5120)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--gpu-memory-mb", type=int, default=0)
    parser.add_argument("--timeout-minutes", type=int, default=15)
    parser.add_argument("--case")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.gpu_memory_mb:
        os.environ["CISEGMENTATION_GPU_MEMORY_LIMIT_MB"] = str(args.gpu_memory_mb)
    if args.case:
        return run_case(args)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for model in args.models:
        model_path = args.output / model.replace(":", "_")
        row = {"model": model, "cases": {}}
        for mode, workers in (("serial", 1), ("parallel", 0)):
            case_path = model_path / mode
            case_path.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                __file__,
                "--input",
                str(args.input),
                "--output",
                str(case_path),
                "--case",
                model,
                "--workers",
                str(workers),
                "--size",
                str(args.size),
                "--tile-size",
                str(args.tile_size),
            ]
            print(f"Starting {model} / {mode}", flush=True)
            try:
                with (case_path / "case.log").open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=os.name == "posix",
                    )
                    try:
                        process.wait(timeout=args.timeout_minutes * 60)
                    except subprocess.TimeoutExpired:
                        # Terminate the case and its tile workers, rather than
                        # leaving GPU models alive while the next case starts.
                        import psutil

                        parent = psutil.Process(process.pid)
                        for child in parent.children(recursive=True):
                            try:
                                child.kill()
                            except psutil.NoSuchProcess:
                                pass
                        process.kill()
                        process.wait()
                        raise
                report_path = case_path / "report.json"
                row["cases"][mode] = (
                    json.loads(report_path.read_text())
                    if report_path.exists()
                    else {"status": "failed", "exit_code": process.returncode}
                )
            except subprocess.TimeoutExpired:
                row["cases"][mode] = {"status": "timeout"}
        if all(case["status"] == "passed" for case in row["cases"].values()):
            row["comparison"] = compare(
                model_path / "serial/raw.zarr", model_path / "parallel/raw.zarr"
            )
        results.append(row)
        save(args.output / "summary.json", {"models": results})
        print(
            json.dumps(
                {
                    "model": model,
                    "statuses": [case["status"] for case in row["cases"].values()],
                    "comparison": row.get("comparison"),
                }
            ),
            flush=True,
        )
    return int(
        any(
            case["status"] != "passed"
            for row in results
            for case in row["cases"].values()
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
