"""Reproducible direct-run tests; use an existing image with this checkout mounted.

The optional GPU limit is a real PyTorch allocator cap, inherited by workers.
It simulates a smaller card; it does not change the physical GPU's capacity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import psutil
import zarr

from cisegmentation.adapters import clear_model_cache, segment_czyx
from cisegmentation.engine import run_workflow
from cisegmentation.ome_zarr_io import ImageResource, read_image
from cisegmentation.registry import get_model_spec
from cisegmentation.resources import apply_gpu_limit, snapshot
from cisegmentation.settings import SegmentationSettings
from cisegmentation.streaming import infer_streamed, raw_array


class Monitor:
    def __init__(self):
        self.stop = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        process = psutil.Process()
        while not self.stop.is_set():
            rss = 0
            for child in [process, *process.children(recursive=True)]:
                try:
                    rss += child.memory_info().rss
                except psutil.Error:
                    pass
            resource = snapshot().to_dict()
            torch = sys.modules.get("torch")
            cuda = (
                int(torch.cuda.memory_reserved())
                if torch and torch.cuda.is_available()
                else 0
            )
            self.samples.append(
                {"time": time.time(), "rss": rss, "cuda_reserved": cuda, **resource}
            )
            self.stop.wait(2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()

    def report(self):
        return {
            "peak_process_tree_rss": max((s["rss"] for s in self.samples), default=0),
            "peak_parent_cuda_reserved": max(
                (s["cuda_reserved"] for s in self.samples), default=0
            ),
            "samples": self.samples,
        }


def validate_output(path):
    root = zarr.open_group(str(path), mode="r")
    result = {}
    for name in root["labels"].attrs["labels"]:
        group = root[f"labels/{name}"]
        multiscale = group.attrs["multiscales"][0]
        arrays = [group[level["path"]] for level in multiscale["datasets"]]
        shapes = [list(array.shape) for array in arrays]
        assert all(array.dtype == np.dtype("uint32") for array in arrays)
        assert all(
            a.shape[-2] >= b.shape[-2] and a.shape[-1] >= b.shape[-1]
            for a, b in pairwise(arrays)
        )
        assert len(shapes) >= 8 if arrays[0].shape[-2:] == (40000, 40000) else True
        # Nearest-neighbor pyramids preserve IDs, including across chunk seams.
        for level, array in enumerate(arrays[1:], 1):
            key = (0,) * (array.ndim - 2) + (slice(0, 32), slice(0, 32))
            base_key = (0,) * (array.ndim - 2) + (
                slice(0, 32 * 2**level, 2**level),
            ) * 2
            np.testing.assert_array_equal(array[key], arrays[0][base_key])
        result[name] = {"shapes": shapes, "dtype": str(arrays[0].dtype)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case", choices=("full", "families", "stress"), default="full"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--measurements", choices=("skip", "duckdb", "sqlite"), default="skip"
    )
    parser.add_argument("--gpu-memory-mb", type=int, default=0)
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    if args.gpu_memory_mb:
        os.environ["CISEGMENTATION_GPU_MEMORY_LIMIT_MB"] = str(args.gpu_memory_mb)
    import torch

    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA is unavailable"
        apply_gpu_limit(torch)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "input": str(args.input),
        "case": args.case,
        "device": args.device,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "resources": snapshot().to_dict(),
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "allocator_fraction": torch.cuda.get_per_process_memory_fraction()
        if torch.cuda.is_available()
        else None,
    }
    started = time.perf_counter()
    try:
        with Monitor() as monitor:
            if args.case == "full":
                outputs = run_workflow(
                    args.input,
                    args.output,
                    SegmentationSettings(
                        cell_model="skip",
                        nucleus_model="stardist:SD_Nuclei_Versatile",
                        nucleus_channel=1,
                        device=args.device,
                        include_original_data=False,
                        measurements_database=args.measurements,
                        max_inference_workers=1,
                        max_measurement_workers=1,
                        remove_border_cells=False,
                    ),
                    log=lambda message: print(message, flush=True),
                )
                report["outputs"] = [str(path) for path in outputs]
                report["labels"] = validate_output(outputs[0])
            elif args.case == "stress":
                from dataclasses import replace

                image = read_image(ImageResource(args.input), lazy=True)
                image = replace(image, data=image.data[:, :, :1, :8192, :8192])
                spec = get_model_spec("stardist:SD_Nuclei_Versatile")
                settings = SegmentationSettings(
                    model=spec.id,
                    target="nuclei",
                    device=args.device,
                    tile_size=8192,
                    tile_overlap=96,
                )
                infos, _ = infer_streamed(
                    image,
                    spec,
                    settings,
                    args.output / "stress-raw.zarr",
                    segment_czyx,
                    log=print,
                )
                report["infos"] = infos
                assert (
                    infos[0]["streaming"]["pressure_splits"]
                    + infos[0]["streaming"]["oom_retries"]
                    > 0
                )
            else:
                image = read_image(ImageResource(args.input), lazy=True)
                # Native model execution for each family, with several overlapping cores.
                model_ids = [
                    "cellpose3:nuclei",
                    "cellpose-sam:cpsam_v2",
                    "stardist:SD_Nuclei_Versatile",
                    "instanseg:single_channel_nuclei",
                    "spotiflow:general",
                    "spotiflow:smfish_3d",
                ]
                model_ids = args.models or model_ids
                report["models"] = {}
                for model_id in model_ids:
                    spec = get_model_spec(model_id)
                    # Retain source metadata; bound this smoke comparison to a 512px ROI.
                    from dataclasses import replace

                    roi = replace(
                        image,
                        data=image.data[
                            :, :, : min(image.data.shape[2], 4), :512, :512
                        ],
                    )
                    if spec.dimensions == "3d" and spec.family == "spotiflow":
                        from copy import deepcopy

                        volume_path = args.output / "volume-fixture.ome.zarr"
                        volume = zarr.open_group(str(volume_path), mode="w")
                        data = np.asarray(roi.data)
                        data = np.repeat(data, 8, axis=2)
                        weights = np.exp(
                            -((np.arange(data.shape[2]) - data.shape[2] // 2) ** 2) / 2
                        )
                        data = (data * weights[None, None, :, None, None]).astype(
                            data.dtype
                        )
                        volume.create_dataset(
                            "0",
                            data=data,
                            chunks=(1, 1, 1, 128, 128),
                        )
                        metadata = deepcopy(image.attrs)
                        multiscale = metadata["multiscales"][0]
                        multiscale["axes"] = [
                            {
                                "name": axis,
                                "type": "channel"
                                if axis == "c"
                                else "time"
                                if axis == "t"
                                else "space",
                            }
                            for axis in "tczyx"
                        ]
                        multiscale["datasets"] = [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {
                                        "type": "scale",
                                        "scale": [
                                            image.scales.get(axis, 1)
                                            for axis in "tczyx"
                                        ],
                                    }
                                ],
                            }
                        ]
                        volume.attrs.update(metadata)
                        roi = read_image(ImageResource(volume_path), lazy=True)
                    settings = SegmentationSettings(
                        model=model_id,
                        target=spec.targets[0],
                        primary_channel=2
                        if spec.family == "spotiflow" and roi.data.shape[1] > 1
                        else 1,
                        device=args.device,
                        tile_size=256,
                        tile_overlap=96,
                        tile_depth=2,
                        tile_overlap_z=1,
                        dimension_mode="slice-2d" if roi.data.shape[2] == 1 else "auto",
                    )
                    path = args.output / (model_id.replace(":", "-") + ".zarr")
                    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
                    infos, _ = infer_streamed(
                        roi, spec, settings, path, segment_czyx, log=print
                    )
                    labels = np.asarray(raw_array(path))
                    assert labels.shape == (roi.data.shape[0], *roi.data.shape[2:])
                    assert infos[0]["device"] == args.device
                    report["models"][model_id] = {
                        "infos": infos,
                        "max_id": int(labels.max()),
                        "peak_cuda_reserved": int(torch.cuda.max_memory_reserved())
                        if torch.cuda.is_available()
                        else 0,
                    }
                    eager, _ = segment_czyx(
                        np.asarray(roi.data[0]), spec, settings, roi.scales
                    )
                    intersection = np.count_nonzero((eager > 0) & (labels[0] > 0))
                    denominator = np.count_nonzero(eager) + np.count_nonzero(labels[0])
                    report["models"][model_id].update(
                        {
                            "untiled_object_count": int(
                                len(np.unique(eager)) - int(np.any(eager == 0))
                            ),
                            "foreground_dice": 2 * intersection / denominator
                            if denominator
                            else 1,
                        }
                    )
                    assert report["models"][model_id]["untiled_object_count"] > 0, (
                        f"{model_id}: this fixture is not a positive control"
                    )
                    assert np.any(labels), (
                        f"{model_id}: tiled prediction lost all reference detections"
                    )
                    if spec.family == "spotiflow":
                        from scipy.spatial import cKDTree

                        reference = np.argwhere(eager > 0)
                        predicted = np.argwhere(labels[0] > 0)
                        # Local normalization can move point predictions by a
                        # voxel; pixel Dice alone is inappropriate for spots.
                        recall = float(
                            np.mean(cKDTree(predicted).query(reference)[0] <= 2)
                        )
                        precision = float(
                            np.mean(cKDTree(reference).query(predicted)[0] <= 2)
                        )
                        report["models"][model_id]["point_recall_within_2px"] = recall
                        report["models"][model_id]["point_precision_within_2px"] = (
                            precision
                        )
                        assert min(recall, precision) >= 0.8, (
                            f"{model_id}: tiled points disagree with the positive reference"
                        )
                    print(f"MODEL PASS: {model_id}", flush=True)
                    (args.output / "model-progress.json").write_text(
                        json.dumps(report, indent=2)
                    )
                    clear_model_cache()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        raise
    finally:
        report["runtime_seconds"] = time.perf_counter() - started
        if "monitor" in locals():
            report["monitor"] = monitor.report()
        (args.output / "resource-report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
