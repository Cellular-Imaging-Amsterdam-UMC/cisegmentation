from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import segment_czyx
from .ome_zarr_io import ImageData, LabelResult, write_label_image
from .registry import ModelSpec, eligible_benchmark_models, get_model_spec
from .settings import SegmentationSettings, parse_model_selection


def center_crop(
    image: np.ndarray, max_xy: int = 1024
) -> tuple[np.ndarray, dict[str, int]]:
    height, width = image.shape[-2:]
    crop_h, crop_w = min(height, max_xy), min(width, max_xy)
    y0, x0 = (height - crop_h) // 2, (width - crop_w) // 2
    return image[..., y0 : y0 + crop_h, x0 : x0 + crop_w], {
        "x": x0,
        "y": y0,
        "width": crop_w,
        "height": crop_h,
    }


def _benchmark_specs(
    settings: SegmentationSettings, channel_count: int
) -> list[ModelSpec]:
    selected = parse_model_selection(settings.benchmark_models)
    if not selected or selected == ["all"]:
        return eligible_benchmark_models(settings.target, channel_count)
    return [get_model_spec(model_id) for model_id in selected]


def run_benchmark(
    image: ImageData, settings: SegmentationSettings, output_dir: str | Path
) -> tuple[Path, bool]:
    first_t, crop = center_crop(image.data[0])
    channel_labels: list[str] = []
    model_labels: list[np.ndarray] = []
    runs: list[dict[str, Any]] = []
    failed = False
    for spec in _benchmark_specs(settings, first_t.shape[0]):
        if settings.target not in spec.targets or first_t.shape[0] < spec.min_channels:
            runs.append(
                {
                    "model": spec.id,
                    "status": "skipped",
                    "reason": "incompatible target or channel count",
                }
            )
            continue
        try:
            labels, info = segment_czyx(first_t, spec, settings, image.scales)
            model_labels.append(labels)
            channel_labels.append(spec.id)
            runs.append({"model": spec.id, "status": "success", **info})
        except Exception as exc:
            failed = True
            runs.append(
                {
                    "model": spec.id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if not model_labels:
        raise RuntimeError("No benchmark model produced a segmentation")
    # C,Z,Y,X -> T,C,Z,Y,X. Each model is one label channel in the gallery.
    labels = np.stack(model_labels, axis=0)[None].astype(np.uint32)
    cropped_source = replace(image, data=first_t[None])
    result = LabelResult(
        labels,
        cropped_source,
        "benchmark-gallery",
        settings.target,
        {
            "benchmark": True,
            "first_timepoint": 0,
            "crop": crop,
            "runs": runs,
            "parameters": settings.to_dict(),
        },
        channel_labels=channel_labels,
    )
    output = Path(output_dir) / f"benchmark_gallery_{image.resource.name}.ome.zarr"
    write_label_image(result, output)
    return output, failed
