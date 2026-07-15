from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .adapters import segment_czyx
from .ome_zarr_io import ImageData, write_rgb_gallery
from .registry import (
    MODEL_REGISTRY,
    ModelSpec,
    eligible_benchmark_models,
    get_model_spec,
)
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
        compatible = eligible_benchmark_models(settings.target, channel_count)
        # Spotiflow is intentionally included in every "all" benchmark: it is a
        # point detector, so its panels use the spots target independently.
        spots = [
            spec
            for spec in MODEL_REGISTRY.values()
            if spec.family == "spotiflow" and channel_count >= spec.min_channels
        ]
        return list(dict.fromkeys([*compatible, *spots]))
    return [get_model_spec(model_id) for model_id in selected]


def _normalize(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = np.percentile(finite, (0.5, 99.8))
    if high <= low:
        high = low + 1.0
    return np.clip((image - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def _fit_panel(
    image: Image.Image, width: int, height: int, nearest=False
) -> Image.Image:
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return image.resize((width, height), method)


def _label_overlay(
    raw: np.ndarray, labels: np.ndarray, width: int, height: int, spots: bool
) -> Image.Image:
    base = np.repeat(_normalize(raw)[..., None], 3, axis=-1)
    label_mip = np.max(labels, axis=0).astype(np.uint32)
    if not spots:
        mask = label_mip > 0
        ids = label_mip[mask].astype(np.uint64)
        colors = np.column_stack(
            ((ids * 37) % 205 + 50, (ids * 73) % 205 + 50, (ids * 109) % 205 + 50)
        ).astype(np.uint8)
        base[mask] = (0.42 * base[mask] + 0.58 * colors).astype(np.uint8)
    panel = _fit_panel(Image.fromarray(base, "RGB"), width, height)
    if spots:
        draw = ImageDraw.Draw(panel)
        source_h, source_w = label_mip.shape
        for y, x in np.argwhere(label_mip > 0):
            px = int(round((x + 0.5) * width / source_w))
            py = int(round((y + 0.5) * height / source_h))
            draw.ellipse(
                (px - 4, py - 4, px + 4, py + 4), outline=(255, 60, 60), width=2
            )
            draw.line((px - 5, py, px + 5, py), fill=(255, 255, 0), width=1)
            draw.line((px, py - 5, px, py + 5), fill=(255, 255, 0), width=1)
    return panel


def _gallery(
    raw: np.ndarray,
    specs: list[ModelSpec],
    runs: list[dict[str, Any]],
    labels_by_model: dict[str, np.ndarray],
) -> np.ndarray:
    panel_w = 280
    image_h = max(140, int(round(panel_w * raw.shape[-2] / raw.shape[-1])))
    header_h, title_h, gap = 38, 42, 12
    panel_h = header_h + image_h
    width = len(specs) * panel_w + (len(specs) + 1) * gap
    height = title_h + 2 * panel_h + 3 * gap
    montage = Image.new("RGB", (max(width, 320), height), "white")
    draw = ImageDraw.Draw(montage)
    font = ImageFont.load_default()
    draw.text((gap, 14), "CI Segmentation benchmark", fill="black", font=font)
    input_panel = _fit_panel(
        Image.fromarray(np.repeat(_normalize(raw)[..., None], 3, axis=-1), "RGB"),
        panel_w,
        image_h,
    )
    runs_by_model = {run["model"]: run for run in runs}
    for index, spec in enumerate(specs):
        x = gap + index * (panel_w + gap)
        for row in range(2):
            y = title_h + gap + row * (panel_h + gap)
            draw.rectangle(
                (x, y, x + panel_w - 1, y + panel_h - 1),
                fill=(245, 245, 245),
                outline=(170, 170, 170),
            )
        draw.text(
            (x + 8, title_h + gap + 13), f"Input | {spec.id}", fill="black", font=font
        )
        montage.paste(input_panel, (x, title_h + gap + header_h))
        run = runs_by_model[spec.id]
        output_y = title_h + gap + panel_h + gap
        status = run["status"]
        suffix = ""
        if status == "success":
            suffix = f" | n={run.get('object_count', 0)} | {run.get('runtime_seconds', 0):.1f}s"
        draw.text(
            (x + 8, output_y + 13),
            f"{status}: {spec.id}{suffix}",
            fill="black",
            font=font,
        )
        labels = labels_by_model.get(spec.id)
        if labels is not None:
            overlay = _label_overlay(
                raw, labels, panel_w, image_h, spec.family == "spotiflow"
            )
            montage.paste(overlay, (x, output_y + header_h))
        else:
            reason = str(run.get("reason") or run.get("error") or "No result")
            draw.multiline_text(
                (x + 12, output_y + header_h + 14),
                reason[:180],
                fill=(170, 30, 30),
                font=font,
                spacing=4,
            )
    return np.moveaxis(np.asarray(montage, dtype=np.uint8), -1, 0)


def run_benchmark(
    image: ImageData, settings: SegmentationSettings, output_dir: str | Path
) -> tuple[Path, bool]:
    first_t, crop = center_crop(image.data[0])
    specs = _benchmark_specs(settings, first_t.shape[0])
    runs: list[dict[str, Any]] = []
    labels_by_model: dict[str, np.ndarray] = {}
    failed = False
    for spec in specs:
        is_spotiflow = spec.family == "spotiflow"
        if (not is_spotiflow and settings.target not in spec.targets) or first_t.shape[
            0
        ] < spec.min_channels:
            runs.append(
                {
                    "model": spec.id,
                    "status": "skipped",
                    "reason": "incompatible target or channel count",
                }
            )
            continue
        model_settings = replace(settings, target="spots") if is_spotiflow else settings
        try:
            labels, info = segment_czyx(first_t, spec, model_settings, image.scales)
            labels_by_model[spec.id] = labels
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
    primary = settings.selected_channels(first_t.shape[0])[0]
    raw_mip = np.max(first_t[primary], axis=0)
    gallery = _gallery(raw_mip, specs, runs, labels_by_model)
    output = Path(output_dir) / f"benchmark_gallery_{image.resource.name}.ome.zarr"
    write_rgb_gallery(
        gallery,
        image,
        {
            "benchmark": True,
            "first_timepoint": 0,
            "crop": crop,
            "runs": runs,
            "parameters": settings.to_dict(),
            "layout": "2d-xy-input-and-segmentation-panels",
        },
        output,
    )
    return output, failed
