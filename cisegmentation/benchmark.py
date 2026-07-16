from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .adapters import segment_czyx
from .ome_zarr_io import ImageData, write_rgb_gallery
from .registry import ModelSpec, get_model_spec
from .settings import (
    CELL_MODELS,
    FOCI_MODELS,
    STEP1_NUCLEUS_MODELS,
    STEP2_NUCLEUS_MODELS,
    SegmentationSettings,
)


@dataclass(frozen=True)
class BenchmarkCase:
    key: str
    step: str
    spec: ModelSpec
    target: str
    primary_channel: int
    nuclei_channel: int = 0


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


def _foci_target(spec: ModelSpec) -> str:
    if spec.family == "spotiflow":
        return "spots"
    if spec.family == "stardist":
        return "foci"
    return "cells"  # Cellpose 3 bacterial instance masks


def _benchmark_cases(settings: SegmentationSettings) -> list[BenchmarkCase]:
    """Expand every enabled workflow step into all models offered by its UI."""
    cases: list[BenchmarkCase] = []

    def add(
        step: str,
        model_ids: tuple[str, ...],
        target: str,
        primary_channel: int,
        nuclei_channel: int = 0,
    ) -> None:
        for model_id in model_ids:
            spec = get_model_spec(model_id)
            cases.append(
                BenchmarkCase(
                    key=f"{step}|ch{primary_channel}|{model_id}",
                    step=step,
                    spec=spec,
                    target=target,
                    primary_channel=primary_channel,
                    nuclei_channel=nuclei_channel,
                )
            )

    if settings.cell_step:
        if settings.cell_method == "cell-expansion":
            add(
                "Step 1 expansion nuclei",
                STEP1_NUCLEUS_MODELS,
                "nuclei",
                settings.cell_channel,
            )
        else:
            add(
                "Step 1 cells",
                CELL_MODELS,
                "cells",
                settings.cell_channel,
                settings.cell_nuclei_channel,
            )
            if settings.cell_nuclei_channel > 0 and not settings.nucleus_step:
                add(
                    "Step 1 nuclei",
                    STEP1_NUCLEUS_MODELS,
                    "nuclei",
                    settings.cell_nuclei_channel,
                )
    if settings.nucleus_step:
        add(
            "Step 2 nuclei",
            STEP2_NUCLEUS_MODELS,
            "nuclei",
            settings.nucleus_channel,
        )
    for slot, _selected_model, channel in settings.enabled_foci_steps():
        step = f"Step 3{chr(96 + slot)} foci"
        for model_id in FOCI_MODELS:
            spec = get_model_spec(model_id)
            cases.append(
                BenchmarkCase(
                    key=f"{step}|ch{channel}|{model_id}",
                    step=step,
                    spec=spec,
                    target=_foci_target(spec),
                    primary_channel=channel,
                )
            )
    return cases


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
    cases: list[BenchmarkCase],
    raw_by_case: dict[str, np.ndarray],
    runs: list[dict[str, Any]],
    labels_by_case: dict[str, np.ndarray],
) -> np.ndarray:
    panel_w = 280
    sample_raw = raw_by_case[cases[0].key]
    image_h = max(
        140, int(round(panel_w * sample_raw.shape[-2] / sample_raw.shape[-1]))
    )
    header_h, title_h, gap = 38, 42, 12
    panel_h = header_h + image_h
    columns = min(4, len(cases))
    rows = math.ceil(len(cases) / columns)
    block_h = 2 * panel_h + gap
    width = columns * panel_w + (columns + 1) * gap
    height = title_h + rows * block_h + (rows + 1) * gap
    montage = Image.new("RGB", (max(width, 320), height), "white")
    draw = ImageDraw.Draw(montage)
    font = ImageFont.load_default()
    draw.text((gap, 14), "CI Segmentation benchmark", fill="black", font=font)
    runs_by_case = {run["case"]: run for run in runs}
    for index, case in enumerate(cases):
        column = index % columns
        block_row = index // columns
        x = gap + column * (panel_w + gap)
        block_y = title_h + gap + block_row * (block_h + gap)
        raw = raw_by_case[case.key]
        input_panel = _fit_panel(
            Image.fromarray(
                np.repeat(_normalize(raw)[..., None], 3, axis=-1), "RGB"
            ),
            panel_w,
            image_h,
        )
        for panel_row in range(2):
            y = block_y + panel_row * (panel_h + gap)
            draw.rectangle(
                (x, y, x + panel_w - 1, y + panel_h - 1),
                fill=(245, 245, 245),
                outline=(170, 170, 170),
            )
        draw.text(
            (x + 8, block_y + 7),
            f"{case.step} | C{case.primary_channel}",
            fill="black",
            font=font,
        )
        draw.text((x + 8, block_y + 21), case.spec.id, fill="black", font=font)
        montage.paste(input_panel, (x, block_y + header_h))
        run = runs_by_case[case.key]
        output_y = block_y + panel_h + gap
        status = run["status"]
        suffix = ""
        if status == "success":
            suffix = f" | n={run.get('object_count', 0)} | {run.get('runtime_seconds', 0):.1f}s"
        draw.text(
            (x + 8, output_y + 13),
            f"{status}: {case.spec.id}{suffix}",
            fill="black",
            font=font,
        )
        labels = labels_by_case.get(case.key)
        if labels is not None:
            overlay = _label_overlay(
                raw, labels, panel_w, image_h, case.target == "spots"
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
    image: ImageData,
    settings: SegmentationSettings,
    output_dir: str | Path,
    *,
    base_timings: dict[str, float] | None = None,
) -> tuple[Path, bool]:
    settings.validate_steps()
    first_t, crop = center_crop(image.data[0])
    cases = _benchmark_cases(settings)
    runs: list[dict[str, Any]] = []
    labels_by_case: dict[str, np.ndarray] = {}
    raw_by_case: dict[str, np.ndarray] = {}
    failed = False
    for case in cases:
        spec = case.spec
        run_base = {
            "case": case.key,
            "step": case.step,
            "model": spec.id,
            "target": case.target,
            "primary_channel": case.primary_channel,
            "nuclei_channel": case.nuclei_channel,
        }
        primary_index = case.primary_channel - 1
        raw_by_case[case.key] = (
            np.max(first_t[primary_index], axis=0)
            if 0 <= primary_index < first_t.shape[0]
            else np.zeros(first_t.shape[-2:], dtype=first_t.dtype)
        )
        required_channels = [case.primary_channel]
        if case.nuclei_channel > 0:
            required_channels.append(case.nuclei_channel)
        if first_t.shape[0] < spec.min_channels or any(
            channel < 1 or channel > first_t.shape[0]
            for channel in required_channels
        ):
            runs.append(
                {
                    **run_base,
                    "status": "skipped",
                    "reason": (
                        f"selected channel outside input channel count "
                        f"{first_t.shape[0]}"
                    ),
                }
            )
            continue
        model_settings = replace(
            settings,
            model=spec.id,
            target=case.target,
            primary_channel=case.primary_channel,
            nuclei_channel=case.nuclei_channel,
        )
        if spec.family == "cellpose3" and "bact" in spec.checkpoint.lower() and settings.diameter == 0:
            model_settings = replace(model_settings, diameter=-1.0)
        try:
            labels, info = segment_czyx(first_t, spec, model_settings, image.scales)
            labels_by_case[case.key] = labels
            runs.append(
                {
                    **run_base,
                    "status": "success",
                    **info,
                }
            )
        except Exception as exc:
            failed = True
            runs.append(
                {
                    **run_base,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    gallery = _gallery(cases, raw_by_case, runs, labels_by_case)
    timings = dict(base_timings or {})
    for run in runs:
        for key, value in run.get("timings", {}).items():
            timings[key] = timings.get(key, 0.0) + float(value)
    output = Path(output_dir) / f"benchmark_gallery_{image.resource.name}.ome.zarr"
    write_rgb_gallery(
        gallery,
        image,
        {
            "benchmark": True,
            "first_timepoint": 0,
            "crop": crop,
            "runs": runs,
            "timings": timings,
            "parameters": settings.to_dict(),
            "layout": "2d-xy-input-and-segmentation-panels",
        },
        output,
    )
    return output, failed
