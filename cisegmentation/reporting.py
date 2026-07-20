from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any

import numpy as np


Emitter = Callable[[str], None] | None


def emit(consumer: Emitter, message: str) -> None:
    if consumer is not None:
        consumer(message)


def label_statistics(labels: np.ndarray, scales: dict[str, float]) -> dict[str, Any]:
    """Return JSON-safe instance counts and size statistics for a label image."""
    array = np.asarray(labels)
    foreground = array[array > 0]
    total_elements = int(array.size)
    if foreground.size:
        _ids, sizes = np.unique(foreground, return_counts=True)
        sizes = sizes.astype(np.float64, copy=False)
    else:
        sizes = np.empty(0, dtype=np.float64)
    result: dict[str, Any] = {
        "label_count": int(sizes.size),
        "foreground_elements": int(foreground.size),
        "foreground_percent": (
            100.0 * float(foreground.size) / total_elements if total_elements else 0.0
        ),
        "mean_size_elements": float(sizes.mean()) if sizes.size else 0.0,
        "median_size_elements": float(np.median(sizes)) if sizes.size else 0.0,
        "min_size_elements": int(sizes.min()) if sizes.size else 0,
        "max_size_elements": int(sizes.max()) if sizes.size else 0,
    }
    x = float(scales.get("x", float("nan")))
    y = float(scales.get("y", float("nan")))
    z = float(scales.get("z", float("nan")))
    depth = int(array.shape[-3]) if array.ndim >= 3 else 1
    result["element_unit"] = "voxels" if depth > 1 else "pixels"
    xy_known = np.isfinite(x) and x > 0 and np.isfinite(y) and y > 0
    physical_size_known = xy_known and (
        depth == 1 or (np.isfinite(z) and z > 0)
    )
    if physical_size_known:
        element_size = x * y
        unit = "um^2"
        if depth > 1:
            element_size *= z
            unit = "um^3"
        mean_physical = float(sizes.mean() * element_size) if sizes.size else 0.0
        median_physical = (
            float(np.median(sizes) * element_size) if sizes.size else 0.0
        )
        median_diameter = (
            2.0 * (3.0 * median_physical / (4.0 * math.pi)) ** (1.0 / 3.0)
            if depth > 1
            else 2.0 * math.sqrt(median_physical / math.pi)
        )
        result.update(
            {
                "physical_size_unit": unit,
                "mean_physical_size": mean_physical,
                "median_physical_size": median_physical,
                "min_physical_size": float(sizes.min() * element_size)
                if sizes.size
                else 0.0,
                "max_physical_size": float(sizes.max() * element_size)
                if sizes.size
                else 0.0,
                "median_equivalent_diameter_um": median_diameter,
            }
        )
    return result


def format_label_statistics(
    statistics: dict[str, Any], *, locations_only: bool = False
) -> str:
    if locations_only:
        return f"locations={statistics['label_count']}"
    if "mean_physical_size" in statistics:
        size_text = (
            f"size {statistics['physical_size_unit']} mean/median/min/max="
            f"{statistics['mean_physical_size']:.2f}/"
            f"{statistics['median_physical_size']:.2f}/"
            f"{statistics['min_physical_size']:.2f}/"
            f"{statistics['max_physical_size']:.2f}, "
            f"median equivalent diameter="
            f"{statistics['median_equivalent_diameter_um']:.2f} um"
        )
    else:
        size_text = (
            f"size {statistics.get('element_unit', 'pixels')} mean/median/min/max="
            f"{statistics['mean_size_elements']:.1f}/"
            f"{statistics['median_size_elements']:.1f}/"
            f"{statistics['min_size_elements']}/"
            f"{statistics['max_size_elements']}"
        )
    return (
        f"labels={statistics['label_count']}, "
        f"foreground={statistics['foreground_percent']:.2f}%, {size_text}"
    )


def step_record(
    *,
    step: str,
    timepoint: int,
    model: str,
    target: str,
    primary_channel: int,
    nuclei_channel: int,
    labels: np.ndarray,
    info: dict[str, Any],
    scales: dict[str, float],
) -> dict[str, Any]:
    return {
        "step": step,
        "timepoint": int(timepoint),
        "model": model,
        "target": target,
        "primary_channel": int(primary_channel),
        "nuclei_channel": int(nuclei_channel),
        "device": info.get("device"),
        "device_name": info.get("device_name"),
        "dimension_mode": info.get("dimension_mode"),
        "runtime_seconds": float(info.get("runtime_seconds", 0.0)),
        "model_cache_hit": bool(info.get("model_cache_hit")),
        "timings": {
            key: float(value) for key, value in info.get("timings", {}).items()
        },
        "effective_parameters": dict(info.get("effective_parameters", {})),
        "label_statistics": label_statistics(labels, scales),
    }


def format_effective_parameters(parameters: dict[str, Any]) -> str | None:
    adapter = parameters.get("adapter")
    if adapter == "cellpose":
        diameter_pixels = parameters.get("diameter_pixels")
        diameter_um = parameters.get("diameter_um")
        diameter = (
            f"{diameter_um:.2f} um / {diameter_pixels:.2f} px"
            if diameter_um is not None and diameter_pixels is not None
            else f"{diameter_pixels:.2f} px"
            if diameter_pixels is not None
            else "model default (value unavailable)"
        )
        return (
            f"effective: diameter={diameter} ({parameters.get('diameter_source')}), "
            f"cell probability={parameters.get('cellprob_threshold'):g}, "
            f"flow={parameters.get('flow_threshold'):g}"
        )
    if adapter == "stardist":
        source_shape = parameters.get("source_yx_shape") or []
        model_shape = parameters.get("model_yx_shape") or []
        rescaled = source_shape != model_shape
        rescale = "none"
        if rescaled:
            rescale = (
                f"YxX={parameters.get('rescale_y_factor'):.3f}x/"
                f"{parameters.get('rescale_x_factor'):.3f}x, "
                f"{source_shape[0]}x{source_shape[1]} -> {model_shape[0]}x{model_shape[1]}"
            )
            if parameters.get("model_y_um") is not None and parameters.get("model_x_um") is not None:
                rescale += (
                    f", model pixel={parameters['model_y_um']:.3f}x"
                    f"{parameters['model_x_um']:.3f} um"
                )
            rescale += ", labels restored to source grid"
        return (
            f"effective: probability={parameters.get('probability_threshold'):g} "
            f"({parameters.get('probability_source')}), NMS={parameters.get('nms_threshold'):g} "
            f"({parameters.get('nms_source')}), rescale={rescale}"
        )
    if adapter == "spotiflow":
        return (
            f"effective: probability={parameters.get('probability_threshold'):g} "
            f"({parameters.get('probability_source')}), minimum distance="
            f"{parameters.get('minimum_distance_pixels')} px / "
            f"{parameters.get('minimum_distance_um'):.3f} um"
        )
    if adapter == "instanseg":
        return f"effective: model pixel size={parameters.get('pixel_size_um'):.3f} um"
    return None


def format_step_record(record: dict[str, Any]) -> list[str]:
    channels = f"C{record['primary_channel']}"
    if record.get("nuclei_channel", 0) > 0 and record["nuclei_channel"] != record["primary_channel"]:
        channels += f" + nucleus C{record['nuclei_channel']}"
    device = str(record.get("device") or "unknown").upper()
    device_name = record.get("device_name")
    if device_name and str(device_name).lower() != device.lower():
        device += f" ({device_name})"
    cache = "cache hit" if record.get("model_cache_hit") else "model loaded"
    lines = [
        f"  {record['step']} | T{record['timepoint'] + 1} | {record['model']} | {channels}",
        f"    device={device}, mode={record.get('dimension_mode') or 'unknown'}, "
        f"runtime={record['runtime_seconds']:.2f}s, {cache}",
        f"    {format_label_statistics(record['label_statistics'], locations_only=record.get('target') == 'spots')}",
    ]
    effective = format_effective_parameters(record.get("effective_parameters", {}))
    if effective:
        lines.insert(2, f"    {effective}")
    timings = record.get("timings", {})
    useful = [
        f"{name.removesuffix('_seconds').replace('_', '-')}={float(timings.get(name, 0.0)):.2f}s"
        for name in (
            "import_seconds",
            "device_setup_seconds",
            "model_load_seconds",
            "inference_seconds",
        )
        if float(timings.get(name, 0.0)) > 0
    ]
    if useful:
        lines.append("    timing: " + ", ".join(useful))
    return lines


def _axis_units(image) -> dict[str, str]:
    multiscales = image.attrs.get("multiscales") or []
    axes = (multiscales[0].get("axes") or []) if multiscales else []
    return {
        str(axis.get("name", "")).lower(): str(axis.get("unit", ""))
        for axis in axes
        if isinstance(axis, dict) and axis.get("name")
    }


def input_report_lines(image, read_seconds: float) -> list[str]:
    t, c, z, y, x = image.data.shape
    units = _axis_units(image)
    scale_parts = []
    for axis in ("x", "y", "z", "t"):
        if axis in image.scales:
            unit = units.get(axis, "")
            scale_parts.append(f"{axis.upper()}={image.scales[axis]:g}{(' ' + unit) if unit else ''}")
    omero = image.attrs.get("omero") or {}
    channel_names = [
        str(channel.get("label") or f"C{index}")
        for index, channel in enumerate(omero.get("channels") or [], start=1)
    ]
    resource = image.resource
    location = str(resource.store_path)
    if resource.image_path:
        location += f" / {resource.image_path}"
    lines = [
        f"Input: {location}",
        f"  shape T={t}, C={c}, Z={z}, Y={y}, X={x}; dtype={image.source_dtype}; read={read_seconds:.2f}s",
        "  scale: " + (", ".join(scale_parts) if scale_parts else "not present in OME-Zarr metadata"),
    ]
    if channel_names:
        lines.append(
            "  channels: "
            + ", ".join(f"C{index}={name}" for index, name in enumerate(channel_names, start=1))
        )
    return lines


def workflow_report_lines(settings) -> list[str]:
    expansion = settings.cell_expansion_model()
    if settings.cell_model == "skip":
        step1 = "Step 1: Skip"
    elif expansion is not None:
        expansion_channel = settings.cell_expansion_channel()
        channel_source = (
            "Step 1 nucleus channel"
            if settings.cell_nuclei_channel > 0
            else "Step 1 Cyto Channel fallback"
        )
        step1 = (
            f"Step 1: expand nuclei with {expansion} on C{expansion_channel} "
            f"({channel_source}); "
            f"distance={settings.cell_expansion_distance:g} um"
        )
    else:
        second = (
            f", optional nucleus input C{settings.cell_nuclei_channel}"
            if settings.cell_nuclei_channel > 0
            and settings.cell_nuclei_channel != settings.cell_channel
            else ""
        )
        step1 = f"Step 1: {settings.cell_model} on C{settings.cell_channel}{second}"
    step2 = (
        "Step 2: Skip"
        if settings.nucleus_model == "skip"
        else f"Step 2: {settings.nucleus_model} on C{settings.nucleus_channel}"
    )
    foci = [
        f"Step 3{chr(96 + slot)}: {model} on C{channel}"
        for slot, model, channel in settings.enabled_foci_steps()
    ]
    if not foci:
        foci = ["Step 3a-3d: Skip"]
    return [
        "Workflow selection:",
        f"  {step1}",
        f"  {step2}",
        *(f"  {line}" for line in foci),
        f"  runtime: requested device={settings.device}, dimensions={settings.dimension_mode}",
        f"  post-processing: remove border cells={settings.remove_border_cells}",
        "  output: "
        + (
            "native OME-Zarr 0.4 labels (original image retained)"
            if settings.write_ome_zarr_labels
            else "labels as image channels"
        ),
        "  effective model parameters are reported for each segmentation below",
    ]
