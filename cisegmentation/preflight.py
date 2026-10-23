"""Metadata-only checks for the complete run, before model probes or output staging."""

from __future__ import annotations

import math

from .calibration import LENGTH_FACTORS_UM, effective_transform
from .ome_zarr_io import enumerate_resources, read_image
from .reporting import emit
from .settings import SKIP


def model_steps(settings):
    """Return the active model, target, primary channel and optional second channel."""
    steps = []
    if settings.cell_model != SKIP:
        expansion = settings.cell_expansion_model()
        steps.append(("Step 1", expansion or settings.cell_model,
                      "nuclei" if expansion else "cells",
                      settings.cell_expansion_channel() if expansion else settings.cell_channel,
                      0 if expansion else settings.cell_nuclei_channel))
    if settings.nucleus_model != SKIP:
        steps.append(("Step 2", settings.nucleus_model, "nuclei", settings.nucleus_channel, 0))
    steps.extend((f"Step 3{chr(96 + slot)}", model, "cells", channel, 0)
                 for slot, model, channel in settings.enabled_foci_steps())
    return steps


def _axis_calibration(image, axis):
    multiscale = image.attrs["multiscales"][0]
    metadata = multiscale.get("axes") or image.axes
    names = [item.get("name") if isinstance(item, dict) else item for item in metadata]
    units = {item["name"]: item.get("unit") for item in metadata if isinstance(item, dict)}
    scales, _offsets = effective_transform(multiscale, names)
    raw, unit = scales.get(axis), units.get(axis)
    try:
        normalized = float(raw) * LENGTH_FACTORS_UM[unit]
    except (TypeError, ValueError, KeyError):
        normalized = float("nan")
    return raw, unit, normalized


def _cellpose_check(image, settings, step, model, target, channels, ram_available):
    source = f"{image.resource.store_path}/{image.resource.image_path}"
    details = []
    pixels = {}
    native_3d = image.data.shape[2] > 1 and settings.dimension_mode != "slice-2d"
    for axis in ("x", "y", "z") if native_3d else ("x", "y"):
        raw, unit, normalized = _axis_calibration(image, axis)
        details.append(f"{axis.upper()}={raw!r} {unit!r} -> {normalized:g} um/pixel")
        pixels[axis] = normalized
        valid = math.isfinite(normalized) and normalized > 0
        if axis in "xy":
            valid = valid and (settings.cellpose_min_pixel_size_um <= normalized
                               <= settings.cellpose_max_pixel_size_um)
        if not valid:
            raise ValueError(
                f"{source}, {step} ({model}): invalid or implausible calibration: "
                + "; ".join(details)
                + f". Cellpose requires calibrated {'X/Y/Z' if native_3d else 'X/Y'} physical units, positive Z for native 3D, "
                f"and X/Y within [{settings.cellpose_min_pixel_size_um:g}, "
                f"{settings.cellpose_max_pixel_size_um:g}] um/pixel. "
                "Correct the source NGFF metadata or embedded OME-XML calibration; "
                "no pixel size is assumed or clamped. The complete run was stopped before inference."
            )
    # 30 pixels is the conservative Cellpose reference diameter; using the nuclei
    # reference of 17 would predict less enlargement. No model is imported here.
    diameter = None if settings.diameter < 0 else (
        (settings.diameter or (12.0 if target == "nuclei" else 25.0))
        / ((pixels["x"] + pixels["y"]) / 2)
    )
    rescale = 1.0 if diameter is None else 30.0 / diameter
    # Floating estimates stay bounded in memory even for maliciously tiny diameters;
    # overflow becomes infinity and is rejected, never an array allocation.
    full_xy = tuple(size * rescale for size in image.data.shape[-2:])
    tile_xy = [min(size, settings.tile_size + 2 * settings.tile_overlap) * rescale
               for size in image.data.shape[-2:]]
    depth = min(image.data.shape[2], settings.tile_depth + 2 * settings.tile_overlap_z) if native_3d else 1
    # Conservative working-array estimate, not a promise of a measured RAM peak.
    estimated = math.prod(tile_xy) * depth * max(channels, 2) * 4 * 8
    unsafe = (not math.isfinite(rescale) or rescale > settings.cellpose_max_rescale
              or max(tile_xy) > 16384
              or (ram_available is not None and estimated > ram_available * 0.8))
    if unsafe:
        raise ValueError(
            f"{source}, {step} ({model}): unsafe Cellpose resampling; "
            + "; ".join(details)
            + f"; effective diameter={diameter!r} pixels, rescale={rescale:g}x, "
            f"projected full XY={full_xy}, maximum working tile XY={tuple(tile_xy)}, "
            f"estimated working arrays={estimated / 1024**2:.1f} MiB, "
            f"available RAM={None if ram_available is None else round(ram_available / 1024**2, 1)} MiB. "
            "Correct calibration/diameter, use diameter=-1 for the supported model-default scale, "
            "or reduce tile dimensions. The complete run was stopped before inference."
        )


def validate_inputs(stores, settings, *, ram_available=None, log=None):
    """Check every field without reading image chunks or importing any model."""
    steps = model_steps(settings)
    checked = fallback_fields = 0
    for store in stores:
        for resource in enumerate_resources(store):
            image = read_image(resource, lazy=True)
            try:
                count = image.data.shape[1]
                for step, model, target, channel, second_channel in steps:
                    for selected in (channel, second_channel) if second_channel else (channel,):
                        if not 1 <= selected <= count:
                            raise ValueError(
                                f"{resource.store_path}/{resource.image_path}, {step}: "
                                f"channel {selected} is outside 1..{count}. Channels start at 1 "
                                "(1 = first channel); 0 only disables the optional extra nucleus channel."
                            )
                    if model.startswith(("cellpose3:", "cellpose-sam:")):
                        _cellpose_check(image, settings, step, model, target,
                                        2 if second_channel else 1, ram_available)
                if settings.spatial_measurements or settings.tracking:
                    from .measurement_extensions import require_spatial_calibration
                    require_spatial_calibration(image)
                checked += 1
                fallback_fields += bool(image.calibration_sources)
            except ValueError as exc:
                emit(log, f"CI Segmentation preflight failed: {exc}")
                raise
            finally:
                image.data.array.store.close()
    return checked, fallback_fields
