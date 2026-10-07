"""Resolve spatial calibration without modifying the input OME-Zarr store."""

import math
from copy import deepcopy
from functools import lru_cache
from xml.etree import ElementTree as ET

LENGTH_FACTORS_UM = {
    "micrometer": 1.0,
    "micrometre": 1.0,
    "um": 1.0,
    "µm": 1.0,
    "μm": 1.0,
    "nanometer": 0.001,
    "nanometre": 0.001,
    "nm": 0.001,
    "millimeter": 1000.0,
    "millimetre": 1000.0,
    "mm": 1000.0,
    "meter": 1e6,
    "metre": 1e6,
    "m": 1e6,
    "centimeter": 1e4,
    "centimetre": 1e4,
    "cm": 1e4,
    "picometer": 1e-6,
    "picometre": 1e-6,
    "pm": 1e-6,
}


def effective_transform(multiscale, axes, dataset=None):
    """Compose dataset transforms followed by the shared NGFF transforms."""
    dataset = dataset if dataset is not None else multiscale["datasets"][0]
    scales, offsets = {}, dict.fromkeys(axes, 0.0)
    transforms = (dataset.get("coordinateTransformations") or []) + (
        multiscale.get("coordinateTransformations") or []
    )
    for transform in transforms:
        kind = transform.get("type")
        values = transform.get(kind)
        if (
            kind not in {"scale", "translation"}
            or not values
            or len(values) != len(axes)
        ):
            raise ValueError(
                "Calibration requires supported NGFF scale/translation transforms"
            )
        for axis, value in zip(axes, values):
            value = float(value)
            if kind == "scale":
                scales[axis] = scales.get(axis, 1.0) * value
                # An invalid size must not turn a zero origin into NaN before
                # OME-XML gets a chance to supply the size.
                offsets[axis] = offsets[axis] * value if offsets[axis] else 0.0
            else:
                offsets[axis] += value
    return scales, offsets


@lru_cache(maxsize=4)
def _ome_metadata(path, mtime_ns, size):
    # Cache per file revision, so a plate's XML is parsed once per process.
    try:
        return ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None


def _xml_pixels(resource, shape, axes):
    local = resource.store_path / resource.image_path / "OME" / "METADATA.ome.xml"
    candidates = [local, resource.store_path / "OME" / "METADATA.ome.xml"]
    for path in dict.fromkeys(candidates):
        try:
            stat = path.stat()
        except OSError:
            continue
        ome = _ome_metadata(str(path), stat.st_mtime_ns, stat.st_size)
        if ome is None:
            continue
        images = ome.findall("{*}Image")
        image = None
        if path == local and len(images) == 1:
            image = images[0]
        elif resource.plate_path:
            row, column, field = resource.plate_path
            plate = (resource.plate_attrs or {}).get("plate") or {}
            try:
                row_index = [str(r["name"]) for r in plate["rows"]].index(row)
                column_index = [str(c["name"]) for c in plate["columns"]].index(column)
                fields = (resource.well_attrs or {}).get("well", {}).get("images", [])
                field_index = (
                    int(field)
                    if field.isdecimal()
                    else [str(f["path"]) for f in fields].index(field)
                )
                plates = ome.findall("{*}Plate")
                if len(plates) != 1:
                    plates = [p for p in plates if p.get("Name") == plate.get("name")]
                if len(plates) != 1:
                    continue
                well = next(
                    w
                    for w in plates[0].findall("{*}Well")
                    if int(w.get("Row")) == row_index
                    and int(w.get("Column")) == column_index
                )
                samples = sorted(
                    well.findall("{*}WellSample"), key=lambda s: int(s.get("Index"))
                )
                ref = samples[field_index].find("{*}ImageRef").get("ID")
                image = next(i for i in images if i.get("ID") == ref)
            except (
                KeyError,
                ValueError,
                TypeError,
                IndexError,
                StopIteration,
                AttributeError,
            ):
                continue
        elif len(images) == 1:
            image = images[0]
        if image is None:
            continue
        pixels = image.find("{*}Pixels")
        if pixels is None:
            continue
        # Do not apply another series' calibration to this array.
        try:
            sizes = dict(zip(axes, shape))
            if any(
                int(pixels.get(f"Size{a.upper()}")) != sizes.get(a, 1) for a in "tczyx"
            ):
                continue
        except (ValueError, TypeError):
            continue
        return pixels, path, image.get("ID")
    return None, None, None


def resolve_spatial_calibration(resource, attrs, axes, shape):
    """Prefer calibrated NGFF axes; fill unavailable axes from matching OME-XML."""
    multiscale = attrs["multiscales"][0]
    scales, _offsets = effective_transform(multiscale, axes)
    units = {
        a["name"]: a.get("unit")
        for a in multiscale.get("axes", [])
        if isinstance(a, dict)
    }
    missing = [
        a
        for a in axes
        if a in "zyx"
        and not (
            units.get(a) in LENGTH_FACTORS_UM
            and math.isfinite(scales.get(a, 0))
            and scales.get(a, 0) > 0
        )
    ]
    fallback, sources = {}, {}
    if missing:
        pixels, path, image_id = _xml_pixels(resource, shape, axes)
        if pixels is not None:
            for axis in missing:
                try:
                    value = float(pixels.get(f"PhysicalSize{axis.upper()}"))
                    unit = pixels.get(f"PhysicalSize{axis.upper()}Unit", "µm")
                    value *= LENGTH_FACTORS_UM[unit]
                except (TypeError, ValueError, KeyError):
                    continue
                if math.isfinite(value) and value > 0:
                    fallback[axis] = value
                    sources[axis] = f"OME-XML: {path} ({image_id})"
    if fallback:
        attrs = deepcopy(attrs)
        multiscale = attrs["multiscales"][0]
        # Flatten shared transforms while preserving other axes and pyramid ratios.
        for dataset in multiscale["datasets"]:
            level_scales, level_offsets = effective_transform(multiscale, axes, dataset)
            for axis, value in fallback.items():
                original = scales.get(axis, 1.0)
                factor = (
                    value / original
                    if math.isfinite(original) and original > 0
                    else None
                )
                level_scales[axis] = (
                    level_scales.get(axis, 1.0) * factor if factor else value
                )
                offset_factor = LENGTH_FACTORS_UM.get(units.get(axis), factor)
                level_offsets[axis] = (
                    level_offsets[axis] * offset_factor if offset_factor else 0.0
                )
            dataset["coordinateTransformations"] = [
                {"type": "scale", "scale": [level_scales.get(a, 1.0) for a in axes]},
                {
                    "type": "translation",
                    "translation": [level_offsets[a] for a in axes],
                },
            ]
        multiscale.pop("coordinateTransformations", None)
        multiscale["axes"] = [
            {**(a if isinstance(a, dict) else {"name": a}), "unit": "micrometer"}
            if (a.get("name") if isinstance(a, dict) else a) in fallback
            else a
            for a in (multiscale.get("axes") or axes)
        ]
        scales, _offsets = effective_transform(multiscale, axes)
        units.update(dict.fromkeys(fallback, "micrometer"))
    # Models and base measurements consume scales in micrometers.
    scales = {
        a: value * LENGTH_FACTORS_UM.get(units.get(a), 1.0) if a in "zyx" else value
        for a, value in scales.items()
    }
    return attrs, scales, sources
