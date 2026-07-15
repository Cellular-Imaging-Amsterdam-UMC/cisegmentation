#!/usr/bin/env python
"""Run the established metadata probe with CI Segmentation compatibility fixes.

The local CIDeconvolve probe assumes that every OMERO rendering channel has an
RGB color. Label images may legitimately have no rendering color, so its
generated OME-XML exporter otherwise calls ``int(None)``. This shim keeps the
source tool unchanged and makes the optional color safe.
"""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_PROBE = Path(
    r"C:\rahoebe\Python\cideconvolve\tools\omero_import_metadata_probe"
    r"\omero_import_metadata_probe.py"
)
COLOR_GUARD = "if not rgb:\n        return None"
SAFE_COLOR_GUARD = (
    "if not rgb or any(component is None for component in rgb[:3]):\n"
    "        return None"
)
ZARR_GROUP_CREATION = 'zarr.open_group(output_dir, mode="w")'
ZARR_V2_GROUP_CREATION = 'zarr.open_group(output_dir, mode="w", zarr_format=2)'
ZARR_ARRAY_CREATION = (
    'arr = root_group.create_dataset("0", shape=shape, chunks=chunks, '
    'dtype=dtype, overwrite=True)'
)
ZARR_ARRAY_WITH_AXES = ZARR_ARRAY_CREATION + (
    '\n    arr.attrs["_ARRAY_DIMENSIONS"] = ["t", "c", "z", "y", "x"]'
)
PROBE_CLEANUP_ARGUMENT = (
    'parser.add_argument("--cleanup", choices=["always", "success", "never"], '
    'default="success", help="Remove imported probe objects after report.")'
)
PROBE_RETAIN_ARGUMENTS = PROBE_CLEANUP_ARGUMENT + (
    '\n    parser.add_argument("--retain-container-staging", action="store_true", '
    'help="Retain a copied input until the caller has finished using the registered Zarr.")'
)
PROBE_STAGING_CLEANUP = (
    'if container_input.get("method") in {"docker_cp_to_tmp", "slurm_input_export"}:'
)
PROBE_CONDITIONAL_STAGING_CLEANUP = (
    'if not args.retain_container_staging and container_input.get("method") '
    'in {"docker_cp_to_tmp", "slurm_input_export"}:'
)
OME_XML_ATTR_FUNCTION = '''def ome_xml_attr(name, value):
    if value is None:
        return ""
    return f' {name}="{xml_escape_attr(value)}"'
'''
OME_XML_SAFE_ATTR_FUNCTION = '''def ome_xml_attr(name, value):
    if value is None:
        return ""
    if name.endswith("Unit"):
        value = {
            "METER": "m",
            "CENTIMETER": "cm",
            "MILLIMETER": "mm",
            "MICROMETER": "µm",
            "NANOMETER": "nm",
            "ANGSTROM": "Å",
            "PICOMETER": "pm",
        }.get(str(value).upper(), value)
    return f' {name}="{xml_escape_attr(value)}"'
'''
CONTAINER_TEMP_STAGING = (
    'container_path = f"/tmp/cideconvolve_omero_probe/{run_uuid}/{input_path.name}"'
)
CONTAINER_SHARED_STAGING = (
    'container_path = f"/data/.cisegmentation_roundtrip/{run_uuid}/{input_path.name}"'
)
CONTAINER_STAGING_CLEANUP_FUNCTION = '''def _cleanup_container_staging(stack: StackConfig, container_path: str) -> None:
    if not container_path.startswith("/tmp/cideconvolve_omero_probe/"):
        return
    stack.compose("exec", "-T", "biomero-importer", "rm", "-rf", str(Path(container_path).parent).replace("\\\\", "/"), timeout=120)
'''
CONTAINER_SAFE_STAGING_CLEANUP_FUNCTION = '''def _cleanup_container_staging(stack: StackConfig, container_path: str) -> None:
    roots = ("/tmp/cideconvolve_omero_probe/", "/data/.cisegmentation_roundtrip/")
    if not any(container_path.startswith(root) for root in roots):
        return
    stack.compose("exec", "-T", "biomero-importer", "rm", "-rf", str(Path(container_path).parent).replace("\\\\", "/"), timeout=120)
'''


def patch_probe_source(source: str) -> str:
    """Allow missing colors and keep NGFF 0.4 output on Zarr v2."""
    if source.count(COLOR_GUARD) != 1:
        raise RuntimeError(
            "The metadata probe color function changed; refusing to apply an "
            "unverified compatibility patch"
        )
    if source.count(ZARR_GROUP_CREATION) != 1:
        raise RuntimeError(
            "The metadata probe Zarr writer changed; refusing to apply an "
            "unverified compatibility patch"
        )
    if source.count(ZARR_ARRAY_CREATION) != 1:
        raise RuntimeError(
            "The metadata probe Zarr array writer changed; refusing to apply "
            "an unverified compatibility patch"
        )
    if source.count(PROBE_CLEANUP_ARGUMENT) != 1 or source.count(PROBE_STAGING_CLEANUP) != 1:
        raise RuntimeError(
            "The metadata probe staging lifecycle changed; refusing to apply "
            "an unverified compatibility patch"
        )
    if source.count(OME_XML_ATTR_FUNCTION) != 1:
        raise RuntimeError(
            "The metadata probe OME-XML writer changed; refusing to apply an "
            "unverified compatibility patch"
        )
    if source.count(CONTAINER_TEMP_STAGING) != 1 or source.count(CONTAINER_STAGING_CLEANUP_FUNCTION) != 1:
        raise RuntimeError(
            "The metadata probe container staging implementation changed; "
            "refusing to apply an unverified compatibility patch"
        )
    return (
        source.replace(COLOR_GUARD, SAFE_COLOR_GUARD, 1)
        .replace(OME_XML_ATTR_FUNCTION, OME_XML_SAFE_ATTR_FUNCTION, 1)
        .replace(CONTAINER_TEMP_STAGING, CONTAINER_SHARED_STAGING, 1)
        .replace(
            CONTAINER_STAGING_CLEANUP_FUNCTION,
            CONTAINER_SAFE_STAGING_CLEANUP_FUNCTION,
            1,
        )
        .replace(ZARR_GROUP_CREATION, ZARR_V2_GROUP_CREATION, 1)
        .replace(ZARR_ARRAY_CREATION, ZARR_ARRAY_WITH_AXES, 1)
        .replace(PROBE_CLEANUP_ARGUMENT, PROBE_RETAIN_ARGUMENTS, 1)
        .replace(PROBE_STAGING_CLEANUP, PROBE_CONDITIONAL_STAGING_CLEANUP, 1)
    )


def main() -> None:
    source_path = Path(os.environ.get("CISEGMENTATION_OMERO_PROBE_SOURCE", DEFAULT_PROBE))
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing OMERO metadata probe: {source_path}")
    source = patch_probe_source(source_path.read_text(encoding="utf-8"))
    namespace = {
        "__name__": "__main__",
        "__file__": str(source_path),
        "__package__": None,
    }
    exec(compile(source, str(source_path), "exec"), namespace)


if __name__ == "__main__":
    main()
