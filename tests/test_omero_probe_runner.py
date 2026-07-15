from pathlib import Path

import pytest

from tools.omero_import_metadata_probe.omero_import_metadata_probe import patch_probe_source
from tools.omero_import_metadata_probe.run_roundtrip import ROOT, build_wrapper_command


def test_probe_patch_allows_missing_rendering_color():
    source = (
        'def ome_xml_attr(name, value):\n    if value is None:\n        return ""\n'
        '    return f\' {name}="{xml_escape_attr(value)}"\'\n\n'
        "def ome_color_int(rgb):\n    if not rgb:\n        return None\n"
        'root_group = zarr.open_group(output_dir, mode="w")\n'
        'arr = root_group.create_dataset("0", shape=shape, chunks=chunks, '
        'dtype=dtype, overwrite=True)\n'
        'parser.add_argument("--cleanup", choices=["always", "success", "never"], '
        'default="success", help="Remove imported probe objects after report.")\n'
        'if container_input.get("method") in {"docker_cp_to_tmp", "slurm_input_export"}:\n'
        'container_path = f"/tmp/cideconvolve_omero_probe/{run_uuid}/{input_path.name}"\n'
        'def _cleanup_container_staging(stack: StackConfig, container_path: str) -> None:\n'
        '    if not container_path.startswith("/tmp/cideconvolve_omero_probe/"):\n'
        '        return\n'
        '    stack.compose("exec", "-T", "biomero-importer", "rm", "-rf", '
        'str(Path(container_path).parent).replace("\\\\", "/"), timeout=120)\n'
    )
    patched = patch_probe_source(source)
    assert "any(component is None for component in rgb[:3])" in patched
    assert 'zarr.open_group(output_dir, mode="w", zarr_format=2)' in patched
    assert 'arr.attrs["_ARRAY_DIMENSIONS"] = ["t", "c", "z", "y", "x"]' in patched
    assert 'parser.add_argument("--retain-container-staging"' in patched
    assert "if not args.retain_container_staging" in patched
    assert '"MICROMETER": "µm"' in patched
    assert "/data/.cisegmentation_roundtrip/{run_uuid}" in patched
    assert 'roots = ("/tmp/cideconvolve_omero_probe/", "/data/.cisegmentation_roundtrip/")' in patched


def test_probe_patch_rejects_an_unknown_source_version():
    with pytest.raises(RuntimeError, match="color function changed"):
        patch_probe_source("def ome_color_int(rgb):\n    return 1\n")


def test_probe_patch_rejects_an_unknown_zarr_writer():
    source = "def ome_color_int(rgb):\n    if not rgb:\n        return None\n"
    with pytest.raises(RuntimeError, match="Zarr writer changed"):
        patch_probe_source(source)


def test_probe_builds_segmentation_wrapper_command():
    command = build_wrapper_command(
        Path("python"),
        Path("input"),
        Path("output"),
        "stardist:SD_Foci_Finn",
        "foci",
        "cuda",
    )
    assert str(ROOT / "wrapper.py") in command
    assert command[command.index("--model") + 1] == "stardist:SD_Foci_Finn"
    assert command[command.index("--target") + 1] == "foci"
    assert command[-1] == "--local"
