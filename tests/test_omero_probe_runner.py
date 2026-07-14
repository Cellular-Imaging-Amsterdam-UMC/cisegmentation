from pathlib import Path

from tools.omero_import_metadata_probe.run_roundtrip import ROOT, build_wrapper_command


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
