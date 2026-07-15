from __future__ import annotations

from pathlib import Path
import shutil

import pytest


TESTS = Path(__file__).resolve().parent
DATA = TESTS / "data"
INPUT = TESTS / "inputfolder"
OUTPUT = TESTS / "outputfolder"


def _clean_test_folder(path: Path) -> None:
    """Remove stale test artifacts, restricted to the tests directory."""
    resolved = path.resolve()
    if resolved.parent != TESTS.resolve() or resolved.name not in {
        "inputfolder",
        "outputfolder",
    }:
        raise RuntimeError(f"Refusing to clean unexpected test path: {resolved}")
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


@pytest.fixture(scope="session")
def inputfolder() -> Path:
    """Stage fresh copies of committed OME-Zarr images for workflow tests."""
    _clean_test_folder(INPUT)
    for source in sorted(DATA.glob("*.ome.zarr")):
        shutil.copytree(source, INPUT / source.name)
    return INPUT


@pytest.fixture
def outputfolder() -> Path:
    """Give every output-writing test a clean, test-local output directory."""
    _clean_test_folder(OUTPUT)
    return OUTPUT
