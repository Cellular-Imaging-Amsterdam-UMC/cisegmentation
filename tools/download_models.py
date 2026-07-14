"""Download and validate the complete CI Segmentation inference model cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request
import zipfile


CP3_MODELS = (
    "cyto3",
    "nuclei",
    "cyto2_cp3",
    "tissuenet_cp3",
    "livecell_cp3",
    "yeast_PhC_cp3",
    "yeast_BF_cp3",
    "bact_phase_cp3",
    "bact_fluor_cp3",
    "deepbacs_cp3",
    "cyto2",
    "cyto",
    "CPx",
    "transformer_cp3",
    "neurips_cellpose_default",
    "neurips_cellpose_transformer",
    "neurips_grayscale_cyto2",
    "CP",
    "TN1",
    "TN2",
    "TN3",
    "LC1",
    "LC2",
    "LC3",
    "LC4",
)
INSTANSEG_MODELS = {
    "brightfield_nuclei": "https://github.com/instanseg/instanseg/releases/download/instanseg_models_v0.1.1/brightfield_nuclei.zip",
    "fluorescence_nuclei_and_cells": "https://github.com/instanseg/instanseg/releases/download/instanseg_models_v0.1.1/fluorescence_nuclei_and_cells.zip",
    "single_channel_nuclei": "https://github.com/instanseg/instanseg/releases/download/instanseg_models_v0.1.2/single_channel_nuclei.zip",
}
SPOTIFLOW_MODELS = (
    "general",
    "hybiss",
    "synth_complex",
    "synth_3d",
    "smfish_3d",
    "fluo_live",
)
CUSTOM_COMMIT = "b280dfebd4910a5678fe4e93534c7c7ae335b96c"
CUSTOM_STARDIST = {
    "SD_Foci_Aggregates": {
        "config.json": "1902d4ef3724c3e9a021a9c67d9efbe46e2a2fed5ed012e68fe722ca7e5c8759",
        "thresholds.json": "97ebdd5eba07dcdbc9ea000be97c7698bd24d4b5b26ae51d23e0bd6d29a6fb51",
        "weights_best.h5": "c5b52b66614fa638271f4ed556efd60cade7f52b22c532df8cff45e2da75cb63",
    },
    "SD_Foci_Finn": {
        "config.json": "74e54fb377896de840c27fada5d6e5c8ccf73cd9603918ca44c0d03c5416527f",
        "thresholds.json": "288804c2345b36b23e2fdd7842315c25e9a4b56ae71ab7d366a4ad919310e960",
        "weights_best.h5": "8ef9add61f8632c77963b0617eaba9ee2ca2bc7e477081aa996f47cfea37d3aa",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_sha256: str | None = None) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        if expected_sha256 is None or _sha256(destination) == expected_sha256:
            return
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, temporary)
    if temporary.stat().st_size == 0:
        raise RuntimeError(f"Empty download: {url}")
    if expected_sha256 is not None:
        actual = _sha256(temporary)
        if actual != expected_sha256:
            temporary.unlink()
            raise RuntimeError(
                f"Checksum mismatch for {url}: expected {expected_sha256}, got {actual}"
            )
    temporary.replace(destination)


def _download_instanseg(root: Path) -> dict:
    state = {}
    for name, url in INSTANSEG_MODELS.items():
        destination = root / "instanseg" / name
        marker = destination / "rdf.yaml"
        if not marker.exists():
            archive = root / "downloads" / f"instanseg_{name}.zip"
            _download(url, archive)
            temporary = destination.with_name(destination.name + ".partial")
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=True)
            with zipfile.ZipFile(archive) as zipped:
                zipped.extractall(temporary)
            candidates = list(temporary.rglob("rdf.yaml"))
            if len(candidates) != 1:
                raise RuntimeError(f"InstanSeg archive {name} has no unique rdf.yaml")
            extracted = candidates[0].parent
            if destination.exists():
                shutil.rmtree(destination)
            if extracted == temporary:
                temporary.replace(destination)
            else:
                shutil.move(str(extracted), str(destination))
                shutil.rmtree(temporary)
        state[name] = _sha256(marker)
    return state


def _download_cellpose(root: Path) -> dict:
    legacy = root / "cellpose3"
    sam = root / "cellpose-sam"
    legacy.mkdir(parents=True, exist_ok=True)
    sam.mkdir(parents=True, exist_ok=True)
    os.environ["CELLPOSE3_LEGACY_LOCAL_MODELS_PATH"] = str(legacy)
    os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = str(sam)
    from cellpose3_legacy import models as cp3

    for name in CP3_MODELS:
        cp3.model_path(name)
    for name in ("cyto", "cyto2", "cyto3", "nuclei"):
        cp3.size_model_path(name)
    from cellpose import models as cp4

    cp4.CellposeModel(gpu=False, pretrained_model="cpsam")
    return {
        "cellpose3_files": len(list(legacy.glob("*"))),
        "cpsam": (sam / "cpsam").stat().st_size,
    }


def _download_stardist(root: Path) -> dict:
    destination = root / "stardist"
    destination.mkdir(parents=True, exist_ok=True)
    versatile = destination / "SD_Nuclei_Versatile"
    if not (versatile / "SD_Nuclei_Versatile.pt").exists():
        from cistardist_pytorch.cli import _download_doi

        versatile.mkdir(parents=True, exist_ok=True)
        _download_doi("10.5281/zenodo.20038194", versatile)
    from cistardist_pytorch import StarDist2D
    from cistardist_pytorch.converter import convert_model_folder

    state = {}
    for name, files in CUSTOM_STARDIST.items():
        folder = destination / name
        folder.mkdir(parents=True, exist_ok=True)
        for filename, expected_sha256 in files.items():
            url = f"https://raw.githubusercontent.com/Cellular-Imaging-Amsterdam-UMC/cistardist_pytorch/{CUSTOM_COMMIT}/models/{name}/{filename}"
            _download(url, folder / filename, expected_sha256)
        checkpoint = folder / f"{name}.pt"
        if not checkpoint.exists():
            convert_model_folder(folder, output_name=checkpoint.name)
        StarDist2D.from_folder(folder, device="cpu")
        state[name] = _sha256(checkpoint)
    StarDist2D.from_folder(versatile, device="cpu")
    state["SD_Nuclei_Versatile"] = _sha256(versatile / "SD_Nuclei_Versatile.pt")
    return state


def _download_spotiflow(root: Path) -> dict:
    destination = root / "spotiflow"
    destination.mkdir(parents=True, exist_ok=True)
    os.environ["SPOTIFLOW_CACHE_DIR"] = str(destination)
    os.environ["SPOTIFLOW_LOCAL_MODELS_PATH"] = str(destination)
    from spotiflow.model.pretrained import get_pretrained_model_path

    return {
        name: str(get_pretrained_model_path(name, cache_dir=destination))
        for name in SPOTIFLOW_MODELS
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(
            os.environ.get("CISEGMENTATION_MODELS", "/opt/cisegmentation/models")
        ),
    )
    parser.add_argument(
        "--skip-cellpose", action="store_true", help="Useful for fast downloader tests"
    )
    parser.add_argument(
        "--skip-spotiflow", action="store_true", help="Useful for fast downloader tests"
    )
    args = parser.parse_args(argv)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    state = {"schema": 1, "custom_stardist_commit": CUSTOM_COMMIT}
    if not args.skip_cellpose:
        state["cellpose"] = _download_cellpose(args.models_dir)
    state["stardist"] = _download_stardist(args.models_dir)
    state["instanseg"] = _download_instanseg(args.models_dir)
    if not args.skip_spotiflow:
        state["spotiflow"] = _download_spotiflow(args.models_dir)
    temporary = args.models_dir / ".complete.json.partial"
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.models_dir / ".complete.json")
    print(f"Model cache complete: {args.models_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
