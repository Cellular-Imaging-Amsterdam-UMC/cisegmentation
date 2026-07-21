import json
import os
import sys
from types import SimpleNamespace

from tools.download_models import (
    CACHE_SCHEMA,
    CUSTOM_COMMIT,
    CUSTOM_STARDIST,
    INSTANSEG_MODELS,
    SPOTIFLOW_MODELS,
    _cache_inventory,
    _complete_cache_state,
    _download_cellpose,
)


def test_custom_stardist_download_contract_is_pinned():
    assert CUSTOM_COMMIT == "b280dfebd4910a5678fe4e93534c7c7ae335b96c"
    assert set(CUSTOM_STARDIST) == {"SD_Foci_Aggregates", "SD_Foci_Finn"}
    for files in CUSTOM_STARDIST.values():
        assert set(files) == {"config.json", "thresholds.json", "weights_best.h5"}
        assert all(len(checksum) == 64 for checksum in files.values())


def test_complete_official_model_groups_are_declared():
    assert CACHE_SCHEMA == 5
    assert len(INSTANSEG_MODELS) == 3
    assert len(SPOTIFLOW_MODELS) == 6


def test_old_cache_schema_is_invalidated(tmp_path):
    (tmp_path / ".complete.json").write_text(
        json.dumps(
            {
                "schema": CACHE_SCHEMA - 1,
                "custom_stardist_commit": CUSTOM_COMMIT,
                "inventory": {},
            }
        ),
        encoding="utf-8",
    )
    assert _complete_cache_state(tmp_path) is None


def test_cellpose_downloader_prepares_both_sam_versions_idempotently(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CELLPOSE3_LEGACY_LOCAL_MODELS_PATH", "restore-after-test")
    monkeypatch.setenv("CELLPOSE_LOCAL_MODELS_PATH", "restore-after-test")

    class LegacyModels:
        @staticmethod
        def model_path(name):
            path = tmp_path / "cellpose3" / name
            path.write_bytes(name.encode())
            return path

        @staticmethod
        def size_model_path(name):
            path = tmp_path / "cellpose3" / f"size_{name}"
            path.write_bytes(name.encode())
            return path

    loaded = []

    class CellposeModel:
        def __init__(self, *, gpu, pretrained_model):
            assert gpu is False
            loaded.append(pretrained_model)
            path = tmp_path / "cellpose-sam" / pretrained_model
            assert str(path.parent) == os.environ["CELLPOSE_LOCAL_MODELS_PATH"]
            path.write_bytes(pretrained_model.encode())

    monkeypatch.setitem(
        sys.modules,
        "cellpose3_legacy",
        SimpleNamespace(models=LegacyModels),
    )
    monkeypatch.setitem(
        sys.modules,
        "cellpose",
        SimpleNamespace(models=SimpleNamespace(CellposeModel=CellposeModel)),
    )

    first = _download_cellpose(tmp_path)
    second = _download_cellpose(tmp_path)

    assert loaded == ["cpsam_v2", "cpsam", "cpsam_v2", "cpsam"]
    assert first == second
    assert first["cpsam_v2"] > 0
    assert first["cpsam"] > 0


def test_instanseg_archives_are_pinned_to_named_releases():
    assert all(url.startswith("https://github.com/instanseg/instanseg/releases/download/") for url in INSTANSEG_MODELS.values())
    assert all(url.endswith(".zip") for url in INSTANSEG_MODELS.values())


def test_complete_cache_inventory_detects_missing_artifacts(tmp_path):
    artifact = tmp_path / "cellpose3" / "nuclei"
    artifact.parent.mkdir()
    artifact.write_bytes(b"checkpoint")
    state = {
        "schema": CACHE_SCHEMA,
        "custom_stardist_commit": CUSTOM_COMMIT,
        "inventory": _cache_inventory(tmp_path),
    }
    (tmp_path / ".complete.json").write_text(json.dumps(state), encoding="utf-8")

    assert _complete_cache_state(tmp_path) == state
    artifact.unlink()
    assert _complete_cache_state(tmp_path) is None
