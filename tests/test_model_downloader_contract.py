import json

from tools.download_models import (
    CACHE_SCHEMA,
    CUSTOM_COMMIT,
    CUSTOM_STARDIST,
    INSTANSEG_MODELS,
    SPOTIFLOW_MODELS,
    _cache_inventory,
    _complete_cache_state,
)


def test_custom_stardist_download_contract_is_pinned():
    assert CUSTOM_COMMIT == "b280dfebd4910a5678fe4e93534c7c7ae335b96c"
    assert set(CUSTOM_STARDIST) == {"SD_Foci_Aggregates", "SD_Foci_Finn"}
    for files in CUSTOM_STARDIST.values():
        assert set(files) == {"config.json", "thresholds.json", "weights_best.h5"}
        assert all(len(checksum) == 64 for checksum in files.values())


def test_complete_official_model_groups_are_declared():
    assert len(INSTANSEG_MODELS) == 3
    assert len(SPOTIFLOW_MODELS) == 6


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
