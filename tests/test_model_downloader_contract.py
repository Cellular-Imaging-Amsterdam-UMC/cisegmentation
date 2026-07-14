from tools.download_models import (
    CUSTOM_COMMIT,
    CUSTOM_STARDIST,
    INSTANSEG_MODELS,
    SPOTIFLOW_MODELS,
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
