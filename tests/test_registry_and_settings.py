import pytest

from cisegmentation.registry import (
    eligible_benchmark_models,
    get_model_spec,
)
from cisegmentation.settings import SegmentationSettings, parse_model_selection


def test_requested_custom_stardist_models_are_registered():
    assert get_model_spec("stardist:SD_Foci_Aggregates").targets == ("foci",)
    assert get_model_spec("stardist:SD_Foci_Finn").targets == ("foci",)


def test_both_cellpose_sam_versions_are_registered():
    v2 = get_model_spec("cellpose-sam:cpsam_v2")
    original = get_model_spec("cellpose-sam:cpsam")
    assert (v2.family, v2.checkpoint, v2.targets, v2.dimensions) == (
        "cellpose-sam",
        "cpsam_v2",
        ("nuclei", "cells"),
        "3d",
    )
    assert original.checkpoint == "cpsam"


def test_foci_eligible_models_include_both_custom_models():
    ids = {spec.id for spec in eligible_benchmark_models("foci", 1)}
    assert {"stardist:SD_Foci_Aggregates", "stardist:SD_Foci_Finn"} <= ids


def test_channel_selection_is_one_based_and_validated():
    settings = SegmentationSettings(primary_channel=2, nuclei_channel=1)
    assert settings.selected_channels(3) == [1, 0]
    with pytest.raises(ValueError):
        SegmentationSettings(primary_channel=4).selected_channels(3)


def test_four_independent_foci_slots_retain_duplicate_channels():
    settings = SegmentationSettings(
        foci_model_1="spotiflow:general",
        foci_channel_1=2,
        foci_model_2="spotiflow:hybiss",
        foci_channel_2=2,
        foci_model_4="stardist:SD_Foci_Finn",
        foci_channel_4=3,
    )
    assert settings.enabled_foci_steps() == [
        (1, "spotiflow:general", 2),
        (2, "spotiflow:hybiss", 2),
        (4, "stardist:SD_Foci_Finn", 3),
    ]


def test_measurement_database_format_is_validated():
    for database_format in ("duckdb", "sqlite", "skip"):
        SegmentationSettings(measurements_database=database_format).validate_steps()
    with pytest.raises(ValueError, match="Measurements Database"):
        SegmentationSettings(measurements_database="csv").validate_steps()


@pytest.mark.parametrize(
    "value, expected",
    [
        ("all", ["all"]),
        ("a,b", ["a", "b"]),
        ('["a", "b"]', ["a", "b"]),
        (["a", "b"], ["a", "b"]),
    ],
)
def test_parse_model_selection(value, expected):
    assert parse_model_selection(value) == expected
