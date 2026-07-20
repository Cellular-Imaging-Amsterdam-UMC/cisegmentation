import numpy as np
import pytest
from types import SimpleNamespace

import cisegmentation.engine as engine
from cisegmentation.settings import SegmentationSettings

from cisegmentation.engine import _match_cells_and_nuclei


def _example_labels():
    cells = np.zeros((1, 8, 8), dtype=np.uint32)
    cells[0, 1:7, 1:4] = 1
    cells[0, 1:4, 0] = 2  # touches the left XY border
    cells[0, 4:7, 5:7] = 3  # has no nucleus
    nuclei = np.zeros_like(cells)
    nuclei[0, 1, 1] = 10
    nuclei[0, 3:5, 2:4] = 11  # larger nucleus in cell 1
    nuclei[0, 2, 0] = 20
    return cells, nuclei


def test_matching_keeps_largest_nucleus_and_shared_compartment_ids():
    cells, nuclei = _example_labels()
    matched_cells, matched_nuclei, cytoplasm, last_id = _match_cells_and_nuclei(
        cells, nuclei, first_id=8
    )

    assert set(np.unique(matched_cells)) == {0, 8, 9}
    assert set(np.unique(matched_nuclei)) == {0, 8, 9}
    assert matched_nuclei[0, 1, 1] == 0  # smaller nucleus 10 was discarded
    assert matched_cells[0, 5, 5] == 0  # cell 3 had no nucleus
    assert np.all(cytoplasm[matched_nuclei > 0] == 0)
    assert set(np.unique(cytoplasm)) <= {0, 8, 9}
    assert last_id == 9


def test_border_cell_removal_propagates_to_nucleus_and_cytoplasm():
    cells, nuclei = _example_labels()
    matched_cells, matched_nuclei, cytoplasm, last_id = _match_cells_and_nuclei(
        cells, nuclei, remove_border_cells=True
    )

    assert set(np.unique(matched_cells)) == {0, 1}
    assert set(np.unique(matched_nuclei)) == {0, 1}
    assert set(np.unique(cytoplasm)) == {0, 1}
    assert matched_nuclei[0, 2, 0] == 0
    assert last_id == 1


def test_multistep_produces_compartment_and_independent_foci_channels(monkeypatch):
    image = SimpleNamespace(
        data=np.zeros((1, 3, 1, 5, 5), dtype=np.uint8),
        scales={"x": 0.5, "y": 0.5},
    )
    calls = []

    def fake_segment(czyx, spec, settings, scales):
        calls.append((settings.target, settings.primary_channel))
        labels = np.zeros((1, 5, 5), dtype=np.uint32)
        if settings.target == "cells":
            labels[0, 1:4, 1:4] = 7
        elif settings.target == "nuclei":
            labels[0, 2, 2] = 4
        else:
            labels[0, 1, 1] = 1
        return labels, {
            "device": "cpu",
            "runtime_seconds": 0.5,
            "model_cache_hit": len(calls) > 1,
            "timings": {
                "import_seconds": 0.1,
                "model_load_seconds": 0.2,
                "inference_seconds": 0.3,
            },
        }

    monkeypatch.setattr(engine, "segment_czyx", fake_segment)
    log = []
    result = engine._segment_multistep_image(
        image,
        SegmentationSettings(
            nucleus_model="cellpose3:nuclei",
            foci_model_1="spotiflow:general",
            foci_channel_1=2,
            foci_model_2="spotiflow:general",
            foci_channel_2=2,
        ),
        startup_seconds=0.4,
        zarr_read_seconds=0.6,
        log=log.append,
    )

    assert result.labels.shape == (1, 5, 1, 5, 5)
    assert result.channel_labels == [
        "labels_cells",
        "labels_nuclei",
        "labels_cytoplasm",
        "labels_spots_channel_2",
        "labels_spots_channel_2",
    ]
    assert calls == [("cells", 1), ("nuclei", 1), ("spots", 2), ("spots", 2)]
    assert result.provenance["timings"]["inference_seconds"] == 1.2
    assert result.provenance["timings"]["startup_seconds"] == 0.4
    assert len(result.provenance["step_runs"]) == 4
    assert len(result.provenance["output_statistics"]) == 5
    assert any("device=CPU" in line for line in log)
    assert any("labels=" in line for line in log)
    assert any("Post-processing" in line for line in log)


@pytest.mark.parametrize(
    "model_id,target,label",
    [
        ("spotiflow:general", "spots", "spots"),
        ("stardist:SD_Foci_Aggregates", "foci", "foci"),
        ("stardist:SD_Foci_Finn", "foci", "foci"),
        ("cellpose3:bact_phase_cp3", "cells", "bacteria"),
        ("cellpose3:bact_fluor_cp3", "cells", "bacteria"),
    ],
)
def test_allowed_multistep_spot_model_dispatch(model_id, target, label):
    spec, resolved_target, resolved_label = engine._spot_model_dispatch(model_id)
    assert spec.id == model_id
    assert (resolved_target, resolved_label) == (target, label)


@pytest.mark.parametrize(
    "model_id", ["stardist:SD_Nuclei_Versatile", "cellpose3:cyto3"]
)
def test_multistep_spot_model_dispatch_rejects_unrelated_models(model_id):
    with pytest.raises(ValueError, match="Multi-step spot models"):
        engine._spot_model_dispatch(model_id)


def test_bacterial_foci_step_uses_supported_target_and_model_diameter(monkeypatch):
    image = SimpleNamespace(
        data=np.zeros((1, 1, 1, 5, 5), dtype=np.uint8),
        scales={"x": 0.5, "y": 0.5},
    )
    observed = {}

    def fake_segment(czyx, spec, settings, scales):
        observed.update(target=settings.target, diameter=settings.diameter)
        return np.zeros((1, 5, 5), dtype=np.uint32), {
            "device": "cpu",
            "runtime_seconds": 0.0,
            "model_cache_hit": False,
            "timings": {},
        }

    monkeypatch.setattr(engine, "segment_czyx", fake_segment)
    result = engine._segment_multistep_image(
        image,
        SegmentationSettings(
            cell_model="skip",
            nucleus_model="skip",
            foci_model_1="cellpose3:bact_phase_cp3",
            foci_channel_1=1,
            diameter=0,
        ),
    )

    assert observed == {"target": "cells", "diameter": -1.0}
    assert result.channel_labels == ["labels_bacteria_channel_1"]


def test_fast_cell_expansion_uses_physical_xy_distance():
    nuclei = np.zeros((1, 5, 7), dtype=np.uint32)
    nuclei[0, 2, 3] = 9
    cells = engine._expand_nuclei_to_cells(
        nuclei, distance_um=1.0, scales={"y": 1.0, "x": 0.5}
    )
    assert cells[0, 2, 1] == 9  # two X pixels are 1 µm
    assert cells[0, 0, 3] == 0  # two Y pixels are 2 µm


def test_step2_nuclei_create_consistent_compartments(monkeypatch):
    image = SimpleNamespace(
        data=np.zeros((1, 3, 1, 5, 5), dtype=np.uint8),
        scales={"x": 0.5, "y": 0.5},
    )
    calls = []

    def fake_segment(czyx, spec, settings, scales):
        calls.append((settings.target, settings.primary_channel, spec.id))
        labels = np.zeros((1, 5, 5), dtype=np.uint32)
        if settings.target == "cells":
            labels[0, 1:4, 1:4] = 8
        else:
            labels[0, 2, 2] = 3
        return labels, {
            "device": "cpu",
            "runtime_seconds": 0.0,
            "model_cache_hit": False,
            "timings": {},
        }

    monkeypatch.setattr(engine, "segment_czyx", fake_segment)
    result = engine._segment_multistep_image(
        image, SegmentationSettings(nucleus_model="cellpose3:nuclei")
    )
    assert result.channel_labels == [
        "labels_cells",
        "labels_nuclei",
        "labels_cytoplasm",
    ]
    assert calls == [
        ("cells", 1, "cellpose3:cyto3"),
        ("nuclei", 1, "cellpose3:nuclei"),
    ]
    assert set(np.unique(result.labels[:, 0])) == {0, 1}
    assert set(np.unique(result.labels[:, 1])) == {0, 1}


def test_no_enabled_steps_is_rejected():
    with pytest.raises(ValueError, match="Select at least one segmentation step"):
        SegmentationSettings(cell_model="skip").validate_steps()


def test_direct_step1_without_step2_produces_cells_only(monkeypatch):
    image = SimpleNamespace(
        data=np.zeros((1, 3, 1, 5, 5), dtype=np.uint8),
        scales={"x": 0.5, "y": 0.5},
    )

    observed = {}

    def fake_segment(czyx, spec, settings, scales):
        observed["nuclei_channel"] = settings.nuclei_channel
        labels = np.zeros((1, 5, 5), dtype=np.uint32)
        labels[0, 1:4, 1:4] = 2
        return labels, {
            "device": "cpu",
            "runtime_seconds": 0.0,
            "model_cache_hit": False,
            "timings": {},
        }

    monkeypatch.setattr(engine, "segment_czyx", fake_segment)
    result = engine._segment_multistep_image(image, SegmentationSettings())
    assert result.channel_labels == ["labels_cells"]
    assert observed["nuclei_channel"] == 0


def test_expansion_and_duplicate_step2_model_both_run(monkeypatch):
    image = SimpleNamespace(
        data=np.zeros((1, 3, 1, 5, 5), dtype=np.uint8),
        scales={"x": 0.5, "y": 0.5},
    )
    calls = []

    def fake_segment(czyx, spec, settings, scales):
        calls.append((spec.id, settings.primary_channel))
        labels = np.zeros((1, 5, 5), dtype=np.uint32)
        labels[0, 2, 2] = 1
        return labels, {
            "device": "cpu",
            "runtime_seconds": 0.0,
            "model_cache_hit": bool(calls[:-1]),
            "timings": {},
        }

    monkeypatch.setattr(engine, "segment_czyx", fake_segment)
    result = engine._segment_multistep_image(
        image,
        SegmentationSettings(
            cell_model="expand:cellpose3:nuclei",
            cell_channel=1,
            cell_nuclei_channel=3,
            nucleus_model="cellpose3:nuclei",
            nucleus_channel=2,
        ),
    )
    assert calls == [("cellpose3:nuclei", 3), ("cellpose3:nuclei", 2)]
    assert result.channel_labels == [
        "labels_cells",
        "labels_nuclei",
        "labels_cytoplasm",
    ]


@pytest.mark.parametrize(
    "cell_model,message",
    [
        ("unknown", "Unknown Step 1 selection"),
        ("expand:", "must include a nucleus model"),
        ("expand:unknown", "Unknown Step 1 expansion model"),
    ],
)
def test_invalid_step1_selector_is_rejected(cell_model, message):
    with pytest.raises(ValueError, match=message):
        SegmentationSettings(cell_model=cell_model).validate_steps()
