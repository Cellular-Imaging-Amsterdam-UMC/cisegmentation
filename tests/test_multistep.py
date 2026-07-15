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


def test_multistep_produces_compartment_and_repeated_spot_channels(monkeypatch):
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
    result = engine._segment_multistep_image(
        image,
        SegmentationSettings(multi_step=True, spot_channels="2,2"),
        startup_seconds=0.4,
        zarr_read_seconds=0.6,
    )

    assert result.labels.shape == (1, 5, 1, 5, 5)
    assert result.channel_labels == [
        "cells",
        "nuclei",
        "cytoplasm",
        "spots channel 2 (1)",
        "spots channel 2 (2)",
    ]
    assert calls == [("cells", 3), ("nuclei", 1), ("spots", 2), ("spots", 2)]
    assert result.provenance["timings"]["inference_seconds"] == 1.2
    assert result.provenance["timings"]["startup_seconds"] == 0.4


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


def test_bacterial_spot_step_uses_supported_target_and_model_diameter(monkeypatch):
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
            multi_step=True,
            cell_step=False,
            nucleus_step=False,
            spot_step=True,
            spot_model="cellpose3:bact_phase_cp3",
            spot_channels=[1],
            diameter=0,
        ),
    )

    assert observed == {"target": "cells", "diameter": -1.0}
    assert result.channel_labels == ["bacteria channel 1 (1)"]
