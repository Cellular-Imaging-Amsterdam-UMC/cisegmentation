from pathlib import Path

import numpy as np
import pytest

from cisegmentation.adapters import clear_model_cache
from cisegmentation.engine import _segment_multistep_image
from cisegmentation.ome_zarr_io import enumerate_resources, read_image
from cisegmentation.settings import SegmentationSettings


@pytest.mark.gpu
def test_nuclei_spots_cytoplasm_expansion_matches_direct_cell_result_structure():
    """Regression test for routing expansion through the Step 1 nucleus channel."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the pinned Cellpose fixture comparison")
    if not Path("models/.complete.json").exists():
        pytest.skip("Downloaded inference models are required for this comparison")

    image = read_image(
        enumerate_resources(Path("tests/data/nuclei-spots-cytoplasm.ome.zarr"))[0]
    )
    expansion_settings = SegmentationSettings(
        cell_model="expand:cellpose3:nuclei",
        cell_channel=1,
        cell_nuclei_channel=0,
        cell_expansion_distance=5.0,
        nucleus_model="cellpose3:nuclei",
        nucleus_channel=1,
        remove_border_cells=False,
        device="cuda",
    )
    direct_settings = SegmentationSettings(
        cell_model="cellpose3:cyto3",
        cell_channel=3,
        cell_nuclei_channel=1,
        nucleus_model="cellpose3:nuclei",
        nucleus_channel=1,
        remove_border_cells=False,
        device="cuda",
    )

    clear_model_cache()
    try:
        expansion = _segment_multistep_image(image, expansion_settings)
        direct = _segment_multistep_image(image, direct_settings)
    finally:
        clear_model_cache()

    expansion_cells, expansion_nuclei, expansion_cytoplasm = expansion.labels[0]
    direct_cells = direct.labels[0, 0]
    assert expansion_settings.cell_expansion_channel() == 1
    assert expansion.channel_labels == [
        "labels_cells",
        "labels_nuclei",
        "labels_cytoplasm",
    ]
    assert len(np.unique(expansion_cells)) - 1 == 4
    assert len(np.unique(direct_cells)) - 1 == 4
    assert set(np.unique(expansion_cells)) == set(np.unique(expansion_nuclei))
    assert np.all(expansion_cells[expansion_nuclei > 0] == expansion_nuclei[expansion_nuclei > 0])
    assert np.all(expansion_cytoplasm[expansion_nuclei > 0] == 0)
    assert np.count_nonzero(expansion_cells) < np.count_nonzero(direct_cells)
