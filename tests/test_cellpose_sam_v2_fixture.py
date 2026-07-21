from pathlib import Path

import numpy as np
import pytest

from cisegmentation.adapters import clear_model_cache
from cisegmentation.engine import _segment_multistep_image
from cisegmentation.ome_zarr_io import enumerate_resources, read_image
from cisegmentation.settings import SegmentationSettings


@pytest.mark.gpu
def test_cellpose_sam_v2_direct_cells_and_nucleus_expansion():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Cellpose-SAM v2 smoke test")
    checkpoint = Path("models/cellpose-sam/cpsam_v2")
    if not checkpoint.exists():
        pytest.skip("Downloaded Cellpose-SAM v2 checkpoint is required")

    image = read_image(
        enumerate_resources(Path("tests/data/nuclei-spots-cytoplasm.ome.zarr"))[0]
    )
    direct_settings = SegmentationSettings(
        cell_model="cellpose-sam:cpsam_v2",
        cell_channel=3,
        cell_nuclei_channel=1,
        nucleus_model="skip",
        remove_border_cells=False,
        device="cuda",
    )
    expansion_settings = SegmentationSettings(
        cell_model="expand:cellpose-sam:cpsam_v2",
        cell_nuclei_channel=1,
        cell_expansion_distance=5.0,
        nucleus_model="skip",
        remove_border_cells=False,
        device="cuda",
    )

    clear_model_cache()
    try:
        direct = _segment_multistep_image(image, direct_settings)
        expansion = _segment_multistep_image(image, expansion_settings)
    finally:
        clear_model_cache()

    assert direct.labels.shape == (1, 1, 1, 545, 423)
    assert direct.labels.dtype == np.uint32
    assert direct.channel_labels == ["labels_cells"]
    assert direct.labels.max() > 0

    assert expansion.labels.shape == (1, 3, 1, 545, 423)
    assert expansion.labels.dtype == np.uint32
    assert expansion.channel_labels == [
        "labels_cells",
        "labels_nuclei",
        "labels_cytoplasm",
    ]
    cells, nuclei, cytoplasm = expansion.labels[0]
    assert cells.max() > 0
    assert set(np.unique(cells)) == set(np.unique(nuclei))
    assert np.all(cells[nuclei > 0] == nuclei[nuclei > 0])
    assert np.all(cytoplasm[nuclei > 0] == 0)
