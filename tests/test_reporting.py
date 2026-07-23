from types import SimpleNamespace

import numpy as np

from cisegmentation.reporting import (
    format_effective_parameters,
    format_label_statistics,
    input_report_lines,
    label_statistics,
    workflow_report_lines,
)
from cisegmentation.settings import SegmentationSettings


def test_label_statistics_report_counts_sizes_and_physical_area():
    labels = np.array([[[0, 1, 1], [0, 2, 0]]], dtype=np.uint32)
    statistics = label_statistics(labels, {"x": 0.5, "y": 0.25})

    assert statistics["label_count"] == 2
    assert statistics["foreground_elements"] == 3
    assert statistics["mean_size_elements"] == 1.5
    assert statistics["median_size_elements"] == 1.5
    assert statistics["min_size_elements"] == 1
    assert statistics["max_size_elements"] == 2
    assert statistics["mean_physical_size"] == 0.1875
    assert statistics["physical_size_unit"] == "um^2"
    assert np.isclose(
        statistics["median_equivalent_diameter_um"],
        2 * np.sqrt(0.1875 / np.pi),
    )
    formatted = format_label_statistics(statistics)
    assert "labels=2" in formatted
    assert "size um^2" in formatted
    assert "pixels" not in formatted
    assert "median equivalent diameter=" in formatted


def test_label_statistics_use_pixels_when_scale_is_unknown():
    labels = np.array([[[0, 1, 1], [0, 2, 0]]], dtype=np.uint32)
    formatted = format_label_statistics(label_statistics(labels, {}))

    assert "size pixels mean/median/min/max=1.5/1.5/1/2" in formatted
    assert "equivalent diameter" not in formatted
    assert format_label_statistics(label_statistics(labels, {}), locations_only=True) == (
        "locations=2"
    )


def test_input_report_contains_shape_dtype_scale_and_channel_names(tmp_path):
    image = SimpleNamespace(
        data=np.zeros((2, 3, 4, 5, 6), dtype=np.uint16),
        source_dtype="uint16",
        scales={"t": 2.0, "z": 1.5, "y": 0.5, "x": 0.25},
        attrs={
            "multiscales": [
                {
                    "axes": [
                        {"name": "t", "unit": "second"},
                        {"name": "c"},
                        {"name": "z", "unit": "micrometer"},
                        {"name": "y", "unit": "micrometer"},
                        {"name": "x", "unit": "micrometer"},
                    ]
                }
            ],
            "omero": {"channels": [{"label": "DAPI"}, {"label": "RNA"}]},
        },
        resource=SimpleNamespace(
            store_path=tmp_path / "input.ome.zarr", image_path="A/1/0"
        ),
    )
    report = "\n".join(input_report_lines(image, 0.75))

    assert "T=2, C=3, Z=4, Y=5, X=6" in report
    assert "dtype=uint16" in report
    assert "X=0.25 micrometer" in report
    assert "C1=DAPI, C2=RNA" in report


def test_workflow_report_lists_selected_steps_and_tuning():
    settings = SegmentationSettings(
        cell_model="expand:cellpose3:nuclei",
        nucleus_model="stardist:SD_Nuclei_Versatile",
        foci_model_1="spotiflow:general",
    )
    report = "\n".join(workflow_report_lines(settings))

    assert "Step 1: expand nuclei with cellpose3:nuclei" in report
    assert "Step 2: stardist:SD_Nuclei_Versatile" in report
    assert "Step 3a: spotiflow:general" in report
    assert "requested device=auto" in report
    assert "output: native OME-Zarr 0.4 labels" in report
    assert "effective model parameters are reported" in report
    assert "Spotiflow output: single-pixel point locations" in report
    assert "labels log info: False" in report

    refined_report = "\n".join(
        workflow_report_lines(
            SegmentationSettings(spotiflow_local_refinement=True)
        )
    )
    assert "bounded local intensity instance masks" in refined_report

    native_report = "\n".join(
        workflow_report_lines(SegmentationSettings(write_ome_zarr_labels=True))
    )
    assert "native OME-Zarr 0.4 labels" in native_report

    channel_report = "\n".join(
        workflow_report_lines(SegmentationSettings(write_ome_zarr_labels=False))
    )
    assert "output: labels as image channels" in channel_report


def test_effective_parameters_report_model_defaults_and_stardist_rescaling():
    spotiflow = format_effective_parameters(
        {
            "adapter": "spotiflow",
            "probability_threshold": 0.5,
            "probability_source": "checkpoint thresholds.yaml",
            "minimum_distance_pixels": 4,
            "minimum_distance_um": 2.0,
        }
    )
    assert "probability=0.5 (checkpoint thresholds.yaml)" in spotiflow
    assert "minimum distance=4 px / 2.000 um" in spotiflow

    refined = format_effective_parameters(
        {
            "adapter": "spotiflow",
            "probability_threshold": 0.5,
            "probability_source": "checkpoint thresholds.yaml",
            "minimum_distance_pixels": 4,
            "minimum_distance_um": 2.0,
            "local_refinement": True,
            "refinement_max_radius_um": 1.0,
            "refinement_radius_y_pixels": 4.0,
            "refinement_radius_x_pixels": 5.0,
            "refinement_noise_sigmas": 3.0,
            "refinement_threshold_fraction": 0.3,
            "detected_points": 12,
            "refined_masks": 10,
            "grown_masks": 8,
            "single_pixel_fallbacks": 2,
            "suppressed_duplicate_seeds": 2,
            "overlap_pixels_removed": 4,
        }
    )
    assert "local mask refinement=bounded intensity growth" in refined
    assert "points=12, masks=10, grown=8" in refined

    stardist = format_effective_parameters(
        {
            "adapter": "stardist",
            "probability_threshold": 0.479,
            "probability_source": "thresholds.json",
            "nms_threshold": 0.3,
            "nms_source": "thresholds.json",
            "source_yx_shape": [80, 100],
            "model_yx_shape": [40, 80],
            "rescale_y_factor": 0.5,
            "rescale_x_factor": 0.8,
            "model_y_um": 0.5,
            "model_x_um": 0.5,
            "label_restoration": "scaled-polygons",
        }
    )
    assert "probability=0.479 (thresholds.json), NMS=0.3" in stardist
    assert "80x100 -> 40x80" in stardist
    assert "model pixel=0.500x0.500 um" in stardist
    assert "restoration=scaled-polygons" in stardist
