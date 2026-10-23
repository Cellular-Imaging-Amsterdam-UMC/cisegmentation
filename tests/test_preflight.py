
import pytest
import zarr

from bilayers_cli import generate_cli_command, load_config
from cisegmentation import engine
from cisegmentation.preflight import validate_inputs
from cisegmentation.settings import SegmentationSettings


def image_store(tmp_path, name="source", *, xy=(0.34, 0.34), unit="micrometer",
                z=1.0, z_unit="micrometer", shape=(1, 1, 1, 8, 8)):
    path = tmp_path / f"{name}.ome.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.create_dataset("0", shape=shape, chunks=(1, 1, 1, 8, 8), dtype="u2")
    root.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [{"name": "t"}, {"name": "c"},
                 {"name": "z", "unit": z_unit},
                 {"name": "y", "unit": unit}, {"name": "x", "unit": unit}],
        "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [1, 1, z, xy[0], xy[1]]}]}],
    }]
    root.store.close()
    return path


def no_inference_or_pixels(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Preflight must not read pixels, initialize models or run inference")
    monkeypatch.setattr(engine, "_run_parallel_store", forbidden)
    monkeypatch.setattr(zarr.Array, "__getitem__", forbidden)


@pytest.mark.parametrize("field,extra", [
    ("cell_channel", {}),
    ("nucleus_channel", {"cell_model": "skip", "nucleus_model": "cellpose3:nuclei"}),
    *[(f"foci_channel_{slot}", {"cell_model": "skip", f"foci_model_{slot}": "spotiflow:general"})
      for slot in range(1, 5)],
])
def test_active_channel_zero_fails_with_one_based_help(field, extra):
    with pytest.raises(ValueError, match="Channels start at 1"):
        SegmentationSettings(**extra, **{field: 0}).validate_steps()


def test_optional_extra_nucleus_zero_is_unused_not_first_channel():
    settings = SegmentationSettings(cell_nuclei_channel=0)
    settings.validate_steps()
    assert settings.selected_channels(1) == [0]


def test_later_field_channel_count_aborts_complete_run(tmp_path, monkeypatch):
    image_store(tmp_path, "a-valid", shape=(1, 2, 1, 8, 8))
    image_store(tmp_path, "b-one-channel")
    no_inference_or_pixels(monkeypatch)
    settings = SegmentationSettings(cell_model="skip", foci_model_1="spotiflow:general", foci_channel_1=2)
    output = tmp_path / "out"
    logs = []
    with pytest.raises(ValueError, match=r"b-one-channel.*channel 2.*1\.\.1"):
        engine.run_workflow(tmp_path, output, settings, log=logs.append)
    assert not list(output.iterdir())
    assert any("preflight failed" in line for line in logs)


@pytest.mark.parametrize("value,unit", [
    (352.7777777778, "micrometer"), (2.0001, "micrometer"),
    (0, "micrometer"), (-1, "micrometer"), (float("nan"), "micrometer"),
    (float("inf"), "micrometer"), (0.34, None), (0.34, "pixel"),
])
def test_bad_calibration_stops_before_pixel_reads(tmp_path, monkeypatch, value, unit):
    path = image_store(tmp_path, xy=(value, value), unit=unit)
    original = (path / ".zattrs").read_bytes()
    no_inference_or_pixels(monkeypatch)
    logs = []
    with pytest.raises(ValueError, match="invalid or implausible calibration"):
        engine.run_workflow(path, tmp_path / "out", SegmentationSettings(), log=logs.append)
    assert any("source.ome.zarr" in line and "complete run" in line for line in logs)
    assert original == (path / ".zattrs").read_bytes()
    assert not list((tmp_path / "out").iterdir())


@pytest.mark.parametrize("xy,unit", [
    ((0.34, 0.34), "micrometer"), ((340, 340), "nanometer"),
    ((0.34, 0.5), "micrometer"), ((2.0, 2.0), "micrometer"),
])
def test_valid_calibration_is_unit_normalized_and_accepts_anisotropy(tmp_path, xy, unit):
    path = image_store(tmp_path, xy=xy, unit=unit)
    assert validate_inputs([path], SegmentationSettings())[0] == 1


def test_invalid_later_field_stops_complete_batch(tmp_path, monkeypatch):
    image_store(tmp_path, "a-valid")
    image_store(tmp_path, "b-invalid", xy=(352.7778, 352.7778))
    no_inference_or_pixels(monkeypatch)
    with pytest.raises(ValueError, match="b-invalid"):
        engine.run_workflow(tmp_path, tmp_path / "out", SegmentationSettings())
    assert not list((tmp_path / "out").iterdir())


def test_native_3d_requires_calibrated_z_but_slice_2d_does_not(tmp_path):
    path = image_store(tmp_path, z_unit=None, shape=(1, 1, 2, 8, 8))
    with pytest.raises(ValueError, match="X/Y/Z"):
        validate_inputs([path], SegmentationSettings())
    assert validate_inputs([path], SegmentationSettings(dimension_mode="slice-2d"))[0] == 1


def test_axial_spacing_is_not_subject_to_lateral_two_micron_limit(tmp_path):
    path = image_store(tmp_path, z=3.0, shape=(1, 1, 2, 8, 8))
    assert validate_inputs([path], SegmentationSettings())[0] == 1


@pytest.mark.parametrize("diameter", [0.001, 1e-300])
def test_unsafe_rescale_is_rejected_without_projected_allocation(tmp_path, monkeypatch, diameter):
    path = image_store(tmp_path)
    no_inference_or_pixels(monkeypatch)
    with pytest.raises(ValueError, match="unsafe Cellpose resampling.*effective diameter"):
        validate_inputs([path], SegmentationSettings(diameter=diameter))


def test_verified_bound_override_still_checks_resampling(tmp_path):
    path = image_store(tmp_path, xy=(352.7778, 352.7778))
    with pytest.raises(ValueError, match="rescale=423"):
        validate_inputs([path], SegmentationSettings(cellpose_max_pixel_size_um=400))


def test_model_default_diameter_does_not_bypass_calibration(tmp_path):
    path = image_store(tmp_path, xy=(352.7778, 352.7778))
    with pytest.raises(ValueError, match="calibration"):
        validate_inputs([path], SegmentationSettings(diameter=-1))


def test_projected_tile_ram_is_checked_without_reading_pixels(tmp_path, monkeypatch):
    path = image_store(tmp_path, shape=(1, 1, 1, 1024, 1024))
    no_inference_or_pixels(monkeypatch)
    with pytest.raises(ValueError, match="estimated working arrays"):
        validate_inputs([path], SegmentationSettings(diameter=5), ram_available=64 * 1024**2)


def test_database_descriptor_hints_and_channel_help():
    config = load_config()
    outputs = {item["name"]: item for item in config["outputs"]}
    assert outputs["measurements_database_out"]["type"] == "measurement"
    assert outputs["measurements_database_out"]["format"] == ["duckdb", "sqlite"]
    assert "--None" not in generate_cli_command(config)
    params = {item["name"]: item for item in config["parameters"]}
    assert params["measurements_database"]["default"] == "duckdb"
    for field in ("cell_channel", "nucleus_channel", *(f"foci_channel_{s}" for s in range(1, 5))):
        assert params[field]["default"] == params[field]["minimum"] == 1
        assert "1-based" in params[field]["label"]
    assert "0 = unused" in params["cell_nuclei_channel"]["label"]
