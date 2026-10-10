import json
import xml.etree.ElementTree as ET

import pytest

from tools.subset_ome_zarr_plate import subset_plate


def test_subset_keeps_pixels_metadata_and_xml_references(tmp_path):
    source = tmp_path / "source.ome.zarr"
    source.mkdir()
    (source / ".zgroup").write_text('{"zarr_format": 2}')
    wells = []
    for index, name in enumerate(["1", "2", "3"]):
        wells.append({"path": "A/" + name, "rowIndex": 0, "columnIndex": index})
        well = source / "A" / name
        (well / "0").mkdir(parents=True)
        (well / ".zattrs").write_text(json.dumps({"well": {"images": [{"path": "0"}]}}))
        (well / "0" / ".zattrs").write_text('{"calibration": 0.345}')
        (well / "0" / "pixels").write_bytes(bytes([index, 42]))
    (source / "A" / ".zgroup").write_text('{"zarr_format": 2}')
    attrs = {
        "plate": {
            "name": "original",
            "columns": [{"name": n} for n in ["1", "2", "3"]],
            "rows": [{"name": "A"}],
            "wells": wells,
        }
    }
    (source / ".zattrs").write_text(json.dumps(attrs))
    original = (source / ".zattrs").read_bytes()
    xml = '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Plate ID="Plate:0">'
    for i in range(3):
        xml += f'<Well ID="Well:{i}" Row="0" Column="{i}"><WellSample ID="WellSample:{i}" Index="{i}"><ImageRef ID="Image:{i}"/></WellSample></Well>'
    xml += (
        '<PlateAcquisition ID="PlateAcquisition:0">'
        + "".join(f'<WellSampleRef ID="WellSample:{i}"/>' for i in range(3))
        + "</PlateAcquisition></Plate>"
    )
    xml += "".join(f'<Image ID="Image:{i}"/>' for i in range(3)) + "</OME>"
    (source / "OME").mkdir()
    (source / "OME" / "METADATA.ome.xml").write_text(xml)
    destination = tmp_path / "subset.ome.zarr"
    result = subset_plate(source, destination)
    assert result["fields"] == result["wells"] == 2
    subset = json.loads((destination / ".zattrs").read_text())
    assert subset["plate"]["columns"] == [{"name": "1"}, {"name": "2"}]
    assert subset["plate"]["wells"] == wells[:2]
    assert not (destination / "A" / "3").exists()
    assert (destination / "A" / "2" / "0" / "pixels").read_bytes() == bytes([1, 42])
    assert (destination / "A" / "2" / "0" / ".zattrs").read_bytes() == (
        source / "A" / "2" / "0" / ".zattrs"
    ).read_bytes()
    assert (source / ".zattrs").read_bytes() == original
    root = ET.parse(destination / "OME" / "METADATA.ome.xml").getroot()
    ns = {"o": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    assert {image.get("ID") for image in root.findall("o:Image", ns)} == {
        "Image:0",
        "Image:1",
    }
    assert len(root.findall(".//o:WellSampleRef", ns)) == 2
    with pytest.raises(FileExistsError):
        subset_plate(source, destination)
    with pytest.raises(ValueError):
        subset_plate(source, source / "nested")
