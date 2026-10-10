"""Copy the first N columns of an NGFF plate, retaining pixels and calibration."""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from uuid import uuid4


def subset_plate(source: Path, destination: Path, columns: int = 2):
    source, destination = source.resolve(), destination.resolve()
    if columns < 1:
        raise ValueError("At least one column is required")
    if destination == source or source in destination.parents:
        raise ValueError("Destination must be outside the source store")
    if destination.exists():
        raise FileExistsError(destination)
    attrs = json.loads((source / ".zattrs").read_text())
    plate = attrs["plate"]
    selected = {c["name"] for c in plate["columns"][:columns]}
    wells = [w for w in plate["wells"] if w["path"].split("/")[1] in selected]
    if not wells:
        raise ValueError("Selected columns contain no wells")
    partial = destination.with_name(destination.name + ".partial")
    partial.mkdir(parents=True, exist_ok=False)
    fields = 0
    for filename in (".zgroup",):
        shutil.copy2(source / filename, partial / filename)
    for well in wells:
        relative = Path(well["path"])
        row_dir = partial / relative.parent
        row_dir.mkdir(exist_ok=True)
        for filename in (".zattrs", ".zgroup"):
            original = source / relative.parent / filename
            if original.exists() and not (row_dir / filename).exists():
                shutil.copy2(original, row_dir / filename)
        shutil.copytree(source / relative, partial / relative)
        well_attrs = json.loads((source / relative / ".zattrs").read_text())
        fields += len(well_attrs["well"]["images"])
    attrs = deepcopy(attrs)
    attrs["plate"]["columns"] = plate["columns"][:columns]
    attrs["plate"]["wells"] = wells
    attrs["plate"]["name"] = destination.name.removesuffix(".ome.zarr")
    attrs["subset_provenance"] = {
        "source": str(source),
        "columns": sorted(selected),
        "fields": fields,
    }
    (partial / ".zattrs").write_text(json.dumps(attrs, indent=2), encoding="utf-8")
    xml_path = source / "OME" / "METADATA.ome.xml"
    if xml_path.exists():
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = {"ome": root.tag.split("}")[0].strip("{")}
        ET.register_namespace("", ns["ome"])
        image_ids = set()
        for xml_plate in root.findall("ome:Plate", ns):
            xml_plate.set("Name", attrs["plate"]["name"])
            xml_plate.set("Columns", str(min(columns, len(plate["columns"]))))
            for well in list(xml_plate.findall("ome:Well", ns)):
                if int(well.get("Column")) >= columns:
                    xml_plate.remove(well)
                else:
                    image_ids.update(
                        ref.get("ID") for ref in well.findall(".//ome:ImageRef", ns)
                    )
            sample_ids = {
                sample.get("ID")
                for sample in xml_plate.findall(".//ome:WellSample", ns)
            }
            for acquisition in xml_plate.findall("ome:PlateAcquisition", ns):
                for ref in list(acquisition.findall("ome:WellSampleRef", ns)):
                    if ref.get("ID") not in sample_ids:
                        acquisition.remove(ref)
        for image in list(root.findall("ome:Image", ns)):
            if image.get("ID") not in image_ids:
                root.remove(image)
        root.set("UUID", "urn:uuid:" + str(uuid4()))
        (partial / "OME").mkdir()
        tree.write(
            partial / "OME" / "METADATA.ome.xml", encoding="utf-8", xml_declaration=True
        )
        if len(image_ids) != fields:
            raise ValueError(
                f"OME XML has {len(image_ids)} selected images, NGFF has {fields}"
            )
    partial.rename(destination)
    return {
        "destination": str(destination),
        "wells": len(wells),
        "fields": fields,
        "columns": len(selected),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--columns", type=int, default=2)
    arguments = parser.parse_args()
    print(
        json.dumps(
            subset_plate(arguments.source, arguments.destination, arguments.columns),
            indent=2,
        )
    )
