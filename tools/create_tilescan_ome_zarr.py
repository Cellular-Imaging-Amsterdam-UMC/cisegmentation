#!/usr/bin/env python
"""Create a chunked, single-channel synthetic OME-Zarr tilescan from a TIFF.

Example:
    python tools/create_tilescan_ome_zarr.py cells.tif cells-tilescan.ome.zarr \
        --height 40000 --width 40000 --seed 20260928

Requires numpy, tifffile, zarr>=2.18,<3, numcodecs>=0.12,<0.16. Reuses the
repository's NGFF axes, scale, channel, OME-XML and atomic-install helpers.
Writes one chunk at a time; it never allocates the complete large image.
The TIFF must be a square, single-channel 2D image. Rotations are multiples
of 90 degrees, avoiding resampling. Without --pixel-size-um, spatial units
remain uncalibrated rather than treating TIFF print resolution as microscopy
calibration. Existing output and partial stores are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

import numpy as np
from numcodecs import Blosc
import tifffile
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cisegmentation.ome_zarr_io import (
    ImageData, ImageResource, LabelResult, _axis_metadata, _install_store,
    _ome_xml, _scale_values, _source_channel_metadata, enumerate_resources,
)


def transformed_tile(tile, quarter_turns, flip_y, flip_x):
    result = np.rot90(tile, int(quarter_turns))
    if flip_y:
        result = result[::-1, :]
    if flip_x:
        result = result[:, ::-1]
    return np.ascontiguousarray(result)


def render_region(tile, transforms, y, x, height, width):
    """Render a window, including windows crossing transformed-tile seams."""
    size = tile.shape[0]
    result = np.empty((height, width), dtype=tile.dtype)
    for row in range(y // size, (y + height - 1) // size + 1):
        for column in range(x // size, (x + width - 1) // size + 1):
            yy, xx = row * size, column * size
            start_y, stop_y = max(y, yy), min(y + height, yy + size)
            start_x, stop_x = max(x, xx), min(x + width, xx + size)
            variant = transformed_tile(tile, *transforms[row, column])
            result[start_y-y:stop_y-y, start_x-x:stop_x-x] = variant[
                start_y-yy:stop_y-yy, start_x-xx:stop_x-xx
            ]
    return result


def mean_downsample(block):
    """Area-average 2x2 pixels, handling odd edges without losing coverage."""
    if block.shape[0] % 2 or block.shape[1] % 2:
        block = np.pad(block, ((0, block.shape[0] % 2), (0, block.shape[1] % 2)), mode='edge')
    average = block.astype(np.float32).reshape(
        block.shape[0] // 2, 2, block.shape[1] // 2, 2
    ).mean(axis=(1, 3))
    if np.issubdtype(block.dtype, np.integer):
        average = np.rint(average)
    return average.astype(block.dtype)


def create_tilescan(source_path, output_path, *, height=40000, width=40000,
                    seed=20260928, chunk_size=512, pyramid_min_size=512,
                    pixel_size_um=None, transforms=None, region_renderer=None,
                    extra_metadata=None, channel_label='Cells',
                    label_renderer=None, label_name='labels_nuclei_reference',
                    label_max_id=None):
    started = time.perf_counter()
    source_path, output_path = Path(source_path), Path(output_path)
    if min(height, width, chunk_size, pyramid_min_size) < 1:
        raise ValueError('Image sizes, chunk size and pyramid minimum must be positive')
    if pixel_size_um is not None and pixel_size_um <= 0:
        raise ValueError('Pixel size must be positive')
    if not output_path.name.lower().endswith('.ome.zarr'):
        raise ValueError('Output must end with .ome.zarr')
    temporary = output_path.with_name(output_path.name + '.partial')
    if output_path.exists() or temporary.exists():
        raise FileExistsError(f'Refusing to overwrite {output_path} or {temporary}')
    with tifffile.TiffFile(source_path) as tif:
        series = tif.series[0]
        if series.axes != 'YX' or len(series.shape) != 2:
            raise ValueError(f'Expected a single-channel YX TIFF, found {series.axes}/{series.shape}')
        tile = series.asarray()
    if tile.shape[0] != tile.shape[1]:
        raise ValueError('Square source tiles are required for 90-degree rotations')
    rows, columns = math.ceil(height / tile.shape[0]), math.ceil(width / tile.shape[1])
    rng = np.random.default_rng(seed)
    if transforms is None:
        transforms = np.stack([
            rng.integers(0, 4, (rows, columns)),
            rng.integers(0, 2, (rows, columns)),
            rng.integers(0, 2, (rows, columns)),
        ], axis=-1)
    transforms = np.asarray(transforms)
    if transforms.shape != (rows, columns, 3):
        raise ValueError('Transform grid must match the output tile grid')
    renderer = region_renderer or render_region
    if label_renderer is not None and (label_max_id is None or
            not 0 <= label_max_id <= np.iinfo(np.uint32).max):
        raise ValueError('Reference label maximum must fit uint32')
    scales = {axis: 1.0 for axis in 'tczyx'}
    if pixel_size_um is not None:
        scales['y'] = scales['x'] = pixel_size_um
    source = ImageData(tile[None,None,None], tuple('tczyx'), scales,
                       {'omero': {'channels': [{'label':channel_label, 'color':'FFFFFF'}]}},
                       ImageResource(source_path), str(tile.dtype))
    metadata_result = LabelResult(np.empty((0,), dtype=np.uint32), source,
                                  'synthetic-tilescan', 'image')
    channels = _source_channel_metadata(metadata_result, 1)
    channels[0]['window']['start'] = float(np.percentile(tile, 1))
    channels[0]['window']['end'] = float(np.percentile(tile, 99.5))
    name = output_path.name.removesuffix('.ome.zarr')
    root = zarr.open_group(str(temporary), mode='w', zarr_version=2)
    compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
    datasets = []
    arrays = []
    level_height, level_width = height, width
    while True:
        index = len(arrays)
        array = root.create_dataset(str(index), shape=(1,1,1,level_height,level_width),
            dtype=tile.dtype, chunks=(1,1,1,min(chunk_size,level_height),min(chunk_size,level_width)),
            compressor=compressor, dimension_separator='/')
        array.attrs['_ARRAY_DIMENSIONS'] = list('tczyx')
        arrays.append(array)
        datasets.append({'path':str(index), 'coordinateTransformations':[
            {'type':'scale', 'scale':_scale_values(source, 2**index)}]})
        if max(level_height, level_width) <= pyramid_min_size:
            break
        level_height, level_width = math.ceil(level_height / 2), math.ceil(level_width / 2)
    axes = _axis_metadata()
    if pixel_size_um is None:
        axes = [{key:value for key,value in axis.items() if key != 'unit'} for axis in axes]
    root.attrs['multiscales'] = [{'version':'0.4','name':name,'axes':axes,'datasets':datasets}]
    root.attrs['omero'] = {'version':'0.4','name':name,'channels':channels,
                           'rdefs':{'defaultT':0,'defaultZ':0,'model':'greyscale'}}
    root.attrs['synthetic_tilescan'] = {
        'source':source_path.name, 'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'source_shape_yx':list(tile.shape), 'target_shape_yx':[height,width], 'seed':seed,
        'tile_grid_yx':[rows,columns], 'transforms':transforms.tolist(),
        'transform_fields':['rotation_quarter_turns','flip_y','flip_x'],
        'edge_policy':'crop to exact target size', 'pyramid_filter':'2x2 mean; integer levels rounded',
        'pixel_size_um':pixel_size_um, 'complete':False,
    }
    if extra_metadata:
        root.attrs['synthetic_tilescan'] = {**dict(root.attrs['synthetic_tilescan']),
                                           'seam_repair': extra_metadata}
    label_arrays = []
    if label_renderer is not None:
        labels = root.require_group('labels')
        labels.attrs['labels'] = [label_name]
        group = labels.require_group(label_name)
        group.attrs['multiscales'] = [{'version':'0.4', 'name':label_name,
                                      'axes':axes, 'datasets':datasets}]
        group.attrs['image-label'] = {'version':'0.4', 'source':{'image':'../../'}}
        group.attrs['omero'] = {'version':'0.4', 'name':label_name,
            'channels':[{'label':label_name, 'color':'FFFFFF', 'active':True,
                'window':{'min':0, 'max':max(1,label_max_id), 'start':0,
                          'end':max(1,label_max_id)}}],
            'rdefs':{'defaultT':0, 'defaultZ':0, 'model':'greyscale'}}
        group.attrs['synthetic_reference'] = {
            'description':'Source segmentation masks and copied donor masks; not manually annotated ground truth',
            'pyramid_filter':'nearest-neighbour; instance IDs unchanged'}
        for index, array in enumerate(arrays):
            labels_array = group.create_dataset(str(index), shape=array.shape,
                dtype=np.uint32, chunks=array.chunks, compressor=compressor,
                dimension_separator='/')
            labels_array.attrs['_ARRAY_DIMENSIONS'] = list('tczyx')
            label_arrays.append(labels_array)
    for index, array in enumerate(arrays):
        h, w = array.shape[-2:]
        for y in range(0, h, chunk_size):
            bh = min(chunk_size, h-y)
            for x in range(0, w, chunk_size):
                bw = min(chunk_size, w-x)
                if index == 0:
                    block = renderer(tile, transforms, y, x, bh, bw)
                else:
                    previous = arrays[index-1]
                    block = mean_downsample(previous[0,0,0,y*2:min((y+bh)*2,previous.shape[-2]),x*2:min((x+bw)*2,previous.shape[-1])])
                array[0,0,0,y:y+bh,x:x+bw] = block
                if label_arrays:
                    if index == 0:
                        label_block = label_renderer(y, x, bh, bw)
                        if label_block.shape != (bh,bw) or label_block.dtype != np.dtype('uint32'):
                            raise ValueError('Reference renderer must return a uint32 array matching the chunk')
                        if int(label_block.max(initial=0)) > label_max_id:
                            raise ValueError('Rendered reference IDs exceed the declared maximum')
                    else:
                        previous_labels = label_arrays[index-1]
                        label_block = previous_labels[0,0,0,
                            y*2:min((y+bh)*2,previous_labels.shape[-2]):2,
                            x*2:min((x+bw)*2,previous_labels.shape[-1]):2]
                    label_arrays[index][0,0,0,y:y+bh,x:x+bw] = label_block
            if y == 0 or (y // chunk_size + 1) % 10 == 0 or y+bh == h:
                print(json.dumps({'level':index, 'shape_yx':[h,w], 'rows_done':y+bh,
                                  'elapsed_seconds':round(time.perf_counter()-started,1)}), flush=True)
    root.require_group('OME')
    xml = _ome_xml(metadata_result, escape(name, {'"':'&quot;'}), arrays[0], channels)
    if pixel_size_um is None:
        ET.register_namespace('', 'http://www.openmicroscopy.org/Schemas/OME/2016-06')
        element = ET.fromstring(xml)
        pixels = element.find('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels')
        for key in ('PhysicalSizeX','PhysicalSizeY','PhysicalSizeZ'):
            pixels.attrib.pop(key, None)
        xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(element, encoding='unicode')
    (temporary/'OME'/'METADATA.ome.xml').write_text(xml, encoding='utf-8')
    # Validate exact pixel values at tile seams, corners and random positions.
    points = [(0,0),(max(0,tile.shape[0]-8),max(0,tile.shape[1]-8)),
              (height-1,width-1),(max(0,height-31),max(0,width-29))]
    check_rng = np.random.default_rng(seed+1)
    points += [(int(check_rng.integers(height)), int(check_rng.integers(width))) for _ in range(16)]
    for y, x in points:
        bh, bw = min(23,height-y), min(29,width-x)
        np.testing.assert_array_equal(arrays[0][0,0,0,y:y+bh,x:x+bw], renderer(tile,transforms,y,x,bh,bw))
        if label_arrays:
            np.testing.assert_array_equal(label_arrays[0][0,0,0,y:y+bh,x:x+bw],
                                          label_renderer(y,x,bh,bw))
    for index in range(1, len(arrays)):
        previous, current = arrays[index-1], arrays[index]
        for y, x in [(0,0),(max(0,current.shape[-2]-17),max(0,current.shape[-1]-19))]:
            bh, bw = min(17,current.shape[-2]-y), min(19,current.shape[-1]-x)
            expected = mean_downsample(previous[0,0,0,y*2:min((y+bh)*2,previous.shape[-2]),x*2:min((x+bw)*2,previous.shape[-1])])
            np.testing.assert_array_equal(current[0,0,0,y:y+bh,x:x+bw], expected)
    manifest = dict(root.attrs['synthetic_tilescan'])
    manifest.update({'complete':True, 'elapsed_seconds':round(time.perf_counter()-started,2),
                     'validation':'exact source windows, tile seams, edges and every pyramid level passed'})
    root.attrs['synthetic_tilescan'] = manifest
    root.store.close()
    _install_store(temporary, output_path)
    assert len(enumerate_resources(output_path)) == 1
    summary = {'output':str(output_path), 'shape_tczyx':[1,1,1,height,width],
               'dtype':str(tile.dtype), 'tile_grid_yx':[rows,columns], 'levels':len(arrays),
               'validation':'PASS', 'elapsed_seconds':manifest['elapsed_seconds']}
    print(json.dumps(summary), flush=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--height', type=int, default=40000)
    parser.add_argument('--width', type=int, default=40000)
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--chunk-size', type=int, default=512)
    parser.add_argument('--pyramid-min-size', type=int, default=512)
    parser.add_argument('--pixel-size-um', type=float)
    args = parser.parse_args()
    create_tilescan(args.source, args.output, height=args.height, width=args.width,
                    seed=args.seed, chunk_size=args.chunk_size,
                    pyramid_min_size=args.pyramid_min_size, pixel_size_um=args.pixel_size_um)


if __name__ == '__main__':
    main()
