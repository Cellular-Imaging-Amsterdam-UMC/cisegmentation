"""Paired-channel seamless fields with complete cell and aggregate support.

Contrast enhancement is detection-only. Both raw channels use exactly the same
donor identities, transforms and placements, including extracellular aggregates.
"""
from pathlib import Path
import json

import numpy as np
from scipy import ndimage as ndi
import tifffile
import zarr

from tools.create_tilescan_ome_zarr import create_tilescan, transformed_tile
from tools.create_seamless_nuclei_tilescan import RandomNucleiPlan, checked_labels


class CellChannelPlan(RandomNucleiPlan):
    """Feather cut cytoplasm margins while preserving nuclei and bright spots."""
    def __init__(self, image, labels, height, width, *, protected, **kwargs):
        labels = checked_labels(labels, image.shape)
        self.protected = protected
        self.source_bounds = ndi.find_objects(labels)
        super().__init__(image, labels, height, width, **kwargs)

    def _patch(self, donor, orientation):
        patch = super()._patch(donor, orientation)
        if not hasattr(patch, 'protected_core'):
            ys, xs = self.source_bounds[patch.source_id-1]
            region = (slice(ys.start-self.halo, ys.stop+self.halo),
                      slice(xs.start-self.halo, xs.stop+self.halo))
            core = self.protected[region] & (self.labels[region] == patch.source_id)
            core = transformed_tile(core, orientation % 4, 0, orientation // 4)
            # Neighbour exclusion can end the outer halo abruptly. Taper inward
            # there, while all nuclear/aggregate pixels stay exactly unchanged.
            available = ndi.distance_transform_edt(np.pad(patch.alpha > 0, 1))[1:-1,1:-1]
            patch.alpha *= np.clip(available/12, 0, 1).astype(np.float32)
            patch.alpha[core] = 1
            patch.protected_core = core
        return patch

    def validate(self):
        for p in self.placements:
            h,w = p.patch.mask.shape
            actual = self.render_labels(p.y,p.x,h,w) == p.instance_id
            np.testing.assert_array_equal(actual,p.patch.mask)
            assert ndi.label(actual)[1] == 1
            pixels = self.render(None,None,p.y,p.x,h,w)
            core = p.patch.protected_core
            np.testing.assert_array_equal(pixels[core],p.patch.image[core])
            assert p.y > 0 and p.x > 0 and p.y+h < self.height and p.x+w < self.width
        return {'validation':'PASS', 'reference_objects':len(self.placements),
                'max_reference_id':self.next_id-1, 'donor_pool':len(self.donors),
                'target_objects':self.target_objects,
                'validation_checks':['identical complete object masks', 'connected cells/objects',
                    'exact native nucleus and aggregate pixels', 'no cropped outer-boundary objects'],
                'cytoplasm_margin_feather_pixels':12}


def read_channels(source, field=None):
    source = Path(source)
    if source.is_dir():
        root = zarr.open_group(str(source), mode='r')
        group = root[field] if field else root
        attrs = dict(group.attrs)
        attrs = attrs.get('ome', attrs)
        multiscale = attrs['multiscales'][0]
        axes = [a['name'] if isinstance(a, dict) else a for a in multiscale['axes']]
        array = group[multiscale['datasets'][0]['path']]
        selected = tuple(slice(None) if a in ('c', 'y', 'x') else 0 for a in axes)
        pixels = np.asarray(array[selected])
        remaining = [a for a in axes if a in ('c', 'y', 'x')]
        if 'c' not in remaining:
            pixels = pixels[None]
            remaining.insert(0, 'c')
        pixels = np.transpose(pixels, [remaining.index(a) for a in 'cyx'])
        channels = attrs.get('omero', {}).get('channels', [])
        names = [channels[i].get('label', f'Channel {i+1}') if i < len(channels) else f'Channel {i+1}' for i in range(len(pixels))]
        transform = next((t['scale'] for t in multiscale['datasets'][0].get('coordinateTransformations', []) if t.get('type') == 'scale'), [])
        scales = {a: float(transform[i]) for i,a in enumerate(axes) if i < len(transform)}
        for i,a in enumerate(axes):
            axis = multiscale['axes'][i]
            if a in ('x','y') and (not isinstance(axis,dict) or axis.get('unit') not in ('micrometer','micrometre')):
                scales.pop(a, None)
        return pixels, names, scales
    with tifffile.TiffFile(source) as tif:
        series = tif.series[0]
        if series.axes != 'CYX':
            raise ValueError('Multichannel TIFF must use CYX axes')
        return series.asarray(), [f'Channel {i+1}' for i in range(series.shape[0])], {}


def detection_contrast(image):
    """Suppress smooth green background and clip bright aggregates for Cellpose."""
    smooth = ndi.gaussian_filter(image.astype(np.float32), 1)
    signal = smooth - ndi.gaussian_filter(smooth, 50)
    low, high = np.percentile(signal, [5, 95])
    enhanced = np.sqrt(np.clip((signal-low)/max(1, high-low), 0, 1))
    return enhanced, {'background_sigma_px': 50, 'smoothing_sigma_px': 1,
                      'percentiles': [5, 95], 'low': float(low), 'high': float(high), 'gamma': 0.5}


def segment_cells(pixels, *, nuclei_channel=1, cell_channel=2, device='cpu', models_root=None, source_scales=None):
    from tools.create_seamless_nuclei_tilescan import segment_source
    from cisegmentation.adapters import segment_czyx
    from cisegmentation.registry import get_model_spec
    from cisegmentation.settings import SegmentationSettings
    if max(nuclei_channel, cell_channel) > len(pixels) or min(nuclei_channel, cell_channel) < 1:
        raise ValueError('Channel numbers are one-based and must exist in the source')
    nuclei, nuclei_info = segment_source(pixels[nuclei_channel-1], device=device, models_root=models_root)
    detection = pixels.astype(np.float32)
    detection[cell_channel-1], contrast = detection_contrast(pixels[cell_channel-1])
    # The adapter normalizes selected channels independently. Its default cell
    # diameter is expressed in micrometres; use the source calibration here.
    model = 'cellpose3:cyto3'
    scales = source_scales or {}
    calibrated = all(np.isfinite(scales.get(a, np.nan)) and scales[a] > 0 for a in ('x','y'))
    cells, cell_info = segment_czyx(detection[:, None], get_model_spec(model),
        SegmentationSettings(model=model, target='cells', primary_channel=cell_channel,
                             nuclei_channel=nuclei_channel, device=device, dimension_mode='slice-2d',
                             diameter=0 if calibrated else -1), scales, bounded=True)
    return nuclei, cells[0], {'nuclei': nuclei_info, 'cells': cell_info, 'detection_contrast': contrast}


def protected_objects(cells, nuclei, aggregate_channel):
    """Join intersecting components so a donor never cuts a nucleus or spot."""
    labels = cells.astype(np.uint32).copy()
    next_id = int(labels.max()) + 1
    def include(mask):
        nonlocal next_id
        overlaps = np.unique(labels[mask]); overlaps = overlaps[overlaps > 0]
        if len(overlaps):
            value = int(overlaps[0])
            if len(overlaps) > 1:
                labels[np.isin(labels, overlaps)] = value
        else:
            value = next_id; next_id += 1
        labels[mask] = value
    for label in np.unique(nuclei):
        if label: include(nuclei == label)
    raw = aggregate_channel.astype(np.float32)
    spots = ndi.gaussian_filter(raw, .7) - ndi.gaussian_filter(raw, 3)
    median = float(np.median(spots))
    noise = float(1.4826 * np.median(np.abs(spots-median)))
    components, _ = ndi.label(spots > median + max(1, 5*noise))
    external = 0
    for label, bounds in enumerate(ndi.find_objects(components), 1):
        if bounds is None: continue
        mask = components == label
        if np.count_nonzero(mask) < 2: continue
        if not np.any(labels[mask]): external += 1
        include(mask)
    from skimage.segmentation import relabel_sequential
    labels = relabel_sequential(labels)[0].astype(np.uint32)
    return labels, {'extracellular_aggregate_components': external,
                    'aggregate_detection': 'DoG 0.7/3 px, median + 5 MAD sigma; >=2 pixels',
                    'aggregate_noise': noise, 'protected_objects': int(labels.max())}


def build_multichannel(source, output, *, source_field=None, rows=4, columns=4,
                       height=None, width=None, seed=20261005, clearance=160, halo=24,
                       gap=1, device='cpu', models_root=None, diagnostics_dir=None,
                       pixel_size_um=.34, chunk_size=512, nuclei_channel=1,
                       cell_channel=2, cell_mask=None, mask=None, **unused):
    from tools.create_seamless_nuclei_tilescan import RandomNucleiPlan, checked_labels
    output = Path(output)
    folder = Path(diagnostics_dir) if diagnostics_dir else output.with_name(output.name.removesuffix('.ome.zarr')+'_diagnostics')
    if output.exists() or folder.exists():
        raise FileExistsError('Output or diagnostics already exists')
    if unused.get('layout', 'random-field') != 'random-field':
        raise ValueError('Multichannel generation uses whole-cell random-field placement')
    if unused.get('export_tiff'):
        raise ValueError('Multichannel TIFF export is not implemented; use the OME-Zarr output')
    pixels, names, source_scales = read_channels(source, source_field)
    if len(pixels) < 2: raise ValueError('This mode requires at least two source channels')
    if (height is None) != (width is None): raise ValueError('Provide both height and width')
    height, width = height or rows*pixels.shape[-2], width or columns*pixels.shape[-1]
    if mask and cell_mask:
        nuclei, cells = tifffile.imread(mask), tifffile.imread(cell_mask)
        inference = {'supplied_nuclei_mask': str(mask), 'supplied_cell_mask': str(cell_mask)}
    elif mask or cell_mask:
        raise ValueError('Supply both nuclei and cell masks, or neither')
    else:
        nuclei, cells, inference = segment_cells(pixels, nuclei_channel=nuclei_channel,
            cell_channel=cell_channel, device=device, models_root=models_root, source_scales=source_scales)
    nuclei = checked_labels(nuclei, pixels.shape[-2:])
    cells = checked_labels(cells, pixels.shape[-2:])
    folder.mkdir(parents=True)
    tifffile.imwrite(folder/'source-nuclei-mask.tif', nuclei, compression='deflate')
    tifffile.imwrite(folder/'source-cell-mask.tif', cells, compression='deflate')
    objects, protection = protected_objects(cells, nuclei, pixels[cell_channel-1])
    tifffile.imwrite(folder/'source-protected-objects.tif', objects, compression='deflate')
    green = pixels[cell_channel-1].astype(np.float32)
    spots = ndi.gaussian_filter(green,.7)-ndi.gaussian_filter(green,3)
    threshold = np.median(spots) + max(1,5*protection['aggregate_noise'])
    protected = (nuclei > 0) | (spots > threshold)
    plans = [CellChannelPlan(image, objects, height, width, protected=protected,
                             seed=seed, clearance=clearance, halo=halo, gap=gap,
                             include_all_sizes=True) for image in pixels]
    geometry = lambda plan: [(p.patch.source_id, p.y, p.x, p.transform, p.instance_id) for p in plan.placements]
    for plan in plans[1:]:
        assert geometry(plan) == geometry(plans[0]), 'Channels lost registration'
    validations = [plan.validate() for plan in plans]
    report = {'source': str(source), 'source_field': source_field, 'shape_cyx': [len(pixels), height, width],
              'pixel_size_um': pixel_size_um, 'source_scales': source_scales, 'source_segmentation': inference, 'protection': protection,
              'channels': names, 'channel_registration': 'identical donor IDs, positions and orientations',
              'validation': 'PASS', 'channel_validation': validations,
              'reference_warning': 'Model-derived synthetic reference masks; not biological ground truth'}
    create_tilescan(source, output, height=height, width=width, seed=seed, chunk_size=chunk_size,
        pixel_size_um=pixel_size_um, transforms=plans[0].transforms, channel_images=pixels,
        channel_renderers=[plan.render for plan in plans], channel_names=names,
        channel_colors=['FFFFFF', '00FF00'] + ['FF00FF'] * max(0,len(pixels)-2),
        label_renderer=plans[0].render_labels, label_name='labels_protected_objects_reference',
        label_max_id=validations[0]['max_reference_id'], extra_metadata=report)
    (folder/'placement-plan.json').write_text(json.dumps(plans[0].manifest(), indent=2), encoding='utf-8')
    (folder/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    save_previews(pixels, cells, nuclei, plans, output, folder)
    print(json.dumps({'output': str(output), 'diagnostics': str(folder), 'validation': 'PASS'}), flush=True)
    return report


def save_previews(pixels, cells, nuclei, plans, output, folder):
    from PIL import Image
    limits = [np.percentile(p, [1, 99.5]) for p in pixels]
    def rgb(data):
        channels = [np.clip((p.astype(float)-lo)/max(1,hi-lo), 0, 1) for p,(lo,hi) in zip(data,limits)]
        return np.uint8(np.stack([channels[0], np.minimum(1, channels[0]+channels[1]), channels[0]],-1)*255)
    original = rgb(pixels)
    boundary = (cells > 0) & (cells != ndi.minimum_filter(cells, size=3))
    overlay = original.copy(); overlay[boundary] = [255,100,40]
    overlay[(nuclei > 0) & (nuclei != ndi.minimum_filter(nuclei, size=3))] = [0,180,255]
    Image.fromarray(original).save(folder/'source-two-channels.png')
    Image.fromarray(overlay).save(folder/'source-cell-nuclei-outlines.png')
    Image.fromarray(rgb(np.stack([p.cleaned for p in plans]))).save(folder/'border-cleaned-two-channels.png')
    root = zarr.open_group(str(output),mode='r')
    datasets = root.attrs['multiscales'][0]['datasets']
    level = next(d['path'] for d in datasets if max(root[d['path']].shape[-2:]) <= 1600)
    Image.fromarray(rgb(root[level][0,:,0])).save(folder/'tilescan-overview.png')
    y, x = plans[0].size-256, plans[0].size-256
    Image.fromarray(rgb(root['0'][0,:,0,y:y+512,x:x+512])).save(folder/'seam-closeup.png')
