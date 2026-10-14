import numpy as np
import tifffile
import zarr

from tools.create_tilescan_ome_zarr import create_tilescan
from tools.create_seamless_nuclei_tilescan import RandomNucleiPlan
from tools.seamless_cells import detection_contrast, protected_objects


def test_two_channels_keep_exact_registration_and_calibration(tmp_path):
    image = np.arange(32*32, dtype=np.uint16).reshape(32, 32)
    source = tmp_path/'source.tif'
    tifffile.imwrite(source, image)
    output = tmp_path/'paired.ome.zarr'
    create_tilescan(source, output, height=64, width=64, chunk_size=17,
        pyramid_min_size=16, pixel_size_um=.34, channel_images=[image, image*3],
        channel_names=['DAPI', 'FITC'], channel_colors=['FFFFFF', '00FF00'])
    root = zarr.open_group(str(output), mode='r')
    assert root['0'].shape == (1, 2, 1, 64, 64)
    np.testing.assert_array_equal(root['0'][0,1,0], root['0'][0,0,0]*3)
    for i, dataset in enumerate(root.attrs['multiscales'][0]['datasets']):
        assert dataset['coordinateTransformations'][0]['scale'][-2:] == [.34*2**i]*2
    xml = (output/'OME/METADATA.ome.xml').read_text()
    assert 'PhysicalSizeX="0.34"' in xml and 'PhysicalSizeY="0.34"' in xml
    assert [c['label'] for c in root.attrs['omero']['channels']] == ['DAPI', 'FITC']


def test_protection_includes_whole_cell_nucleus_and_external_aggregate():
    cells = np.zeros((96,96), np.uint32); cells[20:45,20:45] = 1
    nuclei = np.zeros_like(cells); nuclei[30:50,30:40] = 1
    green = np.full(cells.shape, 100, np.uint16); green[70:73,70:73] = 3000
    before = green.copy()
    objects, report = protected_objects(cells, nuclei, green)
    assert np.all(objects[cells > 0] == objects[30,30])
    assert np.all(objects[nuclei > 0] == objects[30,30])
    assert objects[71,71] > 0 and objects[71,71] != objects[30,30]
    assert report['extracellular_aggregate_components'] > 0
    enhanced, _ = detection_contrast(green)
    assert enhanced.min() >= 0 and enhanced.max() <= 1
    np.testing.assert_array_equal(green, before)


def test_real_cell_patches_have_identical_placement_in_both_channels():
    y,x = np.indices((128,128)); labels = np.zeros((128,128),np.uint32)
    for i,(cy,cx) in enumerate([(32,32),(32,88),(88,32),(88,88)],1):
        labels[(y-cy)**2+(x-cx)**2 < 9**2] = i
    raw = (10 + (labels > 0)*(100+y+x)).astype(np.uint16)
    plans = [RandomNucleiPlan(im, labels, 256, 256, clearance=20,
        include_all_sizes=True) for im in [raw, raw*3]]
    geometry = lambda p: [(d.y,d.x,d.transform,d.patch.source_id) for d in p.placements]
    assert geometry(plans[0]) == geometry(plans[1])
    mask = plans[0].render_labels(0,0,256,256) > 0
    a,b = [p.render(None,None,0,0,256,256) for p in plans]
    np.testing.assert_array_equal(b[mask], a[mask]*3)


def test_cytoplasm_margin_is_soft_but_nuclear_pixels_stay_exact():
    from tools.seamless_cells import CellChannelPlan
    y,x = np.indices((192,192)); labels=np.zeros((192,192),np.uint32)
    protected=np.zeros((192,192),bool)
    for i,(cy,cx) in enumerate([(60,60),(60,130),(130,60),(130,130)],1):
        labels[(y-cy)**2+(x-cx)**2<12**2]=i
        protected |= (y-cy)**2+(x-cx)**2<5**2
    raw=(20+(labels>0)*200+protected*500).astype(np.uint16)
    plan=CellChannelPlan(raw,labels,384,384,protected=protected,clearance=24,halo=12,include_all_sizes=True)
    assert plan.validate()['validation']=='PASS'
    assert any(np.any((p.patch.alpha>0)&(p.patch.alpha<1)) for p in plan.placements)
    for p in plan.placements:
        assert np.all(p.patch.alpha[p.patch.protected_core]==1)
