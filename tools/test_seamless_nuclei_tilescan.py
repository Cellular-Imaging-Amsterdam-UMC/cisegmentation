"""Geometric checks for the synthetic mosaic, independent of inference models."""
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi
import tifffile
import zarr

from tools.create_seamless_nuclei_tilescan import NucleiPlan, RandomNucleiPlan, clean_tile, build
from tools.create_tilescan_ome_zarr import create_tilescan, render_region


@pytest.fixture
def source():
    y,x = np.indices((128,128))
    image = (1000+y*2+x*3).astype(np.uint16)
    labels = np.zeros(image.shape,np.uint32)
    centres = [(40,40),(40,85),(85,40),(85,85),(0,60),(127,60)]
    for i,(cy,cx) in enumerate(centres,1):
        mask = ((y-cy)/6)**2+((x-cx)/5)**2 <= 1
        labels[mask] = i
        image[mask] = 20000+y[mask]*20+x[mask]*10
    return image,labels


def test_cleanup_removes_whole_border_objects_and_preserves_interior(source):
    image,labels = source
    before = image.copy()
    cleaned,kept,info = clean_tile(image,labels,clearance=16,halo=3)
    assert info['border_touching_objects'] == 2
    assert info['removed_objects'] == 2
    assert set(np.unique(kept)) == {0,1,2,3,4}
    assert np.max(cleaned[labels >= 5]) < 3000
    np.testing.assert_array_equal(cleaned[kept > 0],image[kept > 0])
    np.testing.assert_array_equal(image,before)


def test_shared_nuclei_remain_complete_for_all_eight_orientations(source):
    image,labels = source
    plan = NucleiPlan(image,labels,384,384,clearance=16,halo=3,gap=2)
    report = plan.validate()
    assert report['tile_variants_used'] == 8
    assert report['placements_by_kind']['junction'] == 4
    reference = plan.render_labels(0,0,384,384)
    actual = plan.render(None,None,0,0,384,384)
    assert len(np.unique(reference))-1 == report['reference_objects']
    for instance in np.unique(reference)[1:]:
        assert ndi.label(reference == instance)[1] == 1
    for p in plan.placements:
        if p.kind == 'junction':
            h,w = p.patch.mask.shape
            cy,cx = p.y+h//2,p.x+w//2
            assert reference[cy-1,cx-1] == reference[cy,cx] == p.instance_id
    # Arbitrary writer chunks must agree exactly with a single full rendering.
    raw_chunks = np.empty_like(actual)
    label_chunks = np.empty_like(reference)
    for y in range(0,384,37):
        for x in range(0,384,53):
            h,w = min(37,384-y),min(53,384-x)
            raw_chunks[y:y+h,x:x+w] = plan.render(None,None,y,x,h,w)
            label_chunks[y:y+h,x:x+w] = plan.render_labels(y,x,h,w)
    np.testing.assert_array_equal(raw_chunks,actual)
    np.testing.assert_array_equal(label_chunks,reference)


def test_partial_last_tiles_remove_objects_at_the_actual_output_border(source):
    image,labels = source
    plan = NucleiPlan(image,labels,320,350,clearance=16,halo=3,gap=2)
    report = plan.validate()
    reference = plan.render_labels(0,0,320,350)
    assert np.count_nonzero(reference[[0,-1]]) == 0
    assert np.count_nonzero(reference[:,[0,-1]]) == 0
    assert len(np.unique(reference))-1 == report['reference_objects']
    for instance in np.unique(reference)[1:]:
        assert ndi.label(reference == instance)[1] == 1


def test_plan_is_reproducible_and_rejects_unusable_masks(source):
    image,labels = source
    first = NucleiPlan(image,labels,256,256,seed=17,clearance=16,halo=3,gap=2)
    second = NucleiPlan(image,labels,256,256,seed=17,clearance=16,halo=3,gap=2)
    assert first.manifest() == second.manifest()
    with pytest.raises(ValueError,match='No source nuclei'):
        NucleiPlan(image,np.zeros_like(labels),256,256,clearance=16,halo=3)
    bad = labels.astype(np.int32)
    bad[0,0] = -1
    with pytest.raises(ValueError,match='uint32'):
        NucleiPlan(image,bad,256,256,clearance=16,halo=3)


def test_random_field_preserves_core_pixels_despite_overlapping_halos(source):
    image,labels = source
    plan = RandomNucleiPlan(image,labels,384,384,clearance=16,halo=6,gap=1)
    report = plan.validate()
    assert report['base_reference_objects'] == 0
    assert report['reference_objects'] == 54
    assert report['donor_orientations_used'] == 8
    reference = plan.render_labels(0,0,384,384)
    pixels = plan.render(None,None,0,0,384,384)
    assert len(np.unique(reference))-1 == 54
    for p in plan.placements:
        h,w = p.patch.mask.shape
        crop = pixels[p.y:p.y+h,p.x:p.x+w]
        np.testing.assert_array_equal(crop[p.patch.mask],p.patch.image[p.patch.mask])
        assert ndi.label(reference == p.instance_id)[1] == 1
    # The same background and halos are obtained independently of chunk edges.
    stitched = np.empty_like(pixels)
    for y in range(0,384,41):
        for x in range(0,384,59):
            h,w = min(41,384-y),min(59,384-x)
            stitched[y:y+h,x:x+w] = plan.render(None,None,y,x,h,w)
    np.testing.assert_array_equal(stitched,pixels)


def test_random_field_has_no_boundary_density_deficit_or_locked_rows():
    y,x = np.indices((256,256))
    labels = np.zeros((256,256),np.uint32)
    for i,(cy,cx) in enumerate(((cy,cx) for cy in range(12,256,20)
                               for cx in range(12,256,20)),1):
        labels[(y-cy)**2+(x-cx)**2 <= 25] = i
    image = np.where(labels > 0,20000+y*20+x,1000).astype(np.uint16)
    plan = RandomNucleiPlan(image,labels,1024,1024,clearance=16,halo=3,gap=1)
    centres = np.array([(p.y+p.patch.mask.shape[0]/2,p.x+p.patch.mask.shape[1]/2)
                        for p in plan.placements])
    distances = np.min(np.abs(centres[:,:,None]-np.array([256,512,768])),axis=(1,2))
    # Uniform proposals are independent of the former tile grid; sufficiently
    # many objects make a substantial boundary deficit a meaningful regression.
    observed = np.mean(distances < 24)
    expected = 1-(1-6*24/1024)**2
    assert 0.8 < observed/expected < 1.2
    assert len(np.unique(centres[:,0])) > 500
    assert len(np.unique(centres[:,1])) > 500


def test_random_field_reproduces_positions_and_cleans_partial_canvas(source):
    image,labels = source
    a = RandomNucleiPlan(image,labels,320,350,seed=19,clearance=16,halo=3)
    b = RandomNucleiPlan(image,labels,320,350,seed=19,clearance=16,halo=3)
    assert a.manifest() == b.manifest()
    assert a.validate()['reference_objects'] == round(6*320*350/128**2)


def test_writer_retains_32bit_reference_ids_and_odd_pyramid_edges(tmp_path):
    tile = np.arange(256,dtype=np.uint16).reshape(16,16)
    source = tmp_path/'source.tif'
    tifffile.imwrite(source,tile)
    def reference(y,x,h,w):
        yy,xx = np.ogrid[y:y+h,x:x+w]
        return (70000+yy*263+xx).astype(np.uint32)
    output = tmp_path/'labels.ome.zarr'
    create_tilescan(source,output,height=257,width=263,chunk_size=31,
        pyramid_min_size=32,label_renderer=reference,label_max_id=137590)
    root = zarr.open_group(str(output),mode='r')
    np.testing.assert_array_equal(root['labels/labels_nuclei_reference/0'][0,0,0],reference(0,0,257,263))
    group = root['labels/labels_nuclei_reference']
    for i in range(1,len(list(group.array_keys()))):
        assert group[str(i)].dtype == np.dtype('uint32')
        np.testing.assert_array_equal(group[str(i)][:],group[str(i-1)][:,...,::2,::2])
    assert group.attrs['omero']['channels'][0]['window']['end'] == 137590
    root.store.close()


def test_complete_tool_writes_tiff_diagnostics_and_refuses_overwrites(tmp_path,source):
    image,labels = source
    tiff,mask,output = tmp_path/'source.tif',tmp_path/'mask.npy',tmp_path/'test.ome.zarr'
    tifffile.imwrite(tiff,image)
    np.save(mask,labels)
    report = build(tiff,output,rows=2,columns=2,mask=mask,
        clearance=16,halo=3,gap=2,chunk_size=47,export_tiff=True)
    root = zarr.open_group(str(output),mode='r')
    folder = tmp_path/'test_diagnostics'
    np.testing.assert_array_equal(tifffile.imread(folder/'tilescan.tif'),root['0'][0,0,0])
    assert root.attrs['synthetic_tilescan']['seam_repair']['reference_objects'] == report['reference_objects']
    assert root.attrs['omero']['channels'][0]['label'] == 'Nuclei'
    assert (folder/'seam-before-after.png').exists()
    root.store.close()
    with pytest.raises(FileExistsError):
        build(tiff,output,mask=mask)


def test_original_generator_still_repeats_exact_source_pixels(tmp_path):
    tile = np.arange(256,dtype=np.uint16).reshape(16,16)
    source = tmp_path/'old.tif'
    tifffile.imwrite(source,tile)
    output = tmp_path/'old.ome.zarr'
    create_tilescan(source,output,height=35,width=37,chunk_size=13,pyramid_min_size=8)
    root = zarr.open_group(str(output),mode='r')
    transforms = np.array(root.attrs['synthetic_tilescan']['transforms'])
    np.testing.assert_array_equal(root['0'][0,0,0],render_region(tile,transforms,0,0,35,37))
    assert 'labels' not in root
    root.store.close()
