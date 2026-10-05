#!/usr/bin/env python
"""Build a random field of complete real nuclei, without visible tile joins.

Run the 4x4 experiment before scaling up; see SEAMLESS_NUCLEI_TILESCAN.md.
The original tile-repair method is available with --layout tile-repair.
Only the small source and donor patches are held in memory. Image/uint32
reference-mask pyramids are written in chunks.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage as ndi
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.create_tilescan_ome_zarr import create_tilescan, render_region, transformed_tile


def disk(radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    return y*y + x*x <= radius*radius


def edge_distance(shape):
    h, w = shape
    y, x = np.ogrid[:h, :w]
    return np.minimum(np.minimum(y, h-1-y), np.minimum(x, w-1-x))


def checked_labels(labels, shape):
    labels = np.asarray(labels)
    if labels.shape != shape or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Mask must be a 2D integer instance-label image matching the TIFF')
    if labels.min(initial=0) < 0 or labels.max(initial=0) > np.iinfo(np.uint32).max:
        raise ValueError('Source labels must fit uint32 and have background 0')
    # Sparse arbitrary input IDs are compacted before scipy.find_objects, which
    # otherwise allocates a list as long as the largest ID (potentially billions).
    from skimage.segmentation import relabel_sequential
    return np.asarray(relabel_sequential(labels.astype(np.uint32))[0], dtype=np.uint32)


def clean_tile(image, labels, *, clearance, halo, background_level=None):
    """Remove whole instances intersecting an edge corridor, never cut them.

    Inpaint expanded removed masks from nearby non-nuclear background. Fade
    only the empty frame to a common background value so rotated backgrounds
    do not themselves produce bright lines at tile boundaries.
    """
    distance = edge_distance(image.shape)
    removed = np.unique(labels[distance < clearance])
    removed = removed[removed != 0]
    border = np.unique(np.concatenate((labels[0], labels[-1], labels[:,0], labels[:,-1])))
    kept = labels.copy()
    removal = np.isin(labels, removed) if len(removed) else np.zeros(image.shape, bool)
    kept[removal] = 0
    support = ndi.binary_dilation(labels > 0, structure=disk(halo))
    if np.all(support):
        raise ValueError('No non-nuclear background remains; supply a better source mask')
    _, indices = ndi.distance_transform_edt(support, return_indices=True)
    background = image[tuple(indices)].astype(np.float32)
    background = ndi.gaussian_filter(background, sigma=max(1, halo/2))
    if background_level is None:
        background_level = float(np.median(image[~support]))
    erase = ndi.binary_dilation(removal, structure=disk(halo))
    # Preserve every surviving nucleus, including its immediate surroundings.
    erase &= ~ndi.binary_dilation(kept > 0, structure=disk(2))
    cleaned = image.astype(np.float32)
    cleaned[erase] = background[erase]
    fade = np.clip((clearance-distance) / max(1, clearance/2), 0, 1)
    cleaned = cleaned*(1-fade) + background_level*fade
    cleaned = np.rint(cleaned).clip(0, np.iinfo(image.dtype).max).astype(image.dtype)
    assert not np.any(kept[distance < clearance])
    np.testing.assert_array_equal(cleaned[kept > 0], image[kept > 0])
    return cleaned, kept, {
        'source_objects':int(np.count_nonzero(np.unique(labels))),
        'border_touching_objects':int(np.count_nonzero(border)),
        'removed_objects':int(len(removed)),
        'removed_source_ids':removed.tolist(),
        'kept_objects':int(np.count_nonzero(np.unique(kept))),
        'clearance_pixels':int(clearance), 'halo_pixels':int(halo),
        'background_level':background_level,
    }


@dataclass
class Patch:
    image: np.ndarray
    mask: np.ndarray
    alpha: np.ndarray
    source_id: int


@dataclass
class Placement:
    patch: Patch
    y: int
    x: int
    instance_id: int
    transform: int
    kind: str


def extract_donors(image, labels, *, clearance, halo, allow_neighbours=False):
    areas = np.bincount(labels.ravel())
    positive = areas[1:][areas[1:] > 0]
    if len(positive) == 0:
        raise ValueError('No source nuclei found')
    low, high = np.percentile(positive, [5, 95] if allow_neighbours else [10, 85])
    donors = []
    for label_id, bounds in enumerate(ndi.find_objects(labels), start=1):
        if bounds is None or not low <= areas[label_id] <= high:
            continue
        ys, xs = bounds
        y0, y1, x0, x1 = ys.start-halo, ys.stop+halo, xs.start-halo, xs.stop+halo
        if min(y0,x0) < clearance or y1 > image.shape[0]-clearance or x1 > image.shape[1]-clearance:
            continue
        local = labels[y0:y1,x0:x1]
        mask = local == label_id
        if ndi.label(mask)[1] != 1 or max(mask.shape) > 2*clearance:
            continue
        alpha = np.clip(1-ndi.distance_transform_edt(~mask)/halo, 0, 1).astype(np.float32)
        neighbours = (local != 0) & (local != label_id)
        if np.any(neighbours & (alpha > 0)):
            if not allow_neighbours:
                continue
            alpha[neighbours] = 0  # Never copy a different nucleus with this one.
        donors.append(Patch(image[y0:y1,x0:x1].copy(), mask, alpha, label_id))
    if not donors:
        raise ValueError('No isolated, complete donor nuclei fit the clearance; increase it or improve the mask')
    return donors


class NucleiPlan:
    """A small tile library plus globally positioned, shared donor objects.

    Edge repairs depend on neighbours. Eight independently repaired reusable
    tiles cannot guarantee matching halves under arbitrary rotations/flips.
    Instead every donor has one global location/orientation; both sides of
    each join (or all four tiles at a junction) sample that same object.
    """
    def __init__(self, image, labels, height, width, *, seed=20261005,
                 clearance=48, halo=6, gap=2):
        if image.ndim != 2 or image.shape[0] != image.shape[1]:
            raise ValueError('A square, single-channel 2D source is required')
        if image.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
            raise ValueError('The source image must be uint8 or uint16')
        if min(height,width) < 1 or not 0 < halo < clearance < image.shape[0]/2 or gap < 0:
            raise ValueError('Invalid image size, clearance, halo or gap')
        self.image = image
        self.labels = checked_labels(labels, image.shape)
        self.height, self.width = int(height), int(width)
        self.size = image.shape[0]
        self.rows, self.columns = math.ceil(height/self.size), math.ceil(width/self.size)
        self.clearance, self.halo, self.gap = clearance, halo, gap
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.cleaned, self.kept, self.cleaning = clean_tile(
            image,self.labels,clearance=clearance,halo=halo)
        self.donors = extract_donors(image,self.labels,clearance=clearance,halo=halo)
        # Balance the eight distinct dihedral orientations, rather than sampling
        # sixteen rotation/flip combinations that contain duplicate transforms.
        variants = np.concatenate([self.rng.permutation(8)
            for _ in range(math.ceil(self.rows*self.columns/8))])[:self.rows*self.columns]
        self.transforms = np.stack((variants%4,np.zeros_like(variants),variants//4),axis=-1)
        self.transforms = self.transforms.reshape(self.rows,self.columns,3)
        self.stride = int(self.labels.max())
        if self.rows*self.columns*self.stride >= np.iinfo(np.uint32).max:
            raise OverflowError('Base instance IDs exceed uint32')
        self.next_id = self.rows*self.columns*self.stride+1
        self.variants = {}
        self.patch_variants = {}
        self.placements = []
        self.buckets = {}
        self.attempts = 0
        self._place_seams()

    def _variant(self, row, column):
        transform = tuple(map(int,self.transforms[row,column]))
        h, w = min(self.size,self.height-row*self.size), min(self.size,self.width-column*self.size)
        key = (transform,h,w)
        if key not in self.variants:
            if h == self.size and w == self.size:
                raw = transformed_tile(self.cleaned,*transform)
                labels = transformed_tile(self.kept,*transform)
                count = self.cleaning['kept_objects']
            else:
                raw, labels, info = clean_tile(
                    transformed_tile(self.image,*transform)[:h,:w],
                    transformed_tile(self.labels,*transform)[:h,:w],
                    clearance=self.clearance,halo=self.halo,
                    background_level=self.cleaning['background_level'])
                count = info['kept_objects']
            self.variants[key] = (raw,labels,count)
        return self.variants[key]

    def _base(self, y, x, h, w, *, masks):
        if min(y,x) < 0 or min(h,w) < 1 or y+h > self.height or x+w > self.width:
            raise ValueError('Requested region is outside the mosaic')
        out = np.zeros((h,w),np.uint32) if masks else np.empty((h,w),self.image.dtype)
        for row in range(y//self.size,(y+h-1)//self.size+1):
            for column in range(x//self.size,(x+w-1)//self.size+1):
                yy,xx = row*self.size,column*self.size
                ay,by,ax,bx = max(y,yy),min(y+h,yy+self.size),max(x,xx),min(x+w,xx+self.size)
                raw, labels, _ = self._variant(row,column)
                block = (labels if masks else raw)[ay-yy:by-yy,ax-xx:bx-xx]
                if masks:
                    offset = (row*self.columns+column)*self.stride
                    block = np.where(block > 0,block.astype(np.uint64)+offset,0).astype(np.uint32)
                out[ay-y:by-y,ax-x:bx-x] = block
        return out

    def _nearby(self, y, x, h, w):
        ids = set()
        for row in range(y//self.size,(y+h-1)//self.size+1):
            for column in range(x//self.size,(x+w-1)//self.size+1):
                ids.update(self.buckets.get((row,column),()))
        return [self.placements[i] for i in sorted(ids)]

    @staticmethod
    def _intersection(p, y, x, h, w):
        ph,pw = p.patch.mask.shape
        ay,by,ax,bx = max(y,p.y),min(y+h,p.y+ph),max(x,p.x),min(x+w,p.x+pw)
        if by <= ay or bx <= ax:
            return None
        return ((slice(ay-y,by-y),slice(ax-x,bx-x)),
                (slice(ay-p.y,by-p.y),slice(ax-p.x,bx-p.x)))

    def render_labels(self, y, x, h, w):
        out = self._base(y,x,h,w,masks=True)
        for p in self._nearby(y,x,h,w):
            match = self._intersection(p,y,x,h,w)
            if match:
                dst,src = match
                out[dst][p.patch.mask[src]] = p.instance_id
        return out

    def render(self, _tile, _transforms, y, x, h, w):
        out = self._base(y,x,h,w,masks=False).astype(np.float32)
        reference = self.render_labels(y,x,h,w)
        for p in self._nearby(y,x,h,w):
            match = self._intersection(p,y,x,h,w)
            if match:
                dst,src = match
                # Feathered backgrounds may overlap, but each nucleus's pixels
                # belong exclusively to its own patch, regardless of draw order.
                alpha = np.where((reference[dst] == 0) |
                                 (reference[dst] == p.instance_id),p.patch.alpha[src],0)
                out[dst] = out[dst]*(1-alpha)+p.patch.image[src]*alpha
        return np.rint(out).clip(0,np.iinfo(self.image.dtype).max).astype(self.image.dtype)

    def _patch(self, donor, orientation):
        key = (donor,orientation)
        if key not in self.patch_variants:
            d = self.donors[donor]
            transform = (orientation%4,0,orientation//4)
            self.patch_variants[key] = Patch(
                transformed_tile(d.image,*transform),transformed_tile(d.mask,*transform),
                transformed_tile(d.alpha,*transform),d.source_id)
        return self.patch_variants[key]

    def _try_place(self, cy, cx, kind, *, crossing_y=None, crossing_x=None):
        self.attempts += 1
        orientation = int(self.rng.integers(8))
        patch = self._patch(int(self.rng.integers(len(self.donors))),orientation)
        h,w = patch.mask.shape
        y,x = int(cy)-h//2,int(cx)-w//2
        if min(y,x) < self.gap or y+h > self.height-self.gap or x+w > self.width-self.gap:
            return False
        yy,xx = np.nonzero(patch.mask)
        if crossing_y is not None and not y+yy.min() < crossing_y <= y+yy.max():
            return False
        if crossing_x is not None and not x+xx.min() < crossing_x <= x+xx.max():
            return False
        # Dilate the existing masks in a padded window, so proximity just
        # outside the patch is also checked. Halo blending never touches an
        # existing nucleus, preserving every copied nucleus's original pixels.
        ay,ax = max(0,y-self.gap),max(0,x-self.gap)
        by,bx = min(self.height,y+h+self.gap),min(self.width,x+w+self.gap)
        occupied = self.render_labels(ay,ax,by-ay,bx-ax) > 0
        occupied = ndi.binary_dilation(occupied,structure=disk(self.gap))
        occupied = occupied[y-ay:y-ay+h,x-ax:x-ax+w]
        if np.any(occupied & (patch.alpha > 0)):
            return False
        if self.next_id > np.iinfo(np.uint32).max:
            raise OverflowError('Donor instance IDs exceed uint32')
        placement = Placement(patch,y,x,self.next_id,orientation,kind)
        i = len(self.placements)
        self.placements.append(placement)
        self.next_id += 1
        for row in range(y//self.size,(y+h-1)//self.size+1):
            for column in range(x//self.size,(x+w-1)//self.size+1):
                self.buckets.setdefault((row,column),[]).append(i)
        return True

    def _place_seams(self):
        # Junctions first, before either perpendicular edge can occupy them.
        for y in range(self.size,self.height,self.size):
            for x in range(self.size,self.width,self.size):
                for _ in range(30):
                    if self._try_place(y,x,'junction',crossing_y=y,crossing_x=x):
                        break
                else:
                    raise RuntimeError(f'Could not repair junction {y},{x}; increase clearance')
        pitch = max(12,int(round(math.sqrt(self.size**2/self.cleaning['source_objects']))))
        diameter = float(np.median([max(np.ptp(np.nonzero(d.mask)[0]),
                                       np.ptp(np.nonzero(d.mask)[1]))+1 for d in self.donors]))
        extent = int(self.clearance+diameter/2)
        jitter = max(1,min(self.clearance//2,int(diameter/3)))
        for axis, length, normal_length in [('horizontal',self.width,self.height),
                                             ('vertical',self.height,self.width)]:
            for seam in range(self.size,normal_length,self.size):
                before = len(self.placements)
                # Random positions and normal jitter prevent an artificial
                # evenly spaced row of nuclei along the exact tile boundary.
                crossing_target = max(1,length//pitch)
                for _ in range(crossing_target*20):
                    if len(self.placements)-before >= crossing_target:
                        break
                    position = int(self.rng.integers(0,length))
                    offset = int(self.rng.integers(-jitter,jitter+1))
                    cy,cx = (seam+offset,position) if axis == 'horizontal' else (position,seam+offset)
                    self._try_place(cy,cx,axis,
                        crossing_y=seam if axis == 'horizontal' else None,
                        crossing_x=seam if axis == 'vertical' else None)
                target = max(crossing_target,int(round(
                    length*2*extent*self.cleaning['source_objects']/self.size**2)))
                for _ in range(target*20):
                    if len(self.placements)-before >= target:
                        break
                    position = int(self.rng.integers(0,length))
                    offset = int(self.rng.integers(-extent,extent+1))
                    cy,cx = (seam+offset,position) if axis == 'horizontal' else (position,seam+offset)
                    self._try_place(cy,cx,axis+'_fill')
                if len(self.placements) == before:
                    raise RuntimeError(f'No donor fits {axis} seam {seam}')
                print(json.dumps({'seam':axis,'coordinate':seam,
                    'donors_added':len(self.placements)-before}),flush=True)

    def validate(self):
        for p in self.placements:
            h,w = p.patch.mask.shape
            actual = self.render_labels(p.y,p.x,h,w) == p.instance_id
            np.testing.assert_array_equal(actual,p.patch.mask)
            pixels = self.render(None,None,p.y,p.x,h,w)
            np.testing.assert_array_equal(pixels[actual],p.patch.image[actual])
            if ndi.label(actual)[1] != 1:
                raise AssertionError('A donor nucleus was split into disconnected pieces')
        for y in [0,self.height-1]:
            for x in range(0,self.width,512):
                assert not np.any(self.render_labels(y,x,1,min(512,self.width-x)))
        for x in [0,self.width-1]:
            for y in range(0,self.height,512):
                assert not np.any(self.render_labels(y,x,min(512,self.height-y),1))
        count = sum(self._variant(r,c)[2] for r in range(self.rows) for c in range(self.columns))
        kinds = {k:sum(p.kind == k for p in self.placements) for k in sorted({p.kind for p in self.placements})}
        return {'validation':'PASS', 'base_reference_objects':count,
            'inserted_complete_nuclei':len(self.placements),
            'reference_objects':count+len(self.placements),
            'max_reference_id':self.next_id-1,'donor_pool':len(self.donors),
            'placements_by_kind':kinds,'placement_attempts':self.attempts,
            'tile_variants_used':len({tuple(t) for t in self.transforms.reshape(-1,3)}),
            'validation_checks':['intact connected donor masks','exact real donor pixels',
                'no nucleus overlap or cropped outer-border masks','shared global patches across all orientations']}

    def manifest(self):
        return {'seed':self.seed,'source_shape_yx':list(self.image.shape),
            'target_shape_yx':[self.height,self.width], 'transforms':self.transforms.tolist(),
            'cleaning':self.cleaning,'halo_pixels':self.halo,'gap_pixels':self.gap,
            'placements':[{'id':p.instance_id,'source_id':p.patch.source_id,
                'y':p.y,'x':p.x,'shape_yx':list(p.patch.mask.shape),
                'orientation':p.transform,'kind':p.kind} for p in self.placements]}


class RandomNucleiPlan(NucleiPlan):
    """One stationary random field of real nuclei, without tile-edge repairs.

    The tile grid controls output dimensions and comparison previews only.
    Whole nuclei cross those coordinates naturally. Collision tests use the
    instance cores rather than a fluorescence halo, so packing density is not
    artificially reduced around boundaries. Rendering protects every core
    while allowing background halos to blend.
    """
    bucket_size = 128

    def __init__(self,image,labels,height,width,*,seed=20261005,
                 clearance=48,halo=6,gap=1):
        if image.ndim != 2 or image.shape[0] != image.shape[1]:
            raise ValueError('A square, single-channel 2D source is required')
        if image.dtype not in (np.dtype('uint8'),np.dtype('uint16')):
            raise ValueError('The source image must be uint8 or uint16')
        if min(height,width) < 1 or not 0 < halo < clearance < image.shape[0]/2 or gap < 0:
            raise ValueError('Invalid image size, clearance, halo or gap')
        self.image,self.labels = image,checked_labels(labels,image.shape)
        self.height,self.width,self.size = int(height),int(width),image.shape[0]
        self.rows,self.columns = math.ceil(height/self.size),math.ceil(width/self.size)
        self.clearance,self.halo,self.gap,self.seed = clearance,halo,gap,seed
        self.rng = np.random.default_rng(seed)
        self.cleaned,self.kept,self.cleaning = clean_tile(image,self.labels,
            clearance=clearance,halo=halo)
        self.donors = extract_donors(image,self.labels,clearance=clearance,
                                    halo=halo,allow_neighbours=True)
        variants = np.concatenate([self.rng.permutation(8)
            for _ in range(math.ceil(self.rows*self.columns/8))])[:self.rows*self.columns]
        self.transforms = np.stack((variants%4,np.zeros_like(variants),variants//4),axis=-1)
        self.transforms = self.transforms.reshape(self.rows,self.columns,3)
        self.target_objects = max(1,round(self.cleaning['source_objects']*height*width/self.size**2))
        if self.target_objects > np.iinfo(np.uint32).max:
            raise OverflowError('Requested object count exceeds uint32')
        self.next_id = 1
        self.patch_variants,self.clearance_masks = {},{}
        self.placements,self.buckets = [],{}
        self.attempts = 0
        self._prepare_background()
        self._place_field()

    def _prepare_background(self):
        valid = ~ndi.binary_dilation(self.labels > 0,structure=disk(self.halo+2))
        if not np.any(valid):
            raise ValueError('No source background available')
        values = self.image[valid].astype(np.float32)
        self.background_level = float(np.median(values))
        residual = self.image.astype(np.float32)-ndi.gaussian_filter(self.image.astype(np.float32),2)
        highpass = residual[valid]
        self.background_noise = float(1.4826*np.median(np.abs(highpass-np.median(highpass))))
        spread = float(np.diff(np.percentile(values,[16,84]))[0]/2)
        self.background_variation = min(spread/2,self.background_level/4)
        rng = np.random.default_rng(self.seed ^ 0x5741)
        self.background_modes = [(float(rng.uniform(0,2*np.pi)),
            float(rng.uniform(120,700)),float(rng.uniform(0,2*np.pi))) for _ in range(6)]

    def _base(self,y,x,h,w,*,masks):
        if min(y,x) < 0 or min(h,w) < 1 or y+h > self.height or x+w > self.width:
            raise ValueError('Requested region is outside the mosaic')
        if masks:
            return np.zeros((h,w),np.uint32)
        yy,xx = np.ogrid[y:y+h,x:x+w]
        # Coordinate hashing makes background noise independent of writer
        # chunk sizes, with no tiled or repeated background edges.
        code = (yy.astype(np.uint64)*0x85EBCA77+xx.astype(np.uint64)*0x9E3779B1+
                np.uint64(self.seed & 0xFFFFFFFF)) & 0xFFFFFFFF
        code ^= code >> 16
        code = (code*0x7FEB352D) & 0xFFFFFFFF
        code ^= code >> 15
        noise = ((code & 0xFFFF).astype(np.float32)/65535-0.5)*math.sqrt(12)
        out = self.background_level+self.background_noise*noise
        amplitude = self.background_variation*math.sqrt(2/len(self.background_modes))
        for angle,wavelength,phase in self.background_modes:
            out += amplitude*np.sin((yy*np.sin(angle)+xx*np.cos(angle))*
                                     (2*np.pi/wavelength)+phase).astype(np.float32)
        return np.rint(out).clip(0,np.iinfo(self.image.dtype).max).astype(self.image.dtype)

    def _cells(self,y,x,h,w):
        for row in range(y//self.bucket_size,(y+h-1)//self.bucket_size+1):
            for column in range(x//self.bucket_size,(x+w-1)//self.bucket_size+1):
                yield row,column

    def _nearby(self,y,x,h,w):
        ids = set()
        for cell in self._cells(y,x,h,w):
            ids.update(self.buckets.get(cell,()))
        return [self.placements[i] for i in sorted(ids)]

    def _place_field(self):
        limit = self.target_objects*100
        while len(self.placements) < self.target_objects and self.attempts < limit:
            self.attempts += 1
            orientation = int(self.rng.integers(8))
            donor = int(self.rng.integers(len(self.donors)))
            patch = self._patch(donor,orientation)
            h,w = patch.mask.shape
            if h+2*self.gap > self.height or w+2*self.gap > self.width:
                continue
            y = int(self.rng.integers(self.gap,self.height-h-self.gap+1))
            x = int(self.rng.integers(self.gap,self.width-w-self.gap+1))
            key = donor,orientation
            if key not in self.clearance_masks:
                self.clearance_masks[key] = ndi.binary_dilation(
                    np.pad(patch.mask,self.gap),structure=disk(self.gap))
            exclusion = self.clearance_masks[key]
            occupied = False
            for existing in self._nearby(y-self.gap,x-self.gap,h+2*self.gap,w+2*self.gap):
                match = self._intersection(existing,y-self.gap,x-self.gap,h+2*self.gap,w+2*self.gap)
                if match:
                    dst,src = match
                    if np.any(exclusion[dst] & existing.patch.mask[src]):
                        occupied = True
                        break
            if occupied:
                continue
            p = Placement(patch,y,x,self.next_id,orientation,'field')
            i = len(self.placements)
            self.placements.append(p)
            self.next_id += 1
            for cell in self._cells(y,x,h,w):
                self.buckets.setdefault(cell,[]).append(i)
            if self.next_id % 2000 == 0 or len(self.placements) == self.target_objects:
                print(json.dumps({'random_field_objects':len(self.placements),
                    'target':self.target_objects,'attempts':self.attempts}),flush=True)
        if len(self.placements) != self.target_objects:
            raise RuntimeError(f'Could only pack {len(self.placements)} of {self.target_objects} '
                               'nuclei; reduce --gap or use a less dense source mask')

    def _variant(self,row,column):
        # No repeated source nuclei are retained in the new field.
        return None,None,0

    def validate(self):
        report = super().validate()
        report.update({'layout':'random-field','tile_variants_used':0,
            'donor_orientations_used':len({p.transform for p in self.placements}),
            'target_objects':self.target_objects,
            'background':{'method':'stationary coordinate noise and smooth modes; no tile grid',
                'level':self.background_level,'noise_sigma':self.background_noise,
                'variation_sigma':self.background_variation},
            'validation_checks':['intact connected donor masks','exact real donor pixels',
                'no overlapping cores or cropped outer-border masks',
                'uniform whole-canvas positions; no boundary-specific placement']})
        return report

    def manifest(self):
        result = super().manifest()
        result.update({'layout':'random-field','transforms_scope':'comparison preview only',
            'background_level':self.background_level,'background_noise':self.background_noise,
            'background_variation':self.background_variation,'background_modes':self.background_modes})
        return result


def segment_source(image, *, device='cpu', models_root=None):
    import torch
    from cisegmentation.adapters import segment_czyx
    from cisegmentation.registry import get_model_spec
    from cisegmentation.settings import SegmentationSettings
    torch.set_num_threads(4)
    os.environ['CISEGMENTATION_MODELS'] = str(models_root or Path(__file__).resolve().parents[1]/'bundled_models')
    model = 'stardist:SD_Nuclei_Versatile'
    # Detection-only scales avoid resampling the source. The output remains
    # uncalibrated unless --pixel-size-um is explicitly provided by the user.
    labels,info = segment_czyx(image[None,None],get_model_spec(model),
        SegmentationSettings(model=model,target='nuclei',device=device),
        {'x':0.5,'y':0.5,'z':1},bounded=True)
    return labels[0],info


def grid_visibility(store, source_size, *, band_pixels=48):
    """Measure foreground density at former joins against the field interior.

    Use a reference-mask pyramid with at most 2048 pixels per side, excluding
    the outer canvas edge. This is a density diagnostic, not an inference or
    biological segmentation accuracy score.
    """
    import zarr
    root = zarr.open_group(str(store),mode='r')
    group = root['labels/labels_nuclei_reference']
    levels = group.attrs['multiscales'][0]['datasets']
    level = next((d['path'] for d in levels if max(group[d['path']].shape[-2:]) <= 2048),levels[-1]['path'])
    scale = 2**int(level)
    foreground = group[level][0,0,0] > 0
    height,width = root['0'].shape[-2:]
    y = np.arange(foreground.shape[0])*scale
    x = np.arange(foreground.shape[1])*scale
    seams_y = np.arange(source_size,height,source_size)
    seams_x = np.arange(source_size,width,source_size)
    dy = np.min(np.abs(y[:,None]-seams_y),axis=1) if len(seams_y) else np.full(len(y),np.inf)
    dx = np.min(np.abs(x[:,None]-seams_x),axis=1) if len(seams_x) else np.full(len(x),np.inf)
    eligible = ((y[:,None] >= band_pixels*2) & (y[:,None] < height-band_pixels*2) &
                (x[None,:] >= band_pixels*2) & (x[None,:] < width-band_pixels*2))
    near = eligible & ((dy[:,None] < band_pixels) | (dx[None,:] < band_pixels))
    interior = eligible & (dy[:,None] >= band_pixels*2) & (dx[None,:] >= band_pixels*2)
    near_fraction = float(foreground[near].mean()) if np.any(near) else None
    interior_fraction = float(foreground[interior].mean()) if np.any(interior) else None
    result = {'band_pixels':band_pixels,'pyramid_level':int(level),
        'foreground_near_former_joins':near_fraction,'foreground_interior':interior_fraction,
        'join_to_interior_ratio':near_fraction/interior_fraction
            if near_fraction is not None and interior_fraction else None}
    root.store.close()
    return result


def diagnostics(plan, output, folder, report, *, export_tiff=False):
    """Scientific review images use the same contrast for before/after views."""
    from PIL import Image, ImageDraw
    import zarr
    folder.mkdir(parents=True,exist_ok=True)
    tifffile.imwrite(folder/'source-nuclei-mask.tif',plan.labels,compression='deflate')
    tifffile.imwrite(folder/'border-cleaned-source.tif',plan.cleaned,compression='deflate')
    tifffile.imwrite(folder/'border-cleaned-source-mask.tif',plan.kept,compression='deflate')
    (folder/'placement-plan.json').write_text(json.dumps(plan.manifest(),indent=2),encoding='utf-8')
    (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lo,hi = np.percentile(plan.image,[1,99.5])
    def display(a):
        return Image.fromarray(np.clip((a.astype(np.float32)-lo)/max(1,hi-lo)*255,0,255).astype('uint8')).convert('RGB')
    sheet = Image.new('RGB',(2008,1050),'white')
    sheet.paste(display(plan.image).resize((1004,1004)),(0,35))
    sheet.paste(display(plan.cleaned).resize((1004,1004)),(1004,35))
    draw = ImageDraw.Draw(sheet)
    draw.text((10,10),'Original real nuclei image',fill='black')
    draw.text((1014,10),'Cleaned source: whole edge nuclei removed',fill='black')
    sheet.save(folder/'source-before-after.png')
    variants = Image.new('RGB',(4*502,2*537),'white')
    for i in range(8):
        im = display(transformed_tile(plan.cleaned,i%4,0,i//4)).resize((502,502))
        x,y = (i%4)*502,(i//4)*537
        variants.paste(im,(x,y+30))
        ImageDraw.Draw(variants).text((x+8,y+8),f'{i%4*90} deg; flip X={bool(i//4)}',fill='black')
    variants.save(folder/'eight-cleaned-orientations.png')
    root = zarr.open_group(str(output),mode='r')
    levels = root.attrs['multiscales'][0]['datasets']
    level = next((d['path'] for d in levels if max(root[d['path']].shape[-2:]) <= 1600),levels[-1]['path'])
    overview = display(root[level][0,0,0])
    overview.save(folder/'tilescan-overview.png')
    # Inspect horizontal, vertical and four-way joins at full source resolution.
    points = []
    for kind in ('horizontal','vertical','junction'):
        if isinstance(plan,RandomNucleiPlan):
            def crosses(p,axis):
                coords = np.nonzero(p.patch.mask)[axis]
                origin = p.y if axis == 0 else p.x
                return (origin+coords.min())//plan.size != (origin+coords.max())//plan.size
            candidates = [p for p in plan.placements if
                (crosses(p,0) if kind == 'horizontal' else crosses(p,1) if kind == 'vertical'
                 else crosses(p,0) and crosses(p,1))]
        else:
            candidates = [p for p in plan.placements if p.kind == kind]
        target_y = plan.height/2 if kind != 'vertical' else plan.size*(plan.rows//2-0.5)
        target_x = plan.width/2 if kind != 'horizontal' else plan.size*(plan.columns//2-0.5)
        if candidates:
            p = min(candidates,key=lambda p:(p.y-target_y)**2+(p.x-target_x)**2)
            h,w = p.patch.mask.shape
            points.append((kind,p.y+h//2,p.x+w//2))
        elif isinstance(plan,RandomNucleiPlan):
            points.append((kind,int(target_y),int(target_x)))
    contact = Image.new('RGB',(3*384,max(1,len(points))*425),'white')
    for row,(kind,cy,cx) in enumerate(points):
        h,w = min(384,plan.height),min(384,plan.width)
        y,x = max(0,min(cy-h//2,plan.height-h)),max(0,min(cx-w//2,plan.width-w))
        original = render_region(plan.image,plan.transforms,y,x,h,w)
        corrected = root['0'][0,0,0,y:y+h,x:x+w]
        expected = plan.render(None,None,y,x,h,w)
        np.testing.assert_array_equal(corrected,expected)
        masks = root['labels/labels_nuclei_reference/0'][0,0,0,y:y+h,x:x+w]
        np.testing.assert_array_equal(masks,plan.render_labels(y,x,h,w))
        overlay = np.array(display(corrected))
        boundaries = (masks > 0) & (masks != ndi.minimum_filter(masks,size=3))
        overlay[boundaries] = [255,100,70]
        panels = [display(original),display(corrected),Image.fromarray(overlay)]
        for col,(title,panel) in enumerate(zip(['Uncorrected','Shared real nuclei','Reference-mask outlines'],panels)):
            contact.paste(panel,(col*384,row*425+35))
            ImageDraw.Draw(contact).text((col*384+8,row*425+8),f'{kind}: {title}',fill='black')
    contact.save(folder/'seam-before-after.png')
    if export_tiff:
        export_tiff_strips(root['0'],folder/'tilescan.tif')
    root.store.close()


def export_tiff_strips(array, path):
    """Supply precompressed TIFF strips; ndarray iterators represent pages."""
    import zlib
    height,width = array.shape[-2:]
    dtype = array.dtype.newbyteorder('<')
    def strips():
        for y in range(0,height,128):
            pixels = np.asarray(array[0,0,0,y:y+128,:],dtype=dtype)
            yield zlib.compress(pixels.tobytes(order='C'),level=6)
    path = Path(path)
    temporary = path.with_name(path.name+'.partial')
    if temporary.exists():
        raise FileExistsError(f'TIFF partial output already exists: {temporary}')
    tifffile.imwrite(temporary,data=strips(),shape=(height,width),dtype=dtype,
        byteorder='<',rowsperstrip=128,compression='deflate',
        bigtiff=height*width*dtype.itemsize >= 2**32,metadata={'axes':'YX'})
    temporary.replace(path)


def build(source, output, *, rows=4, columns=4, height=None, width=None,
          seed=20261005, clearance=48, halo=6, gap=1, device='cpu',
          models_root=None, mask=None, diagnostics_dir=None,
          pixel_size_um=None, chunk_size=512, export_tiff=False,layout='random-field'):
    source,output = Path(source),Path(output)
    folder = Path(diagnostics_dir) if diagnostics_dir else output.with_name(output.name.removesuffix('.ome.zarr')+'_diagnostics')
    if output.exists() or output.with_name(output.name+'.partial').exists() or folder.exists():
        raise FileExistsError('Output, partial output or diagnostics directory already exists; choose a new name')
    with tifffile.TiffFile(source) as tif:
        if tif.series[0].axes != 'YX':
            raise ValueError('Expected a single-channel YX TIFF')
        image = tif.series[0].asarray()
    if (height is None) != (width is None):
        raise ValueError('Provide both --height and --width')
    height = height if height is not None else rows*image.shape[0]
    width = width if width is not None else columns*image.shape[1]
    if mask:
        mask = Path(mask)
        labels = np.load(mask,allow_pickle=False) if mask.suffix == '.npy' else tifffile.imread(mask)
        segmentation = {'supplied_mask':mask.name,'sha256':hashlib.sha256(mask.read_bytes()).hexdigest()}
    else:
        labels,segmentation = segment_source(image,device=device,models_root=models_root)
    if layout not in ('random-field','tile-repair'):
        raise ValueError('Layout must be random-field or tile-repair')
    cls = RandomNucleiPlan if layout == 'random-field' else NucleiPlan
    plan = cls(image,labels,height,width,seed=seed,clearance=clearance,halo=halo,gap=gap)
    report = {**plan.validate(),'source':str(source),'output':str(output),
        'shape_yx':[height,width],'tile_grid_yx':[plan.rows,plan.columns],
        'cleaning':plan.cleaning,'source_segmentation':segmentation,
        'layout':layout,'method':('uniform whole-canvas placement of complete real nuclei'
            if layout == 'random-field' else 'whole border-object removal; globally shared complete real donor patches'),
        'reference_warning':'Model-derived synthetic reference masks, not manually annotated ground truth'}
    create_tilescan(source,output,height=height,width=width,seed=seed,
        chunk_size=chunk_size,pixel_size_um=pixel_size_um,transforms=plan.transforms,
        region_renderer=plan.render,channel_label='Nuclei',extra_metadata=report,
        label_renderer=plan.render_labels,label_max_id=report['max_reference_id'])
    report['grid_visibility'] = grid_visibility(output,plan.size)
    import zarr
    root = zarr.open_group(str(output),mode='a')
    metadata = dict(root.attrs['synthetic_tilescan'])
    metadata['seam_repair'] = report
    root.attrs['synthetic_tilescan'] = metadata
    root.store.close()
    diagnostics(plan,output,folder,report,export_tiff=export_tiff)
    print(json.dumps({'output':str(output),'diagnostics':str(folder),
                      'reference_objects':report['reference_objects'],
                      'validation':report['validation']}),flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--rows',type=int,default=4)
    p.add_argument('--columns',type=int,default=4)
    p.add_argument('--height',type=int)
    p.add_argument('--width',type=int)
    p.add_argument('--seed',type=int,default=20261005)
    p.add_argument('--clearance',type=int,default=48)
    p.add_argument('--halo',type=int,default=6)
    p.add_argument('--gap',type=int,default=1)
    p.add_argument('--layout',choices=['random-field','tile-repair'],default='random-field',
                   help='Whole-field random placement (default), or the original tile-edge repair')
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu')
    p.add_argument('--models-root',type=Path)
    p.add_argument('--mask',type=Path,help='Existing 2D instance labels (.npy or TIFF), background 0')
    p.add_argument('--diagnostics-dir',type=Path)
    p.add_argument('--pixel-size-um',type=float)
    p.add_argument('--chunk-size',type=int,default=512)
    p.add_argument('--export-tiff',action='store_true',help='Also export a full-resolution TIFF using bounded-memory strips')
    args = vars(p.parse_args())
    build(**args)


if __name__ == '__main__':
    main()
