"""Bounded region I/O and adaptive, overlapping instance-label inference.

The region/core/halo design follows CIDeconvolve's streaming implementation.
Instance labels additionally require overlap matching and global ID remapping.
"""

from __future__ import annotations

import math
import time
from collections import Counter, OrderedDict, deque
from itertools import product
from pathlib import Path

import numpy as np

from .resources import MIB, snapshot


class ArrayView:
    """A basic-indexed disk array that never reads data merely to make a view."""

    streamed = True

    def __init__(self, array, storage_axes, axes=None, selections=None, cache=None):
        self.array = array
        self.cache = cache
        self.storage_axes = tuple(storage_axes)
        self.axes = tuple(storage_axes if axes is None else axes)
        self.selections = dict(
            selections
            or {
                axis: range(array.shape[self.storage_axes.index(axis)])
                if axis in self.storage_axes
                else range(1)
                for axis in self.axes
            }
        )
        self.shape = tuple(len(self.selections[axis]) for axis in self.axes)
        self.ndim = len(self.shape)
        self.dtype = np.dtype(array.dtype).newbyteorder("=")
        self.size = math.prod(self.shape)
        self.nbytes = self.size * self.dtype.itemsize

    def __getitem__(self, keys):
        keys = keys if isinstance(keys, tuple) else (keys,)
        if any(
            not isinstance(key, (int, np.integer, slice)) and key is not Ellipsis
            for key in keys
        ):
            return np.asarray(self)[keys]
        if any(key is Ellipsis for key in keys):
            position = next(i for i, key in enumerate(keys) if key is Ellipsis)
            keys = (
                keys[:position]
                + (slice(None),) * (self.ndim - len(keys) + 1)
                + keys[position + 1 :]
            )
        keys += (slice(None),) * (self.ndim - len(keys))
        if len(keys) != self.ndim:
            raise IndexError("Too many indexes for streamed array")
        selections = dict(self.selections)
        axes = []
        for axis, key in zip(self.axes, keys):
            selections[axis] = selections[axis][key]
            if isinstance(key, slice):
                axes.append(axis)
        if not axes:
            return (
                ArrayView(self.array, self.storage_axes, (), selections, self.cache)
                ._read()
                .item()
            )
        return ArrayView(self.array, self.storage_axes, axes, selections, self.cache)

    def _read(self):
        keys = []
        remaining = []
        for axis in self.storage_axes:
            selection = self.selections[axis]
            if isinstance(selection, range):
                keys.append(slice(selection.start, selection.stop, selection.step))
                remaining.append(axis)
            else:
                keys.append(selection)
        result = (
            self.cache.read(self.array, keys)
            if self.cache
            else np.asarray(self.array[tuple(keys)])
        )
        for axis in self.axes:
            if axis not in remaining:
                result = np.expand_dims(result, 0)
                remaining.insert(0, axis)
        if self.axes:
            result = result.transpose([remaining.index(axis) for axis in self.axes])
        return result.astype(self.dtype, copy=False)

    def __array__(self, dtype=None, copy=None):
        # Catch accidental whole-image copies in downstream processing.
        if self.nbytes > 8 * MIB:
            budget = min(256 * MIB, max(8 * MIB, snapshot().ram_available // 4))
            if self.nbytes > budget:
                raise MemoryError(
                    f"Streamed array read needs {self.nbytes / MIB:.1f} MiB; use bounded regions (budget {budget / MIB:.1f} MiB)"
                )
        result = self._read()
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        return result.copy() if copy else result


class ChunkCache:
    """Bounded decoded chunks for repeated object crops in measurement export.

    Use only with read-only datasets; writing invalidates decoded chunks.
    """

    def __init__(self, limit=64 * MIB):
        self.limit, self.bytes = limit, 0
        self.chunks = OrderedDict()

    def read(self, array, keys):
        if (
            not hasattr(array, "chunks")
            or math.prod(array.chunks) * array.dtype.itemsize > self.limit
        ):
            return np.asarray(array[tuple(keys)])
        selections = [
            range(*key.indices(size)) if isinstance(key, slice) else range(key, key + 1)
            for key, size in zip(keys, array.shape)
        ]
        if any(selection.step <= 0 for selection in selections):
            return np.asarray(array[tuple(keys)])
        result = np.empty(tuple(map(len, selections)), dtype=array.dtype)
        indexes = [
            sorted({value // size for value in selection})
            for selection, size in zip(selections, array.chunks)
        ]
        for index in product(*indexes):
            chunk_key = (id(array), index)
            if chunk_key in self.chunks:
                chunk = self.chunks.pop(chunk_key)
            else:
                slices = tuple(
                    slice(i * size, min(length, (i + 1) * size))
                    for i, size, length in zip(index, array.chunks, array.shape)
                )
                chunk = np.asarray(array[slices])
                while self.chunks and self.bytes + chunk.nbytes > self.limit:
                    _, previous = self.chunks.popitem(last=False)
                    self.bytes -= previous.nbytes
                self.bytes += chunk.nbytes
            self.chunks[chunk_key] = chunk
            source, destination = [], []
            for selection, i, size in zip(selections, index, array.chunks):
                lower = max(0, math.ceil((i * size - selection.start) / selection.step))
                upper = min(
                    len(selection),
                    math.ceil(((i + 1) * size - selection.start) / selection.step),
                )
                first = selection.start + lower * selection.step - i * size
                source.append(
                    slice(
                        first, first + (upper - lower) * selection.step, selection.step
                    )
                )
                destination.append(slice(lower, upper))
            result[tuple(destination)] = chunk[tuple(source)]
        axes = tuple(i for i, key in enumerate(keys) if not isinstance(key, slice))
        return np.squeeze(result, axis=axes)


class ChannelStack:
    """Concatenate one-channel disk views without reading their pixels."""

    streamed = True

    def __init__(self, arrays, channel_axis=1):
        self.arrays = arrays
        self.channel_axis = channel_axis
        shape = list(arrays[0].shape)
        shape[channel_axis] = len(arrays)
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = arrays[0].dtype
        self.size = math.prod(shape)
        self.nbytes = self.size * self.dtype.itemsize

    def __getitem__(self, keys):
        keys = keys if isinstance(keys, tuple) else (keys,)
        keys += (slice(None),) * (self.ndim - len(keys))
        channel = keys[self.channel_axis]
        child_keys = list(keys)
        if isinstance(channel, (int, np.integer)):
            child_keys[self.channel_axis] = 0
            return self.arrays[channel][tuple(child_keys)]
        child_keys[self.channel_axis] = slice(0, 1)
        arrays = [array[tuple(child_keys)] for array in self.arrays[channel]]
        axis = self.channel_axis - sum(
            isinstance(key, (int, np.integer)) for key in keys[: self.channel_axis]
        )
        return ChannelStack(arrays, axis)

    def __array__(self, dtype=None, copy=None):
        if self.nbytes > min(256 * MIB, max(8 * MIB, snapshot().ram_available // 4)):
            raise MemoryError("Read streamed label channels in bounded regions")
        return np.concatenate(
            [np.asarray(array, dtype=dtype) for array in self.arrays],
            axis=self.channel_axis,
        )


def blocks(shape, size=512):
    """Use one plane at a time; bounding memory is independent of Z depth."""
    chunks = [1] * (len(shape) - 2) + [size, size]
    for starts in product(
        *(range(0, length, chunk) for length, chunk in zip(shape, chunks))
    ):
        yield tuple(
            slice(start, min(length, start + chunk))
            for start, length, chunk in zip(starts, shape, chunks)
        )


def should_stream(resource, settings=None):
    import zarr

    root = zarr.open_group(str(resource.store_path), mode="r")
    group = root[resource.image_path] if resource.image_path else root
    multiscale = group.attrs["multiscales"][0]
    array = group[multiscale["datasets"][0]["path"]]
    size = math.prod(array.shape) * np.dtype(array.dtype).itemsize
    return (
        max(array.shape[-2:]) > 4096
        or size > min(256 * MIB, snapshot().ram_available // 12)
        or bool(settings and settings.streaming_mode == "on")
    )


def label_counts(array):
    counts = Counter()
    for key in blocks(array.shape):
        values, sizes = np.unique(np.asarray(array[key]), return_counts=True)
        counts.update(
            {int(value): int(size) for value, size in zip(values, sizes) if value}
        )
    return counts


def overlap_counts(first, second):
    counts = Counter()
    for key in blocks(first.shape):
        a, b = np.asarray(first[key]), np.asarray(second[key])
        foreground = (a != 0) & (b != 0)
        pairs, sizes = np.unique(
            np.column_stack((a[foreground], b[foreground])), axis=0, return_counts=True
        )
        counts.update(
            {(int(a), int(b)): int(size) for (a, b), size in zip(pairs, sizes)}
        )
    return counts


class RegionProxy:
    def __init__(self, region, value, origin):
        self.region, self.label, self.origin = region, value, np.asarray(origin)

    def __getattr__(self, name):
        return getattr(self.region, name)

    @property
    def coords(self):
        return self.region.coords + self.origin

    @property
    def centroid(self):
        return tuple(np.asarray(self.region.centroid) + self.origin)

    @property
    def bbox(self):
        return tuple(np.asarray(self.region.bbox) + np.tile(self.origin, 2))

    @property
    def slice(self):
        return tuple(
            slice(part.start + start, part.stop + start)
            for part, start in zip(self.region.slice, self.origin)
        )


def iter_regions(labels):
    """Find global object bounds in chunks, then measure bounded object crops."""
    from scipy.ndimage import find_objects
    from skimage.measure import regionprops

    boxes = {}
    for key in blocks(labels.shape):
        block = np.asarray(labels[key])
        values = np.unique(block)
        values = values[values != 0]
        mapping = {int(value): index + 1 for index, value in enumerate(values)}
        dense = remap(block, mapping)
        for value, local in zip(values, find_objects(dense)):
            if local is None:
                continue
            lower = [part.start + base.start for part, base in zip(local, key)]
            upper = [part.stop + base.start for part, base in zip(local, key)]
            value = int(value)
            if value in boxes:
                previous_lower, previous_upper = boxes[value]
                lower = [min(a, b) for a, b in zip(lower, previous_lower)]
                upper = [max(a, b) for a, b in zip(upper, previous_upper)]
            boxes[value] = (lower, upper)
    for value, (lower, upper) in sorted(boxes.items()):
        key = tuple(slice(start, end) for start, end in zip(lower, upper))
        mask = (np.asarray(labels[key]) == value).astype(np.uint8)
        for region in regionprops(mask):
            yield RegionProxy(region, value, lower)


class LabelRemap:
    """Prepare global ID mappings once, rather than sorting them per chunk."""

    def __init__(self, mapping):
        if any(
            value < 0 or value > np.iinfo(np.uint32).max for value in mapping.values()
        ):
            raise OverflowError("Final instance IDs exceed uint32")
        self.keys = np.asarray(sorted(mapping), dtype=np.uint32)
        self.targets = np.asarray(
            [mapping[int(key)] for key in self.keys], dtype=np.uint32
        )
        self.lookup = None
        if len(self.keys):
            size = int(self.keys[-1]) + 1
            if size <= max(4096, len(self.keys) * 8) and size * 4 <= 64 * MIB:
                self.lookup = np.zeros(size, dtype=np.uint32)
                self.lookup[self.keys] = self.targets

    def __call__(self, values):
        if not len(self.keys):
            return np.zeros_like(values, dtype=np.uint32)
        if self.lookup is not None:
            valid = values < len(self.lookup)
            return np.where(
                valid, self.lookup[np.minimum(values, len(self.lookup) - 1)], 0
            )
        indices = np.searchsorted(self.keys, values)
        valid = indices < len(self.keys)
        clipped = np.minimum(indices, len(self.keys) - 1)
        valid &= self.keys[clipped] == values
        return np.where(valid, self.targets[clipped], 0).astype(np.uint32)


def remap(values, mapping):
    return (mapping if isinstance(mapping, LabelRemap) else LabelRemap(mapping))(values)


def _root(parents, value):
    while parents[value] != value:
        parents[value] = parents[parents[value]]
        value = parents[value]
    return value


def _stitch(predicted, previous, covered, core, parents, threshold, mapping_out=None):
    """Match mutually strongest instances in the already-completed overlap."""
    labels = np.unique(predicted[core])
    labels = labels[labels != 0]
    local_sizes = dict(zip(*np.unique(predicted[covered], return_counts=True)))
    old_sizes = dict(zip(*np.unique(previous[covered], return_counts=True)))
    overlap = covered & (predicted > 0) & (previous > 0)
    pairs, sizes = np.unique(
        np.column_stack((predicted[overlap], previous[overlap])),
        axis=0,
        return_counts=True,
    )
    best_local, best_old = {}, {}
    for (local, old), size in zip(pairs, sizes):
        local, old, size = int(local), int(old), int(size)
        if size > best_local.get(local, (0, 0))[1]:
            best_local[local] = (old, size)
        if size > best_old.get(old, (0, 0))[1]:
            best_old[old] = (local, size)
    mapping = {}
    for label in labels:
        label = int(label)
        match = best_local.get(label)
        if (
            match
            and best_old[match[0]][0] == label
            and match[1]
            / max(1, min(local_sizes.get(label, 0), old_sizes.get(match[0], 0)))
            >= threshold
        ):
            mapping[label] = _root(parents, match[0])
        else:
            if len(parents) >= np.iinfo(np.uint32).max:
                raise OverflowError("Instance IDs exceed uint32 storage")
            mapping[label] = len(parents)
            parents.append(len(parents))
    if mapping_out is not None:
        mapping_out.update(mapping)
    return remap(predicted[core], mapping)


def _split(core):
    lengths = [end - start for start, end in core]
    eligible = [
        i for i, length in enumerate(lengths) if length >= (2 if i == 0 else 128)
    ]
    if not eligible:
        return []
    axis = max(eligible, key=lambda i: lengths[i])
    start, end = core[axis]
    middle = (start + end) // 2
    children = [list(core), list(core)]
    children[0][axis] = (start, middle)
    children[1][axis] = (middle, end)
    return [tuple(child) for child in children]


def _estimate_bytes(shape, spec, scales):
    # The adapters can downsample, but never upsample the versatile input.
    # Keep the source-sized estimate conservative for preprocessing buffers.
    return int(
        math.prod(shape)
        * {
            "cellpose-sam": 768,
            "cellpose3": 256,
            "stardist": 160,
            "instanseg": 384,
            "spotiflow": 192,
        }[spec.family]
    )


def _cleanup_memory():
    import gc
    import sys

    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def infer_streamed(
    image, spec, settings, path, segment, *, allow_parallel=True, log=None
):
    """Write TZYX raw instances using adaptive tiles for every model adapter."""
    import zarr

    t_size, _, z_size, y_size, x_size = image.data.shape
    Path(str(path) + ".points.sqlite").unlink(missing_ok=True)
    root = zarr.open_group(str(path), mode="w", zarr_version=2)
    shape = (t_size, z_size, y_size, x_size)
    labels = root.create_dataset(
        "labels",
        shape=shape,
        dtype="u4",
        chunks=(1, 1, 512, 512),
        dimension_separator="/",
    )
    coverage = root.create_dataset(
        "coverage",
        shape=shape,
        dtype="bool",
        chunks=(1, 1, 512, 512),
        dimension_separator="/",
    )
    native_3d = spec.dimensions == "3d" and settings.dimension_mode != "slice-2d"
    # The bundled 3D Spotiflow checkpoints use 32-voxel network blocks and
    # two overlap blocks in native prediction. Retain that context even when
    # resource pressure reduces the core depth; thin context-free slabs can
    # silently lose every spot. The resource estimate includes these halos.
    z_halo = settings.tile_overlap_z
    xy_halo = settings.tile_overlap
    if native_3d and spec.family == "spotiflow":
        z_halo = max(z_halo, 64)
        # Wider XY context also avoids the lost/displaced boundary detections
        # observed in the positive 3D checkpoint test at 96-pixel overlap.
        xy_halo = max(xy_halo, 192)
    halos = (
        z_halo if native_3d else 0,
        xy_halo,
        xy_halo,
    )
    core_depth = min(z_size, settings.tile_depth if native_3d else 1)
    infos = []
    read_seconds = 0.0
    for t in range(t_size):
        started = time.perf_counter()
        parents = [0]
        queue = deque(
            tuple(
                (start, min(size, start + step))
                for start, size, step in zip(
                    starts,
                    (z_size, y_size, x_size),
                    (core_depth, settings.tile_size, settings.tile_size),
                )
            )
            for starts in product(
                range(0, z_size, core_depth),
                range(0, y_size, settings.tile_size),
                range(0, x_size, settings.tile_size),
            )
        )
        aggregate = None
        processed = 0
        smallest = settings.tile_size
        last_log = 0.0
        initial = snapshot().to_dict()
        stats = {
            "oom_retries": 0,
            "size_limit_retries": 0,
            "pressure_splits": 0,
            "workers": 1,
            "initial_workers": 1,
            "peak_workers": 1,
            "concurrency_reductions": 0,
            "worker_restarts": 0,
            "memory_waits": 0,
            "spool_seconds": 0.0,
            "peak_worker_rss_mb": 0.0,
            "peak_worker_cuda_mb": 0.0,
            "peak_worker_cuda_allocated_mb": 0.0,
            "peak_worker_cuda_reserved_mb": 0.0,
            "minimum_ram_available": initial["ram_available"],
            "minimum_gpu_available": initial["gpu_available"],
        }
        from contextlib import closing

        from .streaming_tiles import iter_tile_predictions

        with closing(
            iter_tile_predictions(
                image,
                t,
                queue,
                halos,
                spec,
                settings,
                path,
                segment,
                stats,
                allow_parallel=allow_parallel,
                log=log,
            )
        ) as predictions:
            for (
                core,
                extended,
                predicted,
                info,
                tile_read_seconds,
                resources,
            ) in predictions:
                read_seconds += tile_read_seconds
                old = np.asarray(labels[(t, *extended)])
                covered = np.asarray(coverage[(t, *extended)])
                relative = tuple(
                    slice(start - ext.start, end - ext.start)
                    for (start, end), ext in zip(core, extended)
                )
                stitched = _stitch(
                    predicted,
                    old,
                    covered,
                    relative,
                    parents,
                    settings.tile_match_threshold,
                    point_mapping := {},
                )
                points = info.pop("_native_points", None)
                if points is not None:
                    from .point_localizations import append_raw
                    append_raw(str(path) + ".points.sqlite", t, points,
                               point_mapping, tuple(s.start for s in extended))
                destination = (t, *(slice(start, end) for start, end in core))
                labels[destination] = stitched
                coverage[destination] = True
                processed += 1
                smallest = min(
                    smallest, core[1][1] - core[1][0], core[2][1] - core[2][0]
                )
                if aggregate is None:
                    aggregate = dict(info)
                    aggregate["timings"] = {}
                    aggregate["model_cache_hits"] = aggregate["model_cache_misses"] = 0
                for name, value in info.get("timings", {}).items():
                    aggregate["timings"][name] = aggregate["timings"].get(
                        name, 0.0
                    ) + float(value)
                for name in ("model_cache_hits", "model_cache_misses"):
                    aggregate[name] += int(info.get(name, 0))
                now = time.perf_counter()
                if log and now - last_log >= 15:
                    log(
                        f"Streaming {spec.id}: {processed} tiles complete, {stats.get('pending_tiles', len(queue) - processed)} pending; RAM headroom {resources.ram_available / MIB:.0f} MiB, GPU {resources.gpu_available / MIB:.0f} MiB"
                    )
                    last_log = now
                del predicted, old, covered, stitched
        # A new ID is allocated only for an instance present in a written core.
        # IDs are already dense, so another full-image read/write is unnecessary.
        aggregate = aggregate or {
            "device": settings.device,
            "timings": {},
            "effective_parameters": {},
        }
        aggregate.update(
            {
                "object_count": len(parents) - 1,
                "runtime_seconds": time.perf_counter() - started,
                "streaming": {
                    "tiles": processed,
                    **stats,
                    "smallest_core_xy": smallest,
                    "overlap_zyx": list(halos),
                    "initial_resources": initial,
                },
            }
        )
        infos.append(aggregate)
    del root["coverage"]
    root.attrs["complete"] = True
    return infos, read_seconds


def raw_array(path):
    if Path(path).is_dir():
        import zarr

        return zarr.open_group(str(path), mode="r")["labels"]
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _border_labels(array):
    labels = set()
    for z in range(array.shape[0]):
        for key in (
            (z, 0, slice(None)),
            (z, -1, slice(None)),
            (z, slice(None), 0),
            (z, slice(None), -1),
        ):
            labels.update(
                int(value) for value in np.unique(np.asarray(array[key])) if value
            )
    return labels


def _expand_streamed(nuclei, destination, distance, scales):
    from scipy.ndimage import distance_transform_edt

    y_scale = float(scales.get("y") or scales.get("x") or 1.0)
    x_scale = float(scales.get("x") or scales.get("y") or 1.0)
    halo_y, halo_x = (
        math.ceil(distance / y_scale) + 1,
        math.ceil(distance / x_scale) + 1,
    )
    for key in blocks(nuclei.shape):
        z, y, x = key
        extended_y = slice(
            max(0, y.start - halo_y), min(nuclei.shape[-2], y.stop + halo_y)
        )
        extended_x = slice(
            max(0, x.start - halo_x), min(nuclei.shape[-1], x.stop + halo_x)
        )
        plane = np.asarray(nuclei[(z, extended_y, extended_x)])[0]
        if np.any(plane):
            distances, nearest = distance_transform_edt(
                plane == 0, sampling=(y_scale, x_scale), return_indices=True
            )
            expanded = plane[nearest[0], nearest[1]]
            expanded[distances > distance] = 0
            value = expanded[
                y.start - extended_y.start : y.stop - extended_y.start,
                x.start - extended_x.start : x.stop - extended_x.start,
            ]
        else:
            value = np.zeros((y.stop - y.start, x.stop - x.start), dtype=np.uint32)
        destination[key] = value[None]


def finalize_streamed(payload):
    """Chunked global cell/nucleus matching, offsets, expansion and pyramids."""
    from copy import deepcopy

    import zarr

    from . import engine
    from .ome_zarr_io import LabelResult, read_image, write_native_label_groups
    from .parallel_pipeline import _raw_path, _reuse_info, _resource_key
    from .reporting import step_record
    from .settings import SegmentationSettings

    settings = SegmentationSettings(**payload["settings"])
    resource = payload["resource"]
    image = read_image(resource, lazy=True)
    stage = Path(payload["stage_dir"])
    passes, records = payload["passes"], payload["records"]
    consumers, raw = {}, {}
    for model_pass in passes:
        for request in model_pass.requests:
            raw[request.request_id] = ArrayView(
                raw_array(_raw_path(stage, model_pass, request, resource)), "tzyx"
            )
            for consumer in request.consumers:
                consumers.setdefault(consumer.kind, []).append((request, consumer))
    root = zarr.open_group(
        str(
            stage
            / (
                "final-"
                + __import__("hashlib")
                .sha256(resource.image_path.encode())
                .hexdigest()[:16]
                + ".zarr"
            )
        ),
        mode="w",
    )
    names = payload["generated_names"]
    output = root.create_dataset(
        "labels",
        shape=(image.data.shape[0], len(names), *image.data.shape[2:]),
        dtype="u4",
        chunks=(1, 1, 1, 512, 512),
        dimension_separator="/",
    )
    infos, steps, channel_labels = [], [], []
    next_id = 0
    used = {}

    def consume(kind, t, index=0):
        request, consumer = consumers[kind][index]
        info = deepcopy(records[request.request_id][t])
        reuse_key = (t, request.request_id)
        if reuse_key in used:
            info = _reuse_info(info, used[reuse_key])
        else:
            info["result_cache_hit"] = False
            used[reuse_key] = consumer.step
        array = raw[request.request_id][t]
        record = step_record(
            step=consumer.step,
            timepoint=t,
            model=request.model_id,
            target=request.target,
            primary_channel=request.primary_channel,
            nuclei_channel=request.nuclei_channel,
            labels=None,
            info=info,
            scales=image.scales,
            include_label_statistics=False,
        )
        if settings.labels_log_info:
            record["label_statistics"] = streamed_label_statistics(array, image.scales)
        steps.append(record)
        infos.append(info)
        return array

    for t in range(image.data.shape[0]):
        cells = nuclei = seeds = None
        if consumers.get("expansion"):
            seeds = consume("expansion", t)
            expanded = root.create_dataset(
                f"expanded-{t}", shape=seeds.shape, dtype="u4", chunks=(1, 512, 512)
            )
            _expand_streamed(
                seeds, expanded, settings.cell_expansion_distance, image.scales
            )
            cells = ArrayView(expanded, "zyx")
        elif consumers.get("cell"):
            cells = consume("cell", t)
        if consumers.get("nucleus"):
            nuclei = consume("nucleus", t)
        matching = nuclei if nuclei is not None else seeds
        channel = 0
        time_names = []
        if cells is not None and matching is not None:
            sizes = label_counts(matching)
            border = _border_labels(cells) if settings.remove_border_cells else set()
            pairs = overlap_counts(matching, cells)
            best = {}
            for (nucleus, cell), count in sorted(pairs.items()):
                if cell not in border and count > best.get(nucleus, (0, 0))[1]:
                    best[nucleus] = (cell, count)
            chosen = {}
            for nucleus, (cell, _) in sorted(best.items()):
                if cell not in chosen or sizes[nucleus] > sizes[chosen[cell]]:
                    chosen[cell] = nucleus
            cell_map, nucleus_map = {}, {}
            for cell in sorted(chosen):
                next_id += 1
                cell_map[cell] = next_id
                nucleus_map[chosen[cell]] = next_id
            cell_mapping, nucleus_mapping = (
                LabelRemap(cell_map),
                LabelRemap(nucleus_map),
            )
            for key in blocks(cells.shape):
                c, n = (
                    remap(np.asarray(cells[key]), cell_mapping),
                    remap(np.asarray(matching[key]), nucleus_mapping),
                )
                output[(t, 0, *key)] = c
                output[(t, 1, *key)] = n
                output[(t, 2, *key)] = np.where(n > 0, 0, c)
            time_names.extend(("labels_cells", "labels_nuclei", "labels_cytoplasm"))
            channel = 3
        elif cells is not None or nuclei is not None:
            array = cells if cells is not None else nuclei
            counts = label_counts(array)
            border = (
                _border_labels(array)
                if cells is not None and settings.remove_border_cells
                else set()
            )
            mapping = {
                value: value + next_id for value in counts if value not in border
            }
            prepared = LabelRemap(mapping)
            for key in blocks(array.shape):
                output[(t, 0, *key)] = remap(np.asarray(array[key]), prepared)
            next_id = max(mapping.values(), default=next_id)
            channel = 1
            time_names.append("labels_cells" if cells is not None else "labels_nuclei")
        for index, (request, consumer) in enumerate(consumers.get("foci", [])):
            array = consume("foci", t, index)
            mapping = {value: value + next_id for value in label_counts(array)}
            if settings.measurement_extensions_enabled():
                from .point_localizations import finalize_points
                model_pass = next(p for p in passes if any(r.request_id == request.request_id for r in p.requests))
                finalize_points(str(_raw_path(stage, model_pass, request, resource)) + ".points.sqlite",
                                stage / "final-points" / f"{_resource_key(resource)}.sqlite",
                                names[channel], t, mapping)
            prepared = LabelRemap(mapping)
            for key in blocks(array.shape):
                output[(t, channel, *key)] = remap(np.asarray(array[key]), prepared)
            next_id = max(mapping.values(), default=next_id)
            channel += 1
            time_names.append(
                f"labels_{consumer.label_type}_channel_{request.primary_channel}"
            )
        if next_id > np.iinfo(np.uint32).max:
            raise OverflowError("Final instance IDs exceed uint32")
        channel_labels = time_names
    timings = engine._aggregate_timings(
        [dict(info.get("timings", {})) for info in infos]
    )
    timings["zarr_read_seconds"] = float(
        records.get("__zarr_read_seconds__", [{"seconds": 0.0}])[0]["seconds"]
    )
    provenance = {
        "device": infos[-1].get("device") if infos else None,
        "runtime_seconds": sum(float(info.get("runtime_seconds", 0)) for info in infos),
        "segmentation_count": sum(
            not info.get("result_cache_hit", False) for info in infos
        ),
        "model_cache_hits": sum(
            int(info.get("model_cache_hits", 0))
            for info in infos
            if not info.get("result_cache_hit")
        ),
        "model_cache_misses": sum(
            int(info.get("model_cache_misses", 0))
            for info in infos
            if not info.get("result_cache_hit")
        ),
        "result_cache_hits": sum(bool(info.get("result_cache_hit")) for info in infos),
        "timings": timings,
        "step_runs": steps,
        "parameters": settings.to_dict(),
        "shared_instance_ids": ["cells", "nuclei", "cytoplasm"]
        if consumers.get("cell") or consumers.get("expansion")
        else [],
        "streaming": [info["streaming"] for info in infos if "streaming" in info],
    }
    if settings.labels_log_info:
        view = ArrayView(output, "tczyx")
        provenance["output_statistics"] = [
            {
                "timepoint": t,
                "channel": name,
                "locations_only": name.startswith("labels_spots_channel_")
                and not settings.spotiflow_local_refinement,
                "label_statistics": streamed_label_statistics(
                    view[t, index], image.scales
                ),
            }
            for t in range(image.data.shape[0])
            for index, name in enumerate(channel_labels)
        ]
    result = LabelResult(
        ArrayView(output, "tczyx"),
        image,
        "multi-step",
        "multi-step",
        provenance=provenance,
        channel_labels=channel_labels,
        label_origins=["generated"] * len(channel_labels),
    )
    write_started = time.perf_counter()
    write_native_label_groups(
        payload["overlay_path"],
        resource.image_path,
        result,
        names,
        payload["final_names"],
    )
    return {
        "ok": True,
        "resource_path": resource.image_path,
        "provenance": provenance,
        "channel_labels": channel_labels,
        "point_localizations": str(stage / "final-points" / f"{_resource_key(resource)}.sqlite"),
        "zarr_write_seconds": time.perf_counter() - write_started,
    }


def streamed_label_statistics(array, scales):
    counts = label_counts(array)
    sizes = np.asarray(list(counts.values()), dtype=np.float64)
    foreground = int(sizes.sum())
    total = math.prod(array.shape)
    result = {
        "label_count": len(counts),
        "foreground_elements": foreground,
        "foreground_percent": 100 * foreground / total if total else 0,
        "mean_size_elements": float(sizes.mean()) if len(sizes) else 0,
        "median_size_elements": float(np.median(sizes)) if len(sizes) else 0,
        "min_size_elements": int(sizes.min()) if len(sizes) else 0,
        "max_size_elements": int(sizes.max()) if len(sizes) else 0,
        "element_unit": "voxels" if array.shape[-3] > 1 else "pixels",
    }
    depth = array.shape[-3]
    axes = "zyx" if depth > 1 else "yx"
    if all(
        np.isfinite(scales.get(axis, float("nan"))) and scales.get(axis, 0) > 0
        for axis in axes
    ):
        factor = math.prod(scales[axis] for axis in axes)
        physical = sizes * factor
        median = float(np.median(physical)) if len(sizes) else 0
        result.update(
            {
                "physical_size_unit": "um^3" if depth > 1 else "um^2",
                "mean_physical_size": float(physical.mean()) if len(sizes) else 0,
                "median_physical_size": median,
                "min_physical_size": float(physical.min()) if len(sizes) else 0,
                "max_physical_size": float(physical.max()) if len(sizes) else 0,
                "median_equivalent_diameter_um": 2
                * (3 * median / (4 * math.pi)) ** (1 / 3)
                if depth > 1
                else 2 * math.sqrt(median / math.pi),
            }
        )
    return result


def write_streamed_label_groups(
    store_path, resource_path, result, generated_names, final_names
):
    import zarr

    from .ome_zarr_io import _from_tczyx, _source_axis_metadata, _source_scale_values

    root = zarr.open_group(str(store_path), mode="a")
    group = root[resource_path] if resource_path else root
    parent = group.require_group("labels")
    parent.attrs["labels"] = list(final_names)
    axes = result.source.axes
    for channel, (name, display) in enumerate(
        zip(generated_names, result.channel_labels or generated_names)
    ):
        label_group = parent.require_group(name)
        source = result.labels[:, channel : channel + 1]
        canonical_shape = source.shape
        shape = tuple(canonical_shape["tczyx".index(axis)] for axis in axes)
        datasets = []
        previous = None
        level = 0
        while True:
            array = label_group.create_dataset(
                str(level),
                shape=shape,
                dtype="u4",
                chunks=tuple(
                    min(512, length) if axis in "yx" else 1
                    for length, axis in zip(shape, axes)
                ),
                overwrite=True,
                dimension_separator="/",
            )
            chunk_sizes = [512 if axis in "yx" else 1 for axis in axes]
            for starts in product(
                *(range(0, length, size) for length, size in zip(shape, chunk_sizes))
            ):
                key = tuple(
                    slice(start, min(length, start + size))
                    for start, length, size in zip(starts, shape, chunk_sizes)
                )
                if previous is None:
                    native = dict(zip(axes, key))
                    canonical = tuple(native.get(axis, slice(0, 1)) for axis in "tczyx")
                    array[key] = _from_tczyx(np.asarray(source[canonical]), axes)
                else:
                    source_key = tuple(
                        slice(
                            part.start * 2, min(previous.shape[index], part.stop * 2), 2
                        )
                        if axis in "yx"
                        else part
                        for index, (axis, part) in enumerate(zip(axes, key))
                    )
                    array[key] = previous[source_key]
            array.attrs["_ARRAY_DIMENSIONS"] = list(axes)
            datasets.append(
                {
                    "path": str(level),
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": _source_scale_values(result.source, 2**level),
                        }
                    ],
                }
            )
            if min(shape[axes.index(axis)] for axis in "yx") < 512 or level >= 11:
                break
            previous = array
            shape = tuple(
                (length + 1) // 2 if axis in "yx" else length
                for axis, length in zip(axes, shape)
            )
            level += 1
        label_group.attrs["multiscales"] = [
            {
                "version": "0.4",
                "name": display,
                "axes": _source_axis_metadata(result.source),
                "datasets": datasets,
            }
        ]
        label_group.attrs["image-label"] = {
            "version": "0.4",
            "source": {"image": "../../"},
        }
    group.attrs["cisegmentation"] = {
        "model": result.model_id,
        "target": result.target,
        "source": result.source.resource.store_path.name,
        "label_storage_dtype": "uint32",
        "output_layout": "ome-zarr-0.4-labels",
        "label_groups": [f"labels/{name}" for name in generated_names],
        **result.provenance,
    }
