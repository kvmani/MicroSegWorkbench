"""Loading dataset versions for training and evaluation without holding more in memory than allowed.

* ``SplitSource`` reads one split on demand and caches decoded samples up to a fixed number of images.
* ``iter_epoch_batches`` yields deterministic augmented batches lazily. With every training image in memory
  (``images_in_memory = 0``) all patches of an epoch are shuffled together. With a limit, images are shuffled
  and processed in blocks of that many images, patches shuffled within each block, so each image is decoded once
  per epoch. The order depends only on (seed, epoch, limit) - never on the computer's RAM - and the limit is
  part of the recorded training configuration.
* ``prefetch`` prepares the next batches in a background thread while the model trains.
"""

from __future__ import annotations

import queue
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..core.augment import IGNORE_INDEX, AugmentConfig, augment
from ..core.classes import ClassMap
from ..core.dataset import DatasetVersion
from ..core.imageio import load_image, load_mask_values, to_unit_float
from ..core.preprocess import Preprocess, image_stats, normalize, resize_image, resize_labels


@dataclass(frozen=True)
class Sample:
    stem: str
    image: np.ndarray  # float32 (H, W, C) in [0, 1] (dtype-range scaled)
    labels: np.ndarray  # uint8 (H, W) model channel indices, IGNORE_INDEX for ignored
    stats: tuple[np.ndarray, np.ndarray] | None = None  # whole-image mean/std at model scale

    @property
    def nbytes(self) -> int:
        return int(self.image.nbytes + self.labels.nbytes)


def value_to_channel_lut(class_map: ClassMap) -> np.ndarray:
    """Lookup table: mask value -> model channel index (undeclared values -> IGNORE)."""
    return class_map.channel_lut()


def cropped(s: Sample, prep: Preprocess) -> Sample:
    """The analysed region of a sample (excluded border removed)."""
    return Sample(s.stem, prep.crop(s.image), prep.crop(s.labels), s.stats)


def to_training_sample(s: Sample, prep: Preprocess) -> Sample:
    """Crop, resize to model scale and record whole-image statistics for normalization."""
    s = cropped(s, prep)
    shape = prep.scaled_shape(*s.labels.shape)
    image = resize_image(s.image, shape)
    return Sample(s.stem, image, resize_labels(s.labels, shape), image_stats(image))


def to_training_samples(samples: list[Sample], prep: Preprocess) -> list[Sample]:
    return [to_training_sample(s, prep) for s in samples]


class SplitSource:
    """Samples of one split, decoded on demand. ``prep`` given -> training scale with statistics.

    ``max_cached`` = number of decoded samples kept (None = all, 0 = none).
    """

    def __init__(self, version: DatasetVersion, split: str, prep: Preprocess | None = None,
                 max_cached: int | None = None) -> None:
        self.version = version
        self.records = sorted(version.samples(split), key=lambda s: s["stem"])
        self.lut = value_to_channel_lut(version.class_map)
        self.prep = prep
        self.max_cached = max_cached
        self._cache: OrderedDict[int, Sample] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.records)

    @property
    def stems(self) -> list[str]:
        return [r["stem"] for r in self.records]

    def channels(self) -> set[int]:
        return {int(r["channels"]) for r in self.records}

    def estimated_bytes(self) -> int:
        """Memory for all samples at model scale (float32 image + uint8 labels), from the manifest only."""
        f = self.prep.resize_factor if self.prep else 1.0
        return int(sum(r["height"] * r["width"] * f * f * (4 * r["channels"] + 1) for r in self.records))


    def _load(self, i: int) -> Sample:
        r = self.records[i]
        img = to_unit_float(load_image(self.version.image_path(r)).array)
        labels = self.lut[load_mask_values(self.version.mask_path(r))]
        if (labels == IGNORE_INDEX).any():
            raise ValueError(f"Stored mask {r['mask_file']} has values outside the class map; dataset is corrupt.")
        sample = Sample(stem=r["stem"], image=img, labels=labels)
        return to_training_sample(sample, self.prep) if self.prep else sample

    def get(self, i: int) -> Sample:
        with self._lock:
            if i in self._cache:
                self._cache.move_to_end(i)
                return self._cache[i]
        sample = self._load(i)
        if self.max_cached is None or self.max_cached > 0:
            with self._lock:
                self._cache[i] = sample
                while self.max_cached is not None and len(self._cache) > self.max_cached:
                    self._cache.popitem(last=False)
        return sample

    def __iter__(self) -> Iterator[Sample]:
        for i in range(len(self)):
            yield self.get(i)


def load_split(version: DatasetVersion, split: str) -> list[Sample]:
    """Whole split at original resolution (small datasets, tests, evaluation of few images)."""
    return list(SplitSource(version, split, max_cached=0))


def count_batches(n_images: int, patches_per_image: int, batch_size: int) -> int:
    return -(-(n_images * patches_per_image) // batch_size)


def _patch_order(n_images: int, patches_per_image: int, seed: int, epoch: int,
                 images_in_memory: int) -> list[tuple[int, int]]:
    order = [(i, r) for i in range(n_images) for r in range(patches_per_image)]
    if images_in_memory <= 0 or images_in_memory >= n_images:
        perm = np.random.default_rng([seed, epoch, 0]).permutation(len(order))
        return [order[j] for j in perm]
    images = np.random.default_rng([seed, epoch, 2]).permutation(n_images)
    out: list[tuple[int, int]] = []
    for b, start in enumerate(range(0, n_images, images_in_memory)):
        block = [(int(i), r) for i in images[start:start + images_in_memory] for r in range(patches_per_image)]
        perm = np.random.default_rng([seed, epoch, 3, b]).permutation(len(block))
        out.extend(block[j] for j in perm)
    return out


def iter_epoch_batches(source: SplitSource | list[Sample], cfg: AugmentConfig, patches_per_image: int,
                       batch_size: int, seed: int, epoch: int, normalization: str = "dtype_range",
                       images_in_memory: int = 0) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Deterministic augmented batches for one epoch: (N, C, P, P) float32 and (N, P, P) int64."""
    get = source.get if isinstance(source, SplitSource) else source.__getitem__
    order = _patch_order(len(source), patches_per_image, seed, epoch, images_in_memory)
    for start in range(0, len(order), batch_size):
        imgs, labs = [], []
        for i, r in order[start:start + batch_size]:
            s = get(i)
            rng = np.random.default_rng([seed, epoch, 1, i, r])
            img, lab, _ = augment(s.image, s.labels, cfg, rng)
            imgs.append(normalize(img, normalization, s.stats).transpose(2, 0, 1))
            labs.append(lab)
        yield np.stack(imgs).astype(np.float32), np.stack(labs).astype(np.int64)


def epoch_batches(*args, **kwargs) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(iter_epoch_batches(*args, **kwargs))


_DONE = object()


def prefetch(iterator: Iterator, depth: int = 2) -> Iterator:
    """Run ``iterator`` in a background thread, ``depth`` items ahead. Errors are re-raised in the consumer."""
    q: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def produce() -> None:
        try:
            for item in iterator:
                while not stop.is_set():
                    try:
                        q.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
            q.put(_DONE)
        except BaseException as exc:  # handed to the consumer
            q.put(exc)

    thread = threading.Thread(target=produce, name="msw-batch-prefetch", daemon=True)
    thread.start()
    try:
        while True:
            item = q.get()
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        thread.join(timeout=10)
