"""The dataset registry (Stage 8).

Where the training data comes from. v1 shipped a single inline synthetic task; it
now lives here as one entry of a small **registry**, so the same generator feeds
both the trainer and the ``/data/preview`` modal — the preview is exactly the data
training sees — and new datasets (and later the real ones: MNIST/CIFAR/upload)
plug in by adding one :class:`DatasetDef`.

Every dataset is **self-contained and synthetic**: no download, no GPU, provably
learnable, and works for any input shape (a flat vector, an image ``(C,H,W)`` or a
sequence ``(L, F)``), with the class count taken from the network the user built.
A ``Shape`` here is per-sample, without the batch dimension.
"""

from __future__ import annotations

import gzip
import math
import os
import pickle
import shutil
import struct
import tarfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .schema import DatasetInfo, TrainSpec
from .shapes import Shape, numel

_MAX_CUSTOM_ROWS = 10000  # bound the payload / memory for pasted-or-uploaded data

# Local cache for auto-downloaded real datasets (gitignored: backend/data/).
_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
# Only these official hosts may be downloaded from (Stage 10).
_OFFICIAL_HOSTS = {"ossci-datasets.s3.amazonaws.com", "www.cs.toronto.edu"}

#: A generator: (per-sample shape, #classes, train spec, device) -> (X, y).
DatasetFn = Callable[[Shape, int, TrainSpec, "torch.device"], tuple["torch.Tensor", "torch.Tensor"]]


@dataclass(frozen=True)
class DatasetDef:
    id: str
    name: str
    description: str
    make: DatasetFn
    #: Real datasets (MNIST/CIFAR) have a FIXED input shape and class count the
    #: graph must match; adaptive synthetic datasets leave these None.
    fixed_shape: tuple[int, ...] | None = None
    fixed_classes: int | None = None
    kind: str = "synthetic"


def _blobs(in_shape: Shape, num_classes: int, spec: TrainSpec, device) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Separable Gaussian clusters: one random prototype per class + noise.

    This is the exact v1 synthetic task (kept byte-for-byte: same RNG draw order),
    so every existing result is unchanged. Signal std 1.0 vs noise std 0.5 keeps
    the classes well separated, so any sane network learns it quickly.
    """
    g = torch.Generator().manual_seed(spec.seed)
    n = spec.samples
    prototypes = torch.randn(num_classes, *in_shape, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    noise = torch.randn(n, *in_shape, generator=g)
    X = prototypes[y] + 0.5 * noise
    return X.to(device), y.to(device)


def _shells(in_shape: Shape, num_classes: int, spec: TrainSpec, device) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Concentric shells: the class is the sample's *distance from the origin*.

    Each sample points in a random direction and sits on the shell of radius
    ``class + 1`` (plus a little noise). The decision boundary is the vector norm —
    **not** linearly separable — so a bare Linear can't solve it but an MLP/CNN can.
    A nice contrast to ``blobs`` for showing a model learn a curved boundary.
    """
    g = torch.Generator().manual_seed(spec.seed)
    n = spec.samples
    y = torch.randint(0, num_classes, (n,), generator=g)
    direction = torch.randn(n, *in_shape, generator=g)
    norm = direction.flatten(1).norm(dim=1).view(n, *([1] * len(in_shape))) + 1e-8
    unit = direction / norm
    radius = (y.float() + 1.0).view(n, *([1] * len(in_shape)))
    X = unit * radius + 0.15 * torch.randn(n, *in_shape, generator=g)
    return X.to(device), y.to(device)


def _custom(in_shape: Shape, num_classes: int, spec: TrainSpec, device) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Train on the user's OWN rows (Stage 9 — paste or upload your data).

    Each row in ``spec.custom_rows`` is ``[f1, …, fk, label]``: its features then
    an integer class label. Rows are validated against the graph's input shape and
    the network's class count, so a mismatch is a clear message, not a torch crash.
    """
    rows = spec.custom_rows or []
    if not rows:
        raise ValueError(
            "the 'custom' dataset needs rows — paste or upload training examples "
            "(each row = the features, then a whole-number class label)"
        )
    if len(rows) > _MAX_CUSTOM_ROWS:
        raise ValueError(f"too many custom rows ({len(rows)}); the limit is {_MAX_CUSTOM_ROWS}")
    feat = numel(in_shape)
    xs: list[list[float]] = []
    ys: list[int] = []
    for i, row in enumerate(rows):
        if len(row) != feat + 1:
            raise ValueError(
                f"custom row {i + 1}: expected {feat} feature(s) + 1 label = {feat + 1} values, got {len(row)}"
            )
        if any(not math.isfinite(float(v)) for v in row):
            raise ValueError(f"custom row {i + 1}: contains a non-finite value (NaN/inf)")
        if int(row[-1]) != row[-1]:
            raise ValueError(f"custom row {i + 1}: label {row[-1]} must be a whole number")
        label = int(row[-1])
        if not (0 <= label < num_classes):
            raise ValueError(
                f"custom row {i + 1}: label {label} out of range [0, {num_classes}) — "
                f"the network's output layer has {num_classes} classes"
            )
        xs.append([float(v) for v in row[:-1]])
        ys.append(label)
    X = torch.tensor(xs, dtype=torch.float32).reshape(len(rows), *in_shape)
    y = torch.tensor(ys, dtype=torch.long)
    return X.to(device), y.to(device)


# --------------------------------------------------------------------------- #
# Real datasets (Stage 10): auto-downloaded from their official sources, parsed
# with the standard library + numpy (no torchvision), cached under backend/data/.
# --------------------------------------------------------------------------- #
def _download(url: str, dest: str) -> str:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if host not in _OFFICIAL_HOSTS:
        raise ValueError(f"refusing to download a dataset from a non-official host: {host!r}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        req = urllib.request.Request(url, headers={"User-Agent": "ml-constructor"})
        with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:  # noqa: S310 (host allow-listed)
            shutil.copyfileobj(resp, out)
    return dest


_MNIST_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_MNIST_IMAGES = "train-images-idx3-ubyte.gz"
_MNIST_LABELS = "train-labels-idx1-ubyte.gz"


def _load_mnist() -> tuple[np.ndarray, np.ndarray]:
    d = os.path.join(_CACHE, "mnist")
    imgs_gz = _download(_MNIST_BASE + _MNIST_IMAGES, os.path.join(d, _MNIST_IMAGES))
    lbls_gz = _download(_MNIST_BASE + _MNIST_LABELS, os.path.join(d, _MNIST_LABELS))
    with gzip.open(imgs_gz, "rb") as f:
        _magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
    with gzip.open(lbls_gz, "rb") as f:
        struct.unpack(">II", f.read(8))
        lbls = np.frombuffer(f.read(), dtype=np.uint8)
    return imgs, lbls


_CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def _load_cifar10() -> tuple[np.ndarray, np.ndarray]:
    d = os.path.join(_CACHE, "cifar10")
    tgz = _download(_CIFAR_URL, os.path.join(d, "cifar-10-python.tar.gz"))
    batch_dir = os.path.join(d, "cifar-10-batches-py")
    if not os.path.isdir(batch_dir):
        with tarfile.open(tgz, "r:gz") as t:
            t.extractall(d, filter="data")  # py3.12 safe extraction (blocks path traversal)
    data, labels = [], []
    for i in range(1, 6):
        with open(os.path.join(batch_dir, f"data_batch_{i}"), "rb") as f:
            batch = pickle.load(f, encoding="bytes")  # official CIFAR-10 pickle
        data.append(batch[b"data"])
        labels.extend(batch[b"labels"])
    X = np.concatenate(data).reshape(-1, 3, 32, 32)  # (50000, 3, 32, 32) uint8
    y = np.array(labels, dtype=np.uint8)
    return X, y


def _subsample(imgs: np.ndarray, lbls: np.ndarray, spec: TrainSpec, device):
    """A reproducible random subset of `spec.samples` images, scaled to [0, 1]."""
    n = min(int(spec.samples), imgs.shape[0])
    g = torch.Generator().manual_seed(int(spec.seed))
    sel = torch.randperm(imgs.shape[0], generator=g)[:n].numpy()
    X = torch.from_numpy(imgs[sel].astype(np.float32) / 255.0)
    y = torch.from_numpy(lbls[sel].astype(np.int64))
    return X.to(device), y.to(device)


def _check_fixed(name: str, want: tuple[int, ...], in_shape: Shape, num_classes: int):
    if tuple(int(x) for x in in_shape) != want:
        raise ValueError(f"{name} needs an Input of shape {list(want)}; your graph's input is {list(in_shape)}")
    if num_classes < 10:
        raise ValueError(
            f"{name} has 10 classes; your network outputs {num_classes} — make the last layer output at least 10"
        )


def _mnist(in_shape: Shape, num_classes: int, spec: TrainSpec, device):
    _check_fixed("MNIST", (1, 28, 28), in_shape, num_classes)
    imgs, lbls = _load_mnist()
    X, y = _subsample(imgs, lbls, spec, device)  # imgs (n,28,28) -> add channel
    return X.unsqueeze(1), y


def _cifar10(in_shape: Shape, num_classes: int, spec: TrainSpec, device):
    _check_fixed("CIFAR-10", (3, 32, 32), in_shape, num_classes)
    imgs, lbls = _load_cifar10()  # (50000,3,32,32)
    return _subsample(imgs, lbls, spec, device)


_DATASETS: list[DatasetDef] = [
    DatasetDef(
        "blobs",
        "Gaussian blobs",
        "Separable Gaussian clusters — one prototype per class plus noise. "
        "Linearly separable, so any network learns it fast.",
        _blobs,
    ),
    DatasetDef(
        "shells",
        "Concentric shells",
        "The class is the distance from the origin (radius = class). A curved, "
        "non-linear boundary — needs a hidden layer, not just a Linear.",
        _shells,
    ),
    DatasetDef(
        "custom",
        "Your data",
        "Train on rows you paste or upload: each row is its features, then a "
        "whole-number class label. The features must match the Input shape.",
        _custom,
        kind="custom",
    ),
    DatasetDef(
        "mnist",
        "MNIST (digits)",
        "Handwritten digits, 28x28 grayscale, 10 classes. Needs Input [1, 28, 28]. "
        "Auto-downloaded (~11 MB) on first use.",
        _mnist,
        fixed_shape=(1, 28, 28),
        fixed_classes=10,
        kind="real",
    ),
    DatasetDef(
        "cifar10",
        "CIFAR-10",
        "Color images, 32x32 RGB, 10 classes. Needs Input [3, 32, 32]. "
        "Auto-downloaded (~170 MB) on first use; heavy on CPU.",
        _cifar10,
        fixed_shape=(3, 32, 32),
        fixed_classes=10,
        kind="real",
    ),
]

_BY_ID: dict[str, DatasetDef] = {d.id: d for d in _DATASETS}

#: Back-compat: the v1 train spec defaults to ``dataset="synthetic"``; keep it (and
#: an empty value) meaning the original blobs task.
_ALIASES = {"synthetic": "blobs", "": "blobs"}


def canonical_dataset_id(name: str) -> str:
    """The catalogue id for a dataset name (resolves the legacy aliases)."""
    return _ALIASES.get(name, name)


def get_dataset(name: str) -> DatasetDef:
    key = canonical_dataset_id(name)
    try:
        return _BY_ID[key]
    except KeyError:
        known = ", ".join(sorted(_BY_ID))
        raise ValueError(
            f"unknown dataset {name!r}; known: {known} (alias 'synthetic' = blobs)"
        ) from None


def make_dataset(
    name: str, in_shape: Shape, num_classes: int, spec: TrainSpec, device
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Generate ``(X, y)`` for the named dataset — the single entry point the
    trainer and the preview both call, so they can never drift apart."""
    return get_dataset(name).make(in_shape, num_classes, spec, device)


def split_indices(y: torch.Tensor, val_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split row indices into a training part and a held-out validation part (Stage 15).

    The network trains only on the first part; the second measures how it does on
    rows it never saw. The split is deterministic for a seed, stratified (each class
    keeps its share in both parts, by largest remainders) and safe on tiny sets: every
    class keeps at least one row to train on, so the validation part is empty when
    nothing can be held out that way (one row, or one row per class). Returns two
    sorted CPU index tensors. Self-contained on purpose: the PyTorch export embeds
    this very function, so an exported file splits exactly like the app.
    """
    n = int(y.shape[0])
    every = torch.arange(n)
    if n < 2 or not val_fraction or val_fraction <= 0:
        return every, every[:0]
    want = min(max(1, round(n * float(val_fraction))), n - 1)
    g = torch.Generator().manual_seed(int(seed) + 101)  # its own stream: the data generator uses `seed`
    labels = y.detach().to("cpu").long().reshape(-1)
    order = torch.randperm(n, generator=g)  # a random order of the rows ...
    order = order[torch.argsort(labels[order], stable=True)]  # ... grouped by class
    counts = torch.unique_consecutive(labels[order], return_counts=True)[1].tolist()
    # Each class's share of `want`, at most count - 1 so it keeps a row for training.
    take = [min(c * want // n, c - 1) for c in counts]
    left = want - sum(take)
    tie = torch.randperm(len(counts), generator=g).tolist()
    ranked = sorted(range(len(counts)), key=lambda i: (-((counts[i] * want) % n), tie[i]))
    while left > 0:
        progressed = False
        for i in ranked:
            if left > 0 and take[i] < counts[i] - 1:
                take[i] += 1
                left -= 1
                progressed = True
        if not progressed:  # every class is down to its last training row
            break
    is_val = torch.zeros(n, dtype=torch.bool)
    start = 0
    for c, k in zip(counts, take):
        is_val[order[start : start + k]] = True
        start += c
    return every[~is_val], every[is_val]


def list_datasets() -> list[DatasetInfo]:
    """The dataset catalogue for the editor's picker (``GET /datasets``)."""
    return [
        DatasetInfo(
            id=d.id,
            name=d.name,
            description=d.description,
            kind=d.kind,
            input_shape=(list(d.fixed_shape) if d.fixed_shape else None),
        )
        for d in _DATASETS
    ]
