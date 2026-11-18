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

from dataclasses import dataclass
from typing import Callable

import torch

from .schema import DatasetInfo, TrainSpec
from .shapes import Shape

#: A generator: (per-sample shape, #classes, train spec, device) -> (X, y).
DatasetFn = Callable[[Shape, int, TrainSpec, "torch.device"], tuple["torch.Tensor", "torch.Tensor"]]


@dataclass(frozen=True)
class DatasetDef:
    id: str
    name: str
    description: str
    make: DatasetFn


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
]

_BY_ID: dict[str, DatasetDef] = {d.id: d for d in _DATASETS}

#: Back-compat: the v1 train spec defaults to ``dataset="synthetic"``; keep it (and
#: an empty value) meaning the original blobs task.
_ALIASES = {"synthetic": "blobs", "": "blobs"}


def get_dataset(name: str) -> DatasetDef:
    key = _ALIASES.get(name, name)
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


def list_datasets() -> list[DatasetInfo]:
    """The dataset catalogue for the editor's picker (``GET /datasets``)."""
    return [DatasetInfo(id=d.id, name=d.name, description=d.description, kind="synthetic") for d in _DATASETS]
