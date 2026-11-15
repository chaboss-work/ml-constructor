"""Shape utilities shared across the core.

A *shape* is the per-sample tensor shape WITHOUT the batch dimension, written as
a tuple of ints. Examples: an MNIST image is ``(1, 28, 28)``; a flat feature
vector is ``(128,)``. ``None`` means "no shape yet" — e.g. the input side of a
source node before the graph's ``Input`` provides one.
"""

from __future__ import annotations

from math import prod as _prod

Shape = tuple[int, ...]


class ShapeError(ValueError):
    """Raised when a layer receives an input shape it cannot consume.

    This is the seed of the project's "smart adapters" vision: instead of a
    cryptic tensor error deep inside the forward pass, the builder reports
    exactly which connection is invalid and why, pointing at the node id.
    """


def numel(shape: Shape | None) -> int:
    """Number of elements per sample for ``shape`` (0 if unknown)."""
    return int(_prod(shape)) if shape else 0


def as_list(shape: Shape | None) -> list[int] | None:
    return list(shape) if shape is not None else None
