"""Importing this subpackage registers every built-in layer.

Each module below calls :func:`mlconstructor.registry.register` at import time,
so a plain ``from mlconstructor import layers`` populates the whole palette.
To add a layer: create a new module here and import it in this list.
"""

from . import (  # noqa: F401
    activation,
    conv,
    io,
    linear,
    merge,
    norm,
    pooling,
    regularization,
    shaping,
)

__all__ = [
    "activation",
    "conv",
    "io",
    "linear",
    "merge",
    "norm",
    "pooling",
    "regularization",
    "shaping",
]
