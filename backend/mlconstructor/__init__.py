"""ML Constructor core — graph spec → ``nn.Module`` → training.

Public surface:

* :func:`catalog` — the layer palette (for ``GET /layers``).
* :func:`build_model` — turn a :class:`~mlconstructor.schema.GraphSpec` into a
  :class:`~mlconstructor.blocks.NeuralNetwork` plus a per-layer summary.
* :func:`train_model` — run the tiny v1 training loop and return the loss curve.
"""

from __future__ import annotations

from . import layers  # noqa: F401  (import-time side effect: registers all layers)
from .adapters import apply_adapter, suggest_for_graph
from .architectures import get_architecture, list_architectures
from .builder import build_model
from .datasets import list_datasets, make_dataset
from .export import generate_pytorch
from .registry import LAYER_REGISTRY, catalog, get_layer_cls
from .trainer import data_preview, train_model

__version__ = "0.1.0"

__all__ = [
    "build_model",
    "train_model",
    "data_preview",
    "catalog",
    "get_layer_cls",
    "suggest_for_graph",
    "apply_adapter",
    "list_architectures",
    "get_architecture",
    "list_datasets",
    "make_dataset",
    "generate_pytorch",
    "LAYER_REGISTRY",
    "__version__",
]
