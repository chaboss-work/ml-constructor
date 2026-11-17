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
from .builder import build_model
from .registry import LAYER_REGISTRY, catalog, get_layer_cls
from .trainer import train_model

__version__ = "0.1.0"

__all__ = [
    "build_model",
    "train_model",
    "catalog",
    "get_layer_cls",
    "suggest_for_graph",
    "apply_adapter",
    "LAYER_REGISTRY",
    "__version__",
]
