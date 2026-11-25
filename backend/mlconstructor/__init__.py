"""ML Constructor core — graph spec → ``nn.Module`` → training.

Public surface:

* :func:`catalog` — the layer palette (for ``GET /layers``).
* :func:`build_model` — turn a :class:`~mlconstructor.schema.GraphSpec` into a
  :class:`~mlconstructor.blocks.NeuralNetwork` plus a per-layer summary.
* :func:`train_model` — run the tiny v1 training loop and return the loss curve.
* :func:`normalize_project` & co. — open / save versioned project files (Stage 13).
"""

from __future__ import annotations

from . import layers  # noqa: F401  (import-time side effect: registers all layers)
from .adapters import apply_adapter, suggest_for_graph
from .architectures import get_architecture, list_architectures
from .builder import build_model, connection_options
from .datasets import list_datasets, make_dataset
from .export import generate_pytorch
from .projects import (
    create_project,
    delete_project,
    list_projects,
    load_project,
    normalize_project,
    save_project,
    server_projects_enabled,
    slugify,
)
from .registry import LAYER_REGISTRY, catalog, get_layer_cls
from .trainer import data_preview, train_iter, train_model

__version__ = "0.1.0"
#: The latest stage whose HTTP contract this backend serves (``GET /health`` reports
#: it), so the editor can tell when the server runs an older backend than the page.
API_LEVEL = 14

__all__ = [
    "build_model",
    "connection_options",
    "train_model",
    "train_iter",
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
    "normalize_project",
    "create_project",
    "list_projects",
    "load_project",
    "save_project",
    "delete_project",
    "server_projects_enabled",
    "slugify",
    "LAYER_REGISTRY",
    "__version__",
    "API_LEVEL",
]
