"""ML Constructor core — graph spec → ``nn.Module`` → training.

Public surface:

* :func:`catalog` — the layer palette (for ``GET /layers``).
* :func:`build_model` — turn a :class:`~mlconstructor.schema.GraphSpec` into a
  :class:`~mlconstructor.blocks.NeuralNetwork` plus a per-layer summary.
* :func:`train_model` — run the tiny v1 training loop and return the loss curve
  (since Stage 15 with a held-out validation part, see :func:`split_indices`).
* :func:`test_run` / :func:`adopt_weights` — the final test on kept or uploaded
  weights (Stage 16).
* :func:`normalize_project` & co. — open / save versioned project files (Stage 13).
"""

from __future__ import annotations

from .gpu import CUDA_HIDDEN, cuda_available  # first: settles hidden GPUs before torch starts the driver
from . import layers  # noqa: F401  (import-time side effect: registers all layers)
from .adapters import apply_adapter, suggest_for_graph
from .architectures import get_architecture, list_architectures
from .builder import build_model, connection_options
from .datasets import list_datasets, make_dataset, make_rows, split_indices, split_parts, split_rows
from .example_text import EXAMPLE_TEXT
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
from .trainer import (
    MAX_TRAIN_PARAMS,
    adopt_weights,
    check_trainable,
    count_params,
    data_preview,
    generate,
    run_once,
    save_trained,
    start_from_run,
    test_run,
    train_block_reason,
    train_iter,
    train_model,
)
from .weights import MAX_WEIGHTS_BYTES, RunNotFound, read_weights, run_path, weights_bytes

__version__ = "0.1.0"
#: The latest stage whose HTTP contract this backend serves (``GET /health`` reports
#: it), so the editor can tell when the server runs an older backend than the page.
API_LEVEL = 19

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
    "split_indices",
    "split_parts",
    "split_rows",
    "make_rows",
    "generate",
    "EXAMPLE_TEXT",
    "check_trainable",
    "run_once",
    "train_block_reason",
    "count_params",
    "MAX_TRAIN_PARAMS",
    "save_trained",
    "start_from_run",
    "test_run",
    "adopt_weights",
    "read_weights",
    "weights_bytes",
    "run_path",
    "RunNotFound",
    "MAX_WEIGHTS_BYTES",
    "generate_pytorch",
    "normalize_project",
    "create_project",
    "list_projects",
    "load_project",
    "save_project",
    "delete_project",
    "server_projects_enabled",
    "slugify",
    "cuda_available",
    "CUDA_HIDDEN",
    "LAYER_REGISTRY",
    "__version__",
    "API_LEVEL",
]
