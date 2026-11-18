"""The public data contract between the UI and the backend.

Everything crossing the HTTP boundary is defined here as pydantic models:

* :class:`GraphSpec` — the node graph the user builds in the editor.
* :class:`LayerInfo` / :class:`ParamDef` — the layer catalogue the UI draws its
  palette from (``GET /layers``).
* :class:`BuildResult` / :class:`TrainResult` — the responses.

Keeping this file framework-free (no torch) makes the contract easy to read and
reuse from tests and, later, from a TypeScript type generator.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Graph specification (request to /build and /train)
# --------------------------------------------------------------------------- #
class NodeSpec(BaseModel):
    """One node in the graph: a layer instance with its parameters."""

    id: str
    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class TrainSpec(BaseModel):
    """How to train the assembled model. v1 uses a tiny synthetic task."""

    dataset: str = "synthetic"
    task: Literal["classification"] = "classification"
    epochs: int = Field(default=1, ge=1, le=100)
    steps: int | None = Field(default=None, description="hard cap on optimizer steps")
    lr: float = Field(default=1e-2, gt=0)
    batch_size: int = Field(default=64, ge=1, le=4096)
    samples: int = Field(default=1024, ge=16, le=100_000)
    num_classes: int = Field(default=10, ge=2, le=1000)
    optimizer: Literal["adam", "sgd"] = "adam"
    seed: int = 0
    #: Stage 9 — rows for the "custom" dataset: each row is ``[f1, …, fk, label]``
    #: (the class label is the last value). Used only when ``dataset == "custom"``.
    custom_rows: list[list[float]] | None = None


class GraphSpec(BaseModel):
    """A full graph: nodes + directed edges ``[source_id, target_id]`` + train cfg."""

    nodes: list[NodeSpec]
    edges: list[list[str]] = Field(default_factory=list)
    train: TrainSpec = Field(default_factory=TrainSpec)

    @field_validator("edges")
    @classmethod
    def _check_edges(cls, edges: list[list[str]]) -> list[list[str]]:
        for e in edges:
            if len(e) != 2:
                raise ValueError(f"edge must be [source_id, target_id], got {e!r}")
        return edges


# --------------------------------------------------------------------------- #
# Layer catalogue (response of /layers)
# --------------------------------------------------------------------------- #
class ParamDef(BaseModel):
    """One editable parameter of a layer, so the UI can render the right control."""

    name: str
    type: Literal["int", "float", "bool", "str", "intlist"]
    default: Any = None
    label: str | None = None
    min: float | None = None
    max: float | None = None
    help: str | None = None


class LayerInfo(BaseModel):
    type: str
    category: str
    label: str
    doc: str = ""
    params: list[ParamDef] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Responses of /build and /train
# --------------------------------------------------------------------------- #
class LayerSummary(BaseModel):
    id: str
    type: str
    in_shape: list[int] | None
    out_shape: list[int] | None
    num_params: int


class BuildResult(BaseModel):
    ok: bool
    order: list[str]
    layers: list[LayerSummary]
    total_params: int


class TrainResult(BaseModel):
    ok: bool
    loss_curve: list[float]
    #: Full-set accuracy after each completed epoch (Stage 8 — the accuracy curve
    #: next to the per-step loss curve). Empty if training stopped before an epoch.
    acc_curve: list[float] = Field(default_factory=list)
    final_loss: float
    final_accuracy: float
    steps: int
    device: str
    total_params: int


# --------------------------------------------------------------------------- #
# Smart adapters (response of /suggest) — Stage 5
# --------------------------------------------------------------------------- #
class AdapterProposal(BaseModel):
    """A proposed adapter node to insert on a ``source -> target`` edge so the
    target can consume the source's shape (e.g. a Flatten before a Linear)."""

    source_id: str
    target_id: str
    adapter_type: str
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str
    in_shape: list[int]
    out_shape: list[int]


class SuggestResult(BaseModel):
    ok: bool
    proposals: list[AdapterProposal] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Architecture library (response of /architectures) — Stage 7
# --------------------------------------------------------------------------- #
class ArchitectureInfo(BaseModel):
    """A ready public architecture the editor can drop onto the canvas as a full,
    editable graph. The graph itself comes from ``GET /architectures/{id}`` (a
    plain :class:`GraphSpec`); this is the lightweight catalogue entry."""

    id: str
    name: str
    description: str
    category: str = "general"
    tags: list[str] = Field(default_factory=list)
    input_shape: list[int] = Field(default_factory=list)
    node_count: int


# --------------------------------------------------------------------------- #
# Datasets & data preview (response of /datasets and /data/preview) — Stage 8
# --------------------------------------------------------------------------- #
class DatasetInfo(BaseModel):
    """A dataset the trainer can run on. v1 ships self-contained synthetic tasks
    (no download, no GPU); real datasets (MNIST/CIFAR/upload) plug in here next."""

    id: str
    name: str
    description: str
    kind: str = "synthetic"


class DataSampleRow(BaseModel):
    """One example from the training set: its class label and its (flattened,
    rounded, possibly truncated) feature values, for the editor's data modal."""

    label: int
    values: list[float]
    truncated: bool = False
    total: int


class DataPreview(BaseModel):
    """What the training set actually looks like for the graph the user built —
    its format plus a few real sample rows (Stage 8's data modal)."""

    dataset: str
    dataset_name: str
    description: str
    input_shape: list[int]
    dtype: str
    num_classes: int
    num_samples: int
    feature_count: int
    rows: list[DataSampleRow] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Export (response of /export/pytorch) — Stage 8
# --------------------------------------------------------------------------- #
class ExportResult(BaseModel):
    """Standalone, runnable PyTorch source for the graph the user built — the
    architecture as a real ``nn.Module`` (optionally with a tiny training loop),
    depending on nothing but torch."""

    ok: bool
    language: str = "python"
    filename: str
    code: str
