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
    final_loss: float
    final_accuracy: float
    steps: int
    device: str
    total_params: int
