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

from pydantic import BaseModel, Field, field_validator, model_validator


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
    #: Stage 11 — save the best-accuracy weights to a checkpoint during training,
    #: and/or resume from this architecture's last checkpoint before training.
    checkpoint: bool = False
    resume: bool = False
    #: Stage 12 — where to train: "cpu" (default — never touches the shared GPU),
    #: "cuda" (the GPU; a clear error if none is available) or "auto" (GPU if any).
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    #: Stage 15 — the share of rows held out for validation: the network never
    #: trains on them, so their accuracy shows how it does on new data. 0 = off
    #: (train and measure on every row, the pre-Stage-15 behaviour).
    val_fraction: float = Field(default=0.2, ge=0.0, le=0.5)
    #: Stage 16 — the share of rows kept for a final test: used neither for training
    #: nor for judging epochs, measured only by ``POST /test``. 0 = no test part.
    test_fraction: float = Field(default=0.1, ge=0.0, le=0.5)
    #: Stage 16 — start from the weights of an earlier run (a finished training or
    #: an uploaded weights file, see ``run_id``) instead of fresh ones.
    init_run: str | None = None
    #: Stage 18 — the text for the "text" dataset ("Your text"): a language model
    #: learns to continue it, one character at a time.
    text: str | None = Field(default=None, max_length=1_000_000)

    @model_validator(mode="after")
    def _leave_rows_to_train(self) -> "TrainSpec":
        held = self.val_fraction + self.test_fraction
        if held > 0.7 + 1e-9:
            raise ValueError(
                f"validation ({self.val_fraction:.0%}) + test ({self.test_fraction:.0%}) would hold out "
                f"{held:.0%} of the rows; leave at least 30 % for training"
            )
        return self


class EdgeSetting(BaseModel):
    """Stage 14 — how one connection reshapes what flows through it.

    ``transform`` is ``"auto"`` (pick the simplest transform that makes the shapes
    fit — nothing at all when they already do), ``"none"`` (pass through as is) or
    a concrete transform from :mod:`mlconstructor.connections` with its ``params``.
    Edges without a setting use ``"auto"``."""

    source: str
    target: str
    transform: str = "auto"
    params: dict[str, Any] = Field(default_factory=dict)


class GraphSpec(BaseModel):
    """A full graph: nodes + directed edges ``[source_id, target_id]`` + train cfg."""

    nodes: list[NodeSpec]
    edges: list[list[str]] = Field(default_factory=list)
    train: TrainSpec = Field(default_factory=TrainSpec)
    #: Stage 14 — per-connection transforms; edges not listed here use "auto".
    edge_settings: list[EdgeSetting] = Field(default_factory=list)

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
    #: Stage 14 — a fixed set of allowed values (the editor shows a selector).
    choices: list[str] | None = None


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


class EdgeSummary(BaseModel):
    """Stage 14 — what one connection does in the built model."""

    source: str
    target: str
    #: the transform in use ("none" = passed through unchanged)
    transform: str
    #: True when "auto" picked it (the user did not choose one)
    auto: bool
    #: short text for the badge on the wire, e.g. "flatten" or "→ 2×8×8" ("" = nothing to show)
    label: str
    params: dict[str, Any] = Field(default_factory=dict)
    in_shape: list[int]
    out_shape: list[int]
    num_params: int = 0


class BuildResult(BaseModel):
    ok: bool
    order: list[str]
    layers: list[LayerSummary]
    total_params: int
    #: Stage 14 — every connection with its transform (shapes on both ends).
    edges: list[EdgeSummary] = Field(default_factory=list)
    #: Stage 17 — small enough to train (and load weights into) on this server; a
    #: real-scale flagship is built on the meta device to inspect, not to train.
    trainable: bool = True
    max_train_params: int | None = None


class ConnectionOption(BaseModel):
    """One choice for a connection's transform, as the editor's selector shows it."""

    transform: str
    label: str
    description: str
    learnable: bool = False
    params: list[ParamDef] = Field(default_factory=list)
    #: suggested parameter sets, best first (the first one is the default)
    presets: list[dict[str, Any]] = Field(default_factory=list)
    #: whether the target accepts the result (with the default preset)
    fits: bool = False
    out_shape: list[int] | None = None
    #: why it does not fit (or a short remark when it does)
    note: str | None = None


class ConnectionOptionsRequest(BaseModel):
    spec: GraphSpec
    source: str
    target: str


class ConnectionOptions(BaseModel):
    """Everything the connection inspector needs for one edge."""

    source: str
    target: str
    #: the shape arriving from the source (None if it cannot be computed yet)
    in_shape: list[int] | None = None
    #: the edge's current setting ("auto" or a concrete transform)
    current: str = "auto"
    current_params: dict[str, Any] = Field(default_factory=dict)
    #: what "auto" resolves to right now (None if it finds nothing)
    auto_transform: str | None = None
    auto_params: dict[str, Any] = Field(default_factory=dict)
    auto_out_shape: list[int] | None = None
    #: why the connection does not work at the moment (None when it does)
    problem: str | None = None
    options: list[ConnectionOption] = Field(default_factory=list)


class TrainResult(BaseModel):
    ok: bool
    loss_curve: list[float]
    #: Accuracy on the training rows after each completed epoch (Stage 8 — the
    #: accuracy curve next to the per-step loss curve; since Stage 15 the training
    #: part only, every row when validation is off). Empty if training stopped
    #: before an epoch.
    acc_curve: list[float] = Field(default_factory=list)
    #: Loss and accuracy on the training rows at the end.
    final_loss: float
    final_accuracy: float
    steps: int
    device: str
    total_params: int
    #: Stage 11 — the saved checkpoint's key (None if none saved), the best accuracy
    #: seen during training (since Stage 15: on the validation rows when there are
    #: some — the checkpoint keeps those weights — else on the training rows), and
    #: whether training resumed from a checkpoint.
    checkpoint: str | None = None
    best_accuracy: float | None = None
    resumed: bool = False
    #: Stage 15 — the held-out validation part: rows the network never trains on.
    #: Its accuracy is the honest "does it work on new data" number; the fields are
    #: empty / None when validation is off or the set is too small to hold rows out.
    val_acc_curve: list[float] = Field(default_factory=list)
    val_loss_curve: list[float] = Field(default_factory=list)
    final_val_accuracy: float | None = None
    final_val_loss: float | None = None
    train_samples: int | None = None
    val_samples: int = 0
    #: A plain-words verdict on train vs validation accuracy: "generalises",
    #: "overfits", "underfits" or "unknown" (no validation part), and why.
    fit: str | None = None
    fit_note: str | None = None
    #: Things worth knowing about this run (validation off, too few rows, …).
    warnings: list[str] = Field(default_factory=list)
    #: Stage 16 — rows kept for the final test (not trained on, not measured here:
    #: ``POST /test`` measures them), and the id the trained weights are kept under
    #: on the server (download: ``GET /runs/{run_id}/weights``; None if not kept).
    test_samples: int = 0
    run_id: str | None = None


# --------------------------------------------------------------------------- #
# Final test & trained weights (POST /test, /weights/upload) — Stage 16
# --------------------------------------------------------------------------- #
class TestRequest(BaseModel):
    """Measure the weights of a run on the test rows of this graph's data."""

    spec: GraphSpec
    run_id: str


class TestResult(BaseModel):
    """The final test: the run's weights on the rows kept out of training and out
    of the epoch-by-epoch validation — the number to trust once tuning is done."""

    ok: bool = True
    run_id: str
    accuracy: float
    loss: float
    samples: int
    #: The same weights on the validation and training rows, for comparison.
    val_accuracy: float | None = None
    train_accuracy: float | None = None
    val_samples: int = 0
    train_samples: int = 0
    #: True: the weights were trained on this very split; False: on other data or
    #: another split (some test rows may have been trained on); None: unknown
    #: (uploaded weights that carry no record of their training data).
    same_split: bool | None = None
    note: str | None = None
    warnings: list[str] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    """Stage 18 — continue a prompt with a run's weights, trained on "Your text"."""

    spec: GraphSpec
    run_id: str
    prompt: str = Field(default="", max_length=4000)
    length: int = Field(default=200, ge=1, le=2000)
    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    seed: int = 0


class GenerateResult(BaseModel):
    """The continuation of a prompt, written one character at a time."""

    ok: bool = True
    run_id: str
    prompt: str
    text: str
    #: prompt characters the vocabulary does not know (read as "unknown")
    unknown: int = 0
    vocab_size: int
    context: int


class WeightsUpload(BaseModel):
    """A weights file (``.pt``, base64) to load into this graph."""

    spec: GraphSpec
    data: str
    name: str | None = None


class WeightsInfo(BaseModel):
    """Weights loaded and kept on the server as a run, ready to test or train from."""

    ok: bool = True
    run_id: str
    tensors: int
    params: int
    #: "names" — every tensor matched by name; "order" — by position, every shape
    #: lining up (e.g. weights saved by the exported script, whose layer names differ).
    matched: str
    #: Whether the file says which data and split it was trained on, and if that is
    #: this graph's current one.
    data_known: bool = False
    same_split: bool | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    note: str | None = None


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
    #: Stage 17 — the flagships: "mini" (trains here) or "real" (the published model,
    #: sized on the meta device: inspect only). A mini entry points to its real size.
    scale: str | None = None
    real_id: str | None = None
    real_name: str | None = None
    #: the real model's parameter count (Hugging Face; the model card for DeepSeek-V3)
    real_params: int | None = None
    hf_url: str | None = None


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
    #: Fixed input shape a real dataset requires (MNIST [1,28,28]); None = adaptive.
    input_shape: list[int] | None = None


class DataSampleRow(BaseModel):
    """One example from the training set: its class label and its (flattened,
    rounded, possibly truncated) feature values, for the editor's data modal."""

    label: int
    values: list[float]
    truncated: bool = False
    total: int
    #: Stage 15 — which part the row belongs to: "train", "validation" (held out)
    #: or, since Stage 16, "test" (kept for the final test).
    split: str = "train"
    #: Stage 18 — for "Your text": the window as text, and the character that follows.
    text: str | None = None
    label_text: str | None = None


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
    #: Stage 15 — how the rows split: trained on vs held out for validation
    #: (and, since Stage 16, kept for the final test).
    train_samples: int | None = None
    val_samples: int = 0
    test_samples: int = 0
    #: Stage 18 — for "Your text": its length, its different characters, and how many
    #: tokens the network scores (rarer characters share an "unknown" one beyond that).
    text_chars: int | None = None
    distinct_chars: int | None = None
    vocab_size: int | None = None


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


# --------------------------------------------------------------------------- #
# Projects: save / open / autosave (Stage 13)
# --------------------------------------------------------------------------- #
PROJECT_FORMAT = "mlconstructor.project"
#: 1 — Stage 13; 2 — Stage 14 adds per-connection settings (``graph.edge_settings``).
PROJECT_FORMAT_VERSION = 2


class NodePosition(BaseModel):
    x: float
    y: float


class ProjectNode(NodeSpec):
    """A graph node plus where it sits on the canvas (``None`` = lay it out)."""

    position: NodePosition | None = None


class ProjectGraph(BaseModel):
    nodes: list[ProjectNode] = Field(default_factory=list)
    edges: list[list[str]] = Field(default_factory=list)
    edge_settings: list[EdgeSetting] = Field(default_factory=list)

    @field_validator("edges")
    @classmethod
    def _check_edges(cls, edges: list[list[str]]) -> list[list[str]]:
        for e in edges:
            if len(e) != 2:
                raise ValueError(f"edge must be [source_id, target_id], got {e!r}")
        return edges


class ProjectView(BaseModel):
    pan_x: float = 48.0
    pan_y: float = 36.0
    zoom: float = 1.0


class ProjectEditorState(BaseModel):
    """Editor-only state that does not change the model — kept so a reopened project
    looks exactly as it was left."""

    custom_text: str = ""
    view: ProjectView = Field(default_factory=ProjectView)


class ProjectFile(BaseModel):
    """A saved project (``.mlc.json``): the graph with node positions, the training
    setup and metadata. It carries a format version, so older files keep opening."""

    format: Literal["mlconstructor.project"] = PROJECT_FORMAT
    version: int = PROJECT_FORMAT_VERSION
    name: str = Field(default="Untitled", max_length=120)
    description: str = Field(default="", max_length=2000)
    created_at: str | None = None
    updated_at: str | None = None
    graph: ProjectGraph = Field(default_factory=ProjectGraph)
    train: TrainSpec = Field(default_factory=TrainSpec)
    editor: ProjectEditorState = Field(default_factory=ProjectEditorState)


class ProjectWarning(BaseModel):
    """Something worth telling the user when a project opens (it still opens).
    ``code`` is stable, so the UI can translate it later (Stage 16)."""

    code: str
    message: str
    node_id: str | None = None


class ProjectLoadResult(BaseModel):
    ok: bool
    project: ProjectFile
    migrated_from: int | None = None
    warnings: list[ProjectWarning] = Field(default_factory=list)


class ProjectSummary(BaseModel):
    slug: str
    name: str
    updated_at: str | None = None
    node_count: int = 0


class ProjectSaveResult(BaseModel):
    ok: bool
    slug: str
    updated_at: str
