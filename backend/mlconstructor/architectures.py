"""Architecture library (Stage 7): ready public architectures as full graphs.

Each architecture is a **plain graph** in exactly the node/edge vocabulary the
editor already speaks — the same ``GraphSpec`` the UI builds by hand. So a
"ready architecture" is not an opaque black box: it drops onto the canvas as
ordinary, fully editable nodes. Every layer, every repeat, every skip connection
is there to inspect, re-wire and tune. "Start from a known-good net, then make it
your own."

The four starters deliberately span the whole project:

* **MLP** — a classic fully-connected classifier (Stages 0/2: Linear, ReLU, Dropout).
* **CNN** — two convolutional blocks and a dense head (Stages 0/2: Conv2d, pooling).
* **ResNet** — residual blocks whose skip connections are real ``Add`` merge nodes
  (Stage 1's DAG), including a 1x1 projection skip when the channel count changes.
* **GPT** — a tiny transformer: token ``Embedding`` + sinusoidal ``PositionalEncoding``
  (Stages 3/4) and an N-deep stack of encoder blocks expressed with the ``Repeat``
  block (Stage 6). Bump the repeat to make it deeper.

They are defined as builder functions returning a spec dict; :func:`get_architecture`
validates one into a :class:`~mlconstructor.schema.GraphSpec` on demand (cheap and
torch-free — the actual build/train happens through the usual ``/build`` ``/train``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .schema import ArchitectureInfo, GraphSpec

Spec = dict[str, Any]


# --------------------------------------------------------------------------- #
# MLP — fully-connected classifier
# --------------------------------------------------------------------------- #
def _mlp() -> Spec:
    return {
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [64]}},
            {"id": "fc1", "type": "Linear", "params": {"out_features": 128, "bias": True}},
            {"id": "act1", "type": "ReLU", "params": {}},
            {"id": "drop", "type": "Dropout", "params": {"p": 0.1}},
            {"id": "fc2", "type": "Linear", "params": {"out_features": 64, "bias": True}},
            {"id": "act2", "type": "ReLU", "params": {}},
            {"id": "fc3", "type": "Linear", "params": {"out_features": 10, "bias": True}},
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [
            ["in", "fc1"], ["fc1", "act1"], ["act1", "drop"], ["drop", "fc2"],
            ["fc2", "act2"], ["act2", "fc3"], ["fc3", "out"],
        ],
        "train": {"epochs": 6, "lr": 0.01, "batch_size": 64, "samples": 512, "seed": 0},
    }


# --------------------------------------------------------------------------- #
# CNN — two conv blocks + dense head
# --------------------------------------------------------------------------- #
def _cnn() -> Spec:
    return {
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [1, 28, 28]}},
            {"id": "conv1", "type": "Conv2d", "params": {"out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1}},
            {"id": "bn1", "type": "BatchNorm2d", "params": {}},
            {"id": "act1", "type": "ReLU", "params": {}},
            {"id": "pool1", "type": "MaxPool2d", "params": {"kernel_size": 2, "stride": 2}},
            {"id": "conv2", "type": "Conv2d", "params": {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1}},
            {"id": "bn2", "type": "BatchNorm2d", "params": {}},
            {"id": "act2", "type": "ReLU", "params": {}},
            {"id": "pool2", "type": "MaxPool2d", "params": {"kernel_size": 2, "stride": 2}},
            {"id": "flat", "type": "Flatten", "params": {}},
            {"id": "fc1", "type": "Linear", "params": {"out_features": 64, "bias": True}},
            {"id": "act3", "type": "ReLU", "params": {}},
            {"id": "drop", "type": "Dropout", "params": {"p": 0.25}},
            {"id": "fc2", "type": "Linear", "params": {"out_features": 10, "bias": True}},
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [
            ["in", "conv1"], ["conv1", "bn1"], ["bn1", "act1"], ["act1", "pool1"],
            ["pool1", "conv2"], ["conv2", "bn2"], ["bn2", "act2"], ["act2", "pool2"],
            ["pool2", "flat"], ["flat", "fc1"], ["fc1", "act3"], ["act3", "drop"],
            ["drop", "fc2"], ["fc2", "out"],
        ],
        "train": {"epochs": 4, "lr": 0.01, "batch_size": 64, "samples": 512, "seed": 0},
    }


# --------------------------------------------------------------------------- #
# ResNet — residual blocks with real Add skip connections (Stage 1 DAG)
# --------------------------------------------------------------------------- #
def _resnet() -> Spec:
    return {
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [1, 28, 28]}},
            # stem
            {"id": "stem_conv", "type": "Conv2d", "params": {"out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1}},
            {"id": "stem_bn", "type": "BatchNorm2d", "params": {}},
            {"id": "stem_act", "type": "ReLU", "params": {}},
            # residual block 1 — identity skip (shape preserved)
            {"id": "b1_conv1", "type": "Conv2d", "params": {"out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1}},
            {"id": "b1_bn1", "type": "BatchNorm2d", "params": {}},
            {"id": "b1_act1", "type": "ReLU", "params": {}},
            {"id": "b1_conv2", "type": "Conv2d", "params": {"out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1}},
            {"id": "b1_bn2", "type": "BatchNorm2d", "params": {}},
            {"id": "b1_add", "type": "Add", "params": {}},
            {"id": "b1_act2", "type": "ReLU", "params": {}},
            # downsample
            {"id": "pool", "type": "MaxPool2d", "params": {"kernel_size": 2, "stride": 2}},
            # residual block 2 — projection skip (16 -> 32 channels via 1x1 conv)
            {"id": "b2_conv1", "type": "Conv2d", "params": {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1}},
            {"id": "b2_bn1", "type": "BatchNorm2d", "params": {}},
            {"id": "b2_act1", "type": "ReLU", "params": {}},
            {"id": "b2_conv2", "type": "Conv2d", "params": {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1}},
            {"id": "b2_bn2", "type": "BatchNorm2d", "params": {}},
            {"id": "b2_proj", "type": "Conv2d", "params": {"out_channels": 32, "kernel_size": 1, "stride": 1, "padding": 0}},
            {"id": "b2_add", "type": "Add", "params": {}},
            {"id": "b2_act2", "type": "ReLU", "params": {}},
            # head
            {"id": "gap", "type": "AdaptiveAvgPool2d", "params": {"output_size": [1, 1]}},
            {"id": "flat", "type": "Flatten", "params": {}},
            {"id": "fc", "type": "Linear", "params": {"out_features": 10, "bias": True}},
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [
            ["in", "stem_conv"], ["stem_conv", "stem_bn"], ["stem_bn", "stem_act"],
            # block 1 main path
            ["stem_act", "b1_conv1"], ["b1_conv1", "b1_bn1"], ["b1_bn1", "b1_act1"],
            ["b1_act1", "b1_conv2"], ["b1_conv2", "b1_bn2"],
            # block 1 skip: stem output is added back
            ["b1_bn2", "b1_add"], ["stem_act", "b1_add"], ["b1_add", "b1_act2"],
            ["b1_act2", "pool"],
            # block 2 main path
            ["pool", "b2_conv1"], ["b2_conv1", "b2_bn1"], ["b2_bn1", "b2_act1"],
            ["b2_act1", "b2_conv2"], ["b2_conv2", "b2_bn2"],
            # block 2 projection skip: 1x1 conv matches the channel count, then add
            ["pool", "b2_proj"], ["b2_bn2", "b2_add"], ["b2_proj", "b2_add"],
            ["b2_add", "b2_act2"],
            # head
            ["b2_act2", "gap"], ["gap", "flat"], ["flat", "fc"], ["fc", "out"],
        ],
        # 8 epochs: converges on current torch builds too (5 left it at ~0.70 on 2.6)
        "train": {"epochs": 8, "lr": 0.01, "batch_size": 64, "samples": 512, "seed": 0},
    }


# --------------------------------------------------------------------------- #
# GPT — tiny transformer: embedding + positional encoding + N encoder blocks
# The N blocks are a single Repeat (Stage 6): bump "repeat" to go deeper.
# --------------------------------------------------------------------------- #
def _gpt() -> Spec:
    return {
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [16]}},
            {"id": "emb", "type": "Embedding", "params": {"num_embeddings": 256, "embedding_dim": 64}},
            {"id": "pos", "type": "PositionalEncoding", "params": {"max_len": 512, "dropout": 0.0}},
            {"id": "blocks", "type": "Repeat", "params": {
                "repeat": 2,
                "layers": [
                    {"type": "TransformerEncoderLayer",
                     "params": {"nhead": 4, "dim_feedforward": 256, "dropout": 0.1, "activation": "gelu"}},
                ],
            }},
            {"id": "norm", "type": "LayerNorm", "params": {}},
            {"id": "flat", "type": "Flatten", "params": {}},
            {"id": "fc", "type": "Linear", "params": {"out_features": 10, "bias": True}},
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [
            ["in", "emb"], ["emb", "pos"], ["pos", "blocks"], ["blocks", "norm"],
            ["norm", "flat"], ["flat", "fc"], ["fc", "out"],
        ],
        "train": {"epochs": 5, "lr": 0.005, "batch_size": 64, "samples": 512, "seed": 0},
    }


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Arch:
    id: str
    name: str
    description: str
    category: str
    tags: tuple[str, ...]
    build: Callable[[], Spec]


_ARCHITECTURES: list[_Arch] = [
    _Arch("mlp", "MLP", "Fully-connected classifier (3 dense layers).", "basics",
          ("dense", "classifier", "tabular"), _mlp),
    _Arch("cnn", "Simple CNN", "Convolutional classifier: 2 conv blocks + dense head.", "vision",
          ("conv", "image", "classifier"), _cnn),
    _Arch("resnet", "ResNet", "Residual network with real Add skip connections (2 residual blocks).", "vision",
          ("conv", "residual", "skip", "image"), _resnet),
    _Arch("gpt", "GPT (tiny)", "Transformer: embedding + positional encoding + N encoder blocks (Repeat).", "sequence",
          ("transformer", "attention", "embedding", "repeat"), _gpt),
]

_BY_ID: dict[str, _Arch] = {a.id: a for a in _ARCHITECTURES}


def list_architectures() -> list[ArchitectureInfo]:
    """Summaries for the editor's architecture menu (``GET /architectures``)."""
    infos: list[ArchitectureInfo] = []
    for a in _ARCHITECTURES:
        spec = GraphSpec.model_validate(a.build())
        in_node = next((n for n in spec.nodes if n.type == "Input"), None)
        raw_shape = in_node.params.get("shape", []) if in_node else []
        in_shape = [int(d) for d in raw_shape] if isinstance(raw_shape, (list, tuple)) else []
        infos.append(
            ArchitectureInfo(
                id=a.id,
                name=a.name,
                description=a.description,
                category=a.category,
                tags=list(a.tags),
                input_shape=in_shape,
                node_count=len(spec.nodes),
            )
        )
    return infos


def get_architecture(arch_id: str) -> GraphSpec:
    """The full, editable graph for one architecture (``GET /architectures/{id}``)."""
    try:
        arch = _BY_ID[arch_id]
    except KeyError:
        known = ", ".join(sorted(_BY_ID)) or "(none)"
        raise KeyError(f"unknown architecture {arch_id!r}; known: {known}") from None
    return GraphSpec.model_validate(arch.build())
