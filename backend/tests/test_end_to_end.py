"""End-to-end M1 proof — no UI, no network, no dataset download.

Proves the whole backend pipeline from a hand-written graph spec:

* the layer catalogue is populated;
* a graph builds into a model with correct, auto-inferred shapes;
* training the model makes the loss drop (synthetic but learnable task);
* an invalid connection is rejected with a clear, node-pointed error;
* the Block mother-class repeats a group N times.

Run directly:  .venv\\Scripts\\python.exe tests\\test_end_to_end.py
Or via pytest: .venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ on path
# Stage 16: every /train keeps its weights as a run — in a throwaway folder during the suite,
# removed when the process ends (unless the caller chose the folder).
if "MLC_RUNS_DIR" not in os.environ:
    import atexit
    import shutil

    os.environ["MLC_RUNS_DIR"] = tempfile.mkdtemp(prefix="mlc-test-runs-")
    atexit.register(shutil.rmtree, os.environ["MLC_RUNS_DIR"], True)

import torch  # noqa: E402

import mlconstructor as mlc  # noqa: E402
from mlconstructor.adapters import apply_adapter, suggest_for_graph  # noqa: E402
from mlconstructor.architectures import get_architecture, list_architectures  # noqa: E402
from mlconstructor.blocks import Block  # noqa: E402
from mlconstructor.layers.activation import ReLUNode  # noqa: E402
from mlconstructor.layers.linear import LinearNode  # noqa: E402
from mlconstructor.schema import GraphSpec  # noqa: E402
from mlconstructor.shapes import ShapeError  # noqa: E402

CNN_MLP = {
    "nodes": [
        {"id": "in", "type": "Input", "params": {"shape": [1, 28, 28]}},
        {"id": "conv", "type": "Conv2d", "params": {"out_channels": 8, "kernel_size": 3}},
        {"id": "bn", "type": "BatchNorm2d", "params": {}},
        {"id": "act1", "type": "ReLU", "params": {}},
        {"id": "flat", "type": "Flatten", "params": {}},
        {"id": "fc1", "type": "Linear", "params": {"out_features": 32}},
        {"id": "act2", "type": "ReLU", "params": {}},
        {"id": "fc2", "type": "Linear", "params": {"out_features": 10}},
        {"id": "out", "type": "Output", "params": {}},
    ],
    "edges": [
        ["in", "conv"], ["conv", "bn"], ["bn", "act1"], ["act1", "flat"],
        ["flat", "fc1"], ["fc1", "act2"], ["act2", "fc2"], ["fc2", "out"],
    ],
    "train": {"epochs": 3, "lr": 0.01, "batch_size": 64, "samples": 512, "seed": 0},
}


def _pinned(graph: dict, src: str, dst: str, transform: str = "none", **params) -> dict:
    """``graph`` with the connection ``src -> dst`` pinned to ``transform`` (Stage 14).
    Pinned to "none" (Direct), a connection passes data through unchanged — the way
    every connection behaved before Stage 14 made them adapt shapes on their own."""
    settings = [s for s in graph.get("edge_settings", []) if (s["source"], s["target"]) != (src, dst)]
    settings.append({"source": src, "target": dst, "transform": transform, "params": params})
    return {**graph, "edge_settings": settings}


def test_layer_catalog():
    cat = mlc.catalog()
    types = {c.type for c in cat}
    assert {"Input", "Output", "Linear", "Conv2d", "ReLU", "BatchNorm2d", "Flatten"} <= types
    assert {"Add", "Concat"} <= types, "Stage 1 merge nodes should be in the catalogue"
    assert {
        "Dropout", "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d", "GELU", "LeakyReLU",
        "Softmax", "BatchNorm1d", "LayerNorm", "GroupNorm", "ConvTranspose2d",
    } <= types, "Stage 2 layers should be in the catalogue"
    assert {"Embedding", "RNN", "LSTM", "GRU"} <= types, "Stage 3 sequence layers should be in the catalogue"
    assert {"MultiheadAttention", "TransformerEncoderLayer", "PositionalEncoding"} <= types, (
        "Stage 4 attention layers should be in the catalogue"
    )
    assert "Repeat" in types, "Stage 6 block (Repeat) should be in the catalogue"
    print(f"[catalog] {len(cat)} layers: {sorted(types)}")


def test_build_summary_shapes():
    spec = GraphSpec.model_validate(CNN_MLP)
    _model, build = mlc.build_model(spec)
    assert build.ok and build.total_params > 0
    out = {s.id: s.out_shape for s in build.layers}
    assert out["conv"] == [8, 26, 26]        # (1,28,28), k=3, no pad -> (8,26,26)
    assert out["flat"] == [8 * 26 * 26]       # 5408
    assert out["fc2"] == [10]
    print(f"[build] total_params={build.total_params:,}  order={build.order}")


def test_train_loss_drops():
    spec = GraphSpec.model_validate(CNN_MLP)
    model, build = mlc.build_model(spec)
    in_shape = tuple(build.layers[0].out_shape)
    res = mlc.train_model(model, in_shape, spec.train)
    print(
        f"[train] steps={res.steps}  loss {res.loss_curve[0]:.3f} -> {res.final_loss:.3f}"
        f"  acc={res.final_accuracy:.3f}  device={res.device}"
    )
    assert res.steps > 0 and res.loss_curve
    assert res.final_loss < res.loss_curve[0], (res.final_loss, res.loss_curve[0])
    assert res.final_accuracy > 0.8


def test_shape_error_is_clear():
    bad = {
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [1, 28, 28]}},
            {"id": "fc", "type": "Linear", "params": {"out_features": 10}},  # no Flatten!
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [["in", "fc"], ["fc", "out"]],
    }
    # Pinned to "Direct", the connection cannot work: a clear error that points at it.
    try:
        mlc.build_model(GraphSpec.model_validate(_pinned(bad, "in", "fc")))
        raise AssertionError("expected a ShapeError for a Linear fed a 3-D feature map")
    except ShapeError as exc:
        assert "fc" in str(exc) and exc.edge == ("in", "fc") and exc.node_id == "fc" and exc.hint, exc
        print(f"[validate] caught clear ShapeError: {exc}")
    # Left on "auto" (the default since Stage 14), the connection flattens the image itself.
    _m, build = mlc.build_model(GraphSpec.model_validate(bad))
    edge = next(e for e in build.edges if (e.source, e.target) == ("in", "fc"))
    assert (edge.transform, edge.auto, edge.label, edge.out_shape) == ("flatten", True, "flatten", [784]), edge
    print("[validate] on auto the same connection flattens 1×28×28 -> 784 by itself")


def test_block_repeat():
    block = Block(lambda: [LinearNode("l", {"out_features": 16}), ReLUNode("r")], in_shape=(16,), repeat=3)
    assert len(block.layers) == 6             # 3 x (Linear + ReLU)
    y = block(torch.zeros(4, 16))
    assert tuple(y.shape) == (4, 16)
    print(f"[block] repeat=3 -> {len(block.layers)} modules, out_shape={block.out_shape}")


BRANCH_ADD = {
    "nodes": [
        {"id": "in", "type": "Input", "params": {"shape": [16]}},
        {"id": "fc1", "type": "Linear", "params": {"out_features": 32}},
        {"id": "pa", "type": "ReLU", "params": {}},
        {"id": "pb", "type": "Tanh", "params": {}},
        {"id": "sum", "type": "Add", "params": {}},
        {"id": "fc2", "type": "Linear", "params": {"out_features": 4}},
        {"id": "out", "type": "Output", "params": {}},
    ],
    "edges": [
        ["in", "fc1"],
        ["fc1", "pa"], ["fc1", "pb"],        # branch: fc1 fans out to two paths
        ["pa", "sum"], ["pb", "sum"],         # merge: Add combines the two paths
        ["sum", "fc2"], ["fc2", "out"],
    ],
    "train": {"epochs": 3, "lr": 0.01, "batch_size": 64, "samples": 512, "seed": 0},
}


def test_branch_add_builds_and_trains():
    spec = GraphSpec.model_validate(BRANCH_ADD)
    model, build = mlc.build_model(spec)
    out = {s.id: s.out_shape for s in build.layers}
    assert out["fc1"] == [32] and out["sum"] == [32] and out["fc2"] == [4], out
    in_shape = tuple(build.layers[0].out_shape)
    res = mlc.train_model(model, in_shape, spec.train)
    assert res.final_loss < res.loss_curve[0] and res.final_accuracy > 0.8
    print(f"[branch+Add] params={build.total_params} acc={res.final_accuracy:.3f} loss {res.loss_curve[0]:.3f}->{res.final_loss:.3f}")


def test_concat_merge():
    spec = GraphSpec.model_validate({
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [16]}},
            {"id": "l1", "type": "Linear", "params": {"out_features": 8}},
            {"id": "l2", "type": "Linear", "params": {"out_features": 8}},
            {"id": "cc", "type": "Concat", "params": {"dim": 0}},
            {"id": "fc", "type": "Linear", "params": {"out_features": 3}},
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [["in", "l1"], ["in", "l2"], ["l1", "cc"], ["l2", "cc"], ["cc", "fc"], ["fc", "out"]],
    })
    model, build = mlc.build_model(spec)
    out = {s.id: s.out_shape for s in build.layers}
    assert out["cc"] == [16], out            # concat of two (8,) along dim 0 -> (16,)
    y = model(torch.zeros(2, 16))
    assert tuple(y.shape) == (2, 3)
    print(f"[Concat] cc out_shape={out['cc']}, forward ok")


def test_merge_needs_two_inputs():
    spec = GraphSpec.model_validate({
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [8]}},
            {"id": "sum", "type": "Add", "params": {}},
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [["in", "sum"], ["sum", "out"]],
    })
    try:
        mlc.build_model(spec)
    except ShapeError as exc:
        assert "sum" in str(exc) and "only 1 is connected" in str(exc), exc
        assert exc.node_id == "sum" and "two branches" in exc.hint, (exc.node_id, exc.hint)
        print(f"[validate] Add with 1 input rejected: {exc}")
        return
    raise AssertionError("expected a ShapeError for a merge node with one input")


def test_layer_without_input_says_how_to_connect():
    spec = GraphSpec.model_validate({
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [8]}},
            {"id": "fc", "type": "Linear", "params": {"out_features": 3}},
            {"id": "lone", "type": "ReLU", "params": {}},
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [["in", "fc"], ["fc", "out"], ["lone", "out"]],
    })
    from mlconstructor.builder import plan_graph

    plan = plan_graph(spec, tolerant=True)
    lone = next(e for e in plan.errors if e.node_id == "lone")
    # a layer with nothing connected: say so and how to connect — not "use a merge node"
    assert "has no input" in str(lone) and "port" in lone.hint and "merge" not in str(lone), (str(lone), lone.hint)
    print(f"[validate] a layer with no input: {lone} → {lone.hint}")


def test_stage2_layers():
    spec = GraphSpec.model_validate({
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [1, 28, 28]}},
            {"id": "c", "type": "Conv2d", "params": {"out_channels": 8, "kernel_size": 3}},
            {"id": "bn", "type": "BatchNorm2d", "params": {}},
            {"id": "g", "type": "GELU", "params": {}},
            {"id": "mp", "type": "MaxPool2d", "params": {"kernel_size": 2, "stride": 2}},
            {"id": "gn", "type": "GroupNorm", "params": {"num_groups": 4}},
            {"id": "ap", "type": "AdaptiveAvgPool2d", "params": {"output_size": [4, 4]}},
            {"id": "fl", "type": "Flatten", "params": {}},
            {"id": "dp", "type": "Dropout", "params": {"p": 0.3}},
            {"id": "ln", "type": "LayerNorm", "params": {}},
            {"id": "fc", "type": "Linear", "params": {"out_features": 10}},
            {"id": "sm", "type": "Softmax", "params": {"dim": -1}},
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [
            ["in", "c"], ["c", "bn"], ["bn", "g"], ["g", "mp"], ["mp", "gn"],
            ["gn", "ap"], ["ap", "fl"], ["fl", "dp"], ["dp", "ln"], ["ln", "fc"],
            ["fc", "sm"], ["sm", "out"],
        ],
    })
    model, build = mlc.build_model(spec)
    out = {s.id: s.out_shape for s in build.layers}
    assert out["c"] == [8, 26, 26] and out["mp"] == [8, 13, 13] and out["ap"] == [8, 4, 4], out
    assert out["fl"] == [128] and out["fc"] == [10], out
    y = model(torch.zeros(2, 1, 28, 28))
    assert tuple(y.shape) == (2, 10)
    print(f"[stage2] {len(build.layers)} nodes, shapes ok, params={build.total_params:,}")


def test_conv_transpose_upsamples():
    spec = GraphSpec.model_validate({
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [4, 8, 8]}},
            {"id": "up", "type": "ConvTranspose2d", "params": {"out_channels": 2, "kernel_size": 2, "stride": 2}},
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [["in", "up"], ["up", "out"]],
    })
    _model, build = mlc.build_model(spec)
    out = {s.id: s.out_shape for s in build.layers}
    assert out["up"] == [2, 16, 16], out            # (8-1)*2 - 0 + 2 = 16
    print(f"[ConvTranspose2d] (4,8,8) -> {out['up']}")


# --------------------------------------------------------------------------- #
# Stage 3 — sequence models (Embedding, RNN, LSTM, GRU)
# Shapes are per-sample (L, F): the batched tensor is (B, L, F), batch_first.
# --------------------------------------------------------------------------- #
SEQ_LSTM = {
    "nodes": [
        {"id": "in", "type": "Input", "params": {"shape": [8, 4]}},       # (L=8, F=4)
        {"id": "lstm", "type": "LSTM", "params": {"hidden_size": 16}},     # last step -> (16,)
        {"id": "fc", "type": "Linear", "params": {"out_features": 3}},
        {"id": "out", "type": "Output", "params": {}},
    ],
    "edges": [["in", "lstm"], ["lstm", "fc"], ["fc", "out"]],
    "train": {"epochs": 4, "lr": 0.01, "batch_size": 64, "samples": 512, "seed": 0},
}


def test_lstm_sequence_trains():
    spec = GraphSpec.model_validate(SEQ_LSTM)
    model, build = mlc.build_model(spec)
    out = {s.id: s.out_shape for s in build.layers}
    assert out["lstm"] == [16] and out["fc"] == [3], out
    in_shape = tuple(build.layers[0].out_shape)
    res = mlc.train_model(model, in_shape, spec.train)
    assert res.final_loss < res.loss_curve[0] and res.final_accuracy > 0.8
    print(f"[LSTM] (8,4)->last(16)->3  acc={res.final_accuracy:.3f} loss {res.loss_curve[0]:.3f}->{res.final_loss:.3f}")


def test_embedding_lstm_classifier_trains():
    spec = GraphSpec.model_validate({
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [6]}},                              # (L=6,) token ids
            {"id": "emb", "type": "Embedding", "params": {"num_embeddings": 20, "embedding_dim": 8}},
            {"id": "lstm", "type": "LSTM", "params": {"hidden_size": 16}},
            {"id": "fc", "type": "Linear", "params": {"out_features": 4}},
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [["in", "emb"], ["emb", "lstm"], ["lstm", "fc"], ["fc", "out"]],
        "train": {"epochs": 6, "lr": 0.01, "batch_size": 64, "samples": 512, "seed": 0},
    })
    model, build = mlc.build_model(spec, seed=0)  # fixed starting weights: not hostage to the tests before it
    out = {s.id: s.out_shape for s in build.layers}
    assert out["emb"] == [6, 8] and out["lstm"] == [16] and out["fc"] == [4], out
    in_shape = tuple(build.layers[0].out_shape)
    res = mlc.train_model(model, in_shape, spec.train)
    assert res.final_loss < res.loss_curve[0] and res.final_accuracy > 0.8
    print(f"[Embedding+LSTM] (6,)->emb(6,8)->lstm(16)->4  acc={res.final_accuracy:.3f} loss {res.loss_curve[0]:.3f}->{res.final_loss:.3f}")


def test_gru_return_sequence_then_flatten():
    spec = GraphSpec.model_validate({
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [5, 3]}},
            {"id": "gru", "type": "GRU", "params": {"hidden_size": 7, "return_sequence": True}},   # -> (5,7)
            {"id": "flat", "type": "Flatten", "params": {}},                                        # -> (35,)
            {"id": "fc", "type": "Linear", "params": {"out_features": 2}},
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [["in", "gru"], ["gru", "flat"], ["flat", "fc"], ["fc", "out"]],
    })
    model, build = mlc.build_model(spec)
    out = {s.id: s.out_shape for s in build.layers}
    assert out["gru"] == [5, 7] and out["flat"] == [35] and out["fc"] == [2], out
    y = model(torch.zeros(2, 5, 3))
    assert tuple(y.shape) == (2, 2)
    print("[GRU seq] (5,3)->gru(5,7)->flat(35)->2, forward ok")


def test_rnn_relu_and_bad_nonlinearity():
    ok = GraphSpec.model_validate({
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [4, 3]}},
            {"id": "rnn", "type": "RNN", "params": {"hidden_size": 5, "nonlinearity": "relu"}},
            {"id": "fc", "type": "Linear", "params": {"out_features": 2}},
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [["in", "rnn"], ["rnn", "fc"], ["fc", "out"]],
    })
    model, build = mlc.build_model(ok)
    assert {s.id: s.out_shape for s in build.layers}["rnn"] == [5]
    assert tuple(model(torch.zeros(2, 4, 3)).shape) == (2, 2)

    bad = GraphSpec.model_validate({
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [4, 3]}},
            {"id": "rnn", "type": "RNN", "params": {"nonlinearity": "swish"}},
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [["in", "rnn"], ["rnn", "out"]],
    })
    try:
        mlc.build_model(bad)
    except ShapeError as exc:
        assert "rnn" in str(exc)
        print(f"[validate] bad RNN nonlinearity rejected: {exc}")
        return
    raise AssertionError("expected a ShapeError for an unknown RNN nonlinearity")


def test_embedding_needs_1d_sequence():
    spec = GraphSpec.model_validate({
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [3, 4]}},    # 2-D, not a (L,) index sequence
            {"id": "emb", "type": "Embedding", "params": {"num_embeddings": 10}},
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [["in", "emb"], ["emb", "out"]],
    })
    try:
        mlc.build_model(GraphSpec.model_validate(_pinned(spec.model_dump(), "in", "emb")))
        raise AssertionError("expected a ShapeError for an Embedding fed a 2-D shape")
    except ShapeError as exc:
        assert "emb" in str(exc) and exc.edge == ("in", "emb"), exc
        print(f"[validate] Embedding fed 2-D rejected: {exc}")
    _m, build = mlc.build_model(spec)  # auto: the 3×4 grid is read as 12 token ids
    assert build.edges[0].transform == "flatten" and build.edges[0].out_shape == [12], build.edges[0]


# --------------------------------------------------------------------------- #
# Stage 4 — attention & transformers (self-attention + encoder block + pos-enc)
# All shape-preserving over the (L, E) sequence convention; a Flatten + Linear
# turns the sequence into class logits.
# --------------------------------------------------------------------------- #
def test_multihead_attention_shape():
    spec = GraphSpec.model_validate({
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [6, 8]}},
            {"id": "att", "type": "MultiheadAttention", "params": {"num_heads": 2}},   # -> (6,8)
            {"id": "flat", "type": "Flatten", "params": {}},                            # -> (48,)
            {"id": "fc", "type": "Linear", "params": {"out_features": 3}},
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [["in", "att"], ["att", "flat"], ["flat", "fc"], ["fc", "out"]],
    })
    model, build = mlc.build_model(spec)
    out = {s.id: s.out_shape for s in build.layers}
    assert out["att"] == [6, 8] and out["flat"] == [48] and out["fc"] == [3], out
    y = model(torch.zeros(2, 6, 8))
    assert tuple(y.shape) == (2, 3)
    print("[MHA] (6,8) self-attn -> (6,8) -> flat(48) -> 3, forward ok")


def test_transformer_encoder_trains():
    spec = GraphSpec.model_validate({
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [5, 8]}},
            {"id": "pe", "type": "PositionalEncoding", "params": {}},
            {"id": "enc", "type": "TransformerEncoderLayer",
             "params": {"nhead": 2, "dim_feedforward": 16, "dropout": 0.0}},
            {"id": "flat", "type": "Flatten", "params": {}},
            {"id": "fc", "type": "Linear", "params": {"out_features": 3}},
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [["in", "pe"], ["pe", "enc"], ["enc", "flat"], ["flat", "fc"], ["fc", "out"]],
        "train": {"epochs": 8, "lr": 0.01, "batch_size": 64, "samples": 512, "seed": 0},
    })
    model, build = mlc.build_model(spec)
    out = {s.id: s.out_shape for s in build.layers}
    assert out["pe"] == [5, 8] and out["enc"] == [5, 8] and out["flat"] == [40], out
    in_shape = tuple(build.layers[0].out_shape)
    res = mlc.train_model(model, in_shape, spec.train)
    assert res.final_loss < res.loss_curve[0] and res.final_accuracy > 0.8
    print(f"[Transformer] (5,8)->PE->encoder->flat->3  acc={res.final_accuracy:.3f} loss {res.loss_curve[0]:.3f}->{res.final_loss:.3f}")


def test_embedding_transformer_classifier_builds():
    spec = GraphSpec.model_validate({
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [6]}},                                 # (L=6,) token ids
            {"id": "emb", "type": "Embedding", "params": {"num_embeddings": 20, "embedding_dim": 8}},
            {"id": "pe", "type": "PositionalEncoding", "params": {}},
            {"id": "enc", "type": "TransformerEncoderLayer",
             "params": {"nhead": 2, "dim_feedforward": 16, "dropout": 0.0}},
            {"id": "flat", "type": "Flatten", "params": {}},
            {"id": "fc", "type": "Linear", "params": {"out_features": 4}},
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [["in", "emb"], ["emb", "pe"], ["pe", "enc"], ["enc", "flat"], ["flat", "fc"], ["fc", "out"]],
    })
    model, build = mlc.build_model(spec)
    out = {s.id: s.out_shape for s in build.layers}
    assert out["emb"] == [6, 8] and out["enc"] == [6, 8] and out["fc"] == [4], out
    y = model(torch.zeros(2, 6))
    assert tuple(y.shape) == (2, 4)
    print("[Embedding+Transformer] (6,)->emb->PE->encoder->flat->4, forward ok")


def test_attention_head_divisibility():
    spec = GraphSpec.model_validate({
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [6, 8]}},
            {"id": "att", "type": "MultiheadAttention", "params": {"num_heads": 3}},   # 8 % 3 != 0
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [["in", "att"], ["att", "out"]],
    })
    try:
        mlc.build_model(spec)
    except ShapeError as exc:
        assert "att" in str(exc) and "num_heads" in str(exc)
        print(f"[validate] embed_dim indivisible by heads rejected: {exc}")
        return
    raise AssertionError("expected a ShapeError for embed_dim not divisible by num_heads")


# --------------------------------------------------------------------------- #
# Stage 5 — smart adapters: a mismatched connection is proposed a fix (Flatten),
# and applying the proposal makes the graph build.
# --------------------------------------------------------------------------- #
CONV_NO_FLATTEN = {
    "nodes": [
        {"id": "in", "type": "Input", "params": {"shape": [1, 28, 28]}},
        {"id": "conv", "type": "Conv2d", "params": {"out_channels": 4, "kernel_size": 3}},
        {"id": "fc", "type": "Linear", "params": {"out_features": 10}},    # fed a (4,26,26) map, no Flatten
        {"id": "out", "type": "Output", "params": {}},
    ],
    "edges": [["in", "conv"], ["conv", "fc"], ["fc", "out"]],
}


def test_suggest_flatten_for_conv_to_linear():
    # since Stage 14 an "auto" connection flattens by itself; a Direct one still needs the adapter node
    spec = GraphSpec.model_validate(_pinned(CONV_NO_FLATTEN, "conv", "fc"))
    props = suggest_for_graph(spec)
    assert len(props) == 1, props
    p = props[0]
    assert p.adapter_type == "Flatten" and p.source_id == "conv" and p.target_id == "fc"
    assert p.in_shape == [4, 26, 26] and p.out_shape == [4 * 26 * 26]
    print(f"[suggest] {p.reason}")


def test_apply_adapter_makes_graph_build():
    spec = GraphSpec.model_validate(_pinned(CONV_NO_FLATTEN, "conv", "fc"))
    try:
        mlc.build_model(spec)
        raise AssertionError("expected the un-flattened Conv->Linear graph to fail to build")
    except ShapeError:
        pass
    fixed = apply_adapter(spec, suggest_for_graph(spec)[0])
    _model, build = mlc.build_model(fixed)
    flats = [s for s in build.layers if s.type == "Flatten"]
    assert len(flats) == 1 and flats[0].out_shape == [4 * 26 * 26], build.order
    # the inserted Flatten sits on the conv->fc edge
    assert build.order.index("conv") < build.order.index(flats[0].id) < build.order.index("fc")
    assert fixed.edge_settings == [], fixed.edge_settings  # the replaced edge's Direct pin went with it
    print(f"[apply] inserted {flats[0].id} -> graph builds, params={build.total_params:,}")


def test_suggest_empty_for_valid_graph():
    assert suggest_for_graph(GraphSpec.model_validate(CNN_MLP)) == []
    print("[suggest] valid graph -> no proposals")


def test_suggest_flatten_for_sequence_to_linear():
    spec = GraphSpec.model_validate({
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [5, 4]}},
            {"id": "lstm", "type": "LSTM", "params": {"hidden_size": 6, "return_sequence": True}},  # (5,6)
            {"id": "fc", "type": "Linear", "params": {"out_features": 3}},
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [["in", "lstm"], ["lstm", "fc"], ["fc", "out"]],
    })
    assert suggest_for_graph(spec) == []  # auto (the default) adapts the sequence by itself
    spec = GraphSpec.model_validate(_pinned(spec.model_dump(), "lstm", "fc"))
    props = suggest_for_graph(spec)
    assert len(props) == 1 and props[0].adapter_type == "Flatten"
    assert props[0].in_shape == [5, 6] and props[0].out_shape == [30]
    print(f"[suggest] seq->Linear: {props[0].reason}")


# --------------------------------------------------------------------------- #
# Stage 6 — reusable blocks: a Repeat node stamps an inner linear chain N times
# (reusing the Block mother class), behaving like any single-input layer.
# --------------------------------------------------------------------------- #
REPEAT_MLP = {
    "nodes": [
        {"id": "in", "type": "Input", "params": {"shape": [16]}},
        {"id": "blk", "type": "Repeat", "params": {"repeat": 3, "layers": [
            {"type": "Linear", "params": {"out_features": 16}},
            {"type": "ReLU", "params": {}},
        ]}},
        {"id": "fc", "type": "Linear", "params": {"out_features": 4}},
        {"id": "out", "type": "Output", "params": {}},
    ],
    "edges": [["in", "blk"], ["blk", "fc"], ["fc", "out"]],
    "train": {"epochs": 4, "lr": 0.01, "batch_size": 64, "samples": 512, "seed": 0},
}


def test_repeat_block_builds_and_trains():
    spec = GraphSpec.model_validate(REPEAT_MLP)
    model, build = mlc.build_model(spec)
    out = {s.id: s.out_shape for s in build.layers}
    assert out["blk"] == [16] and out["fc"] == [4], out
    blk = next(s for s in build.layers if s.id == "blk")
    assert blk.num_params == 3 * (16 * 16 + 16), blk.num_params  # 3 stamps of Linear(16->16)+bias
    in_shape = tuple(build.layers[0].out_shape)
    res = mlc.train_model(model, in_shape, spec.train)
    assert res.final_loss < res.loss_curve[0] and res.final_accuracy > 0.8
    print(f"[Repeat] [Linear,ReLU]x3 block params={blk.num_params} acc={res.final_accuracy:.3f}")


def test_repeat_block_shape_changes_each_stamp():
    spec = GraphSpec.model_validate({
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [16]}},
            {"id": "blk", "type": "Repeat", "params": {"repeat": 2, "layers": [
                {"type": "Linear", "params": {"out_features": 8}},
                {"type": "ReLU", "params": {}},
            ]}},
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [["in", "blk"], ["blk", "out"]],
    })
    _model, build = mlc.build_model(spec)
    out = {s.id: s.out_shape for s in build.layers}
    assert out["blk"] == [8], out            # (16)->L8->(8) then (8)->L8->(8)
    blk = next(s for s in build.layers if s.id == "blk")
    assert blk.num_params == (16 * 8 + 8) + (8 * 8 + 8), blk.num_params
    print(f"[Repeat] shape-changing block (16->8)x2 params={blk.num_params}, out={out['blk']}")


def test_empty_block_rejected():
    spec = GraphSpec.model_validate({
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [8]}},
            {"id": "blk", "type": "Repeat", "params": {"repeat": 2, "layers": []}},
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [["in", "blk"], ["blk", "out"]],
    })
    try:
        mlc.build_model(spec)
    except ShapeError as exc:
        assert "blk" in str(exc) and "empty" in str(exc)
        print(f"[validate] empty block rejected: {exc}")
        return
    raise AssertionError("expected an empty block to be rejected")


def test_block_rejects_inner_io_node():
    spec = GraphSpec.model_validate({
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [8]}},
            {"id": "blk", "type": "Repeat", "params": {"repeat": 1, "layers": [
                {"type": "Output", "params": {}},
            ]}},
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [["in", "blk"], ["blk", "out"]],
    })
    try:
        mlc.build_model(spec)
    except ShapeError as exc:
        assert "blk" in str(exc)
        print(f"[validate] inner Output rejected: {exc}")
        return
    raise AssertionError("expected an inner Output to be rejected")


# --------------------------------------------------------------------------- #
# Stage 7 — architecture library: ready public architectures (MLP, CNN, ResNet,
# GPT) drop onto the canvas as full, EDITABLE graphs — ordinary nodes, real
# Add skip-merges, a Repeat-powered transformer stack — and each one builds and
# trains end to end.
# --------------------------------------------------------------------------- #
ARCH_IDS = ("mlp", "cnn", "resnet", "gpt")


def test_architecture_catalog():
    lib = list_architectures()
    ids = {a.id for a in lib}
    assert set(ARCH_IDS) <= ids, ids
    for a in lib:
        assert a.name and a.description and a.node_count > 0 and a.input_shape
    print(f"[arch] library: {[(a.id, a.node_count) for a in lib]}")


def test_architecture_nodes_are_editable_layers():
    # Stage 7's promise: a ready architecture is NOT a black box — every node is
    # an ordinary registered layer (and so are the inner layers of a Repeat block),
    # so the user can select and deeply edit each one.
    for aid in ARCH_IDS:
        spec = get_architecture(aid)
        for n in spec.nodes:
            assert n.type in mlc.LAYER_REGISTRY, (aid, n.id, n.type)
            if n.type == "Repeat":
                for inner in n.params.get("layers", []):
                    assert inner["type"] in mlc.LAYER_REGISTRY, (aid, inner)
    print("[arch] every node is an editable registered layer")


def test_architectures_build_with_expected_shapes():
    _m, b = mlc.build_model(get_architecture("mlp"))
    assert {s.id: s.out_shape for s in b.layers}["fc3"] == [10]

    _m, b = mlc.build_model(get_architecture("cnn"))
    out = {s.id: s.out_shape for s in b.layers}
    assert out["flat"] == [32 * 7 * 7] and out["fc2"] == [10], out

    model, b = mlc.build_model(get_architecture("resnet"))
    out = {s.id: s.out_shape for s in b.layers}
    assert out["b1_add"] == [16, 28, 28] and out["b2_add"] == [32, 14, 14], out
    assert out["gap"] == [32, 1, 1] and out["flat"] == [32] and out["fc"] == [10], out
    assert tuple(model(torch.zeros(2, 1, 28, 28)).shape) == (2, 10)   # Add skip-merges run

    model, b = mlc.build_model(get_architecture("gpt"))
    out = {s.id: s.out_shape for s in b.layers}
    assert out["emb"] == [16, 64] and out["blocks"] == [16, 64], out
    assert out["flat"] == [16 * 64] and out["fc"] == [10], out
    assert tuple(model(torch.zeros(2, 16)).shape) == (2, 10)
    print("[arch] MLP/CNN/ResNet/GPT build with the expected shapes")


def test_each_architecture_trains():
    for aid in ARCH_IDS:
        spec = get_architecture(aid)
        model, build = mlc.build_model(spec)
        in_shape = tuple(build.layers[0].out_shape)
        res = mlc.train_model(model, in_shape, spec.train)
        assert res.steps > 0 and res.loss_curve
        assert res.final_loss < res.loss_curve[0], (aid, res.final_loss, res.loss_curve[0])
        assert res.final_accuracy > 0.5, (aid, res.final_accuracy)
        print(
            f"[arch:{aid}] params={build.total_params:,} acc={res.final_accuracy:.3f} "
            f"loss {res.loss_curve[0]:.3f}->{res.final_loss:.3f}"
        )


# --------------------------------------------------------------------------- #
# Stage 8 — datasets, data preview, the accuracy curve, and standalone PyTorch
# export. The headline proof: a graph built in the app exports to a .py that
# depends on nothing but torch and rebuilds to the same model (params + output).
# --------------------------------------------------------------------------- #
def test_dataset_catalog():
    ds = mlc.list_datasets()
    kinds = {d.id: d.kind for d in ds}
    assert {"blobs", "shells"} <= set(kinds), kinds
    assert kinds["blobs"] == "synthetic" and kinds["shells"] == "synthetic"
    for d in ds:
        assert d.name and d.description and d.kind
    print(f"[datasets] {[d.id for d in ds]}")


def test_datasets_generate_valid_tensors():
    cfg = GraphSpec.model_validate(CNN_MLP).train
    dev = torch.device("cpu")
    for name in ("blobs", "shells", "synthetic"):
        for in_shape in [(8,), (1, 12, 12)]:
            X, y = mlc.make_dataset(name, in_shape, 5, cfg, dev)
            assert tuple(X.shape) == (cfg.samples, *in_shape), (name, in_shape, X.shape)
            assert tuple(y.shape) == (cfg.samples,) and 0 <= int(y.min()) and int(y.max()) < 5
            X2, y2 = mlc.make_dataset(name, in_shape, 5, cfg, dev)
            assert torch.equal(X, X2) and torch.equal(y, y2), name  # deterministic per seed
    a = mlc.make_dataset("synthetic", (8,), 5, cfg, dev)
    b = mlc.make_dataset("blobs", (8,), 5, cfg, dev)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])  # 'synthetic' aliases blobs
    print("[datasets] blobs/shells valid + deterministic; 'synthetic' == 'blobs'")


def test_trainer_reports_accuracy_curve():
    spec = GraphSpec.model_validate(CNN_MLP)
    model, build = mlc.build_model(spec)
    in_shape = tuple(build.layers[0].out_shape)
    res = mlc.train_model(model, in_shape, spec.train)
    assert len(res.acc_curve) == spec.train.epochs, (len(res.acc_curve), spec.train.epochs)
    assert all(0.0 <= a <= 1.0 for a in res.acc_curve)
    assert abs(res.acc_curve[-1] - res.final_accuracy) < 1e-6  # last epoch == final state
    print(f"[curve] acc per epoch: {[round(a, 3) for a in res.acc_curve]}")


def test_shells_dataset_trains():
    spec = GraphSpec.model_validate({
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [8]}},
            {"id": "fc1", "type": "Linear", "params": {"out_features": 32}},
            {"id": "a1", "type": "ReLU", "params": {}},
            {"id": "fc2", "type": "Linear", "params": {"out_features": 16}},
            {"id": "a2", "type": "ReLU", "params": {}},
            {"id": "fc3", "type": "Linear", "params": {"out_features": 4}},
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [["in", "fc1"], ["fc1", "a1"], ["a1", "fc2"], ["fc2", "a2"], ["a2", "fc3"], ["fc3", "out"]],
        "train": {"dataset": "shells", "epochs": 25, "lr": 0.01, "batch_size": 64, "samples": 512, "seed": 0},
    })
    model, build = mlc.build_model(spec)
    in_shape = tuple(build.layers[0].out_shape)
    res = mlc.train_model(model, in_shape, spec.train)
    assert res.final_loss < res.loss_curve[0]
    assert res.final_accuracy > 0.6, res.final_accuracy  # non-linear radial task, needs a hidden layer
    print(f"[shells] radial task acc={res.final_accuracy:.3f} loss {res.loss_curve[0]:.3f}->{res.final_loss:.3f}")


def test_data_preview_is_the_training_data():
    spec = GraphSpec.model_validate(CNN_MLP)
    prev = mlc.data_preview(spec, rows=5)
    assert prev.input_shape == [1, 28, 28] and prev.feature_count == 784
    assert prev.num_classes == 10 and prev.dataset == "blobs" and prev.num_samples == spec.train.samples
    assert len(prev.rows) == 5
    for r in prev.rows:
        assert 0 <= r.label < 10 and r.total == 784 and r.truncated and len(r.values) == 12
    # the preview rows are literally the first rows of the real training set
    X, y = mlc.make_dataset(spec.train.dataset, (1, 28, 28), 10, spec.train, torch.device("cpu"))
    assert prev.rows[0].label == int(y[0])
    for a, b in zip(prev.rows[0].values, X[0].reshape(-1)[:12].tolist()):
        assert abs(a - round(float(b), 3)) < 1e-6
    print(f"[preview] {prev.dataset_name}: {prev.input_shape} x{prev.num_classes}, rows == training data")


def _exec_exported(code: str) -> dict:
    ns: dict = {"__name__": "exported_model"}
    exec(compile(code, "<exported_model>", "exec"), ns)
    return ns


CONCAT_GRAPH = {
    "nodes": [
        {"id": "in", "type": "Input", "params": {"shape": [16]}},
        {"id": "l1", "type": "Linear", "params": {"out_features": 8}},
        {"id": "l2", "type": "Linear", "params": {"out_features": 8}},
        {"id": "cc", "type": "Concat", "params": {"dim": 0}},
        {"id": "fc", "type": "Linear", "params": {"out_features": 3}},
        {"id": "out", "type": "Output", "params": {}},
    ],
    "edges": [["in", "l1"], ["in", "l2"], ["l1", "cc"], ["l2", "cc"], ["cc", "fc"], ["fc", "out"]],
}

TRANSFORMER_GRAPH = {
    "nodes": [
        {"id": "in", "type": "Input", "params": {"shape": [5, 8]}},
        {"id": "pe", "type": "PositionalEncoding", "params": {}},
        {"id": "enc", "type": "TransformerEncoderLayer", "params": {"nhead": 2, "dim_feedforward": 16, "dropout": 0.0}},
        {"id": "flat", "type": "Flatten", "params": {}},
        {"id": "fc", "type": "Linear", "params": {"out_features": 3}},
        {"id": "out", "type": "Output", "params": {}},
    ],
    "edges": [["in", "pe"], ["pe", "enc"], ["enc", "flat"], ["flat", "fc"], ["fc", "out"]],
}

MHA_GRAPH = {
    "nodes": [
        {"id": "in", "type": "Input", "params": {"shape": [6, 8]}},
        {"id": "att", "type": "MultiheadAttention", "params": {"num_heads": 2}},
        {"id": "flat", "type": "Flatten", "params": {}},
        {"id": "fc", "type": "Linear", "params": {"out_features": 3}},
        {"id": "out", "type": "Output", "params": {}},
    ],
    "edges": [["in", "att"], ["att", "flat"], ["flat", "fc"], ["fc", "out"]],
}


def test_export_is_standalone_and_equivalent():
    graphs = {aid: get_architecture(aid) for aid in ARCH_IDS}
    graphs["branch_add"] = GraphSpec.model_validate(BRANCH_ADD)
    graphs["concat"] = GraphSpec.model_validate(CONCAT_GRAPH)
    graphs["lstm"] = GraphSpec.model_validate(SEQ_LSTM)
    graphs["transformer"] = GraphSpec.model_validate(TRANSFORMER_GRAPH)
    graphs["mha"] = GraphSpec.model_validate(MHA_GRAPH)
    for name, spec in graphs.items():
        model, build = mlc.build_model(spec)
        in_shape = tuple(build.layers[0].out_shape)
        ncls = int(model(torch.zeros(2, *in_shape)).shape[1])
        code = mlc.generate_pytorch(spec, include_training=False)
        assert "mlconstructor" not in code, name  # truly standalone
        gen = _exec_exported(code)["Model"]()
        gp = sum(p.numel() for p in gen.parameters())
        assert gp == build.total_params, (name, gp, build.total_params)  # same parameters
        y = gen(torch.zeros(2, *in_shape))
        assert tuple(y.shape) == (2, ncls), (name, tuple(y.shape))  # same output
        print(f"[export:{name}] standalone Model: params={gp:,} out={tuple(y.shape)}")


def test_export_file_trains_end_to_end():
    code = mlc.generate_pytorch(get_architecture("mlp"), include_training=True)
    assert "import mlconstructor" not in code and "def make_data" in code and "def main" in code
    _exec_exported(code)["main"]()  # runs the generated training loop on synthetic data, start to finish
    print("[export] generated MLP file trains end to end")


# --------------------------------------------------------------------------- #
# Stage 9 — bring your own data (the "custom" dataset) + state_dict / ONNX export.
# --------------------------------------------------------------------------- #
from mlconstructor.schema import TrainSpec  # noqa: E402

CUSTOM_MLP_NODES = [
    {"id": "in", "type": "Input", "params": {"shape": [4]}},
    {"id": "fc1", "type": "Linear", "params": {"out_features": 16}},
    {"id": "a1", "type": "ReLU", "params": {}},
    {"id": "fc2", "type": "Linear", "params": {"out_features": 3}},
    {"id": "out", "type": "Output", "params": {}},
]
CUSTOM_MLP_EDGES = [["in", "fc1"], ["fc1", "a1"], ["a1", "fc2"], ["fc2", "out"]]
# 12 linearly separable rows in 4-D, 3 classes (features..., label):
CUSTOM_ROWS = [
    [2.0, 0.0, 0.0, 0.0, 0], [2.0, 0.1, 0.0, 0.0, 0], [1.8, 0.0, 0.1, 0.0, 0], [2.1, 0.0, 0.0, 0.1, 0],
    [0.0, 2.0, 0.0, 0.0, 1], [0.1, 2.0, 0.0, 0.0, 1], [0.0, 1.8, 0.1, 0.0, 1], [0.0, 2.1, 0.0, 0.0, 1],
    [0.0, 0.0, 2.0, 0.0, 2], [0.0, 0.1, 2.0, 0.0, 2], [0.0, 0.0, 1.8, 0.1, 2], [0.0, 0.0, 2.1, 0.0, 2],
]


def _custom_graph(rows, **train):
    return GraphSpec.model_validate({
        "nodes": CUSTOM_MLP_NODES,
        "edges": CUSTOM_MLP_EDGES,
        "train": {"dataset": "custom", "custom_rows": rows, **train},
    })


def test_custom_dataset_in_catalog():
    assert "custom" in {d.id for d in mlc.list_datasets()}
    print("[custom] dataset in catalogue")


def test_custom_dataset_trains_on_your_rows():
    spec = _custom_graph(CUSTOM_ROWS, epochs=60, lr=0.05, batch_size=64, seed=0)
    model, build = mlc.build_model(spec)
    in_shape = tuple(build.layers[0].out_shape)
    res = mlc.train_model(model, in_shape, spec.train)
    assert res.steps == 60, res.steps  # 12 rows, one batch per epoch -> uses EXACTLY our data
    assert res.final_loss < res.loss_curve[0] and res.final_accuracy > 0.8
    print(f"[custom] trained on 12 pasted rows: acc={res.final_accuracy:.3f} loss {res.loss_curve[0]:.3f}->{res.final_loss:.3f}")


def test_custom_dataset_validation():
    dev = torch.device("cpu")
    # wrong row length (4 features expected + 1 label = 5)
    try:
        mlc.make_dataset("custom", (4,), 3, _custom_graph([[1.0, 2.0, 3.0, 0]]).train, dev)
        raise AssertionError("expected a length error")
    except ValueError as exc:
        assert "expected 4 feature" in str(exc), exc
    # label out of range for a 3-class network
    try:
        mlc.make_dataset("custom", (4,), 3, _custom_graph([[0.0, 0.0, 0.0, 0.0, 5]]).train, dev)
        raise AssertionError("expected an out-of-range error")
    except ValueError as exc:
        assert "out of range" in str(exc), exc
    # no rows at all
    try:
        mlc.make_dataset("custom", (4,), 3, _custom_graph([]).train, dev)
        raise AssertionError("expected a needs-rows error")
    except ValueError as exc:
        assert "needs rows" in str(exc), exc
    # non-finite value
    try:
        bad = TrainSpec(dataset="custom", custom_rows=[[0.0, 0.0, 0.0, float("nan"), 0]])
        mlc.make_dataset("custom", (4,), 3, bad, dev)
        raise AssertionError("expected a non-finite error")
    except ValueError as exc:
        assert "non-finite" in str(exc), exc
    print("[custom] bad rows rejected with clear messages")


def test_custom_data_preview_echoes_your_rows():
    spec = _custom_graph(CUSTOM_ROWS)
    prev = mlc.data_preview(spec, rows=4)
    assert prev.dataset == "custom" and prev.input_shape == [4] and prev.num_classes == 3
    assert len(prev.rows) == 4 and prev.rows[0].total == 4
    for a, b in zip(prev.rows[0].values, CUSTOM_ROWS[0][:4]):
        assert abs(a - b) < 1e-6
    assert prev.rows[0].label == CUSTOM_ROWS[0][4]
    print("[custom] preview echoes the pasted rows")


def test_export_custom_embeds_your_data():
    spec = _custom_graph(CUSTOM_ROWS, epochs=5, lr=0.05)
    code = mlc.generate_pytorch(spec, include_training=True)
    assert "CUSTOM_ROWS = " in code and "def make_data" in code and "mlconstructor" not in code
    model = _exec_exported(code)["main"]()  # trains on the embedded rows
    _m, build = mlc.build_model(spec)
    assert sum(p.numel() for p in model.parameters()) == build.total_params
    print("[export:custom] embeds your rows and trains end to end")


def test_export_saves_state_dict_and_onnx():
    import os
    import shutil
    import tempfile

    code = mlc.generate_pytorch(get_architecture("mlp"), include_training=True)
    assert "def export_model" in code and "torch.save" in code and "torch.onnx.export" in code
    ns = _exec_exported(code)
    d = tempfile.mkdtemp()
    cwd = os.getcwd()
    try:
        os.chdir(d)
        ns["export_model"](ns["main"]())
        assert os.path.exists("model.pt"), "state_dict should be saved"
        onnx_ok = os.path.exists("model.onnx")
    finally:
        os.chdir(cwd)
        shutil.rmtree(d, ignore_errors=True)
    print(f"[export] state_dict saved; onnx={'yes' if onnx_ok else 'skipped (best-effort)'}")


# --------------------------------------------------------------------------- #
# Stage 10 — built-in real datasets (MNIST / CIFAR-10). These offline checks cover
# the catalogue and the fixed-shape validation (which runs BEFORE any download).
# The actual download + train is verified separately (scratchpad/verify_real.py),
# so the always-run suite stays offline and fast.
# --------------------------------------------------------------------------- #
def test_real_datasets_in_catalog():
    by = {d.id: d for d in mlc.list_datasets()}
    assert "mnist" in by and "cifar10" in by, list(by)
    assert by["mnist"].kind == "real" and by["mnist"].input_shape == [1, 28, 28]
    assert by["cifar10"].kind == "real" and by["cifar10"].input_shape == [3, 32, 32]
    assert by["custom"].kind == "custom"
    assert by["blobs"].input_shape is None  # adaptive datasets advertise no fixed shape
    print(f"[real] catalogue: mnist{by['mnist'].input_shape} cifar10{by['cifar10'].input_shape}")


def test_real_dataset_shape_validation():
    dev = torch.device("cpu")
    cfg = TrainSpec(dataset="mnist", samples=16, seed=0)
    try:  # wrong input shape -> clear error, WITHOUT downloading (validated first)
        mlc.make_dataset("mnist", (4,), 10, cfg, dev)
        raise AssertionError("expected a shape error")
    except ValueError as exc:
        assert "Input of shape [1, 28, 28]" in str(exc), exc
    try:  # too few output classes -> clear error, also before any download
        mlc.make_dataset("mnist", (1, 28, 28), 3, cfg, dev)
        raise AssertionError("expected a class-count error")
    except ValueError as exc:
        assert "10 classes" in str(exc), exc
    print("[real] fixed-shape / class validation rejects mismatches (no download)")


# --------------------------------------------------------------------------- #
# Stage 11 — live training: streamed progress events + checkpoints (save/resume).
# --------------------------------------------------------------------------- #
def test_train_iter_streams_events():
    spec = GraphSpec.model_validate(BRANCH_ADD)
    model, build = mlc.build_model(spec)
    in_shape = tuple(build.layers[0].out_shape)
    events = list(mlc.train_iter(model, in_shape, spec.train))
    assert events[0]["type"] == "start" and events[0]["total_params"] > 0, events[0]
    steps = [e for e in events if e["type"] == "step"]
    epochs = [e for e in events if e["type"] == "epoch"]
    done = [e for e in events if e["type"] == "done"]
    assert len(done) == 1 and len(steps) == done[0]["result"].steps
    assert len(epochs) == spec.train.epochs  # 3 full epochs, no step cap
    assert steps[-1]["loss"] < steps[0]["loss"]  # the loss drops as events stream
    res = done[0]["result"]
    assert res.final_loss < res.loss_curve[0] and res.final_accuracy > 0.8
    print(f"[stream] {len(steps)} step + {len(epochs)} epoch events; acc={res.final_accuracy:.3f}")


CKPT_MLP = {
    "nodes": [
        {"id": "in", "type": "Input", "params": {"shape": [12]}},
        {"id": "fc1", "type": "Linear", "params": {"out_features": 24}},
        {"id": "a1", "type": "ReLU", "params": {}},
        {"id": "fc2", "type": "Linear", "params": {"out_features": 6}},
        {"id": "out", "type": "Output", "params": {}},
    ],
    "edges": [["in", "fc1"], ["fc1", "a1"], ["a1", "fc2"], ["fc2", "out"]],
    "train": {"dataset": "blobs", "epochs": 4, "lr": 0.05, "batch_size": 64, "samples": 256, "seed": 0},
}


def _train(graph, **train_over):
    spec = GraphSpec.model_validate({**graph, "train": {**graph["train"], **train_over}})
    model, build = mlc.build_model(spec)
    return mlc.train_model(model, tuple(build.layers[0].out_shape), spec.train)


def test_checkpoint_save_and_resume():
    import os as _os
    from mlconstructor.trainer import _CKPT_DIR

    fresh = _train(CKPT_MLP)                      # no checkpoint
    saved = _train(CKPT_MLP, checkpoint=True)     # save the best-accuracy weights
    assert saved.checkpoint is not None and saved.best_accuracy is not None
    path = _os.path.join(_CKPT_DIR, saved.checkpoint + ".pt")
    assert _os.path.exists(path), path
    try:
        resumed = _train(CKPT_MLP, resume=True)   # load those weights, keep training
        assert resumed.resumed is True
        # resuming from trained weights starts far below a fresh run's first loss
        assert resumed.loss_curve[0] < fresh.loss_curve[0], (resumed.loss_curve[0], fresh.loss_curve[0])
    finally:
        if _os.path.exists(path):
            _os.remove(path)
    print(f"[checkpoint] saved {saved.checkpoint}; resumed first-loss "
          f"{resumed.loss_curve[0]:.3f} << fresh {fresh.loss_curve[0]:.3f}")


# --------------------------------------------------------------------------- #
# Stage 12 — GPU as an explicit opt-in. The default never touches the (shared) GPU;
# "cuda" fails clearly when no CUDA device exists; "auto" falls back to the CPU. The
# real on-card run is verified separately, inside a coordinated GPU window.
# --------------------------------------------------------------------------- #
def test_device_default_cpu_and_cuda_optin():
    import torch as _torch

    res = _train(CKPT_MLP)
    assert res.device == "cpu", res.device  # default: never touches the GPU
    real = _torch.cuda.is_available
    try:
        _torch.cuda.is_available = lambda: False  # simulate a backend without CUDA
        try:
            _train(CKPT_MLP, device="cuda")
            raise AssertionError("expected a clear no-CUDA error")
        except ValueError as exc:
            assert "CUDA" in str(exc), exc
        assert _train(CKPT_MLP, device="auto").device == "cpu"  # auto falls back
    finally:
        _torch.cuda.is_available = real
    print("[device] default cpu; device=cuda errors clearly without CUDA; auto -> cpu")


def test_probe_runs_on_the_model_device():
    # Regression (found by the promotion session recording Stage 11 on a CUDA venv):
    # the class-count probe must run on the MODEL's device. Stage 11 built it on the
    # CPU, so a model moved to CUDA crashed with "Input type (torch.FloatTensor) and
    # weight type (torch.cuda.FloatTensor)". The meta device exercises it, GPU-free.
    from mlconstructor.trainer import _infer_num_classes

    model, build = mlc.build_model(get_architecture("cnn"))
    model = model.to("meta")
    assert _infer_num_classes(model, tuple(build.layers[0].out_shape)) == 10
    print("[probe] class-count probe follows the model's device (meta) — Stage 11 CUDA crash fixed")


# --------------------------------------------------------------------------- #
# Stage 13 — projects: a versioned file format with migrations and warnings, plus
# the optional server-side store (exercised in a temp folder, and over HTTP).
# --------------------------------------------------------------------------- #
def _project_from(graph: dict, name: str = "Test project") -> dict:
    """A current-format project dict for a graph, every node placed on the canvas."""
    from mlconstructor.schema import PROJECT_FORMAT_VERSION

    return {
        "format": "mlconstructor.project",
        "version": PROJECT_FORMAT_VERSION,
        "name": name,
        "description": "made by the test suite",
        "graph": {
            "nodes": [
                {**n, "position": {"x": 70.0 + 13.5 * i, "y": 24.0 + 112.25 * i}}
                for i, n in enumerate(graph["nodes"])
            ],
            "edges": [list(e) for e in graph["edges"]],
            "edge_settings": [dict(s) for s in graph.get("edge_settings", [])],
        },
        "train": dict(graph.get("train", {})),
        "editor": {"custom_text": "1, 2, 0\n# a comment", "view": {"pan_x": -12.5, "pan_y": 40.0, "zoom": 0.8}},
    }


def _graph_spec_of(project) -> GraphSpec:
    """The plain GraphSpec a saved project trains with (positions dropped)."""
    return GraphSpec.model_validate({
        "nodes": [n.model_dump(exclude={"position"}) for n in project.graph.nodes],
        "edges": project.graph.edges,
        "train": project.train.model_dump(),
        "edge_settings": [s.model_dump() for s in project.graph.edge_settings],
    })


def test_project_round_trip_is_exact():
    gpt = get_architecture("gpt").model_dump()  # nested params: a Repeat block's inner layers
    custom = _custom_graph(CUSTOM_ROWS, epochs=7).model_dump()  # pasted rows travel with the project
    for graph in (CNN_MLP, gpt, custom):
        raw = _project_from(graph)
        first = mlc.normalize_project(raw)
        assert first.ok and first.migrated_from is None and first.warnings == [], first.warnings
        saved = json.loads(first.project.model_dump_json())   # what "Save" writes
        again = mlc.normalize_project(saved)                  # what "Open" reads back
        assert again.project == first.project and again.warnings == []
        # ids, types, params (nested ones too), positions, edges, train setup and view survive exactly
        for got, want in zip(again.project.graph.nodes, raw["graph"]["nodes"], strict=True):
            assert (got.id, got.type, got.params) == (want["id"], want["type"], want["params"])
            assert (got.position.x, got.position.y) == (want["position"]["x"], want["position"]["y"])
        assert again.project.graph.edges == raw["graph"]["edges"]
        for key, value in raw["train"].items():
            if key == "dataset":  # the legacy default "synthetic" opens as its catalogue id
                value = {"synthetic": "blobs"}.get(value, value)
            assert getattr(again.project.train, key) == value, key
        assert again.project.editor.model_dump() == raw["editor"]
        assert (again.project.name, again.project.description) == (raw["name"], raw["description"])
        # ... and the reopened project builds into exactly the same model
        _m, b_open = mlc.build_model(_graph_spec_of(again.project))
        _m, b_orig = mlc.build_model(GraphSpec.model_validate(graph))
        assert (b_open.total_params, b_open.order) == (b_orig.total_params, b_orig.order)
    assert again.project.train.custom_rows == CUSTOM_ROWS and again.project.train.dataset == "custom"
    print("[project] save -> open is exact for CNN / GPT (Repeat) / custom rows; reopened graphs build identically")


def test_project_opens_a_bare_graph_spec():
    # "Version 0" = a bare graph spec (the JSON of /architectures/{id} or /build):
    # it opens, is upgraded, and its nodes have no positions (the editor lays them out).
    bare = get_architecture("resnet").model_dump()
    res = mlc.normalize_project(bare)
    from mlconstructor.schema import PROJECT_FORMAT_VERSION

    assert res.migrated_from == 0 and res.project.version == PROJECT_FORMAT_VERSION
    assert res.project.name == "Imported graph"
    assert {w.code for w in res.warnings} == {"migrated", "missing_positions"}, res.warnings
    assert [n.id for n in res.project.graph.nodes] == [n["id"] for n in bare["nodes"]]
    assert all(n.position is None for n in res.project.graph.nodes)
    assert res.project.graph.edges == bare["edges"]
    assert bare["train"]["dataset"] == "synthetic" and res.project.train.dataset == "blobs"  # legacy alias
    assert res.project.train == TrainSpec.model_validate({**bare["train"], "dataset": "blobs"})
    _m, b = mlc.build_model(_graph_spec_of(res.project))
    print(f"[project] bare graph spec (format 0) opens: upgraded to {PROJECT_FORMAT_VERSION}, {len(bare['nodes'])} nodes to lay out, "
          f"{b.total_params:,} params")


def test_project_warnings_and_repairs():
    raw = _project_from(CNN_MLP)
    raw["graph"]["nodes"].append({"id": "mystery", "type": "QuantumLayer", "params": {}, "position": {"x": 1, "y": 2}})
    raw["graph"]["nodes"].append({"id": "loose", "type": "ReLU", "params": {}})  # no position
    raw["graph"]["edges"] += [["fc2", "ghost"], ["mystery", "loose"]]
    res = mlc.normalize_project(raw)
    codes = [w.code for w in res.warnings]
    assert sorted(codes) == ["dangling_edge", "missing_positions", "unknown_layer"], codes
    unknown = next(w for w in res.warnings if w.code == "unknown_layer")
    assert unknown.node_id == "mystery" and "QuantumLayer" in unknown.message
    assert ["fc2", "ghost"] not in res.project.graph.edges  # the broken edge is dropped...
    assert ["mystery", "loose"] in res.project.graph.edges  # ...a valid one to an unknown layer is kept
    assert len(res.project.graph.nodes) == len(CNN_MLP["nodes"]) + 2  # nothing the user made is lost
    print("[project] unknown layer kept + warned, dangling edge dropped + warned, missing position warned")


def test_project_rejects_bad_files():
    from mlconstructor import projects as P

    dup = _project_from(CNN_MLP)
    dup["graph"]["nodes"].append(dict(dup["graph"]["nodes"][0]))
    bad = {
        "not an object": [1, 2, 3],
        "not a project": {"hello": "world"},
        "newer format": {"format": "mlconstructor.project", "version": 99},
        "bad version": {"format": "mlconstructor.project", "version": "1"},
        "bad edge": {"format": "mlconstructor.project", "version": 1, "graph": {"edges": [["a"]]}},
        "bad train": {"format": "mlconstructor.project", "version": 1, "train": {"epochs": 0}},
        "duplicate ids": dup,
    }
    msgs = {}
    for case, raw in bad.items():
        try:
            mlc.normalize_project(raw)
            raise AssertionError(f"{case}: expected a clear error")
        except ValueError as exc:
            msgs[case] = str(exc)
    assert "newer version" in msgs["newer format"] and "update" in msgs["newer format"]
    assert "invalid format version" in msgs["bad version"]
    assert "graph.edges" in msgs["bad edge"] and "train.epochs" in msgs["bad train"]
    assert "duplicate node ids: in" in msgs["duplicate ids"]
    old = P.MAX_NODES
    P.MAX_NODES = 3
    try:
        try:
            mlc.normalize_project(_project_from(CNN_MLP))
            raise AssertionError("expected the node limit to trip")
        except ValueError as exc:
            assert "limit is 3" in str(exc), exc
    finally:
        P.MAX_NODES = old
    print(f"[project] {len(msgs) + 1} kinds of bad file rejected with clear messages")


def test_server_project_store():
    import tempfile

    from mlconstructor import projects as P

    loaded = mlc.normalize_project(_project_from(CNN_MLP, name="Моя сеть")).project
    with tempfile.TemporaryDirectory() as root:
        assert mlc.list_projects(root) == []
        first = mlc.create_project(loaded, root)
        second = mlc.create_project(loaded, root)  # the same title gets a fresh name, never overwrites
        assert (first.slug, second.slug) == ("moya-set", "moya-set-2"), (first.slug, second.slug)
        assert not [f for f in os.listdir(root) if f.endswith(".tmp")]  # atomic writes leave no temp files
        listing = mlc.list_projects(root)
        assert {s.slug for s in listing} == {"moya-set", "moya-set-2"}
        assert all(s.name == "Моя сеть" and s.node_count == len(CNN_MLP["nodes"]) for s in listing)
        back = mlc.load_project("moya-set", root)
        assert back.project.graph == loaded.graph and back.project.train == loaded.train
        assert back.project.created_at and back.project.updated_at
        again = mlc.save_project("moya-set", back.project, root)  # overwrite keeps the creation date
        assert mlc.load_project("moya-set", root).project.created_at == back.project.created_at
        assert again.updated_at >= back.project.updated_at
        mlc.delete_project("moya-set-2", root)
        assert [s.slug for s in mlc.list_projects(root)] == ["moya-set"]
        # path traversal and junk names are refused before any file is touched
        for evil in ["../evil", "..", "a/b", "a\\b", "", "Caps", "-x", "x.y", "x" * 65, "c:evil"]:
            for op in (mlc.load_project, mlc.delete_project):
                try:
                    op(evil, root)
                    raise AssertionError(f"{evil!r} accepted by {op.__name__}")
                except ValueError:
                    pass
            try:
                mlc.save_project(evil, loaded, root)
                raise AssertionError(f"{evil!r} accepted by save_project")
            except ValueError:
                pass
        try:
            mlc.load_project("missing", root)
            raise AssertionError("expected KeyError for a missing project")
        except KeyError:
            pass
        # limits: the file size and the number of stored projects
        old_bytes, old_count = P.MAX_PROJECT_BYTES, P.MAX_PROJECTS
        P.MAX_PROJECT_BYTES, P.MAX_PROJECTS = 8192, 2
        try:
            big = loaded.model_copy(update={"train": loaded.train.model_copy(update={"custom_rows": CUSTOM_ROWS * 40})})
            try:
                mlc.save_project("big", big, root)
                raise AssertionError("expected the size limit to trip")
            except ValueError as exc:
                assert "too large" in str(exc), exc
            mlc.create_project(loaded, root)  # 2 of 2
            try:
                mlc.create_project(loaded, root)
                raise AssertionError("expected the project-count limit to trip")
            except ValueError as exc:
                assert "limit 2" in str(exc), exc
        finally:
            P.MAX_PROJECT_BYTES, P.MAX_PROJECTS = old_bytes, old_count
        # a damaged file never breaks the listing
        with open(os.path.join(root, "broken.mlc.json"), "w", encoding="utf-8") as f:
            f.write("{not json")
        assert sorted(s.slug for s in mlc.list_projects(root)) == ["moya-set", "moya-set-2"]
    assert mlc.slugify("Моя сеть!") == "moya-set" and mlc.slugify("***") == "project"
    print("[project] server store: create/list/load/overwrite/delete, unique names, traversal refused, limits hold")


def _asgi(method: str, path: str, body: bytes = b"", content_length: bool = True) -> tuple[int, object]:
    """Call the FastAPI app in-process over raw ASGI (no network, no httpx)."""
    import asyncio

    from app import app as asgi_app

    headers = [(b"content-type", b"application/json")]
    if content_length:
        headers.append((b"content-length", str(len(body)).encode()))
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
        "scheme": "http", "path": path, "raw_path": path.encode(), "root_path": "", "query_string": b"",
        "headers": headers, "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 8050),
    }
    pending = [body[i:i + 65536] for i in range(0, len(body), 65536)] or [b""]
    out = {"status": 0, "body": b""}

    async def receive():
        if pending:
            chunk = pending.pop(0)
            return {"type": "http.request", "body": chunk, "more_body": bool(pending)}
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            out["status"] = message["status"]
        elif message["type"] == "http.response.body":
            out["body"] += message.get("body", b"")

    asyncio.run(asgi_app(scope, receive, send))
    return out["status"], (json.loads(out["body"]) if out["body"] else None)


def test_project_http_endpoints():
    import tempfile

    from mlconstructor import projects as P

    raw = json.dumps(_project_from(CNN_MLP, name="Через HTTP")).encode()
    old_dir = P._PROJECTS_DIR
    with tempfile.TemporaryDirectory() as root:
        P._PROJECTS_DIR = root  # keep the real store untouched
        try:
            st, health = _asgi("GET", "/health")
            assert st == 200 and health["server_projects"] is True, health
            st, res = _asgi("POST", "/projects/normalize", raw)
            assert st == 200 and res["migrated_from"] is None and res["warnings"] == [], res
            st, res = _asgi("POST", "/projects/normalize", json.dumps(get_architecture("mlp").model_dump()).encode())
            assert st == 200 and res["migrated_from"] == 0, res
            st, res = _asgi("POST", "/projects/normalize", b"{oops")
            assert st == 422 and "JSON" in res["detail"], res
            st, res = _asgi("POST", "/projects/normalize", b'{"format": "mlconstructor.project", "version": 7}')
            assert st == 422 and "newer version" in res["detail"], res
            huge = b" " * (P.MAX_PROJECT_BYTES + 1)
            st, _ = _asgi("POST", "/projects/normalize", huge)  # refused from the declared size
            assert st == 413, st
            st, _ = _asgi("POST", "/projects/normalize", huge, content_length=False)  # and while streaming
            assert st == 413, st
            st, saved = _asgi("POST", "/projects", raw)
            assert st == 200 and saved["slug"] == "cherez-http", saved
            st, listing = _asgi("GET", "/projects")
            assert st == 200 and [p["slug"] for p in listing] == ["cherez-http"], listing
            st, opened = _asgi("GET", "/projects/cherez-http")
            assert st == 200 and opened["project"]["graph"] == json.loads(raw)["graph"], opened
            st, saved = _asgi("PUT", "/projects/cherez-http", raw)
            assert st == 200 and saved["slug"] == "cherez-http", saved
            st, res = _asgi("GET", "/projects/..%2Fevil")
            assert st in (404, 422), (st, res)
            st, res = _asgi("PUT", "/projects/Bad.Name", raw)
            assert st == 422 and "invalid project name" in res["detail"], res
            st, res = _asgi("GET", "/projects/missing")
            assert st == 404, (st, res)
            st, res = _asgi("DELETE", "/projects/cherez-http")
            assert st == 200 and res["ok"] is True, res
            st, listing = _asgi("GET", "/projects")
            assert listing == [], listing
            os.environ["MLC_SERVER_PROJECTS"] = "0"  # the store can be switched off...
            try:
                assert _asgi("GET", "/projects")[0] == 404
                assert _asgi("POST", "/projects", raw)[0] == 404
                assert _asgi("POST", "/projects/normalize", raw)[0] == 200  # ...opening files still works
                assert _asgi("GET", "/health")[1]["server_projects"] is False
            finally:
                del os.environ["MLC_SERVER_PROJECTS"]
        finally:
            P._PROJECTS_DIR = old_dir
    print("[project] HTTP: normalize (any version, 413 / 422), server CRUD, bad names refused, store switch")


# --------------------------------------------------------------------------- #
# Stage 14 — smart connections: every edge adapts shapes ("auto"), or uses the
# transform the user picked, with its parameters; a connection can be inspected.
# --------------------------------------------------------------------------- #
#: The graph from the user's screenshot (25.11): an MLP with a Conv2d and an LSTM
#: wired onto a dense layer. Before Stage 14 it failed with "conv2d_12 expects a
#: 3-D input (C,H,W), got shape (128,)".
SCREENSHOT_GRAPH = {
    "nodes": [
        {"id": "in", "type": "Input", "params": {"shape": [64]}},
        {"id": "fc1", "type": "Linear", "params": {"out_features": 128}},
        {"id": "act1", "type": "ReLU", "params": {}},
        {"id": "drop", "type": "Dropout", "params": {"p": 0.1}},
        {"id": "fc2", "type": "Linear", "params": {"out_features": 64}},
        {"id": "conv2d_12", "type": "Conv2d", "params": {"out_channels": 16, "kernel_size": 3}},
        {"id": "lstm_13", "type": "LSTM", "params": {"hidden_size": 64}},
        {"id": "concat_10", "type": "Concat", "params": {}},
        {"id": "relu_11", "type": "ReLU", "params": {}},
        {"id": "fc3", "type": "Linear", "params": {"out_features": 10}},
        {"id": "out", "type": "Output", "params": {}},
    ],
    "edges": [
        ["in", "fc1"], ["fc1", "act1"], ["act1", "drop"], ["drop", "fc2"], ["drop", "conv2d_12"],
        ["conv2d_12", "lstm_13"], ["fc2", "concat_10"], ["lstm_13", "concat_10"], ["concat_10", "relu_11"],
        ["relu_11", "fc3"], ["fc3", "out"],
    ],
    "train": {"epochs": 3, "lr": 0.01, "batch_size": 64, "samples": 256, "seed": 0},
}


def test_screenshot_graph_builds_and_trains():
    spec = GraphSpec.model_validate(SCREENSHOT_GRAPH)
    model, build = mlc.build_model(spec)
    adapted = {(e.source, e.target): e for e in build.edges if e.transform != "none"}
    assert set(adapted) == {("drop", "conv2d_12"), ("conv2d_12", "lstm_13")}, adapted
    to_img = adapted[("drop", "conv2d_12")]
    assert (to_img.transform, to_img.out_shape, to_img.label, to_img.auto) == ("to_image", [2, 8, 8], "→ 2×8×8", True)
    rows = adapted[("conv2d_12", "lstm_13")]
    assert (rows.transform, rows.params, rows.out_shape) == ("image_to_sequence", {"mode": "rows"}, [6, 96])
    res = mlc.train_model(model, tuple(build.layers[0].out_shape), spec.train)
    assert res.final_accuracy > 0.9 and res.final_loss < res.loss_curve[0], res
    print(f"[connections] screenshot graph builds by itself (→2×8×8, rows) and trains: acc={res.final_accuracy:.3f}")


def _auto_edge(in_shape, target: dict, then: list[dict] | None = None):
    """What "auto" puts on Input(in_shape) -> target (followed by ``then`` and an Output)."""
    from mlconstructor.builder import plan_graph

    nodes = [{"id": "in", "type": "Input", "params": {"shape": list(in_shape)}}, {"id": "t", **target}]
    edges = [["in", "t"]]
    prev = "t"
    for i, extra in enumerate(then or []):
        nodes.append({"id": f"x{i}", **extra})
        edges.append([prev, f"x{i}"])
        prev = f"x{i}"
    nodes.append({"id": "out", "type": "Output", "params": {}})
    edges.append([prev, "out"])
    plan = plan_graph(GraphSpec.model_validate({"nodes": nodes, "edges": edges}))
    r = plan.edges[("in", "t")]
    return r.transform.kind, r.transform.resolved_params(r.in_shape), list(r.out_shape)


def test_connection_auto_choices():
    linear = {"type": "Linear", "params": {"out_features": 4}}
    conv = {"type": "Conv2d", "params": {"out_channels": 4, "kernel_size": 3}}
    lstm = {"type": "LSTM", "params": {"hidden_size": 8}}
    cases = [
        ((1, 28, 28), linear, None, ("flatten", {}, [784])),
        ((16, 6, 6), lstm, None, ("image_to_sequence", {"mode": "rows"}, [6, 96])),
        ((128,), conv, None, ("to_image", {"channels": 2, "height": 8, "width": 8}, [2, 8, 8])),
        ((784,), conv, None, ("to_image", {"channels": 1, "height": 28, "width": 28}, [1, 28, 28])),
        ((10,), conv, None, ("project", {"shape": [1, 8, 8]}, [1, 8, 8])),  # no fitting picture -> learn one
        ((5, 6), linear, None, ("flatten", {}, [30])),
        ((64,), lstm, None, ("to_sequence", {"length": 1}, [1, 64])),
        ((5, 6), conv, None, ("to_image", {"channels": 1, "height": 5, "width": 6}, [1, 5, 6])),
        ((7, 3, 3), linear, [{"type": "ReLU", "params": {}}], ("flatten", {}, [63])),
        ((32,), linear, None, ("none", {}, [32])),  # shapes already fit: nothing at all
    ]
    for in_shape, target, then, expected in cases:
        got = _auto_edge(in_shape, target, then)
        assert got == expected, (in_shape, target["type"], got, expected)
    # a sequence into the Output is flattened too (Output takes one score per class)
    _m, b = mlc.build_model(GraphSpec.model_validate({
        "nodes": [{"id": "in", "type": "Input", "params": {"shape": [5, 6]}},
                  {"id": "rnn", "type": "GRU", "params": {"hidden_size": 3, "return_sequence": True}},
                  {"id": "out", "type": "Output", "params": {}}],
        "edges": [["in", "rnn"], ["rnn", "out"]],
    }))
    assert (b.edges[-1].transform, b.edges[-1].out_shape) == ("flatten", [15]), b.edges[-1]
    print(f"[connections] auto picks the simplest fitting transform in {len(cases) + 1} situations")


def test_every_transform_matches_its_module():
    from mlconstructor.connections import make_transform

    cases = [
        ("none", {}, (5,)), ("flatten", {}, (3, 4, 4)), ("flatten", {}, (6, 2)),
        ("global_pool", {"mode": "avg"}, (3, 5, 5)), ("global_pool", {"mode": "max"}, (3, 5, 5)),
        ("seq_pool", {"mode": "last"}, (4, 6)), ("seq_pool", {"mode": "first"}, (4, 6)),
        ("seq_pool", {"mode": "mean"}, (4, 6)), ("seq_pool", {"mode": "max"}, (4, 6)),
        ("to_image", {"channels": 2, "height": 4, "width": 4}, (32,)),
        ("to_image", {"channels": 1, "height": 4, "width": 6}, (4, 6)),
        ("to_sequence", {"length": 1}, (12,)), ("to_sequence", {"length": 4}, (12,)),
        ("image_to_sequence", {"mode": "rows"}, (3, 4, 5)), ("image_to_sequence", {"mode": "columns"}, (3, 4, 5)),
        ("image_to_sequence", {"mode": "pixels"}, (3, 4, 5)),
        ("image_to_sequence", {"mode": "patches", "patch": 2}, (3, 4, 6)),
        ("reshape", {"shape": [2, 3]}, (6,)), ("reshape", {"shape": [6]}, (2, 3)),
        ("project", {"shape": [7]}, (5,)), ("project", {"shape": [4, 9]}, (4, 6)),
        ("project", {"shape": [8, 5, 5]}, (3, 5, 5)), ("project", {"shape": [8, 2, 2]}, (3, 5, 5)),
        ("project", {"shape": [1, 8, 8]}, (10,)),
    ]
    x_seq = torch.arange(2 * 3 * 4 * 5, dtype=torch.float).reshape(2, 3, 4, 5)
    for kind, params, in_shape in cases:
        t = make_transform(kind, params)
        out = t.out_shape(in_shape)
        mod = t.build(in_shape)
        y = torch.randn(2, *in_shape) if mod is not None else None
        if mod is not None:
            assert tuple(mod(y).shape[1:]) == tuple(out), (kind, params, in_shape, tuple(mod(y).shape), out)
    # the rows / pixels readings put the right values in the right steps
    from mlconstructor.connections import ImageToSeq

    rows = ImageToSeq("rows")(x_seq)
    assert torch.equal(rows[0, 1], x_seq[0, :, 1, :].reshape(-1))  # step 1 = row 1 of every channel
    pixels = ImageToSeq("pixels")(x_seq)
    assert torch.equal(pixels[0, 7], x_seq[0, :, 1, 2])  # step 7 = the pixel (1, 2), all channels
    print(f"[connections] {len(cases)} transform settings: declared shape == module output, values in place")


def test_connection_choice_and_clear_errors():
    seq_graph = {
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [5, 4]}},
            {"id": "lstm", "type": "LSTM", "params": {"hidden_size": 6, "return_sequence": True}},
            {"id": "fc", "type": "Linear", "params": {"out_features": 3}},
            {"id": "out", "type": "Output", "params": {}},
        ],
        "edges": [["in", "lstm"], ["lstm", "fc"], ["fc", "out"]],
    }
    # the user's choice is used (mean over time instead of auto's flatten)
    _m, b = mlc.build_model(GraphSpec.model_validate(_pinned(seq_graph, "lstm", "fc", "seq_pool", mode="mean")))
    e = next(x for x in b.edges if x.target == "fc")
    assert (e.transform, e.auto, e.params, e.out_shape, e.label) == ("seq_pool", False, {"mode": "mean"}, [6], "mean")
    # a choice that cannot take the data, or whose result the layer refuses: clear, pointed errors
    bad_cases = [
        (_pinned(seq_graph, "lstm", "fc", "global_pool"), "cannot take"),  # pooling an image, but this is a sequence
        (_pinned(seq_graph, "lstm", "fc", "to_sequence", length=1), "cannot take"),
        (_pinned(seq_graph, "lstm", "fc", "image_to_sequence"), "cannot take"),
        (_pinned(seq_graph, "lstm", "fc", "reshape", shape=[5, 6]), "cannot take that"),  # still 2-D for a Linear
        (_pinned(seq_graph, "lstm", "fc", "to_image", channels=1, height=3, width=3), "multiply to 30"),
    ]
    for graph, phrase in bad_cases:
        try:
            mlc.build_model(GraphSpec.model_validate(graph))
            raise AssertionError(f"expected an error for {graph['edge_settings']}")
        except ShapeError as exc:
            assert phrase in str(exc) and exc.edge == ("lstm", "fc") and exc.hint, (phrase, str(exc))
    try:
        mlc.build_model(GraphSpec.model_validate(_pinned(seq_graph, "lstm", "fc", "teleport")))
        raise AssertionError("expected an unknown-transform error")
    except ValueError as exc:
        assert "unknown connection transform" in str(exc), exc
    print("[connections] the picked transform is used; bad picks fail clearly, pointing at the connection")


def test_merge_inputs_are_matched_automatically():
    head = [{"id": "fc", "type": "Linear", "params": {"out_features": 3}}, {"id": "out", "type": "Output", "params": {}}]
    tail = [["m", "fc"], ["fc", "out"]]
    # Add: two vectors of different length -> a learnable projection on the second input
    add_vec = GraphSpec.model_validate({
        "nodes": [{"id": "in", "type": "Input", "params": {"shape": [16]}},
                  {"id": "a", "type": "Linear", "params": {"out_features": 24}},
                  {"id": "b", "type": "Linear", "params": {"out_features": 16}},
                  {"id": "m", "type": "Add", "params": {}}] + head,
        "edges": [["in", "a"], ["in", "b"], ["a", "m"], ["b", "m"]] + tail,
    })
    _m, b = mlc.build_model(add_vec)
    e = next(x for x in b.edges if (x.source, x.target) == ("b", "m"))
    assert (e.transform, e.out_shape, e.num_params) == ("project", [24], 16 * 24 + 24), e
    # Add: an image and a vector holding the same number of values -> a free reshape
    add_img = GraphSpec.model_validate({
        "nodes": [{"id": "in", "type": "Input", "params": {"shape": [2, 4, 4]}},
                  {"id": "c", "type": "Conv2d", "params": {"out_channels": 2, "kernel_size": 3, "padding": 1}},
                  {"id": "f", "type": "Flatten", "params": {}},
                  {"id": "m", "type": "Add", "params": {}}] + head,
        "edges": [["in", "c"], ["in", "f"], ["c", "m"], ["f", "m"]] + tail,
    })
    _m, b = mlc.build_model(add_img)
    e = next(x for x in b.edges if (x.source, x.target) == ("f", "m"))
    assert (e.transform, e.out_shape, e.num_params) == ("reshape", [2, 4, 4], 0), e
    # Concat: an image and a vector -> the image is flattened, the vector stays
    concat = GraphSpec.model_validate({
        "nodes": [{"id": "in", "type": "Input", "params": {"shape": [1, 8, 8]}},
                  {"id": "c", "type": "Conv2d", "params": {"out_channels": 2, "kernel_size": 3, "padding": 1}},
                  {"id": "f", "type": "Flatten", "params": {}},
                  {"id": "l", "type": "Linear", "params": {"out_features": 5}},
                  {"id": "m", "type": "Concat", "params": {}}] + head,
        "edges": [["in", "c"], ["in", "f"], ["f", "l"], ["c", "m"], ["l", "m"]] + tail,
    })
    model, b = mlc.build_model(concat)
    by = {(x.source, x.target): x for x in b.edges}
    assert (by[("c", "m")].transform, by[("c", "m")].out_shape) == ("flatten", [128]), by[("c", "m")]
    assert by[("l", "m")].transform == "none"
    assert next(s for s in b.layers if s.id == "m").out_shape == [133]
    assert tuple(model(torch.zeros(2, 1, 8, 8)).shape) == (2, 3)
    # ResNet's own skip connections still need nothing (shapes already match)
    _m, b = mlc.build_model(get_architecture("resnet"))
    assert all(x.transform == "none" for x in b.edges), [x for x in b.edges if x.transform != "none"]
    print("[connections] Add inputs projected / reshaped, Concat inputs flattened; ResNet untouched")


def test_architectures_unchanged_by_connections():
    expected_params = {}
    for aid in ARCH_IDS:
        _m, b = mlc.build_model(get_architecture(aid))
        assert all(e.transform == "none" for e in b.edges), (aid, [e for e in b.edges if e.transform != "none"])
        expected_params[aid] = b.total_params
    print(f"[connections] every library architecture builds with plain connections: {expected_params}")


def test_export_reproduces_connections_exactly():
    graphs = {
        "screenshot": GraphSpec.model_validate(SCREENSHOT_GRAPH),
        "merge": GraphSpec.model_validate(_pinned({
            "nodes": [
                {"id": "in", "type": "Input", "params": {"shape": [64]}},
                {"id": "fc1", "type": "Linear", "params": {"out_features": 128}},
                {"id": "conv", "type": "Conv2d", "params": {"out_channels": 8, "kernel_size": 3}},
                {"id": "lstm", "type": "LSTM", "params": {"hidden_size": 16, "return_sequence": True}},
                {"id": "fc2", "type": "Linear", "params": {"out_features": 24}},
                {"id": "add", "type": "Add", "params": {}},
                {"id": "out", "type": "Output", "params": {}},
            ],
            "edges": [["in", "fc1"], ["fc1", "conv"], ["conv", "lstm"], ["lstm", "add"], ["fc1", "fc2"],
                      ["fc2", "add"], ["add", "out"]],
        }, "lstm", "add", "seq_pool", mode="mean")),
        "patches": GraphSpec.model_validate(_pinned({
            "nodes": [
                {"id": "in", "type": "Input", "params": {"shape": [1, 28, 28]}},
                {"id": "att", "type": "MultiheadAttention", "params": {"num_heads": 2}},
                {"id": "out", "type": "Output", "params": {}},
            ],
            "edges": [["in", "att"], ["att", "out"]],
        }, "in", "att", "image_to_sequence", mode="patches", patch=4)),
    }
    for name, spec in graphs.items():
        model, build = mlc.build_model(spec)
        code = mlc.generate_pytorch(spec, include_training=False)
        assert "mlconstructor" not in code, name
        gen = _exec_exported(code)["Model"]()
        with torch.no_grad():
            for p_gen, p_app in zip(gen.parameters(), model.parameters(), strict=True):
                p_gen.copy_(p_app)  # same weights in the same order...
        model.eval()
        gen.eval()
        x = torch.randn(3, *build.layers[0].out_shape)
        assert torch.allclose(gen(x), model(x), atol=1e-6), name  # ...give the same outputs
        print(f"[export:{name}] connections exported exactly: params={build.total_params:,}")


def test_connection_options_for_the_inspector():
    spec = GraphSpec.model_validate(SCREENSHOT_GRAPH)
    opts = mlc.connection_options(spec, "drop", "conv2d_12")
    assert opts.in_shape == [128] and opts.current == "auto" and opts.problem is None
    assert (opts.auto_transform, opts.auto_out_shape) == ("to_image", [2, 8, 8])
    by = {o.transform: o for o in opts.options}
    assert "flatten" not in by and "image_to_sequence" not in by  # they need an image / a 2-D input
    assert by["to_image"].fits and by["to_image"].presets[0] == {"channels": 2, "height": 8, "width": 8}
    assert not by["none"].fits and "3-D" in by["none"].note  # Direct: Conv2d needs C×H×W
    assert not by["to_sequence"].fits and by["project"].learnable
    assert by["project"].presets[0] == {"shape": [2, 8, 8]}  # offered with auto's shape
    # into a merge node: Concat with the other input as it is
    opts = mlc.connection_options(spec, "lstm_13", "concat_10")
    assert opts.in_shape == [64] and opts.auto_transform == "none"
    # a connection pinned to a bad transform reports the problem, and the options still come
    bad = GraphSpec.model_validate(_pinned(SCREENSHOT_GRAPH, "drop", "conv2d_12"))
    opts = mlc.connection_options(bad, "drop", "conv2d_12")
    assert opts.current == "none" and opts.problem and "3-D" in opts.problem and opts.options
    try:
        mlc.connection_options(spec, "in", "out")
        raise AssertionError("expected an error for a connection that does not exist")
    except ValueError:
        pass
    print("[connections] inspector: shape in, auto's pick, every option with fits / presets / reasons")


def test_project_format_2_keeps_connection_settings():
    from mlconstructor.schema import PROJECT_FORMAT_VERSION

    pinned = _pinned(SCREENSHOT_GRAPH, "conv2d_12", "lstm_13", "image_to_sequence", mode="columns")
    raw = _project_from(pinned)
    res = mlc.normalize_project(json.loads(mlc.normalize_project(raw).project.model_dump_json()))
    assert res.warnings == [] and [s.model_dump() for s in res.project.graph.edge_settings] == pinned["edge_settings"]
    _m, b = mlc.build_model(_graph_spec_of(res.project))
    e = next(x for x in b.edges if x.target == "lstm_13")
    assert (e.transform, e.params, e.auto) == ("image_to_sequence", {"mode": "columns"}, False)
    # a format-1 file (Stage 13) opens and is upgraded: its connections are all auto
    v1 = {**_project_from(SCREENSHOT_GRAPH), "version": 1}
    del v1["graph"]["edge_settings"]
    res = mlc.normalize_project(v1)
    assert res.migrated_from == 1 and res.project.version == PROJECT_FORMAT_VERSION
    assert [w.code for w in res.warnings] == ["migrated"] and res.project.graph.edge_settings == []
    # settings for a missing connection, or with an unknown transform, are dropped with a warning
    junk = _project_from({**SCREENSHOT_GRAPH, "edge_settings": [
        {"source": "fc1", "target": "out", "transform": "flatten", "params": {}},
        {"source": "fc3", "target": "out", "transform": "teleport", "params": {}},
    ]})
    res = mlc.normalize_project(junk)
    assert sorted(w.code for w in res.warnings) == ["dangling_setting", "unknown_transform"], res.warnings
    assert res.project.graph.edge_settings == []
    print("[project] format 2 keeps connection settings; format 1 upgrades; junk settings dropped with warnings")


def test_connection_http_endpoints():
    spec = json.dumps({"spec": SCREENSHOT_GRAPH, "source": "drop", "target": "conv2d_12"}).encode()
    st, res = _asgi("POST", "/connections/options", spec)
    assert st == 200 and res["auto_transform"] == "to_image" and res["in_shape"] == [128], res
    st, health = _asgi("GET", "/health")
    assert st == 200 and health["api"] == mlc.API_LEVEL >= 14, health  # the editor warns below this level
    st, res = _asgi("POST", "/build", json.dumps(SCREENSHOT_GRAPH).encode())
    assert st == 200 and any(e["transform"] == "to_image" for e in res["edges"]), res
    bad = _pinned(SCREENSHOT_GRAPH, "drop", "conv2d_12")
    st, res = _asgi("POST", "/build", json.dumps(bad).encode())
    assert st == 422 and isinstance(res["detail"], str), res  # the plain message stays where it was...
    assert res["error"]["edge"] == ["drop", "conv2d_12"] and res["error"]["node_id"] == "conv2d_12", res
    assert res["error"]["hint"], res  # ...and the structured part says where and what to do
    print("[connections] HTTP: /connections/options, /build edges, structured 422 (edge, node, hint)")


# --------------------------------------------------------------------------- #
# Stage 15 — honest accuracy: a held-out validation part the network never trains on.
# --------------------------------------------------------------------------- #
from mlconstructor.datasets import split_indices  # noqa: E402

SPLIT_MLP = {
    "nodes": [
        {"id": "in", "type": "Input", "params": {"shape": [12]}},
        {"id": "fc1", "type": "Linear", "params": {"out_features": 32}},
        {"id": "a1", "type": "ReLU", "params": {}},
        {"id": "fc2", "type": "Linear", "params": {"out_features": 4}},
        {"id": "out", "type": "Output", "params": {}},
    ],
    "edges": [["in", "fc1"], ["fc1", "a1"], ["a1", "fc2"], ["fc2", "out"]],
    "train": {"dataset": "blobs", "epochs": 5, "lr": 0.01, "batch_size": 64, "samples": 512, "seed": 0,
              "test_fraction": 0},  # Stage 15's checks are about validation alone
}

INSEPARABLE_MLP = {
    "nodes": [
        {"id": "in", "type": "Input", "params": {"shape": [2]}},
        {"id": "fc1", "type": "Linear", "params": {"out_features": 32}},
        {"id": "a1", "type": "ReLU", "params": {}},
        {"id": "fc2", "type": "Linear", "params": {"out_features": 32}},
        {"id": "a2", "type": "ReLU", "params": {}},
        {"id": "fc3", "type": "Linear", "params": {"out_features": 2}},
        {"id": "out", "type": "Output", "params": {}},
    ],
    "edges": [["in", "fc1"], ["fc1", "a1"], ["a1", "fc2"], ["fc2", "a2"], ["a2", "fc3"], ["fc3", "out"]],
    "train": {"dataset": "custom", "epochs": 40, "lr": 0.02, "batch_size": 64, "seed": 0, "test_fraction": 0},
}


def _inseparable_rows(seed: int = 7) -> list[list[float]]:
    """2 features, 2 classes, 500 rows that NO network can fit perfectly: 100 points
    on the diagonal x1 = x2 appear twice (rows i and i + 100), once labelled 0 and
    once 1, plus 300 clean points labelled x1 > x2. The best possible error over
    all 500 rows is exactly 20 % — one row of each identical pair must be wrong."""
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(100, generator=g) * 2 - 1
    rows = [[float(a), float(a), 0] for a in u] + [[float(a), float(a), 1] for a in u]
    p = torch.rand(300, 2, generator=g) * 2 - 1
    rows += [[float(a), float(b), int(a > b)] for a, b in p]
    return rows


def _legacy_loss_curve(model, in_shape, s) -> list[float]:
    """The pre-Stage-15 training loop (every row, no hold-out), for the byte-for-byte check."""
    torch.manual_seed(s.seed)
    model.eval()
    with torch.no_grad():
        num_classes = int(model(torch.zeros(2, *in_shape)).shape[1])
    model.train()
    X, y = mlc.make_dataset(s.dataset, in_shape, num_classes, s, torch.device("cpu"))
    n = X.shape[0]
    bs = min(s.batch_size, n)
    opt = torch.optim.Adam(model.parameters(), lr=s.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    curve: list[float] = []
    for _ in range(s.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
            curve.append(float(loss.detach()))
    return curve


def test_split_is_stratified_deterministic_and_safe():
    y = torch.tensor([0] * 600 + [1] * 300 + [2] * 100)
    tr, va = split_indices(y, 0.2, 0)
    assert (len(tr), len(va)) == (800, 200), (len(tr), len(va))
    assert [int((y[va] == c).sum()) for c in range(3)] == [120, 60, 20]  # every class keeps its share
    assert sorted(tr.tolist() + va.tolist()) == list(range(1000))  # disjoint, nothing lost
    assert torch.equal(va, split_indices(y, 0.2, 0)[1])  # the same seed -> the same split
    assert not torch.equal(va, split_indices(y, 0.2, 1)[1])  # another seed -> another split
    assert len(split_indices(y, 0.0, 0)[1]) == 0  # 0 = validation off
    # Tiny sets: every class keeps a row to train on, so nothing is held out when that is impossible.
    for ys, n_val in (([0], 0), ([0, 1], 0), ([0, 0], 1), ([0, 0, 1], 1), ([0, 1, 2, 3, 4], 0)):
        tr, va = split_indices(torch.tensor(ys), 0.2, 0)
        assert len(va) == n_val and {ys[i] for i in tr.tolist()} == set(ys), (ys, tr, va)
    # Randomised: the guarantees hold on 1000 random label sets and fractions.
    g = torch.Generator().manual_seed(1)
    for trial in range(1000):
        n = int(torch.randint(1, 80, (1,), generator=g))
        ys = torch.randint(0, int(torch.randint(1, 12, (1,), generator=g)), (n,), generator=g)
        f = round(float(torch.rand(1, generator=g)) * 0.5, 2)
        tr, va = split_indices(ys, f, trial)
        assert sorted(tr.tolist() + va.tolist()) == list(range(n)), trial
        assert set(ys[tr].tolist()) == set(ys.tolist()), trial  # no class vanishes from training
        want = min(max(1, round(n * f)), n - 1) if n >= 2 and f > 0 else 0
        assert len(va) <= want, (trial, len(va), want)
        for c in set(ys.tolist()):  # stratified: close to the class's proportional share
            nc, vc = int((ys == c).sum()), int((ys[va] == c).sum())
            assert vc <= nc - 1 and vc >= nc * len(va) // n - 1, (trial, c, nc, vc)
    print("[split] stratified 120/60/20, deterministic per seed, tiny sets safe, 1000 random sets hold")


def test_blobs_generalises_to_held_out_rows():
    res = _train(SPLIT_MLP)
    assert (res.train_samples, res.val_samples) == (410, 102), (res.train_samples, res.val_samples)
    assert res.final_accuracy >= 0.95 and res.final_val_accuracy is not None and res.final_val_accuracy >= 0.9, res
    assert len(res.acc_curve) == len(res.val_acc_curve) == len(res.val_loss_curve) == 5
    assert abs(res.val_acc_curve[-1] - res.final_val_accuracy) < 1e-6  # the last epoch == the final state
    assert res.final_val_loss is not None and res.final_val_loss < 0.5, res.final_val_loss
    assert res.fit == "generalises" and res.fit_note and not res.warnings, (res.fit, res.warnings)
    print(f"[validation] blobs: train {res.final_accuracy:.3f}, held-out {res.final_val_accuracy:.3f} "
          f"({res.val_samples} rows it never saw) -> {res.fit}")


def test_inseparable_data_keeps_an_error():
    rows = _inseparable_rows()
    res = _train(INSEPARABLE_MLP, custom_rows=rows)
    assert (res.train_samples, res.val_samples) == (400, 100), (res.train_samples, res.val_samples)
    assert res.final_val_accuracy is not None  # measured at all
    # The floor: identical inputs with both labels inside one part -> one of each pair must be wrong.
    tr, va = split_indices(torch.tensor([int(r[-1]) for r in rows]), 0.2, 0)

    def floor(idx: torch.Tensor) -> float:
        part = set(idx.tolist())
        return sum(1 for i in range(100) if i in part and i + 100 in part) / len(part)

    assert 1 - res.final_accuracy >= floor(tr) - 1e-6 and floor(tr) > 0.1, (res.final_accuracy, floor(tr))
    assert 1 - res.final_val_accuracy >= floor(va) - 1e-6, (res.final_val_accuracy, floor(va))
    assert 0.6 < res.final_accuracy < 0.9 and 0.6 < res.final_val_accuracy < 0.9, res  # an error stays on both
    assert res.fit != "generalises", res.fit
    print(f"[validation] inseparable set: train {res.final_accuracy:.3f} (floor {1 - floor(tr):.3f}), "
          f"held-out {res.final_val_accuracy:.3f} -> {res.fit}; the error never reaches 0")


def test_overfitting_shows_on_held_out_rows():
    # shells: the MLP aces its training rows but not new ones -- the gap the old full-set accuracy hid
    spec = GraphSpec.model_validate({**SPLIT_MLP, "train": {**SPLIT_MLP["train"], "dataset": "shells", "epochs": 25}})
    model, build = mlc.build_model(spec)
    res = mlc.train_model(model, (12,), spec.train)
    assert res.fit == "overfits" and res.final_accuracy - res.final_val_accuracy > 0.1, res
    assert "Overfitting" in res.fit_note, res.fit_note
    # the numbers are what the trained model really scores on exactly those rows
    X, y = mlc.make_dataset("shells", (12,), 4, spec.train, torch.device("cpu"))
    tr, va = split_indices(y, 0.2, 0)
    model.eval()
    with torch.no_grad():
        acc_tr = float((model(X[tr]).argmax(1) == y[tr]).float().mean())
        acc_va = float((model(X[va]).argmax(1) == y[va]).float().mean())
    assert abs(acc_tr - res.final_accuracy) < 1e-6 and abs(acc_va - res.final_val_accuracy) < 1e-6, (
        acc_tr, res.final_accuracy, acc_va, res.final_val_accuracy)
    print(f"[validation] shells: train {res.final_accuracy:.3f} vs held-out {res.final_val_accuracy:.3f} "
          f"-> {res.fit} (the old full-set number hid this)")


def test_validation_off_is_the_original_loop():
    import copy

    spec = GraphSpec.model_validate({**SPLIT_MLP, "train": {**SPLIT_MLP["train"], "samples": 256, "seed": 3,
                                                            "epochs": 3, "val_fraction": 0}})
    model, build = mlc.build_model(spec)
    a, b = copy.deepcopy(model), copy.deepcopy(model)
    res = mlc.train_model(a, (12,), spec.train)
    assert res.loss_curve == _legacy_loss_curve(b, (12,), spec.train)  # byte for byte
    assert res.train_samples == 256 and res.val_samples == 0 and res.final_val_accuracy is None
    assert res.val_acc_curve == [] and res.fit == "unknown" and "Validation is off" in res.warnings[0], res
    print("[validation] val_fraction = 0 reproduces the pre-Stage-15 loop byte for byte, with a clear note")


def test_tiny_custom_sets_fall_back_clearly():
    for n, n_val in ((1, 0), (2, 0), (3, 1), (12, 2)):
        rows = [[float(i), float(-i), i % 2] for i in range(n)]
        res = _train(INSEPARABLE_MLP, custom_rows=rows, epochs=5)
        assert (res.train_samples, res.val_samples) == (n - n_val, n_val), (n, res.train_samples, res.val_samples)
        assert res.fit == "unknown" and len(res.warnings) == 1, (n, res.fit, res.warnings)
        if n_val:
            assert res.final_val_accuracy is not None and "Only" in res.warnings[0], res.warnings
        else:
            assert res.final_val_accuracy is None and "Too few rows" in res.warnings[0], res.warnings
    # A set too small to split still trains on what it has, with no error.
    st, out = _asgi("POST", "/train", json.dumps({**INSEPARABLE_MLP, "train": {
        **INSEPARABLE_MLP["train"], "epochs": 2, "custom_rows": [[0.0, 1.0, 0]]}}).encode())
    assert st == 200 and out["val_samples"] == 0 and out["warnings"], (st, out)
    print("[validation] 1-2 rows: nothing held out (a warning says why); 3 / 12 rows: 1 / 2 held out, no verdict")


def test_stream_and_checkpoint_use_validation():
    import os as _os

    from mlconstructor.trainer import _CKPT_DIR

    # shells, so the training and held-out curves differ and the checkpoint's choice shows
    spec = GraphSpec.model_validate({**CKPT_MLP, "train": {**CKPT_MLP["train"], "dataset": "shells", "epochs": 6,
                                                           "checkpoint": True, "test_fraction": 0}})
    model, build = mlc.build_model(spec)
    events = list(mlc.train_iter(model, tuple(build.layers[0].out_shape), spec.train))
    start, done = events[0], events[-1]["result"]
    try:
        assert start["samples"] == 256 and (start["train_samples"], start["val_samples"]) == (205, 51), start
        epochs = [e for e in events if e["type"] == "epoch"]
        assert len(epochs) == 6 and all(e["val_acc"] is not None and e["val_loss"] is not None for e in epochs)
        assert [e["val_acc"] for e in epochs] == done.val_acc_curve
        assert [e["acc"] for e in epochs] == done.acc_curve and done.acc_curve != done.val_acc_curve
        assert max(done.acc_curve) != max(done.val_acc_curve), (done.acc_curve, done.val_acc_curve)
        assert done.best_accuracy == max(done.val_acc_curve), (done.best_accuracy, done.val_acc_curve)
    finally:
        path = _os.path.join(_CKPT_DIR, f"{done.checkpoint}.pt")
        if done.checkpoint and _os.path.exists(path):
            _os.remove(path)
    print(f"[validation] stream: start {start['train_samples']}+{start['val_samples']} rows, val_acc per epoch; "
          f"checkpoint keeps the best held-out accuracy {done.best_accuracy:.3f}")


def test_preview_and_export_share_the_split():
    import contextlib
    import io

    spec = GraphSpec.model_validate(SPLIT_MLP)
    prev = mlc.data_preview(spec, rows=40)
    _, va = split_indices(mlc.make_dataset("blobs", (12,), 4, spec.train, torch.device("cpu"))[1], 0.2, 0)
    held = set(va.tolist())
    assert (prev.train_samples, prev.val_samples) == (410, 102), (prev.train_samples, prev.val_samples)
    assert [r.split for r in prev.rows] == ["validation" if i in held else "train" for i in range(40)]
    assert {r.split for r in prev.rows} == {"train", "validation"}
    code = mlc.generate_pytorch(spec, include_training=True)
    assert "VAL_FRACTION = 0.2" in code and "def split_parts" in code and "mlconstructor" not in code
    ns = _exec_exported(code)
    for ys in (torch.randint(0, 5, (97,)), torch.tensor([0, 0, 1]), torch.tensor([3])):
        for f in (0.0, 0.2, 0.5):
            assert all(torch.equal(p, q) for p, q in zip(ns["split_parts"](ys, f, 0.0, 4), split_indices(ys, f, 4)))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns["main"]()
    lines = buf.getvalue().strip().splitlines()
    assert len(lines) == 5 and all("train_acc" in ln and "val_acc" in ln for ln in lines), lines
    print(f"[validation] preview marks held-out rows; the export splits like the app: {lines[-1].strip()}")


def test_stage15_http():
    st, health = _asgi("GET", "/health")
    assert st == 200 and health["api"] == mlc.API_LEVEL >= 15, health  # the editor needs this level
    st, res = _asgi("POST", "/train", json.dumps(SPLIT_MLP).encode())
    assert st == 200 and res["val_samples"] == 102 and res["final_val_accuracy"] >= 0.9, res
    assert len(res["val_acc_curve"]) == 5 and res["fit"] == "generalises", res
    st, prev = _asgi("POST", "/data/preview", json.dumps(SPLIT_MLP).encode())
    assert st == 200 and prev["val_samples"] == 102 and all(r["split"] in ("train", "validation") for r in prev["rows"])
    bad = {**SPLIT_MLP, "train": {**SPLIT_MLP["train"], "val_fraction": 0.9}}
    st, res = _asgi("POST", "/train", json.dumps(bad).encode())
    assert st == 422, (st, res)  # at most half the rows can be held out
    print("[validation] HTTP: /health api 15, /train held-out fields + verdict, /data/preview split, 0.9 rejected")


# --------------------------------------------------------------------------- #
# Stage 16 — the same seed, the same numbers; a test part; keep / load / test weights.
# --------------------------------------------------------------------------- #
import base64  # noqa: E402
import io  # noqa: E402

from mlconstructor import weights as W  # noqa: E402
from mlconstructor.datasets import split_parts  # noqa: E402

RUN_MLP = {
    "nodes": SPLIT_MLP["nodes"],
    "edges": SPLIT_MLP["edges"],
    "train": {"dataset": "blobs", "epochs": 3, "lr": 0.01, "batch_size": 64, "samples": 512, "seed": 0},
}


def _post(path: str, obj) -> tuple[int, object]:
    return _asgi("POST", path, json.dumps(obj).encode())


def _get_raw(path: str, query: str = "") -> tuple[int, bytes, dict]:
    """GET over raw ASGI, returning the body as bytes (a file download)."""
    import asyncio

    from app import app as asgi_app

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET", "scheme": "http",
        "path": path, "raw_path": path.encode(), "root_path": "", "query_string": query.encode(), "headers": [],
        "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 8050),
    }
    out = {"status": 0, "body": b"", "headers": {}}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            out["status"] = message["status"]
            out["headers"] = {k.decode().lower(): v.decode() for k, v in message.get("headers", [])}
        elif message["type"] == "http.response.body":
            out["body"] += message.get("body", b"")

    asyncio.run(asgi_app(scope, receive, send))
    return out["status"], out["body"], out["headers"]


def _upload(spec: dict, raw: bytes) -> tuple[int, object]:
    return _post("/weights/upload", {"spec": spec, "data": base64.b64encode(raw).decode()})


def test_same_seed_same_weights_and_numbers():
    spec = GraphSpec.model_validate(RUN_MLP)
    before = torch.get_rng_state()
    a, _ = mlc.build_model(spec, seed=3)
    b, _ = mlc.build_model(spec, seed=3)
    c, _ = mlc.build_model(spec, seed=4)
    assert torch.equal(torch.get_rng_state(), before)  # the global random state is left alone
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)  # the same seed -> the same starting weights
    assert not all(torch.equal(sa[k], sc[k]) for k in sa)  # another seed -> other weights
    # Through the app: Train twice, the same numbers — curves, validation, the final test.
    (s1, r1), (s2, r2) = _post("/train", RUN_MLP), _post("/train", RUN_MLP)
    assert s1 == s2 == 200 and r1["loss_curve"] == r2["loss_curve"], (s1, s2)
    assert r1["val_acc_curve"] == r2["val_acc_curve"] and r1["run_id"] != r2["run_id"]
    t1, t2 = (_post("/test", {"spec": RUN_MLP, "run_id": r["run_id"]})[1] for r in (r1, r2))
    assert (t1["accuracy"], t1["loss"]) == (t2["accuracy"], t2["loss"]), (t1, t2)
    assert not torch.cuda.is_initialized()  # seeding never opens CUDA
    print(f"[seed] the same seed -> the same weights, curves and test ({t1['accuracy']:.3f}); global RNG untouched")


def test_three_part_split_keeps_its_guarantees():
    y = torch.tensor([0] * 600 + [1] * 300 + [2] * 100)
    tr, va, te = split_parts(y, 0.2, 0.1, 0)
    assert (len(tr), len(va), len(te)) == (700, 200, 100), (len(tr), len(va), len(te))
    assert [int((y[te] == c).sum()) for c in range(3)] == [60, 30, 10]  # stratified test part
    assert [int((y[va] == c).sum()) for c in range(3)] == [120, 60, 20]  # and validation part
    assert sorted(tr.tolist() + va.tolist() + te.tolist()) == list(range(1000))  # disjoint, nothing lost
    assert all(torch.equal(p, q) for p, q in zip(split_parts(y, 0.2, 0.1, 0), split_parts(y, 0.2, 0.1, 0)))
    g = torch.Generator().manual_seed(2)
    for trial in range(400):
        n = int(torch.randint(1, 90, (1,), generator=g))
        ys = torch.randint(0, int(torch.randint(1, 10, (1,), generator=g)), (n,), generator=g)
        fv = round(float(torch.rand(1, generator=g)) * 0.4, 2)
        ft = round(float(torch.rand(1, generator=g)) * 0.3, 2)
        tr, va, te = split_parts(ys, fv, ft, trial)
        assert sorted(tr.tolist() + va.tolist() + te.tolist()) == list(range(n)), trial
        assert set(ys[tr].tolist()) == set(ys.tolist()), trial  # no class vanishes from training
        old = split_indices(ys, fv, trial)  # without a test part it is exactly Stage 15's split
        assert all(torch.equal(p, q) for p, q in zip(split_parts(ys, fv, 0.0, trial)[:2], old)), trial
    for ys in ([0], [0, 1], [0, 0, 1]):  # tiny sets: training keeps every class
        tr, va, te = split_parts(torch.tensor(ys), 0.2, 0.1, 0)
        assert {ys[i] for i in tr.tolist()} == set(ys), (ys, tr, va, te)
    try:
        GraphSpec.model_validate({**RUN_MLP, "train": {**RUN_MLP["train"], "val_fraction": 0.5, "test_fraction": 0.3}})
        raise AssertionError("80 % held out should be refused")
    except ValueError as exc:
        assert "at least 30 %" in str(exc), exc
    print("[split] train / validation / test 700/200/100 stratified; test 0 == Stage 15; 400 random sets hold")


def test_weights_download_upload_and_test():
    st, res = _post("/train", RUN_MLP)
    assert st == 200 and (res["train_samples"], res["val_samples"], res["test_samples"]) == (359, 102, 51), res
    st, raw, headers = _get_raw(f"/runs/{res['run_id']}/weights", "name=My%20net")
    assert st == 200 and headers["content-type"] == "application/octet-stream", (st, headers)
    assert 'filename="My net-weights.pt"' in headers["content-disposition"], headers
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)  # plain tensors and values
    assert payload["format"] == W.WEIGHTS_FORMAT and payload["data"] and payload["metrics"]["steps"] == res["steps"]
    # the very weights the run ended with (training repeats exactly, so retrain and compare)
    spec = GraphSpec.model_validate(RUN_MLP)
    model, build = mlc.build_model(spec, seed=0)
    mlc.train_model(model, (12,), spec.train)
    assert all(torch.equal(payload["state_dict"][k], v) for k, v in model.state_dict().items())
    st, test = _post("/test", {"spec": RUN_MLP, "run_id": res["run_id"]})
    assert st == 200 and test["samples"] == 51 and test["same_split"] is True and test["accuracy"] >= 0.9, test
    st, up = _upload(RUN_MLP, raw)  # the downloaded file loads back, knows its data, tests the same
    assert st == 200 and up["matched"] == "names" and up["data_known"] and up["same_split"] is True, up
    st, again = _post("/test", {"spec": RUN_MLP, "run_id": up["run_id"]})
    assert (again["accuracy"], again["loss"]) == (test["accuracy"], test["loss"]), (again, test)
    assert _get_raw("/runs/0123456789abcdef/weights")[0] == 404 and _get_raw("/runs/..%2Fx/weights")[0] == 404
    print(f"[weights] download -> {len(raw):,} bytes (safe to load), upload -> the same test {test['accuracy']:.3f}")


def test_final_test_reports_honestly():
    st, res = _post("/train", RUN_MLP)
    st, test = _post("/test", {"spec": RUN_MLP, "run_id": res["run_id"]})
    assert test["note"].startswith("Confirmed") and not test["warnings"], test
    other = {**RUN_MLP, "train": {**RUN_MLP["train"], "seed": 9}}  # another split: some test rows were trained on
    st, unfair = _post("/test", {"spec": other, "run_id": res["run_id"]})
    assert st == 200 and unfair["same_split"] is False and "not a fair check" in unfair["warnings"][0], unfair
    st, none = _post("/test", {"spec": {**RUN_MLP, "train": {**RUN_MLP["train"], "test_fraction": 0}},
                               "run_id": res["run_id"]})
    assert st == 422 and "no test part" in none["detail"], none
    st, gone = _post("/test", {"spec": RUN_MLP, "run_id": "0123456789abcdef"})
    assert st == 404 and "no longer on the server" in gone["detail"], gone
    # the inseparable set keeps an error on the test rows too
    insep = {**INSEPARABLE_MLP, "train": {**INSEPARABLE_MLP["train"], "test_fraction": 0.1,
                                          "custom_rows": _inseparable_rows()}}
    st, r2 = _post("/train", insep)
    st, t2 = _post("/test", {"spec": insep, "run_id": r2["run_id"]})
    assert st == 200 and t2["samples"] == 50 and 0.5 < t2["accuracy"] < 0.95, t2
    # a tiny test part: a warning, no reading
    tiny = {**RUN_MLP, "train": {**RUN_MLP["train"], "samples": 100}}
    st, r3 = _post("/train", tiny)
    st, t3 = _post("/test", {"spec": tiny, "run_id": r3["run_id"]})
    assert t3["samples"] == 10 and t3["note"] is None and "Only 10 rows" in t3["warnings"][-1], t3
    print(f"[test] blobs confirmed; another split flagged unfair; inseparable test {t2['accuracy']:.3f}; "
          "no test part / gone run / tiny part explained")


def test_exported_script_runs_app_weights():
    graphs = {aid: get_architecture(aid) for aid in ("mlp", "cnn", "gpt")}
    graphs["screenshot"] = GraphSpec.model_validate(SCREENSHOT_GRAPH)
    for name, spec in graphs.items():
        model, build = mlc.build_model(spec, seed=1)
        in_shape = tuple(build.layers[0].out_shape)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "w.pt")
            with open(path, "wb") as f:
                f.write(W.weights_bytes(model, data=None))
            ns = _exec_exported(mlc.generate_pytorch(spec, include_training=False))
            gen = ns["load_weights"](ns["Model"](), path)
        model.eval()
        gen.eval()
        x = (torch.randint(0, 10, (3, *in_shape)) if name == "gpt" else torch.randn(3, *in_shape))
        with torch.no_grad():
            assert torch.allclose(gen(x), model(x), atol=1e-6), name  # the same network, the same answers
    print("[weights] the exported model.py loads app weights (mlp, cnn + BatchNorm, gpt + Repeat, screenshot)")


def test_uploads_are_checked_and_safe():
    model, _ = mlc.build_model(GraphSpec.model_validate(RUN_MLP), seed=0)
    good = W.weights_bytes(model, data=None)
    wider = {**RUN_MLP, "nodes": [dict(n, params={"out_features": 64}) if n["id"] == "fc1" else n
                                  for n in RUN_MLP["nodes"]]}
    st, res = _upload(wider, good)
    assert st == 422 and "fc1.weight is [32, 12] in the file but [64, 12] here" in res["detail"], res
    st, res = _upload({**RUN_MLP, "nodes": SPLIT_MLP["nodes"][:3] + [
        {"id": "fc9", "type": "Linear", "params": {"out_features": 4}}, SPLIT_MLP["nodes"][4]],
        "edges": [["in", "fc1"], ["fc1", "a1"], ["a1", "fc9"], ["fc9", "out"]]}, good)
    assert st == 200 and res["matched"] == "order", res  # renamed layer, same shapes: matched by position
    st, res = _upload(get_architecture("cnn").model_dump(), good)
    assert st == 422 and "different network" in res["detail"], res
    bare = io.BytesIO()
    torch.save({k.removeprefix("_mods."): v for k, v in model.state_dict().items()}, bare)  # the exported script's
    st, res = _upload(RUN_MLP, bare.getvalue())
    assert st == 200 and res["matched"] == "order" and res["data_known"] is False and res["same_split"] is None
    st, res = _upload(RUN_MLP, b"definitely not a torch file")
    assert st == 422 and "not a PyTorch weights file" in res["detail"], res
    marker = os.path.join(tempfile.gettempdir(), f"mlc-pwned-{os.getpid()}")

    class Evil:
        def __reduce__(self):
            return (os.makedirs, (marker,))

    evil = io.BytesIO()
    torch.save({"w": Evil()}, evil)
    st, res = _upload(RUN_MLP, evil.getvalue())
    assert st == 422 and "nothing in it was run" in res["detail"] and not os.path.exists(marker), res
    st, res = _post("/weights/upload", {"spec": RUN_MLP, "data": "@@not base64@@"})
    assert st == 422 and "bad base64" in res["detail"], res
    big = json.dumps({"spec": RUN_MLP, "data": "A" * 16}).encode()
    st, res = _asgi("POST", "/weights/upload", big[:10], content_length=False)  # truncated JSON
    assert st == 422, (st, res)
    import app as app_module

    old = app_module._MAX_UPLOAD_JSON
    app_module._MAX_UPLOAD_JSON = 1000
    try:
        st, res = _upload(RUN_MLP, good)
        assert st == 413, (st, res)
    finally:
        app_module._MAX_UPLOAD_JSON = old
    print("[weights] uploads: a resized layer named, renamed / bare files by position, another network refused, "
          "junk refused, a code-running file refused before it runs, too big -> 413")


def test_training_starts_from_saved_weights():
    st, first = _post("/train", RUN_MLP)
    st, cont = _post("/train", {**RUN_MLP, "train": {**RUN_MLP["train"], "init_run": first["run_id"]}})
    assert st == 200 and cont["loss_curve"][0] < 0.25 * first["loss_curve"][0], (cont["loss_curve"][0],
                                                                                   first["loss_curve"][0])
    st, stale = _post("/train", {**RUN_MLP, "train": {**RUN_MLP["train"], "init_run": "ffffffffffffffff"}})
    assert st == 422 and "no longer on the server" in stale["detail"], stale
    print(f"[weights] training from saved weights starts at loss {cont['loss_curve'][0]:.3f} "
          f"(fresh {first['loss_curve'][0]:.3f}); a stale id is explained")


def test_runs_are_pruned():
    old_max = W.MAX_RUNS
    with tempfile.TemporaryDirectory() as d:
        os.environ["MLC_RUNS_DIR"], prev = d, os.environ["MLC_RUNS_DIR"]
        W.MAX_RUNS = 3
        try:
            model, _ = mlc.build_model(GraphSpec.model_validate(RUN_MLP), seed=0)
            ids = [W.save_run(W.weights_bytes(model, data=None)) for _ in range(5)]
            left = sorted(f[:-3] for f in os.listdir(d) if f.endswith(".pt"))
            assert sorted(ids[-3:]) == left, (ids, left)  # exactly the newest three survive
            # an id is never a path: a .pt next to the store stays out of reach
            outside = os.path.join(os.path.dirname(d), f"outside-{os.getpid()}.pt")
            with open(outside, "wb") as f:
                f.write(W.weights_bytes(model, data=None))
            try:
                for bad in ("../" + os.path.basename(outside)[:-3], "..\\" + os.path.basename(outside)[:-3]):
                    st, res = _post("/test", {"spec": RUN_MLP, "run_id": bad})
                    assert st == 404, (bad, st, res)
            finally:
                os.remove(outside)
        finally:
            W.MAX_RUNS = old_max
            os.environ["MLC_RUNS_DIR"] = prev
    print("[weights] the server keeps the last runs only (oldest removed first)")


def test_stage16_http():
    st, health = _asgi("GET", "/health")
    assert st == 200 and health["api"] == mlc.API_LEVEL >= 16, health
    st, prev = _post("/data/preview", RUN_MLP)
    assert st == 200 and (prev["train_samples"], prev["val_samples"], prev["test_samples"]) == (359, 102, 51), prev
    assert {r["split"] for r in prev["rows"]} <= {"train", "validation", "test"}
    code = mlc.generate_pytorch(GraphSpec.model_validate(RUN_MLP), include_training=True)
    assert "TEST_FRACTION = 0.1" in code and "def load_weights" in code
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _exec_exported(code)["main"]()
    assert "final test on 51 rows" in buf.getvalue(), buf.getvalue()
    print("[stage16] HTTP: /health api 16, preview marks train / validation / test, export tests once at the end")


# --------------------------------------------------------------------------- #
# Stage 17 — flagship language models: real sizes on "meta", mini models that train.
# --------------------------------------------------------------------------- #
from mlconstructor.flagships import FLAGSHIPS, mini_spec, real_spec  # noqa: E402
from mlconstructor.shapes import ShapeError as _ShapeError  # noqa: E402

GQA_GRAPH = {
    "nodes": [
        {"id": "in", "type": "Input", "params": {"shape": [8]}},
        {"id": "emb", "type": "Embedding", "params": {"num_embeddings": 32, "embedding_dim": 32}},
        {"id": "att", "type": "GQAttention", "params": {"heads": 4, "kv_heads": 2, "head_dim": 8}},
        {"id": "mlp", "type": "GatedMLP", "params": {"hidden": 64}},
        {"id": "norm", "type": "RMSNorm", "params": {}},
        {"id": "head", "type": "LMHead", "params": {"vocab": 32}},
        {"id": "out", "type": "Output", "params": {}},
    ],
    "edges": [["in", "emb"], ["emb", "att"], ["att", "mlp"], ["mlp", "norm"], ["norm", "head"], ["head", "out"]],
    "edge_settings": [{"source": "norm", "target": "head", "transform": "seq_pool", "params": {"mode": "last"}}],
    "train": {"dataset": "tokens", "epochs": 1, "samples": 64, "seed": 0},
}


def _with(graph: dict, node_id: str, **params) -> dict:
    g = json.loads(json.dumps(graph))
    for n in g["nodes"]:
        if n["id"] == node_id:
            n["params"].update(params)
    return g


def test_flagship_real_sizes_match_hugging_face():
    for fam in FLAGSHIPS:
        model, build = mlc.build_model(GraphSpec.model_validate(real_spec(fam)), meta=True)
        total = sum(p.numel() for p in model.parameters())
        assert total == build.total_params, (fam["id"], total, build.total_params)
        assert all(p.is_meta for p in model.parameters()), fam["id"]  # sized, not allocated
        if fam["id"] == "deepseek_v3":  # the Hub's 684.5B adds the multi-token-prediction module
            assert total == 671_026_419_200 and abs(total - fam["official"]) / fam["official"] < 0.01, total
        else:
            assert total == fam["official"], (fam["real_name"], total, fam["official"])  # to the parameter
    print("[flagships] real sizes on meta == Hugging Face: " + ", ".join(
        f"{f['real_name']} {f['official'] / 1e9:.2f}B" for f in FLAGSHIPS))


def test_flagship_minis_train_on_next_token():
    for fam in FLAGSHIPS:  # every mini model learns (a short run each)
        raw = mini_spec(fam)
        raw["train"].update({"samples": 1200, "epochs": 2})
        spec = GraphSpec.model_validate(raw)
        model, build = mlc.build_model(spec, seed=0)
        res = mlc.train_model(model, (16,), spec.train)
        assert build.total_params < 400_000 and res.final_loss < res.loss_curve[0], (fam["id"], res.final_loss)
        assert res.final_val_accuracy > 0.15, (fam["id"], res.final_val_accuracy)  # chance is 1/32
    spec = GraphSpec.model_validate(mini_spec(FLAGSHIPS[3]))  # Llama 3 mini, its full default run
    model, _ = mlc.build_model(spec, seed=0)
    res = mlc.train_model(model, (16,), spec.train)
    assert res.final_val_accuracy >= 0.85, res.final_val_accuracy
    print(f"[flagships] 8 mini models learn the next token; Llama 3 mini on new data {res.final_val_accuracy:.3f}")


def test_flagship_minis_export_exactly():
    for fam in FLAGSHIPS:
        spec = GraphSpec.model_validate(mini_spec(fam))
        model, _ = mlc.build_model(spec, seed=1)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "w.pt")
            with open(path, "wb") as f:
                f.write(W.weights_bytes(model, data=None))
            ns = _exec_exported(mlc.generate_pytorch(spec, include_training=False))
            gen = ns["load_weights"](ns["Model"](), path)
        model.eval()
        gen.eval()
        x = torch.randint(0, 32, (3, 16)).float()
        with torch.no_grad():
            assert torch.allclose(gen(x), model(x), atol=1e-5), fam["id"]
        assert sum(p.numel() for p in gen.parameters()) == sum(p.numel() for p in model.parameters()), fam["id"]
    print("[flagships] every mini model exports to model.py: the same parameters, the same answers")


def test_llm_layers_behave_and_explain():
    model, build = mlc.build_model(GraphSpec.model_validate(GQA_GRAPH), seed=0)
    model.eval()
    x = torch.randint(0, 32, (2, 8)).float()
    att = model._mods["n_att"]
    h = model._mods["n_emb"](x)
    with torch.no_grad():
        later = h.clone()
        later[:, -1] += 1.0  # change the last token only
        a, b = att(h), att(later)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6) and not torch.allclose(a[:, -1], b[:, -1])  # causal
    win = mlc.llm_modules.GroupedQueryAttention(32, 4, 2, 8, sliding_window=2).eval()
    with torch.no_grad():
        far = h.clone()
        far[:, 0] += 1.0  # token 0 is out of the last position's 2-token window
        assert torch.allclose(win(h)[:, -1], win(far)[:, -1], atol=1e-6)
    # latent attention with unequal head sizes: the exact parameter formula, and it runs
    mla = mlc.llm_modules.LatentAttention(32, 2, q_lora_rank=16, kv_lora_rank=8, qk_nope_dim=6, qk_rope_dim=4,
                                          v_dim=10).eval()
    expected = (32 + 32 * 16 + 16 + 16 * 2 * (6 + 4)  # pre-norm, query down / norm / up
                + 32 * (8 + 4) + 8 + 8 * 2 * (6 + 10)  # key/value latent + shared rotary key, norm, up
                + 2 * 10 * 32)  # output projection
    assert sum(p.numel() for p in mla.parameters()) == expected
    with torch.no_grad():
        assert tuple(mla(h).shape) == (2, 8, 32)
    # a tied head shares the embedding table: counted once, the very same tensor
    tied = GraphSpec.model_validate(_with(GQA_GRAPH, "head", tie_embedding=True))
    tm, tb = mlc.build_model(tied, seed=0)
    assert tm._mods["n_head"].weight is tm._mods["n_emb"].emb.weight
    assert tb.total_params == build.total_params - 32 * 32 == sum(p.numel() for p in tm.parameters())
    # clear messages for contradictory settings — the layer is named, not the connection
    cases = [
        (_with(GQA_GRAPH, "att", heads=4, kv_heads=3), "cannot be shared evenly", "multiple of the key/value heads"),
        (_with(GQA_GRAPH, "att", head_dim=7), "must be even", "even head size"),
        (_with(GQA_GRAPH, "head", tie_embedding=True, vocab=64), "sizes differ", "embedding's vocabulary size"),
    ]
    for graph, msg, hint in cases:
        try:
            mlc.build_model(GraphSpec.model_validate(graph))
            raise AssertionError(f"expected an error: {msg}")
        except _ShapeError as exc:
            assert msg in str(exc) and hint in (exc.hint or ""), (str(exc), exc.hint)
    moe = _with(GQA_GRAPH, "mlp")
    moe["nodes"][3] = {"id": "mlp", "type": "MoE", "params": {"experts": 2, "top_k": 3}}
    st, res = _asgi("POST", "/build", json.dumps(moe).encode())
    assert st == 422 and "each token goes to 3 experts" in res["detail"] and res["error"]["node_id"] == "mlp", res
    assert "no automatic transform" not in res["detail"] and "between 1 and the number" in res["error"]["hint"], res
    print("[llm layers] causal + sliding window hold; a tied head shares the table; clashing settings named")


def test_real_scale_is_inspect_only():
    real = get_architecture("gemma3_real").model_dump()
    st, b = _asgi("POST", "/build", json.dumps(real).encode())
    assert st == 200 and b["total_params"] == 999_885_952 and b["trainable"] is False, b
    st, res = _asgi("POST", "/train", json.dumps(real).encode())
    assert st == 422 and "far too big to train" in res["detail"] and "mini version" in res["detail"], res
    st, res = _upload(real, b"PK\x03\x04 not really")
    assert st == 422 and "far too big" in res["detail"], res
    st, prev = _asgi("POST", "/data/preview", json.dumps(real).encode())
    assert st == 200 and prev["num_classes"] == 262144, prev  # sized on meta: no weights made
    st, ex = _asgi("POST", "/export/pytorch", json.dumps(real).encode())
    assert st == 200 and "class GroupedQueryAttention" in ex["code"] and ".emb.weight" in ex["code"], st
    st, b = _asgi("POST", "/build", json.dumps(get_architecture("gemma3").model_dump()).encode())
    assert b["trainable"] is True and b["max_train_params"] == mlc.MAX_TRAIN_PARAMS
    print("[flagships] a real-size model builds and exports (inspect), but never trains or loads weights")


def test_next_token_dataset():
    spec = GraphSpec.model_validate(GQA_GRAPH)
    X, y = mlc.make_dataset("tokens", (8,), 32, spec.train, torch.device("cpu"))
    assert X.shape == (64, 8) and y.shape == (64,) and int(X.max()) < 32 and int(y.max()) < 32
    for row, nxt in zip(X.long().tolist(), y.tolist()):  # the label continues the repeating pattern
        period = next(p for p in (2, 3, 4) if all(row[i] == row[i - p] for i in range(p, 8)))
        assert nxt == row[8 - period], (row, nxt)
    X2, y2 = mlc.make_dataset("tokens", (8,), 32, spec.train, torch.device("cpu"))
    assert torch.equal(X, X2) and torch.equal(y, y2)
    for bad in ((1, 28, 28), (2,)):
        try:
            mlc.make_dataset("tokens", bad, 32, spec.train, torch.device("cpu"))
            raise AssertionError(bad)
        except ValueError as exc:
            assert "next token" in str(exc), exc
    print("[data] next token: each label continues its row's pattern; deterministic; wrong inputs explained")


def test_stage17_http():
    st, health = _asgi("GET", "/health")
    assert st == 200 and health["api"] == mlc.API_LEVEL >= 17, health
    st, lib = _asgi("GET", "/architectures")
    flags = {a["id"]: a for a in lib if a.get("scale") == "mini"}
    assert set(flags) == {f["id"] for f in FLAGSHIPS}, sorted(flags)
    ds = flags["deepseek_v3"]
    assert ds["real_id"] == "deepseek_v3_real" and ds["real_params"] == 671_000_000_000
    assert ds["hf_url"] == "https://huggingface.co/deepseek-ai/DeepSeek-V3" and "moe" in ds["tags"], ds
    st, real = _asgi("GET", "/architectures/qwen3_real")
    assert st == 200 and [n["type"] for n in real["nodes"]][:3] == ["Input", "Embedding", "Repeat"], real
    print(f"[stage17] HTTP: /health api 17; the library lists {len(flags)} flagships with real size + model card")


# --------------------------------------------------------------------------- #
# Stage 18 — your text: a GPT that learns your text at every position and writes.
# --------------------------------------------------------------------------- #
from mlconstructor import text as TX  # noqa: E402
from mlconstructor.datasets import split_rows  # noqa: E402


def _your_gpt(**train) -> dict:
    spec = get_architecture("your_gpt").model_dump()
    spec["train"].update(train)
    return spec


def test_text_vocab_and_windows():
    vocab = TX.text_vocab("abracadabra!", 10)
    assert vocab[:3] == ["a", "b", "r"] and set(vocab) == set("abracadabra!")  # most frequent first
    small = TX.text_vocab("abracadabra!", 3)
    assert small == ["a", "b", TX.UNKNOWN], small  # the rarest share one unknown token
    assert TX.decode_text(TX.encode_text("abcd", small), small) == "ab" + TX.UNKNOWN * 2
    text = mlc.EXAMPLE_TEXT
    X, y, groups = TX.text_windows(text, 64, 96, 10**6, 0)
    vocab = TX.text_vocab(text, 96)
    ids = TX.encode_text(text, vocab)
    assert X.shape == y.shape == (len(groups), 64)
    assert torch.equal(y[:, :-1], X[:, 1:].long())  # every position's label is the next character
    block = TX.text_block(len(text), 64)
    for row, g in zip(X.long().tolist()[::97], groups.tolist()[::97]):
        start = next(i for i in range(g * block, (g + 1) * block) if ids[i:i + 64] == row)
        assert start + 64 < (g + 1) * block, (start, g)  # a window and its labels stay inside the block
    X2, y2, g2 = TX.text_windows(text, 64, 96, 500, 0)
    assert len(X2) == 500 and torch.equal(X2, TX.text_windows(text, 64, 96, 500, 0)[0])  # capped, deterministic
    print(f"[text] vocabulary by frequency (+unknown), {len(X)} windows of 64 with next-character labels, in blocks")


def test_text_split_keeps_blocks_whole():
    X, y, groups = TX.text_windows(mlc.EXAMPLE_TEXT, 16, 96, 10**6, 3)
    tr, va, te = split_rows(y, groups, 0.2, 0.1, 3)
    parts = [set(groups[p].tolist()) for p in (tr, va, te)]
    assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])  # whole blocks
    assert sorted(tr.tolist() + va.tolist() + te.tolist()) == list(range(len(y)))
    n_blocks = len(set(groups.tolist()))
    assert len(parts[1]) == round(n_blocks * 0.2) and len(parts[2]) == round(n_blocks * 0.1), (len(parts[1]), n_blocks)
    yy = torch.randint(0, 4, (200,))
    assert all(torch.equal(a, b) for a, b in zip(split_rows(yy, None, 0.2, 0.1, 5), split_parts(yy, 0.2, 0.1, 5)))
    print(f"[text] the split keeps {n_blocks} blocks whole: no held-out window has a neighbour in training")


def test_your_gpt_learns_and_writes():
    spec = _your_gpt(samples=1600, epochs=3)
    st, res = _post("/train", spec)
    assert st == 200 and res["final_loss"] < res["loss_curve"][0] * 0.75, res["loss_curve"][:3]
    assert res["final_accuracy"] > 0.3 and res["test_samples"] > 0, res  # every position scored: it learns fast
    st, g = _post("/generate", {"spec": spec, "run_id": res["run_id"], "prompt": "The little", "length": 60,
                                "temperature": 0.0})
    vocab = set(mlc.EXAMPLE_TEXT)
    assert st == 200 and len(g["text"]) == 60 and set(g["text"]) <= vocab and g["context"] == 64, g
    again = _post("/generate", {"spec": spec, "run_id": res["run_id"], "prompt": "The little", "length": 60,
                                "temperature": 0.0})[1]
    assert again["text"] == g["text"]  # temperature 0: the most likely character every time
    hot = _post("/generate", {"spec": spec, "run_id": res["run_id"], "prompt": "", "length": 40, "seed": 2})[1]
    assert hot["prompt"] == mlc.EXAMPLE_TEXT[:64] and len(hot["text"]) == 40  # no prompt: the text's start
    st, test = _post("/test", {"spec": spec, "run_id": res["run_id"]})
    assert st == 200 and test["same_split"] is True and 0 < test["accuracy"] < 1, test
    other = _your_gpt(samples=1600, epochs=3, text=mlc.EXAMPLE_TEXT.replace("network", "machine"))
    st, unfair = _post("/test", {"spec": other, "run_id": res["run_id"]})
    assert st == 200 and unfair["same_split"] is False, unfair  # trained on another text: not a fair check
    print(f"[your gpt] learns the text at every position (train {res['final_accuracy']:.2f}) and writes: "
          f"«The little{g['text'][:40]}»")


def test_text_export_writes_like_the_app():
    spec = GraphSpec.model_validate(_your_gpt(samples=800, epochs=1))
    model, _ = mlc.build_model(spec, seed=0)
    code = mlc.generate_pytorch(spec, include_training=True)
    assert "TEXT = " in code and "def write" in code and "def split_rows" in code and "mlconstructor" not in code
    ns = _exec_exported(code)
    X, y, groups = ns["make_data"](800, (64,), 96, 0)
    Xa, ya, ga = mlc.make_rows("text", (64,), 96, spec.train, torch.device("cpu"))
    assert torch.equal(X, Xa) and torch.equal(y, ya) and torch.equal(groups, ga)  # the same rows
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "w.pt")
        with open(path, "wb") as f:
            f.write(W.weights_bytes(model, data=None))
        gen = ns["load_weights"](ns["Model"](), path)
    vocab = TX.text_vocab(spec.train.text, 96)
    model.eval()
    with torch.no_grad():  # the next character is what the LAST position scores highest
        scores = model(torch.tensor([TX.encode_text("Once", vocab)], dtype=torch.float32))[0, -1, : len(vocab)]
    assert TX.generate_text(model, vocab, 64, "Once", 1, 0.0) == vocab[int(scores.argmax())]
    app_text = TX.generate_text(model, vocab, 64, "Once", 30, 0.0)
    assert ns["write"](gen, "Once", 30, 0.0) == app_text  # the exported file writes exactly like the app
    print("[your gpt] the exported model.py builds the same rows and writes the same text")


def test_text_errors_are_clear():
    from mlconstructor.trainer import _fit_labels

    out, y = torch.randn(2, 5, 7), torch.randint(0, 7, (2, 5))
    o, t = _fit_labels(out, y)  # scores at every position learn from every next token
    assert o.shape == (10, 7) and torch.equal(t, y.reshape(-1))
    o, t = _fit_labels(out[:, -1], y)  # a last-position model learns from the last label
    assert o.shape == (2, 7) and torch.equal(t, y[:, -1])
    st, res = _post("/train", {**get_architecture("cnn").model_dump(),
                               "train": {"dataset": "text", "text": mlc.EXAMPLE_TEXT}})
    assert st == 422 and "needs an Input of token ids" in res["detail"], res
    st, res = _post("/train", _your_gpt(text="far too short"))
    assert st == 422 and "the text is too short" in res["detail"], res
    st, res = _post("/train", _your_gpt(text=""))
    assert st == 422 and "has no text yet" in res["detail"], res
    st, res = _post("/train", {**_your_gpt(), "train": {**_your_gpt()["train"], "dataset": "tokens"}})
    assert st == 422 and "scores every position" in res["detail"], res  # per-position output, one label per row
    small = _your_gpt(samples=400, epochs=1)
    for n in small["nodes"]:
        if n["id"] in ("emb", "head"):
            n["params"]["num_embeddings" if n["id"] == "emb" else "vocab"] = 20
    st, res = _post("/train", small)
    assert st == 200 and "the 24 rarest read as one" in res["warnings"][-1], res["warnings"]
    st, rr = _post("/train", {**get_architecture("qwen3").model_dump(),
                              "train": {**get_architecture("qwen3").model_dump()["train"], "samples": 300, "epochs": 1}})
    st, res = _post("/generate", {"spec": get_architecture("qwen3").model_dump(), "run_id": rr["run_id"]})
    assert st == 422 and "Generate continues text" in res["detail"], res
    st, b = _post("/build", _your_gpt())
    assert st == 200 and b["layers"][-1]["out_shape"] == [64, 96], b["layers"][-1]  # scores at every position
    print("[text] clear messages: wrong input, too short / no text, per-position vs one label, small vocabulary")


def test_stage18_http():
    st, health = _asgi("GET", "/health")
    assert st == 200 and health["api"] == mlc.API_LEVEL >= 18, health
    st, ex = _asgi("GET", "/text/example")
    assert st == 200 and ex["text"] == mlc.EXAMPLE_TEXT
    st, prev = _post("/data/preview", _your_gpt())
    assert st == 200 and prev["text_chars"] == len(mlc.EXAMPLE_TEXT) and prev["distinct_chars"] == 43, prev
    row = prev["rows"][0]
    assert len(row["text"]) == 64 and row["label_text"] == mlc.EXAMPLE_TEXT[64], row  # the window, then what follows
    st, lib = _asgi("GET", "/architectures")
    assert any(a["id"] == "your_gpt" for a in lib)
    print("[stage18] HTTP: /health api 18, /text/example, preview shows windows as text, Your GPT in the library")


def test_meta_builds_take_turns():
    # Two meta builds at once crashed a CPU-only backend (a native access violation in torch's
    # Python meta kernels): every meta build holds the lock, so they run one at a time.
    import threading as _th

    from mlconstructor import builder as B

    seen = []
    real = B._build_model

    def probe(spec):
        free = []

        def try_lock():  # from another thread: is the lock free right now?
            ok = B.META_LOCK.acquire(blocking=False)
            if ok:
                B.META_LOCK.release()
            free.append(ok)

        t = _th.Thread(target=try_lock)
        t.start()
        t.join()
        seen.append(free[0])
        return real(spec)

    B._build_model = probe
    try:
        mlc.build_model(get_architecture("gemma3"), meta=True)
        mlc.build_model(get_architecture("gemma3"))  # a CPU build does not need it
    finally:
        B._build_model = real
    assert seen == [False, True], seen  # taken during the meta build, free during the CPU one
    # ... and a meta build runs no kernel at all: initialization is skipped, so torch's meta
    # kernels (whose Python implementations crashed a CPU-only server) are never entered
    from torch.utils._python_dispatch import TorchDispatchMode

    class AtenOps(TorchDispatchMode):  # the operators that really reach torch's dispatcher
        def __init__(self):
            super().__init__()
            self.ops = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.ops.append(func.overloadpacket.__name__)
            return func(*args, **(kwargs or {}))

    with AtenOps() as meta_ops:
        model, build = mlc.build_model(get_architecture("gpt"), meta=True)  # Linear, Embedding, attention
    fills = [o for o in meta_ops.ops if o.endswith("_")]  # uniform_, normal_, fill_, zero_, copy_ ...
    assert not fills and all(p.is_meta for p in model.parameters()), sorted(set(fills))
    assert build.total_params == mlc.build_model(get_architecture("gpt"))[1].total_params
    with AtenOps() as cpu_ops:
        mlc.build_model(get_architecture("mlp"))
    assert any(o.endswith("_") for o in cpu_ops.ops), "a CPU build still initializes its weights"
    print("[meta] meta builds take turns and run no kernel (initialization skipped); CPU builds unchanged")


if __name__ == "__main__":
    test_layer_catalog()
    test_build_summary_shapes()
    test_train_loss_drops()
    test_shape_error_is_clear()
    test_block_repeat()
    test_branch_add_builds_and_trains()
    test_concat_merge()
    test_merge_needs_two_inputs()
    test_layer_without_input_says_how_to_connect()
    test_stage2_layers()
    test_conv_transpose_upsamples()
    test_lstm_sequence_trains()
    test_embedding_lstm_classifier_trains()
    test_gru_return_sequence_then_flatten()
    test_rnn_relu_and_bad_nonlinearity()
    test_embedding_needs_1d_sequence()
    test_multihead_attention_shape()
    test_transformer_encoder_trains()
    test_embedding_transformer_classifier_builds()
    test_attention_head_divisibility()
    test_suggest_flatten_for_conv_to_linear()
    test_apply_adapter_makes_graph_build()
    test_suggest_empty_for_valid_graph()
    test_suggest_flatten_for_sequence_to_linear()
    test_repeat_block_builds_and_trains()
    test_repeat_block_shape_changes_each_stamp()
    test_empty_block_rejected()
    test_block_rejects_inner_io_node()
    test_architecture_catalog()
    test_architecture_nodes_are_editable_layers()
    test_architectures_build_with_expected_shapes()
    test_each_architecture_trains()
    test_dataset_catalog()
    test_datasets_generate_valid_tensors()
    test_trainer_reports_accuracy_curve()
    test_shells_dataset_trains()
    test_data_preview_is_the_training_data()
    test_export_is_standalone_and_equivalent()
    test_export_file_trains_end_to_end()
    test_custom_dataset_in_catalog()
    test_custom_dataset_trains_on_your_rows()
    test_custom_dataset_validation()
    test_custom_data_preview_echoes_your_rows()
    test_export_custom_embeds_your_data()
    test_export_saves_state_dict_and_onnx()
    test_real_datasets_in_catalog()
    test_real_dataset_shape_validation()
    test_train_iter_streams_events()
    test_checkpoint_save_and_resume()
    test_device_default_cpu_and_cuda_optin()
    test_probe_runs_on_the_model_device()
    test_project_round_trip_is_exact()
    test_project_opens_a_bare_graph_spec()
    test_project_warnings_and_repairs()
    test_project_rejects_bad_files()
    test_server_project_store()
    test_project_http_endpoints()
    test_screenshot_graph_builds_and_trains()
    test_connection_auto_choices()
    test_every_transform_matches_its_module()
    test_connection_choice_and_clear_errors()
    test_merge_inputs_are_matched_automatically()
    test_architectures_unchanged_by_connections()
    test_export_reproduces_connections_exactly()
    test_connection_options_for_the_inspector()
    test_project_format_2_keeps_connection_settings()
    test_connection_http_endpoints()
    test_split_is_stratified_deterministic_and_safe()
    test_blobs_generalises_to_held_out_rows()
    test_inseparable_data_keeps_an_error()
    test_overfitting_shows_on_held_out_rows()
    test_validation_off_is_the_original_loop()
    test_tiny_custom_sets_fall_back_clearly()
    test_stream_and_checkpoint_use_validation()
    test_preview_and_export_share_the_split()
    test_stage15_http()
    test_same_seed_same_weights_and_numbers()
    test_three_part_split_keeps_its_guarantees()
    test_weights_download_upload_and_test()
    test_final_test_reports_honestly()
    test_exported_script_runs_app_weights()
    test_uploads_are_checked_and_safe()
    test_training_starts_from_saved_weights()
    test_runs_are_pruned()
    test_stage16_http()
    test_flagship_real_sizes_match_hugging_face()
    test_flagship_minis_train_on_next_token()
    test_flagship_minis_export_exactly()
    test_llm_layers_behave_and_explain()
    test_real_scale_is_inspect_only()
    test_next_token_dataset()
    test_stage17_http()
    test_text_vocab_and_windows()
    test_text_split_keeps_blocks_whole()
    test_your_gpt_learns_and_writes()
    test_text_export_writes_like_the_app()
    test_text_errors_are_clear()
    test_stage18_http()
    test_meta_builds_take_turns()
    print(
        "\nALL CHECKS PASSED (v1 + Stage 1 + Stage 2 + Stage 3 + Stage 4 + Stage 5 + Stage 6 "
        "+ Stage 7 + Stage 8 + Stage 9 + Stage 10 offline + Stage 11 + Stage 12 + Stage 13 + Stage 14 "
        "+ Stage 15 + Stage 16 + Stage 17 + Stage 18)"
    )
