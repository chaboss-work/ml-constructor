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

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ on path

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
    spec = GraphSpec.model_validate(bad)
    try:
        mlc.build_model(spec)
    except ShapeError as exc:
        assert "fc" in str(exc)
        print(f"[validate] caught clear ShapeError: {exc}")
        return
    raise AssertionError("expected a ShapeError for a Linear fed a 3-D feature map")


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
        assert "sum" in str(exc) and ">= 2" in str(exc)
        print(f"[validate] Add with 1 input rejected: {exc}")
        return
    raise AssertionError("expected a ShapeError for a merge node with one input")


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
    model, build = mlc.build_model(spec)
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
        mlc.build_model(spec)
    except ShapeError as exc:
        assert "emb" in str(exc)
        print(f"[validate] Embedding fed 2-D rejected: {exc}")
        return
    raise AssertionError("expected a ShapeError for an Embedding fed a 2-D shape")


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
    spec = GraphSpec.model_validate(CONV_NO_FLATTEN)
    props = suggest_for_graph(spec)
    assert len(props) == 1, props
    p = props[0]
    assert p.adapter_type == "Flatten" and p.source_id == "conv" and p.target_id == "fc"
    assert p.in_shape == [4, 26, 26] and p.out_shape == [4 * 26 * 26]
    print(f"[suggest] {p.reason}")


def test_apply_adapter_makes_graph_build():
    spec = GraphSpec.model_validate(CONV_NO_FLATTEN)
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
    ids = {d.id for d in ds}
    assert {"blobs", "shells"} <= ids, ids
    for d in ds:
        assert d.name and d.description and d.kind == "synthetic"
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


if __name__ == "__main__":
    test_layer_catalog()
    test_build_summary_shapes()
    test_train_loss_drops()
    test_shape_error_is_clear()
    test_block_repeat()
    test_branch_add_builds_and_trains()
    test_concat_merge()
    test_merge_needs_two_inputs()
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
    print(
        "\nALL CHECKS PASSED "
        "(v1 + Stage 1 + Stage 2 + Stage 3 + Stage 4 + Stage 5 + Stage 6 + Stage 7 + Stage 8)"
    )
