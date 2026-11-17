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
    print("\nALL CHECKS PASSED (v1 + Stage 1 + Stage 2 + Stage 3 + Stage 4)")
