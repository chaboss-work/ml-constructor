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


if __name__ == "__main__":
    test_layer_catalog()
    test_build_summary_shapes()
    test_train_loss_drops()
    test_shape_error_is_clear()
    test_block_repeat()
    print("\nALL M1 CHECKS PASSED")
