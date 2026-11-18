"""The training loop and the data preview.

Tiny and self-contained: it trains the assembled model on a **synthetic**
classification task so the whole pipeline (graph → model → training → result)
can be proven with no dataset download and no GPU. The data now comes from the
:mod:`~mlconstructor.datasets` registry (Stage 8) — the default ``blobs`` task is
the original separable Gaussian clusters, kept byte-for-byte — so the trainer and
the ``/data/preview`` modal always agree on what the network sees.

The number of classes is read from the model's own output width (a dry probe
forward), so it always matches the network the user built. Alongside the per-step
loss curve the loop reports a per-epoch **accuracy curve** (Stage 8).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .builder import build_model
from .datasets import get_dataset, make_dataset
from .schema import DataPreview, DataSampleRow, GraphSpec, TrainResult, TrainSpec
from .shapes import Shape, ShapeError, numel

_PREVIEW_MAX_VALUES = 12  # feature values shown per sample row before truncating


def _infer_num_classes(model: nn.Module, in_shape: Shape) -> int:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        probe = model(torch.zeros(2, *in_shape))  # batch=2 keeps BatchNorm happy
    if was_training:
        model.train()
    if probe.dim() != 2:
        raise ShapeError(
            f"network output must be 2-D (batch, classes) for the v1 classification "
            f"task; got {tuple(probe.shape)} — add a Flatten + Linear so the last "
            f"layer outputs class logits"
        )
    return int(probe.shape[1])


def _input_shape(build) -> Shape:
    """The graph's per-sample input shape. The sole source (Input) has in-degree
    0, so the topological order lists it first."""
    first = build.layers[0]
    if not build.layers or first.out_shape is None:
        raise ShapeError("graph has no Input node to take a shape from")
    return tuple(int(d) for d in first.out_shape)


def train_model(model: nn.Module, in_shape: Shape, spec: TrainSpec) -> TrainResult:
    torch.manual_seed(spec.seed)
    device = torch.device("cpu")  # v1: CPU, tiny — no GPU window needed
    model = model.to(device)

    in_shape = tuple(int(d) for d in in_shape)
    num_classes = _infer_num_classes(model, in_shape)
    X, y = make_dataset(spec.dataset, in_shape, num_classes, spec, device)

    n = X.shape[0]
    bs = min(spec.batch_size, n)
    opt = (
        torch.optim.Adam(model.parameters(), lr=spec.lr)
        if spec.optimizer == "adam"
        else torch.optim.SGD(model.parameters(), lr=spec.lr)
    )
    loss_fn = nn.CrossEntropyLoss()

    def _full_set_accuracy() -> float:
        """Accuracy over the whole set, in eval mode (so Dropout/BatchNorm behave).

        Run under ``no_grad`` in eval mode, it consumes no RNG and touches no
        gradients, so inserting it between epochs leaves the training trajectory
        (and every existing result) exactly as it was."""
        was_training = model.training
        model.eval()
        with torch.no_grad():
            acc = float((model(X).argmax(1) == y).float().mean())
        if was_training:
            model.train()
        return acc

    loss_curve: list[float] = []
    acc_curve: list[float] = []
    step = 0
    stop = False
    model.train()
    for _epoch in range(spec.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
            loss_curve.append(float(loss.detach()))
            step += 1
            if spec.steps and step >= spec.steps:
                stop = True
                break
        if not stop:  # a full epoch finished — record its accuracy
            acc_curve.append(_full_set_accuracy())
        if stop:
            break

    model.eval()
    with torch.no_grad():
        logits = model(X)
        final_loss = float(loss_fn(logits, y))
        accuracy = float((logits.argmax(1) == y).float().mean())
    if not acc_curve:  # step cap stopped us before any epoch completed
        acc_curve.append(accuracy)

    total_params = sum(p.numel() for p in model.parameters())
    return TrainResult(
        ok=True,
        loss_curve=loss_curve,
        acc_curve=acc_curve,
        final_loss=final_loss,
        final_accuracy=accuracy,
        steps=step,
        device=str(device),
        total_params=int(total_params),
    )


def data_preview(spec: GraphSpec, rows: int = 6) -> DataPreview:
    """A few real rows of the training set for the graph the user built, plus its
    format (Stage 8's data modal). Uses the *same* generator and seed as training,
    so these are exactly the first rows the model will be trained on."""
    model, build = build_model(spec)
    in_shape = _input_shape(build)
    num_classes = _infer_num_classes(model, in_shape)
    device = torch.device("cpu")

    ds = get_dataset(spec.train.dataset)
    k = max(1, min(int(rows), 50))
    # Draw from the SAME generator, seed and sample count as training (the same
    # spec), so these are exactly the first rows the model will be trained on — the
    # noise is drawn after the labels, so a smaller draw would NOT line up.
    X, y = ds.make(in_shape, num_classes, spec.train, device)
    k = min(k, X.shape[0])

    flat = X[:k].reshape(k, -1)
    out_rows: list[DataSampleRow] = []
    for i in range(k):
        vals = flat[i].tolist()
        out_rows.append(
            DataSampleRow(
                label=int(y[i]),
                values=[round(float(v), 3) for v in vals[:_PREVIEW_MAX_VALUES]],
                truncated=len(vals) > _PREVIEW_MAX_VALUES,
                total=len(vals),
            )
        )

    return DataPreview(
        dataset=ds.id,
        dataset_name=ds.name,
        description=ds.description,
        input_shape=list(in_shape),
        dtype=str(X.dtype).replace("torch.", ""),
        num_classes=num_classes,
        num_samples=int(X.shape[0]),  # actual rows (synthetic = samples; custom = your row count)
        feature_count=numel(in_shape),
        rows=out_rows,
    )
