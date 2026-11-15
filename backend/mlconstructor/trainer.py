"""The v1 training loop.

Tiny and self-contained: it trains the assembled model on a **synthetic**
classification task so the whole pipeline (graph → model → training → result)
can be proven with no dataset download and no GPU. The task is separable
Gaussian clusters — one random prototype per class plus noise — so any sane
network learns it quickly and the loss provably drops.

The number of classes is read from the model's own output width (a dry probe
forward), so it always matches the network the user built.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .schema import TrainResult, TrainSpec
from .shapes import Shape, ShapeError


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


def _synthetic_classification(in_shape: Shape, num_classes: int, spec: TrainSpec, device):
    """Separable Gaussian clusters: one random prototype per class + noise."""
    g = torch.Generator().manual_seed(spec.seed)
    n = spec.samples
    prototypes = torch.randn(num_classes, *in_shape, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    noise = torch.randn(n, *in_shape, generator=g)
    X = prototypes[y] + 0.5 * noise  # signal std 1.0 vs noise std 0.5 -> well separated
    return X.to(device), y.to(device)


def train_model(model: nn.Module, in_shape: Shape, spec: TrainSpec) -> TrainResult:
    torch.manual_seed(spec.seed)
    device = torch.device("cpu")  # v1: CPU, tiny — no GPU window needed
    model = model.to(device)

    in_shape = tuple(int(d) for d in in_shape)
    num_classes = _infer_num_classes(model, in_shape)
    X, y = _synthetic_classification(in_shape, num_classes, spec, device)

    n = X.shape[0]
    bs = min(spec.batch_size, n)
    opt = (
        torch.optim.Adam(model.parameters(), lr=spec.lr)
        if spec.optimizer == "adam"
        else torch.optim.SGD(model.parameters(), lr=spec.lr)
    )
    loss_fn = nn.CrossEntropyLoss()

    loss_curve: list[float] = []
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
        if stop:
            break

    model.eval()
    with torch.no_grad():
        logits = model(X)
        final_loss = float(loss_fn(logits, y))
        accuracy = float((logits.argmax(1) == y).float().mean())

    total_params = sum(p.numel() for p in model.parameters())
    return TrainResult(
        ok=True,
        loss_curve=loss_curve,
        final_loss=final_loss,
        final_accuracy=accuracy,
        steps=step,
        device=str(device),
        total_params=int(total_params),
    )
