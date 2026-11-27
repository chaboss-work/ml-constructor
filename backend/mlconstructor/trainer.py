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

Since Stage 15 a share of the rows (``val_fraction``, 20 % by default) is **held
out**: the network trains on the rest only, and the held-out rows measure how it
does on data it never saw — on separable synthetic sets the training accuracy
alone is always ~100 %, which says nothing. ``val_fraction = 0`` is the original
loop, byte for byte.

Since Stage 16 a further share (``test_fraction``, 10 % by default) is kept for a
**final test**: it is neither trained on nor measured during training — only
:func:`test_run` measures it, on the weights a run kept (:func:`save_trained`) or
on uploaded ones (:func:`adopt_weights`). With both shares at 0 the loop is the
original one.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import build_model
from .datasets import canonical_dataset_id, get_dataset, make_rows, split_rows, text_stats
from .generative import (
    EMA,
    UNCOND_PROB,
    ddim_sample,
    ema_decay,
    guide,
    image_items,
    input_roles,
    kl_total,
    lr_factor,
    noise_schedule,
    noisy_batch,
    recon_loss,
    resolve_task,
    set_prior,
)
from .gpu import CUDA_HIDDEN, cuda_available, hidden_message
from .schema import (
    DataPreview,
    DataSampleRow,
    DrawResult,
    GalleryImage,
    GenerateResult,
    GraphSpec,
    RunInputInfo,
    RunResult,
    TestResult,
    TrainResult,
    TrainSpec,
    WeightsInfo,
)
from .shapes import GraphError, Shape, ShapeError, numel
from .text import UNKNOWN, generate_text, text_vocab
from .weights import arch_key, data_key, fit_state_dict, load_run, read_weights, save_run, weights_bytes

_PREVIEW_MAX_VALUES = 12  # feature values shown per sample row before truncating
# Checkpoints live under the gitignored data cache (Stage 11).
_CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "checkpoints")
#: Stage 15 — below this many held-out rows each one moves validation accuracy a lot.
_SMALL_VAL = 20
#: Training minus validation accuracy above this reads as overfitting; validation at
#: or above _GOOD_VAL (with a small gap) reads as "generalises".
_OVERFIT_GAP = 0.10
_GOOD_VAL = 0.90
#: Stage 16 — test accuracy this far below validation reads as "tuned to the validation rows".
_TEST_GAP = 0.05
#: Stage 17 — the largest network trained (or loaded) here: real-scale flagships are
#: built on the meta device to inspect; their mini versions train.
MAX_TRAIN_PARAMS = 25_000_000


def _human(n: int) -> str:
    for size, word in ((10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (10**3, "thousand")):
        if n >= size:
            v = n / size
            return f"{v:.1f} {word}".replace(".0 ", " ")
    return str(n)


def count_params(spec: GraphSpec) -> int:
    """The network's parameter count, sized on the meta device (no memory taken)."""
    return build_model(spec, meta=True)[1].total_params


def train_block_reason(spec: GraphSpec, build) -> str | None:
    """Stage 19 — why this network cannot be trained (or tested) here yet, in plain
    words; None when it can. Training here teaches one score per class (or the next
    token) from one input: a network with several inputs, or one whose output is a
    picture, a latent or points, builds, runs once and exports instead."""
    return resolve_task(spec, build).reason


def check_trainable(spec: GraphSpec, what: str = "train") -> int:
    """Refuse a network too big for this server before a single weight is made —
    and (Stage 19) one that cannot be trained here, to train or test it."""
    build = build_model(spec, meta=True)[1]
    n = build.total_params
    if n > MAX_TRAIN_PARAMS:
        raise GraphError(
            f"this network has {n:,} parameters ({_human(n)}), far too big to {what} on this server (the limit is "
            f"{_human(MAX_TRAIN_PARAMS)}). It is here to inspect: every layer and shape of the real model. The "
            "flagships in the library come with a mini version that trains here; or shrink the layers",
            code="too_big", facts={"params": n, "limit": MAX_TRAIN_PARAMS},
        )
    if what in ("train", "test"):
        plan = resolve_task(spec, build)
        if plan.reason:
            raise GraphError(plan.reason, code=plan.code or "task_inputs")
    return n


def _plural(n: int, word: str, many: str | None = None) -> str:
    return f"{n} {word if n == 1 else (many or word + 's')}"


def _split_warnings(spec: TrainSpec, n_rows: int, n_val: int, n_test: int, y: torch.Tensor) -> list[str]:
    """What to know about the train / validation / test split of this run (Stages 15–16)."""
    out: list[str] = []
    if not spec.val_fraction:
        out.append(
            "Validation is off (0 %): accuracy is measured on the rows the network trained on, so it "
            "shows how well it memorised them, not how it does on new data."
        )
    elif n_val == 0:
        classes = int(torch.unique(y).numel())
        out.append(
            f"Too few rows to hold any out for validation ({_plural(n_rows, 'row')}, "
            f"{_plural(classes, 'class', 'classes')}; every class needs a row to train on), so accuracy is "
            "measured on the training rows themselves. Add more rows to see how it does on new data."
        )
    elif n_val < _SMALL_VAL:
        out.append(
            f"Only {_plural(n_val, 'row')} held out for validation, so each one moves validation accuracy "
            f"by {100 / n_val:.0f} %: too coarse to tell whether it generalises. Add rows for a reliable verdict."
        )
    if spec.test_fraction and n_test == 0:
        out.append(
            "Too few rows to keep a test part aside as well, so Test has nothing to measure. Add rows, or set "
            "Final test to Off."
        )
    return out


def _text_warnings(spec: TrainSpec, num_classes: int) -> list[str]:
    """Stage 18 — "Your text" with more characters than the network scores."""
    if canonical_dataset_id(spec.dataset) != "text" or not spec.text:
        return []
    distinct = len(set(spec.text))
    if distinct <= num_classes:
        return []
    return [
        f"Your text uses {distinct} different characters, but the network scores {num_classes}: the "
        f"{distinct - num_classes + 1} rarest read as one \"unknown\" character. Press \"Fit vocabulary\" in Data to "
        "give every character a token of its own."
    ]


def _fit_verdict(train_acc: float, val_acc: float | None, n_val: int) -> tuple[str, str | None]:
    """Training vs validation accuracy in plain words (Stage 15)."""
    if val_acc is None or n_val < _SMALL_VAL:  # nothing (or too little) held out — the warnings say why
        return "unknown", None
    t, v = round(100 * train_acc), round(100 * val_acc)
    if train_acc - val_acc > _OVERFIT_GAP:
        return "overfits", (
            f"Overfitting: {t}% on the rows it trained on, but {v}% on rows it never saw. It memorises "
            "its training rows instead of learning the rule: try fewer epochs, a Dropout layer or more samples."
        )
    if val_acc >= _GOOD_VAL:
        return "generalises", (
            f"Generalises: {v}% on rows it never saw, close to the {t}% on its training rows. "
            "It learned the rule, not just the rows."
        )
    return "underfits", (
        f"Not learned yet: {t}% on its training rows, {v}% on new ones. More epochs, a higher learning "
        "rate or a bigger network may help, unless the data itself mixes the classes (identical inputs "
        "with different labels), which no network can fully separate."
    )


def _infer_num_classes(model: nn.Module, in_shape: Shape) -> int:
    was_training = model.training
    model.eval()
    dev = next((p.device for p in model.parameters()), torch.device("cpu"))
    with torch.no_grad():
        probe = model(torch.zeros(2, *in_shape, device=dev))  # batch=2 keeps BatchNorm happy
    if was_training:
        model.train()
    if probe.dim() not in (2, 3):
        raise ShapeError(
            f"network output must be 2-D (batch, classes) for the v1 classification "
            f"task; got {tuple(probe.shape)} — add a Flatten + Linear so the last "
            f"layer outputs class logits",
            code="output_not_classes", facts={"got": list(probe.shape[1:])},
        )
    return int(probe.shape[-1])  # Stage 18: (batch, L, vocab) scores every position


def _fit_labels(out: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage 18 — match the scores to the labels: a language model scoring every
    position learns from every next token; one scoring only the last position (or a
    classifier) from the last label."""
    if out.dim() == 3:
        if y.dim() != 2:
            raise GraphError(
                "the network scores every position, but this dataset gives one label per row: switch the "
                "Output's \"Every position\" off, or train on Your text",
                code="every_position",
            )
        return out.reshape(-1, out.shape[-1]), y.reshape(-1)
    return out, (y[:, -1] if y.dim() == 2 else y)


def _input_shape(build) -> Shape:
    """The graph's per-sample input shape. The sole source (Input) has in-degree
    0, so the topological order lists it first."""
    inputs = list(getattr(build, "inputs", None) or [])
    if len(inputs) > 1:  # Stage 19
        names = ", ".join(i.name or i.id for i in inputs)
        raise GraphError(
            f"this network takes {len(inputs)} inputs ({names}); the data, training and the test work with one input "
            "so far. Run once feeds it random inputs",
            code="task_inputs",
        )
    first = build.layers[0]
    if not build.layers or first.out_shape is None:
        raise ShapeError("graph has no Input node to take a shape from", code="no_input")
    return tuple(int(d) for d in first.out_shape)


def _checkpoint_key(model: nn.Module) -> str:
    """A stable key per architecture — a hash of parameter names and shapes — so a
    checkpoint only ever reloads into the same network."""
    return arch_key(model.state_dict())


def _checkpoint_path(model: nn.Module) -> str:
    return os.path.join(_CKPT_DIR, _checkpoint_key(model) + ".pt")


def _device(want: str = "cpu") -> "torch.device":
    """Where to train (Stage 12). ``"cpu"`` (the default) never touches the GPU — the
    card is shared on this host, so GPU use is an explicit opt-in. ``"cuda"`` asks
    for the GPU and fails clearly if there is none; ``"auto"`` takes the GPU when
    available, else the CPU. A backend started with every GPU hidden never uses one
    (:mod:`~mlconstructor.gpu`)."""
    if want == "cuda":
        if CUDA_HIDDEN:
            raise GraphError(hidden_message(), code="no_gpu")
        if not cuda_available():
            raise GraphError(
                "GPU (CUDA) training was requested, but no CUDA device is available on this backend",
                code="no_gpu",
            )
        return torch.device("cuda")
    if want == "auto":
        return torch.device("cuda" if cuda_available() else "cpu")
    return torch.device("cpu")


def train_iter(model: nn.Module, in_shape: Shape, spec: TrainSpec) -> Iterator[dict]:
    """Train, yielding progress as it happens (Stage 11 live streaming).

    Events: ``{"type":"start", …}`` once; ``{"type":"step", step, loss}`` per
    optimizer step; ``{"type":"epoch", epoch, acc, loss, val_acc, val_loss}`` per
    finished epoch (``acc`` on the training rows, ``val_*`` on the held-out rows or
    None); and finally ``{"type":"done", "result": TrainResult}``. :func:`train_model`
    just drains this, so the streaming and plain ``/train`` paths share one loop
    (hence one trajectory — with ``checkpoint``/``resume`` off and ``val_fraction = 0``
    it is the original loop, byte for byte)."""
    plan = getattr(model, "task_plan", None)
    if plan is not None and plan.task in ("reconstruction", "diffusion"):  # Stage 20
        yield from _generative_iter(model, plan, spec)
        return
    torch.manual_seed(spec.seed)
    device = _device(spec.device)
    model = model.to(device)

    in_shape = tuple(int(d) for d in in_shape)
    num_classes = _infer_num_classes(model, in_shape)
    X, y, groups = make_rows(spec.dataset, in_shape, num_classes, spec, device)

    # Stages 15–16 — hold rows out for validation and for the final test; train on the rest only
    # (Stage 18: rows that overlap, like the windows of a text, are split block by block).
    tr_idx, va_idx, te_idx = split_rows(y, groups, spec.val_fraction, spec.test_fraction, spec.seed)
    n_rows, n_val, n_test = int(X.shape[0]), int(va_idx.numel()), int(te_idx.numel())
    warnings = _split_warnings(spec, n_rows, n_val, n_test, y) + _text_warnings(spec, num_classes)
    Xv = yv = None
    if n_val or n_test:
        if n_val:
            Xv, yv = X[va_idx.to(X.device)], y[va_idx.to(y.device)]
        X, y = X[tr_idx.to(X.device)], y[tr_idx.to(y.device)]  # the test rows are not even kept here
    # else: nothing held out — train (and measure) on every row, exactly as before Stage 15

    resumed = False
    if spec.resume and os.path.exists(_checkpoint_path(model)):
        try:
            model.load_state_dict(torch.load(_checkpoint_path(model), map_location=device))
            resumed = True
        except Exception:  # architecture changed since the checkpoint — start fresh
            resumed = False

    n = X.shape[0]
    bs = min(spec.batch_size, n)
    opt = (
        torch.optim.Adam(model.parameters(), lr=spec.lr)
        if spec.optimizer == "adam"
        else torch.optim.SGD(model.parameters(), lr=spec.lr)
    )
    loss_fn = nn.CrossEntropyLoss()
    total_params = int(sum(p.numel() for p in model.parameters()))

    yield {
        "type": "start", "device": str(device), "resumed": resumed,
        "num_classes": num_classes, "samples": n_rows, "total_params": total_params,
        "train_samples": int(n), "val_samples": n_val, "test_samples": n_test,
    }

    def _evaluate(Xs: torch.Tensor, ys: torch.Tensor) -> tuple[float, float]:
        """Loss and accuracy on a set in eval mode (no RNG, no grads), so measuring
        between epochs leaves the training trajectory exactly as it was."""
        was_training = model.training
        model.eval()
        with torch.no_grad():
            logits, target = _fit_labels(model(Xs), ys)
            loss_v = float(loss_fn(logits, target))
            acc_v = float((logits.argmax(-1) == target).float().mean())
        if was_training:
            model.train()
        return loss_v, acc_v

    def _save() -> str:
        os.makedirs(_CKPT_DIR, exist_ok=True)
        torch.save(model.state_dict(), _checkpoint_path(model))
        return _checkpoint_key(model)

    loss_curve: list[float] = []
    acc_curve: list[float] = []
    val_acc_curve: list[float] = []
    val_loss_curve: list[float] = []
    step = 0
    stop = False
    best_acc = -1.0
    ckpt_key: str | None = None
    model.train()
    for epoch in range(spec.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss = loss_fn(*_fit_labels(model(X[idx]), y[idx]))
            loss.backward()
            opt.step()
            lv = float(loss.detach())
            loss_curve.append(lv)
            step += 1
            yield {"type": "step", "step": step, "loss": lv}
            if spec.steps and step >= spec.steps:
                stop = True
                break
        if not stop:  # a full epoch finished — record its accuracy (and maybe checkpoint)
            acc = _evaluate(X, y)[1]
            acc_curve.append(acc)
            val_loss = val_acc = None
            if Xv is not None:
                val_loss, val_acc = _evaluate(Xv, yv)
                val_acc_curve.append(val_acc)
                val_loss_curve.append(val_loss)
            yield {
                "type": "epoch", "epoch": epoch + 1, "acc": acc,
                "loss": loss_curve[-1] if loss_curve else None,
                "val_acc": val_acc, "val_loss": val_loss,
            }
            score = acc if val_acc is None else val_acc  # keep the weights that do best on new data
            if spec.checkpoint and score > best_acc:
                best_acc = score
                ckpt_key = _save()
        if stop:
            break

    model.eval()
    final_loss, accuracy = _evaluate(X, y)
    final_val_loss = final_val_accuracy = None
    if Xv is not None:
        final_val_loss, final_val_accuracy = _evaluate(Xv, yv)
    if not acc_curve:  # step cap stopped us before any epoch completed
        acc_curve.append(accuracy)
        if final_val_accuracy is not None:
            val_acc_curve.append(final_val_accuracy)
            val_loss_curve.append(final_val_loss)
    if spec.checkpoint and ckpt_key is None:  # stopped before any epoch finished — save final
        ckpt_key = _save()
    if device.type == "cuda":  # hand the shared card back: weights, data and cached blocks
        model.to("cpu")
        del X, y, Xv, yv
        torch.cuda.empty_cache()

    final_score = accuracy if final_val_accuracy is None else final_val_accuracy
    fit, fit_note = _fit_verdict(accuracy, final_val_accuracy, n_val)
    result = TrainResult(
        ok=True,
        loss_curve=loss_curve,
        acc_curve=acc_curve,
        final_loss=final_loss,
        final_accuracy=accuracy,
        steps=step,
        device=str(device),
        total_params=total_params,
        checkpoint=ckpt_key,
        best_accuracy=(best_acc if best_acc >= 0 else final_score),
        resumed=resumed,
        val_acc_curve=val_acc_curve,
        val_loss_curve=val_loss_curve,
        final_val_accuracy=final_val_accuracy,
        final_val_loss=final_val_loss,
        train_samples=int(n),
        val_samples=n_val,
        fit=fit,
        fit_note=fit_note,
        warnings=warnings,
        test_samples=n_test,
    )
    yield {"type": "done", "result": result}


def train_model(model: nn.Module, in_shape: Shape, spec: TrainSpec) -> TrainResult:
    """Train and return the final result (drains :func:`train_iter`)."""
    result: TrainResult | None = None
    for event in train_iter(model, in_shape, spec):
        if event["type"] == "done":
            result = event["result"]
    assert result is not None, "train_iter did not yield a result"
    return result


def data_preview(spec: GraphSpec, rows: int = 6) -> DataPreview:
    """A few real rows of the training set for the graph the user built, plus its
    format (Stage 8's data modal). Uses the *same* generator and seed as training,
    so these are exactly the first rows the model will be trained on."""
    model, build = build_model(spec, meta=True)  # only the shapes are needed: no weights made (Stage 17)
    plan = model.task_plan
    if plan.task in ("reconstruction", "diffusion"):  # Stage 20: the pictures it learns from
        in_shape = plan.in_shapes[plan.image_input]
        num_classes = int(spec.train.num_classes)
    else:
        in_shape = _input_shape(build)
        # the class count is the Output's last dimension — no forward pass on meta tensors needed
        out = next((s for s in reversed(build.layers) if s.type == "Output" and s.out_shape), None)
        num_classes = int(out.out_shape[-1]) if out else _infer_num_classes(build_model(spec)[0], in_shape)
    device = torch.device("cpu")

    ds = get_dataset(spec.train.dataset)
    k = max(1, min(int(rows), 50))
    # Draw from the SAME generator, seed and sample count as training (the same
    # spec), so these are exactly the first rows of the training data — the noise
    # is drawn after the labels, so a smaller draw would NOT line up.
    X, y, groups = make_rows(spec.train.dataset, in_shape, num_classes, spec.train, device)
    k = min(k, X.shape[0])
    # Stages 15–16 — the same split as training: mark the validation and test rows.
    tr_idx, va_idx, te_idx = split_rows(y, groups, spec.train.val_fraction, spec.train.test_fraction, spec.train.seed)
    is_text = ds.id == "text"
    vocab = text_vocab(spec.train.text or "", num_classes) if is_text else None
    held = torch.zeros(int(X.shape[0]), dtype=torch.bool)
    held[va_idx] = True
    tested = torch.zeros(int(X.shape[0]), dtype=torch.bool)
    tested[te_idx] = True

    flat = X[:k].reshape(k, -1)
    out_rows: list[DataSampleRow] = []
    for i in range(k):
        vals = flat[i].tolist()
        out_rows.append(
            DataSampleRow(
                label=int(y[i]) if y.dim() == 1 else int(y[i, -1]),
                values=[round(float(v), 3) for v in vals[:_PREVIEW_MAX_VALUES]],
                truncated=len(vals) > _PREVIEW_MAX_VALUES,
                total=len(vals),
                split="test" if bool(tested[i]) else ("validation" if bool(held[i]) else "train"),
                text="".join(vocab[int(v)] for v in X[i].tolist()) if vocab else None,
                label_text=vocab[int(y[i] if y.dim() == 1 else y[i, -1])] if vocab else None,
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
        train_samples=int(tr_idx.numel()),
        val_samples=int(va_idx.numel()),
        test_samples=int(te_idx.numel()),
        **(text_stats(spec.train, num_classes) if is_text else {}),
    )


# --------------------------------------------------------------------------- #
# Stage 16 — keep the trained weights, start from saved ones, the final test
# --------------------------------------------------------------------------- #
def save_trained(model: nn.Module, in_shape: Shape, spec: TrainSpec, result: TrainResult) -> TrainResult:
    """Keep the trained weights on the server as a run and put its id into the
    result (download, Test, or the next run's start). A storage failure never
    fails the training — the result says what could not be kept."""
    try:
        model = model.to("cpu")
        plan = getattr(model, "task_plan", None)
        if plan is not None and plan.task in ("reconstruction", "diffusion"):  # Stage 20: the pictures' rows
            in_shape = plan.in_shapes[plan.image_input]
            num_classes = int(spec.num_classes)
        else:
            num_classes = _infer_num_classes(model, tuple(int(d) for d in in_shape))
        metrics = {
            "dataset": canonical_dataset_id(spec.dataset),
            "final_accuracy": result.final_accuracy,
            "final_val_accuracy": result.final_val_accuracy,
            "final_loss": result.final_loss,
            "steps": result.steps,
        }
        raw = weights_bytes(model, data=data_key(spec, in_shape, num_classes), metrics=metrics)
        result.run_id = save_run(raw)
    except OSError as exc:
        result.warnings.append(
            f"The trained weights could not be kept on the server ({exc.strerror or exc}), so Save weights "
            "and Test are unavailable for this run."
        )
    return result


def start_from_run(model: nn.Module, spec: TrainSpec) -> TrainSpec:
    """Load the weights of ``spec.init_run`` into the model (they must fit the
    graph) and turn ``resume`` off, so a checkpoint cannot overwrite them."""
    if not spec.init_run:
        return spec
    payload = load_run(spec.init_run)
    mapped, _how = fit_state_dict(model, payload["state_dict"])
    model.load_state_dict(mapped)
    return spec.model_copy(update={"resume": False})


def _test_reading(test_acc: float, val_acc: float | None, n_test: int, same: bool | None) -> tuple[list[str], str | None]:
    """The final test in plain words: is it a fair check, and how does it compare."""
    warnings: list[str] = []
    if same is False:
        warnings.append(
            "These weights were trained on other data or another split, so some of the test rows may have been "
            "among their training rows: this test is not a fair check. Train again with the current settings, "
            "then test."
        )
    elif same is None:
        warnings.append(
            "These weights carry no record of the data they were trained on, so there is no telling whether the "
            "test rows were among it."
        )
    if n_test < _SMALL_VAL:
        warnings.append(
            f"Only {_plural(n_test, 'row')} in the test part, so each one moves test accuracy by "
            f"{100 / n_test:.0f} %: too coarse for a reading. Add rows for a reliable test."
        )
        return warnings, None
    t = round(100 * test_acc)
    if val_acc is None:
        return warnings, f"{t}% on the test rows: rows used neither for training nor for judging the epochs."
    v = round(100 * val_acc)
    if test_acc < val_acc - _TEST_GAP:
        return warnings, (
            f"Lower than on validation ({t}% vs {v}%): the settings may have been tuned to the validation rows. "
            "Trust the test number: it is the honest one."
        )
    if test_acc > val_acc + _TEST_GAP:
        return warnings, (
            f"{t}% on the test rows vs {v}% on validation: small parts vary by chance, so neither number is exact."
        )
    return warnings, (
        f"Confirmed: {t}% on the test rows, in line with {v}% on validation. The numbers hold on rows used for "
        "nothing else."
    )


def test_run(spec: GraphSpec, run_id: str) -> TestResult:
    """The final test: a run's weights on this graph's test rows (and, for
    comparison, on its validation and training rows). Says whether the weights
    were trained on this very split — only then is it a fair check."""
    payload = load_run(run_id)  # a stale id is "not found", whatever the graph
    check_trainable(spec, "test")
    model, build = build_model(spec)
    if model.task_plan.task in ("reconstruction", "diffusion"):  # Stage 20
        return _generative_test(spec, run_id, payload, model)
    in_shape = _input_shape(build)
    mapped, _how = fit_state_dict(model, payload["state_dict"])
    model.load_state_dict(mapped)
    model.eval()
    num_classes = _infer_num_classes(model, in_shape)
    t = spec.train
    X, y, groups = make_rows(t.dataset, in_shape, num_classes, t, torch.device("cpu"))
    tr, va, te = split_rows(y, groups, t.val_fraction, t.test_fraction, t.seed)
    if not te.numel():
        raise GraphError(
            "there is no test part to measure: set Final test to 10 % or more and train again (the test rows "
            "have to be kept out of training), then press Test"
            if not t.test_fraction else
            "too few rows to keep a test part aside: add rows (every class needs a row to train on first)",
            code="no_test_part" if not t.test_fraction else "few_rows",
        )
    loss_fn = nn.CrossEntropyLoss()

    def evaluate(idx: torch.Tensor) -> tuple[float, float]:
        with torch.no_grad():
            logits, target = _fit_labels(model(X[idx]), y[idx])
            return float(loss_fn(logits, target)), float((logits.argmax(-1) == target).float().mean())

    test_loss, test_acc = evaluate(te)
    val_acc = evaluate(va)[1] if va.numel() else None
    train_acc = evaluate(tr)[1] if tr.numel() else None
    same = None if payload["data"] is None else payload["data"] == data_key(t, in_shape, num_classes)
    warnings, note = _test_reading(test_acc, val_acc, int(te.numel()), same)
    return TestResult(
        run_id=run_id,
        accuracy=test_acc,
        loss=test_loss,
        samples=int(te.numel()),
        val_accuracy=val_acc,
        train_accuracy=train_acc,
        val_samples=int(va.numel()),
        train_samples=int(tr.numel()),
        same_split=same,
        note=note,
        warnings=warnings,
    )


def adopt_weights(spec: GraphSpec, raw: bytes) -> WeightsInfo:
    """Load a weights file into this graph (it must fit), keep it as a run and
    say how it matched and whether it knows its training data."""
    check_trainable(spec, "load weights into")  # a real-scale graph is refused before the file is read
    payload = read_weights(raw)
    model, build = build_model(spec)
    in_shape = _input_shape(build)
    mapped, how = fit_state_dict(model, payload["state_dict"])
    model.load_state_dict(mapped)
    num_classes = _infer_num_classes(model, in_shape)
    data = payload["data"]
    same = None if data is None else data == data_key(spec.train, in_shape, num_classes)
    run_id = save_run(weights_bytes(mapped, data=data, metrics=payload["metrics"]))
    note = (
        f"Weights loaded: all {len(mapped)} tensors match this graph by name."
        if how == "names" else
        f"Weights loaded by position: the names differ (as in a file saved by the exported script), "
        f"but all {len(mapped)} shapes match."
    )
    if same is False:
        note += " They were trained on other data or another split, so Test will not be a fair check."
    elif same is None:
        note += " The file does not say which data it was trained on."
    return WeightsInfo(
        run_id=run_id,
        tensors=len(mapped),
        params=int(sum(p.numel() for p in model.parameters())),
        matched=how,
        data_known=data is not None,
        same_split=same,
        metrics=payload["metrics"],
        note=note,
    )


#: Stage 19 — what random values suit each Input (moved to mlconstructor.generative in Stage 20)
_input_kinds = input_roles


# --------------------------------------------------------------------------- #
# Stage 20 — teach it to generate: reconstruction (autoencoder, VAE) and diffusion
# --------------------------------------------------------------------------- #
_GEN_EVAL_ROWS = 2048  # rows measured after every epoch (a fixed subset: the same rows each time)
_GALLERY_STEPS = 25    # sampler steps for the pictures shown while training


class _Gen:
    """Everything a generative run needs to know, in one place: the data, how its
    pictures map to the network's signal and back, the loss, and drawing."""

    def __init__(self, model: nn.Module, plan, spec: TrainSpec, device: torch.device):
        self.model, self.plan, self.spec, self.device = model, plan, spec, device
        self.order = list(model.input_ids)
        self.diffusion = plan.task == "diffusion"
        shape = plan.in_shapes[plan.image_input]
        X, y, self.groups = make_rows(spec.dataset, shape, int(spec.num_classes), spec, device)
        self.X, self.y = X.float(), y
        self.unit = bool(self.X.numel()) and float(self.X.min()) >= 0.0 and float(self.X.max()) <= 1.0
        self.classes = int(self.y.max()) + 1 if self.y.numel() and self.y.dim() == 1 else 0
        self.null = None
        if self.diffusion:
            self.alphas = noise_schedule(int(spec.diffusion_steps)).to(device)
            if plan.class_input:
                if plan.class_rows < self.classes:
                    raise GraphError(
                        f"the class embedding has {plan.class_rows} rows, but the data has {self.classes} classes: give "
                        f"it at least {self.classes} (one more, {self.classes + 1}, lets it also draw without a class, "
                        "which makes the pictures sharper)",
                        code="class_rows", facts={"classes": self.classes, "input": plan.class_input},
                    )
                if plan.class_rows > self.classes:
                    self.null = plan.class_rows - 1  # the spare row: "no class"
            self.kind = "mse"
            self.loss_name = "error in seeing the noise (mean squared)"
        else:
            want = spec.recon_loss
            self.kind = want if want != "auto" else ("bce" if plan.sigmoid_out and self.unit else "mse")
            self.per_example = max(1, math.prod(plan.out_shape))
            what = "cross-entropy" if self.kind == "bce" else "mean squared error"
            self.loss_name = f"reconstruction error ({what})" + (" + KL" if plan.vae else "")

    # -- signal <-> picture
    def signal(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2 - 1 if (self.diffusion and self.unit) else x

    def picture(self, x: torch.Tensor) -> torch.Tensor:
        if self.unit:
            return ((x + 1) / 2 if self.diffusion else x).clamp(0, 1)
        flat = x.flatten(1)
        lo, hi = flat.min(1).values.view(-1, *[1] * (x.dim() - 1)), flat.max(1).values.view(-1, *[1] * (x.dim() - 1))
        return (x - lo) / (hi - lo + 1e-8)

    # -- the loss of a batch: (total, reconstruction, kl); the last two only for reconstruction
    def loss(self, xb: torch.Tensor, yb: torch.Tensor, gen: torch.Generator, train: bool):
        if self.diffusion:
            t, noise, xt = noisy_batch(self.signal(xb), self.alphas, gen)
            feeds = {self.plan.image_input: xt, self.plan.timestep_input: t.float().view(-1, 1)}
            if self.plan.class_input:
                lab = yb.float().clone()
                if train and self.null is not None:
                    drop = (torch.rand(lab.shape[0], generator=gen) < UNCOND_PROB).to(lab.device)
                    lab[drop] = float(self.null)
                feeds[self.plan.class_input] = lab.view(-1, 1)
            return F.mse_loss(self.model(*[feeds[i] for i in self.order]), noise), None, None
        out = self.model(xb)
        rec = recon_loss(out, xb, self.kind)
        kl = kl_total(self.model)
        kl = kl / self.per_example if torch.is_tensor(kl) else torch.zeros((), device=xb.device)
        return rec + kl, rec, kl

    def evaluate(self, Xs: torch.Tensor, ys: torch.Tensor) -> tuple[float, float | None, float | None]:
        """The loss on (at most _GEN_EVAL_ROWS of) a set, in eval mode, with the same
        noise every time — so epochs compare fairly and training's RNG is untouched."""
        was = self.model.training
        self.model.eval()
        gen = torch.Generator().manual_seed(int(self.spec.seed) + 707)
        Xs, ys = Xs[:_GEN_EVAL_ROWS], ys[:_GEN_EVAL_ROWS]
        tot = rec = kl = 0.0
        with torch.no_grad():
            for i in range(0, int(Xs.shape[0]), 256):
                xb, yb = Xs[i:i + 256], ys[i:i + 256]
                lv, rv, kv = self.loss(xb, yb, gen, train=False)
                m = int(xb.shape[0])
                tot += float(lv) * m
                if rv is not None:
                    rec += float(rv) * m
                    kl += float(kv) * m
        if was:
            self.model.train()
        n = max(1, int(Xs.shape[0]))
        return tot / n, (None if self.diffusion else rec / n), (None if self.diffusion else kl / n)

    # -- drawing
    def draw(self, count: int, labels: torch.Tensor | None, steps: int, guidance: float,
             gen: torch.Generator) -> torch.Tensor:
        """Pictures (count, C, H, W) in 0…1: diffusion samples (for `labels`, or free),
        or a VAE's decodings of random codes."""
        was = self.model.training
        self.model.eval()
        try:
            if not self.diffusion:
                # random codes from this draw's own seed: training's random numbers stay untouched
                seed = int(torch.randint(0, 2**31 - 1, (1,), generator=gen))
                devices = [self.device.index or 0] if self.device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(seed)
                    set_prior(self.model, True)
                    try:
                        with torch.no_grad():
                            shape = (count, *self.plan.in_shapes[self.plan.image_input])
                            return self.picture(self.model(torch.zeros(shape, device=self.device)))
                    finally:
                        set_prior(self.model, False)
            plan, dev = self.plan, self.device
            lab = None
            if plan.class_input:
                if labels is not None:
                    lab = labels.float().view(-1, 1).to(dev)
                elif self.null is None:  # no "no class" row: draw random classes
                    lab = torch.randint(0, max(self.classes, 1), (count, 1), generator=gen).float().to(dev)
            free = None if self.null is None else torch.full((count, 1), float(self.null), device=dev)

            def predict(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                feeds = {plan.image_input: x, plan.timestep_input: t}
                if plan.class_input:
                    feeds[plan.class_input] = lab if lab is not None else free
                eps = self.model(*[feeds[i] for i in self.order])
                if plan.class_input and lab is not None and free is not None and guidance != 1.0:
                    feeds[plan.class_input] = free
                    plain = self.model(*[feeds[i] for i in self.order])
                    eps = guide(eps, plain, guidance)
                return eps

            shape = (count, *plan.in_shapes[plan.image_input])
            return self.picture(ddim_sample(predict, shape, self.alphas, steps, gen, dev))
        finally:
            if was:
                self.model.train()

    def gallery(self, steps: int, Xv: torch.Tensor | None) -> list[dict]:
        """What the gallery shows: diffusion — one picture per class (or ten free
        ones); reconstruction — originals over their reconstructions, and for a VAE
        new pictures from random codes."""
        gen = torch.Generator().manual_seed(int(self.spec.seed) + 909)
        if self.diffusion:
            if self.plan.class_input and self.classes:
                labels = torch.arange(10) % self.classes
                pics = self.draw(10, labels, steps, float(self.spec.guidance), gen)
                return image_items(pics, [str(int(v)) for v in labels])
            return image_items(self.draw(10, None, steps, float(self.spec.guidance), gen), ["new"] * 10)
        src = (Xv if Xv is not None and Xv.shape[0] else self.X)[:8]
        was = self.model.training
        self.model.eval()
        with torch.no_grad():
            rec = self.model(src)
        if was:
            self.model.train()
        items = image_items(self.picture(src), ["original"] * len(src)) + \
            image_items(self.picture(rec), ["reconstruction"] * len(src))
        if self.plan.vae:
            items += image_items(self.draw(8, None, steps, 1.0, gen), ["new"] * 8)
        return items


def _gen_fit(train_loss: float, val_loss: float | None, n_val: int) -> tuple[str, str | None]:
    """Stage 20 — a verdict for a generative run: its error on new pictures vs on the
    training ones."""
    if val_loss is None or n_val < _SMALL_VAL:
        return "unknown", None
    gap = (val_loss - train_loss) / max(abs(train_loss), 1e-8)
    if gap > 0.15:
        return "overfits", (f"Its error on new pictures ({val_loss:.4f}) is {gap:.0%} higher than on the training ones "
                            f"({train_loss:.4f}): it is starting to learn them by heart.")
    return "generalises", (f"Its error on new pictures ({val_loss:.4f}) is about the same as on the training ones "
                           f"({train_loss:.4f}): it learned what such pictures look like, not the pictures themselves.")


def _generative_iter(model: nn.Module, plan, spec: TrainSpec) -> Iterator[dict]:
    """Stage 20 — the training loop for reconstruction and diffusion. The same events
    as classification (start / step / epoch / done; the epoch carries losses, no
    accuracy) plus ``{"type": "samples", "epoch", "images"}`` — the gallery after
    every epoch."""
    torch.manual_seed(spec.seed)
    device = _device(spec.device)
    model = model.to(device)
    g = _Gen(model, plan, spec, device)
    X, y = g.X, g.y
    tr_idx, va_idx, te_idx = split_rows(y, g.groups, spec.val_fraction, spec.test_fraction, spec.seed)
    n_rows, n_val, n_test = int(X.shape[0]), int(va_idx.numel()), int(te_idx.numel())
    warnings = _split_warnings(spec, n_rows, n_val, n_test, y)
    Xv = yv = None
    if n_val or n_test:
        if n_val:
            Xv, yv = X[va_idx.to(X.device)], y[va_idx.to(y.device)]
        X, y = X[tr_idx.to(X.device)], y[tr_idx.to(y.device)]
    g.X, g.y = X, y

    resumed = False
    if spec.resume and os.path.exists(_checkpoint_path(model)):
        try:
            model.load_state_dict(torch.load(_checkpoint_path(model), map_location=device))
            resumed = True
        except Exception:
            resumed = False

    n = int(X.shape[0])
    bs = min(spec.batch_size, n)
    opt = (torch.optim.Adam(model.parameters(), lr=spec.lr) if spec.optimizer == "adam"
           else torch.optim.SGD(model.parameters(), lr=spec.lr))
    total_params = int(sum(p.numel() for p in model.parameters()))
    gen = torch.Generator().manual_seed(int(spec.seed) + 303)  # timesteps, noise, dropped labels: per seed
    # the learning rate falls along a cosine to a tenth; a diffusion model draws with its averaged weights
    total_steps = spec.epochs * math.ceil(n / bs)
    if spec.steps:
        total_steps = min(total_steps, int(spec.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total_steps))
    ema = EMA(model, ema_decay(total_steps)) if plan.task == "diffusion" else None
    yield {
        "type": "start", "device": str(device), "resumed": resumed, "task": plan.task, "num_classes": g.classes,
        "samples": n_rows, "total_params": total_params, "train_samples": n, "val_samples": n_val,
        "test_samples": n_test,
    }

    def _save() -> str:
        os.makedirs(_CKPT_DIR, exist_ok=True)
        torch.save(model.state_dict(), _checkpoint_path(model))
        return _checkpoint_key(model)

    loss_curve: list[float] = []
    val_loss_curve: list[float] = []
    step = 0
    stop = False
    best = math.inf
    ckpt_key: str | None = None
    model.train()
    for epoch in range(spec.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss, _rec, _kl = g.loss(X[idx], y[idx], gen, train=True)
            loss.backward()
            opt.step()
            sched.step()
            if ema is not None:
                ema.update(model)
            lv = float(loss.detach())
            loss_curve.append(lv)
            step += 1
            yield {"type": "step", "step": step, "loss": lv}
            if spec.steps and step >= spec.steps:
                stop = True
                break
        if not stop:
            tr_loss, tr_rec, tr_kl = g.evaluate(X, y)
            val_loss = val_rec = val_kl = None
            if Xv is not None:
                val_loss, val_rec, val_kl = g.evaluate(Xv, yv)
                val_loss_curve.append(val_loss)
            yield {"type": "epoch", "epoch": epoch + 1, "acc": None, "loss": tr_loss, "val_acc": None,
                   "val_loss": val_loss, "recon": val_rec if val_rec is not None else tr_rec,
                   "kl": val_kl if val_kl is not None else tr_kl}
            if ema is not None:
                ema.swap(model)  # the gallery draws with the averaged weights
            images = g.gallery(min(_GALLERY_STEPS, spec.sample_steps), Xv)
            if ema is not None:
                ema.swap(model)
            yield {"type": "samples", "epoch": epoch + 1, "images": images}
            score = tr_loss if val_loss is None else val_loss
            if spec.checkpoint and score < best:
                best = score
                ckpt_key = _save()
        if stop:
            break

    if ema is not None:
        ema.swap(model)  # from here on the network is its averaged weights: measured, drawn with and kept
    model.eval()
    final_loss, final_rec, final_kl = g.evaluate(X, y)
    final_val = val_rec = val_kl = None
    if Xv is not None:
        final_val, val_rec, val_kl = g.evaluate(Xv, yv)
        if not val_loss_curve:
            val_loss_curve.append(final_val)
    if spec.checkpoint and ckpt_key is None:
        ckpt_key = _save()
    gallery = g.gallery(int(spec.sample_steps), Xv)
    fit, fit_note = _gen_fit(final_loss, final_val, n_val)
    draw_classes = g.classes if (plan.task == "diffusion" and plan.class_input) else 0
    result = TrainResult(
        ok=True, loss_curve=loss_curve, acc_curve=[], final_loss=final_loss, final_accuracy=0.0, steps=step,
        device=str(device), total_params=total_params, checkpoint=ckpt_key, best_accuracy=None, resumed=resumed,
        val_acc_curve=[], val_loss_curve=val_loss_curve, final_val_accuracy=None, final_val_loss=final_val,
        train_samples=n, val_samples=n_val, fit=fit, fit_note=fit_note, warnings=warnings, test_samples=n_test,
        task=plan.task, loss_name=g.loss_name,
        final_recon=val_rec if val_rec is not None else final_rec, final_kl=val_kl if val_kl is not None else final_kl,
        gallery=[GalleryImage(**it) for it in gallery], draw_classes=draw_classes,
        draw_free=(plan.task == "diffusion" and (not plan.class_input or g.null is not None)) or bool(plan.vae),
    )
    if device.type == "cuda":
        model.to("cpu")
        del X, y, Xv, yv
        g.X = g.y = None
        torch.cuda.empty_cache()
    yield {"type": "done", "result": result}


def _generative_test(spec: GraphSpec, run_id: str, payload: dict, model: nn.Module) -> TestResult:
    """Stage 20 — the final test of a generative run: its error on the test pictures
    (and, to compare, on the validation and training ones)."""
    mapped, _how = fit_state_dict(model, payload["state_dict"])
    model.load_state_dict(mapped)
    model.eval()
    t = spec.train
    g = _Gen(model, model.task_plan, t, torch.device("cpu"))
    tr, va, te = split_rows(g.y, g.groups, t.val_fraction, t.test_fraction, t.seed)
    if not te.numel():
        raise GraphError(
            "there is no test part to measure: set Final test to 10 % or more and train again (the test rows have to be "
            "kept out of training), then press Test",
            code="no_test_part",
        )
    test_loss = g.evaluate(g.X[te], g.y[te])[0]
    val_loss = g.evaluate(g.X[va], g.y[va])[0] if va.numel() else None
    train_loss = g.evaluate(g.X[tr], g.y[tr])[0] if tr.numel() else None
    shape = model.task_plan.in_shapes[model.task_plan.image_input]
    same = None if payload["data"] is None else payload["data"] == data_key(t, shape, int(t.num_classes))
    note = (f"Error on pictures it never saw: {test_loss:.4f}"
            + (f" (new data while training: {val_loss:.4f})" if val_loss is not None else "") + ".")
    warnings = [] if same is not False else [
        "These weights were trained on other data or another split: some test pictures may have been trained on."]
    return TestResult(run_id=run_id, accuracy=0.0, loss=test_loss, samples=int(te.numel()), val_accuracy=None,
                      train_accuracy=None, val_samples=int(va.numel()), train_samples=int(tr.numel()), same_split=same,
                      note=note, warnings=warnings, task=model.task_plan.task, val_loss=val_loss, train_loss=train_loss)


def draw(spec: GraphSpec, run_id: str, count: int = 10, label: int | None = None, steps: int = 50,
         guidance: float = 1.0, seed: int = 0) -> DrawResult:
    """Stage 20 — draw pictures with a run's weights: diffusion removes noise step by
    step (for one class, or one of each); a VAE decodes random codes; a plain
    autoencoder can only reconstruct, so it shows originals and reconstructions."""
    payload = load_run(run_id)
    check_trainable(spec, "run")
    model, _build = build_model(spec)
    plan = model.task_plan
    if plan.task not in ("reconstruction", "diffusion"):
        raise GraphError("Draw needs a network that learned to generate: a VAE (reconstruction) or a diffusion model",
                         code="draw_needs_generative")
    mapped, _how = fit_state_dict(model, payload["state_dict"])
    model.load_state_dict(mapped)
    g = _Gen(model, plan, spec.train, torch.device("cpu"))
    gen = torch.Generator().manual_seed(int(seed))
    start = time.perf_counter()
    note = None
    if plan.task == "diffusion":
        labels = None
        if plan.class_input and g.classes:
            if label is not None:
                if not 0 <= int(label) < g.classes:
                    raise GraphError(f"the class must be between 0 and {g.classes - 1}", code="draw_class")
                labels = torch.full((count,), int(label))
            else:
                labels = torch.arange(count) % g.classes
        pics = g.draw(count, labels, steps, guidance, gen)
        names = [str(int(v)) for v in labels] if labels is not None else ["new"] * count
        items = image_items(pics, names)
    elif plan.vae:
        items = image_items(g.draw(count, None, steps, 1.0, gen), ["new"] * count)
    else:
        idx = torch.randperm(int(g.X.shape[0]), generator=gen)[:count]
        src = g.X[idx]
        model.eval()
        with torch.no_grad():
            rec = model(src)
        items = image_items(g.picture(src), ["original"] * len(src)) + image_items(g.picture(rec), ["reconstruction"] * len(src))
        note = ("A plain autoencoder only gives back what it is shown: add a VAE sampling layer between its encoder and "
                "decoder to draw new pictures from noise.")
    return DrawResult(run_id=run_id, task=plan.task, images=[GalleryImage(**it) for it in items],
                      ms=round((time.perf_counter() - start) * 1000.0, 1), note=note)


def run_once(spec: GraphSpec) -> RunResult:
    """Stage 19 — one forward pass on random inputs (a batch of two, from the run's
    seed), with the weights the network starts from: the proof that it really runs,
    what comes out and how long it takes. Works for a network with several inputs;
    a real-scale flagship is refused (too big to run here)."""
    check_trainable(spec, "run")
    model, build = build_model(spec, seed=spec.train.seed)
    model.eval()
    g = torch.Generator().manual_seed(int(spec.train.seed))
    kinds = _input_kinds(spec, build)
    feeds: list[torch.Tensor] = []
    infos: list[RunInputInfo] = []
    for inp in build.inputs:
        kind, top = kinds[inp.id]
        shape = (2, *inp.shape)
        if kind == "tokens":
            x = torch.randint(0, max(int(top), 1), shape, generator=g).float()
        elif kind == "timestep" and top <= 1.0:
            x = torch.rand(shape, generator=g)
        elif kind == "timestep":
            x = torch.randint(0, max(int(top), 1), shape, generator=g).float()
        else:
            x = torch.randn(shape, generator=g)
        feeds.append(x)
        infos.append(RunInputInfo(id=inp.id, name=inp.name, shape=list(inp.shape), kind=kind))
    start = time.perf_counter()
    with torch.no_grad():
        y = model(*feeds)
    ms = (time.perf_counter() - start) * 1000.0
    first = y[0].float().flatten()
    finite = bool(torch.isfinite(first).all())
    vals = first[torch.isfinite(first)]
    stat = (lambda f: float(f(vals))) if vals.numel() else (lambda f: 0.0)
    return RunResult(
        inputs=infos, output_shape=list(y.shape[1:]), mean=stat(torch.mean),
        std=float(vals.std()) if vals.numel() > 1 else 0.0, min=stat(torch.min), max=stat(torch.max),
        sample=[round(float(v), 5) for v in first[:8]], finite=finite, ms=round(ms, 2),
        total_params=build.total_params,
    )


def generate(spec: GraphSpec, run_id: str, prompt: str = "", length: int = 200, temperature: float = 0.8,
             seed: int = 0) -> GenerateResult:
    """Stage 18 — continue a prompt with the weights of a run trained on "Your text":
    the model reads the last L characters, scores every character of the text's
    vocabulary, and one is drawn; repeat. An empty prompt starts from the text's
    first characters."""
    payload = load_run(run_id)  # a stale id is "not found", whatever the graph
    t = spec.train
    if canonical_dataset_id(t.dataset) != "text" or not (t.text or "").strip():
        raise GraphError(
            "Generate continues text: choose \"Your text\" as the dataset, add a text in Data, train, then generate",
            code="generate_needs_text",
        )
    check_trainable(spec, "run")
    model, build = build_model(spec)
    in_shape = _input_shape(build)
    if len(in_shape) != 1:
        raise GraphError("Generate needs a language model: Input [L] -> Embedding -> ... -> LM head",
                         code="generate_needs_lm")
    mapped, _how = fit_state_dict(model, payload["state_dict"])
    model.load_state_dict(mapped)
    model.eval()
    num_classes = _infer_num_classes(model, in_shape)
    vocab = text_vocab(t.text, num_classes)
    context = int(in_shape[0])
    if not prompt:
        prompt = t.text[:context]
    known = set(vocab)
    unknown = sum(1 for c in prompt if c not in known)
    text = generate_text(model, vocab, context, prompt, int(length), float(temperature), int(seed))
    return GenerateResult(run_id=run_id, prompt=prompt, text=text.replace(UNKNOWN, "\ufffd"), unknown=unknown,
                          vocab_size=len(vocab), context=context)
