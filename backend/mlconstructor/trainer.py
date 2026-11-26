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

import os
from collections.abc import Iterator

import torch
import torch.nn as nn

from .builder import build_model
from .datasets import canonical_dataset_id, get_dataset, make_dataset, split_parts
from .schema import DataPreview, DataSampleRow, GraphSpec, TestResult, TrainResult, TrainSpec, WeightsInfo
from .shapes import Shape, ShapeError, numel
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


def check_trainable(spec: GraphSpec, what: str = "train") -> int:
    """Refuse a network too big for this server before a single weight is made."""
    n = count_params(spec)
    if n > MAX_TRAIN_PARAMS:
        raise ValueError(
            f"this network has {n:,} parameters ({_human(n)}), far too big to {what} on this server (the limit is "
            f"{_human(MAX_TRAIN_PARAMS)}). It is here to inspect: every layer and shape of the real model. The "
            "flagships in the library come with a mini version that trains here; or shrink the layers"
        )
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
    available, else the CPU."""
    if want == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "GPU (CUDA) training was requested, but no CUDA device is available on this backend"
            )
        return torch.device("cuda")
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    torch.manual_seed(spec.seed)
    device = _device(spec.device)
    model = model.to(device)

    in_shape = tuple(int(d) for d in in_shape)
    num_classes = _infer_num_classes(model, in_shape)
    X, y = make_dataset(spec.dataset, in_shape, num_classes, spec, device)

    # Stages 15–16 — hold rows out for validation and for the final test; train on the rest only.
    tr_idx, va_idx, te_idx = split_parts(y, spec.val_fraction, spec.test_fraction, spec.seed)
    n_rows, n_val, n_test = int(X.shape[0]), int(va_idx.numel()), int(te_idx.numel())
    warnings = _split_warnings(spec, n_rows, n_val, n_test, y)
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
            logits = model(Xs)
            loss_v = float(loss_fn(logits, ys))
            acc_v = float((logits.argmax(1) == ys).float().mean())
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
            loss = loss_fn(model(X[idx]), y[idx])
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
    in_shape = _input_shape(build)
    num_classes = _infer_num_classes(model, in_shape)
    device = torch.device("cpu")

    ds = get_dataset(spec.train.dataset)
    k = max(1, min(int(rows), 50))
    # Draw from the SAME generator, seed and sample count as training (the same
    # spec), so these are exactly the first rows of the training data — the noise
    # is drawn after the labels, so a smaller draw would NOT line up.
    X, y = ds.make(in_shape, num_classes, spec.train, device)
    k = min(k, X.shape[0])
    # Stages 15–16 — the same split as training: mark the validation and test rows.
    tr_idx, va_idx, te_idx = split_parts(y, spec.train.val_fraction, spec.train.test_fraction, spec.train.seed)
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
                label=int(y[i]),
                values=[round(float(v), 3) for v in vals[:_PREVIEW_MAX_VALUES]],
                truncated=len(vals) > _PREVIEW_MAX_VALUES,
                total=len(vals),
                split="test" if bool(tested[i]) else ("validation" if bool(held[i]) else "train"),
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
    in_shape = _input_shape(build)
    mapped, _how = fit_state_dict(model, payload["state_dict"])
    model.load_state_dict(mapped)
    model.eval()
    num_classes = _infer_num_classes(model, in_shape)
    t = spec.train
    X, y = make_dataset(t.dataset, in_shape, num_classes, t, torch.device("cpu"))
    tr, va, te = split_parts(y, t.val_fraction, t.test_fraction, t.seed)
    if not te.numel():
        raise ValueError(
            "there is no test part to measure: set Final test to 10 % or more and train again (the test rows "
            "have to be kept out of training), then press Test"
            if not t.test_fraction else
            "too few rows to keep a test part aside: add rows (every class needs a row to train on first)"
        )
    loss_fn = nn.CrossEntropyLoss()

    def evaluate(idx: torch.Tensor) -> tuple[float, float]:
        with torch.no_grad():
            logits = model(X[idx])
            return float(loss_fn(logits, y[idx])), float((logits.argmax(1) == y[idx]).float().mean())

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
