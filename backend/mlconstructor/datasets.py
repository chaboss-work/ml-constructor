"""The dataset registry (Stage 8).

Where the training data comes from. v1 shipped a single inline synthetic task; it
now lives here as one entry of a small **registry**, so the same generator feeds
both the trainer and the ``/data/preview`` modal — the preview is exactly the data
training sees — and new datasets (and later the real ones: MNIST/CIFAR/upload)
plug in by adding one :class:`DatasetDef`.

Every dataset is **self-contained and synthetic**: no download, no GPU, provably
learnable, and works for any input shape (a flat vector, an image ``(C,H,W)`` or a
sequence ``(L, F)``), with the class count taken from the network the user built.
A ``Shape`` here is per-sample, without the batch dimension.
"""

from __future__ import annotations

import gzip
import math
import os
import pickle
import shutil
import struct
import tarfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .schema import DatasetInfo, TrainSpec
from .shapes import GraphError, Shape, numel
from .text import text_vocab, text_windows

_MAX_CUSTOM_ROWS = 10000  # bound the payload / memory for pasted-or-uploaded data

# Local cache for auto-downloaded real datasets (gitignored: backend/data/).
_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
# Only these official hosts may be downloaded from (Stage 10).
_OFFICIAL_HOSTS = {"ossci-datasets.s3.amazonaws.com", "www.cs.toronto.edu"}

#: A generator: (per-sample shape, #classes, train spec, device) -> (X, y).
DatasetFn = Callable[[Shape, int, TrainSpec, "torch.device"], tuple["torch.Tensor", "torch.Tensor"]]


@dataclass(frozen=True)
class DatasetDef:
    id: str
    name: str
    description: str
    make: DatasetFn
    #: Real datasets (MNIST/CIFAR) have a FIXED input shape and class count the
    #: graph must match; adaptive synthetic datasets leave these None.
    fixed_shape: tuple[int, ...] | None = None
    fixed_classes: int | None = None
    kind: str = "synthetic"


def _blobs(in_shape: Shape, num_classes: int, spec: TrainSpec, device) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Separable Gaussian clusters: one random prototype per class + noise.

    This is the exact v1 synthetic task (kept byte-for-byte: same RNG draw order),
    so every existing result is unchanged. Signal std 1.0 vs noise std 0.5 keeps
    the classes well separated, so any sane network learns it quickly.
    """
    g = torch.Generator().manual_seed(spec.seed)
    n = spec.samples
    prototypes = torch.randn(num_classes, *in_shape, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    noise = torch.randn(n, *in_shape, generator=g)
    X = prototypes[y] + 0.5 * noise
    return X.to(device), y.to(device)


def _shells(in_shape: Shape, num_classes: int, spec: TrainSpec, device) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Concentric shells: the class is the sample's *distance from the origin*.

    Each sample points in a random direction and sits on the shell of radius
    ``class + 1`` (plus a little noise). The decision boundary is the vector norm —
    **not** linearly separable — so a bare Linear can't solve it but an MLP/CNN can.
    A nice contrast to ``blobs`` for showing a model learn a curved boundary.
    """
    g = torch.Generator().manual_seed(spec.seed)
    n = spec.samples
    y = torch.randint(0, num_classes, (n,), generator=g)
    direction = torch.randn(n, *in_shape, generator=g)
    norm = direction.flatten(1).norm(dim=1).view(n, *([1] * len(in_shape))) + 1e-8
    unit = direction / norm
    radius = (y.float() + 1.0).view(n, *([1] * len(in_shape)))
    X = unit * radius + 0.15 * torch.randn(n, *in_shape, generator=g)
    return X.to(device), y.to(device)


def _tokens(in_shape: Shape, num_classes: int, spec: TrainSpec, device) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Next token (Stage 17): the language-model task in miniature.

    Each row is a sequence of token ids that repeats a short random pattern (2 to 4
    tokens long); the label is the token that comes next. A language model solves it
    the way real ones continue text: by finding where the current token appeared
    before and copying what followed it. The vocabulary is the network's own output
    size, so the head scores every possible next token.
    """
    length = _check_tokens(in_shape)
    g = torch.Generator().manual_seed(spec.seed)
    n = spec.samples
    period = torch.randint(2, min(4, length - 1) + 1, (n,), generator=g)
    pattern = torch.randint(0, num_classes, (n, 4), generator=g)
    position = torch.arange(length + 1)
    seq = pattern.gather(1, position[None, :] % period[:, None])
    return seq[:, :length].float().to(device), seq[:, length].to(device)


def _check_tokens(in_shape: Shape) -> int:
    if len(in_shape) != 1:
        raise GraphError(
            f"the 'next token' dataset needs an Input of token ids, shape [L]; your graph's input is "
            f"{list(in_shape)} — put an Embedding after a 1-D Input",
            code="dataset_needs_tokens", facts={"dataset": "tokens"},
        )
    length = int(in_shape[0])
    if length < 3:
        raise GraphError("the 'next token' dataset needs sequences of at least 3 tokens (Input [L] with L >= 3)",
                         code="tokens_too_short", facts={"length": length})
    return length


def _check_text(in_shape: Shape, spec: TrainSpec) -> int:
    text = spec.text or ""
    if len(in_shape) != 1:
        raise GraphError(
            f"'Your text' needs an Input of token ids, shape [L]; your graph's input is {list(in_shape)}. "
            "Take a language model from the library (e.g. Your GPT, or a flagship's mini version)",
            code="dataset_needs_tokens", facts={"dataset": "text"},
        )
    length = int(in_shape[0])
    if not text.strip():
        raise GraphError("'Your text' has no text yet: open Data, paste or upload a text (or use the example), Apply",
                         code="text_missing")
    need = 8 * (length + 1)
    if len(text) < need:
        raise GraphError(
            f"the text is too short: {len(text):,} characters, and windows of {length} need at least {need:,}. "
            "Paste more text, or make the Input shorter",
            code="text_too_short", facts={"chars": len(text), "length": length},
        )
    return length


def _text(in_shape: Shape, num_classes: int, spec: TrainSpec, device):
    """Your text (Stage 18): a language model learns to continue it.

    Rows are windows of ``L`` characters (token ids), labelled with the character
    that follows; the vocabulary is the text's own characters (as many as the
    network's output scores). Rows carry the text block they come from, so the
    train / validation / test split keeps every block whole — no held-out row has a
    near-copy among the training rows.
    """
    text = spec.text or ""
    length = _check_text(in_shape, spec)
    X, y, groups = text_windows(text, length, num_classes, int(spec.samples), int(spec.seed))
    return X.to(device), y.to(device), groups


def _custom(in_shape: Shape, num_classes: int, spec: TrainSpec, device) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Train on the user's OWN rows (Stage 9 — paste or upload your data).

    Each row in ``spec.custom_rows`` is ``[f1, …, fk, label]``: its features then
    an integer class label. Rows are validated against the graph's input shape and
    the network's class count, so a mismatch is a clear message, not a torch crash.
    """
    rows = _check_custom(in_shape, num_classes, spec)
    xs = [[float(v) for v in row[:-1]] for row in rows]
    ys = [int(row[-1]) for row in rows]
    X = torch.tensor(xs, dtype=torch.float32).reshape(len(rows), *in_shape)
    y = torch.tensor(ys, dtype=torch.long)
    return X.to(device), y.to(device)


def _whole(v) -> int | None:
    try:
        f = float(v)
        return int(f) if math.isfinite(f) and int(f) == f else None
    except (TypeError, ValueError, OverflowError):
        return None


def _check_custom(in_shape: Shape, num_classes: int, spec: TrainSpec) -> list:
    """Stage 21 — every row checked against the Input's shape and the class count
    (training and the live diagnosis both run this)."""
    rows = spec.custom_rows or []
    if not rows:
        raise GraphError(
            "the 'custom' dataset needs rows — paste or upload training examples "
            "(each row = the features, then a whole-number class label)",
            code="custom_missing",
        )
    if len(rows) > _MAX_CUSTOM_ROWS:
        raise GraphError(f"too many custom rows ({len(rows)}); the limit is {_MAX_CUSTOM_ROWS}", code="custom_too_many")
    feat = numel(in_shape)
    for i, row in enumerate(rows):
        if len(row) != feat + 1:
            lengths = {len(r) for r in rows}
            raise GraphError(
                f"custom row {i + 1}: expected {feat} feature(s) + 1 label = {feat + 1} values, got {len(row)}",
                code="custom_row_length",
                # every row agrees with the others: the Input is what is off
                facts={"row": i + 1, "features": feat, "row_features": lengths.pop() - 1 if len(lengths) == 1 else None},
            )
        if any(not math.isfinite(float(v)) for v in row):
            raise GraphError(f"custom row {i + 1}: contains a non-finite value (NaN/inf)", code="custom_value",
                             facts={"row": i + 1})
        if int(row[-1]) != row[-1]:
            raise GraphError(f"custom row {i + 1}: label {row[-1]} must be a whole number", code="custom_label",
                             facts={"row": i + 1})
        label = int(row[-1])
        if not (0 <= label < num_classes):
            labels = [_whole(r[-1]) for r in rows if len(r) == feat + 1]
            top = max([v for v in labels if v is not None] or [label])
            raise GraphError(
                f"custom row {i + 1}: label {label} out of range [0, {num_classes}) — "
                f"the network's output layer has {num_classes} classes",
                code="custom_label_range", facts={"row": i + 1, "classes": num_classes, "needed": top + 1},
            )
    return rows


def check_dataset(name: str, in_shape: Shape, num_classes: int, spec: TrainSpec) -> None:
    """Stage 21 — would training on ``name`` refuse this network? The same checks the
    datasets run before making their rows (raises their :class:`GraphError`), with
    nothing generated or downloaded: the editor runs it on every edit."""
    d = get_dataset(name)
    if d.id == "tokens":
        _check_tokens(in_shape)
    elif d.id == "text":
        _check_text(in_shape, spec)
    elif d.id == "custom":
        _check_custom(in_shape, num_classes, spec)
    elif d.id == "mnist":
        _check_fixed("MNIST", (1, 28, 28), in_shape, num_classes)
    elif d.id == "cifar10":
        _check_fixed("CIFAR-10", (3, 32, 32), in_shape, num_classes)


# --------------------------------------------------------------------------- #
# Real datasets (Stage 10): auto-downloaded from their official sources, parsed
# with the standard library + numpy (no torchvision), cached under backend/data/.
# --------------------------------------------------------------------------- #
def _download(url: str, dest: str) -> str:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if host not in _OFFICIAL_HOSTS:
        raise ValueError(f"refusing to download a dataset from a non-official host: {host!r}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        req = urllib.request.Request(url, headers={"User-Agent": "ml-constructor"})
        with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:  # noqa: S310 (host allow-listed)
            shutil.copyfileobj(resp, out)
    return dest


_MNIST_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_MNIST_IMAGES = "train-images-idx3-ubyte.gz"
_MNIST_LABELS = "train-labels-idx1-ubyte.gz"


def _load_mnist() -> tuple[np.ndarray, np.ndarray]:
    d = os.path.join(_CACHE, "mnist")
    imgs_gz = _download(_MNIST_BASE + _MNIST_IMAGES, os.path.join(d, _MNIST_IMAGES))
    lbls_gz = _download(_MNIST_BASE + _MNIST_LABELS, os.path.join(d, _MNIST_LABELS))
    with gzip.open(imgs_gz, "rb") as f:
        _magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
    with gzip.open(lbls_gz, "rb") as f:
        struct.unpack(">II", f.read(8))
        lbls = np.frombuffer(f.read(), dtype=np.uint8)
    return imgs, lbls


_CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def _load_cifar10() -> tuple[np.ndarray, np.ndarray]:
    d = os.path.join(_CACHE, "cifar10")
    tgz = _download(_CIFAR_URL, os.path.join(d, "cifar-10-python.tar.gz"))
    batch_dir = os.path.join(d, "cifar-10-batches-py")
    if not os.path.isdir(batch_dir):
        with tarfile.open(tgz, "r:gz") as t:
            t.extractall(d, filter="data")  # py3.12 safe extraction (blocks path traversal)
    data, labels = [], []
    for i in range(1, 6):
        with open(os.path.join(batch_dir, f"data_batch_{i}"), "rb") as f:
            batch = pickle.load(f, encoding="bytes")  # official CIFAR-10 pickle
        data.append(batch[b"data"])
        labels.extend(batch[b"labels"])
    X = np.concatenate(data).reshape(-1, 3, 32, 32)  # (50000, 3, 32, 32) uint8
    y = np.array(labels, dtype=np.uint8)
    return X, y


def _subsample(imgs: np.ndarray, lbls: np.ndarray, spec: TrainSpec, device):
    """A reproducible random subset of `spec.samples` images, scaled to [0, 1]."""
    n = min(int(spec.samples), imgs.shape[0])
    g = torch.Generator().manual_seed(int(spec.seed))
    sel = torch.randperm(imgs.shape[0], generator=g)[:n].numpy()
    X = torch.from_numpy(imgs[sel].astype(np.float32) / 255.0)
    y = torch.from_numpy(lbls[sel].astype(np.int64))
    return X.to(device), y.to(device)


def _check_fixed(name: str, want: tuple[int, ...], in_shape: Shape, num_classes: int):
    if tuple(int(x) for x in in_shape) != want:
        raise GraphError(f"{name} needs an Input of shape {list(want)}; your graph's input is {list(in_shape)}",
                         code="dataset_shape", facts={"want": list(want), "dataset": name})
    if num_classes < 10:
        raise GraphError(
            f"{name} has 10 classes; your network outputs {num_classes} — make the last layer output at least 10",
            code="dataset_classes", facts={"classes": 10, "got": num_classes},
        )


def _mnist(in_shape: Shape, num_classes: int, spec: TrainSpec, device):
    _check_fixed("MNIST", (1, 28, 28), in_shape, num_classes)
    imgs, lbls = _load_mnist()
    X, y = _subsample(imgs, lbls, spec, device)  # imgs (n,28,28) -> add channel
    return X.unsqueeze(1), y


def _cifar10(in_shape: Shape, num_classes: int, spec: TrainSpec, device):
    _check_fixed("CIFAR-10", (3, 32, 32), in_shape, num_classes)
    imgs, lbls = _load_cifar10()  # (50000,3,32,32)
    return _subsample(imgs, lbls, spec, device)


_DATASETS: list[DatasetDef] = [
    DatasetDef(
        "blobs",
        "Gaussian blobs",
        "Separable Gaussian clusters — one prototype per class plus noise. "
        "Linearly separable, so any network learns it fast.",
        _blobs,
    ),
    DatasetDef(
        "shells",
        "Concentric shells",
        "The class is the distance from the origin (radius = class). A curved, "
        "non-linear boundary — needs a hidden layer, not just a Linear.",
        _shells,
    ),
    DatasetDef(
        "tokens",
        "Next token",
        "The language-model task in miniature: sequences of token ids that repeat a short pattern; "
        "the label is the token that comes next. Needs Input [L] and an Embedding.",
        _tokens,
    ),
    DatasetDef(
        "text",
        "Your text",
        "Teach a language model your own text: it learns to continue it, one character at a time, "
        "then writes from your prompt. Needs Input [L], an Embedding and an LM head.",
        _text,
        kind="custom",
    ),
    DatasetDef(
        "custom",
        "Your data",
        "Train on rows you paste or upload: each row is its features, then a "
        "whole-number class label. The features must match the Input shape.",
        _custom,
        kind="custom",
    ),
    DatasetDef(
        "mnist",
        "MNIST (digits)",
        "Handwritten digits, 28x28 grayscale, 10 classes. Needs Input [1, 28, 28]. "
        "Auto-downloaded (~11 MB) on first use.",
        _mnist,
        fixed_shape=(1, 28, 28),
        fixed_classes=10,
        kind="real",
    ),
    DatasetDef(
        "cifar10",
        "CIFAR-10",
        "Color images, 32x32 RGB, 10 classes. Needs Input [3, 32, 32]. "
        "Auto-downloaded (~170 MB) on first use; heavy on CPU.",
        _cifar10,
        fixed_shape=(3, 32, 32),
        fixed_classes=10,
        kind="real",
    ),
]

_BY_ID: dict[str, DatasetDef] = {d.id: d for d in _DATASETS}

#: Back-compat: the v1 train spec defaults to ``dataset="synthetic"``; keep it (and
#: an empty value) meaning the original blobs task.
_ALIASES = {"synthetic": "blobs", "": "blobs"}


def canonical_dataset_id(name: str) -> str:
    """The catalogue id for a dataset name (resolves the legacy aliases)."""
    return _ALIASES.get(name, name)


def get_dataset(name: str) -> DatasetDef:
    key = canonical_dataset_id(name)
    try:
        return _BY_ID[key]
    except KeyError:
        known = ", ".join(sorted(_BY_ID))
        raise GraphError(
            f"unknown dataset {name!r}; known: {known} (alias 'synthetic' = blobs)",
            code="unknown_dataset",
        ) from None


def make_dataset(
    name: str, in_shape: Shape, num_classes: int, spec: TrainSpec, device
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Generate ``(X, y)`` for the named dataset — the single entry point the
    trainer and the preview both call, so they can never drift apart."""
    return make_rows(name, in_shape, num_classes, spec, device)[:2]


def make_rows(name: str, in_shape: Shape, num_classes: int, spec: TrainSpec, device):
    """``(X, y, groups)``: the rows, and — for a dataset whose rows overlap, like the
    windows of a text (Stage 18) — the group each row belongs to (else None)."""
    out = get_dataset(name).make(in_shape, num_classes, spec, device)
    return (out[0], out[1], out[2] if len(out) > 2 else None)


def text_stats(spec: TrainSpec, num_classes: int) -> dict:
    """Stage 18 — what the "Your text" dataset makes of the text: its length, its
    different characters and how many of them get a token of their own."""
    text = spec.text or ""
    distinct = len(set(text))
    return {"text_chars": len(text), "distinct_chars": distinct,
            "vocab_size": len(text_vocab(text, num_classes)) if text else 0}


def split_parts(
    y: torch.Tensor, val_fraction: float, test_fraction: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split row indices into training, validation and test parts (Stages 15–16).

    The network trains on the first part only. The validation part is measured after
    every epoch (how it does on rows it never trained on); the test part is kept for
    one final check and touches neither training nor the choice of weights. Both
    shares are of ALL rows. The test part is set aside first, then validation from
    the rest, so with ``test_fraction = 0`` the split is exactly Stage 15's.

    Deterministic for a seed, stratified (each class keeps its share in every part,
    by largest remainders) and safe on tiny sets: every class keeps at least one row
    to train on, so a part stays empty when nothing can be held out that way.
    Returns three sorted CPU index tensors. Self-contained on purpose: the PyTorch
    export embeds this very function, so an exported file splits exactly like the app.
    """
    n = int(y.shape[0])
    labels = y.detach().to("cpu").long().reshape(-1)
    every = torch.arange(n)

    def share(fraction: float) -> int:
        return min(max(1, round(n * float(fraction))), n - 1) if n >= 2 and fraction and fraction > 0 else 0

    def hold_out(idx: torch.Tensor, want: int, g: torch.Generator) -> torch.Tensor:
        m = int(idx.numel())
        if m < 2 or want <= 0:
            return idx[:0]
        want = min(want, m - 1)
        lab = labels[idx]
        order = torch.randperm(m, generator=g)  # a random order of the rows ...
        order = order[torch.argsort(lab[order], stable=True)]  # ... grouped by class
        counts = torch.unique_consecutive(lab[order], return_counts=True)[1].tolist()
        # Each class's share of `want`, at most count - 1 so it keeps a row for training.
        take = [min(c * want // m, c - 1) for c in counts]
        left = want - sum(take)
        tie = torch.randperm(len(counts), generator=g).tolist()
        ranked = sorted(range(len(counts)), key=lambda i: (-((counts[i] * want) % m), tie[i]))
        while left > 0:
            progressed = False
            for i in ranked:
                if left > 0 and take[i] < counts[i] - 1:
                    take[i] += 1
                    left -= 1
                    progressed = True
            if not progressed:  # every class is down to its last training row
                break
        picked = torch.zeros(m, dtype=torch.bool)
        start = 0
        for c, k in zip(counts, take):
            picked[order[start : start + k]] = True
            start += c
        return idx[picked]

    # Own random streams (the data generator uses `seed`): the test part first, then validation.
    test = hold_out(every, share(test_fraction), torch.Generator().manual_seed(int(seed) + 202))
    rest = torch.ones(n, dtype=torch.bool)
    rest[test] = False
    val = hold_out(every[rest], share(val_fraction), torch.Generator().manual_seed(int(seed) + 101))
    train = rest.clone()
    train[val] = False
    return every[train], val, test


def split_rows(y, groups, val_fraction, test_fraction, seed):
    """Split rows into train / validation / test (Stage 18): row by row as
    :func:`split_parts` does, or — when the rows come in groups (overlapping windows
    of one block of a text) — whole groups at a time, so no part holds a neighbour of
    another part's rows. Self-contained like split_parts (the export embeds it)."""
    if groups is None:
        return split_parts(y, val_fraction, test_fraction, seed)
    groups = groups.detach().to("cpu").long().reshape(-1)
    ids, member = torch.unique(groups, return_inverse=True)
    g_train, g_val, g_test = split_parts(torch.zeros(len(ids), dtype=torch.long), val_fraction, test_fraction, seed)
    part = torch.zeros(len(ids), dtype=torch.long)
    part[g_val], part[g_test] = 1, 2
    rows = torch.arange(len(groups))
    return rows[part[member] == 0], rows[part[member] == 1], rows[part[member] == 2]


def split_indices(y: torch.Tensor, val_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Training and validation parts only (Stage 15) — :func:`split_parts` without a test part."""
    train, val, _ = split_parts(y, val_fraction, 0.0, seed)
    return train, val


def list_datasets() -> list[DatasetInfo]:
    """The dataset catalogue for the editor's picker (``GET /datasets``)."""
    return [
        DatasetInfo(
            id=d.id,
            name=d.name,
            description=d.description,
            kind=d.kind,
            input_shape=(list(d.fixed_shape) if d.fixed_shape else None),
        )
        for d in _DATASETS
    ]
