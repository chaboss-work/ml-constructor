"""Trained weights: keep, download, load back (Stage 16).

Every training run on the server keeps its final weights as a **run** — one file
under the gitignored data cache — so they can be downloaded (``.pt``), checked on
the test rows, or used as the starting point of the next run. A weights file is a
plain ``torch.save`` of a small dict::

    {"format": "mlconstructor.weights", "version": 1,
     "state_dict": {...},   # the app model's own tensor names
     "arch": <names-and-shapes key>, "data": <data-and-split key or None>,
     "metrics": {...}, "created": <UTC time>}

It is read back with ``torch.load(weights_only=True)`` — tensors and plain values
only, never pickled code — so loading a file from anywhere is safe. A bare
``state_dict`` loads too (e.g. the ``model.pt`` the exported script writes): by
name when the names match, else by position when every shape lines up.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import secrets
from datetime import datetime, timezone

import torch
import torch.nn as nn

from .datasets import canonical_dataset_id
from .schema import TrainSpec
from .shapes import Shape

WEIGHTS_FORMAT = "mlconstructor.weights"
WEIGHTS_VERSION = 1
#: The largest weights file accepted (an upload is base64 in JSON, a third bigger).
MAX_WEIGHTS_BYTES = 64 * 2**20
#: Runs kept on the server; older ones are removed as new ones arrive.
MAX_RUNS = 30
_RUN_ID = re.compile(r"^[0-9a-f]{16}$")
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


class RunNotFound(KeyError):
    """The run id is unknown here (never existed, malformed, or already pruned)."""

    def __str__(self) -> str:  # KeyError would quote the message
        return str(self.args[0]) if self.args else "unknown run"


def runs_dir() -> str:
    """Where runs live (``MLC_RUNS_DIR`` moves it; read on every call, so tests can)."""
    return os.environ.get("MLC_RUNS_DIR") or os.path.join(_DATA_DIR, "runs")


def arch_key(state_dict: dict) -> str:
    """A stable key per architecture — a hash of tensor names and shapes."""
    sig = [(name, tuple(t.shape)) for name, t in state_dict.items()]
    return hashlib.sha1(repr(sig).encode("utf-8")).hexdigest()[:16]


def data_key(spec: TrainSpec, in_shape: Shape, num_classes: int) -> str:
    """What the rows and their split depend on — equal keys mean the very same
    training / validation / test rows (dataset, seed, sample count, shapes, the
    shares held out, and the pasted rows themselves for custom data)."""
    dataset = canonical_dataset_id(spec.dataset)
    payload = {
        "dataset": dataset,
        "seed": int(spec.seed),
        "samples": None if dataset == "custom" else int(spec.samples),
        "in_shape": [int(d) for d in in_shape],
        "classes": int(num_classes),
        "val": float(spec.val_fraction),
        "test": float(spec.test_fraction),
        "rows": (
            hashlib.sha1(json.dumps(spec.custom_rows or []).encode("utf-8")).hexdigest()
            if dataset == "custom" else None
        ),
    }
    if dataset == "text":  # Stage 18 — "Your text": the very same text (other datasets keep their old keys)
        payload["text"] = hashlib.sha1((spec.text or "").encode("utf-8")).hexdigest()
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def weights_bytes(model: nn.Module | dict, *, data: str | None, metrics: dict | None = None) -> bytes:
    """A weights file for a model (or a ready state_dict), as bytes."""
    sd = model.state_dict() if isinstance(model, nn.Module) else model
    sd = {k: v.detach().to("cpu").clone() for k, v in sd.items()}
    payload = {
        "format": WEIGHTS_FORMAT,
        "version": WEIGHTS_VERSION,
        "state_dict": sd,
        "arch": arch_key(sd),
        "data": data,
        "metrics": {k: v for k, v in (metrics or {}).items() if isinstance(v, (int, float, str, type(None)))},
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


def read_weights(raw: bytes) -> dict:
    """Parse a weights file safely: ``{"state_dict", "data", "metrics", "created", "bare"}``.

    ``weights_only=True`` lets through tensors and plain values only, so a file that
    tries to run code on loading is refused before anything in it runs."""
    if len(raw) > MAX_WEIGHTS_BYTES:
        raise ValueError(f"the weights file is over {MAX_WEIGHTS_BYTES // 2**20} MB")
    if not (raw[:4] == b"PK\x03\x04" or raw[:1] == b"\x80"):  # a torch.save zip, or the legacy pickle format
        raise ValueError("this is not a PyTorch weights file (.pt): load a file saved by Save weights or torch.save")
    try:
        obj = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as exc:  # broken, or holding more than tensors and plain values
        if "weights only" in str(exc).lower() or "weights_only" in str(exc):
            raise ValueError(
                "this file holds more than weights (Python objects, possibly code), and the app only opens files "
                "with plain tensors, for safety: nothing in it was run. Save the weights alone (a state_dict) and "
                "load that"
            ) from None
        raise ValueError("this PyTorch file is damaged or incomplete: save the weights again") from None
    if isinstance(obj, dict) and obj.get("format") == WEIGHTS_FORMAT:
        if int(obj.get("version") or 0) > WEIGHTS_VERSION:
            raise ValueError("this weights file comes from a newer version of the app")
        sd, bare = obj.get("state_dict"), False
    else:
        sd, bare = obj, True
    if not isinstance(sd, dict) or not sd:
        raise ValueError("the file holds no weights (expected tensor names mapped to tensors)")
    if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in sd.items()):
        raise ValueError("the file holds something other than weights (every entry must be a named tensor)")
    meta = obj if not bare else {}
    data = meta.get("data")
    return {
        "state_dict": sd,
        "data": data if isinstance(data, str) else None,
        "metrics": meta.get("metrics") if isinstance(meta.get("metrics"), dict) else {},
        "created": meta.get("created") if isinstance(meta.get("created"), str) else None,
        "bare": bare,
    }


def _pretty(name: str) -> str:
    """``_mods.n_fc1.weight`` -> ``fc1.weight`` (the layer id the user sees)."""
    return re.sub(r"(^|\.)n_", r"\1", name.removeprefix("_mods."))


def fit_state_dict(model: nn.Module, sd: dict) -> tuple[dict, str]:
    """Map a file's tensors onto ``model``: by name when the names match, else by
    position when every shape lines up. Raises ValueError saying what differs."""
    target = model.state_dict()
    if set(sd) == set(target):
        bad = [k for k in target if tuple(sd[k].shape) != tuple(target[k].shape)]
        if not bad:
            return {k: sd[k] for k in target}, "names"
        shown = "; ".join(
            f"{_pretty(k)} is {list(sd[k].shape)} in the file but {list(target[k].shape)} here" for k in bad[:3]
        )
        more = f" (and {len(bad) - 3} more)" if len(bad) > 3 else ""
        raise ValueError(
            f"these weights do not fit this graph: {shown}{more}. A layer's size was changed since they were "
            "saved — set it back, or train again"
        )
    if len(sd) == len(target) and all(tuple(a.shape) == tuple(b.shape) for a, b in zip(sd.values(), target.values())):
        return dict(zip(target.keys(), sd.values())), "order"
    n_file = sum(int(t.numel()) for t in sd.values())
    n_here = sum(int(t.numel()) for t in target.values())
    raise ValueError(
        f"these weights belong to a different network: the file has {len(sd)} tensors ({n_file:,} values), "
        f"this graph needs {len(target)} ({n_here:,}). Load weights saved from this same graph "
        "(same layers, same sizes)"
    )


# --------------------------------------------------------------------------- #
# The run store
# --------------------------------------------------------------------------- #
def run_path(run_id: str) -> str:
    """The file of a run; :class:`RunNotFound` when there is none."""
    if not isinstance(run_id, str) or not _RUN_ID.match(run_id):
        raise RunNotFound(f"no saved weights with the id {run_id!r}")
    path = os.path.join(runs_dir(), run_id + ".pt")
    if not os.path.isfile(path):
        raise RunNotFound(
            f"the weights {run_id} are no longer on the server (it keeps the last {MAX_RUNS} runs) — "
            "train again, or load the saved .pt file"
        )
    return path


def save_run(raw: bytes) -> str:
    """Keep a weights file as a new run; returns its id. Old runs beyond
    :data:`MAX_RUNS` are removed."""
    d = runs_dir()
    os.makedirs(d, exist_ok=True)
    run_id = secrets.token_hex(8)
    path = os.path.join(d, run_id + ".pt")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(raw)
    os.replace(tmp, path)
    _prune(d, keep=run_id)
    return run_id


def load_run(run_id: str) -> dict:
    """A run's weights, parsed as by :func:`read_weights`."""
    with open(run_path(run_id), "rb") as f:
        return read_weights(f.read())


def _prune(d: str, keep: str) -> None:
    files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".pt") and _RUN_ID.match(f[:-3])]
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for old in files[MAX_RUNS:]:
        if os.path.basename(old)[:-3] != keep:
            try:
                os.remove(old)
            except OSError:
                pass
