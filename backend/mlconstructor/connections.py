"""Smart connections (Stage 14) — every edge can reshape what flows through it.

A connection carries a *transform* that turns the source's output into an input
the target layer can take: flatten an image for a dense layer, read an image row
by row as a sequence for an LSTM, fold a vector into a C×H×W picture for a
convolution, pool a sequence into one vector, or match the shapes feeding an Add.

``auto`` (the default for every edge) picks the simplest transform that works —
nothing at all when the shapes already fit — and reports what it picked. When
there are several sensible choices the editor offers them in the connection's
settings, each with its own parameters. Every transform is a plain ``torch.nn``
module, so training and the PyTorch export see exactly what the editor shows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isqrt, prod
from typing import Any, ClassVar, Iterable

import torch
import torch.nn as nn

from .schema import ParamDef
from .shapes import Shape, ShapeError

AUTO = "auto"
#: A learnable projection is picked automatically only up to this many weights;
#: a bigger one stays available in the selector but is never chosen silently.
MAX_AUTO_PARAMS = 1_000_000


def fmt_shape(shape: Iterable[int] | None) -> str:
    return "×".join(str(int(d)) for d in shape) if shape is not None else "?"


# --------------------------------------------------------------------------- #
# Runtime modules for the transforms torch has no ready-made layer for
# --------------------------------------------------------------------------- #
class SeqPool(nn.Module):
    """Collapse a (B, L, F) sequence to (B, F): the last or first step, the mean or the max."""

    def __init__(self, mode: str = "last"):
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "last":
            return x[:, -1]
        if self.mode == "first":
            return x[:, 0]
        if self.mode == "mean":
            return x.mean(dim=1)
        return x.amax(dim=1)

    def extra_repr(self) -> str:
        return f"mode={self.mode!r}"


class ImageToSeq(nn.Module):
    """Read a (B, C, H, W) feature map as a sequence: ``rows`` → (B, H, C·W),
    ``columns`` → (B, W, C·H), ``pixels`` → (B, H·W, C), ``patches`` → (B, N, C·p·p)."""

    def __init__(self, mode: str = "rows", patch: int = 4):
        super().__init__()
        self.mode = mode
        self.patch = patch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if self.mode == "rows":
            return x.permute(0, 2, 1, 3).reshape(b, h, c * w)
        if self.mode == "columns":
            return x.permute(0, 3, 1, 2).reshape(b, w, c * h)
        if self.mode == "pixels":
            return x.flatten(2).transpose(1, 2)
        p = self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5)  # (B, H/p, W/p, C, p, p)
        return x.reshape(b, (h // p) * (w // p), c * p * p)

    def extra_repr(self) -> str:
        return f"mode={self.mode!r}, patch={self.patch}"


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
class Transform(ABC):
    """One way a connection can reshape data. Instances carry their parameters."""

    kind: ClassVar[str] = ""
    label: ClassVar[str] = ""  # selector text
    description: ClassVar[str] = ""  # one sentence: what it does to the data
    learnable: ClassVar[bool] = False

    def __init__(self, params: dict | None = None):
        self.params = dict(params or {})

    @classmethod
    def accepts(cls, in_shape: Shape) -> bool:
        """Whether this transform can take ``in_shape`` at all (by rank)."""
        return True

    @classmethod
    def param_defs(cls, in_shape: Shape) -> list[ParamDef]:
        return []

    @classmethod
    def presets(cls, in_shape: Shape) -> list[dict]:
        """Parameter sets worth offering / trying automatically, best first."""
        return [{}]

    @abstractmethod
    def out_shape(self, in_shape: Shape) -> Shape: ...

    @abstractmethod
    def build(self, in_shape: Shape) -> nn.Module | None: ...

    def badge(self, in_shape: Shape) -> str:
        """Short text for the wire (empty = nothing to show)."""
        return self.label

    def resolved_params(self, in_shape: Shape) -> dict:
        """The parameters in effect (defaults filled in), for the editor to show."""
        return dict(self.params)

    def _int(self, name: str, default: int) -> int:
        try:
            return int(self.params.get(name, default))
        except (TypeError, ValueError):
            raise ShapeError(f"parameter {name!r} of “{self.label}” must be a whole number") from None


class Direct(Transform):
    kind = "none"
    label = "Direct — no change"
    description = "Passes the data through exactly as it is."

    def out_shape(self, in_shape: Shape) -> Shape:
        return tuple(in_shape)

    def build(self, in_shape: Shape) -> None:
        return None

    def badge(self, in_shape: Shape) -> str:
        return ""


class Flatten(Transform):
    kind = "flatten"
    label = "Flatten to a vector"
    description = "Lays every value out in one long row, e.g. an image 16×6×6 becomes 576 numbers."

    @classmethod
    def accepts(cls, in_shape: Shape) -> bool:
        return len(in_shape) >= 2

    def out_shape(self, in_shape: Shape) -> Shape:
        if len(in_shape) < 2:
            raise ShapeError("it is already a flat vector")
        return (int(prod(in_shape)),)

    def build(self, in_shape: Shape) -> nn.Module:
        return nn.Flatten()

    def badge(self, in_shape: Shape) -> str:
        return "flatten"


class GlobalPool(Transform):
    kind = "global_pool"
    label = "Pool each channel to one value"
    description = "Averages (or takes the maximum of) each channel over the whole image: C×H×W → C."

    @classmethod
    def accepts(cls, in_shape: Shape) -> bool:
        return len(in_shape) == 3

    @classmethod
    def param_defs(cls, in_shape: Shape) -> list[ParamDef]:
        return [ParamDef(name="mode", type="str", default="avg", label="Mode", choices=["avg", "max"])]

    @classmethod
    def presets(cls, in_shape: Shape) -> list[dict]:
        return [{"mode": "avg"}, {"mode": "max"}]

    def _mode(self) -> str:
        mode = str(self.params.get("mode", "avg"))
        if mode not in ("avg", "max"):
            raise ShapeError(f"pooling mode must be 'avg' or 'max', got {mode!r}")
        return mode

    def out_shape(self, in_shape: Shape) -> Shape:
        if len(in_shape) != 3:
            raise ShapeError("it needs an image (C×H×W)")
        self._mode()
        return (int(in_shape[0]),)

    def build(self, in_shape: Shape) -> nn.Module:
        pool = nn.AdaptiveAvgPool2d(1) if self._mode() == "avg" else nn.AdaptiveMaxPool2d(1)
        return nn.Sequential(pool, nn.Flatten())

    def badge(self, in_shape: Shape) -> str:
        return f"{self._mode()} pool"

    def resolved_params(self, in_shape: Shape) -> dict:
        return {"mode": self._mode()}


class SequenceToVector(Transform):
    kind = "seq_pool"
    label = "Sequence → one vector"
    description = "Keeps the last (or first) step of a sequence, or its mean or maximum over time: L×F → F."

    @classmethod
    def accepts(cls, in_shape: Shape) -> bool:
        return len(in_shape) == 2

    @classmethod
    def param_defs(cls, in_shape: Shape) -> list[ParamDef]:
        return [
            ParamDef(name="mode", type="str", default="last", label="Keep", choices=["last", "first", "mean", "max"])
        ]

    @classmethod
    def presets(cls, in_shape: Shape) -> list[dict]:
        return [{"mode": m} for m in ("last", "mean", "max", "first")]

    def _mode(self) -> str:
        mode = str(self.params.get("mode", "last"))
        if mode not in ("last", "first", "mean", "max"):
            raise ShapeError(f"mode must be last, first, mean or max, got {mode!r}")
        return mode

    def out_shape(self, in_shape: Shape) -> Shape:
        if len(in_shape) != 2:
            raise ShapeError("it needs a sequence (L×F)")
        self._mode()
        return (int(in_shape[1]),)

    def build(self, in_shape: Shape) -> nn.Module:
        return SeqPool(self._mode())

    def badge(self, in_shape: Shape) -> str:
        return {"last": "last step", "first": "first step"}.get(self._mode(), self._mode())

    def resolved_params(self, in_shape: Shape) -> dict:
        return {"mode": self._mode()}


def _image_factorizations(n: int, limit: int = 8) -> list[tuple[int, int, int]]:
    """(C, H, W) with C·H·W = n, H and W ≥ 2, best first: square pictures with few
    channels, then near-square ones. 128 → (2, 8, 8) first; 784 → (1, 28, 28)."""
    found: list[tuple[int, int, int]] = []
    for c in range(1, n + 1):
        if n % c:
            continue
        hw = n // c
        for h in range(2, isqrt(hw) + 1):
            if hw % h == 0 and hw // h >= 2:
                found.append((c, h, hw // h))
    # tiny pictures (H < 4) last: most convolutions need room for their kernel
    found.sort(key=lambda t: (t[1] < 4, t[1] != t[2], t[2] / t[1], t[0], -t[1]))
    return found[:limit]


class ToImage(Transform):
    kind = "to_image"
    label = "Fold into an image (C×H×W)"
    description = "Rearranges the values into channels × height × width, e.g. 128 numbers → a 2×8×8 picture."

    @classmethod
    def param_defs(cls, in_shape: Shape) -> list[ParamDef]:
        return [
            ParamDef(name="channels", type="int", default=1, min=1, label="Channels (C)"),
            ParamDef(name="height", type="int", default=1, min=1, label="Height (H)"),
            ParamDef(name="width", type="int", default=1, min=1, label="Width (W)"),
        ]

    @classmethod
    def presets(cls, in_shape: Shape) -> list[dict]:
        out: list[dict] = []
        if len(in_shape) == 2:  # a sequence: a one-channel picture of it first
            out.append({"channels": 1, "height": int(in_shape[0]), "width": int(in_shape[1])})
        for c, h, w in _image_factorizations(int(prod(in_shape))):
            p = {"channels": c, "height": h, "width": w}
            if p not in out:
                out.append(p)
        return out or [{"channels": 1, "height": 1, "width": int(prod(in_shape))}]

    def _dims(self, in_shape: Shape) -> tuple[int, int, int]:
        default = self.presets(in_shape)[0]
        dims = (
            self._int("channels", default["channels"]),
            self._int("height", default["height"]),
            self._int("width", default["width"]),
        )
        return dims

    def out_shape(self, in_shape: Shape) -> Shape:
        c, h, w = self._dims(in_shape)
        n = int(prod(in_shape))
        if min(c, h, w) < 1 or c * h * w != n:
            raise ShapeError(f"C×H×W must multiply to {n} (the number of values arriving); {c}×{h}×{w} = {c * h * w}")
        return (c, h, w)

    def build(self, in_shape: Shape) -> nn.Module:
        dims = self.out_shape(in_shape)
        if len(in_shape) == 1:
            return nn.Unflatten(1, dims)
        return nn.Sequential(nn.Flatten(), nn.Unflatten(1, dims))

    def badge(self, in_shape: Shape) -> str:
        return "→ " + fmt_shape(self.out_shape(in_shape))

    def resolved_params(self, in_shape: Shape) -> dict:
        c, h, w = self._dims(in_shape)
        return {"channels": c, "height": h, "width": w}


class ToSequence(Transform):
    kind = "to_sequence"
    label = "Vector → sequence"
    description = "Cuts a vector into steps for a sequence layer: one step with all values, or one value per step."

    @classmethod
    def accepts(cls, in_shape: Shape) -> bool:
        return len(in_shape) == 1

    @classmethod
    def param_defs(cls, in_shape: Shape) -> list[ParamDef]:
        return [
            ParamDef(
                name="length",
                type="int",
                default=1,
                min=1,
                label="Steps (L)",
                help="How many time steps to cut the vector into; it must divide the vector length.",
            )
        ]

    @classmethod
    def presets(cls, in_shape: Shape) -> list[dict]:
        n = int(in_shape[0]) if in_shape else 1
        return [{"length": 1}] + ([{"length": n}] if n > 1 else [])

    def out_shape(self, in_shape: Shape) -> Shape:
        if len(in_shape) != 1:
            raise ShapeError("it needs a flat vector")
        n = int(in_shape[0])
        length = self._int("length", 1)
        if length < 1 or n % length:
            raise ShapeError(f"the number of steps must divide {n}; {length} does not")
        return (length, n // length)

    def build(self, in_shape: Shape) -> nn.Module:
        return nn.Unflatten(1, self.out_shape(in_shape))

    def badge(self, in_shape: Shape) -> str:
        return "→ " + fmt_shape(self.out_shape(in_shape))

    def resolved_params(self, in_shape: Shape) -> dict:
        return {"length": self._int("length", 1)}


class ImageToSequence(Transform):
    kind = "image_to_sequence"
    label = "Read the image as a sequence"
    description = (
        "Turns an image into steps: rows (each row is a step), columns, pixels, "
        "or square patches like a vision transformer."
    )

    @classmethod
    def accepts(cls, in_shape: Shape) -> bool:
        return len(in_shape) == 3

    @classmethod
    def param_defs(cls, in_shape: Shape) -> list[ParamDef]:
        return [
            ParamDef(name="mode", type="str", default="rows", label="Read by", choices=["rows", "columns", "pixels", "patches"]),
            ParamDef(name="patch", type="int", default=4, min=1, label="Patch size", help="Used with “patches”; must divide H and W."),
        ]

    @classmethod
    def presets(cls, in_shape: Shape) -> list[dict]:
        out = [{"mode": m} for m in ("rows", "pixels", "columns")]
        if len(in_shape) == 3:
            _c, h, w = in_shape
            for p in (2, 4, 7, 8, 14, 16):
                if p < min(h, w) and h % p == 0 and w % p == 0:
                    out.append({"mode": "patches", "patch": p})
        return out

    def _mode(self) -> str:
        mode = str(self.params.get("mode", "rows"))
        if mode not in ("rows", "columns", "pixels", "patches"):
            raise ShapeError(f"read-by must be rows, columns, pixels or patches, got {mode!r}")
        return mode

    def out_shape(self, in_shape: Shape) -> Shape:
        if len(in_shape) != 3:
            raise ShapeError("it needs an image (C×H×W)")
        c, h, w = (int(d) for d in in_shape)
        mode = self._mode()
        if mode == "rows":
            return (h, c * w)
        if mode == "columns":
            return (w, c * h)
        if mode == "pixels":
            return (h * w, c)
        p = self._int("patch", 4)
        if p < 1 or h % p or w % p:
            raise ShapeError(f"the patch size must divide the image height {h} and width {w}; {p} does not")
        return ((h // p) * (w // p), c * p * p)

    def build(self, in_shape: Shape) -> nn.Module:
        self.out_shape(in_shape)
        return ImageToSeq(self._mode(), self._int("patch", 4))

    def badge(self, in_shape: Shape) -> str:
        mode = self._mode()
        return f"patches {self._int('patch', 4)}²" if mode == "patches" else mode

    def resolved_params(self, in_shape: Shape) -> dict:
        out: dict[str, Any] = {"mode": self._mode()}
        if out["mode"] == "patches":
            out["patch"] = self._int("patch", 4)
        return out


def _target_shape(params: dict, name: str = "shape") -> Shape:
    raw = params.get(name)
    if isinstance(raw, str):
        raw = [s for s in raw.replace("×", ",").replace("x", ",").split(",") if s.strip()]
    try:
        shape = tuple(int(d) for d in (raw or []))
    except (TypeError, ValueError):
        raise ShapeError("the target shape must be whole numbers, e.g. 64 or 1, 8, 8") from None
    if not shape or any(d < 1 for d in shape) or len(shape) > 3:
        raise ShapeError("the target shape must be 1 to 3 positive numbers, e.g. 64 or 1, 8, 8")
    return shape


class Reshape(Transform):
    kind = "reshape"
    label = "Reshape (same values)"
    description = "Rearranges the same values into another shape with the same total count."

    @classmethod
    def param_defs(cls, in_shape: Shape) -> list[ParamDef]:
        return [ParamDef(name="shape", type="intlist", default=[int(prod(in_shape))], label="Shape")]

    @classmethod
    def presets(cls, in_shape: Shape) -> list[dict]:
        return [{"shape": [int(prod(in_shape))]}]

    def out_shape(self, in_shape: Shape) -> Shape:
        shape = _target_shape(self.params) if self.params.get("shape") else (int(prod(in_shape)),)
        if prod(shape) != prod(in_shape):
            raise ShapeError(
                f"the new shape must hold the same {int(prod(in_shape))} values; {fmt_shape(shape)} holds {int(prod(shape))}"
            )
        return shape

    def build(self, in_shape: Shape) -> nn.Module | None:
        shape = self.out_shape(in_shape)
        if tuple(shape) == tuple(in_shape):
            return None
        if len(shape) == 1:
            return nn.Flatten()
        return nn.Sequential(nn.Flatten(), nn.Unflatten(1, shape))

    def badge(self, in_shape: Shape) -> str:
        return "→ " + fmt_shape(self.out_shape(in_shape))

    def resolved_params(self, in_shape: Shape) -> dict:
        return {"shape": list(self.out_shape(in_shape))}


def projection_params(in_shape: Shape, target: Shape) -> int:
    """How many weights :class:`Project` adds for ``in_shape`` → ``target``."""
    if len(in_shape) == len(target) == 1:
        return in_shape[0] * target[0] + target[0]
    if len(in_shape) == len(target) == 2 and in_shape[0] == target[0]:
        return in_shape[1] * target[1] + target[1]
    if len(in_shape) == len(target) == 3:
        return in_shape[0] * target[0] + target[0]
    return int(prod(in_shape)) * int(prod(target)) + int(prod(target))


class Project(Transform):
    kind = "project"
    label = "Learnable projection"
    description = (
        "Learns a mapping to the shape you give: a dense layer for vectors and sequences, "
        "a 1×1 convolution (plus resizing) for images."
    )
    learnable = True

    @classmethod
    def param_defs(cls, in_shape: Shape) -> list[ParamDef]:
        return [
            ParamDef(
                name="shape",
                type="intlist",
                default=list(in_shape),
                label="Target shape",
                help="e.g. 64 for a vector, or 32, 14, 14 for an image",
            )
        ]

    @classmethod
    def presets(cls, in_shape: Shape) -> list[dict]:
        return [{"shape": list(in_shape)}]

    def out_shape(self, in_shape: Shape) -> Shape:
        return _target_shape(self.params) if self.params.get("shape") else tuple(in_shape)

    def build(self, in_shape: Shape) -> nn.Module:
        t = self.out_shape(in_shape)
        if len(in_shape) == len(t) == 1:
            return nn.Linear(in_shape[0], t[0])
        if len(in_shape) == len(t) == 2 and in_shape[0] == t[0]:
            return nn.Linear(in_shape[1], t[1])  # applied to every step
        if len(in_shape) == len(t) == 3:
            conv = nn.Conv2d(in_shape[0], t[0], kernel_size=1)
            if tuple(in_shape[1:]) == tuple(t[1:]):
                return conv
            return nn.Sequential(nn.AdaptiveAvgPool2d((t[1], t[2])), conv)
        mods: list[nn.Module] = [nn.Flatten(), nn.Linear(int(prod(in_shape)), int(prod(t)))]
        if len(t) > 1:
            mods.append(nn.Unflatten(1, t))
        return nn.Sequential(*mods)

    def badge(self, in_shape: Shape) -> str:
        return "proj → " + fmt_shape(self.out_shape(in_shape))

    def resolved_params(self, in_shape: Shape) -> dict:
        return {"shape": list(self.out_shape(in_shape))}


TRANSFORMS: dict[str, type[Transform]] = {
    t.kind: t
    for t in (Direct, Flatten, GlobalPool, SequenceToVector, ToImage, ToSequence, ImageToSequence, Reshape, Project)
}


def make_transform(kind: str, params: dict | None = None) -> Transform:
    try:
        return TRANSFORMS[kind](params)
    except KeyError:
        known = ", ".join([AUTO, *TRANSFORMS])
        raise ValueError(f"unknown connection transform {kind!r}; known: {known}") from None


# --------------------------------------------------------------------------- #
# Resolution: which transform an edge uses
# --------------------------------------------------------------------------- #
@dataclass
class Resolved:
    transform: Transform
    in_shape: Shape
    out_shape: Shape
    auto: bool


def _auto_plan(in_shape: Shape) -> list[tuple[str, dict]]:
    """Transforms "auto" tries for a single-input layer, simplest first."""
    plan: list[tuple[str, dict]] = [("none", {})]
    rank = len(in_shape)
    if rank == 3:
        plan += [
            ("flatten", {}),
            ("image_to_sequence", {"mode": "rows"}),
            ("image_to_sequence", {"mode": "pixels"}),
            ("image_to_sequence", {"mode": "columns"}),
            ("global_pool", {"mode": "avg"}),
        ]
    elif rank == 2:
        plan += [("flatten", {}), ("to_image", ToImage.presets(in_shape)[0]), ("seq_pool", {"mode": "last"})]
    elif rank == 1:
        plan += [("to_sequence", {"length": 1})]
        plan += [("to_image", p) for p in ToImage.presets(in_shape)]
        if in_shape[0] > 1:
            plan += [("to_sequence", {"length": int(in_shape[0])})]
        # last resort for a vector into an image layer: learn an 8×8 picture from it
        if projection_params(tuple(in_shape), (1, 8, 8)) <= MAX_AUTO_PARAMS:
            plan += [("project", {"shape": [1, 8, 8]})]
    return plan


def _why(exc: Exception) -> str:
    """The layer's own reason for refusing."""
    return str(exc) or "it does not fit"


def resolve_single(kind: str, params: dict, in_shape: Shape, target, edge: tuple[str, str]) -> Resolved:
    """The transform an edge into a single-input layer uses. Raises
    :class:`ShapeError` (pointing at the edge) when nothing makes the shapes fit."""
    src, dst = edge
    if kind != AUTO:
        t = make_transform(kind, params)
        try:
            adapted = t.out_shape(in_shape)
        except ShapeError as exc:
            raise ShapeError(
                f"the connection {src} → {dst} is set to “{t.label}”, which cannot take {fmt_shape(in_shape)}: {_why(exc)}",
                edge=edge,
                node_id=dst,
                hint="Pick another transform for this connection, or set it back to Auto.",
            ) from None
        try:
            target.out_shape(adapted)
        except ShapeError as exc:
            raise ShapeError(
                f"the connection {src} → {dst} is set to “{t.label}”, which gives {fmt_shape(adapted)}, "
                f"but {dst} ({target.type_name}) cannot take that: {_why(exc)}",
                edge=edge,
                node_id=dst,
                hint="Pick another transform for this connection, or set it back to Auto.",
            ) from None
        return Resolved(t, tuple(in_shape), tuple(adapted), auto=False)

    first_error: Exception | None = None
    for k, p in _auto_plan(in_shape):
        t = make_transform(k, p)
        try:
            adapted = t.out_shape(in_shape)
            target.out_shape(adapted)
        except ShapeError as exc:
            if first_error is None:
                first_error = exc
            continue
        return Resolved(t, tuple(in_shape), tuple(adapted), auto=True)
    raise ShapeError(
        f"{dst} ({target.type_name}) cannot take what {src} sends ({fmt_shape(in_shape)}), "
        f"and no automatic transform fits: {_why(first_error) if first_error else 'no transform applies'}",
        edge=edge,
        node_id=dst,
        hint="Change the layer's settings, or connect it to a layer with a fitting output.",
    )


def _add_auto(in_shape: Shape, ref: Shape) -> Transform:
    """What "auto" uses to make one Add input match the reference shape."""
    if tuple(in_shape) == tuple(ref):
        return make_transform("none")
    if prod(in_shape) == prod(ref):
        return make_transform("reshape", {"shape": list(ref)})
    if projection_params(in_shape, ref) <= MAX_AUTO_PARAMS:
        return make_transform("project", {"shape": list(ref)})
    raise ShapeError(
        f"a learnable projection {fmt_shape(in_shape)} → {fmt_shape(ref)} would need "
        f"{projection_params(in_shape, ref):,} weights — too many to add automatically"
    )


def resolve_merge(
    settings: list[tuple[str, dict]],
    in_shapes: list[Shape],
    node,
    sources: list[str],
) -> list[Resolved]:
    """Transforms for every edge into a merge node (Add / Concat), in input order."""
    dst = node.id
    n = len(in_shapes)
    chosen: list[Resolved | None] = [None] * n
    for i, (kind, params) in enumerate(settings):
        if kind != AUTO:
            t = make_transform(kind, params)
            try:
                adapted = t.out_shape(in_shapes[i])
            except ShapeError as exc:
                raise ShapeError(
                    f"the connection {sources[i]} → {dst} is set to “{t.label}”, which cannot take "
                    f"{fmt_shape(in_shapes[i])}: {_why(exc)}",
                    edge=(sources[i], dst),
                    node_id=dst,
                    hint="Pick another transform for this connection, or set it back to Auto.",
                ) from None
            chosen[i] = Resolved(t, tuple(in_shapes[i]), tuple(adapted), auto=False)

    def finish(auto_kinds: list[Transform]) -> list[Resolved]:
        out: list[Resolved] = []
        j = 0
        for i in range(n):
            if chosen[i] is not None:
                out.append(chosen[i])
            else:
                t = auto_kinds[j]
                j += 1
                out.append(Resolved(t, tuple(in_shapes[i]), tuple(t.out_shape(in_shapes[i])), auto=True))
        return out

    auto_idx = [i for i in range(n) if chosen[i] is None]
    # 1) nothing to do?
    direct = finish([make_transform("none") for _ in auto_idx])
    try:
        node.out_shape_merge([r.out_shape for r in direct])
        return direct
    except ShapeError as exc:
        first_error = exc
    # 2) adapt the automatic inputs
    if node.type_name == "Add":
        ref_i = next((i for i in range(n) if chosen[i] is not None), 0)
        ref = chosen[ref_i].out_shape if chosen[ref_i] is not None else tuple(in_shapes[ref_i])
        try:
            fixed = finish([_add_auto(in_shapes[i], ref) for i in auto_idx])
            node.out_shape_merge([r.out_shape for r in fixed])
            return fixed
        except ShapeError as exc:
            first_error = exc
    elif node.type_name == "Concat":
        fixed_ts = [make_transform("flatten") if len(in_shapes[i]) >= 2 else make_transform("none") for i in auto_idx]
        try:
            fixed = finish(fixed_ts)
            node.out_shape_merge([r.out_shape for r in fixed])
            return fixed
        except ShapeError as exc:
            first_error = exc
    shapes = ", ".join(fmt_shape(s) for s in in_shapes)
    raise ShapeError(
        f"{dst} ({node.type_name}) cannot combine its inputs ({shapes}), and no automatic transform fits: "
        f"{_why(first_error)}",
        node_id=dst,
        hint="Set a transform on one of the connections into it, or change the layers that feed it.",
    )
