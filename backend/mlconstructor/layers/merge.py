"""Merge nodes — combine SEVERAL inputs into one (the heart of non-linear graphs).

A merge node has ``is_merge = True``; the builder gives it the list of its input
shapes (``out_shape_merge``) and a function that combines the input tensors
(``merge_fn``). Shapes here exclude the batch dimension, so a feature-axis ``dim``
maps to tensor dim ``dim + 1`` at runtime.
"""

from __future__ import annotations

from typing import Callable

import torch

from ..registry import register
from ..schema import ParamDef
from ..shapes import Shape, ShapeError
from .base import LayerNode


class _Merge(LayerNode):
    category = "merge"
    is_merge = True

    # The single-input methods are never used for merge nodes, but LayerNode is
    # abstract, so provide them. The builder dispatches on ``is_merge``.
    def out_shape(self, in_shape: Shape | None) -> Shape:
        raise ShapeError(f"{self.type_name} {self.id!r} is a merge node — wire several inputs into it")

    def build(self, in_shape: Shape | None):
        return None

    # -- to implement by each merge node -------------------------------------
    def out_shape_merge(self, in_shapes: list[Shape]) -> Shape:  # pragma: no cover - overridden
        raise NotImplementedError

    def merge_fn(self) -> Callable[[list["torch.Tensor"]], "torch.Tensor"]:  # pragma: no cover
        raise NotImplementedError


@register("Add")
class AddNode(_Merge):
    """Element-wise sum of its inputs. All inputs must have the same shape
    (use it for residual / skip connections)."""

    label = "Add"

    def out_shape_merge(self, in_shapes: list[Shape]) -> Shape:
        if len(in_shapes) < 2:
            raise ShapeError(f"Add {self.id!r} needs >= 2 inputs, got {len(in_shapes)}")
        first = tuple(in_shapes[0])
        for s in in_shapes[1:]:
            if tuple(s) != first:
                raise ShapeError(
                    f"Add {self.id!r}: all inputs must have the SAME shape; got "
                    f"{[tuple(x) for x in in_shapes]}"
                )
        return first

    def merge_fn(self) -> Callable:
        def _add(xs: list) -> "torch.Tensor":
            out = xs[0]
            for t in xs[1:]:
                out = out + t
            return out

        return _add


@register("Concat")
class ConcatNode(_Merge):
    """Concatenate inputs along a feature axis (``dim``, 0-based, excluding the
    batch dim). Inputs must match on every other axis."""

    label = "Concat"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(
                name="dim",
                type="int",
                default=0,
                min=0,
                label="Dim",
                help="Feature axis to concatenate along (0-based, NOT counting the batch dim)",
            )
        ]

    def out_shape_merge(self, in_shapes: list[Shape]) -> Shape:
        if len(in_shapes) < 2:
            raise ShapeError(f"Concat {self.id!r} needs >= 2 inputs, got {len(in_shapes)}")
        dim = int(self.p("dim", 0))
        shapes = [tuple(s) for s in in_shapes]
        rank = len(shapes[0])
        if not (0 <= dim < rank):
            raise ShapeError(f"Concat {self.id!r}: dim {dim} out of range for {rank}-D inputs")
        for s in shapes:
            if len(s) != rank:
                raise ShapeError(f"Concat {self.id!r}: inputs must share rank; got {shapes}")
            for ax in range(rank):
                if ax != dim and s[ax] != shapes[0][ax]:
                    raise ShapeError(
                        f"Concat {self.id!r}: inputs must match on all axes except dim {dim}; got {shapes}"
                    )
        out = list(shapes[0])
        out[dim] = sum(s[dim] for s in shapes)
        return tuple(out)

    def merge_fn(self) -> Callable:
        dim = int(self.p("dim", 0))

        def _cat(xs: list) -> "torch.Tensor":
            return torch.cat(xs, dim=dim + 1)  # +1: tensors carry a batch dim at axis 0

        return _cat
