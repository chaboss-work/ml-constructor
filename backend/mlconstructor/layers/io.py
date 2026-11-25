"""Source and sink nodes: ``Input`` (declares the tensor shape) and ``Output``."""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from ..registry import register
from ..schema import ParamDef
from ..shapes import Shape, ShapeError
from .base import LayerNode


@register("Input")
class InputNode(LayerNode):
    """Declares the per-sample input shape (no batch dim), e.g. (1,28,28)."""

    category = "io"
    label = "Input"
    needs_input = False

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(
                name="shape",
                type="intlist",
                default=[1, 28, 28],
                label="Shape",
                help="Per-sample shape without the batch dimension: (C,H,W) or (features,)",
            )
        ]

    def out_shape(self, in_shape: Shape | None) -> Shape:
        shape = self.p("shape", [1, 28, 28])
        if not shape or any(int(d) <= 0 for d in shape):
            raise ShapeError(f"Input node {self.id!r} needs a positive 'shape', got {shape!r}")
        return tuple(int(d) for d in shape)

    def build(self, in_shape: Shape | None) -> Optional[nn.Module]:
        return None  # a source contributes no module


@register("Output")
class OutputNode(LayerNode):
    """Marks the network output: one score per class. Pass-through (no parameters).

    It takes a flat vector — a classifier's class scores. Since Stage 14 a
    connection into it flattens a sequence or a feature map automatically."""

    category = "io"
    label = "Output"

    def out_shape(self, in_shape: Shape | None) -> Shape:
        return self._require_shape(in_shape, ndim=1, what="(one score per class)")

    def build(self, in_shape: Shape | None) -> Optional[nn.Module]:
        return None
