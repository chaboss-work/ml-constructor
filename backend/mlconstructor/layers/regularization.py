"""Regularization layers."""

from __future__ import annotations

import torch.nn as nn

from ..registry import register
from ..schema import ParamDef
from ..shapes import Shape
from .base import LayerNode


@register("Dropout")
class DropoutNode(LayerNode):
    """Randomly zeroes a fraction ``p`` of elements during training. Shape-preserving."""

    category = "reg"
    label = "Dropout"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [ParamDef(name="p", type="float", default=0.5, min=0, max=1, label="Drop probability")]

    def out_shape(self, in_shape: Shape | None) -> Shape:
        return self._require_shape(in_shape)

    def build(self, in_shape: Shape | None) -> nn.Module:
        return nn.Dropout(float(self.p("p", 0.5)))
