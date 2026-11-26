"""Fully-connected (dense) layer."""

from __future__ import annotations

import torch.nn as nn

from ..registry import register
from ..schema import ParamDef
from ..shapes import Shape
from .base import LayerNode


@register("Linear")
class LinearNode(LayerNode):
    """Linear / fully-connected layer. ``in_features`` is inferred automatically
    from the incoming shape (which must be 1-D — put a Flatten before it)."""

    category = "dense"
    label = "Linear"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="out_features", type="int", default=64, min=1, label="Out features"),
            ParamDef(name="bias", type="bool", default=True, label="Bias"),
            # Stage 19: the same Linear on every token of a sequence (or every point)
            ParamDef(name="per_token", type="bool", default=False, label="Every position (last axis)",
                     help="On: applied to each token of a sequence (L, F) -> (L, out), as transformers do. "
                          "Off: takes one flat vector."),
        ]

    def out_shape(self, in_shape: Shape | None) -> Shape:
        if bool(self.p("per_token", False)):
            sh = self._require_shape(in_shape)
            return (*[int(d) for d in sh[:-1]], int(self.p("out_features", 64)))
        self._require_shape(in_shape, ndim=1, what="(use Flatten before Linear)")
        return (int(self.p("out_features", 64)),)

    def build(self, in_shape: Shape | None) -> nn.Module:
        if bool(self.p("per_token", False)):
            shape = self._require_shape(in_shape)
            return nn.Linear(int(shape[-1]), int(self.p("out_features", 64)), bias=bool(self.p("bias", True)))
        shape = self._require_shape(in_shape, ndim=1, what="(use Flatten before Linear)")
        return nn.Linear(shape[0], int(self.p("out_features", 64)), bias=bool(self.p("bias", True)))
