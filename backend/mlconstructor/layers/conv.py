"""2-D convolution."""

from __future__ import annotations

import torch.nn as nn

from ..registry import register
from ..schema import ParamDef
from ..shapes import Shape, ShapeError
from .base import LayerNode


@register("Conv2d")
class Conv2dNode(LayerNode):
    """2-D convolution over a (C,H,W) feature map. ``in_channels`` is inferred
    from the incoming shape; output H/W are computed and validated."""

    category = "conv"
    label = "Conv2d"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="out_channels", type="int", default=16, min=1, label="Out channels"),
            ParamDef(name="kernel_size", type="int", default=3, min=1, label="Kernel"),
            ParamDef(name="stride", type="int", default=1, min=1, label="Stride"),
            ParamDef(name="padding", type="int", default=0, min=0, label="Padding"),
        ]

    def _geom(self):
        return (
            int(self.p("out_channels", 16)),
            int(self.p("kernel_size", 3)),
            int(self.p("stride", 1)),
            int(self.p("padding", 0)),
        )

    def out_shape(self, in_shape: Shape | None) -> Shape:
        c, h, w = self._require_shape(in_shape, ndim=3, what="(C,H,W)")
        oc, k, s, pad = self._geom()
        ho = (h + 2 * pad - k) // s + 1
        wo = (w + 2 * pad - k) // s + 1
        if ho <= 0 or wo <= 0:
            raise ShapeError(
                f"Conv2d {self.id!r}: non-positive output ({ho}x{wo}) for input {tuple(in_shape)} "
                f"with kernel {k}, stride {s}, padding {pad}"
            )
        return (oc, ho, wo)

    def build(self, in_shape: Shape | None) -> nn.Module:
        c, h, w = self._require_shape(in_shape, ndim=3, what="(C,H,W)")
        oc, k, s, pad = self._geom()
        return nn.Conv2d(c, oc, kernel_size=k, stride=s, padding=pad)
