"""Pooling layers — down-sample a (C,H,W) feature map over its spatial dims."""

from __future__ import annotations

import torch.nn as nn

from ..registry import register
from ..schema import ParamDef
from ..shapes import Shape, ShapeError
from .base import LayerNode


class _Pool2d(LayerNode):
    category = "pool"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="kernel_size", type="int", default=2, min=1, label="Kernel"),
            ParamDef(name="stride", type="int", default=2, min=1, label="Stride"),
            ParamDef(name="padding", type="int", default=0, min=0, label="Padding"),
        ]

    def _geom(self):
        return int(self.p("kernel_size", 2)), int(self.p("stride", 2)), int(self.p("padding", 0))

    def out_shape(self, in_shape: Shape | None) -> Shape:
        c, h, w = self._require_shape(in_shape, ndim=3, what="(C,H,W)")
        k, s, p = self._geom()
        ho = (h + 2 * p - k) // s + 1
        wo = (w + 2 * p - k) // s + 1
        if ho <= 0 or wo <= 0:
            raise ShapeError(f"{self.type_name} {self.id!r}: non-positive output ({ho}x{wo})")
        return (c, ho, wo)


@register("MaxPool2d")
class MaxPool2dNode(_Pool2d):
    """2-D max pooling."""

    label = "MaxPool2d"

    def build(self, in_shape: Shape | None) -> nn.Module:
        self._require_shape(in_shape, ndim=3, what="(C,H,W)")
        k, s, p = self._geom()
        return nn.MaxPool2d(kernel_size=k, stride=s, padding=p)


@register("AvgPool2d")
class AvgPool2dNode(_Pool2d):
    """2-D average pooling."""

    label = "AvgPool2d"

    def build(self, in_shape: Shape | None) -> nn.Module:
        self._require_shape(in_shape, ndim=3, what="(C,H,W)")
        k, s, p = self._geom()
        return nn.AvgPool2d(kernel_size=k, stride=s, padding=p)


@register("AdaptiveAvgPool2d")
class AdaptiveAvgPool2dNode(LayerNode):
    """2-D adaptive average pooling to a fixed (H,W) output size."""

    category = "pool"
    label = "AdaptiveAvgPool2d"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [ParamDef(name="output_size", type="intlist", default=[1, 1], label="Output size (H,W)")]

    def _size(self):
        sz = self.p("output_size", [1, 1])
        if isinstance(sz, int):
            return (int(sz), int(sz))
        return (int(sz[0]), int(sz[-1]))

    def out_shape(self, in_shape: Shape | None) -> Shape:
        c, _h, _w = self._require_shape(in_shape, ndim=3, what="(C,H,W)")
        oh, ow = self._size()
        return (c, oh, ow)

    def build(self, in_shape: Shape | None) -> nn.Module:
        self._require_shape(in_shape, ndim=3, what="(C,H,W)")
        return nn.AdaptiveAvgPool2d(self._size())
