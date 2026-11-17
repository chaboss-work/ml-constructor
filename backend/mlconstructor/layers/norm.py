"""Normalization layers."""

from __future__ import annotations

import torch.nn as nn

from ..registry import register
from ..schema import ParamDef
from ..shapes import Shape, ShapeError
from .base import LayerNode


@register("BatchNorm2d")
class BatchNorm2dNode(LayerNode):
    """Batch normalization over the channels of a (C,H,W) feature map."""

    category = "norm"
    label = "BatchNorm2d"

    def out_shape(self, in_shape: Shape | None) -> Shape:
        return self._require_shape(in_shape, ndim=3, what="(C,H,W)")

    def build(self, in_shape: Shape | None) -> nn.Module:
        c, _h, _w = self._require_shape(in_shape, ndim=3, what="(C,H,W)")
        return nn.BatchNorm2d(c)


@register("BatchNorm1d")
class BatchNorm1dNode(LayerNode):
    """Batch normalization for 1-D data: (features,) or (C, L)."""

    category = "norm"
    label = "BatchNorm1d"

    def _channels(self, in_shape: Shape | None) -> int:
        sh = self._require_shape(in_shape)
        if len(sh) not in (1, 2):
            raise ShapeError(f"BatchNorm1d {self.id!r} expects (features,) or (C,L), got {tuple(sh)}")
        return sh[0]

    def out_shape(self, in_shape: Shape | None) -> Shape:
        self._channels(in_shape)
        return self._require_shape(in_shape)

    def build(self, in_shape: Shape | None) -> nn.Module:
        return nn.BatchNorm1d(self._channels(in_shape))


@register("LayerNorm")
class LayerNormNode(LayerNode):
    """Layer normalization over the whole per-sample shape."""

    category = "norm"
    label = "LayerNorm"

    def out_shape(self, in_shape: Shape | None) -> Shape:
        return self._require_shape(in_shape)

    def build(self, in_shape: Shape | None) -> nn.Module:
        sh = self._require_shape(in_shape)
        return nn.LayerNorm(tuple(int(x) for x in sh))


@register("GroupNorm")
class GroupNormNode(LayerNode):
    """Group normalization over channels (first axis). ``num_groups`` must divide channels."""

    category = "norm"
    label = "GroupNorm"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [ParamDef(name="num_groups", type="int", default=8, min=1, label="Groups")]

    def _check(self, in_shape: Shape):
        c = in_shape[0]
        g = int(self.p("num_groups", 8))
        if c % g != 0:
            raise ShapeError(f"GroupNorm {self.id!r}: channels {c} not divisible by num_groups {g}")
        return c, g

    def out_shape(self, in_shape: Shape | None) -> Shape:
        sh = self._require_shape(in_shape)
        self._check(sh)
        return sh

    def build(self, in_shape: Shape | None) -> nn.Module:
        sh = self._require_shape(in_shape)
        c, g = self._check(sh)
        return nn.GroupNorm(g, c)
