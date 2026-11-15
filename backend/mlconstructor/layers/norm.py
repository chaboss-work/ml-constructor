"""Normalization layers."""

from __future__ import annotations

import torch.nn as nn

from ..registry import register
from ..shapes import Shape
from .base import LayerNode


@register("BatchNorm2d")
class BatchNorm2dNode(LayerNode):
    """Batch normalization over the channels of a (C,H,W) feature map.
    ``num_features`` is inferred from the incoming channel count."""

    category = "norm"
    label = "BatchNorm2d"

    def out_shape(self, in_shape: Shape | None) -> Shape:
        return self._require_shape(in_shape, ndim=3, what="(C,H,W)")

    def build(self, in_shape: Shape | None) -> nn.Module:
        c, _h, _w = self._require_shape(in_shape, ndim=3, what="(C,H,W)")
        return nn.BatchNorm2d(c)
