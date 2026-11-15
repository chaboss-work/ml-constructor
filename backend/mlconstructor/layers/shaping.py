"""Shape-changing utility layers."""

from __future__ import annotations

import torch.nn as nn

from ..registry import register
from ..shapes import Shape, numel
from .base import LayerNode


@register("Flatten")
class FlattenNode(LayerNode):
    """Flatten all non-batch dimensions into a single feature vector, e.g.
    (C,H,W) -> (C*H*W,). Typically placed before a Linear layer."""

    category = "shape"
    label = "Flatten"

    def out_shape(self, in_shape: Shape | None) -> Shape:
        shape = self._require_shape(in_shape)
        return (numel(shape),)

    def build(self, in_shape: Shape | None) -> nn.Module:
        return nn.Flatten()
