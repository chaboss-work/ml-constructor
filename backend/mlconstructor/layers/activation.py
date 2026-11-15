"""Activation functions — shape-preserving element-wise nodes."""

from __future__ import annotations

import torch.nn as nn

from ..registry import register
from ..shapes import Shape
from .base import LayerNode


class _Activation(LayerNode):
    """Common base: activations keep the shape and own no parameters."""

    category = "activation"
    _module: type[nn.Module] = nn.Identity

    def out_shape(self, in_shape: Shape | None) -> Shape:
        return self._require_shape(in_shape)

    def build(self, in_shape: Shape | None) -> nn.Module:
        return self._module()


@register("ReLU")
class ReLUNode(_Activation):
    """Rectified Linear Unit."""

    label = "ReLU"
    _module = nn.ReLU


@register("Sigmoid")
class SigmoidNode(_Activation):
    """Logistic sigmoid."""

    label = "Sigmoid"
    _module = nn.Sigmoid


@register("Tanh")
class TanhNode(_Activation):
    """Hyperbolic tangent."""

    label = "Tanh"
    _module = nn.Tanh
