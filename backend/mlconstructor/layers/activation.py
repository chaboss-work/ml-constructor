"""Activation functions — shape-preserving element-wise nodes."""

from __future__ import annotations

import torch.nn as nn

from ..registry import register
from ..schema import ParamDef
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


@register("LeakyReLU")
class LeakyReLUNode(_Activation):
    """Leaky ReLU with a small negative slope."""

    label = "LeakyReLU"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [ParamDef(name="negative_slope", type="float", default=0.01, min=0, label="Negative slope")]

    def build(self, in_shape: Shape | None) -> nn.Module:
        return nn.LeakyReLU(float(self.p("negative_slope", 0.01)))


@register("GELU")
class GELUNode(_Activation):
    """Gaussian Error Linear Unit (common in transformers)."""

    label = "GELU"
    _module = nn.GELU


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


@register("Softmax")
class SoftmaxNode(_Activation):
    """Softmax over one axis (default: the last feature axis)."""

    label = "Softmax"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [ParamDef(name="dim", type="int", default=-1, label="Dim", help="Axis for softmax; -1 = last")]

    def build(self, in_shape: Shape | None) -> nn.Module:
        return nn.Softmax(dim=int(self.p("dim", -1)))
