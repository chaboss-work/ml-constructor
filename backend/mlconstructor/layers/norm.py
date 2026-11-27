"""Normalization layers."""

from __future__ import annotations

import torch.nn as nn

from ..registry import register
from ..schema import ParamDef
from .. import fixes as fx
from ..shapes import Shape, ShapeError, describe_shape
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
            raise ShapeError(f"BatchNorm1d “{self.id}” reads a vector or a sequence, but receives {describe_shape(sh)}",
                             code="wrong_rank", node_id=self.id, facts={"want_rank": 1, "got": list(sh)})
        return sh[0]

    def out_shape(self, in_shape: Shape | None) -> Shape:
        self._channels(in_shape)
        return self._require_shape(in_shape)

    def build(self, in_shape: Shape | None) -> nn.Module:
        return nn.BatchNorm1d(self._channels(in_shape))


@register("LayerNorm")
class LayerNormNode(LayerNode):
    """Layer normalization over the whole per-sample shape — or, with "Each position",
    over the last axis only, every token on its own (as transformers do)."""

    category = "norm"
    label = "LayerNorm"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            # Stage 19: transformers normalize every token on its own
            ParamDef(name="last_axis", type="bool", default=False, label="Each position (last axis)",
                     help="On: every token of a sequence is normalized on its own (ViT, CLIP, Whisper). "
                          "Off: the whole example at once."),
            ParamDef(name="eps", type="float", default=1e-5, min=0.0, label="Epsilon"),
        ]

    def out_shape(self, in_shape: Shape | None) -> Shape:
        return self._require_shape(in_shape)

    def build(self, in_shape: Shape | None) -> nn.Module:
        sh = self._require_shape(in_shape)
        dims = (int(sh[-1]),) if bool(self.p("last_axis", False)) else tuple(int(x) for x in sh)
        return nn.LayerNorm(dims, eps=float(self.p("eps", 1e-5)))


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
            raise ShapeError(f"GroupNorm “{self.id}”: {c} channels do not split into {g} equal groups",
                             code="not_divisible", param="num_groups",
                             fixes=fx.divisor_fixes(self.id, "num_groups", c, g, " channels per group"))
        return c, g

    def out_shape(self, in_shape: Shape | None) -> Shape:
        sh = self._require_shape(in_shape)
        self._check(sh)
        return sh

    def build(self, in_shape: Shape | None) -> nn.Module:
        sh = self._require_shape(in_shape)
        c, g = self._check(sh)
        return nn.GroupNorm(g, c)
