"""Convolution layers."""

from __future__ import annotations

import torch.nn as nn

from ..registry import register
from ..schema import ParamDef
from .. import fixes as fx
from ..shapes import Shape, ShapeError, fmt
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
                f"Conv2d “{self.id}”: a {k}×{k} kernel does not fit in the {h}×{w} picture arriving "
                f"(padding {pad}, stride {s}) — nothing would be left",
                code="too_small", param="kernel_size", fixes=window_fixes(self.id, h, w, k, s, pad),
            )
        return (oc, ho, wo)

    def build(self, in_shape: Shape | None) -> nn.Module:
        c, _h, _w = self._require_shape(in_shape, ndim=3, what="(C,H,W)")
        oc, k, s, pad = self._geom()
        return nn.Conv2d(c, oc, kernel_size=k, stride=s, padding=pad)


def window_fixes(nid: str, h: int, w: int, k: int, s: int, pad: int) -> list[dict]:
    """Stage 21 — ways to make a k×k window (stride s, padding pad) fit an h×w picture:
    the smallest change first."""
    def fits(k2: int, s2: int, p2: int) -> bool:
        return (min(h, w) + 2 * p2 - k2) // s2 + 1 >= 1

    out = []
    if fits(k, s, k // 2) and k // 2 != pad:
        out.append(fx.set_param(nid, "padding", k // 2, f"Add padding {k // 2} (the picture keeps its size)"))
    k2 = min(h, w) + 2 * pad
    if 1 <= k2 < k and fits(k2, s, pad):
        out.append(fx.set_param(nid, "kernel_size", k2, f"Use a {k2}×{k2} kernel"))
    if s > 1 and fits(k, 1, pad):
        out.append(fx.set_param(nid, "stride", 1, "Set stride to 1"))
    return out


@register("ConvTranspose2d")
class ConvTranspose2dNode(LayerNode):
    """2-D transposed convolution (learnable up-sampling). ``in_channels`` is
    inferred; output H/W grow with the stride."""

    category = "conv"
    label = "ConvTranspose2d"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="out_channels", type="int", default=16, min=1, label="Out channels"),
            ParamDef(name="kernel_size", type="int", default=2, min=1, label="Kernel"),
            ParamDef(name="stride", type="int", default=2, min=1, label="Stride"),
            ParamDef(name="padding", type="int", default=0, min=0, label="Padding"),
        ]

    def _geom(self):
        return (
            int(self.p("out_channels", 16)),
            int(self.p("kernel_size", 2)),
            int(self.p("stride", 2)),
            int(self.p("padding", 0)),
        )

    def out_shape(self, in_shape: Shape | None) -> Shape:
        c, h, w = self._require_shape(in_shape, ndim=3, what="(C,H,W)")
        oc, k, s, pad = self._geom()
        ho = (h - 1) * s - 2 * pad + k
        wo = (w - 1) * s - 2 * pad + k
        if ho <= 0 or wo <= 0:
            raise ShapeError(
                f"ConvTranspose2d “{self.id}”: padding {pad} cuts more than the {h}×{w} picture grows to — "
                f"nothing would be left", code="too_small", param="padding",
                fixes=[fx.set_param(self.id, "padding", 0, "Set padding to 0")],
            )
        return (oc, ho, wo)

    def build(self, in_shape: Shape | None) -> nn.Module:
        c, _h, _w = self._require_shape(in_shape, ndim=3, what="(C,H,W)")
        oc, k, s, pad = self._geom()
        return nn.ConvTranspose2d(c, oc, kernel_size=k, stride=s, padding=pad)
