"""Source and sink nodes: ``Input`` (declares the tensor shape) and ``Output``."""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from ..registry import register
from ..schema import ParamDef
from ..shapes import Shape, ShapeError
from .base import LayerNode


@register("Input")
class InputNode(LayerNode):
    """Declares the per-sample input shape (no batch dim), e.g. (1,28,28)."""

    category = "io"
    label = "Input"
    needs_input = False

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(
                name="shape",
                type="intlist",
                default=[1, 28, 28],
                label="Shape",
                help="Per-sample shape without the batch dimension: (C,H,W) or (features,)",
            ),
            # Stage 19: a model can take several inputs (an image, a text, a timestep)
            ParamDef(name="name", type="str", default="", label="Name",
                     help="What this input is (image, text, timestep…): shown on the node and used as the "
                          "argument's name in the exported code"),
        ]

    def out_shape(self, in_shape: Shape | None) -> Shape:
        shape = self.p("shape", [1, 28, 28])
        if not shape or any(int(d) <= 0 for d in shape):
            raise ShapeError(f"Input node {self.id!r} needs a positive 'shape', got {shape!r}")
        return tuple(int(d) for d in shape)

    def build(self, in_shape: Shape | None) -> Optional[nn.Module]:
        return None  # a source contributes no module


@register("Output")
class OutputNode(LayerNode):
    """Marks the network output: one score per class. Pass-through (no parameters).

    It takes a flat vector — a classifier's class scores. Since Stage 14 a
    connection into it flattens a sequence or a feature map automatically.

    Stage 18 — "every position": a language model's (L, vocab) scores, the next
    token guessed at every position at once — how a GPT trains."""

    category = "io"
    label = "Output"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [ParamDef(name="per_token", type="bool", default=False, label="Every position (a language model)",
                         help="On: the next token is scored at every position (L x vocab), as a GPT trains on "
                              "every character at once. Off: one score per class."),
                # Stage 19: generators output a picture, a latent, points…
                ParamDef(name="any_shape", type="bool", default=False, label="Any shape (a picture, a latent…)",
                         help="On: the network's output can be anything — a generated image, a latent, 3-D points. "
                              "Such a network builds and runs here; training it to generate comes later.")]

    def out_shape(self, in_shape: Shape | None) -> Shape:
        if bool(self.p("any_shape", False)):
            return self._require_shape(in_shape)
        if bool(self.p("per_token", False)):
            return self._require_shape(in_shape, ndim=2, what="(L, vocab): scores at every position")
        return self._require_shape(in_shape, ndim=1, what="(one score per class)")

    def build(self, in_shape: Shape | None) -> Optional[nn.Module]:
        return None
