"""Reusable block: a linear sub-chain of layers, repeated N times (Stage 6).

This exposes the :class:`~mlconstructor.blocks.Block` mother class in the graph.
A ``Repeat`` node carries an inner linear chain (``layers``) and a ``repeat``
count and behaves like any single-input layer: its output shape is the inner
chain applied ``repeat`` times, and it builds into a Block module that stamps the
chain that many times — each repeat gets its own fresh parameters. "Edit the
block once, stamp it many times."
"""

from __future__ import annotations

import torch.nn as nn

from ..blocks import Block
from ..registry import get_layer_cls, register
from ..schema import ParamDef
from ..shapes import Shape, ShapeError
from .base import LayerNode


@register("Repeat")
class RepeatNode(LayerNode):
    """A sub-chain of layers repeated ``repeat`` times.

    ``params``: ``layers`` is the inner chain ``[{"type", "params"}, ...]`` and
    ``repeat`` is how many times it is stamped. The inner layers form a linear
    chain — no Input/Output/merge inside.
    """

    category = "block"
    label = "Block (repeat)"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        # `layers` is a structured sub-graph the editor edits with its own UI,
        # so only the repeat count is a plain, catalogue-editable param here.
        return [ParamDef(name="repeat", type="int", default=2, min=1, label="Repeat ×N")]

    def _repeat(self) -> int:
        reps = int(self.p("repeat", 2))
        if reps < 1:
            raise ShapeError(f"Block {self.id!r}: repeat must be >= 1, got {reps}")
        return reps

    def _make_inner_nodes(self) -> list[LayerNode]:
        """A fresh set of inner LayerNode instances (one repetition's worth)."""
        specs = self.p("layers", []) or []
        if not specs:
            raise ShapeError(f"Block {self.id!r} is empty — add at least one inner layer")
        nodes: list[LayerNode] = []
        for i, spec in enumerate(specs):
            try:
                cls = get_layer_cls(spec["type"])
            except (KeyError, TypeError) as exc:
                raise ShapeError(f"Block {self.id!r}: bad inner layer #{i + 1}: {exc}") from None
            node = cls(f"{self.id}__inner{i}", spec.get("params", {}))
            if not node.needs_input or node.is_merge or node.type_name in ("Input", "Output"):
                raise ShapeError(
                    f"Block {self.id!r}: inner layer {node.type_name!r} is not allowed inside a block "
                    f"(a block is a linear chain — no Input/Output/merge)"
                )
            nodes.append(node)
        return nodes

    def out_shape(self, in_shape: Shape | None) -> Shape:
        sh = self._require_shape(in_shape)
        for _ in range(self._repeat()):
            for node in self._make_inner_nodes():
                sh = node.out_shape(sh)
        return sh

    def build(self, in_shape: Shape | None) -> nn.Module:
        sh = self._require_shape(in_shape)
        return Block(self._make_inner_nodes, sh, repeat=self._repeat())
