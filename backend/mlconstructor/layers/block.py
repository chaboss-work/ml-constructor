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

from .. import fixes as fx
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
            raise ShapeError(f"Block {self.id!r}: repeat must be at least 1, got {reps}", code="param_range",
                             param="repeat", node_id=self.id, fixes=[fx.set_param(self.id, "repeat", 1)])
        return reps

    def _rehome(self, exc: ShapeError, i: int) -> ShapeError:
        """Stage 21 — an error of the i-th inner layer points at this block (the editor
        has no node for an inner layer), and its fixes edit that inner layer."""
        inner_id = f"{self.id}__inner{i}"
        if exc.node_id not in (None, inner_id):
            return exc
        exc.node_id, exc.inner = self.id, i
        exc.args = (str(exc).replace(f"“{inner_id}”", f"(layer {i + 1} inside the block “{self.id}”)")
                    .replace(repr(inner_id), f"(layer {i + 1} inside the block “{self.id}”)"),)
        moved = []
        for f in exc.fixes:
            ops = []
            for o in f.get("ops") or []:
                if o.get("op") == "set_param" and o.get("node") == inner_id:
                    ops.append(fx.op_set_inner_param(self.id, i, o["name"], o["value"]))
                elif inner_id in (o.get("node"), o.get("source"), o.get("target")):
                    ops = None  # an edit of the inner layer's connections: not possible inside a block
                    break
                else:
                    ops.append(o)
            if ops:
                moved.append({"label": f"{f['label']} (layer {i + 1} of the block)", "ops": ops})
        exc.fixes = moved
        return exc

    def _inner_out(self, i: int, node: LayerNode, sh, extra: dict | None = None):
        try:
            ports = node.input_ports()
            if extra is not None and ports:
                return node.out_shape_ports([sh] + [extra[n] for n in ports[1:]])
            return node.out_shape(sh)
        except ShapeError as exc:
            raise self._rehome(exc, i) from None

    def _make_inner_nodes(self) -> list[LayerNode]:
        """A fresh set of inner LayerNode instances (one repetition's worth)."""
        specs = self.p("layers", []) or []
        if not specs:
            raise ShapeError(f"Block {self.id!r} is empty — add at least one inner layer", code="block_empty",
                             node_id=self.id)
        nodes: list[LayerNode] = []
        for i, spec in enumerate(specs):
            remove = fx.fix(f"Remove layer {i + 1} from the block", fx.op_remove_inner(self.id, i))
            try:
                cls = get_layer_cls(spec["type"])
            except (KeyError, TypeError) as exc:
                raise ShapeError(f"Block {self.id!r}: inner layer #{i + 1} is of an unknown type: {exc}",
                                 code="unknown_layer", node_id=self.id, fixes=[remove]) from None
            node = cls(f"{self.id}__inner{i}", spec.get("params", {}))
            if not node.needs_input or node.is_merge or node.type_name in ("Input", "Output"):
                raise ShapeError(
                    f"Block {self.id!r}: inner layer {node.type_name!r} is not allowed inside a block "
                    f"(a block is a linear chain — no Input/Output/merge)",
                    code="block_bad_inner", node_id=self.id, fixes=[remove],
                )
            nodes.append(node)
        return nodes

    def out_shape(self, in_shape: Shape | None) -> Shape:
        if self.input_ports():
            names = ", ".join(self.input_ports()[1:])
            raise ShapeError(
                f"Block {self.id!r} also needs {names} — connect it as a second input of the block",
                node_id=self.id,
                code="block_needs_input",
                hint="The layers inside read another input: wire the main data first, then that input.",
            )
        sh = self._require_shape(in_shape)
        for _ in range(self._repeat()):
            for i, node in enumerate(self._make_inner_nodes()):
                sh = self._inner_out(i, node, sh)
        return sh

    def build(self, in_shape: Shape | None) -> nn.Module:
        sh = self._require_shape(in_shape)
        return Block(self._make_inner_nodes, sh, repeat=self._repeat())

    # -- Stage 19: inner layers with several inputs (a DiT block reads a condition,
    # a decoder block the audio): the block takes those inputs once and hands the
    # same value to every layer that reads it.
    def input_ports(self) -> tuple[str, ...]:
        try:
            nodes = self._make_inner_nodes()
        except ShapeError:
            return ()
        main: str | None = None
        extra: list[str] = []
        for node in nodes:
            ports = node.input_ports()
            if ports:
                main = main or ports[0]
                extra += [n for n in ports[1:] if n not in extra]
        return (main or "input", *extra) if extra else ()

    def check_port(self, index: int, shape: Shape) -> Shape:
        if index == 0:
            return tuple(shape)
        name = self.input_ports()[index]
        for i, node in enumerate(self._make_inner_nodes()):
            ports = node.input_ports()
            if name in ports[1:]:
                try:
                    node.check_port(ports.index(name), shape)
                except ShapeError as exc:
                    raise self._rehome(exc, i) from None
        return tuple(shape)

    def out_shape_ports(self, shapes: list[Shape]) -> Shape:
        extra = dict(zip(self.input_ports()[1:], shapes[1:]))
        sh = self._require_shape(shapes[0])
        for _ in range(self._repeat()):
            for i, node in enumerate(self._make_inner_nodes()):
                sh = self._inner_out(i, node, sh, extra)
        return sh

    def build_ports(self, shapes: list[Shape]) -> nn.Module:
        return Block(self._make_inner_nodes, self._require_shape(shapes[0]), repeat=self._repeat(),
                     extra_shapes=shapes[1:], extra_names=self.input_ports()[1:])
