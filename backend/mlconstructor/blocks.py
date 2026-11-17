"""The *mother* modules that compose per-layer modules into bigger units.

* :class:`NeuralNetwork` — a plain sequential model (a linear chain).
* :class:`Block` — a reusable, repeatable group of layers ("N× transformer blocks").
* :class:`GraphModule` — executes an arbitrary DAG: branches and merge nodes
  (Add/Concat). This is what :func:`~mlconstructor.builder.build_model` returns.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn

from .layers.base import LayerNode
from .shapes import Shape


class NeuralNetwork(nn.Module):
    """Top-level model: a sequence of built layer modules (a linear chain)."""

    def __init__(self, ordered_modules: Sequence[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(ordered_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Block(nn.Module):
    """An ordered group of layers, stacked ``repeat`` times.

    ``make_nodes`` is a zero-argument factory returning a *fresh* list of
    :class:`LayerNode` on each call, so every repeat gets its own modules. Shapes
    are propagated across repeats; the output shape is exposed as :attr:`out_shape`.
    """

    def __init__(self, make_nodes: Callable[[], list[LayerNode]], in_shape: Shape, repeat: int = 1):
        super().__init__()
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}")
        from .assembly import build_chain  # local import avoids an import cycle

        modules: list[nn.Module] = []
        shape: Shape | None = in_shape
        for _ in range(repeat):
            chain = build_chain(make_nodes(), shape)
            modules.extend(chain.modules)
            shape = chain.out_shape
        self.layers = nn.ModuleList(modules)
        self.repeat = repeat
        self.out_shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class GraphModule(nn.Module):
    """Executes an arbitrary DAG of nodes.

    For each node in topological order: take the predecessors' outputs (combined
    by the node's merge function when it is a merge node, else the single
    predecessor's output), optionally apply the node's module, and store the
    result for downstream nodes. Returns the output node's value.
    """

    def __init__(
        self,
        order: Sequence[str],
        preds: dict[str, list[str]],
        modules_by_id: dict[str, nn.Module],
        merges: dict[str, Callable],
        input_id: str,
        output_id: str,
    ):
        super().__init__()
        self.order = list(order)
        self.preds = {k: list(v) for k, v in preds.items()}
        self.input_id = input_id
        self.output_id = output_id
        self.merge_ids = set(merges.keys())
        self._merges = dict(merges)  # id -> callable(list[Tensor]) -> Tensor (plain functions)
        self._mods = nn.ModuleDict({self._key(k): v for k, v in modules_by_id.items()})

    @staticmethod
    def _key(node_id: str) -> str:
        # ModuleDict keys cannot contain '.'; make a safe, unique-ish key.
        return "n_" + "".join(c if c.isalnum() else "_" for c in node_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs: dict[str, torch.Tensor] = {}
        for nid in self.order:
            ps = self.preds[nid]
            if nid == self.input_id:
                v = x
            elif nid in self.merge_ids:
                v = self._merges[nid]([outs[p] for p in ps])
            else:
                v = outs[ps[0]]
            key = self._key(nid)
            if key in self._mods:
                v = self._mods[key](v)
            outs[nid] = v
        return outs[self.output_id]
