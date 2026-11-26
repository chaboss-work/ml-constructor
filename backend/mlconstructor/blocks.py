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

    def __init__(
        self,
        make_nodes: Callable[[], list[LayerNode]],
        in_shape: Shape,
        repeat: int = 1,
        extra_shapes: Sequence[Shape] = (),
        extra_names: Sequence[str] = (),
    ):
        super().__init__()
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}")
        from .assembly import build_chain  # local import avoids an import cycle

        modules: list[nn.Module] = []
        port_maps: list[tuple[int, ...]] = []
        shape: Shape | None = in_shape
        for _ in range(repeat):
            chain = build_chain(make_nodes(), shape, list(extra_shapes), tuple(extra_names))
            modules.extend(chain.modules)
            port_maps.extend(chain.port_maps)
            shape = chain.out_shape
        self.layers = nn.ModuleList(modules)
        self.repeat = repeat
        self.out_shape = shape
        #: Stage 19 — which extra inputs (a condition, the audio) each layer also reads
        self.port_maps = port_maps

    def forward(self, x: torch.Tensor, *extra: torch.Tensor) -> torch.Tensor:
        for layer, idx in zip(self.layers, self.port_maps):
            x = layer(x, *[extra[i] for i in idx]) if idx else layer(x)
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
        edge_modules: dict[tuple[str, str], nn.Module] | None = None,
        input_ids: Sequence[str] | None = None,
        port_ids: Sequence[str] = (),
    ):
        super().__init__()
        self.order = list(order)
        self.preds = {k: list(v) for k, v in preds.items()}
        self.input_id = input_id
        #: Stage 19 — every Input node, in the order the model takes them
        self.input_ids = list(input_ids) if input_ids else [input_id]
        #: Stage 19 — layers with several inputs: they get every predecessor's value
        self.port_ids = set(port_ids)
        self.output_id = output_id
        self.merge_ids = set(merges.keys())
        self._merges = dict(merges)  # id -> callable(list[Tensor]) -> Tensor (plain functions)
        self._mods = nn.ModuleDict({self._key(k): v for k, v in modules_by_id.items()})
        # Stage 14: a connection's transform (flatten, fold into an image, …) runs on
        # the value travelling along that edge, before the target node sees it.
        self._edge_keys: dict[tuple[str, str], str] = {}
        edge_mods: dict[str, nn.Module] = {}
        for i, ((src, dst), mod) in enumerate((edge_modules or {}).items()):
            key = f"e{i}_{self._key(src)[2:]}__{self._key(dst)[2:]}"
            self._edge_keys[(src, dst)] = key
            edge_mods[key] = mod
        self._edge_mods = nn.ModuleDict(edge_mods)

    @staticmethod
    def _key(node_id: str) -> str:
        # ModuleDict keys cannot contain '.'; make a safe, unique-ish key.
        return "n_" + "".join(c if c.isalnum() else "_" for c in node_id)

    def _along(self, outs: dict[str, torch.Tensor], src: str, dst: str) -> torch.Tensor:
        """The value ``src`` sends to ``dst``, after the connection's transform."""
        key = self._edge_keys.get((src, dst))
        return self._edge_mods[key](outs[src]) if key is not None else outs[src]

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        if len(xs) != len(self.input_ids):
            raise TypeError(f"this model takes {len(self.input_ids)} input(s) ({', '.join(self.input_ids)}), got {len(xs)}")
        feed = dict(zip(self.input_ids, xs))
        outs: dict[str, torch.Tensor] = {}
        for nid in self.order:
            ps = self.preds[nid]
            key = self._key(nid)
            if nid in feed:
                v = feed[nid]
            elif nid in self.port_ids:  # Stage 19: one value per input, in connection order
                outs[nid] = self._mods[key](*[self._along(outs, p, nid) for p in ps])
                continue
            elif nid in self.merge_ids:
                v = self._merges[nid]([self._along(outs, p, nid) for p in ps])
            else:
                v = self._along(outs, ps[0], nid)
            if key in self._mods:
                v = self._mods[key](v)
            outs[nid] = v
        return outs[self.output_id]
