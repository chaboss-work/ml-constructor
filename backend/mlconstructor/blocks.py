"""The *mother* modules that compose per-layer modules into bigger units.

* :class:`NeuralNetwork` — the top-level model assembled from the whole graph
  (mirrors the user's ``class NeuralNetwork(nn.Module)``): it runs its ordered
  layers in sequence.
* :class:`Block` — a reusable, **repeatable** sub-unit: an ordered group of
  layers stacked ``repeat`` times (think "N× transformer blocks"). Each repeat is
  an independent copy, with shapes propagated so the copies chain correctly.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn

from .layers.base import LayerNode
from .shapes import Shape


class NeuralNetwork(nn.Module):
    """Top-level model: a sequence of built layer modules."""

    def __init__(self, ordered_modules: Sequence[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(ordered_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Block(nn.Module):
    """An ordered group of layers, stacked ``repeat`` times.

    ``make_nodes`` is a zero-argument factory that returns a *fresh* list of
    :class:`LayerNode` on each call, so every repeat gets its own independent
    modules. Shapes are propagated across repeats; the resulting output shape is
    exposed as :attr:`out_shape`.
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
