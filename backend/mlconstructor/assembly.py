"""The shared chain assembler.

Given an ordered list of :class:`~mlconstructor.layers.base.LayerNode` and a
starting shape, it propagates shapes node-by-node, builds each torch module, and
records a per-node summary. Both the graph :mod:`~mlconstructor.builder` and the
repeatable :class:`~mlconstructor.blocks.Block` are built on top of this one
function, so shape inference and parameter counting live in a single place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn

from .layers.base import LayerNode
from .schema import LayerSummary
from .shapes import Shape, as_list


@dataclass
class ChainResult:
    modules: list[nn.Module] = field(default_factory=list)   # non-null, in order
    summaries: list[LayerSummary] = field(default_factory=list)
    out_shape: Shape | None = None


def build_chain(nodes: list[LayerNode], in_shape: Shape | None) -> ChainResult:
    """Propagate shapes through ``nodes`` and build their modules.

    Raises :class:`~mlconstructor.shapes.ShapeError` (with the offending node id)
    if a connection is invalid — the backend never lets a bad graph reach torch.
    """
    res = ChainResult(out_shape=in_shape)
    shape = in_shape
    for node in nodes:
        node_in: Shape | None = None if not node.needs_input else shape
        out = node.out_shape(shape)      # may raise ShapeError
        module = node.build(shape)       # may raise ShapeError
        nparams = sum(p.numel() for p in module.parameters()) if module is not None else 0
        if module is not None:
            res.modules.append(module)
        res.summaries.append(
            LayerSummary(
                id=node.id,
                type=node.type_name,
                in_shape=as_list(node_in),
                out_shape=as_list(out),
                num_params=int(nparams),
            )
        )
        shape = out
    res.out_shape = shape
    return res
