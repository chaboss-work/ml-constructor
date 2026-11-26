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
    #: Stage 19 — for each module: which of the chain's extra inputs it also takes
    port_maps: list[tuple[int, ...]] = field(default_factory=list)


def build_chain(
    nodes: list[LayerNode],
    in_shape: Shape | None,
    extra_shapes: list[Shape] | None = None,
    extra_names: tuple[str, ...] = (),
) -> ChainResult:
    """Propagate shapes through ``nodes`` and build their modules.

    Raises :class:`~mlconstructor.shapes.ShapeError` (with the offending node id)
    if a connection is invalid — the backend never lets a bad graph reach torch.

    Stage 19: a layer with several inputs (e.g. a DiT block reads a condition)
    takes the chain's value as its first input and the others by name from
    ``extra_names`` / ``extra_shapes`` — the same for every layer of the chain.
    """
    res = ChainResult(out_shape=in_shape)
    extra = dict(zip(extra_names, extra_shapes or []))
    shape = in_shape
    for node in nodes:
        node_in: Shape | None = None if not node.needs_input else shape
        ports = node.input_ports()
        idx: tuple[int, ...] = ()
        if ports:
            missing = [n for n in ports[1:] if n not in extra]
            if missing:
                from .shapes import ShapeError
                raise ShapeError(
                    f"{node.id} ({node.type_name}) needs the input(s) {', '.join(missing)}, which this chain does not get",
                    node_id=node.id,
                )
            shapes = [node.check_port(0, shape)] + [node.check_port(i + 1, extra[n]) for i, n in enumerate(ports[1:])]
            out = node.out_shape_ports(shapes)
            module = node.build_ports(shapes)
            idx = tuple(extra_names.index(n) for n in ports[1:])
        else:
            out = node.out_shape(shape)      # may raise ShapeError
            module = node.build(shape)       # may raise ShapeError
        nparams = sum(p.numel() for p in module.parameters()) if module is not None else 0
        if module is not None:
            res.modules.append(module)
            res.port_maps.append(idx)
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
