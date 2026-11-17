"""Compile a :class:`~mlconstructor.schema.GraphSpec` into a runnable model.

Supports a general **directed acyclic graph**: a node's output may fan out to
several nodes (branch), and **merge** nodes (Add/Concat) combine several inputs.
The topology is validated and tensor shapes are propagated through the whole
graph, so invalid wiring is rejected with a clear, node-pointed error before any
training happens.
"""

from __future__ import annotations

from collections import deque

from .blocks import GraphModule
from .layers.base import LayerNode
from .registry import get_layer_cls
from .schema import BuildResult, GraphSpec, LayerSummary, NodeSpec
from .shapes import Shape, ShapeError, as_list


def _node_from_spec(spec: NodeSpec) -> LayerNode:
    return get_layer_cls(spec.type)(spec.id, spec.params)


def topology(spec: GraphSpec):
    """Topological order (Kahn) plus predecessor/successor maps."""
    ids = [n.id for n in spec.nodes]
    idset = set(ids)
    if len(idset) != len(ids):
        raise ValueError("duplicate node ids in graph")

    indeg = {i: 0 for i in ids}
    succ: dict[str, list[str]] = {i: [] for i in ids}
    preds: dict[str, list[str]] = {i: [] for i in ids}
    for edge in spec.edges:
        src, dst = edge
        if src not in idset or dst not in idset:
            raise ValueError(f"edge references unknown node: {edge}")
        succ[src].append(dst)
        preds[dst].append(src)
        indeg[dst] += 1

    queue = deque(i for i in ids if indeg[i] == 0)
    order: list[str] = []
    deg = dict(indeg)
    while queue:
        n = queue.popleft()
        order.append(n)
        for m in succ[n]:
            deg[m] -= 1
            if deg[m] == 0:
                queue.append(m)
    if len(order) != len(ids):
        raise ValueError("graph has a cycle")
    return order, preds, succ


def build_model(spec: GraphSpec) -> tuple[GraphModule, BuildResult]:
    nodes = {n.id: _node_from_spec(n) for n in spec.nodes}
    order, preds, succ = topology(spec)

    source_ids = [i for i in nodes if not nodes[i].needs_input]
    if len(source_ids) != 1:
        raise ShapeError(f"graph must have exactly one Input node, found {len(source_ids)}")
    input_id = source_ids[0]

    sinks = [i for i in nodes if len(succ[i]) == 0]
    outputs = [i for i in sinks if nodes[i].type_name == "Output"] or sinks
    if len(outputs) != 1:
        raise ShapeError(f"graph must have exactly one output (sink), found {len(sinks)}")
    output_id = outputs[0]

    out_shape: dict[str, Shape] = {}
    modules: dict[str, object] = {}
    merges: dict[str, object] = {}
    summaries: list[LayerSummary] = []

    for nid in order:
        node = nodes[nid]
        ps = preds[nid]
        if not node.needs_input:  # Input / source
            if ps:
                raise ShapeError(f"source node {nid!r} ({node.type_name}) must have no inputs, has {len(ps)}")
            sh = node.out_shape(None)
            mod = node.build(None)
            node_in: Shape | None = None
        elif node.is_merge:  # Add / Concat
            if len(ps) < 2:
                raise ShapeError(
                    f"merge node {nid!r} ({node.type_name}) needs >= 2 inputs, got {len(ps)} — wire several edges into it"
                )
            sh = node.out_shape_merge([out_shape[p] for p in ps])
            mod = None
            merges[nid] = node.merge_fn()
            node_in = None
        else:  # ordinary single-input layer
            if len(ps) != 1:
                raise ShapeError(
                    f"node {nid!r} ({node.type_name}) needs exactly 1 input, got {len(ps)}; "
                    f"use a merge node (Add/Concat) to combine branches"
                )
            in_sh = out_shape[ps[0]]
            sh = node.out_shape(in_sh)
            mod = node.build(in_sh)
            node_in = in_sh

        out_shape[nid] = sh
        if mod is not None:
            modules[nid] = mod
        nparams = sum(p.numel() for p in mod.parameters()) if mod is not None else 0
        summaries.append(
            LayerSummary(
                id=nid, type=node.type_name, in_shape=as_list(node_in), out_shape=as_list(sh), num_params=int(nparams)
            )
        )

    model = GraphModule(
        order=order, preds=preds, modules_by_id=modules, merges=merges, input_id=input_id, output_id=output_id
    )
    total = sum(s.num_params for s in summaries)
    return model, BuildResult(ok=True, order=order, layers=summaries, total_params=total)
