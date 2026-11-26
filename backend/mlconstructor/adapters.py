"""Smart adapters (Stage 5).

When a connection's source shape does not fit the target layer, instead of a
dead end the builder's own shape logic is reused to *propose* an adapter node to
drop in between — e.g. a Flatten before a Linear. The editor shows the proposal
inline and inserts it on one click; :func:`apply_adapter` performs the same
insertion on a spec (used by tests and any server-side auto-fix).

The framework is open: each candidate is a pure shape transform paired with the
node it corresponds to. Flatten (collapse a feature map or a sequence to a
vector) is the first concrete adapter; reshape / pooling adapters slot in the
same way — add a candidate and the proposal machinery picks it up.
"""

from __future__ import annotations

from math import prod

from .builder import plan_graph
from .layers.base import LayerNode
from .schema import AdapterProposal, GraphSpec, NodeSpec
from .shapes import Shape, ShapeError


def _flatten_candidate(shape: Shape):
    """Collapse a rank>=2 shape — (C,H,W), (L,E), … — into a 1-D vector."""
    if len(shape) >= 2:
        return "Flatten", {}, (int(prod(shape)),)
    return None


#: Ordered candidate adapters. Each maps an input shape to
#: ``(adapter_type, params, out_shape)``, or ``None`` when it does not apply.
_CANDIDATES = [_flatten_candidate]


def _reason(adapter_type: str, target_type: str, in_shape: Shape, out_shape: Shape) -> str:
    if adapter_type == "Flatten":
        return (
            f"{target_type} expects a 1-D input, but receives {tuple(in_shape)} — "
            f"Flatten collapses it to a {out_shape[0]}-feature vector"
        )
    return f"Insert {adapter_type} to reconcile shapes {tuple(in_shape)} → {tuple(out_shape)}"


def _first_fixing_adapter(src_shape: Shape, target: LayerNode):
    """The first candidate whose output ``target`` accepts, or ``None``."""
    for candidate in _CANDIDATES:
        res = candidate(src_shape)
        if res is None:
            continue
        atype, params, out_shape = res
        try:
            target.out_shape(out_shape)
        except ShapeError:
            continue
        return atype, params, out_shape
    return None


def suggest_for_graph(spec: GraphSpec) -> list[AdapterProposal]:
    """For every single-input edge whose target cannot consume the source shape,
    propose an adapter node if one fixes it.

    Since Stage 14 connections adapt shapes on their own ("auto"), so this only
    fires where a connection is pinned to a transform that does not fit (e.g.
    "Direct"). It never raises on a shape mismatch and tolerates a partial graph
    (any number of Input/Output nodes), so the editor can call it while wiring.
    """
    plan = plan_graph(spec, tolerant=True)
    proposals: list[AdapterProposal] = []
    for exc in plan.errors:
        if not exc.edge:
            continue
        src, dst = exc.edge
        node = plan.nodes.get(dst)
        if node is None or node.is_merge or node.input_ports() or src not in plan.out_shapes:
            continue
        in_sh = plan.out_shapes[src]
        fix = _first_fixing_adapter(in_sh, node)
        if fix is None:
            continue
        atype, params, adapted = fix
        proposals.append(
            AdapterProposal(
                source_id=src,
                target_id=dst,
                adapter_type=atype,
                params=params,
                reason=_reason(atype, node.type_name, in_sh, adapted),
                in_shape=list(in_sh),
                out_shape=list(adapted),
            )
        )
    return proposals


def _fresh_adapter_id(existing: set[str], adapter_type: str) -> str:
    base = adapter_type.lower()
    i = 1
    while f"{base}_auto_{i}" in existing:
        i += 1
    return f"{base}_auto_{i}"


def apply_adapter(spec: GraphSpec, proposal: AdapterProposal) -> GraphSpec:
    """Return a new spec with the adapter node inserted on its edge:
    ``source -> adapter -> target`` replaces ``source -> target``."""
    existing = {n.id for n in spec.nodes}
    new_id = _fresh_adapter_id(existing, proposal.adapter_type)
    nodes = list(spec.nodes) + [NodeSpec(id=new_id, type=proposal.adapter_type, params=dict(proposal.params))]
    edges: list[list[str]] = []
    replaced = False
    for src, dst in spec.edges:
        if src == proposal.source_id and dst == proposal.target_id:
            edges.extend([[proposal.source_id, new_id], [new_id, proposal.target_id]])
            replaced = True
        else:
            edges.append([src, dst])
    if not replaced:
        raise ValueError(f"edge {proposal.source_id!r} -> {proposal.target_id!r} not found in the graph")
    # the replaced edge's own setting goes with it; every other connection keeps its setting
    settings = [
        s for s in spec.edge_settings if not (s.source == proposal.source_id and s.target == proposal.target_id)
    ]
    return GraphSpec(nodes=nodes, edges=edges, train=spec.train, edge_settings=settings)
