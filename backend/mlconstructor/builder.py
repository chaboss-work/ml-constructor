"""Turn a :class:`~mlconstructor.schema.GraphSpec` into a runnable model.

v1 supports a single linear chain (one source ``Input`` → one sink). Branching
and merge nodes (Add, Concat, residuals) are a planned extension; the topology
check below rejects them with a clear message rather than silently mis-building.
"""

from __future__ import annotations

from collections import deque

from .assembly import build_chain
from .blocks import NeuralNetwork
from .layers.base import LayerNode
from .registry import get_layer_cls
from .schema import BuildResult, GraphSpec, NodeSpec
from .shapes import ShapeError


def _node_from_spec(spec: NodeSpec) -> LayerNode:
    return get_layer_cls(spec.type)(spec.id, spec.params)


def topological_order(spec: GraphSpec) -> list[str]:
    """Kahn's algorithm with the v1 single-chain constraint."""
    ids = [n.id for n in spec.nodes]
    idset = set(ids)
    if len(idset) != len(ids):
        raise ValueError("duplicate node ids in graph")

    indeg = {i: 0 for i in ids}
    succ: dict[str, list[str]] = {i: [] for i in ids}
    pred: dict[str, list[str]] = {i: [] for i in ids}
    for src, dst in spec.edges:
        if src not in idset or dst not in idset:
            raise ValueError(f"edge references unknown node: {[src, dst]}")
        succ[src].append(dst)
        pred[dst].append(src)
        indeg[dst] += 1

    for i in ids:
        if len(pred[i]) > 1 or len(succ[i]) > 1:
            raise ShapeError(
                f"node {i!r} has {len(pred[i])} input(s) and {len(succ[i])} output(s); "
                f"v1 supports a single linear chain (branch/merge nodes coming later)"
            )

    sources = [i for i in ids if indeg[i] == 0]
    if len(sources) != 1:
        raise ShapeError(f"expected exactly one source (Input) node, found {len(sources)}")

    queue = deque(sources)
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
    return order


def build_model(spec: GraphSpec) -> tuple[NeuralNetwork, BuildResult]:
    """Assemble the model and a per-layer summary (shapes + parameter counts)."""
    order = topological_order(spec)
    by_id = {n.id: n for n in spec.nodes}
    nodes = [_node_from_spec(by_id[i]) for i in order]
    chain = build_chain(nodes, in_shape=None)
    model = NeuralNetwork(chain.modules)
    total = sum(s.num_params for s in chain.summaries)
    result = BuildResult(ok=True, order=order, layers=chain.summaries, total_params=total)
    return model, result
