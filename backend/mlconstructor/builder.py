"""Compile a :class:`~mlconstructor.schema.GraphSpec` into a runnable model.

Supports a general **directed acyclic graph**: a node's output may fan out to
several nodes (branch), and **merge** nodes (Add/Concat) combine several inputs.
The topology is validated and tensor shapes are propagated through the whole
graph, so invalid wiring is rejected with a clear, node-pointed error before any
training happens.

Stage 14 — **smart connections**: every edge carries a transform from
:mod:`mlconstructor.connections` (``"auto"`` unless the user picked one), so a
connection between layers with different shapes is adapted instead of rejected.
Building is two passes: :func:`plan_graph` works out every shape and every edge's
transform (it can also run *tolerantly*, collecting problems instead of stopping,
for the editor's connection inspector), then :func:`build_model` creates the
modules — layers first, in topological order, then the connections' own modules.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._device import _device_constructors

from .blocks import GraphModule
from .connections import AUTO, Resolved, resolve_merge, resolve_single
from .layers.base import LayerNode
from .registry import get_layer_cls
from .schema import BuildResult, EdgeSummary, GraphSpec, LayerSummary, NodeSpec
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


def edge_settings_map(spec: GraphSpec) -> dict[tuple[str, str], tuple[str, dict]]:
    """``(source, target) -> (transform, params)`` for the edges that have a setting."""
    return {(s.source, s.target): (s.transform or AUTO, dict(s.params)) for s in spec.edge_settings}


@dataclass
class GraphPlan:
    """Every shape and every connection's transform, before any module exists."""

    nodes: dict[str, LayerNode]
    order: list[str]
    preds: dict[str, list[str]]
    succ: dict[str, list[str]]
    input_id: str | None = None
    output_id: str | None = None
    #: what a single-input layer receives (after its connection's transform)
    node_in: dict[str, Shape | None] = field(default_factory=dict)
    out_shapes: dict[str, Shape] = field(default_factory=dict)
    edges: dict[tuple[str, str], Resolved] = field(default_factory=dict)
    #: problems found in tolerant mode (strict mode raises the first one instead)
    errors: list[ShapeError] = field(default_factory=list)


def _at(exc: ShapeError, nid: str) -> ShapeError:
    if exc.node_id is None:
        exc.node_id = nid
    return exc


def plan_graph(spec: GraphSpec, tolerant: bool = False) -> GraphPlan:
    """Propagate shapes through the graph and resolve every connection's transform.

    Strict (the default) raises the first :class:`ShapeError`; ``tolerant=True``
    records each problem in ``plan.errors`` and carries on with whatever can still
    be computed (nodes downstream of a broken one simply get no shape)."""
    nodes = {n.id: _node_from_spec(n) for n in spec.nodes}
    order, preds, succ = topology(spec)
    plan = GraphPlan(nodes=nodes, order=order, preds=preds, succ=succ)
    settings = edge_settings_map(spec)

    def fail(exc: ShapeError) -> None:
        if not tolerant:
            raise exc
        plan.errors.append(exc)

    source_ids = [i for i in nodes if not nodes[i].needs_input]
    if len(source_ids) == 1:
        plan.input_id = source_ids[0]
    else:
        fail(ShapeError(
            f"graph must have exactly one Input node, found {len(source_ids)}",
            hint="Keep a single Input node — it declares the shape of one example.",
        ))
    sinks = [i for i in nodes if len(succ[i]) == 0]
    outputs = [i for i in sinks if nodes[i].type_name == "Output"] or sinks
    if len(outputs) == 1:
        plan.output_id = outputs[0]
    else:
        fail(ShapeError(
            f"graph must have exactly one output (sink), found {len(sinks)}",
            hint="Connect every branch onward so that the graph ends in a single Output node.",
        ))

    for nid in order:
        node = nodes[nid]
        ps = preds[nid]
        try:
            if not node.needs_input:  # Input / source
                if ps:
                    raise ShapeError(f"source node {nid!r} ({node.type_name}) must have no inputs, has {len(ps)}")
                plan.out_shapes[nid] = node.out_shape(None)
                plan.node_in[nid] = None
            elif node.is_merge:  # Add / Concat
                if len(ps) < 2:
                    raise ShapeError(
                        f"{nid} ({node.type_name}) combines several branches, but only {len(ps)} "
                        f"{'is' if len(ps) == 1 else 'are'} connected",
                        hint="Connect at least two branches into it.",
                    )
                if any(p not in plan.out_shapes for p in ps):
                    continue  # an input is broken upstream (tolerant mode)
                resolved = resolve_merge(
                    [settings.get((p, nid), (AUTO, {})) for p in ps],
                    [plan.out_shapes[p] for p in ps],
                    node,
                    ps,
                )
                for p, r in zip(ps, resolved):
                    plan.edges[(p, nid)] = r
                plan.out_shapes[nid] = node.out_shape_merge([r.out_shape for r in resolved])
                plan.node_in[nid] = None
            else:  # ordinary single-input layer
                if not ps:
                    raise ShapeError(
                        f"{nid} ({node.type_name}) has no input",
                        hint="Connect a layer to it: click that layer's bottom port, then this layer's top port.",
                    )
                if len(ps) > 1:
                    raise ShapeError(
                        f"{nid} ({node.type_name}) takes one input, but {len(ps)} are connected",
                        hint="To combine branches, connect them into an Add or Concat node, and that node into this layer.",
                    )
                if ps[0] not in plan.out_shapes:
                    continue  # its input is broken upstream (tolerant mode)
                kind, params = settings.get((ps[0], nid), (AUTO, {}))
                r = resolve_single(kind, params, plan.out_shapes[ps[0]], node, (ps[0], nid))
                plan.edges[(ps[0], nid)] = r
                plan.node_in[nid] = r.out_shape
                plan.out_shapes[nid] = node.out_shape(r.out_shape)
        except ShapeError as exc:
            fail(_at(exc, nid))
        except ValueError as exc:  # e.g. an unknown transform name in a setting
            fail(ShapeError(str(exc), node_id=nid))
    return plan


#: Meta-device builds run one at a time (a second guard, see _MetaBuild).
META_LOCK = threading.RLock()


class _MetaBuild(TorchFunctionMode):
    """Build on the meta device without running a single meta kernel.

    Factories (torch.empty & co.) make meta tensors, as ``with torch.device("meta")``
    does; but weight initialization — torch.nn.init and any in-place op on a meta
    tensor — is skipped: without values it does nothing, and with a CPU-only torch
    (or CUDA hidden) it runs through torch's Python-implemented meta kernels, which
    crashed the server (a native access violation) when another request was using
    torch at the same moment. The mode is thread-local: other requests are untouched."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if getattr(func, "__module__", "") == "torch.nn.init":  # kaiming_uniform_, normal_, xavier_… : skip
            return args[0] if args else kwargs.get("tensor")
        name = getattr(func, "__name__", "")
        on_meta = bool(args) and isinstance(args[0], torch.Tensor) and args[0].is_meta
        if on_meta and name.endswith("_") and not name.endswith("__"):  # uniform_, fill_, zero_, copy_, … : skip
            return args[0]
        if on_meta and name == "__setitem__":  # filling a buffer slice (positional tables): nothing to fill
            return None
        if func in _device_constructors() and kwargs.get("device") is None:
            kwargs["device"] = torch.device("meta")
        return func(*args, **kwargs)


def build_model(spec: GraphSpec, seed: int | None = None, meta: bool = False) -> tuple[GraphModule, BuildResult]:
    """Assemble the graph into a model plus its per-layer summary.

    ``seed`` (Stage 16) makes the starting weights reproducible — the same seed,
    the same weights, so a training run repeats exactly — without touching the
    global random state (nor CUDA's). Without it the weights come from the global
    random state, as before.

    ``meta`` (Stage 17) builds on PyTorch's "meta" device: every module and shape
    is real, but no weight takes any memory — how a 671-billion-parameter model is
    sized on a laptop. Such a model can be inspected, not run."""
    if meta:
        with META_LOCK, _MetaBuild():
            return _build_model(spec)
    if seed is None:
        return _build_model(spec)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return _build_model(spec)


def _tie_heads(plan: GraphPlan, modules: dict[str, object], summaries: list[LayerSummary]) -> list[tuple[str, str]]:
    """Stage 17 — an LM head marked "tied" takes the graph's Embedding table as its
    weight (one shared parameter, counted once). Returns the (head, embedding) pairs."""
    tied: list[tuple[str, str]] = []
    heads = [nid for nid in plan.order if plan.nodes[nid].type_name == "LMHead"
             and bool(plan.nodes[nid].p("tie_embedding", False))]
    if not heads:
        return tied
    emb_id = next((nid for nid in plan.order if plan.nodes[nid].type_name == "Embedding"), None)
    for nid in heads:
        if emb_id is None:
            raise ShapeError(
                f"LM head {nid!r} is tied to the Embedding, but the graph has no Embedding",
                node_id=nid, hint="Add an Embedding at the start of the network, or switch off \"Tie to the Embedding\".",
            )
        table = modules[emb_id].emb.weight
        head = modules[nid]
        if tuple(table.shape) != tuple(head.weight.shape):
            raise ShapeError(
                f"LM head {nid!r} is tied to the Embedding {emb_id!r}, but their sizes differ: the embedding is "
                f"{table.shape[0]} x {table.shape[1]} (vocabulary x size) and the head needs "
                f"{head.weight.shape[0]} x {head.weight.shape[1]}",
                node_id=nid, hint="Give the head the embedding's vocabulary size, or untie it.",
            )
        head.weight = table  # the very same parameter: no copy, no extra values
        tied.append((nid, emb_id))
        for summary in summaries:
            if summary.id == nid:
                summary.num_params = 0  # shares the embedding's weights
    return tied


def _build_model(spec: GraphSpec) -> tuple[GraphModule, BuildResult]:
    plan = plan_graph(spec)

    modules: dict[str, object] = {}
    merges: dict[str, object] = {}
    summaries: list[LayerSummary] = []
    for nid in plan.order:  # layers first, in topological order (same weight init as before Stage 14)
        node = plan.nodes[nid]
        if node.is_merge:
            merges[nid] = node.merge_fn()
            mod = None
        else:
            try:
                mod = node.build(plan.node_in[nid])
            except ShapeError as exc:
                raise _at(exc, nid)
        if mod is not None:
            modules[nid] = mod
        nparams = sum(p.numel() for p in mod.parameters()) if mod is not None else 0
        summaries.append(
            LayerSummary(
                id=nid,
                type=node.type_name,
                in_shape=as_list(plan.node_in[nid]),
                out_shape=as_list(plan.out_shapes[nid]),
                num_params=int(nparams),
            )
        )

    tied = _tie_heads(plan, modules, summaries)

    edge_modules: dict[tuple[str, str], object] = {}
    edge_summaries: list[EdgeSummary] = []
    for src, dst in spec.edges:  # then the connections' own modules, in edge order
        r = plan.edges.get((src, dst))
        if r is None:
            continue
        mod = r.transform.build(r.in_shape)
        if mod is not None:
            edge_modules[(src, dst)] = mod
        nparams = sum(p.numel() for p in mod.parameters()) if mod is not None else 0
        edge_summaries.append(
            EdgeSummary(
                source=src,
                target=dst,
                transform=r.transform.kind,
                auto=r.auto,
                label=r.transform.badge(r.in_shape),
                params=r.transform.resolved_params(r.in_shape),
                in_shape=list(r.in_shape),
                out_shape=list(r.out_shape),
                num_params=int(nparams),
            )
        )

    model = GraphModule(
        order=plan.order,
        preds=plan.preds,
        modules_by_id=modules,
        merges=merges,
        input_id=plan.input_id,
        output_id=plan.output_id,
        edge_modules=edge_modules,
    )
    model.tied = tied  # Stage 17: (LM head, Embedding) pairs sharing one weight — the export repeats it
    total = sum(s.num_params for s in summaries) + sum(e.num_params for e in edge_summaries)
    return model, BuildResult(ok=True, order=plan.order, layers=summaries, total_params=total, edges=edge_summaries)


# --------------------------------------------------------------------------- #
# The connection inspector: every way one edge could reshape its data
# --------------------------------------------------------------------------- #
def connection_options(spec: GraphSpec, source: str, target: str):
    """What the editor's connection settings show for the edge ``source -> target``:
    the shape arriving, what "auto" picks right now, and every transform that
    applies — with its parameters, suggested presets and whether the target
    accepts the result (with a reason when it does not)."""
    from .connections import TRANSFORMS
    from .schema import ConnectionOption, ConnectionOptions

    if [source, target] not in [list(e) for e in spec.edges]:
        raise ValueError(f"there is no connection {source} → {target} in this graph")
    plan = plan_graph(spec, tolerant=True)
    settings = edge_settings_map(spec)
    kind, params = settings.get((source, target), (AUTO, {}))
    res = ConnectionOptions(source=source, target=target, current=kind, current_params=params)
    problems = [e for e in plan.errors if e.edge == (source, target) or (e.edge is None and e.node_id == target)]
    res.problem = str(problems[0]) if problems else None
    in_shape = plan.out_shapes.get(source)
    if in_shape is None:
        res.problem = res.problem or "the shape arriving here is unknown until the layers before it work"
        return res
    res.in_shape = list(in_shape)
    node = plan.nodes[target]
    preds = plan.preds[target]

    def other_inputs() -> list | None:
        shapes: list = []
        for p in preds:
            if p == source:
                shapes.append(None)
            elif (p, target) in plan.edges:
                shapes.append(plan.edges[(p, target)].out_shape)
            elif p in plan.out_shapes:
                shapes.append(plan.out_shapes[p])
            else:
                return None
        return shapes

    def fits(adapted) -> tuple[bool, str | None]:
        try:
            if node.is_merge:
                shapes = other_inputs()
                if shapes is None:
                    return True, None  # the other inputs are not known yet
                node.out_shape_merge([adapted if s is None else s for s in shapes])
            else:
                node.out_shape(adapted)
            return True, None
        except ShapeError as exc:
            return False, str(exc)

    # what "auto" resolves to (the rest of the graph as it is)
    ref: Shape | None = None
    try:
        if node.is_merge:
            shapes_in = [plan.out_shapes.get(p) for p in preds]
            if all(s is not None for s in shapes_in):
                chosen = [settings.get((p, target), (AUTO, {})) if p != source else (AUTO, {}) for p in preds]
                resolved = resolve_merge(chosen, shapes_in, node, preds)
                mine = resolved[preds.index(source)]
                res.auto_transform = mine.transform.kind
                res.auto_params = mine.transform.resolved_params(mine.in_shape)
                res.auto_out_shape = list(mine.out_shape)
                others = [r.out_shape for p, r in zip(preds, resolved) if p != source]
                ref = others[0] if others else None
        else:
            mine = resolve_single(AUTO, {}, in_shape, node, (source, target))
            res.auto_transform = mine.transform.kind
            res.auto_params = mine.transform.resolved_params(in_shape)
            res.auto_out_shape = list(mine.out_shape)
            ref = mine.out_shape
    except ShapeError as exc:
        res.problem = res.problem or str(exc)

    for cls in TRANSFORMS.values():
        if not cls.accepts(in_shape):
            continue
        presets = cls.presets(in_shape)
        if cls.kind in ("project", "reshape") and ref is not None:
            presets = [{"shape": list(ref)}] + [p for p in presets if p.get("shape") != list(ref)]
        best = None
        for preset in presets or [{}]:
            try:
                out = cls(preset).out_shape(in_shape)
                ok, note = fits(out)
            except ShapeError as exc:
                out, ok, note = None, False, str(exc)
            if best is None or (ok and not best[2]):
                best = (preset, list(out) if out is not None else None, ok, note)
            if ok:
                break
        preset, out, ok, note = best
        ordered = [preset] + [p for p in presets if p != preset]
        res.options.append(
            ConnectionOption(
                transform=cls.kind,
                label=cls.label,
                description=cls.description,
                learnable=cls.learnable,
                params=cls.param_defs(in_shape),
                presets=ordered,
                fits=ok,
                out_shape=out,
                note=note,
            )
        )
    return res
