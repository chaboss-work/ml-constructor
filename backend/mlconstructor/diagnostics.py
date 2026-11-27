"""Diagnostics (Stage 21): every problem of a graph, explained, with one-click fixes.

:func:`diagnose` is what the editor calls after each edit. It reports **every**
problem at once (the builder's tolerant plan, then the build itself, then whether the
chosen data would refuse the network), **warnings** for graphs that build but are
probably wrong (a dropout of 0.9, a tiny hidden layer, a branch that leads nowhere,
layers after the Output…), and for each a title in plain words, why, what to do and
ready **fixes** from :mod:`mlconstructor.fixes`.

A fix is only offered once it has been tried on a copy of the graph and removed the
problem; the ones that leave the graph with no problem at all come first. Fixes that
only the editor can carry out (open Data, the library…) cannot be tried here and come
last.

:func:`describe_error` gives the same explanation for an error raised by Build,
Train, Test, Export… (the 422 responses carry it).
"""

from __future__ import annotations

import json
from collections import deque
from itertools import permutations
from math import prod

from . import fixes as fx
from .builder import _UnknownNode, build_model, connection_options, plan_graph
from .error_codes import CATALOG, CATEGORIES, GLOSSARY, entry, terms_in
from .connections import AUTO, MAX_AUTO_PARAMS, projection_params, resolve_single
from .datasets import canonical_dataset_id, check_dataset, get_dataset
from .generative import resolve_task
from .registry import get_layer_cls
from .schema import CatalogEntryInfo, Diagnosis, ErrorCatalog, FixInfo, GraphSpec, Problem
from .shapes import GraphError, fmt

#: at most this many fixes per problem (the best first)
MAX_FIXES = 4
#: problems found only when the modules are made (the plan cannot see them)
_BUILD_CODES = {"tie_no_embedding", "tie_size"}
#: problems that concern the whole graph, not one node (a fix may move where they show)
_GLOBAL_CODES = {"no_input", "several_outputs", "cycle", "duplicate_id"}
#: what training on the chosen data would say
_DATA_CODES = {
    "dataset_shape", "dataset_classes", "dataset_needs_tokens", "tokens_too_short", "text_missing", "text_too_short",
    "custom_missing", "custom_too_many", "custom_row_length", "custom_value", "custom_label", "custom_label_range",
    "unknown_dataset", "every_position", "class_rows",
}
#: shape-preserving layers a class count passes through unchanged (walking back from the Output)
_PASS_THROUGH = {"ReLU", "LeakyReLU", "GELU", "SiLU", "Sigmoid", "Tanh", "Softmax", "Dropout", "LayerNorm", "RMSNorm",
                 "BatchNorm1d"}


# --------------------------------------------------------------------------- #
# The graph, as a fixer sees it
# --------------------------------------------------------------------------- #
class _Ctx:
    def __init__(self, spec: GraphSpec, plan=None):
        self.spec = spec
        self.plan = plan
        self.by_id = {n.id: n for n in spec.nodes}

    def type(self, nid: str) -> str:
        n = self.by_id.get(nid)
        return n.type if n is not None else ""

    def params(self, nid: str) -> dict:
        n = self.by_id.get(nid)
        return dict(n.params) if n is not None else {}

    def cls(self, nid: str):
        try:
            return get_layer_cls(self.type(nid))
        except KeyError:
            return None

    def is_source(self, nid: str) -> bool:
        cls = self.cls(nid)
        return cls is not None and not cls.needs_input

    def preds(self, nid: str) -> list[str]:
        return [s for s, d in self.spec.edges if d == nid]

    def succs(self, nid: str) -> list[str]:
        return [d for s, d in self.spec.edges if s == nid]

    def _walk(self, start: str, step) -> set[str]:
        seen: set[str] = set()
        queue = deque(step(start))
        while queue:
            n = queue.popleft()
            if n not in seen:
                seen.add(n)
                queue.extend(step(n))
        return seen

    def descendants(self, nid: str) -> set[str]:
        return self._walk(nid, self.succs)

    def ancestors(self, nid: str) -> set[str]:
        return self._walk(nid, self.preds)

    def new_id(self, type_name: str, taken: set[str] | None = None) -> str:
        return fx.fresh_id({n.id for n in self.spec.nodes} | set(taken or ()), type_name)

    def shape(self, nid: str):
        if self.plan is None:
            return None
        return self.plan.out_shapes.get(nid)

    def output_node(self) -> str | None:
        outs = [n.id for n in self.spec.nodes if n.type == "Output"]
        return outs[0] if len(outs) == 1 else None

    def inputs(self) -> list[str]:
        return [n.id for n in self.spec.nodes if self.is_source(n.id)]

    def defaults(self, nid: str) -> dict:
        cls = self.cls(nid)
        return {p.name: p.default for p in cls.param_defs()} if cls is not None else {}

    def width_layer(self) -> tuple[str, str] | None:
        """The layer that sets how many scores the network gives: walking back from the
        Output through shape-keeping layers to a Linear (``out_features``) or an LM head
        (``vocab``)."""
        node = self.output_node()
        seen: set[str] = set()
        while node is not None and node not in seen:
            seen.add(node)
            t = self.type(node)
            if t == "Linear":
                return node, "out_features"
            if t == "LMHead":
                return node, "vocab"
            ps = self.preds(node)
            if len(ps) != 1 or (t not in _PASS_THROUGH and t != "Output"):
                return None
            node = ps[0]
        return None

    def width_fixes(self, k: int, why: str) -> list[dict]:
        found = self.width_layer()
        if found is None:
            return []
        nid, param = found
        return [fx.set_param(nid, param, int(k), f"Give “{nid}” {k} outputs ({why})")]


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _key(exc: GraphError) -> tuple:
    if exc.code in _GLOBAL_CODES:
        return (exc.code,)
    return (exc.code, exc.node_id, tuple(exc.edge) if exc.edge else None, exc.inner)


# --------------------------------------------------------------------------- #
# Fixers: the fixes a problem's code calls for, made for the very graph
# --------------------------------------------------------------------------- #
def _input_shape_guess(c: _Ctx, root: str | None) -> list[int]:
    d = canonical_dataset_id(c.spec.train.dataset)
    if d == "mnist":
        return [1, 28, 28]
    if d == "cifar10":
        return [3, 32, 32]
    if d in ("tokens", "text"):
        return [16]
    cls = c.cls(root) if root else None
    category = getattr(cls, "category", "") if cls is not None else ""
    if c.type(root or "") == "Embedding":
        return [16]
    if category in ("conv", "pool") or c.type(root or "") in ("BatchNorm2d", "PatchEmbed"):
        return [1, 28, 28]
    if category in ("sequence", "attention", "llm"):
        return [16, 32]
    return [784]


def _fix_no_input(exc, c: _Ctx) -> list[dict]:
    roots = [n.id for n in c.spec.nodes if not c.is_source(n.id) and not c.preds(n.id) and n.type != "Output"]
    root = roots[0] if roots else None
    shape = _input_shape_guess(c, root)
    nid = c.new_id("Input")
    ops = [fx.op_add_node(nid, "Input", {"shape": shape}, near=root)]
    if root:
        ops.append(fx.op_add_edge(nid, root))
    return [fx.fix(f"Add an Input ({fmt(shape)})" + (f" before “{root}”" if root else ""), *ops)]


def _branch_of(c: _Ctx, end: str, ends: list[str]) -> list[str]:
    """The nodes whose data reaches ``end`` and no other end (removing them loses nothing else)."""
    mine = c.ancestors(end) | {end}
    others: set[str] = set()
    for e in ends:
        if e != end:
            others |= c.ancestors(e) | {e}
    return [n.id for n in c.spec.nodes if n.id in mine and n.id not in others]


def _fix_several_outputs(exc, c: _Ctx) -> list[dict]:
    ends = list(exc.facts.get("ends") or [])
    if len(ends) < 2:
        return []
    outputs = [e for e in ends if c.type(e) == "Output"]
    keep = outputs[0] if outputs else None
    cid = c.new_id("Concat")
    ops = [fx.op_add_node(cid, "Concat", {"dim": 0}, near=ends[0])]
    for e in ends:
        if c.type(e) == "Output":
            for p in c.preds(e):
                ops += [fx.op_remove_edge(p, e), fx.op_add_edge(p, cid)]
        else:
            ops.append(fx.op_add_edge(e, cid))
    ops += [fx.op_remove_node(e) for e in outputs[1:]]
    if keep:
        ops.append(fx.op_add_edge(cid, keep))
    else:
        oid = c.new_id("Output", {cid})
        ops += [fx.op_add_node(oid, "Output", {}, near=cid), fx.op_add_edge(cid, oid)]
    join = fx.fix("Join the ends with a Concat into one Output", *ops)
    removals = []
    for e in reversed(ends):
        branch = _branch_of(c, e, ends)
        if branch:
            removals.append(fx.fix(f"Remove the branch ending at “{e}” ({_plural(len(branch), 'node')})",
                                   *[fx.op_remove_node(b) for b in branch]))
    # two Outputs: one of them is most likely a leftover; loose ends without an Output: join them
    return removals[:2] + [join] if len(outputs) >= 2 else [join] + removals[:2]


def _fix_no_connection(exc, c: _Ctx) -> list[dict]:
    x = exc.node_id
    if not x or x not in c.by_id:
        return []
    below = c.descendants(x) | {x}
    ends = [n.id for n in c.spec.nodes if n.id not in below and not c.succs(n.id) and n.type != "Output"
            and not c.is_source(n.id)]
    inputs = [i for i in c.inputs() if i not in below]
    candidates = list(reversed(ends)) + [i for i in inputs if i not in ends]
    out = [fx.fix(f"Connect “{a}” → “{x}”", fx.op_add_edge(a, x)) for a in candidates[:3]]
    if not c.succs(x) and c.type(x) != "Output":  # a lone layer, connected to nothing: most likely left over
        out.insert(0, fx.fix(f"Delete “{x}”", fx.op_remove_node(x)))
    return out


def _fix_too_many_inputs(exc, c: _Ctx) -> list[dict]:
    x = exc.node_id
    ps = c.preds(x)
    if len(ps) < 2:
        return []
    out = []
    for merge, how in (("Add", "sum them"), ("Concat", "put them side by side")):
        mid = c.new_id(merge)
        ops = [fx.op_add_node(mid, merge, {}, near=x)]
        for p in ps:
            ops += [fx.op_remove_edge(p, x), fx.op_add_edge(p, mid)]
        ops.append(fx.op_add_edge(mid, x))
        out.append(fx.fix(f"Join them with {merge} ({how}), then into “{x}”", *ops))
    out.append(fx.fix(f"Keep only “{ps[0]}” → “{x}”", *[fx.op_remove_edge(p, x) for p in ps[1:]]))
    return out


def _fix_merge_needs_two(exc, c: _Ctx) -> list[dict]:
    m = exc.node_id
    ps = c.preds(m)
    if not m or m not in c.by_id:
        return []
    if not ps:
        return [fx.fix(f"Delete “{m}”", fx.op_remove_node(m))]
    p = ps[0]
    below = c.descendants(m) | {m}
    out = []
    back = [q for q in c.preds(p) if q not in below]
    if back:  # one step back: the classic skip connection around "p"
        out.append(fx.fix(f"Connect “{back[0]}” into “{m}” too (a skip connection around “{p}”)",
                          fx.op_add_edge(back[0], m)))
    out.append(fx.fix(f"Remove “{m}” and connect “{p}” straight on", fx.op_remove_node(m, bridge=True)))
    return out


def _port_shape(c: _Ctx, x: str, index: int) -> list[int] | None:
    """A shape the ``index``-th input of ``x`` takes: tried on the layer itself."""
    if c.plan is None or x not in c.plan.nodes:
        return None
    node = c.plan.nodes[x]
    first = c.plan.out_shapes.get(c.preds(x)[0]) if c.preds(x) else None
    tries = [(1,)] + [c.plan.out_shapes[i] for i in c.inputs() if i in c.plan.out_shapes]
    if first is not None:
        tries.append(tuple(first))
    tries += [(77, 768), (16, 64), (1, 64), (64,)]
    for shape in tries:
        try:
            node.check_port(index, tuple(shape))
            return [int(d) for d in shape]
        except (GraphError, IndexError, ValueError, TypeError):
            continue
    return None


def _fix_ports(exc, c: _Ctx) -> list[dict]:
    x = exc.node_id
    if not x or x not in c.by_id:
        return []
    ports = list(exc.facts.get("ports") or [])
    ps = c.preds(x)
    if not ports:
        return []
    if len(ps) > len(ports):
        extra = ps[len(ports):]
        return [fx.fix(f"Remove the extra connection{'s' if len(extra) > 1 else ''} from "
                       + ", ".join(f"“{e}”" for e in extra), *[fx.op_remove_edge(e, x) for e in extra])]
    below = c.descendants(x) | {x}
    loose = [i for i in c.inputs() if not c.succs(i) and i not in below]
    ops: list[dict] = []
    names: list[str] = []
    taken: set[str] = set()
    for k, port in enumerate(ports[len(ps):]):
        if loose:
            src = loose.pop(0)
            ops.append(fx.op_add_edge(src, x))
            names.append(f"“{src}” as “{port}”")
            continue
        shape = _port_shape(c, x, len(ps) + k)
        if shape is None:
            return []
        nid = c.new_id("Input", taken)
        taken.add(nid)
        ops += [fx.op_add_node(nid, "Input", {"shape": shape, "name": port}, near=x), fx.op_add_edge(nid, x)]
        names.append(f"a new Input for “{port}” ({fmt(shape)})")
    return [fx.fix("Connect " + ", ".join(names), *ops)] if ops else []


def _preset_text(params: dict) -> str:
    if not params:
        return ""
    if "shape" in params:
        return fmt(params["shape"])
    if "channels" in params:
        return f"{params['channels']}×{params['height']}×{params['width']}"
    return ", ".join(f"{k} {v}" for k, v in params.items())


def _edge_options(c: _Ctx, s: str, t: str, limit: int = 3) -> list[dict]:
    """The connection inspector's transforms that make ``s -> t`` fit."""
    try:
        opts = connection_options(c.spec, s, t)
    except Exception:  # noqa: BLE001 - a fixer never breaks the diagnosis
        return []
    out = []
    for o in opts.options:
        if not o.fits or o.transform == opts.current:
            continue
        params = dict(o.presets[0]) if o.presets else {}
        if o.transform == "project" and opts.in_shape and params.get("shape") and projection_params(
                tuple(opts.in_shape), tuple(params["shape"])) > MAX_AUTO_PARAMS:
            continue  # a learnable projection the size of a model is no fix
        extra = _preset_text(params)
        out.append(fx.fix(f"Set the connection {s} → {t} to “{o.label}”" + (f" ({extra})" if extra else ""),
                          fx.op_set_transform(s, t, o.transform, params)))
    return out[:limit]


def _fix_port_order(c: _Ctx, x: str) -> list[dict]:
    """A several-input layer whose inputs fit in another order: reconnect them so."""
    if c.plan is None or x not in c.plan.nodes:
        return []
    node = c.plan.nodes[x]
    ports = node.input_ports()
    ps = c.preds(x)
    if not ports or len(ps) != len(ports) or len(ps) > 4 or any(p not in c.plan.out_shapes for p in ps):
        return []
    from .builder import PortTarget

    for order in permutations(ps):
        if list(order) == ps:
            continue
        try:
            shapes = [resolve_single(AUTO, {}, c.plan.out_shapes[p], PortTarget(node, i), (p, x)).out_shape
                      for i, p in enumerate(order)]
            node.out_shape_ports(shapes)
        except (GraphError, ValueError, TypeError, IndexError):
            continue
        return [fx.fix("Reconnect its inputs in this order: " + " → ".join(f"{p} as “{port}”" for p, port in
                                                                          zip(order, ports)),
                       fx.op_reorder_inputs(x, list(order)))]
    return []


def _fix_shape(exc, c: _Ctx) -> list[dict]:
    """wrong_rank / wrong_shape / pinned_transform / no_auto_transform / transform_param."""
    out: list[dict] = []
    x = exc.node_id
    if x and exc.inner is None:
        out += _fix_port_order(c, x)
    single = (x is not None and exc.inner is None and c.plan is not None and x in c.plan.nodes
              and len(c.preds(x)) == 1 and not c.plan.nodes[x].input_ports())
    if exc.edge:
        out += _edge_options(c, exc.edge[0], exc.edge[1])
    elif single:
        out += _edge_options(c, c.preds(x)[0], x)
    if exc.inner is not None and exc.code == "wrong_rank":
        i = exc.inner
        out.append(fx.fix(f"Insert a Flatten inside the block, before layer {i + 1}", fx.op_add_inner(x, "Flatten", {}, i)))
    return out


def _fix_merge_shapes(exc, c: _Ctx) -> list[dict]:
    m = exc.node_id
    ps = c.preds(m)
    shapes = {p: c.shape(p) for p in ps}
    out: list[dict] = []
    if c.type(m) == "Add" and len(ps) >= 2 and all(shapes.values()):
        ref = shapes[ps[0]]
        for p in ps[1:]:
            if tuple(shapes[p]) == tuple(ref):
                continue
            if prod(shapes[p]) == prod(ref):
                out.append(fx.fix(f"Reshape “{p}” to {fmt(ref)} on its way into “{m}”",
                                  fx.op_set_transform(p, m, "reshape", {"shape": list(ref)})))
            elif projection_params(tuple(shapes[p]), tuple(ref)) <= MAX_AUTO_PARAMS:
                out.append(fx.fix(f"Project “{p}” to {fmt(ref)} on its way into “{m}” (learns a Linear)",
                                  fx.op_set_transform(p, m, "project", {"shape": list(ref)})))
        # sums need equal shapes; side by side, anything goes
        cid = c.new_id("Concat")
        ops = [fx.op_add_node(cid, "Concat", {}, near=m)] + [fx.op_add_edge(p, cid) for p in ps]
        ops += [fx.op_add_edge(cid, s) for s in c.succs(m)] + [fx.op_remove_node(m)]
        out.append(fx.fix(f"Use a Concat instead of “{m}” (put the branches side by side)", *ops))
    for p in ps:
        out += [f for f in _edge_options(c, p, m, limit=2) if not _too_big_projection(f, shapes.get(p))][:1]
    return out


def _too_big_projection(f: dict, shape) -> bool:
    """A learnable projection the size of a model is no fix."""
    for o in f["ops"]:
        if o.get("op") == "set_transform" and o.get("transform") == "project" and shape:
            target = tuple((o.get("params") or {}).get("shape") or ())
            if target and projection_params(tuple(shape), target) > MAX_AUTO_PARAMS:
                return True
    return False


def _fix_too_small(exc, c: _Ctx) -> list[dict]:
    x = exc.node_id
    if not x or exc.inner is not None or len(c.preds(x)) != 1 or not c.succs(x):
        return []
    return [fx.fix(f"Delete “{x}” (connect straight through)", fx.op_remove_node(x, bridge=True))]


def _fix_output_not_classes(exc, c: _Ctx) -> list[dict]:
    out_id = c.output_node()
    ps = c.preds(out_id) if out_id else []
    if not ps:
        return []
    k = get_dataset(c.spec.train.dataset).fixed_classes or 10
    fid = c.new_id("Flatten")
    lid = c.new_id("Linear", {fid})
    return [fx.fix(f"End with a Flatten and a Linear of {k} outputs",
                   fx.op_insert_node(ps[0], out_id, fid, "Flatten"),
                   fx.op_insert_node(fid, out_id, lid, "Linear", {"out_features": k}))]


def _fix_reset(exc, c: _Ctx) -> list[dict]:
    """The last resort for a setting gone wrong: the layer's default settings."""
    x = exc.node_id
    if not x or exc.inner is not None or x not in c.by_id:
        return []
    current = c.params(x)
    ops = [fx.op_set_param(x, k, v) for k, v in c.defaults(x).items() if current.get(k) != v]
    return [fx.fix(f"Reset “{x}” to its default settings", *ops)] if ops else []


def _fix_block_empty(exc, c: _Ctx) -> list[dict]:
    b = exc.node_id
    ps = c.preds(b)
    shape = c.shape(ps[0]) if ps else None
    rank = len(shape) if shape else 1
    if rank == 3:
        layers = [("Conv2d", {"out_channels": int(shape[0]), "kernel_size": 3, "padding": 1}), ("ReLU", {})]
    elif rank == 2:
        layers = [("LayerNorm", {}), ("GELU", {})]
    else:
        layers = [("Linear", {"out_features": int(shape[0]) if shape else 64}), ("ReLU", {})]
    out = [fx.fix(f"Put {' + '.join(t for t, _ in layers)} inside the block",
                  *[fx.op_add_inner(b, t, p) for t, p in layers])]
    if len(ps) == 1:
        out.append(fx.fix(f"Delete the block “{b}” (connect straight through)", fx.op_remove_node(b, bridge=True)))
    return out


def _single_input(c: _Ctx) -> str | None:
    ins = c.inputs()
    return ins[0] if len(ins) == 1 else None


def _fix_dataset_shape(exc, c: _Ctx) -> list[dict]:
    want, inp = exc.facts.get("want"), _single_input(c)
    out = []
    if want and inp:
        out.append(fx.set_param(inp, "shape", list(want),
                                f"Set the Input to {fmt(want)}, as {exc.facts.get('dataset', 'the data')} needs"))
    out.append(fx.set_train("dataset", "blobs", "Use Gaussian blobs instead (they fit any shape)"))
    return out


def _fix_dataset_classes(exc, c: _Ctx) -> list[dict]:
    k = int(exc.facts.get("classes") or 10)
    return c.width_fixes(k, "one per class")


def _fix_needs_tokens(exc, c: _Ctx) -> list[dict]:
    return [fx.ui("open_library", "Take a language model from the library"),
            fx.set_train("dataset", "blobs", "Use Gaussian blobs instead (they fit this network)")]


def _fix_tokens_short(exc, c: _Ctx) -> list[dict]:
    inp = _single_input(c)
    return [fx.set_param(inp, "shape", [16], "Make the Input 16 tokens long")] if inp else []


def _fix_text_short(exc, c: _Ctx) -> list[dict]:
    inp = _single_input(c)
    chars = int(exc.facts.get("chars") or 0)
    length = max(3, chars // 8 - 1)
    if inp and length < int(exc.facts.get("length") or 0):
        return [fx.set_param(inp, "shape", [length], f"Make the Input {length} characters long")]
    return []


def _fix_row_length(exc, c: _Ctx) -> list[dict]:
    rf, inp = exc.facts.get("row_features"), _single_input(c)
    if rf and inp:
        return [fx.set_param(inp, "shape", [int(rf)], f"Set the Input to {rf} features, as your rows have")]
    return []


def _fix_label_range(exc, c: _Ctx) -> list[dict]:
    need = int(exc.facts.get("needed") or 0)
    return c.width_fixes(need, "one per class in your rows") if need else []


def _fix_unknown_dataset(exc, c: _Ctx) -> list[dict]:
    return [fx.set_train("dataset", "blobs", "Use Gaussian blobs")]


def _fix_every_position(exc, c: _Ctx) -> list[dict]:
    out_id = c.output_node()
    out = [fx.set_param(out_id, "per_token", False, "Switch the Output's “Every position” off")] if out_id else []
    return out + [fx.set_train("dataset", "text", "Train on Your text instead")]


def _fix_class_rows(exc, c: _Ctx) -> list[dict]:
    k = int(exc.facts.get("classes") or 0)
    inp = exc.facts.get("input")
    emb = next((d for s, d in c.spec.edges if s == inp and c.type(d) == "Embedding"), None)
    if not emb or not k:
        return []
    return [fx.set_param(emb, "num_embeddings", k + 1, f"Give “{emb}” {k + 1} rows (one per class, plus “no class”)")]


def _fix_task(exc, c: _Ctx) -> list[dict]:
    out_id = c.output_node()
    if not out_id:
        return []
    if exc.code == "task_needs_any_shape":
        return [fx.set_param(out_id, "any_shape", True, "Switch the Output's “Any shape” on")]
    return [fx.set_param(out_id, "any_shape", False, "Switch the Output's “Any shape” off (classify)")]


def _fix_few_rows(exc, c: _Ctx) -> list[dict]:
    n = min(100_000, max(int(c.spec.train.samples) * 2, 1024))
    return [fx.set_train("samples", n, f"Use {n:,} rows")]


def _fix_weights_mismatch(exc, c: _Ctx) -> list[dict]:
    ops = []
    for name, in_file, here in exc.facts.get("sizes") or []:
        node, _, what = str(name).rpartition(".")
        t = c.type(node)
        if what != "weight" or len(in_file) != len(here):
            continue
        if t == "Linear" and len(in_file) == 2 and in_file[1] == here[1]:
            ops.append(fx.op_set_param(node, "out_features", int(in_file[0])))
        elif t == "Conv2d" and len(in_file) == 4 and in_file[1:] == here[1:]:
            ops.append(fx.op_set_param(node, "out_channels", int(in_file[0])))
    return [fx.fix("Set the layers back to the sizes in the file", *ops)] if ops else []


_FIXERS = {
    "no_input": _fix_no_input,
    "several_outputs": _fix_several_outputs,
    "no_connection": _fix_no_connection,
    "too_many_inputs": _fix_too_many_inputs,
    "merge_needs_two": _fix_merge_needs_two,
    "ports_count": _fix_ports,
    "block_needs_input": _fix_ports,
    "wrong_rank": _fix_shape,
    "wrong_shape": _fix_shape,
    "pinned_transform": _fix_shape,
    "no_auto_transform": _fix_shape,
    "transform_param": _fix_shape,
    "merge_shapes": _fix_merge_shapes,
    "too_small": _fix_too_small,
    "output_not_classes": _fix_output_not_classes,
    "not_divisible": _fix_reset,
    "param_range": _fix_reset,
    "param_choice": _fix_reset,
    "param_list": _fix_reset,
    "settings_conflict": _fix_reset,
    "block_empty": _fix_block_empty,
    "dataset_shape": _fix_dataset_shape,
    "dataset_classes": _fix_dataset_classes,
    "dataset_needs_tokens": _fix_needs_tokens,
    "tokens_too_short": _fix_tokens_short,
    "text_too_short": _fix_text_short,
    "custom_row_length": _fix_row_length,
    "custom_label_range": _fix_label_range,
    "unknown_dataset": _fix_unknown_dataset,
    "every_position": _fix_every_position,
    "class_rows": _fix_class_rows,
    "task_output_shape": _fix_task,
    "task_needs_any_shape": _fix_task,
    "task_any_shape_on": _fix_task,
    "few_rows": _fix_few_rows,
    "weights_mismatch": _fix_weights_mismatch,
}


# --------------------------------------------------------------------------- #
# Checking a fix on a copy of the graph
# --------------------------------------------------------------------------- #
def _keys(spec: GraphSpec, deep: bool, data: bool) -> set[tuple]:
    """Every problem of ``spec``, as keys: the plan's, then (``deep``) the build's,
    then (``data``) what training on its data would say."""
    try:
        plan = plan_graph(spec, tolerant=True)
    except GraphError as exc:
        return {_key(exc)}
    keys = {_key(e) for e in plan.errors}
    if keys or not (deep or data):
        return keys
    try:
        _model, build = build_model(spec, meta=True)
    except GraphError as exc:
        return {_key(exc)}
    if data:
        keys |= {_key(e) for e in _data_errors(spec, build, _model.task_plan)}
    return keys


def _checked(problem: GraphError, fixes: list[dict], spec: GraphSpec | None, warning: bool = False) -> list[dict]:
    """The fixes that remove ``problem`` from a copy of the graph — those leaving no
    problem at all first; the editor-only ones (they cannot be tried here) last."""
    unique: list[dict] = []
    seen: set[str] = set()
    for f in fixes:
        sig = json.dumps(f.get("ops"), sort_keys=True, default=str)
        if f.get("ops") and sig not in seen:
            seen.add(sig)
            unique.append(f)
    if spec is None:
        return unique[:MAX_FIXES]
    deep = problem.code in _BUILD_CODES
    data = problem.code in _DATA_CODES
    key = _key(problem)
    ranked: list[tuple[int, int, dict]] = []
    for i, f in enumerate(unique):
        if any(o.get("op") == "ui" for o in f["ops"]):
            ranked.append((3, i, f))
            continue
        try:
            after = fx.apply_fix(spec, f)
        except (GraphError, ValueError, KeyError, IndexError, TypeError):
            continue  # it does not even apply to this graph
        try:
            if warning:
                left = _keys(after, deep=False, data=False)
                if left or key in {_key(w) for w in _lint(after)}:
                    continue
                ranked.append((0, i, f))
                continue
            left = _keys(after, deep=deep, data=data)
        except Exception:  # noqa: BLE001 - a fix that breaks the check is no fix
            continue
        if key in left:
            continue
        ranked.append((0 if not left else 1, i, f))
    ranked.sort(key=lambda r: (r[0], r[1]))
    return [f for _, _, f in ranked][:MAX_FIXES]


# --------------------------------------------------------------------------- #
# Problems in the editor's words
# --------------------------------------------------------------------------- #
def _terms(*texts: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for t in texts:
        for w in terms_in(t):
            if w in GLOSSARY:
                out[w] = GLOSSARY[w]
    return out


def _problem(exc: Exception, spec: GraphSpec | None = None, plan=None, severity: str | None = None,
             check: bool = True) -> Problem:
    code = getattr(exc, "code", None) if isinstance(exc, GraphError) or hasattr(exc, "code") else None
    e = entry(code)
    fixes: list[dict] = list(getattr(exc, "fixes", None) or [])
    if isinstance(exc, GraphError) and spec is not None:
        c = _Ctx(spec, plan)
        fixer = _FIXERS.get(code or "")
        if fixer is not None:
            try:
                fixes += fixer(exc, c)
            except Exception:  # noqa: BLE001 - a fixer never breaks the diagnosis
                pass
        if not fixes and code in ("not_divisible", "param_range", "param_choice", "param_list", "settings_conflict"):
            fixes += _fix_reset(exc, c)
    fixes += [dict(f) for f in e.fixes]
    sev = severity or e.severity
    if check and isinstance(exc, GraphError):
        fixes = _checked(exc, fixes, spec, warning=sev == "warning")
    else:
        fixes = fixes[:MAX_FIXES]
    hint = getattr(exc, "hint", None)
    todo = hint or e.todo
    return Problem(
        code=e.code,
        severity=sev,
        category=e.category,
        title=e.title,
        message=str(exc) if not isinstance(exc, KeyError) else str(exc.args[0] if exc.args else exc),
        why=e.why,
        todo=todo,
        terms=_terms(e.why, todo),
        node_id=getattr(exc, "node_id", None),
        edge=list(exc.edge) if getattr(exc, "edge", None) else None,
        inner=getattr(exc, "inner", None),
        param=getattr(exc, "param", None),
        fixes=[FixInfo(label=f["label"], ops=f["ops"]) for f in fixes],
    )


def describe_error(exc: Exception, spec: GraphSpec | None = None) -> dict:
    """The ``error`` of a 422 response: the problem in plain words with its fixes
    (checked on ``spec`` when given), plus the Stage 14 keys (``message``,
    ``node_id``, ``edge``, ``hint``) as before."""
    plan = None
    if spec is not None:
        try:
            plan = plan_graph(spec, tolerant=True)
        except GraphError:
            plan = None
    out = _problem(exc, spec, plan).model_dump()
    out["hint"] = getattr(exc, "hint", None)
    return out


# --------------------------------------------------------------------------- #
# Would the data refuse this network?
# --------------------------------------------------------------------------- #
def _data_errors(spec: GraphSpec, build, task_plan) -> list[GraphError]:
    """What training on the chosen data would refuse — found without making a row."""
    t = spec.train
    try:
        d = get_dataset(t.dataset)
    except GraphError as exc:
        return [exc]
    task = task_plan.task if task_plan is not None else None
    if task is None or not build.trainable:
        return []
    errs: list[GraphError] = []
    if task == "classification":
        if len(build.inputs) != 1:
            return []
        inp = build.inputs[0]
        out_node = next((n for n in spec.nodes if n.type == "Output"), None)
        out_shape = next((s.out_shape for s in build.layers if out_node is not None and s.id == out_node.id), None)
        if not out_shape:
            return []
        per_token = bool(out_node.params.get("per_token", False)) if out_node is not None else False
        classes = int(out_shape[-1])
        try:
            check_dataset(t.dataset, tuple(inp.shape), classes, t)
        except GraphError as exc:
            if exc.code in ("dataset_shape", "tokens_too_short", "text_too_short", "custom_row_length"):
                exc.node_id = exc.node_id or inp.id
            elif exc.code in ("dataset_classes", "custom_label_range"):
                found = _Ctx(spec).width_layer()
                exc.node_id = exc.node_id or (found[0] if found else out_node.id if out_node else None)
            elif exc.code == "dataset_needs_tokens":
                exc.node_id = exc.node_id or inp.id
            errs.append(exc)
        if per_token and d.id != "text" and not errs:
            errs.append(GraphError(
                "the network scores every position, but this dataset gives one label per row: switch the Output's "
                "\"Every position\" off, or train on Your text", code="every_position",
                node_id=out_node.id if out_node is not None else None))
        return errs
    # generative: the pictures come from the dataset, with its own classes
    shape = task_plan.in_shapes.get(task_plan.image_input) if task_plan.image_input else None
    if shape is None:
        return []
    try:
        check_dataset(t.dataset, tuple(shape), int(t.num_classes), t)
    except GraphError as exc:
        if exc.code in ("dataset_shape", "custom_row_length"):
            exc.node_id = exc.node_id or task_plan.image_input
        errs.append(exc)
    if task == "diffusion" and task_plan.class_input and d.fixed_classes and task_plan.class_rows < d.fixed_classes:
        k = d.fixed_classes
        emb = next((dd for s, dd in spec.edges if s == task_plan.class_input
                    and next((n.type for n in spec.nodes if n.id == dd), "") == "Embedding"), None)
        errs.append(GraphError(
            f"the class embedding has {task_plan.class_rows} rows, but the data has {k} classes: give it at least {k} "
            f"(one more, {k + 1}, lets it also draw without a class, which makes the pictures sharper)",
            code="class_rows", node_id=emb, facts={"classes": k, "input": task_plan.class_input}))
    return errs


# --------------------------------------------------------------------------- #
# Warnings: graphs that build, but are probably not what was meant
# --------------------------------------------------------------------------- #
def _lint(spec: GraphSpec, plan=None, build=None, task: str | None = None) -> list[GraphError]:
    c = _Ctx(spec, plan)
    out: list[GraphError] = []

    def warn(code, message, node=None, fixes=(), param=None, edge=None):
        out.append(GraphError(message, code=code, node_id=node, edge=edge, param=param, fixes=list(fixes)))

    # a very high dropout
    for n in spec.nodes:
        for param in ("p", "dropout"):
            if param == "p" and n.type != "Dropout":
                continue
            v = n.params.get(param)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.7 <= float(v) < 1.0 + 1e-9:
                warn("dropout_high", f"“{n.id}” drops {float(v):.0%} of its numbers at every training step", n.id,
                     [fx.set_param(n.id, param, 0.5 if param == "p" else 0.1),
                      fx.set_param(n.id, param, 0.2 if param == "p" else 0.0)], param)

    out_id = c.output_node()
    classes = None
    if build is not None and out_id and task == "classification":
        shape = next((s.out_shape for s in build.layers if s.id == out_id), None)
        classes = int(shape[-1]) if shape else None

    # a narrow hidden Linear, smaller than the answer it must give
    if classes:
        last = c.width_layer()
        for n in spec.nodes:
            if n.type != "Linear" or (last and n.id == last[0]):
                continue
            width = n.params.get("out_features")
            if isinstance(width, int) and width < min(classes, 8) and any(
                    c.type(d) in ("Linear", "LMHead") for d in c.descendants(n.id)):
                warn("tiny_hidden", f"“{n.id}” squeezes everything into {width} number{'s' if width != 1 else ''}, "
                     f"fewer than the {classes} classes the network must tell apart", n.id,
                     [fx.set_param(n.id, "out_features", 64), fx.set_param(n.id, "out_features", max(16, classes * 2))],
                     "out_features")

    # layers after the Output
    outs = [n.id for n in spec.nodes if n.type == "Output"]
    for o in outs:
        after = c.descendants(o)
        if not after:
            continue
        ps = c.preds(o)
        fixes = []
        if len(ps) == 1:
            ends = [d for d in after if not c.succs(d)]
            if len(ends) == 1:
                ops = []
                for f in c.succs(o):
                    ops += [fx.op_remove_edge(o, f), fx.op_add_edge(ps[0], f)]
                ops += [fx.op_remove_edge(ps[0], o), fx.op_add_edge(ends[0], o)]
                fixes.append(fx.fix(f"Move the Output to the end (after “{ends[0]}”)", *ops))
        fixes.append(fx.fix(f"Remove the {_plural(len(after), 'node')} after the Output",
                            *[fx.op_remove_node(d) for d in after]))
        warn("after_output", f"the Output “{o}” is followed by {_plural(len(after), 'more node')}: "
             + ", ".join(f"“{d}”" for d in sorted(after)), o, fixes)

    # branches that never reach the end (while the graph has errors, a cut-off branch is their consequence)
    if plan is not None and plan.output_id and not plan.errors:
        feeds = c.ancestors(plan.output_id) | {plan.output_id}
        loose = [n.id for n in spec.nodes if n.id not in feeds and not (outs and n.id in c.descendants(outs[0]))]
        if loose and len(loose) < len(spec.nodes):
            warn("unused_branch", f"{_plural(len(loose), 'node')} never reach the Output: "
                 + ", ".join(f"“{n}”" for n in loose), loose[0],
                 [fx.fix(f"Remove the unused {'node' if len(loose) == 1 else 'branch'} ("
                         + ", ".join(loose) + ")", *[fx.op_remove_node(n) for n in loose])])

    # Softmax right before the Output of a classifier
    if out_id and task == "classification":
        for p in c.preds(out_id):
            if c.type(p) == "Softmax" and len(c.preds(p)) == 1:
                warn("softmax_before_loss", f"“{p}” turns the class scores into probabilities, and training does it "
                     "again inside its loss", p, [fx.fix(f"Remove “{p}”", fx.op_remove_node(p, bridge=True))])

    # two Linear layers in a row
    for s, d in spec.edges:
        if c.type(s) == "Linear" and c.type(d) == "Linear" and c.succs(s) == [d] and c.preds(d) == [s]:
            rid = c.new_id("ReLU")
            warn("linear_linear", f"“{s}” feeds “{d}” directly: two Linear layers in a row act as one", d,
                 [fx.fix(f"Insert a ReLU between “{s}” and “{d}”", fx.op_insert_node(s, d, rid, "ReLU"))], edge=(s, d))

    # no Output node
    if not outs and plan is not None and plan.output_id and not plan.errors:
        oid = c.new_id("Output")
        shape = c.shape(plan.output_id)
        params = {} if shape is not None and len(shape) == 1 else {"any_shape": True}
        warn("no_output_node", f"the network ends at “{plan.output_id}”, with no Output node", plan.output_id,
             [fx.fix(f"Add an Output after “{plan.output_id}”", fx.op_add_node(oid, "Output", params, near=plan.output_id),
                     fx.op_add_edge(plan.output_id, oid))])

    # training settings
    t = spec.train
    if t.lr > 0.1 or t.lr < 1e-5:
        good = 0.001 if t.optimizer == "adam" else 0.01
        warn("lr_extreme", f"the learning rate is {t.lr:g}", None,
             [fx.set_train("lr", good, f"Set the learning rate to {good:g}")])
    train_rows = int(t.samples * (1 - float(t.val_fraction) - float(t.test_fraction)))
    if task is not None and t.batch_size > train_rows > 0:
        good = max(1, min(64, train_rows))
        warn("batch_too_big", f"the batch of {t.batch_size} is bigger than the {train_rows:,} training rows", None,
             [fx.set_train("batch_size", good, f"Use batches of {good}")])
    return out


# --------------------------------------------------------------------------- #
# The editor's live check
# --------------------------------------------------------------------------- #
def diagnose(spec: GraphSpec) -> Diagnosis:
    """Every problem and warning of ``spec``, explained, with checked fixes — plus the
    build (shapes, parameter counts) when it builds, as ``POST /build`` gives it."""
    from .trainer import MAX_TRAIN_PARAMS

    try:
        plan = plan_graph(spec, tolerant=True)
    except GraphError as exc:  # the wiring itself: two nodes alike, a dangling connection, a loop
        return Diagnosis(ok=False, problems=[_problem(exc, spec, None)])
    problems = [_problem(exc, spec, plan) for exc in plan.errors]
    shapes = {nid: [int(d) for d in sh] for nid, sh in plan.out_shapes.items() if sh is not None}
    build = None
    task_plan = None
    train_issue = None
    if not plan.errors:
        try:
            model, build = build_model(spec, meta=True, plan=plan)
            task_plan = model.task_plan
            build.task = task_plan.task
            build.runnable = build.total_params <= MAX_TRAIN_PARAMS
            build.train_note = task_plan.reason if build.runnable else None
            build.trainable = build.runnable and build.train_note is None
            build.max_train_params = MAX_TRAIN_PARAMS
        except GraphError as exc:
            problems.append(_problem(exc, spec, plan))
            build = None
    if build is not None:
        problems += [_problem(e, spec, plan) for e in _data_errors(spec, build, task_plan)]
        if build.train_note and task_plan is not None and task_plan.code:
            train_issue = _problem(GraphError(build.train_note, code=task_plan.code), spec, plan)
    warnings = []
    if not any(p.code in _GLOBAL_CODES for p in problems):
        task = task_plan.task if task_plan is not None else None
        warnings = [_problem(w, spec, plan, severity="warning") for w in _lint(spec, plan, build, task)]
    return Diagnosis(ok=not problems, problems=problems, warnings=warnings, build=build, train_issue=train_issue,
                     shapes=shapes)


def lint(spec: GraphSpec) -> list[Problem]:
    """The warnings alone (tests and tools)."""
    return diagnose(spec).warnings


def error_catalog() -> ErrorCatalog:
    """``GET /errors``: every code with its words and its always-applicable fixes."""
    entries = [
        CatalogEntryInfo(code=e.code, category=e.category, severity=e.severity, title=e.title, why=e.why,
                         todo=e.todo, terms=terms_in(e.why + " " + e.todo),
                         fixes=[FixInfo(label=f["label"], ops=f["ops"]) for f in e.fixes])
        for e in CATALOG.values()
    ]
    return ErrorCatalog(entries=entries, glossary=dict(GLOSSARY), categories=dict(CATEGORIES))


__all__ = ["diagnose", "describe_error", "error_catalog", "lint", "_UnknownNode"]
