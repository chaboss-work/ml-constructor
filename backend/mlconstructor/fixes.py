"""One-click fixes (Stage 21).

A fix is a short label (what the button says) plus a list of small **graph edits**
— the same list the editor applies on a click and :func:`apply_fix` applies here
(the tests apply every fix and check the problem is gone). Each edit is a dict with
an ``op``:

``set_param``        a node's setting: ``node``, ``name``, ``value``
``set_inner_param``  a setting of a layer inside a Repeat block: ``node``, ``index``, ``name``, ``value``
``add_inner``        a layer put inside a Repeat block: ``node``, ``type``, ``params`` (``index``: where; none = at the end)
``remove_inner``     a layer removed from a Repeat block: ``node``, ``index``
``set_transform``    a connection's transform: ``source``, ``target``, ``transform`` (``auto`` clears it), ``params``
``insert_node``      a new layer on a connection, ``source -> id -> target``: ``source``, ``target``, ``id``, ``type``, ``params``
``add_node``         a new node: ``id``, ``type``, ``params`` (``near``: a node to put it next to)
``remove_node``      a node and its connections: ``node``; with ``bridge`` its one input is wired to what it fed
``add_edge``         a connection ``source -> target`` (it becomes the target's last input)
``remove_edge``      a connection: ``source``, ``target``
``reorder_inputs``   a layer's inputs reconnected in this order: ``node``, ``sources``
``rename_node``      the ``occurrence``-th node called ``node`` gets ``new_id`` (two nodes shared one id)
``set_train``        a training setting: ``name``, ``value``
``ui``               something only the editor does (open Data, the library…): ``action`` — no graph edit

Self-contained on purpose (no builder import), so the builder and the layers can
make fixes where they find a problem.
"""

from __future__ import annotations

from typing import Any

#: every op an edit may have (the editor implements the same set)
OPS = (
    "set_param", "set_inner_param", "add_inner", "remove_inner", "set_transform", "insert_node", "add_node",
    "remove_node", "add_edge", "remove_edge", "reorder_inputs", "rename_node", "set_train", "ui",
)
#: what a "ui" edit may ask the editor to do
UI_ACTIONS = ("open_data", "use_example_text", "open_library", "open_weights", "train", "run_once", "draw_each")


def fix(label: str, *ops: dict) -> dict:
    return {"label": label, "ops": [dict(o) for o in ops]}


def _show(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (list, tuple)):
        return "×".join(str(v) for v in value)
    return str(value)


# --- single edits ---------------------------------------------------------------- #
def op_set_param(node: str, name: str, value: Any) -> dict:
    return {"op": "set_param", "node": node, "name": name, "value": value}


def op_set_inner_param(node: str, index: int, name: str, value: Any) -> dict:
    return {"op": "set_inner_param", "node": node, "index": int(index), "name": name, "value": value}


def op_add_inner(node: str, type_name: str, params: dict | None = None, index: int | None = None) -> dict:
    return {"op": "add_inner", "node": node, "type": type_name, "params": dict(params or {}), "index": index}


def op_remove_inner(node: str, index: int) -> dict:
    return {"op": "remove_inner", "node": node, "index": int(index)}


def op_set_transform(source: str, target: str, transform: str, params: dict | None = None) -> dict:
    return {"op": "set_transform", "source": source, "target": target, "transform": transform,
            "params": dict(params or {})}


def op_insert_node(source: str, target: str, new_id: str, type_name: str, params: dict | None = None) -> dict:
    return {"op": "insert_node", "source": source, "target": target, "id": new_id, "type": type_name,
            "params": dict(params or {})}


def op_add_node(new_id: str, type_name: str, params: dict | None = None, near: str | None = None) -> dict:
    return {"op": "add_node", "id": new_id, "type": type_name, "params": dict(params or {}), "near": near}


def op_remove_node(node: str, bridge: bool = False) -> dict:
    return {"op": "remove_node", "node": node, "bridge": bool(bridge)}


def op_add_edge(source: str, target: str) -> dict:
    return {"op": "add_edge", "source": source, "target": target}


def op_remove_edge(source: str, target: str) -> dict:
    return {"op": "remove_edge", "source": source, "target": target}


def op_reorder_inputs(node: str, sources: list[str]) -> dict:
    return {"op": "reorder_inputs", "node": node, "sources": list(sources)}


def op_rename_node(node: str, occurrence: int, new_id: str) -> dict:
    return {"op": "rename_node", "node": node, "occurrence": int(occurrence), "new_id": new_id}


def op_set_train(name: str, value: Any) -> dict:
    return {"op": "set_train", "name": name, "value": value}


def op_ui(action: str) -> dict:
    assert action in UI_ACTIONS, action
    return {"op": "ui", "action": action}


# --- common fixes ---------------------------------------------------------------- #
def set_param(node: str, name: str, value: Any, label: str | None = None) -> dict:
    return fix(label or f"Set {name} to {_show(value)}", op_set_param(node, name, value))


def set_inner_param(node: str, index: int, name: str, value: Any, label: str | None = None) -> dict:
    return fix(label or f"Set {name} to {_show(value)} (layer {index + 1} of the block)",
               op_set_inner_param(node, index, name, value))


def set_train(name: str, value: Any, label: str) -> dict:
    return fix(label, op_set_train(name, value))


def ui(action: str, label: str) -> dict:
    return fix(label, op_ui(action))


def nearest_divisors(n: int, k: int, also: int | None = None) -> list[int]:
    """Divisors of ``n`` (and of ``also``, when given) closest to ``k`` — the nearest
    below and the nearest above, ``k`` itself left out — closest first."""
    n = int(n)
    divs = [d for d in range(1, n + 1) if n % d == 0 and d != k and (also is None or int(also) % d == 0)]
    below = [d for d in divs if d < k]
    above = [d for d in divs if d > k]
    out = ([below[-1]] if below else []) + ([above[0]] if above else [])
    return sorted(out, key=lambda d: (abs(d - k), -d))


def _singular(what: str) -> str:
    """" numbers per head" -> " number per head" (for a count of one)."""
    words = what.split(" ")
    for i, w in enumerate(words):
        if w.endswith("es") and w[:-2].endswith(("ch", "sh", "x")):
            words[i] = w[:-2]
            break
        if w.endswith("s") and len(w) > 2:
            words[i] = w[:-1]
            break
    return " ".join(words)


def divisor_fixes(node: str, param: str, n: int, k: int, what: str = "", also: int | None = None) -> list[dict]:
    """"Set heads to 5 (30 ÷ 5 = 6 numbers per head)" — the nearest values of ``param`` that divide ``n``."""
    out = []
    for d in nearest_divisors(n, k, also):
        q = n // d
        out.append(set_param(node, param, d, f"Set {param} to {d} ({n} ÷ {d} = {q}{what if q != 1 else _singular(what)})"))
    return out


def fresh_id(existing: set[str] | list[str], type_name: str) -> str:
    """A free node id, as the editor makes them: ``flatten_1``, ``flatten_2``…"""
    taken = set(existing)
    base = type_name.lower()
    k = 1
    while f"{base}_{k}" in taken:
        k += 1
    return f"{base}_{k}"


def fix_kind(f: dict) -> str:
    """The first graph edit's op ("ui" when the fix only asks the editor to do something)."""
    ops = f.get("ops") or []
    return ops[0]["op"] if ops else "ui"


# --- applying a fix to a graph spec ---------------------------------------------- #
def apply_fix(spec, f: dict):
    """A new :class:`~mlconstructor.schema.GraphSpec` with the fix's edits applied
    (``ui`` edits change nothing here). The spec passed in is not modified."""
    return apply_ops(spec, f.get("ops") or [])


def apply_ops(spec, ops: list[dict]):
    from .schema import EdgeSetting, GraphSpec, NodeSpec

    nodes = [NodeSpec(id=n.id, type=n.type, params=_copy(n.params)) for n in spec.nodes]
    edges = [list(e) for e in spec.edges]
    settings = {(s.source, s.target): (s.transform, _copy(s.params)) for s in spec.edge_settings}
    train = spec.train.model_dump()

    def node(nid: str) -> NodeSpec:
        for n in nodes:
            if n.id == nid:
                return n
        raise ValueError(f"the fix names a node that is not in the graph: {nid!r}")

    def inner_layers(nid: str) -> list:
        n = node(nid)
        layers = [dict(l) for l in (n.params.get("layers") or [])]
        n.params["layers"] = layers
        return layers

    for o in ops:
        kind = o.get("op")
        if kind == "set_param":
            node(o["node"]).params[o["name"]] = _copy(o["value"])
        elif kind == "set_inner_param":
            layers = inner_layers(o["node"])
            layer = layers[int(o["index"])]
            layer["params"] = {**(layer.get("params") or {}), o["name"]: _copy(o["value"])}
        elif kind == "add_inner":
            layers = inner_layers(o["node"])
            at = len(layers) if o.get("index") is None else int(o["index"])
            layers.insert(at, {"type": o["type"], "params": _copy(o.get("params") or {})})
        elif kind == "remove_inner":
            del inner_layers(o["node"])[int(o["index"])]
        elif kind == "set_transform":
            key = (o["source"], o["target"])
            if o.get("transform", "auto") == "auto":
                settings.pop(key, None)
            else:
                settings[key] = (o["transform"], _copy(o.get("params") or {}))
        elif kind == "insert_node":
            s, t, new = o["source"], o["target"], o["id"]
            if any(n.id == new for n in nodes):
                raise ValueError(f"the fix adds a node whose id is taken: {new!r}")
            nodes.append(NodeSpec(id=new, type=o["type"], params=_copy(o.get("params") or {})))
            at = edges.index([s, t])  # the new connection into the target keeps the old one's place (its port)
            edges[at:at + 1] = [[s, new], [new, t]]
            settings.pop((s, t), None)
        elif kind == "add_node":
            if any(n.id == o["id"] for n in nodes):
                raise ValueError(f"the fix adds a node whose id is taken: {o['id']!r}")
            nodes.append(NodeSpec(id=o["id"], type=o["type"], params=_copy(o.get("params") or {})))
        elif kind == "remove_node":
            nid = o["node"]
            node(nid)
            ins = [s for s, d in edges if d == nid]
            new_edges: list[list[str]] = []
            for s, d in edges:
                if d == nid:
                    continue
                if s == nid:
                    if o.get("bridge") and len(ins) == 1 and [ins[0], d] not in new_edges and [ins[0], d] not in edges:
                        new_edges.append([ins[0], d])  # in the removed connection's place (keeps the port)
                    continue
                new_edges.append([s, d])
            edges = new_edges
            nodes = [n for n in nodes if n.id != nid]
            settings = {k: v for k, v in settings.items() if nid not in k}
        elif kind == "add_edge":
            if [o["source"], o["target"]] not in edges:
                edges.append([o["source"], o["target"]])
        elif kind == "remove_edge":
            edges = [e for e in edges if e != [o["source"], o["target"]]]
            settings.pop((o["source"], o["target"]), None)
        elif kind == "reorder_inputs":
            nid, order = o["node"], list(o["sources"])
            current = [s for s, d in edges if d == nid]
            if sorted(current) != sorted(order):
                raise ValueError(f"the fix reorders inputs {nid!r} does not have: {order}")
            slots = iter(order)  # the same places in the list, the sources in the new order
            edges = [[next(slots), d] if d == nid else [s, d] for s, d in edges]
        elif kind == "rename_node":
            seen = 0
            for n in nodes:
                if n.id == o["node"]:
                    if seen == int(o.get("occurrence", 1)):
                        n.id = o["new_id"]
                        break
                    seen += 1
        elif kind == "set_train":
            train[o["name"]] = _copy(o["value"])
        elif kind == "ui":
            pass
        else:
            raise ValueError(f"unknown fix edit {kind!r}")

    return GraphSpec(
        nodes=nodes,
        edges=edges,
        train=type(spec.train).model_validate(train),
        edge_settings=[EdgeSetting(source=s, target=t, transform=k, params=p) for (s, t), (k, p) in settings.items()],
    )


def _copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy(v) for v in value]
    return value
