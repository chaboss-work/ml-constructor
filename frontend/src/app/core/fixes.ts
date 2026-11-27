// Stage 21 — one-click fixes, applied in the editor.
//
// A fix is a list of small graph edits (the backend's mlconstructor/fixes.py
// documents every op). The backend tries each fix on a copy of the graph before it
// offers it; this module applies the very same edits to the editor's nodes and
// connections — same order of connections (a several-input layer tells its inputs
// apart by it), same ids — and places new nodes next to the ones they join.
// fix-cases.json holds cases the backend wrote; both sides are tested against them.
//
// Framework-free on purpose (unit-tested with vitest, like project.ts).
import { EditorEdge, EditorNode, FixOp } from './graph.models';

export interface FixState {
  nodes: EditorNode[];
  edges: EditorEdge[];
}

export interface FixResult extends FixState {
  /** training settings the fix changes (name -> value) */
  train: Record<string, unknown>;
  /** things only the editor can do (open Data, the library…), in order */
  ui: string[];
  /** ids of the nodes the fix added */
  added: string[];
}

export interface FixLayout {
  nodeW: number;
  nodeH: number;
}

const DEFAULT_LAYOUT: FixLayout = { nodeW: 174, nodeH: 78 };

function clone<T>(v: T): T {
  return v === undefined ? v : (JSON.parse(JSON.stringify(v)) as T);
}

/** A free id like the editor's own (``flatten_1``, ``flatten_2``…). */
export function freshNodeId(taken: Set<string>, type: string): string {
  const base = type.toLowerCase();
  let k = 1;
  while (taken.has(`${base}_${k}`)) k += 1;
  return `${base}_${k}`;
}

/** Apply a fix's edits. The state passed in is not modified. */
export function applyFixOps(state: FixState, ops: FixOp[], layout: FixLayout = DEFAULT_LAYOUT): FixResult {
  let nodes: EditorNode[] = state.nodes.map((n) => ({ ...n, params: clone(n.params) }));
  let edges: EditorEdge[] = state.edges.map((e) => ({ ...e, params: e.params ? clone(e.params) : e.params }));
  const train: Record<string, unknown> = {};
  const ui: string[] = [];
  const added: string[] = [];
  /** a new node whose suggested id was taken here gets another one, and later edits follow it */
  const renamed = new Map<string, string>();
  const id = (v: unknown): string => {
    const s = String(v ?? '');
    return renamed.get(s) ?? s;
  };
  const node = (nid: string): EditorNode => {
    const n = nodes.find((x) => x.id === nid);
    if (!n) throw new Error(`the fix names a node that is not in the graph: ${nid}`);
    return n;
  };
  const innerLayers = (nid: string): { type: string; params: Record<string, unknown> }[] => {
    const n = node(nid);
    const layers = ((n.params['layers'] as { type: string; params: Record<string, unknown> }[]) ?? []).map((l) => ({
      ...l,
      params: { ...(l.params ?? {}) },
    }));
    n.params = { ...n.params, layers };
    return layers;
  };
  const newId = (wanted: unknown, type: string): string => {
    const taken = new Set(nodes.map((n) => n.id));
    const want = String(wanted ?? '');
    const got = want && !taken.has(want) ? want : freshNodeId(taken, type);
    if (want && got !== want) renamed.set(want, got);
    return got;
  };
  const place = (type: string, near: string | null): { x: number; y: number } => {
    const n = near ? nodes.find((x) => x.id === near) : undefined;
    if (!n) return { x: 40, y: 40 + nodes.length * 12 };
    const step = layout.nodeH + 44;
    if (type === 'Input') return { x: n.x, y: Math.max(0, n.y - step) };
    if (type === 'Output') return { x: n.x, y: n.y + step };
    return { x: n.x + layout.nodeW + 40, y: n.y };
  };

  for (const o of ops) {
    switch (o.op) {
      case 'set_param': {
        const n = node(id(o['node']));
        n.params = { ...n.params, [String(o['name'])]: clone(o['value']) };
        break;
      }
      case 'set_inner_param': {
        const layers = innerLayers(id(o['node']));
        const layer = layers[Number(o['index'])];
        if (!layer) throw new Error('the fix names a layer the block does not have');
        layer.params = { ...layer.params, [String(o['name'])]: clone(o['value']) };
        break;
      }
      case 'add_inner': {
        const layers = innerLayers(id(o['node']));
        const at = o['index'] === null || o['index'] === undefined ? layers.length : Number(o['index']);
        layers.splice(at, 0, { type: String(o['type']), params: clone((o['params'] as Record<string, unknown>) ?? {}) });
        break;
      }
      case 'remove_inner': {
        innerLayers(id(o['node'])).splice(Number(o['index']), 1);
        break;
      }
      case 'set_transform': {
        const s = id(o['source']);
        const t = id(o['target']);
        const e = edges.find((x) => x.source === s && x.target === t);
        if (!e) throw new Error(`the fix names a connection that is not in the graph: ${s} → ${t}`);
        const kind = String(o['transform'] ?? 'auto');
        if (kind === 'auto') {
          delete e.transform;
          delete e.params;
        } else {
          e.transform = kind;
          e.params = clone((o['params'] as Record<string, unknown>) ?? {});
        }
        break;
      }
      case 'insert_node': {
        const s = id(o['source']);
        const t = id(o['target']);
        const at = edges.findIndex((x) => x.source === s && x.target === t);
        if (at < 0) throw new Error(`the fix names a connection that is not in the graph: ${s} → ${t}`);
        const type = String(o['type']);
        const nid = newId(o['id'], type);
        const a = node(s);
        const b = node(t);
        nodes.push({
          id: nid,
          type,
          x: Math.round((a.x + b.x) / 2),
          y: Math.round((a.y + b.y) / 2),
          params: clone((o['params'] as Record<string, unknown>) ?? {}),
        });
        added.push(nid);
        // the new connection into the target keeps the old one's place (its input port)
        edges.splice(at, 1, { source: s, target: nid }, { source: nid, target: t });
        break;
      }
      case 'add_node': {
        const type = String(o['type']);
        const near = o['near'] ? id(o['near']) : null;
        const pos = place(type, near);
        const nid = newId(o['id'], type);
        nodes.push({ id: nid, type, ...pos, params: clone((o['params'] as Record<string, unknown>) ?? {}) });
        added.push(nid);
        break;
      }
      case 'remove_node': {
        const nid = id(o['node']);
        node(nid);
        const ins = edges.filter((e) => e.target === nid).map((e) => e.source);
        const next: EditorEdge[] = [];
        for (const e of edges) {
          if (e.target === nid) continue;
          if (e.source === nid) {
            const from = ins[0];
            if (
              o['bridge'] &&
              ins.length === 1 &&
              !next.some((x) => x.source === from && x.target === e.target) &&
              !edges.some((x) => x.source === from && x.target === e.target)
            ) {
              next.push({ source: from, target: e.target }); // in the removed connection's place
            }
            continue;
          }
          next.push(e);
        }
        edges = next;
        nodes = nodes.filter((n) => n.id !== nid);
        break;
      }
      case 'add_edge': {
        const s = id(o['source']);
        const t = id(o['target']);
        if (!edges.some((e) => e.source === s && e.target === t)) edges.push({ source: s, target: t });
        break;
      }
      case 'remove_edge': {
        const s = id(o['source']);
        const t = id(o['target']);
        edges = edges.filter((e) => !(e.source === s && e.target === t));
        break;
      }
      case 'reorder_inputs': {
        const nid = id(o['node']);
        const order = ((o['sources'] as unknown[]) ?? []).map(id);
        const into = edges.filter((e) => e.target === nid);
        const byFrom = new Map(into.map((e) => [e.source, e]));
        if (order.length !== into.length || order.some((s) => !byFrom.has(s))) {
          throw new Error('the fix reorders inputs the layer does not have');
        }
        let k = 0; // the same places in the list, the sources in the new order (each keeps its setting)
        edges = edges.map((e) => (e.target === nid ? byFrom.get(order[k++])! : e));
        break;
      }
      case 'rename_node': {
        let seen = 0;
        for (const n of nodes) {
          if (n.id === String(o['node'])) {
            if (seen === Number(o['occurrence'] ?? 1)) {
              n.id = String(o['new_id']);
              break;
            }
            seen += 1;
          }
        }
        break;
      }
      case 'set_train':
        train[String(o['name'])] = clone(o['value']);
        break;
      case 'ui':
        ui.push(String(o['action']));
        break;
      default:
        throw new Error(`unknown fix edit ${o.op}`);
    }
  }
  return { nodes, edges, train, ui, added };
}

/** "Why" text with glossary words marked [[like this]] -> pieces to show (a word is clickable). */
export function textParts(text: string): { text: string; term: string | null }[] {
  const out: { text: string; term: string | null }[] = [];
  const re = /\[\[([^\]]+)\]\]/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push({ text: text.slice(last, m.index), term: null });
    out.push({ text: m[1], term: m[1] });
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({ text: text.slice(last), term: null });
  return out;
}
