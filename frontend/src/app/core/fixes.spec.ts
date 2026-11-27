// Unit tests for the editor's one-click fixes (Stage 21).
// Run alone:  node_modules/.bin/vitest run src/app/core/fixes.spec.ts --globals
//
// fix-cases.json is written by the backend (tests/test_end_to_end.py, _fix_cases):
// a broken graph, a fix the backend offered for it, and the graph after the backend
// applied it. The editor must make exactly the same graph.
import cases from './fix-cases.json';
import { EditorEdge, EditorNode, FixOp } from './graph.models';
import { applyFixOps, freshNodeId, textParts } from './fixes';

interface SpecNode {
  id: string;
  type: string;
  params: Record<string, unknown>;
}
interface Setting {
  source: string;
  target: string;
  transform: string;
  params: Record<string, unknown>;
}
interface Case {
  name: string;
  before: { nodes: SpecNode[]; edges: string[][]; edge_settings: Setting[] };
  ops: FixOp[];
  after: { nodes: SpecNode[]; edges: string[][]; edge_settings: Setting[]; train: Record<string, unknown> };
}

function toEditor(g: Case['before']): { nodes: EditorNode[]; edges: EditorEdge[] } {
  const settings = new Map(g.edge_settings.map((s) => [`${s.source}->${s.target}`, s]));
  return {
    nodes: g.nodes.map((n, i) => ({ id: n.id, type: n.type, x: 40 + 200 * (i % 4), y: 40 + 110 * i, params: n.params })),
    edges: g.edges.map(([source, target]) => {
      const s = settings.get(`${source}->${target}`);
      return s ? { source, target, transform: s.transform, params: s.params } : { source, target };
    }),
  };
}

describe('one-click fixes (Stage 21)', () => {
  const all = cases as unknown as Case[];

  it('the backend wrote cases with every kind of edit', () => {
    const kinds = new Set(all.flatMap((c) => c.ops.map((o) => o.op)));
    for (const k of ['set_param', 'set_inner_param', 'add_inner', 'remove_inner', 'set_transform', 'insert_node',
      'add_node', 'remove_node', 'add_edge', 'remove_edge', 'reorder_inputs', 'rename_node', 'set_train']) {
      expect(kinds.has(k)).toBe(true);
    }
    expect(all.length).toBeGreaterThan(40);
  });

  for (const c of cases as unknown as Case[]) {
    it(`makes the backend's graph: ${c.name}`, () => {
      const res = applyFixOps(toEditor(c.before), c.ops);
      const want = toEditor({ nodes: c.after.nodes, edges: c.after.edges, edge_settings: c.after.edge_settings });
      expect(res.nodes.map((n) => [n.id, n.type, n.params])).toEqual(want.nodes.map((n) => [n.id, n.type, n.params]));
      expect(res.edges.map((e) => [e.source, e.target, e.transform ?? 'auto', e.params ?? {}])).toEqual(
        want.edges.map((e) => [e.source, e.target, e.transform ?? 'auto', e.params ?? {}]),
      );
      expect(res.train).toEqual(c.after.train);
      // new nodes land on the canvas, near what they join
      for (const id of res.added) {
        const n = res.nodes.find((x) => x.id === id)!;
        expect(Number.isFinite(n.x) && Number.isFinite(n.y)).toBe(true);
      }
    });
  }

  it('leaves the graph it was given untouched', () => {
    const c = all.find((x) => x.ops.some((o) => o.op === 'set_param'))!;
    const state = toEditor(c.before);
    const copy = JSON.stringify(state);
    applyFixOps(state, c.ops);
    expect(JSON.stringify(state)).toBe(copy);
  });

  it('gives a new node another id when the suggested one is taken, and later edits follow it', () => {
    const state = {
      nodes: [
        { id: 'in', type: 'Input', x: 0, y: 0, params: {} },
        { id: 'relu_1', type: 'ReLU', x: 0, y: 100, params: {} },
        { id: 'fc', type: 'Linear', x: 0, y: 200, params: {} },
      ],
      edges: [{ source: 'in', target: 'fc' }],
    };
    const res = applyFixOps(state, [
      { op: 'add_node', id: 'relu_1', type: 'ReLU', params: {}, near: 'fc' },
      { op: 'add_edge', source: 'fc', target: 'relu_1' },
    ]);
    expect(res.added).toEqual(['relu_2']);
    expect(res.edges.map((e) => `${e.source}>${e.target}`)).toEqual(['in>fc', 'fc>relu_2']);
    expect(freshNodeId(new Set(['flatten_1']), 'Flatten')).toBe('flatten_2');
  });

  it('collects what only the editor can do', () => {
    const res = applyFixOps({ nodes: [], edges: [] }, [{ op: 'ui', action: 'open_data' }]);
    expect(res.ui).toEqual(['open_data']);
  });

  it('refuses a fix that no longer fits the graph', () => {
    expect(() => applyFixOps({ nodes: [], edges: [] }, [{ op: 'set_param', node: 'gone', name: 'p', value: 1 }])).toThrow();
  });

  it('splits a "why" into words and glossary terms', () => {
    expect(textParts('a [[C×H×W]] picture')).toEqual([
      { text: 'a ', term: null },
      { text: 'C×H×W', term: 'C×H×W' },
      { text: ' picture', term: null },
    ]);
  });
});
