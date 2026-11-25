// Unit tests for the framework-free project module (Stage 13).
// Run alone:  node_modules/.bin/vitest run src/app/core/project.spec.ts --globals
import { ProjectFile } from './graph.models';
import {
  DEFAULT_VIEW,
  EditorSnapshot,
  MAX_ZOOM,
  PROJECT_FORMAT,
  PROJECT_FORMAT_VERSION,
  ProjectStorage,
  fingerprint,
  fromProjectFile,
  layoutGraph,
  projectFileName,
  readCurrentFormat,
  toProjectFile,
} from './project';

/** An in-memory Storage with an optional size quota (in characters). */
class MemoryStorage {
  private data = new Map<string, string>();
  constructor(private readonly quota = Infinity) {}
  get length(): number {
    return this.data.size;
  }
  clear(): void {
    this.data.clear();
  }
  key(i: number): string | null {
    return [...this.data.keys()][i] ?? null;
  }
  getItem(k: string): string | null {
    return this.data.has(k) ? this.data.get(k)! : null;
  }
  removeItem(k: string): void {
    this.data.delete(k);
  }
  setItem(k: string, v: string): void {
    let used = 0;
    for (const [kk, vv] of this.data) if (kk !== k) used += kk.length + vv.length;
    if (used + k.length + v.length > this.quota) throw new Error('QuotaExceededError');
    this.data.set(k, v);
  }
}

const storage = (quota = Infinity) => new ProjectStorage(new MemoryStorage(quota) as unknown as Storage);

const SNAP: EditorSnapshot = {
  name: 'Моя сеть: CNN + блок',
  description: 'Пример с кириллицей\nи переводом строки',
  createdAt: '2026-11-25T10:00:00Z',
  nodes: [
    { id: 'input_1', type: 'Input', x: 70, y: 24, params: { shape: [1, 28, 28] } },
    { id: 'conv2d_2', type: 'Conv2d', x: 70.5, y: 136.25, params: { out_channels: 8, kernel_size: 3 } },
    {
      id: 'repeat_3',
      type: 'Repeat',
      x: 301.123456789,
      y: 248,
      params: { repeat: 2, layers: [{ type: 'Linear', params: { out_features: 16 } }, { type: 'ReLU', params: {} }] },
    },
    { id: 'output_4', type: 'Output', x: -12, y: 360, params: {} },
  ],
  edges: [
    { source: 'input_1', target: 'conv2d_2' },
    { source: 'conv2d_2', target: 'repeat_3' },
    { source: 'repeat_3', target: 'output_4' },
  ],
  train: {
    dataset: 'custom',
    epochs: 7,
    lr: 0.003,
    batch_size: 32,
    samples: 256,
    custom_rows: [
      [1.5, 0, 0],
      [0, 2.25, 1],
    ],
    checkpoint: true,
    device: 'auto',
  },
  trainExtra: { optimizer: 'sgd', seed: 42, steps: null },
  customText: '1.5, 0, 0\n0, 2.25, 1\n# комментарий',
  view: { pan_x: -40.5, pan_y: 12, zoom: 0.8 },
};

describe('project file <-> editor state', () => {
  it('save -> open reproduces the editor state exactly', () => {
    const file = toProjectFile(SNAP);
    expect(file.format).toBe(PROJECT_FORMAT);
    expect(file.version).toBe(PROJECT_FORMAT_VERSION);
    const onDisk = JSON.parse(JSON.stringify(file)); // what a saved file reads back as
    const reopened = fromProjectFile(readCurrentFormat(onDisk)!);
    expect(reopened).toEqual(SNAP);
    // and saving the reopened state again gives the very same file
    expect(toProjectFile(reopened)).toEqual(file);
  });

  it('keeps nested params independent of the file object', () => {
    const file = toProjectFile(SNAP);
    const reopened = fromProjectFile(file);
    (reopened.nodes[2].params['layers'] as { type: string }[])[0].type = 'GELU';
    expect((file.graph.nodes[2].params['layers'] as { type: string }[])[0].type).toBe('Linear');
  });

  it('lays out nodes that come without positions (a bare graph spec)', () => {
    const file = toProjectFile(SNAP);
    file.graph.nodes = file.graph.nodes.map(({ position: _p, ...n }) => n);
    const s = fromProjectFile(file);
    const ys = s.nodes.map((n) => n.y);
    expect(ys).toEqual([...ys].sort((a, b) => a - b)); // a chain flows top -> bottom
    expect(new Set(ys).size).toBe(4);
    expect(s.nodes.every((n) => n.x === 70)).toBe(true);
  });

  it('puts laid-out nodes to the right of placed ones when only some have positions', () => {
    const file = toProjectFile(SNAP);
    delete file.graph.nodes[3].position;
    const s = fromProjectFile(file);
    expect(s.nodes.slice(0, 3).map((n) => [n.x, n.y])).toEqual(SNAP.nodes.slice(0, 3).map((n) => [n.x, n.y]));
    expect(s.nodes[3].x).toBeGreaterThan(Math.max(...SNAP.nodes.slice(0, 3).map((n) => n.x)));
  });

  it('falls back to safe defaults for a broken view and clamps the zoom', () => {
    const file = toProjectFile(SNAP);
    file.editor = { view: { pan_x: Number.NaN, pan_y: 0, zoom: 1 } };
    expect(fromProjectFile(file).view).toEqual(DEFAULT_VIEW);
    file.editor = { view: { pan_x: 0, pan_y: 0, zoom: 99 } };
    expect(fromProjectFile(file).view.zoom).toBe(MAX_ZOOM);
  });

  it('splits train settings into editor-controlled and carried-along ones', () => {
    const s = fromProjectFile(toProjectFile(SNAP));
    expect(Object.keys(s.trainExtra).sort()).toEqual(['optimizer', 'seed', 'steps']);
    expect(s.train.custom_rows).toEqual(SNAP.train.custom_rows);
  });

  it('reads only the current format directly', () => {
    const file = toProjectFile(SNAP);
    expect(readCurrentFormat(file)).not.toBeNull();
    expect(readCurrentFormat({ ...file, version: 2 })).toBeNull(); // newer -> backend decides
    expect(readCurrentFormat({ nodes: [], edges: [] })).toBeNull(); // bare spec -> backend migrates
    expect(readCurrentFormat([1, 2])).toBeNull();
    expect(readCurrentFormat({ ...file, graph: { nodes: [{ id: 1 }], edges: [] } })).toBeNull();
  });
});

describe('fingerprint (unsaved changes)', () => {
  it('ignores dates but sees every content change', () => {
    const a = toProjectFile(SNAP);
    const b: ProjectFile = { ...toProjectFile(SNAP), created_at: 'x', updated_at: 'y' };
    expect(fingerprint(a)).toBe(fingerprint(b));
    const moved = toProjectFile({ ...SNAP, nodes: SNAP.nodes.map((n, i) => (i ? n : { ...n, x: n.x + 1 })) });
    expect(fingerprint(moved)).not.toBe(fingerprint(a));
    expect(fingerprint(toProjectFile({ ...SNAP, name: 'Другое имя' }))).not.toBe(fingerprint(a));
    expect(fingerprint(toProjectFile({ ...SNAP, customText: 'edited' }))).not.toBe(fingerprint(a));
  });

  it('does not count panning or zooming the canvas as a change', () => {
    const a = toProjectFile(SNAP);
    const panned = toProjectFile({ ...SNAP, view: { pan_x: 500, pan_y: -20, zoom: 2 } });
    expect(fingerprint(panned)).toBe(fingerprint(a));
  });
});

describe('download file name', () => {
  it('keeps the title, drops characters file systems refuse', () => {
    expect(projectFileName('Моя сеть')).toBe('Моя сеть.mlc.json');
    expect(projectFileName('a/b\\c:d*e?"f"<g>|h')).toBe('a b c d e f g h.mlc.json');
    expect(projectFileName('  ...  ')).toBe('project.mlc.json');
    expect(projectFileName('trailing dot.')).toBe('trailing dot.mlc.json');
  });
});

describe('layoutGraph', () => {
  it('spreads branches side by side and survives a cycle', () => {
    const nodes = ['in', 'a', 'b', 'add'].map((id) => ({ id, type: 'X', params: {} }));
    const laid = layoutGraph(nodes, [
      ['in', 'a'],
      ['in', 'b'],
      ['a', 'add'],
      ['b', 'add'],
    ]);
    expect(laid[1].y).toBe(laid[2].y);
    expect(laid[1].x).not.toBe(laid[2].x);
    expect(laid[3].y).toBeGreaterThan(laid[1].y);
    expect(layoutGraph(nodes.slice(0, 2), [['in', 'a'], ['a', 'in']])).toHaveLength(2);
  });
});

describe('browser storage', () => {
  it('saves, lists (most recent first), overwrites and removes projects', () => {
    const st = storage();
    const p = toProjectFile({ ...SNAP, createdAt: null });
    const first = st.save(null, p, '2026-11-25T10:00:00Z');
    const second = st.save(null, { ...p, name: 'Second' }, '2026-11-25T11:00:00Z');
    expect(st.list().map((m) => m.name)).toEqual(['Second', SNAP.name]);
    expect(st.list()[1].node_count).toBe(4);
    expect(readCurrentFormat(st.load(first.id))).toEqual(first.project);
    // overwriting keeps the creation date and moves the project to the top
    const again = st.save(first.id, { ...first.project, name: 'Renamed' }, '2026-11-25T12:00:00Z');
    expect(again.project.created_at).toBe('2026-11-25T10:00:00Z');
    expect(st.list().map((m) => m.name)).toEqual(['Renamed', 'Second']);
    st.remove(second.id);
    expect(st.list().map((m) => m.id)).toEqual([first.id]);
    expect(st.load(second.id)).toBeNull();
  });

  it('fails a save cleanly when storage is full', () => {
    const st = storage(20_000);
    const small = toProjectFile(SNAP);
    const big = toProjectFile({ ...SNAP, customText: 'x'.repeat(50_000) });
    const kept = st.save(null, small);
    expect(() => st.save(null, big)).toThrow();
    expect(st.list().map((m) => m.id)).toEqual([kept.id]); // nothing half-saved
  });

  it('rolls a new project back when the project list cannot be updated', () => {
    const now = '2026-11-25T10:00:00Z';
    const project = toProjectFile(SNAP);
    const itemChars = JSON.stringify({ ...project, created_at: project.created_at ?? now, updated_at: now }).length;
    // room for the project itself (+ its ~40-char key), but not for the list entry after it
    const mem = new MemoryStorage(itemChars + 60);
    const st = new ProjectStorage(mem as unknown as Storage);
    expect(() => st.save(null, project, now)).toThrow();
    expect(mem.length).toBe(0); // no orphan the list could never show
  });

  it('round-trips the autosaved draft', () => {
    const st = storage();
    expect(st.readAutosave()).toBeNull();
    const project = toProjectFile(SNAP);
    st.writeAutosave({ project, origin: 'browser', localId: 'p1', serverSlug: null, savedFp: fingerprint(project), at: 'now' });
    const rec = st.readAutosave()!;
    expect(fromProjectFile(readCurrentFormat(rec.project)!)).toEqual(SNAP);
    expect(rec.savedFp).toBe(fingerprint(project));
  });

  it('works without storage at all (reads empty, writes throw)', () => {
    const st = new ProjectStorage(null);
    expect(st.available).toBe(false);
    expect(st.list()).toEqual([]);
    expect(st.readAutosave()).toBeNull();
    expect(() => st.save(null, toProjectFile(SNAP))).toThrow();
  });
});
