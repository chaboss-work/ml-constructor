/**
 * Projects (Stage 13): the versioned `.mlc.json` file, converting it to and from the
 * editor's state, and browser storage (autosave + a small project library).
 *
 * Framework-free on purpose, so it is unit-tested without Angular (project.spec.ts).
 * The file format and its migrations live in the backend
 * (backend/mlconstructor/projects.py): files from disk or the server are opened
 * through POST /projects/normalize. This module only reads the *current* format
 * directly — what this very app wrote into browser storage.
 */

import { EdgeSetting, EditorEdge, EditorNode, ProjectFile, ProjectNode, TrainCfg } from './graph.models';

export const PROJECT_FORMAT = 'mlconstructor.project';
/** 1 — Stage 13; 2 — Stage 14 adds per-connection settings (graph.edge_settings). */
export const PROJECT_FORMAT_VERSION = 2;
/** A connection with no setting adapts shapes on its own. */
export const AUTO = 'auto';
export const PROJECT_FILE_SUFFIX = '.mlc.json';
/** Same cap as the backend (one project file, pasted rows included). */
export const MAX_PROJECT_BYTES = 5 * 1024 * 1024;
export const MAX_NAME_LENGTH = 120;
export const MAX_DESCRIPTION_LENGTH = 2000;

/** Training settings the editor has its own controls for. Anything else in a
 * project's `train` (optimizer, seed, …) is carried along untouched. */
export const UI_TRAIN_KEYS: readonly string[] = [
  'dataset',
  'epochs',
  'lr',
  'batch_size',
  'samples',
  'custom_rows',
  'checkpoint',
  'resume',
  'device',
  'val_fraction',
  'test_fraction',
  'text',
];

export interface EditorView {
  pan_x: number;
  pan_y: number;
  zoom: number;
}

export const DEFAULT_VIEW: EditorView = { pan_x: 48, pan_y: 36, zoom: 1 };
export const MIN_ZOOM = 0.3;
export const MAX_ZOOM = 2.5;

/** Everything a project holds, in the editor's own terms. */
export interface EditorSnapshot {
  name: string;
  description: string;
  createdAt: string | null;
  nodes: EditorNode[];
  edges: EditorEdge[];
  /** The settings the editor controls (see UI_TRAIN_KEYS). */
  train: Partial<TrainCfg>;
  /** Every other training setting, carried along untouched. */
  trainExtra: Record<string, unknown>;
  customText: string;
  view: EditorView;
}

/** Where the open project lives — decides what "Save" does. */
export type ProjectOrigin = 'new' | 'browser' | 'server' | 'file';

// --------------------------------------------------------------------------- //
// Editor state <-> project file
// --------------------------------------------------------------------------- //
export function toProjectFile(s: EditorSnapshot): ProjectFile {
  const train: Record<string, unknown> = { ...s.trainExtra };
  for (const [key, value] of Object.entries(s.train)) {
    if (value !== undefined) train[key] = value;
  }
  return {
    format: PROJECT_FORMAT,
    version: PROJECT_FORMAT_VERSION,
    name: s.name.trim().slice(0, MAX_NAME_LENGTH) || 'Untitled',
    description: s.description.slice(0, MAX_DESCRIPTION_LENGTH),
    created_at: s.createdAt,
    updated_at: null,
    graph: {
      nodes: s.nodes.map((n) => ({ id: n.id, type: n.type, params: n.params, position: { x: n.x, y: n.y } })),
      edges: s.edges.map((e) => [e.source, e.target]),
      edge_settings: edgeSettings(s.edges),
    },
    train,
    editor: { custom_text: s.customText, view: { ...s.view } },
  };
}

export function fromProjectFile(p: ProjectFile): EditorSnapshot {
  const specNodes: ProjectNode[] = p.graph?.nodes ?? [];
  const specEdges = (p.graph?.edges ?? []).filter((e) => Array.isArray(e) && e.length === 2);
  const settings = new Map((p.graph?.edge_settings ?? []).map((s) => [`${s.source}->${s.target}`, s]));
  // Nodes saved with a position keep it. Nodes without one (a bare graph spec, or a
  // hand-edited file) are laid out by depth; when some nodes do have positions, the
  // laid-out ones go to the right of them so nothing lands on top of the user's work.
  const laid = layoutGraph(specNodes, specEdges);
  const placed = specNodes.filter((n) => isPoint(n.position));
  const shiftX = placed.length ? Math.max(...placed.map((n) => n.position!.x)) + 240 : 0;
  const nodes: EditorNode[] = specNodes.map((n, i) =>
    isPoint(n.position)
      ? { id: n.id, type: n.type, x: n.position!.x, y: n.position!.y, params: structuredClone(n.params ?? {}) }
      : { ...laid[i], x: laid[i].x + shiftX },
  );
  const train: Record<string, unknown> = {};
  const trainExtra: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(p.train ?? {})) {
    (UI_TRAIN_KEYS.includes(key) ? train : trainExtra)[key] = value;
  }
  const v = p.editor?.view;
  const view: EditorView =
    v && [v.pan_x, v.pan_y, v.zoom].every((x) => typeof x === 'number' && Number.isFinite(x))
      ? { pan_x: v.pan_x, pan_y: v.pan_y, zoom: Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, v.zoom)) }
      : { ...DEFAULT_VIEW };
  return {
    name: p.name || 'Untitled',
    description: p.description ?? '',
    createdAt: p.created_at ?? null,
    nodes,
    edges: specEdges.map(([source, target]) => {
      const s = settings.get(`${source}->${target}`);
      return s && s.transform && s.transform !== AUTO
        ? { source, target, transform: s.transform, params: structuredClone(s.params ?? {}) }
        : { source, target };
    }),
    train: train as Partial<TrainCfg>,
    trainExtra,
    customText: p.editor?.custom_text ?? '',
    view,
  };
}

/** The connections that do not simply use "auto", as the backend expects them. */
export function edgeSettings(edges: EditorEdge[]): EdgeSetting[] {
  return edges
    .filter((e) => e.transform && e.transform !== AUTO)
    .map((e) => ({ source: e.source, target: e.target, transform: e.transform as string, params: e.params ?? {} }));
}

function isPoint(pos: ProjectNode['position']): boolean {
  return !!pos && Number.isFinite(pos.x) && Number.isFinite(pos.y);
}

/** Read a project that is already in the current format (what this app wrote to
 * browser storage). Returns null for anything else — older or newer formats and
 * foreign files go through the backend's migrations instead. */
export function readCurrentFormat(raw: unknown): ProjectFile | null {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
  const p = raw as ProjectFile;
  if (p.format !== PROJECT_FORMAT || p.version !== PROJECT_FORMAT_VERSION) return null;
  if (!p.graph || !Array.isArray(p.graph.nodes) || !Array.isArray(p.graph.edges)) return null;
  if (!p.graph.nodes.every((n) => n && typeof n.id === 'string' && typeof n.type === 'string')) return null;
  return p;
}

/** A short fingerprint of a project's content, to tell whether there are unsaved
 * changes. Dates and the canvas view (pan / zoom) are left out — moving around is
 * not an edit. FNV-1a over the JSON, plus its length. */
export function fingerprint(p: ProjectFile): string {
  const { created_at: _c, updated_at: _u, editor, ...content } = p;
  const s = JSON.stringify({ ...content, custom_text: editor?.custom_text ?? '' });
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return `${(h >>> 0).toString(16).padStart(8, '0')}:${s.length.toString(36)}`;
}

/** A download name for a project: its title (any language) minus the characters
 * file systems refuse, plus the `.mlc.json` suffix. */
export function projectFileName(name: string): string {
  const base = name
    .replace(/[\\/:*?"<>|\u0000-\u001f]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 80)
    .replace(/[. ]+$/, '');
  return (base || 'project') + PROJECT_FILE_SUFFIX;
}

/** Position a position-free graph on the canvas: layer by longest-path depth
 * (top→bottom, matching the edge flow), siblings spread left→right. */
export function layoutGraph(
  specNodes: { id: string; type: string; params?: Record<string, unknown> }[],
  specEdges: string[][],
): EditorNode[] {
  const V_STEP = 112;
  const H_STEP = 212;
  const X0 = 70;
  const Y0 = 24;
  const ids = specNodes.map((n) => n.id);
  const idset = new Set(ids);
  const succ = new Map<string, string[]>(ids.map((id) => [id, []]));
  const indeg = new Map<string, number>(ids.map((id) => [id, 0]));
  for (const [s, t] of specEdges) {
    if (!idset.has(s) || !idset.has(t)) continue;
    succ.get(s)!.push(t);
    indeg.set(t, (indeg.get(t) ?? 0) + 1);
  }
  // Kahn topological order
  const deg = new Map(indeg);
  const queue = ids.filter((id) => deg.get(id) === 0);
  const order: string[] = [];
  for (let i = 0; i < queue.length; i++) {
    const n = queue[i];
    order.push(n);
    for (const m of succ.get(n)!) {
      deg.set(m, (deg.get(m) ?? 0) - 1);
      if (deg.get(m) === 0) queue.push(m);
    }
  }
  const topo = order.length === ids.length ? order : ids; // cycle fallback
  // longest-path depth from the sources
  const depth = new Map<string, number>(ids.map((id) => [id, 0]));
  for (const id of topo) {
    const d = depth.get(id)!;
    for (const m of succ.get(id)!) {
      if (depth.get(m)! < d + 1) depth.set(m, d + 1);
    }
  }
  // column within each depth, by spec insertion order (stable, few crossings)
  const colCount = new Map<number, number>();
  return specNodes.map((n) => {
    const d = depth.get(n.id)!;
    const col = colCount.get(d) ?? 0;
    colCount.set(d, col + 1);
    return { id: n.id, type: n.type, x: X0 + col * H_STEP, y: Y0 + d * V_STEP, params: structuredClone(n.params ?? {}) };
  });
}

// --------------------------------------------------------------------------- //
// Browser storage: the autosaved draft + a small library of saved projects
// --------------------------------------------------------------------------- //
export interface LocalProjectMeta {
  id: string;
  name: string;
  updated_at: string;
  node_count: number;
}

/** The autosaved editor state: the project plus where it came from. */
export interface AutosaveRecord {
  project: ProjectFile;
  origin: ProjectOrigin;
  localId: string | null;
  serverSlug: string | null;
  /** Fingerprint of the last explicitly saved / opened version (unsaved-changes flag). */
  savedFp: string | null;
  at: string;
}

const AUTOSAVE_KEY = 'mlc.autosave.v1';
const INDEX_KEY = 'mlc.projects.index.v1';
const itemKey = (id: string) => `mlc.projects.item.v1.${id}`;

/** localStorage-backed project storage. Every write may throw (quota exceeded,
 * storage disabled) — callers show that to the user. Reads never throw. */
export class ProjectStorage {
  constructor(private readonly store: Storage | null) {}

  static browser(): ProjectStorage {
    try {
      const s = globalThis.localStorage;
      s.getItem(INDEX_KEY); // throws where storage is blocked
      return new ProjectStorage(s);
    } catch {
      return new ProjectStorage(null);
    }
  }

  get available(): boolean {
    return this.store !== null;
  }

  readAutosave(): AutosaveRecord | null {
    const rec = this.readJson<AutosaveRecord>(AUTOSAVE_KEY);
    return rec && typeof rec === 'object' && rec.project ? rec : null;
  }

  writeAutosave(rec: AutosaveRecord): void {
    this.writeJson(AUTOSAVE_KEY, rec);
  }

  /** Saved projects, most recently saved first. */
  list(): LocalProjectMeta[] {
    const idx = this.readJson<LocalProjectMeta[]>(INDEX_KEY);
    if (!Array.isArray(idx)) return [];
    return idx
      .filter((m) => m && typeof m.id === 'string')
      .sort((a, b) => String(b.updated_at).localeCompare(String(a.updated_at)));
  }

  load(id: string): unknown {
    return this.readJson<unknown>(itemKey(id));
  }

  /** Save a project under `id` (a new id when null). Keeps the creation date of an
   * existing project. Returns the id and the project as stored. */
  save(id: string | null, project: ProjectFile, now = new Date().toISOString()): { id: string; project: ProjectFile } {
    if (!this.store) throw new Error('browser storage is not available');
    const pid = id ?? newLocalId();
    const isNew = this.store.getItem(itemKey(pid)) === null;
    const stored: ProjectFile = { ...project, created_at: project.created_at ?? now, updated_at: now };
    this.writeJson(itemKey(pid), stored);
    const idx = this.list().filter((m) => m.id !== pid);
    idx.unshift({ id: pid, name: stored.name, updated_at: now, node_count: stored.graph.nodes.length });
    try {
      this.writeJson(INDEX_KEY, idx);
    } catch (err) {
      if (isNew) this.store.removeItem(itemKey(pid)); // no orphan the list cannot show
      throw err;
    }
    return { id: pid, project: stored };
  }

  remove(id: string): void {
    if (!this.store) return;
    this.store.removeItem(itemKey(id));
    this.writeJson(
      INDEX_KEY,
      this.list().filter((m) => m.id !== id),
    );
  }

  private readJson<T>(key: string): T | null {
    if (!this.store) return null;
    try {
      const text = this.store.getItem(key);
      return text === null ? null : (JSON.parse(text) as T);
    } catch {
      return null;
    }
  }

  private writeJson(key: string, value: unknown): void {
    if (!this.store) throw new Error('browser storage is not available');
    this.store.setItem(key, JSON.stringify(value));
  }
}

function newLocalId(): string {
  return `p${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}
