/**
 * Stage 22 — the look of a layer: its family (colour + icon), the line of key settings
 * the detailed node view shows, how the inspector groups its parameters, and how deep
 * each node sits in the graph (the training wave runs top to bottom, then back up).
 * No Angular here — the editor and the unit tests use the same functions.
 */
import DATA from './families.json';
import { ParamDef } from './graph.models';

export interface Family {
  key: string;
  label: string;
  /** the symbol id in the editor's icon sprite, without the "i-" prefix */
  icon: string;
}

interface FamilyData {
  families: Record<string, { label: string; icon: string }>;
  categories: Record<string, string>;
  typeIcons: Record<string, string>;
  architectures: Record<string, { family: string; icon: string }>;
  advancedParams: string[];
}

const D = DATA as unknown as FamilyData;

/** Every family, in the palette's order. */
export const FAMILIES: readonly Family[] = Object.entries(D.families).map(([key, f]) => ({ key, ...f }));
const BY_KEY = new Map(FAMILIES.map((f) => [f.key, f]));
const OTHER = BY_KEY.get('other')!;
const ADVANCED = new Set(D.advancedParams);

/** The family of a layer category ("conv" → Convolution); unknown categories are "other". */
export function familyOf(category: string | null | undefined): Family {
  const key = category ? D.categories[category] : undefined;
  return (key && BY_KEY.get(key)) || OTHER;
}

/** A node's icon: its own for a few types (Input, Output…), else its family's. */
export function iconOf(type: string, category: string | null | undefined): string {
  return D.typeIcons[type] ?? familyOf(category).icon;
}

/** A library entry's family and icon, by the architecture's category. */
export function archLook(category: string | null | undefined): { family: string; icon: string } {
  const a = category ? D.architectures[category] : undefined;
  return a ?? { family: OTHER.key, icon: OTHER.icon };
}

/** Rarely touched settings (eps, rope theta…): the inspector folds them under "Advanced". */
export function isAdvanced(name: string): boolean {
  return ADVANCED.has(name);
}

/** The inspector's groups: settings first, switches next, the advanced ones folded. */
export function groupParams(defs: readonly ParamDef[]): { main: ParamDef[]; options: ParamDef[]; advanced: ParamDef[] } {
  const main: ParamDef[] = [];
  const options: ParamDef[] = [];
  const advanced: ParamDef[] = [];
  for (const d of defs) {
    if (isAdvanced(d.name)) advanced.push(d);
    else if (d.type === 'bool') options.push(d);
    else main.push(d);
  }
  return { main, options, advanced };
}

/** Short names for the detailed view's line of settings. */
const SHORT: Record<string, string> = {
  out_channels: 'ch',
  channels: 'ch',
  block_channels: 'ch',
  latent_channels: 'latent',
  kernel_size: 'k',
  stride: 's',
  padding: 'pad',
  out_features: 'out',
  output_size: 'out',
  hidden_size: 'hidden',
  hidden: 'hidden',
  num_layers: 'layers',
  layers_per_block: 'layers',
  num_heads: 'heads',
  nhead: 'heads',
  heads: 'heads',
  kv_heads: 'kv',
  head_dim: 'head',
  embedding_dim: 'dim',
  num_embeddings: 'vocab',
  vocab: 'vocab',
  dim_feedforward: 'ff',
  expert_hidden: 'expert',
  shared_experts: 'shared',
  top_k: 'top',
  patch_size: 'patch',
  max_len: 'len',
  length: 'len',
  num_groups: 'groups',
  dropout: 'drop',
  repeat: '×',
  txt_len: 'text',
  mlp_ratio: 'mlp',
  v_dim: 'v',
  negative_slope: 'slope',
  kl_weight: 'kl',
};

/** A number as the node shows it: 1e-5, 0.001, 50000. */
export function fmtNum(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  if (v !== 0 && Math.abs(v) < 1e-3) return Number(v.toPrecision(3)).toExponential();
  return String(Number(v.toPrecision(6)));
}

/** The detailed view's line: a node's key settings ("ch 32 · k 3 · s 1"), at most `max` of them. */
export function keyParams(params: Record<string, unknown>, defs: readonly ParamDef[] | undefined, max = 4): string {
  const parts: string[] = [];
  const flags: string[] = [];
  for (const d of defs ?? []) {
    if (isAdvanced(d.name) || d.name === 'name' || d.name === 'layers') continue;
    const v = params[d.name] ?? d.default;
    if (d.type === 'bool') {
      if (v === true) flags.push((d.label || d.name).replace(/_/g, ' ').toLowerCase());
      continue;
    }
    if (v == null || v === '' || (Array.isArray(v) && !v.length)) continue;
    const val = Array.isArray(v)
      ? v.join(d.name === 'shape' ? '×' : ',')
      : typeof v === 'number'
        ? fmtNum(v)
        : String(v);
    const label = SHORT[d.name] ?? d.name.replace(/_/g, ' ');
    parts.push(label === '×' ? `×${val}` : `${label} ${val}`);
  }
  return [...parts, ...flags].slice(0, max).join(' · ');
}

/**
 * How deep each node sits: the longest path from a node without inputs (0) — the order
 * a batch flows in. A loop (the backend reports it as a problem) leaves its nodes at 0.
 */
export function graphDepths(ids: readonly string[], edges: readonly (readonly [string, string])[]): Map<string, number> {
  const known = new Set(ids);
  const succ = new Map<string, string[]>(ids.map((id) => [id, []]));
  const indeg = new Map<string, number>(ids.map((id) => [id, 0]));
  for (const [s, t] of edges) {
    if (!known.has(s) || !known.has(t) || s === t) continue;
    succ.get(s)!.push(t);
    indeg.set(t, indeg.get(t)! + 1);
  }
  const depth = new Map<string, number>(ids.map((id) => [id, 0]));
  const queue = ids.filter((id) => indeg.get(id) === 0);
  for (let i = 0; i < queue.length; i++) {
    const n = queue[i];
    for (const m of succ.get(n)!) {
      depth.set(m, Math.max(depth.get(m)!, depth.get(n)! + 1));
      indeg.set(m, indeg.get(m)! - 1);
      if (indeg.get(m) === 0) queue.push(m);
    }
  }
  return depth;
}

/** Each node's place in the training wave: 0 at the inputs, 1 at the deepest node. */
export function waveOf(depths: ReadonlyMap<string, number>): Map<string, number> {
  const max = Math.max(0, ...depths.values());
  const out = new Map<string, number>();
  for (const [id, d] of depths) out.set(id, max ? Math.round((d / max) * 1000) / 1000 : 0);
  return out;
}

/** The palette's groups: layers of each family together, families in their fixed order. */
export function groupByFamily<T extends { category: string }>(layers: readonly T[]): { family: Family; layers: T[] }[] {
  const groups = new Map<string, T[]>();
  for (const l of layers) {
    const key = familyOf(l.category).key;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(l);
  }
  return FAMILIES.filter((f) => groups.has(f.key)).map((family) => ({ family, layers: groups.get(family.key)! }));
}
