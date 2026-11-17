import { CommonModule } from '@angular/common';
import { Component, HostListener, computed, inject, signal } from '@angular/core';
import {
  IonBadge,
  IonButton,
  IonButtons,
  IonCheckbox,
  IonContent,
  IonHeader,
  IonInput,
  IonItem,
  IonLabel,
  IonList,
  IonNote,
  IonSearchbar,
  IonTitle,
  IonToolbar,
} from '@ionic/angular';

import { ApiService } from '../core/api.service';
import {
  AdapterProposal,
  BuildResult,
  EditorEdge,
  EditorNode,
  GraphSpec,
  LayerInfo,
  ParamDef,
  TrainResult,
} from '../core/graph.models';

const NODE_W = 174;
const NODE_H = 78;

@Component({
  selector: 'app-editor',
  standalone: true,
  imports: [
    CommonModule,
    IonHeader,
    IonToolbar,
    IonTitle,
    IonButtons,
    IonButton,
    IonContent,
    IonList,
    IonItem,
    IonLabel,
    IonInput,
    IonCheckbox,
    IonBadge,
    IonNote,
    IonSearchbar,
  ],
  templateUrl: './editor.page.html',
  styleUrl: './editor.page.scss',
})
export class EditorPage {
  private api = inject(ApiService);

  readonly NODE_W = NODE_W;
  readonly NODE_H = NODE_H;

  // ---- palette ----
  layers = signal<LayerInfo[]>([]);
  layersError = signal<string | null>(null);
  paletteQuery = signal('');

  // ---- graph ----
  nodes = signal<EditorNode[]>([]);
  edges = signal<EditorEdge[]>([]);
  selectedId = signal<string | null>(null);
  connectFrom = signal<string | null>(null);

  // ---- view ----
  pan = signal<{ x: number; y: number }>({ x: 48, y: 36 });
  zoom = signal(1);

  // ---- run ----
  busy = signal(false);
  buildResult = signal<BuildResult | null>(null);
  trainResult = signal<TrainResult | null>(null);
  runError = signal<string | null>(null);
  proposals = signal<AdapterProposal[]>([]);

  // ---- train config ----
  epochs = signal(3);
  lr = signal(0.01);
  batchSize = signal(64);
  samples = signal(512);

  // ---- interaction internals ----
  private mode: 'idle' | 'drag' | 'pan' = 'idle';
  private dragId: string | null = null;
  private lastX = 0;
  private lastY = 0;
  private seq = 0;

  constructor() {
    this.api.layers().subscribe({
      next: (ls) => this.layers.set(ls),
      error: (e) => this.layersError.set(this.errMsg(e)),
    });
    this.seedExample();
  }

  // ---------------------------------------------------------------- derived
  selectedNode = computed(() => this.nodes().find((n) => n.id === this.selectedId()) ?? null);

  selectedInfo = computed<LayerInfo | null>(() => {
    const n = this.selectedNode();
    return n ? this.layers().find((l) => l.type === n.type) ?? null : null;
  });

  // Palette filtered by the search box (by label, type or category) — the
  // catalogue is now 29 layers, so a filter keeps it navigable.
  filteredLayers = computed<LayerInfo[]>(() => {
    const q = this.paletteQuery().trim().toLowerCase();
    const ls = this.layers();
    if (!q) return ls;
    return ls.filter(
      (l) =>
        l.label.toLowerCase().includes(q) ||
        l.type.toLowerCase().includes(q) ||
        l.category.toLowerCase().includes(q),
    );
  });

  shapeById = computed(() => {
    const map = new Map<string, number[] | null>();
    const b = this.buildResult();
    if (b) for (const s of b.layers) map.set(s.id, s.out_shape);
    return map;
  });

  edgeViews = computed(() => {
    const byId = new Map(this.nodes().map((n) => [n.id, n]));
    const views: { id: string; d: string }[] = [];
    for (const e of this.edges()) {
      const s = byId.get(e.source);
      const t = byId.get(e.target);
      if (!s || !t) continue;
      const x1 = s.x + NODE_W / 2;
      const y1 = s.y + NODE_H;
      const x2 = t.x + NODE_W / 2;
      const y2 = t.y;
      const k = Math.max(36, Math.abs(y2 - y1) / 2);
      views.push({ id: `${e.source}->${e.target}`, d: `M ${x1} ${y1} C ${x1} ${y1 + k}, ${x2} ${y2 - k}, ${x2} ${y2}` });
    }
    return views;
  });

  canvasTransform = computed(() => {
    const p = this.pan();
    return `translate(${p.x}px, ${p.y}px) scale(${this.zoom()})`;
  });

  lossPoints = computed(() => {
    const r = this.trainResult();
    if (!r || r.loss_curve.length === 0) return '';
    const c = r.loss_curve;
    const W = 220;
    const H = 60;
    const pad = 4;
    const max = Math.max(...c);
    const min = Math.min(...c);
    const span = max - min || 1;
    const n = c.length - 1 || 1;
    return c
      .map((v, i) => {
        const x = pad + (i / n) * (W - 2 * pad);
        const y = pad + (1 - (v - min) / span) * (H - 2 * pad);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  });

  // ---------------------------------------------------------------- palette
  addNode(layer: LayerInfo): void {
    const id = this.freshId(layer.type);
    const params: Record<string, unknown> = {};
    for (const p of layer.params) params[p.name] = structuredClone(p.default);
    const i = this.nodes().length;
    const node: EditorNode = { id, type: layer.type, x: 70 + (i % 4) * 26, y: 60 + i * 22, params };
    this.nodes.update((ns) => [...ns, node]);
    this.selectedId.set(id);
    this.invalidateBuild();
  }

  private freshId(type: string): string {
    this.seq += 1;
    return `${type.toLowerCase()}_${this.seq}`;
  }

  // -------------------------------------------------------------- selection
  select(id: string): void {
    this.selectedId.set(id);
  }

  deleteSelected(): void {
    const id = this.selectedId();
    if (!id) return;
    this.nodes.update((ns) => ns.filter((n) => n.id !== id));
    this.edges.update((es) => es.filter((e) => e.source !== id && e.target !== id));
    this.selectedId.set(null);
    this.invalidateBuild();
  }

  clearGraph(): void {
    this.nodes.set([]);
    this.edges.set([]);
    this.selectedId.set(null);
    this.connectFrom.set(null);
    this.invalidateBuild();
    this.trainResult.set(null);
    this.runError.set(null);
  }

  // ---------------------------------------------------------------- connect
  startConnect(id: string, ev: Event): void {
    ev.stopPropagation();
    this.connectFrom.set(id);
  }

  finishConnect(targetId: string, ev: Event): void {
    ev.stopPropagation();
    const src = this.connectFrom();
    this.connectFrom.set(null);
    if (!src || src === targetId) return;
    if (this.edges().some((e) => e.source === src && e.target === targetId)) return;
    // A non-merge node accepts exactly one input; merge nodes (Add/Concat) take many.
    const target = this.nodes().find((n) => n.id === targetId);
    const info = target ? this.layers().find((l) => l.type === target.type) : null;
    const isMerge = info?.category === 'merge';
    const hasInput = this.edges().some((e) => e.target === targetId);
    if (hasInput && !isMerge) {
      this.runError.set(
        `«${target?.type}» принимает один вход. Чтобы объединить ветки, используйте merge-ноду (Add или Concat).`,
      );
      return;
    }
    this.edges.update((es) => [...es, { source: src, target: targetId }]);
    this.invalidateBuild();
  }

  cancelConnect(): void {
    this.connectFrom.set(null);
  }

  // -------------------------------------------------------------- dragging
  nodePointerDown(id: string, ev: PointerEvent): void {
    ev.stopPropagation();
    this.select(id);
    this.mode = 'drag';
    this.dragId = id;
    this.lastX = ev.clientX;
    this.lastY = ev.clientY;
  }

  canvasDown(ev: PointerEvent): void {
    this.mode = 'pan';
    this.lastX = ev.clientX;
    this.lastY = ev.clientY;
    this.selectedId.set(null);
    this.cancelConnect();
  }

  @HostListener('document:pointermove', ['$event'])
  onPointerMove(ev: PointerEvent): void {
    if (this.mode === 'idle') return;
    const dx = ev.clientX - this.lastX;
    const dy = ev.clientY - this.lastY;
    this.lastX = ev.clientX;
    this.lastY = ev.clientY;
    if (this.mode === 'drag' && this.dragId) {
      const z = this.zoom();
      const id = this.dragId;
      this.nodes.update((ns) => ns.map((n) => (n.id === id ? { ...n, x: n.x + dx / z, y: n.y + dy / z } : n)));
    } else if (this.mode === 'pan') {
      this.pan.update((p) => ({ x: p.x + dx, y: p.y + dy }));
    }
  }

  @HostListener('document:pointerup')
  onPointerUp(): void {
    this.mode = 'idle';
    this.dragId = null;
  }

  onWheel(ev: WheelEvent): void {
    ev.preventDefault();
    const factor = ev.deltaY < 0 ? 1.1 : 1 / 1.1;
    this.zoom.update((z) => Math.min(2.5, Math.max(0.3, z * factor)));
  }

  // ---------------------------------------------------------------- params
  setParam(nodeId: string, p: ParamDef, raw: unknown): void {
    let val: unknown = raw;
    if (p.type === 'int') val = Math.round(Number(raw));
    else if (p.type === 'float') val = Number(raw);
    else if (p.type === 'bool') val = !!raw;
    else if (p.type === 'intlist') {
      val = String(raw)
        .split(',')
        .map((s) => parseInt(s.trim(), 10))
        .filter((x) => !Number.isNaN(x));
    } else val = String(raw);
    this.nodes.update((ns) => ns.map((n) => (n.id === nodeId ? { ...n, params: { ...n.params, [p.name]: val } } : n)));
    this.invalidateBuild();
  }

  paramDisplay(n: EditorNode, p: ParamDef): string | number {
    const v = n.params[p.name];
    if (p.type === 'intlist') return Array.isArray(v) ? v.join(', ') : String(v ?? '');
    return v as string | number;
  }

  // ------------------------------------------------------------------- run
  private toSpec(): GraphSpec {
    return {
      nodes: this.nodes().map((n) => ({ id: n.id, type: n.type, params: n.params })),
      edges: this.edges().map((e) => [e.source, e.target]),
      train: {
        dataset: 'synthetic',
        epochs: this.epochs(),
        lr: this.lr(),
        batch_size: this.batchSize(),
        samples: this.samples(),
      },
    };
  }

  onBuild(): void {
    this.busy.set(true);
    this.runError.set(null);
    this.proposals.set([]);
    this.trainResult.set(null);
    const spec = this.toSpec();
    this.api.build(spec).subscribe({
      next: (r) => {
        this.buildResult.set(r);
        this.busy.set(false);
      },
      error: (e) => {
        this.buildResult.set(null);
        this.runError.set(this.errMsg(e));
        this.busy.set(false);
        this.fetchSuggestions(spec); // offer a one-click adapter if a shape mismatch is fixable
      },
    });
  }

  onTrain(): void {
    this.busy.set(true);
    this.runError.set(null);
    this.proposals.set([]);
    this.api.train(this.toSpec()).subscribe({
      next: (r) => {
        this.trainResult.set(r);
        this.busy.set(false);
      },
      error: (e) => {
        this.runError.set(this.errMsg(e));
        this.busy.set(false);
      },
    });
  }

  // ---------------------------------------------------------- smart adapters
  private fetchSuggestions(spec: GraphSpec): void {
    this.api.suggest(spec).subscribe({
      next: (s) => this.proposals.set(s.proposals ?? []),
      error: () => this.proposals.set([]),
    });
  }

  /** Insert the proposed adapter node on its edge and rebuild (one-click fix). */
  applyProposal(p: AdapterProposal): void {
    const id = this.freshId(p.adapter_type);
    const src = this.nodes().find((n) => n.id === p.source_id);
    const tgt = this.nodes().find((n) => n.id === p.target_id);
    const x = src && tgt ? Math.round((src.x + tgt.x) / 2) : 90;
    const y = src && tgt ? Math.round((src.y + tgt.y) / 2) : 90;
    const node: EditorNode = { id, type: p.adapter_type, x, y, params: { ...p.params } };
    this.nodes.update((ns) => [...ns, node]);
    this.edges.update((es) => [
      ...es.filter((e) => !(e.source === p.source_id && e.target === p.target_id)),
      { source: p.source_id, target: id },
      { source: id, target: p.target_id },
    ]);
    this.proposals.set([]);
    this.runError.set(null);
    this.onBuild();
  }

  private invalidateBuild(): void {
    this.buildResult.set(null);
    this.proposals.set([]);
  }

  private errMsg(e: unknown): string {
    const err = e as { error?: { detail?: string }; message?: string; status?: number };
    if (err?.error?.detail) return err.error.detail;
    if (err?.status === 0) return 'Нет связи с бэкендом на :8050 — запущен ли FastAPI?';
    return err?.message ?? 'Request failed';
  }

  // -------------------------------------------------------------- example
  private seedExample(): void {
    const mk = (id: string, type: string, x: number, y: number, params: Record<string, unknown> = {}): EditorNode => ({
      id,
      type,
      x,
      y,
      params,
    });
    this.nodes.set([
      mk('input_1', 'Input', 70, 24, { shape: [1, 28, 28] }),
      mk('flatten_2', 'Flatten', 70, 136, {}),
      mk('linear_3', 'Linear', 70, 248, { out_features: 64, bias: true }),
      mk('relu_4', 'ReLU', 70, 360, {}),
      mk('linear_5', 'Linear', 70, 472, { out_features: 10, bias: true }),
      mk('output_6', 'Output', 70, 584, {}),
    ]);
    this.edges.set([
      { source: 'input_1', target: 'flatten_2' },
      { source: 'flatten_2', target: 'linear_3' },
      { source: 'linear_3', target: 'relu_4' },
      { source: 'relu_4', target: 'linear_5' },
      { source: 'linear_5', target: 'output_6' },
    ]);
    this.seq = 6;
  }
}
