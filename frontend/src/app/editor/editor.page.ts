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
  ArchitectureInfo,
  BuildResult,
  DataPreview,
  DataSampleRow,
  DatasetInfo,
  EditorEdge,
  EditorNode,
  GraphSpec,
  LayerInfo,
  ParamDef,
  TrainEvent,
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

  // ---- architecture library (Stage 7) ----
  architectures = signal<ArchitectureInfo[]>([]);

  // ---- datasets, data preview & export (Stage 8) ----
  // `stage8` turns the new UI on only once the backend answers /datasets, so the
  // live app keeps working against an older backend (features just stay hidden).
  stage8 = signal(false);
  datasets = signal<DatasetInfo[]>([]);
  dataset = signal<string>('synthetic');
  dataPreview = signal<DataPreview | null>(null);
  dataModalOpen = signal(false);
  exportCode = signal<string>('');
  exportModalOpen = signal(false);
  exportCopied = signal(false);
  overlayBusy = signal(false);
  overlayError = signal<string | null>(null);

  // ---- custom data: bring your own rows (Stage 9) ----
  customText = signal<string>('');
  customRows = signal<number[][]>([]);
  customError = signal<string | null>(null);

  // ---- live training + checkpoints (Stage 11) ----
  liveLoss = signal<number[]>([]);
  liveAcc = signal<number[]>([]);
  saveCheckpoint = signal(false);
  resumeCheckpoint = signal(false);

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
  innerSel = signal<number | null>(null); // selected inner layer of a Block (Repeat) node

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
    this.api.architectures().subscribe({
      next: (a) => this.architectures.set(a),
      error: () => this.architectures.set([]), // non-fatal: the templates menu just stays hidden
    });
    this.api.datasets().subscribe({
      next: (ds) => {
        this.datasets.set(ds);
        if (ds.length) {
          this.dataset.set(ds[0].id);
          this.stage8.set(true); // backend supports Stage 8 — show the dataset picker, Data & Export
        }
      },
      error: () => this.stage8.set(false), // older backend: hide the Stage 8 UI, app still works
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

  // ---- Block (Repeat) node: the inner linear chain it stamps N times ----
  blockLayers = computed<{ type: string; params: Record<string, unknown> }[]>(() => {
    const n = this.selectedNode();
    return n?.type === 'Repeat' ? ((n.params['layers'] as { type: string; params: Record<string, unknown> }[]) ?? []) : [];
  });

  // Layer types allowed inside a block: no Input/Output (io), no merges, no nested blocks.
  innerTypes = computed<string[]>(() =>
    this.layers()
      .filter((l) => l.category !== 'io' && l.category !== 'merge' && l.type !== 'Repeat')
      .map((l) => l.type),
  );

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

  // Live training curves (Stage 11): prefer the streamed arrays, else the final result.
  curveLoss = computed(() => {
    const live = this.liveLoss();
    return live.length ? live : (this.trainResult()?.loss_curve ?? []);
  });
  curveAcc = computed(() => {
    const live = this.liveAcc();
    return live.length ? live : (this.trainResult()?.acc_curve ?? []);
  });

  lossPoints = computed(() => this.pointsOf(this.curveLoss(), false));
  // Accuracy on a fixed 0..1 axis so the height reads as accuracy.
  accPoints = computed(() => this.pointsOf(this.curveAcc(), true));

  private pointsOf(c: number[], fixed01: boolean): string {
    if (c.length === 0) return '';
    const W = 220;
    const H = 60;
    const pad = 4;
    const max = fixed01 ? 1 : Math.max(...c);
    const min = fixed01 ? 0 : Math.min(...c);
    const span = max - min || 1;
    const n = c.length - 1 || 1;
    return c
      .map((v, i) => {
        const x = pad + (i / n) * (W - 2 * pad);
        const y = pad + (1 - (Math.min(max, Math.max(min, v)) - min) / span) * (H - 2 * pad);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  }

  // ---------------------------------------------------------------- palette
  addNode(layer: LayerInfo): void {
    const id = this.freshId(layer.type);
    const params: Record<string, unknown> = {};
    for (const p of layer.params) params[p.name] = structuredClone(p.default);
    if (layer.type === 'Repeat' && !Array.isArray(params['layers'])) params['layers'] = []; // block starts empty
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
        `«${target?.type}» takes a single input. To combine branches, use a merge node (Add or Concat).`,
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
  private coerceParam(p: ParamDef, raw: unknown): unknown {
    if (p.type === 'int') return Math.round(Number(raw));
    if (p.type === 'float') return Number(raw);
    if (p.type === 'bool') return !!raw;
    if (p.type === 'intlist') {
      return String(raw)
        .split(',')
        .map((s) => parseInt(s.trim(), 10))
        .filter((x) => !Number.isNaN(x));
    }
    return String(raw);
  }

  setParam(nodeId: string, p: ParamDef, raw: unknown): void {
    const val = this.coerceParam(p, raw);
    this.nodes.update((ns) => ns.map((n) => (n.id === nodeId ? { ...n, params: { ...n.params, [p.name]: val } } : n)));
    this.invalidateBuild();
  }

  // -------------------------------------------------------- block inner chain
  nodeTitle(n: EditorNode): string {
    return n.type === 'Repeat' ? `Block ×${n.params['repeat'] ?? 1}` : n.type;
  }

  innerSummary(n: EditorNode): string {
    const ls = (n.params['layers'] as { type: string }[]) ?? [];
    return ls.length ? ls.map((l) => l.type).join(' → ') : '(empty)';
  }

  innerInfo(type: string): LayerInfo | null {
    return this.layers().find((l) => l.type === type) ?? null;
  }

  addInner(type: string): void {
    if (!type) return;
    const info = this.innerInfo(type);
    const params: Record<string, unknown> = {};
    if (info) for (const p of info.params) params[p.name] = structuredClone(p.default);
    const blkId = this.selectedId();
    this.nodes.update((ns) =>
      ns.map((n) => {
        if (n.id !== blkId) return n;
        const layers = [...((n.params['layers'] as unknown[]) ?? []), { type, params }];
        return { ...n, params: { ...n.params, layers } };
      }),
    );
    this.innerSel.set(this.blockLayers().length - 1);
    this.invalidateBuild();
  }

  removeInner(idx: number): void {
    const blkId = this.selectedId();
    this.nodes.update((ns) =>
      ns.map((n) => {
        if (n.id !== blkId) return n;
        const layers = ((n.params['layers'] as unknown[]) ?? []).filter((_, i) => i !== idx);
        return { ...n, params: { ...n.params, layers } };
      }),
    );
    this.innerSel.set(null);
    this.invalidateBuild();
  }

  setInnerParam(idx: number, p: ParamDef, raw: unknown): void {
    const val = this.coerceParam(p, raw);
    const blkId = this.selectedId();
    this.nodes.update((ns) =>
      ns.map((n) => {
        if (n.id !== blkId) return n;
        const layers = [...((n.params['layers'] as { type: string; params: Record<string, unknown> }[]) ?? [])];
        if (!layers[idx]) return n;
        layers[idx] = { ...layers[idx], params: { ...layers[idx].params, [p.name]: val } };
        return { ...n, params: { ...n.params, layers } };
      }),
    );
    this.invalidateBuild();
  }

  paramDisplay(n: EditorNode, p: ParamDef): string | number {
    const v = n.params[p.name];
    if (p.type === 'intlist') return Array.isArray(v) ? v.join(', ') : String(v ?? '');
    return v as string | number;
  }

  // ------------------------------------------------------------------- run
  private toSpec(): GraphSpec {
    const train: GraphSpec['train'] = {
      dataset: this.dataset(),
      epochs: this.epochs(),
      lr: this.lr(),
      batch_size: this.batchSize(),
      samples: this.samples(),
    };
    if (this.dataset() === 'custom') train.custom_rows = this.customRows();
    if (this.saveCheckpoint()) train.checkpoint = true;
    if (this.resumeCheckpoint()) train.resume = true;
    return {
      nodes: this.nodes().map((n) => ({ id: n.id, type: n.type, params: n.params })),
      edges: this.edges().map((e) => [e.source, e.target]),
      train,
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
    if (this.dataset() === 'custom' && !this.customRows().length) {
      this.runError.set('Select some data first: open “Data”, paste your rows, and Apply.');
      return;
    }
    this.busy.set(true);
    this.runError.set(null);
    this.proposals.set([]);
    this.trainResult.set(null);
    this.liveLoss.set([]);
    this.liveAcc.set([]);
    const spec = this.toSpec();
    this.api
      .trainStream(spec, (e) => {
        if (e.type === 'start') {
          this.liveLoss.set([]);
          this.liveAcc.set([]);
        } else if (e.type === 'step' && e.loss != null) {
          this.liveLoss.update((a) => [...a, e.loss as number]);
        } else if (e.type === 'epoch' && e.acc != null) {
          this.liveAcc.update((a) => [...a, e.acc as number]);
        } else if (e.type === 'done' && e.result) {
          this.trainResult.set(e.result);
          this.busy.set(false);
        } else if (e.type === 'error') {
          this.runError.set(e.detail ?? 'Training failed');
          this.busy.set(false);
        }
      })
      .then(() => {
        if (this.busy() && !this.trainResult()) this.busy.set(false); // stream ended without a result
      })
      .catch((err) => {
        if (err?.fallback) {
          // older backend without /train/stream — fall back to the plain endpoint
          this.api.train(spec).subscribe({
            next: (r) => {
              this.trainResult.set(r);
              this.busy.set(false);
            },
            error: (e) => {
              this.runError.set(this.errMsg(e));
              this.busy.set(false);
            },
          });
        } else {
          this.runError.set(err?.detail || this.errMsg(err));
          this.busy.set(false);
        }
      });
  }

  // ------------------------------------------------ data & export (Stage 8)
  setDataset(id: string): void {
    if (!id) return;
    this.dataset.set(id);
    this.trainResult.set(null);
  }

  datasetInfo(id: string) {
    return this.datasets().find((d) => d.id === id) ?? null;
  }

  rowValues(r: DataSampleRow): string {
    const base = r.values.join(', ');
    return r.truncated ? `[${base}, … +${r.total - r.values.length} more]` : `[${base}]`;
  }

  /** Show the format and a few real rows of the training set for this graph.
   * For the "custom" dataset this opens the paste/upload editor instead; the
   * preview then runs once rows are applied. */
  openData(): void {
    this.dataModalOpen.set(true);
    this.overlayError.set(null);
    this.dataPreview.set(null);
    if (this.dataset() === 'custom') {
      this.overlayBusy.set(false);
      if (this.customRows().length) this.previewData();
      return;
    }
    this.previewData();
  }

  private previewData(): void {
    this.overlayBusy.set(true);
    this.overlayError.set(null);
    this.dataPreview.set(null);
    this.api.dataPreview(this.toSpec(), 6).subscribe({
      next: (p) => {
        this.dataPreview.set(p);
        this.overlayBusy.set(false);
      },
      error: (e) => {
        this.overlayError.set(this.errMsg(e));
        this.overlayBusy.set(false);
      },
    });
  }

  /** Parse the pasted/uploaded text into numeric rows (features…, then a label)
   * and preview them (the backend validates against the graph). */
  applyCustom(): void {
    const lines = this.customText()
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter((l) => l && !l.startsWith('#'));
    if (!lines.length) {
      this.customError.set('Paste some rows first — one example per line.');
      return;
    }
    const rows: number[][] = [];
    for (let i = 0; i < lines.length; i++) {
      const nums = lines[i].split(/[,\s]+/).filter((s) => s.length).map(Number);
      if (nums.some((n) => Number.isNaN(n))) {
        this.customError.set(`Line ${i + 1}: not all values are numbers.`);
        return;
      }
      if (nums.length < 2) {
        this.customError.set(`Line ${i + 1}: need at least one feature and a label.`);
        return;
      }
      rows.push(nums);
    }
    this.customError.set(null);
    this.customRows.set(rows);
    this.previewData(); // backend validates the rows against the graph and echoes them
  }

  onCustomFile(ev: Event): void {
    const input = ev.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      this.customText.set(String(reader.result ?? ''));
      this.applyCustom();
    };
    reader.readAsText(file);
    input.value = '';
  }

  closeData(): void {
    this.dataModalOpen.set(false);
  }

  /** Generate standalone PyTorch source for this graph and show it. */
  openExport(): void {
    this.exportModalOpen.set(true);
    this.overlayError.set(null);
    this.overlayBusy.set(true);
    this.exportCode.set('');
    this.exportCopied.set(false);
    this.api.exportPytorch(this.toSpec(), true).subscribe({
      next: (r) => {
        this.exportCode.set(r.code);
        this.overlayBusy.set(false);
      },
      error: (e) => {
        this.overlayError.set(this.errMsg(e));
        this.overlayBusy.set(false);
      },
    });
  }

  closeExport(): void {
    this.exportModalOpen.set(false);
  }

  copyCode(): void {
    const code = this.exportCode();
    if (!code) return;
    navigator.clipboard?.writeText(code).then(
      () => {
        this.exportCopied.set(true);
        setTimeout(() => this.exportCopied.set(false), 1500);
      },
      () => this.overlayError.set('Copy failed — select the text and copy it manually.'),
    );
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
    if (err?.status === 0) return 'No connection to the backend on :8050 — is FastAPI running?';
    return err?.message ?? 'Request failed';
  }

  // ------------------------------------------------ architecture library (S7)
  /** Load a ready architecture: fetch its full graph, lay it out on the canvas,
   * and adopt its recommended training config. The nodes are ordinary, editable
   * layer nodes — the user can re-wire and tune every one. */
  loadArchitecture(id: string): void {
    this.busy.set(true);
    this.runError.set(null);
    this.proposals.set([]);
    this.api.architecture(id).subscribe({
      next: (spec) => {
        this.nodes.set(this.layoutGraph(spec.nodes, spec.edges));
        this.edges.set(spec.edges.map(([source, target]) => ({ source, target })));
        this.selectedId.set(null);
        this.innerSel.set(null);
        this.connectFrom.set(null);
        this.seq = spec.nodes.length;
        const t = spec.train ?? {};
        if (t.epochs != null) this.epochs.set(t.epochs);
        if (t.lr != null) this.lr.set(t.lr);
        if (t.batch_size != null) this.batchSize.set(t.batch_size);
        if (t.samples != null) this.samples.set(t.samples);
        this.pan.set({ x: 48, y: 24 });
        this.zoom.set(0.8);
        this.buildResult.set(null);
        this.trainResult.set(null);
        this.busy.set(false);
      },
      error: (e) => {
        this.runError.set(this.errMsg(e));
        this.busy.set(false);
      },
    });
  }

  /** Position a position-free backend graph on the canvas: layer by longest-path
   * depth (top→bottom, matching the edge flow), siblings spread left→right. */
  private layoutGraph(
    specNodes: { id: string; type: string; params: Record<string, unknown> }[],
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
      return { id: n.id, type: n.type, x: X0 + col * H_STEP, y: Y0 + d * V_STEP, params: structuredClone(n.params) };
    });
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
