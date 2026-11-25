import { CommonModule } from '@angular/common';
import { Component, HostListener, computed, effect, inject, signal } from '@angular/core';
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
  ProjectLoadResult,
  ProjectSummary,
  TrainCfg,
  TrainEvent,
  TrainResult,
} from '../core/graph.models';
import {
  DEFAULT_VIEW,
  EditorSnapshot,
  LocalProjectMeta,
  MAX_PROJECT_BYTES,
  ProjectOrigin,
  ProjectStorage,
  UI_TRAIN_KEYS,
  fingerprint,
  fromProjectFile,
  layoutGraph,
  projectFileName,
  readCurrentFormat,
  toProjectFile,
} from '../core/project';

const NODE_W = 174;
const NODE_H = 78;
const AUTOSAVE_DELAY_MS = 700;
const ORIGINS: readonly ProjectOrigin[] = ['new', 'browser', 'server', 'file'];

interface Notice {
  kind: 'info' | 'warn' | 'error';
  text: string;
  details: string[];
}

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
  styleUrls: ['./editor.page.scss', './projects.scss'],
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

  // ---- GPU as an explicit opt-in (Stage 12) ----
  cudaAvailable = signal(false); // from /health — the Device picker only shows when true
  device = signal<'cpu' | 'cuda' | 'auto'>('cpu');

  // ---- projects: save / open / autosave (Stage 13) ----
  projectName = signal('Untitled');
  projectDesc = signal('');
  private projectCreated = signal<string | null>(null);
  /** Training settings without an editor control (optimizer, seed, …), kept as loaded. */
  private trainExtra = signal<Record<string, unknown>>({});
  origin = signal<ProjectOrigin>('new');
  localId = signal<string | null>(null);
  serverSlug = signal<string | null>(null);
  private savedFp = signal<string | null>(null);
  dirty = signal(false);
  lastAutosave = signal<Date | null>(null);
  autosaveError = signal<string | null>(null);
  projectsOpen = signal(false);
  localProjects = signal<LocalProjectMeta[]>([]);
  serverEnabled = signal(false); // from /health — the server section only shows when true
  serverProjects = signal<ProjectSummary[]>([]);
  serverError = signal<string | null>(null);
  serverBusy = signal(false);
  notice = signal<Notice | null>(null);
  dropHover = signal(false);
  private store = ProjectStorage.browser();
  private autosaveTimer: ReturnType<typeof setTimeout> | null = null;
  private autosaveArmed = false;
  private noticeTimer: ReturnType<typeof setTimeout> | null = null;

  // ---- graph ----
  nodes = signal<EditorNode[]>([]);
  edges = signal<EditorEdge[]>([]);
  selectedId = signal<string | null>(null);
  connectFrom = signal<string | null>(null);

  // ---- view ----
  pan = signal<{ x: number; y: number }>({ x: DEFAULT_VIEW.pan_x, y: DEFAULT_VIEW.pan_y });
  zoom = signal(DEFAULT_VIEW.zoom);

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
          // Keep a restored project's dataset; otherwise start on the first one. That
          // swap is housekeeping, not an edit, so a clean project stays clean.
          if (!ds.some((d) => d.id === this.dataset())) {
            const wasClean = !this.isDirtyNow();
            this.dataset.set(ds[0].id);
            if (wasClean) this.savedFp.set(this.currentFp());
          }
          this.stage8.set(true); // backend supports Stage 8 — show the dataset picker, Data & Export
        }
      },
      error: () => this.stage8.set(false), // older backend: hide the Stage 8 UI, app still works
    });
    this.api.health().subscribe({
      next: (h) => {
        this.cudaAvailable.set(!!h.cuda_available);
        this.serverEnabled.set(!!h.server_projects);
      },
      error: () => {
        this.cudaAvailable.set(false);
        this.serverEnabled.set(false);
      },
    });

    // Bring back the autosaved draft (your work survives a reload); first visit -> the example.
    const restored = this.restoreAutosave();
    if (restored !== 'restored') this.startFresh(this.exampleSnapshot());
    effect(() => {
      this.snapshot(); // track every piece of project state...
      if (this.autosaveArmed) this.scheduleAutosave(); // ...and save a draft shortly after a change
    });
    this.autosaveArmed = restored !== 'pending';
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

  // Everything a project file holds, as one value (autosave tracks it).
  private snapshot = computed<EditorSnapshot>(() => {
    const p = this.pan();
    return {
      name: this.projectName(),
      description: this.projectDesc(),
      createdAt: this.projectCreated(),
      nodes: this.nodes(),
      edges: this.edges(),
      train: this.uiTrain(),
      trainExtra: this.trainExtra(),
      customText: this.customText(),
      view: { pan_x: p.x, pan_y: p.y, zoom: this.zoom() },
    };
  });

  // One short word on the toolbar for where the work stands; the tooltip says more.
  saveStatus = computed(() => {
    if (this.autosaveError()) return '⚠ autosave off';
    if (this.dirty()) return '● edited · autosaved';
    switch (this.origin()) {
      case 'browser':
        return '✓ saved in browser';
      case 'server':
        return '✓ saved on server';
      case 'file':
        return '✓ file';
      default:
        return 'draft · autosaved';
    }
  });

  saveTooltip = computed(() => {
    const err = this.autosaveError();
    if (err) return err;
    const at = this.lastAutosave();
    const when = at ? ` (last at ${at.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })})` : '';
    const where =
      this.origin() === 'server'
        ? 'Save (Ctrl+S) stores it on the server.'
        : 'Save (Ctrl+S) keeps it in this browser; Download makes a file.';
    return `Your draft is autosaved in this browser and comes back after a reload${when}. ${where}`;
  });

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

  /** A new node id that no node in the graph uses yet (opened projects bring their own ids). */
  private freshId(type: string): string {
    const taken = new Set(this.nodes().map((n) => n.id));
    let id: string;
    do {
      this.seq += 1;
      id = `${type.toLowerCase()}_${this.seq}`;
    } while (taken.has(id));
    return id;
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
    if (this.nodes().length && !confirm('Clear the canvas? Every node and connection is removed (there is no undo yet).')) {
      return;
    }
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
      ...(this.trainExtra() as Partial<TrainCfg>), // optimizer, seed, … as the project carries them
      dataset: this.dataset(),
      epochs: this.epochs(),
      lr: this.lr(),
      batch_size: this.batchSize(),
      samples: this.samples(),
    };
    if (this.dataset() === 'custom') train.custom_rows = this.customRows();
    if (this.saveCheckpoint()) train.checkpoint = true;
    if (this.resumeCheckpoint()) train.resume = true;
    // The GPU only where this backend has one: a project saved on a CUDA machine still trains elsewhere.
    if (this.device() !== 'cpu' && this.cudaAvailable()) train.device = this.device();
    return {
      nodes: this.nodes().map((n) => ({ id: n.id, type: n.type, params: n.params })),
      edges: this.edges().map((e) => [e.source, e.target]),
      train,
    };
  }

  /** The training settings the editor's controls hold, as saved in a project. */
  private uiTrain(): Partial<TrainCfg> {
    const t: Partial<TrainCfg> = {
      dataset: this.dataset(),
      epochs: this.epochs(),
      lr: this.lr(),
      batch_size: this.batchSize(),
      samples: this.samples(),
    };
    if (this.customRows().length) t.custom_rows = this.customRows(); // pasted rows travel with the project
    if (this.saveCheckpoint()) t.checkpoint = true;
    if (this.resumeCheckpoint()) t.resume = true;
    if (this.device() !== 'cpu') t.device = this.device();
    return t;
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
  /** Start a new project from a ready architecture: fetch its full graph, lay it
   * out on the canvas, and adopt its recommended training config. The nodes are
   * ordinary, editable layer nodes — the user can re-wire and tune every one. */
  loadArchitecture(id: string): void {
    if (this.busy() || !this.confirmDiscard()) return;
    this.busy.set(true);
    this.runError.set(null);
    this.proposals.set([]);
    this.api.architecture(id).subscribe({
      next: (spec) => {
        const info = this.architectures().find((a) => a.id === id);
        const t = spec.train ?? {};
        const trainExtra: Record<string, unknown> = {};
        for (const [k, v] of Object.entries(t)) if (!UI_TRAIN_KEYS.includes(k)) trainExtra[k] = v;
        this.busy.set(false);
        this.startFresh({
          name: info?.name ?? id,
          description: info?.description ?? '',
          createdAt: null,
          nodes: layoutGraph(spec.nodes, spec.edges),
          edges: spec.edges.map(([source, target]) => ({ source, target })),
          train: {
            // the template brings its recommended schedule; your data choice stays
            ...this.uiTrain(),
            epochs: t.epochs ?? this.epochs(),
            lr: t.lr ?? this.lr(),
            batch_size: t.batch_size ?? this.batchSize(),
            samples: t.samples ?? this.samples(),
          },
          trainExtra,
          customText: this.customText(),
          view: { pan_x: 48, pan_y: 24, zoom: 0.8 },
        });
        this.showNotice('info', `New project from the «${info?.name ?? id}» template.`);
      },
      error: (e) => {
        this.runError.set(this.errMsg(e));
        this.busy.set(false);
      },
    });
  }

  // ------------------------------------------------ projects (Stage 13)
  openProjects(): void {
    this.localProjects.set(this.store.list());
    this.projectsOpen.set(true);
    if (this.serverEnabled()) this.refreshServerProjects();
  }

  closeProjects(): void {
    this.projectsOpen.set(false);
  }

  /** New, empty project (asks first if there are unsaved changes). */
  newProject(): void {
    if (this.busy() || !this.confirmDiscard()) return;
    this.startFresh(this.blankSnapshot());
    this.projectsOpen.set(false);
    this.showNotice('info', 'New project — add layers from the palette.');
  }

  /** Ctrl+S / Save: save where the project lives — the server for a server
   * project, otherwise this browser's project library. */
  saveProject(): void {
    if (this.origin() === 'server' && this.serverSlug() && this.serverEnabled()) this.saveToServer();
    else this.saveLocal(false);
  }

  /** Save into this browser's project library — in place, or as a new copy. */
  saveLocal(asCopy: boolean): void {
    if (!this.store.available) {
      this.showNotice('error', 'This browser does not allow storage here — use Download to keep a file.');
      return;
    }
    const prevName = this.projectName();
    const prevCreated = this.projectCreated();
    if (asCopy) {
      this.projectName.set(`${prevName} (copy)`.slice(0, 120));
      this.projectCreated.set(null);
    }
    try {
      const res = this.store.save(asCopy ? null : this.localId(), toProjectFile(this.snapshot()));
      this.localId.set(res.id);
      this.serverSlug.set(null);
      this.origin.set('browser');
      this.projectCreated.set(res.project.created_at ?? null);
      this.markSaved();
      this.localProjects.set(this.store.list());
      this.showNotice('info', asCopy ? `Saved a copy: «${res.project.name}».` : `Saved «${res.project.name}» in this browser.`);
    } catch {
      this.projectName.set(prevName);
      this.projectCreated.set(prevCreated);
      this.showNotice('error', 'Could not save: browser storage is full. Download the project as a file, or delete old projects.');
    }
  }

  openLocal(id: string): void {
    if (this.busy() || !this.confirmDiscard()) return;
    const raw = this.store.load(id);
    if (raw == null) {
      this.showNotice('error', 'That project is no longer in this browser.');
      this.localProjects.set(this.store.list());
      return;
    }
    const current = readCurrentFormat(raw);
    if (current) {
      this.adoptLoaded({ ok: true, project: current, migrated_from: null, warnings: [] }, 'browser', { localId: id });
      return;
    }
    this.api.normalizeProject(raw).subscribe({
      next: (res) => this.adoptLoaded(res, 'browser', { localId: id }),
      error: (e) => this.showNotice('error', `Could not open it: ${this.errMsg(e)}`),
    });
  }

  deleteLocal(p: LocalProjectMeta): void {
    if (!confirm(`Delete «${p.name}» from this browser? This cannot be undone.`)) return;
    try {
      this.store.remove(p.id);
    } catch {
      this.showNotice('error', 'Could not update browser storage.');
    }
    if (this.localId() === p.id) this.detachFromStore();
    this.localProjects.set(this.store.list());
  }

  /** Download the project as a `.mlc.json` file (opens on any ML Constructor). */
  downloadProject(): void {
    const project = toProjectFile(this.snapshot());
    const now = new Date().toISOString();
    project.created_at = project.created_at ?? now;
    project.updated_at = now;
    const blob = new Blob([JSON.stringify(project, null, 1)], { type: 'application/json' });
    if (blob.size > MAX_PROJECT_BYTES) {
      this.showNotice('error', 'This project is over 5 MB and would not open again — trim the pasted rows first.');
      return;
    }
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = projectFileName(project.name);
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    if (this.origin() === 'new' || this.origin() === 'file') {
      this.origin.set('file');
      this.markSaved(fingerprint(project));
    }
    this.showNotice('info', `Downloaded ${a.download}`);
  }

  onProjectFile(ev: Event): void {
    const input = ev.target as HTMLInputElement;
    const file = input.files?.[0];
    input.value = '';
    if (file) this.openProjectFile(file);
  }

  /** Open a `.mlc.json` file — any format version, or a bare graph spec. The
   * backend migrates and checks it; warnings are shown, nothing is lost silently. */
  openProjectFile(file: File): void {
    if (this.busy()) return;
    if (file.size > MAX_PROJECT_BYTES) {
      this.showNotice('error', `«${file.name}» is too large (the limit is 5 MB).`);
      return;
    }
    if (!this.confirmDiscard()) return;
    const reader = new FileReader();
    reader.onload = () => {
      let raw: unknown;
      try {
        raw = JSON.parse(String(reader.result ?? ''));
      } catch {
        this.showNotice('error', `«${file.name}» is not a JSON file.`);
        return;
      }
      this.api.normalizeProject(raw).subscribe({
        next: (res) => this.adoptLoaded(res, 'file', {}),
        error: (e) => {
          // An older backend without /projects/normalize still opens current-format files.
          const current = e?.status === 404 || e?.status === 405 ? readCurrentFormat(raw) : null;
          if (current) this.adoptLoaded({ ok: true, project: current, migrated_from: null, warnings: [] }, 'file', {});
          else this.showNotice('error', `Could not open «${file.name}»: ${this.errMsg(e)}`);
        },
      });
    };
    reader.onerror = () => this.showNotice('error', `Could not read «${file.name}».`);
    reader.readAsText(file);
  }

  // Drop a project file anywhere on the canvas to open it.
  onDragOver(ev: DragEvent): void {
    if (!ev.dataTransfer?.types?.includes('Files')) return;
    ev.preventDefault();
    ev.dataTransfer.dropEffect = 'copy';
    this.dropHover.set(true);
  }

  onDragLeave(ev: DragEvent): void {
    const to = ev.relatedTarget as Node | null;
    if (!to || !(ev.currentTarget as HTMLElement).contains(to)) this.dropHover.set(false);
  }

  onDrop(ev: DragEvent): void {
    ev.preventDefault();
    this.dropHover.set(false);
    const file = ev.dataTransfer?.files?.[0];
    if (file) this.openProjectFile(file);
  }

  refreshServerProjects(): void {
    this.api.listProjects().subscribe({
      next: (list) => {
        this.serverProjects.set(list);
        this.serverError.set(null);
      },
      error: (e) => this.serverError.set(this.errMsg(e)),
    });
  }

  /** Save on the server: in place for a server project, else as a new one. */
  saveToServer(): void {
    if (!this.serverEnabled() || this.serverBusy()) return;
    const project = toProjectFile(this.snapshot());
    const slug = this.origin() === 'server' ? this.serverSlug() : null;
    this.serverBusy.set(true);
    (slug ? this.api.saveProject(slug, project) : this.api.createProject(project)).subscribe({
      next: (r) => {
        this.serverBusy.set(false);
        this.serverSlug.set(r.slug);
        this.localId.set(null);
        this.origin.set('server');
        this.markSaved(fingerprint(project)); // what was sent, even if you kept editing meanwhile
        this.refreshServerProjects();
        this.showNotice('info', `Saved «${project.name}» on the server.`);
      },
      error: (e) => {
        this.serverBusy.set(false);
        this.showNotice('error', `Could not save on the server: ${this.errMsg(e)}`);
      },
    });
  }

  openServer(slug: string): void {
    if (this.busy() || !this.confirmDiscard()) return;
    this.api.getProject(slug).subscribe({
      next: (res) => this.adoptLoaded(res, 'server', { serverSlug: slug }),
      error: (e) => this.showNotice('error', `Could not open «${slug}»: ${this.errMsg(e)}`),
    });
  }

  deleteServer(p: ProjectSummary): void {
    if (!confirm(`Delete «${p.name}» from the server? This cannot be undone.`)) return;
    this.api.deleteProject(p.slug).subscribe({
      next: () => {
        if (this.origin() === 'server' && this.serverSlug() === p.slug) this.detachFromStore();
        this.refreshServerProjects();
      },
      error: (e) => this.showNotice('error', `Could not delete «${p.name}»: ${this.errMsg(e)}`),
    });
  }

  @HostListener('document:keydown', ['$event'])
  onKeyDown(ev: KeyboardEvent): void {
    // Ctrl+S / Cmd+S — by physical key, so it works on any keyboard layout
    if ((ev.ctrlKey || ev.metaKey) && !ev.altKey && !ev.shiftKey && ev.code === 'KeyS') {
      ev.preventDefault();
      this.saveProject();
    }
  }

  // Never lose the last few hundred milliseconds of work: write a pending draft
  // right away when the tab is hidden or closed.
  @HostListener('window:beforeunload')
  @HostListener('document:visibilitychange')
  flushPendingAutosave(): void {
    if (this.autosaveTimer) this.flushAutosave();
  }

  /** Load a snapshot into every piece of editor state. */
  private applySnapshot(s: EditorSnapshot): void {
    const num = (v: unknown, fallback: number) => (typeof v === 'number' && Number.isFinite(v) ? v : fallback);
    const t = s.train;
    this.projectName.set(s.name);
    this.projectDesc.set(s.description);
    this.projectCreated.set(s.createdAt);
    this.nodes.set(s.nodes);
    this.edges.set(s.edges);
    this.dataset.set(typeof t.dataset === 'string' && t.dataset ? t.dataset : (this.datasets()[0]?.id ?? 'synthetic'));
    this.epochs.set(num(t.epochs, 3));
    this.lr.set(num(t.lr, 0.01));
    this.batchSize.set(num(t.batch_size, 64));
    this.samples.set(num(t.samples, 512));
    this.customRows.set(Array.isArray(t.custom_rows) ? t.custom_rows : []);
    this.saveCheckpoint.set(t.checkpoint === true);
    this.resumeCheckpoint.set(t.resume === true);
    this.device.set(t.device === 'cuda' || t.device === 'auto' ? t.device : 'cpu');
    this.trainExtra.set({ ...s.trainExtra });
    this.customText.set(s.customText);
    this.customError.set(null);
    this.pan.set({ x: s.view.pan_x, y: s.view.pan_y });
    this.zoom.set(s.view.zoom);
    this.selectedId.set(null);
    this.innerSel.set(null);
    this.connectFrom.set(null);
    this.buildResult.set(null);
    this.trainResult.set(null);
    this.runError.set(null);
    this.proposals.set([]);
    this.liveLoss.set([]);
    this.liveAcc.set([]);
    this.seq = s.nodes.length;
  }

  private startFresh(s: EditorSnapshot): void {
    this.applySnapshot(s);
    this.origin.set('new');
    this.localId.set(null);
    this.serverSlug.set(null);
    this.markSaved();
  }

  private adoptLoaded(
    res: ProjectLoadResult,
    origin: ProjectOrigin,
    ids: { localId?: string | null; serverSlug?: string | null },
  ): void {
    this.applySnapshot(fromProjectFile(res.project));
    this.origin.set(origin);
    this.localId.set(ids.localId ?? null);
    this.serverSlug.set(ids.serverSlug ?? null);
    this.markSaved();
    this.projectsOpen.set(false);
    const details = (res.warnings ?? []).filter((w) => w.code !== 'migrated').map((w) => w.message);
    const ds = this.datasets();
    if (ds.length && !ds.some((d) => d.id === this.dataset())) {
      details.push(`The dataset «${this.dataset()}» is not available here — pick another one under Training.`);
    }
    const upgraded = res.migrated_from != null ? ` (upgraded from format ${res.migrated_from})` : '';
    this.showNotice(details.length ? 'warn' : 'info', `Opened «${res.project.name}»${upgraded}.`, details);
  }

  /** The stored copy is gone (deleted): keep the work on the canvas as an unsaved draft. */
  private detachFromStore(): void {
    this.origin.set('new');
    this.localId.set(null);
    this.serverSlug.set(null);
    this.savedFp.set(null);
    this.flushAutosave();
  }

  /** Remember the current state (or the given fingerprint) as saved. */
  private markSaved(fp: string = this.currentFp()): void {
    this.savedFp.set(fp);
    this.flushAutosave();
  }

  private currentFp(): string {
    return fingerprint(toProjectFile(this.snapshot()));
  }

  private isDirtyNow(): boolean {
    return this.currentFp() !== this.savedFp();
  }

  /** Ask before replacing work that is not saved anywhere but the autosaved draft. */
  private confirmDiscard(): boolean {
    if (!this.nodes().length || !this.isDirtyNow()) return true;
    return confirm(`«${this.projectName()}» has unsaved changes. Replace it anyway? (Save first to keep them.)`);
  }

  private scheduleAutosave(): void {
    if (this.autosaveTimer) clearTimeout(this.autosaveTimer);
    this.autosaveTimer = setTimeout(() => this.flushAutosave(), AUTOSAVE_DELAY_MS);
  }

  private flushAutosave(): void {
    if (this.autosaveTimer) {
      clearTimeout(this.autosaveTimer);
      this.autosaveTimer = null;
    }
    const project = toProjectFile(this.snapshot());
    this.dirty.set(fingerprint(project) !== this.savedFp());
    if (!this.autosaveArmed) return;
    if (!this.store.available) {
      this.autosaveError.set('Autosave is off: this browser blocks local storage here. Use Download to keep your work.');
      return;
    }
    try {
      this.store.writeAutosave({
        project,
        origin: this.origin(),
        localId: this.localId(),
        serverSlug: this.serverSlug(),
        savedFp: this.savedFp(),
        at: new Date().toISOString(),
      });
      this.autosaveError.set(null);
      this.lastAutosave.set(new Date());
    } catch {
      this.autosaveError.set('Autosave is off: this project is too big for browser storage. Use Download to keep your work.');
    }
  }

  /** Bring back the autosaved draft. `pending`: it is an older format the backend
   * migrates first — autosave stays off until then, so the old draft is never
   * overwritten by the example. */
  private restoreAutosave(): 'restored' | 'pending' | 'none' {
    const rec = this.store.readAutosave();
    if (!rec) return 'none';
    const origin = ORIGINS.includes(rec.origin) ? rec.origin : 'new';
    const ids = { localId: rec.localId ?? null, serverSlug: rec.serverSlug ?? null };
    const current = readCurrentFormat(rec.project);
    if (current) {
      this.applySnapshot(fromProjectFile(current));
      this.origin.set(origin);
      this.localId.set(ids.localId);
      this.serverSlug.set(ids.serverSlug);
      this.savedFp.set(rec.savedFp ?? null);
      this.dirty.set(this.isDirtyNow());
      this.lastAutosave.set(rec.at ? new Date(rec.at) : null);
      return 'restored';
    }
    this.api.normalizeProject(rec.project).subscribe({
      next: (res) => {
        this.autosaveArmed = true;
        this.adoptLoaded(res, origin, ids);
      },
      error: () => {
        this.autosaveArmed = true;
        this.showNotice('warn', 'The autosaved draft from an older version could not be opened.');
      },
    });
    return 'pending';
  }

  private showNotice(kind: Notice['kind'], text: string, details: string[] = []): void {
    const n: Notice = { kind, text, details };
    this.notice.set(n);
    if (this.noticeTimer) clearTimeout(this.noticeTimer);
    if (kind === 'info') {
      this.noticeTimer = setTimeout(() => {
        if (this.notice() === n) this.notice.set(null);
      }, 3500);
    }
  }

  // -------------------------------------------------------------- example
  private blankSnapshot(): EditorSnapshot {
    return {
      name: 'Untitled',
      description: '',
      createdAt: null,
      nodes: [],
      edges: [],
      train: { dataset: this.datasets()[0]?.id ?? this.dataset(), epochs: 3, lr: 0.01, batch_size: 64, samples: 512 },
      trainExtra: {},
      customText: '',
      view: { ...DEFAULT_VIEW },
    };
  }

  private exampleSnapshot(): EditorSnapshot {
    const mk = (id: string, type: string, x: number, y: number, params: Record<string, unknown> = {}): EditorNode => ({
      id,
      type,
      x,
      y,
      params,
    });
    return {
      ...this.blankSnapshot(),
      nodes: [
        mk('input_1', 'Input', 70, 24, { shape: [1, 28, 28] }),
        mk('flatten_2', 'Flatten', 70, 136, {}),
        mk('linear_3', 'Linear', 70, 248, { out_features: 64, bias: true }),
        mk('relu_4', 'ReLU', 70, 360, {}),
        mk('linear_5', 'Linear', 70, 472, { out_features: 10, bias: true }),
        mk('output_6', 'Output', 70, 584, {}),
      ],
      edges: [
        { source: 'input_1', target: 'flatten_2' },
        { source: 'flatten_2', target: 'linear_3' },
        { source: 'linear_3', target: 'relu_4' },
        { source: 'relu_4', target: 'linear_5' },
        { source: 'linear_5', target: 'output_6' },
      ],
    };
  }
}
