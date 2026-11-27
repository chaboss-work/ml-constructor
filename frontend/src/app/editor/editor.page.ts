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
  ConnectionOption,
  ConnectionOptions,
  DataPreview,
  DataSampleRow,
  DatasetInfo,
  Diagnosis,
  DrawResult,
  EdgeSummary,
  EditorEdge,
  EditorNode,
  FixInfo,
  GalleryImage,
  GenerateResult,
  GraphErrorInfo,
  GraphSpec,
  LayerInfo,
  ParamDef,
  Problem,
  ProjectLoadResult,
  ProjectSummary,
  RunResult,
  TrainCfg,
  TrainEvent,
  TestResult,
  TrainResult,
  WeightsInfo,
} from '../core/graph.models';
import {
  AUTO,
  DEFAULT_VIEW,
  EditorSnapshot,
  LocalProjectMeta,
  MAX_PROJECT_BYTES,
  ProjectOrigin,
  ProjectStorage,
  UI_TRAIN_KEYS,
  edgeSettings,
  fingerprint,
  fromProjectFile,
  layoutGraph,
  projectFileName,
  readCurrentFormat,
  toProjectFile,
} from '../core/project';
import { applyFixOps, textParts } from '../core/fixes';

const NODE_W = 174;
const NODE_H = 78;
const AUTOSAVE_DELAY_MS = 700;
const ANALYSIS_DELAY_MS = 450;
/** The backend API this page needs: Stage 21 (every problem explained, with one-click fixes). */
const REQUIRED_API = 21;
const OUTDATED_MSG =
  'The server still runs an older backend: connection settings, automatic adaptation, hints, validation accuracy, the final test, saved weights, the flagship models, Your text, Run once, models with several inputs, generative training (Draw) and errors explained with one-click fixes need a restart of the backend.';
/** Stage 21 — how many problems the panel shows before "more". */
const PROBLEMS_SHOWN = 3;
/** Stage 18 — the longest text Your text takes (characters). */
const MAX_TEXT_CHARS = 1_000_000;

/** Stage 17 — a parameter count in words for the library: 8.2B, 671B, 999.9M, 152K. */
export function humanCount(n: number | null | undefined): string {
  if (n == null) return '';
  for (const [size, unit] of [
    [1e12, 'T'],
    [1e9, 'B'],
    [1e6, 'M'],
    [1e3, 'K'],
  ] as const) {
    if (n >= size * 0.9995) {
      // 999.9M reads as 1B, not 1000M
      const v = n / size;
      return `${v >= 100 ? Math.round(v) : Number(v.toFixed(1))}${unit}`;
    }
  }
  return String(n);
}
/** Stage 16 — the share of rows kept for the final test unless the project says otherwise. */
const DEFAULT_TEST_FRACTION = 0.1;
/** The choices of the "Final test" picker. */
const TEST_CHOICES: { value: number; label: string }[] = [
  { value: 0, label: 'Off — no final test' },
  { value: 0.1, label: '10 % of rows (default)' },
  { value: 0.2, label: '20 % of rows' },
  { value: 0.3, label: '30 % of rows' },
];
/** The largest weights file the backend accepts (64 MB). */
const MAX_WEIGHTS_BYTES = 64 * 1024 * 1024;

/** A file's bytes as base64, in chunks (no call-stack limit on big files). */
function toBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}
/** Stage 15 — the share of rows held out for validation unless the project says otherwise. */
const DEFAULT_VAL_FRACTION = 0.2;
/** The choices of the "Check on new data" picker. */
const VAL_CHOICES: { value: number; label: string }[] = [
  { value: 0, label: 'Off — measure on the training rows' },
  { value: 0.1, label: '10 % of rows' },
  { value: 0.2, label: '20 % of rows (default)' },
  { value: 0.3, label: '30 % of rows' },
  { value: 0.5, label: '50 % of rows' },
];
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
  styleUrls: ['./editor.page.scss', './projects.scss', './connections.scss', './training.scss', './problems.scss'],
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

  // ---- Your text (Stage 18): a GPT learns it, then writes ----
  textData = signal<string>('');
  textError = signal<string | null>(null);
  genPrompt = signal('');
  genLength = signal(200);
  genTemperature = signal(0.8);
  genBusy = signal(false);
  genError = signal<string | null>(null);
  genResult = signal<GenerateResult | null>(null);
  private genSeed = 0;
  /** the text's different characters (code points, as the backend counts them) */
  textDistinct = computed(() => new Set(Array.from(this.textData())).size);
  /** how many tokens the network scores: its LM head (else its Embedding) */
  networkVocab = computed(() => {
    const head = this.nodes().find((n) => n.type === 'LMHead');
    const emb = this.nodes().find((n) => n.type === 'Embedding');
    const v = head ? head.params['vocab'] : emb ? emb.params['num_embeddings'] : null;
    return typeof v === 'number' ? v : null;
  });

  // ---- custom data: bring your own rows (Stage 9) ----
  customText = signal<string>('');
  customRows = signal<number[][]>([]);
  customError = signal<string | null>(null);

  // ---- live training + checkpoints (Stage 11) ----
  liveLoss = signal<number[]>([]);
  liveAcc = signal<number[]>([]);
  /** Stage 15 — held-out accuracy per epoch while training streams. */
  liveValAcc = signal<number[]>([]);
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

  // ---- smart connections (Stage 14) ----
  /** the selected connection, as "source->target" */
  selectedEdge = signal<string | null>(null);
  /** the live build after each edit: shapes on the nodes, transforms on the wires */
  analysis = signal<BuildResult | null>(null);
  /** why the graph does not build yet (soft: shown without pressing Build) */
  analysisError = signal<GraphErrorInfo | null>(null);
  /** where the last explicit Build / Train failed (for the red highlight) */
  runErrorInfo = signal<GraphErrorInfo | null>(null);
  connOptions = signal<ConnectionOptions | null>(null);
  connError = signal<string | null>(null);

  // ---- clear errors with fixes (Stage 21) ----
  /** the live check after each edit: every problem and warning, explained, with checked fixes */
  diagnosis = signal<Diagnosis | null>(null);
  problemsOpen = signal(true);
  problemsAll = signal(false);
  /** the problem the canvas points at (pulsing outline) */
  focusKey = signal<string | null>(null);
  /** a glossary word opened under a problem */
  openTerm = signal<{ key: string; term: string } | null>(null);
  fixNote = signal<string | null>(null);
  libraryFlash = signal(false);
  private fixNoteTimer: ReturnType<typeof setTimeout> | null = null;
  readonly PROBLEMS_SHOWN = PROBLEMS_SHOWN;
  /** The server runs an older backend than this page (a restart updates it). */
  backendOutdated = signal(false);
  private analysisTimer: ReturnType<typeof setTimeout> | null = null;
  private analysisSeq = 0;

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
  /** Stage 15 — share of rows held out for validation (the network never trains on them). */
  valFraction = signal(DEFAULT_VAL_FRACTION);
  readonly valChoices = VAL_CHOICES;
  /** Stage 16 — share of rows kept for the final test (never trained on, not judged per epoch). */
  testFraction = signal(DEFAULT_TEST_FRACTION);
  readonly testChoices = TEST_CHOICES;
  /** The last final test (shown only while it belongs to the current weights). */
  testResult = signal<TestResult | null>(null);
  testBusy = signal(false);
  testError = signal<string | null>(null);
  /** Weights loaded from a file: Test checks them, the next Train starts from them. */
  loadedWeights = signal<{ run_id: string; name: string; info: WeightsInfo } | null>(null);
  weightsBusy = signal(false);

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
        this.backendOutdated.set((h.api ?? 0) < REQUIRED_API);
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
    effect(() => {
      this.graphKey(); // layers, parameters and connections — not positions...
      this.scheduleAnalysis(); // ...rebuild shortly after a change: live shapes and transforms
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

  /** The explicit Build's result, else the live one. */
  liveBuild = computed(() => this.buildResult() ?? this.analysis());

  shapeById = computed(() => {
    const map = new Map<string, number[] | null>();
    const b = this.liveBuild();
    if (b) for (const s of b.layers) map.set(s.id, s.out_shape);
    // Stage 21: a broken graph still shows the shapes worked out before the problem
    else for (const [id, sh] of Object.entries(this.diagnosis()?.shapes ?? {})) map.set(id, sh);
    return map;
  });

  edgeInfo = computed(() => new Map((this.liveBuild()?.edges ?? []).map((e) => [`${e.source}->${e.target}`, e])));

  /** Where the current problem is: the explicit Build's error first, else the live one. */
  private problem = computed(() => this.runErrorInfo() ?? this.analysisError());

  // ---- Stage 21: every problem, highlighted where it is ----
  /** What the Problems panel lists: the explicit Build / Train error first, then the live ones, then warnings. */
  problemList = computed<Problem[]>(() => {
    const d = this.diagnosis();
    const out: Problem[] = [];
    const run = this.runErrorInfo();
    if (run?.code) out.push(this.asProblem(run));
    for (const p of [...(d?.problems ?? []), ...(d?.warnings ?? [])]) {
      if (!out.some((q) => this.problemKey(q) === this.problemKey(p))) out.push(p);
    }
    if (!d && !run?.code) {
      const soft = this.problem(); // an older backend: its one message, without the catalogue
      if (soft) out.push(this.asProblem(soft));
    }
    return out;
  });
  errorCount = computed(() => this.problemList().filter((p) => p.severity !== 'warning').length);
  warningCount = computed(() => this.problemList().filter((p) => p.severity === 'warning').length);
  shownProblems = computed(() =>
    this.problemsAll() ? this.problemList() : this.problemList().slice(0, PROBLEMS_SHOWN),
  );
  /** node id -> the worst problem on it (a layer inside a block marks the block) */
  issueByNode = computed(() => {
    const m = new Map<string, 'error' | 'warning'>();
    for (const p of this.problemList()) {
      const ids = p.node_id ? [p.node_id] : [];
      if (p.code === 'unused_branch' && p.fixes[0]) {
        for (const o of p.fixes[0].ops) if (o.op === 'remove_node') ids.push(String(o['node']));
      }
      for (const id of ids) if (m.get(id) !== 'error') m.set(id, p.severity === 'warning' ? 'warning' : 'error');
    }
    return m;
  });
  issueByEdge = computed(() => {
    const m = new Map<string, 'error' | 'warning'>();
    for (const p of this.problemList()) {
      if (p.edge?.length !== 2) continue;
      const k = `${p.edge[0]}->${p.edge[1]}`;
      if (m.get(k) !== 'error') m.set(k, p.severity === 'warning' ? 'warning' : 'error');
    }
    return m;
  });
  private focused = computed(() => this.problemList().find((p) => this.problemKey(p) === this.focusKey()) ?? null);
  focusNode = computed(() => {
    const p = this.focused();
    return p && !(p.edge?.length === 2) ? p.node_id : null;
  });
  focusEdge = computed(() => {
    const e = this.focused()?.edge;
    return e?.length === 2 ? `${e[0]}->${e[1]}` : null;
  });

  edgeViews = computed(() => {
    const byId = new Map(this.nodes().map((n) => [n.id, n]));
    const info = this.edgeInfo();
    const selected = this.selectedEdge();
    const issues = this.issueByEdge();
    const focus = this.focusEdge();
    const views: {
      id: string;
      d: string;
      mx: number;
      my: number;
      label: string;
      auto: boolean;
      selected: boolean;
      problem: boolean;
      warn: boolean;
      focus: boolean;
      title: string;
    }[] = [];
    for (const e of this.edges()) {
      const s = byId.get(e.source);
      const t = byId.get(e.target);
      if (!s || !t) continue;
      const x1 = s.x + NODE_W / 2;
      const y1 = s.y + NODE_H;
      const x2 = t.x + NODE_W / 2;
      const y2 = t.y;
      const k = Math.max(36, Math.abs(y2 - y1) / 2);
      const id = `${e.source}->${e.target}`;
      const live = info.get(id);
      const pinned = !!e.transform && e.transform !== AUTO;
      const label = live?.label ?? (pinned ? (e.transform as string) : '');
      const shapes = live ? ` · [${live.in_shape.join('×')}] → [${live.out_shape.join('×')}]` : '';
      views.push({
        id,
        d: `M ${x1} ${y1} C ${x1} ${y1 + k}, ${x2} ${y2 - k}, ${x2} ${y2}`,
        mx: (x1 + x2) / 2, // the curve's midpoint (its control points share the ends' x)
        my: (y1 + y2) / 2,
        label,
        auto: live ? live.auto : !pinned,
        selected: id === selected,
        problem: issues.get(id) === 'error',
        warn: issues.get(id) === 'warning',
        focus: id === focus,
        title: `${e.source} → ${e.target}${label ? ' · ' + label : ''}${shapes} — click to set up`,
      });
    }
    return views;
  });

  // ---- the connection inspector ----
  selectedEdgeObj = computed(() => {
    const key = this.selectedEdge();
    return key ? (this.edges().find((e) => `${e.source}->${e.target}` === key) ?? null) : null;
  });

  selectedEdgeInfo = computed<EdgeSummary | null>(() => {
    const key = this.selectedEdge();
    return key ? (this.edgeInfo().get(key) ?? null) : null;
  });

  /** The option matching the selected connection's pinned transform (null on auto). */
  currentOption = computed<ConnectionOption | null>(() => {
    const e = this.selectedEdgeObj();
    if (!e?.transform || e.transform === AUTO) return null;
    return this.connOptions()?.options.find((o) => o.transform === e.transform) ?? null;
  });

  /** What "auto" does right now, for the selector's first entry. */
  autoSummary = computed(() => {
    const o = this.connOptions();
    if (!o?.auto_transform) return '';
    if (o.auto_transform === 'none') return ' — now: no change needed';
    const label = o.options.find((x) => x.transform === o.auto_transform)?.label ?? o.auto_transform;
    return ` — now: ${label}${o.auto_out_shape ? ' → ' + o.auto_out_shape.join('×') : ''}`;
  });

  // layers, parameters and connection settings — what the live build depends on (not positions)
  private graphKey = computed(() =>
    JSON.stringify({
      n: this.nodes().map((n) => [n.id, n.type, n.params]),
      e: this.edges().map((e) => [e.source, e.target, e.transform ?? AUTO, e.params ?? {}]),
      i: this.nodes().length,
      // Stage 21: the live check also asks whether the chosen data and settings would refuse the network
      t: [
        this.dataset(),
        this.lr(),
        this.batchSize(),
        this.samples(),
        this.valFraction(),
        this.testFraction(),
        this.customRows().length,
        JSON.stringify(this.customRows().slice(0, 1)),
        this.textData().length,
      ],
    }),
  );

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
  curveValAcc = computed(() => {
    const live = this.liveValAcc();
    return live.length ? live : (this.trainResult()?.val_acc_curve ?? []);
  });

  lossPoints = computed(() => this.pointsOf(this.curveLoss(), false));
  // Accuracy on a fixed 0..1 axis so the height reads as accuracy.
  accPoints = computed(() => this.pointsOf(this.curveAcc(), true));
  valAccPoints = computed(() => this.pointsOf(this.curveValAcc(), true));
  /** Stage 15 — the picker's label for a share that is not one of its choices (from a project). */
  valChoiceMissing = computed(() => !VAL_CHOICES.some((c) => c.value === this.valFraction()));
  testChoiceMissing = computed(() => !TEST_CHOICES.some((c) => c.value === this.testFraction()));
  /** Stage 16 — the weights Test and Save weights act on: the last training, else the loaded file. */
  currentRunId = computed(() => this.trainResult()?.run_id ?? this.loadedWeights()?.run_id ?? null);
  /** The final test, while it still belongs to the current weights. */
  shownTest = computed(() => {
    const t = this.testResult();
    return t && t.run_id === this.currentRunId() ? t : null;
  });

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
    this.selectedEdge.set(null);
  }

  deleteSelected(): void {
    const id = this.selectedId();
    if (!id) return;
    this.nodes.update((ns) => ns.filter((n) => n.id !== id));
    this.edges.update((es) => es.filter((e) => e.source !== id && e.target !== id));
    this.selectedId.set(null);
    this.selectedEdge.set(null);
    this.invalidateBuild();
  }

  // ------------------------------------------------- connections (Stage 14)
  selectEdge(key: string, ev: Event): void {
    ev.stopPropagation();
    this.selectedEdge.set(key);
    this.selectedId.set(null);
    this.connectFrom.set(null);
    this.connOptions.set(null);
    this.loadConnectionOptions();
  }

  deleteSelectedEdge(): void {
    const key = this.selectedEdge();
    if (!key) return;
    this.edges.update((es) => es.filter((e) => `${e.source}->${e.target}` !== key));
    this.selectedEdge.set(null);
    this.connOptions.set(null);
    this.invalidateBuild();
  }

  /** Ask the backend what this connection can do: the shape arriving, auto's pick, every option. */
  private loadConnectionOptions(): void {
    const e = this.selectedEdgeObj();
    if (!e) {
      this.connOptions.set(null);
      return;
    }
    const key = `${e.source}->${e.target}`;
    this.connError.set(null);
    this.api.connectionOptions(this.toSpec(), e.source, e.target).subscribe({
      next: (o) => {
        if (this.selectedEdge() === key) this.connOptions.set(o);
      },
      error: (err) => {
        if (this.selectedEdge() !== key) return;
        this.connOptions.set(null);
        this.connError.set(
          err?.status === 404 || err?.status === 405 ? OUTDATED_MSG : this.errMsg(err),
        );
      },
    });
  }

  private updateSelectedEdge(patch: (e: EditorEdge) => EditorEdge): void {
    const key = this.selectedEdge();
    if (!key) return;
    this.edges.update((es) => es.map((e) => (`${e.source}->${e.target}` === key ? patch(e) : e)));
    this.invalidateBuild();
  }

  /** Pick the connection's transform: "auto", or one option (with its first preset). */
  setEdgeTransform(kind: string): void {
    if (!kind || kind === AUTO) {
      this.updateSelectedEdge(({ source, target }) => ({ source, target }));
      return;
    }
    const opt = this.connOptions()?.options.find((o) => o.transform === kind);
    const params = structuredClone(opt?.presets[0] ?? {});
    this.updateSelectedEdge((e) => ({ source: e.source, target: e.target, transform: kind, params }));
  }

  applyEdgePreset(preset: Record<string, unknown>): void {
    this.updateSelectedEdge((e) => ({ ...e, params: structuredClone(preset) }));
  }

  setEdgeParam(p: ParamDef, raw: unknown): void {
    const val = this.coerceParam(p, raw);
    this.updateSelectedEdge((e) => ({ ...e, params: { ...(e.params ?? {}), [p.name]: val } }));
  }

  /** The value an edge parameter shows: its own, else the option's default preset, else the def's default. */
  edgeParam(p: ParamDef): string | number | boolean {
    const own = this.selectedEdgeObj()?.params?.[p.name];
    const v = own ?? this.currentOption()?.presets[0]?.[p.name] ?? p.default;
    if (p.type === 'intlist') return Array.isArray(v) ? v.join(', ') : String(v ?? '');
    return v as string | number | boolean;
  }

  presetLabel(p: Record<string, unknown>): string {
    if ('channels' in p) return `${p['channels']}×${p['height']}×${p['width']}`;
    if ('shape' in p) return Array.isArray(p['shape']) ? (p['shape'] as number[]).join('×') : String(p['shape']);
    if ('length' in p) return `${p['length']} step${p['length'] === 1 ? '' : 's'}`;
    if ('mode' in p) return p['mode'] === 'patches' ? `patches ${p['patch']}²` : String(p['mode']);
    return Object.values(p).join(', ');
  }

  isPresetActive(p: Record<string, unknown>): boolean {
    const own = this.selectedEdgeObj()?.params ?? {};
    return Object.entries(p).every(([k, v]) => JSON.stringify(own[k]) === JSON.stringify(v));
  }

  private scheduleAnalysis(): void {
    if (this.analysisTimer) clearTimeout(this.analysisTimer);
    this.analysisTimer = setTimeout(() => this.runAnalysis(), ANALYSIS_DELAY_MS);
  }

  /** Build the graph in the background after each edit: live shapes on the nodes,
   * live transforms on the wires, and a gentle note when it does not build yet. */
  private runAnalysis(): void {
    this.analysisTimer = null;
    const seq = ++this.analysisSeq;
    if (!this.nodes().length) {
      this.analysis.set(null);
      this.analysisError.set(null);
      return;
    }
    const spec = this.toSpec();
    this.api.diagnose(spec).subscribe({
      next: (d) => {
        if (seq !== this.analysisSeq) return; // a newer edit is on its way
        this.diagnosis.set(d);
        this.analysis.set(d.build);
        const first = d.problems[0];
        this.analysisError.set(first ? { ...first, hint: null } : null);
      },
      error: (e) => {
        if (seq !== this.analysisSeq) return;
        this.diagnosis.set(null);
        if (e?.status === 404 || e?.status === 405) {
          this.buildAnalysis(seq, spec); // an older backend without /diagnose: the Stage 14 live build
          return;
        }
        this.analysis.set(null);
        this.analysisError.set(e?.status === 0 ? null : this.errInfo(e));
      },
    });
    if (this.selectedEdge()) this.loadConnectionOptions();
  }

  private buildAnalysis(seq: number, spec: GraphSpec): void {
    this.api.build(spec).subscribe({
      next: (r) => {
        if (seq !== this.analysisSeq) return;
        this.analysis.set(r);
        this.analysisError.set(null);
      },
      error: (e) => {
        if (seq !== this.analysisSeq) return;
        this.analysis.set(null);
        this.analysisError.set(e?.status === 0 ? null : this.errInfo(e));
      },
    });
  }

  // ------------------------------------------- clear errors with fixes (Stage 21)
  problemKey(p: Problem): string {
    return `${p.code}|${p.node_id ?? ''}|${(p.edge ?? []).join('>')}|${p.inner ?? ''}`;
  }

  /** A 422's `error` as a problem card (an older backend's has no catalogue words). */
  private asProblem(e: GraphErrorInfo): Problem {
    return {
      code: e.code ?? 'other',
      severity: e.severity ?? 'error',
      category: e.category ?? '',
      title: e.title ?? 'The graph does not build yet',
      message: e.message,
      why: e.why ?? '',
      todo: e.todo ?? e.hint ?? '',
      terms: e.terms ?? {},
      node_id: e.node_id,
      edge: e.edge,
      inner: e.inner ?? null,
      param: e.param ?? null,
      fixes: e.fixes ?? [],
    };
  }

  /** The "why" / "what to do" text in pieces: glossary words become buttons. */
  parts(text: string): { text: string; term: string | null }[] {
    return textParts(text ?? '');
  }

  toggleTerm(p: Problem, term: string, ev: Event): void {
    ev.stopPropagation();
    const key = this.problemKey(p);
    const open = this.openTerm();
    this.openTerm.set(open && open.key === key && open.term === term ? null : { key, term });
  }

  termText(p: Problem): { term: string; text: string } | null {
    const open = this.openTerm();
    if (!open || open.key !== this.problemKey(p)) return null;
    const text = p.terms[open.term];
    return text ? { term: open.term, text } : null;
  }

  /** Point the canvas at a problem: select its node or connection and bring it into view. */
  focusProblem(p: Problem, ev?: Event): void {
    ev?.stopPropagation();
    this.focusKey.set(this.problemKey(p));
    const e = p.edge?.length === 2 ? p.edge : null;
    const key = e ? `${e[0]}->${e[1]}` : null;
    if (e && key && this.edges().some((x) => `${x.source}->${x.target}` === key)) {
      this.selectedEdge.set(key);
      this.selectedId.set(null);
      this.connOptions.set(null);
      this.loadConnectionOptions();
      const a = this.nodes().find((n) => n.id === e[0]);
      const b = this.nodes().find((n) => n.id === e[1]);
      if (a && b) this.centerOn((a.x + b.x) / 2 + NODE_W / 2, (a.y + NODE_H + b.y) / 2);
      return;
    }
    const n = p.node_id ? this.nodes().find((x) => x.id === p.node_id) : null;
    if (n) {
      this.select(n.id);
      if (p.inner != null) this.innerSel.set(p.inner);
      this.centerOn(n.x + NODE_W / 2, n.y + NODE_H / 2);
    }
  }

  /** Pan so that the canvas point (x, y) sits in the upper middle of the visible canvas. */
  private centerOn(x: number, y: number): void {
    const wrap = document.querySelector('.canvas-wrap') as HTMLElement | null;
    const w = wrap?.clientWidth ?? 900;
    const h = wrap?.clientHeight ?? 600;
    const z = this.zoom();
    this.pan.set({ x: Math.round(w / 2 - x * z), y: Math.round(h * 0.38 - y * z) });
  }

  /** One click: apply a fix's edits to the graph (and do what only the editor can do). */
  applyFix(f: FixInfo, ev?: Event): void {
    ev?.stopPropagation();
    let res;
    try {
      res = applyFixOps({ nodes: this.nodes(), edges: this.edges() }, f.ops, { nodeW: NODE_W, nodeH: NODE_H });
    } catch {
      this.showFixNote('The graph has changed since — the check runs again.');
      this.scheduleAnalysis();
      return;
    }
    const graphEdit = f.ops.some((o) => o.op !== 'ui' && o.op !== 'set_train');
    if (graphEdit) {
      this.nodes.set(res.nodes);
      this.edges.set(res.edges);
      if (this.selectedId() && !res.nodes.some((n) => n.id === this.selectedId())) this.selectedId.set(null);
      const se = this.selectedEdge();
      if (se && !res.edges.some((e) => `${e.source}->${e.target}` === se)) {
        this.selectedEdge.set(null);
        this.connOptions.set(null);
      }
      this.invalidateBuild();
      this.trainResult.set(null);
    }
    for (const [k, v] of Object.entries(res.train)) this.setTrainValue(k, v);
    this.runError.set(null);
    this.runErrorInfo.set(null);
    this.focusKey.set(null);
    this.openTerm.set(null);
    this.showFixNote(`✓ ${f.label}`);
    for (const a of res.ui) this.runUiAction(a);
    this.scheduleAnalysis();
  }

  private showFixNote(text: string): void {
    this.fixNote.set(text);
    if (this.fixNoteTimer) clearTimeout(this.fixNoteTimer);
    this.fixNoteTimer = setTimeout(() => this.fixNote.set(null), 3500);
  }

  /** A training setting a fix changes. */
  private setTrainValue(name: string, v: unknown): void {
    switch (name) {
      case 'dataset':
        this.setDataset(String(v));
        break;
      case 'epochs':
        this.epochs.set(Number(v));
        break;
      case 'lr':
        this.lr.set(Number(v));
        break;
      case 'batch_size':
        this.batchSize.set(Number(v));
        break;
      case 'samples':
        this.samples.set(Number(v));
        break;
      case 'val_fraction':
        this.valFraction.set(Number(v));
        break;
      case 'test_fraction':
        this.testFraction.set(Number(v));
        break;
      case 'device':
        this.device.set(v as 'cpu' | 'cuda' | 'auto');
        break;
      default:
        this.trainExtra.update((t) => ({ ...t, [name]: v }));
    }
  }

  /** What a fix asks of the editor itself. */
  private runUiAction(a: string): void {
    switch (a) {
      case 'open_data':
        this.openData();
        break;
      case 'use_example_text':
        if (this.dataset() !== 'text') this.setDataset('text');
        this.useExampleText();
        break;
      case 'open_library':
        document.querySelector('.arch-list')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        this.libraryFlash.set(true);
        setTimeout(() => this.libraryFlash.set(false), 1800);
        break;
      case 'open_weights':
        (document.querySelector('.weights-file') as HTMLInputElement | null)?.click();
        break;
      case 'train':
        this.onTrain();
        break;
      case 'run_once':
        this.onRunOnce();
        break;
    }
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
    this.selectedEdge.set(null);
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
    if (n.type === 'Input' && typeof n.params['name'] === 'string' && n.params['name'].trim()) {
      return `Input · ${n.params['name'].trim()}`; // Stage 19: a model can take several inputs
    }
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
      val_fraction: this.valFraction(),
      test_fraction: this.testFraction(),
    };
    // Start from loaded weights only while they are loaded (a stale id in a project is dropped).
    delete train.init_run;
    const loaded = this.loadedWeights();
    if (loaded) train.init_run = loaded.run_id;
    if (this.dataset() === 'custom') train.custom_rows = this.customRows();
    if (this.dataset() === 'text') train.text = this.textData();
    if (this.saveCheckpoint()) train.checkpoint = true;
    if (this.resumeCheckpoint()) train.resume = true;
    // The GPU only where this backend has one: a project saved on a CUDA machine still trains elsewhere.
    if (this.device() !== 'cpu' && this.cudaAvailable()) train.device = this.device();
    const spec: GraphSpec = {
      nodes: this.nodes().map((n) => ({ id: n.id, type: n.type, params: n.params })),
      edges: this.edges().map((e) => [e.source, e.target]),
      train,
    };
    const settings = edgeSettings(this.edges());
    if (settings.length) spec.edge_settings = settings; // only when a connection is not on auto
    return spec;
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
    if (this.textData()) t.text = this.textData(); // and so does your text (Stage 18)
    if (this.saveCheckpoint()) t.checkpoint = true;
    if (this.resumeCheckpoint()) t.resume = true;
    if (this.device() !== 'cpu') t.device = this.device();
    if (this.valFraction() !== DEFAULT_VAL_FRACTION) t.val_fraction = this.valFraction();
    if (this.testFraction() !== DEFAULT_TEST_FRACTION) t.test_fraction = this.testFraction();
    return t;
  }

  /** Stage 17 — a parameter count in words (8.2B, 671B) for the library. */
  human(n: number | null | undefined): string {
    return humanCount(n);
  }

  /** Stage 17 — the live build says the network is too big to train here (a real-size flagship). */
  inspectOnly = computed(() => this.analysis()?.trainable === false);

  /** Stage 17 — where the flagship group starts in the library list. */
  firstFlagship = computed(() =>
    this.architectures().findIndex((a) => a.scale === 'mini' && a.category === 'language'),
  );
  /** Stage 19 — where the image / 3D / sound / vision flagships start. */
  firstVisionFlagship = computed(() =>
    this.architectures().findIndex((a) => a.scale === 'mini' && a.category !== 'language'),
  );
  /** Stage 19 — why the network cannot be trained here yet (several inputs, a picture as output). */
  trainNote = computed(() => this.analysis()?.train_note ?? null);
  /** Stage 19 — small enough to run once here (a real-size flagship is not). */
  runnable = computed(() => this.analysis()?.runnable !== false);
  /** Stage 20 — what training will teach this graph (from the live build). */
  taskName = computed(() => this.analysis()?.task ?? null);
  /** Stage 20 — the gallery while training: the pictures after the latest epoch. */
  liveGallery = signal<GalleryImage[]>([]);
  liveGalleryEpoch = signal(0);
  /** Stage 20 — pictures drawn with the trained weights, and the Draw controls. */
  drawResult = signal<DrawResult | null>(null);
  drawBusy = signal(false);
  drawError = signal<string | null>(null);
  drawLabel = signal('');
  drawCount = signal(10);
  drawSteps = signal(50);
  drawGuidance = signal(1);
  private drawSeed = 0;
  private imgCache = new Map<string, string>();
  drawClassOptions = computed(() => Array.from({ length: this.trainResult()?.draw_classes ?? 0 }, (_, i) => i));

  /** Stage 20 — the VAE sampling layer (it lets a reconstruction network draw new pictures). */
  isSampler = (l: { type: string }): boolean => l.type === 'Reparameterize';

  isGenerative(task: string | null | undefined): boolean {
    return task === 'reconstruction' || task === 'diffusion';
  }

  /** A gallery picture as a data URL (painted once on a canvas, then cached). */
  imgUrl(g: GalleryImage): string {
    const hit = this.imgCache.get(g.data);
    if (hit) return hit;
    const bytes = Uint8Array.from(atob(g.data), (ch) => ch.charCodeAt(0));
    const canvas = document.createElement('canvas');
    canvas.width = g.w;
    canvas.height = g.h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return '';
    const img = ctx.createImageData(g.w, g.h);
    for (let i = 0; i < g.w * g.h; i++) {
      const r = bytes[i * g.c];
      img.data[i * 4] = r;
      img.data[i * 4 + 1] = g.c === 3 ? bytes[i * 3 + 1] : r;
      img.data[i * 4 + 2] = g.c === 3 ? bytes[i * 3 + 2] : r;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    const url = canvas.toDataURL('image/png');
    if (this.imgCache.size > 600) this.imgCache.clear();
    this.imgCache.set(g.data, url);
    return url;
  }

  /** Stage 20 — draw new pictures with the trained weights. */
  onDraw(): void {
    const runId = this.currentRunId();
    if (!runId) return;
    this.drawBusy.set(true);
    this.drawError.set(null);
    this.drawSeed += 1;
    const label = this.drawLabel() === '' ? null : Number(this.drawLabel());
    this.api
      .draw(this.toSpec(), runId, this.drawCount(), label, this.drawSteps(), this.drawGuidance(), this.drawSeed)
      .subscribe({
        next: (r) => {
          this.drawResult.set(r);
          this.drawBusy.set(false);
        },
        error: (err) => {
          this.drawError.set(err?.status === 405 ? OUTDATED_MSG : this.errMsg(err));
          this.drawBusy.set(false);
        },
      });
  }

  /** Stage 19 — the last forward pass on random inputs. */
  onceResult = signal<RunResult | null>(null);
  onceBusy = signal(false);
  onceError = signal<string | null>(null);

  /** Stage 19 — run the network once on random inputs: does it run, what comes out, how fast. */
  onRunOnce(): void {
    if (!this.nodes().length) return;
    this.onceBusy.set(true);
    this.onceError.set(null);
    this.api.run(this.toSpec()).subscribe({
      next: (r) => {
        this.onceResult.set(r);
        this.onceBusy.set(false);
      },
      error: (err) => {
        this.onceResult.set(null);
        this.onceError.set(err?.status === 404 || err?.status === 405 ? OUTDATED_MSG : this.errMsg(err));
        const info = this.errInfo(err);
        if (info.code) this.showRunProblem(info);
        this.onceBusy.set(false);
      },
    });
  }

  shapeStr(s: number[] | null | undefined): string {
    return (s ?? []).join('×');
  }

  /** What Run once fed an input, in words. */
  kindText(kind: string): string {
    return kind === 'tokens' ? 'random token ids' : kind === 'timestep' ? 'random steps' : 'random numbers';
  }

  sampleStr(v: number[]): string {
    return v.map((x) => (Math.abs(x) >= 100 ? x.toExponential(2) : x.toFixed(3))).join('  ');
  }

  /** Stage 16 — the "Final test" picker. */
  setTestFraction(v: string | number): void {
    const f = Number(v);
    if (!Number.isFinite(f) || f < 0 || f > 0.5) return;
    this.testFraction.set(f);
    this.trainResult.set(null);
  }

  /** Stage 16 — the final test on the current weights (the last training, or the loaded file). */
  onTest(): void {
    const runId = this.currentRunId();
    if (!runId) return;
    this.testBusy.set(true);
    this.testError.set(null);
    this.api.test(this.toSpec(), runId).subscribe({
      next: (r) => {
        this.testResult.set(r);
        this.testBusy.set(false);
        // bring the result into view: it sits under the training numbers
        setTimeout(() => document.querySelector('.final-test')?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }), 60);
      },
      error: (err) => {
        this.testError.set(this.errMsg(err));
        const info = this.errInfo(err);
        if (info.code) this.showRunProblem(info); // Stage 21: e.g. no test part -> "Set Final test to 10 %"
        this.testBusy.set(false);
      },
    });
  }

  /** Stage 16 — download the current weights as a .pt file. */
  onSaveWeights(): void {
    const runId = this.currentRunId();
    if (!runId) return;
    const a = document.createElement('a');
    a.href = this.api.weightsUrl(runId, projectFileName(this.projectName()).replace(/\.mlc\.json$/, ''));
    a.download = '';
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  /** Stage 16 — load a weights file (.pt) into this graph. */
  onWeightsFile(ev: Event): void {
    const input = ev.target as HTMLInputElement;
    const file = input.files?.[0];
    input.value = '';
    if (!file) return;
    if (file.size > MAX_WEIGHTS_BYTES) {
      this.showNotice('error', `«${file.name}» is over 64 MB — too big to load here.`);
      return;
    }
    this.weightsBusy.set(true);
    file
      .arrayBuffer()
      .then((buf) => {
        this.api.uploadWeights(this.toSpec(), toBase64(buf), file.name).subscribe({
          next: (info) => {
            this.loadedWeights.set({ run_id: info.run_id, name: file.name, info });
            this.trainResult.set(null);
            this.weightsBusy.set(false);
            this.showNotice('info', `${info.note ?? 'Weights loaded.'} Test checks them; Train continues from them.`);
          },
          error: (err) => {
            this.weightsBusy.set(false);
            this.showNotice('error', `Could not load «${file.name}»: ${this.errMsg(err)}`);
          },
        });
      })
      .catch(() => {
        this.weightsBusy.set(false);
        this.showNotice('error', `Could not read «${file.name}».`);
      });
  }

  /** Stop starting from the loaded weights (the next Train starts fresh). */
  clearLoadedWeights(): void {
    this.loadedWeights.set(null);
  }

  /** Stage 15 — the "Check on new data" picker. */
  setValFraction(v: string | number): void {
    const f = Number(v);
    if (!Number.isFinite(f) || f < 0 || f > 0.5) return;
    this.valFraction.set(f);
    this.trainResult.set(null);
  }

  /** Stage 15 — a share as a percentage for the picker ("25 %"). */
  pct(v: number): string {
    return `${Math.round(v * 100)} %`;
  }

  /** Stage 15 — the verdict's symbol: ✓ generalises, ⚠ otherwise. */
  fitIcon(fit: string | null | undefined): string {
    return fit === 'generalises' ? '✓' : '⚠';
  }

  onBuild(): void {
    this.busy.set(true);
    this.runError.set(null);
    this.runErrorInfo.set(null);
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
        const info = this.errInfo(e);
        this.runErrorInfo.set(info);
        this.busy.set(false);
        if (info.code) {
          // Stage 21: the Problems panel explains it, with its fixes; the view jumps to the culprit
          this.problemsOpen.set(true);
          this.focusProblem(this.asProblem(info));
        } else {
          this.fetchSuggestions(spec); // an older backend: offer a one-click adapter if a shape mismatch is fixable
        }
      },
    });
  }

  onTrain(): void {
    if (this.dataset() === 'custom' && !this.customRows().length) {
      this.runError.set('Select some data first: open “Data”, paste your rows, and Apply.');
      return;
    }
    if (this.dataset() === 'text' && !this.textData().trim()) {
      this.runError.set('Add your text first: open “Data”, paste or upload a text (or use the example).');
      return;
    }
    this.genResult.set(null);
    this.busy.set(true);
    this.runError.set(null);
    this.runErrorInfo.set(null);
    this.proposals.set([]);
    this.trainResult.set(null);
    this.liveLoss.set([]);
    this.liveAcc.set([]);
    this.liveValAcc.set([]);
    this.liveGallery.set([]);
    this.drawResult.set(null);
    this.drawError.set(null);
    const spec = this.toSpec();
    this.api
      .trainStream(spec, (e) => {
        if (e.type === 'start') {
          this.liveLoss.set([]);
          this.liveAcc.set([]);
          this.liveValAcc.set([]);
        } else if (e.type === 'step' && e.loss != null) {
          this.liveLoss.update((a) => [...a, e.loss as number]);
        } else if (e.type === 'epoch' && e.acc != null) {
          this.liveAcc.update((a) => [...a, e.acc as number]);
          if (e.val_acc != null) this.liveValAcc.update((a) => [...a, e.val_acc as number]);
        } else if (e.type === 'samples' && e.images) {
          this.liveGallery.set(e.images); // Stage 20: the pictures after this epoch
          this.liveGalleryEpoch.set(e.epoch ?? 0);
        } else if (e.type === 'done' && e.result) {
          this.trainResult.set(e.result);
          if (e.result.run_id) this.loadedWeights.set(null); // the loaded weights went into this run
          this.busy.set(false);
        } else if (e.type === 'error') {
          this.runError.set(e.detail ?? 'Training failed');
          if (e.error?.code) this.showRunProblem(e.error);
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
          if (err?.error?.code) this.showRunProblem(err.error);
          this.busy.set(false);
        }
      });
  }

  /** Stage 21 — an error from Train / Run once / Test, as a card in the Problems panel. */
  private showRunProblem(info: GraphErrorInfo): void {
    this.runErrorInfo.set(info);
    this.problemsOpen.set(true);
    if (info.node_id || info.edge) this.focusProblem(this.asProblem(info));
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
    if (this.dataset() === 'text') {
      this.overlayBusy.set(false);
      if (this.textData().trim()) this.previewData();
      return;
    }
    this.previewData();
  }

  // ------------------------------------------------ Your text (Stage 18)
  setTextData(v: string): void {
    if (v.length > MAX_TEXT_CHARS) {
      this.textError.set(`That is ${v.length.toLocaleString('en-US')} characters; Your text takes up to 1,000,000.`);
      return;
    }
    this.textError.set(null);
    this.textData.set(v);
    this.trainResult.set(null);
  }

  applyText(): void {
    if (!this.textData().trim()) {
      this.textError.set('Paste or upload a text first (or use the example).');
      return;
    }
    this.previewData();
  }

  onTextFile(ev: Event): void {
    const input = ev.target as HTMLInputElement;
    const file = input.files?.[0];
    input.value = '';
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      this.setTextData(String(reader.result ?? ''));
      if (!this.textError()) this.applyText();
    };
    reader.readAsText(file);
  }

  useExampleText(): void {
    this.api.textExample().subscribe({
      next: (r) => {
        this.setTextData(r.text);
        this.applyText();
      },
      error: (e) => this.textError.set(this.errMsg(e)),
    });
  }

  /** Give every character of the text a token of its own: the Embedding and the LM head
   * get the text's number of different characters. */
  fitVocabulary(): void {
    const v = Math.max(2, this.textDistinct());
    this.nodes.update((ns) =>
      ns.map((n) =>
        n.type === 'LMHead'
          ? { ...n, params: { ...n.params, vocab: v } }
          : n.type === 'Embedding'
            ? { ...n, params: { ...n.params, num_embeddings: v } }
            : n,
      ),
    );
    this.trainResult.set(null);
    this.showNotice('info', `Vocabulary set to ${v}: every character of your text has its own token.`);
    this.previewData();
  }

  /** Your GPT writes: continue the prompt with the current weights. */
  onGenerate(): void {
    const runId = this.currentRunId();
    if (!runId) return;
    this.genBusy.set(true);
    this.genError.set(null);
    this.genSeed += 1;
    this.api
      .generate(this.toSpec(), runId, this.genPrompt(), this.genLength(), this.genTemperature(), this.genSeed)
      .subscribe({
        next: (r) => {
          this.genResult.set(r);
          this.genBusy.set(false);
        },
        error: (err) => {
          this.genError.set(this.errMsg(err));
          this.genBusy.set(false);
        },
      });
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
    this.runErrorInfo.set(null);
    this.onceResult.set(null); // Stage 19: a forward pass of the old graph says nothing about the new one
    this.onceError.set(null);
  }

  /** The structured part of a 422 (where the problem is, what to do), or just the message. */
  private errInfo(e: unknown): GraphErrorInfo {
    const err = e as { error?: { error?: GraphErrorInfo } };
    return err?.error?.error ?? { message: this.errMsg(e), node_id: null, edge: null, hint: null };
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
  /** Open an architecture from the library. A flagship's real size comes from its mini entry
   * (`from`): the published model, sized on the meta device — inspect only (Stage 17). */
  loadArchitecture(id: string, from?: ArchitectureInfo): void {
    if (this.busy() || !this.confirmDiscard()) return;
    this.busy.set(true);
    this.runError.set(null);
    this.proposals.set([]);
    this.api.architecture(id).subscribe({
      next: (spec) => {
        const info = this.architectures().find((a) => a.id === id);
        const real = !!from && from.real_id === id;
        const t = spec.train ?? {};
        const trainExtra: Record<string, unknown> = {};
        for (const [k, v] of Object.entries(t)) if (!UI_TRAIN_KEYS.includes(k)) trainExtra[k] = v;
        // Stage 17: a template's connection settings come along (e.g. the LM head reads the last token)
        const settings = new Map((spec.edge_settings ?? []).map((s) => [`${s.source}->${s.target}`, s]));
        const name = real ? (from!.real_name ?? id) : (info?.name ?? id);
        this.busy.set(false);
        this.startFresh({
          name,
          description: real ? `${from!.description} Real size: ${from!.hf_url ?? ''}` : (info?.description ?? ''),
          createdAt: null,
          nodes: layoutGraph(spec.nodes, spec.edges),
          edges: spec.edges.map(([source, target]) => {
            const s = settings.get(`${source}->${target}`);
            return s ? { source, target, transform: s.transform, params: { ...(s.params ?? {}) } } : { source, target };
          }),
          train: {
            // the template brings its recommended schedule; your data choice stays, unless the
            // template needs its own task (the flagships learn "next token")
            ...this.uiTrain(),
            ...(t.dataset && t.dataset !== 'synthetic' ? { dataset: t.dataset } : {}),
            ...(t.text ? { text: this.textData() || t.text } : {}),
            epochs: t.epochs ?? this.epochs(),
            lr: t.lr ?? this.lr(),
            batch_size: t.batch_size ?? this.batchSize(),
            samples: t.samples ?? this.samples(),
          },
          trainExtra,
          customText: this.customText(),
          view: { pan_x: 48, pan_y: 24, zoom: 0.8 },
        });
        this.showNotice(
          'info',
          real
            ? `Real size: ${name}, ${(from!.real_params ?? 0).toLocaleString('en-US')} parameters, built on the "meta" device — every layer and shape, no memory. Inspect it here; the mini version trains.`
            : `New project from the «${name}» template.`,
        );
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
      return;
    }
    // Delete / Backspace removes the selected connection or node — unless you are typing
    if (ev.key === 'Delete' || ev.key === 'Backspace') {
      const el = ev.target as HTMLElement | null;
      const typing =
        !!el &&
        (['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName) ||
          el.isContentEditable ||
          !!el.closest?.('ion-input, ion-searchbar, ion-textarea, ion-select'));
      if (typing || this.projectsOpen() || this.dataModalOpen() || this.exportModalOpen()) return;
      if (this.selectedEdge()) {
        ev.preventDefault();
        this.deleteSelectedEdge();
      } else if (this.selectedId()) {
        ev.preventDefault();
        this.deleteSelected();
      }
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
    this.textData.set(typeof t.text === 'string' ? t.text : '');
    this.textError.set(null);
    this.genResult.set(null);
    this.genError.set(null);
    const vf = num(t.val_fraction, DEFAULT_VAL_FRACTION);
    this.valFraction.set(vf >= 0 && vf <= 0.5 ? vf : DEFAULT_VAL_FRACTION);
    const tf = num(t.test_fraction, DEFAULT_TEST_FRACTION);
    this.testFraction.set(tf >= 0 && tf <= 0.5 ? tf : DEFAULT_TEST_FRACTION);
    this.loadedWeights.set(null);
    this.testResult.set(null);
    this.testError.set(null);
    this.saveCheckpoint.set(t.checkpoint === true);
    this.resumeCheckpoint.set(t.resume === true);
    this.device.set(t.device === 'cuda' || t.device === 'auto' ? t.device : 'cpu');
    this.trainExtra.set({ ...s.trainExtra });
    this.customText.set(s.customText);
    this.customError.set(null);
    this.pan.set({ x: s.view.pan_x, y: s.view.pan_y });
    this.zoom.set(s.view.zoom);
    this.selectedId.set(null);
    this.selectedEdge.set(null);
    this.innerSel.set(null);
    this.connectFrom.set(null);
    this.buildResult.set(null);
    this.trainResult.set(null);
    this.runError.set(null);
    this.runErrorInfo.set(null);
    this.analysis.set(null);
    this.analysisError.set(null);
    this.proposals.set([]);
    this.liveLoss.set([]);
    this.liveAcc.set([]);
    this.liveValAcc.set([]);
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
