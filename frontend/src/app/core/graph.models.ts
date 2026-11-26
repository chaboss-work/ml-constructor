/**
 * TypeScript mirror of the backend contract (mlconstructor/schema.py).
 * Keep these in sync with the FastAPI models.
 */

export type ParamType = 'int' | 'float' | 'bool' | 'str' | 'intlist';

export interface ParamDef {
  name: string;
  type: ParamType;
  default: unknown;
  label?: string | null;
  min?: number | null;
  max?: number | null;
  help?: string | null;
  /** Stage 14 — a fixed set of allowed values (rendered as a selector). */
  choices?: string[] | null;
}

export interface LayerInfo {
  type: string;
  category: string;
  label: string;
  doc: string;
  params: ParamDef[];
}

/** A node as held by the editor (adds canvas position). */
export interface EditorNode {
  id: string;
  type: string;
  x: number;
  y: number;
  params: Record<string, unknown>;
}

export interface EditorEdge {
  source: string;
  target: string;
  /** Stage 14 — how this connection reshapes its data: "auto" (default), "none" or a transform. */
  transform?: string;
  params?: Record<string, unknown>;
}

/** Stage 14 — a connection's transform, as sent to the backend (edges without one use "auto"). */
export interface EdgeSetting {
  source: string;
  target: string;
  transform: string;
  params: Record<string, unknown>;
}

/** Stage 14 — what one connection does in the built model. */
export interface EdgeSummary {
  source: string;
  target: string;
  /** the transform in use ("none" = passed through unchanged) */
  transform: string;
  /** true when "auto" picked it */
  auto: boolean;
  /** short text for the wire ("" = nothing to show) */
  label: string;
  params: Record<string, unknown>;
  in_shape: number[];
  out_shape: number[];
  num_params: number;
}

/** Stage 14 — one choice in a connection's transform selector. */
export interface ConnectionOption {
  transform: string;
  label: string;
  description: string;
  learnable: boolean;
  params: ParamDef[];
  /** suggested parameter sets, best first */
  presets: Record<string, unknown>[];
  /** whether the target accepts the result (with the first preset) */
  fits: boolean;
  out_shape: number[] | null;
  note: string | null;
}

/** Stage 14 — everything the connection inspector shows for one edge. */
export interface ConnectionOptions {
  source: string;
  target: string;
  in_shape: number[] | null;
  current: string;
  current_params: Record<string, unknown>;
  auto_transform: string | null;
  auto_params: Record<string, unknown>;
  auto_out_shape: number[] | null;
  problem: string | null;
  options: ConnectionOption[];
}

/** Stage 14 — the structured part of a 422 response: where the problem is and what to do. */
export interface GraphErrorInfo {
  message: string;
  node_id: string | null;
  edge: string[] | null;
  hint: string | null;
}

export interface TrainCfg {
  dataset: string;
  epochs: number;
  lr: number;
  batch_size: number;
  samples: number;
  seed: number;
  /** Stage 9 — rows for the "custom" dataset: each is [f1, …, fk, label]. */
  custom_rows?: number[][];
  /** Stage 11 — save best weights during training / resume from the last checkpoint. */
  checkpoint?: boolean;
  resume?: boolean;
  /** Stage 12 — where to train; the backend defaults to "cpu" (GPU is opt-in). */
  device?: 'cpu' | 'cuda' | 'auto';
  /** Stage 15 — share of rows held out for validation (0 = off; the backend defaults to 0.2). */
  val_fraction?: number;
  /** Stage 16 — share of rows kept for the final test (0 = none; the backend defaults to 0.1). */
  test_fraction?: number;
  /** Stage 16 — start from the weights of an earlier run (a finished training or loaded file). */
  init_run?: string | null;
}

/** Stage 11 — one streamed progress event from POST /train/stream. */
export interface TrainEvent {
  type: 'start' | 'step' | 'epoch' | 'done' | 'error';
  step?: number;
  loss?: number;
  epoch?: number;
  acc?: number;
  /** Stage 15 — loss and accuracy on the held-out rows after an epoch (null when none). */
  val_acc?: number | null;
  val_loss?: number | null;
  device?: string;
  resumed?: boolean;
  total_params?: number;
  samples?: number;
  /** Stage 15 — how the rows split: trained on vs held out. */
  train_samples?: number;
  val_samples?: number;
  /** Stage 16 — rows kept for the final test. */
  test_samples?: number;
  num_classes?: number;
  result?: TrainResult;
  detail?: string;
}

/** The payload sent to /build and /train. */
export interface GraphSpec {
  nodes: { id: string; type: string; params: Record<string, unknown> }[];
  edges: string[][];
  train: Partial<TrainCfg>;
  /** Stage 14 — per-connection transforms; edges not listed use "auto". */
  edge_settings?: EdgeSetting[];
}

export interface LayerSummary {
  id: string;
  type: string;
  in_shape: number[] | null;
  out_shape: number[] | null;
  num_params: number;
}

export interface BuildResult {
  ok: boolean;
  order: string[];
  layers: LayerSummary[];
  total_params: number;
  /** Stage 14 — every connection with its transform. Optional: older backends omit it. */
  edges?: EdgeSummary[];
}

export interface TrainResult {
  ok: boolean;
  loss_curve: number[];
  /** Accuracy on the training rows after each epoch (Stage 8). Optional: older backends omit it. */
  acc_curve?: number[];
  /** Loss and accuracy on the training rows at the end. */
  final_loss: number;
  final_accuracy: number;
  steps: number;
  device: string;
  total_params: number;
  /** Stage 11 — saved checkpoint key (or null), best accuracy (held-out when there is a
   * validation part), resumed flag. */
  checkpoint?: string | null;
  best_accuracy?: number | null;
  resumed?: boolean;
  /** Stage 15 — the held-out validation part (rows the network never trained on). */
  val_acc_curve?: number[];
  val_loss_curve?: number[];
  final_val_accuracy?: number | null;
  final_val_loss?: number | null;
  train_samples?: number | null;
  val_samples?: number;
  /** "generalises" | "overfits" | "underfits" | "unknown", and why in plain words. */
  fit?: string | null;
  fit_note?: string | null;
  warnings?: string[];
  /** Stage 16 — rows kept for the final test, and the id the trained weights are kept under. */
  test_samples?: number;
  run_id?: string | null;
}

/** Stage 16 — the final test: a run's weights on the rows used for nothing else. */
export interface TestResult {
  ok: boolean;
  run_id: string;
  accuracy: number;
  loss: number;
  samples: number;
  val_accuracy?: number | null;
  train_accuracy?: number | null;
  val_samples?: number;
  train_samples?: number;
  /** true: trained on this very split; false: on other data / another split; null: unknown. */
  same_split?: boolean | null;
  note?: string | null;
  warnings?: string[];
}

/** Stage 16 — a weights file loaded into the graph (kept on the server as a run). */
export interface WeightsInfo {
  ok: boolean;
  run_id: string;
  tensors: number;
  params: number;
  matched: 'names' | 'order' | string;
  data_known?: boolean;
  same_split?: boolean | null;
  metrics?: Record<string, unknown>;
  note?: string | null;
}

/** Smart adapters (Stage 5): a node to insert so a connection's shapes line up. */
export interface AdapterProposal {
  source_id: string;
  target_id: string;
  adapter_type: string;
  params: Record<string, unknown>;
  reason: string;
  in_shape: number[];
  out_shape: number[];
}

export interface SuggestResult {
  ok: boolean;
  proposals: AdapterProposal[];
}

/** Architecture library (Stage 7): a ready public net the editor can drop onto
 * the canvas as a full, editable graph. The graph itself comes from
 * GET /architectures/{id} (a plain GraphSpec). */
export interface ArchitectureInfo {
  id: string;
  name: string;
  description: string;
  category: string;
  tags: string[];
  input_shape: number[];
  node_count: number;
}

/** Datasets & data preview (Stage 8). */
export interface DatasetInfo {
  id: string;
  name: string;
  description: string;
  kind: string;
  /** Fixed input shape a real dataset needs (MNIST [1,28,28]); null = adaptive. */
  input_shape?: number[] | null;
}

export interface DataSampleRow {
  label: number;
  values: number[];
  truncated: boolean;
  total: number;
  /** Stage 15 — "train", "validation" (held out) or "test" (kept for the final test, Stage 16). */
  split?: string;
}

export interface DataPreview {
  dataset: string;
  dataset_name: string;
  description: string;
  input_shape: number[];
  dtype: string;
  num_classes: number;
  num_samples: number;
  feature_count: number;
  rows: DataSampleRow[];
  /** Stage 15 — rows trained on vs held out for validation (+ the final test, Stage 16). */
  train_samples?: number | null;
  val_samples?: number;
  test_samples?: number;
}

/** Export (Stage 8): standalone PyTorch source for the current graph. */
export interface ExportResult {
  ok: boolean;
  language: string;
  filename: string;
  code: string;
}

/** Projects (Stage 13): the versioned `.mlc.json` file — the graph with node
 * positions, the training setup and metadata. Mirrors schema.ProjectFile. */
export interface ProjectNode {
  id: string;
  type: string;
  params: Record<string, unknown>;
  /** Where the node sits on the canvas; null/absent = lay it out automatically. */
  position?: { x: number; y: number } | null;
}

export interface ProjectView {
  pan_x: number;
  pan_y: number;
  zoom: number;
}

export interface ProjectFile {
  format: 'mlconstructor.project';
  version: number;
  name: string;
  description?: string;
  created_at?: string | null;
  updated_at?: string | null;
  graph: { nodes: ProjectNode[]; edges: string[][]; edge_settings?: EdgeSetting[] };
  train: Partial<TrainCfg> & Record<string, unknown>;
  editor?: { custom_text?: string; view?: ProjectView };
}

/** Something worth telling the user when a project opens (it still opens). */
export interface ProjectWarning {
  code: string;
  message: string;
  node_id?: string | null;
}

export interface ProjectLoadResult {
  ok: boolean;
  project: ProjectFile;
  /** The format version the file was upgraded from (null = it already was current). */
  migrated_from?: number | null;
  warnings: ProjectWarning[];
}

/** A project in the optional server-side store. */
export interface ProjectSummary {
  slug: string;
  name: string;
  updated_at?: string | null;
  node_count: number;
}

export interface ProjectSaveResult {
  ok: boolean;
  slug: string;
  updated_at: string;
}

/** GET /health. `server_projects` (Stage 13) is absent on older backends. */
export interface HealthInfo {
  status: string;
  version: string;
  layers: number;
  cuda_available?: boolean;
  server_projects?: boolean;
  /** Stage 14 — the latest stage whose API this backend serves (absent on older ones). */
  api?: number;
}
