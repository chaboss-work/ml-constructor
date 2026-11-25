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
}

/** Stage 11 — one streamed progress event from POST /train/stream. */
export interface TrainEvent {
  type: 'start' | 'step' | 'epoch' | 'done' | 'error';
  step?: number;
  loss?: number;
  epoch?: number;
  acc?: number;
  device?: string;
  resumed?: boolean;
  total_params?: number;
  samples?: number;
  num_classes?: number;
  result?: TrainResult;
  detail?: string;
}

/** The payload sent to /build and /train. */
export interface GraphSpec {
  nodes: { id: string; type: string; params: Record<string, unknown> }[];
  edges: string[][];
  train: Partial<TrainCfg>;
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
}

export interface TrainResult {
  ok: boolean;
  loss_curve: number[];
  /** Full-set accuracy after each epoch (Stage 8). Optional: older backends omit it. */
  acc_curve?: number[];
  final_loss: number;
  final_accuracy: number;
  steps: number;
  device: string;
  total_params: number;
  /** Stage 11 — saved checkpoint key (or null), best full-set accuracy, resumed flag. */
  checkpoint?: string | null;
  best_accuracy?: number | null;
  resumed?: boolean;
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
  graph: { nodes: ProjectNode[]; edges: string[][] };
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
}
