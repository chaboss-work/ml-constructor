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
