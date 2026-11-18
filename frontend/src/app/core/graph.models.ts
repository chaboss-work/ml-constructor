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
