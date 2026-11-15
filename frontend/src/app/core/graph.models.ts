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
