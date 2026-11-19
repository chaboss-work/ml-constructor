import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import {
  ArchitectureInfo,
  BuildResult,
  DataPreview,
  DatasetInfo,
  ExportResult,
  GraphSpec,
  LayerInfo,
  SuggestResult,
  TrainEvent,
  TrainResult,
} from './graph.models';

/**
 * Base URL of the PyTorch backend. It is served on port 8050 on the same host
 * that serves this app, so LAN access works too (backend must bind 0.0.0.0).
 * For a custom host, override `API_BASE`.
 */
const API_BASE = `http://${location.hostname || 'localhost'}:8050`;

@Injectable({ providedIn: 'root' })
export class ApiService {
  private http = inject(HttpClient);
  readonly base = API_BASE;

  health(): Observable<{ status: string; version: string; layers: number; cuda_available?: boolean }> {
    return this.http.get<{ status: string; version: string; layers: number; cuda_available?: boolean }>(
      `${API_BASE}/health`,
    );
  }

  layers(): Observable<LayerInfo[]> {
    return this.http.get<LayerInfo[]>(`${API_BASE}/layers`);
  }

  build(spec: GraphSpec): Observable<BuildResult> {
    return this.http.post<BuildResult>(`${API_BASE}/build`, spec);
  }

  train(spec: GraphSpec): Observable<TrainResult> {
    return this.http.post<TrainResult>(`${API_BASE}/train`, spec);
  }

  suggest(spec: GraphSpec): Observable<SuggestResult> {
    return this.http.post<SuggestResult>(`${API_BASE}/suggest`, spec);
  }

  /** Architecture library (Stage 7): the menu of ready public architectures. */
  architectures(): Observable<ArchitectureInfo[]> {
    return this.http.get<ArchitectureInfo[]>(`${API_BASE}/architectures`);
  }

  /** The full, editable graph for one ready architecture. */
  architecture(id: string): Observable<GraphSpec> {
    return this.http.get<GraphSpec>(`${API_BASE}/architectures/${id}`);
  }

  /** Datasets the trainer can run on (Stage 8). */
  datasets(): Observable<DatasetInfo[]> {
    return this.http.get<DatasetInfo[]>(`${API_BASE}/datasets`);
  }

  /** A few real rows of the training set for a graph, plus its format (Stage 8). */
  dataPreview(spec: GraphSpec, rows = 6): Observable<DataPreview> {
    return this.http.post<DataPreview>(`${API_BASE}/data/preview?rows=${rows}`, spec);
  }

  /** Standalone PyTorch source for a graph (Stage 8). */
  exportPytorch(spec: GraphSpec, includeTraining = true): Observable<ExportResult> {
    return this.http.post<ExportResult>(
      `${API_BASE}/export/pytorch?include_training=${includeTraining}`,
      spec,
    );
  }

  /** Live training (Stage 11): POST the graph and stream NDJSON progress events,
   * invoking `onEvent` for each. Rejects with `{status, fallback, detail}` — set
   * `fallback` when the backend has no streaming endpoint (404), so the caller can
   * fall back to the plain `train()`. */
  async trainStream(spec: GraphSpec, onEvent: (e: TrainEvent) => void): Promise<void> {
    const resp = await fetch(`${API_BASE}/train/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(spec),
    });
    if (!resp.ok || !resp.body) {
      let detail = '';
      try {
        detail = (await resp.json())?.detail ?? '';
      } catch {
        /* no JSON body */
      }
      throw { status: resp.status, fallback: resp.status === 404, detail };
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (line) onEvent(JSON.parse(line) as TrainEvent);
      }
    }
  }
}
