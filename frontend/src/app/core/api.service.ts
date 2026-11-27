import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import {
  ArchitectureInfo,
  BuildResult,
  ConnectionOptions,
  DataPreview,
  DatasetInfo,
  ExportResult,
  DrawResult,
  GenerateResult,
  GraphSpec,
  HealthInfo,
  RunResult,
  LayerInfo,
  ProjectFile,
  ProjectLoadResult,
  ProjectSaveResult,
  ProjectSummary,
  SuggestResult,
  TestResult,
  TrainEvent,
  TrainResult,
  WeightsInfo,
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

  health(): Observable<HealthInfo> {
    return this.http.get<HealthInfo>(`${API_BASE}/health`);
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

  /** The connection inspector (Stage 14): the shape arriving on one edge, what
   * "auto" picks, and every transform that applies (with presets and reasons). */
  connectionOptions(spec: GraphSpec, source: string, target: string): Observable<ConnectionOptions> {
    return this.http.post<ConnectionOptions>(`${API_BASE}/connections/options`, { spec, source, target });
  }

  // ---- final test & trained weights (Stage 16) ----
  /** The final test: a run's weights on the graph's test rows. */
  test(spec: GraphSpec, runId: string): Observable<TestResult> {
    return this.http.post<TestResult>(`${API_BASE}/test`, { spec, run_id: runId });
  }

  /** Load a weights file (.pt, base64) into the graph; the server keeps it as a run. */
  uploadWeights(spec: GraphSpec, data: string, name?: string): Observable<WeightsInfo> {
    return this.http.post<WeightsInfo>(`${API_BASE}/weights/upload`, { spec, data, name });
  }

  /** Stage 18 — continue a prompt with weights trained on your text. */
  generate(spec: GraphSpec, runId: string, prompt: string, length: number, temperature: number, seed: number) {
    return this.http.post<GenerateResult>(`${API_BASE}/generate`, {
      spec,
      run_id: runId,
      prompt,
      length,
      temperature,
      seed,
    });
  }

  /** Stage 20 — draw pictures with a run's weights (a diffusion model or a VAE). */
  draw(spec: GraphSpec, runId: string, count: number, label: number | null, steps: number, guidance: number,
       seed: number): Observable<DrawResult> {
    return this.http.post<DrawResult>(`${API_BASE}/generate/images`, {
      spec,
      run_id: runId,
      count,
      label,
      steps,
      guidance,
      seed,
    });
  }

  /** Stage 19 — one forward pass on random inputs: does the network run, what comes out. */
  run(spec: GraphSpec): Observable<RunResult> {
    return this.http.post<RunResult>(`${API_BASE}/run`, spec);
  }

  /** Stage 18 — the example text for Your text. */
  textExample(): Observable<{ text: string }> {
    return this.http.get<{ text: string }>(`${API_BASE}/text/example`);
  }

  /** Where a run's weights download from (a .pt file). */
  weightsUrl(runId: string, name: string): string {
    return `${API_BASE}/runs/${encodeURIComponent(runId)}/weights?name=${encodeURIComponent(name)}`;
  }

  // ---- projects (Stage 13) ----
  /** Open a project file of any supported version (or a bare graph spec): the
   * backend migrates it to the current format and reports warnings. */
  normalizeProject(raw: unknown): Observable<ProjectLoadResult> {
    return this.http.post<ProjectLoadResult>(`${API_BASE}/projects/normalize`, raw);
  }

  /** Projects stored on the server (when the instance enables them). */
  listProjects(): Observable<ProjectSummary[]> {
    return this.http.get<ProjectSummary[]>(`${API_BASE}/projects`);
  }

  getProject(slug: string): Observable<ProjectLoadResult> {
    return this.http.get<ProjectLoadResult>(`${API_BASE}/projects/${encodeURIComponent(slug)}`);
  }

  /** Store a new project on the server; it picks a free name from the title. */
  createProject(project: ProjectFile): Observable<ProjectSaveResult> {
    return this.http.post<ProjectSaveResult>(`${API_BASE}/projects`, project);
  }

  saveProject(slug: string, project: ProjectFile): Observable<ProjectSaveResult> {
    return this.http.put<ProjectSaveResult>(`${API_BASE}/projects/${encodeURIComponent(slug)}`, project);
  }

  deleteProject(slug: string): Observable<{ ok: boolean; slug: string }> {
    return this.http.delete<{ ok: boolean; slug: string }>(`${API_BASE}/projects/${encodeURIComponent(slug)}`);
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
