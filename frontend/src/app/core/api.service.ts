import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import { BuildResult, GraphSpec, LayerInfo, SuggestResult, TrainResult } from './graph.models';

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

  health(): Observable<{ status: string; version: string; layers: number }> {
    return this.http.get<{ status: string; version: string; layers: number }>(`${API_BASE}/health`);
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
}
