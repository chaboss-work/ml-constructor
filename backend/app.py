"""FastAPI entry point for the ML Constructor backend.

Run from the ``backend/`` directory:

    .venv\\Scripts\\python.exe -m uvicorn app:app --port 8050 --reload

Endpoints:
    GET  /health   liveness + layer count
    GET  /layers   the layer palette the UI renders
    POST /build    graph spec -> model summary (shapes, param counts) or 422
    POST /train    graph spec -> tiny training run -> loss curve + accuracy
"""

from __future__ import annotations

import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

import mlconstructor as mlc
from mlconstructor.schema import (
    ArchitectureInfo,
    BuildResult,
    DataPreview,
    DatasetInfo,
    ExportResult,
    GraphSpec,
    LayerInfo,
    SuggestResult,
    TrainResult,
)
from mlconstructor.shapes import ShapeError

app = FastAPI(title="ML Constructor Backend", version=mlc.__version__)

# v1 dev: allow any origin so the Angular frontend (any port / host) can call in.
# Tighten to the real frontend origin before any public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_USER_ERRORS = (ShapeError, KeyError, ValueError)


@app.get("/health")
def health() -> dict:
    import torch  # is_available() queries the driver; it does not open a CUDA context

    return {
        "status": "ok",
        "version": mlc.__version__,
        "layers": len(mlc.LAYER_REGISTRY),
        "cuda_available": bool(torch.cuda.is_available()),
    }


@app.get("/layers", response_model=list[LayerInfo])
def layers() -> list[LayerInfo]:
    return mlc.catalog()


@app.post("/build", response_model=BuildResult)
def build(spec: GraphSpec) -> BuildResult:
    try:
        _model, result = mlc.build_model(spec)
        return result
    except _USER_ERRORS as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/train", response_model=TrainResult)
def train(spec: GraphSpec) -> TrainResult:
    try:
        model, result = mlc.build_model(spec)
        if not result.layers or result.layers[0].out_shape is None:
            raise ShapeError("graph has no Input node to take a shape from")
        in_shape = tuple(result.layers[0].out_shape)
        return mlc.train_model(model, in_shape, spec.train)
    except _USER_ERRORS as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/train/stream")
def train_stream(spec: GraphSpec) -> StreamingResponse:
    """Live training (Stage 11): stream newline-delimited JSON progress events —
    ``start`` once, a ``step`` per optimizer step, an ``epoch`` per finished epoch,
    then ``done`` with the full TrainResult. A bad graph still fails fast with 422."""
    try:
        model, result = mlc.build_model(spec)
        if not result.layers or result.layers[0].out_shape is None:
            raise ShapeError("graph has no Input node to take a shape from")
        in_shape = tuple(result.layers[0].out_shape)
    except _USER_ERRORS as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    def events():
        try:
            for event in mlc.train_iter(model, in_shape, spec.train):
                if event["type"] == "done":
                    event = {"type": "done", "result": event["result"].model_dump()}
                yield json.dumps(event) + "\n"
        except _USER_ERRORS as exc:  # a training-time error -> stream it, don't 500
            yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/suggest", response_model=SuggestResult)
def suggest(spec: GraphSpec) -> SuggestResult:
    """Smart adapters: propose nodes (e.g. Flatten) to reconcile mismatched
    connections. Never fails on a shape mismatch — it returns the fixes."""
    try:
        return SuggestResult(ok=True, proposals=mlc.suggest_for_graph(spec))
    except _USER_ERRORS as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/architectures", response_model=list[ArchitectureInfo])
def architectures() -> list[ArchitectureInfo]:
    """The library of ready public architectures (MLP, CNN, ResNet, GPT, …) the
    editor offers to drop onto the canvas as full, editable graphs."""
    return mlc.list_architectures()


@app.get("/architectures/{arch_id}", response_model=GraphSpec)
def architecture(arch_id: str) -> GraphSpec:
    """The full graph for one architecture — plain, editable nodes and edges."""
    try:
        return mlc.get_architecture(arch_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/datasets", response_model=list[DatasetInfo])
def datasets() -> list[DatasetInfo]:
    """The datasets the trainer can run on (Stage 8). v1 = self-contained synthetic
    tasks (no download, no GPU); real datasets plug in behind the same contract."""
    return mlc.list_datasets()


@app.post("/data/preview", response_model=DataPreview)
def data_preview(spec: GraphSpec, rows: int = 6) -> DataPreview:
    """A few real rows of the training set for this graph, plus its format — the
    exact data training will see (Stage 8's data modal)."""
    try:
        return mlc.data_preview(spec, rows)
    except _USER_ERRORS as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/export/pytorch", response_model=ExportResult)
def export_pytorch(spec: GraphSpec, include_training: bool = True) -> ExportResult:
    """Standalone PyTorch source for this graph — a real ``nn.Module`` (optionally
    with a tiny training loop) that depends on nothing but torch (Stage 8)."""
    try:
        code = mlc.generate_pytorch(spec, include_training=include_training)
        return ExportResult(ok=True, filename="model.py", code=code)
    except _USER_ERRORS as exc:
        raise HTTPException(status_code=422, detail=str(exc))


# --- Serve the built frontend so one process hosts the whole app on :8050 -----
# The API routes above are registered first, so they win over this catch-all
# mount; everything else (index.html, JS/CSS chunks, SPA routes) is served from
# the Angular production build. Requires `ng build` in ../frontend first.
_FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "dist", "mlconstructor", "browser"
)
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
