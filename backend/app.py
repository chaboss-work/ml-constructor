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

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import mlconstructor as mlc
from mlconstructor.schema import BuildResult, GraphSpec, LayerInfo, SuggestResult, TrainResult
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
    return {"status": "ok", "version": mlc.__version__, "layers": len(mlc.LAYER_REGISTRY)}


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


@app.post("/suggest", response_model=SuggestResult)
def suggest(spec: GraphSpec) -> SuggestResult:
    """Smart adapters: propose nodes (e.g. Flatten) to reconcile mismatched
    connections. Never fails on a shape mismatch — it returns the fixes."""
    try:
        return SuggestResult(ok=True, proposals=mlc.suggest_for_graph(spec))
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
