"""FastAPI entry point for the ML Constructor backend.

Run from the ``backend/`` directory:

    .venv\\Scripts\\python.exe -m uvicorn app:app --port 8050 --reload

Endpoints:
    GET  /health   liveness + layer count
    GET  /layers   the layer palette the UI renders
    POST /build    graph spec -> model summary (shapes, param counts) or 422
    POST /train    graph spec -> tiny training run -> loss curve + accuracy
    POST /projects/normalize   any project file version -> current format + warnings
    GET/PUT/DELETE /projects[/{slug}]   the optional server-side project store
"""

from __future__ import annotations

import json
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

import mlconstructor as mlc
from mlconstructor.projects import MAX_PROJECT_BYTES
from mlconstructor.schema import (
    ArchitectureInfo,
    BuildResult,
    DataPreview,
    DatasetInfo,
    ExportResult,
    GraphSpec,
    LayerInfo,
    ProjectLoadResult,
    ProjectSaveResult,
    ProjectSummary,
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
        "server_projects": mlc.server_projects_enabled(),
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


# --- Projects: open any format version, optional server-side store (Stage 13) --
async def _read_project_json(request: Request) -> object:
    """The request body as JSON, refusing anything over the project size limit
    before it is parsed. Raw JSON on purpose: older formats (and a bare graph spec)
    must be migrated *before* they are validated against the current model."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_PROJECT_BYTES:
        raise HTTPException(status_code=413, detail=f"project file is over {MAX_PROJECT_BYTES // 2**20} MB")
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_PROJECT_BYTES:
            raise HTTPException(status_code=413, detail=f"project file is over {MAX_PROJECT_BYTES // 2**20} MB")
    try:
        return json.loads(bytes(body).decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise HTTPException(status_code=422, detail="not a valid JSON file")


def _require_server_projects() -> None:
    if not mlc.server_projects_enabled():
        raise HTTPException(status_code=404, detail="server-side projects are switched off on this instance")


@app.post("/projects/normalize", response_model=ProjectLoadResult)
async def projects_normalize(request: Request) -> ProjectLoadResult:
    """Open a project file of any supported version — or a bare graph spec — and
    return it in the current format with warnings (migrated, unknown layers, broken
    edges, missing positions). Stateless: nothing is stored."""
    raw = await _read_project_json(request)
    try:
        return await run_in_threadpool(mlc.normalize_project, raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/projects", response_model=list[ProjectSummary])
def projects_list() -> list[ProjectSummary]:
    """Projects stored on this server, most recently saved first."""
    _require_server_projects()
    return mlc.list_projects()


@app.post("/projects", response_model=ProjectSaveResult)
async def projects_create(request: Request) -> ProjectSaveResult:
    """Store a new project on the server under a fresh name derived from its title."""
    _require_server_projects()
    raw = await _read_project_json(request)

    def _create() -> ProjectSaveResult:
        return mlc.create_project(mlc.normalize_project(raw).project)

    try:
        return await run_in_threadpool(_create)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/projects/{slug}", response_model=ProjectLoadResult)
def projects_get(slug: str) -> ProjectLoadResult:
    _require_server_projects()
    try:
        return mlc.load_project(slug)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=exc.args[0])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.put("/projects/{slug}", response_model=ProjectSaveResult)
async def projects_put(slug: str, request: Request) -> ProjectSaveResult:
    """Save (create or overwrite) a project on the server. The body is a project
    file of any supported version; it is stored in the current format."""
    _require_server_projects()
    raw = await _read_project_json(request)

    def _save() -> ProjectSaveResult:
        return mlc.save_project(slug, mlc.normalize_project(raw).project)

    try:
        return await run_in_threadpool(_save)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.delete("/projects/{slug}")
def projects_delete(slug: str) -> dict:
    _require_server_projects()
    try:
        mlc.delete_project(slug)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=exc.args[0])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"ok": True, "slug": slug}


# --- Serve the built frontend so one process hosts the whole app on :8050 -----
# The API routes above are registered first, so they win over this catch-all
# mount; everything else (index.html, JS/CSS chunks, SPA routes) is served from
# the Angular production build. Requires `ng build` in ../frontend first.
_FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "dist", "mlconstructor", "browser"
)
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
