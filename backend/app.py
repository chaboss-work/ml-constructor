"""FastAPI entry point for the ML Constructor backend.

Run from the ``backend/`` directory:

    .venv\\Scripts\\python.exe -m uvicorn app:app --port 8050 --reload

Endpoints:
    GET  /health   liveness + layer count
    GET  /layers   the layer palette the UI renders
    POST /build    graph spec -> model summary (shapes, param counts) or 422
    POST /train    graph spec -> tiny training run -> loss curve + accuracy
    POST /test     the final test: a run's weights on the test rows (Stage 16)
    POST /generate continue a prompt with weights trained on your text (Stage 18)
    GET  /runs/{id}/weights   download a run's weights (.pt)
    POST /weights/upload      load a weights file into the graph (kept as a run)
    POST /projects/normalize   any project file version -> current format + warnings
    GET/PUT/DELETE /projects[/{slug}]   the optional server-side project store
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import urllib.parse

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

import mlconstructor as mlc
from mlconstructor.projects import MAX_PROJECT_BYTES
from mlconstructor.schema import (
    ArchitectureInfo,
    BuildResult,
    ConnectionOptions,
    ConnectionOptionsRequest,
    DataPreview,
    DatasetInfo,
    ExportResult,
    GraphSpec,
    LayerInfo,
    ProjectLoadResult,
    ProjectSaveResult,
    ProjectSummary,
    GenerateRequest,
    GenerateResult,
    SuggestResult,
    TestRequest,
    TestResult,
    TrainResult,
    WeightsInfo,
    WeightsUpload,
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


def _error_info(exc: Exception) -> dict:
    if isinstance(exc, ShapeError):
        return exc.to_detail()
    return {"message": str(exc), "node_id": None, "edge": None, "hint": None}


def _unprocessable(exc: Exception) -> JSONResponse:
    """422 for a mistake in the graph: ``detail`` is the message (as before) and,
    since Stage 14, ``error`` says where it is (node / connection) and what to do."""
    return JSONResponse(status_code=422, content={"detail": str(exc), "error": _error_info(exc)})


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": mlc.__version__,
        "layers": len(mlc.LAYER_REGISTRY),
        "cuda_available": mlc.cuda_available(),  # no when the GPUs were hidden at start (see mlconstructor.gpu)
        "server_projects": mlc.server_projects_enabled(),
        "api": mlc.API_LEVEL,
    }


@app.get("/layers", response_model=list[LayerInfo])
def layers() -> list[LayerInfo]:
    return mlc.catalog()


@app.post("/build", response_model=BuildResult)
def build(spec: GraphSpec) -> BuildResult:
    """Shapes and parameter counts. Built on the meta device (Stage 17): no weight is
    made, so even a 671-billion-parameter flagship is sized in seconds."""
    try:
        _model, result = mlc.build_model(spec, meta=True)
        result.trainable = result.total_params <= mlc.MAX_TRAIN_PARAMS
        result.max_train_params = mlc.MAX_TRAIN_PARAMS
        return result
    except _USER_ERRORS as exc:
        return _unprocessable(exc)


def _prepare_training(spec: GraphSpec):
    """Build the model with reproducible starting weights (Stage 16: the run's seed)
    and load the weights to start from, if any. Returns (model, in_shape, train spec)."""
    mlc.check_trainable(spec)  # sized on the meta device first: a real-scale flagship is refused here
    model, result = mlc.build_model(spec, seed=spec.train.seed)
    if not result.layers or result.layers[0].out_shape is None:
        raise ShapeError("graph has no Input node to take a shape from")
    in_shape = tuple(result.layers[0].out_shape)
    return model, in_shape, mlc.start_from_run(model, spec.train)


@app.post("/train", response_model=TrainResult)
def train(spec: GraphSpec) -> TrainResult:
    try:
        model, in_shape, train_spec = _prepare_training(spec)
        result = mlc.train_model(model, in_shape, train_spec)
        return mlc.save_trained(model, in_shape, train_spec, result)
    except _USER_ERRORS as exc:
        return _unprocessable(exc)


@app.post("/train/stream")
def train_stream(spec: GraphSpec) -> StreamingResponse:
    """Live training (Stage 11): stream newline-delimited JSON progress events —
    ``start`` once, a ``step`` per optimizer step, an ``epoch`` per finished epoch,
    then ``done`` with the full TrainResult. A bad graph still fails fast with 422."""
    try:
        model, in_shape, train_spec = _prepare_training(spec)
    except _USER_ERRORS as exc:
        return _unprocessable(exc)

    def events():
        try:
            for event in mlc.train_iter(model, in_shape, train_spec):
                if event["type"] == "done":  # keep the trained weights first (Stage 16): the result names them
                    result = mlc.save_trained(model, in_shape, train_spec, event["result"])
                    event = {"type": "done", "result": result.model_dump()}
                yield json.dumps(event) + "\n"
        except _USER_ERRORS as exc:  # a training-time error -> stream it, don't 500
            yield json.dumps({"type": "error", "detail": str(exc), "error": _error_info(exc)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/suggest", response_model=SuggestResult)
def suggest(spec: GraphSpec) -> SuggestResult:
    """Smart adapters: propose nodes (e.g. Flatten) to reconcile mismatched
    connections. Never fails on a shape mismatch — it returns the fixes."""
    try:
        return SuggestResult(ok=True, proposals=mlc.suggest_for_graph(spec))
    except _USER_ERRORS as exc:
        return _unprocessable(exc)


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
        return _unprocessable(exc)


@app.post("/export/pytorch", response_model=ExportResult)
def export_pytorch(spec: GraphSpec, include_training: bool = True) -> ExportResult:
    """Standalone PyTorch source for this graph — a real ``nn.Module`` (optionally
    with a tiny training loop) that depends on nothing but torch (Stage 8)."""
    try:
        code = mlc.generate_pytorch(spec, include_training=include_training)
        return ExportResult(ok=True, filename="model.py", code=code)
    except _USER_ERRORS as exc:
        return _unprocessable(exc)


@app.post("/connections/options", response_model=ConnectionOptions)
def connections_options(req: ConnectionOptionsRequest) -> ConnectionOptions:
    """The connection inspector (Stage 14): for one edge, the shape arriving, what
    "auto" picks, and every transform that applies — with parameters, presets and
    whether the target accepts the result."""
    try:
        return mlc.connection_options(req.spec, req.source, req.target)
    except _USER_ERRORS as exc:
        return _unprocessable(exc)


# --- Final test & trained weights (Stage 16) --------------------------------------
@app.post("/test", response_model=TestResult)
def test(req: TestRequest) -> TestResult:
    """The final test: a run's weights on the rows kept out of training and out of
    the epoch-by-epoch validation, plus whether it is a fair check (same split)."""
    try:
        return mlc.test_run(req.spec, req.run_id)
    except mlc.RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except _USER_ERRORS as exc:
        return _unprocessable(exc)


@app.post("/generate", response_model=GenerateResult)
def generate(req: GenerateRequest) -> GenerateResult:
    """Stage 18 — your GPT writes: continue a prompt with a run's weights, trained on
    "Your text", one character at a time."""
    try:
        return mlc.generate(req.spec, req.run_id, req.prompt, req.length, req.temperature, req.seed)
    except mlc.RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except _USER_ERRORS as exc:
        return _unprocessable(exc)


@app.get("/text/example")
def text_example() -> dict:
    """Stage 18 — the example text for "Your text" (an original short story)."""
    return {"text": mlc.EXAMPLE_TEXT}


@app.get("/runs/{run_id}/weights")
def run_weights(run_id: str, name: str = "model") -> Response:
    """Download a run's weights as a ``.pt`` file (``torch.load(weights_only=True)``
    reads it; so does ``load_weights()`` in the exported ``model.py``)."""
    try:
        path = mlc.run_path(run_id)
    except mlc.RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    with open(path, "rb") as f:
        raw = f.read()
    base = (name or "model").strip()[:80] or "model"
    ascii_base = "".join(c if c.isascii() and (c.isalnum() or c in "-_ .") else "_" for c in base).strip()
    if not any(c.isalnum() for c in ascii_base):  # e.g. a Cyrillic name: the UTF-8 filename* carries it
        ascii_base = "model"
    quoted = urllib.parse.quote(base + "-weights.pt")
    disposition = f'attachment; filename="{ascii_base}-weights.pt"; filename*=UTF-8' + "''" + quoted
    return Response(content=raw, media_type="application/octet-stream", headers={"Content-Disposition": disposition})


# base64 makes an upload a third bigger than the file; the graph rides along.
_MAX_UPLOAD_JSON = mlc.MAX_WEIGHTS_BYTES * 4 // 3 + MAX_PROJECT_BYTES


@app.post("/weights/upload", response_model=WeightsInfo)
async def weights_upload(request: Request) -> WeightsInfo:
    """Load a weights file (``.pt``, base64 in ``data``) into the graph in ``spec``:
    it must fit; it is read safely (tensors only) and kept as a run to test or to
    train from."""
    too_big = f"weights file is over {mlc.MAX_WEIGHTS_BYTES // 2**20} MB"
    raw = await _read_json(request, _MAX_UPLOAD_JSON, too_big)
    try:
        req = WeightsUpload.model_validate(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"not a weights upload: {exc}")
    try:
        data = base64.b64decode(req.data, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=422, detail="the weights file did not arrive intact (bad base64)")
    if len(data) > mlc.MAX_WEIGHTS_BYTES:
        raise HTTPException(status_code=413, detail=too_big)
    try:
        return await run_in_threadpool(mlc.adopt_weights, req.spec, data)
    except _USER_ERRORS as exc:
        return _unprocessable(exc)


# --- Projects: open any format version, optional server-side store (Stage 13) --
async def _read_json(request: Request, limit: int, too_big: str) -> object:
    """The request body as JSON, refusing anything over ``limit`` bytes before it
    is parsed."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise HTTPException(status_code=413, detail=too_big)
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise HTTPException(status_code=413, detail=too_big)
    try:
        return json.loads(bytes(body).decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise HTTPException(status_code=422, detail="not a valid JSON file")


async def _read_project_json(request: Request) -> object:
    """A project file from the body, within the project size limit. Raw JSON on
    purpose: older formats (and a bare graph spec) must be migrated *before* they
    are validated against the current model."""
    return await _read_json(request, MAX_PROJECT_BYTES, f"project file is over {MAX_PROJECT_BYTES // 2**20} MB")


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
