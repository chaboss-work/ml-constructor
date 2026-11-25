"""Projects: a versioned save format, migrations and an optional server store (Stage 13).

A project file (``.mlc.json``) holds the graph *with node positions*, the training
setup and some metadata. It carries a format version, and every older version is
migrated forward when it is opened, so files saved by earlier releases keep opening.
A bare graph spec — the JSON of ``/build`` or ``/architectures/{id}`` — counts as
"version 0" and opens too (its nodes have no positions, so the editor lays them out).

The optional server store keeps projects under ``backend/data/projects/`` so the same
project can be opened from another device of a self-hosted instance. Names are strict
slugs (no path traversal), files are size-capped and the store is bounded. Set the
environment variable ``MLC_SERVER_PROJECTS=0`` to switch the store off, or
``MLC_PROJECTS_DIR`` to keep it in another folder.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import ValidationError

from .datasets import canonical_dataset_id
from .registry import LAYER_REGISTRY
from .schema import (
    PROJECT_FORMAT,
    PROJECT_FORMAT_VERSION,
    ProjectFile,
    ProjectLoadResult,
    ProjectSaveResult,
    ProjectSummary,
    ProjectWarning,
)

_PROJECTS_DIR = os.environ.get("MLC_PROJECTS_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "projects"
)
_SUFFIX = ".mlc.json"
MAX_PROJECT_BYTES = 5 * 1024 * 1024  # one project file (custom rows included)
MAX_PROJECTS = 500  # files in the server store
MAX_NODES = 2000  # nodes in one graph
_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

# Russian -> Latin, so a project called "Моя сеть" gets the slug "moya-set".
_TRANSLIT = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
})


def server_projects_enabled() -> bool:
    return os.environ.get("MLC_SERVER_PROJECTS", "1") != "0"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Format versions & migrations
# --------------------------------------------------------------------------- #
def _v0_to_v1(raw: dict) -> dict:
    """A bare graph spec (``nodes`` / ``edges`` / ``train``, no positions) -> v1."""
    return {
        "format": PROJECT_FORMAT,
        "version": 1,
        "name": raw.get("name") or "Imported graph",
        "graph": {"nodes": raw.get("nodes", []), "edges": raw.get("edges", [])},
        "train": raw.get("train", {}) or {},
    }


#: ``version -> function`` upgrading a raw dict by exactly one version.
_MIGRATIONS: dict[int, Callable[[dict], dict]] = {0: _v0_to_v1}


def detect_version(raw: Any) -> int:
    if not isinstance(raw, dict):
        raise ValueError("not an ML Constructor project: expected a JSON object")
    if raw.get("format") == PROJECT_FORMAT:
        version = raw.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise ValueError(f"project file has an invalid format version: {version!r}")
        return version
    if "nodes" in raw:  # a bare graph spec
        return 0
    raise ValueError("not an ML Constructor project or graph (it has neither 'format' nor 'nodes')")


def migrate(raw: Any) -> tuple[dict, int | None]:
    """Upgrade ``raw`` to the current format. Returns the upgraded dict and the
    version it came from (``None`` if it already was current)."""
    version = detect_version(raw)
    if version > PROJECT_FORMAT_VERSION:
        raise ValueError(
            f"this project was saved by a newer version of ML Constructor (format {version}); "
            f"this one reads up to format {PROJECT_FORMAT_VERSION} — please update"
        )
    source = version
    while version < PROJECT_FORMAT_VERSION:
        raw = _MIGRATIONS[version](raw)
        version += 1
    return raw, (None if source == PROJECT_FORMAT_VERSION else source)


# --------------------------------------------------------------------------- #
# Normalize = migrate + validate + warn
# --------------------------------------------------------------------------- #
def normalize_project(raw: Any) -> ProjectLoadResult:
    """Open a project of any supported version: migrate it, validate its structure
    and collect warnings. A graph that does not build still opens — saving
    half-finished work is the whole point — but broken references are reported."""
    upgraded, source = migrate(raw)
    try:
        project = ProjectFile.model_validate(upgraded)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ())) or "(root)"
        raise ValueError(f"invalid project file at {where}: {first.get('msg')}") from None

    # The legacy default "synthetic" means blobs: name it the way the dataset picker does.
    project.train.dataset = canonical_dataset_id(project.train.dataset)

    warnings: list[ProjectWarning] = []
    if source is not None:
        warnings.append(ProjectWarning(
            code="migrated",
            message=f"Opened a format-{source} file and upgraded it to format {PROJECT_FORMAT_VERSION}.",
        ))

    nodes = project.graph.nodes
    if len(nodes) > MAX_NODES:
        raise ValueError(f"project has {len(nodes)} nodes; the limit is {MAX_NODES}")
    duplicates = sorted(i for i, n in Counter(n.id for n in nodes).items() if n > 1)
    if duplicates:
        raise ValueError(f"project has duplicate node ids: {', '.join(duplicates)}")

    for node in nodes:
        if node.type not in LAYER_REGISTRY:
            warnings.append(ProjectWarning(
                code="unknown_layer",
                message=f"Node {node.id!r} has an unknown layer type {node.type!r}; it is kept but will not build.",
                node_id=node.id,
            ))

    ids = {n.id for n in nodes}
    kept: list[list[str]] = []
    for edge in project.graph.edges:
        if edge[0] in ids and edge[1] in ids:
            kept.append(edge)
        else:
            warnings.append(ProjectWarning(
                code="dangling_edge",
                message=f"Dropped the edge {edge[0]!r} → {edge[1]!r}: it points at a missing node.",
            ))
    project.graph.edges = kept

    if nodes and any(n.position is None for n in nodes):
        warnings.append(ProjectWarning(
            code="missing_positions",
            message="Some nodes have no position; they are laid out automatically.",
        ))
    return ProjectLoadResult(ok=True, project=project, migrated_from=source, warnings=warnings)


# --------------------------------------------------------------------------- #
# Optional server store
# --------------------------------------------------------------------------- #
def slugify(name: str) -> str:
    """A safe store name for a project title (Russian is transliterated)."""
    s = re.sub(r"[^a-z0-9_-]+", "-", str(name).lower().translate(_TRANSLIT)).strip("-_")
    return (s or "project")[:64]


def _slug_path(slug: str, root: str | None = None) -> str:
    if not isinstance(slug, str) or not _SLUG.match(slug):
        raise ValueError(
            "invalid project name: use 1–64 lowercase letters, digits, '-' or '_', "
            "starting with a letter or a digit"
        )
    base = os.path.realpath(root or _PROJECTS_DIR)
    path = os.path.realpath(os.path.join(base, slug + _SUFFIX))
    if os.path.dirname(path) != base:  # belt and braces: the slug regex already forbids this
        raise ValueError("invalid project name")
    return path


def save_project(slug: str, project: ProjectFile, root: str | None = None) -> ProjectSaveResult:
    path = _slug_path(slug, root)
    base = os.path.dirname(path)
    os.makedirs(base, exist_ok=True)
    if not os.path.exists(path):
        count = sum(1 for f in os.listdir(base) if f.endswith(_SUFFIX))
        if count >= MAX_PROJECTS:
            raise ValueError(f"the server already holds {count} projects (limit {MAX_PROJECTS}); delete some first")
    now = _now()
    project = project.model_copy(update={"updated_at": now, "created_at": project.created_at or now})
    data = project.model_dump_json(indent=1)
    size = len(data.encode("utf-8"))
    if size > MAX_PROJECT_BYTES:
        raise ValueError(
            f"project is too large to store ({size // 1024} KB; the limit is {MAX_PROJECT_BYTES // 2**20} MB)"
        )
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written project
    return ProjectSaveResult(ok=True, slug=slug, updated_at=now)


def unique_slug(name: str, root: str | None = None) -> str:
    """A free store name for a new project: ``slugify(name)``, then ``-2``, ``-3``…"""
    base = slugify(name)
    folder = os.path.realpath(root or _PROJECTS_DIR)
    slug, n = base, 2
    while os.path.exists(os.path.join(folder, slug + _SUFFIX)):
        suffix = f"-{n}"
        slug = base[: 64 - len(suffix)] + suffix
        n += 1
    return slug


def create_project(project: ProjectFile, root: str | None = None) -> ProjectSaveResult:
    """Store a new project under a fresh name derived from its title (never overwrites)."""
    return save_project(unique_slug(project.name, root), project, root)


def list_projects(root: str | None = None) -> list[ProjectSummary]:
    base = os.path.realpath(root or _PROJECTS_DIR)
    if not os.path.isdir(base):
        return []
    summaries: list[ProjectSummary] = []
    for fn in os.listdir(base):
        if not fn.endswith(_SUFFIX):
            continue
        slug = fn[: -len(_SUFFIX)]
        if not _SLUG.match(slug):
            continue
        try:
            with open(os.path.join(base, fn), encoding="utf-8") as f:
                raw = json.load(f)
            graph = raw.get("graph") if isinstance(raw.get("graph"), dict) else {}
            summaries.append(ProjectSummary(
                slug=slug,
                name=str(raw.get("name") or slug),
                updated_at=raw.get("updated_at"),
                node_count=len(graph.get("nodes") or []),
            ))
        except (OSError, ValueError, AttributeError):
            continue  # an unreadable file never breaks the listing
    summaries.sort(key=lambda s: s.updated_at or "", reverse=True)
    return summaries


def load_project(slug: str, root: str | None = None) -> ProjectLoadResult:
    path = _slug_path(slug, root)
    if not os.path.exists(path):
        raise KeyError(f"no project named {slug!r} on the server")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return normalize_project(raw)


def delete_project(slug: str, root: str | None = None) -> None:
    path = _slug_path(slug, root)
    if not os.path.exists(path):
        raise KeyError(f"no project named {slug!r} on the server")
    os.remove(path)
