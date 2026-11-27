"""Shape utilities shared across the core.

A *shape* is the per-sample tensor shape WITHOUT the batch dimension, written as
a tuple of ints. Examples: an MNIST image is ``(1, 28, 28)``; a flat feature
vector is ``(128,)``. ``None`` means "no shape yet" — e.g. the input side of a
source node before the graph's ``Input`` provides one.
"""

from __future__ import annotations

from math import prod as _prod

Shape = tuple[int, ...]


class GraphError(ValueError):
    """A mistake the user can fix: in the graph, its settings, the data or training.

    Stage 21 — every one carries a stable ``code`` (its entry in
    :mod:`mlconstructor.error_codes`: a title in plain words, why it happens, what to
    do) and can carry ready **fixes** — each a list of small graph edits
    (:mod:`mlconstructor.fixes`) the editor applies on one click.

    ``node_id`` / ``edge`` say where it is, ``hint`` what to do about it in this
    very case, ``param`` which setting of the node is at fault, ``facts`` anything
    a fix needs to know (the shape a dataset wants, the class count…).
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        node_id: str | None = None,
        edge: tuple[str, str] | None = None,
        hint: str | None = None,
        settings: bool = False,
        param: str | None = None,
        fixes: list[dict] | None = None,
        facts: dict | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.node_id = node_id
        self.edge = edge
        self.hint = hint
        self.settings = settings
        self.param = param
        self.fixes = list(fixes or [])
        self.facts = dict(facts or {})
        #: Stage 21 — a layer inside a Repeat block: its index (``node_id`` is the block)
        self.inner: int | None = None

    def to_detail(self) -> dict:
        """The structured part of a 422 response (the catalogue's words are added by
        :func:`mlconstructor.diagnostics.describe_error`)."""
        return {
            "message": str(self),
            "code": self.code,
            "node_id": self.node_id,
            "edge": list(self.edge) if self.edge else None,
            "hint": self.hint,
        }


class ShapeError(GraphError):
    """Raised when a layer receives an input shape it cannot consume.

    This is the seed of the project's "smart adapters" vision: instead of a
    cryptic tensor error deep inside the forward pass, the builder reports
    exactly which connection is invalid and why, pointing at the node id.

    Stage 14 adds machine-readable context so the editor can point at the
    culprit: ``node_id`` (the layer), ``edge`` (``(source, target)`` of the
    connection) and ``hint`` (what to do about it).

    Stage 17: ``settings=True`` marks a contradiction in the layer's own settings
    (e.g. 4 query heads over 3 key/value groups) — no transform on the incoming
    connection can fix that, so it is reported as it is.

    Stage 21: a :class:`GraphError` — with a ``code`` and possibly ready fixes.
    """


#: what a layer reads, by the number of axes of one example (Stage 21: plain words)
RANK_WORDS = {
    1: "a vector (one row of numbers)",
    2: "a sequence (L steps × D numbers each)",
    3: "a picture (C×H×W: channels × height × width)",
    4: "a volume (C×D×H×W)",
}


def fmt(shape: Shape | list | None) -> str:
    return "×".join(str(int(d)) for d in shape) if shape else "()"


def describe_shape(shape: Shape | list | None) -> str:
    """One example's shape in plain words: "a picture 16×14×14", "a vector of 784 numbers"."""
    if shape is None:
        return "nothing"
    shape = tuple(int(d) for d in shape)
    if len(shape) == 1:
        return f"a vector of {shape[0]:,} number{'' if shape[0] == 1 else 's'}"
    if len(shape) == 2:
        return f"a sequence {fmt(shape)} ({shape[0]:,} steps × {shape[1]:,} numbers)"
    if len(shape) == 3:
        return f"a picture {fmt(shape)} ({shape[0]:,} channel{'' if shape[0] == 1 else 's'}, {shape[1]}×{shape[2]} pixels)"
    return f"a {len(shape)}-axis block {fmt(shape)}"


def numel(shape: Shape | None) -> int:
    """Number of elements per sample for ``shape`` (0 if unknown)."""
    return int(_prod(shape)) if shape else 0


def as_list(shape: Shape | None) -> list[int] | None:
    return list(shape) if shape is not None else None
