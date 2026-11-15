"""Abstract base for a graph node = one layer instance.

Subclasses do two things:

* :meth:`out_shape` — infer the output shape (no batch dim) from the input
  shape. Runs *before* building, so the builder can validate every connection
  and auto-fill dimensions (e.g. a Linear's ``in_features``) from the incoming
  shape. This is the first concrete step toward the project's "smart adapters".
* :meth:`build` — construct the actual ``torch.nn`` module (or ``None`` for a
  pass-through source/sink that owns no parameters).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Optional

import torch.nn as nn

from ..schema import ParamDef
from ..shapes import Shape, ShapeError


class LayerNode(ABC):
    type_name: ClassVar[str] = ""          # set by @register
    category: ClassVar[str] = "misc"
    label: ClassVar[str] = ""
    #: Input/source nodes (e.g. ``Input``) set this False — they produce a shape
    #: without consuming one.
    needs_input: ClassVar[bool] = True

    def __init__(self, node_id: str, params: dict | None = None):
        self.id = node_id
        self.params = dict(params or {})

    # -- parameters -------------------------------------------------------- #
    def p(self, name: str, default=None):
        return self.params.get(name, default)

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        """Editable parameters for the UI palette. Override per layer."""
        return []

    # -- shape helpers ----------------------------------------------------- #
    def _require_shape(self, in_shape: Shape | None, ndim: int | None = None, what: str = "") -> Shape:
        """Validate the incoming shape, raising :class:`ShapeError` with node
        context on mismatch (clear message instead of a deep torch crash)."""
        if in_shape is None:
            raise ShapeError(f"node {self.id!r} ({self.type_name}) expects an input but has none")
        if ndim is not None and len(in_shape) != ndim:
            extra = f" {what}" if what else ""
            raise ShapeError(
                f"node {self.id!r} ({self.type_name}) expects a {ndim}-D input{extra}, "
                f"got shape {tuple(in_shape)}"
            )
        return in_shape

    # -- to implement ------------------------------------------------------ #
    @abstractmethod
    def out_shape(self, in_shape: Shape | None) -> Shape:
        ...

    @abstractmethod
    def build(self, in_shape: Shape | None) -> Optional[nn.Module]:
        ...

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"{type(self).__name__}(id={self.id!r}, params={self.params!r})"
