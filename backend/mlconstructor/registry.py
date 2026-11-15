"""The layer registry.

Every layer class registers itself with :func:`register` so the rest of the
system can look it up by its graph ``type`` string and the UI can enumerate the
whole palette via :func:`catalog`. This decoupling is what lets "one class per
layer" scale: adding a layer is dropping a new file in ``layers/`` — no central
list to edit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from .schema import LayerInfo

if TYPE_CHECKING:  # avoid an import cycle at runtime (base imports this module)
    from .layers.base import LayerNode

LAYER_REGISTRY: dict[str, "type[LayerNode]"] = {}


def register(type_name: str) -> Callable[[type], type]:
    """Class decorator: register a :class:`LayerNode` under ``type_name``."""

    def deco(cls: type) -> type:
        cls.type_name = type_name
        if type_name in LAYER_REGISTRY:
            raise RuntimeError(f"duplicate layer type registered: {type_name!r}")
        LAYER_REGISTRY[type_name] = cls
        return cls

    return deco


def get_layer_cls(type_name: str) -> "type[LayerNode]":
    try:
        return LAYER_REGISTRY[type_name]
    except KeyError:
        known = ", ".join(sorted(LAYER_REGISTRY)) or "(none loaded)"
        raise KeyError(f"unknown layer type {type_name!r}; known: {known}") from None


def catalog() -> list[LayerInfo]:
    """The full palette the UI draws from (``GET /layers``)."""
    infos: list[LayerInfo] = []
    for name, cls in sorted(LAYER_REGISTRY.items()):
        infos.append(
            LayerInfo(
                type=name,
                category=cls.category,
                label=cls.label or name,
                doc=(cls.__doc__ or "").strip().split("\n")[0],
                params=cls.param_defs(),
            )
        )
    return infos
