"""Attention & transformer layers (Stage 4).

Built on the Stage 3 sequence convention: the per-sample shape is ``(L, E)`` —
sequence length by embedding/feature size — so the batched tensor is
``(B, L, E)`` (``batch_first``). These layers are shape-preserving: multi-head
self-attention and a transformer-encoder block map ``(L, E) -> (L, E)``, and
sinusoidal positional encoding adds position information without changing the
shape. Put a Flatten + Linear (a pooling head comes later) after them to turn
the sequence into class logits.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..registry import register
from ..schema import ParamDef
from .. import fixes as fx
from ..shapes import Shape, ShapeError
from .base import LayerNode


class _SelfAttention(nn.Module):
    """Multi-head *self*-attention: query = key = value = x; returns the attended
    sequence (the attention weights are discarded)."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x, need_weights=False)
        return out


@register("MultiheadAttention")
class MultiheadAttentionNode(LayerNode):
    """Multi-head self-attention over a (L, E) sequence. ``embed_dim`` = E is
    inferred and must be divisible by ``num_heads``. Shape-preserving."""

    category = "attention"
    label = "MultiheadAttention"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="num_heads", type="int", default=4, min=1, label="Heads"),
            ParamDef(name="dropout", type="float", default=0.0, min=0.0, max=1.0, label="Dropout"),
        ]

    def _check(self, in_shape: Shape | None):
        length, embed = self._require_shape(in_shape, ndim=2, what="(L, embed_dim)")
        heads = int(self.p("num_heads", 4))
        if embed % heads != 0:
            raise ShapeError(
                f"MultiheadAttention “{self.id}”: its vectors of {embed} numbers (embed_dim) do not split into "
                f"{heads} equal heads (num_heads)",
                code="not_divisible", param="num_heads",
                fixes=fx.divisor_fixes(self.id, "num_heads", embed, heads, " numbers per head"),
            )
        return int(length), int(embed), heads

    def out_shape(self, in_shape: Shape | None) -> Shape:
        length, embed, _h = self._check(in_shape)
        return (length, embed)

    def build(self, in_shape: Shape | None) -> nn.Module:
        _l, embed, heads = self._check(in_shape)
        return _SelfAttention(embed, heads, float(self.p("dropout", 0.0)))


@register("TransformerEncoderLayer")
class TransformerEncoderLayerNode(LayerNode):
    """One transformer-encoder block (self-attention + feed-forward, each with a
    residual connection and LayerNorm) over a (L, D) sequence. ``d_model`` = D is
    inferred and must be divisible by ``nhead``. Shape-preserving."""

    category = "attention"
    label = "TransformerEncoderLayer"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="nhead", type="int", default=4, min=1, label="Heads"),
            ParamDef(name="dim_feedforward", type="int", default=256, min=1, label="FFN dim"),
            ParamDef(name="dropout", type="float", default=0.1, min=0.0, max=1.0, label="Dropout"),
            ParamDef(name="activation", type="str", default="relu", label="Activation", help="relu or gelu"),
        ]

    def _check(self, in_shape: Shape | None):
        length, dmodel = self._require_shape(in_shape, ndim=2, what="(L, d_model)")
        heads = int(self.p("nhead", 4))
        if dmodel % heads != 0:
            raise ShapeError(
                f"TransformerEncoderLayer “{self.id}”: its vectors of {dmodel} numbers (d_model) do not split into "
                f"{heads} equal heads (nhead)",
                code="not_divisible", param="nhead",
                fixes=fx.divisor_fixes(self.id, "nhead", dmodel, heads, " numbers per head"),
            )
        act = str(self.p("activation", "relu"))
        if act not in ("relu", "gelu"):
            raise ShapeError(
                f"TransformerEncoderLayer {self.id!r}: activation must be 'relu' or 'gelu', got {act!r}",
                code="param_choice", param="activation",
                fixes=[fx.set_param(self.id, "activation", "relu"), fx.set_param(self.id, "activation", "gelu")],
            )
        return int(length), int(dmodel), heads, act

    def out_shape(self, in_shape: Shape | None) -> Shape:
        length, dmodel, _h, _a = self._check(in_shape)
        return (length, dmodel)

    def build(self, in_shape: Shape | None) -> nn.Module:
        _l, dmodel, heads, act = self._check(in_shape)
        return nn.TransformerEncoderLayer(
            d_model=dmodel,
            nhead=heads,
            dim_feedforward=int(self.p("dim_feedforward", 256)),
            dropout=float(self.p("dropout", 0.1)),
            activation=act,
            batch_first=True,
        )


class _PositionalEncoding(nn.Module):
    """Adds fixed sinusoidal positional encodings to a (B, L, E) sequence."""

    def __init__(self, embed_dim: int, max_len: int, dropout: float):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        half = embed_dim // 2
        div = torch.exp(torch.arange(0, 2 * half, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim))
        pe[:, 0 : 2 * half : 2] = torch.sin(pos * div)
        pe[:, 1 : 2 * half : 2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, E)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.drop(x)


@register("PositionalEncoding")
class PositionalEncodingNode(LayerNode):
    """Adds sinusoidal positional encoding to a (L, E) sequence (shape-preserving,
    no learnable parameters). Usually placed right after an Embedding."""

    category = "attention"
    label = "PositionalEncoding"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="max_len", type="int", default=512, min=1, label="Max length"),
            ParamDef(name="dropout", type="float", default=0.0, min=0.0, max=1.0, label="Dropout"),
        ]

    def out_shape(self, in_shape: Shape | None) -> Shape:
        return self._require_shape(in_shape, ndim=2, what="(L, embed_dim)")

    def build(self, in_shape: Shape | None) -> nn.Module:
        length, embed = self._require_shape(in_shape, ndim=2, what="(L, embed_dim)")
        max_len = max(int(self.p("max_len", 512)), int(length))  # never shorter than the sequence
        return _PositionalEncoding(int(embed), max_len, float(self.p("dropout", 0.0)))
