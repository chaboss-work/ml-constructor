"""Sequence layers: an embedding table and the recurrent nets (RNN, LSTM, GRU).

Shape convention for sequences: the per-sample shape is ``(L, F)`` — sequence
length ``L`` by feature size ``F`` — so the batched tensor is ``(B, L, F)``
(PyTorch ``batch_first=True``). ``Embedding`` turns a length-``L`` sequence of
token indices ``(L,)`` into ``(L, embedding_dim)``; the recurrent layers map
``(L, input_size)`` to either the full output sequence ``(L, hidden_size)`` or
just the last time-step ``(hidden_size,)`` — the latter feeds straight into a
Linear for a classic sequence classifier.
"""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn as nn

from ..registry import register
from ..schema import ParamDef
from ..shapes import Shape, ShapeError
from .base import LayerNode


class _EmbeddingModule(nn.Module):
    """Wraps ``nn.Embedding`` so the graph's float tensors become valid indices.

    The DAG carries float tensors, but an embedding lookup needs integer token
    ids. We cast to long and take them modulo ``num_embeddings``: for real token
    ids already in ``[0, num_embeddings)`` this is the identity, and it keeps the
    synthetic trainer (and any accidental out-of-range id) from crashing the
    lookup with an index error.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.emb = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.remainder(x.long(), self.num_embeddings)
        return self.emb(idx)


@register("Embedding")
class EmbeddingNode(LayerNode):
    """Token-id lookup table: a (L,) sequence of indices -> (L, embedding_dim)."""

    category = "sequence"
    label = "Embedding"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="num_embeddings", type="int", default=1000, min=1, label="Vocab size"),
            ParamDef(name="embedding_dim", type="int", default=32, min=1, label="Embedding dim"),
        ]

    def out_shape(self, in_shape: Shape | None) -> Shape:
        (length,) = self._require_shape(in_shape, ndim=1, what="(L,) token indices")
        return (int(length), int(self.p("embedding_dim", 32)))

    def build(self, in_shape: Shape | None) -> nn.Module:
        self._require_shape(in_shape, ndim=1, what="(L,) token indices")
        return _EmbeddingModule(int(self.p("num_embeddings", 1000)), int(self.p("embedding_dim", 32)))


class _RecurrentModule(nn.Module):
    """Runs a torch recurrent layer and returns either the full output sequence
    or just the last time-step (dropping the hidden-state tuple the layer also
    returns, which differs in shape between RNN/GRU and LSTM)."""

    def __init__(self, rnn: nn.Module, return_sequence: bool):
        super().__init__()
        self.rnn = rnn
        self.return_sequence = return_sequence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)  # out: (B, L, hidden); _ is h_n (and c_n for LSTM)
        return out if self.return_sequence else out[:, -1]


class _RecurrentNode(LayerNode):
    """Shared base for RNN/LSTM/GRU over a ``(L, input_size)`` sequence.

    ``input_size`` is inferred from the incoming shape; the output is the full
    sequence ``(L, hidden_size)`` when *Return sequence* is on, otherwise just
    the last step ``(hidden_size,)``.
    """

    category = "sequence"
    #: torch recurrent class, overridden per subclass
    _torch_cls: ClassVar[type[nn.Module]] = nn.RNN

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="hidden_size", type="int", default=64, min=1, label="Hidden size"),
            ParamDef(name="num_layers", type="int", default=1, min=1, label="Layers"),
            ParamDef(
                name="return_sequence",
                type="bool",
                default=False,
                label="Return sequence",
                help="On: output the full sequence (L, hidden). Off: only the last step (hidden,).",
            ),
        ]

    def _geom(self, in_shape: Shape | None):
        length, feat = self._require_shape(in_shape, ndim=2, what="(L, input_size)")
        return (
            int(length),
            int(feat),
            int(self.p("hidden_size", 64)),
            int(self.p("num_layers", 1)),
            bool(self.p("return_sequence", False)),
        )

    def out_shape(self, in_shape: Shape | None) -> Shape:
        length, _feat, hidden, _nl, ret = self._geom(in_shape)
        return (length, hidden) if ret else (hidden,)

    def _make_rnn(self, input_size: int, hidden: int, num_layers: int) -> nn.Module:
        return self._torch_cls(input_size, hidden, num_layers, batch_first=True)

    def build(self, in_shape: Shape | None) -> nn.Module:
        _length, feat, hidden, num_layers, ret = self._geom(in_shape)
        return _RecurrentModule(self._make_rnn(feat, hidden, num_layers), ret)


@register("RNN")
class RNNNode(_RecurrentNode):
    """Elman RNN over a (L, input_size) sequence."""

    label = "RNN"
    _torch_cls = nn.RNN

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        defs = super().param_defs()
        defs.append(
            ParamDef(name="nonlinearity", type="str", default="tanh", label="Nonlinearity", help="tanh or relu")
        )
        return defs

    def _make_rnn(self, input_size: int, hidden: int, num_layers: int) -> nn.Module:
        nl = str(self.p("nonlinearity", "tanh"))
        if nl not in ("tanh", "relu"):
            raise ShapeError(f"RNN {self.id!r}: nonlinearity must be 'tanh' or 'relu', got {nl!r}")
        return nn.RNN(input_size, hidden, num_layers, batch_first=True, nonlinearity=nl)


@register("LSTM")
class LSTMNode(_RecurrentNode):
    """Long Short-Term Memory over a (L, input_size) sequence."""

    label = "LSTM"
    _torch_cls = nn.LSTM


@register("GRU")
class GRUNode(_RecurrentNode):
    """Gated Recurrent Unit over a (L, input_size) sequence."""

    label = "GRU"
    _torch_cls = nn.GRU
