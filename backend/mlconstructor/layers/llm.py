"""Layers of today's large language models (Stage 17).

Graph nodes for the blocks in :mod:`mlconstructor.llm_modules`: RMSNorm, grouped-
query attention with rotary positions (GQA), DeepSeek's multi-head latent
attention (MLA), the gated MLP (SwiGLU / GeGLU), a mixture of experts and the
vocabulary head. The attention and feed-forward nodes include their own pre-norm
and residual connection, so a decoder layer is a chain of two nodes and a whole
model one ``Repeat`` of it. Sequences are ``(L, D)`` per sample.
"""

from __future__ import annotations

import torch.nn as nn

from .. import llm_modules as M
from ..registry import register
from ..schema import ParamDef
from .. import fixes as fx
from ..shapes import Shape, ShapeError
from .base import LayerNode

_EPS = ParamDef(name="eps", type="float", default=1e-6, min=0.0, label="Norm epsilon")


def _where(node_id: str) -> str:
    """'layers__inner0' -> "layer 1 of the block 'layers'"; a plain id stays quoted."""
    block, sep, rest = node_id.rpartition("__inner")
    if sep and rest.isdigit():
        return f"layer {int(rest) + 1} of the block {block!r}"
    return repr(node_id)


@register("RMSNorm")
class RMSNormNode(LayerNode):
    """Root-mean-square normalization over the last dimension — the norm of modern
    LLMs (no mean subtraction, no bias). Shape-preserving."""

    category = "llm"
    label = "RMSNorm"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [_EPS]

    def out_shape(self, in_shape: Shape | None) -> Shape:
        return self._require_shape(in_shape)

    def build(self, in_shape: Shape | None) -> nn.Module:
        sh = self._require_shape(in_shape)
        return nn.RMSNorm(int(sh[-1]), eps=float(self.p("eps", 1e-6)))


class _SeqNode(LayerNode):
    """A shape-preserving block over a (L, D) token sequence."""

    category = "llm"

    def _dims(self, in_shape: Shape | None) -> tuple[int, int]:
        length, dim = self._require_shape(in_shape, ndim=2, what="(L, D): a sequence of token vectors")
        return int(length), int(dim)

    def out_shape(self, in_shape: Shape | None) -> Shape:
        return self._dims(in_shape)

    def _fail(self, msg: str, hint: str | None = None, *, code: str = "settings_conflict", param: str | None = None,
              fixes: list | None = None):
        raise ShapeError(f"{self.label} ({_where(self.id)}): {msg}", node_id=self.id, hint=hint, settings=True,
                         code=code, param=param, fixes=fixes)


@register("GQAttention")
class GQAttentionNode(_SeqNode):
    """Self-attention with rotary positions (RoPE) and grouped key/value heads —
    Llama 3, Qwen3, Mistral, Gemma 3, Phi-4. Includes the pre-norm and the residual:
    x + attention(RMSNorm(x)). Causal by default (a token sees only earlier ones)."""

    label = "Attention (GQA + RoPE)"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="heads", type="int", default=4, min=1, label="Query heads"),
            ParamDef(name="kv_heads", type="int", default=2, min=1, label="Key/value heads",
                     help="Groups of query heads share one key/value head (GQA). Equal to heads = classic attention."),
            ParamDef(name="head_dim", type="int", default=0, min=0, label="Head size",
                     help="0 = model size / heads"),
            ParamDef(name="rope_theta", type="float", default=10000.0, min=1.0, label="RoPE base"),
            ParamDef(name="causal", type="bool", default=True, label="Causal (only earlier tokens)"),
            ParamDef(name="sliding_window", type="int", default=0, min=0, label="Sliding window",
                     help="0 = see the whole sequence; N = only the last N tokens"),
            ParamDef(name="qk_norm", type="bool", default=False, label="Query/key norm (Qwen3, Gemma 3)"),
            ParamDef(name="bias", type="bool", default=False, label="Bias on q/k/v"),
            ParamDef(name="post_norm", type="bool", default=False, label="Post-norm (Gemma)"),
            _EPS,
        ]

    def _geom(self, in_shape: Shape | None) -> dict:
        _length, dim = self._dims(in_shape)
        heads, kv = int(self.p("heads", 4)), int(self.p("kv_heads", 2))
        head_dim = int(self.p("head_dim", 0)) or (dim // heads if heads else 0)
        if heads < 1 or kv < 1:
            self._fail("needs at least one query head and one key/value head", code="param_range",
                       param="heads" if heads < 1 else "kv_heads",
                       fixes=[fx.fix("Use 4 query heads and 2 key/value heads", fx.op_set_param(self.id, "heads", 4),
                                     fx.op_set_param(self.id, "kv_heads", 2))])
        if heads % kv:
            self._fail(f"{heads} query heads cannot be shared evenly by {kv} key/value heads",
                       hint="Make the query heads a multiple of the key/value heads (e.g. 8 and 2).",
                       code="not_divisible", param="kv_heads",
                       fixes=fx.divisor_fixes(self.id, "kv_heads", heads, kv, " query heads per key/value head"))
        if not int(self.p("head_dim", 0)) and dim % heads:
            self._fail(f"model size {dim} does not split into {heads} heads",
                       hint="Set the head size explicitly, or pick a head count that divides the model size.",
                       code="not_divisible", param="heads",
                       fixes=[f for f in fx.divisor_fixes(self.id, "heads", dim, heads, " numbers per head")
                              if f["ops"][0]["value"] % kv == 0] or
                             fx.divisor_fixes(self.id, "heads", dim, heads, " numbers per head"))
        if head_dim % 2:
            self._fail(f"head size {head_dim} must be even (rotary positions rotate pairs)",
                       hint="Use an even head size.", param="head_dim",
                       fixes=[fx.set_param(self.id, "head_dim", head_dim + 1, f"Set the head size to {head_dim + 1}")])
        return dict(dim=dim, heads=heads, kv_heads=kv, head_dim=head_dim,
                    rope_theta=float(self.p("rope_theta", 10000.0)), causal=bool(self.p("causal", True)),
                    sliding_window=int(self.p("sliding_window", 0)), qk_norm=bool(self.p("qk_norm", False)),
                    bias=bool(self.p("bias", False)), post_norm=bool(self.p("post_norm", False)),
                    eps=float(self.p("eps", 1e-6)))

    def out_shape(self, in_shape: Shape | None) -> Shape:
        self._geom(in_shape)
        return self._dims(in_shape)

    def build(self, in_shape: Shape | None) -> nn.Module:
        return M.GroupedQueryAttention(**self._geom(in_shape))


@register("LatentAttention")
class LatentAttentionNode(_SeqNode):
    """DeepSeek's multi-head latent attention (MLA): keys and values pass through a
    small shared latent (what DeepSeek-V3 caches), with a separate rotary slice per
    head. Includes the pre-norm and the residual."""

    label = "Latent attention (MLA)"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="heads", type="int", default=4, min=1, label="Heads"),
            ParamDef(name="q_lora_rank", type="int", default=32, min=0, label="Query latent size",
                     help="0 = no query compression"),
            ParamDef(name="kv_lora_rank", type="int", default=16, min=1, label="Key/value latent size"),
            ParamDef(name="qk_nope_dim", type="int", default=8, min=1, label="Head size without rotation"),
            ParamDef(name="qk_rope_dim", type="int", default=8, min=2, label="Rotary head size"),
            ParamDef(name="v_dim", type="int", default=8, min=1, label="Value head size"),
            ParamDef(name="rope_theta", type="float", default=10000.0, min=1.0, label="RoPE base"),
            ParamDef(name="causal", type="bool", default=True, label="Causal (only earlier tokens)"),
            _EPS,
        ]

    def _geom(self, in_shape: Shape | None) -> dict:
        _length, dim = self._dims(in_shape)
        g = dict(dim=dim, heads=int(self.p("heads", 4)), q_lora_rank=int(self.p("q_lora_rank", 32)),
                 kv_lora_rank=int(self.p("kv_lora_rank", 16)), qk_nope_dim=int(self.p("qk_nope_dim", 8)),
                 qk_rope_dim=int(self.p("qk_rope_dim", 8)), v_dim=int(self.p("v_dim", 8)),
                 rope_theta=float(self.p("rope_theta", 10000.0)), causal=bool(self.p("causal", True)),
                 eps=float(self.p("eps", 1e-6)))
        if g["qk_rope_dim"] % 2:
            self._fail(f"rotary head size {g['qk_rope_dim']} must be even", hint="Use an even rotary head size.",
                       param="qk_rope_dim", fixes=[fx.set_param(self.id, "qk_rope_dim", g["qk_rope_dim"] + 1)])
        if min(g["heads"], g["kv_lora_rank"], g["qk_nope_dim"], g["v_dim"]) < 1:
            low = [k for k in ("heads", "kv_lora_rank", "qk_nope_dim", "v_dim") if g[k] < 1]
            self._fail("heads, the key/value latent and the head sizes must all be at least 1", code="param_range",
                       param=low[0], fixes=[fx.fix("Set " + ", ".join(low) + " to 1",
                                                   *[fx.op_set_param(self.id, k, 1) for k in low])])
        return g

    def out_shape(self, in_shape: Shape | None) -> Shape:
        self._geom(in_shape)
        return self._dims(in_shape)

    def build(self, in_shape: Shape | None) -> nn.Module:
        return M.LatentAttention(**self._geom(in_shape))


_ACT = ParamDef(name="activation", type="str", default="silu", label="Activation",
                help="silu = SwiGLU (Llama, Qwen, DeepSeek); gelu = GeGLU (Gemma)", choices=["silu", "gelu"])


@register("GatedMLP")
class GatedMLPNode(_SeqNode):
    """The gated feed-forward of modern LLMs: down(act(gate(x)) * up(x)) — SwiGLU or
    GeGLU. Includes the pre-norm and the residual."""

    label = "Gated MLP (SwiGLU)"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="hidden", type="int", default=128, min=1, label="Hidden size"),
            _ACT,
            ParamDef(name="post_norm", type="bool", default=False, label="Post-norm (Gemma)"),
            _EPS,
        ]

    def build(self, in_shape: Shape | None) -> nn.Module:
        _length, dim = self._dims(in_shape)
        act = str(self.p("activation", "silu"))
        if act not in ("silu", "gelu"):
            self._fail(f"activation must be 'silu' or 'gelu', got {act!r}", code="param_choice", param="activation",
                       fixes=[fx.set_param(self.id, "activation", "silu"), fx.set_param(self.id, "activation", "gelu")])
        return M.GatedMLP(dim, int(self.p("hidden", 128)), act, bool(self.p("post_norm", False)),
                          float(self.p("eps", 1e-6)))


@register("MoE")
class MoENode(_SeqNode):
    """A mixture of experts: a router sends each token to its top-k experts (small
    gated MLPs) and mixes their answers; shared experts see every token (DeepSeek).
    Includes the pre-norm and the residual."""

    label = "Mixture of experts (MoE)"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="experts", type="int", default=8, min=1, label="Experts"),
            ParamDef(name="top_k", type="int", default=2, min=1, label="Experts per token"),
            ParamDef(name="expert_hidden", type="int", default=64, min=1, label="Expert hidden size"),
            ParamDef(name="shared_experts", type="int", default=0, min=0, label="Shared experts"),
            ParamDef(name="router", type="str", default="softmax", label="Router",
                     help="softmax (Mixtral, Qwen3) or sigmoid with a balancing bias (DeepSeek-V3)",
                     choices=["softmax", "sigmoid"]),
            ParamDef(name="norm_topk", type="bool", default=True, label="Renormalize the chosen weights"),
            ParamDef(name="scaling", type="float", default=1.0, min=0.0, label="Output scaling"),
            _ACT,
            _EPS,
        ]

    def _geom(self, in_shape: Shape | None) -> dict:
        _length, dim = self._dims(in_shape)
        g = dict(dim=dim, experts=int(self.p("experts", 8)), top_k=int(self.p("top_k", 2)),
                 expert_hidden=int(self.p("expert_hidden", 64)), shared_experts=int(self.p("shared_experts", 0)),
                 router=str(self.p("router", "softmax")), norm_topk=bool(self.p("norm_topk", True)),
                 scaling=float(self.p("scaling", 1.0)), activation=str(self.p("activation", "silu")),
                 eps=float(self.p("eps", 1e-6)))
        if not 1 <= g["top_k"] <= g["experts"]:
            k = min(max(g["top_k"], 1), max(g["experts"], 1))
            self._fail(f"each token goes to {g['top_k']} experts, but there are {g['experts']}",
                       hint="Experts per token must be between 1 and the number of experts.", code="param_range",
                       param="top_k",
                       fixes=[fx.set_param(self.id, "top_k", k, f"Send each token to {k} expert{'s' if k > 1 else ''}")]
                       + ([fx.set_param(self.id, "experts", g["top_k"], f"Use {g['top_k']} experts")]
                          if g["top_k"] >= 1 else []))
        if g["router"] not in ("softmax", "sigmoid"):
            self._fail(f"router must be 'softmax' or 'sigmoid', got {g['router']!r}", code="param_choice",
                       param="router", fixes=[fx.set_param(self.id, "router", "softmax"),
                                              fx.set_param(self.id, "router", "sigmoid")])
        if g["activation"] not in ("silu", "gelu"):
            self._fail(f"activation must be 'silu' or 'gelu', got {g['activation']!r}", code="param_choice",
                       param="activation", fixes=[fx.set_param(self.id, "activation", "silu"),
                                                  fx.set_param(self.id, "activation", "gelu")])
        return g

    def out_shape(self, in_shape: Shape | None) -> Shape:
        self._geom(in_shape)
        return self._dims(in_shape)

    def build(self, in_shape: Shape | None) -> nn.Module:
        return M.MixtureOfExperts(**self._geom(in_shape))


@register("LMHead")
class LMHeadNode(LayerNode):
    """The vocabulary head: one score per token of the vocabulary (a Linear without
    bias) over the last dimension. Tied, it reuses the graph's Embedding table as
    its weight — no extra parameters (Gemma, small Qwen3 and Llama 3.2)."""

    category = "llm"
    label = "LM head"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="vocab", type="int", default=256, min=1, label="Vocabulary size"),
            ParamDef(name="tie_embedding", type="bool", default=False, label="Tie to the Embedding",
                     help="Reuse the Embedding table as this layer's weight (needs the same vocabulary and size)."),
        ]

    def out_shape(self, in_shape: Shape | None) -> Shape:
        sh = self._require_shape(in_shape)
        return (*sh[:-1], int(self.p("vocab", 256)))

    def build(self, in_shape: Shape | None) -> nn.Module:
        sh = self._require_shape(in_shape)
        return M.LMHead(int(sh[-1]), int(self.p("vocab", 256)), bool(self.p("tie_embedding", False)))
