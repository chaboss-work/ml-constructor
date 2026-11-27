"""Layers of today's image, 3D, speech and vision models (Stage 19).

Graph nodes for the blocks in :mod:`mlconstructor.gen_modules`: patch embeddings
and position tables (ViT, CLIP, DiT), pre-norm attention / cross-attention / MLP
blocks (ViT, CLIP, Whisper), the diffusion parts — timestep embedding, adaptive
norm, DiT and FLUX blocks, Stable Diffusion's U-Net and VAE, adding noise and
predicting the clean latent —, the triplane parts of a 3-D generator (TripoSR),
CLIP's similarity, and small generic layers: Conv1d, Conv3d, Upsample, SiLU,
Transpose, Slice tokens.

Several of them take more than one input — a U-Net reads a noisy latent, a
timestep and a text. Their inputs (``ports``) arrive in the order the connections
were made: the first connection feeds the first port, and so on.
"""

from __future__ import annotations

import math

import torch.nn as nn

from .. import gen_modules as G
from ..registry import register
from ..schema import ParamDef
from ..shapes import Shape, ShapeError
from .base import LayerNode
from .llm import _where


def _fmt(shape) -> str:
    return "×".join(str(int(d)) for d in shape) if shape else "()"


_EPS5 = ParamDef(name="eps", type="float", default=1e-5, min=0.0, label="Norm epsilon")
_EPS6 = ParamDef(name="eps", type="float", default=1e-6, min=0.0, label="Norm epsilon")


class _Gen(LayerNode):
    """Helpers shared by the Stage 19 layers."""

    def _fail(self, msg: str, hint: str | None = None):
        """The layer's own settings clash: no connection transform can help."""
        raise ShapeError(f"{self.label} ({_where(self.id)}): {msg}", node_id=self.id, hint=hint, settings=True)

    def _bad(self, msg: str, hint: str | None = None):
        """What arrives does not fit (a connection transform might make it fit)."""
        raise ShapeError(f"{self.label} ({_where(self.id)}): {msg}", node_id=self.id, hint=hint)

    def _int(self, name: str, default: int, low: int = 1) -> int:
        v = int(self.p(name, default))
        if v < low:
            self._fail(f"{name} must be at least {low}, got {v}")
        return v

    def _choice(self, name: str, default: str, choices: tuple[str, ...]) -> str:
        v = str(self.p(name, default))
        if v not in choices:
            self._fail(f"{name} must be one of {', '.join(choices)}, got {v!r}")
        return v

    def _ints(self, name: str, default: list[int]) -> list[int]:
        raw = self.p(name, default)
        try:
            vals = [int(v) for v in raw]
        except (TypeError, ValueError):
            self._fail(f"{name} must be a list of whole numbers, got {raw!r}")
        if not vals:
            self._fail(f"{name} must not be empty")
        return vals


class _Ported(_Gen):
    """A layer with several named inputs (``ports``), arriving in connection order."""

    def _needs_ports(self):
        ports = self.input_ports()
        raise ShapeError(
            f"{self.label} ({_where(self.id)}) takes {len(ports)} inputs: {', '.join(ports)}",
            node_id=self.id,
            hint=f"Connect them in this order: {' → '.join(ports)}.",
        )

    def out_shape(self, in_shape: Shape | None) -> Shape:
        self._needs_ports()

    def build(self, in_shape: Shape | None):
        self._needs_ports()

    def _rank(self, index: int, shape: Shape, ndim: int, what: str) -> Shape:
        if len(shape) != ndim:
            self._bad(f"its input “{self.input_ports()[index]}” must be {what}, got {_fmt(shape)}")
        return tuple(int(d) for d in shape)

    def _one_number(self, index: int, shape: Shape) -> Shape:
        if math.prod(int(d) for d in shape) != 1:
            self._bad(f"its input “{self.input_ports()[index]}” is one number per example (Input [1]), got {_fmt(shape)}")
        return tuple(int(d) for d in shape)


# --------------------------------------------------------------------------- #
# Vision: patches, positions, CLIP's similarity
# --------------------------------------------------------------------------- #
@register("PatchEmbed")
class PatchEmbedNode(_Gen):
    """Cuts an image into patches, one token per patch, with an optional class
    token and position table — how ViT, CLIP, DiT and DINO read a picture.
    (C, H, W) -> (N, dim)."""

    category = "vision"
    label = "Patch embedding"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="patch_size", type="int", default=16, min=1, label="Patch size"),
            ParamDef(name="dim", type="int", default=768, min=1, label="Token size"),
            ParamDef(name="cls_token", type="bool", default=False, label="Class token in front (ViT, CLIP)"),
            ParamDef(name="position", type="str", default="learned", label="Position table",
                     help="learned (ViT, CLIP); sincos = DiT's fixed sine-cosine table; none",
                     choices=["learned", "sincos", "none"]),
            ParamDef(name="pos_grid", type="int", default=0, min=0, label="Table made for a grid of",
                     help="0 = the image's grid. DINO's table is for 14×14 patches; a bigger image gets it resized."),
            ParamDef(name="bias", type="bool", default=True, label="Bias", help="CLIP's patch convolution has none"),
        ]

    def _geom(self, in_shape: Shape | None):
        c, h, w = self._require_shape(in_shape, ndim=3, what="(C, H, W): an image")
        p = self._int("patch_size", 16)
        if h % p or w % p:
            self._bad(f"the image {h}×{w} does not split into {p}×{p} patches",
                      hint="Pick a patch size that divides the image's height and width.")
        gh, gw = h // p, w // p
        pos = self._choice("position", "learned", ("learned", "sincos", "none"))
        if pos != "none" and gh != gw:
            self._bad(f"a position table needs a square grid of patches, got {gh}×{gw}")
        return int(c), p, gh, gw, pos, int(self.p("pos_grid", 0) or 0)

    def out_shape(self, in_shape: Shape | None) -> Shape:
        _c, _p, gh, gw, _pos, _g = self._geom(in_shape)
        return (gh * gw + (1 if bool(self.p("cls_token", False)) else 0), self._int("dim", 768))

    def build(self, in_shape: Shape | None) -> nn.Module:
        c, p, gh, _gw, pos, pos_grid = self._geom(in_shape)
        return G.PatchEmbed(c, self._int("dim", 768), p, gh, bool(self.p("cls_token", False)), pos,
                            bool(self.p("bias", True)), pos_grid)


@register("Unpatchify")
class UnpatchifyNode(_Gen):
    """Patch tokens back into an image — the inverse of a patch embedding (DiT's
    last step). (N, patch·patch·C) -> (C, H, W); the N patches form a square."""

    category = "vision"
    label = "Unpatchify"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="patch_size", type="int", default=2, min=1, label="Patch size"),
            ParamDef(name="channels", type="int", default=4, min=1, label="Image channels"),
        ]

    def out_shape(self, in_shape: Shape | None) -> Shape:
        n, f = self._require_shape(in_shape, ndim=2, what="(N, patch·patch·C): patch tokens")
        p, c = self._int("patch_size", 2), self._int("channels", 4)
        if f != p * p * c:
            self._bad(f"each token must hold {p}×{p}×{c} = {p * p * c} numbers (patch × patch × channels), got {f}",
                      hint="Set the channels so that patch × patch × channels equals the token size.")
        g = int(round(math.sqrt(n)))
        if g * g != n:
            self._bad(f"{n} patches do not form a square")
        return (c, g * p, g * p)

    def build(self, in_shape: Shape | None) -> nn.Module:
        self.out_shape(in_shape)
        return G.Unpatchify(self._int("patch_size", 2), self._int("channels", 4))


@register("PositionEmbedding")
class PositionEmbeddingNode(_Gen):
    """Adds a position table to a (L, D) sequence: learned (CLIP's text encoder,
    Whisper's decoder) or fixed sinusoids (Whisper's encoder). Shape-preserving."""

    category = "sequence"
    label = "Position embedding"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="length", type="int", default=0, min=0, label="Table length",
                     help="0 = the sequence's length; a longer table also serves shorter sequences"),
            ParamDef(name="kind", type="str", default="learned", label="Kind",
                     help="learned, or fixed sinusoids (Whisper's encoder)", choices=["learned", "sinusoidal"]),
        ]

    def _dims(self, in_shape: Shape | None):
        length, dim = self._require_shape(in_shape, ndim=2, what="(L, D): a sequence of token vectors")
        n = int(self.p("length", 0) or 0)
        if n and n < length:
            self._fail(f"the table has {n} positions, but the sequence is {length} long",
                       hint="Make the table at least as long as the sequence (0 = the same length).")
        return int(length), int(dim), n or int(length), self._choice("kind", "learned", ("learned", "sinusoidal"))

    def out_shape(self, in_shape: Shape | None) -> Shape:
        length, dim, _n, _k = self._dims(in_shape)
        return (length, dim)

    def build(self, in_shape: Shape | None) -> nn.Module:
        _length, dim, n, kind = self._dims(in_shape)
        return G.PositionEmbedding(n, dim, kind)


@register("Similarity")
class SimilarityNode(_Ported):
    """CLIP's score for an image and a text: the cosine of their embeddings times a
    learned temperature. Two vectors of the same size -> one number."""

    category = "vision"
    label = "Similarity (CLIP)"
    ports = ("image", "text")

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [ParamDef(name="logit_scale", type="float", default=2.6592, label="Temperature (log)",
                         help="The score is the cosine times e^this; CLIP starts at log(1 / 0.07) = 2.6592")]

    def check_port(self, index: int, shape: Shape) -> Shape:
        return self._rank(index, shape, 1, "a vector (an embedding)")

    def out_shape_ports(self, shapes: list[Shape]) -> Shape:
        a, b = (self.check_port(i, s) for i, s in enumerate(shapes))
        if a != b:
            self._bad(f"the image and the text embeddings must have the same size, got {a[0]} and {b[0]}",
                      hint="Give both projections the same output size.")
        return (1,)

    def build_ports(self, shapes: list[Shape]) -> nn.Module:
        self.out_shape_ports(shapes)
        return G.Similarity(float(self.p("logit_scale", 2.6592)))


# --------------------------------------------------------------------------- #
# Transformer blocks with a pre-norm and a residual (ViT, CLIP, Whisper)
# --------------------------------------------------------------------------- #
def _heads(node: _Gen, dim: int) -> int:
    heads = node._int("heads", 8)
    if dim % heads:
        node._fail(f"token size {dim} does not split into {heads} heads",
                   hint="Pick a head count that divides the token size.")
    return heads


@register("AttentionBlock")
class AttentionBlockNode(_Gen):
    """Pre-norm multi-head self-attention with a residual — the attention of ViT,
    CLIP and Whisper: x + attention(LayerNorm(x)). Shape-preserving over (L, D)."""

    category = "attention"
    label = "Attention block"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="heads", type="int", default=8, min=1, label="Heads"),
            ParamDef(name="causal", type="bool", default=False, label="Causal (only earlier tokens)",
                     help="On for text that is read left to right (CLIP's text encoder, Whisper's decoder)"),
            ParamDef(name="k_bias", type="bool", default=True, label="Bias on keys", help="Off in Whisper"),
            _EPS5,
        ]

    def out_shape(self, in_shape: Shape | None) -> Shape:
        length, dim = self._require_shape(in_shape, ndim=2, what="(L, D): a sequence of token vectors")
        _heads(self, int(dim))
        return (int(length), int(dim))

    def build(self, in_shape: Shape | None) -> nn.Module:
        _length, dim = self.out_shape(in_shape)
        return G.AttentionBlock(dim, _heads(self, dim), bool(self.p("causal", False)), bool(self.p("k_bias", True)),
                                float(self.p("eps", 1e-5)))


@register("CrossAttentionBlock")
class CrossAttentionBlockNode(_Ported):
    """Pre-norm cross-attention with a residual: the tokens ask, another sequence
    (the context) answers — how Whisper's decoder reads the audio.
    Inputs: tokens (L, D), then the context (L', D')."""

    category = "attention"
    label = "Cross-attention block"
    ports = ("tokens", "context")

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="heads", type="int", default=8, min=1, label="Heads"),
            ParamDef(name="k_bias", type="bool", default=True, label="Bias on keys", help="Off in Whisper"),
            _EPS5,
        ]

    def check_port(self, index: int, shape: Shape) -> Shape:
        return self._rank(index, shape, 2, "a sequence (L, D)")

    def out_shape_ports(self, shapes: list[Shape]) -> Shape:
        x, _ctx = (self.check_port(i, s) for i, s in enumerate(shapes))
        _heads(self, x[1])
        return x

    def build_ports(self, shapes: list[Shape]) -> nn.Module:
        x = self.out_shape_ports(shapes)
        return G.CrossAttentionBlock(x[1], int(shapes[1][1]), _heads(self, x[1]), bool(self.p("k_bias", True)),
                                     float(self.p("eps", 1e-5)))


_GELU_KINDS = ["gelu", "gelu_tanh", "quick_gelu", "silu"]


@register("MLPBlock")
class MLPBlockNode(_Gen):
    """Pre-norm feed-forward with a residual: x + fc2(act(fc1(LayerNorm(x)))) — GELU
    in ViT and Whisper, quick GELU in CLIP. Works on the last axis."""

    category = "attention"
    label = "MLP block"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="hidden", type="int", default=2048, min=1, label="Hidden size"),
            ParamDef(name="activation", type="str", default="gelu", label="Activation",
                     help="gelu (ViT, Whisper), quick_gelu (CLIP), gelu_tanh (DiT), silu", choices=_GELU_KINDS),
            _EPS5,
        ]

    def out_shape(self, in_shape: Shape | None) -> Shape:
        sh = self._require_shape(in_shape)
        self._choice("activation", "gelu", tuple(_GELU_KINDS))
        return sh

    def build(self, in_shape: Shape | None) -> nn.Module:
        sh = self.out_shape(in_shape)
        return G.MLPBlock(int(sh[-1]), self._int("hidden", 2048), str(self.p("activation", "gelu")),
                          float(self.p("eps", 1e-5)))


# --------------------------------------------------------------------------- #
# Diffusion: timesteps, conditions, DiT / FLUX blocks, the U-Net and the VAE
# --------------------------------------------------------------------------- #
@register("TimestepEmbedding")
class TimestepEmbeddingNode(_Gen):
    """A diffusion timestep as a vector: sinusoidal features, then Linear - SiLU -
    Linear (DiT, FLUX). One number (Input [1]) -> (dim,)."""

    category = "diffusion"
    label = "Timestep embedding"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="dim", type="int", default=256, min=1, label="Vector size"),
            ParamDef(name="freq_dim", type="int", default=256, min=2, label="Sinusoid features"),
            ParamDef(name="time_factor", type="float", default=1.0, label="Time factor",
                     help="FLUX: 1000 (its timesteps run from 0 to 1)"),
        ]

    def out_shape(self, in_shape: Shape | None) -> Shape:
        sh = self._require_shape(in_shape)
        if math.prod(int(d) for d in sh) != 1:
            self._bad(f"it takes one number per example — the timestep, Input [1] — got {_fmt(sh)}")
        return (self._int("dim", 256),)

    def build(self, in_shape: Shape | None) -> nn.Module:
        (dim,) = self.out_shape(in_shape)
        return G.TimestepEmbedding(dim, self._int("freq_dim", 256, low=2), float(self.p("time_factor", 1.0)))


@register("AdaLayerNorm")
class AdaLayerNormNode(_Ported):
    """Adaptive layer norm: normalize the tokens, then shift and scale them by
    amounts computed from a condition vector — DiT's and FLUX's final norm.
    Inputs: tokens, then the condition (D,)."""

    category = "diffusion"
    label = "Adaptive LayerNorm"
    ports = ("tokens", "condition")

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [_EPS6]

    def check_port(self, index: int, shape: Shape) -> Shape:
        if index == 1:
            return self._rank(1, shape, 1, "a vector (the condition)")
        if not shape:
            self._bad("its input “tokens” is empty")
        return tuple(int(d) for d in shape)

    def out_shape_ports(self, shapes: list[Shape]) -> Shape:
        return self.check_port(0, shapes[0])

    def build_ports(self, shapes: list[Shape]) -> nn.Module:
        x, c = self.check_port(0, shapes[0]), self.check_port(1, shapes[1])
        return G.AdaLayerNorm(x[-1], c[0], float(self.p("eps", 1e-6)))


class _CondBlock(_Ported):
    """A transformer block over (L, D) tokens steered by a condition vector."""

    category = "diffusion"
    ports = ("tokens", "condition")

    def check_port(self, index: int, shape: Shape) -> Shape:
        if index == 1:
            return self._rank(1, shape, 1, "a vector (the condition: timestep, class, text)")
        return self._rank(0, shape, 2, "a sequence of tokens (L, D)")


@register("DiTBlock")
class DiTBlockNode(_CondBlock):
    """A DiT block: self-attention and an MLP, each steered by the condition (the
    timestep and the class) through adaLN-Zero. Inputs: tokens, then the condition."""

    label = "DiT block"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="heads", type="int", default=16, min=1, label="Heads"),
            ParamDef(name="mlp_ratio", type="float", default=4.0, min=0.1, label="MLP width (× token size)"),
            _EPS6,
        ]

    def out_shape_ports(self, shapes: list[Shape]) -> Shape:
        x, _c = (self.check_port(i, s) for i, s in enumerate(shapes))
        _heads(self, x[1])
        return x

    def build_ports(self, shapes: list[Shape]) -> nn.Module:
        x = self.out_shape_ports(shapes)
        return G.DiTBlock(x[1], int(shapes[1][0]), _heads(self, x[1]), float(self.p("mlp_ratio", 4.0)),
                          float(self.p("eps", 1e-6)))


class _FluxNode(_CondBlock):
    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="heads", type="int", default=24, min=1, label="Heads"),
            ParamDef(name="head_dim", type="int", default=128, min=2, label="Head size"),
            ParamDef(name="txt_len", type="int", default=512, min=0, label="Text tokens in front",
                     help="The sequence is [text; image]: how many of its first tokens are the text"),
            ParamDef(name="mlp_ratio", type="float", default=4.0, min=0.1, label="MLP width (× token size)"),
            ParamDef(name="axes", type="intlist", default=[16, 56, 56], label="Rotary axes",
                     help="How the head size splits between the position axes (time, row, column); adds up to the head size"),
            ParamDef(name="theta", type="float", default=10000.0, min=1.0, label="RoPE base"),
        ]

    def _geom(self, shapes: list[Shape]) -> dict:
        x, c = (self.check_port(i, s) for i, s in enumerate(shapes))
        heads, head_dim = self._int("heads", 24), self._int("head_dim", 128, low=2)
        txt_len, axes = self._int("txt_len", 512, low=0), self._ints("axes", [16, 56, 56])
        if sum(axes) != head_dim or any(a % 2 or a < 2 for a in axes):
            self._fail(f"the rotary axes {axes} must be even and add up to the head size {head_dim}",
                       hint="E.g. 16, 56, 56 for heads of 128.")
        if txt_len >= x[0]:
            self._bad(f"the sequence has {x[0]} tokens, not more than its {txt_len} text tokens",
                      hint="Join the text and the image tokens first (Concat), and set the number of text tokens.")
        return dict(dim=x[1], cond_dim=c[0], heads=heads, head_dim=head_dim, txt_len=txt_len,
                    mlp_ratio=float(self.p("mlp_ratio", 4.0)), axes_dims=axes, theta=float(self.p("theta", 10000.0)))

    def out_shape_ports(self, shapes: list[Shape]) -> Shape:
        self._geom(shapes)
        return tuple(int(d) for d in shapes[0])


@register("DoubleStreamBlock")
class DoubleStreamBlockNode(_FluxNode):
    """FLUX's double-stream block: text and image tokens keep their own weights but
    attend jointly over one sequence [text; image]. Inputs: the joint tokens, then
    the condition (timestep + pooled text)."""

    label = "Double-stream block (FLUX)"

    def build_ports(self, shapes: list[Shape]) -> nn.Module:
        return G.DoubleStreamBlock(**self._geom(shapes))


@register("SingleStreamBlock")
class SingleStreamBlockNode(_FluxNode):
    """FLUX's single-stream block: one set of weights for the whole [text; image]
    sequence, attention and MLP side by side. Inputs: tokens, then the condition."""

    label = "Single-stream block (FLUX)"

    def build_ports(self, shapes: list[Shape]) -> nn.Module:
        return G.SingleStreamBlock(**self._geom(shapes))


def _levels(node: _Gen, name: str, default: list[int], groups: int) -> list[int]:
    chans = node._ints(name, default)
    for ch in chans:
        if ch < 1 or ch % groups:
            node._fail(f"every level's channels must divide by the {groups} norm groups, got {chans}",
                       hint="Use multiples of the group count (e.g. 32: 128, 256, 512).")
    return chans


@register("UNet2D")
class UNet2DNode(_Ported):
    """Stable Diffusion's denoiser: from a noisy latent, its timestep and a context
    it predicts the noise. Levels of ResNet blocks go down and back up with skip
    connections across; transformers read the context by cross-attention.
    Inputs: the latent (C, H, W), the timestep, the context (L, D) — the text from a
    text encoder, or a class embedding (a digit to draw)."""

    category = "diffusion"
    label = "U-Net (Stable Diffusion)"
    ports = ("latent", "timestep", "context")

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="out_channels", type="int", default=4, min=1, label="Out channels"),
            ParamDef(name="block_channels", type="intlist", default=[320, 640, 1280, 1280], label="Channels per level"),
            ParamDef(name="layers_per_block", type="int", default=2, min=1, label="ResNet blocks per level"),
            ParamDef(name="attention_levels", type="intlist", default=[1, 1, 1, 0], label="Cross-attention per level",
                     help="1 = the level reads the text (transformer blocks), 0 = convolutions only"),
            ParamDef(name="heads", type="int", default=8, min=1, label="Heads"),
            ParamDef(name="groups", type="int", default=32, min=1, label="Norm groups"),
        ]

    def check_port(self, index: int, shape: Shape) -> Shape:
        if index == 0:
            return self._rank(0, shape, 3, "a latent image (C, H, W)")
        if index == 1:
            return self._one_number(1, shape)
        if len(shape) == 1:  # one vector: a class, added to the timestep's embedding too
            return tuple(int(d) for d in shape)
        return self._rank(2, shape, 2, "a context (L, D) — the text from a text encoder — or one vector (a class)")

    def _geom(self, shapes: list[Shape]) -> dict:
        lat, _t, txt = (self.check_port(i, s) for i, s in enumerate(shapes))
        groups = self._int("groups", 32)
        chans = _levels(self, "block_channels", [320, 640, 1280, 1280], groups)
        attn = self._ints("attention_levels", [1, 1, 1, 0])
        if len(attn) != len(chans):
            self._fail(f"{len(chans)} levels but {len(attn)} cross-attention flags",
                       hint="Give one 0/1 flag per level.")
        heads = self._int("heads", 8)
        for ch in chans:
            if ch % heads:
                self._fail(f"channels {ch} do not split into {heads} heads", hint="Use a head count that divides every level.")
        factor = 2 ** (len(chans) - 1)
        if lat[1] % factor or lat[2] % factor:
            self._bad(f"the latent {lat[1]}×{lat[2]} must divide by {factor} (it is halved {len(chans) - 1} times)",
                      hint="Use a latent size that is a multiple of that, or fewer levels.")
        geom = dict(in_channels=lat[0], out_channels=self._int("out_channels", 4), block_channels=chans,
                    layers_per_block=self._int("layers_per_block", 2), attention_levels=[1 if a else 0 for a in attn],
                    heads=heads, context_dim=txt[-1], groups=groups)
        if len(txt) == 1:
            geom["class_dim"] = txt[0]
        return geom

    def out_shape_ports(self, shapes: list[Shape]) -> Shape:
        g = self._geom(shapes)
        return (g["out_channels"], int(shapes[0][1]), int(shapes[0][2]))

    def build_ports(self, shapes: list[Shape]) -> nn.Module:
        return G.UNet2DCondition(**self._geom(shapes))


class _VAENode(_Gen):
    category = "diffusion"

    @classmethod
    def _common(cls) -> list[ParamDef]:
        return [
            ParamDef(name="block_channels", type="intlist", default=[128, 256, 512, 512], label="Channels per level"),
            ParamDef(name="layers_per_block", type="int", default=2, min=1, label="ResNet blocks per level"),
            ParamDef(name="groups", type="int", default=32, min=1, label="Norm groups"),
            ParamDef(name="scaling", type="float", default=0.18215, label="Latent scale",
                     help="Stable Diffusion multiplies the latent by 0.18215 (and divides before decoding)"),
        ]

    def _chans(self) -> list[int]:
        return _levels(self, "block_channels", [128, 256, 512, 512], self._int("groups", 32))


@register("VAEEncoder")
class VAEEncoderNode(_VAENode):
    """Stable Diffusion's VAE encoder: a picture -> its latent, 8 times smaller per
    side (with 4 levels) and 4 channels. (C, H, W) -> (latent, H/8, W/8)."""

    label = "VAE encoder"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [ParamDef(name="latent_channels", type="int", default=4, min=1, label="Latent channels"), *cls._common()]

    def out_shape(self, in_shape: Shape | None) -> Shape:
        _c, h, w = self._require_shape(in_shape, ndim=3, what="(C, H, W): an image")
        factor = 2 ** (len(self._chans()) - 1)
        if h % factor or w % factor:
            self._bad(f"the image {h}×{w} must divide by {factor} (it is halved {len(self._chans()) - 1} times)")
        return (self._int("latent_channels", 4), h // factor, w // factor)

    def build(self, in_shape: Shape | None) -> nn.Module:
        self.out_shape(in_shape)
        return G.VAEEncoder(int(in_shape[0]), self._int("latent_channels", 4), self._chans(),
                            self._int("layers_per_block", 2), self._int("groups", 32), 1e-6,
                            float(self.p("scaling", 0.18215)))


@register("VAEDecoder")
class VAEDecoderNode(_VAENode):
    """Stable Diffusion's VAE decoder: a latent -> the picture, 8 times larger per
    side (with 4 levels). (latent, h, w) -> (C, 8h, 8w)."""

    label = "VAE decoder"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [ParamDef(name="out_channels", type="int", default=3, min=1, label="Picture channels"), *cls._common()]

    def out_shape(self, in_shape: Shape | None) -> Shape:
        _c, h, w = self._require_shape(in_shape, ndim=3, what="(C, h, w): a latent")
        factor = 2 ** (len(self._chans()) - 1)
        return (self._int("out_channels", 3), h * factor, w * factor)

    def build(self, in_shape: Shape | None) -> nn.Module:
        self.out_shape(in_shape)
        return G.VAEDecoder(self._int("out_channels", 3), int(in_shape[0]), self._chans(),
                            self._int("layers_per_block", 2), self._int("groups", 32), 1e-6,
                            float(self.p("scaling", 0.18215)))


class _NoiseNode(_Ported):
    category = "diffusion"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="steps", type="int", default=1000, min=1, label="Steps in the schedule"),
            ParamDef(name="beta_start", type="float", default=0.00085, min=0.0, label="Noise at the first step"),
            ParamDef(name="beta_end", type="float", default=0.012, min=0.0, label="Noise at the last step"),
            ParamDef(name="schedule", type="str", default="scaled_linear", label="Schedule",
                     help="scaled_linear (Stable Diffusion) or linear (DDPM)", choices=["scaled_linear", "linear"]),
        ]

    def check_port(self, index: int, shape: Shape) -> Shape:
        if index == 2:
            return self._one_number(2, shape)
        if not shape:
            self._bad(f"its input “{self.input_ports()[index]}” is empty")
        return tuple(int(d) for d in shape)

    def out_shape_ports(self, shapes: list[Shape]) -> Shape:
        a, b, _t = (self.check_port(i, s) for i, s in enumerate(shapes))
        if a != b:
            self._bad(f"the noise ({_fmt(b)}) must have the latent's shape ({_fmt(a)})")
        return a

    def build_ports(self, shapes: list[Shape]) -> nn.Module:
        self.out_shape_ports(shapes)
        cls = G.PredictClean if self.type_name == "PredictClean" else G.AddNoise
        return cls(self._int("steps", 1000), float(self.p("beta_start", 0.00085)), float(self.p("beta_end", 0.012)),
                   self._choice("schedule", "scaled_linear", ("scaled_linear", "linear")))


@register("AddNoise")
class AddNoiseNode(_NoiseNode):
    """Mixes a clean latent with noise for a timestep, as a diffusion model is
    trained (no parameters). Inputs: the clean latent, the noise, the timestep."""

    label = "Add noise"
    ports = ("clean latent", "noise", "timestep")


@register("PredictClean")
class PredictCleanNode(_NoiseNode):
    """The guess of the clean latent from a noisy one and the predicted noise — one
    denoising step at once (no parameters). Inputs: the noisy latent, the
    predicted noise, the timestep."""

    label = "Predict the clean latent"
    ports = ("noisy latent", "predicted noise", "timestep")


@register("Reparameterize")
class ReparameterizeNode(_Gen):
    """A VAE's sampling step: half of its input is a mean, half a log-variance. It
    passes on a sample while training, the mean otherwise, and random codes when
    drawing new pictures; its KL term, added to the loss, keeps the codes near
    N(0, I) — where new pictures come from. (2k, …) -> (k, …)."""

    category = "diffusion"
    label = "VAE sampling"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [ParamDef(name="kl_weight", type="float", default=1.0, min=0.0, label="KL weight (beta)",
                         help="How strongly the codes are kept near N(0, I): 1 = a standard VAE, 0 = a plain "
                              "autoencoder, more = smoother but blurrier pictures")]

    def out_shape(self, in_shape: Shape | None) -> Shape:
        sh = self._require_shape(in_shape)
        if int(sh[0]) % 2:
            self._bad(f"its first axis ({sh[0]}) must be even: half a mean, half a log-variance",
                      hint="Give the layer before it twice the code size (e.g. a Linear of 32 for a code of 16).")
        return (int(sh[0]) // 2, *[int(d) for d in sh[1:]])

    def build(self, in_shape: Shape | None) -> nn.Module:
        self.out_shape(in_shape)
        return G.Reparameterize(float(self.p("kl_weight", 1.0)))


# --------------------------------------------------------------------------- #
# 3-D: TripoSR's triplane
# --------------------------------------------------------------------------- #
@register("TriplaneTokens")
class TriplaneTokensNode(_Gen):
    """TripoSR's learned triplane: three planes (xy, xz, yz) of size×size cells, one
    token per cell -> (3·size·size, channels). Its input only gives the batch
    size; the image is read later, by cross-attention."""

    category = "3d"
    label = "Triplane tokens"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="size", type="int", default=32, min=1, label="Cells per side"),
            ParamDef(name="channels", type="int", default=1024, min=1, label="Token size"),
        ]

    def out_shape(self, in_shape: Shape | None) -> Shape:
        self._require_shape(in_shape)
        size = self._int("size", 32)
        return (3 * size * size, self._int("channels", 1024))

    def build(self, in_shape: Shape | None) -> nn.Module:
        self.out_shape(in_shape)
        return G.TriplaneTokens(3, self._int("size", 32), self._int("channels", 1024))


@register("TriplaneTransformer")
class TriplaneTransformerNode(_Ported):
    """TripoSR's backbone: the triplane tokens attend to each other and, by
    cross-attention, to the image. Inputs: the triplane tokens, then the image tokens."""

    category = "3d"
    label = "Triplane transformer"
    ports = ("triplane", "image")

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="layers", type="int", default=16, min=1, label="Layers"),
            ParamDef(name="heads", type="int", default=16, min=1, label="Heads"),
            ParamDef(name="head_dim", type="int", default=64, min=1, label="Head size"),
            ParamDef(name="groups", type="int", default=32, min=1, label="Norm groups"),
        ]

    def check_port(self, index: int, shape: Shape) -> Shape:
        return self._rank(index, shape, 2, "a sequence of tokens (L, D)")

    def _geom(self, shapes: list[Shape]) -> dict:
        x, ctx = (self.check_port(i, s) for i, s in enumerate(shapes))
        groups = self._int("groups", 32)
        if x[1] % groups:
            self._fail(f"token size {x[1]} does not divide by the {groups} norm groups")
        return dict(dim=x[1], context_dim=ctx[1], layers=self._int("layers", 16), heads=self._int("heads", 16),
                    head_dim=self._int("head_dim", 64), groups=groups)

    def out_shape_ports(self, shapes: list[Shape]) -> Shape:
        self._geom(shapes)
        return tuple(int(d) for d in shapes[0])

    def build_ports(self, shapes: list[Shape]) -> nn.Module:
        return G.TriplaneTransformer(**self._geom(shapes))


@register("TriplaneUpsample")
class TriplaneUpsampleNode(_Gen):
    """Triplane tokens back into three feature planes, twice as fine (TripoSR's
    post-processor). (3·s·s, C) -> (3, out, 2s, 2s)."""

    category = "3d"
    label = "Triplane upsample"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [ParamDef(name="out_channels", type="int", default=40, min=1, label="Features per plane")]

    def out_shape(self, in_shape: Shape | None) -> Shape:
        n, _c = self._require_shape(in_shape, ndim=2, what="(3·s·s, C): triplane tokens")
        s = int(round(math.sqrt(n // 3))) if n % 3 == 0 else 0
        if not s or 3 * s * s != n:
            self._bad(f"{n} tokens are not three square planes")
        return (3, self._int("out_channels", 40), 2 * s, 2 * s)

    def build(self, in_shape: Shape | None) -> nn.Module:
        self.out_shape(in_shape)
        return G.TriplaneUpsample(int(in_shape[1]), self._int("out_channels", 40), 3)


@register("TriplaneSample")
class TriplaneSampleNode(_Ported):
    """Reads the triplane at 3-D points: each point is projected onto the three
    planes and the features there are joined -> (N, 3·C), what a small MLP turns
    into density and colour. Inputs: the planes (3, C, H, W), then the points (N, 3)."""

    category = "3d"
    label = "Triplane sample"
    ports = ("triplane", "points")

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [ParamDef(name="radius", type="float", default=0.87, min=0.0, label="Scene radius",
                         help="Points from -radius to +radius span the planes")]

    def check_port(self, index: int, shape: Shape) -> Shape:
        if index == 0:
            sh = self._rank(0, shape, 4, "three feature planes (3, C, H, W)")
            if sh[0] != 3:
                self._bad(f"its input “triplane” must hold 3 planes, got {sh[0]}")
            return sh
        sh = self._rank(1, shape, 2, "points (N, 3)")
        if sh[1] != 3:
            self._bad(f"each point needs 3 coordinates (x, y, z), got {sh[1]}")
        return sh

    def out_shape_ports(self, shapes: list[Shape]) -> Shape:
        planes, pts = (self.check_port(i, s) for i, s in enumerate(shapes))
        return (pts[0], 3 * planes[1])

    def build_ports(self, shapes: list[Shape]) -> nn.Module:
        self.out_shape_ports(shapes)
        return G.TriplaneSample(float(self.p("radius", 0.87)))


# --------------------------------------------------------------------------- #
# Small generic layers
# --------------------------------------------------------------------------- #
@register("Transpose")
class TransposeNode(_Gen):
    """Swaps the two axes of each example: (A, B) -> (B, A) — audio features
    (channels, time) become a sequence (time, channels)."""

    category = "shape"
    label = "Transpose"

    def out_shape(self, in_shape: Shape | None) -> Shape:
        a, b = self._require_shape(in_shape, ndim=2, what="(A, B)")
        return (int(b), int(a))

    def build(self, in_shape: Shape | None) -> nn.Module:
        self.out_shape(in_shape)
        return G.Transpose()


@register("SliceTokens")
class SliceTokensNode(_Gen):
    """Keeps part of a sequence: tokens start … start + length (length 0 = to the
    end) — FLUX keeps the image tokens after its joint blocks."""

    category = "sequence"
    label = "Slice tokens"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="start", type="int", default=0, min=0, label="From token"),
            ParamDef(name="length", type="int", default=0, min=0, label="How many", help="0 = to the end"),
        ]

    def out_shape(self, in_shape: Shape | None) -> Shape:
        sh = self._require_shape(in_shape)
        start, length = self._int("start", 0, low=0), self._int("length", 0, low=0)
        n = int(sh[0])
        end = start + length if length else n
        if start >= n or end > n:
            self._bad(f"tokens {start}…{end - 1} do not fit in a sequence of {n}",
                      hint="Pick a start and a length inside the sequence.")
        return (end - start, *[int(d) for d in sh[1:]])

    def build(self, in_shape: Shape | None) -> nn.Module:
        self.out_shape(in_shape)
        return G.SliceTokens(self._int("start", 0, low=0), self._int("length", 0, low=0))


def _conv_len(n: int, k: int, s: int, p: int) -> int:
    return (n + 2 * p - k) // s + 1


class _ConvND(_Gen):
    category = "conv"
    ndim = 1

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="out_channels", type="int", default=64, min=1, label="Out channels"),
            ParamDef(name="kernel_size", type="int", default=3, min=1, label="Kernel size"),
            ParamDef(name="stride", type="int", default=1, min=1, label="Stride"),
            ParamDef(name="padding", type="int", default=1, min=0, label="Padding"),
        ]

    def _what(self) -> str:
        return "(C, L): channels × time" if self.ndim == 1 else "(C, D, H, W): channels × depth × height × width"

    def out_shape(self, in_shape: Shape | None) -> Shape:
        sh = self._require_shape(in_shape, ndim=self.ndim + 1, what=self._what())
        k, s, p = self._int("kernel_size", 3), self._int("stride", 1), self._int("padding", 1, low=0)
        sizes = [_conv_len(int(n), k, s, p) for n in sh[1:]]
        if min(sizes) < 1:
            self._bad(f"a {k}-wide kernel does not fit in {_fmt(sh[1:])}", hint="Use a smaller kernel or more padding.")
        return (self._int("out_channels", 64), *sizes)

    def build(self, in_shape: Shape | None) -> nn.Module:
        self.out_shape(in_shape)
        cls = nn.Conv1d if self.ndim == 1 else nn.Conv3d
        return cls(int(in_shape[0]), self._int("out_channels", 64), self._int("kernel_size", 3),
                   stride=self._int("stride", 1), padding=self._int("padding", 1, low=0))


@register("Conv1d")
class Conv1dNode(_ConvND):
    """1-D convolution over time: (C, L) -> (out, L') — Whisper's front end reads the
    audio spectrogram with two of them."""

    label = "Conv1d"
    ndim = 1


@register("Conv3d")
class Conv3dNode(_ConvND):
    """3-D convolution over volumes or video: (C, D, H, W) -> (out, D', H', W')."""

    label = "Conv3d"
    ndim = 3


@register("Upsample")
class UpsampleNode(_Gen):
    """Makes a feature map larger: (C, H, W) -> (C, H·s, W·s), copying each pixel
    (nearest) or blending neighbours (bilinear) — how decoders grow a picture."""

    category = "conv"
    label = "Upsample"

    @classmethod
    def param_defs(cls) -> list[ParamDef]:
        return [
            ParamDef(name="scale", type="int", default=2, min=1, label="Scale"),
            ParamDef(name="mode", type="str", default="nearest", label="Mode", choices=["nearest", "bilinear"]),
        ]

    def out_shape(self, in_shape: Shape | None) -> Shape:
        c, h, w = self._require_shape(in_shape, ndim=3, what="(C, H, W): a feature map")
        s = self._int("scale", 2)
        self._choice("mode", "nearest", ("nearest", "bilinear"))
        return (int(c), int(h) * s, int(w) * s)

    def build(self, in_shape: Shape | None) -> nn.Module:
        self.out_shape(in_shape)
        return nn.Upsample(scale_factor=float(self._int("scale", 2)), mode=str(self.p("mode", "nearest")))


@register("SiLU")
class SiLUNode(_Gen):
    """SiLU (swish): x · sigmoid(x) — the activation of diffusion models and of
    TripoSR's NeRF MLP. Shape-preserving."""

    category = "activation"
    label = "SiLU"

    def out_shape(self, in_shape: Shape | None) -> Shape:
        return self._require_shape(in_shape)

    def build(self, in_shape: Shape | None) -> nn.Module:
        return nn.SiLU()
