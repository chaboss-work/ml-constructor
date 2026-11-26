"""The building blocks of today's image, 3D, speech and vision models (Stage 19).

Plain torch modules for the layers inside ViT and CLIP, Stable Diffusion (its
text encoder, U-Net and VAE), DiT and FLUX, TripoSR and Whisper: patch
embeddings, position tables, pre-norm attention / cross-attention / MLP blocks,
timestep embeddings, adaptive norms, the diffusion transformer blocks, the U-Net
and the VAE, the triplane parts of a 3-D generator, and a few small helpers.

The structures follow the published implementations (diffusers, the official
DiT and FLUX code, TripoSR, Hugging Face's ViT / CLIP / Whisper) closely enough
that a real configuration has exactly the published number of parameters.

Like :mod:`mlconstructor.llm_modules` this file is self-contained (torch only):
the PyTorch export copies everything below the "export" marker verbatim into
``model.py``, and every module keeps the arguments it was made with in
``self.config``, so the exporter can write the same constructor call back out.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- export: begin ---


def timestep_features(t, dim, max_period=10000.0, time_factor=1.0):
    """Sinusoidal features of a diffusion timestep, (B, 1) -> (B, dim): cosines,
    then sines (DiT, FLUX, and Stable Diffusion's U-Net with flip_sin_to_cos)."""
    t = t.reshape(-1).float() * time_factor
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / max(half, 1))
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def sinusoids(length, channels, max_timescale=10000.0):
    """Whisper's fixed position table (length, channels): sines, then cosines."""
    inc = math.log(max_timescale) / max(channels // 2 - 1, 1)
    inv = torch.exp(-inc * torch.arange(channels // 2, dtype=torch.float32))
    scaled = torch.arange(length, dtype=torch.float32)[:, None] * inv[None, :]
    table = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)
    if channels % 2:
        table = torch.cat([table, torch.zeros(length, 1)], dim=1)
    return table


def sincos_2d(dim, grid):
    """DiT's fixed 2-D sine-cosine position table (grid * grid, dim): half the
    numbers encode the row, half the column."""
    def one_axis(d, pos):
        omega = 1.0 / 10000 ** (torch.arange(d // 2, dtype=torch.float64) / max(d / 2.0, 1.0))
        out = pos.reshape(-1).double()[:, None] * omega[None]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    coords = torch.arange(grid, dtype=torch.float64)
    gw, gh = torch.meshgrid(coords, coords, indexing="xy")
    table = torch.cat([one_axis(dim // 2, gh), one_axis(dim - dim // 2, gw)], dim=1)
    if table.shape[1] < dim:
        table = torch.cat([table, torch.zeros(table.shape[0], dim - table.shape[1], dtype=table.dtype)], dim=1)
    return table.float()


def activate(x, kind):
    """gelu (ViT, Whisper), gelu_tanh (DiT, FLUX), quick_gelu (CLIP) or silu."""
    if kind == "gelu_tanh":
        return F.gelu(x, approximate="tanh")
    if kind == "quick_gelu":
        return x * torch.sigmoid(1.702 * x)
    if kind == "silu":
        return F.silu(x)
    return F.gelu(x)


def modulate(x, shift, scale):
    """Adaptive norm: x * (1 + scale) + shift, with one (shift, scale) per example."""
    if x.dim() == 3:
        shift, scale = shift[:, None], scale[:, None]
    return x * (1 + scale) + shift


def heads_split(x, heads):
    """(B, L, H * D) -> (B, H, L, D)."""
    b, n, _ = x.shape
    return x.view(b, n, heads, -1).transpose(1, 2)


def heads_merge(x):
    """(B, H, L, D) -> (B, L, H * D)."""
    b, h, n, d = x.shape
    return x.transpose(1, 2).reshape(b, n, h * d)


class TimestepEmbedding(nn.Module):
    """A diffusion timestep as a vector: sinusoidal features, then Linear - SiLU -
    Linear (DiT's t_embedder, FLUX's time_in, the U-Net's time embedding)."""

    def __init__(self, dim, freq_dim=256, time_factor=1.0, max_period=10000.0):
        super().__init__()
        self.config = dict(dim=dim, freq_dim=freq_dim, time_factor=time_factor, max_period=max_period)
        self.freq_dim, self.time_factor, self.max_period = int(freq_dim), float(time_factor), float(max_period)
        self.linear_1 = nn.Linear(freq_dim, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, t):
        h = timestep_features(t, self.freq_dim, self.max_period, self.time_factor).to(self.linear_1.weight.dtype)
        return self.linear_2(F.silu(self.linear_1(h)))


class PatchEmbed(nn.Module):
    """Cuts an image into patches, one token per patch: a convolution whose stride
    is the patch size, (C, H, W) -> (N, dim). Optionally a class token in front and
    a position table: learned (ViT, CLIP), or DiT's fixed sine-cosine one (kept as
    a frozen parameter, as DiT does). A table made for another grid (pos_grid) is
    resized to the image's grid (how TripoSR feeds 512-pixel images to DINO)."""

    def __init__(self, in_channels, dim, patch, grid, cls_token=False, position="learned", bias=True, pos_grid=0):
        super().__init__()
        self.config = dict(in_channels=in_channels, dim=dim, patch=patch, grid=grid, cls_token=cls_token,
                           position=position, bias=bias, pos_grid=pos_grid)
        self.table_grid = int(pos_grid or grid)
        self.proj = nn.Conv2d(in_channels, dim, patch, stride=patch, bias=bias)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim)) if cls_token else None
        rows = self.table_grid * self.table_grid + (1 if cls_token else 0)
        self.pos = None
        if position == "learned":
            self.pos = nn.Parameter(torch.zeros(1, rows, dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
        elif position == "sincos":
            self.pos = nn.Parameter(torch.zeros(1, rows, dim), requires_grad=False)
            if not self.pos.is_meta:
                with torch.no_grad():
                    self.pos[0, rows - self.table_grid ** 2:] = sincos_2d(dim, self.table_grid)

    def _table(self, gh, gw):
        g = self.table_grid
        if (gh, gw) == (g, g):
            return self.pos
        extra = 1 if self.cls is not None else 0
        head, cells = self.pos[:, :extra], self.pos[:, extra:]
        cells = cells.reshape(1, g, g, -1).permute(0, 3, 1, 2)
        cells = F.interpolate(cells, size=(gh, gw), mode="bicubic", align_corners=False)
        return torch.cat([head, cells.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)], dim=1)

    def forward(self, x):
        x = self.proj(x)
        b, _, gh, gw = x.shape
        x = x.flatten(2).transpose(1, 2)
        if self.cls is not None:
            x = torch.cat([self.cls.expand(b, -1, -1), x], dim=1)
        if self.pos is not None:
            x = x + self._table(gh, gw)
        return x


class Unpatchify(nn.Module):
    """Patch tokens back into an image, (N, patch * patch * C) -> (C, H, W): the
    inverse of a patch embedding (DiT's last step). The patches form a square."""

    def __init__(self, patch, channels):
        super().__init__()
        self.config = dict(patch=patch, channels=channels)
        self.patch, self.channels = int(patch), int(channels)

    def forward(self, x):
        b, n, _ = x.shape
        g, p, c = int(round(n ** 0.5)), self.patch, self.channels
        x = x.reshape(b, g, g, p, p, c).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(b, c, g * p, g * p)


class PositionEmbedding(nn.Module):
    """Adds a position table to a (L, D) sequence: learned (CLIP's text encoder,
    Whisper's decoder) or fixed sinusoids (Whisper's encoder, stored as a frozen
    parameter, as Hugging Face does)."""

    def __init__(self, length, dim, kind="learned"):
        super().__init__()
        self.config = dict(length=length, dim=dim, kind=kind)
        self.weight = nn.Parameter(torch.empty(length, dim), requires_grad=kind == "learned")
        if kind == "learned":
            nn.init.normal_(self.weight, std=0.02)
        elif not self.weight.is_meta:
            with torch.no_grad():
                self.weight.copy_(sinusoids(length, dim))

    def forward(self, x):
        return x + self.weight[: x.shape[1]]


class AttentionBlock(nn.Module):
    """Pre-norm multi-head self-attention with a residual: x + attention(LayerNorm(x))
    — the attention of ViT, CLIP and Whisper. causal: a token sees only earlier ones
    (CLIP's text encoder, Whisper's decoder); k_bias=False: Whisper's keys have no bias."""

    def __init__(self, dim, heads, causal=False, k_bias=True, eps=1e-5):
        super().__init__()
        self.config = dict(dim=dim, heads=heads, causal=causal, k_bias=k_bias, eps=eps)
        self.heads, self.causal = int(heads), bool(causal)
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim, bias=k_bias)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = (heads_split(p(h), self.heads) for p in (self.q_proj, self.k_proj, self.v_proj))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return x + self.out_proj(heads_merge(out))


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention with a residual: the queries come from x, the keys
    and values from another sequence, the context (Whisper's decoder reads the
    audio this way). x + attention(LayerNorm(x), context)."""

    def __init__(self, dim, context_dim, heads, k_bias=True, eps=1e-5):
        super().__init__()
        self.config = dict(dim=dim, context_dim=context_dim, heads=heads, k_bias=k_bias, eps=eps)
        self.heads = int(heads)
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(context_dim, dim, bias=k_bias)
        self.v_proj = nn.Linear(context_dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, context):
        q = heads_split(self.q_proj(self.norm(x)), self.heads)
        k, v = heads_split(self.k_proj(context), self.heads), heads_split(self.v_proj(context), self.heads)
        return x + self.out_proj(heads_merge(F.scaled_dot_product_attention(q, k, v)))


class MLPBlock(nn.Module):
    """Pre-norm feed-forward with a residual: x + fc2(act(fc1(LayerNorm(x)))) — GELU
    in ViT and Whisper, quick GELU in CLIP."""

    def __init__(self, dim, hidden, activation="gelu", eps=1e-5):
        super().__init__()
        self.config = dict(dim=dim, hidden=hidden, activation=activation, eps=eps)
        self.activation = activation
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return x + self.fc2(activate(self.fc1(self.norm(x)), self.activation))


class AdaLayerNorm(nn.Module):
    """Adaptive layer norm: a LayerNorm without its own scale, then a shift and a
    scale computed from a condition — shift, scale = Linear(SiLU(c)) — DiT's and
    FLUX's final norm."""

    def __init__(self, dim, cond_dim, eps=1e-6):
        super().__init__()
        self.config = dict(dim=dim, cond_dim=cond_dim, eps=eps)
        self.dim, self.eps = int(dim), float(eps)
        self.linear = nn.Linear(cond_dim, 2 * dim)

    def forward(self, x, cond):
        shift, scale = self.linear(F.silu(cond)).chunk(2, dim=-1)
        return modulate(F.layer_norm(x, (self.dim,), eps=self.eps), shift, scale)


class DiTBlock(nn.Module):
    """A DiT block: self-attention and an MLP, each steered by the condition (the
    timestep and the class) through adaLN-Zero — a shift, a scale and a gate per
    branch, all computed from the condition and starting at zero."""

    def __init__(self, dim, cond_dim, heads, mlp_ratio=4.0, eps=1e-6):
        super().__init__()
        self.config = dict(dim=dim, cond_dim=cond_dim, heads=heads, mlp_ratio=mlp_ratio, eps=eps)
        self.dim, self.heads, self.eps = int(dim), int(heads), float(eps)
        hidden = int(dim * mlp_ratio)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.adaLN = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.adaLN(F.silu(c)).chunk(6, dim=-1)
        h = modulate(F.layer_norm(x, (self.dim,), eps=self.eps), s1, sc1)
        q, k, v = (heads_split(t, self.heads) for t in self.qkv(h).chunk(3, dim=-1))
        x = x + g1[:, None] * self.proj(heads_merge(F.scaled_dot_product_attention(q, k, v)))
        h = modulate(F.layer_norm(x, (self.dim,), eps=self.eps), s2, sc2)
        return x + g2[:, None] * self.fc2(F.gelu(self.fc1(h), approximate="tanh"))


def flux_positions(txt_len, img_len, axes, device):
    """FLUX's position ids (L, axes): text tokens at (0, 0, 0), image tokens at
    (0, row, column) of their square grid."""
    ids = torch.zeros(txt_len + img_len, axes, device=device)
    side = int(round(img_len ** 0.5))
    r = torch.arange(img_len, device=device)
    if axes >= 3 and side * side == img_len:
        ids[txt_len:, 1] = (r // side).float()
        ids[txt_len:, 2] = (r % side).float()
    else:
        ids[txt_len:, axes - 1] = r.float()
    return ids


def rope_axes(ids, axes_dims, theta):
    """Rotary angles for several position axes, (L, head_dim / 2): each axis turns
    its own slice of the pairs (FLUX: 16 numbers for the "time" axis, 56 + 56 for
    the image's rows and columns)."""
    parts = []
    for i, d in enumerate(axes_dims):
        freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32, device=ids.device) / d))
        parts.append(ids[:, i].float()[:, None] * freqs[None])
    angles = torch.cat(parts, dim=-1)
    return angles.cos(), angles.sin()


def rope_pairs(x, cos, sin):
    """Rotates neighbouring pairs (x0, x1), (x2, x3), … of (B, H, L, D) by their angles."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2).to(x.dtype)


class _FluxAttention(nn.Module):
    """What FLUX's two block kinds share: query/key norms and joint attention with
    3-axis rotary positions over [text; image]."""

    def _setup(self, heads, head_dim, txt_len, axes_dims, theta):
        self.heads, self.head_dim, self.txt_len = int(heads), int(head_dim), int(txt_len)
        self.axes_dims, self.theta = [int(a) for a in axes_dims], float(theta)

    def _qkv(self, qkv, q_norm, k_norm):
        q, k, v = (heads_split(t, self.heads) for t in qkv.chunk(3, dim=-1))
        return q_norm(q), k_norm(k), v

    def _attend(self, q, k, v):
        n = q.shape[2]
        ids = flux_positions(self.txt_len, n - self.txt_len, len(self.axes_dims), q.device)
        cos, sin = rope_axes(ids, self.axes_dims, self.theta)
        q, k = rope_pairs(q, cos, sin), rope_pairs(k, cos, sin)
        return heads_merge(F.scaled_dot_product_attention(q, k, v))


class DoubleStreamBlock(_FluxAttention):
    """FLUX's double-stream block: text and image tokens keep separate weights
    (modulation, attention projections, MLP) but attend jointly, over one
    sequence [text; image]. The joint sequence comes in whole; its first txt_len
    tokens are the text."""

    def __init__(self, dim, cond_dim, heads, head_dim, txt_len, mlp_ratio=4.0, axes_dims=(16, 56, 56),
                 theta=10000.0):
        super().__init__()
        self.config = dict(dim=dim, cond_dim=cond_dim, heads=heads, head_dim=head_dim, txt_len=txt_len,
                           mlp_ratio=mlp_ratio, axes_dims=list(axes_dims), theta=theta)
        self._setup(heads, head_dim, txt_len, axes_dims, theta)
        self.dim = int(dim)
        inner, hidden = heads * head_dim, int(dim * mlp_ratio)
        for s in ("img", "txt"):
            setattr(self, f"{s}_mod", nn.Linear(cond_dim, 6 * dim))
            setattr(self, f"{s}_qkv", nn.Linear(dim, 3 * inner))
            setattr(self, f"{s}_q_norm", nn.RMSNorm(head_dim, eps=1e-6))
            setattr(self, f"{s}_k_norm", nn.RMSNorm(head_dim, eps=1e-6))
            setattr(self, f"{s}_proj", nn.Linear(inner, dim))
            setattr(self, f"{s}_fc1", nn.Linear(dim, hidden))
            setattr(self, f"{s}_fc2", nn.Linear(hidden, dim))

    def _norm(self, x):
        return F.layer_norm(x, (self.dim,), eps=1e-6)

    def forward(self, x, vec):
        t = self.txt_len
        streams = {"txt": x[:, :t], "img": x[:, t:]}
        mods, qkv = {}, {}
        for s, h in streams.items():
            mods[s] = getattr(self, f"{s}_mod")(F.silu(vec)).chunk(6, dim=-1)
            shift, scale = mods[s][0], mods[s][1]
            qkv[s] = self._qkv(getattr(self, f"{s}_qkv")(modulate(self._norm(h), shift, scale)),
                               getattr(self, f"{s}_q_norm"), getattr(self, f"{s}_k_norm"))
        q, k, v = (torch.cat([qkv["txt"][i], qkv["img"][i]], dim=2) for i in range(3))
        attn = self._attend(q, k, v)
        attn_part = {"txt": attn[:, :t], "img": attn[:, t:]}
        out = {}
        for s, h in streams.items():
            _, _, g1, s2, sc2, g2 = mods[s]
            h = h + g1[:, None] * getattr(self, f"{s}_proj")(attn_part[s])
            m = getattr(self, f"{s}_fc2")(F.gelu(getattr(self, f"{s}_fc1")(modulate(self._norm(h), s2, sc2)),
                                                 approximate="tanh"))
            out[s] = h + g2[:, None] * m
        return torch.cat([out["txt"], out["img"]], dim=1)


class SingleStreamBlock(_FluxAttention):
    """FLUX's single-stream block: one set of weights for the whole [text; image]
    sequence; the attention and the MLP run side by side (one projection in,
    one out), steered by the condition."""

    def __init__(self, dim, cond_dim, heads, head_dim, txt_len, mlp_ratio=4.0, axes_dims=(16, 56, 56),
                 theta=10000.0):
        super().__init__()
        self.config = dict(dim=dim, cond_dim=cond_dim, heads=heads, head_dim=head_dim, txt_len=txt_len,
                           mlp_ratio=mlp_ratio, axes_dims=list(axes_dims), theta=theta)
        self._setup(heads, head_dim, txt_len, axes_dims, theta)
        self.dim = int(dim)
        self.inner, self.hidden = heads * head_dim, int(dim * mlp_ratio)
        self.modulation = nn.Linear(cond_dim, 3 * dim)
        self.linear1 = nn.Linear(dim, 3 * self.inner + self.hidden)
        self.linear2 = nn.Linear(self.inner + self.hidden, dim)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)

    def forward(self, x, vec):
        shift, scale, gate = self.modulation(F.silu(vec)).chunk(3, dim=-1)
        h = modulate(F.layer_norm(x, (self.dim,), eps=1e-6), shift, scale)
        qkv, mlp = torch.split(self.linear1(h), [3 * self.inner, self.hidden], dim=-1)
        attn = self._attend(*self._qkv(qkv, self.q_norm, self.k_norm))
        return x + gate[:, None] * self.linear2(torch.cat([attn, F.gelu(mlp, approximate="tanh")], dim=-1))


class ResnetBlock2D(nn.Module):
    """The convolutional block of the U-Net and the VAE: two GroupNorm - SiLU - 3x3
    convolutions with a residual (a 1x1 convolution when the channels change); in
    the U-Net the timestep is added in between."""

    def __init__(self, in_ch, out_ch, temb_ch=0, groups=32, eps=1e-5):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch, eps=eps)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_emb_proj = nn.Linear(temb_ch, out_ch) if temb_ch else None
        self.norm2 = nn.GroupNorm(groups, out_ch, eps=eps)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x, temb=None):
        h = self.conv1(F.silu(self.norm1(x)))
        if self.time_emb_proj is not None:
            h = h + self.time_emb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return (x if self.conv_shortcut is None else self.conv_shortcut(x)) + h


class Attention(nn.Module):
    """Multi-head attention as diffusers builds it: q from x, k and v from x or a
    context, no bias on q/k/v, a bias on the output."""

    def __init__(self, dim, heads, head_dim, context_dim=None, qkv_bias=False):
        super().__init__()
        inner = heads * head_dim
        self.heads = int(heads)
        self.to_q = nn.Linear(dim, inner, bias=qkv_bias)
        self.to_k = nn.Linear(context_dim or dim, inner, bias=qkv_bias)
        self.to_v = nn.Linear(context_dim or dim, inner, bias=qkv_bias)
        self.to_out = nn.Linear(inner, dim)

    def forward(self, x, context=None):
        ctx = x if context is None else context
        q, k, v = heads_split(self.to_q(x), self.heads), heads_split(self.to_k(ctx), self.heads), \
            heads_split(self.to_v(ctx), self.heads)
        return self.to_out(heads_merge(F.scaled_dot_product_attention(q, k, v)))


class BasicTransformerBlock(nn.Module):
    """Self-attention, cross-attention to a context (the text, the image) and a GEGLU
    feed-forward, each after a LayerNorm and with a residual — the transformer
    inside Stable Diffusion's U-Net and TripoSR's backbone."""

    def __init__(self, dim, heads, head_dim, context_dim, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(dim, heads, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = Attention(dim, heads, head_dim, context_dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ff_proj = nn.Linear(dim, 2 * ff_mult * dim)
        self.ff_out = nn.Linear(ff_mult * dim, dim)

    def forward(self, x, context):
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        a, gate = self.ff_proj(self.norm3(x)).chunk(2, dim=-1)
        return x + self.ff_out(a * F.gelu(gate))


class Transformer2D(nn.Module):
    """A transformer over the pixels of a feature map: GroupNorm, a 1x1 projection,
    transformer blocks reading the text, a 1x1 projection back, a residual."""

    def __init__(self, ch, heads, context_dim, groups=32, depth=1):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch, eps=1e-6)
        self.proj_in = nn.Conv2d(ch, ch, 1)
        self.blocks = nn.ModuleList([BasicTransformerBlock(ch, heads, ch // heads, context_dim) for _ in range(depth)])
        self.proj_out = nn.Conv2d(ch, ch, 1)

    def forward(self, x, context):
        b, c, h, w = x.shape
        y = self.proj_in(self.norm(x)).permute(0, 2, 3, 1).reshape(b, h * w, c)
        for blk in self.blocks:
            y = blk(y, context)
        return x + self.proj_out(y.reshape(b, h, w, c).permute(0, 3, 1, 2))


class UNet2DCondition(nn.Module):
    """Stable Diffusion's denoiser: from a noisy latent, its timestep and the text
    (the text encoder's output) it predicts the noise. Levels of ResNet blocks
    halve the size on the way down and double it on the way up; skip connections
    carry every down-level output across; transformers with cross-attention read
    the text at the levels marked in attention_levels and in the middle."""

    def __init__(self, in_channels=4, out_channels=4, block_channels=(320, 640, 1280, 1280), layers_per_block=2,
                 attention_levels=(1, 1, 1, 0), heads=8, context_dim=768, groups=32, eps=1e-5):
        super().__init__()
        self.config = dict(in_channels=in_channels, out_channels=out_channels, block_channels=list(block_channels),
                           layers_per_block=layers_per_block, attention_levels=list(attention_levels), heads=heads,
                           context_dim=context_dim, groups=groups, eps=eps)
        c0, n = block_channels[0], len(block_channels)
        tdim = 4 * c0
        self.time_embedding = TimestepEmbedding(tdim, freq_dim=c0)
        self.conv_in = nn.Conv2d(in_channels, c0, 3, padding=1)
        self.down = nn.ModuleList()
        skips, ch = [c0], c0
        for i, out in enumerate(block_channels):
            level = nn.Module()
            level.resnets = nn.ModuleList(
                [ResnetBlock2D(ch if j == 0 else out, out, tdim, groups, eps) for j in range(layers_per_block)])
            level.attentions = nn.ModuleList(
                [Transformer2D(out, heads, context_dim, groups) for _ in range(layers_per_block)]
            ) if attention_levels[i] else None
            ch = out
            skips += [out] * layers_per_block
            level.downsample = nn.Conv2d(out, out, 3, stride=2, padding=1) if i < n - 1 else None
            if level.downsample is not None:
                skips.append(out)
            self.down.append(level)
        self.mid_res1 = ResnetBlock2D(ch, ch, tdim, groups, eps)
        self.mid_attn = Transformer2D(ch, heads, context_dim, groups)
        self.mid_res2 = ResnetBlock2D(ch, ch, tdim, groups, eps)
        self.up = nn.ModuleList()
        for i, (out, attn) in enumerate(zip(reversed(block_channels), reversed(attention_levels))):
            level = nn.Module()
            resnets = []
            for _ in range(layers_per_block + 1):
                resnets.append(ResnetBlock2D(ch + skips.pop(), out, tdim, groups, eps))
                ch = out
            level.resnets = nn.ModuleList(resnets)
            level.attentions = nn.ModuleList(
                [Transformer2D(out, heads, context_dim, groups) for _ in range(layers_per_block + 1)]
            ) if attn else None
            level.upsample = nn.Conv2d(out, out, 3, padding=1) if i < n - 1 else None
            self.up.append(level)
        self.conv_norm_out = nn.GroupNorm(groups, c0, eps=eps)
        self.conv_out = nn.Conv2d(c0, out_channels, 3, padding=1)

    def forward(self, sample, timestep, context):
        temb = self.time_embedding(timestep)
        h = self.conv_in(sample)
        skips = [h]
        for level in self.down:
            for j, res in enumerate(level.resnets):
                h = res(h, temb)
                if level.attentions is not None:
                    h = level.attentions[j](h, context)
                skips.append(h)
            if level.downsample is not None:
                h = level.downsample(h)
                skips.append(h)
        h = self.mid_res2(self.mid_attn(self.mid_res1(h, temb), context), temb)
        for level in self.up:
            for j, res in enumerate(level.resnets):
                h = res(torch.cat([h, skips.pop()], dim=1), temb)
                if level.attentions is not None:
                    h = level.attentions[j](h, context)
            if level.upsample is not None:
                h = level.upsample(F.interpolate(h, scale_factor=2.0, mode="nearest"))
        return self.conv_out(F.silu(self.conv_norm_out(h)))


class VAEAttention(nn.Module):
    """The VAE's single-head self-attention over all pixels, with a GroupNorm and a residual."""

    def __init__(self, ch, groups=32, eps=1e-6):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, ch, eps=eps)
        self.to_q = nn.Linear(ch, ch)
        self.to_k = nn.Linear(ch, ch)
        self.to_v = nn.Linear(ch, ch)
        self.to_out = nn.Linear(ch, ch)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.group_norm(x).view(b, c, h * w).transpose(1, 2)
        out = F.scaled_dot_product_attention(self.to_q(y)[:, None], self.to_k(y)[:, None], self.to_v(y)[:, None])
        return x + self.to_out(out[:, 0]).transpose(1, 2).reshape(b, c, h, w)


class VAEEncoder(nn.Module):
    """Stable Diffusion's VAE encoder: a picture -> its latent, 8 times smaller per
    side with 4 channels. ResNet blocks that halve the size, an attention in the
    middle, then the latent's mean, multiplied by scaling_factor."""

    def __init__(self, in_channels=3, latent_channels=4, block_channels=(128, 256, 512, 512), layers_per_block=2,
                 groups=32, eps=1e-6, scaling_factor=0.18215):
        super().__init__()
        self.config = dict(in_channels=in_channels, latent_channels=latent_channels,
                           block_channels=list(block_channels), layers_per_block=layers_per_block, groups=groups,
                           eps=eps, scaling_factor=scaling_factor)
        self.scaling_factor = float(scaling_factor)
        self.conv_in = nn.Conv2d(in_channels, block_channels[0], 3, padding=1)
        self.down = nn.ModuleList()
        ch = block_channels[0]
        for i, out in enumerate(block_channels):
            level = nn.Module()
            level.resnets = nn.ModuleList(
                [ResnetBlock2D(ch if j == 0 else out, out, 0, groups, eps) for j in range(layers_per_block)])
            ch = out
            level.downsample = nn.Conv2d(out, out, 3, stride=2, padding=0) if i < len(block_channels) - 1 else None
            self.down.append(level)
        self.mid_res1 = ResnetBlock2D(ch, ch, 0, groups, eps)
        self.mid_attn = VAEAttention(ch, groups, eps)
        self.mid_res2 = ResnetBlock2D(ch, ch, 0, groups, eps)
        self.conv_norm_out = nn.GroupNorm(groups, ch, eps=eps)
        self.conv_out = nn.Conv2d(ch, 2 * latent_channels, 3, padding=1)
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)

    def forward(self, x):
        h = self.conv_in(x)
        for level in self.down:
            for res in level.resnets:
                h = res(h)
            if level.downsample is not None:
                h = level.downsample(F.pad(h, (0, 1, 0, 1)))
        h = self.mid_res2(self.mid_attn(self.mid_res1(h)))
        moments = self.quant_conv(self.conv_out(F.silu(self.conv_norm_out(h))))
        mean, _logvar = moments.chunk(2, dim=1)
        return mean * self.scaling_factor


class VAEDecoder(nn.Module):
    """Stable Diffusion's VAE decoder: a latent -> the picture, 8 times larger per
    side. An attention in the middle, then ResNet blocks that double the size."""

    def __init__(self, out_channels=3, latent_channels=4, block_channels=(128, 256, 512, 512), layers_per_block=2,
                 groups=32, eps=1e-6, scaling_factor=0.18215):
        super().__init__()
        self.config = dict(out_channels=out_channels, latent_channels=latent_channels,
                           block_channels=list(block_channels), layers_per_block=layers_per_block, groups=groups,
                           eps=eps, scaling_factor=scaling_factor)
        self.scaling_factor = float(scaling_factor)
        rev = list(reversed(block_channels))
        ch = rev[0]
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
        self.conv_in = nn.Conv2d(latent_channels, ch, 3, padding=1)
        self.mid_res1 = ResnetBlock2D(ch, ch, 0, groups, eps)
        self.mid_attn = VAEAttention(ch, groups, eps)
        self.mid_res2 = ResnetBlock2D(ch, ch, 0, groups, eps)
        self.up = nn.ModuleList()
        for i, out in enumerate(rev):
            level = nn.Module()
            level.resnets = nn.ModuleList(
                [ResnetBlock2D(ch if j == 0 else out, out, 0, groups, eps) for j in range(layers_per_block + 1)])
            ch = out
            level.upsample = nn.Conv2d(out, out, 3, padding=1) if i < len(rev) - 1 else None
            self.up.append(level)
        self.conv_norm_out = nn.GroupNorm(groups, ch, eps=eps)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, z):
        h = self.conv_in(self.post_quant_conv(z / self.scaling_factor))
        h = self.mid_res2(self.mid_attn(self.mid_res1(h)))
        for level in self.up:
            for res in level.resnets:
                h = res(h)
            if level.upsample is not None:
                h = level.upsample(F.interpolate(h, scale_factor=2.0, mode="nearest"))
        return self.conv_out(F.silu(self.conv_norm_out(h)))


def alphas_cumprod(steps, beta_start, beta_end, schedule):
    """How much of the clean signal survives at each step of the noise schedule."""
    if schedule == "scaled_linear":  # Stable Diffusion's
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, steps, dtype=torch.float64) ** 2
    else:
        betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float64)
    return torch.cumprod(1.0 - betas, dim=0).float()


class AddNoise(nn.Module):
    """Mixes a clean latent with noise for a timestep, as a diffusion model is
    trained: sqrt(a_t) * latent + sqrt(1 - a_t) * noise, a_t from the noise schedule.
    No parameters."""

    def __init__(self, steps=1000, beta_start=0.00085, beta_end=0.012, schedule="scaled_linear"):
        super().__init__()
        self.config = dict(steps=steps, beta_start=beta_start, beta_end=beta_end, schedule=schedule)
        self.steps = int(steps)
        self.register_buffer("alphas_cumprod", alphas_cumprod(int(steps), beta_start, beta_end, schedule),
                             persistent=False)

    def _a(self, t, like):
        idx = t.reshape(-1).round().long().clamp(0, self.steps - 1)
        a = self.alphas_cumprod.to(like.device)[idx]
        return a.view(-1, *([1] * (like.dim() - 1))).to(like.dtype)

    def forward(self, latent, noise, timestep):
        a = self._a(timestep, latent)
        return a.sqrt() * latent + (1 - a).sqrt() * noise


class PredictClean(AddNoise):
    """The guess of the clean latent from a noisy one and the predicted noise:
    (x_t - sqrt(1 - a_t) * noise) / sqrt(a_t) — one denoising step, all at once.
    No parameters."""

    def forward(self, latent, noise, timestep):
        a = self._a(timestep, latent)
        return (latent - (1 - a).sqrt() * noise) / a.sqrt()


class Similarity(nn.Module):
    """CLIP's score for an image and a text: the cosine of their embeddings times a
    learned temperature, exp(logit_scale)."""

    def __init__(self, logit_scale=2.6592):
        super().__init__()
        self.config = dict(logit_scale=logit_scale)
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale)))

    def forward(self, a, b):
        a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
        return (a * b).sum(dim=-1, keepdim=True) * self.logit_scale.exp()


class TriplaneTokens(nn.Module):
    """TripoSR's learned triplane: three planes (xy, xz, yz) of size x size cells,
    one token of `channels` numbers per cell -> (3 * size * size, channels). The
    input only gives the batch size — the image is read later, by cross-attention."""

    def __init__(self, planes=3, size=32, channels=1024):
        super().__init__()
        self.config = dict(planes=planes, size=size, channels=channels)
        self.embeddings = nn.Parameter(torch.randn(planes, channels, size, size) / math.sqrt(channels))

    def forward(self, x):
        p, c = self.embeddings.shape[0], self.embeddings.shape[1]
        tokens = self.embeddings.flatten(2).permute(0, 2, 1).reshape(-1, c)
        return tokens.unsqueeze(0).expand(x.shape[0], -1, -1)


class TriplaneTransformer(nn.Module):
    """TripoSR's backbone: the triplane tokens attend to each other and, through
    cross-attention, to the image tokens — GroupNorm, a projection, N transformer
    blocks (self-attention, cross-attention, GEGLU feed-forward), a projection back
    and a residual."""

    def __init__(self, dim, context_dim, layers=16, heads=16, head_dim=64, groups=32):
        super().__init__()
        self.config = dict(dim=dim, context_dim=context_dim, layers=layers, heads=heads, head_dim=head_dim,
                           groups=groups)
        inner = heads * head_dim
        self.norm = nn.GroupNorm(groups, dim, eps=1e-6)
        self.proj_in = nn.Linear(dim, inner)
        self.blocks = nn.ModuleList([BasicTransformerBlock(inner, heads, head_dim, context_dim) for _ in range(layers)])
        self.proj_out = nn.Linear(inner, dim)

    def forward(self, x, context):
        h = self.proj_in(self.norm(x.transpose(1, 2)).transpose(1, 2))
        for blk in self.blocks:
            h = blk(h, context)
        return x + self.proj_out(h)


class TriplaneUpsample(nn.Module):
    """Triplane tokens back into three feature planes, twice as fine:
    (3 * s * s, C) -> (3, out_channels, 2s, 2s), with a transposed convolution
    (TripoSR's post-processor)."""

    def __init__(self, in_channels=1024, out_channels=40, planes=3):
        super().__init__()
        self.config = dict(in_channels=in_channels, out_channels=out_channels, planes=planes)
        self.planes = int(planes)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)

    def forward(self, x):
        b, n, c = x.shape
        s = int(round((n // self.planes) ** 0.5))
        h = x.reshape(b, self.planes, s, s, c).permute(0, 1, 4, 2, 3).reshape(b * self.planes, c, s, s)
        h = self.upsample(h)
        return h.reshape(b, self.planes, *h.shape[1:])


class TriplaneSample(nn.Module):
    """Reads the triplane at 3-D points: each point (x, y, z) is projected onto the
    xy, xz and yz planes, the features there are sampled (bilinear) and the three
    are joined -> (N, 3 * C), what TripoSR's NeRF MLP turns into density and colour."""

    def __init__(self, radius=0.87):
        super().__init__()
        self.config = dict(radius=radius)
        self.radius = float(radius)

    def forward(self, planes, points):
        b, n_planes, c, h, w = planes.shape
        p = points / self.radius
        coords = torch.stack([p[..., [0, 1]], p[..., [0, 2]], p[..., [1, 2]]], dim=1)
        feats = F.grid_sample(planes.reshape(b * n_planes, c, h, w), coords.reshape(b * n_planes, 1, -1, 2),
                              mode="bilinear", padding_mode="zeros", align_corners=False)
        return feats.reshape(b, n_planes, c, -1).permute(0, 3, 1, 2).reshape(b, -1, n_planes * c)


class Transpose(nn.Module):
    """Swaps the two axes of each example, (A, B) -> (B, A): audio features
    (channels, time) become a sequence (time, channels)."""

    def __init__(self):
        super().__init__()
        self.config = {}

    def forward(self, x):
        return x.transpose(1, 2)


class SliceTokens(nn.Module):
    """Keeps part of a sequence: tokens start … start + length (length 0 = to the
    end). FLUX keeps the image tokens after its joint blocks."""

    def __init__(self, start=0, length=0):
        super().__init__()
        self.config = dict(start=start, length=length)
        self.start, self.length = int(start), int(length)

    def forward(self, x):
        return x[:, self.start:self.start + self.length] if self.length else x[:, self.start:]


class Chain(nn.Module):
    """Layers in a row, where some also read extra inputs (a condition, the audio):
    ports[i] lists the extra inputs layer i takes."""

    def __init__(self, layers, ports):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.port_maps = [tuple(p) for p in ports]

    def forward(self, x, *extra):
        for layer, idx in zip(self.layers, self.port_maps):
            x = layer(x, *[extra[i] for i in idx]) if idx else layer(x)
        return x
