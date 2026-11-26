"""The building blocks of today's large language models (Stage 17).

Plain torch modules for the layers inside Llama 3, Qwen3, Mistral / Mixtral,
Gemma 3, Phi-4 and DeepSeek-V3: rotary position embedding (RoPE), grouped-query
attention (GQA), DeepSeek's multi-head latent attention (MLA), the gated
feed-forward (SwiGLU / GeGLU), a mixture of experts (MoE) and the vocabulary head.

Every block carries its own pre-norm and residual connection — ``x + f(norm(x))``
— so a decoder layer is just a chain of two nodes (attention, then feed-forward)
and a whole model is one ``Repeat`` of that chain. Each module also keeps the
arguments it was made with in ``self.config``, so the exporter can write the same
constructor call back out.

This file is self-contained on purpose (torch only): the PyTorch export copies
everything below the "export" marker verbatim into ``model.py``, so the exported
network is exactly the one built here.
"""

import math  # noqa: F401  (kept for the exported copy)

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- export: begin ---


def rope_cos_sin(length, dim, theta, device, dtype):
    """Rotary position embedding tables (L, dim): the angle of position p in pair i
    is p / theta^(2i / dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    freqs = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    """Rotate (B, H, L, D) queries or keys by their position (the "rotate half" layout)."""
    return x * cos + rotate_half(x) * sin


def attention_mask(length, causal, window, device):
    """Which positions may attend to which (L, L, True = allowed), or None for all.
    Causal: only to earlier tokens. Window: only to the last `window` tokens."""
    if not causal and not window:
        return None
    i = torch.arange(length, device=device)[:, None]
    j = torch.arange(length, device=device)[None, :]
    allowed = torch.ones(length, length, dtype=torch.bool, device=device)
    if causal:
        allowed = allowed & (j <= i)
    if window:
        allowed = allowed & ((i - j).abs() < window)
    return allowed


class GroupedQueryAttention(nn.Module):
    """Self-attention with rotary positions and grouped key/value heads (GQA): several
    query heads share one key/value head. The attention of Llama 3, Qwen3, Mistral,
    Gemma 3 and Phi-4. Returns x + attention(RMSNorm(x)).

    Options: qk_norm normalizes queries and keys per head (Qwen3, Gemma 3);
    sliding_window limits how far back a token looks (Mistral v0.1, Gemma 3's local
    layers); post_norm adds Gemma's second norm after the attention; bias puts biases
    on the q/k/v projections (Qwen2)."""

    def __init__(self, dim, heads, kv_heads, head_dim, rope_theta=10000.0, causal=True, sliding_window=0,
                 qk_norm=False, bias=False, post_norm=False, eps=1e-6):
        super().__init__()
        self.config = dict(dim=dim, heads=heads, kv_heads=kv_heads, head_dim=head_dim, rope_theta=rope_theta,
                           causal=causal, sliding_window=sliding_window, qk_norm=qk_norm, bias=bias,
                           post_norm=post_norm, eps=eps)
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.rope_theta, self.causal, self.window = float(rope_theta), bool(causal), int(sliding_window)
        self.norm = nn.RMSNorm(dim, eps=eps)
        self.q_proj = nn.Linear(dim, heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(heads * head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps) if qk_norm else None
        self.k_norm = nn.RMSNorm(head_dim, eps=eps) if qk_norm else None
        self.post_norm = nn.RMSNorm(dim, eps=eps) if post_norm else None

    def forward(self, x):
        b, n, _ = x.shape
        h = self.norm(x)
        q = self.q_proj(h).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, n, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, n, self.kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = rope_cos_sin(n, self.head_dim, self.rope_theta, x.device, q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.kv_heads != self.heads:  # each key/value head serves a group of query heads
            group = self.heads // self.kv_heads
            k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask(n, self.causal, self.window, x.device))
        out = self.o_proj(out.transpose(1, 2).reshape(b, n, self.heads * self.head_dim))
        if self.post_norm is not None:
            out = self.post_norm(out)
        return x + out


class LatentAttention(nn.Module):
    """DeepSeek's multi-head latent attention (MLA): keys and values are squeezed
    through a small shared latent vector (kv_lora_rank), queries through another
    (q_lora_rank), and a separate slice of each head carries the rotary position.
    That small latent is what DeepSeek-V3 caches instead of full keys and values.
    Returns x + attention(RMSNorm(x))."""

    def __init__(self, dim, heads, q_lora_rank, kv_lora_rank, qk_nope_dim, qk_rope_dim, v_dim,
                 rope_theta=10000.0, causal=True, eps=1e-6):
        super().__init__()
        self.config = dict(dim=dim, heads=heads, q_lora_rank=q_lora_rank, kv_lora_rank=kv_lora_rank,
                           qk_nope_dim=qk_nope_dim, qk_rope_dim=qk_rope_dim, v_dim=v_dim, rope_theta=rope_theta,
                           causal=causal, eps=eps)
        self.heads, self.kv_lora_rank = heads, kv_lora_rank
        self.nope, self.rope, self.v_dim = qk_nope_dim, qk_rope_dim, v_dim
        self.rope_theta, self.causal = float(rope_theta), bool(causal)
        qk_dim = qk_nope_dim + qk_rope_dim
        self.norm = nn.RMSNorm(dim, eps=eps)
        if q_lora_rank:
            self.q_a_proj = nn.Linear(dim, q_lora_rank, bias=False)
            self.q_a_norm = nn.RMSNorm(q_lora_rank, eps=eps)
            self.q_b_proj = nn.Linear(q_lora_rank, heads * qk_dim, bias=False)
        else:
            self.q_proj = nn.Linear(dim, heads * qk_dim, bias=False)
        self.kv_a_proj = nn.Linear(dim, kv_lora_rank + qk_rope_dim, bias=False)
        self.kv_a_norm = nn.RMSNorm(kv_lora_rank, eps=eps)
        self.kv_b_proj = nn.Linear(kv_lora_rank, heads * (qk_nope_dim + v_dim), bias=False)
        self.o_proj = nn.Linear(heads * v_dim, dim, bias=False)

    def forward(self, x):
        b, n, _ = x.shape
        h = self.norm(x)
        q = self.q_b_proj(self.q_a_norm(self.q_a_proj(h))) if hasattr(self, "q_a_proj") else self.q_proj(h)
        q = q.view(b, n, self.heads, self.nope + self.rope).transpose(1, 2)
        q_nope, q_pe = q.split([self.nope, self.rope], dim=-1)
        c_kv, k_pe = self.kv_a_proj(h).split([self.kv_lora_rank, self.rope], dim=-1)
        k_pe = k_pe.view(b, n, 1, self.rope).transpose(1, 2)  # one rotary key, shared by every head
        kv = self.kv_b_proj(self.kv_a_norm(c_kv)).view(b, n, self.heads, self.nope + self.v_dim).transpose(1, 2)
        k_nope, v = kv.split([self.nope, self.v_dim], dim=-1)
        cos, sin = rope_cos_sin(n, self.rope, self.rope_theta, x.device, q.dtype)
        q_pe, k_pe = apply_rope(q_pe, cos, sin), apply_rope(k_pe, cos, sin)
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, self.heads, -1, -1)], dim=-1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask(n, self.causal, 0, x.device))
        return x + self.o_proj(out.transpose(1, 2).reshape(b, n, self.heads * self.v_dim))


def _activate(x, activation):
    return F.silu(x) if activation == "silu" else F.gelu(x, approximate="tanh")


class Expert(nn.Module):
    """One gated MLP: down(act(gate(x)) * up(x)) — SwiGLU with SiLU, GeGLU with GELU."""

    def __init__(self, dim, hidden, activation="silu"):
        super().__init__()
        self.activation = activation
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down_proj(_activate(self.gate_proj(x), self.activation) * self.up_proj(x))


class GatedMLP(nn.Module):
    """The gated feed-forward of modern LLMs (SwiGLU in Llama, Qwen, Mistral, DeepSeek,
    Phi; GeGLU in Gemma). Returns x + MLP(RMSNorm(x)); post_norm adds Gemma's second
    norm after it."""

    def __init__(self, dim, hidden, activation="silu", post_norm=False, eps=1e-6):
        super().__init__()
        self.config = dict(dim=dim, hidden=hidden, activation=activation, post_norm=post_norm, eps=eps)
        self.norm = nn.RMSNorm(dim, eps=eps)
        self.mlp = Expert(dim, hidden, activation)
        self.post_norm = nn.RMSNorm(dim, eps=eps) if post_norm else None

    def forward(self, x):
        out = self.mlp(self.norm(x))
        if self.post_norm is not None:
            out = self.post_norm(out)
        return x + out


class MixtureOfExperts(nn.Module):
    """A sparse feed-forward: a router sends every token to its top_k experts (small
    gated MLPs) and mixes their answers; shared experts see every token. Mixtral and
    Qwen3-MoE route with a softmax; DeepSeek-V3 scores experts with a sigmoid plus a
    learned balancing bias (its group-limited routing is left out here). Returns
    x + MoE(RMSNorm(x))."""

    def __init__(self, dim, experts, top_k, expert_hidden, shared_experts=0, router="softmax", norm_topk=True,
                 scaling=1.0, activation="silu", eps=1e-6):
        super().__init__()
        self.config = dict(dim=dim, experts=experts, top_k=top_k, expert_hidden=expert_hidden,
                           shared_experts=shared_experts, router=router, norm_topk=norm_topk, scaling=scaling,
                           activation=activation, eps=eps)
        self.top_k, self.router_kind, self.norm_topk, self.scaling = top_k, router, bool(norm_topk), float(scaling)
        self.norm = nn.RMSNorm(dim, eps=eps)
        self.router = nn.Linear(dim, experts, bias=False)
        self.router_bias = nn.Parameter(torch.zeros(experts)) if router == "sigmoid" else None
        self.experts = nn.ModuleList([Expert(dim, expert_hidden, activation) for _ in range(experts)])
        self.shared = Expert(dim, expert_hidden * shared_experts, activation) if shared_experts else None

    def forward(self, x):
        b, n, d = x.shape
        h = self.norm(x).reshape(-1, d)
        if h.device.type == "meta":  # shapes only (e.g. sizing a real-scale model): no routing to do
            out = self.experts[0](h)
        else:
            logits = self.router(h)
            if self.router_kind == "sigmoid":
                scores = torch.sigmoid(logits)
                chosen = torch.topk(scores + self.router_bias, self.top_k, dim=-1).indices
                weights = scores.gather(-1, chosen)
            else:
                weights, chosen = torch.topk(logits.softmax(dim=-1), self.top_k, dim=-1)
            if self.norm_topk:
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            weights = weights * self.scaling
            out = torch.zeros_like(h)
            for e, expert in enumerate(self.experts):
                hit = chosen == e
                rows = hit.any(dim=-1).nonzero().squeeze(-1)
                if rows.numel():
                    w = (weights * hit).sum(dim=-1)[rows]
                    out = out.index_add(0, rows, expert(h[rows]) * w[:, None])
        if self.shared is not None:
            out = out + self.shared(h)
        return x + out.view(b, n, d)


class LMHead(nn.Module):
    """Scores every token of the vocabulary — a Linear without bias. Tied, it reuses
    the embedding table as its weight (Gemma, small Qwen3 and Llama 3.2): the same
    shape, no extra parameters."""

    def __init__(self, dim, vocab, tie=False):
        super().__init__()
        self.config = dict(dim=dim, vocab=vocab, tie=tie)
        self.weight = nn.Parameter(torch.empty(vocab, dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        return F.linear(x, self.weight)
