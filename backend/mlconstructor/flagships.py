"""Flagship language models (Stage 17): Qwen3, DeepSeek-V3, Llama 3, Mistral,
Mixtral, Gemma 3 and Phi-4 as plain, editable graphs.

Every family is one decoder-only graph built from its configuration — token
embedding -> N x (attention, feed-forward) -> RMSNorm -> vocabulary head — in two
scales:

* **mini** — the same structure (GQA or MLA, SwiGLU or MoE, query/key norms,
  sliding windows, tied heads, …) at a size that trains here in seconds, on the
  "next token" task;
* **real** — the published ``config.json`` from Hugging Face, sized on PyTorch's
  "meta" device: every layer and shape of the real model, its exact parameter
  count, and no memory taken. It can be inspected, not trained.

The real configurations were read from the models' ``config.json`` on Hugging
Face (2026-11-26); ``official`` is the parameter total the Hub reports for the
weights (``safetensors.total``), or the model card's figure where the Hub's total
includes extra modules (DeepSeek-V3's multi-token-prediction layer).
"""

from __future__ import annotations

from typing import Any

Spec = dict[str, Any]
#: The mini models read 16 tokens from a 32-token vocabulary (the "next token" task).
MINI_SEQ, MINI_VOCAB = 16, 32


def _attn(cfg: dict, local: bool | None = None) -> dict:
    """One attention node: GQA (with its options) or DeepSeek's latent attention."""
    a = cfg["attn"]
    if a["kind"] == "mla":
        return {"type": "LatentAttention", "params": {
            "heads": a["heads"], "q_lora_rank": a["q_lora_rank"], "kv_lora_rank": a["kv_lora_rank"],
            "qk_nope_dim": a["qk_nope_dim"], "qk_rope_dim": a["qk_rope_dim"], "v_dim": a["v_dim"],
            "rope_theta": a["rope_theta"], "eps": cfg["eps"]}}
    params = {"heads": a["heads"], "kv_heads": a["kv_heads"], "head_dim": a["head_dim"],
              "rope_theta": a["rope_theta"], "qk_norm": a.get("qk_norm", False), "eps": cfg["eps"]}
    if a.get("sliding_window") and local is not False:
        params["sliding_window"] = a["sliding_window"]
    if local is True and a.get("local_rope_theta"):
        params["rope_theta"] = a["local_rope_theta"]
    if a.get("post_norm"):
        params["post_norm"] = True
    if a.get("bias"):
        params["bias"] = True
    return {"type": "GQAttention", "params": params}


def _ffn(cfg: dict, dense_hidden: int | None = None) -> dict:
    """One feed-forward node: a gated MLP, or a mixture of experts."""
    f = cfg["ffn"]
    if f["kind"] == "moe" and dense_hidden is None:
        return {"type": "MoE", "params": {
            "experts": f["experts"], "top_k": f["top_k"], "expert_hidden": f["expert_hidden"],
            "shared_experts": f.get("shared", 0), "router": f.get("router", "softmax"),
            "norm_topk": f.get("norm_topk", True), "scaling": f.get("scaling", 1.0), "eps": cfg["eps"]}}
    params = {"hidden": dense_hidden or f["hidden"], "activation": f.get("activation", "silu"), "eps": cfg["eps"]}
    if f.get("post_norm"):
        params["post_norm"] = True
    return {"type": "GatedMLP", "params": params}


def _blocks(cfg: dict) -> list[tuple[str, int, list[dict]]]:
    """The decoder layers as Repeat nodes: (id, repeat, inner chain)."""
    layers = cfg["layers"]
    pattern = cfg["attn"].get("pattern")  # Gemma 3: every pattern-th layer is global, the rest local
    dense_first = cfg["ffn"].get("dense_first", 0)  # DeepSeek-V3: the first layers keep a dense MLP
    if pattern:
        group = []
        for i in range(pattern):
            group += [_attn(cfg, local=i < pattern - 1), _ffn(cfg)]
        full, rest = divmod(layers, pattern)
        out = [("layers", full, group)] if full else []
        if rest:
            tail = []
            for _ in range(rest):
                tail += [_attn(cfg, local=True), _ffn(cfg)]
            out.append(("layers_tail", 1, tail))
        return out
    if dense_first:
        return [("dense_layers", dense_first, [_attn(cfg), _ffn(cfg, dense_hidden=cfg["ffn"]["dense_hidden"])]),
                ("moe_layers", layers - dense_first, [_attn(cfg), _ffn(cfg)])]
    return [("layers", layers, [_attn(cfg), _ffn(cfg)])]


def lm_spec(cfg: dict, train: dict | None = None) -> Spec:
    """A decoder-only language model as an editable graph. The head reads the last
    token (the connection takes the last step of the sequence): its scores are the
    model's guess for the next token."""
    nodes: list[dict] = [
        {"id": "in", "type": "Input", "params": {"shape": [cfg["seq"]]}},
        {"id": "emb", "type": "Embedding", "params": {"num_embeddings": cfg["vocab"], "embedding_dim": cfg["dim"]}},
    ]
    chain = ["in", "emb"]
    for bid, repeat, inner in _blocks(cfg):
        nodes.append({"id": bid, "type": "Repeat", "params": {"repeat": repeat, "layers": inner}})
        chain.append(bid)
    nodes += [
        {"id": "norm", "type": "RMSNorm", "params": {"eps": cfg["eps"]}},
        {"id": "head", "type": "LMHead", "params": {"vocab": cfg["vocab"], "tie_embedding": cfg.get("tie", False)}},
        {"id": "out", "type": "Output", "params": {}},
    ]
    chain += ["norm", "head", "out"]
    spec: Spec = {
        "nodes": nodes,
        "edges": [[a, b] for a, b in zip(chain, chain[1:])],
        # the head scores the next token from the last position of the sequence
        "edge_settings": [{"source": "norm", "target": "head", "transform": "seq_pool", "params": {"mode": "last"}}],
    }
    if train is not None:
        spec["train"] = train
    return spec


_MINI_TRAIN = {"dataset": "tokens", "epochs": 8, "lr": 0.003, "batch_size": 64, "samples": 6000, "seed": 0}


def _mini(attn: dict, ffn: dict, **extra) -> dict:
    return {"seq": MINI_SEQ, "vocab": MINI_VOCAB, "dim": 64, "layers": 2, "eps": 1e-6, "attn": attn, "ffn": ffn,
            **extra}


#: id, name, what it is, Hugging Face repo, official parameter count, mini config, real config.
FLAGSHIPS: list[dict] = [
    {
        "id": "qwen3", "name": "Qwen3", "real_name": "Qwen3-8B", "hf": "Qwen/Qwen3-8B", "official": 8_190_735_360,
        "description": "Alibaba's Qwen3: grouped-query attention with query/key norms, RoPE, SwiGLU, RMSNorm.",
        "mini": _mini({"kind": "gqa", "heads": 4, "kv_heads": 2, "head_dim": 16, "rope_theta": 1e6, "qk_norm": True},
                      {"kind": "dense", "hidden": 192}),
        "real": {"seq": 16, "vocab": 151936, "dim": 4096, "layers": 36, "eps": 1e-6,
                 "attn": {"kind": "gqa", "heads": 32, "kv_heads": 8, "head_dim": 128, "rope_theta": 1e6,
                          "qk_norm": True},
                 "ffn": {"kind": "dense", "hidden": 12288}},
    },
    {
        "id": "qwen3_moe", "name": "Qwen3-MoE", "real_name": "Qwen3-30B-A3B", "hf": "Qwen/Qwen3-30B-A3B",
        "official": 30_532_122_624,
        "description": "Qwen3 with a mixture of experts: every token goes to 8 of 128 experts (3B of 30B active).",
        "mini": _mini({"kind": "gqa", "heads": 4, "kv_heads": 2, "head_dim": 16, "rope_theta": 1e6, "qk_norm": True},
                      {"kind": "moe", "experts": 8, "top_k": 2, "expert_hidden": 32}),
        "real": {"seq": 16, "vocab": 151936, "dim": 2048, "layers": 48, "eps": 1e-6,
                 "attn": {"kind": "gqa", "heads": 32, "kv_heads": 4, "head_dim": 128, "rope_theta": 1e6,
                          "qk_norm": True},
                 "ffn": {"kind": "moe", "experts": 128, "top_k": 8, "expert_hidden": 768}},
    },
    {
        "id": "deepseek_v3", "name": "DeepSeek-V3 / R1", "real_name": "DeepSeek-V3", "hf": "deepseek-ai/DeepSeek-V3",
        "official": 671_000_000_000,
        "description": "Multi-head latent attention (MLA) and a mixture of 256 experts plus a shared one; "
                       "R1 is the same network, trained to reason.",
        "mini": _mini({"kind": "mla", "heads": 4, "q_lora_rank": 32, "kv_lora_rank": 16, "qk_nope_dim": 8,
                       "qk_rope_dim": 8, "v_dim": 8, "rope_theta": 10000.0},
                      {"kind": "moe", "experts": 8, "top_k": 2, "expert_hidden": 32, "shared": 1,
                       "router": "sigmoid", "scaling": 2.5, "dense_first": 1, "dense_hidden": 128}),
        "real": {"seq": 16, "vocab": 129280, "dim": 7168, "layers": 61, "eps": 1e-6,
                 "attn": {"kind": "mla", "heads": 128, "q_lora_rank": 1536, "kv_lora_rank": 512,
                          "qk_nope_dim": 128, "qk_rope_dim": 64, "v_dim": 128, "rope_theta": 10000.0},
                 "ffn": {"kind": "moe", "experts": 256, "top_k": 8, "expert_hidden": 2048, "shared": 1,
                         "router": "sigmoid", "scaling": 2.5, "dense_first": 3, "dense_hidden": 18432}},
    },
    {
        "id": "llama3", "name": "Llama 3", "real_name": "Llama-3.1-8B", "hf": "meta-llama/Llama-3.1-8B",
        "official": 8_030_261_248,
        "description": "Meta's Llama 3: grouped-query attention, RoPE (base 500,000), SwiGLU, RMSNorm.",
        "mini": _mini({"kind": "gqa", "heads": 4, "kv_heads": 2, "head_dim": 16, "rope_theta": 500000.0},
                      {"kind": "dense", "hidden": 192}),
        "real": {"seq": 16, "vocab": 128256, "dim": 4096, "layers": 32, "eps": 1e-5,
                 "attn": {"kind": "gqa", "heads": 32, "kv_heads": 8, "head_dim": 128, "rope_theta": 500000.0},
                 "ffn": {"kind": "dense", "hidden": 14336}},
    },
    {
        "id": "mistral", "name": "Mistral", "real_name": "Mistral-7B-v0.3", "hf": "mistralai/Mistral-7B-v0.3",
        "official": 7_248_023_552,
        "description": "Mistral 7B: grouped-query attention and SwiGLU; the mini one uses a sliding window, "
                       "as v0.1 did.",
        "mini": _mini({"kind": "gqa", "heads": 4, "kv_heads": 2, "head_dim": 16, "rope_theta": 1e6,
                       "sliding_window": 8},
                      {"kind": "dense", "hidden": 192}),
        "real": {"seq": 16, "vocab": 32768, "dim": 4096, "layers": 32, "eps": 1e-5,
                 "attn": {"kind": "gqa", "heads": 32, "kv_heads": 8, "head_dim": 128, "rope_theta": 1e6},
                 "ffn": {"kind": "dense", "hidden": 14336}},
    },
    {
        "id": "mixtral", "name": "Mixtral", "real_name": "Mixtral-8x7B", "hf": "mistralai/Mixtral-8x7B-v0.1",
        "official": 46_702_792_704,
        "description": "Mistral's mixture of experts: 8 expert MLPs per layer, every token uses 2.",
        "mini": _mini({"kind": "gqa", "heads": 4, "kv_heads": 2, "head_dim": 16, "rope_theta": 1e6},
                      {"kind": "moe", "experts": 8, "top_k": 2, "expert_hidden": 64}),
        "real": {"seq": 16, "vocab": 32000, "dim": 4096, "layers": 32, "eps": 1e-5,
                 "attn": {"kind": "gqa", "heads": 32, "kv_heads": 8, "head_dim": 128, "rope_theta": 1e6},
                 "ffn": {"kind": "moe", "experts": 8, "top_k": 2, "expert_hidden": 14336}},
    },
    {
        "id": "gemma3", "name": "Gemma 3", "real_name": "Gemma-3-1B", "hf": "google/gemma-3-1b-pt",
        "official": 999_885_952,
        "description": "Google's Gemma 3: 5 local (sliding-window) layers to 1 global, query/key norms, norms "
                       "before and after each block, GeGLU, a head tied to the embedding.",
        "mini": _mini({"kind": "gqa", "heads": 2, "kv_heads": 1, "head_dim": 32, "rope_theta": 1e6, "qk_norm": True,
                       "post_norm": True, "sliding_window": 8, "pattern": 3, "local_rope_theta": 10000.0},
                      {"kind": "dense", "hidden": 192, "activation": "gelu", "post_norm": True},
                      layers=3, tie=True),
        "real": {"seq": 16, "vocab": 262144, "dim": 1152, "layers": 26, "eps": 1e-6, "tie": True,
                 "attn": {"kind": "gqa", "heads": 4, "kv_heads": 1, "head_dim": 256, "rope_theta": 1e6,
                          "qk_norm": True, "post_norm": True, "sliding_window": 512, "pattern": 6,
                          "local_rope_theta": 10000.0},
                 "ffn": {"kind": "dense", "hidden": 6912, "activation": "gelu", "post_norm": True}},
    },
    {
        "id": "phi4", "name": "Phi-4", "real_name": "Phi-4", "hf": "microsoft/phi-4", "official": 14_659_507_200,
        "description": "Microsoft's Phi-4: a dense 14B model, 4 query heads per key/value head, SwiGLU.",
        "mini": _mini({"kind": "gqa", "heads": 4, "kv_heads": 1, "head_dim": 16, "rope_theta": 250000.0},
                      {"kind": "dense", "hidden": 224}),
        "real": {"seq": 16, "vocab": 100352, "dim": 5120, "layers": 40, "eps": 1e-5,
                 "attn": {"kind": "gqa", "heads": 40, "kv_heads": 10, "head_dim": 128, "rope_theta": 250000.0},
                 "ffn": {"kind": "dense", "hidden": 17920}},
    },
]


def mini_spec(fam: dict) -> Spec:
    return lm_spec(fam["mini"], dict(_MINI_TRAIN))


def real_spec(fam: dict) -> Spec:
    return lm_spec(fam["real"], dict(_MINI_TRAIN))


def hf_url(fam: dict) -> str:
    return f"https://huggingface.co/{fam['hf']}"
