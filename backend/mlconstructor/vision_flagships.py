"""Flagship image, 3D, speech and vision models (Stage 19): ViT, CLIP, Stable
Diffusion, DiT, FLUX, TripoSR and Whisper as plain, editable graphs.

The same two scales as the language models of Stage 17:

* **mini** — the same structure at a size that builds and runs here in a moment
  (the ViT mini also trains, on MNIST);
* **real** — the published configuration, sized on PyTorch's "meta" device: every
  layer and shape of the real model and its exact parameter count, no memory
  taken. It can be inspected, not run.

Most of these models take several inputs — Stable Diffusion a picture, noise, a
timestep and a text; CLIP an image and a text; Whisper the audio and the text so
far. Each is one graph with several Input nodes, wired in the order the layers
expect them.

The real configurations come from the models' published configs on Hugging
Face (read 2026-11-26). ``official`` is the parameter total the Hub reports for
the weights where it has one, otherwise the figure from the paper or the files
— ``source`` says which.
"""

from __future__ import annotations

from typing import Any, Callable

Spec = dict[str, Any]


def _node(nid: str, typ: str, **params) -> dict:
    return {"id": nid, "type": typ, "params": params}


def _input(nid: str, shape: list[int], name: str) -> dict:
    return _node(nid, "Input", shape=list(shape), name=name)


def _pool(source: str, target: str, mode: str) -> dict:
    """A connection that keeps one token of a sequence (the class token, the last one)."""
    return {"source": source, "target": target, "transform": "seq_pool", "params": {"mode": mode}}


def _graph(nodes: list[dict], edges: list[list[str]], settings: list[dict] | None = None,
           train: dict | None = None) -> Spec:
    spec: Spec = {"nodes": nodes, "edges": edges, "edge_settings": settings or []}
    spec["train"] = dict(train) if train else {"seed": 0}
    return spec


def _encoder_layers(nid: str, c: dict, causal: bool = False, cross: bool = False) -> dict:
    """N transformer layers as one Repeat: attention (+ cross-attention) + MLP."""
    inner = [{"type": "AttentionBlock", "params": {"heads": c["heads"], "causal": causal,
                                                   "k_bias": c.get("k_bias", True), "eps": c["eps"]}}]
    if cross:
        inner.append({"type": "CrossAttentionBlock", "params": {"heads": c["heads"], "k_bias": c.get("k_bias", True),
                                                                "eps": c["eps"]}})
    inner.append({"type": "MLPBlock", "params": {"hidden": c["mlp"], "activation": c.get("act", "gelu"),
                                                 "eps": c["eps"]}})
    return _node(nid, "Repeat", repeat=c["layers"], layers=inner)


# --------------------------------------------------------------------------- #
# ViT — an image classifier made of a transformer
# --------------------------------------------------------------------------- #
def vit_spec(c: dict, train: dict | None = None) -> Spec:
    nodes = [
        _input("image", c["image"], "image"),
        _node("patches", "PatchEmbed", patch_size=c["patch"], dim=c["dim"], cls_token=True, position="learned"),
        _encoder_layers("layers", c),
        _node("norm", "LayerNorm", last_axis=True, eps=c["eps"]),
        _node("head", "Linear", out_features=c["classes"], bias=True),
        _node("out", "Output"),
    ]
    chain = ["image", "patches", "layers", "norm", "head", "out"]
    # the class token (first) carries the answer
    return _graph(nodes, [[a, b] for a, b in zip(chain, chain[1:])], [_pool("norm", "head", "first")], train)


# --------------------------------------------------------------------------- #
# CLIP — an image tower and a text tower meeting in one embedding space
# --------------------------------------------------------------------------- #
def clip_spec(c: dict, train: dict | None = None) -> Spec:
    v, t = c["vision"], c["text"]
    nodes = [
        _input("image", c["image"], "image"),
        _node("patches", "PatchEmbed", patch_size=v["patch"], dim=v["dim"], cls_token=True, position="learned",
              bias=False),
        _node("pre_norm", "LayerNorm", last_axis=True, eps=v["eps"]),
        _encoder_layers("vision_layers", v),
        _node("post_norm", "LayerNorm", last_axis=True, eps=v["eps"]),
        _node("image_proj", "Linear", out_features=c["proj"], bias=False),
        _input("text", [t["length"]], "text"),
        _node("tokens", "Embedding", num_embeddings=t["vocab"], embedding_dim=t["dim"]),
        _node("positions", "PositionEmbedding", length=t["length"], kind="learned"),
        _encoder_layers("text_layers", t, causal=True),
        _node("text_norm", "LayerNorm", last_axis=True, eps=t["eps"]),
        _node("text_proj", "Linear", out_features=c["proj"], bias=False),
        _node("score", "Similarity", logit_scale=2.6592),
        _node("out", "Output"),
    ]
    edges = [
        ["image", "patches"], ["patches", "pre_norm"], ["pre_norm", "vision_layers"], ["vision_layers", "post_norm"],
        ["post_norm", "image_proj"],
        ["text", "tokens"], ["tokens", "positions"], ["positions", "text_layers"], ["text_layers", "text_norm"],
        ["text_norm", "text_proj"],
        ["image_proj", "score"], ["text_proj", "score"], ["score", "out"],
    ]
    # the image is summed up by its class token, the text by its last token
    settings = [_pool("vision_layers", "post_norm", "first"), _pool("text_norm", "text_proj", "last")]
    return _graph(nodes, edges, settings, train)


# --------------------------------------------------------------------------- #
# Stable Diffusion — text encoder + U-Net + VAE, one denoising step end to end
# --------------------------------------------------------------------------- #
def sd_spec(c: dict, train: dict | None = None) -> Spec:
    vae, t, u = c["vae"], c["text"], c["unet"]
    lat = c["latent"]
    nodes = [
        _input("image", c["image"], "image"),
        _node("vae_encoder", "VAEEncoder", latent_channels=lat[0], block_channels=vae["channels"],
              layers_per_block=vae["layers"], groups=vae["groups"], scaling=0.18215),
        _input("noise", lat, "noise"),
        _input("timestep", [1], "timestep"),
        _node("add_noise", "AddNoise"),
        _input("text", [t["length"]], "text"),
        _node("tokens", "Embedding", num_embeddings=t["vocab"], embedding_dim=t["dim"]),
        _node("positions", "PositionEmbedding", length=t["length"], kind="learned"),
        _encoder_layers("text_layers", t, causal=True),
        _node("text_norm", "LayerNorm", last_axis=True, eps=t["eps"]),
        _node("unet", "UNet2D", out_channels=lat[0], block_channels=u["channels"], layers_per_block=u["layers"],
              attention_levels=u["attention"], heads=u["heads"], groups=u["groups"]),
        _node("predict_clean", "PredictClean"),
        _node("vae_decoder", "VAEDecoder", out_channels=c["image"][0], block_channels=vae["channels"],
              layers_per_block=vae["layers"], groups=vae["groups"], scaling=0.18215),
        _node("out", "Output", any_shape=True),
    ]
    edges = [
        ["image", "vae_encoder"],
        # add noise: the clean latent, the noise, the timestep — in this order
        ["vae_encoder", "add_noise"], ["noise", "add_noise"], ["timestep", "add_noise"],
        ["text", "tokens"], ["tokens", "positions"], ["positions", "text_layers"], ["text_layers", "text_norm"],
        # the U-Net: the noisy latent, the timestep, the text
        ["add_noise", "unet"], ["timestep", "unet"], ["text_norm", "unet"],
        # the clean latent it implies: the noisy latent, the predicted noise, the timestep
        ["add_noise", "predict_clean"], ["unet", "predict_clean"], ["timestep", "predict_clean"],
        ["predict_clean", "vae_decoder"], ["vae_decoder", "out"],
    ]
    return _graph(nodes, edges, [], train)


# --------------------------------------------------------------------------- #
# DiT — a diffusion transformer over latent patches, steered by timestep + class
# --------------------------------------------------------------------------- #
def dit_spec(c: dict, train: dict | None = None) -> Spec:
    lat, p = c["latent"], c["patch"]
    nodes = [
        _input("latent", lat, "noisy latent"),
        _node("patches", "PatchEmbed", patch_size=p, dim=c["dim"], cls_token=False, position="sincos"),
        _input("timestep", [1], "timestep"),
        _node("t_embed", "TimestepEmbedding", dim=c["dim"], freq_dim=256),
        _input("label", [1], "class"),
        _node("y_embed", "Embedding", num_embeddings=c["classes"] + 1, embedding_dim=c["dim"]),
        _node("condition", "Add"),
        _node("blocks", "Repeat", repeat=c["layers"], layers=[
            {"type": "DiTBlock", "params": {"heads": c["heads"], "mlp_ratio": 4.0}}]),
        _node("final_norm", "AdaLayerNorm"),
        _node("final", "Linear", out_features=p * p * c["out_channels"], per_token=True),
        _node("unpatchify", "Unpatchify", patch_size=p, channels=c["out_channels"]),
        _node("out", "Output", any_shape=True),
    ]
    edges = [
        ["latent", "patches"], ["timestep", "t_embed"], ["label", "y_embed"],
        ["t_embed", "condition"], ["y_embed", "condition"],
        ["patches", "blocks"], ["condition", "blocks"],
        ["blocks", "final_norm"], ["condition", "final_norm"],
        ["final_norm", "final"], ["final", "unpatchify"], ["unpatchify", "out"],
    ]
    return _graph(nodes, edges, [], train)


# --------------------------------------------------------------------------- #
# FLUX — a rectified-flow transformer over [text; image] tokens
# --------------------------------------------------------------------------- #
def flux_spec(c: dict, train: dict | None = None) -> Spec:
    d = c["dim"]
    block = {"heads": c["heads"], "head_dim": c["head_dim"], "txt_len": c["txt_len"], "mlp_ratio": 4.0,
             "axes": c["axes"], "theta": 10000.0}
    nodes = [
        _input("image", [c["img_tokens"], c["in_channels"]], "image tokens"),
        _node("img_in", "Linear", out_features=d, per_token=True),
        _input("text", [c["txt_len"], c["text_dim"]], "text (T5)"),
        _node("txt_in", "Linear", out_features=d, per_token=True),
        _input("pooled", [c["pooled_dim"]], "pooled text (CLIP)"),
        _node("vec_in", "Linear", out_features=d),
        _node("vec_act", "SiLU"),
        _node("vec_out", "Linear", out_features=d),
        _input("timestep", [1], "timestep"),
        _node("time_in", "TimestepEmbedding", dim=d, freq_dim=256, time_factor=1000.0),
        _node("condition", "Add"),
        _node("joint", "Concat", dim=0),
        _node("double_blocks", "Repeat", repeat=c["double"], layers=[{"type": "DoubleStreamBlock", "params": block}]),
        _node("single_blocks", "Repeat", repeat=c["single"], layers=[{"type": "SingleStreamBlock", "params": block}]),
        _node("image_only", "SliceTokens", start=c["txt_len"]),
        _node("final_norm", "AdaLayerNorm"),
        _node("final", "Linear", out_features=c["in_channels"], per_token=True),
        _node("out", "Output", any_shape=True),
    ]
    edges = [
        ["image", "img_in"], ["text", "txt_in"], ["pooled", "vec_in"], ["vec_in", "vec_act"], ["vec_act", "vec_out"],
        ["timestep", "time_in"],
    ]
    if c.get("guidance"):  # FLUX.1 [dev] also embeds the guidance strength
        nodes.insert(nodes.index(next(n for n in nodes if n["id"] == "condition")),
                     _input("guidance", [1], "guidance"))
        nodes.insert(nodes.index(next(n for n in nodes if n["id"] == "condition")),
                     _node("guidance_in", "TimestepEmbedding", dim=d, freq_dim=256, time_factor=1000.0))
        edges += [["guidance", "guidance_in"], ["time_in", "condition"], ["guidance_in", "condition"]]
    else:
        edges += [["time_in", "condition"]]
    edges += [
        ["vec_out", "condition"],
        # the joint sequence: the text first, then the image
        ["txt_in", "joint"], ["img_in", "joint"],
        ["joint", "double_blocks"], ["condition", "double_blocks"],
        ["double_blocks", "single_blocks"], ["condition", "single_blocks"],
        ["single_blocks", "image_only"],
        ["image_only", "final_norm"], ["condition", "final_norm"],
        ["final_norm", "final"], ["final", "out"],
    ]
    return _graph(nodes, edges, [], train)


# --------------------------------------------------------------------------- #
# TripoSR — one picture -> a triplane -> density and colour at any 3-D point
# --------------------------------------------------------------------------- #
def triposr_spec(c: dict, train: dict | None = None) -> Spec:
    v, b = c["vision"], c["backbone"]
    nodes = [
        _input("image", c["image"], "image"),
        _node("patches", "PatchEmbed", patch_size=v["patch"], dim=v["dim"], cls_token=True, position="learned",
              pos_grid=v.get("pos_grid", 0)),
        _encoder_layers("image_layers", v),
        _node("image_tokens", "LayerNorm", last_axis=True, eps=v["eps"]),
        _node("triplane", "TriplaneTokens", size=c["plane"], channels=b["dim"]),
        _node("backbone", "TriplaneTransformer", layers=b["layers"], heads=b["heads"], head_dim=b["head_dim"],
              groups=b["groups"]),
        _node("planes", "TriplaneUpsample", out_channels=c["features"]),
        _input("points", [c["points"], 3], "3-D points"),
        _node("sample", "TriplaneSample", radius=0.87),
        _node("mlp_in", "Linear", out_features=c["mlp"], per_token=True),
        _node("mlp_act", "SiLU"),
        _node("mlp_hidden", "Repeat", repeat=c["mlp_layers"] - 1, layers=[
            {"type": "Linear", "params": {"out_features": c["mlp"], "per_token": True}},
            {"type": "SiLU", "params": {}}]),
        _node("density_rgb", "Linear", out_features=4, per_token=True),
        _node("out", "Output", any_shape=True),
    ]
    edges = [
        ["image", "patches"], ["patches", "image_layers"], ["image_layers", "image_tokens"],
        ["image_tokens", "triplane"],
        # the backbone: the triplane tokens, then the image tokens they read
        ["triplane", "backbone"], ["image_tokens", "backbone"],
        ["backbone", "planes"],
        ["planes", "sample"], ["points", "sample"],
        ["sample", "mlp_in"], ["mlp_in", "mlp_act"], ["mlp_act", "mlp_hidden"], ["mlp_hidden", "density_rgb"],
        ["density_rgb", "out"],
    ]
    return _graph(nodes, edges, [], train)


# --------------------------------------------------------------------------- #
# Whisper — speech to text: an audio encoder and a text decoder that reads it
# --------------------------------------------------------------------------- #
def whisper_spec(c: dict, train: dict | None = None) -> Spec:
    d = c["dim"]
    layer = {"heads": c["heads"], "mlp": c["mlp"], "eps": 1e-5, "k_bias": False, "act": "gelu"}
    nodes = [
        _input("audio", [c["mels"], c["frames"]], "audio (log-mel)"),
        _node("conv1", "Conv1d", out_channels=d, kernel_size=3, stride=1, padding=1),
        _node("gelu1", "GELU"),
        _node("conv2", "Conv1d", out_channels=d, kernel_size=3, stride=2, padding=1),
        _node("gelu2", "GELU"),
        _node("to_sequence", "Transpose"),
        _node("audio_positions", "PositionEmbedding", length=c["frames"] // 2, kind="sinusoidal"),
        _encoder_layers("encoder", {**layer, "layers": c["enc_layers"]}),
        _node("audio_features", "LayerNorm", last_axis=True, eps=1e-5),
        _input("text", [c["text_len"]], "text so far"),
        _node("tokens", "Embedding", num_embeddings=c["vocab"], embedding_dim=d),
        _node("text_positions", "PositionEmbedding", length=c["text_len"], kind="learned"),
        _encoder_layers("decoder", {**layer, "layers": c["dec_layers"]}, causal=True, cross=True),
        _node("norm", "LayerNorm", last_axis=True, eps=1e-5),
        _node("head", "LMHead", vocab=c["vocab"], tie_embedding=True),
        _node("out", "Output", per_token=True),
    ]
    edges = [
        ["audio", "conv1"], ["conv1", "gelu1"], ["gelu1", "conv2"], ["conv2", "gelu2"], ["gelu2", "to_sequence"],
        ["to_sequence", "audio_positions"], ["audio_positions", "encoder"], ["encoder", "audio_features"],
        ["text", "tokens"], ["tokens", "text_positions"],
        # the decoder: the text, then the audio it listens to
        ["text_positions", "decoder"], ["audio_features", "decoder"],
        ["decoder", "norm"], ["norm", "head"], ["head", "out"],
    ]
    return _graph(nodes, edges, [], train)


# --------------------------------------------------------------------------- #
# The library entries
# --------------------------------------------------------------------------- #
# ~17 s on a laptop CPU: 0.90-0.93 on new digits (0.94-0.97 on the training ones), for seeds 0 and 1
_VIT_TRAIN = {"dataset": "mnist", "epochs": 10, "lr": 0.002, "batch_size": 64, "samples": 8000, "seed": 0}

#: id, names, what it is, Hugging Face repo, official parameter count (and where it comes
#: from), the graph builder, the mini and the real configuration.
VISION_FLAGSHIPS: list[dict] = [
    {
        "id": "vit", "name": "ViT", "real_name": "ViT-B/16", "hf": "google/vit-base-patch16-224",
        "official": 86_567_656, "source": "Hugging Face (safetensors total)",
        "category": "vision", "tags": ["vision", "transformer", "patches", "classifier"],
        "description": "Google's Vision Transformer: the picture cut into 16×16 patches, one token each, read by a "
                       "transformer; the class token gives the answer. The mini one learns MNIST here.",
        "build": vit_spec, "mini_train": _VIT_TRAIN,
        "mini": {"image": [1, 28, 28], "patch": 7, "dim": 64, "layers": 3, "heads": 4, "mlp": 128, "eps": 1e-6,
                 "classes": 10},
        "real": {"image": [3, 224, 224], "patch": 16, "dim": 768, "layers": 12, "heads": 12, "mlp": 3072,
                 "eps": 1e-12, "classes": 1000},
    },
    {
        "id": "clip", "name": "CLIP", "real_name": "CLIP ViT-L/14", "hf": "openai/clip-vit-large-patch14",
        "official": 427_616_513,
        "source": "Hugging Face: 427,616,512 weights + 1 for the logit scale, a single number it lists apart "
                  "(plus 334 integer position ids, not weights)",
        "category": "vision", "tags": ["vision", "text", "contrastive", "two-towers", "quick-gelu"],
        "description": "OpenAI's CLIP: an image transformer and a text transformer that map a picture and a "
                       "caption into one space; the score says how well they match. Its text half is Stable "
                       "Diffusion's text encoder.",
        "build": clip_spec,
        "mini": {"image": [3, 32, 32], "proj": 32,
                 "vision": {"patch": 8, "dim": 64, "layers": 2, "heads": 4, "mlp": 128, "eps": 1e-5,
                            "act": "quick_gelu"},
                 "text": {"length": 16, "vocab": 128, "dim": 64, "layers": 2, "heads": 4, "mlp": 128, "eps": 1e-5,
                          "act": "quick_gelu"}},
        "real": {"image": [3, 224, 224], "proj": 768,
                 "vision": {"patch": 14, "dim": 1024, "layers": 24, "heads": 16, "mlp": 4096, "eps": 1e-5,
                            "act": "quick_gelu"},
                 "text": {"length": 77, "vocab": 49408, "dim": 768, "layers": 12, "heads": 12, "mlp": 3072,
                          "eps": 1e-5, "act": "quick_gelu"}},
    },
    {
        "id": "stable_diffusion", "name": "Stable Diffusion", "real_name": "Stable Diffusion 1.5",
        "hf": "stable-diffusion-v1-5/stable-diffusion-v1-5", "official": 1_066_235_307,
        "parts": {"unet": 859_520_964, "text": 123_060_480, "vae": 83_653_863},
        "source": "U-Net 859,520,964 (Hugging Face) + text encoder 123,060,480 + VAE 83,653,863 (both match the "
                  "size of their published weight files)",
        "category": "generation", "tags": ["diffusion", "u-net", "vae", "cross-attention", "text-to-image"],
        "description": "Text to image: a CLIP text encoder, a U-Net that removes noise from a latent while reading "
                       "the text, and a VAE that turns latents into pictures. Here as one denoising step: picture "
                       "→ latent + noise → the U-Net's guess → picture.",
        "build": sd_spec,
        "mini": {"image": [3, 32, 32], "latent": [4, 16, 16],
                 "vae": {"channels": [32, 64], "layers": 1, "groups": 8},
                 "text": {"length": 16, "vocab": 128, "dim": 64, "layers": 2, "heads": 4, "mlp": 128, "eps": 1e-5,
                          "act": "quick_gelu"},
                 "unet": {"channels": [32, 64], "layers": 1, "attention": [1, 0], "heads": 4, "groups": 8}},
        "real": {"image": [3, 512, 512], "latent": [4, 64, 64],
                 "vae": {"channels": [128, 256, 512, 512], "layers": 2, "groups": 32},
                 "text": {"length": 77, "vocab": 49408, "dim": 768, "layers": 12, "heads": 12, "mlp": 3072,
                          "eps": 1e-5, "act": "quick_gelu"},
                 "unet": {"channels": [320, 640, 1280, 1280], "layers": 2, "attention": [1, 1, 1, 0], "heads": 8,
                          "groups": 32}},
    },
    {
        "id": "dit", "name": "DiT", "real_name": "DiT-XL/2", "hf": "facebook/DiT-XL-2-256", "official": 675_129_632,
        "source": "the paper's 675M; 675,129,632 counted the way the official code counts (its fixed position "
                  "table included)",
        "category": "generation", "tags": ["diffusion", "transformer", "adaln-zero", "class-conditional"],
        "description": "Meta's Diffusion Transformer: the noisy latent cut into patches and denoised by "
                       "transformer blocks, each steered by the timestep and the class through adaptive norms "
                       "(adaLN-Zero) — the design behind Sora and Stable Diffusion 3.",
        "build": dit_spec,
        "mini": {"latent": [4, 8, 8], "patch": 2, "dim": 64, "layers": 2, "heads": 4, "classes": 10,
                 "out_channels": 8},
        "real": {"latent": [4, 32, 32], "patch": 2, "dim": 1152, "layers": 28, "heads": 16, "classes": 1000,
                 "out_channels": 8},
    },
    {
        "id": "flux", "name": "FLUX.1", "real_name": "FLUX.1 [schnell]", "hf": "black-forest-labs/FLUX.1-schnell",
        "official": 11_891_178_560, "source": "Hugging Face (safetensors total)",
        "category": "generation", "tags": ["diffusion", "flow", "transformer", "mmdit", "rope"],
        "description": "Black Forest Labs' FLUX.1: text and image tokens in one sequence — 19 double-stream blocks "
                       "(separate weights, joint attention), then 38 single-stream blocks — steered by the "
                       "timestep and the pooled text. [dev] adds a guidance input (11,901,408,320).",
        "build": flux_spec,
        "mini": {"img_tokens": 16, "in_channels": 16, "txt_len": 8, "text_dim": 32, "pooled_dim": 16, "dim": 64,
                 "heads": 2, "head_dim": 32, "axes": [8, 12, 12], "double": 2, "single": 2},
        "real": {"img_tokens": 4096, "in_channels": 64, "txt_len": 512, "text_dim": 4096, "pooled_dim": 768,
                 "dim": 3072, "heads": 24, "head_dim": 128, "axes": [16, 56, 56], "double": 19, "single": 38},
    },
    {
        "id": "triposr", "name": "TripoSR", "real_name": "TripoSR", "hf": "stabilityai/TripoSR",
        "official": 418_685_036,
        "source": "no published count: counted from the published config and code; the 1.68 GB checkpoint holds "
                  "these numbers plus DINO's unused pooler (590,592) and about 140 KB of file structure",
        "category": "3d", "tags": ["3d", "triplane", "nerf", "image-to-3d", "cross-attention"],
        "description": "One picture to a 3-D object: a DINO image transformer, a transformer that turns learned "
                       "triplane tokens into three feature planes by reading the image, and a small MLP that gives "
                       "density and colour at any 3-D point.",
        "build": triposr_spec,
        "mini": {"image": [3, 64, 64], "plane": 8, "features": 8, "points": 256, "mlp": 32, "mlp_layers": 3,
                 "vision": {"patch": 8, "dim": 64, "layers": 2, "heads": 4, "mlp": 128, "eps": 1e-6},
                 "backbone": {"dim": 64, "layers": 2, "heads": 4, "head_dim": 16, "groups": 8}},
        "real": {"image": [3, 512, 512], "plane": 32, "features": 40, "points": 4096, "mlp": 64, "mlp_layers": 9,
                 "vision": {"patch": 16, "dim": 768, "layers": 12, "heads": 12, "mlp": 3072, "eps": 1e-12,
                            "pos_grid": 14},
                 "backbone": {"dim": 1024, "layers": 16, "heads": 16, "head_dim": 64, "groups": 32}},
    },
    {
        "id": "whisper", "name": "Whisper", "real_name": "Whisper large-v3", "hf": "openai/whisper-large-v3",
        "official": 1_543_490_560, "source": "Hugging Face (safetensors total)",
        "category": "audio", "tags": ["speech", "audio", "encoder-decoder", "cross-attention", "tied-head"],
        "description": "OpenAI's Whisper: speech to text. Two convolutions and a transformer encode 30 s of "
                       "audio; a text decoder writes the words, reading the audio through cross-attention.",
        "build": whisper_spec,
        "mini": {"mels": 16, "frames": 64, "dim": 64, "enc_layers": 2, "dec_layers": 2, "heads": 4, "mlp": 128,
                 "vocab": 64, "text_len": 16},
        "real": {"mels": 128, "frames": 3000, "dim": 1280, "enc_layers": 32, "dec_layers": 32, "heads": 20,
                 "mlp": 5120, "vocab": 51866, "text_len": 448},
    },
]


def mini_spec(fam: dict) -> Spec:
    build: Callable[..., Spec] = fam["build"]
    return build(fam["mini"], fam.get("mini_train"))


def real_spec(fam: dict) -> Spec:
    build: Callable[..., Spec] = fam["build"]
    return build(fam["real"], None)


def hf_url(fam: dict) -> str:
    return f"https://huggingface.co/{fam['hf']}"
