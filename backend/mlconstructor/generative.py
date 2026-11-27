"""Teach a network to generate (Stage 20): reconstruction and denoising diffusion.

Until now training taught one thing — scores for classes (or the next token).
Stage 20 adds two more, chosen from the graph itself (``train.task = "auto"``):

* **reconstruction** — one input, an output of the input's shape: the network
  learns to give back its input (an autoencoder). With a VAE sampling layer
  (:class:`~mlconstructor.gen_modules.Reparameterize`) its codes are kept close to
  N(0, I), so new pictures can be drawn from noise.
* **diffusion** — a noisy picture, a timestep and (optionally) a class in, a
  picture of the same shape out: the network learns to see the noise that was
  added (DDPM's noise prediction); a sampler (DDIM) then removes noise step by
  step, from pure noise to a picture. With a class it draws what you ask for;
  with one spare "no class" row in the class embedding it also learns
  unconditionally, which gives classifier-free guidance.

The part below the "export" marker is self-contained (torch only): the PyTorch
export copies it verbatim, so the exported script trains and draws like the app.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .registry import get_layer_cls

#: The tasks training knows; "auto" picks one from the graph.
TASKS = ("classification", "reconstruction", "diffusion")
#: Diffusion with a class: how often a label is replaced by "no class" in training
#: (so the same network also learns without a class: classifier-free guidance).
UNCOND_PROB = 0.1


def input_roles(spec, build) -> dict[str, tuple[str, float]]:
    """What each Input feeds, to know what values suit it: ``("tokens", vocab)`` for an
    Embedding (token ids, or a class), ``("timestep", top)`` for a timestep (top = 1
    for a flow model such as FLUX, else the number of steps), ``("noise", 0)`` else."""
    nodes = {n.id: n for n in spec.nodes}
    roles: dict[str, tuple[str, float]] = {}
    for inp in build.inputs:
        role: tuple[str, float] = ("noise", 0.0)
        for src, dst in spec.edges:
            if src != inp.id or dst not in nodes:
                continue
            node = nodes[dst]
            if node.type == "Embedding":
                role = ("tokens", float(int(node.params.get("num_embeddings", 2))))
                break
            if node.type == "TimestepEmbedding":
                role = ("timestep", 1.0 if float(node.params.get("time_factor", 1.0)) >= 100 else 1000.0)
                break
            layer = get_layer_cls(node.type)(node.id, node.params)
            ports = layer.input_ports()
            preds = [s for s, d in spec.edges if d == dst]
            if ports and len(preds) == len(ports) and ports[preds.index(src)] == "timestep":
                role = ("timestep", float(int(node.params.get("steps", 1000))))
                break
        roles[inp.id] = role
    return roles


@dataclass
class TaskPlan:
    """What training does with a graph — or, with ``task`` None, why it cannot."""

    task: str | None
    reason: str | None = None
    #: Stage 21 — the catalogue code of ``reason``
    code: str | None = None
    #: the Input with the pictures (reconstruction: the only one; diffusion: the noisy picture)
    image_input: str | None = None
    timestep_input: str | None = None
    #: diffusion with a class: the Input with the class, and its embedding's rows
    class_input: str | None = None
    class_rows: int = 0
    out_shape: tuple = ()
    in_shapes: dict = field(default_factory=dict)
    #: reconstruction: the network has a VAE sampling layer (it can draw from noise)
    vae: bool = False
    #: the network ends in a Sigmoid (outputs in 0…1): reconstruct with cross-entropy
    sigmoid_out: bool = False


def _shape(shape) -> str:
    return "×".join(str(int(d)) for d in shape) if shape else "()"


def _has_sampler(spec) -> bool:
    for n in spec.nodes:
        if n.type == "Reparameterize":
            return True
        if n.type == "Repeat" and any((l or {}).get("type") == "Reparameterize" for l in n.params.get("layers", []) or []):
            return True
    return False


def resolve_task(spec, build) -> TaskPlan:
    """The training task for this graph: ``train.task`` when given, else the one its
    shape asks for (one input and class scores: classification; one input and an
    output of its shape: reconstruction; a noisy picture + a timestep (+ a class) in,
    the picture's shape out: diffusion)."""
    want = getattr(spec.train, "task", "auto") or "auto"
    inputs = list(getattr(build, "inputs", None) or [])
    in_shapes = {i.id: tuple(int(d) for d in i.shape) for i in inputs}
    out_node = next((n for n in spec.nodes if n.type == "Output"), None)
    any_shape = bool(out_node is not None and out_node.params.get("any_shape", False))
    out_shape = tuple(next((s.out_shape for s in build.layers if out_node is not None and s.id == out_node.id), None)
                      or ())
    before_out = [s for s, d in spec.edges if out_node is not None and d == out_node.id]
    sigmoid_out = bool(before_out) and next((n.type for n in spec.nodes if n.id == before_out[0]), "") == "Sigmoid"
    names = ", ".join(i.name or i.id for i in inputs)
    plan = dict(out_shape=out_shape, in_shapes=in_shapes, sigmoid_out=sigmoid_out)

    if len(inputs) == 1 and want in ("auto", "classification") and not any_shape:
        return TaskPlan("classification", image_input=inputs[0].id, **plan)
    if len(inputs) == 1 and want in ("auto", "reconstruction") and any_shape:
        if in_shapes[inputs[0].id] == out_shape:
            return TaskPlan("reconstruction", image_input=inputs[0].id, vae=_has_sampler(spec), **plan)
        return TaskPlan(None, code="task_output_shape", reason=(
            f"This network's output is {_shape(out_shape)} — a picture, a latent or points, not one score per class. "
            f"To learn to reconstruct its input, the output must have the input's shape ({_shape(in_shapes[inputs[0].id])})."
            " As it is, it builds, runs once on random inputs (Run once) and exports."), **plan)
    if len(inputs) == 1:
        if want == "reconstruction":
            return TaskPlan(None, code="task_needs_any_shape",
                            reason="To learn to reconstruct its input, set the Output to “Any shape” and give it the "
                                   "input's shape.", **plan)
        return TaskPlan(None, code="task_any_shape_on",
                        reason="Classification needs one score per class: switch the Output's “Any shape” off "
                               "(or let the task be chosen automatically).", **plan)

    # several inputs: diffusion is the only way to train them here
    roles = input_roles(spec, build)
    times = [i for i in inputs if roles[i.id][0] == "timestep"]
    classes = [i for i in inputs if roles[i.id][0] == "tokens" and math.prod(in_shapes[i.id]) == 1]
    images = [i for i in inputs if in_shapes[i.id] == out_shape and i not in times and i not in classes]
    problem = None
    if want not in ("auto", "diffusion"):
        problem = f"the task is set to {want}, which takes one input"
    elif not any_shape:
        problem = "its Output is not “Any shape” (it should give a picture: the noise it sees)"
    elif len(times) != 1:
        problem = "none of its inputs is a timestep (an Input [1] into a Timestep embedding or a U-Net's “timestep”)" \
            if not times else "it has more than one timestep input"
    elif not images:
        problem = f"none of its inputs has the output's shape {_shape(out_shape)} (the noisy picture)"
    elif len(classes) > 1:
        problem = "it has more than one class input"
    else:
        others = [i for i in inputs if i is not images[0] and i not in times and i not in classes]
        if others:
            problem = f"it also takes {', '.join(i.name or i.id for i in others)}, which diffusion training here cannot fill"
    if problem is None:
        rows = 0
        if classes:
            emb = next((n for s, d in spec.edges if s == classes[0].id
                        for n in spec.nodes if n.id == d and n.type == "Embedding"), None)
            rows = int(emb.params.get("num_embeddings", 0)) if emb is not None else 0
        return TaskPlan("diffusion", image_input=images[0].id, timestep_input=times[0].id,
                        class_input=classes[0].id if classes else None, class_rows=rows, **plan)
    return TaskPlan(None, code="task_inputs", reason=(
        f"This network takes {len(inputs)} inputs ({names}). Training here learns from one input (to classify it or to "
        "reconstruct it), or learns to generate by diffusion — a noisy picture, a timestep and maybe a class in, a "
        f"picture of the same shape out — but {problem}. It builds, runs once on random inputs (Run once) and exports."
    ), **plan)


def image_items(x: torch.Tensor, labels: list[str]) -> list[dict]:
    """Pictures for the gallery: (n, C, H, W) values in 0…1 -> small records with the
    pixels as base64 bytes (channels last), drawn by the editor on a canvas."""
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if x.dim() != 4 or x.shape[1] not in (1, 3):
        return []
    px = (x.detach().float().clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu()
    out = []
    for i in range(px.shape[0]):
        out.append({"label": labels[i] if i < len(labels) else "", "w": int(px.shape[2]), "h": int(px.shape[1]),
                    "c": int(px.shape[3]), "data": base64.b64encode(px[i].numpy().tobytes()).decode("ascii")})
    return out


# --- export: begin ---


def lr_factor(step, total):
    """The learning rate's share at a step: a cosine from 1 at the start down to 0.1 at
    the end — big steps first, careful ones at the end."""
    if total <= 1:
        return 1.0
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step, total) / total))


def ema_decay(total):
    """How slowly the averaged weights follow the trained ones: over about the last
    tenth of the run, between 0.99 and 0.999."""
    return min(0.999, max(0.99, 1 - 10 / max(total, 1)))


class EMA:
    """A running average of a network's weights. A diffusion model draws much cleaner
    pictures with it than with its last, still jittery weights."""

    def __init__(self, model, decay):
        self.decay = float(decay)
        self.shadow = [p.detach().clone() for p in model.parameters()]

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow, model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def swap(self, model):
        """Exchange the network's weights and the averaged ones (swap again to undo)."""
        for s, p in zip(self.shadow, model.parameters()):
            keep = p.detach().clone()
            p.copy_(s)
            s.copy_(keep)


def noise_schedule(steps=1000, beta_start=1e-4, beta_end=0.02):
    """DDPM's linear noise schedule: how much of the clean picture survives at each
    step (alphas_cumprod), from almost all at step 0 to almost nothing at the end."""
    betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float64)
    return torch.cumprod(1.0 - betas, dim=0).float()


def noisy_batch(x0, alphas, generator=None):
    """A training example for diffusion: a random step t for every picture, random
    noise, and the picture with that much noise mixed in."""
    n = x0.shape[0]
    t = torch.randint(0, alphas.numel(), (n,), generator=generator).to(x0.device)
    noise = torch.randn(x0.shape, generator=generator).to(x0.device)
    a = alphas.to(x0.device)[t].view(-1, *([1] * (x0.dim() - 1)))
    return t, noise, a.sqrt() * x0 + (1 - a).sqrt() * noise


@torch.no_grad()
def ddim_sample(predict_noise, shape, alphas, steps=50, generator=None, device="cpu"):
    """Draw pictures from pure noise: at each of `steps` steps the network names the
    noise, the clean picture it implies is computed, and a little less noise is mixed
    back in (DDIM, deterministic). predict_noise(x, t) -> the noise in x at step t."""
    x = torch.randn(shape, generator=generator).to(device)
    ts = torch.linspace(alphas.numel() - 1, 0, max(int(steps), 1)).round().long().tolist()
    alphas = alphas.to(device)
    x0 = x
    for i, t in enumerate(ts):
        a = alphas[t]
        a_prev = alphas[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
        eps = predict_noise(x, torch.full((shape[0], 1), float(t), device=device))
        x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
    return x0


def guide(with_class, without, scale):
    """Classifier-free guidance: start from the noise seen without a class and move
    `scale` times as far towards the noise seen with it (1 = just the class)."""
    return without + scale * (with_class - without)


def recon_loss(out, target, kind="mse"):
    """How far a reconstruction is from the original, per number: squared error, or
    binary cross-entropy for pictures in 0…1 made by a Sigmoid."""
    if kind == "bce":
        return F.binary_cross_entropy(out.clamp(1e-6, 1 - 1e-6), target.clamp(0, 1))
    return F.mse_loss(out, target)


def kl_total(model):
    """The KL terms the VAE sampling layers recorded in the last forward pass (0 without any)."""
    terms = [m.kl for m in model.modules() if getattr(m, "kl", None) is not None]
    return sum(terms) if terms else 0.0


def set_prior(model, on):
    """VAE sampling layers draw codes from N(0, I) (on) or from the input (off)."""
    for m in model.modules():
        if hasattr(m, "from_prior"):
            m.from_prior = bool(on)
