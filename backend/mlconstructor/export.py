"""Export a graph as standalone PyTorch source (Stage 8).

"Build it visually, then take the code with you." Given a :class:`GraphSpec`, this
emits a self-contained ``.py`` — a real ``nn.Module`` that reconstructs the whole
directed graph (branches, ``Add``/``Concat`` merges, ``Repeat`` blocks and all),
plus an optional tiny training loop on the same synthetic task — depending on
**nothing but torch**. No ``mlconstructor`` import; edit freely from there.

How it stays faithful: the graph is built once with the normal builder, and each
concrete ``nn`` module is turned back into its constructor expression — standard
layers from their (eval-able) ``repr``, the few custom wrappers from explicit
emitters that mirror the originals exactly. The result builds to a model with the
same parameters and the same output shape as the one trained in the app.
"""

from __future__ import annotations

import inspect

import torch.nn as nn

from .blocks import Block
from .builder import build_model
from .connections import ImageToSeq, SeqPool
from .datasets import split_indices
from .layers.attention import _PositionalEncoding, _SelfAttention
from .layers.sequence import _EmbeddingModule, _RecurrentModule
from .schema import GraphSpec


class ExportError(ValueError):
    """A graph uses a construct the exporter does not yet know how to emit."""


def _key(node_id: str) -> str:
    """A safe, unique-ish attribute name for a node id (matches GraphModule)."""
    return "n_" + "".join(c if c.isalnum() else "_" for c in node_id)


# --------------------------------------------------------------------------- #
# Inlined helper classes — emitted only when the graph actually uses them, so the
# generated file carries exactly the wrappers it needs and nothing more. Each is a
# faithful copy of its mlconstructor counterpart.
# --------------------------------------------------------------------------- #
_HELPERS: dict[str, str] = {
    "TokenEmbedding": '''
class TokenEmbedding(nn.Module):
    """Embedding lookup that tolerates float / out-of-range ids (cast + modulo)."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.emb = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        return self.emb(torch.remainder(x.long(), self.num_embeddings))
''',
    "Recurrent": '''
class Recurrent(nn.Module):
    """Run a recurrent layer; return the full sequence or just the last step."""

    def __init__(self, rnn, return_sequence):
        super().__init__()
        self.rnn = rnn
        self.return_sequence = return_sequence

    def forward(self, x):
        out, _ = self.rnn(x)
        return out if self.return_sequence else out[:, -1]
''',
    "SelfAttention": '''
class SelfAttention(nn.Module):
    """Multi-head self-attention: query = key = value = x."""

    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        out, _ = self.attn(x, x, x, need_weights=False)
        return out
''',
    "PositionalEncoding": '''
class PositionalEncoding(nn.Module):
    """Add fixed sinusoidal positional encodings to a (B, L, E) sequence."""

    def __init__(self, embed_dim, max_len, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        half = embed_dim // 2
        div = torch.exp(torch.arange(0, 2 * half, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim))
        pe[:, 0 : 2 * half : 2] = torch.sin(pos * div)
        pe[:, 1 : 2 * half : 2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, : x.size(1)])
''',
    "SeqPool": '''
class SeqPool(nn.Module):
    """Collapse a (B, L, F) sequence to (B, F): the last or first step, the mean or the max."""

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        if self.mode == "last":
            return x[:, -1]
        if self.mode == "first":
            return x[:, 0]
        if self.mode == "mean":
            return x.mean(dim=1)
        return x.amax(dim=1)
''',
    "ImageToSeq": '''
class ImageToSeq(nn.Module):
    """Read a (B, C, H, W) feature map as a sequence: rows, columns, pixels or patches."""

    def __init__(self, mode, patch=4):
        super().__init__()
        self.mode = mode
        self.patch = patch

    def forward(self, x):
        b, c, h, w = x.shape
        if self.mode == "rows":
            return x.permute(0, 2, 1, 3).reshape(b, h, c * w)
        if self.mode == "columns":
            return x.permute(0, 3, 1, 2).reshape(b, w, c * h)
        if self.mode == "pixels":
            return x.flatten(2).transpose(1, 2)
        p = self.patch
        x = x.unfold(2, p, p).unfold(3, p, p).permute(0, 2, 3, 1, 4, 5)
        return x.reshape(b, (h // p) * (w // p), c * p * p)
''',
}


def _transformer_src(m: nn.TransformerEncoderLayer) -> str:
    d_model = int(m.linear1.in_features)
    dim_feedforward = int(m.linear1.out_features)
    nhead = int(m.self_attn.num_heads)
    dropout = float(m.dropout.p)
    act_fn = getattr(m, "activation", None)
    activation = "gelu" if getattr(act_fn, "__name__", "") == "gelu" else "relu"
    return (
        f"nn.TransformerEncoderLayer(d_model={d_model}, nhead={nhead}, "
        f"dim_feedforward={dim_feedforward}, dropout={dropout}, "
        f"activation={activation!r}, batch_first=True)"
    )


def _module_expr(m: nn.Module, needs: set[str], indent: str = "            ") -> str:
    """The constructor expression that re-creates module ``m`` in standalone code.

    Records any inlined helper the expression needs in ``needs``. Raises
    :class:`ExportError` for a module whose ``repr`` is not a single eval-able line
    (so an unsupported new layer fails loudly instead of emitting broken code)."""
    if isinstance(m, _PositionalEncoding):
        needs.add("PositionalEncoding")
        embed = int(m.pe.shape[2])
        max_len = int(m.pe.shape[1])
        return f"PositionalEncoding({embed}, {max_len}, {float(m.drop.p)})"
    if isinstance(m, _SelfAttention):
        needs.add("SelfAttention")
        a = m.attn
        return f"SelfAttention({int(a.embed_dim)}, {int(a.num_heads)}, {float(a.dropout)})"
    if isinstance(m, _EmbeddingModule):
        needs.add("TokenEmbedding")
        return f"TokenEmbedding({int(m.num_embeddings)}, {int(m.emb.embedding_dim)})"
    if isinstance(m, _RecurrentModule):
        needs.add("Recurrent")
        return f"Recurrent(nn.{m.rnn!r}, {bool(m.return_sequence)})"
    if isinstance(m, nn.TransformerEncoderLayer):
        return _transformer_src(m)
    if isinstance(m, SeqPool):  # Stage 14 connection transforms
        needs.add("SeqPool")
        return f"SeqPool({m.mode!r})"
    if isinstance(m, ImageToSeq):
        needs.add("ImageToSeq")
        return f"ImageToSeq({m.mode!r}, {int(m.patch)})"
    if isinstance(m, nn.Sequential) and not isinstance(m, Block):  # e.g. a connection's flatten + unflatten
        inner = [_module_expr(sub, needs, indent + "    ") for sub in m]
        joined = (",\n" + indent + "    ").join(inner)
        return "nn.Sequential(\n" + indent + "    " + joined + ",\n" + indent + ")"
    if isinstance(m, Block):  # a Repeat node: its flattened inner chain, in order
        inner = [_module_expr(sub, needs, indent + "    ") for sub in m.layers]
        joined = (",\n" + indent + "    ").join(inner)
        return "nn.Sequential(\n" + indent + "    " + joined + ",\n" + indent + ")"
    r = repr(m)
    if "\n" in r:
        raise ExportError(
            f"export does not yet support layer {type(m).__name__} (its repr is not "
            f"a single constructor line)"
        )
    return "nn." + r


# --------------------------------------------------------------------------- #
# Synthetic-data generators, mirroring mlconstructor.datasets, for the optional
# training loop in the exported file.
# --------------------------------------------------------------------------- #
_DATA_SRC: dict[str, str] = {
    "blobs": '''
def make_data(n, in_shape, num_classes, seed=0):
    g = torch.Generator().manual_seed(seed)
    proto = torch.randn(num_classes, *in_shape, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    noise = torch.randn(n, *in_shape, generator=g)
    return proto[y] + 0.5 * noise, y
''',
    "shells": '''
def make_data(n, in_shape, num_classes, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, num_classes, (n,), generator=g)
    d = torch.randn(n, *in_shape, generator=g)
    norm = d.flatten(1).norm(dim=1).view(n, *([1] * len(in_shape))) + 1e-8
    unit = d / norm
    radius = (y.float() + 1.0).view(n, *([1] * len(in_shape)))
    return unit * radius + 0.15 * torch.randn(n, *in_shape, generator=g), y
''',
}

# Custom data is embedded as a CUSTOM_ROWS literal above this snippet.
_CUSTOM_DATA_SRC = '''
def make_data(n, in_shape, num_classes, seed=0):
    feats = [row[:-1] for row in CUSTOM_ROWS]
    labels = [int(row[-1]) for row in CUSTOM_ROWS]
    X = torch.tensor(feats, dtype=torch.float).reshape(len(CUSTOM_ROWS), *in_shape)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y
'''

_TRAIN_BODY = '''
def _num_classes(model):
    with torch.no_grad():
        return int(model(torch.zeros(2, *IN_SHAPE)).shape[1])


def main():
    torch.manual_seed(SEED)
    model = Model()
    num_classes = _num_classes(model)
    X, y = make_data(SAMPLES, IN_SHAPE, num_classes, seed=SEED)
    # Hold rows out for validation, exactly as the app does: train on the rest only.
    train_idx, val_idx = split_indices(y, VAL_FRACTION, SEED)
    Xv, yv = X[val_idx], y[val_idx]
    X, y = X[train_idx], y[train_idx]
    opt = OPTIMIZER(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    n = X.shape[0]
    bs = min(BATCH_SIZE, n)
    model.train()
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(X).argmax(1) == y).float().mean().item()
            val = f"  val_acc {(model(Xv).argmax(1) == yv).float().mean().item():.3f}" if len(yv) else ""
        model.train()
        print(f"epoch {epoch + 1}/{EPOCHS}  loss {loss.item():.3f}  train_acc {acc:.3f}{val}")
    return model


def export_model(model, onnx=True):
    """Save the trained weights, and (best effort) an ONNX graph -- Stage 9."""
    model.eval()
    torch.save(model.state_dict(), "model.pt")
    print("saved weights -> model.pt")
    if onnx:
        try:
            dummy = torch.zeros(1, *IN_SHAPE)
            names = dict(input_names=["input"], output_names=["logits"],
                        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
            try:
                # Legacy exporter: no onnxscript needed (the dynamo path requires it).
                torch.onnx.export(model, dummy, "model.onnx", dynamo=False, **names)
            except TypeError:
                torch.onnx.export(model, dummy, "model.onnx", **names)  # older torch
            print("exported -> model.onnx")
        except Exception as exc:
            print(f"ONNX export skipped: {exc}")


if __name__ == "__main__":
    export_model(main())
'''


def generate_pytorch(spec: GraphSpec, include_training: bool = True) -> str:
    """Standalone PyTorch source for ``spec``. See the module docstring."""
    from .datasets import get_dataset  # local: avoid import at module load

    model, build = build_model(spec)
    order = model.order
    preds = model.preds
    input_id = model.input_id
    output_id = model.output_id
    merge_ids = model.merge_ids
    mods = model._mods
    node_by_id = {n.id: n for n in spec.nodes}
    in_shape = tuple(int(d) for d in build.layers[0].out_shape)

    needs: set[str] = set()
    init_lines: list[str] = []
    has_module: dict[str, bool] = {}
    for nid in order:
        key = _key(nid)
        if key in mods:
            init_lines.append(f"        self.{key} = {_module_expr(mods[key], needs)}")
            has_module[nid] = True
        else:
            has_module[nid] = False
    # Stage 14: the connections' transforms, registered after the layers (as in the app)
    edge_keys = model._edge_keys
    for key in edge_keys.values():
        init_lines.append(f"        self.{key} = {_module_expr(model._edge_mods[key], needs)}")

    def along(src: str, dst: str) -> str:
        """What ``src`` sends to ``dst`` in the generated forward()."""
        key = edge_keys.get((src, dst))
        return f"self.{key}(h[{src!r}])" if key else f"h[{src!r}]"

    # forward: replicate the DAG walk (merges combine predecessors; everyone else
    # takes its single predecessor), applying each node's module where it has one.
    fwd_lines: list[str] = ["        h = {}"]
    for nid in order:
        ref = repr(nid)
        if nid == input_id:
            rhs = "x"
        elif nid in merge_ids:
            node = node_by_id[nid]
            parts = [along(p, nid) for p in preds[nid]]
            if node.type == "Concat":
                dim = int(node.params.get("dim", 0)) + 1  # +1: tensors carry a batch dim
                rhs = f"torch.cat([{', '.join(parts)}], dim={dim})"
            else:  # Add
                rhs = " + ".join(parts)
        else:
            rhs = along(preds[nid][0], nid)
        if has_module[nid]:
            fwd_lines.append(f"        h[{ref}] = self.{_key(nid)}({rhs})")
        else:
            fwd_lines.append(f"        h[{ref}] = {rhs}")
    fwd_lines.append(f"        return h[{output_id!r}]")

    # ---- assemble the file ----
    out: list[str] = []
    out.append('"""Standalone PyTorch model exported from ML Constructor.')
    out.append("")
    out.append("Depends on nothing but torch. The architecture below is the graph you")
    out.append("built, as a plain nn.Module — edit it freely from here.")
    out.append('"""')
    out.append("")
    out.append("import torch")
    out.append("import torch.nn as nn")
    if "PositionalEncoding" in needs:
        out.append("import math")
    out.append("")
    for name in ("TokenEmbedding", "Recurrent", "SelfAttention", "PositionalEncoding", "SeqPool", "ImageToSeq"):
        if name in needs:
            out.append(_HELPERS[name].strip("\n"))
            out.append("")
            out.append("")
    out.append("class Model(nn.Module):")
    out.append("    def __init__(self):")
    out.append("        super().__init__()")
    out.extend(init_lines or ["        pass"])
    out.append("")
    out.append("    def forward(self, x):")
    out.extend(fwd_lines)

    if include_training:
        ds = get_dataset(spec.train.dataset)
        opt = "torch.optim.Adam" if spec.train.optimizer == "adam" else "torch.optim.SGD"
        rows = spec.train.custom_rows or []
        if ds.id == "custom" and rows:
            comment = "# ---- training on your custom data ----"
            data_block = "CUSTOM_ROWS = " + repr(rows) + "\n\n\n" + _CUSTOM_DATA_SRC.strip("\n")
        elif ds.id == "custom":
            comment = "# ---- no custom rows provided; using synthetic 'blobs' as a placeholder ----"
            data_block = _DATA_SRC["blobs"].strip("\n")
        else:
            comment = f"# ---- training on the same synthetic task the app used ({ds.id}) ----"
            data_block = _DATA_SRC.get(ds.id, _DATA_SRC["blobs"]).strip("\n")
        consts = [
            "",
            "",
            comment,
            f"IN_SHAPE = {in_shape!r}",
            f"SEED = {int(spec.train.seed)}",
            f"SAMPLES = {int(spec.train.samples)}",
            f"EPOCHS = {int(spec.train.epochs)}",
            f"LR = {float(spec.train.lr)}",
            f"BATCH_SIZE = {int(spec.train.batch_size)}",
            f"OPTIMIZER = {opt}",
            f"VAL_FRACTION = {float(spec.train.val_fraction)}  # share of rows held out for validation",
        ]
        out.extend(consts)
        out.append(data_block)
        out.append("")
        out.append("")
        out.append(inspect.getsource(split_indices).strip("\n"))  # the app's own split, verbatim
        out.append("")
        out.append("")
        out.append(_TRAIN_BODY.strip("\n"))

    return "\n".join(out) + "\n"
