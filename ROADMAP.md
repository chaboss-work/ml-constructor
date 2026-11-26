# ML Constructor — Project Description & Roadmap

*A visual IDE for building neural networks — "ComfyUI for neural nets."*

[English](ROADMAP.md) · Русская версия — в конце файла / Russian version at the bottom.

---

## What it is

ML Constructor turns building a neural network into a **node-graph on a canvas**.
You drag layer blocks (Input, Conv2d, Linear, attention, …), wire them into an
architecture, press **Train**, and a **PyTorch backend** assembles the graph into
a real `torch.nn.Module`, trains it, and streams the result back to the page — no
code required.

It is a **standalone, self-hostable, source-available project** (PolyForm
Noncommercial). Anyone can clone it and stand it up from the README.

### Why it exists

Designing and experimenting with architectures usually means writing boilerplate
`nn.Module` code, re-deriving tensor shapes by hand, and restarting on every tweak.
ML Constructor makes the architecture **visual, inspectable and runnable**:

* **See the shapes.** Every connection is validated and tensor shapes are inferred
  and shown on each node *before* training — no more guessing `in_features`.
* **Build, don't boilerplate.** The graph *is* the model; the backend compiles it.
* **Learn by doing.** Great for teaching, prototyping and sharing architectures.

### How it's built

* **Backend** — Python, **FastAPI + PyTorch**. One class per layer, self-registering
  into a catalogue; "mother" classes compose layers into blocks (with repeats); a
  builder compiles the graph spec into an `nn.Module`; a trainer runs it.
* **Frontend** — **Angular 22 + Ionic 9**, a lightweight custom SVG node editor
  (signals, zoneless). The palette is generated from the backend's `/layers`.
* **Contract** — a single JSON graph spec is the whole UI⇄backend API.

### Architecture at a glance

```
 ┌──────────────┐     graph spec (JSON)      ┌─────────────────────────────┐
 │  Node editor │  ───────────────────────▶  │  FastAPI backend (PyTorch)  │
 │ (Angular 22  │   /layers /build /train    │  spec → nn.Module → train   │
 │  / Ionic 9)  │  ◀───────────────────────  │  shape inference, DAG build │
 └──────────────┘   shapes / loss / accuracy └─────────────────────────────┘
```

---

## Development roadmap

Development proceeds in **stages**. Each finished stage is a self-contained,
demoable increment — documented, tested, deployed to the live app, published to
GitHub, and turned into a narrated video on the project's channels.

### Stage 0 — v1 thin slice · **DONE (2026-11-15)**
The whole pipeline end-to-end: node editor → PyTorch → train → result on the page.
9 layers (Input, Conv2d, BatchNorm2d, Flatten, Linear, ReLU, Sigmoid, Tanh, Output),
single linear chain, shape inference, synthetic-data training. Live demo reached
accuracy 1.000. Hosted on the machine and linked from the dashboard.

### Stage 1 — Branching & merges (DAG) · **DONE (2026-11-17)**
Move from a single chain to a **directed acyclic graph**: a node's output can fan
out to several nodes, and **merge nodes** (`Add`, `Concat`) combine several inputs.
Enables skip/residual connections, parallel paths and multi-branch designs. The
editor supports branches and guides merges; the builder compiles arbitrary DAGs
with shape propagation through the merges.

### Stage 2 — Richer layer catalogue · **DONE (2026-11-17)**
Dropout; pooling (MaxPool2d, AvgPool2d, AdaptiveAvgPool2d); more normalization
(BatchNorm1d, LayerNorm, GroupNorm); more activations (GELU, LeakyReLU, Softmax);
`ConvTranspose2d`. Fuller per-layer parameter control.

### Stage 3 — Sequence models · **DONE (2026-11-17)**
`Embedding`, `RNN`, `LSTM`, `GRU` on the `(L, F)` sequence convention
(`batch_first`). Embedding maps `(L,)` token ids to `(L, embedding_dim)`; the
recurrent layers map `(L, input_size)` to the last step `(hidden_size,)` or the
full sequence `(L, hidden_size)`. Build and train a small sequence model in the
editor (e.g. Input → Embedding → LSTM → Linear).

### Stage 4 — Attention & Transformers · **DONE (2026-11-17)**
`MultiheadAttention` (self-attention), `TransformerEncoderLayer`, sinusoidal
`PositionalEncoding` — all shape-preserving over the `(L, E)` convention — plus
`LayerNorm` from Stage 2. Enough to assemble and train a small transformer, e.g.
Input → Embedding → PositionalEncoding → TransformerEncoderLayer → Flatten → Linear.

### Stage 5 — Smart adapters · **DONE (2026-11-17)**
Shape reconciliation on connections: when a source shape doesn't fit the target,
the backend `/suggest` endpoint proposes an adapter node to insert — Flatten is
the first (it collapses a feature map or a sequence to a vector) — and the editor
shows the fix inline and inserts it on one click. Fewer manual Flattens; the
candidate framework is open for reshape / pooling / dtype adapters next.

### Stage 6 — Reusable blocks & repeats · **DONE (2026-11-17)**
A `Repeat` node holds an inner linear chain and stamps it **N×** (reusing the
`Block` mother class), behaving like any single-input layer — its output shape is
the chain applied N times and each stamp gets fresh parameters. The editor adds a
Block node with an inner-chain editor (add / remove / edit inner layers, set N).
"Edit the block once, stamp it many times."

### Stage 7 — Architecture library · **DONE (2026-11-18)**
A library of ready public architectures — **MLP**, **simple CNN**, **ResNet**
(whose skip connections are real `Add` merge nodes) and a **tiny GPT** (token
embedding + sinusoidal positional encoding + an N-deep stack of transformer blocks
expressed with the Stage 6 `Repeat` block) — served from `GET /architectures`. Pick
one and it drops onto the canvas as a full graph of **ordinary, editable nodes**:
every layer, every repeat, every skip is there to re-wire and tune — "start from a
known-good net, then make it your own." Each one builds and trains end to end.

### Stage 8 — Datasets, data preview & export · **DONE (2026-11-18)**
See the data, and take the model with you. A **dataset layer** feeds both the
trainer and a new **data preview** (`/data/preview`): a modal showing the exact
format — input shape, dtype, class count, sample count — and a few **real sample
rows** the model will train on (the headline "data modal"). A **dataset picker**
chooses the synthetic task (`blobs` — separable clusters; `shells` — a non-linear
radial boundary), the result now plots an **accuracy curve** per epoch next to the
loss, and one click **exports standalone PyTorch** (`/export/pytorch`): a real
`nn.Module` reconstructing the whole graph — branches, `Add`/`Concat` merges,
`Repeat` blocks and all — plus an optional training loop, depending on nothing but
torch. Verified end to end: every architecture exports to code that rebuilds to the
same parameters and output.

### Stage 9 — Bring your own data & model export · **DONE (2026-11-19)**
Train on your own data, and take the trained model out. A **custom dataset** lets you
**paste or upload** training rows (each row: its features, then a class label); the
backend parses and validates them against the graph (clear messages on a bad shape or
label), the **Data** modal previews exactly what you gave, and training runs on
precisely those rows. Export grew **more targets**: the generated, dependency-free
script now also saves the trained weights (`torch.save` → `model.pt`) and exports
**ONNX** (`model.onnx`, best-effort). Verified end to end — custom rows train to
accuracy 1.000, bad rows are rejected, and the exported file writes a real `state_dict`
and ONNX graph.

### Stage 10 — Built-in real datasets (MNIST / CIFAR-10) · **DONE (2026-11-19)**
Real datasets behind the same dataset contract, auto-downloaded from their official
sources (no torchvision) and cached locally: **MNIST** (28×28 digits) and
**CIFAR-10** (32×32 colour), each parsed with the standard library, shape/class
validated against the graph (a clear message if your Input doesn't match), previewed
like any dataset and trained on a reproducible subset. Verified end to end — MNIST
reaches ~0.96 accuracy on a small CPU run; CIFAR-10 downloads, parses and trains
(loss drops). The dataset picker lists them and hints the Input shape they need.

### Stage 11 — Live training: streaming curves & checkpoints · **DONE (2026-11-19)**
Training streams its progress live: `POST /train/stream` emits newline-delimited JSON
(`start` → a per-step `step` → a per-epoch `epoch` → `done`), and the editor plots the
loss and accuracy curves as they arrive. **Checkpoints** save the best-accuracy weights
during training (keyed by the architecture) and **resume** from them on the next run —
verified end to end (resuming starts far below a fresh run's loss). The trainer is
**device-aware** (uses CUDA automatically when a CUDA PyTorch build is present); the
bundled backend is CPU-only, so actually running on the GPU is Stage 12.

### Stage 12 — GPU training (explicit opt-in) · **DONE (2026-11-19)**
Training can run on the GPU: the backend runs on a CUDA PyTorch build, and the
trainer takes a `device` — `cpu` (the default, so a shared server never grabs the card
on its own), `cuda`, or `auto`; asking for `cuda` without a CUDA device fails with a
clear message, and after a GPU run the weights and data are handed back to the card.
`GET /health` reports `cuda_available` (without opening a CUDA context), and the editor
shows a **Device** picker only when a GPU is there. Verified on an RTX 3060 Laptop GPU:
a small MNIST CNN trains **~9.7× faster** than on the CPU (0.63 s vs 6.15 s for 3
epochs), and live streaming works on CUDA. This stage also fixes a latent Stage 11 bug
— the class-count probe ran on the CPU while the model sat on the GPU — covered by a
GPU-free regression test.

### Ongoing
Professional OOP and engineering practices, tests, documentation, performance, and
accessibility — maintained across every stage.

---

## Phase 2 — from a builder to a workbench · *in progress*

Phase 1 (Stages 0–12) built the core: draw a network, train it, export it. Phase 2
turns the builder into a workbench you can live in — work you can save, connections
that adapt by themselves and errors in plain words, today's **flagship architectures**
(Qwen, DeepSeek, Stable Diffusion…) and training that generates, a polished
interactive editor, a real **data workbench** (tables, parsers, a column mapper with
warnings), a look **inside** the trained model, and **your own layers** with your own
math. The rule stays the same: every stage is a self-contained increment — tested,
deployed to the live app, published on GitHub and turned into a narrated video on the
project channels.

The stages come in six blocks, ordered by dependency (Stages 14 and 17–20 were added
on 2026-11-25 and Stages 15–16 on 2026-11-26, at the user's request, before the rest —
the original later stages moved up by seven; the same day the flagships, 17–19, moved ahead
of clear errors, now 20):

- **A · Foundation** (13–16) — save your work, connections that adapt by themselves,
  honest accuracy on held-out data, repeatable runs with saved weights and a final test.
- **B · Flagship architectures** (17–19) — today's language models, the models behind
  images, 3D, sound and vision, and training that generates.
- **C · Clarity, look & feel** (20–23) — errors in plain words with one-click fixes, a new
  look, a truly interactive editor, a Russian interface.
- **D · The model in action** (24–25) — try the trained model on your own input and
  see what happens inside it.
- **E · Data workbench** (26–29) — datasets as tables, a universal column → input
  mapper with warnings, and parsers that build datasets from raw material (text for
  language models, exported price candles for time series).
- **F · Custom layers** (30–31) — your own activation formulas, gradients and weighted
  layers, saved under your own names.

| Stage | Block | In one line | Video |
|---|---|---|---|
| 13 | A | Projects: save, open, autosave — **done** | Close the tab, come back — it's all there |
| 14 | A | Smart connections: wires that adapt the data, with settings — **done** | Plug anything into anything |
| 15 | A | Honest accuracy: held-out data, training vs new-data curves, a verdict — **done** | Learning or memorising? |
| 16 | A | The same seed, the same numbers; save / load weights; a final test — **done** | Train, check, test — and keep it |
| 17 | B | Flagship language models: Qwen3, DeepSeek-V3, Llama, Mistral, Gemma — **done** | Qwen and DeepSeek from the inside |
| 18 | B | Flagships for images, 3D, sound, vision: Stable Diffusion, DiT, ViT, Whisper | What is inside Stable Diffusion |
| 19 | B | Teach it to generate: autoencoder, VAE, diffusion | A network draws digits |
| 20 | C | Clear errors: plain words, why, what to do, one-click fixes | Errors you understand without a degree |
| 21 | C | A new look: design system, themes, live training on the canvas | Before / after |
| 22 | C | Interactive editor: drag & drop, multi-select, undo / redo, mini-map | A ResNet in under a minute |
| 23 | C | Russian interface (EN / RU) | The app in Russian |
| 24 | D | Try the model: draw a digit, see the prediction | Draw a digit — it guesses |
| 25 | D | Inside the model: confusion matrix, activations, early stopping | What is inside a neural network |
| 26 | E | Datasets as tables: browse, statistics, edit, import | Fix your data like a spreadsheet |
| 27 | E | Column → input mapper with warnings; regression | Plug any table into a network |
| 28 | E | Text → dataset for language models; text generation | Feed it a book |
| 29 | E | Time series from exported candles (e.g. SOLUSDT) | A network on SOLUSDT candles |
| 30 | F | Your own activations: a formula editor | Invent an activation function |
| 31 | F | Your own gradients and weighted layers | A layer from scratch |

Cross-cutting rules for Phase 2: user formulas are read by a **safe math parser** and
are never executed as code; file parsers have size limits; project files carry a format
version, so older files keep opening; from Stage 23 on, every new string ships in
English and Russian.

### Stage 13 — Projects: save, open, autosave · **DONE (2026-11-25)**
Your work stops disappearing on a page refresh.
- A versioned project file (`.mlc.json`): the graph with node positions, training
  settings, the chosen dataset (including pasted custom rows) and metadata (name,
  description, dates).
- **Save** to a file, **open** from a file, **autosave** in the browser with restore
  after a reload, **recent projects**, new / save as / rename / duplicate.
- Optional **server-side projects** for a self-hosted instance (open the same project
  from another device).
- Format migrations, so files from older versions keep opening.
- *Done when:* save → open reproduces the graph exactly (tested), autosave survives a
  reload, and an older-format file opens.
- *Video:* build a network, close the tab, come back — everything is there; send the
  file to a friend.
- **Shipped:** the versioned `.mlc.json` format (format 1; a bare graph spec opens as
  format 0 and is upgraded). Migrations live in the backend — `POST /projects/normalize`
  opens any supported version and reports unknown layers, dropped broken edges and
  missing positions. The draft autosaves in the browser and comes back after a reload;
  **Save** (Ctrl+S) keeps projects in the browser (the list doubles as recent projects),
  with save as copy, rename / description and delete; **Download** a file and **open**
  one (or drop it onto the canvas); optional server-side projects
  (`GET/POST/PUT/DELETE /projects`, strict names, size and count limits,
  `MLC_SERVER_PROJECTS=0` to switch off, `MLC_PROJECTS_DIR` to move). A template from the
  library now starts a new project. Tested: save → open is exact for a CNN, a GPT with a
  nested Repeat block and pasted custom rows, and the reopened graphs build identically
  (6 backend tests + 16 front-end unit tests); checked live in the browser — the
  autosave survives a reload and an old-format file opens.

### Stage 14 — Smart connections · **DONE (2026-11-25)**
Connect any layer to any other — the connection adapts the data by itself.
- A connection is an object of its own: click a wire to select it, set it up in the
  inspector, delete it (a button or the Delete key).
- Every connection carries a **transform**. *Auto* (the default) picks the simplest
  change that lets the target take what the source sends — nothing at all when the
  shapes already fit: flatten an image for a dense layer, fold a vector into a C×H×W
  picture for a convolution, read an image row by row (or by columns, pixels, patches)
  as a sequence, keep the last step / the mean / the max of a sequence, match the
  inputs of Add and Concat.
- Where there are several sensible choices, a **selector** with parameters and presets
  (e.g. 2×8×8 or 8×4×4); options that do not fit are shown with the reason.
- The transform's label sits right on the wire; the shapes on nodes and wires update
  after every edit, without pressing Build; when the graph does not build yet, a gentle
  note says where (the node / wire lights up) and what to do.
- Transforms are plain torch modules, so training and the PyTorch export use exactly
  what the editor shows. Project files move to format 2 (connection settings); format-1
  files are upgraded on opening.
- *Done when:* the graph from the user's screenshot (an MLP with a Conv2d and an LSTM
  wired onto a dense layer) builds and trains with no manual fix, and every automatic
  transform is visible and can be changed.
- *Video:* plug anything into anything — the network adapts.
- **Shipped:** nine transforms (direct, flatten, pool each channel, sequence → vector,
  fold into an image, vector → sequence, read the image as a sequence, reshape, a
  learnable projection) with automatic resolution for single inputs and for Add /
  Concat; `POST /connections/options` for the inspector; a structured 422 (`error`:
  node, connection, hint); the Output takes class scores (a sequence or an image is
  flattened into it automatically); a layer with no input now says how to connect it.
  Tested: the screenshot graph builds by itself (→ 2×8×8, rows) and trains to accuracy
  1.000; auto picks the expected transform in 11 situations; 24 transform settings
  match their modules; the PyTorch export reproduces the connections exactly (identical
  outputs with the same weights); every library architecture is untouched; format-2
  projects keep the settings (11 new backend tests + 17 front-end unit tests); checked
  live in the browser.

### Stage 15 — Honest accuracy: held-out validation data · **DONE (2026-11-26)**
Is the network learning the rule, or just memorising its rows?
- Before training, a share of the rows (20 % by default; off / 10 / 20 / 30 / 50 % in the
  training panel's **Check on new data**) is **held out**: the network never trains on
  them. The split is deterministic for a seed and stratified (every class keeps its share
  in both parts); on tiny sets every class keeps a row to train on, and when nothing can
  be held out the result says so instead of failing.
- The result leads with the accuracy **on new data**, next to the accuracy on the
  training rows, both losses and the row counts; the accuracy curve shows both per epoch
  (live while training streams). A plain-words **verdict** — generalises, overfitting,
  not learned yet — and a warning instead when the held-out part is too small for one.
- The data window marks the held-out rows; the checkpoint keeps the weights that do best
  on new data; the PyTorch export splits exactly like the app (it embeds the same split
  function) and prints both accuracies.
- *Why:* on the separable synthetic sets the old full-set accuracy was always ~100 % — it
  measured memorising, not learning.
- *Done when:* on blobs both accuracies are high; on a set no network can fit (the same
  inputs with different labels) both errors stay above zero; `val_fraction = 0`
  reproduces the old loop exactly.
- *Video:* learning or memorising?
- **Shipped:** `train.val_fraction` (0–0.5, default 0.2) in the graph spec and in project
  files (no format change: older files get the default); `TrainResult` gains
  `val_acc_curve`, `val_loss_curve`, `final_val_accuracy`, `final_val_loss`,
  `train_samples` / `val_samples`, `fit` / `fit_note` and `warnings`; streamed epochs carry
  `val_acc` / `val_loss`; `/data/preview` marks each row's part; `/health` reports API 15.
  **Behaviour change:** training uses 80 % of the rows by default, so an epoch takes fewer
  steps, and `final_accuracy` / `acc_curve` / `final_loss` now describe the training rows.
  Tested on real runs: blobs — training 1.000, new data 1.000 (generalises); concentric
  shells — training 0.98, new data 0.75 (overfitting the old number hid); a 500-row set
  with a 20 % error floor — 0.80 / 0.80, the error never reaches 0; the split keeps its
  guarantees on 1000 random label sets; `val_fraction = 0` matches the pre-Stage-15 loop
  byte for byte; every library architecture still trains (9 new backend tests, the
  front-end project tests cover the new setting; the tests caught all 10 bugs planted on
  purpose in a copy of the code); checked live in the browser.

### Stage 16 — The same seed, the same numbers; saved weights; a final test · **DONE (2026-11-26)**
Train, check on new data, test once at the end — and keep what you trained.
- **Repeatable runs:** the starting weights now come from the run's seed too, so Train
  with the same settings gives the same numbers every time (before, the weights depended
  on whatever the server had done earlier). The global random state is left alone and
  CUDA is never touched.
- **A test part:** besides validation, a share of the rows (10 % by default; off / 10 /
  20 / 30 % in **Final test**) is kept for one final check — used neither for training
  nor for judging epochs. **Test** measures the weights on it once you are done tuning,
  next to their validation accuracy, and says whether it is a fair check: weights trained
  on another split (some test rows were training rows) or of unknown origin are flagged.
- **Saved weights:** every training run keeps its weights on the server; **Save weights**
  downloads them (`.pt`), **Load weights** brings a file back into the same graph — by
  name, or by position for files saved by the exported script — to test it or to train
  on from it. Files are read with `weights_only=True`: plain tensors only, nothing in a
  file ever runs. The exported `model.py` gains `load_weights()`, so the trained network
  runs anywhere.
- *Done when:* two runs with the same settings match number for number; Test reports the
  test rows honestly; a downloaded file loads back (and into `model.py`) and gives the
  same answers; a file that tries to run code is refused before it runs.
- *Video:* train, check, test — and keep the weights.
- **Shipped:** `build_model(spec, seed=…)` (the app passes the run's seed);
  `datasets.split_parts` (train / validation / test; with no test part it is exactly
  Stage 15's split); `train.test_fraction` (0–0.5, default 0.1; validation + test may hold
  out at most 70 %) and `train.init_run`; `TrainResult.test_samples` / `run_id`;
  `POST /test`, `GET /runs/{id}/weights`, `POST /weights/upload` (64 MB at most; the last
  30 runs are kept); a weights file records which data and split it was trained on;
  `/health` reports API 16. **Behaviour change:** by default training uses 70 % of the
  rows (20 % validation, 10 % test). Tested: the same seed gives the same weights, curves
  and test; the split keeps its guarantees on 400 random label sets; downloaded weights
  equal the trained ones and load back to the same test; the exported `model.py` loads
  them for an MLP, a CNN with BatchNorm, the GPT with a Repeat block and the screenshot
  graph; a resized layer is named, a renamed one matched by position, another network
  refused, a code-running file refused before it runs, an id is never a path (9 new
  backend tests; the tests caught all 14 bugs planted on purpose in a copy of the code;
  front-end project tests 17/17); checked live in the browser — train, test, save, load,
  train on from the loaded weights.

### Stage 17 — Flagship language models · **DONE (2026-11-26)**
The architectures behind today's chatbots, as editable graphs.
- New layers: RMSNorm, a SwiGLU feed-forward, rotary position embedding (RoPE),
  grouped-query attention (GQA) with a causal mask, a mixture of experts (MoE) with a
  router and shared experts, multi-head latent attention (MLA).
- Library entries: **Qwen3** (dense and MoE), **DeepSeek-V3 / R1**, **Llama 3**,
  **Mistral / Mixtral**, **Gemma 3**, **Phi-4** — each in two scales: a **mini** one that
  trains here, and the **real** one built from its published `config.json` (Hugging
  Face) on PyTorch's memory-free "meta" device, with the true parameter count and every
  shape; a link to the model card.
- *Done when:* each real-scale model's parameter count matches the official figure
  (within 1 %), and each mini model trains.
- *Video:* Qwen and DeepSeek from the inside.
- **Shipped:** six new layers — RMSNorm; attention with rotary positions and grouped
  key/value heads (GQA, with query/key norms, a sliding window, Gemma's post-norm, q/k/v
  bias); DeepSeek's multi-head latent attention (MLA); the gated MLP (SwiGLU / GeGLU); a
  mixture of experts (softmax or sigmoid routing with a balancing bias, shared experts);
  the vocabulary head, optionally tied to the embedding. Each attention and feed-forward
  block carries its pre-norm and residual, so a decoder is one Repeat of two nodes. Eight
  flagships in the library — Qwen3, Qwen3-MoE, DeepSeek-V3 / R1, Llama 3, Mistral, Mixtral,
  Gemma 3, Phi-4 — each a **mini** model (100–230 thousand parameters) that learns the new
  **Next token** task here in 15–45 s (about 0.9 on new data; Gemma 3 0.99), and its
  **real** size from the published config.json, built on the meta device: Qwen3-8B
  8,190,735,360, Qwen3-30B-A3B 30,532,122,624, Llama-3.1-8B 8,030,261,248, Mistral-7B-v0.3
  7,248,023,552, Mixtral-8x7B 46,702,792,704, Gemma-3-1B 999,885,952, Phi-4 14,659,507,200
  — the Hub's count to the parameter — and DeepSeek-V3 671,026,419,200 (the model card's
  671B; the Hub's 684.5B adds the multi-token-prediction module). A real-size model builds,
  previews and exports but never trains (more than 25 million parameters is refused with
  an explanation); `/build` now always sizes on the meta device. The export copies the LLM
  blocks verbatim. A clash in a layer's own settings (e.g. 4 query heads over 3 key/value
  groups) is now reported as such, not as a connection problem. Tested: 7 new backend tests
  (real sizes, minis learn, exact export, causal and sliding-window attention, tied heads,
  clear errors, inspect-only real sizes, the next-token data); the tests caught all 10 bugs
  planted on purpose in a copy of the code; checked live in the browser.

### Stage 18 — Flagships for images, 3D, sound and vision · *planned*
- New layers: patch embedding, Conv3d, up-sampling, a timestep embedding,
  cross-attention, adaptive normalization.
- Library entries: **ViT**, **CLIP**, **Stable Diffusion** (VAE + U-Net with
  cross-attention + text encoder), **DiT / FLUX-style** diffusion transformers, **3D
  generators** (triplane / point-cloud models in the spirit of TripoSR), **Whisper**
  (speech) — the same two scales (mini / real on "meta") and model-card links.
- *Done when:* each real-scale model matches its published parameter count, and each
  mini model builds and runs a forward pass.
- *Video:* what is inside Stable Diffusion.

### Stage 19 — Teach it to generate · *planned*
Training beyond classification.
- New training tasks: **reconstruction** (autoencoder, VAE) and **denoising diffusion**,
  with the right losses and samplers.
- Generate pictures right in the app: MNIST digits from a small VAE and a small
  diffusion model, with a gallery that fills in as training goes.
- *Done when:* a small diffusion model trained here draws recognizable digits, and a
  VAE reconstructs them.
- *Video:* a network draws digits.

### Stage 20 — Clear errors with fixes · *planned*
Every message says what happened, why, and what to do — in plain words.
- A catalogue of every error the editor can meet (shapes, wiring, parameters, data,
  training): each gets a short title in plain words, a "why" — with a small glossary
  (what C×H×W, a sequence, an embedding are) — and "what to do".
- **One-click fixes** next to the message: set a transform on the connection, change a
  parameter, insert a layer, reconnect.
- The culprit node or connection is highlighted and the view jumps to it; the checks
  run as you edit, before Build.
- Warnings for graphs that build but are probably wrong (a dropout of 0.9, a tiny
  hidden layer, an unused branch, a layer after the Output…).
- *Done when:* every error code has a plain-words explanation and at least one fix,
  covered by tests; a newcomer repairs every broken graph of a test set without
  reading code.
- *Video:* errors you understand without a degree.

### Stage 21 — A new look · *planned*
An editor that is attractive at first sight.
- A design system — palette, typography, spacing, elevation — with **dark and light
  themes** and a switch.
- Nodes colour-coded by family (input / output, convolution, dense, activation,
  normalization, sequence, attention, merge, block), with icons, a compact and a
  detailed view, and tensor-shape and parameter-count badges.
- Edges as smooth curves with an animated **data flow**, a hover highlight and the
  tensor shape shown right on the wire.
- **Live training on the canvas:** nodes glow as a batch passes through, a progress
  ring, and the final accuracy shown on the Output node.
- A cleaner inspector — grouped parameters, sliders and toggles, inline help — with
  smooth transitions and micro-interactions throughout.
- *Done when:* every screen uses the design system in both themes, and the training
  animation does not slow training down.
- *Video:* before / after.

### Stage 22 — An interactive editor · *planned*
Editing that feels like a real tool.
- **Drag & drop** layers from the palette onto the canvas; connect by **dragging from
  port to port**, with valid targets highlighted and shape mismatches flagged while you
  hover (reusing the Stage 5 smart adapters).
- **Multi-select** (box / Shift), move and delete groups; **copy / paste / duplicate**;
  **undo / redo**, built on the Stage 13 serialization.
- A **mini-map**, fit-to-screen, **auto-layout** on demand, snap-to-grid and alignment.
- A **context menu** (insert a layer, wrap in a Repeat block, disable, delete),
  **keyboard shortcuts** with a cheat sheet, and search on the canvas.
- *Done when:* every action can be undone, the shortcuts work, and editor end-to-end
  tests cover drag, connect, copy / paste and undo.
- *Video:* building a ResNet in under a minute with drag & drop and shortcuts.

### Stage 23 — Russian interface (EN / RU) · *planned*
- A translation layer for the UI with a language switch, remembered per browser.
- Every UI string, layer description and parameter hint in English and Russian; the
  backend's layer catalogue and error messages become translatable (stable error codes
  plus translated text).
- From this stage on, every new string ships in both languages.
- *Done when:* the whole app works in either language with no untranslated strings (a
  check fails the build on a missing key).
- *Video:* the app in Russian, for the Russian-speaking audience.

### Stage 24 — Try the model · *planned*
Use the network you have just trained.
- The backend already keeps every run's weights (Stage 16) and loads a saved weights
  file; `POST /predict` returns class probabilities for a run.
- A **Try it** panel: for images a **drawing canvas** (MNIST-style preprocessing —
  centring and scaling to 28×28) and image upload; for vectors and custom data, a value
  or row input.
- Probability bars for every class, the top 3 and the confidence; compare two models on
  the same input.
- *Done when:* a drawn digit is classified by the trained MNIST model in the live app,
  and the predictions match the backend's offline result.
- *Video:* draw a digit — the network guesses it.

### Stage 25 — Inside the model · *planned*
See what the network has learned — and where it struggles.
- **Early stopping** on the held-out loss, and per-class numbers on the held-out rows (the
  train / validation split and its curves arrived in Stage 15).
- A **confusion matrix** and per-class accuracy; a **gallery of mistakes** (what gets
  mistaken for what).
- **Activation maps** of the convolution layers for a chosen example, and the
  first-layer filters.
- **Weight and gradient histograms** per layer over time, with warnings for vanishing or
  exploding gradients.
- *Done when:* all views work for the library architectures on MNIST and CIFAR-10 and
  are covered by tests.
- *Video:* what is going on inside a neural network.

### Stage 26 — Datasets as tables · *planned*
A real view of the data instead of a few preview rows.
- One table model for every dataset — columns with types (number, category, text,
  image, time) and rows — built-in datasets included.
- Visual browsing: images as a thumbnail grid, text as text, time series as charts,
  numbers as a table.
- Column statistics: min / max / mean / standard deviation, missing values, unique
  values, histograms, class balance.
- **Editing:** cells, add / delete rows and columns, filter, sort, search.
- Import CSV / TSV / JSON (more formats later), export, and a preview of the train /
  validation / test split (the split itself arrived in Stages 15–16).
- *Done when:* a CSV can be imported, inspected, edited and saved with the project, and
  large tables page smoothly.
- *Video:* look at and fix your data like in a spreadsheet.

### Stage 27 — Column mapper & data warnings · *planned*
A universal editor that wires table columns to the model.
- Columns on one side, the model's inputs on the other, connected by dragging; one
  column is picked as the **target**. The task follows from the target:
  **classification** (categories) or **regression** (numbers — the trainer gains MSE /
  MAE losses).
- Transforms in place: scaling (min-max / z-score), category encoding (one-hot / label),
  missing-value filling, log transform.
- A **validator** with errors, warnings and hints — each with a one-click fix where
  possible:
  - a model input has no column (not selected);
  - a column is constant or all zeros (it carries no information);
  - one column feeds several inputs, or two columns are identical (duplicates);
  - the selected columns do not match the Input shape;
  - a non-numeric column without an encoding; missing values, NaN or infinity;
  - the target column is among the inputs, or an input looks suspiciously like the
    target (target leakage);
  - class imbalance or too few rows per class; wildly different feature scales;
  - for time series, a feature that peeks into the future.
- The mapping is saved with the project.
- *Done when:* every warning has a test that triggers it, and a mapped CSV trains end to
  end in both task types.
- *Video:* plug any table into a network — the editor warns you before you train.

### Stage 28 — Text → dataset for language models · *planned*
Feed raw material, get a dataset built for the model.
- Load material (txt / md; pdf and docx converted to text) or paste it.
- A parser that cleans the text, tokenizes it (by characters, or with a BPE vocabulary
  trained on the material itself — no external service), and cuts fixed-length context
  windows into "context → next token" training pairs.
- A preview: tokens in colour, vocabulary size, number of windows; warnings for too
  little text, a vocabulary too large for the model, or a window longer than the text.
- One click wires the dataset to the library GPT, with the vocabulary and output size
  adjusted automatically.
- **Generation:** the trained model continues your text (temperature, top-k).
- *Done when:* a small text trains a GPT in the live app and it writes in the
  material's style; parser tests cover tokenization and windowing.
- *Video:* feed it a book — the model writes in its style.

### Stage 29 — Time series: price candles · *planned*
Build a dataset from an exported timeframe, e.g. SOLUSDT candles.
- Import exported candles (CSV: time, open, high, low, close, volume); columns are
  detected and gaps in time are reported.
- A parser with lookback windows, configurable features (returns, log price, volume,
  SMA / EMA / RSI / volatility) and a target: the direction after H candles (up / down /
  flat by a threshold — classification) or the return (regression).
- A **strictly chronological split** (no shuffling) and warnings for look-ahead leakage,
  non-stationarity and class imbalance.
- A candlestick chart with the target and the windows marked and, after training, the
  predictions over the test period.
- Honest framing: a learning and research example — not a trading system and not
  financial advice; no orders, no wallets.
- *Done when:* an exported candle file goes from import to a trained model and a
  test-period chart, and the leakage checks are tested.
- *Video:* a neural network on SOLUSDT candles — and why it is harder than it looks.

### Stage 30 — Your own activations: a formula editor · *planned*
Create layers with your own math and keep them under your own names.
- **Create a layer from…**: take any layer as the parent (ReLU, GELU, Sigmoid, …) and
  open its formula.
- Write `f(x; a, b, …)` with coefficients that are fixed or **trainable**, with initial
  values and ranges. A **safe math parser** — an allow-list of operators and functions,
  never executed as code — compiles it to torch operations.
- A live plot of `f(x)` and its derivative `f'(x)` (via autograd), coefficient sliders
  that redraw instantly, and checks for NaN / infinity, a gradient that vanishes over a
  wide range (a "dead" function) and explosive growth.
- Save to **My layers** with a name, a description and a family: the layer appears in
  the palette and behaves like any other (shapes, parameters; the PyTorch export
  generates a class with your formula).
- Stored with the project and in a personal layer library.
- *Done when:* a custom activation trains a network in the live app and exports to
  standalone PyTorch, and the parser rejects anything outside the allow-list (tested).
- *Video:* invent an activation function and train a network with it.

### Stage 31 — Your own gradients & weighted layers · *planned*
- A **custom backward**: write your own derivative (a surrogate gradient) for step,
  threshold, quantization or spiking functions, and compare it with the automatic one.
- **Layers with weights**: a formula over the input and trainable tensors (W, b, …)
  with inferred shapes — e.g. a polynomial layer, a gate, your own normalization.
- **Save a sub-graph as a layer** — a composite custom layer built on the Repeat / Block
  machinery.
- Sharing: export / import a layer library as a file, with version checks.
- *Done when:* a surrogate-gradient layer and a weighted custom layer train end to end
  and export to PyTorch, and a saved sub-graph reloads as a single layer.
- *Video:* a layer from scratch — formula, weights and gradient.

---

## Как это работает (кратко, по-русски)

ML Constructor — визуальная IDE для сборки нейросетей, «ComfyUI для нейросетей»:
перетаскиваешь блоки-слои на холст, соединяешь в архитектуру, жмёшь **Train** —
Python-бэкенд на **PyTorch** собирает граф в настоящий `nn.Module`, обучает его и
возвращает результат на страницу, без единой строки кода. Самостоятельный
опенсорс-проект (PolyForm Noncommercial), поднимается по README.

### Этапы развития
Разработка идёт **этапами**; каждый завершённый этап — самостоятельный демо-инкремент:
документируется, тестируется, деплоится в живое приложение, публикуется на GitHub и
превращается в озвученное видео на каналах проекта.

* **Этап 0 — v1, сквозной срез · ГОТОВО (15.11):** редактор → PyTorch → обучение →
  результат; 9 слоёв, одна цепочка, вывод форм, accuracy 1.0 в живом демо.
* **Этап 1 — Ветвление и слияния (DAG) · ГОТОВО (17.11):** от цепочки к направленному
  графу; ноды-слияния `Add`/`Concat`, ветки, skip/residual-связи, параллельные пути.
* **Этап 2 — Богаче каталог слоёв · ГОТОВО (17.11):** Dropout, пулинг, доп. нормализации
  и активации, `ConvTranspose2d`.
* **Этап 3 — Последовательности · ГОТОВО (17.11):** `Embedding`, `RNN`/`LSTM`/`GRU` на
  соглашении `(L, F)` (`batch_first`); выход — последний шаг `(hidden,)` или вся
  последовательность `(L, hidden)`.
* **Этап 4 — Внимание и трансформеры · ГОТОВО (17.11):** `MultiheadAttention`
  (self-attention), `TransformerEncoderLayer`, синусоидальное `PositionalEncoding`
  — всё сохраняет форму `(L, E)`; собирается и обучается небольшой трансформер.
* **Этап 5 — Умные адаптеры · ГОТОВО (17.11):** согласование форм на стыках — бэкенд
  `/suggest` предлагает ноду-адаптер (первый — Flatten), редактор показывает подсказку
  и вставляет её в один клик; каркас открыт для reshape/пулинг/dtype-адаптеров.
* **Этап 6 — Блоки и повторы · ГОТОВО (17.11):** нода `Repeat` держит внутреннюю
  линейную цепочку и штампует её **N×** (переиспользует материнский класс `Block`),
  ведёт себя как обычный слой; в редакторе — нода Block с редактором внутренней
  цепочки (добавить / убрать / править слои, задать N).
* **Этап 7 — Библиотека архитектур · ГОТОВО (18.11):** готовые **MLP**, **CNN**,
  **ResNet** (skip-связи — настоящие ноды-слияния `Add`) и **мини-GPT** (эмбеддинг
  + позиционное кодирование + стопка трансформер-блоков через `Repeat` из Этапа 6),
  отдаются по `GET /architectures`; выбираешь — и архитектура ложится на холст
  обычными **редактируемыми нодами**: каждый слой, каждый повтор, каждый skip можно
  перенастроить. Каждая собирается и обучается от начала до конца.
* **Этап 8 — Датасеты, превью данных и экспорт · ГОТОВО (18.11):** увидеть данные и
  забрать модель с собой. Слой датасетов кормит и тренер, и новое **превью данных**
  (`/data/preview`): модалка с точным форматом (форма входа, dtype, число классов и
  примеров) и несколькими **настоящими строками** выборки, на которых учится сеть.
  **Выбор датасета** (`blobs` — разделимые кластеры; `shells` — нелинейная радиальная
  граница), результат рисует **кривую точности** по эпохам рядом с loss, и в один клик —
  **экспорт автономного PyTorch** (`/export/pytorch`): настоящий `nn.Module`, собирающий
  весь граф (ветки, слияния `Add`/`Concat`, блоки `Repeat`), плюс цикл обучения — без
  зависимостей, кроме torch. Проверено сквозными тестами: каждая архитектура
  экспортируется в код, который собирается в ту же модель.
* **Этап 9 — Свои данные и экспорт модели · ГОТОВО (19.11):** обучай на своих данных и
  забирай обученную модель. **Кастомный датасет**: вставь или загрузи строки (в каждой —
  признаки, затем метка класса); бэкенд парсит и проверяет их по графу (понятные ошибки при
  неверной форме/метке), модалка **Data** показывает ровно то, что ты дал, и обучение идёт
  именно на этих строках. Экспорт получил **новые форматы**: сгенерированный скрипт без
  зависимостей теперь ещё и сохраняет веса (`torch.save` → `model.pt`) и экспортирует **ONNX**
  (`model.onnx`, best-effort). Проверено сквозными тестами: свои строки обучаются до acc 1.000,
  плохие строки отклоняются, экспортированный файл пишет настоящий `state_dict` и ONNX-граф.
* **Этап 10 — Встроенные реальные датасеты (MNIST / CIFAR-10) · ГОТОВО (19.11):** реальные
  датасеты за тем же контрактом, авто-скачиваемые с официальных источников (без torchvision) и
  кэшируемые локально: **MNIST** (цифры 28×28) и **CIFAR-10** (цвет 32×32), парсинг стандартной
  библиотекой, проверка формы/классов по графу (понятное сообщение, если Input не совпал),
  превью как у любого датасета и обучение на воспроизводимом подмножестве. Проверено сквозь:
  MNIST ~0.96 на небольшом CPU-прогоне; CIFAR-10 скачивается, парсится и обучается (loss падает).
  Пикер показывает их и подсказывает нужную форму Input.
* **Этап 11 — Живое обучение: потоковые кривые и чекпоинты · ГОТОВО (19.11):** обучение стримит
  прогресс вживую — `POST /train/stream` шлёт NDJSON (`start` → по-шагово `step` → по-эпохно
  `epoch` → `done`), редактор рисует кривые loss/accuracy по мере поступления. **Чекпоинты**
  сохраняют лучшие по точности веса (ключ — по архитектуре) и **возобновляют** из них в следующем
  запуске — проверено сквозь (возобновление стартует намного ниже loss свежего запуска). Тренер
  **device-aware** (берёт CUDA автоматически, если стоит CUDA-сборка PyTorch); встроенный бэкенд
  CPU-only, поэтому реальный прогон на GPU — Этап 12.
* **Этап 12 — Обучение на GPU (явный opt-in) · ГОТОВО (19.11):** обучение может идти на
  видеокарте: бэкенд на CUDA-сборке PyTorch, а тренер принимает `device` — `cpu` (по
  умолчанию, чтобы общий сервер сам не занимал карту), `cuda` или `auto`; `cuda` без
  CUDA-устройства — понятная ошибка, после GPU-прогона веса и данные отдаются карте обратно.
  `GET /health` сообщает `cuda_available` (не открывая CUDA-контекст), редактор показывает
  выбор **Device** только при наличии GPU. Проверено на RTX 3060 Laptop GPU: небольшая CNN на
  MNIST учится **в ~9.7 раза быстрее**, чем на CPU (0.63 с против 6.15 с за 3 эпохи), живой
  стрим на CUDA работает. Заодно исправлен скрытый баг Этапа 11 — проба числа классов шла на
  CPU, когда модель уже на GPU, — с регрессионным тестом без GPU.
* **Постоянно:** высокий уровень ООП, тесты, документация, производительность.

### Фаза 2 — от конструктора к мастерской (идёт)

Фаза 1 (Этапы 0–12) построила ядро: нарисовать сеть, обучить, экспортировать. Фаза 2
превращает конструктор в мастерскую, в которой удобно жить: сохранение работы,
соединения, которые подстраиваются сами, и ошибки простыми словами, **флагманские
архитектуры** наших дней (Qwen, DeepSeek, Stable Diffusion…) и обучение генерации,
красивый интерактивный редактор, полноценный **цех данных** (таблицы, парсеры, редактор
«колонки → входы» с предупреждениями), взгляд **внутрь** обученной модели и **свои слои**
со своей математикой. Правило прежнее: каждый этап — самостоятельный инкремент: тесты,
деплой в живое приложение, GitHub и озвученный ролик в каналах проекта.

Шесть блоков, по порядку зависимостей (Этапы 14 и 17–20 добавлены 25.11, Этапы 15–16 —
26.11, всё по просьбе пользователя, и идут раньше остальных — прежние последующие этапы
сдвинулись на семь; в тот же день флагманы, 17–19, встали раньше понятных ошибок, теперь 20):

- **A · Фундамент** (13–16) — сохранение работы, соединения, которые подстраиваются сами,
  честная точность на отложенных данных, повторяемые запуски с сохранением весов и
  финальным тестом.
- **B · Флагманские архитектуры** (17–19) — языковые модели наших дней, модели для
  картинок, 3D, звука и зрения и обучение, которое генерирует.
- **C · Понятность, облик и удобство** (20–23) — ошибки простыми словами с исправлением в
  один клик, новый облик, по-настоящему интерактивный редактор, русский интерфейс.
- **D · Модель в деле** (24–25) — проверить обученную сеть на своём примере и заглянуть
  внутрь.
- **E · Цех данных** (26–29) — датасеты как таблицы, универсальный редактор «колонки →
  входы» с предупреждениями и парсеры, которые собирают датасет из сырого материала
  (текст для языковых моделей, выгруженные свечи для временных рядов).
- **F · Свои слои** (30–31) — свои формулы активации, градиенты и слои с весами — под
  своими названиями.

| Этап | Блок | Коротко | Ролик |
|---|---|---|---|
| 13 | A | Проекты: сохранить, открыть, автосохранение — **готово** | Закрыл вкладку, вернулся — всё на месте |
| 14 | A | Умные соединения: связи, которые сами подгоняют данные, с настройками — **готово** | Соединяй что угодно с чем угодно |
| 15 | A | Честная точность: отложенная выборка, кривые «обучение / новые данные», вердикт — **готово** | Учится или зубрит? |
| 16 | A | Один seed — одни цифры; сохранить / загрузить веса; финальный тест — **готово** | Обучили, проверили, протестировали — и сохранили |
| 17 | B | Флагманские языковые модели: Qwen3, DeepSeek-V3, Llama, Mistral, Gemma — **готово** | Qwen и DeepSeek изнутри |
| 18 | B | Флагманы для картинок, 3D, звука и зрения: Stable Diffusion, DiT, ViT, Whisper | Что внутри у Stable Diffusion |
| 19 | B | Учим генерировать: автоэнкодер, VAE, диффузия | Нейросеть рисует цифры |
| 20 | C | Понятные ошибки: простыми словами, почему, что делать, исправление в один клик | Ошибки, понятные без диплома |
| 21 | C | Новый облик: дизайн-система, темы, живое обучение на холсте | Было / стало |
| 22 | C | Интерактивный редактор: перетаскивание, мультивыделение, отмена, мини-карта | ResNet меньше чем за минуту |
| 23 | C | Русский интерфейс (EN / RU) | Приложение на русском |
| 24 | D | Проверка модели: нарисуй цифру — увидишь предсказание | Нарисовал цифру — сеть угадала |
| 25 | D | Внутри модели: матрица ошибок, активации, ранняя остановка | Что внутри у нейросети |
| 26 | E | Датасеты как таблицы: просмотр, статистика, правка, импорт | Правим данные, как в таблице |
| 27 | E | Редактор «колонки → входы» с предупреждениями; регрессия | Подключаем любую таблицу к сети |
| 28 | E | Текст → датасет для языковых моделей; генерация текста | Скормили книгу |
| 29 | E | Временные ряды из выгруженных свечей (например, SOLUSDT) | Нейросеть на свечах SOLUSDT |
| 30 | F | Свои функции активации: редактор формул | Придумали свою активацию |
| 31 | F | Свои градиенты и слои с весами | Слой с нуля |

Сквозные правила фазы 2: пользовательские формулы разбирает **безопасный математический
парсер** — они никогда не исполняются как код; у парсеров файлов есть лимиты размера; у
файлов проектов есть версия формата, поэтому старые файлы продолжают открываться; с
Этапа 23 каждая новая строка интерфейса — сразу на английском и русском.

#### Этап 13 — Проекты: сохранить, открыть, автосохранение · ГОТОВО (25.11)
Работа больше не пропадает при обновлении страницы.
- Файл проекта с версией формата (`.mlc.json`): граф с позициями нод, настройки обучения,
  выбранный датасет (включая вставленные свои строки) и метаданные (название, описание,
  даты).
- **Сохранить** в файл, **открыть** из файла, **автосохранение** в браузере с
  восстановлением после перезагрузки, **недавние проекты**, новый / сохранить как /
  переименовать / дублировать.
- По желанию — **проекты на сервере** для своей установки (открыть тот же проект с
  другого устройства).
- Миграции формата: файлы старых версий продолжают открываться.
- *Готово, когда:* «сохранить → открыть» воспроизводит граф один в один (проверено
  тестами), автосохранение переживает перезагрузку, файл старого формата открывается.
- *Ролик:* собрал сеть, закрыл вкладку, вернулся — всё на месте; отправил файл другу.
- **Сделано:** формат `.mlc.json` с версией (формат 1; «голая» спецификация графа
  открывается как формат 0 и обновляется). Миграции живут в бэкенде —
  `POST /projects/normalize` открывает любую поддерживаемую версию и сообщает о
  неизвестных слоях, удалённых битых связях и нодах без позиций. Черновик автоматически
  сохраняется в браузере и возвращается после перезагрузки; **Save** (Ctrl+S) хранит
  проекты в браузере (этот список и есть недавние проекты), есть «сохранить копию»,
  переименование / описание и удаление; **скачать** файл и **открыть** его (или бросить
  на холст); по желанию — проекты на сервере (`GET/POST/PUT/DELETE /projects`, строгие
  имена, лимиты размера и количества, `MLC_SERVER_PROJECTS=0` выключает,
  `MLC_PROJECTS_DIR` переносит). Шаблон из библиотеки теперь начинает новый проект.
  Проверено: «сохранить → открыть» точно для CNN, GPT со вложенным блоком Repeat и
  вставленных своих строк, открытые графы собираются в те же модели (6 тестов бэкенда +
  16 юнит-тестов фронта); вживую в браузере — автосохранение переживает перезагрузку,
  файл старого формата открывается.

#### Этап 14 — Умные соединения · ГОТОВО (25.11)
Соединяй любой слой с любым — соединение само подгоняет данные.
- Соединение — самостоятельный объект: кликни по линии, чтобы выбрать её, настрой в
  инспекторе, удали (кнопкой или клавишей Delete).
- На каждом соединении есть **преобразование**. *Авто* (по умолчанию) подбирает
  простейшую замену, при которой слой-получатель примет то, что отправляет источник, —
  ничего не делает, если формы уже совпадают: развернуть картинку в вектор для полносвязного
  слоя, свернуть вектор в картинку C×H×W для свёртки, прочитать картинку построчно (или по
  столбцам, пикселям, патчам) как последовательность, взять у последовательности последний
  шаг / среднее / максимум, подогнать входы Add и Concat.
- Если разумных вариантов несколько — **селектор** с параметрами и пресетами (например,
  2×8×8 или 8×4×4); неподходящие варианты видны вместе с причиной.
- Подпись преобразования — прямо на линии; формы на нодах и линиях пересчитываются после
  каждой правки, без кнопки Build; если граф пока не собирается, мягкая подсказка говорит,
  где (нода / линия подсвечивается) и что сделать.
- Преобразования — обычные модули torch, поэтому обучение и экспорт в PyTorch используют
  ровно то, что показывает редактор. Файлы проектов переходят на формат 2 (настройки
  соединений); файлы формата 1 обновляются при открытии.
- *Готово, когда:* граф со скриншота пользователя (MLP, к которому подключены Conv2d и
  LSTM) собирается и обучается без единой ручной правки, а каждое автоматическое
  преобразование видно и меняется.
- *Ролик:* соединяй что угодно с чем угодно — сеть подстроится.
- **Сделано:** девять преобразований (без изменений, развернуть в вектор, свести каждый
  канал к числу, последовательность → вектор, свернуть в картинку, вектор →
  последовательность, прочитать картинку как последовательность, изменить форму,
  обучаемая проекция) с автоматическим выбором для обычных входов и для Add / Concat;
  `POST /connections/options` для инспектора; структурированный ответ 422 (`error`: нода,
  соединение, подсказка); Output принимает оценки классов (последовательность или картинка
  разворачиваются перед ним автоматически); слой без входа теперь подсказывает, как его
  подключить. Проверено: граф со скриншота собирается сам (→ 2×8×8, построчно) и обучается
  до точности 1.000; «авто» выбирает ожидаемое преобразование в 11 ситуациях; 24 настройки
  преобразований совпадают со своими модулями; экспорт в PyTorch воспроизводит соединения
  в точности (одинаковые выходы при одинаковых весах); все архитектуры библиотеки не
  затронуты; проекты формата 2 хранят настройки (11 новых тестов бэкенда + 17 юнит-тестов
  фронта); вживую в браузере.

#### Этап 15 — Честная точность: отложенная выборка · ГОТОВО (26.11)
Сеть выучила правило или просто зазубрила свои строки?
- Перед обучением часть строк (по умолчанию 20 %; в панели обучения **Check on new data**:
  выкл. / 10 / 20 / 30 / 50 %) **откладывается**: на них сеть никогда не учится. Разбиение
  одинаково при одном seed и стратифицировано (у каждого класса своя доля в обеих частях);
  на крошечных наборах у каждого класса остаётся строка для обучения, а если отложить
  нечего, результат прямо об этом говорит, а не падает.
- Результат начинается с точности **на новых данных**, рядом — точность на обучающих
  строках, обе потери и число строк; кривая точности показывает обе по эпохам (вживую, пока
  идёт обучение). **Вердикт** простыми словами — обобщает, переобучение, ещё не выучила, —
  а если отложенных строк слишком мало для вердикта, вместо него предупреждение.
- Окно данных помечает отложенные строки; чекпоинт хранит веса, лучшие на новых данных;
  экспорт в PyTorch делит строки ровно как приложение (в него встроена та же функция
  разбиения) и печатает обе точности.
- *Зачем:* на разделимых синтетических наборах прежняя точность «на всём наборе» всегда была
  ~100 % — она мерила зубрёжку, а не обучение.
- *Готово, когда:* на blobs обе точности высокие; на наборе, который не подогнать никакой
  сетью (одинаковые входы с разными метками), обе ошибки остаются выше нуля;
  `val_fraction = 0` в точности повторяет прежний цикл.
- *Ролик:* учится или зубрит?
- **Сделано:** `train.val_fraction` (0–0.5, по умолчанию 0.2) в спецификации графа и в
  файлах проектов (формат не меняется: старые файлы получают значение по умолчанию); в
  `TrainResult` появились `val_acc_curve`, `val_loss_curve`, `final_val_accuracy`,
  `final_val_loss`, `train_samples` / `val_samples`, `fit` / `fit_note` и `warnings`;
  события эпох в потоке несут `val_acc` / `val_loss`; `/data/preview` помечает часть каждой
  строки; `/health` сообщает API 15. **Изменение поведения:** по умолчанию обучение идёт на
  80 % строк, поэтому в эпохе меньше шагов, а `final_accuracy` / `acc_curve` / `final_loss`
  теперь относятся к обучающим строкам. Проверено настоящими прогонами: blobs — обучение
  1.000, новые данные 1.000 (обобщает); концентрические оболочки — обучение 0.98, новые
  данные 0.75 (переобучение, которое прятала прежняя цифра); набор из 500 строк с полом
  ошибки 20 % — 0.80 / 0.80, ошибка не доходит до нуля; разбиение держит гарантии на 1000
  случайных наборов меток; `val_fraction = 0` совпадает с прежним циклом байт в байт; все
  архитектуры библиотеки обучаются (9 новых тестов бэкенда, тесты проектов во фронте
  покрывают новую настройку; тесты поймали все 10 ошибок, нарочно внесённых в копию кода);
  вживую в браузере.

#### Этап 16 — Один seed — одни цифры; сохранённые веса; финальный тест · ГОТОВО (26.11)
Обучили, проверили на новых данных, один раз протестировали в конце — и сохранили результат.
- **Повторяемые запуски:** начальные веса теперь тоже берутся из seed запуска, поэтому Train
  с теми же настройками каждый раз даёт те же цифры (раньше веса зависели от того, что
  сервер делал до этого). Общее состояние генератора не меняется, CUDA не задействуется.
- **Тестовая часть:** кроме валидации, часть строк (по умолчанию 10 %; в **Final test**:
  выкл. / 10 / 20 / 30 %) откладывается для одной финальной проверки — не для обучения и не
  для оценки эпох. **Test** меряет на ней веса, когда настройка закончена, рядом с их
  точностью на валидации, и говорит, честная ли это проверка: веса, обученные на другом
  разбиении (часть тестовых строк была в обучении) или неизвестного происхождения,
  помечаются.
- **Сохранение весов:** каждый запуск хранит свои веса на сервере; **Save weights** скачивает
  их (`.pt`), **Load weights** возвращает файл в тот же граф — по именам или, для файлов из
  экспортированного скрипта, по позиции, — чтобы протестировать или продолжить обучение.
  Файлы читаются с `weights_only=True`: только тензоры, ничего из файла не исполняется. В
  экспортированном `model.py` появилась `load_weights()`, так что обученная сеть работает
  где угодно.
- *Готово, когда:* два запуска с одинаковыми настройками совпадают цифра в цифру; Test честно
  сообщает о тестовых строках; скачанный файл загружается обратно (и в `model.py`) и даёт те
  же ответы; файл, пытающийся выполнить код, отклоняется до выполнения.
- *Ролик:* обучили, проверили, протестировали — и сохранили веса.
- **Сделано:** `build_model(spec, seed=…)` (приложение передаёт seed запуска);
  `datasets.split_parts` (обучение / валидация / тест; без тестовой части — ровно разбиение
  Этапа 15); `train.test_fraction` (0–0.5, по умолчанию 0.1; валидация + тест — не больше
  70 % строк) и `train.init_run`; `TrainResult.test_samples` / `run_id`; `POST /test`,
  `GET /runs/{id}/weights`, `POST /weights/upload` (до 64 МБ; хранятся последние 30
  запусков); файл весов помнит, на каких данных и разбиении обучен; `/health` сообщает
  API 16. **Изменение поведения:** по умолчанию обучение идёт на 70 % строк (20 % —
  валидация, 10 % — тест). Проверено: одинаковый seed даёт одинаковые веса, кривые и тест;
  разбиение держит гарантии на 400 случайных наборах меток; скачанные веса равны обученным
  и после загрузки дают тот же тест; экспортированный `model.py` загружает их для MLP, CNN с
  BatchNorm, GPT с блоком Repeat и графа со скриншота; изменённый размер слоя называется по
  имени, переименованный слой сопоставляется по позиции, чужая сеть отклоняется, файл с
  попыткой выполнить код отклоняется до выполнения, id никогда не бывает путём (9 новых
  тестов бэкенда; тесты поймали все 14 ошибок, нарочно внесённых в копию кода; тесты
  проектов во фронте 17/17); вживую в браузере — обучить, протестировать, сохранить,
  загрузить, продолжить обучение с загруженных весов.

#### Этап 17 — Флагманские языковые модели · ГОТОВО (26.11)
Архитектуры, на которых работают современные чат-боты, — как редактируемые графы.
- Новые слои: RMSNorm, SwiGLU-блок, вращающиеся позиционные эмбеддинги (RoPE),
  внимание с группами запросов (GQA) и казуальной маской, смесь экспертов (MoE) с
  роутером и общими экспертами, латентное внимание (MLA).
- В библиотеке: **Qwen3** (плотная и MoE), **DeepSeek-V3 / R1**, **Llama 3**,
  **Mistral / Mixtral**, **Gemma 3**, **Phi-4** — каждая в двух масштабах: **мини**, которая
  обучается прямо здесь, и **настоящая**, собранная по опубликованному `config.json`
  (Hugging Face) на «meta»-устройстве PyTorch без расхода памяти — с настоящим числом
  параметров и всеми формами; ссылка на карточку модели.
- *Готово, когда:* число параметров у каждой модели настоящего масштаба совпадает с
  официальным (в пределах 1 %), а каждая мини-модель обучается.
- *Ролик:* Qwen и DeepSeek изнутри.
- **Сделано:** шесть новых слоёв — RMSNorm; внимание с поворотными позициями и группами
  ключей/значений (GQA: нормы запросов и ключей, скользящее окно, пост-норма Gemma, смещения
  q/k/v); латентное внимание DeepSeek (MLA); вентильный MLP (SwiGLU / GeGLU); смесь экспертов
  (маршрутизация softmax или sigmoid с балансирующим смещением, общие эксперты); выходной
  слой словаря, по желанию связанный с эмбеддингом. Блоки внимания и прямого прохода несут
  свою пре-норму и остаточную связь, поэтому декодер — один Repeat из двух узлов. Восемь
  флагманов в библиотеке — Qwen3, Qwen3-MoE, DeepSeek-V3 / R1, Llama 3, Mistral, Mixtral,
  Gemma 3, Phi-4: у каждого **мини**-модель (100–230 тысяч параметров), которая учит новую
  задачу **Next token** прямо здесь за 15–45 с (около 0.9 на новых данных; Gemma 3 — 0.99), и
  **настоящий** размер по опубликованному config.json, собранный на meta-устройстве:
  Qwen3-8B 8 190 735 360, Qwen3-30B-A3B 30 532 122 624, Llama-3.1-8B 8 030 261 248,
  Mistral-7B-v0.3 7 248 023 552, Mixtral-8x7B 46 702 792 704, Gemma-3-1B 999 885 952,
  Phi-4 14 659 507 200 — до параметра как на Hub, — и DeepSeek-V3 671 026 419 200 (671 млрд
  по карточке модели; 684,5 млрд на Hub включают модуль предсказания нескольких токенов).
  Модель настоящего размера собирается, показывает данные и экспортируется, но не обучается
  (больше 25 млн параметров — отказ с объяснением); `/build` теперь всегда считает на
  meta-устройстве. Экспорт копирует блоки LLM дословно. Противоречие в собственных
  настройках слоя (например, 4 головы запросов на 3 группы ключей/значений) теперь так и
  называется, а не выдаётся за проблему соединения. Проверено: 7 новых тестов бэкенда
  (настоящие размеры, мини учатся, точный экспорт, причинное внимание и скользящее окно,
  связанный выходной слой, понятные ошибки, настоящий размер — только просмотр, данные
  «следующий токен»); тесты поймали все 10 ошибок, нарочно внесённых в копию кода; вживую в
  браузере.

#### Этап 18 — Флагманы для картинок, 3D, звука и зрения (в планах)
- Новые слои: нарезка на патчи, Conv3d, повышение разрешения, эмбеддинг шага диффузии,
  перекрёстное внимание, адаптивная нормализация.
- В библиотеке: **ViT**, **CLIP**, **Stable Diffusion** (VAE + U-Net с перекрёстным
  вниманием + текстовый энкодер), диффузионные трансформеры **DiT / в духе FLUX**,
  **3D-генераторы** (модели на трёх плоскостях / облаках точек в духе TripoSR),
  **Whisper** (речь) — в тех же двух масштабах (мини / настоящий на «meta») и со ссылками
  на карточки моделей.
- *Готово, когда:* у каждой модели настоящего масштаба число параметров совпадает с
  опубликованным, а каждая мини-модель собирается и выполняет прямой проход.
- *Ролик:* что внутри у Stable Diffusion.

#### Этап 19 — Учим генерировать (в планах)
Обучение не только классификации.
- Новые задачи обучения: **восстановление** (автоэнкодер, VAE) и **диффузионное
  шумоподавление** — с правильными функциями потерь и сэмплерами.
- Картинки генерируются прямо в приложении: цифры MNIST от маленького VAE и маленькой
  диффузионной модели, галерея пополняется по ходу обучения.
- *Готово, когда:* маленькая диффузионная модель, обученная здесь, рисует узнаваемые
  цифры, а VAE их восстанавливает.
- *Ролик:* нейросеть рисует цифры.

#### Этап 20 — Понятные ошибки с исправлениями (в планах)
Каждое сообщение говорит, что случилось, почему и что сделать — простыми словами.
- Каталог всех ошибок, которые встречаются в редакторе (формы, связи, параметры, данные,
  обучение): у каждой — короткий заголовок простыми словами, «почему» — с маленьким
  словарём (что такое C×H×W, последовательность, эмбеддинг) — и «что сделать».
- **Исправление в один клик** рядом с сообщением: поставить преобразование на соединение,
  поменять параметр, вставить слой, переподключить.
- Виноватая нода или соединение подсвечивается, вид переходит к ней; проверки идут прямо во
  время правки, до Build.
- Предупреждения для графов, которые собираются, но скорее всего ошибочны (dropout 0.9,
  крошечный скрытый слой, неиспользуемая ветка, слой после Output…).
- *Готово, когда:* у каждого кода ошибки есть объяснение простыми словами и хотя бы одно
  исправление, всё покрыто тестами; новичок чинит каждый сломанный граф из тестового
  набора, не читая код.
- *Ролик:* ошибки, понятные без диплома.

#### Этап 21 — Новый облик
Редактор, который цепляет с первого взгляда.
- Дизайн-система — палитра, типографика, отступы, тени — с **тёмной и светлой темами** и
  переключателем.
- Ноды раскрашены по семействам (вход / выход, свёртки, плотные, активации, нормализация,
  последовательности, внимание, слияния, блоки), с иконками, компактным и подробным видом,
  бейджами формы тензора и числа параметров.
- Рёбра — плавные кривые с анимацией **потока данных**, подсветкой при наведении и формой
  тензора прямо на связи.
- **Живое обучение на холсте:** ноды светятся, когда через них проходит батч, кольцо
  прогресса, итоговая точность на ноде Output.
- Аккуратный инспектор — группы параметров, слайдеры и переключатели, подсказки на месте —
  и плавные переходы с микроанимациями повсюду.
- *Готово, когда:* все экраны на дизайн-системе в обеих темах, а анимация обучения не
  замедляет само обучение.
- *Ролик:* было / стало.

#### Этап 22 — Интерактивный редактор
Редактирование, как в настоящем инструменте.
- **Перетаскивание** слоёв из палитры на холст; соединение **протягиванием от порта к
  порту** — допустимые цели подсвечиваются, несовпадение форм видно ещё при наведении (на
  базе умных адаптеров Этапа 5).
- **Мультивыделение** (рамкой / Shift), перенос и удаление групп; **копировать / вставить /
  дублировать**; **отменить / повторить** — на сериализации Этапа 13.
- **Мини-карта**, «вписать в экран», **авто-раскладка** по кнопке, привязка к сетке и
  выравнивание.
- **Контекстное меню** (вставить слой, обернуть в блок Repeat, отключить, удалить),
  **горячие клавиши** со шпаргалкой, поиск по холсту.
- *Готово, когда:* любое действие отменяется, горячие клавиши работают, а сквозные тесты
  редактора покрывают перетаскивание, соединение, копирование и отмену.
- *Ролик:* собираем ResNet меньше чем за минуту мышкой и горячими клавишами.

#### Этап 23 — Русский интерфейс (EN / RU)
- Слой переводов интерфейса с переключателем языка (запоминается в браузере).
- Все строки интерфейса, описания слоёв и подсказки параметров — на английском и русском;
  каталог слоёв и сообщения об ошибках бэкенда становятся переводимыми (стабильные коды
  ошибок плюс переведённый текст).
- С этого этапа каждая новая строка — сразу на двух языках.
- *Готово, когда:* всё приложение работает на любом из языков без непереведённых строк
  (проверка роняет сборку при пропущенном ключе).
- *Ролик:* приложение на русском — для русскоязычной аудитории.

#### Этап 24 — Проверка модели
Применить только что обученную сеть.
- Бэкенд уже хранит веса каждого запуска (Этап 16) и загружает сохранённый файл весов;
  `POST /predict` возвращает вероятности классов для запуска.
- Панель **«Проверить»**: для картинок — **холст для рисования** (предобработка как у
  MNIST: центрирование и масштаб до 28×28) и загрузка изображения; для векторов и своих
  данных — ввод значений или строки.
- Столбики вероятностей по всем классам, топ-3 и уверенность; сравнение двух моделей на
  одном примере.
- *Готово, когда:* нарисованную цифру распознаёт обученная MNIST-модель в живом
  приложении, а предсказания совпадают с офлайн-результатом бэкенда.
- *Ролик:* нарисовал цифру — сеть угадала.

#### Этап 25 — Внутри модели
Увидеть, что выучила сеть и где она ошибается.
- **Ранняя остановка** по потерям на отложенной части и точность по классам на ней (само
  разделение и его кривые появились в Этапе 15).
- **Матрица ошибок** и точность по классам; **галерея ошибок** (что с чем путается).
- **Карты активаций** свёрточных слоёв для выбранного примера и фильтры первого слоя.
- **Гистограммы весов и градиентов** по слоям во времени — с предупреждениями о затухающих
  или взрывающихся градиентах.
- *Готово, когда:* все виды работают для архитектур библиотеки на MNIST и CIFAR-10 и
  покрыты тестами.
- *Ролик:* что происходит внутри нейросети.

#### Этап 26 — Датасеты как таблицы
Настоящий просмотр данных вместо пары строк превью.
- Единая табличная модель для любого датасета — колонки с типами (число, категория,
  текст, изображение, время) и строки, — включая встроенные датасеты.
- Визуальный просмотр: картинки — сеткой миниатюр, текст — текстом, временные ряды —
  графиками, числа — таблицей.
- Статистика колонок: минимум / максимум / среднее / разброс, пропуски, уникальные
  значения, гистограммы, баланс классов.
- **Редактирование:** ячейки, добавить / удалить строки и колонки, фильтр, сортировка,
  поиск.
- Импорт CSV / TSV / JSON (другие форматы — позже), экспорт и предпросмотр разбиения на
  обучение / валидацию / тест (само разбиение появилось в Этапах 15–16).
- *Готово, когда:* CSV импортируется, просматривается, правится и сохраняется вместе с
  проектом, а большие таблицы листаются плавно.
- *Ролик:* смотрим и правим свои данные, как в таблице.

#### Этап 27 — Редактор «колонки → входы» и предупреждения
Универсальный редактор, который подключает колонки таблицы к модели.
- Колонки — с одной стороны, входы модели — с другой, соединяются перетаскиванием; одна
  колонка выбирается **целевой**. Тип задачи следует из цели: **классификация**
  (категории) или **регрессия** (числа — тренер получает функции потерь MSE / MAE).
- Преобразования на месте: масштабирование (min-max / z-score), кодирование категорий
  (one-hot / номер), заполнение пропусков, логарифм.
- **Валидатор** с ошибками, предупреждениями и подсказками — где можно, с исправлением в
  один клик:
  - вход модели не получил колонку (не выбран);
  - колонка константная или целиком из нулей (не несёт информации);
  - одна колонка подана на несколько входов или две колонки одинаковы (дубликаты);
  - выбранные колонки не совпадают с формой Input;
  - нечисловая колонка без кодирования; пропуски, NaN или бесконечность;
  - целевая колонка оказалась среди входов или вход подозрительно похож на цель (утечка
    цели);
  - дисбаланс классов или слишком мало строк на класс; сильно разный масштаб признаков;
  - для временных рядов — признак, который подглядывает в будущее.
- Маппинг сохраняется в проект.
- *Готово, когда:* у каждого предупреждения есть тест, который его вызывает, а размеченный
  CSV обучается от начала до конца в обоих типах задач.
- *Ролик:* подключаем любую таблицу к сети — редактор предупреждает об ошибках ещё до
  обучения.

#### Этап 28 — Текст → датасет для языковых моделей
Подаём сырой материал — получаем датасет, собранный под модель.
- Загрузка материала (txt / md; pdf и docx — с переводом в текст) или вставка текста.
- Парсер чистит текст, токенизирует его (по символам или словарём BPE, обученным на самом
  материале, — без внешних сервисов) и нарезает окна контекста фиксированной длины на
  пары «контекст → следующий токен».
- Предпросмотр: токены цветом, размер словаря, число окон; предупреждения: слишком мало
  текста, словарь слишком велик для модели, окно длиннее текста.
- В один клик датасет подключается к GPT из библиотеки — размер словаря и выхода
  подстраиваются сами.
- **Генерация:** обученная модель дописывает ваш текст (температура, top-k).
- *Готово, когда:* небольшой текст обучает GPT в живом приложении, и она пишет в стиле
  материала; тесты парсера покрывают токенизацию и нарезку окон.
- *Ролик:* скормили книгу — модель пишет в её стиле.

#### Этап 29 — Временные ряды: свечи
Датасет из выгруженного таймфрейма, например свечей SOLUSDT.
- Импорт выгруженных свечей (CSV: время, open, high, low, close, volume); колонки
  распознаются, дыры во времени — показываются.
- Парсер: окна истории (lookback), настраиваемые признаки (доходности, логарифм цены,
  объём, SMA / EMA / RSI / волатильность) и цель — направление через H свечей (рост /
  падение / флэт по порогу — классификация) или доходность (регрессия).
- **Строго хронологическое разбиение** (без перемешивания) и предупреждения об утечке
  будущего, нестационарности и дисбалансе классов.
- Свечной график с разметкой цели и окон, а после обучения — предсказания на тестовом
  периоде.
- Честная подача: учебный и исследовательский пример — не торговая система и не
  финансовый совет; никаких ордеров и кошельков.
- *Готово, когда:* выгруженный файл свечей проходит путь от импорта до обученной модели и
  графика тестового периода, а проверки утечки покрыты тестами.
- *Ролик:* нейросеть на свечах SOLUSDT — и почему это сложнее, чем кажется.

#### Этап 30 — Свои функции активации: редактор формул
Создавать слои со своей математикой и хранить их под своими названиями.
- **«Создать слой на основе…»:** берём любой слой как родительский (ReLU, GELU, Sigmoid и
  т. д.) и открываем его формулу.
- Пишем `f(x; a, b, …)` с коэффициентами — фиксированными или **обучаемыми**, с
  начальными значениями и диапазонами. **Безопасный математический парсер** — белый
  список операций и функций, никогда не исполняется как код — переводит формулу в
  операции torch.
- Живой график `f(x)` и её производной `f'(x)` (через autograd), слайдеры коэффициентов с
  мгновенной перерисовкой и проверки на NaN / бесконечность, на градиент, который исчезает
  на широком диапазоне («мёртвая» функция), и на взрывной рост.
- Сохранение в **«Мои слои»** с названием, описанием и семейством: слой появляется в
  палитре и ведёт себя как любой другой (формы, параметры; экспорт в PyTorch генерирует
  класс с вашей формулой).
- Хранится в проекте и в личной библиотеке слоёв.
- *Готово, когда:* своя активация обучает сеть в живом приложении и экспортируется в
  автономный PyTorch, а парсер отклоняет всё вне белого списка (проверено тестами).
- *Ролик:* придумали свою функцию активации и обучили на ней сеть.

#### Этап 31 — Свои градиенты и слои с весами
- **Свой backward:** собственная формула производной (surrogate gradient) для
  ступенчатых, пороговых, квантующих или спайковых функций и сравнение с автоматической
  производной.
- **Слои с весами:** формула над входом и обучаемыми тензорами (W, b, …) с выводом форм —
  например, полиномиальный слой, гейт, своя нормализация.
- **Сохранить подграф как слой** — составной свой слой на механике Repeat / Block.
- Обмен: экспорт / импорт библиотеки слоёв файлом с проверкой версий.
- *Готово, когда:* слой с surrogate-градиентом и свой слой с весами обучаются от начала
  до конца и экспортируются в PyTorch, а сохранённый подграф загружается обратно одним
  слоем.
- *Ролик:* слой с нуля — формула, веса и градиент.
