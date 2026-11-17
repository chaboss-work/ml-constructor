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

### Stage 1 — Branching & merges (DAG) · **IN PROGRESS**
Move from a single chain to a **directed acyclic graph**: a node's output can fan
out to several nodes, and **merge nodes** (`Add`, `Concat`) combine several inputs.
Enables skip/residual connections, parallel paths and multi-branch designs. The
editor supports branches and guides merges; the builder compiles arbitrary DAGs
with shape propagation through the merges.

### Stage 2 — Richer layer catalogue
Dropout; pooling (MaxPool2d, AvgPool2d, AdaptiveAvgPool2d); more normalization
(BatchNorm1d, LayerNorm, GroupNorm); more activations (GELU, LeakyReLU, Softmax);
`ConvTranspose2d`. Fuller per-layer parameter control.

### Stage 3 — Sequence models
`Embedding`, `RNN`, `LSTM`, `GRU`, and the sequence-shaped tensors they need.
Build and train a small sequence model in the editor.

### Stage 4 — Attention & Transformers
`MultiheadAttention`, `TransformerEncoderLayer`, positional encoding, `LayerNorm` —
enough to assemble and train a small transformer block from nodes.

### Stage 5 — Smart adapters
Automatic shape/type reconciliation on connections (auto-insert Flatten/reshape,
dtype casts), with inline validation and one-click fixes — in the spirit of
Unsloth Studio. Fewer manual Flattens; the editor proposes the adapter.

### Stage 6 — Reusable blocks & repeats
Group a sub-graph into a named **Block** and **repeat it N×** (as in "N× transformer
blocks"). Exposes the backend's mother-class composition in the UI; edit a block
once, stamp it many times.

### Stage 7 — Architecture library
A library of ready public architectures (MLP, simple CNN, **ResNet**, **GPT-like**
LLM, …) you can drop onto the canvas and **deeply edit** — every layer, every repeat.

### Stage 8 — Real datasets & training UX
A **data modal** on the Input / Output nodes showing the training set's format and
a few sample rows from it, later upgraded into a **parser for custom training
examples** you paste or upload. Plus a dataset picker (MNIST / CIFAR / custom
upload), live loss & accuracy curves, checkpoints, **GPU** training, and export
(generated PyTorch code / `state_dict` / ONNX).

### Ongoing
Professional OOP and engineering practices, tests, documentation, performance, and
accessibility — maintained across every stage.

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
* **Этап 1 — Ветвление и слияния (DAG) · В РАБОТЕ:** от цепочки к направленному
  графу; ноды-слияния `Add`/`Concat`, ветки, skip/residual-связи, параллельные пути.
* **Этап 2 — Богаче каталог слоёв:** Dropout, пулинг, доп. нормализации и активации,
  `ConvTranspose2d`.
* **Этап 3 — Последовательности:** `Embedding`, `RNN`/`LSTM`/`GRU`.
* **Этап 4 — Внимание и трансформеры:** `MultiheadAttention`,
  `TransformerEncoderLayer`, позиционное кодирование.
* **Этап 5 — Умные адаптеры:** авто-согласование форм/типов на стыках, подсказки и
  починка в один клик (в духе Unsloth Studio).
* **Этап 6 — Блоки и повторы:** группировка подграфа в блок и повтор N× («N×
  трансформерных блоков»).
* **Этап 7 — Библиотека архитектур:** готовые ResNet, GPT-подобные LLM и др. —
  добавил на холст и глубоко редактируешь.
* **Этап 8 — Реальные датасеты и UX обучения:** **модалка с данными** на нодах
  Input/Output — формат обучающей выборки и примеры строк из неё, с апгрейдом в
  **парсер кастомных примеров** для обучения; выбор датасета (MNIST/CIFAR/свой),
  живые кривые loss/accuracy, чекпоинты, обучение на GPU, экспорт (код/`state_dict`/ONNX).
* **Постоянно:** высокий уровень ООП, тесты, документация, производительность.
