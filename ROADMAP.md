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
* **Этап 8 — Реальные датасеты и UX обучения:** **модалка с данными** на нодах
  Input/Output — формат обучающей выборки и примеры строк из неё, с апгрейдом в
  **парсер кастомных примеров** для обучения; выбор датасета (MNIST/CIFAR/свой),
  живые кривые loss/accuracy, чекпоинты, обучение на GPU, экспорт (код/`state_dict`/ONNX).
* **Постоянно:** высокий уровень ООП, тесты, документация, производительность.
