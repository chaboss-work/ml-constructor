"""The catalogue of every error and warning the editor can meet (Stage 21).

Each entry has a stable **code** (what the backend raises and the editor keys on;
Stage 24 translates by it), a short **title** in plain words, **why** it happens and
**what to do**. Words of the small glossary are written ``[[like this]]``: the editor
underlines them and shows what they mean on a click.

The fixes themselves live where the problem is understood (the layers, the builder,
:mod:`mlconstructor.diagnostics`); an entry's ``fixes`` are the ones that always
apply (opening Data, training again…).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import fixes as fx

CATEGORIES = {
    "wiring": "Connections",
    "shapes": "Shapes",
    "params": "Settings",
    "data": "Data",
    "training": "Training",
}


@dataclass(frozen=True)
class Entry:
    code: str
    category: str
    title: str
    why: str
    todo: str
    severity: str = "error"  # or "warning": the graph builds, but is probably not what you meant
    #: fixes that always apply (most fixes are made for the very case, by the code that finds it)
    fixes: tuple = field(default_factory=tuple)


GLOSSARY: dict[str, str] = {
    "shape": "How the numbers of one example are arranged: 784 is a vector, 1×28×28 a picture, 16×64 a sequence. "
             "The batch (many examples at once) is never part of it.",
    "vector": "One row of numbers — e.g. 784 pixel values, or 10 class scores. A Linear layer reads a vector.",
    "sequence": "A list of steps, each a vector: L×D is L steps (tokens, time steps) of D numbers each. "
                "Attention, LSTM and transformer layers read sequences.",
    "C×H×W": "A picture: C channels (1 for gray, 3 for colour, or many feature maps) of H×W pixels. "
             "Convolutions and pooling read pictures.",
    "channel": "One layer of a picture: a gray image has 1 channel, a colour one 3 (red, green, blue). "
               "A convolution makes as many channels as it has filters.",
    "token": "One piece of a text — here usually one character — written as a whole number (its id).",
    "token ids": "Text written as whole numbers, one per token. An Embedding turns each id into a vector.",
    "embedding": "A table that turns a whole number (a token id, a class) into a vector of numbers the network can learn.",
    "class scores": "The network's answer for classification: one number per class (also called logits); "
                    "the biggest one is its guess.",
    "batch": "Many examples processed at once. Shapes in the editor are for one example; the batch is added on top.",
    "kernel": "The small window a convolution or pooling slides over a picture, e.g. 3×3 pixels.",
    "stride": "How far the kernel moves each step: stride 2 halves the picture's height and width.",
    "padding": "Empty pixels added around the picture's border, so a kernel also fits at the edges.",
    "head": "Attention splits each vector into several heads that look at the sequence separately; "
            "the vector's size must divide evenly between them.",
    "merge": "Add or Concat: a node that combines several branches into one.",
    "connection transform": "What a connection does to the data on its way: nothing (Direct), flatten it, "
                            "reshape it, read a picture as a sequence… Auto picks one when the shapes differ.",
    "Auto": "A connection that adapts the data to the next layer by itself (flatten, reshape…), "
            "picking the transform that keeps the meaning.",
    "Repeat block": "A short chain of layers stamped several times, each copy with its own weights.",
    "parameter": "A number the network learns (a weight). Their count is the network's size.",
    "epoch": "One pass over all the training rows.",
    "learning rate": "How big a step training takes after each batch: too big jumps around, too small barely moves.",
    "dropout": "During training, a random share of the numbers is set to zero, so the network cannot lean on any "
               "single one. 0.1–0.5 is usual.",
    "timestep": "In diffusion, how much noise was added to a picture (0 = clean, 1000 = pure noise).",
    "latent": "A compressed picture: a smaller grid of numbers a decoder turns back into the image.",
    "vocabulary": "All the different tokens a language model knows; its last layer scores each of them.",
    "test part": "Rows kept away from training and from the epoch-by-epoch check: the final, fair exam.",
    "weights": "All the learned numbers of a network, saved together in a .pt file.",
}


def _e(code, category, title, why, todo, severity="error", fixes=()):
    return Entry(code, category, title, why, todo, severity, tuple(fixes))


_OPEN_DATA = fx.ui("open_data", "Open Data")
_TRAIN = fx.ui("train", "Train again")
_LOAD = fx.ui("open_weights", "Load another weights file")
_LIBRARY = fx.ui("open_library", "Open the library (each flagship has a mini version)")

_ENTRIES = [
    # ------------------------------------------------------------------ wiring
    _e("duplicate_id", "wiring", "Two nodes share one name",
       "Connections name the nodes they join, so every node needs an id of its own; with two alike the "
       "wiring is ambiguous.",
       "Rename one of them."),
    _e("unknown_node", "wiring", "A connection leads to a node that is not there",
       "The connection names a node that was deleted or never existed, so the data has nowhere to go.",
       "Remove the connection."),
    _e("cycle", "wiring", "The connections go round in a circle",
       "Data flows one way, from the Input to the Output. In a loop a layer would wait for its own output forever.",
       "Remove one connection of the loop."),
    _e("no_input", "wiring", "There is no Input",
       "The Input declares the [[shape]] of one example — a [[C×H×W]] picture, a [[vector]], [[token ids]]. Every "
       "other shape in the network is worked out from it.",
       "Add an Input and connect it to the first layer."),
    _e("several_outputs", "wiring", "The network has more than one end",
       "A network gives one answer, so every branch has to flow into one Output. Branches can be joined with a "
       "[[merge]] (Add or Concat).",
       "Join the loose ends into the Output, or remove the extra branch."),
    _e("input_has_input", "wiring", "Something is connected into an Input",
       "An Input is where the data enters the network; nothing can flow into it.",
       "Remove the connection into the Input."),
    _e("no_connection", "wiring", "This layer has no input",
       "A layer works on the data that reaches it. With nothing connected into its top port it has nothing to do, "
       "and every shape after it stays unknown.",
       "Connect the layer before it into its top port, or delete it."),
    _e("too_many_inputs", "wiring", "Several connections go into a layer that takes one",
       "Most layers read one stream of data. Branches must be combined first: Add sums data of the same "
       "[[shape]], Concat puts it side by side.",
       "Join the branches with Add or Concat, or keep one connection."),
    _e("merge_needs_two", "wiring", "Add / Concat needs at least two branches",
       "A [[merge]] combines branches; with one input there is nothing to combine.",
       "Connect a second branch into it, or remove it and connect straight through."),
    _e("ports_count", "wiring", "This layer needs a different number of inputs",
       "It reads several different things — e.g. a noisy picture, a [[timestep]] and a text — and tells them apart "
       "by the order the connections were made: the first connection feeds its first input, and so on.",
       "Connect exactly its inputs, in the order its message gives."),
    _e("block_needs_input", "wiring", "The block needs one more input",
       "A layer inside this [[Repeat block]] reads a second input (a condition, the audio…). The block takes it "
       "once, as its second connection, and hands it to every copy.",
       "Connect the main data first, then that input."),
    # ------------------------------------------------------------------ shapes
    _e("wrong_rank", "shapes", "The data arrives in the wrong form",
       "Each layer reads one form of data: a Linear reads a [[vector]], a Conv2d a [[C×H×W]] picture, attention a "
       "[[sequence]]. What arrives here has another [[shape]].",
       "Let the connection reshape it (Auto, or pick a [[connection transform]]), or feed the layer from one whose "
       "output fits."),
    _e("wrong_shape", "shapes", "The data does not fit this layer",
       "The data has the right form but the wrong size for this layer — e.g. a picture whose side does not split "
       "into whole patches, or an embedding of another size.",
       "Change the layer's settings, or the layers that feed it, so the sizes agree."),
    _e("pinned_transform", "shapes", "The connection's transform does not fit",
       "This connection was set to a fixed [[connection transform]]. The data arriving cannot go through it, or "
       "comes out in a form the next layer cannot read.",
       "Set the connection back to Auto, or pick a transform that fits."),
    _e("no_auto_transform", "shapes", "Nothing can adapt this connection by itself",
       "[[Auto]] adapts data when the change keeps its meaning — flattening a picture, reading it as a sequence. "
       "Here no such change gives what the layer needs (or the learnable projection would be too big).",
       "Pick a transform for the connection, change the layer's settings, or feed it from a layer whose output fits."),
    _e("merge_shapes", "shapes", "The branches cannot be combined",
       "Add sums the branches number by number, so they must have the same [[shape]]; Concat puts them side by "
       "side along one axis, so every other axis must match.",
       "Set a transform on one connection into it, change the layers that feed it, or pick another axis."),
    _e("too_small", "shapes", "The picture is too small for this layer",
       "A [[kernel]] bigger than the picture leaves nothing, and every [[stride]] 2 step halves the picture — after "
       "a few, a 28×28 digit is down to a single pixel.",
       "Use a smaller kernel or stride, add [[padding]], or remove a pooling layer before it."),
    _e("output_not_classes", "shapes", "The output is not one score per class",
       "To classify, the network must end in a [[vector]] with one number per class — the [[class scores]].",
       "End the network with Flatten and a Linear that has one output per class."),
    # ------------------------------------------------------------------ params
    _e("not_divisible", "params", "Two settings do not divide evenly",
       "Some layers split their numbers into equal parts: attention into [[head]]s, GroupNorm into groups, a "
       "picture into patches. The size must divide by the number of parts.",
       "Pick a number of parts that divides the size (the fixes below do)."),
    _e("param_range", "params", "A setting is out of range",
       "Some settings only make sense within limits — a size of at least 1, a probability between 0 and 1, no more "
       "experts chosen than there are.",
       "Set it within the allowed range."),
    _e("param_choice", "params", "A setting has a value the layer does not know",
       "This setting takes one of a few named values, and the one given is not among them.",
       "Pick one of the values the message lists."),
    _e("param_list", "params", "A list setting is not valid",
       "This setting is a list of whole numbers (e.g. channels per level), and what it holds is not.",
       "Enter whole numbers separated by commas, or go back to the default."),
    _e("settings_conflict", "params", "The layer's settings contradict each other",
       "Two settings of the layer cannot both hold — e.g. rotary sizes that do not add up to the head size. No "
       "connection can fix that: the settings have to agree.",
       "Change one of the settings the message names."),
    _e("input_shape", "params", "The Input's shape is not valid",
       "The Input's [[shape]] is the size of one example — every number must be at least 1: 784 for a vector, "
       "1, 28, 28 for a gray picture.",
       "Enter the shape of one example."),
    _e("block_empty", "params", "The Repeat block is empty",
       "A [[Repeat block]] stamps its inner layers several times; with none inside it has nothing to stamp.",
       "Add a layer inside the block, or delete the block."),
    _e("block_bad_inner", "params", "This layer cannot go inside a Repeat block",
       "A [[Repeat block]] is a straight chain of ordinary layers. An Input, an Output or a [[merge]] needs "
       "connections of its own, so it cannot be stamped inside.",
       "Remove that layer from the block."),
    _e("unknown_layer", "params", "A layer type this version does not know",
       "The project names a layer this version of the app does not have — made by a newer version, or misspelt.",
       "Delete the node, or replace it with a known layer."),
    _e("tie_no_embedding", "params", "The LM head is tied to an Embedding that is not there",
       "A tied LM head reuses the [[embedding]] table as its own weights — but the network has no Embedding.",
       "Switch off “Tie to the Embedding”, or add an Embedding at the start."),
    _e("tie_size", "params", "The tied LM head and the Embedding differ in size",
       "Tying shares one table between the [[embedding]] and the LM head, so both need the same "
       "[[vocabulary]] and the same vector size.",
       "Give the head the embedding's vocabulary, or untie it."),
    _e("bad_transform", "params", "The connection names an unknown transform",
       "The connection asks for a [[connection transform]] this version does not have.",
       "Set the connection back to Auto."),
    _e("transform_param", "params", "A connection setting is not valid",
       "The connection's [[connection transform]] has a setting that cannot work with the data arriving — e.g. a "
       "target shape that holds a different number of values.",
       "Set the connection back to Auto, or pick one of its suggested settings."),
    # ------------------------------------------------------------------ data
    _e("dataset_shape", "data", "The data does not fit the Input",
       "A real dataset has fixed pictures: MNIST digits are 1×28×28 (one gray [[channel]]), CIFAR-10 photos "
       "3×32×32. The Input must declare exactly that [[shape]].",
       "Set the Input to the dataset's shape, or pick a dataset that adapts to any shape."),
    _e("dataset_classes", "data", "The network has fewer outputs than the data has classes",
       "The data has more classes than the network has [[class scores]]: the classes without a score could never be "
       "guessed.",
       "Give the last layer one output per class."),
    _e("dataset_needs_tokens", "data", "This data needs a network that reads token ids",
       "Next token and Your text feed the network [[token ids]]: its Input must be one number per position "
       "(Input [L]), followed by an [[embedding]].",
       "Take a language model from the library, or pick a dataset for this network."),
    _e("tokens_too_short", "data", "The sequences are too short",
       "The next-token task needs at least 3 [[token]]s per row to show a pattern to continue.",
       "Make the Input longer (16 is a good start)."),
    _e("text_missing", "data", "There is no text yet",
       "Your text teaches the network the text you give it — and none has been given yet.",
       "Open Data and paste or upload a text, or use the example.",
       fixes=[fx.ui("use_example_text", "Use the example text"), _OPEN_DATA]),
    _e("text_too_short", "data", "The text is too short",
       "Training cuts the text into windows as long as the Input; a short text gives too few of them to learn from.",
       "Paste more text, or make the Input shorter.",
       fixes=[_OPEN_DATA]),
    _e("custom_missing", "data", "There are no rows yet",
       "Your data trains on rows you give it, and none have been given yet.",
       "Open Data and paste or upload rows: each row is the features, then a whole-number class.",
       fixes=[_OPEN_DATA]),
    _e("custom_too_many", "data", "Too many rows",
       "This server takes up to 10,000 pasted rows.",
       "Keep fewer rows.",
       fixes=[_OPEN_DATA]),
    _e("custom_row_length", "data", "A row has the wrong number of values",
       "Each row is the features of one example, then its class. The number of features must match the Input's "
       "[[shape]] (an Input of 4 takes 4 features + 1 class = 5 values).",
       "Fix the row, or set the Input to the number of features your rows have.",
       fixes=[_OPEN_DATA]),
    _e("custom_value", "data", "A row holds something that is not a number",
       "Training needs ordinary numbers; an empty cell, NaN or infinity cannot be learned from.",
       "Fix or remove that row.",
       fixes=[_OPEN_DATA]),
    _e("custom_label", "data", "A class is not a whole number",
       "The last value of each row is its class: 0, 1, 2… — a whole number.",
       "Fix that row's last value.",
       fixes=[_OPEN_DATA]),
    _e("custom_label_range", "data", "A class is bigger than the network can score",
       "Classes run from 0 to one less than the number of [[class scores]] the network gives.",
       "Give the last layer more outputs, or fix the row.",
       fixes=[_OPEN_DATA]),
    _e("unknown_dataset", "data", "The project names an unknown dataset",
       "The dataset named in the project is not one this version offers.",
       "Pick a dataset from the list."),
    _e("every_position", "data", "The network scores every position, but the data gives one label per row",
       "A language model can guess the next [[token]] at every position at once — but this data has one answer "
       "per row.",
       "Switch the Output's “Every position” off, or train on Your text."),
    _e("class_rows", "data", "The class embedding has fewer rows than the data has classes",
       "Diffusion turns the class into a vector with an [[embedding]]; it needs a row per class (and one more to "
       "draw without a class).",
       "Give the embedding one row per class, plus one."),
    # ------------------------------------------------------------------ training
    _e("too_big", "training", "Too big to train here",
       "This network has more [[parameter]]s than this server can train. It is here to inspect: every layer and "
       "shape of the real model.",
       "Open its mini version from the library, or make the layers smaller.",
       fixes=[_LIBRARY]),
    _e("task_inputs", "training", "This network cannot be trained here",
       "Training here learns from one input (to classify or to reconstruct it), or learns to generate by "
       "diffusion — a noisy picture, a [[timestep]] and maybe a class in. This network takes other inputs.",
       "Run it once on random inputs, or change its inputs.",
       fixes=[fx.ui("run_once", "Run once instead")]),
    _e("task_output_shape", "training", "The output cannot be learned",
       "To learn to reconstruct its input, the output must have the input's [[shape]]; to classify, it must be one "
       "number per class.",
       "Make the output the input's shape, or switch the Output's “Any shape” off to classify."),
    _e("task_needs_any_shape", "training", "Reconstruction needs an Output of any shape",
       "To give back a picture, the Output must accept any [[shape]] — by default it takes [[class scores]].",
       "Switch the Output's “Any shape” on."),
    _e("task_any_shape_on", "training", "Classification needs one score per class",
       "The Output is set to “Any shape”, which is for networks that give back pictures — classification needs "
       "[[class scores]].",
       "Switch the Output's “Any shape” off."),
    _e("no_gpu", "training", "There is no GPU to train on",
       "Training was asked to run on a GPU, but this server has none it can use.",
       "Train on the CPU.",
       fixes=[fx.set_train("device", "cpu", "Train on the CPU")]),
    _e("no_test_part", "training", "There is no test part to measure",
       "The final test uses the [[test part]] — rows kept out of training. With Final test off there are none.",
       "Set Final test to 10 % and train again.",
       fixes=[fx.set_train("test_fraction", 0.1, "Set Final test to 10 %")]),
    _e("few_rows", "training", "Too few rows to keep a test part aside",
       "Every class needs rows to train on first; with this few rows none are left for the [[test part]].",
       "Use more rows."),
    _e("draw_needs_generative", "training", "This network cannot draw",
       "Draw needs a network that learned to generate: a VAE (it reconstructs its input) or a diffusion model.",
       "Train a generative network — e.g. Digit VAE or Digit diffusion from the library.",
       fixes=[_LIBRARY]),
    _e("draw_class", "training", "No such class to draw",
       "The network learned the classes 0 to one less than their number; this one is not among them.",
       "Pick a class from the list.",
       fixes=[fx.ui("draw_each", "Draw one of each class")]),
    _e("generate_needs_text", "training", "Generate continues text",
       "Generate writes like the text the network learned — so it needs a network trained on Your text.",
       "Choose Your text as the dataset, add a text in Data, train, then generate.",
       fixes=[fx.set_train("dataset", "text", "Use Your text as the dataset")]),
    _e("generate_needs_lm", "training", "Generate needs a language model",
       "Only a language model — [[token ids]] in, an [[embedding]], a score for every token out — can write text.",
       "Take Your GPT from the library.",
       fixes=[_LIBRARY]),
    _e("run_missing", "training", "These weights are gone",
       "The server keeps the [[weights]] of its last runs only; these were removed to make room.",
       "Train again, or load the saved .pt file.",
       fixes=[_TRAIN, _LOAD]),
    _e("weights_file", "training", "This is not a usable weights file",
       "A [[weights]] file is a .pt file saved by Save weights (or torch.save of a state_dict); this one is not, "
       "or it is damaged.",
       "Load a .pt file saved from this app.",
       fixes=[_LOAD]),
    _e("weights_unsafe", "training", "The file holds more than weights",
       "It also holds Python objects, possibly code. For safety the app only opens files with plain numbers — "
       "nothing in it was run.",
       "Save the weights alone (a state_dict) and load that.",
       fixes=[_LOAD]),
    _e("weights_newer", "training", "The weights come from a newer version of the app",
       "This version cannot read the newer file format.",
       "Update the app, or load weights saved by this version.",
       fixes=[_LOAD]),
    _e("weights_mismatch", "training", "These weights do not fit this network",
       "[[weights]] fit only the network they were trained in: the same layers with the same sizes.",
       "Set the layers back to the sizes they had, load weights from this network, or train again.",
       fixes=[_TRAIN, _LOAD]),
    # ------------------------------------------------------------------ warnings
    _e("dropout_high", "params", "A very high dropout",
       "[[dropout]] this high throws most numbers away at every step — the network barely learns.",
       "Use 0.1–0.5.", severity="warning"),
    _e("tiny_hidden", "params", "A very narrow layer in the middle",
       "Everything the network knows has to pass through this layer; a handful of numbers is a bottleneck "
       "smaller than the answer it must give.",
       "Make the layer wider.", severity="warning"),
    _e("unused_branch", "wiring", "A branch leads nowhere",
       "These layers are computed (and trained) but their result never reaches the Output, so it is lost.",
       "Connect the branch into the network, or remove it.", severity="warning"),
    _e("after_output", "wiring", "Layers come after the Output",
       "The Output should be the network's last node; the layers after it change the answer, and the Output is no "
       "longer the end.",
       "Move the Output to the end, or remove the layers after it.", severity="warning"),
    _e("softmax_before_loss", "params", "Softmax just before the Output",
       "Training already turns the [[class scores]] into probabilities inside its loss; a Softmax before it does "
       "it twice and slows learning down.",
       "Remove the Softmax.", severity="warning"),
    _e("linear_linear", "params", "Two Linear layers in a row",
       "Two Linear layers with nothing between them act as one Linear: the extra [[parameter]]s add no power. A "
       "non-linearity (ReLU) between them lets the network learn curves.",
       "Insert a ReLU between them.", severity="warning"),
    _e("no_output_node", "wiring", "There is no Output node",
       "Without an Output node the last layer is taken as the answer — it is clearer to mark it.",
       "Add an Output after the last layer.", severity="warning"),
    _e("lr_extreme", "training", "An unusual learning rate",
       "A [[learning rate]] far above 0.1 makes training jump around (the loss can blow up); far below 0.00001 it "
       "barely moves.",
       "Use 0.001–0.01 with Adam.", severity="warning"),
    _e("batch_too_big", "training", "The batch is bigger than the data",
       "A [[batch]] bigger than the training rows means one step per [[epoch]] — the network learns very slowly.",
       "Use a smaller batch.", severity="warning"),
]

CATALOG: dict[str, Entry] = {e.code: e for e in _ENTRIES}
assert len(CATALOG) == len(_ENTRIES), "duplicate code in the catalogue"

#: the code of an error that came without one (a bug, or a check that predates the catalogue)
OTHER = Entry("other", "training", "Something went wrong",
              "The app met a problem it has no explanation for yet.",
              "Read the message; if it does not help, undo the last change.")


def entry(code: str | None) -> Entry:
    return CATALOG.get(code or "", OTHER)


def terms_in(text: str) -> list[str]:
    """The glossary words a text marks ``[[like this]]``, in order, each once."""
    out: list[str] = []
    rest = text
    while "[[" in rest:
        start = rest.index("[[") + 2
        end = rest.index("]]", start)
        word = rest[start:end]
        base = word[:-1] if word.endswith("s") and word[:-1] in GLOSSARY else word
        if base not in out:
            out.append(base)
        rest = rest[end + 2:]
    return out
