"""Your own text for a language model (Stage 18): a character vocabulary, training
windows and generation.

The simplest honest tokenizer: every character is a token. A text becomes training
rows of ``L`` characters, each labelled with the character that follows — exactly
what a GPT learns, at the character level. After training, :func:`generate_text`
continues a prompt one character at a time.

The windows are cut inside **blocks** of the text, and each row carries its block
as a group: the split into training / validation / test then keeps a block whole,
so no held-out row has a near-copy (its neighbour window) among the training rows.

Self-contained on purpose (torch only): the PyTorch export copies these functions
verbatim, so an exported ``model.py`` builds the same rows and writes the same way.
"""

from collections import Counter

import torch

#: The token rare characters share when the network's vocabulary is too small.
UNKNOWN = "�"


def text_vocab(text, size):
    """The text's characters, most frequent first (ties by code point), as at most
    `size` tokens: if the text has more characters, the rarest share one UNKNOWN
    token (the last)."""
    counts = Counter(text)
    chars = sorted(counts, key=lambda c: (-counts[c], ord(c)))
    if len(chars) > size:
        chars = chars[: size - 1] + [UNKNOWN]
    return chars


def encode_text(text, vocab):
    """Characters -> token ids (characters outside the vocabulary -> UNKNOWN, else 0)."""
    index = {c: i for i, c in enumerate(vocab)}
    fallback = index.get(UNKNOWN, 0)
    return [index.get(c, fallback) for c in text]


def decode_text(ids, vocab):
    return "".join(vocab[int(i)] if 0 <= int(i) < len(vocab) else UNKNOWN for i in ids)


def text_block(n_chars, length):
    """How long a block of the text is: about a twentieth of it, between 2 and 32
    windows long."""
    return max(2 * (length + 1), min(32 * (length + 1), n_chars // 20))


def text_windows(text, length, vocab_size, samples, seed):
    """Rows for next-character training: X = `length` characters (as float token
    ids), y = the character after each of them (the window shifted by one: a GPT
    learns at every position; a model that scores only the last position uses the
    last), groups = the block each row comes from. At most `samples` rows, a random
    choice (fixed by `seed`) kept in text order."""
    vocab = text_vocab(text, vocab_size)
    ids = torch.tensor(encode_text(text, vocab), dtype=torch.long)
    block = text_block(len(ids), length)
    n_blocks = len(ids) // block
    per_block = block - length  # windows that stay inside their block (and keep their label there)
    starts = (torch.arange(n_blocks)[:, None] * block + torch.arange(per_block)[None, :]).reshape(-1)
    groups = torch.arange(n_blocks)[:, None].expand(n_blocks, per_block).reshape(-1)
    if starts.numel() > samples:
        g = torch.Generator().manual_seed(int(seed))
        keep = torch.randperm(starts.numel(), generator=g)[:samples].sort().values
        starts, groups = starts[keep], groups[keep]
    X = ids[starts[:, None] + torch.arange(length)[None, :]]
    y = ids[starts[:, None] + 1 + torch.arange(length)[None, :]]
    return X.float(), y, groups


def generate_text(model, vocab, context, prompt, steps, temperature=0.8, seed=0):
    """Continue `prompt` by `steps` characters: the model reads the last `context`
    characters, scores every token, and one is drawn (temperature 0 = always the
    most likely). Returns the continuation only."""
    g = torch.Generator().manual_seed(int(seed))
    pad = encode_text(" ", vocab)[0] if " " in vocab else 0
    ids = encode_text(prompt, vocab)
    out = []
    model.eval()
    with torch.no_grad():
        for _ in range(int(steps)):
            window = ids[-context:] or [pad]
            try:  # a GPT reads a short prompt as it is ...
                logits = model(torch.tensor([window], dtype=torch.float32))[0]
            except RuntimeError:  # ... a network fixed to `context` tokens gets it padded with spaces
                window = ([pad] * context + window)[-context:]
                logits = model(torch.tensor([window], dtype=torch.float32))[0]
            if logits.dim() == 2:  # scores at every position: the next character is the last one's
                logits = logits[-1]
            logits = logits[: len(vocab)]  # only real tokens
            if temperature <= 0:
                nxt = int(logits.argmax())
            else:
                probs = torch.softmax(logits.float() / float(temperature), dim=-1)
                nxt = int(torch.multinomial(probs, 1, generator=g))
            ids.append(nxt)
            out.append(nxt)
    return decode_text(out, vocab)
