"""Is there a GPU to train on? — with every GPU hidden, never; and no driver crash.

A backend started with every GPU hidden (``CUDA_VISIBLE_DEVICES=-1``, ``""``, …) never
uses CUDA: :func:`cuda_available` says no, ``/health`` reports no GPU, and a "cuda"
device is refused with a clear message.

On Windows the variable itself is also taken away, at import, before torch first
starts the NVIDIA driver. A driver started with no visible device crashed the whole
process (an access violation in the driver's own thread) a minute or so later — found
26.11 on the demo backend. torch starts the driver in ``torch.cuda.is_available()``,
which every optimizer step calls, and in the first backward pass even on the CPU
(autograd asks for the device count by itself) — so asking less often cannot help.
With the card visible the driver starts normally, and CPU work opens no CUDA context:
no video memory is taken (the tests check it at the driver level).
"""

from __future__ import annotations

import logging
import os
import re
import sys


def hides_every_gpu(value: str | None) -> bool:
    """Whether a ``CUDA_VISIBLE_DEVICES`` value leaves no GPU visible.

    Read the way CUDA reads it: the list stops at its first entry that is not a device
    number or UUID, so ``"-1"``, ``""`` or ``"none"`` hide everything, ``"0,-1"`` keeps
    device 0, and an unset variable hides nothing."""
    if value is None:
        return False
    first = value.split(",")[0].strip()
    if first.startswith(("GPU-", "MIG-")):  # a device picked by its UUID
        return False
    number = re.match(r"[+-]?\d+", first)
    return number is None or int(number.group()) < 0


#: The value that hid every GPU when this backend started (None: GPUs not hidden).
HIDDEN_BY: str | None = os.environ.get("CUDA_VISIBLE_DEVICES")
if not hides_every_gpu(HIDDEN_BY):
    HIDDEN_BY = None
#: Every GPU was hidden when this backend started: it never uses CUDA.
CUDA_HIDDEN = HIDDEN_BY is not None

if CUDA_HIDDEN and sys.platform == "win32":
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    logging.getLogger(__name__).warning(
        "CUDA_VISIBLE_DEVICES=%r hides every GPU: this backend will not use CUDA. The variable "
        "itself is removed in this process, because on Windows the NVIDIA driver, started with no "
        "visible device, crashes the process; CPU work opens no CUDA context, so no video memory "
        "is taken.",
        HIDDEN_BY,
    )


def cuda_available() -> bool:
    """Whether training on the GPU can be offered (never when the GPUs were hidden at start)."""
    if CUDA_HIDDEN:
        return False
    import torch

    return bool(torch.cuda.is_available())


def hidden_message() -> str:
    """Why a "cuda" device is refused on a backend started with every GPU hidden."""
    return (
        "GPU (CUDA) training was requested, but this backend was started with every GPU hidden "
        f"(CUDA_VISIBLE_DEVICES={HIDDEN_BY!r}); restart it without that variable to train on the GPU"
    )
