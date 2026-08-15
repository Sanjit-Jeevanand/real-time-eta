"""One OpenMP runtime, loaded in the one order that does not deadlock.

LightGBM and PyTorch both link `@rpath/libomp.dylib`, and the environment ships
several copies (torch, sklearn, treelite, tl2cgen). Loading two of them into one
process gives two OpenMP thread pools that contend on the same fork barrier.

Measured on this machine, with a 30-round LightGBM fit and 20 torch steps:

    import torch, then lightgbm  ->  both run, 0.1s / 0.3s / 0.1s
    import lightgbm, then torch  ->  LightGBM fits, then the first torch
                                     backward pass hangs at 0% CPU, forever

It hangs rather than raising, which is the dangerous part: a long training run
looks like it is still working. So the order is asserted here rather than left to
whichever module happened to be imported first.
"""

from __future__ import annotations

import sys

__all__ = ["preload_torch_before_lightgbm"]


def preload_torch_before_lightgbm() -> str:
    """Import torch first if it is installed. Returns what it did, for logging.

    Raises if lightgbm is already loaded and torch is not, because at that point
    the safe order is no longer achievable in this process and importing torch
    later would hang instead of failing.
    """
    if "torch" in sys.modules:
        return "torch already loaded"

    try:
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            return "torch not installed -- single OpenMP runtime, nothing to order"
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return "torch not importable"

    if "lightgbm" in sys.modules:
        msg = (
            "lightgbm was imported before torch. Both link libomp, and this order "
            "deadlocks the first torch backward pass at 0% CPU. Import "
            "eta.models.quantile (or torch itself) before anything touches lightgbm."
        )
        raise RuntimeError(msg)

    import torch  # noqa: F401

    return "torch preloaded ahead of lightgbm"
