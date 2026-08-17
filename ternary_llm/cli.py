"""Shared argparse helpers for the three training entrypoints.

``train.py`` (STE), ``train_discrete.py`` (gradient-free local rules) and
``train_baseline_backprop.py`` (backprop baseline) each declare a large set of
overlapping flags (data source, preset, flip mechanics). The helpers below
declare those once so the scripts stay in sync; every flag keeps its original
name, default and help string so existing invocation lines are unchanged.

A helper is only extracted when at least two scripts use the same flag name.
Flags that differ across scripts (default or help) take their default as a
parameter; per-script-specialized flags stay declared in the owning script.

``_UNSET`` is the sentinel for "do not add this flag"; passing ``None`` (the
usual sentinel for 'no default') still adds the flag, which is what train.py
needs for flags whose CLI default is literally ``None``.
"""

import argparse
import os

import torch

_UNSET = object()

_cpu_configured = False  # module-level: avoid double warnings on re-entry


def _is_cpu(device) -> bool:
    """True when the resolved device is CPU (or unset / invalid)."""
    if not device:
        return True
    dev = str(device).lower()
    return dev in ("cpu", "cpu:0")


def configure_cpu_runtime(device, *, dtype: str | None = None) -> int:
    """Tune the torch runtime for CPU training with no quality trade-off.

    PyTorch's default intra-op thread count often under-uses the machine (it
    may default to a fraction of the cores, e.g. 4 on an 8-core box), and
    float16 matmuls are dramatically slower than float32 on CPU. Both are pure
    wins to correct at runtime:

      * set the intra-op thread count to ``cpu_count()`` (capped at 32, where
        OpenMP sync overhead stops paying for these small GEMMs);
      * warn when ``--dtype float16`` was requested on CPU: fp16 on x86 runs
        through slow scalar/emulated kernels, typically 2-3x slower than fp32
        for the same result.

    Call this as early as possible (before the model and dataloaders are
    built) so the thread pool is sized once. Returns the thread count used.
    """
    if not _is_cpu(device):
        return torch.get_num_threads()

    global _cpu_configured
    n = max(1, min(os.cpu_count() or 4, 32))
    torch.set_num_threads(n)

    if dtype == "float16" and not _cpu_configured:
        print(
            "WARNING: --dtype float16 on CPU is 2-3x slower than float32 "
            "(x86 has no fast fp16 matmul path). Use --dtype float32."
        )
    _cpu_configured = True

    return n


def add_data_args(
    parser: argparse.ArgumentParser,
    *,
    preset_choices=None,
    steps=_UNSET,
    block_size=_UNSET,
    batch_size=_UNSET,
    val_split=_UNSET,
    data_cache=_UNSET,
    device=_UNSET,
    device_choices=None,
    seed=_UNSET,
) -> None:
    """Data source / model-size flags shared across the training scripts.

    Pass ``_UNSET`` (default) to omit a flag; pass any other value — including
    ``None`` — to add it with that CLI default.
    """
    if preset_choices is not None:
        parser.add_argument("--preset", type=str, default=None, choices=list(preset_choices),
                            help="Model size preset (overrides hidden/layers/heads/ffn)")
    if steps is not _UNSET:
        parser.add_argument("--steps", type=int, default=steps,
                            help="Max training steps")
    if block_size is not _UNSET:
        parser.add_argument("--block-size", type=int, default=block_size,
                            help="Block size (context length)")
    if batch_size is not _UNSET:
        parser.add_argument("--batch-size", type=int, default=batch_size,
                            help="Batch size")
    if val_split is not _UNSET:
        parser.add_argument("--val-split", type=float, default=val_split,
                            help="Validation split fraction")
    if data_cache is not _UNSET:
        parser.add_argument("--data-cache", type=str, default=data_cache,
                            help="Data directory (multi-source 'data/' cache or tinydata/)")
    if device is not _UNSET:
        parser.add_argument("--device", type=str, default=device,
                            choices=device_choices,
                            help="Training device")
    if seed is not _UNSET:
        parser.add_argument("--seed", type=int, default=seed,
                            help="RNG seed")


def add_flip_args(
    parser: argparse.ArgumentParser,
    *,
    threshold=_UNSET,
    acc_decay=_UNSET,
    adaptive_thr=_UNSET,
    acc_energy: bool = False,
) -> None:
    """Bit-flip mechanics shared by the discrete & backprop trainers (and the
    stochastic mode of train.py). ``--acc-energy`` is only added when asked
    (it is a store_true flag with no default variance)."""
    if threshold is not _UNSET:
        parser.add_argument("--threshold", type=float, default=threshold,
                            help="Bit-flip threshold")
    if acc_decay is not _UNSET:
        parser.add_argument("--acc-decay", type=float, default=acc_decay,
                            help="Leaky accumulator decay per step (0.99 recommended)")
    if adaptive_thr is not _UNSET:
        parser.add_argument("--adaptive-thr", type=float, default=adaptive_thr,
                            help="Exp 3: adaptive flip threshold k (tau = k*RMS(acc) "
                                 "per channel). None = fixed scalar threshold")
    if acc_energy:
        parser.add_argument("--acc-energy", action="store_true",
                            help="Exp 3: energy accumulator (leaky EMA of -grad) "
                                 "instead of ±1 sign votes")