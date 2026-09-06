"""Deterministic seeding across random, numpy, and torch (CPU + CUDA)."""

from __future__ import annotations

import random

import numpy as np
import torch

__all__ = ["derive_generator", "derive_subseed", "seed_everything"]

_MIX = 0x9E3779B97F4A7C15


def _splitmix64(value: int) -> int:
    """One round of SplitMix64.

    Parameters:
        value: Input state.

    Returns:
        The 64-bit output word.
    """
    state = (value + _MIX) & 0xFFFFFFFFFFFFFFFF
    state = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    state = ((state ^ (state >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (state ^ (state >> 31)) & 0xFFFFFFFFFFFFFFFF


def seed_everything(seed: int) -> None:
    """Seed python, numpy, torch CPU and all CUDA generators.

    Parameters:
        seed: Base seed (>= 0).
    """
    if seed < 0:
        raise ValueError("seed must be >= 0")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_subseed(base_seed: int, *parts: int | str) -> int:
    """Derive a deterministic sub-seed for a window/chunk.

    Parameters:
        base_seed: The prediction-level seed.
        *parts: Discriminating parts (ints or strings).

    Returns:
        A derived seed in [0, 2^31 - 1].
    """
    payload = base_seed
    for part in parts:
        if isinstance(part, str):
            for char in part:
                payload = _splitmix64(payload ^ ord(char))
        else:
            payload = _splitmix64(payload ^ (part & 0xFFFFFFFFFFFFFFFF))
    for _ in range(4):
        payload = _splitmix64(payload)
    return payload % (2**31)


def derive_generator(base_seed: int, *parts: int | str) -> torch.Generator:
    """Create a torch.Generator deterministically seeded for one window.

    Parameters:
        base_seed: The prediction-level seed.
        *parts: Discriminating parts.

    Returns:
        A seeded torch.Generator (CPU device).
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(derive_subseed(base_seed, *parts)))
    return generator
