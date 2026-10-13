"""Small, dependency-free statistics helpers shared by Robot CI pipelines.

Only numpy is used. All resampling uses a fixed RNG seed so reports are
reproducible run to run.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 12345


def summarize(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Mean, standard deviation (sample, ddof=1), min, max and count."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": n,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if n > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Optional[List[float]]:
    """Percentile bootstrap CI of the mean. None when fewer than 2 values."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return [float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))]


def bootstrap_delta_ci(
    baseline: Sequence[float],
    candidate: Sequence[float],
    confidence: float = 0.95,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Optional[List[float]]:
    """Percentile bootstrap CI of mean(candidate) - mean(baseline).

    Groups are resampled independently. Returns None if either group has
    fewer than 2 values (a CI would be meaningless).
    """
    b = np.asarray([v for v in baseline if v is not None and np.isfinite(v)], dtype=float)
    c = np.asarray([v for v in candidate if v is not None and np.isfinite(v)], dtype=float)
    if b.size < 2 or c.size < 2:
        return None
    rng = np.random.default_rng(seed)
    bi = rng.integers(0, b.size, size=(n_boot, b.size))
    ci = rng.integers(0, c.size, size=(n_boot, c.size))
    deltas = c[ci].mean(axis=1) - b[bi].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return [float(np.quantile(deltas, alpha)), float(np.quantile(deltas, 1.0 - alpha))]
