"""Resampling utilities for the particle filter."""

from __future__ import annotations

import numpy as np


def compute_ess(weights: np.ndarray) -> float:
    """Compute the Effective Sample Size (ESS) from particle weights.

    ESS = 1 / sum(w_i^2) where w_i are the normalised weights.
    ESS == N when all weights are equal; ESS == 1 when one particle has all weight.

    Args:
        weights: Particle weight array, shape (N,).  Need not be normalised.

    Returns:
        ESS as a float in [1, N].
    """
    w = weights / weights.sum()
    return float(1.0 / np.sum(w ** 2))


def systematic_resample(weights: np.ndarray) -> np.ndarray:
    """Systematic resampling of particle indices.

    Systematic resampling uses a single random offset to generate N evenly-spaced
    points on the CDF, giving O(N) cost and lower variance than multinomial
    resampling.

    Args:
        weights: Particle weight array, shape (N,).  Need not be normalised.

    Returns:
        Index array, shape (N,), with values in [0, N).
    """
    n = len(weights)
    w = weights / weights.sum()
    cumsum = np.cumsum(w)

    # Single uniform draw shifts all N evenly-spaced positions
    u0 = np.random.uniform(0.0, 1.0 / n)
    positions = u0 + np.arange(n) / n

    return np.searchsorted(cumsum, positions).astype(np.int64)
