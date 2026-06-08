"""Observation likelihood for inter-onset interval (IOI) events."""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import midi_tempo_hmm.config as _cfg


def compute_likelihood(
    ioi_sec: float,
    tempos: np.ndarray,
    beat_phases: np.ndarray,  # reserved for future phase-based observation
    config: ModuleType,
) -> np.ndarray:
    """Compute per-particle observation likelihood for a given IOI.

    For each candidate beat ratio r in config.BEAT_RATIOS, the expected IOI is
    ``beat_period * r`` where ``beat_period = 60 / tempo``.  A Gaussian centred
    at the expected IOI with std ``expected_ioi * config.SIGMA_OBS_RATIO`` is
    evaluated at *ioi_sec*.  The final likelihood is the prior-weighted sum over
    all ratio candidates.

    Args:
        ioi_sec: Observed inter-onset interval in seconds (scalar).
        tempos: Particle tempo array, shape (N,), units BPM.
        beat_phases: Particle beat-phase array, shape (N,); currently unused.
        config: Module containing BEAT_RATIOS, BEAT_RATIO_WEIGHTS, SIGMA_OBS_RATIO.

    Returns:
        Likelihood array, shape (N,).  Values are positive but not normalised.
    """
    beat_periods = 60.0 / tempos  # (N,) seconds per beat

    ratios = np.asarray(config.BEAT_RATIOS, dtype=np.float64)       # (R,)
    ratio_weights = np.asarray(config.BEAT_RATIO_WEIGHTS, dtype=np.float64)  # (R,)

    # expected_iois: (N, R) = beat_periods[:, None] * ratios[None, :]
    expected_iois = beat_periods[:, np.newaxis] * ratios[np.newaxis, :]

    # Gaussian std per (particle, ratio): sigma = expected_ioi * SIGMA_OBS_RATIO
    sigmas = expected_iois * config.SIGMA_OBS_RATIO

    diff = ioi_sec - expected_iois  # (N, R)
    gauss = np.exp(-0.5 * (diff / sigmas) ** 2) / (sigmas * np.sqrt(2.0 * np.pi))

    # Weighted sum over ratio dimension → (N,)
    likelihoods = gauss @ ratio_weights

    # Prevent exact zeros to avoid weight collapse
    np.clip(likelihoods, 1e-300, None, out=likelihoods)
    return likelihoods
