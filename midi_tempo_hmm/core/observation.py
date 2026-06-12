"""Observation likelihood for inter-onset interval (IOI) events."""

from __future__ import annotations

from types import ModuleType

import numpy as np

from midi_tempo_hmm.core.instrument_category import InstrumentCategory


def compute_likelihood_with_params(
    ioi_sec:      float,
    tempos:       np.ndarray,
    ratios:       np.ndarray,
    ratio_weights: np.ndarray,
    sigma_ratio:  float,
    drum_weight:  float = 1.0,
) -> np.ndarray:
    """Compute per-particle observation likelihood with explicit model parameters.

    Args:
        ioi_sec:       Observed inter-onset interval in seconds.
        tempos:        Particle tempo array, shape (N,), units BPM.
        ratios:        Beat ratio candidates, shape (R,).
        ratio_weights: Prior weights for each ratio, shape (R,).
        sigma_ratio:   Observation noise as fraction of expected IOI.
        drum_weight:   Scaling factor applied to the final likelihood array.

    Returns:
        Likelihood array, shape (N,).
    """
    beat_periods  = 60.0 / tempos
    expected_iois = beat_periods[:, np.newaxis] * ratios[np.newaxis, :]
    sigmas        = expected_iois * sigma_ratio
    diff          = ioi_sec - expected_iois
    gauss         = np.exp(-0.5 * (diff / sigmas) ** 2) / (sigmas * np.sqrt(2.0 * np.pi))
    likelihoods   = gauss @ ratio_weights
    likelihoods  *= drum_weight
    np.clip(likelihoods, 1e-300, None, out=likelihoods)
    return likelihoods


def compute_likelihood(
    ioi_sec:     float,
    tempos:      np.ndarray,
    category:    InstrumentCategory,
    config:      ModuleType,
    drum_weight: float = 1.0,
) -> np.ndarray:
    """Compute per-particle observation likelihood for a given IOI.

    Beat ratios and their prior weights are selected from *config* based on
    *category*:

    ========= ======================== =========================
    Category  Ratios constant          Weights constant
    ========= ======================== =========================
    KICK      KICK_BEAT_RATIOS         KICK_BEAT_WEIGHTS
    SNARE     SNARE_BEAT_RATIOS        SNARE_BEAT_WEIGHTS
    HIHAT     HIHAT_BEAT_RATIOS        HIHAT_BEAT_WEIGHTS
    OTHERS    OTHERS_BEAT_RATIOS       OTHERS_BEAT_WEIGHTS
    ========= ======================== =========================

    Falls back to the global ``BEAT_RATIOS`` / ``BEAT_RATIO_WEIGHTS`` when
    category-specific constants are absent (e.g. during tests that patch
    config).

    Args:
        ioi_sec:     Observed inter-onset interval in seconds.
        tempos:      Particle tempo array, shape (N,), units BPM.
        category:    Instrument category for model selection.
        config:      Module containing beat-ratio constants and SIGMA_OBS_RATIO.
        drum_weight: Scaling factor applied to the final likelihood array.
                     Pass 1.0 for non-drum or fully trusted instruments.

    Returns:
        Likelihood array, shape (N,).  Values are positive but not normalised.
    """
    _PREFIX = {
        InstrumentCategory.KICK:   'KICK',
        InstrumentCategory.SNARE:  'SNARE',
        InstrumentCategory.HIHAT:  'HIHAT',
        InstrumentCategory.OTHERS: 'OTHERS',
    }
    prefix = _PREFIX[category]

    ratios = np.asarray(
        getattr(config, f'{prefix}_BEAT_RATIOS',  config.BEAT_RATIOS),
        dtype=np.float64,
    )
    ratio_weights = np.asarray(
        getattr(config, f'{prefix}_BEAT_WEIGHTS', config.BEAT_RATIO_WEIGHTS),
        dtype=np.float64,
    )
    sigma_ratio = getattr(config, f'{prefix}_SIGMA_OBS_RATIO', config.SIGMA_OBS_RATIO)

    return compute_likelihood_with_params(
        ioi_sec, tempos, ratios, ratio_weights, sigma_ratio, drum_weight
    )
