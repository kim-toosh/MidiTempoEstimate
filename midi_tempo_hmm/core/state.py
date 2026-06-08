"""Particle state representation for the tempo particle filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ParticleState:
    """Single-particle state snapshot.

    Attributes:
        tempo: Tempo estimate in BPM.
        beat_phase: Phase within the current beat, in [0.0, 1.0).
        meter_phase: Beat index within the current measure (0-based).
        weight: Unnormalised particle weight.
    """

    tempo: float
    beat_phase: float
    meter_phase: int
    weight: float


# ---------------------------------------------------------------------------
# Array-based helpers (vectorised over N particles)
# ---------------------------------------------------------------------------

def make_particle_arrays(
    n: int,
    tempo_min: float,
    tempo_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialise particle arrays with uniform priors.

    Returns:
        Tuple of (tempos, beat_phases, meter_phases, weights), all shape (n,).
        tempos are drawn uniformly from [tempo_min, tempo_max].
        beat_phases are drawn uniformly from [0, 1).
        meter_phases are all zero.
        weights are uniform 1/n.
    """
    tempos = np.random.uniform(tempo_min, tempo_max, n)
    beat_phases = np.random.uniform(0.0, 1.0, n)
    meter_phases = np.zeros(n, dtype=np.int32)
    weights = np.ones(n) / n
    return tempos, beat_phases, meter_phases, weights


def arrays_to_states(
    tempos: np.ndarray,
    beat_phases: np.ndarray,
    meter_phases: np.ndarray,
    weights: np.ndarray,
) -> list[ParticleState]:
    """Convert parallel arrays to a list of ParticleState objects (for inspection)."""
    return [
        ParticleState(float(t), float(b), int(m), float(w))
        for t, b, m, w in zip(tempos, beat_phases, meter_phases, weights)
    ]
