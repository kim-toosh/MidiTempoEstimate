"""Particle filter for real-time MIDI tempo, beat, and measure estimation."""

from __future__ import annotations

import logging
import time
from types import ModuleType
from typing import Optional

import numpy as np

from midi_tempo_hmm.core.observation import compute_likelihood
from midi_tempo_hmm.core.resampling import compute_ess, systematic_resample
from midi_tempo_hmm.core.state import make_particle_arrays
from midi_tempo_hmm.output.estimator_result import EstimatorResult

logger = logging.getLogger(__name__)


class ParticleFilter:
    """Sequential Monte Carlo estimator for MIDI tempo, beat phase, and meter.

    Each particle represents a hypothesis about the current tempo (BPM),
    position within the beat (beat_phase ∈ [0, 1)), and position within the
    measure (meter_phase ∈ {0, …, METER_NUMERATOR-1}).

    Call :meth:`update` once per received MIDI note-on event.  The first call
    only stores the reference timestamp; subsequent calls perform the full
    predict–weight–resample cycle and return an :class:`EstimatorResult`.

    Args:
        config: Module (e.g. ``midi_tempo_hmm.config``) containing the global
                constants N_PARTICLES, TEMPO_MIN, TEMPO_MAX, SIGMA_TEMPO,
                SIGMA_OBS_RATIO, BEAT_RATIOS, BEAT_RATIO_WEIGHTS,
                ESS_THRESHOLD_RATIO, METER_NUMERATOR, TEMPO_PRECISION.
    """

    def __init__(self, config: ModuleType) -> None:
        self.config = config
        self.n_particles: int = config.N_PARTICLES
        self._prev_timestamp: Optional[float] = None
        self._initialize_particles()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, timestamp_sec: float) -> Optional[EstimatorResult]:
        """Process a MIDI note-on event and return the current tempo estimate.

        The first call initialises the reference time and returns *None*
        because no IOI is yet available.

        Args:
            timestamp_sec: Absolute timestamp of the event in seconds.

        Returns:
            :class:`EstimatorResult` after the second and subsequent calls,
            or *None* for the very first call.
        """
        t0 = time.perf_counter()

        if self._prev_timestamp is None:
            self._prev_timestamp = timestamp_sec
            logger.debug("First event at %.4f s — initialising reference time.", timestamp_sec)
            return None

        ioi_sec = timestamp_sec - self._prev_timestamp
        self._prev_timestamp = timestamp_sec

        if ioi_sec <= 0.0:
            logger.warning("Non-positive IOI (%.6f s) ignored.", ioi_sec)
            return None

        self._predict(ioi_sec)
        self._update_weights(ioi_sec)
        self._resample_if_needed()

        processing_time_ms = (time.perf_counter() - t0) * 1000.0
        result = self._estimate(processing_time_ms)
        logger.debug(
            "tempo=%.2f BPM  beat_pos=%.3f  measure_beat=%d  conf=%.3f  t=%.2f ms",
            result.tempo_bpm,
            result.beat_position,
            result.measure_beat,
            result.confidence,
            result.processing_time_ms,
        )
        return result

    def reset(self) -> None:
        """Re-initialise all particles and clear the reference timestamp."""
        self._prev_timestamp = None
        self._initialize_particles()

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _initialize_particles(self) -> None:
        (
            self.tempos,
            self.beat_phases,
            self.meter_phases,
            self.weights,
        ) = make_particle_arrays(
            self.n_particles,
            self.config.TEMPO_MIN,
            self.config.TEMPO_MAX,
        )

    # Fraction of particles resampled from the prior each step.
    # This mixture component lets the filter recover from abrupt tempo changes
    # that a pure Gaussian random walk (SIGMA_TEMPO = 0.5 BPM) cannot track.
    _EXPLORER_FRACTION: float = 0.02

    def _predict(self, ioi_sec: float) -> None:
        """Propagate particles through the state transition model.

        Tempo undergoes a mixture transition:
          - (1 - α) particles follow a Gaussian random walk with std SIGMA_TEMPO.
          - α particles jump to a fresh uniform draw from [TEMPO_MIN, TEMPO_MAX].
        The explorer component maintains global diversity and allows recovery from
        abrupt tempo changes without requiring a large SIGMA_TEMPO.
        Beat phase advances by ioi_sec / beat_period; integer crossings increment
        the meter phase.
        """
        # Tempo transition: Gaussian random walk
        noise = np.random.normal(0.0, self.config.SIGMA_TEMPO, self.n_particles)
        self.tempos += noise
        np.clip(self.tempos, self.config.TEMPO_MIN, self.config.TEMPO_MAX, out=self.tempos)

        # Explorer mixture: a small fraction jump to the prior to handle abrupt changes
        n_explorers = max(1, int(self._EXPLORER_FRACTION * self.n_particles))
        explorer_idx = np.random.choice(self.n_particles, n_explorers, replace=False)
        self.tempos[explorer_idx] = np.random.uniform(
            self.config.TEMPO_MIN, self.config.TEMPO_MAX, n_explorers
        )
        self.beat_phases[explorer_idx] = np.random.uniform(0.0, 1.0, n_explorers)

        # Phase advance
        beat_periods = 60.0 / self.tempos          # seconds per beat, (N,)
        phase_advance = ioi_sec / beat_periods      # (N,)

        self.beat_phases += phase_advance

        # Count complete beats that elapsed (could be 0 or 1, rarely 2+)
        beat_crossings = np.floor(self.beat_phases).astype(np.int32)
        self.meter_phases = (self.meter_phases + beat_crossings) % self.config.METER_NUMERATOR

        # Normalise phase to [0, 1)
        self.beat_phases = self.beat_phases % 1.0

    def _update_weights(self, ioi_sec: float) -> None:
        """Multiply particle weights by the observation likelihood."""
        likelihoods = compute_likelihood(
            ioi_sec, self.tempos, self.beat_phases, self.config
        )
        self.weights *= likelihoods
        total = self.weights.sum()
        if total > 0.0:
            self.weights /= total
        else:
            # Weight collapse — reset to uniform
            self.weights[:] = 1.0 / self.n_particles
            logger.warning("Weight collapse detected; resetting to uniform.")

    def _resample_if_needed(self) -> None:
        """Perform systematic resampling when ESS falls below the threshold."""
        ess = compute_ess(self.weights)
        threshold = self.n_particles * self.config.ESS_THRESHOLD_RATIO
        if ess < threshold:
            indices = systematic_resample(self.weights)
            self.tempos = self.tempos[indices]
            self.beat_phases = self.beat_phases[indices]
            self.meter_phases = self.meter_phases[indices]
            self.weights = np.ones(self.n_particles) / self.n_particles
            logger.debug("Resampled (ESS=%.1f < threshold=%.1f).", ess, threshold)

    def _estimate(self, processing_time_ms: float) -> EstimatorResult:
        """Compute weighted estimates from the current particle distribution.

        Tempo uses the weighted arithmetic mean.
        Beat phase uses the weighted circular mean via arctan2 to handle the
        [0, 1) wrap-around correctly.
        Measure beat uses the weighted mode of integer meter phases.
        Confidence is the normalised ESS.
        """
        # Tempo: weighted arithmetic mean
        tempo = float(np.dot(self.weights, self.tempos))
        tempo = round(tempo, self.config.TEMPO_PRECISION)

        # Beat phase: weighted circular mean
        angles = 2.0 * np.pi * self.beat_phases
        sin_mean = float(np.dot(self.weights, np.sin(angles)))
        cos_mean = float(np.dot(self.weights, np.cos(angles)))
        beat_position = float(np.arctan2(sin_mean, cos_mean) / (2.0 * np.pi)) % 1.0

        # Measure beat: weighted mode
        counts = np.bincount(
            self.meter_phases,
            weights=self.weights,
            minlength=self.config.METER_NUMERATOR,
        )
        measure_beat = int(np.argmax(counts))

        # Confidence: normalised ESS
        ess = compute_ess(self.weights)
        confidence = float(ess / self.n_particles)

        return EstimatorResult(
            tempo_bpm=tempo,
            beat_position=beat_position,
            measure_beat=measure_beat,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
        )
