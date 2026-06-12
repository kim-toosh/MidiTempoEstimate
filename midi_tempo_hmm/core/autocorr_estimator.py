"""Autocorrelation-based initial tempo estimator.

Before the particle filter converges, this module provides coarse tempo
hypotheses by analysing the pairwise time-differences of recent MIDI events.
The algorithm is a weighted histogram over all inter-event intervals, with
harmonic correction (k = 1..4) to handle events that arrive at sub-beat or
multi-beat intervals.

Drum instrument weights (from :mod:`~midi_tempo_hmm.core.drum_map`) are used
so that strong-beat instruments (kick, snare) contribute more to the estimate
than subdivision instruments (hi-hat, ride).  The weight map is injectable at
construction time and replaceable at runtime via :meth:`set_weight_map`, which
leaves room for future auto-generated or learned maps.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import ModuleType
from typing import Optional

import numpy as np

from midi_tempo_hmm.core.drum_map import get_drum_weight


@dataclass
class AutocorrResult:
    """Structured result from :meth:`AutocorrEstimator.estimate_result`.

    Attributes:
        tempo_candidates: BPM candidates sorted by score descending, with
                          harmonically-related peaks suppressed.
        scores: Normalised score for each candidate (top is always 1.0).
        best_tempo: BPM of the top candidate.
        confidence: ``scores[0] / sum(scores)`` — proportion of evidence
                    concentrated in the best candidate.  Approaches 1.0 for
                    unambiguous single-tempo input.
    """

    tempo_candidates: list[float]
    scores: list[float]
    best_tempo: float
    confidence: float


class AutocorrEstimator:
    """Estimate tempo from the autocorrelation of recent MIDI event timings.

    Args:
        config: Config module with AUTOCORR_BUFFER_SIZE, AUTOCORR_MIN_EVENTS,
                AUTOCORR_N_PEAKS, TEMPO_MIN, TEMPO_MAX.
        weight_map: Optional custom drum weight dict.  Pass ``None`` to use the
                    built-in GM map.  Can be updated later via
                    :meth:`set_weight_map` (e.g. to inject an auto-generated map).
    """

    # Harmonic multiples to check: an event at lag k beats produces
    # a pairwise delta of k * beat_period seconds.
    _HARMONICS: tuple[int, ...] = (1, 2, 3, 4)

    def __init__(self, config: ModuleType, weight_map: Optional[dict[int, float]] = None) -> None:
        self._config    = config
        self._weight_map: Optional[dict[int, float]] = weight_map

        # Buffer stores (timestamp, note, velocity, drum_weight)
        self._buffer: deque[tuple[float, int, int, float]] = deque(
            maxlen=config.AUTOCORR_BUFFER_SIZE
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def set_weight_map(self, weight_map: dict[int, float]) -> None:
        """Replace the drum weight map at runtime.

        Allows a learning algorithm to inject an auto-generated weight map
        without recreating the estimator.  The new map takes effect from the
        next :meth:`add_event` call onward; existing buffer entries keep their
        original weights.

        Args:
            weight_map: Note-number → weight dict.  Keys not present fall back
                        to :data:`~midi_tempo_hmm.core.drum_map.DEFAULT_DRUM_WEIGHT`.
        """
        self._weight_map = weight_map

    def add_event(self, timestamp: float, note: int, velocity: int) -> None:
        """Append a MIDI NOTE_ON event to the rolling buffer.

        The drum weight for *note* is computed immediately using the current
        weight map and stored alongside the event so that :meth:`estimate` does
        not need to re-evaluate it.

        Args:
            timestamp: Absolute timestamp in seconds (e.g. from
                       ``time.perf_counter()``).
            note: MIDI note number (0–127).
            velocity: MIDI velocity (1–127); stored but not currently used in
                      the estimation; reserved for velocity-weighted extensions.
        """
        weight = get_drum_weight(
            note, is_drum_channel=True, weight_map=self._weight_map
        )
        self._buffer.append((timestamp, note, velocity, weight))

    def estimate(self) -> list[tuple[float, float]]:
        """Estimate the most likely tempos from the current buffer.

        Returns:
            List of ``(bpm, normalized_strength)`` tuples sorted by strength
            descending, at most ``config.AUTOCORR_N_PEAKS`` entries.
            Returns an empty list if fewer than ``config.AUTOCORR_MIN_EVENTS``
            events are in the buffer.
        """
        if len(self._buffer) < self._config.AUTOCORR_MIN_EVENTS:
            return []

        timestamps = np.array([e[0] for e in self._buffer])  # (N,)
        weights    = np.array([e[3] for e in self._buffer])  # (N,)

        return self._histogram_autocorr(timestamps, weights)

    def estimate_result(self) -> Optional[AutocorrResult]:
        """Return a structured :class:`AutocorrResult` with harmonic suppression.

        Harmonically related peaks (octaves, 3/2 multiples) of the strongest
        candidate are removed before computing confidence, making the metric
        more meaningful for rhythmic tempo estimation.

        Returns:
            :class:`AutocorrResult`, or *None* when fewer than
            ``AUTOCORR_MIN_EVENTS`` events are in the buffer.
        """
        raw = self.estimate()
        if not raw:
            return None

        cleaned = self._suppress_harmonic_peaks(raw)
        tempo_candidates = [bpm for bpm, _ in cleaned]
        scores = [s for _, s in cleaned]
        total = sum(scores)
        confidence = scores[0] / total if total > 0 else 0.0

        return AutocorrResult(
            tempo_candidates=tempo_candidates,
            scores=scores,
            best_tempo=tempo_candidates[0],
            confidence=confidence,
        )

    def reset(self) -> None:
        """Clear the event buffer."""
        self._buffer.clear()

    # ── Algorithm ─────────────────────────────────────────────────────────────

    def _histogram_autocorr(
        self,
        timestamps: np.ndarray,
        weights: np.ndarray,
    ) -> list[tuple[float, float]]:
        """Weighted histogram autocorrelation.

        1. Compute all pairwise deltas Δ = t_j − t_i  (upper triangle, O(N²))
        2. Weight each pair by w_i * w_j (product of drum weights)
        3. For each harmonic k ∈ {1,2,3,4}: BPM = 60*k / Δ, clipped to
           [TEMPO_MIN, TEMPO_MAX], then accumulated into a 1-BPM-resolution
           weighted histogram
        4. Local maxima of the histogram become tempo candidates
        """
        bpm_min = self._config.TEMPO_MIN
        bpm_max = self._config.TEMPO_MAX

        # ── Pairwise deltas (vectorised) ──────────────────────────────────────
        t_col = timestamps[:, np.newaxis]   # (N, 1)
        t_row = timestamps[np.newaxis, :]   # (1, N)
        delta_mat = t_row - t_col           # (N, N); upper triangle has Δ > 0

        w_col   = weights[:, np.newaxis]
        w_row   = weights[np.newaxis, :]
        w_mat   = w_col * w_row             # (N, N) product weights

        upper = np.triu(np.ones_like(delta_mat, dtype=bool), k=1)
        deltas  = delta_mat[upper]          # (M,) where M = N*(N-1)/2
        w_pairs = w_mat[upper]              # (M,)

        if deltas.size == 0:
            return []

        # ── Histogram accumulation ────────────────────────────────────────────
        # BPM bins: integer boundaries from floor(bpm_min) to ceil(bpm_max)+1
        bin_lo  = int(np.floor(bpm_min))
        bin_hi  = int(np.ceil(bpm_max)) + 1
        bpm_bins = np.arange(bin_lo, bin_hi + 1, dtype=np.float64)   # edges
        hist    = np.zeros(len(bpm_bins) - 1)

        for k in self._HARMONICS:
            delta_lo = 60.0 * k / bpm_max
            delta_hi = 60.0 * k / bpm_min
            mask = (deltas >= delta_lo) & (deltas <= delta_hi)
            if not mask.any():
                continue
            bpms_k     = (60.0 * k) / deltas[mask]
            inv_w      = w_pairs[mask] / deltas[mask]   # inverse-delta: prefer short intervals
            h, _       = np.histogram(bpms_k, bins=bpm_bins, weights=inv_w)
            hist  += h

        if hist.max() == 0.0:
            return []

        # ── Peak detection ────────────────────────────────────────────────────
        # Local maxima: strictly greater than both neighbours
        is_peak = np.zeros(len(hist), dtype=bool)
        is_peak[1:-1] = (hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:])
        # Allow endpoints to be peaks
        if len(hist) > 1:
            is_peak[0]  = hist[0] > hist[1]
            is_peak[-1] = hist[-1] > hist[-2]

        peak_indices   = np.where(is_peak)[0]
        if peak_indices.size == 0:
            return []

        peak_bpms      = bin_lo + peak_indices + 0.5   # bin centres
        peak_strengths = hist[peak_indices]

        # Sort by strength descending and normalise
        order    = np.argsort(-peak_strengths)
        max_str  = peak_strengths[order[0]]
        results  = [
            (float(peak_bpms[i]), float(peak_strengths[i] / max_str))
            for i in order[: self._config.AUTOCORR_N_PEAKS]
        ]
        return results

    # Musical harmonic ratios to check for suppression (reference:candidate)
    _HARMONIC_RATIOS: tuple[float, ...] = (2.0, 0.5, 3.0, 1/3, 1.5, 2/3)
    _HARMONIC_TOLERANCE: float = 0.06  # 6% relative tolerance

    def _suppress_harmonic_peaks(
        self,
        candidates: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Remove candidates that are harmonic multiples of a stronger peak.

        Keeps the top candidate always.  Each subsequent candidate is removed
        if its BPM ratio to any already-kept candidate falls within 6% of one
        of the musical harmonic ratios (2x, 0.5x, 1.5x, 2/3x, 3x, 1/3x).
        """
        if len(candidates) <= 1:
            return candidates

        kept: list[tuple[float, float]] = [candidates[0]]
        for bpm, strength in candidates[1:]:
            is_harmonic = False
            for ref_bpm, _ in kept:
                ratio = bpm / ref_bpm
                for h in self._HARMONIC_RATIOS:
                    if abs(ratio - h) < self._HARMONIC_TOLERANCE * h:
                        is_harmonic = True
                        break
                if is_harmonic:
                    break
            if not is_harmonic:
                kept.append((bpm, strength))

        return kept
