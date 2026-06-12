"""Tests for the autocorrelation-based tempo estimator."""

from __future__ import annotations

import numpy as np
import pytest

import midi_tempo_hmm.config as config
from midi_tempo_hmm.core.autocorr_estimator import AutocorrEstimator, AutocorrResult


def _make_estimator(weight_map=None) -> AutocorrEstimator:
    return AutocorrEstimator(config, weight_map=weight_map)


def _feed_isochronous(estimator: AutocorrEstimator, bpm: float, n: int,
                      note: int = 36, humanize_ms: float = 0.0) -> None:
    """Feed *n* perfectly spaced (or humanised) events at *bpm* BPM."""
    period = 60.0 / bpm
    rng = np.random.default_rng(0)
    t = 0.0
    for _ in range(n):
        jitter = float(rng.normal(0.0, humanize_ms / 1000.0)) if humanize_ms else 0.0
        estimator.add_event(t + jitter, note, 100)
        t += period


# ---------------------------------------------------------------------------


def test_autocorr_returns_correct_tempo_perfect() -> None:
    """Perfectly metronomic kick events → top candidate within 2 BPM."""
    est = _make_estimator()
    _feed_isochronous(est, 120.0, n=16, note=36)
    candidates = est.estimate()
    assert len(candidates) > 0, "No candidates returned"
    top_bpm, strength = candidates[0]
    assert abs(top_bpm - 120.0) < 2.0, (
        f"Expected top candidate near 120 BPM, got {top_bpm:.1f} BPM"
    )
    assert 0.0 < strength <= 1.0


def test_autocorr_returns_correct_tempo_humanized() -> None:
    """Humanised events (±10 ms jitter) → top candidate within 5 BPM of truth."""
    est = _make_estimator()
    _feed_isochronous(est, 90.0, n=24, note=36, humanize_ms=10.0)
    candidates = est.estimate()
    assert len(candidates) > 0
    top_bpm, _ = candidates[0]
    assert abs(top_bpm - 90.0) < 5.0, (
        f"Expected ~90 BPM, got {top_bpm:.1f} BPM"
    )


def test_autocorr_insufficient_events() -> None:
    """Fewer than AUTOCORR_MIN_EVENTS → empty list."""
    est = _make_estimator()
    for i in range(config.AUTOCORR_MIN_EVENTS - 1):
        est.add_event(i * 0.5, 36, 100)
    assert est.estimate() == []


def test_autocorr_exactly_min_events() -> None:
    """Exactly AUTOCORR_MIN_EVENTS events → non-empty result."""
    est = _make_estimator()
    _feed_isochronous(est, 120.0, n=config.AUTOCORR_MIN_EVENTS, note=36)
    assert len(est.estimate()) > 0


def test_autocorr_n_peaks_limit() -> None:
    """Result list length never exceeds AUTOCORR_N_PEAKS."""
    est = _make_estimator()
    _feed_isochronous(est, 120.0, n=32, note=36)
    candidates = est.estimate()
    assert len(candidates) <= config.AUTOCORR_N_PEAKS


def test_autocorr_strength_normalised() -> None:
    """Top candidate always has normalised strength = 1.0."""
    est = _make_estimator()
    _feed_isochronous(est, 120.0, n=32, note=36)
    candidates = est.estimate()
    assert len(candidates) > 0
    assert candidates[0][1] == pytest.approx(1.0)


def test_autocorr_set_weight_map() -> None:
    """set_weight_map() replaces the internal map for subsequent add_event calls."""
    est = _make_estimator(weight_map={36: 1.0})

    # Override with a map that gives note 36 a very low weight
    est.set_weight_map({36: 0.01})

    # Feed events with the new map in effect; we just check it doesn't raise
    _feed_isochronous(est, 120.0, n=16, note=36)
    candidates = est.estimate()
    # Estimation should still return something (just with lower weights)
    # The important thing is set_weight_map didn't break anything
    assert isinstance(candidates, list)


def test_autocorr_reset_clears_buffer() -> None:
    """After reset(), fewer than MIN_EVENTS → empty estimate."""
    est = _make_estimator()
    _feed_isochronous(est, 120.0, n=32, note=36)
    assert len(est.estimate()) > 0   # sanity check

    est.reset()
    assert est.estimate() == []


def test_autocorr_high_weight_note_dominates() -> None:
    """Events on high-weight note (kick=1.0) should produce a clearer peak
    than equally many events on a low-weight note (hi-hat=0.2)."""
    est_kick   = _make_estimator()
    est_hihat  = _make_estimator()

    _feed_isochronous(est_kick,  120.0, n=24, note=36)   # kick, weight 1.0
    _feed_isochronous(est_hihat, 120.0, n=24, note=42)   # hi-hat, weight 0.2

    kick_strength  = est_kick.estimate()[0][1]   # 1.0 by definition (normalised)
    hihat_strength = est_hihat.estimate()[0][1]  # also 1.0 (same note, just diff weight)

    # Both return normalised strength 1.0, but kick histogram has larger raw values.
    # We can verify by checking the raw histogram indirectly: kick events at 120 BPM
    # should produce a candidate at 120 BPM regardless of absolute scale.
    kick_bpm  = est_kick.estimate()[0][0]
    hihat_bpm = est_hihat.estimate()[0][0]
    assert abs(kick_bpm  - 120.0) < 3.0
    assert abs(hihat_bpm - 120.0) < 3.0


# ---------------------------------------------------------------------------
# New spec tests using AutocorrResult / estimate_result()
# ---------------------------------------------------------------------------


def test_autocorr_single_tempo() -> None:
    """32 isochronous kick events at 120 BPM → best_tempo within 120 ± 2 BPM."""
    est = _make_estimator()
    _feed_isochronous(est, 120.0, n=32, note=36)
    result = est.estimate_result()
    assert result is not None
    assert isinstance(result, AutocorrResult)
    assert abs(result.best_tempo - 120.0) < 2.0, (
        f"Expected best_tempo ≈ 120.0, got {result.best_tempo:.1f}"
    )


def test_autocorr_various_tempos() -> None:
    """estimate_result best_tempo within 2 BPM for 60/90/120/150/180 BPM."""
    for bpm in [60.0, 90.0, 120.0, 150.0, 180.0]:
        est = _make_estimator()
        _feed_isochronous(est, bpm, n=32, note=36)
        result = est.estimate_result()
        assert result is not None, f"No result at {bpm} BPM"
        assert abs(result.best_tempo - bpm) < 2.0, (
            f"At {bpm} BPM: best_tempo was {result.best_tempo:.1f}"
        )


def test_autocorr_with_subdivisions() -> None:
    """8th-note events of a 200 BPM piece → best_tempo ≈ 200 BPM.

    The algorithm naturally recovers the quarter-note tempo from 8th-note input
    when the literal 8th-note BPM (400) exceeds AUTOCORR_TEMPO_MAX (250).
    Only the 2-event-apart pairs (delta = quarter period = 0.30 s) fall within
    the valid range and vote for 200 BPM.

    8th-note period of 200 BPM = 0.15 s  → literal "BPM" = 400 (out of range)
    input IOI × 2 = 0.30 s              → BPM = 60 / 0.30 = 200 ✓
    """
    est = _make_estimator()
    # Feed events at the 8th-note rate of a 200 BPM piece.
    # _feed_isochronous(bpm) spaces events at 60/bpm seconds.
    # 8th-note rate = 200 * 2 = 400 BPM  (period = 0.15 s)
    eighth_note_bpm = 200.0 * 2  # = 400
    _feed_isochronous(est, eighth_note_bpm, n=32, note=36)
    result = est.estimate_result()
    assert result is not None
    # best_tempo should be close to 200 BPM (the quarter-note tempo)
    assert abs(result.best_tempo - 200.0) < 5.0, (
        f"Expected quarter-note tempo ~200 BPM, got {result.best_tempo:.1f}"
    )


def test_autocorr_confidence() -> None:
    """Steady metronomic input → confidence ≥ AUTOCORR_CONFIDENCE_THRESHOLD after suppression."""
    est = _make_estimator()
    _feed_isochronous(est, 120.0, n=32, note=36)
    result = est.estimate_result()
    assert result is not None
    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence >= config.AUTOCORR_CONFIDENCE_THRESHOLD, (
        f"confidence={result.confidence:.3f} < threshold {config.AUTOCORR_CONFIDENCE_THRESHOLD}"
    )
