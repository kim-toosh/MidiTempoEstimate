"""Unit tests for the Kalman gate module."""

from __future__ import annotations

import math
import random

import pytest

import midi_tempo_hmm.config as config
from midi_tempo_hmm.core.kalman_gate import KalmanGate


# ── helpers ────────────────────────────────────────────────────────────────────

def _warm_up(gate: KalmanGate, tempo: float, n: int = 20, confidence: float = 0.8) -> None:
    """Feed *n* clean observations at *tempo* BPM to establish the Kalman mean."""
    for _ in range(n):
        gate.update([(tempo, confidence)])


# ── tests ──────────────────────────────────────────────────────────────────────

def test_accept_steady_tempo() -> None:
    """120 ± 1 BPM noise for 32 events → at least 90% accepted."""
    gate = KalmanGate(config)
    rng  = random.Random(42)
    n    = 32

    accepts = sum(
        1 for _ in range(n)
        if gate.update([(120.0 + rng.gauss(0.0, 1.0), 0.8)]).accepted
    )
    assert accepts / n >= 0.90, f"Accept rate {accepts/n:.2%} < 90%"


def test_reject_octave_jump() -> None:
    """240 BPM candidate while settled at 120 BPM → 'octave' reject."""
    gate = KalmanGate(config)
    _warm_up(gate, 120.0)
    result = gate.update([(240.0, 0.8)])
    assert result.accepted is False
    assert result.reject_reason == "octave"


def test_reject_mahal_outlier() -> None:
    """200 BPM candidate while settled at 120 BPM → 'mahal' reject (not octave)."""
    gate = KalmanGate(config)
    _warm_up(gate, 120.0)
    result = gate.update([(200.0, 0.8)])
    assert result.accepted is False
    assert result.reject_reason == "mahal"


def test_reject_low_confidence() -> None:
    """Low confidence (0.3) → 'confidence' reject on first call."""
    gate   = KalmanGate(config)
    result = gate.update([(120.0, 0.3)])
    assert result.accepted is False
    assert result.reject_reason == "confidence"


def test_tempo_rms_error_improvement() -> None:
    """Kalman-gated tempo stream should have lower std-dev than raw candidates."""
    gate = KalmanGate(config)
    rng  = random.Random(7)
    raw_list   : list[float] = []
    gated_list : list[float] = []

    for _ in range(60):
        # Mostly clean signal; occasional large outliers
        if rng.random() < 0.1:
            candidate = 120.0 + rng.choice([-60.0, 60.0])
        else:
            candidate = 120.0 + rng.gauss(0.0, 3.0)
        result = gate.update([(candidate, 0.8)])
        raw_list.append(candidate)
        gated_list.append(result.gated_tempo)

    def _std(vals: list[float]) -> float:
        mean = sum(vals) / len(vals)
        return math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))

    raw_std   = _std(raw_list)
    gated_std = _std(gated_list)
    assert gated_std < raw_std, (
        f"Gated std {gated_std:.3f} should be less than raw std {raw_std:.3f}"
    )


def test_variance_convergence() -> None:
    """Posterior variance should converge below KALMAN_R after steady observations."""
    gate = KalmanGate(config)
    result = None
    for _ in range(40):
        result = gate.update([(120.0, 0.8)])
    assert result is not None
    assert result.current_var < config.KALMAN_R, (
        f"Variance {result.current_var:.4f} did not converge below KALMAN_R={config.KALMAN_R}"
    )


def test_reset() -> None:
    """After reset(), gate should return to KALMAN_INIT_TEMPO with KALMAN_INIT_VAR."""
    gate = KalmanGate(config)
    _warm_up(gate, 120.0)

    gate.reset()

    assert gate._mean == config.KALMAN_INIT_TEMPO
    assert gate._var == config.KALMAN_INIT_VAR


def test_confidence_interval_initial() -> None:
    """At initialisation, CI should be centred on KALMAN_INIT_TEMPO."""
    gate = KalmanGate(config)
    lo, hi = gate.get_confidence_interval(sigma=2.0)
    mid = (lo + hi) / 2.0
    assert abs(mid - config.KALMAN_INIT_TEMPO) < 0.01
    # Width = 2 * sigma * sqrt(KALMAN_INIT_VAR) = 2*2*10 = 40 BPM
    assert abs((hi - lo) - 4.0 * math.sqrt(config.KALMAN_INIT_VAR)) < 0.01


def test_confidence_interval_converged() -> None:
    """After warm-up at 120 BPM, 2σ CI should be narrow (< 20 BPM wide)."""
    gate = KalmanGate(config)
    _warm_up(gate, 120.0, n=30)
    lo, hi = gate.get_confidence_interval(sigma=2.0)
    width = hi - lo
    assert width < 20.0, f"CI width {width:.2f} is too wide after convergence"
    assert lo < 120.0 < hi, f"120 BPM not inside CI ({lo:.2f}, {hi:.2f})"
