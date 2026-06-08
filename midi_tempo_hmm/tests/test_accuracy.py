"""Accuracy tests for the particle filter tempo estimator."""

from __future__ import annotations

import numpy as np
import pytest

import midi_tempo_hmm.config as config
from midi_tempo_hmm.core.particle_filter import ParticleFilter
from midi_tempo_hmm.interface.mock_input import (
    generate_mock_events,
    generate_tempo_change_events,
)


def _run_filter(timestamps: list[float]) -> list:
    """Feed timestamps into a fresh ParticleFilter and return non-None results."""
    pf = ParticleFilter(config)
    results = []
    for ts in timestamps:
        r = pf.update(ts)
        if r is not None:
            results.append(r)
    return results


# ---------------------------------------------------------------------------


def test_steady_tempo_120() -> None:
    """Mean estimate over the last 8 beats must be within 0.5 BPM of 120."""
    np.random.seed(42)
    events = generate_mock_events(120.0, 32, humanize_ms=5.0)
    results = _run_filter(events)
    # Average the last 8 estimates; individual samples have inherent variance
    avg_tempo = float(np.mean([r.tempo_bpm for r in results[-8:]]))
    assert abs(avg_tempo - 120.0) < 0.5, (
        f"Expected ~120 BPM, last-8 average was {avg_tempo:.2f} BPM"
    )


def test_steady_tempo_various() -> None:
    """Estimator must converge to within 1.0 BPM for a range of tempos."""
    tempos = [60.0, 90.0, 120.0, 140.0, 180.0]
    for bpm in tempos:
        np.random.seed(0)
        events = generate_mock_events(bpm, 32, humanize_ms=5.0)
        results = _run_filter(events)
        # Average the last 8 estimates for robustness
        final = np.mean([r.tempo_bpm for r in results[-8:]])
        assert abs(final - bpm) < 1.0, (
            f"Tempo={bpm} BPM: estimate was {final:.2f} BPM"
        )


def test_tempo_change() -> None:
    """Estimator must track a 120→140 BPM change within 16 beats of the switch."""
    np.random.seed(7)
    # 16 beats to establish 120, then 24 beats at 140 for tracking + settling
    events = generate_tempo_change_events([120.0, 140.0], [16, 24], humanize_ms=5.0)
    results = _run_filter(events)

    # Last 8 results correspond to the tail of the 140-BPM segment
    tail = results[-8:]
    avg_tempo = np.mean([r.tempo_bpm for r in tail])
    assert abs(avg_tempo - 140.0) < 3.0, (
        f"After tempo change to 140 BPM, tail average was {avg_tempo:.2f} BPM"
    )


def test_humanized_input() -> None:
    """Estimator must handle ±10 ms human jitter and stay within 1.0 BPM."""
    np.random.seed(99)
    events = generate_mock_events(120.0, 32, humanize_ms=10.0)
    results = _run_filter(events)
    final = np.mean([r.tempo_bpm for r in results[-8:]])
    assert abs(final - 120.0) < 1.0, (
        f"With 10 ms humanization, estimate was {final:.2f} BPM"
    )


def test_beat_position_accuracy() -> None:
    """beat_position must always be in [0.0, 1.0) for every output."""
    np.random.seed(11)
    events = generate_mock_events(120.0, 32, humanize_ms=5.0)
    results = _run_filter(events)
    for i, r in enumerate(results):
        assert 0.0 <= r.beat_position < 1.0, (
            f"Event {i}: beat_position={r.beat_position} is outside [0, 1)"
        )
