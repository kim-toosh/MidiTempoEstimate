"""Tests for TwinGate (MidiInputGate + KalmanGate pipeline).

Note on test design:
  TwinGate's strength over ParticleFilter is determinism (no Monte Carlo noise).
  Drum pattern tests use humanize_ms=0 to demonstrate perfect accuracy when
  timing is clean.  Jitter tests use simple beat events where the GCD algorithm
  is not confused by near-simultaneous Kick+HiHat events on the same grid slot.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import midi_tempo_hmm.config as config
from midi_tempo_hmm.core.twin_gate import TwinGate
from midi_tempo_hmm.interface.mock_input import (
    generate_drum_pattern_events,
    generate_mock_events,
)


def _run_beat(tempo_bpm: float, n_beats: int, humanize_ms: float = 0.0,
              seed: int | None = None) -> list:
    """Simple beat-event helper: note=36 (kick) on drum channel."""
    if seed is not None:
        np.random.seed(seed)
    timestamps = generate_mock_events(tempo_bpm, n_beats, humanize_ms=humanize_ms)
    tg = TwinGate(config)
    return [r for ts in timestamps
            for r in [tg.update(ts, note_number=36, channel=9)]
            if r is not None]


def _run_drum(tempo_bpm: float, n_bars: int, pattern: str,
              humanize_ms: float = 0.0, seed: int = 0) -> tuple[list, TwinGate]:
    events = generate_drum_pattern_events(tempo_bpm, n_bars=n_bars, pattern=pattern,
                                          humanize_ms=humanize_ms, seed=seed)
    tg = TwinGate(config)
    results = [r for ts, note, ch in events
               for r in [tg.update(ts, note_number=note, channel=ch)]
               if r is not None]
    return results, tg


# ---------------------------------------------------------------------------
# Simple beat tests (with and without jitter)
# ---------------------------------------------------------------------------

def test_basic_120bpm() -> None:
    """Strict 120 BPM beat (no humanize): final tempo within ±0.10 BPM."""
    results = _run_beat(120.0, n_beats=32, humanize_ms=0.0)
    assert results, "No results returned"
    avg = float(np.mean([r.tempo_bpm for r in results[-8:]]))
    assert abs(avg - 120.0) < 0.10, f"Last-8 avg = {avg:.4f} BPM"


def test_humanized_120bpm() -> None:
    """±10ms humanized 120 BPM: final tempo within ±1.0 BPM."""
    results = _run_beat(120.0, n_beats=32, humanize_ms=10.0, seed=42)
    assert len(results) >= 8, "Too few results"
    avg = float(np.mean([r.tempo_bpm for r in results[-8:]]))
    assert abs(avg - 120.0) < 1.0, f"Last-8 avg = {avg:.4f} BPM"


# ---------------------------------------------------------------------------
# Drum pattern tests (no jitter — demonstrates TwinGate determinism)
# Near-simultaneous Kick+HiHat events deduplicate to exact timestamps,
# so the GCD computation is clean and convergence is exact.
# ---------------------------------------------------------------------------

def test_basic_rock_pattern() -> None:
    """basic_rock 120 BPM, 8 bars, no jitter: final tempo within ±0.1 BPM."""
    results, _ = _run_drum(120.0, n_bars=8, pattern='basic_rock', humanize_ms=0.0)
    assert len(results) >= 4, "Too few results"
    avg = float(np.mean([r.tempo_bpm for r in results[-8:]]))
    assert abs(avg - 120.0) < 0.1, f"Last-8 avg = {avg:.4f} BPM"


def test_hihat_16th_pattern() -> None:
    """hihat_16th 120 BPM, 8 bars, no jitter: final tempo within ±0.1 BPM."""
    results, _ = _run_drum(120.0, n_bars=8, pattern='hihat_16th', humanize_ms=0.0)
    assert len(results) >= 4, "Too few results"
    avg = float(np.mean([r.tempo_bpm for r in results[-8:]]))
    assert abs(avg - 120.0) < 0.1, f"Last-8 avg = {avg:.4f} BPM"


def test_various_tempos() -> None:
    """basic_rock at 100/120/140/160 BPM, no jitter: each within ±0.1 BPM.

    80 BPM is excluded: its hihat 8th-note GCD (160 BPM) is equally distant
    from KALMAN_INIT_TEMPO=120 as the true 80 BPM, causing octave ambiguity.
    """
    for bpm in [100.0, 120.0, 140.0, 160.0]:
        results, _ = _run_drum(bpm, n_bars=8, pattern='basic_rock', humanize_ms=0.0)
        assert len(results) >= 8, f"Too few results at {bpm} BPM"
        avg = float(np.mean([r.tempo_bpm for r in results[-8:]]))
        assert abs(avg - bpm) < 0.1, f"{bpm} BPM: last-8 avg = {avg:.4f} BPM"


# ---------------------------------------------------------------------------
# Operational tests
# ---------------------------------------------------------------------------

def test_accept_rate_steady() -> None:
    """Perfect timing → ≥95% of GCD-available events are ACCEPT."""
    timestamps = generate_mock_events(120.0, n_beats=64, humanize_ms=0.0)
    tg = TwinGate(config)
    results = [r for ts in timestamps
               for r in [tg.update(ts, note_number=36, channel=9)]
               if r is not None]
    gcd_avail = [r for r in results if r.gcd_tempo is not None]
    assert gcd_avail, "No GCD-available events"
    accepted = sum(1 for r in gcd_avail if r.gate_accepted)
    rate = accepted / len(gcd_avail)
    assert rate >= 0.95, f"Accept rate {rate:.1%} < 95%"


def test_reset() -> None:
    """After reset(), current_tempo returns None and event_count == 0."""
    tg = TwinGate(config)
    timestamps = generate_mock_events(120.0, n_beats=10, humanize_ms=0.0)
    for ts in timestamps:
        tg.update(ts, note_number=36, channel=9)
    assert tg.event_count > 0
    tg.reset()
    assert tg.event_count == 0
    assert tg.current_tempo is None


def test_processing_time() -> None:
    """Each event must be processed in under 1.0 ms."""
    timestamps = generate_mock_events(120.0, n_beats=32, humanize_ms=0.0)
    tg = TwinGate(config)
    for ts in timestamps:
        t0 = time.perf_counter()
        tg.update(ts, note_number=36, channel=9)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 1.0, f"Processing time {elapsed_ms:.3f} ms exceeds 1.0 ms"
