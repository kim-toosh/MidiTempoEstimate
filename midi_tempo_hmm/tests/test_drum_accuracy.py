"""Accuracy tests for the particle filter with drum pattern input."""

from __future__ import annotations

import time
import types

import numpy as np
import pytest

import midi_tempo_hmm.config as config
from midi_tempo_hmm.core.particle_filter import ParticleFilter
from midi_tempo_hmm.interface.mock_input import generate_drum_pattern_events


def _run_filter_drum(events: list[tuple[float, int, int]]) -> list:
    """Feed (timestamp, note, channel) triples into a fresh ParticleFilter."""
    pf = ParticleFilter(config)
    results = []
    for ts, note, channel in events:
        r = pf.update(ts, note_number=note, channel=channel)
        if r is not None:
            results.append(r)
    return results


# ---------------------------------------------------------------------------
# Original accuracy tests (updated for 3-tuple event format)
# ---------------------------------------------------------------------------


def test_drum_basic_rock_120() -> None:
    """basic_rock pattern at 120 BPM: last-8 average within 2.0 BPM."""
    np.random.seed(42)
    events = generate_drum_pattern_events(120.0, n_bars=16, pattern='basic_rock',
                                          humanize_ms=8.0)
    results = _run_filter_drum(events)
    assert len(results) >= 8, "Too few results to evaluate"
    avg = float(np.mean([r.tempo_bpm for r in results[-8:]]))
    assert abs(avg - 120.0) < 2.0, (
        f"basic_rock 120 BPM: last-8 average was {avg:.2f} BPM"
    )


def test_drum_basic_rock_90() -> None:
    """basic_rock pattern at 90 BPM: last-8 average within 2.0 BPM."""
    np.random.seed(7)
    events = generate_drum_pattern_events(90.0, n_bars=16, pattern='basic_rock',
                                          humanize_ms=8.0)
    results = _run_filter_drum(events)
    assert len(results) >= 8, "Too few results to evaluate"
    avg = float(np.mean([r.tempo_bpm for r in results[-8:]]))
    assert abs(avg - 90.0) < 2.0, (
        f"basic_rock 90 BPM: last-8 average was {avg:.2f} BPM"
    )


def test_drum_straight_8th_140() -> None:
    """straight_8th pattern at 140 BPM: last-8 average within 3.0 BPM."""
    np.random.seed(99)
    events = generate_drum_pattern_events(140.0, n_bars=16, pattern='straight_8th',
                                          humanize_ms=8.0)
    results = _run_filter_drum(events)
    assert len(results) >= 8, "Too few results to evaluate"
    avg = float(np.mean([r.tempo_bpm for r in results[-8:]]))
    assert abs(avg - 140.0) < 3.0, (
        f"straight_8th 140 BPM: last-8 average was {avg:.2f} BPM"
    )


def test_drum_pattern_unknown_raises() -> None:
    """generate_drum_pattern_events raises ValueError for unknown pattern."""
    with pytest.raises(ValueError, match="Unknown pattern"):
        generate_drum_pattern_events(120.0, pattern='nonexistent')


def test_drum_beat_position_always_valid() -> None:
    """beat_position must stay in [0.0, 1.0) for every event."""
    np.random.seed(11)
    events = generate_drum_pattern_events(120.0, n_bars=8, humanize_ms=5.0)
    results = _run_filter_drum(events)
    for i, r in enumerate(results):
        assert 0.0 <= r.beat_position < 1.0, (
            f"Event {i}: beat_position={r.beat_position} outside [0, 1)"
        )


# ---------------------------------------------------------------------------
# New spec tests
# ---------------------------------------------------------------------------


def test_basic_rock_pattern_120() -> None:
    """basic_rock at 120 BPM with n_bars=8: error < 1.0 BPM and is_converged."""
    np.random.seed(42)
    events = generate_drum_pattern_events(120.0, n_bars=8, pattern='basic_rock',
                                          humanize_ms=8.0)
    results = _run_filter_drum(events)
    assert len(results) >= 8, "Too few results to evaluate"
    avg = float(np.mean([r.tempo_bpm for r in results[-8:]]))
    assert abs(avg - 120.0) < 1.0, (
        f"basic_rock 120 BPM: last-8 average was {avg:.2f} BPM"
    )
    assert results[-1].is_converged, "Filter should have converged after 8 bars"


def test_basic_rock_pattern_various() -> None:
    """basic_rock pattern at 80/100/120/140/160 BPM: error < 1.0 BPM each."""
    for bpm in [80.0, 100.0, 120.0, 140.0, 160.0]:
        np.random.seed(42)
        events = generate_drum_pattern_events(bpm, n_bars=16, pattern='basic_rock',
                                              humanize_ms=8.0, seed=42)
        results = _run_filter_drum(events)
        assert len(results) >= 8, f"Too few results at {bpm} BPM"
        avg = float(np.mean([r.tempo_bpm for r in results[-8:]]))
        assert abs(avg - bpm) < 1.0, (
            f"basic_rock {bpm} BPM: last-8 average was {avg:.2f} BPM"
        )


def test_hihat_16th_pattern() -> None:
    """hihat_16th pattern: 16th-note dense hi-hat, strong beats drive filter."""
    np.random.seed(55)
    events = generate_drum_pattern_events(120.0, n_bars=16, pattern='hihat_16th',
                                          humanize_ms=8.0)
    results = _run_filter_drum(events)
    assert len(results) >= 8, "Too few results to evaluate"
    avg = float(np.mean([r.tempo_bpm for r in results[-8:]]))
    assert abs(avg - 120.0) < 2.0, (
        f"hihat_16th 120 BPM: last-8 average was {avg:.2f} BPM"
    )


def test_sparse_pattern() -> None:
    """sparse pattern (2 events/bar): record events to convergence and report."""
    np.random.seed(13)
    events = generate_drum_pattern_events(120.0, n_bars=32, pattern='sparse',
                                          humanize_ms=8.0)
    pf = ParticleFilter(config)
    converge_event: int | None = None
    results = []
    for i, (ts, note, channel) in enumerate(events):
        r = pf.update(ts, note_number=note, channel=channel)
        if r is not None:
            results.append(r)
            if r.is_converged and converge_event is None:
                converge_event = i + 1

    print(f"\n[sparse] convergence at event {converge_event} / {len(events)}")
    # Sparse patterns may take many events but should not fail to process
    assert len(results) > 0, "No results generated from sparse pattern"


def test_drum_vs_nondrum_mode() -> None:
    """Drum-weight mode converges; compare with non-drum mode."""
    np.random.seed(42)
    events = generate_drum_pattern_events(120.0, n_bars=16, pattern='basic_rock',
                                          humanize_ms=8.0)

    # Drum mode
    pf_drum = ParticleFilter(config)
    converge_drum: int | None = None
    for i, (ts, note, ch) in enumerate(events):
        r = pf_drum.update(ts, note_number=note, channel=ch)
        if r is not None and r.is_converged and converge_drum is None:
            converge_drum = i + 1

    # Non-drum mode: disable drum weights (also disables autocorr)
    cfg_nodrum = types.SimpleNamespace(
        **{k: getattr(config, k) for k in dir(config) if not k.startswith('_')}
    )
    cfg_nodrum.USE_DRUM_WEIGHTS = False

    np.random.seed(42)
    pf_nodrum = ParticleFilter(cfg_nodrum)
    converge_nodrum: int | None = None
    for i, (ts, note, ch) in enumerate(events):
        r = pf_nodrum.update(ts, note_number=note, channel=ch)
        if r is not None and r.is_converged and converge_nodrum is None:
            converge_nodrum = i + 1

    print(f"\n[drum_vs_nondrum] drum={converge_drum}  nodrum={converge_nodrum}")
    # Drum mode should converge at some point
    assert converge_drum is not None, "Drum-weight mode did not converge"


def test_processing_time_drum() -> None:
    """Each drum pattern event must be processed in under 10 ms."""
    np.random.seed(0)
    events = generate_drum_pattern_events(120.0, n_bars=8, pattern='basic_rock',
                                          humanize_ms=0.0)
    pf = ParticleFilter(config)
    for ts, note, channel in events:
        t0 = time.perf_counter()
        pf.update(ts, note_number=note, channel=channel)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 10.0, f"Processing time {elapsed_ms:.2f} ms exceeds 10 ms"
