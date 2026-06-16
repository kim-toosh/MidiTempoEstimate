"""Tests for the category-based observation likelihood model."""

from __future__ import annotations

import numpy as np
import pytest

import midi_tempo_hmm.config as config
from midi_tempo_hmm.core.instrument_category import InstrumentCategory
from midi_tempo_hmm.core.observation import compute_likelihood
from midi_tempo_hmm.core.particle_filter import ParticleFilter
from midi_tempo_hmm.interface.mock_input import generate_drum_pattern_events

_TEMPOS = np.linspace(config.TEMPO_MIN, config.TEMPO_MAX, 500)


def test_hihat_no_large_ratio() -> None:
    """HIHAT likelihood for a 1-beat IOI (ratio=1.0) must be lower than KICK likelihood."""
    ioi_sec = 60.0 / 120.0   # 1 beat at 120 BPM = 0.5 s

    lik_hihat = compute_likelihood(ioi_sec, _TEMPOS.copy(), InstrumentCategory.HIHAT, config, 1.0)
    lik_kick  = compute_likelihood(ioi_sec, _TEMPOS.copy(), InstrumentCategory.KICK,  config, 1.0)

    # Peak HIHAT likelihood should be lower because it has no ratio=1.0 term
    assert lik_hihat.max() < lik_kick.max(), (
        "HIHAT likelihood at 1-beat IOI should be lower than KICK"
    )


def test_kick_emphasizes_beat() -> None:
    """At the correct tempo, KICK ratio=1.0 should dominate the likelihood."""
    # Feed 1-beat IOI at 120 BPM → ratio=1.0 should be the strongest contributor
    ioi_sec = 60.0 / 120.0
    tempos = np.array([120.0])

    lik_at_correct  = compute_likelihood(ioi_sec, tempos, InstrumentCategory.KICK, config, 1.0)
    lik_at_double   = compute_likelihood(ioi_sec, np.array([240.0]), InstrumentCategory.KICK, config, 1.0)

    assert lik_at_correct[0] > lik_at_double[0], (
        "KICK likelihood should be higher at true tempo than at double tempo"
    )


def test_hihat_emphasizes_subdivision() -> None:
    """HIHAT should produce higher likelihood for 8th-note IOI than 1-beat IOI."""
    tempo = np.array([120.0])
    beat_period = 60.0 / 120.0        # 0.5 s

    ioi_eighth = beat_period * 0.5    # 8th note = ratio 0.5
    ioi_beat   = beat_period * 1.0    # 1 beat   = ratio 1.0 (not in HIHAT model)

    lik_eighth = compute_likelihood(ioi_eighth, tempo, InstrumentCategory.HIHAT, config, 1.0)
    lik_beat   = compute_likelihood(ioi_beat,   tempo, InstrumentCategory.HIHAT, config, 1.0)

    assert lik_eighth[0] > lik_beat[0], (
        "HIHAT should favor 8th-note IOI over beat IOI"
    )


def test_drum_weight_applied() -> None:
    """HIHAT_DRUM_WEIGHT(0.15) should scale likelihood vs KICK_DRUM_WEIGHT(1.0)."""
    ioi_sec = 60.0 / 120.0 * 0.5   # 8th note at 120 BPM
    tempos = _TEMPOS.copy()

    lik_kick  = compute_likelihood(ioi_sec, tempos, InstrumentCategory.KICK,  config,
                                   config.KICK_DRUM_WEIGHT)
    lik_hihat = compute_likelihood(ioi_sec, tempos, InstrumentCategory.HIHAT, config,
                                   config.HIHAT_DRUM_WEIGHT)

    assert lik_kick.max() > lik_hihat.max(), (
        "KICK (weight=1.0) should have higher peak likelihood than HIHAT (weight=0.15)"
    )


def test_category_based_accuracy() -> None:
    """Compare convergence speed with category-based vs default model (report only)."""
    np.random.seed(42)
    tempo_bpm = 120.0
    events = generate_drum_pattern_events(tempo_bpm, n_bars=16, pattern='basic_rock',
                                          humanize_ms=8.0)

    pf = ParticleFilter(config)
    results = []
    converge_event: int | None = None
    for i, (ts, note, ch) in enumerate(events):
        r = pf.update(ts, note_number=note, channel=ch)
        if r is not None:
            results.append(r)
            if r.is_converged and converge_event is None:
                converge_event = i + 1
    final = pf.flush(events[-1][0] + pf.span_same_time)
    if final is not None:
        results.append(final)
        if final.is_converged and converge_event is None:
            converge_event = len(events)

    assert len(results) > 0, "No results generated"
    final_avg = float(np.mean([r.tempo_bpm for r in results[-8:]]))
    error = abs(final_avg - tempo_bpm)

    print(
        f"\n[category_accuracy] convergence at event {converge_event}/{len(events)}"
        f"  final_avg={final_avg:.2f}  error={error:.2f} BPM"
    )
    assert error < 2.0, f"Category-based model error {error:.2f} BPM exceeds 2.0 BPM"
