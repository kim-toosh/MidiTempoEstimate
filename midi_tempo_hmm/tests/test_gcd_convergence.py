"""Convergence speed tests for GCD-accelerated tempo estimation."""

from __future__ import annotations

import numpy as np
import pytest

import midi_tempo_hmm.config as config
from midi_tempo_hmm.core.autocorr_estimator import AutocorrEstimator
from midi_tempo_hmm.core.instrument_category import InstrumentCategory
from midi_tempo_hmm.core.particle_filter import ParticleFilter
from midi_tempo_hmm.interface.mock_input import generate_drum_pattern_events


def test_convergence_within_4_events() -> None:
    """120BPMのbasic_rockでKick+Snareが4個揃った時点でgcd_confidence >= 閾値になる。"""
    np.random.seed(42)
    events = generate_drum_pattern_events(120.0, n_bars=4, pattern='basic_rock', humanize_ms=8.0)
    pf = ParticleFilter(config)

    gcd_reach_event: int | None = None
    for i, (ts, note, channel) in enumerate(events):
        pf.update(ts, note_number=note, channel=channel)
        total_events = len(pf.midi_gate.gcd_timestamps)
        gcd_conf = (pf.midi_gate.last_gcd_candidates[0][1]
                    if pf.midi_gate.last_gcd_candidates else 0.0)
        if (gcd_reach_event is None
                and total_events >= config.GCD_MIN_EVENTS
                and gcd_conf >= config.GCD_CONFIDENCE_THRESHOLD):
            gcd_reach_event = i + 1

    assert gcd_reach_event is not None, (
        "GCD should reach confidence threshold after GCD_MIN_EVENTS K+S events"
    )
    assert gcd_reach_event <= 20, (
        f"GCD should reach threshold quickly, took {gcd_reach_event} events"
    )


def test_convergence_hihat_16th() -> None:
    """120BPMのhihat_16thでGCD候補リスト内に120±4.5 BPMの候補が含まれる。

    GCDはKick/Snareタイムスタンプのみを使用するため hihat_16th でも
    0.5s IOI（120BPM）が正しく検出される。
    estimate_tempo_from_timestamps_multi はオクターブ展開済みの候補を返すため、
    gcd_candidates のいずれかが120 BPM付近にあることを確認する。
    """
    np.random.seed(0)
    events = generate_drum_pattern_events(120.0, n_bars=4, pattern='hihat_16th', humanize_ms=8.0, seed=0)
    pf = ParticleFilter(config)

    found_near_120: bool = False
    for ts, note, channel in events:
        pf.update(ts, note_number=note, channel=channel)
        cands = pf.midi_gate.last_gcd_candidates
        if (cands and cands[0][1] >= config.GCD_CONFIDENCE_THRESHOLD
                and any(abs(t - 120.0) < 4.5 for t, _ in cands)):
            found_near_120 = True
            break

    assert found_near_120, "GCD candidates should contain ~120 BPM for hihat_16th"


def test_gcd_vs_autocorr_speed() -> None:
    """GCDはK+S 4イベントで確信度に達し、Autocorrは全イベント数がそれ以上必要。

    比較基準: GCDが使用するK+Sイベント数 vs Autocorrが使用する全イベント数。
    GCDはKick+Snareのみで動作するため、全楽器を使うAutocorrより
    少ない「強拍イベント」で収束できることを確認する。
    """
    np.random.seed(42)
    events = generate_drum_pattern_events(120.0, n_bars=8, pattern='basic_rock', humanize_ms=8.0)

    pf       = ParticleFilter(config)
    autocorr = AutocorrEstimator(config)

    gcd_ks_reach:        int | None = None  # GCDが閾値到達時のK+Sイベント累積数
    autocorr_total_reach: int | None = None  # Autocorrが閾値到達時の全イベント数

    for i, (ts, note, channel) in enumerate(events):
        pf.update(ts, note_number=note, channel=channel)
        autocorr.add_event(ts, note, 100)

        # GCDが閾値到達した時点のK+Sイベント数を記録
        _gcd_conf = (pf.midi_gate.last_gcd_candidates[0][1]
                     if pf.midi_gate.last_gcd_candidates else 0.0)
        if (gcd_ks_reach is None
                and _gcd_conf >= config.GCD_CONFIDENCE_THRESHOLD):
            ks = (pf.midi_gate.event_count_by_category.get(InstrumentCategory.KICK, 0)
                  + pf.midi_gate.event_count_by_category.get(InstrumentCategory.SNARE, 0))
            gcd_ks_reach = ks

        if autocorr_total_reach is None:
            ar = autocorr.estimate_result()
            if ar is not None and ar.confidence >= config.AUTOCORR_CONFIDENCE_THRESHOLD:
                autocorr_total_reach = i + 1

    assert gcd_ks_reach is not None, "GCD should reach confidence threshold"
    assert autocorr_total_reach is not None, "Autocorr should reach confidence threshold"
    assert gcd_ks_reach < autocorr_total_reach, (
        f"GCD uses only {gcd_ks_reach} K+S events; "
        f"autocorr needs {autocorr_total_reach} total events"
    )


def test_various_tempos() -> None:
    """[80,100,120,140,160] BPMそれぞれでGCD候補リスト内に±6%以内の候補が含まれる。

    estimate_tempo_from_timestamps_multi はオクターブ展開済み候補を返すため、
    last_gcd_candidates のいずれかが目標BPMの±6%以内にあることを確認する。
    """
    for bpm in [80.0, 100.0, 120.0, 140.0, 160.0]:
        np.random.seed(0)
        events = generate_drum_pattern_events(bpm, n_bars=4, pattern='basic_rock', humanize_ms=8.0, seed=0)
        pf = ParticleFilter(config)

        found: bool = False
        for ts, note, channel in events:
            pf.update(ts, note_number=note, channel=channel)
            cands = pf.midi_gate.last_gcd_candidates
            if (cands and cands[0][1] >= config.GCD_CONFIDENCE_THRESHOLD
                    and any(abs(t - bpm) < bpm * 0.06 for t, _ in cands)):
                found = True
                break

        assert found, (
            f"GCD candidates should contain a tempo within 6% of {bpm} BPM. "
            f"Last candidates: {pf.midi_gate.last_gcd_candidates}"
        )


def test_full_pipeline_basic_rock() -> None:
    """basic_rock 1小節以内にParticleFilterが収束し最終テンポが120±3.0 BPM以内。"""
    np.random.seed(42)
    events = generate_drum_pattern_events(120.0, n_bars=1, pattern='basic_rock', humanize_ms=8.0, seed=42)
    pf = ParticleFilter(config)

    results = []
    for ts, note, channel in events:
        r = pf.update(ts, note_number=note, channel=channel)
        if r is not None:
            results.append(r)
    final = pf.flush(events[-1][0] + pf.span_same_time)
    if final is not None:
        results.append(final)

    assert len(results) > 0, "Should produce at least one result"
    assert results[-1].is_converged, (
        f"Filter should converge within 1 bar of basic_rock; "
        f"last confidence={results[-1].confidence:.3f}"
    )
    assert abs(results[-1].tempo_bpm - 120.0) < 3.0, (
        f"Final tempo {results[-1].tempo_bpm:.2f} not within 3 BPM of 120"
    )


def test_full_pipeline_hihat_16th() -> None:
    """hihat_16th 2小節以内に収束し最終テンポが120±3.0 BPM以内。"""
    np.random.seed(55)
    events = generate_drum_pattern_events(120.0, n_bars=2, pattern='hihat_16th', humanize_ms=8.0)
    pf = ParticleFilter(config)

    results = []
    for ts, note, channel in events:
        r = pf.update(ts, note_number=note, channel=channel)
        if r is not None:
            results.append(r)
    final = pf.flush(events[-1][0] + pf.span_same_time)
    if final is not None:
        results.append(final)

    assert len(results) > 0, "Should produce at least one result"
    assert results[-1].is_converged, (
        f"Filter should converge within 2 bars of hihat_16th; "
        f"last confidence={results[-1].confidence:.3f}"
    )
    assert abs(results[-1].tempo_bpm - 120.0) < 3.0, (
        f"Final tempo {results[-1].tempo_bpm:.2f} not within 3 BPM of 120"
    )
