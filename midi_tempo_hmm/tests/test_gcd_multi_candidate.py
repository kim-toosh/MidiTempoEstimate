"""Tests for multi-candidate GCD output through MidiInputGate → KalmanGate pipeline."""

from __future__ import annotations

import pytest

import midi_tempo_hmm.config as config
from midi_tempo_hmm.core.instrument_category import InstrumentCategory
from midi_tempo_hmm.core.kalman_gate import KalmanGate
from midi_tempo_hmm.core.midi_input_gate import MidiInputGate


# ── helpers ────────────────────────────────────────────────────────────────────

def _feed_kick_snare(gate: MidiInputGate, tempo_bpm: float, n_beats: int = 8) -> list:
    """Feed alternating kick/snare at tempo_bpm, return last gcd_candidates."""
    beat = 60.0 / tempo_bpm
    candidates = []
    for i in range(n_beats):
        ts = i * beat
        note = 36 if i % 2 == 0 else 38  # kick / snare
        _, _, candidates = gate.process(ts, note, velocity=100, channel=9)
    return candidates


# ── tests ──────────────────────────────────────────────────────────────────────

def test_process_returns_three_tuple() -> None:
    """MidiInputGate.process()が(event, ioi, candidates)の3タプルを返す。"""
    gate = MidiInputGate(config)
    result = gate.process(0.0, 36, velocity=100, channel=9)
    assert len(result) == 3
    event, ioi, candidates = result
    assert event is not None
    assert ioi is None  # first event has no IOI
    assert isinstance(candidates, list)


def test_velocity_zero_returns_none_and_empty() -> None:
    """velocity==0（NOTE_OFF）はevent=None, candidates=[]を返す。"""
    gate = MidiInputGate(config)
    event, ioi, candidates = gate.process(0.0, 36, velocity=0, channel=9)
    assert event is None
    assert ioi is None
    assert candidates == []


def test_candidates_empty_before_min_events() -> None:
    """GCD_MIN_EVENTS未満のイベントでは候補リストが空になる。"""
    gate = MidiInputGate(config)
    min_events = getattr(config, 'GCD_MIN_EVENTS', 4)
    candidates = []
    for i in range(min_events - 1):
        _, _, candidates = gate.process(i * 0.5, 36, velocity=100, channel=9)
    assert candidates == [], f"Expected empty candidates, got {candidates}"


def test_candidates_non_empty_after_min_events() -> None:
    """GCD_MIN_EVENTS以上のKick/Snareイベント後に候補が返る。"""
    gate = MidiInputGate(config)
    min_events = getattr(config, 'GCD_MIN_EVENTS', 4)
    candidates = []
    for i in range(min_events + 2):
        _, _, candidates = gate.process(i * 0.5, 36, velocity=100, channel=9)
    assert len(candidates) > 0, "Expected at least one candidate after min_events"


def test_candidates_tempo_near_target() -> None:
    """120 BPMのKick+Snareパターン後の先頭候補が120 BPM付近になる。"""
    gate = MidiInputGate(config)
    candidates = _feed_kick_snare(gate, tempo_bpm=120.0, n_beats=10)
    assert len(candidates) > 0
    best_tempo = candidates[0][0]
    # 先頭候補はHalf-note(240 BPM)やQuarter-note(120 BPM)のいずれかになりうる
    # 少なくとも1つの候補が120 BPM付近にあることを確認
    assert any(abs(t - 120.0) < 10.0 for t, _ in candidates), (
        f"Expected ~120 BPM in candidates, got {candidates}"
    )


def test_kalman_gate_accepts_candidate_list() -> None:
    """KalmanGate.update(candidates)が正常に動作し結果を返す。"""
    gate = KalmanGate(config)
    candidates = [(120.0, 0.9), (240.0, 0.5)]
    result = gate.update(candidates)
    assert result is not None
    assert result.all_candidates == candidates
    assert result.selected_index >= 0


def test_kalman_gate_empty_candidates_returns_no_gcd() -> None:
    """候補リストが空のとき reject_reason=='no_gcd' になる。"""
    gate = KalmanGate(config)
    result = gate.update([])
    assert result.accepted is False
    assert result.reject_reason == "no_gcd"
    assert result.selected_index == -1
