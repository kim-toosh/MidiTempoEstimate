"""Unit tests for MidiInputGate classification and IOI computation."""

from __future__ import annotations

import midi_tempo_hmm.config as config
from midi_tempo_hmm.core.instrument_category import InstrumentCategory
from midi_tempo_hmm.core.midi_input_gate import MidiInputGate

_DRUM_CH = config.DRUM_CHANNEL  # 9


def _gate() -> MidiInputGate:
    return MidiInputGate(config)


def test_kick_classification() -> None:
    gate = _gate()
    for note in (35, 36):
        ev, *_ = gate.process(0.0, note, 100, _DRUM_CH)
        assert ev is not None
        assert ev.category == InstrumentCategory.KICK, f"note {note} should be KICK"


def test_snare_classification() -> None:
    gate = _gate()
    for note in (37, 38, 39, 40):
        ev, *_ = gate.process(0.0, note, 100, _DRUM_CH)
        assert ev is not None
        assert ev.category == InstrumentCategory.SNARE, f"note {note} should be SNARE"


def test_hihat_classification() -> None:
    gate = _gate()
    for note in (42, 44, 46):
        ev, *_ = gate.process(0.0, note, 100, _DRUM_CH)
        assert ev is not None
        assert ev.category == InstrumentCategory.HIHAT, f"note {note} should be HIHAT"


def test_others_classification() -> None:
    gate = _gate()
    ev, *_ = gate.process(0.0, 49, 100, _DRUM_CH)  # Crash cymbal
    assert ev is not None
    assert ev.category == InstrumentCategory.OTHERS


def test_ioi_same_category_only() -> None:
    """Kick → HiHat → Kick: only the second Kick produces a Kick IOI."""
    gate = _gate()

    _, ioi_kick1, _, _ = gate.process(0.0,  36, 100, _DRUM_CH)  # KICK
    _, ioi_hihat, _, _ = gate.process(0.25, 42, 100, _DRUM_CH)  # HIHAT
    _, ioi_kick2, _, _ = gate.process(1.0,  36, 100, _DRUM_CH)  # KICK

    assert ioi_kick1 is None, "First Kick should have no IOI"
    assert ioi_hihat is None, "First HiHat should have no IOI (separate from Kick)"
    assert ioi_kick2 is not None
    assert abs(ioi_kick2 - 1.0) < 1e-9, f"Kick→Kick IOI should be 1.0 s, got {ioi_kick2}"


def test_noteoff_returns_none() -> None:
    gate = _gate()
    ev, ioi, gcd_t, gcd_c = gate.process(0.0, 36, 0, _DRUM_CH)  # velocity=0 → NOTE_OFF
    assert ev is None
    assert ioi is None
    assert gcd_t is None
    assert gcd_c == 0.0


def test_non_drum_channel_returns_others() -> None:
    gate = _gate()
    ev, *_ = gate.process(0.0, 36, 100, channel=0)  # non-drum channel
    assert ev is not None
    assert ev.category == InstrumentCategory.OTHERS
    assert ev.drum_weight == 1.0, "Non-drum channel should have drum_weight=1.0"


def test_drum_weight_non_drum_channel() -> None:
    """Non-drum-channel events always get drum_weight=1.0 regardless of note."""
    gate = _gate()
    for note in (36, 38, 42, 49):
        ev, *_ = gate.process(0.0, note, 100, channel=0)
        assert ev.drum_weight == 1.0, f"note {note} on ch0 should have weight 1.0"


def test_event_counts_increment() -> None:
    gate = _gate()
    gate.process(0.0, 36, 100, _DRUM_CH)   # KICK
    gate.process(0.5, 36, 100, _DRUM_CH)   # KICK
    gate.process(0.25, 42, 100, _DRUM_CH)  # HIHAT
    stats = gate.get_stats()
    assert stats[InstrumentCategory.KICK] == 2
    assert stats[InstrumentCategory.HIHAT] == 1


def test_reset_clears_state() -> None:
    gate = _gate()
    gate.process(0.0, 36, 100, _DRUM_CH)
    gate.reset()
    stats = gate.get_stats()
    assert all(v == 0 for v in stats.values())
    # After reset, next Kick has no IOI
    _, ioi, _, _ = gate.process(0.0, 36, 100, _DRUM_CH)
    assert ioi is None


def test_gcd_tempo_after_enough_events() -> None:
    """4件のKick+Snareイベント後にGCDテンポが返される。"""
    gate = _gate()
    # 120 BPM: beat_period=0.5s, K at 0.0/1.0, S at 0.5/1.5
    gate.process(0.0, 36, 100, _DRUM_CH)   # K1 - no GCD yet
    gate.process(0.5, 38, 100, _DRUM_CH)   # S1 - 2 events
    gate.process(1.0, 36, 100, _DRUM_CH)   # K2 - 3 events
    _, _, gcd_t, gcd_c = gate.process(1.5, 38, 100, _DRUM_CH)  # S2 - 4 events
    assert gcd_t is not None, "GCD tempo should be estimated after 4 events"
    assert abs(gcd_t - 120.0) < 5.0, f"GCD tempo {gcd_t:.1f} not near 120 BPM"
    assert gcd_c >= 0.5, f"GCD confidence {gcd_c:.2f} too low"


def test_gcd_returns_none_before_min_events() -> None:
    """最低イベント数未満ではGCDがNoneを返す。"""
    gate = _gate()
    gate.process(0.0, 36, 100, _DRUM_CH)
    gate.process(0.5, 38, 100, _DRUM_CH)
    _, _, gcd_t, gcd_c = gate.process(1.0, 36, 100, _DRUM_CH)  # 3 events < GCD_MIN_EVENTS=4
    assert gcd_t is None
    assert gcd_c == 0.0


def test_reset_clears_gcd_state() -> None:
    """reset()後はGCD状態もクリアされる。"""
    gate = _gate()
    for ts, note in [(0.0, 36), (0.5, 38), (1.0, 36), (1.5, 38)]:
        gate.process(ts, note, 100, _DRUM_CH)
    gate.reset()
    assert gate.last_gcd_tempo is None
    assert gate.last_gcd_confidence == 0.0
    assert len(gate.gcd_timestamps) == 0
