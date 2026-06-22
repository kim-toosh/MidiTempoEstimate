"""Tests for PhaseOscillator."""

from __future__ import annotations

import pytest
import midi_tempo_hmm.config as config
from midi_tempo_hmm.core.phase_oscillator import PhaseOscillator


def _pco() -> PhaseOscillator:
    return PhaseOscillator(config)


PERIOD = 0.5      # 120 BPM
STRONG = config.PCO_ETA_PHASE_STRONG
WEAK   = config.PCO_ETA_PHASE_WEAK


def test_init_first_event() -> None:
    pco = _pco()
    r = pco.update(0.0, PERIOD, STRONG)
    assert r.phase == 0.0
    assert pco.period_sec == PERIOD
    assert r.phase_error == 0.0
    assert r.beat_count == 0


def test_phase_advances_correctly() -> None:
    pco = _pco()
    pco.update(0.0, PERIOD, 0.0)        # 初回：phase=0.0、補正なし
    r = pco.update(PERIOD / 2, PERIOD, 0.0)  # dt = 0.25s = half period
    assert abs(r.phase - 0.5) < 1e-9


def test_phase_sync_on_beat() -> None:
    """拍頭ちょうどのタイミングでイベントを続けるとis_synced=Trueになる。"""
    pco = _pco()
    pco.update(0.0, PERIOD, STRONG)
    for i in range(1, 20):
        r = pco.update(i * PERIOD, PERIOD, STRONG)
    assert r.is_synced
    assert abs(r.phase_error) < config.PCO_SYNC_THRESHOLD


def test_phase_correction_late_event() -> None:
    """拍頭より少し遅れて到着したイベントは位相が拍頭方向に引き込まれる。"""
    pco = _pco()
    pco.update(0.0, PERIOD, 0.0)
    # dt = PERIOD + 0.05 → 自然進行後 phase = 0.1（遅れ）
    r = pco.update(PERIOD + 0.05, PERIOD, STRONG)
    # 補正後は 0.1 - STRONG*0.1 = 0.1 * (1 - STRONG) → 減少している
    assert r.phase < 0.1


def test_phase_correction_early_event() -> None:
    """拍頭より少し早く到着したイベントは位相が次拍頭方向に補正される。"""
    pco = _pco()
    pco.update(0.0, PERIOD, 0.0)
    # dt = PERIOD - 0.05 → 自然進行後 phase = 0.9（早い → error = -0.1）
    r = pco.update(PERIOD - 0.05, PERIOD, STRONG)
    # 補正後: phase = 0.9 - STRONG*(-0.1) = 0.9 + STRONG*0.1 → 増加（1.0に近づく）
    assert r.phase > 0.9


def test_next_beat_prediction() -> None:
    """phase=0.5 のとき next_beat_time = ts + PERIOD/2 (補正なしの場合)。"""
    pco = _pco()
    pco.update(0.0, PERIOD, 0.0)
    # dt = PERIOD/2 → phase=0.5 (補正なし)
    r = pco.update(PERIOD / 2, PERIOD, 0.0)
    assert r.next_beat_time is not None
    expected = PERIOD / 2 + (1.0 - r.phase) * PERIOD
    assert abs(r.next_beat_time - expected) < 1e-9


def test_beat_count_increments() -> None:
    """dt = 1.5 * period → 1拍分を通過するので beat_count が1増加する。"""
    pco = _pco()
    pco.update(0.0, PERIOD, 0.0)
    r = pco.update(1.5 * PERIOD, PERIOD, 0.0)
    assert r.beat_count == 1


def test_sync_strength_effect() -> None:
    """sync_strengthが大きいほど位相補正量が大きい。"""
    def _correction(strength: float) -> float:
        pco = _pco()
        pco.update(0.0, PERIOD, 0.0)
        # dt=PERIOD+0.05 → phase≈0.1（遅れ）
        r = pco.update(PERIOD + 0.05, PERIOD, strength)
        return r.phase  # 補正後のphase（小さいほど補正が大きい）

    phase_strong = _correction(STRONG)
    phase_weak   = _correction(WEAK)
    assert phase_strong < phase_weak  # strongの方が拍頭に近い（位相が小さい）


def test_reset() -> None:
    pco = _pco()
    pco.update(0.0, PERIOD, STRONG)
    pco.update(PERIOD, PERIOD, STRONG)
    pco.reset()
    assert pco.phase      == 0.0
    assert pco.period_sec is None
    assert pco.last_time  is None
    assert pco.beat_count == 0
    assert pco.is_synced  is False
    assert len(pco._error_history) == 0


def test_full_pipeline_phase_tracking() -> None:
    """TwinGate経由でbasic_rockを処理し、収束後is_phase_synced=True になること。
    またKickイベント到着時のphaseが拍頭付近（PCO_SYNC_THRESHOLDの2倍以内）になること。
    """
    from midi_tempo_hmm.core.twin_gate import TwinGate
    from midi_tempo_hmm.core.instrument_category import InstrumentCategory
    from midi_tempo_hmm.interface.mock_input import generate_drum_pattern_events

    tg = TwinGate(config)
    events = generate_drum_pattern_events(120.0, n_bars=8, pattern='basic_rock',
                                           humanize_ms=5.0)
    results = []
    for ts, note, ch in events:
        r = tg.update(ts, note_number=note, velocity=100, channel=ch)
        if r is not None:
            results.append(r)

    assert results, "イベントが処理されていない"

    # 後半の結果で確認（収束後）
    tail = results[len(results) // 2:]
    synced = [r for r in tail if r.is_phase_synced]
    assert len(synced) > 0, "収束後もis_phase_synced=Trueになるフレームがない"

    # Kickイベントで位相が拍頭付近（誤差が SYNC_THRESHOLD*2 以内）かを確認
    kick_results = [r for r in tail
                    if r.category == InstrumentCategory.KICK
                    and r.phase is not None and r.phase_error is not None]
    assert kick_results, "後半にKickイベントがない"
    kick_errors = [abs(r.phase_error) for r in kick_results]
    assert min(kick_errors) < config.PCO_SYNC_THRESHOLD * 2
