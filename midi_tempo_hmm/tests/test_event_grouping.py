"""Tests for ParticleFilter's same-timing event grouping (delayed-average + flush API)."""

from __future__ import annotations

import midi_tempo_hmm.config as config
from midi_tempo_hmm.core.particle_filter import ParticleFilter

_DRUM_CH = config.DRUM_CHANNEL  # 9


def _pf() -> ParticleFilter:
    return ParticleFilter(config)


def test_group_finalizes_with_average_timestamp() -> None:
    """グループ確定時、平均タイムスタンプが次の同カテゴリIOIの基準になる。"""
    pf = _pf()

    # Kick(t=0.00) と HiHat(t=0.02) は span_same_time(0.05s)内 → 同一グループ
    assert pf.update(0.00, note_number=36, channel=_DRUM_CH) is None  # KICK
    assert pf.update(0.02, note_number=42, channel=_DRUM_CH) is None  # HIHAT

    # 0.50sのKickはspan外 → 旧グループ(avg_ts=0.01)を確定してから新グループ開始
    assert pf.update(0.50, note_number=36, channel=_DRUM_CH) is None
    assert pf.midi_gate._last_time[pf.midi_gate._classify(36, _DRUM_CH)] == 0.01

    # 孤立した最後のKickをflushで確定 → IOIはavg_ts(0.01)基準で0.49sになる
    result = pf.flush(0.50 + pf.span_same_time + 0.01)
    assert result is not None
    assert abs(result.ioi_sec - 0.49) < 1e-9, (
        f"ioi_sec should be computed against the group average (0.01s), got {result.ioi_sec}"
    )


def test_isolated_event_flush() -> None:
    """孤立イベントはspan_same_time経過後にflushで確定し、保留グループが空になる。"""
    pf = _pf()

    assert pf.update(0.00, note_number=36, channel=_DRUM_CH) is None
    assert len(pf._pending_group) == 1

    # span_same_time以内ではflushしても確定しない
    assert pf.flush(0.00 + pf.span_same_time * 0.5) is None
    assert len(pf._pending_group) == 1

    # span_same_time超過後はflushで確定（結果はIOIなしのためNoneだが保留グループは空になる）
    result = pf.flush(0.00 + pf.span_same_time + 0.01)
    assert result is None, "First-ever event has no IOI yet"
    assert pf._pending_group == []
    assert pf._pending_group_start is None


def test_same_note_repeat_not_grouped() -> None:
    """span_same_time内の同一ノート連続はグルーピングされず、IOIが0にならない。"""
    pf = _pf()

    assert pf.update(0.00, note_number=36, channel=_DRUM_CH) is None
    # 同じノート(36)が即座に再来 → 旧グループ(t=0.00のみ)を確定してから新グループ開始
    assert pf.update(0.02, note_number=36, channel=_DRUM_CH) is None
    assert len(pf._pending_group) == 1
    assert pf._pending_group[0][0] == 0.02

    # 3回目のKick → 旧グループ(t=0.02)を確定。ioi_sec=0.02s(< min_ioi)で結果はNone。
    assert pf.update(1.00, note_number=36, channel=_DRUM_CH) is None

    # 最後のグループ(t=1.00)をflushで確定。IOIは前回確定値(0.02)基準の0.98sになる。
    result = pf.flush(1.00 + pf.span_same_time + 0.01)
    assert result is not None
    assert abs(result.ioi_sec - 0.98) < 1e-9, (
        f"ioi_sec should be 1.00 - 0.02 = 0.98s (not forced to 0), got {result.ioi_sec}"
    )


def test_span_same_time_runtime_change() -> None:
    """span_same_timeを実行時に変更すると、グルーピングの窓が変わる。"""
    pf = _pf()
    pf.span_same_time = 0.1

    pf.update(0.00, note_number=36, channel=_DRUM_CH)
    pf.update(0.08, note_number=42, channel=_DRUM_CH)
    assert len(pf._pending_group) == 2, (
        "0.08s should be within the widened span_same_time=0.1"
    )

    # デフォルト(0.05)なら0.08sはspan外 → 別グループになる
    pf_default = _pf()
    pf_default.update(0.00, note_number=36, channel=_DRUM_CH)
    pf_default.update(0.08, note_number=42, channel=_DRUM_CH)
    assert len(pf_default._pending_group) == 1
    assert pf_default._pending_group[0][0] == 0.08


def test_reset_clears_pending_group() -> None:
    """reset()で保留グループがクリアされる。"""
    pf = _pf()
    pf.update(0.00, note_number=36, channel=_DRUM_CH)
    assert pf._pending_group != []

    pf.reset()
    assert pf._pending_group == []
    assert pf._pending_group_start is None
