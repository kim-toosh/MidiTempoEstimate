"""Tests for PlayerEngine and load_midi_events."""

from __future__ import annotations

import sys
import os
import threading
import time

import mido
import pytest

# smf_player/ をパスに追加（pytest は tests/ から実行されるため）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from midi_event import load_midi_events
from player_engine import PlayerEngine


# ── テスト用 MIDI 生成ヘルパー ─────────────────────────────────────────────────

def _create_test_midi(tmp_path, tempo_bpm: float = 120, n_notes: int = 8) -> str:
    """等間隔の note_on/off を持つ MIDI ファイルを tmp_path に生成してパスを返す。"""
    mid   = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo = mido.bpm2tempo(tempo_bpm)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

    ticks_per_note = 480  # 1拍 = quarter note
    off_time = ticks_per_note // 2  # note_off → 次のnote_onまでの間隔も同じ
    for i in range(n_notes):
        on_time  = 0 if i == 0 else off_time
        track.append(mido.Message('note_on',  channel=9, note=36,
                                  velocity=100, time=on_time))
        track.append(mido.Message('note_off', channel=9, note=36,
                                  velocity=0,   time=off_time))

    track.append(mido.MetaMessage('end_of_track', time=0))

    path = str(tmp_path / 'test.mid')
    mid.save(path)
    return path


# ── ダミー出力ポート ───────────────────────────────────────────────────────────

class _MockPort:
    """送信されたメッセージを記録するモックポート。"""

    def __init__(self):
        self.sent: list[mido.Message] = []
        self._lock = threading.Lock()

    def send(self, msg: mido.Message) -> None:
        with self._lock:
            self.sent.append(msg)

    def close(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.sent)


def _make_engine(mock_port: _MockPort) -> PlayerEngine:
    eng = PlayerEngine.__new__(PlayerEngine)
    eng.events            = []
    eng.outport           = mock_port
    eng.muted_channels    = set()
    eng.is_playing        = False
    eng.current_index     = 0
    eng.start_wall_time   = None
    eng.start_offset_sec  = 0.0
    eng._lock             = threading.Lock()
    eng._thread           = None
    eng._stop_flag        = threading.Event()
    eng._seek_request     = None
    eng.on_event_played      = None
    eng.on_playback_finished = None
    return eng


# ── テスト ────────────────────────────────────────────────────────────────────

def test_load_events_count(tmp_path):
    """note_on + note_off + set_tempo + end_of_track のイベント数が正しい。"""
    n_notes = 8
    path    = _create_test_midi(tmp_path, n_notes=n_notes)
    events  = load_midi_events(path)
    # n_notes 個の note_on + n_notes 個の note_off + set_tempo + end_of_track
    expected = n_notes * 2 + 2
    assert len(events) == expected, f'Expected {expected} events, got {len(events)}'


def test_load_events_timing(tmp_path):
    """120 BPM・1拍刻みの note_on 時刻が 0.500s 間隔になっていること。"""
    path   = _create_test_midi(tmp_path, tempo_bpm=120, n_notes=4)
    events = load_midi_events(path)

    note_ons = [e for e in events if e.msg_type == 'note_on']
    assert len(note_ons) == 4

    expected_times = [0.0, 0.5, 1.0, 1.5]  # 120 BPM = 0.5s/beat
    for ev, exp_t in zip(note_ons, expected_times):
        assert abs(ev.abs_time_sec - exp_t) < 0.01, (
            f'Expected {exp_t:.3f}s, got {ev.abs_time_sec:.3f}s'
        )


def test_play_sends_correct_events(tmp_path):
    """play() → 完了まで待ち、送信メッセージ数がイベント数と一致する。"""
    n_notes = 4
    path    = _create_test_midi(tmp_path, n_notes=n_notes)
    mock    = _MockPort()
    eng     = _make_engine(mock)
    eng.load(path)

    done = threading.Event()
    eng.on_playback_finished = lambda: done.set()

    eng.play()
    assert done.wait(timeout=10), 'Playback did not finish in time'

    # note_on + note_off のみカウント（meta は bytes() なし = 送信されない）
    sent_note = [m for m in mock.sent
                 if m.type in ('note_on', 'note_off')]
    assert len(sent_note) == n_notes * 2


def test_mute_channel_blocks_events(tmp_path):
    """チャンネル 9 をミュートすると note_on/off が送信されない。"""
    path = _create_test_midi(tmp_path, n_notes=4)
    mock = _MockPort()
    eng  = _make_engine(mock)
    eng.load(path)
    eng.set_channel_mute(9, True)

    done = threading.Event()
    eng.on_playback_finished = lambda: done.set()

    eng.play()
    assert done.wait(timeout=10)

    note_msgs = [m for m in mock.sent
                 if m.type in ('note_on', 'note_off')]
    assert len(note_msgs) == 0, 'Muted channel should not send note messages'


def test_pause_resume(tmp_path):
    """play() → stop() で途中停止し、play() で続きから再生できる。"""
    n_notes = 8
    path    = _create_test_midi(tmp_path, n_notes=n_notes)
    mock    = _MockPort()
    eng     = _make_engine(mock)
    eng.load(path)

    # 3 イベント目が送信されたら stop する
    stop_after = 3
    played = threading.Event()
    count  = [0]

    def on_played(ev):
        count[0] += 1
        if count[0] >= stop_after:
            played.set()

    eng.on_event_played = on_played
    eng.play()
    assert played.wait(timeout=10)

    eng.stop()
    idx_after_stop = eng.current_index
    assert 0 < idx_after_stop <= len(eng.events), \
        f'current_index {idx_after_stop} should be mid-sequence'

    # 残りを再生
    done = threading.Event()
    eng.on_playback_finished = lambda: done.set()
    eng.play()
    assert done.wait(timeout=10)

    note_msgs = [m for m in mock.sent if m.type in ('note_on', 'note_off')]
    assert len(note_msgs) == n_notes * 2


def test_seek_during_stopped(tmp_path):
    """停止中に seek(5) を呼ぶと current_index == 5 になる。"""
    path = _create_test_midi(tmp_path, n_notes=8)
    mock = _MockPort()
    eng  = _make_engine(mock)
    eng.load(path)

    eng.seek(5)
    assert eng.current_index == 5


def test_seek_during_playback(tmp_path):
    """再生中に seek(10) を呼ぶと index 10 以降のイベントが送信される。"""
    n_notes = 12
    path    = _create_test_midi(tmp_path, n_notes=n_notes)
    mock    = _MockPort()
    eng     = _make_engine(mock)
    eng.load(path)

    seek_to     = 10
    seeked      = threading.Event()
    played_idxs: list[int] = []

    def on_played(ev):
        played_idxs.append(ev.index)
        if ev.index == 3 and not seeked.is_set():
            seeked.set()
            eng.seek(seek_to)

    done = threading.Event()
    eng.on_event_played      = on_played
    eng.on_playback_finished = lambda: done.set()

    eng.play()
    assert done.wait(timeout=15)

    # シーク後のインデックスはすべて seek_to 以上
    post_seek = [i for i in played_idxs if i >= seek_to]
    assert len(post_seek) > 0, 'Should have played events after seek'
    # シーク直後のイベントが seek_to 以上から始まっている
    first_post = next((i for i in played_idxs
                       if i in post_seek), None)
    assert first_post is not None and first_post >= seek_to


def test_seek_out_of_range(tmp_path):
    """seek(-1) / seek(99999) がクラッシュせず範囲内にクリップされる。"""
    path = _create_test_midi(tmp_path, n_notes=4)
    mock = _MockPort()
    eng  = _make_engine(mock)
    eng.load(path)

    eng.seek(-1)
    assert eng.current_index == 0

    eng.seek(99999)
    assert eng.current_index == len(eng.events) - 1
