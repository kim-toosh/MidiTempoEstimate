"""MIDI event dataclass and SMF loader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mido


class BeatMap:
    """Converts abs_time_sec to (measure, beat) using the file's tempo map."""

    def __init__(self, segs: list[tuple[float, float, float]],
                 beats_per_measure: int) -> None:
        self._segs = segs  # [(start_sec, start_beat, sec_per_beat), ...]
        self.beats_per_measure = beats_per_measure

    def _seg_at(self, sec: float) -> tuple[float, float, float]:
        seg = self._segs[0]
        for s in self._segs:
            if s[0] > sec:
                break
            seg = s
        return seg

    def sec_to_pos(self, sec: float) -> tuple[int, int]:
        """Return 1-indexed (measure, beat)."""
        seg_sec, seg_beat, spb = self._seg_at(sec)
        total_beat = seg_beat + (sec - seg_sec) / spb
        measure = int(total_beat / self.beats_per_measure)
        beat    = int(total_beat % self.beats_per_measure)
        return measure + 1, beat + 1

    def sec_to_bpm(self, sec: float) -> float:
        _, _, spb = self._seg_at(sec)
        return 60.0 / spb


class MeasureMap:
    """Converts absolute MIDI ticks to 1-indexed (measure, beat, tick_within_beat).

    Handles time signature changes.  Measure and beat are 1-indexed;
    tick_within_beat is 0-indexed (0 … ticks_per_beat-1).
    """

    def __init__(self, ticks_per_beat: int,
                 sig_changes: list[tuple[int, int]]) -> None:
        # sig_changes: [(abs_tick, numerator), ...] sorted ascending, starts at tick 0
        self._tpb = ticks_per_beat
        # Build segments: (start_tick, beats_per_measure, start_measure_0indexed)
        segs: list[tuple[int, int, int]] = []
        cur_meas = 0
        for i, (tick, num) in enumerate(sig_changes):
            if i > 0:
                prev_tick, prev_num = sig_changes[i - 1]
                cur_meas += (tick - prev_tick) // (prev_num * ticks_per_beat)
            segs.append((tick, num, cur_meas))
        self._segs: list[tuple[int, int, int]] = segs if segs else [(0, 4, 0)]

    def tick_to_mbt(self, abs_tick: int) -> tuple[int, int, int]:
        """Return 1-indexed (measure, beat) and 0-indexed tick_within_beat."""
        seg_tick, beats_per_meas, seg_meas = self._segs[0]
        for s in self._segs:
            if s[0] > abs_tick:
                break
            seg_tick, beats_per_meas, seg_meas = s
        dt             = abs_tick - seg_tick
        ticks_per_meas = beats_per_meas * self._tpb
        meas           = seg_meas + dt // ticks_per_meas
        rem            = dt % ticks_per_meas
        beat           = rem // self._tpb
        tick           = rem % self._tpb
        return meas + 1, beat + 1, tick


def build_beat_map(filepath: str) -> BeatMap:
    """Build a BeatMap by parsing tempo/time_signature meta events from the file."""
    mid = mido.MidiFile(filepath)
    ticks_per_beat = mid.ticks_per_beat

    raw: list[tuple[int, Any]] = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type in ('set_tempo', 'time_signature'):
                raw.append((abs_tick, msg))
    raw.sort(key=lambda x: x[0])

    tempo             = 500_000  # 120 BPM
    beats_per_measure = 4
    segs: list[tuple[float, float, float]] = []
    prev_tick = prev_sec = prev_beat = 0.0

    for tick, msg in raw:
        dt       = tick - prev_tick
        abs_sec  = prev_sec  + mido.tick2second(dt, ticks_per_beat, int(tempo))
        abs_beat = prev_beat + dt / ticks_per_beat

        if msg.type == 'set_tempo':
            tempo = msg.tempo
            segs.append((abs_sec, abs_beat, tempo / 1_000_000))
        elif msg.type == 'time_signature':
            beats_per_measure = msg.numerator

        prev_tick = tick
        prev_sec  = abs_sec
        prev_beat = abs_beat

    if not segs or segs[0][0] > 0.0:
        segs.insert(0, (0.0, 0.0, 500_000 / 1_000_000))

    return BeatMap(segs, beats_per_measure)

# GM Percussion Map（簡易表示用）
_DRUM_NAMES: dict[int, str] = {
    35: 'Kick2', 36: 'Kick',  37: 'RimShot', 38: 'Snare',
    39: 'ClpSnr', 40: 'ESnare', 41: 'LFlTom', 42: 'HHCls',
    43: 'HFlTom', 44: 'HHFt',  45: 'LMTom',  46: 'HHOpn',
    47: 'LHTom',  48: 'HMTom', 49: 'Crash1', 50: 'HiTom',
    51: 'Ride1',  52: 'ChinaCy',53: 'RideBl', 54: 'Tamb',
    55: 'SplCy',  56: 'Cowbel', 57: 'Crash2', 58: 'VibraSlp',
    59: 'Ride2',  60: 'HiBongo',61: 'LoBongo',62: 'MtHiCga',
    63: 'OpHiCga',64: 'LoConga',65: 'HiTmb', 66: 'LoTmb',
    67: 'HiAgogo',68: 'LoAgogo',69: 'Cabasa', 70: 'Maracas',
    71: 'ShrtWsl',72: 'LngWsl', 73: 'ShrtGu', 74: 'LngGui',
    75: 'Claves', 76: 'HiWdBlk',77: 'LoWdBlk',78: 'MtCureg',
    79: 'OpCureg',80: 'MtTri',  81: 'OpTri',
}

_DRUM_CHANNEL = 9  # 0-indexed


@dataclass
class PlaybackEvent:
    index        : int
    abs_time_sec : float
    abs_tick     : int
    diff_ms      : float
    channel      : int | None
    msg_type     : str
    note         : int | None
    velocity     : int | None
    raw_msg      : Any
    display_text : str


def _note_name(note: int, channel: int | None) -> str:
    if channel == _DRUM_CHANNEL and note in _DRUM_NAMES:
        return f'{_DRUM_NAMES[note]}({note})'
    return f'Note{note}'


def _format_display(index: int, mbt: tuple[int, int, int],
                    event: PlaybackEvent) -> str:
    meas, beat, tick = mbt
    pos_str  = f'{meas:03d}:{beat}:{tick:03d}'
    ms_str   = f'{event.abs_time_sec * 1000:8.0f}ms'
    diff_str = f'd{event.diff_ms:5.0f}ms'
    ch_str   = f'Ch{(event.channel + 1):02d}' if event.channel is not None else '----'
    type_str = event.msg_type.replace('_', '').title()[:8]

    if event.note is not None and event.velocity is not None:
        note_str = _note_name(event.note, event.channel)
        detail   = f'{note_str:<14} vel={event.velocity}'
    elif event.note is not None:
        detail = _note_name(event.note, event.channel)
    else:
        detail = ''

    return (
        f'{index:04d}  {pos_str}  {ms_str}  {diff_str}  {ch_str}  '
        f'{type_str:<9} {detail}'
    )


def load_midi_events(filepath: str) -> list[PlaybackEvent]:
    """SMFを読み込み、絶対時刻順のPlaybackEventリストを返す。"""
    mid = mido.MidiFile(filepath)
    ticks_per_beat: int = mid.ticks_per_beat

    # Collect time signature changes from all tracks (deduplicated, sorted)
    sig_map: dict[int, int] = {}   # abs_tick → numerator
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'time_signature':
                sig_map[abs_tick] = msg.numerator
    sig_changes = sorted(sig_map.items())
    if not sig_changes or sig_changes[0][0] > 0:
        sig_changes.insert(0, (0, 4))
    mmap = MeasureMap(ticks_per_beat, sig_changes)

    all_events: list[tuple[float, int, mido.Message, int]] = []

    for track in mid.tracks:
        current_tick = 0
        tempo = 500000  # default: 120 BPM

        # 各トラック内でabsoluteタイムに変換する
        tick_events: list[tuple[int, mido.Message]] = []
        for msg in track:
            current_tick += msg.time
            tick_events.append((current_tick, msg))

        # tickを秒に変換（テンポチェンジを考慮）
        prev_tick = 0
        prev_sec  = 0.0
        for tick, msg in tick_events:
            delta_tick = tick - prev_tick
            delta_sec  = mido.tick2second(delta_tick, ticks_per_beat, tempo)
            abs_sec    = prev_sec + delta_sec

            all_events.append((abs_sec, id(msg), msg, tick))

            if msg.type == 'set_tempo':
                tempo = msg.tempo

            prev_tick = tick
            prev_sec  = abs_sec

    # 絶対時刻でソート（同時刻はファイル順を維持）
    all_events.sort(key=lambda x: x[0])

    result: list[PlaybackEvent] = []
    prev_sec = 0.0
    for i, (abs_sec, _, msg, abs_tick) in enumerate(all_events):
        channel  = getattr(msg, 'channel', None)
        note     = getattr(msg, 'note',    None)
        velocity = getattr(msg, 'velocity', None) if msg.type == 'note_on' else None

        ev = PlaybackEvent(
            index        = i,
            abs_time_sec = abs_sec,
            abs_tick     = abs_tick,
            diff_ms      = (abs_sec - prev_sec) * 1000,
            channel      = channel,
            msg_type     = msg.type,
            note         = note,
            velocity     = velocity,
            raw_msg      = msg,
            display_text = '',
        )
        ev.display_text = _format_display(i, mmap.tick_to_mbt(abs_tick), ev)
        result.append(ev)
        prev_sec = abs_sec

    return result
