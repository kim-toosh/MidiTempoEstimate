"""SMF Player engine: loads MIDI events and plays them to a MIDI output port."""

from __future__ import annotations

import threading
import time
from typing import Callable

import mido

from midi_event import PlaybackEvent, load_midi_events


class PlayerEngine:
    """再生エンジン。GUIとは独立して動作確認できる設計。"""

    def __init__(self, port_name: str) -> None:
        self.events: list[PlaybackEvent] = []
        self.outport = mido.open_output(port_name)
        self.muted_channels: set[int] = set()

        self.is_playing       : bool          = False
        self.current_index    : int           = 0
        self.start_wall_time  : float | None  = None
        self.start_offset_sec : float         = 0.0

        self._lock          = threading.Lock()
        self._thread        : threading.Thread | None = None
        self._stop_flag     = threading.Event()
        self._seek_request  : int | None = None  # プレイバックスレッド内シーク用

        self.on_event_played      : Callable[[PlaybackEvent], None] | None = None
        self.on_playback_finished : Callable[[], None] | None              = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, filepath: str) -> None:
        self.events             = load_midi_events(filepath)
        self.current_index      = 0
        self.start_offset_sec   = 0.0

    def play(self, start_index: int | None = None) -> None:
        if self.is_playing:
            return

        if start_index is not None:
            idx = max(0, min(start_index, len(self.events) - 1)) if self.events else 0
            self.current_index    = idx
            self.start_offset_sec = self.events[idx].abs_time_sec if self.events else 0.0

        self.is_playing      = True
        self.start_wall_time = time.perf_counter()
        self._stop_flag.clear()

        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.is_playing = False

    def seek(self, index: int) -> None:
        if not self.events:
            return
        index = max(0, min(index, len(self.events) - 1))

        if self.is_playing:
            if (self._thread is not None
                    and self._thread is threading.current_thread()):
                # プレイバックスレッド内から呼ばれた場合：ループにシーク要求を渡す
                with self._lock:
                    self._seek_request = index
            else:
                self.stop()
                self.play(start_index=index)
        else:
            with self._lock:
                self.current_index    = index
                self.start_offset_sec = self.events[index].abs_time_sec

    def set_channel_mute(self, channel: int, muted: bool) -> None:
        if muted:
            self.muted_channels.add(channel)
        else:
            self.muted_channels.discard(channel)

    def close(self) -> None:
        self.stop()
        self.outport.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _playback_loop(self) -> None:
        assert self.start_wall_time is not None

        while not self._stop_flag.is_set():
            with self._lock:
                idx = self.current_index

            if idx >= len(self.events):
                break

            event   = self.events[idx]
            elapsed = time.perf_counter() - self.start_wall_time
            target  = event.abs_time_sec - self.start_offset_sec

            wait = target - elapsed
            if wait > 0:
                interrupted = self._sleep_interruptible(wait)
                if interrupted:
                    break

            if (event.channel is None
                    or event.channel not in self.muted_channels):
                if isinstance(event.raw_msg, mido.Message):
                    self.outport.send(event.raw_msg)

            if self.on_event_played:
                self.on_event_played(event)

            # シーク要求があればジャンプ、なければ次のイベントへ進む
            with self._lock:
                if self._seek_request is not None:
                    seek_idx              = self._seek_request
                    self._seek_request    = None
                    self.current_index    = seek_idx
                    self.start_offset_sec = self.events[seek_idx].abs_time_sec
                    self.start_wall_time  = time.perf_counter()
                else:
                    self.current_index += 1

        with self._lock:
            self.is_playing = False

        with self._lock:
            finished = (self.current_index >= len(self.events)
                        and not self._stop_flag.is_set()
                        and self._seek_request is None)
        if finished and self.on_playback_finished:
            self.on_playback_finished()

    def _sleep_interruptible(self, duration: float) -> bool:
        """durationだけ待機。_stop_flagがセットされたらTrueを返す。"""
        deadline = time.perf_counter() + duration
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return False
            if self._stop_flag.wait(timeout=min(0.01, remaining)):
                return True
