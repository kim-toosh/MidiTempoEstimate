"""MIDI input gate: classify note-on events by instrument category and track per-category IOI."""

from __future__ import annotations

from collections import deque
from types import ModuleType
from typing import Optional

import numpy as np

from midi_tempo_hmm.core.approx_gcd import estimate_tempo_from_timestamps
from midi_tempo_hmm.core.instrument_category import CategorizedEvent, InstrumentCategory

# GM standard drum note mappings
_KICK_NOTES  = frozenset({35, 36})
_SNARE_NOTES = frozenset({37, 38, 39, 40})
_HIHAT_NOTES = frozenset({42, 44, 46})


class MidiInputGate:
    """Classify MIDI note-on events into :class:`InstrumentCategory` and compute
    same-category inter-onset intervals (IOIs).

    IOI is only computed between two consecutive events of the *same* category
    (e.g. Kick→Kick).  Cross-category intervals are intentionally ignored to
    avoid mixing different rhythmic roles in the observation model.

    Args:
        config: Module containing DRUM_CHANNEL, USE_DRUM_WEIGHTS, GCD_* constants,
                and the category drum-weight constants.
    """

    def __init__(self, config: ModuleType) -> None:
        self._cfg = config
        self._drum_channel: int  = getattr(config, 'DRUM_CHANNEL', 9)
        self._use_drum: bool     = getattr(config, 'USE_DRUM_WEIGHTS', True)
        self._crash_notes        = frozenset(getattr(config, 'CRASH_NOTE_NUMBERS', [49, 52, 55, 57]))

        self._cat_weight: dict[InstrumentCategory, float] = {
            InstrumentCategory.KICK:   getattr(config, 'KICK_DRUM_WEIGHT',   1.0),
            InstrumentCategory.SNARE:  getattr(config, 'SNARE_DRUM_WEIGHT',  0.9),
            InstrumentCategory.HIHAT:  getattr(config, 'HIHAT_DRUM_WEIGHT',  0.15),
            InstrumentCategory.OTHERS: getattr(config, 'OTHERS_DRUM_WEIGHT', 0.4),
        }
        # Crash overrides OTHERS weight when note is in CRASH_NOTE_NUMBERS
        self._crash_weight: float = getattr(config, 'CRASH_DRUM_WEIGHT', 0.8)

        self.event_count_by_category: dict[InstrumentCategory, int] = {
            c: 0 for c in InstrumentCategory
        }
        self._last_time: dict[InstrumentCategory, Optional[float]] = {
            c: None for c in InstrumentCategory
        }

        # GCD推定用タイムスタンプバッファ（Kick/Snare/HiHatを時系列順に1本で保持）
        _buf_size = getattr(config, 'GCD_BUFFER_SIZE', 8)
        self.gcd_timestamps: deque[float] = deque(maxlen=_buf_size)

        self.last_gcd_tempo:      Optional[float] = None
        self.last_gcd_confidence: float = 0.0
        self.last_gcd_iois:       list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        timestamp_sec: float,
        note_number:   int,
        velocity:      int,
        channel:       int,
    ) -> tuple[Optional[CategorizedEvent], Optional[float], Optional[float], float]:
        """Classify a MIDI event, compute same-category IOI, and estimate GCD tempo.

        Returns:
            (event, ioi_sec, gcd_tempo, gcd_confidence)
            event is ``None`` when *velocity* == 0 (NOTE_OFF).
        """
        if velocity == 0:
            return (None, None, None, 0.0)

        category    = self._classify(note_number, channel)
        drum_weight = self._drum_weight(note_number, channel, category)

        self.event_count_by_category[category] += 1

        event = CategorizedEvent(
            timestamp_sec=timestamp_sec,
            note_number=note_number,
            velocity=velocity,
            channel=channel,
            category=category,
            drum_weight=drum_weight,
        )

        # GCD推定用タイムスタンプ蓄積（Kick/Snare/HiHat）
        if category in (InstrumentCategory.KICK, InstrumentCategory.SNARE, InstrumentCategory.HIHAT):
            self.gcd_timestamps.append(timestamp_sec)

        # 同カテゴリIOI計算
        prev = self._last_time[category]
        self._last_time[category] = timestamp_sec
        ioi_sec = None if prev is None else timestamp_sec - prev

        # Kick+Snare合算で近似GCDを計算
        gcd_tempo, gcd_confidence = self._calc_gcd_tempo()
        self.last_gcd_tempo       = gcd_tempo
        self.last_gcd_confidence  = gcd_confidence

        return (event, ioi_sec, gcd_tempo, gcd_confidence)

    def get_ioi(self, event: CategorizedEvent) -> Optional[float]:
        """Return the same-category IOI in seconds, or ``None`` on the first event.

        Updates the stored last-event timestamp for the event's category.

        Note: When using process() directly, IOI is already included in the return
        value. This method is retained for external use and backward compatibility.
        """
        prev = self._last_time[event.category]
        self._last_time[event.category] = event.timestamp_sec
        if prev is None:
            return None
        return event.timestamp_sec - prev

    def reset(self) -> None:
        """Clear all per-category state."""
        for c in InstrumentCategory:
            self.event_count_by_category[c] = 0
            self._last_time[c] = None
        self.gcd_timestamps.clear()
        self.last_gcd_tempo      = None
        self.last_gcd_confidence = 0.0
        self.last_gcd_iois       = []

    def get_stats(self) -> dict:
        """Return a copy of per-category event counts."""
        return dict(self.event_count_by_category)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify(self, note_number: int, channel: int) -> InstrumentCategory:
        if not self._use_drum or channel != self._drum_channel:
            return InstrumentCategory.OTHERS
        if note_number in _KICK_NOTES:
            return InstrumentCategory.KICK
        if note_number in _SNARE_NOTES:
            return InstrumentCategory.SNARE
        if note_number in _HIHAT_NOTES:
            return InstrumentCategory.HIHAT
        return InstrumentCategory.OTHERS

    def _drum_weight(
        self,
        note_number: int,
        channel: int,
        category: InstrumentCategory,
    ) -> float:
        if not self._use_drum or channel != self._drum_channel:
            return 1.0
        if note_number in self._crash_notes:
            return self._crash_weight
        return self._cat_weight[category]

    def _calc_gcd_tempo(self) -> tuple[Optional[float], float]:
        """Kick・Snare・HiHatのタイムスタンプを合算して近似GCDを計算する。"""
        # Kick/Snare/HiHatを単一バッファに時系列順で蓄積しているため、
        # ソートし直す必要はないが、念のため明示的にソートしておく。
        # 単一バッファにまとめることで、一方のカテゴリの入力が
        # 一定時間途絶えても、もう一方の古いタイムスタンプが
        # バッファに居座り続けてGCD計算に影響することを防ぐ。
        combined = sorted(self.gcd_timestamps)

        # 推定に必要な最小イベント数（IOI換算でmin_events-1個）に達していない
        # 場合は、GCD推定が不安定になるため計算をスキップしてNoneを返す。
        min_events = getattr(self._cfg, 'GCD_MIN_EVENTS', 4)
        if len(combined) < min_events:
            self.last_gcd_iois = []
            return (None, 0.0)

        # GUI表示用に、GCD推定の入力となるIOI列をそのまま保持しておく
        # （estimate_tempo_from_timestamps内部でも同じ np.diff を計算するが、
        #  戻り値が(tempo, confidence)のみのため、表示用に別途計算する）
        self.last_gcd_iois = list(np.diff(np.asarray(combined, dtype=float)))

        # approx_gcd.estimate_tempo_from_timestamps に委譲:
        #   1. combinedの隣接差分(IOI)を取り、approx_gcd()で粗→精密の
        #      2段階探索により近似GCD周期を求める
        #   2. refine_gcd()で加重中央値による反復精密化
        #   3. calc_gcd_confidence()でGCDに対する各IOIの残差から信頼度を算出
        #   4. tempo = 60/gcd に変換。TEMPO_MIN/MAX範囲外なら(None, 0.0)を返す
        return estimate_tempo_from_timestamps(combined, self._cfg)
