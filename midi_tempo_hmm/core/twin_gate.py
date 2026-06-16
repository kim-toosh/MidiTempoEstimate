"""TwinGate: MidiInputGate + KalmanGate direct-connected tempo estimator."""

from __future__ import annotations

import time
from collections import deque
from types import ModuleType
from typing import Optional

import numpy as np

from midi_tempo_hmm.core.instrument_category import InstrumentCategory
from midi_tempo_hmm.core.kalman_gate import KalmanGate
from midi_tempo_hmm.core.midi_input_gate import MidiInputGate
from midi_tempo_hmm.output.twin_gate_result import TwinGateResult


class TwinGate:
    """MidiInputGate と KalmanGate を直結したテンポ推定エンジン。

    GCDテンポはパーティクルフィルタと同じオクターブ補正を適用して
    KalmanGateに渡す。ParticleFilterを使用しないため処理が高速かつ
    決定論的で、数値揺らぎが発生しない。
    """

    def __init__(self, config: ModuleType) -> None:
        self.midi_gate   = MidiInputGate(config)
        self.kalman_gate = KalmanGate(config)
        self.config      = config
        self.event_count = 0

        _buf_size = getattr(config, 'GCD_BUFFER_SIZE', 8)
        self._gcd_cat_buf: deque[tuple[float, InstrumentCategory]] = deque(maxlen=_buf_size)
        self._last_event_ts: Optional[float] = None

        self._gcd_group_start: Optional[float] = None
        self._gcd_group_sum:   float = 0.0
        self._gcd_group_count: int   = 0
        self._gcd_group_cat:   Optional[InstrumentCategory] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        timestamp_sec: float,
        note_number:   int,
        velocity:      int = 100,
        channel:       int = 9,
    ) -> Optional[TwinGateResult]:
        """MIDIイベントを処理してTwinGateResultを返す。

        velocity==0 (NOTE_OFF) の場合は None を返す。
        GCD未収束の場合は KalmanGate を更新せず現在値を返す。
        """
        t0 = time.perf_counter()

        event, raw_ioi, gcd_tempo_raw, gcd_confidence = self.midi_gate.process(
            timestamp_sec, note_number, velocity, channel
        )

        if event is None:
            return None

        self.event_count += 1

        # Mirror MidiInputGate's gcd_timestamps accumulation with category tracking
        event_timeout = getattr(self.config, 'EVENT_TIMEOUT_SEC', 24.0)
        if (self._last_event_ts is not None
                and timestamp_sec - self._last_event_ts > event_timeout):
            self._gcd_cat_buf.clear()
            self._gcd_group_start = None
            self._gcd_group_count = 0
        self._last_event_ts = timestamp_sec

        if event.category in (InstrumentCategory.KICK,
                               InstrumentCategory.SNARE,
                               InstrumentCategory.HIHAT):
            span = getattr(self.config, 'SPAN_SAME_TIME_SEC', 0.05)
            if (self._gcd_cat_buf
                    and self._gcd_group_start is not None
                    and timestamp_sec - self._gcd_group_start <= span):
                self._gcd_group_sum   += timestamp_sec
                self._gcd_group_count += 1
                avg_ts = self._gcd_group_sum / self._gcd_group_count
                self._gcd_cat_buf[-1] = (avg_ts, self._gcd_group_cat)
            else:
                self._gcd_cat_buf.append((timestamp_sec, event.category))
                self._gcd_group_start = timestamp_sec
                self._gcd_group_sum   = timestamp_sec
                self._gcd_group_count = 1
                self._gcd_group_cat   = event.category

        counts       = self.midi_gate.event_count_by_category
        kick_count   = counts.get(InstrumentCategory.KICK,   0)
        snare_count  = counts.get(InstrumentCategory.SNARE,  0)
        hihat_count  = counts.get(InstrumentCategory.HIHAT,  0)
        others_count = counts.get(InstrumentCategory.OTHERS, 0)

        if gcd_tempo_raw is not None:
            # オクターブ補正: GCDはサブディビジョン周期を返すことがあるため、
            # Kalman平均に最も近いビート倍率候補を選択する。
            gcd_tempo = self._correct_gcd_octave(gcd_tempo_raw)
        else:
            gcd_tempo = None

        gcd_period = (60.0 / gcd_tempo) if gcd_tempo is not None else None

        if gcd_tempo is None:
            gate_accepted   = False
            reject_reason   = "no_gcd"
            predicted_tempo = self.kalman_gate.mean
            innovation      = None
            mahal_distance  = None
            mahal_threshold = self.config.KALMAN_GATE_SIGMA ** 2
            kalman_variance = self.kalman_gate._var
            kalman_gain     = 0.0
            tempo_bpm       = self.kalman_gate.mean
        else:
            gr = self.kalman_gate.update(candidate=gcd_tempo, confidence=gcd_confidence)
            gate_accepted   = gr.accepted
            reject_reason   = gr.reject_reason
            predicted_tempo = gr.predicted_tempo
            innovation      = gr.innovation
            mahal_distance  = gr.mahal_distance
            mahal_threshold = gr.mahal_threshold
            kalman_variance = gr.current_var
            kalman_gain     = gr.kalman_gain
            tempo_bpm       = gr.gated_tempo

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return TwinGateResult(
            tempo_bpm         = tempo_bpm,
            tempo_bpm_str     = f"{tempo_bpm:.2f}",
            category          = event.category,
            raw_ioi           = raw_ioi,
            gcd_tempo         = gcd_tempo,
            gcd_confidence    = gcd_confidence,
            gcd_period        = gcd_period,
            kick_count        = kick_count,
            snare_count       = snare_count,
            hihat_count       = hihat_count,
            others_count      = others_count,
            gate_accepted     = gate_accepted,
            reject_reason     = reject_reason,
            predicted_tempo   = predicted_tempo,
            innovation        = innovation,
            mahal_distance    = mahal_distance,
            mahal_threshold   = mahal_threshold,
            kalman_variance   = kalman_variance,
            kalman_gain       = kalman_gain,
            event_count       = self.event_count,
            processing_time_ms = elapsed_ms,
            gcd_buffer        = list(self._gcd_cat_buf),
        )

    def reset(self) -> None:
        self.midi_gate.reset()
        self.kalman_gate.reset()
        self.event_count = 0
        self._gcd_cat_buf.clear()
        self._last_event_ts   = None
        self._gcd_group_start = None
        self._gcd_group_sum   = 0.0
        self._gcd_group_count = 0
        self._gcd_group_cat   = None

    @property
    def current_tempo(self) -> Optional[float]:
        """現在のKalman平均テンポ。update()未呼び出しの場合はNone。"""
        return self.kalman_gate.mean if self.event_count > 0 else None

    @property
    def current_variance(self) -> float:
        return self.kalman_gate._var

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _correct_gcd_octave(self, gcd_tempo: float) -> float:
        """GCDテンポをGCD_OCTAVE_RATIOSで割り、Kalman平均に最も近い候補を返す。"""
        ratios = np.asarray(self.config.GCD_OCTAVE_RATIOS, dtype=np.float64)
        candidates = gcd_tempo / ratios
        return float(candidates[np.argmin(np.abs(candidates - self.kalman_gate.mean))])
