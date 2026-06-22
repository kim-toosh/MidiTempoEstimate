"""Phase-Coupled Oscillator (PCO) — Large & Kolen circle map."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import mean
from types import ModuleType
from typing import Optional


@dataclass
class PhaseOscillatorResult:
    phase           : float         # 現在の拍内位相 [0.0, 1.0)
    period_sec      : float         # 使用中の1拍の長さ [秒]
    phase_error     : float         # 位相誤差 [-0.5, 0.5]（正=遅れ、負=早い）
    is_synced       : bool          # 同期確立フラグ
    next_beat_time  : Optional[float]  # 次拍頭の予測絶対時刻 [秒]
    beat_count      : int           # 拍頭通過累計回数
    sync_confidence : float         # 同期信頼度 [0.0, 1.0]


class PhaseOscillator:
    """Large & Kolen circle map に基づく位相結合オシレータ。

    period（拍周期）はKalmanGateから外部供給される。
    PCOはphaseの同期のみに専念し、periodの推定は行わない。
    """

    def __init__(self, config: ModuleType) -> None:
        self._cfg           = config
        self.phase          : float          = 0.0
        self.period_sec     : Optional[float] = None
        self.last_time      : Optional[float] = None
        self.beat_count     : int            = 0
        self.is_synced      : bool           = False
        self._error_history : deque[float]   = deque(maxlen=8)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        timestamp_sec : float,
        period_sec    : float,
        sync_strength : float,
    ) -> PhaseOscillatorResult:
        """位相を更新してPhaseOscillatorResultを返す。

        Args:
            timestamp_sec: イベントの絶対時刻 [秒]
            period_sec:    KalmanGateが推定した1拍の長さ [秒]
            sync_strength: 位相同期の強さ [0〜1]（カテゴリに応じて変える）
        """
        if self.last_time is None:
            # 初回イベント：位相を0にリセットして開始
            self.phase      = 0.0
            self.period_sec = period_sec
            self.last_time  = timestamp_sec
            return PhaseOscillatorResult(
                phase          = self.phase,
                period_sec     = period_sec,
                phase_error    = 0.0,
                is_synced      = False,
                next_beat_time = self._next_beat(timestamp_sec, period_sec),
                beat_count     = self.beat_count,
                sync_confidence = 0.0,
            )

        # ① 周期更新（KalmanGateの最新値を信頼）
        self.period_sec = period_sec

        # ② 自然進行（経過時間で位相を前進）
        dt                  = timestamp_sec - self.last_time
        new_phase_unwrapped = self.phase + dt / self.period_sec
        beats_crossed       = int(new_phase_unwrapped)
        self.phase          = new_phase_unwrapped % 1.0
        self.beat_count    += beats_crossed

        # ③ 位相誤差（拍頭への最短距離）
        if self.phase < 0.5:
            phase_error = self.phase          # 遅れ：拍頭を少し過ぎた
        else:
            phase_error = self.phase - 1.0    # 早い：次の拍頭より少し前

        # ④ 位相補正（拍頭方向に引き込む）
        self.phase = (self.phase - sync_strength * phase_error) % 1.0

        # ⑤ 同期判定（位相誤差の移動平均 — 弱拍系は除外）
        if sync_strength >= getattr(self._cfg, 'PCO_ETA_PHASE', 0.15):
            self._error_history.append(abs(phase_error))
        mean_error = mean(self._error_history) if self._error_history else 1.0
        self.is_synced  = mean_error < self._cfg.PCO_SYNC_THRESHOLD
        sync_confidence = max(0.0, 1.0 - mean_error / 0.5)

        # ⑥ 次Beat予測
        next_beat_time = self._next_beat(timestamp_sec, self.period_sec)

        # ⑦ 時刻更新
        self.last_time = timestamp_sec

        return PhaseOscillatorResult(
            phase          = self.phase,
            period_sec     = self.period_sec,
            phase_error    = phase_error,
            is_synced      = self.is_synced,
            next_beat_time = next_beat_time,
            beat_count     = self.beat_count,
            sync_confidence = sync_confidence,
        )

    def get_current_phase(self, now_sec: float) -> Optional[float]:
        """現在時刻における位相を外挿して返す（GUI表示用）。"""
        if self.last_time is None or self.period_sec is None:
            return None
        return (self.phase + (now_sec - self.last_time) / self.period_sec) % 1.0

    def reset(self) -> None:
        self.phase          = 0.0
        self.period_sec     = None
        self.last_time      = None
        self.beat_count     = 0
        self.is_synced      = False
        self._error_history.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _next_beat(self, timestamp_sec: float, period_sec: float) -> Optional[float]:
        if not getattr(self._cfg, 'PCO_PREDICTION_ENABLE', True):
            return None
        return timestamp_sec + (1.0 - self.phase) * period_sec
