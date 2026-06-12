"""1-D Kalman gate for particle-filter tempo output."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import ModuleType
from typing import Optional


@dataclass
class KalmanGateResult:
    """Result of a single Kalman-gate update step.

    Attributes:
        accepted:        Whether the candidate passed all gate checks.
        raw_candidate:   Raw tempo from the particle filter (BPM).
        gated_tempo:     Kalman-smoothed tempo (BPM); equals Kalman mean after
                         an accepted update, or the previous mean on reject.
        predicted_tempo: Kalman predicted mean before incorporating the candidate.
        innovation:      raw_candidate − predicted_tempo.
        innovation_var:  Predicted variance + observation noise R.
        mahal_distance:  innovation² / innovation_var (Mahalanobis²).
        mahal_threshold: Gate threshold = KALMAN_GATE_SIGMA².
        kalman_gain:     K = predicted_var / innovation_var.
        current_var:     Posterior variance after this step.
        reject_reason:   "confidence" | "octave" | "mahal" | "" (accepted).
    """

    accepted: bool
    raw_candidate: float
    gated_tempo: float
    predicted_tempo: float
    innovation: float
    innovation_var: float
    mahal_distance: float
    mahal_threshold: float
    kalman_gain: float
    current_var: float
    reject_reason: str


class KalmanGate:
    """1-D Kalman filter acting as an outlier gate on particle-filter output.

    The filter is initialised at KALMAN_INIT_TEMPO (default 120 BPM) with
    variance KALMAN_INIT_VAR.  Gate checks are applied on every call:

      1. "confidence" — candidate confidence < KALMAN_MIN_CONFIDENCE
      2. "octave"     — candidate/mean ≈ one of OCTAVE_RATIOS (±OCTAVE_TOLERANCE)
      3. "mahal"      — Mahalanobis² > KALMAN_GATE_SIGMA²

    Variance grows by KALMAN_Q on every call, including rejects.

    Args:
        config: Module containing Kalman constants.
    """

    def __init__(self, config: ModuleType) -> None:
        self._cfg = config
        self._initial_tempo: float = config.KALMAN_INIT_TEMPO
        self._mean: float = self._initial_tempo
        self._var: float = config.KALMAN_INIT_VAR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def mean(self) -> float:
        """Current Kalman mean estimate (BPM)."""
        return self._mean

    def reset(self) -> None:
        """Restore initial state (mean = KALMAN_INIT_TEMPO, var = KALMAN_INIT_VAR)."""
        self._mean = self._initial_tempo
        self._var = self._cfg.KALMAN_INIT_VAR

    def get_confidence_interval(self, sigma: float = 2.0) -> tuple[float, float]:
        """Return (lower, upper) tempo bounds for the current Kalman estimate."""
        half = sigma * math.sqrt(self._var)
        return (self._mean - half, self._mean + half)

    def update(self, candidate: float, confidence: float) -> KalmanGateResult:
        """Process one tempo candidate through the gate.

        Predict → gate checks → Kalman update (if accepted).

        Args:
            candidate:  Raw tempo estimate from the particle filter (BPM).
            confidence: Particle-filter confidence in [0.0, 1.0].

        Returns:
            :class:`KalmanGateResult` describing the outcome.
        """
        cfg = self._cfg

        # ── 1. 予測ステップ ──────────────────────────────────────────────────
        # 前回の事後平均をそのまま予測平均として使う（定数モデル）
        predicted_mean = self._mean
        # プロセスノイズ Q を加えて予測分散を更新（時間経過による不確かさの増加）
        predicted_var  = self._var + cfg.KALMAN_Q

        # 観測値（候補テンポ）と予測平均の差＝イノベーション
        innovation      = candidate - predicted_mean
        # イノベーションの分散＝予測分散＋観測ノイズ R
        innovation_var  = predicted_var + cfg.KALMAN_R
        # カルマンゲイン K＝予測分散 / イノベーション分散（0〜1、高いほど観測を信頼）
        kalman_gain     = predicted_var / innovation_var
        # マハラノビス距離²＝イノベーション² / イノベーション分散（標準化された外れ値指標）
        mahal_distance  = (innovation ** 2) / innovation_var
        # ゲート閾値＝KALMAN_GATE_SIGMA²（マハラノビス距離がこれを超えたら棄却）
        mahal_threshold = cfg.KALMAN_GATE_SIGMA ** 2

        # ── 2. ゲート判定（棄却チェック）────────────────────────────────────
        reject_reason = ""
        # チェック①：パーティクルフィルタの信頼度が低すぎる場合は棄却
        if confidence < cfg.KALMAN_MIN_CONFIDENCE:
            reject_reason = "confidence"
        else:
            # チェック②：候補テンポが現在の推定値のオクターブ倍率に近い場合は棄却
            # （例: 120BPM → 60BPM や 240BPM への跳躍はテンポ誤認と見なす）
            ratio = candidate / self._mean
            for r in cfg.OCTAVE_RATIOS:
                if abs(ratio / r - 1.0) <= cfg.OCTAVE_TOLERANCE:
                    reject_reason = "octave"
                    break
            # チェック③：マハラノビス距離が閾値を超えた場合は外れ値として棄却
            if not reject_reason and mahal_distance > mahal_threshold:
                reject_reason = "mahal"

        accepted = (reject_reason == "")

        # ── 3. カルマン更新ステップ ──────────────────────────────────────────
        if accepted:
            # 受理された場合：予測平均＋カルマンゲイン×イノベーション で事後平均を更新
            self._mean = predicted_mean + kalman_gain * innovation
            # 事後分散を更新（ゲインが大きいほど分散が縮小し確信度が上がる）
            self._var  = (1.0 - kalman_gain) * predicted_var
            gated_tempo = self._mean
        else:
            # 棄却された場合：平均は据え置き、分散だけ予測値で更新
            # （観測なしの時間経過として不確かさを増加させる）
            self._var = predicted_var
            gated_tempo = self._mean

        precision = getattr(cfg, 'TEMPO_PRECISION', 2)

        return KalmanGateResult(
            accepted=accepted,
            raw_candidate=candidate,
            gated_tempo=round(gated_tempo, precision),
            predicted_tempo=predicted_mean,
            innovation=innovation,
            innovation_var=innovation_var,
            mahal_distance=mahal_distance,
            mahal_threshold=mahal_threshold,
            kalman_gain=kalman_gain,
            current_var=self._var,
            reject_reason=reject_reason,
        )
