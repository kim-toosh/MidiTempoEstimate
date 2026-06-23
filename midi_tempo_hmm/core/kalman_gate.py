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
        reject_reason:   "confidence" | "octave" | "mahal" | "no_gcd" | "" (accepted).
        all_candidates:  All input (tempo, confidence) candidates.
        selected_index:  Index of the chosen candidate in all_candidates (-1 if none).
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
    all_candidates: list
    selected_index: int


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

    def select_best_candidate(
        self,
        candidates: list[tuple[float, float]],
    ) -> tuple:
        """複数のGCD候補から最適な1つを選ぶ。

        Args:
            candidates: [(tempo_bpm, confidence), ...] スコア良い順

        Returns:
            (tempo, confidence, selected_index)
            candidatesが空の場合は (None, 0.0, -1)
        """
        if not candidates:
            return (None, 0.0, -1)

        # Kalman meanがない（初回）またはmean=0: confidence最大の候補を選ぶ
        if self._mean is None or self._mean <= 0:
            idx = max(range(len(candidates)), key=lambda i: candidates[i][1])
            return (*candidates[idx], idx)

        # Kalman meanに最も近い候補を選ぶ
        distances = [abs(t - self._mean) for t, _ in candidates]
        min_dist  = min(distances)

        # 全候補がmeanから大きく離れている場合（mean比50%以上）はconfidence最大にフォールバック
        if min_dist > 0.5 * self._mean:
            idx = max(range(len(candidates)), key=lambda i: candidates[i][1])
            return (*candidates[idx], idx)

        idx = int(distances.index(min_dist))

        # Harmonic disambiguation: if the closest candidate is a spurious
        # harmonic of another candidate within 4× the minimum distance,
        # prefer the musically-correct one.
        #   4/3 downward: best = t × (4/3) → prefer the lower t
        #     e.g. 133.33 = 100 × 4/3 with init=120 → prefer 100
        #   3/2 upward:   t = best × (3/2) → prefer the higher t
        #     e.g. 160 = 106.67 × 3/2 with init=120 → prefer 160
        _TOL   = 0.05
        _LIMIT = 4.0 * min_dist
        best_t = candidates[idx][0]
        for j, (t, _) in enumerate(candidates):
            if j == idx or t <= 0:
                continue
            if distances[j] >= _LIMIT:
                continue
            # Switch DOWN: best is the 4/3 upper harmonic of t
            if abs(best_t / t - 4.0 / 3.0) < _TOL:
                idx = j
                break
            # Switch UP: t is the 3/2 upper harmonic of best
            if abs(t / best_t - 3.0 / 2.0) < _TOL:
                idx = j
                break

        return (*candidates[idx], idx)

    def update(self, candidates: list[tuple[float, float]]) -> KalmanGateResult:
        """複数のGCD候補をゲート処理する。

        select_best_candidate() で1つに絞り込んだ後、
        既存のオクターブ・マハラノビス・信頼度チェックを適用する。

        Args:
            candidates: [(tempo_bpm, confidence), ...] 空リストはno_gcd扱い

        Returns:
            :class:`KalmanGateResult` describing the outcome.
        """
        cfg = self._cfg
        precision       = getattr(cfg, 'TEMPO_PRECISION', 2)
        mahal_threshold = cfg.KALMAN_GATE_SIGMA ** 2
        predicted_var   = self._var + cfg.KALMAN_Q

        # 候補なし
        if not candidates:
            self._var = predicted_var
            return KalmanGateResult(
                accepted        = False,
                raw_candidate   = 0.0,
                gated_tempo     = round(self._mean, precision),
                predicted_tempo = self._mean,
                innovation      = 0.0,
                innovation_var  = predicted_var + cfg.KALMAN_R,
                mahal_distance  = 0.0,
                mahal_threshold = mahal_threshold,
                kalman_gain     = 0.0,
                current_var     = self._var,
                reject_reason   = "no_gcd",
                all_candidates  = [],
                selected_index  = -1,
            )

        candidate, confidence, sel_idx = self.select_best_candidate(candidates)

        # ── 1. 予測ステップ ──────────────────────────────────────────────────
        predicted_mean  = self._mean
        innovation      = candidate - predicted_mean
        innovation_var  = predicted_var + cfg.KALMAN_R
        kalman_gain     = predicted_var / innovation_var
        mahal_distance  = (innovation ** 2) / innovation_var

        # ── 2. ゲート判定（棄却チェック）────────────────────────────────────
        reject_reason = ""
        if confidence < cfg.KALMAN_MIN_CONFIDENCE:
            reject_reason = "confidence"
        else:
            ratio = candidate / self._mean
            for r in cfg.OCTAVE_RATIOS:
                if abs(ratio / r - 1.0) <= cfg.OCTAVE_TOLERANCE:
                    reject_reason = "octave"
                    break
            if not reject_reason and mahal_distance > mahal_threshold:
                reject_reason = "mahal"

        accepted = (reject_reason == "")

        # ── 3. カルマン更新ステップ ──────────────────────────────────────────
        if accepted:
            self._mean  = predicted_mean + kalman_gain * innovation
            self._var   = (1.0 - kalman_gain) * predicted_var
            gated_tempo = self._mean
        else:
            self._var   = predicted_var
            gated_tempo = self._mean

        return KalmanGateResult(
            accepted        = accepted,
            raw_candidate   = candidate,
            gated_tempo     = round(gated_tempo, precision),
            predicted_tempo = predicted_mean,
            innovation      = innovation,
            innovation_var  = innovation_var,
            mahal_distance  = mahal_distance,
            mahal_threshold = mahal_threshold,
            kalman_gain     = kalman_gain,
            current_var     = self._var,
            reject_reason   = reject_reason,
            all_candidates  = list(candidates),
            selected_index  = sel_idx,
        )
