"""Approximate GCD-based tempo estimation from event timestamps."""

from __future__ import annotations

from types import ModuleType
from typing import Optional

import numpy as np


def approx_gcd(
    values: np.ndarray,
    g_min: float,
    g_max: float,
    resolution: float,
) -> tuple[float, float]:
    """浮動小数点数列から近似GCDを求める（2段階探索）。

    Stage1: resolution*10 の粗い探索で最良候補を特定する。
    Stage2: 最良候補 ± coarse_resolution*2 の範囲を resolution で精密探索する。
    合計候補数は resolution=0.002 のとき 200個以下になる。

    Args:
        values:     IOIまたはタイムスタンプ差分の配列
        g_min:      探索下限 [秒]
        g_max:      探索上限 [秒]
        resolution: 候補GCDの刻み幅 [秒]

    Returns:
        (best_gcd [秒], best_score) — best_scoreは平均残差（小さいほど良い）
    """
    coarse_res = resolution * 10

    # ── Stage 1: 粗い探索 ──────────────────────────────────────────────────
    coarse = np.arange(g_min, g_max, coarse_res)          # shape: [~88]
    n1     = np.maximum(1, np.round(values[np.newaxis, :] / coarse[:, np.newaxis]))
    r1     = np.abs(values[np.newaxis, :] - n1 * coarse[:, np.newaxis]) / coarse[:, np.newaxis]
    mr1    = r1.mean(axis=1)
    min1   = float(mr1.min())
    # 最小残差が複数候補に並ぶ場合、最大GCDを選ぶ（n が最小 = より自然な解釈）
    best_c = float(coarse[np.where(mr1 <= min1 + 1e-9)[0][-1]])

    # ── Stage 2: 精密探索 ──────────────────────────────────────────────────
    fine_lo = max(g_min, best_c - coarse_res * 2)
    fine_hi = min(g_max, best_c + coarse_res * 2)
    fine    = np.arange(fine_lo, fine_hi, resolution)      # shape: [~40]

    if len(fine) == 0:
        return (best_c, min1)

    n2   = np.maximum(1, np.round(values[np.newaxis, :] / fine[:, np.newaxis]))
    r2   = np.abs(values[np.newaxis, :] - n2 * fine[:, np.newaxis]) / fine[:, np.newaxis]
    mr2  = r2.mean(axis=1)
    min2 = float(mr2.min())
    best_idx = int(np.where(mr2 <= min2 + 1e-9)[0][-1])

    return float(fine[best_idx]), float(mr2[best_idx])


def refine_gcd(
    values: np.ndarray,
    g_init: float,
    n_iter: int = 3,
) -> float:
    """初期候補 g_init を加重中央値で精密化する。

    Args:
        values: IOIまたはタイムスタンプ差分の配列
        g_init: 初期GCD推定値 [秒]
        n_iter: 反復回数

    Returns:
        精密化されたGCD [秒]
    """
    g = g_init
    for _ in range(n_iter):
        n      = np.maximum(1, np.round(values / g).astype(int))
        ratios = values / n
        g      = float(np.median(ratios))
    return g


def calc_gcd_confidence(
    values: np.ndarray,
    g: float,
    tolerance: float,
) -> float:
    """GCDの信頼度を計算する。

    Args:
        values:    IOIまたはタイムスタンプ差分の配列
        g:         GCD推定値 [秒]
        tolerance: 残差の許容割合 (例: 0.15 = 15%)

    Returns:
        confidence [0.0〜1.0]
    """
    n         = np.maximum(1, np.round(values / g).astype(int))
    residuals = np.abs(values - n * g) / g
    return float(np.mean(residuals < tolerance))


def estimate_tempo_from_timestamps(
    timestamps: list[float],
    config: ModuleType,
) -> tuple[float, float] | tuple[None, float]:
    """タイムスタンプリスト（ソート済み）から近似GCDでテンポを推定する。

    Args:
        timestamps: ソート済みタイムスタンプリスト [秒]
        config:     TEMPO_MIN/MAX, GCD_RESOLUTION, GCD_TOLERANCE, GCD_REFINE_ITER,
                    GCD_OCTAVE_RATIOS を持つconfig

    Returns:
        (tempo_bpm, confidence) — 推定不能な場合は (None, 0.0)
    """
    iois = np.diff(np.asarray(timestamps, dtype=float))
    if len(iois) == 0 or float(np.max(iois)) <= 0.0:
        return (None, 0.0)
    # 重複タイムスタンプ（IOI=0）を除去
    iois = iois[iois > 0.0]
    if len(iois) == 0:
        return (None, 0.0)

    # gcdはbeat周期そのものとは限らず、GCD_OCTAVE_RATIOS（最大16）倍まで細かい
    # リズム単位（8th/16th note等）になり得るため、探索下限・最終レンジの上限を
    # その分だけ広げておく（実際のbeatテンポへの補正はparticle_filter.py側で行う）。
    max_subdiv = max(config.GCD_OCTAVE_RATIOS)
    g_min = 60.0 / (config.TEMPO_MAX * max_subdiv)
    g_max = 60.0 / config.TEMPO_MIN

    gcd, _      = approx_gcd(iois, g_min, g_max, config.GCD_RESOLUTION)
    gcd         = refine_gcd(iois, gcd, config.GCD_REFINE_ITER)
    confidence  = calc_gcd_confidence(iois, gcd, config.GCD_TOLERANCE)

    tempo = 60.0 / gcd

    if not (config.TEMPO_MIN <= tempo <= config.TEMPO_MAX * max_subdiv):
        return (None, 0.0)

    return (tempo, confidence)
