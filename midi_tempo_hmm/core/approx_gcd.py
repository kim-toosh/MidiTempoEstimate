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


def approx_gcd_top_n(
    values: np.ndarray,
    g_min: float,
    g_max: float,
    resolution: float,
    ratios: np.ndarray,
    n_candidates: int,
    min_gap: float,
) -> list[tuple[float, float]]:
    """ratiosベースのスコアリングで上位n_candidates件のGCD候補を返す（2段階探索）。

    正規化方式: residual = |v - g*r| / g  (rはratiosの中で最良のもの)
    これによりgの大きさによらず公平なスコア比較が可能になる。

    Args:
        values:       IOI配列 [秒]
        g_min:        探索下限 [秒]
        g_max:        探索上限 [秒]
        resolution:   候補GCDの刻み幅 [秒]
        ratios:       ratio候補配列 (例: [0.5, 1.0, 1.5, 2.0])
        n_candidates: 返す候補数の上限
        min_gap:      候補間の最小差 [秒]（近すぎる候補を除外）

    Returns:
        [(gcd_period, score), ...] スコア昇順（良い順）、最大n_candidates件
    """
    coarse_res = resolution * 10
    ratios = np.asarray(ratios, dtype=float)

    def _score_all(candidates: np.ndarray) -> np.ndarray:
        # candidates (M,), values (N,), ratios (K,)
        # expected: (M, 1, K) * broadcast → (M, N, K) via values (1, N, 1)
        expected  = candidates[:, None, None] * ratios[None, None, :]  # (M,1,K)
        diff      = np.abs(values[None, :, None] - expected)           # (M,N,K)
        residuals = diff / candidates[:, None, None]                   # (M,N,K)
        best_per  = residuals.min(axis=2)                              # (M,N)
        return best_per.mean(axis=1)                                   # (M,)

    # 全探索範囲を fine resolution で一気にスコアリング。
    # 2段階探索（粗→精）だとStage1の勝者周辺しかStage2が見えず、
    # オクターブ違いの候補を取りこぼすため。
    fine = np.arange(g_min, g_max, resolution)
    if len(fine) == 0:
        return []

    scores = _score_all(fine)

    # スコア昇順にソートして上位候補を greedy に選択
    order = np.argsort(scores)
    selected: list[tuple[float, float]] = []
    for idx in order:
        period = float(fine[idx])
        score  = float(scores[idx])
        # 採用済み候補との差がmin_gap未満なら重複とみなしスキップ
        if any(abs(period - s[0]) < min_gap for s in selected):
            continue
        selected.append((period, score))
        if len(selected) >= n_candidates:
            break

    return selected


def refine_gcd(
    values: np.ndarray,
    g_init: float,
    ratios: Optional[np.ndarray] = None,
    n_iter: int = 3,
) -> float:
    """初期候補 g_init を精密化する。

    Args:
        values: IOIまたはタイムスタンプ差分の配列
        g_init: 初期GCD推定値 [秒]
        ratios: ratio候補配列。Noneの場合は従来の整数n方式を使用
        n_iter: 反復回数

    Returns:
        精密化されたGCD [秒]
    """
    g = g_init
    if ratios is None:
        for _ in range(n_iter):
            n = np.maximum(1, np.round(values / g).astype(int))
            g = float(np.median(values / n))
    else:
        ratios = np.asarray(ratios, dtype=float)
        for _ in range(n_iter):
            best_idx = np.argmin(np.abs(values[:, None] - g * ratios[None, :]), axis=1)
            best_r   = ratios[best_idx]
            valid    = best_r > 0
            if not valid.any():
                break
            g = float(np.median(values[valid] / best_r[valid]))
    return g


def calc_gcd_confidence(
    values: np.ndarray,
    g: float,
    tolerance: float,
    ratios: Optional[np.ndarray] = None,
) -> float:
    """GCDの信頼度を計算する。

    Args:
        values:    IOIまたはタイムスタンプ差分の配列
        g:         GCD推定値 [秒]
        tolerance: 残差の許容割合 (例: 0.15 = 15%)
        ratios:    ratio候補配列。Noneの場合は従来の整数n方式を使用

    Returns:
        confidence [0.0〜1.0]
    """
    if ratios is None:
        n         = np.maximum(1, np.round(values / g).astype(int))
        residuals = np.abs(values - n * g) / g
    else:
        ratios    = np.asarray(ratios, dtype=float)
        best_idx  = np.argmin(np.abs(values[:, None] - g * ratios[None, :]), axis=1)
        best_r    = ratios[best_idx]
        residuals = np.abs(values - g * best_r) / g
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
    gcd         = refine_gcd(iois, gcd, n_iter=config.GCD_REFINE_ITER)
    confidence  = calc_gcd_confidence(iois, gcd, config.GCD_TOLERANCE)

    tempo = 60.0 / gcd

    if not (config.TEMPO_MIN <= tempo <= config.TEMPO_MAX * max_subdiv):
        return (None, 0.0)

    return (tempo, confidence)


def estimate_tempo_from_timestamps_multi(
    timestamps: list[float],
    config: ModuleType,
) -> list[tuple[float, float]]:
    """ratiosベースのスコアリングでテンポ候補を複数返す。

    Args:
        timestamps: ソート済みタイムスタンプリスト [秒]
        config:     GCD_RATIOS, GCD_N_CANDIDATES, GCD_CANDIDATE_MIN_GAP 等を持つconfig

    Returns:
        [(tempo_bpm, confidence), ...] スコア良い順、最大GCD_N_CANDIDATES件
        推定不能な場合は空リスト
    """
    iois = np.diff(np.asarray(timestamps, dtype=float))
    if len(iois) == 0 or float(np.max(iois)) <= 0.0:
        return []
    iois = iois[iois > 0.0]
    if len(iois) == 0:
        return []

    ratios      = np.asarray(getattr(config, 'GCD_RATIOS', [0.5, 1.0, 2.0]), dtype=float)
    max_subdiv  = max(config.GCD_OCTAVE_RATIOS)
    g_min       = 60.0 / (config.TEMPO_MAX * max_subdiv)
    g_max       = 60.0 / config.TEMPO_MIN
    n_cands     = getattr(config, 'GCD_N_CANDIDATES', 3)
    min_gap     = getattr(config, 'GCD_CANDIDATE_MIN_GAP', 0.05)

    raw = approx_gcd_top_n(iois, g_min, g_max, config.GCD_RESOLUTION,
                            ratios, n_cands, min_gap)

    # GCD候補を各オクターブ倍率で展開し、TEMPO_MIN〜TEMPO_MAX範囲の候補をすべて収集。
    # 大きなg（低テンポ）が先頭になっても、正しいオクターブ候補がリストに含まれる。
    oct_ratios = getattr(config, 'GCD_OCTAVE_RATIOS', [1, 2, 4, 8, 16])
    seen: list[float] = []
    results: list[tuple[float, float]] = []

    for gcd_period, _ in raw:
        gcd_r = refine_gcd(iois, gcd_period, ratios=ratios, n_iter=config.GCD_REFINE_ITER)
        conf  = calc_gcd_confidence(iois, gcd_r, config.GCD_TOLERANCE, ratios=ratios)
        for oct_r in oct_ratios:
            beat_period = gcd_r / oct_r
            tempo = 60.0 / beat_period
            if config.TEMPO_MIN <= tempo <= config.TEMPO_MAX:
                if not any(abs(tempo - t) < 2.0 for t in seen):
                    results.append((tempo, conf))
                    seen.append(tempo)
                    if len(results) >= n_cands:
                        return results

    return results
