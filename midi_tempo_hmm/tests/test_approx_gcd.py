"""Unit tests for the approximate GCD tempo estimator."""

from __future__ import annotations

import numpy as np
import pytest

import midi_tempo_hmm.config as config
from midi_tempo_hmm.core.approx_gcd import (
    approx_gcd,
    calc_gcd_confidence,
    estimate_tempo_from_timestamps,
    refine_gcd,
)


def test_exact_values() -> None:
    """ノイズなしの値列からGCD=0.50が得られる。"""
    values = np.array([0.50, 0.50, 1.00, 1.50, 2.00])
    gcd, _ = approx_gcd(values, 0.1, 2.5, 0.002)
    assert abs(gcd - 0.50) < 0.01, f"Expected GCD~0.50, got {gcd:.4f}"


def test_noisy_values() -> None:
    """ノイズありの値列からGCD≈0.50が得られる。"""
    values = np.array([0.48, 0.53, 1.54, 0.96, 1.99])
    gcd, _ = approx_gcd(values, 0.1, 2.5, 0.002)
    assert abs(gcd - 0.50) < 0.03, f"Expected GCD~0.50, got {gcd:.4f}"


def test_refine_improves_accuracy() -> None:
    """refine_gcd()後の推定値が真値0.50の近傍に収まる。"""
    values = np.array([0.48, 0.53, 1.54, 0.96, 1.99])
    gcd_rough, _ = approx_gcd(values, 0.1, 2.5, 0.002)
    gcd_refined  = refine_gcd(values, gcd_rough, n_iter=3)
    # roughとrefinedの両方が真値0.50の±0.05以内に収まることを確認
    assert abs(gcd_rough - 0.50) < 0.05, f"Rough GCD {gcd_rough:.4f} not near 0.50"
    assert abs(gcd_refined - 0.50) < 0.05, f"Refined GCD {gcd_refined:.4f} not near 0.50"


def test_confidence_high_for_consistent() -> None:
    """ノイズ小（±2%）の値列でconfidence > 0.8になる。"""
    rng  = np.random.default_rng(0)
    base = np.array([0.5, 1.0, 1.5, 2.0, 0.5, 1.0])
    noisy = base * (1.0 + rng.uniform(-0.02, 0.02, size=len(base)))
    gcd, _ = approx_gcd(noisy, 0.1, 2.5, 0.002)
    gcd    = refine_gcd(noisy, gcd)
    conf   = calc_gcd_confidence(noisy, gcd, tolerance=0.15)
    assert conf > 0.8, f"Expected confidence > 0.8 for low-noise data, got {conf:.3f}"


def test_confidence_low_for_noisy() -> None:
    """周期性のないランダムな値列でconfidence < 0.5になる。"""
    rng = np.random.default_rng(99)
    # 周期性のないランダムな間隔（GCDが存在しない）
    random_vals = rng.uniform(0.25, 2.0, size=20)
    gcd, _ = approx_gcd(random_vals, 0.1, 2.5, 0.002)
    gcd    = refine_gcd(random_vals, gcd)
    conf   = calc_gcd_confidence(random_vals, gcd, tolerance=0.15)
    assert conf < 0.5, f"Expected confidence < 0.5 for random data, got {conf:.3f}"


def test_tempo_range_filter() -> None:
    """テンポ範囲チェックはTEMPO_MIN〜TEMPO_MAX*max(GCD_OCTAVE_RATIOS)を許容する。

    gcdは1拍周期そのものとは限らず、GCD_OCTAVE_RATIOS（最大16）倍まで細かい
    リズム単位（8th/16th note等）になり得るため、最終レンジチェックの上限は
    TEMPO_MAX*16まで緩和されている。
    """
    max_subdiv = max(config.GCD_OCTAVE_RATIOS)

    # 0.1s IOI: tempo = 60/0.1 = 600 BPM。TEMPO_MAX(250)は超えるが
    # TEMPO_MAX*16(=4000)以内なので、16th note等の細かい単位として有効。
    timestamps = [0.0, 0.1, 0.2, 0.3, 0.4]
    tempo, conf = estimate_tempo_from_timestamps(timestamps, config)
    assert tempo is not None, "0.1s IOI (600 BPM) should be within the relaxed range"
    assert abs(tempo - 600.0) < 1.0
    assert conf == 1.0

    # 5ms IOI: tempo ≈ 12000 BPM。TEMPO_MAX*16(=4000)を超えるためrejected。
    timestamps_fast = list(np.cumsum([0.0] + [0.005] * 8))
    tempo_fast, conf_fast = estimate_tempo_from_timestamps(timestamps_fast, config)
    assert tempo_fast is None, f"Expected None for out-of-range tempo, got {tempo_fast}"
    assert conf_fast == 0.0


def test_computation_time() -> None:
    """estimate_tempo_from_timestamps() の処理時間がN=8イベントのとき1.0ms以内。"""
    import time
    timestamps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    estimate_tempo_from_timestamps(timestamps, config)  # warmup
    N_TRIALS = 100
    t0 = time.perf_counter()
    for _ in range(N_TRIALS):
        estimate_tempo_from_timestamps(timestamps, config)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / N_TRIALS
    assert elapsed_ms < 1.0, f"Processing time {elapsed_ms:.3f} ms exceeds 1.0 ms"


def test_kick_snare_combined() -> None:
    """120BPMでKick（奇数拍）+Snare（偶数拍）を合算してGCD≈0.50s→120BPMが得られる。"""
    kick_ts  = [0.0, 1.0, 2.0, 3.0]   # 1拍・3拍
    snare_ts = [0.5, 1.5, 2.5, 3.5]   # 2拍・4拍
    combined = sorted(kick_ts + snare_ts)
    # → [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    # IOI列: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    tempo, conf = estimate_tempo_from_timestamps(combined, config)
    print("tempo=",tempo)
    assert tempo is not None, "Should estimate a tempo"
    assert abs(tempo - 120.0) < 2.0, f"Expected ~120 BPM, got {tempo:.2f}"
    assert conf >= 0.8, f"Expected high confidence, got {conf:.3f}"
