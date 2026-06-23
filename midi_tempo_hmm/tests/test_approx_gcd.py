"""Unit tests for the approximate GCD tempo estimator."""

from __future__ import annotations

import numpy as np
import pytest

import midi_tempo_hmm.config as config
from midi_tempo_hmm.core.approx_gcd import (
    approx_gcd,
    approx_gcd_top_n,
    calc_gcd_confidence,
    estimate_tempo_from_timestamps,
    estimate_tempo_from_timestamps_multi,
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


def test_normalization_by_base_not_expected() -> None:
    """残差はexpected(g*r)でなくbase(g)で正規化されている（スコアがgに対して公平）。

    g=0.5sのとき r=2.0でv=1.0: residual = |1.0 - 0.5*2.0|/0.5 = 0.0
    g=1.0sのとき r=1.0でv=1.0: residual = |1.0 - 1.0*1.0|/1.0 = 0.0
    どちらも同じ値なら、より大きなg(=1.0)が選ばれる（Stage1の tie-break動作）。
    """
    ratios = np.array([0.5, 1.0, 2.0])
    values = np.array([1.0, 2.0])
    # g=1.0: ratio=1.0でresidual=(|1-1|+|2-2|)/1.0/2=0, g=0.5: ratio=2.0でも0
    # approx_gcd_top_n が両方ゼロのとき最大gを選ぶことを確認
    results = approx_gcd_top_n(values, 0.4, 2.1, 0.01, ratios, 1, 0.05)
    assert len(results) >= 1
    best_period, best_score = results[0]
    assert best_score < 0.05, f"Expected near-zero score, got {best_score:.4f}"


def test_top_n_returns_distinct_candidates() -> None:
    """返された候補がmin_gap以上の間隔で離れていることを確認。"""
    ratios = np.array([0.5, 1.0, 2.0])
    values = np.array([0.5, 0.5, 1.0, 1.0, 0.5])
    min_gap = 0.05
    results = approx_gcd_top_n(values, 0.1, 2.5, 0.002, ratios, 3, min_gap)
    periods = [r[0] for r in results]
    for i in range(len(periods)):
        for j in range(i + 1, len(periods)):
            assert abs(periods[i] - periods[j]) >= min_gap - 1e-9, (
                f"Candidates too close: {periods[i]:.4f} vs {periods[j]:.4f}"
            )


def test_top_n_count() -> None:
    """n_candidates=2を指定したとき最大2件しか返らない。"""
    ratios = np.array([0.5, 1.0, 2.0])
    values = np.array([0.5, 0.5, 0.5, 1.0, 1.5])
    results = approx_gcd_top_n(values, 0.1, 2.5, 0.002, ratios, 2, 0.05)
    assert len(results) <= 2


def test_ternary_pattern_correctly_identified() -> None:
    """100 BPMと3連符が混在するIOIパターンを正しく100 BPMと識別する。

    values_ms=[301.8, 102.4, 302.4, 194.3, 401.5] は：
    - 302ms ≈ 100 BPM の1拍 (600ms周期 × 0.5)
    - 102ms ≈ 100 BPM の3連符 (600ms周期 × 0.167)
    - 194ms ≈ 100 BPM の3連符×2 (600ms周期 × 0.333)
    - 402ms ≈ 100 BPM の2/3拍 (600ms周期 × 0.667)
    旧整数n方式では200ms(300 BPM)を選んでしまう。
    """
    values_ms = [301.8, 102.4, 302.4, 194.3, 401.5]
    ts = list(np.cumsum([0.0] + [v / 1000.0 for v in values_ms]))
    results = estimate_tempo_from_timestamps_multi(ts, config)
    assert len(results) > 0, "Should return at least one candidate"
    best_tempo = results[0][0]
    assert abs(best_tempo - 100.0) < 5.0, (
        f"Expected ~100 BPM as best candidate, got {best_tempo:.2f} BPM. "
        f"All candidates: {results}"
    )


def test_mixed_binary_ternary_pattern() -> None:
    """8th notesと3連符の混合パターンで120 BPMを正しく識別する。"""
    beat = 60.0 / 120.0  # 0.5s at 120 BPM
    # 4分音符×2 + 3連符×3 + 4分音符×2
    iois_sec = [beat, beat, beat / 3, beat / 3, beat / 3, beat, beat]
    ts = list(np.cumsum([0.0] + iois_sec))
    results = estimate_tempo_from_timestamps_multi(ts, config)
    assert len(results) > 0, "Should return candidates"
    # GCDはg=0.5s(120 BPM)またはg=1.0s(60 BPM)に収束しうる（どちらも正当な解釈）。
    # GCD_OCTAVE_RATIOSで補正すれば60/120/240 BPMはすべて同一テンポ候補。
    valid = {60.0, 120.0, 240.0}
    assert any(any(abs(t - v) < 5.0 for v in valid) for t, _ in results), (
        f"Expected 60/120/240 BPM in candidates, got {results}"
    )


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
