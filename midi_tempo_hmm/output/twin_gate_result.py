"""Data class for TwinGate tempo estimation output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from midi_tempo_hmm.core.instrument_category import InstrumentCategory


@dataclass
class TwinGateResult:
    """Result of a single TwinGate (MidiInputGate → KalmanGate) update step.

    gcd_tempo is octave-adjusted (same correction as particle_filter.py) so it
    represents a beat-period tempo rather than the raw subdivision GCD.
    """

    # --- 確定出力 ---
    tempo_bpm         : float           # KalmanGate後の確定テンポ [BPM]
    tempo_bpm_str     : str             # f"{tempo_bpm:.2f}" 表示用

    # --- MidiInputGate情報 ---
    category          : InstrumentCategory
    raw_ioi           : Optional[float]   # 同カテゴリ生IOI [秒]
    gcd_tempo         : Optional[float]   # オクターブ補正済みGCDテンポ [BPM]
    gcd_confidence    : float             # GCD信頼度 [0〜1]
    gcd_period        : Optional[float]   # gcd_tempo対応の周期 [秒]
    kick_count        : int
    snare_count       : int
    hihat_count       : int
    others_count      : int

    # --- KalmanGate情報 ---
    gate_accepted     : bool
    reject_reason     : str               # ""/"octave"/"mahal"/"confidence"/"no_gcd"
    predicted_tempo   : Optional[float]   # カルマン予測値 [BPM]
    innovation        : Optional[float]   # 候補 − 予測値 [BPM]
    mahal_distance    : Optional[float]   # マハラノビス距離²
    mahal_threshold   : float             # ゲート閾値
    kalman_variance   : float             # 事後分散
    kalman_gain       : float             # カルマンゲイン (ACCEPT時のみ有効)

    # --- メタ情報 ---
    event_count       : int
    processing_time_ms: float

    # --- GCDバッファスナップショット ---
    gcd_buffer        : list  # list[tuple[float, InstrumentCategory]] - category-tagged timestamps
