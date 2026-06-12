"""Data class for tempo estimation output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from midi_tempo_hmm.core.instrument_category import InstrumentCategory

if TYPE_CHECKING:
    from midi_tempo_hmm.core.kalman_gate import KalmanGateResult


@dataclass
class EstimatorResult:
    """Result of a single particle filter update step.

    Attributes:
        tempo_bpm: Estimated tempo in BPM, rounded to TEMPO_PRECISION decimal places.
        beat_position: Phase within the current beat, in [0.0, 1.0).
        measure_beat: Beat index within the current measure, in [0, METER_NUMERATOR).
        confidence: Normalised ESS in [0.0, 1.0]; higher means more certain.
        processing_time_ms: Wall-clock time spent in the update() call (milliseconds).
        is_converged: True once the particle filter has reached CONVERGENCE_THRESHOLD.
        autocorr_tempo: Best autocorrelation tempo estimate (BPM) before convergence,
                        None after convergence or when unavailable.
        gate_result: Kalman-gate outcome for this step; None if gate not available.
    """

    tempo_bpm: float
    beat_position: float
    measure_beat: int
    confidence: float
    processing_time_ms: float
    is_converged: bool = False
    autocorr_tempo: Optional[float] = None
    gate_result: Optional[KalmanGateResult] = None
    last_category: Optional[InstrumentCategory] = None  # category of the last processed event
    category_counts: Optional[dict] = None              # per-category event counts from MidiInputGate
    gcd_tempo:      Optional[float] = None              # 近似GCD推定テンポ [BPM]
    gcd_confidence: float = 0.0                         # GCD信頼度 [0〜1]
    ioi_sec:         Optional[float] = None             # 直前イベントの同カテゴリIOI [秒]
    note_number:     Optional[int] = None               # 直前イベントの MIDI note number
    gcd_iois:        Optional[list[float]] = None       # GCD推定に使用したIOI列 [秒]
