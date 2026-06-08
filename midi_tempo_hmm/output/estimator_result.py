"""Data class for tempo estimation output."""

from dataclasses import dataclass


@dataclass
class EstimatorResult:
    """Result of a single particle filter update step.

    Attributes:
        tempo_bpm: Estimated tempo in BPM, rounded to TEMPO_PRECISION decimal places.
        beat_position: Phase within the current beat, in [0.0, 1.0).
        measure_beat: Beat index within the current measure, in [0, METER_NUMERATOR).
        confidence: Normalised ESS in [0.0, 1.0]; higher means more certain.
        processing_time_ms: Wall-clock time spent in the update() call (milliseconds).
    """

    tempo_bpm: float
    beat_position: float
    measure_beat: int
    confidence: float
    processing_time_ms: float
