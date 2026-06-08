"""Global configuration constants for the MIDI tempo estimation system."""

N_PARTICLES: int = 500

TEMPO_MIN: float = 30.0   # BPM
TEMPO_MAX: float = 250.0  # BPM
SIGMA_TEMPO: float = 0.5  # BPM, transition noise

SIGMA_OBS_RATIO: float = 0.05  # observation noise as fraction of beat period

BEAT_RATIOS: list[float] = [0.5, 0.667, 1.0, 1.5, 2.0, 3.0, 4.0]
BEAT_RATIO_WEIGHTS: list[float] = [0.1, 0.1, 0.4, 0.15, 0.15, 0.05, 0.05]

ESS_THRESHOLD_RATIO: float = 0.5  # resample when ESS < N_PARTICLES * this value

METER_NUMERATOR: int = 4   # beats per measure (4/4 time)

TEMPO_PRECISION: int = 2   # decimal places for BPM output (0.01 BPM resolution)
