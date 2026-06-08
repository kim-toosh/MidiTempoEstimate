"""Mock MIDI event generators for testing and PoC runs."""

from __future__ import annotations

import numpy as np


def generate_mock_events(
    tempo_bpm: float,
    n_beats: int,
    humanize_ms: float = 10.0,
    meter: int = 4,
) -> list[float]:
    """Generate a list of note-on timestamps at a fixed tempo.

    Timestamps are spaced by ``60 / tempo_bpm`` seconds, with optional
    Gaussian jitter added to simulate human timing variation.

    Args:
        tempo_bpm: Target tempo in BPM.
        n_beats: Number of beat events to generate.
        humanize_ms: Standard deviation of the Gaussian timing jitter in
                     milliseconds.  Set to 0 for perfectly metronomic input.
        meter: Beats per measure (informational only; not used here).

    Returns:
        List of timestamps in seconds, length *n_beats*, monotonically
        non-decreasing with high probability for realistic humanize_ms values.
    """
    period_sec = 60.0 / tempo_bpm
    ideal = np.arange(n_beats, dtype=np.float64) * period_sec

    if humanize_ms > 0.0:
        jitter = np.random.normal(0.0, humanize_ms / 1000.0, n_beats)
        timestamps = ideal + jitter
        # Keep first event non-negative
        timestamps = timestamps - min(timestamps[0], 0.0)
    else:
        timestamps = ideal

    return timestamps.tolist()


def generate_tempo_change_events(
    tempo_sequence: list[float],
    beats_per_tempo: list[int],
    humanize_ms: float = 10.0,
) -> list[float]:
    """Generate note-on timestamps with a mid-sequence tempo change.

    Each element of *tempo_sequence* paired with the corresponding element
    of *beats_per_tempo* defines one constant-tempo segment.  Segments are
    concatenated in time so the resulting list is one continuous sequence.

    Args:
        tempo_sequence: List of tempos in BPM, one per segment.
        beats_per_tempo: Number of beats for each segment.
        humanize_ms: Standard deviation of timing jitter in milliseconds.

    Returns:
        List of timestamps in seconds, length ``sum(beats_per_tempo)``.
    """
    all_timestamps: list[float] = []
    current_time = 0.0

    rng = np.random.default_rng()

    for tempo_bpm, n_beats in zip(tempo_sequence, beats_per_tempo):
        period_sec = 60.0 / tempo_bpm
        for _ in range(n_beats):
            jitter = float(rng.normal(0.0, humanize_ms / 1000.0)) if humanize_ms > 0.0 else 0.0
            all_timestamps.append(current_time + jitter)
            current_time += period_sec

    return all_timestamps
