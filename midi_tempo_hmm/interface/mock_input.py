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


# ---------------------------------------------------------------------------
# Drum pattern generators
# ---------------------------------------------------------------------------

# Pre-defined patterns: list of (grid_position, note_number) per bar.
# Grid positions are in 16th-note units (0–15 per bar).
_PATTERNS: dict[str, list[tuple[int, int]]] = {
    'basic_rock': [
        # Kick on beats 1 & 3 (grid 0, 8)
        (0,  36), (8,  36),
        # Snare on beats 2 & 4 (grid 4, 12)
        (4,  38), (12, 38),
        # Closed hi-hat on every 8th note (grid 0,2,4,6,8,10,12,14)
        (0,  42), (2,  42), (4,  42), (6,  42),
        (8,  42), (10, 42), (12, 42), (14, 42),
    ],
    'straight_8th': [
        # Kick on beat 1 only (grid 0)
        (0,  36),
        # Snare on beats 2 & 4 (grid 4, 12)
        (4,  38), (12, 38),
        # Closed hi-hat on every 8th note
        (0,  42), (2,  42), (4,  42), (6,  42),
        (8,  42), (10, 42), (12, 42), (14, 42),
    ],
    'hihat_16th': [
        # Kick on beats 1 & 3 (grid 0, 8)
        (0,  36), (8,  36),
        # Snare on beats 2 & 4 (grid 4, 12)
        (4,  38), (12, 38),
        # Closed hi-hat on every 16th note (all 16 grid positions)
        (0,  42), (1,  42), (2,  42), (3,  42),
        (4,  42), (5,  42), (6,  42), (7,  42),
        (8,  42), (9,  42), (10, 42), (11, 42),
        (12, 42), (13, 42), (14, 42), (15, 42),
    ],
    'sparse': [
        # Bass Drum on beat 1 only (grid 0)
        (0,  36),
        # Snare on beat 3 only (grid 8)
        (8,  38),
    ],
}


_DRUM_CHANNEL: int = 9  # MIDI channel 10, 0-indexed


def generate_drum_pattern_events(
    tempo_bpm: float,
    n_bars: int = 8,
    pattern: str = 'basic_rock',
    humanize_ms: float = 10.0,
    seed: int | None = None,
) -> list[tuple[float, int, int]]:
    """Generate note-on timestamps for a repeating drum pattern.

    Each bar is divided into 16 equal 16th-note grid slots.  The chosen
    *pattern* places notes at specific grid positions with the given MIDI note
    numbers.  Multiple notes that fall on the same grid position (e.g. kick and
    hi-hat both on beat 1) are emitted at the same nominal time, giving the
    particle filter realistic "chord" events to process.

    Args:
        tempo_bpm: Target tempo in BPM.
        n_bars: Number of bars to generate.
        pattern: Name of the pattern; one of ``'basic_rock'``, ``'straight_8th'``,
                 ``'hihat_16th'``, or ``'sparse'``.
        humanize_ms: Standard deviation of Gaussian timing jitter in
                     milliseconds applied independently to each event.
        seed: Optional integer seed for the random number generator.  Pass an
              integer to make event generation deterministic; omit or pass
              *None* for non-deterministic behaviour (default).

    Returns:
        List of ``(timestamp_sec, note_number, channel)`` tuples sorted by
        timestamp.  *channel* is always 9 (MIDI drum channel 10, 0-indexed).

    Raises:
        ValueError: If *pattern* is not a recognised pattern name.
    """
    if pattern not in _PATTERNS:
        raise ValueError(
            f"Unknown pattern '{pattern}'. Available: {list(_PATTERNS)}"
        )

    grid = _PATTERNS[pattern]
    sixteenth_sec = (60.0 / tempo_bpm) / 4.0   # duration of one 16th note

    rng = np.random.default_rng(seed)
    events: list[tuple[float, int, int]] = []

    for bar in range(n_bars):
        bar_start = bar * 16 * sixteenth_sec
        for grid_pos, note in grid:
            ideal_t = bar_start + grid_pos * sixteenth_sec
            jitter  = float(rng.normal(0.0, humanize_ms / 1000.0)) if humanize_ms > 0.0 else 0.0
            events.append((ideal_t + jitter, note, _DRUM_CHANNEL))

    events.sort(key=lambda x: x[0])
    return events
