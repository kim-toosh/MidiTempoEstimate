"""Drum instrument identification and beat-correlation weight mapping.

Each General MIDI percussion note is assigned a weight that reflects how
strongly it correlates with the beat grid:

  1.0  strong-beat instruments (kick, snare) — high influence on likelihood
  0.5  mixed-subdivision instruments (toms, some cymbals)
  0.2  subdivision-only instruments (hi-hats, ride) — lower influence

The weight map is designed to be injectable: pass a custom dict to
``get_drum_weight()`` to override the defaults.  This leaves room for
future auto-generated or learned weight maps without changing the API.
"""

from __future__ import annotations

# General MIDI Percussion Map: note number → beat-correlation weight
DRUM_WEIGHT_MAP: dict[int, float] = {
    35: 1.0,   # Acoustic Bass Drum
    36: 1.0,   # Bass Drum 1          ← strong beat marker (beats 1 & 3)
    37: 0.5,   # Side Stick
    38: 0.9,   # Acoustic Snare       ← strong beat marker (beats 2 & 4)
    39: 0.4,   # Hand Clap
    40: 0.9,   # Electric Snare
    41: 0.5,   # Low Floor Tom
    42: 0.2,   # Closed Hi-Hat        ← subdivision only
    43: 0.5,   # High Floor Tom
    44: 0.3,   # Pedal Hi-Hat
    45: 0.5,   # Low Tom
    46: 0.2,   # Open Hi-Hat          ← subdivision only
    47: 0.5,   # Low-Mid Tom
    48: 0.5,   # Hi-Mid Tom
    49: 0.8,   # Crash Cymbal 1       ← downbeat marker
    50: 0.5,   # High Tom
    51: 0.2,   # Ride Cymbal 1        ← subdivision only
    52: 0.7,   # Chinese Cymbal
    53: 0.4,   # Ride Bell
    54: 0.3,   # Tambourine
    55: 0.7,   # Splash Cymbal
    56: 0.4,   # Cowbell
    57: 0.8,   # Crash Cymbal 2
    58: 0.4,   # Vibraslap
    59: 0.2,   # Ride Cymbal 2
}

DEFAULT_DRUM_WEIGHT: float = 0.4     # fallback for notes not in the map
DEFAULT_NON_DRUM_WEIGHT: float = 1.0  # used when not on a drum channel


def get_drum_weight(
    note_number: int,
    is_drum_channel: bool,
    weight_map: dict[int, float] | None = None,
) -> float:
    """Return the beat-correlation weight for a MIDI note.

    Args:
        note_number: MIDI note number (0–127).
        is_drum_channel: True when the event originates from the drum channel
                         (MIDI channel 10, 0-indexed as 9).
        weight_map: Optional custom weight dict.  When *None*, the built-in
                    :data:`DRUM_WEIGHT_MAP` is used.  Pass a dict generated
                    by a learning algorithm to override defaults at runtime.

    Returns:
        Weight in (0.0, 1.0].  Non-drum channels always return
        :data:`DEFAULT_NON_DRUM_WEIGHT` (1.0).
    """
    if not is_drum_channel:
        return DEFAULT_NON_DRUM_WEIGHT

    effective_map = weight_map if weight_map is not None else DRUM_WEIGHT_MAP
    return effective_map.get(note_number, DEFAULT_DRUM_WEIGHT)
