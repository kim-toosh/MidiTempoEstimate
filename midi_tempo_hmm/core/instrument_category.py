"""Drum instrument categories for per-category observation models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class InstrumentCategory(Enum):
    KICK   = auto()
    SNARE  = auto()
    HIHAT  = auto()
    OTHERS = auto()


@dataclass
class CategorizedEvent:
    timestamp_sec: float
    note_number:   int
    velocity:      int
    channel:       int
    category:      InstrumentCategory
    drum_weight:   float  # 1.0 for non-drum channel; category-specific for drum channel
