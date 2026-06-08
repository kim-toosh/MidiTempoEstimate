"""Performance tests for the particle filter tempo estimator."""

from __future__ import annotations

import time

import numpy as np
import pytest

import midi_tempo_hmm.config as config
from midi_tempo_hmm.core.particle_filter import ParticleFilter
from midi_tempo_hmm.interface.mock_input import generate_mock_events


def test_single_event_time() -> None:
    """A single update() call must complete in under 10 ms."""
    np.random.seed(0)
    pf = ParticleFilter(config)
    events = generate_mock_events(120.0, 2, humanize_ms=0.0)

    # Prime the filter (first event sets reference)
    pf.update(events[0])

    t0 = time.perf_counter()
    pf.update(events[1])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 10.0, f"Single event took {elapsed_ms:.2f} ms (limit 10 ms)"


def test_batch_event_time() -> None:
    """100 consecutive events: average < 5 ms, maximum < 10 ms per event."""
    np.random.seed(1)
    n_events = 100
    pf = ParticleFilter(config)
    events = generate_mock_events(120.0, n_events, humanize_ms=5.0)

    times_ms: list[float] = []
    for ts in events:
        t0 = time.perf_counter()
        pf.update(ts)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(times_ms))
    max_ms = float(np.max(times_ms))

    print(
        f"\n[Performance] 100 events — avg={avg_ms:.3f} ms, max={max_ms:.3f} ms"
    )

    assert avg_ms < 5.0, f"Average processing time {avg_ms:.3f} ms exceeds 5 ms"
    assert max_ms < 10.0, f"Max processing time {max_ms:.3f} ms exceeds 10 ms"
