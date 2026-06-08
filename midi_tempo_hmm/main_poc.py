"""Proof-of-concept CLI runner for the MIDI tempo estimator."""

from __future__ import annotations

import argparse
import sys

import numpy as np

import midi_tempo_hmm.config as config
from midi_tempo_hmm.core.particle_filter import ParticleFilter
from midi_tempo_hmm.interface.mock_input import generate_mock_events
from midi_tempo_hmm.output.estimator_result import EstimatorResult


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MIDI tempo estimation PoC using a particle filter."
    )
    p.add_argument("--tempo", type=float, default=120.0, help="Target tempo in BPM (default: 120)")
    p.add_argument("--beats", type=int, default=32, help="Number of beat events (default: 32)")
    p.add_argument(
        "--humanize",
        type=float,
        default=10.0,
        help="Timing jitter std-dev in ms (default: 10.0)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Target tempo : {args.tempo:.1f} BPM")
    print(f"Beats        : {args.beats}")
    print(f"Humanize     : {args.humanize:.1f} ms\n")

    events = generate_mock_events(
        args.tempo, args.beats, humanize_ms=args.humanize
    )

    pf = ParticleFilter(config)
    results: list[EstimatorResult] = []

    header = f"{'Beat':>5}  {'Tempo':>8}  {'Phase':>6}  {'MBeat':>6}  {'Conf':>6}  {'ms':>6}"
    print(header)
    print("-" * len(header))

    for i, ts in enumerate(events):
        r = pf.update(ts)
        if r is None:
            print(f"{i + 1:>5}  (first event — reference timestamp set)")
            continue
        results.append(r)
        print(
            f"{i + 1:>5}  {r.tempo_bpm:>8.2f}  {r.beat_position:>6.3f}"
            f"  {r.measure_beat:>6}  {r.confidence:>6.3f}  {r.processing_time_ms:>6.2f}"
        )

    if not results:
        print("No results — need at least 2 events.")
        return 1

    tempos = np.array([r.tempo_bpm for r in results])
    errors = np.abs(tempos - args.tempo)
    proc_times = np.array([r.processing_time_ms for r in results])

    print("\n" + "=" * 50)
    print("Accuracy summary")
    print("=" * 50)
    print(f"  Mean BPM error   : {errors.mean():.3f} BPM")
    print(f"  Max  BPM error   : {errors.max():.3f} BPM")
    print(f"  Final estimate   : {results[-1].tempo_bpm:.2f} BPM  (target {args.tempo:.1f})")
    print(f"  Mean proc time   : {proc_times.mean():.3f} ms")
    print(f"  Max  proc time   : {proc_times.max():.3f} ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
