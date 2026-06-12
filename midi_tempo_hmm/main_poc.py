"""Proof-of-concept CLI runner for the MIDI tempo estimator."""

from __future__ import annotations

import argparse
import sys

import numpy as np

import midi_tempo_hmm.config as config
from midi_tempo_hmm.core.particle_filter import ParticleFilter
from midi_tempo_hmm.interface.mock_input import (
    generate_drum_pattern_events,
    generate_mock_events,
)
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
    p.add_argument(
        "--drum-pattern",
        type=str,
        default="none",
        choices=["none", "basic_rock", "hihat_16th", "sparse"],
        help="Drum pattern to use instead of simple beat events (default: none)",
    )
    p.add_argument(
        "--bars",
        type=int,
        default=8,
        help="Number of bars for drum pattern (default: 8; ignored when --drum-pattern=none)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Target tempo : {args.tempo:.1f} BPM")
    print(f"Humanize     : {args.humanize:.1f} ms")

    # Build event list as (timestamp, note_or_None, channel_or_None)
    if args.drum_pattern != "none":
        print(f"Drum pattern : {args.drum_pattern}  ({args.bars} bars)\n")
        raw = generate_drum_pattern_events(
            args.tempo, n_bars=args.bars,
            pattern=args.drum_pattern,
            humanize_ms=args.humanize,
        )
        events: list[tuple[float, int | None, int | None]] = [
            (ts, note, ch) for ts, note, ch in raw
        ]
    else:
        print(f"Beats        : {args.beats}\n")
        timestamps = generate_mock_events(
            args.tempo, args.beats, humanize_ms=args.humanize
        )
        events = [(ts, None, None) for ts in timestamps]

    pf = ParticleFilter(config)
    results: list[EstimatorResult] = []

    header = (
        f"{'#':>5}  {'Tempo':>8}  {'Phase':>6}  {'MBeat':>6}"
        f"  {'Conf':>6}  {'Conv':>5}  {'ACORR':>7}  {'GCD':>7}  {'GCDConf':>7}"
        f"  {'ms':>6}"
    )
    print(header)
    print("-" * len(header))

    for i, (ts, note, channel) in enumerate(events):
        r = pf.update(ts, note_number=note, channel=channel)
        if r is None:
            continue  # subdivision event or first strong-beat event
        results.append(r)
        conv_str     = "YES" if r.is_converged else "no"
        acorr_str    = f"{r.autocorr_tempo:.1f}" if r.autocorr_tempo is not None else "---"
        gcd_str      = f"{r.gcd_tempo:.1f}" if r.gcd_tempo is not None else "---"
        gcd_conf_str = f"{r.gcd_confidence:.2f}"
        print(
            f"{i + 1:>5}  {r.tempo_bpm:>8.2f}  {r.beat_position:>6.3f}"
            f"  {r.measure_beat:>6}  {r.confidence:>6.3f}"
            f"  {conv_str:>5}  {acorr_str:>7}  {gcd_str:>7}  {gcd_conf_str:>7}"
            f"  {r.processing_time_ms:>6.2f}"
        )

    if not results:
        print("No results — need at least 2 strong-beat events.")
        return 1

    tempos = np.array([r.tempo_bpm for r in results])
    errors = np.abs(tempos - args.tempo)
    proc_times = np.array([r.processing_time_ms for r in results])

    print("\n" + "=" * 55)
    print("Accuracy summary")
    print("=" * 55)
    print(f"  Mean BPM error   : {errors.mean():.3f} BPM")
    print(f"  Max  BPM error   : {errors.max():.3f} BPM")
    print(f"  Final estimate   : {results[-1].tempo_bpm:.2f} BPM  (target {args.tempo:.1f})")
    print(f"  Converged        : {results[-1].is_converged}")
    print(f"  Mean proc time   : {proc_times.mean():.3f} ms")
    print(f"  Max  proc time   : {proc_times.max():.3f} ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
