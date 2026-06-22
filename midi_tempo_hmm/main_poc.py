"""Proof-of-concept CLI runner for the MIDI tempo estimator."""

from __future__ import annotations

import argparse
import sys

import numpy as np

import midi_tempo_hmm.config as config
from midi_tempo_hmm.interface.mock_input import (
    generate_drum_pattern_events,
    generate_mock_events,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MIDI tempo estimation PoC."
    )
    p.add_argument("--engine", type=str, default="twin",
                   choices=["twin", "particle"],
                   help="Estimation engine: twin (default) or particle")
    p.add_argument("--tempo", type=float, default=120.0,
                   help="Target tempo in BPM (default: 120)")
    p.add_argument("--beats", type=int, default=32,
                   help="Number of beat events (default: 32; ignored with --drum-pattern)")
    p.add_argument("--humanize", type=float, default=10.0,
                   help="Timing jitter std-dev in ms (default: 10.0)")
    p.add_argument("--drum-pattern", type=str, default="none",
                   choices=["none", "basic_rock", "hihat_16th", "sparse"],
                   help="Drum pattern to use instead of simple beat events (default: none)")
    p.add_argument("--bars", type=int, default=8,
                   help="Number of bars for drum pattern (default: 8)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# TwinGate engine
# ---------------------------------------------------------------------------

def _build_events(args: argparse.Namespace) -> list[tuple[float, int, int]]:
    if args.drum_pattern != "none":
        print(f"Drum pattern : {args.drum_pattern}  ({args.bars} bars)\n")
        raw = generate_drum_pattern_events(
            args.tempo, n_bars=args.bars,
            pattern=args.drum_pattern,
            humanize_ms=args.humanize,
        )
        return [(ts, note, ch) for ts, note, ch in raw]
    else:
        print(f"Beats        : {args.beats}\n")
        timestamps = generate_mock_events(args.tempo, args.beats, humanize_ms=args.humanize)
        return [(ts, 36, 9) for ts in timestamps]   # note 36 = Bass Drum on drum channel


def _run_twin(args: argparse.Namespace) -> int:
    from midi_tempo_hmm.core.twin_gate import TwinGate
    from midi_tempo_hmm.output.twin_gate_result import TwinGateResult

    events = _build_events(args)
    tg = TwinGate(config)
    results: list[TwinGateResult] = []

    header = (
        f"{'#':>5}  {'Tempo':>8}  {'GCD_T':>7}  {'GCDConf':>7}"
        f"  {'Gate':>4}  {'KVar':>6}  {'Innov':>8}"
        f"  {'Phase':>6}  {'PErr':>6}  {'Sync':>4}  {'ms':>6}"
    )
    sep = "─" * len(header)
    print(header)
    print(sep)

    def _fmt_result(idx: int, r: TwinGateResult) -> None:
        gcd_str   = f"{r.gcd_tempo:.2f}" if r.gcd_tempo is not None else "---"
        gate_str  = "AC" if r.gate_accepted else "RE"
        innov_str = f"{r.innovation:+.2f}" if r.innovation is not None else "---"
        phase_str = f"{r.phase:.3f}" if r.phase is not None else "---"
        perr_str  = f"{r.phase_error:+.3f}" if r.phase_error is not None else "---"
        sync_str  = "YES" if r.is_phase_synced else "no"
        print(
            f"{idx:>5}  {r.tempo_bpm:>8.2f}  {gcd_str:>7}  {r.gcd_confidence:>7.2f}"
            f"  {gate_str:>4}  {r.kalman_variance:>6.2f}  {innov_str:>8}"
            f"  {phase_str:>6}  {perr_str:>6}  {sync_str:>4}  {r.processing_time_ms:>6.2f}"
        )

    reject_counts: dict[str, int] = {}
    for i, (ts, note, channel) in enumerate(events):
        r = tg.update(ts, note_number=note, velocity=100, channel=channel)
        if r is None:
            continue
        results.append(r)
        _fmt_result(i + 1, r)
        if not r.gate_accepted and r.reject_reason not in ("no_gcd",):
            reject_counts[r.reject_reason] = reject_counts.get(r.reject_reason, 0) + 1

    if not results:
        print("No results — need at least 2 events for GCD estimation.")
        return 1

    tempos     = np.array([r.tempo_bpm for r in results])
    errors     = np.abs(tempos - args.tempo)
    proc_times = np.array([r.processing_time_ms for r in results])

    accepted     = sum(1 for r in results if r.gate_accepted)
    gcd_avail    = sum(1 for r in results if r.gcd_tempo is not None)
    accept_rate  = accepted / max(1, gcd_avail) * 100

    print("\n" + "=" * 55)
    print("TwinGate summary")
    print("=" * 55)
    print(f"  Total events     : {len(results)}")
    print(f"  ACCEPT rate      : {accept_rate:.1f}%  ({accepted}/{gcd_avail} GCD-available)")
    print(f"  REJECT (mahal)   : {reject_counts.get('mahal', 0)}")
    print(f"  REJECT (octave)  : {reject_counts.get('octave', 0)}")
    print(f"  REJECT (conf)    : {reject_counts.get('confidence', 0)}")
    print(f"  Mean BPM error   : {errors.mean():.3f} BPM")
    print(f"  Max  BPM error   : {errors.max():.3f} BPM")
    print(f"  Final estimate   : {results[-1].tempo_bpm:.2f} BPM  (target {args.tempo:.1f})")
    print(f"  Kalman variance  : {results[-1].kalman_variance:.2f}")
    print(f"  Mean proc time   : {proc_times.mean():.3f} ms")
    print(f"  Max  proc time   : {proc_times.max():.3f} ms")
    return 0


# ---------------------------------------------------------------------------
# Particle filter engine (legacy)
# ---------------------------------------------------------------------------

def _run_particle(args: argparse.Namespace) -> int:
    from midi_tempo_hmm.core.particle_filter import ParticleFilter
    from midi_tempo_hmm.output.estimator_result import EstimatorResult

    print("Engine       : particle\n")
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
        timestamps = generate_mock_events(args.tempo, args.beats, humanize_ms=args.humanize)
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

    def print_result(idx: int, r: EstimatorResult) -> None:
        conv_str     = "YES" if r.is_converged else "no"
        acorr_str    = f"{r.autocorr_tempo:.1f}" if r.autocorr_tempo is not None else "---"
        gcd_str      = f"{r.gcd_tempo:.1f}" if r.gcd_tempo is not None else "---"
        gcd_conf_str = f"{r.gcd_confidence:.2f}"
        print(
            f"{idx:>5}  {r.tempo_bpm:>8.2f}  {r.beat_position:>6.3f}"
            f"  {r.measure_beat:>6}  {r.confidence:>6.3f}"
            f"  {conv_str:>5}  {acorr_str:>7}  {gcd_str:>7}  {gcd_conf_str:>7}"
            f"  {r.processing_time_ms:>6.2f}"
        )

    for i, (ts, note, channel) in enumerate(events):
        r = pf.update(ts, note_number=note, channel=channel)
        if r is None:
            continue
        results.append(r)
        print_result(i + 1, r)

    final = pf.flush(events[-1][0] + pf.span_same_time)
    if final is not None:
        results.append(final)
        print_result(len(events), final)

    if not results:
        print("No results — need at least 2 strong-beat events.")
        return 1

    tempos = np.array([r.tempo_bpm for r in results])
    errors = np.abs(tempos - args.tempo)
    proc_times = np.array([r.processing_time_ms for r in results])

    print("\n" + "=" * 55)
    print("Accuracy summary (particle)")
    print("=" * 55)
    print(f"  Mean BPM error   : {errors.mean():.3f} BPM")
    print(f"  Max  BPM error   : {errors.max():.3f} BPM")
    print(f"  Final estimate   : {results[-1].tempo_bpm:.2f} BPM  (target {args.tempo:.1f})")
    print(f"  Converged        : {results[-1].is_converged}")
    print(f"  Mean proc time   : {proc_times.mean():.3f} ms")
    print(f"  Max  proc time   : {proc_times.max():.3f} ms")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    print(f"Engine       : {args.engine}")
    print(f"Target tempo : {args.tempo:.1f} BPM")
    print(f"Humanize     : {args.humanize:.1f} ms")

    if args.engine == "twin":
        return _run_twin(args)
    else:
        return _run_particle(args)


if __name__ == "__main__":
    sys.exit(main())
