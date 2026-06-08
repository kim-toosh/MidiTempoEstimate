"""Real-time MIDI tempo estimation from a live MIDI input port."""

from __future__ import annotations

import argparse
import sys
import time

import midi_tempo_hmm.config as config
from midi_tempo_hmm.core.particle_filter import ParticleFilter
from midi_tempo_hmm.interface.midi_input import MidiInputHandler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time MIDI tempo estimator (particle filter)."
    )
    p.add_argument(
        "--port-index",
        type=int,
        default=0,
        metavar="N",
        help="MIDI input port index (default: 0)",
    )
    p.add_argument(
        "--list-ports",
        action="store_true",
        help="Print available MIDI input ports and exit",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_ports:
        ports = MidiInputHandler.list_ports()
        if not ports:
            print("No MIDI input ports found.")
            return 1
        print("Available MIDI input ports:")
        for i, name in enumerate(ports):
            print(f"  {i}: {name}")
        return 0

    pf = ParticleFilter(config)
    handler = MidiInputHandler(pf, port_index=args.port_index)

    try:
        handler.start()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        handler.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
