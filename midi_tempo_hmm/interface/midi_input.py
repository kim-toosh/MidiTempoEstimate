"""Real-time MIDI input handler using python-rtmidi."""

from __future__ import annotations

import time
from typing import Callable, Optional

import rtmidi

from midi_tempo_hmm.core.particle_filter import ParticleFilter
from midi_tempo_hmm.output.estimator_result import EstimatorResult


class MidiInputHandler:
    """Receive MIDI note-on events and feed them into a ParticleFilter in real time.

    Callbacks are invoked from rtmidi's internal MIDI thread.  CPython's GIL
    serialises the NumPy operations inside ParticleFilter.update(), so no
    additional locking is required for correctness.  If update() ever becomes
    slow enough to risk dropping MIDI events, switch to the Queue design
    described in _on_midi_message() below.

    Args:
        particle_filter: A fully initialised ParticleFilter instance.
        port_index: Zero-based index into the list returned by list_ports().
        on_result: Optional callable invoked with each EstimatorResult from the
                   MIDI thread.  When provided, console printing is suppressed.
                   Typical use: pass ``queue.Queue.put`` to route results to a
                   GUI thread safely.
    """

    # MIDI status byte masks
    _NOTE_ON = 0x90
    _STATUS_MASK = 0xF0

    def __init__(
        self,
        particle_filter: ParticleFilter,
        port_index: int = 0,
        on_result: Optional[Callable[[EstimatorResult], None]] = None,
    ) -> None:
        self._pf = particle_filter
        self._port_index = port_index
        self._on_result = on_result
        self._midi_in: Optional[rtmidi.MidiIn] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def list_ports() -> list[str]:
        """Return the names of all available MIDI input ports.

        Returns:
            List of port name strings, in the same order as their indices.
        """
        tmp = rtmidi.MidiIn()
        ports = [tmp.get_port_name(i) for i in range(tmp.get_port_count())]
        del tmp
        return ports

    def start(self) -> None:
        """Open the configured port and register the MIDI callback.

        Raises:
            RuntimeError: If no MIDI ports are available or the port index is
                          out of range.
        """
        ports = self.list_ports()
        if not ports:
            raise RuntimeError("No MIDI input ports found.")
        if self._port_index >= len(ports):
            raise RuntimeError(
                f"Port index {self._port_index} out of range "
                f"(available: {len(ports)} ports)."
            )

        self._midi_in = rtmidi.MidiIn()
        self._midi_in.open_port(self._port_index)
        self._midi_in.set_callback(self._on_midi_message)
        print(f"[MidiInputHandler] Listening on port {self._port_index}: {ports[self._port_index]}")

    def stop(self) -> None:
        """Close the MIDI port and remove the callback."""
        if self._midi_in is not None:
            self._midi_in.cancel_callback()
            self._midi_in.close_port()
            self._midi_in = None
            print("[MidiInputHandler] Stopped.")

    # ------------------------------------------------------------------
    # MIDI callback
    # ------------------------------------------------------------------

    def _on_midi_message(self, message: tuple, data: object = None) -> None:
        """Process a single incoming MIDI message.

        Called from rtmidi's internal MIDI thread for every received message.
        Only NOTE_ON events with velocity > 0 are forwarded to the filter;
        all other messages are silently discarded.

        rtmidi passes ``message`` as ``(bytes_list, delta_time_seconds)``.
        The delta_time value is intentionally ignored; instead, the absolute
        wall-clock time from time.perf_counter() is used so that timestamps
        are consistent across the session and unaffected by rtmidi's internal
        clock drift.

        Thread-safety note
        ------------------
        ParticleFilter.update() mutates NumPy arrays, but CPython's GIL
        ensures that only one thread runs Python bytecode at a time.  Since
        all heavy work inside update() is within NumPy C extensions (which
        release the GIL only briefly and atomically per array operation), the
        result is effectively serialised and safe without an explicit lock.

        Alternative — Queue-based async design
        ---------------------------------------
        If update() ever becomes a bottleneck (e.g. larger N_PARTICLES or
        additional processing), consider:

            import queue, threading

            # In __init__:
            self._queue: queue.Queue = queue.Queue()
            self._worker = threading.Thread(target=self._process_loop, daemon=True)

            # In _on_midi_message (lightweight — just enqueue the timestamp):
            self._queue.put(time.perf_counter())

            # In _process_loop (runs in a dedicated thread):
            while True:
                ts = self._queue.get()
                result = self._pf.update(ts)
                ...

        This keeps the MIDI callback thread free to receive the next event
        immediately, at the cost of a small latency increase.

        Args:
            message: Tuple of ``(midi_bytes, delta_time)`` provided by rtmidi.
            data: Optional user data registered with set_callback(); unused here.
        """
        midi_bytes, _delta_time = message

        # Filter: NOTE_ON with velocity > 0 only (velocity == 0 is a NOTE_OFF)
        status = midi_bytes[0] & self._STATUS_MASK
        velocity = midi_bytes[2] if len(midi_bytes) >= 3 else 0
        if status != self._NOTE_ON or velocity == 0:
            return

        timestamp_sec = time.perf_counter()
        result = self._pf.update(timestamp_sec)

        if result is None:
            print("[midi] first event — reference time set", flush=True)
            return

        if self._on_result is not None:
            # Route to GUI / queue; suppress console output
            self._on_result(result)
        else:
            print(
                f"tempo={result.tempo_bpm:>7.2f} BPM  "
                f"beat={result.beat_position:.3f}  "
                f"m_beat={result.measure_beat}  "
                f"conf={result.confidence:.3f}  "
                f"({result.processing_time_ms:.2f} ms)",
                flush=True,
            )
