"""Drum MIDI event IOI map visualizer.

Left panel  — scrolling IOI map:
  - X-axis : time  (right edge = now, scrolls left as time passes)
  - Y-axis : MIDI note number with GM drum labels
  - Bar    : horizontal line spanning the IOI for each note hit
  - Color  : plasma colormap mapped to velocity (soft=dark, hard=bright)

Right panel — IOI history (latest 8 per note):
  - Y-axis : MIDI note number
  - X-axis : accumulated IOI seconds
  - Bars   : stacked horizontal segments, oldest=dark → newest=bright (plasma)
  - Text   : sum and average of the latest 8 IOI values

Run:
    python -m midi_tempo_hmm.drum_visualizer
    python -m midi_tempo_hmm.drum_visualizer --window 20
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

import tkinter as tk
from tkinter import ttk

# Allow running as a plain script (`python drum_visualizer.py`) in addition to
# the canonical `python -m midi_tempo_hmm.drum_visualizer`.  When executed
# directly, __package__ is None and the project root is not on sys.path, so
# we insert it here before any package-relative import.
if __package__ is None or __package__ == '':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection
import numpy as np
import rtmidi

import midi_tempo_hmm.config as config

# ── GM Drum note labels ───────────────────────────────────────────────────────
GM_DRUMS: dict[int, str] = {
    35: 'AcBD',   36: 'BD',     37: 'Rim',    38: 'Snare',
    39: 'Clap',   40: 'Snr2',   41: 'Tom4',   42: 'HH-C',
    43: 'Tom3',   44: 'HH-P',   45: 'Tom2',   46: 'HH-O',
    47: 'Tom1',   48: 'TomH2',  49: 'Crash1', 50: 'TomH',
    51: 'Ride',   52: 'China',  53: 'RdBell', 54: 'Tamb',
    55: 'Splash', 56: 'Cowbel', 57: 'Crash2', 59: 'Ride2',
}

def _note_label(n: int) -> str:
    return f"{n} {GM_DRUMS[n]}" if n in GM_DRUMS else str(n)

# ── Theme ─────────────────────────────────────────────────────────────────────
BG        = '#1E1E2E'
BG_GRAPH  = '#181825'
BG_WIDGET = '#313244'
BG_HOVER  = '#45475A'
FG        = '#CDD6F4'
FG_DIM    = '#6C7086'
BORDER    = '#45475A'

CMAP            = mpl.colormaps['plasma']  # shared by both graphs
IOI_HISTORY_LEN = 8                        # max IOI samples kept per note


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class DrumEvent:
    timestamp: float           # time.perf_counter() at the moment of the hit
    note:      int             # MIDI note number (0–127)
    velocity:  int             # MIDI velocity (1–127)
    ioi:       Optional[float] # seconds since previous hit of the same note
                               # None for the first hit of each note


# ── MIDI capture ──────────────────────────────────────────────────────────────

class RawMidiCapture:
    """Minimal rtmidi wrapper: NOTE_ON only → callback(timestamp, note, velocity)."""

    _NOTE_ON     = 0x90
    _STATUS_MASK = 0xF0

    def __init__(self, port_index: int, on_event) -> None:
        self._port_index = port_index
        self._on_event   = on_event   # callable(float, int, int)
        self._midi_in    = None

    @staticmethod
    def list_ports() -> list[str]:
        tmp   = rtmidi.MidiIn()
        ports = [tmp.get_port_name(i) for i in range(tmp.get_port_count())]
        del tmp
        return ports

    def start(self) -> None:
        ports = self.list_ports()
        if not ports:
            raise RuntimeError("No MIDI input ports found.")
        if self._port_index >= len(ports):
            raise RuntimeError(f"Port index {self._port_index} out of range.")
        self._midi_in = rtmidi.MidiIn()
        self._midi_in.open_port(self._port_index)
        self._midi_in.set_callback(self._callback)

    def stop(self) -> None:
        if self._midi_in is not None:
            self._midi_in.cancel_callback()
            self._midi_in.close_port()
            self._midi_in = None

    def _callback(self, message, data=None) -> None:
        midi_bytes, _ = message
        if (midi_bytes[0] & self._STATUS_MASK) != self._NOTE_ON:
            return
        note     = midi_bytes[1]
        velocity = midi_bytes[2] if len(midi_bytes) >= 3 else 0
        if velocity == 0:
            return   # velocity-0 NOTE_ON = NOTE_OFF
        self._on_event(time.perf_counter(), note, velocity)


# ── Main window ───────────────────────────────────────────────────────────────

class DrumVisualizerApp:
    """Two-panel drum visualizer: scrolling IOI map (left) + IOI history bars (right)."""

    def __init__(self, root: tk.Tk, window_sec: float = 10.0) -> None:
        self.root        = root
        self._window_sec = window_sec

        # Time-series state
        self._events:     list[DrumEvent]  = []
        self._last_time:  dict[int, float] = {}
        self._seen_notes: set[int]         = set()
        self._queue:      queue.Queue      = queue.Queue()
        self._capture:    Optional[RawMidiCapture] = None
        self._total_events: int = 0

        # IOI history state (right panel)
        self._ioi_history: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=IOI_HISTORY_LEN)
        )
        self._ioi_changed: bool = False

        root.title("Drum MIDI Visualizer")
        root.minsize(1000, 500)
        root.configure(bg=BG)

        self._configure_ttk_style()
        self._build_ui()

        root.protocol('WM_DELETE_WINDOW', self._on_close)
        root.after(0,  self._auto_connect)
        root.after(50, self._tick)

    def _configure_ttk_style(self) -> None:
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TCombobox',
                        fieldbackground=BG_WIDGET, background=BG_WIDGET,
                        foreground=FG, selectbackground=BG_WIDGET,
                        selectforeground=FG, arrowcolor=FG)
        style.map('TCombobox',
                  fieldbackground=[('readonly', BG_WIDGET)],
                  background=[('readonly', BG_WIDGET)],
                  foreground=[('readonly', FG)])

    def _mk_btn(self, parent: tk.Widget, text: str, command,
                width: int = 0) -> tk.Button:
        kw: dict = dict(text=text, command=command,
                        bg=BG_WIDGET, fg=FG,
                        activebackground=BG_HOVER, activeforeground=FG,
                        relief=tk.FLAT, bd=0, padx=10, pady=4,
                        font=('Helvetica', 12))
        if width:
            kw['width'] = width
        btn = tk.Button(parent, **kw)
        btn.bind('<Enter>', lambda e: btn.config(bg=BG_HOVER))
        btn.bind('<Leave>', lambda e: btn.config(bg=BG_WIDGET))
        return btn

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        toolbar = tk.Frame(self.root, bg=BG)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(8, 4))
        self._build_toolbar(toolbar)

        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                               bg=BORDER, sashwidth=2, sashrelief=tk.FLAT)
        paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        left_host  = tk.Frame(paned, bg=BG)
        right_host = tk.Frame(paned, bg=BG)
        paned.add(left_host,  width=820, stretch='always')
        paned.add(right_host, width=480, stretch='always')

        self._build_left_canvas(left_host)
        self._build_right_canvas(right_host)

    def _build_toolbar(self, parent: tk.Frame) -> None:
        tk.Label(parent, text="MIDI PORT:", bg=BG, fg=FG,
                 font=('Helvetica', 11)).pack(side=tk.LEFT, padx=(0, 4))

        self._port_var   = tk.StringVar()
        self._port_combo = ttk.Combobox(parent, textvariable=self._port_var,
                                        state='readonly', width=30,
                                        font=('Helvetica', 11))
        self._port_combo.pack(side=tk.LEFT, padx=(0, 4))

        self._mk_btn(parent, "⟳", self._refresh_ports,
                     width=2).pack(side=tk.LEFT, padx=(0, 4))

        self._conn_btn = self._mk_btn(parent, "Connect", self._toggle_connect)
        self._conn_btn.pack(side=tk.LEFT, padx=(0, 4))

        self._mk_btn(parent, "Clear", self._clear).pack(side=tk.LEFT)

        self._count_var = tk.StringVar(value="Events: 0")
        tk.Label(parent, textvariable=self._count_var, bg=BG, fg=FG,
                 font=('Helvetica', 10)).pack(side=tk.RIGHT)

        self._refresh_ports()

    def _build_left_canvas(self, parent: tk.Frame) -> None:
        """Scrolling IOI time-series."""
        self.fig = Figure(facecolor=BG, tight_layout=True)
        self.ax  = self.fig.add_subplot(111)

        self.ax.set_facecolor(BG_GRAPH)
        self.ax.tick_params(colors=FG, labelcolor=FG, which='both')
        for sp in self.ax.spines.values():
            sp.set_edgecolor(BORDER)

        self.ax.set_xlabel('Time   (past ←  →  now)', color=FG)
        self.ax.set_xlim(-self._window_sec, 0.3)
        self.ax.set_ylim(34, 82)
        self.ax.axvline(0, color=FG_DIM, linewidth=1, linestyle='--', alpha=0.5, zorder=1)
        self.ax.grid(True, color=BG_WIDGET, linestyle='--', linewidth=0.5, axis='x', zorder=0)

        self._bar_collection = LineCollection([], zorder=3)
        self.ax.add_collection(self._bar_collection)
        self._dot_scatter = self.ax.scatter([], [], zorder=5, linewidths=0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_right_canvas(self, parent: tk.Frame) -> None:
        """IOI history stacked bar chart."""
        self.fig2 = Figure(facecolor=BG, tight_layout=True)
        self.ax2  = self.fig2.add_subplot(111)

        self.ax2.set_facecolor(BG_GRAPH)
        self.ax2.tick_params(colors=FG, labelcolor=FG, which='both')
        for sp in self.ax2.spines.values():
            sp.set_edgecolor(BORDER)

        self.ax2.set_xlabel(f'IOI (s)   oldest → newest  (last {IOI_HISTORY_LEN})', color=FG)
        self.ax2.set_xlim(0, 2.0)
        self.ax2.set_ylim(34, 82)
        self.ax2.grid(True, color=BG_WIDGET, linestyle='--', linewidth=0.5, axis='x', zorder=0)

        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=parent)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── Port helpers ──────────────────────────────────────────────────────────

    def _refresh_ports(self) -> None:
        ports = RawMidiCapture.list_ports()
        if ports:
            self._port_combo['values'] = ports
            self._port_combo.current(0)
        else:
            self._port_combo['values'] = ['(no ports found)']
            self._port_combo.current(0)

    def _auto_connect(self) -> None:
        vals = self._port_combo['values']
        if vals and vals[0] != '(no ports found)':
            self._connect()

    def _toggle_connect(self) -> None:
        if self._capture is None:
            self._connect()
        else:
            self._disconnect()

    def _connect(self) -> None:
        idx = self._port_combo.current()
        if idx < 0:
            return
        try:
            self._capture = RawMidiCapture(
                idx,
                on_event=lambda t, n, v: self._queue.put((t, n, v)),
            )
            self._capture.start()
            self._conn_btn.config(text="Disconnect")
        except RuntimeError as e:
            print(f"[drum_viz] {e}", flush=True)

    def _disconnect(self) -> None:
        if self._capture is not None:
            self._capture.stop()
            self._capture = None
        self._conn_btn.config(text="Connect")

    def _clear(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._events.clear()
        self._last_time.clear()
        self._seen_notes.clear()
        self._ioi_history.clear()
        self._ioi_changed = False
        self._total_events = 0
        self._count_var.set("Events: 0")

        self._bar_collection.set_segments([])
        self._dot_scatter.set_offsets(np.zeros((0, 2)))
        self.canvas.draw_idle()

        self.ax2.cla()
        self._style_ax2()
        self.canvas2.draw_idle()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        """Drain the MIDI queue then redraw. Rescheduled every 50 ms."""
        try:
            while True:
                timestamp, note, velocity = self._queue.get_nowait()
                ioi = timestamp - self._last_time[note] if note in self._last_time else None
                self._last_time[note] = timestamp
                self._seen_notes.add(note)
                self._events.append(DrumEvent(timestamp, note, velocity, ioi))
                self._total_events += 1

                if ioi is not None:
                    bpm = 60.0 / ioi
                    if config.TEMPO_MIN <= bpm <= config.TEMPO_MAX:
                        self._ioi_history[note].append(ioi)
                    else:
                        self._ioi_history[note].clear()
                    self._ioi_changed = True
        except queue.Empty:
            pass

        self._draw_timeseries()
        if self._ioi_changed:
            self._draw_ioi_history()
            self._ioi_changed = False

        self.root.after(50, self._tick)

    # ── Left panel: time-series ───────────────────────────────────────────────

    def _draw_timeseries(self) -> None:
        now  = time.perf_counter()
        tmin = now - self._window_sec

        while self._events and self._events[0].timestamp < tmin:
            self._events.pop(0)

        bar_segments: list = []
        bar_colors:   list = []
        bar_widths:   list = []
        dot_xy:       list = []
        dot_colors:   list = []
        dot_sizes:    list = []

        for ev in self._events:
            x    = ev.timestamp - now
            c    = CMAP(ev.velocity / 127)
            v    = ev.velocity / 127          # 0.0 〜 1.0
            lw   = 1.0 + v * 15.0            # linewidth: 1.0 (soft) 〜 16.0 (hard)
            ds   = (lw + 5) ** 2             # dot diameter always larger than linewidth
            dot_xy.append([x, ev.note])
            dot_colors.append(c)
            dot_sizes.append(ds)
            if ev.ioi is not None:
                x_start = max(ev.timestamp - ev.ioi, tmin) - now
                bar_segments.append([(x_start, ev.note), (x, ev.note)])
                bar_colors.append(c)
                bar_widths.append(lw)

        self._bar_collection.set_segments(bar_segments)
        if bar_colors:
            self._bar_collection.set_colors(bar_colors)
            self._bar_collection.set_linewidths(bar_widths)

        if dot_xy:
            self._dot_scatter.set_offsets(np.array(dot_xy))
            self._dot_scatter.set_facecolors(dot_colors)
            self._dot_scatter.set_sizes(dot_sizes)
        else:
            self._dot_scatter.set_offsets(np.zeros((0, 2)))

        if self._seen_notes:
            notes = sorted(self._seen_notes)
            self.ax.set_ylim(notes[0] - 1.5, notes[-1] + 1.5)
            self.ax.set_yticks(notes)
            self.ax.set_yticklabels([_note_label(n) for n in notes], fontsize=8)
            self.ax.tick_params(axis='y', labelcolor=FG, colors=FG)

        self._count_var.set(f"Events: {self._total_events}")
        self.canvas.draw_idle()

    # ── Right panel: IOI history ──────────────────────────────────────────────

    def _style_ax2(self) -> None:
        """Re-apply axis style after cla()."""
        self.ax2.set_facecolor(BG_GRAPH)
        self.ax2.tick_params(colors=FG, labelcolor=FG, which='both')
        for sp in self.ax2.spines.values():
            sp.set_edgecolor(BORDER)
        self.ax2.set_xlabel(
            f'IOI (s)   oldest → newest  (last {IOI_HISTORY_LEN})', color=FG
        )
        self.ax2.grid(
            True, color=BG_WIDGET, linestyle='--', linewidth=0.5, axis='x', zorder=0
        )

    def _draw_ioi_history(self) -> None:
        self.ax2.cla()
        self._style_ax2()

        notes = sorted(self._seen_notes)
        if not notes:
            self.ax2.set_xlim(0, 2.0)
            self.canvas2.draw_idle()
            return

        max_x = 0.0

        for note in notes:
            history = list(self._ioi_history.get(note, []))
            if not history:
                continue

            n   = len(history)
            x   = 0.0
            for i, ioi in enumerate(history):   # oldest first
                norm = i / (IOI_HISTORY_LEN - 1) if IOI_HISTORY_LEN > 1 else 1.0
                self.ax2.barh(
                    note, ioi, left=x, height=0.6,
                    color=CMAP(norm), alpha=0.85, zorder=3,
                )
                x += ioi

            total = sum(history)
            avg   = total / n
            max_x = max(max_x, total)

            self.ax2.text(
                total + max(max_x * 0.02, 0.01),
                note,
                f"∑{total:.2f}s  μ{avg:.3f}s  {60/avg:.1f}BPM",
                va='center', ha='left', fontsize=7, color=FG, zorder=4,
            )

        # Y-axis: mirror the left panel
        self.ax2.set_ylim(notes[0] - 1.5, notes[-1] + 1.5)
        self.ax2.set_yticks(notes)
        self.ax2.set_yticklabels([_note_label(n) for n in notes], fontsize=8)
        self.ax2.tick_params(axis='y', labelcolor=FG, colors=FG)

        # X-axis: leave 35% extra room for text labels
        self.ax2.set_xlim(0, max(max_x * 1.45, 0.5))

        self.canvas2.draw_idle()

    # ── Close ─────────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        self._disconnect()
        self.root.destroy()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Drum MIDI IOI Visualizer')
    p.add_argument('--port-index', type=int,   default=0,
                   help='MIDI input port index (default: 0)')
    p.add_argument('--window',     type=float, default=10.0,
                   help='visible time window in seconds (default: 10)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    DrumVisualizerApp(root, window_sec=args.window)
    root.mainloop()


if __name__ == '__main__':
    main()
