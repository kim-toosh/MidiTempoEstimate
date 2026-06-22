"""GUI front-end for the MIDI tempo estimator — TwinGate engine (Tkinter + matplotlib).

Modes
-----
Mock (Beat) : 単純なビートイベント列を自動生成。
Mock (Rock) : basic_rock ドラムパターンを自動生成。
Mock (16th) : hihat_16th ドラムパターンを自動生成。
MIDI Input  : リアルタイム MIDI 入力。START でポートを開く。

Threading model
---------------
Mock  : ワーカースレッドが result_queue に put
MIDI  : rtmidi コールバックスレッドが result_queue に put
Main  : root.after(50ms) ごとに _poll_results() でキューをドレイン
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
from collections import deque
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import midi_tempo_hmm.config as config
from midi_tempo_hmm.core.instrument_category import InstrumentCategory
from midi_tempo_hmm.core.twin_gate import TwinGate
from midi_tempo_hmm.interface.mock_input import (
    generate_drum_pattern_events,
    generate_mock_events,
)
from midi_tempo_hmm.output.twin_gate_result import TwinGateResult

# ── Colours ───────────────────────────────────────────────────────────────────
BG            = '#1E1E2E'
BG_GRAPH      = '#181825'
BG_WIDGET     = '#313244'
BG_HOVER      = '#45475A'
FG            = '#CDD6F4'
FG_DIM        = '#6C7086'
BORDER        = '#45475A'
ACCENT_RED    = '#E74C3C'
ACCENT_BLUE   = '#3498DB'
ACCENT_GREEN  = '#2ECC71'
ACCENT_ORANGE = '#E67E22'
COLOR_BROWN   = '#8B4513'
COLOR_PURPLE  = '#9B59B6'
COLOR_YELLOW  = '#F39C12'

_CAT_COLOR = {
    InstrumentCategory.KICK:   ACCENT_RED,
    InstrumentCategory.SNARE:  ACCENT_BLUE,
    InstrumentCategory.HIHAT:  ACCENT_GREEN,
    InstrumentCategory.OTHERS: FG_DIM,
}

_MODE_ITEMS = ['Mock (Beat)', 'Mock (Rock)', 'Mock (16th)', 'MIDI Input']
_MIDI_INDEX = 3


# ── MIDI listener ─────────────────────────────────────────────────────────────

class _TwinGateMidiListener:
    _NOTE_ON     = 0x90
    _STATUS_MASK = 0xF0

    def __init__(self, twin_gate: TwinGate, port_index: int = 0,
                 on_result=None) -> None:
        self._tg         = twin_gate
        self._port_index = port_index
        self._on_result  = on_result
        self._midi_in    = None

    @staticmethod
    def list_ports() -> list[str]:
        try:
            import rtmidi
            tmp   = rtmidi.MidiIn()
            ports = [tmp.get_port_name(i) for i in range(tmp.get_port_count())]
            del tmp
            return ports
        except Exception:
            return []

    def start(self) -> None:
        import rtmidi
        ports = self.list_ports()
        if not ports:
            raise RuntimeError("No MIDI input ports found.")
        if self._port_index >= len(ports):
            raise RuntimeError(f"Port index {self._port_index} out of range.")
        self._midi_in = rtmidi.MidiIn()
        self._midi_in.open_port(self._port_index)
        self._midi_in.set_callback(self._on_midi_message)
        print(f"[midi] listening on port {self._port_index}: {ports[self._port_index]}", flush=True)

    def stop(self) -> None:
        if self._midi_in is not None:
            self._midi_in.cancel_callback()
            self._midi_in.close_port()
            self._midi_in = None
            print("[midi] stopped.", flush=True)

    def _on_midi_message(self, message, data=None) -> None:
        midi_bytes, _ = message
        status   = midi_bytes[0] & self._STATUS_MASK
        velocity = midi_bytes[2] if len(midi_bytes) >= 3 else 0
        if status != self._NOTE_ON or velocity == 0:
            return
        note    = midi_bytes[1]
        channel = midi_bytes[0] & 0x0F
        ts      = time.perf_counter()
        result  = self._tg.update(ts, note_number=note, velocity=velocity, channel=channel)
        if result is not None and self._on_result is not None:
            self._on_result(result)


# ── Main window ───────────────────────────────────────────────────────────────

class TempoEstimatorApp:
    """TwinGate tempo estimator GUI (Tkinter + matplotlib)."""

    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args

        root.title("MIDI Tempo Estimator - TwinGate")
        root.minsize(1000, 560)
        root.configure(bg=BG)

        # ── Engine & state ────────────────────────────────────────────────────
        self.twin_gate    = TwinGate(config)
        self.is_running   = False
        self.lock         = threading.Lock()
        self.result_queue: queue.Queue = queue.Queue()

        self._history: deque[TwinGateResult]            = deque(maxlen=64)
        self._gcd_buf_history: deque[tuple[int, list]]  = deque(maxlen=32)
        self._midi_listener: Optional[_TwinGateMidiListener] = None

        self._total_events  = 0
        self._accept_count  = 0
        self._reject_mahal  = 0
        self._reject_octave = 0
        self._reject_conf   = 0

        self._prev_beat_count = 0

        self._configure_ttk_style()
        self._build_ui()

        root.protocol('WM_DELETE_WINDOW', self._on_close)
        root.after(50, self._poll_results)
        root.after(0, self._auto_start)

    # ── ttk style ─────────────────────────────────────────────────────────────

    def _configure_ttk_style(self) -> None:
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TCombobox',
                        fieldbackground=BG_WIDGET, background=BG_WIDGET,
                        foreground=FG, selectbackground=BG_WIDGET,
                        selectforeground=FG, arrowcolor=FG)
        style.map('TCombobox',
                  fieldbackground=[('readonly', BG_WIDGET)],
                  foreground=[('readonly', FG)],
                  selectbackground=[('readonly', BG_WIDGET)],
                  selectforeground=[('readonly', FG)])
        style.configure('TSeparator', background=BORDER)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self._pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                                    bg=BORDER, sashwidth=2, sashrelief=tk.FLAT)
        self._pane.pack(fill=tk.BOTH, expand=True)

        left  = tk.Frame(self._pane, bg=BG, width=230)
        right = tk.Frame(self._pane, bg=BG)
        self._pane.add(left,  minsize=180, stretch='never')
        self._pane.add(right, minsize=500, stretch='always')

        self._build_left(left)
        self._build_right(right)

        self.root.after(100, lambda: self._pane.sash_place(0, 230, 0))

    def _build_left(self, parent: tk.Frame) -> None:
        # Bottom controls (packed first so they stay at bottom)
        bottom = tk.Frame(parent, bg=BG)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=14, pady=14)

        # Top data display
        top = tk.Frame(parent, bg=BG)
        top.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=14, pady=(14, 0))

        # ── TOP: data labels ─────────────────────────────────────────────────

        def _hdr(text: str, margin_top: int = 8) -> None:
            tk.Label(top, text=text, bg=BG, fg=FG_DIM,
                     font=('Courier New', 9)).pack(
                anchor='w', pady=(margin_top, 0))

        _hdr('TEMPO', margin_top=0)
        self._tempo_var = tk.StringVar(value='---')
        tk.Label(top, textvariable=self._tempo_var, bg=BG, fg=FG,
                 font=('Courier New', 28, 'bold')).pack(anchor='w')

        _hdr('GATE')
        self._gate_lbl = tk.Label(top, text='---', bg=BG, fg=FG_DIM,
                                   font=('Courier New', 13, 'bold'))
        self._gate_lbl.pack(anchor='w')

        _hdr('GCD TEMPO')
        self._gcd_tempo_lbl = tk.Label(top, text='---', bg=BG, fg=ACCENT_BLUE,
                                        font=('Courier New', 13, 'bold'))
        self._gcd_tempo_lbl.pack(anchor='w')
        self._gcd_conf_lbl = tk.Label(top, text='---', bg=BG, fg=FG_DIM,
                                       font=('Courier New', 11))
        self._gcd_conf_lbl.pack(anchor='w')

        _hdr('KALMAN')
        self._kalman_var_lbl   = tk.Label(top, text='var:   ---', bg=BG, fg=FG,
                                           font=('Courier New', 11))
        self._kalman_pred_lbl  = tk.Label(top, text='pred:  ---', bg=BG, fg=FG,
                                           font=('Courier New', 11))
        self._kalman_innov_lbl = tk.Label(top, text='innov: ---', bg=BG, fg=FG,
                                           font=('Courier New', 11))
        for lbl in (self._kalman_var_lbl, self._kalman_pred_lbl, self._kalman_innov_lbl):
            lbl.pack(anchor='w')

        _hdr('EVENT COUNTS')
        self._counts_lbl = tk.Label(top, text='K:  0 S:  0\nH:  0 O:  0',
                                     bg=BG, fg=FG_DIM, font=('Courier New', 11),
                                     justify='left')
        self._counts_lbl.pack(anchor='w')

        _hdr('LAST HIT')
        self._last_hit_lbl = tk.Label(top, text='---', bg=BG, fg=FG_DIM,
                                       font=('Courier New', 13, 'bold'))
        self._last_hit_lbl.pack(anchor='w')

        _hdr('PHASE')
        self._phase_lbl = tk.Label(top, text='phase: ---', bg=BG, fg=FG,
                                    font=('Courier New', 11))
        self._phase_lbl.pack(anchor='w')
        self._phase_canvas = tk.Canvas(top, height=10, bg=BG_GRAPH,
                                        highlightthickness=0)
        self._phase_canvas.pack(fill=tk.X, pady=(2, 0))
        self._phase_err_lbl = tk.Label(top, text='err:   ---', bg=BG, fg=FG_DIM,
                                        font=('Courier New', 11))
        self._phase_err_lbl.pack(anchor='w')

        beat_row = tk.Frame(top, bg=BG)
        beat_row.pack(anchor='w', pady=(2, 0))
        tk.Label(beat_row, text='BEAT  ', bg=BG, fg=FG_DIM,
                 font=('Courier New', 11)).pack(side=tk.LEFT)
        self._beat_canvas = tk.Canvas(beat_row, width=18, height=18,
                                       bg=BG, highlightthickness=0)
        self._beat_canvas.pack(side=tk.LEFT)
        self._beat_oval = self._beat_canvas.create_oval(2, 2, 16, 16,
                                                         fill=FG_DIM, outline='')

        self._next_beat_lbl = tk.Label(top, text='next:  --- ms', bg=BG, fg=FG_DIM,
                                        font=('Courier New', 11))
        self._next_beat_lbl.pack(anchor='w')

        # ── BOTTOM: controls ─────────────────────────────────────────────────

        self._run_status_lbl = tk.Label(bottom, text='■  STOPPED',
                                         bg=BG, fg=FG_DIM,
                                         font=('Courier New', 12))
        self._run_status_lbl.pack(anchor='w', pady=(0, 4))

        for text, cmd in [
            ('▶  START', self._start),
            ('■  STOP',  self._stop_and_summarize),
            ('↺  RESET', self._reset),
        ]:
            self._mk_btn(bottom, text, cmd).pack(fill=tk.X, pady=2)

        # INPUT MODE
        tk.Label(bottom, text='INPUT MODE', bg=BG, fg=FG_DIM,
                 font=('Courier New', 9)).pack(anchor='w', pady=(10, 0))

        if self.args.mode == 'mock':
            if self.args.drum_pattern == 'basic_rock':
                initial = 1
            elif self.args.drum_pattern == 'hihat_16th':
                initial = 2
            else:
                initial = 0
        else:
            initial = _MIDI_INDEX

        self._mode_var   = tk.StringVar(value=_MODE_ITEMS[initial])
        self._mode_combo = ttk.Combobox(bottom, textvariable=self._mode_var,
                                        values=_MODE_ITEMS, state='readonly',
                                        font=('Helvetica', 11))
        self._mode_combo.pack(fill=tk.X, pady=4)
        self._mode_combo.bind('<<ComboboxSelected>>', lambda e: self._on_mode_changed())

        # MIDI PORT section (shown only in MIDI Input mode)
        self._port_section = tk.Frame(bottom, bg=BG)
        tk.Label(self._port_section, text='MIDI PORT', bg=BG, fg=FG_DIM,
                 font=('Courier New', 9)).pack(anchor='w')
        port_row = tk.Frame(self._port_section, bg=BG)
        port_row.pack(fill=tk.X)
        self._port_combo = ttk.Combobox(port_row, state='readonly',
                                        font=('Helvetica', 11))
        self._port_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._mk_btn(port_row, '⟳', self._refresh_ports,
                     width=2).pack(side=tk.LEFT, padx=(4, 0))

        self._refresh_ports()
        self._on_mode_changed()

    def _build_right(self, parent: tk.Frame) -> None:
        self.fig = Figure(facecolor=BG, tight_layout=False)
        self.fig.subplots_adjust(left=0.08, right=0.88, top=0.95,
                                  bottom=0.06, hspace=0.85)
        gs = self.fig.add_gridspec(4, 1, height_ratios=[3, 1, 2, 1])

        # ── 上段: テンポ推移 ──────────────────────────────────────────────────
        self.ax = self.fig.add_subplot(gs[0])
        self._style_ax(self.ax, ylabel='Tempo (BPM)')
        self.ax.set_ylim(70, 150)

        self._line_gcd,    = self.ax.plot([], [], color=ACCENT_BLUE, linewidth=1.5,
                                           linestyle='--', alpha=0.85,
                                           label='GCD tempo', zorder=2)
        self._line_kalman, = self.ax.plot([], [], color=ACCENT_RED, linewidth=2.5,
                                           label='Kalman tempo', zorder=3)
        self._line_reject, = self.ax.plot([], [], color=ACCENT_ORANGE, linestyle='none',
                                           marker='^', markersize=9,
                                           label='REJECT', zorder=4)
        self._ci_fill = None
        self.ax.legend(loc='upper left', fontsize=8, framealpha=0.3,
                       facecolor=BG_WIDGET, edgecolor=BORDER, labelcolor=FG)

        # ── 下段: デバッグ ────────────────────────────────────────────────────
        self.ax_debug  = self.fig.add_subplot(gs[1])
        self.ax_debug2 = self.ax_debug.twinx()
        self._style_debug_axes()

        self._line_conf,  = self.ax_debug.plot([], [], color=ACCENT_GREEN, linewidth=1.5,
                                                 label='GCD conf', zorder=3)
        self._line_var,   = self.ax_debug2.plot([], [], color=COLOR_BROWN, linewidth=1.5,
                                                  label='K var', zorder=2)
        self._line_mahal, = self.ax_debug2.plot([], [], color=COLOR_PURPLE, linewidth=1.5,
                                                  label='Mahal', zorder=3)
        self._line_thr,   = self.ax_debug2.plot([], [], color=ACCENT_RED, linewidth=1.0,
                                                  linestyle='--', alpha=0.7,
                                                  label='threshold', zorder=2)

        # ── 3段: GCDバッファIOI履歴 ───────────────────────────────────────────
        self.ax_gcd_hist = self.fig.add_subplot(gs[2])
        self.ax_gcd_hist.set_facecolor(BG_GRAPH)
        self.ax_gcd_hist.tick_params(colors=FG, labelcolor=FG, which='both', labelsize=7)
        for spine in self.ax_gcd_hist.spines.values():
            spine.set_edgecolor(BORDER)
        self.ax_gcd_hist.set_xlabel('Event #', color=FG, fontsize=8)
        self.ax_gcd_hist.set_ylabel('IOI (s)', color=FG, fontsize=8)
        self.ax_gcd_hist.grid(True, color=BG_WIDGET, linestyle='--', linewidth=0.5)

        # ── 4段: Phase Oscillator ─────────────────────────────────────────────
        self.ax_phase = self.fig.add_subplot(gs[3])
        self.ax_phase.set_facecolor(BG_GRAPH)
        self.ax_phase.tick_params(colors=FG, labelcolor=FG, which='both', labelsize=7)
        for spine in self.ax_phase.spines.values():
            spine.set_edgecolor(BORDER)
        self.ax_phase.set_xlabel('Event #', color=FG, fontsize=8)
        self.ax_phase.set_ylabel('Phase', color=COLOR_PURPLE, fontsize=8)
        self.ax_phase.set_ylim(0, 1)
        self.ax_phase.grid(True, color=BG_WIDGET, linestyle='--', linewidth=0.5)
        self._line_phase, = self.ax_phase.plot([], [], color=COLOR_PURPLE,
                                                linewidth=1.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _style_ax(self, ax, ylabel: str = '') -> None:
        ax.set_facecolor(BG_GRAPH)
        ax.tick_params(colors=FG, labelcolor=FG, which='both')
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.set_xlabel('Event #', color=FG, fontsize=9)
        ax.set_ylabel(ylabel, color=FG, fontsize=9)
        ax.grid(True, color=BG_WIDGET, linestyle='--', linewidth=0.5)

    def _style_debug_axes(self) -> None:
        ax  = self.ax_debug
        ax2 = self.ax_debug2
        ax.set_facecolor(BG_GRAPH)
        for a in (ax, ax2):
            a.tick_params(colors=FG, labelcolor=FG, which='both', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.set_xlabel('Event #', color=FG, fontsize=9)
        ax.set_ylabel('Conf', color=ACCENT_GREEN, fontsize=8)
        ax2.set_ylabel('Var / Mahal', color=COLOR_PURPLE, fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.grid(True, color=BG_WIDGET, linestyle='--', linewidth=0.5)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _mk_btn(self, parent: tk.Widget, text: str, cmd,
                width: int | None = None) -> tk.Button:
        kw: dict = {}
        if width is not None:
            kw['width'] = width
        return tk.Button(
            parent, text=text, command=cmd,
            bg=BG_WIDGET, fg=FG,
            activebackground=BG_HOVER, activeforeground=FG,
            relief=tk.FLAT, padx=6, pady=8,
            font=('Helvetica', 13), cursor='hand2',
            **kw,
        )

    # ── Port / mode helpers ───────────────────────────────────────────────────

    def _refresh_ports(self) -> None:
        self._port_combo['values'] = ()
        ports = _TwinGateMidiListener.list_ports()
        if ports:
            self._port_combo['values'] = ports
            self._port_combo.current(0)
        else:
            self._port_combo['values'] = ('(no ports found)',)
            self._port_combo.current(0)

    def _on_mode_changed(self) -> None:
        is_midi = (self._mode_var.get() == _MODE_ITEMS[_MIDI_INDEX])
        if is_midi:
            self._port_section.pack(fill=tk.X, pady=(4, 0))
        else:
            self._port_section.pack_forget()

    def _current_mode_idx(self) -> int:
        try:
            return _MODE_ITEMS.index(self._mode_var.get())
        except ValueError:
            return 0

    # ── GUI update (main thread only) ─────────────────────────────────────────

    def _update_labels(self, r: TwinGateResult) -> None:
        self._tempo_var.set(f'{r.tempo_bpm:.2f} BPM')

        if r.gcd_tempo is not None:
            if r.gate_accepted:
                self._gate_lbl.config(text='✓ ACCEPT', fg=ACCENT_GREEN)
            else:
                reason = r.reject_reason or '?'
                self._gate_lbl.config(text=f'✗ REJECT [{reason}]', fg=ACCENT_RED)
            self._gcd_tempo_lbl.config(text=f'{r.gcd_tempo:.2f} BPM')
            self._gcd_conf_lbl.config(text=f'conf: {r.gcd_confidence:.2f}')
        else:
            self._gate_lbl.config(text='(no GCD)', fg=FG_DIM)
            self._gcd_tempo_lbl.config(text='---')
            self._gcd_conf_lbl.config(text='---')

        var = r.kalman_variance
        var_color = FG if var < 2.0 else (COLOR_YELLOW if var < 10.0 else ACCENT_RED)
        self._kalman_var_lbl.config(text=f'var:   {var:.2f}', fg=var_color)

        pred_str = f'{r.predicted_tempo:.2f}' if r.predicted_tempo is not None else '---'
        self._kalman_pred_lbl.config(text=f'pred:  {pred_str}')

        if r.innovation is not None:
            sign = '+' if r.innovation >= 0 else ''
            self._kalman_innov_lbl.config(text=f'innov: {sign}{r.innovation:.2f}')
        else:
            self._kalman_innov_lbl.config(text='innov: ---')

        self._counts_lbl.config(
            text=(f'K:{r.kick_count:3d} S:{r.snare_count:3d}\n'
                  f'H:{r.hihat_count:3d} O:{r.others_count:3d}')
        )

        col = _CAT_COLOR.get(r.category, FG_DIM)
        self._last_hit_lbl.config(text=r.category.name, fg=col)

        if r.phase is not None:
            self._phase_lbl.config(text=f'phase: {r.phase:.3f}')
            phase_err = r.phase_error if r.phase_error is not None else 0.0
            self._phase_err_lbl.config(text=f'err:  {phase_err:+.3f}')
            bar_color = ACCENT_GREEN if r.is_phase_synced else COLOR_YELLOW
            w = self._phase_canvas.winfo_width() or 160
            self._phase_canvas.delete('all')
            self._phase_canvas.create_rectangle(
                0, 0, int(w * r.phase), 10, fill=bar_color, outline='')
            self._next_beat_lbl.config(text=f'beat#: {r.beat_count}')
            if r.beat_count > self._prev_beat_count:
                self._beat_canvas.itemconfig(self._beat_oval, fill=ACCENT_GREEN)
                self.root.after(120, lambda: self._beat_canvas.itemconfig(
                    self._beat_oval, fill=FG_DIM))
            self._prev_beat_count = r.beat_count
        else:
            self._phase_lbl.config(text='phase: ---')
            self._phase_err_lbl.config(text='err:   ---')

    def _update_graph(self, r: TwinGateResult) -> None:
        self._history.append(r)
        xs = [h.event_count for h in self._history]

        xs_g = [h.event_count for h in self._history if h.gcd_tempo is not None]
        ys_g = [h.gcd_tempo   for h in self._history if h.gcd_tempo is not None]
        self._line_gcd.set_xdata(xs_g)
        self._line_gcd.set_ydata(ys_g)

        self._line_kalman.set_xdata(xs)
        self._line_kalman.set_ydata([h.tempo_bpm for h in self._history])

        if self._ci_fill is not None:
            self._ci_fill.remove()
            self._ci_fill = None
        ys_lo = [h.tempo_bpm - 2 * np.sqrt(max(0, h.kalman_variance)) for h in self._history]
        ys_hi = [h.tempo_bpm + 2 * np.sqrt(max(0, h.kalman_variance)) for h in self._history]
        self._ci_fill = self.ax.fill_between(xs, ys_lo, ys_hi, alpha=0.12, color=ACCENT_BLUE)

        xs_rej = [h.event_count for h in self._history
                  if not h.gate_accepted and h.gcd_tempo is not None
                  and h.reject_reason != 'no_gcd']
        ys_rej = [h.gcd_tempo for h in self._history
                  if not h.gate_accepted and h.gcd_tempo is not None
                  and h.reject_reason != 'no_gcd']
        self._line_reject.set_xdata(xs_rej)
        self._line_reject.set_ydata(ys_rej)

        self.ax.set_ylim(70, 150)
        if xs:
            self.ax.set_xlim(xs[0], max(xs[-1], xs[0] + 1))

        ys_conf  = [h.gcd_confidence for h in self._history]
        ys_var   = [h.kalman_variance for h in self._history]
        ys_mahal = [h.mahal_distance if h.mahal_distance is not None else 0
                    for h in self._history]
        thr = r.mahal_threshold

        self._line_conf.set_xdata(xs);  self._line_conf.set_ydata(ys_conf)
        self._line_var.set_xdata(xs);   self._line_var.set_ydata(ys_var)
        self._line_mahal.set_xdata(xs); self._line_mahal.set_ydata(ys_mahal)
        self._line_thr.set_xdata(xs);   self._line_thr.set_ydata([thr] * len(xs))

        if xs:
            self.ax_debug.set_xlim(xs[0], max(xs[-1], xs[0] + 1))
            self.ax_debug2.set_xlim(xs[0], max(xs[-1], xs[0] + 1))

        combined = [v for v in ys_var + ys_mahal if v is not None] + [thr]
        if combined:
            top = max(combined) * 1.15
            self.ax_debug2.set_ylim(0, max(top, 0.1))

        self._update_gcd_hist(r)

        xs_p = [h.event_count for h in self._history if h.phase is not None]
        ys_p = [h.phase       for h in self._history if h.phase is not None]
        self._line_phase.set_xdata(xs_p)
        self._line_phase.set_ydata(ys_p)
        if xs_p:
            self.ax_phase.set_xlim(xs_p[0], max(xs_p[-1], xs_p[0] + 1))

        self.canvas.draw_idle()

    def _update_gcd_hist(self, r: TwinGateResult) -> None:
        if r.gcd_buffer:
            self._gcd_buf_history.append((r.event_count, list(r.gcd_buffer)))

        ax = self.ax_gcd_hist
        ax.cla()
        ax.set_facecolor(BG_GRAPH)
        ax.tick_params(colors=FG, labelcolor=FG, which='both', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.set_xlabel('Event #', color=FG, fontsize=8)
        ax.set_ylabel('IOI (s)', color=FG, fontsize=8)
        ax.grid(True, color=BG_WIDGET, linestyle='--', linewidth=0.5)

        if not self._gcd_buf_history:
            return

        max_total = 0.0
        for ev_idx, buf in self._gcd_buf_history:
            if len(buf) < 2:
                continue
            bottom = 0.0
            for i in range(1, len(buf)):
                ioi   = buf[i][0] - buf[i - 1][0]
                cat   = buf[i][1]
                color = _CAT_COLOR.get(cat, FG_DIM)
                ax.bar(ev_idx, ioi, bottom=bottom, color=color, width=0.8, alpha=0.75)
                bottom += ioi
            max_total = max(max_total, bottom)

        if r.gcd_period is not None and 0 < r.gcd_period <= 2.0:
            ax.axhline(y=r.gcd_period, color=ACCENT_BLUE, linestyle='--',
                       linewidth=1.0, alpha=0.85, label=f'beat {r.gcd_period:.3f}s')
            ax.legend(loc='upper left', fontsize=7, framealpha=0.3,
                      facecolor=BG_WIDGET, edgecolor=BORDER, labelcolor=FG)

        if max_total > 0:
            ax.set_ylim(0, max_total * 1.15)

        ev_indices = [ev for ev, _ in self._gcd_buf_history]
        ax.set_xlim(min(ev_indices) - 0.5, max(ev_indices) + 0.5)

        _, last_buf = self._gcd_buf_history[-1]
        if len(last_buf) >= 2:
            n = len(last_buf) - 1
            for i in range(1, len(last_buf)):
                ioi   = last_buf[i][0] - last_buf[i - 1][0]
                cat   = last_buf[i][1]
                color = _CAT_COLOR.get(cat, FG_DIM)
                x_pos = (i - 0.5) / n
                ax.text(x_pos, 0.98, f'{ioi:.3f}\n{cat.name[0]}',
                        transform=ax.transAxes,
                        color=color, fontsize=6.5, ha='center', va='top',
                        fontfamily='monospace')

    # ── Queue polling (root.after loop) ──────────────────────────────────────

    def _handle_result(self, r: TwinGateResult) -> None:
        self._total_events += 1
        if r.gate_accepted:
            self._accept_count += 1
        elif r.reject_reason == 'mahal':
            self._reject_mahal += 1
        elif r.reject_reason == 'octave':
            self._reject_octave += 1
        elif r.reject_reason == 'confidence':
            self._reject_conf += 1
        self._update_labels(r)
        self._update_graph(r)

    def _poll_results(self) -> None:
        processed = 0
        try:
            while processed < 10:
                r = self.result_queue.get_nowait()
                self._handle_result(r)
                processed += 1
        except queue.Empty:
            pass
        self.root.after(50, self._poll_results)  # reschedule

    # ── Control ───────────────────────────────────────────────────────────────

    def _start(self) -> None:
        if self.is_running:
            return
        idx = self._current_mode_idx()
        if idx == _MIDI_INDEX:
            self._start_midi()
        else:
            self._start_mock(idx)

    def _start_midi(self) -> None:
        port_index = self._port_combo.current()
        if port_index < 0:
            print("[midi] no port selected", flush=True)
            return
        try:
            self._midi_listener = _TwinGateMidiListener(
                self.twin_gate,
                port_index=port_index,
                on_result=self.result_queue.put,
            )
            self._midi_listener.start()
            self.is_running = True
            self._run_status_lbl.config(text='▶  RUNNING', fg=ACCENT_GREEN)
        except RuntimeError as e:
            print(f"[midi] ERROR: {e}", flush=True)

    def _start_mock(self, mode_idx: int) -> None:
        self.is_running = True
        self._run_status_lbl.config(text='▶  RUNNING', fg=ACCENT_GREEN)
        threading.Thread(target=self._mock_thread, args=(mode_idx,), daemon=True).start()

    def _stop(self) -> None:
        self.is_running = False
        if self._midi_listener is not None:
            self._midi_listener.stop()
            self._midi_listener = None
        self._run_status_lbl.config(text='■  STOPPED', fg=FG_DIM)

    def _stop_and_summarize(self) -> None:
        self._stop()
        print("\n=== TwinGate Summary ===", flush=True)
        print(f"Total events    : {self._total_events}", flush=True)
        gcd_avail = (self._accept_count + self._reject_mahal
                     + self._reject_octave + self._reject_conf)
        rate = self._accept_count / max(1, gcd_avail) * 100
        print(f"ACCEPT rate     : {rate:.1f}%", flush=True)
        print(f"REJECT (mahal)  : {self._reject_mahal}", flush=True)
        print(f"REJECT (octave) : {self._reject_octave}", flush=True)
        print(f"REJECT (conf)   : {self._reject_conf}", flush=True)
        if self._history:
            last = self._history[-1]
            print(f"Final tempo     : {last.tempo_bpm:.2f} BPM", flush=True)
            print(f"Kalman variance : {last.kalman_variance:.2f}", flush=True)

    def _reset(self) -> None:
        self._stop()
        with self.lock:
            self.twin_gate.reset()
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

        self._history.clear()
        self._gcd_buf_history.clear()
        self._total_events  = 0
        self._accept_count  = 0
        self._reject_mahal  = 0
        self._reject_octave = 0
        self._reject_conf   = 0
        self._prev_beat_count = 0

        # Reset labels
        self._tempo_var.set('---')
        self._gate_lbl.config(text='---', fg=FG_DIM)
        self._gcd_tempo_lbl.config(text='---')
        self._gcd_conf_lbl.config(text='---')
        self._kalman_var_lbl.config(text='var:   ---', fg=FG)
        self._kalman_pred_lbl.config(text='pred:  ---')
        self._kalman_innov_lbl.config(text='innov: ---')
        self._counts_lbl.config(text='K:  0 S:  0\nH:  0 O:  0')
        self._last_hit_lbl.config(text='---', fg=FG_DIM)
        self._phase_lbl.config(text='phase: ---')
        self._phase_err_lbl.config(text='err:   ---')
        self._next_beat_lbl.config(text='beat#: ---')
        self._phase_canvas.delete('all')
        self._beat_canvas.itemconfig(self._beat_oval, fill=FG_DIM)
        self._prev_beat_count = 0

        # Reset graph
        for line in (self._line_gcd, self._line_kalman, self._line_reject,
                     self._line_conf, self._line_var, self._line_mahal, self._line_thr,
                     self._line_phase):
            line.set_xdata([])
            line.set_ydata([])

        if self._ci_fill is not None:
            self._ci_fill.remove()
            self._ci_fill = None

        self.ax.set_ylim(70, 150)

        self.ax_gcd_hist.cla()
        self.ax_gcd_hist.set_facecolor(BG_GRAPH)
        for spine in self.ax_gcd_hist.spines.values():
            spine.set_edgecolor(BORDER)

        self.canvas.draw_idle()

    def _auto_start(self) -> None:
        if self.args.mode == 'mock' or self.args.autostart:
            self._start()

    def _on_close(self) -> None:
        self._stop()
        self.root.destroy()

    # ── Mock worker thread ────────────────────────────────────────────────────

    def _mock_thread(self, mode_idx: int) -> None:
        tempo    = self.args.tempo
        humanize = self.args.humanize

        if mode_idx == 0:
            timestamps = generate_mock_events(tempo, n_beats=256, humanize_ms=humanize)
            event_list = [(ts, 36, 9) for ts in timestamps]
            print(f"[mock] beat: {len(event_list)} events at {tempo} BPM", flush=True)
        elif mode_idx == 1:
            raw = generate_drum_pattern_events(tempo, n_bars=32, pattern='basic_rock',
                                               humanize_ms=humanize)
            event_list = list(raw)
            print(f"[mock] basic_rock: {len(event_list)} events at {tempo} BPM", flush=True)
        else:
            raw = generate_drum_pattern_events(tempo, n_bars=32, pattern='hihat_16th',
                                               humanize_ms=humanize)
            event_list = list(raw)
            print(f"[mock] hihat_16th: {len(event_list)} events at {tempo} BPM", flush=True)

        prev_ts   = None
        prev_wall = time.perf_counter()

        for ts, note, channel in event_list:
            if not self.is_running:
                print("[mock] stopped.", flush=True)
                break

            if prev_ts is not None:
                ideal_dt  = ts - prev_ts
                actual_dt = time.perf_counter() - prev_wall
                sleep_sec = ideal_dt - actual_dt
                if sleep_sec > 0.001:
                    time.sleep(sleep_sec)
            prev_wall = time.perf_counter()
            prev_ts   = ts

            with self.lock:
                result = self.twin_gate.update(ts, note_number=note, channel=channel)
            if result is not None:
                self.result_queue.put(result)

        self.is_running = False
        print("[mock] done.", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='MIDI Tempo Estimator — TwinGate GUI (Tkinter)')
    p.add_argument('--tempo',    type=float, default=120.0)
    p.add_argument('--humanize', type=float, default=10.0)
    p.add_argument('--mode',     type=str,   default='midi',
                   choices=['midi', 'mock'])
    p.add_argument('--drum-pattern', type=str, default='none',
                   choices=['none', 'basic_rock', 'hihat_16th'])
    p.add_argument('--autostart', action='store_true')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    TempoEstimatorApp(root, args)
    root.mainloop()


if __name__ == '__main__':
    main()
