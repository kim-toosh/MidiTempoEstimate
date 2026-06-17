"""GUI front-end for the MIDI tempo estimator — TwinGate engine (PyQt6 + matplotlib).

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
Main  : QTimer が 50ms ごとに _poll_results() でキューをドレイン
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSplitter, QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

import midi_tempo_hmm.config as config
from midi_tempo_hmm.core.instrument_category import InstrumentCategory
from midi_tempo_hmm.core.twin_gate import TwinGate
from midi_tempo_hmm.interface.mock_input import (
    generate_drum_pattern_events,
    generate_mock_events,
)
from midi_tempo_hmm.output.twin_gate_result import TwinGateResult

# ── Colours ───────────────────────────────────────────────────────────────────
BG           = '#1E1E2E'
BG_GRAPH     = '#181825'
BG_WIDGET    = '#313244'
BG_HOVER     = '#45475A'
FG           = '#CDD6F4'
FG_DIM       = '#6C7086'
BORDER       = '#45475A'
ACCENT_RED   = '#E74C3C'
ACCENT_BLUE  = '#3498DB'
ACCENT_GREEN = '#2ECC71'
ACCENT_ORANGE = '#E67E22'
COLOR_BROWN  = '#8B4513'
COLOR_PURPLE = '#9B59B6'
COLOR_YELLOW = '#F39C12'

_QSS = f"""
QMainWindow, QWidget            {{ background: {BG}; color: {FG}; }}
QLabel                          {{ background: {BG}; color: {FG}; }}
QComboBox {{
    background: {BG_WIDGET}; color: {FG};
    border: 1px solid {BORDER}; border-radius: 4px;
    padding: 4px 8px; font-size: 12px;
}}
QComboBox QAbstractItemView     {{ background: {BG_WIDGET}; color: {FG};
                                   selection-background-color: {BG_HOVER}; }}
QPushButton {{
    background: {BG_WIDGET}; color: {FG};
    border: none; border-radius: 4px;
    padding: 8px 6px; font-size: 13px;
}}
QPushButton:hover   {{ background: {BG_HOVER}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QSplitter::handle   {{ background: {BORDER}; width: 2px; }}
"""

_CAT_COLOR = {
    InstrumentCategory.KICK:   ACCENT_RED,
    InstrumentCategory.SNARE:  ACCENT_BLUE,
    InstrumentCategory.HIHAT:  ACCENT_GREEN,
    InstrumentCategory.OTHERS: FG_DIM,
}

_MODE_ITEMS = ['Mock (Beat)', 'Mock (Rock)', 'Mock (16th)', 'MIDI Input']
_MIDI_INDEX = 3


# ── MIDI listener (thin rtmidi wrapper, TwinGate edition) ─────────────────────

class _TwinGateMidiListener:
    _NOTE_ON    = 0x90
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

class TempoEstimatorApp(QMainWindow):
    """TwinGate tempo estimator GUI (PyQt6 + matplotlib)."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args

        self.setWindowTitle("MIDI Tempo Estimator - TwinGate")
        self.resize(1000, 560)
        self.setStyleSheet(_QSS)

        # ── Engine & state ────────────────────────────────────────────────────
        self.twin_gate    = TwinGate(config)
        self.is_running   = False
        self.lock         = threading.Lock()
        self.result_queue: queue.Queue = queue.Queue()

        self._history: deque[TwinGateResult] = deque(maxlen=64)
        self._gcd_buf_history: deque[tuple[int, list]] = deque(maxlen=32)
        self._midi_listener: Optional[_TwinGateMidiListener] = None

        # STOP summary counters
        self._total_events  = 0
        self._accept_count  = 0
        self._reject_mahal  = 0
        self._reject_octave = 0
        self._reject_conf   = 0

        self._build_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_results)
        self._timer.start(50)

        QTimer.singleShot(0, self._auto_start)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        self.setCentralWidget(splitter)

        left  = QWidget()
        right = QWidget()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([230, 770])

        self._build_left(left)
        self._build_right(right)

    def _build_left(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(0)

        font_dim    = QFont('Courier New', 9)
        font_tempo  = QFont('Courier New', 28); font_tempo.setBold(True)
        font_medium = QFont('Courier New', 13); font_medium.setBold(True)
        font_small  = QFont('Courier New', 11)
        font_combo  = QFont('Helvetica', 11)

        def _hdr(text: str, margin_top: int = 8) -> None:
            lbl = QLabel(text)
            lbl.setFont(font_dim)
            lbl.setStyleSheet(f'color: {FG_DIM}; margin-top: {margin_top}px;')
            layout.addWidget(lbl)

        # ── TEMPO ─────────────────────────────────────────────────────────────
        _hdr('TEMPO', margin_top=0)
        self._tempo_lbl = QLabel('---')
        self._tempo_lbl.setFont(font_tempo)
        layout.addWidget(self._tempo_lbl)

        # ── GATE ──────────────────────────────────────────────────────────────
        _hdr('GATE')
        self._gate_lbl = QLabel('---')
        self._gate_lbl.setFont(font_medium)
        self._gate_lbl.setStyleSheet(f'color: {FG_DIM};')
        layout.addWidget(self._gate_lbl)

        # ── GCD TEMPO ─────────────────────────────────────────────────────────
        _hdr('GCD TEMPO')
        self._gcd_tempo_lbl = QLabel('---')
        self._gcd_tempo_lbl.setFont(font_medium)
        self._gcd_tempo_lbl.setStyleSheet(f'color: {ACCENT_BLUE};')
        layout.addWidget(self._gcd_tempo_lbl)
        self._gcd_conf_lbl = QLabel('---')
        self._gcd_conf_lbl.setFont(font_small)
        self._gcd_conf_lbl.setStyleSheet(f'color: {FG_DIM};')
        layout.addWidget(self._gcd_conf_lbl)

        # ── KALMAN ────────────────────────────────────────────────────────────
        _hdr('KALMAN')
        self._kalman_var_lbl   = QLabel('var:   ---')
        self._kalman_pred_lbl  = QLabel('pred:  ---')
        self._kalman_innov_lbl = QLabel('innov: ---')
        for lbl in (self._kalman_var_lbl, self._kalman_pred_lbl, self._kalman_innov_lbl):
            lbl.setFont(font_small)
            layout.addWidget(lbl)

        # ── EVENT COUNTS ──────────────────────────────────────────────────────
        _hdr('EVENT COUNTS')
        self._counts_lbl = QLabel('K:  0 S:  0\nH:  0 O:  0')
        self._counts_lbl.setFont(font_small)
        self._counts_lbl.setStyleSheet(f'color: {FG_DIM};')
        layout.addWidget(self._counts_lbl)

        # ── LAST HIT ──────────────────────────────────────────────────────────
        _hdr('LAST HIT')
        self._last_hit_lbl = QLabel('---')
        self._last_hit_lbl.setFont(font_medium)
        self._last_hit_lbl.setStyleSheet(f'color: {FG_DIM};')
        layout.addWidget(self._last_hit_lbl)

        layout.addStretch(1)

        # ── Status ────────────────────────────────────────────────────────────
        self._run_status_lbl = QLabel('■  STOPPED')
        self._run_status_lbl.setFont(QFont('Courier New', 12))
        self._run_status_lbl.setStyleSheet(f'color: {FG_DIM};')
        layout.addWidget(self._run_status_lbl)
        layout.addSpacing(4)

        # ── Buttons ───────────────────────────────────────────────────────────
        for text, slot in [
            ('▶  START', self._start),
            ('■  STOP',  self._stop_and_summarize),
            ('↺  RESET', self._reset),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            layout.addWidget(btn)
            layout.addSpacing(4)

        # ── INPUT MODE ────────────────────────────────────────────────────────
        mode_hdr = QLabel('INPUT MODE')
        mode_hdr.setFont(font_dim)
        mode_hdr.setStyleSheet(f'color: {FG_DIM}; margin-top: 10px;')
        layout.addWidget(mode_hdr)

        self._mode_combo = QComboBox()
        self._mode_combo.setFont(font_combo)
        self._mode_combo.addItems(_MODE_ITEMS)
        # Select from CLI args
        if self.args.mode == 'mock':
            if self.args.drum_pattern == 'basic_rock':
                self._mode_combo.setCurrentIndex(1)
            elif self.args.drum_pattern == 'hihat_16th':
                self._mode_combo.setCurrentIndex(2)
            else:
                self._mode_combo.setCurrentIndex(0)
        else:
            self._mode_combo.setCurrentIndex(_MIDI_INDEX)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        layout.addWidget(self._mode_combo)
        layout.addSpacing(6)

        # ── MIDI PORT ─────────────────────────────────────────────────────────
        self._port_section = QWidget()
        port_layout = QVBoxLayout(self._port_section)
        port_layout.setContentsMargins(0, 0, 0, 0)
        port_layout.setSpacing(4)

        port_hdr = QLabel('MIDI PORT')
        port_hdr.setFont(font_dim)
        port_hdr.setStyleSheet(f'color: {FG_DIM};')
        port_layout.addWidget(port_hdr)

        port_row = QWidget()
        port_row_layout = QHBoxLayout(port_row)
        port_row_layout.setContentsMargins(0, 0, 0, 0)
        port_row_layout.setSpacing(4)

        self._port_combo = QComboBox()
        self._port_combo.setFont(font_combo)
        port_row_layout.addWidget(self._port_combo, stretch=1)

        refresh_btn = QPushButton('⟳')
        refresh_btn.setFixedWidth(36)
        refresh_btn.clicked.connect(self._refresh_ports)
        port_row_layout.addWidget(refresh_btn)

        port_layout.addWidget(port_row)
        layout.addWidget(self._port_section)
        layout.addSpacing(10)

        self._refresh_ports()
        self._on_mode_changed()

    def _build_right(self, parent: QWidget) -> None:
        self.fig = Figure(facecolor=BG, tight_layout=False)
        self.fig.subplots_adjust(left=0.08, right=0.88, top=0.95, bottom=0.07, hspace=0.75)
        gs = self.fig.add_gridspec(3, 1, height_ratios=[3, 1, 2])

        # ── 上段: テンポ推移グラフ ─────────────────────────────────────────
        self.ax = self.fig.add_subplot(gs[0])
        self._style_ax(self.ax, ylabel='Tempo (BPM)')
        self.ax.set_ylim(70, 150)

        self._line_gcd,    = self.ax.plot([], [], color=ACCENT_BLUE, linewidth=1.5,
                                           linestyle='--', alpha=0.85, label='GCD tempo', zorder=2)
        self._line_kalman, = self.ax.plot([], [], color=ACCENT_RED,  linewidth=2.5,
                                           label='Kalman tempo', zorder=3)
        self._line_reject, = self.ax.plot([], [], color=ACCENT_ORANGE, linestyle='none',
                                           marker='^', markersize=9,
                                           label='REJECT', zorder=4)
        self._ci_fill = None
        self.ax.legend(loc='upper left', fontsize=8, framealpha=0.3,
                       facecolor=BG_WIDGET, edgecolor=BORDER, labelcolor=FG)

        # Data accumulators for upper graph
        self._xs_gcd:    list[int]   = []
        self._ys_gcd:    list[float] = []
        self._xs_kalman: list[int]   = []
        self._ys_kalman: list[float] = []
        self._xs_ci:     list[int]   = []
        self._ys_ci_lo:  list[float] = []
        self._ys_ci_hi:  list[float] = []
        self._xs_rej:    list[int]   = []
        self._ys_rej:    list[float] = []

        # ── 下段: デバッググラフ ──────────────────────────────────────────────
        self.ax_debug  = self.fig.add_subplot(gs[1])
        self.ax_debug2 = self.ax_debug.twinx()
        self._style_debug_axes()

        self._line_conf,  = self.ax_debug.plot([], [], color=ACCENT_GREEN, linewidth=1.5,
                                                 label='GCD conf', zorder=3)
        self._line_var,   = self.ax_debug2.plot([], [], color=COLOR_BROWN,  linewidth=1.5,
                                                  label='K var', zorder=2)
        self._line_mahal, = self.ax_debug2.plot([], [], color=COLOR_PURPLE, linewidth=1.5,
                                                  label='Mahal', zorder=3)
        self._line_thr,   = self.ax_debug2.plot([], [], color=ACCENT_RED, linewidth=1.0,
                                                  linestyle='--', alpha=0.7,
                                                  label='threshold', zorder=2)

        # ── 3段: GCDバッファIOI履歴グラフ ──────────────────────────────────────
        self.ax_gcd_hist = self.fig.add_subplot(gs[2])
        self.ax_gcd_hist.set_facecolor(BG_GRAPH)
        self.ax_gcd_hist.tick_params(colors=FG, labelcolor=FG, which='both', labelsize=7)
        for spine in self.ax_gcd_hist.spines.values():
            spine.set_edgecolor(BORDER)
        self.ax_gcd_hist.set_xlabel('Event #', color=FG, fontsize=8)
        self.ax_gcd_hist.set_ylabel('IOI (s)', color=FG, fontsize=8)
        self.ax_gcd_hist.grid(True, color=BG_WIDGET, linestyle='--', linewidth=0.5)

        self.canvas = FigureCanvasQTAgg(self.fig)
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

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

    # ── Port / mode helpers ───────────────────────────────────────────────────

    def _refresh_ports(self) -> None:
        self._port_combo.clear()
        ports = _TwinGateMidiListener.list_ports()
        if ports:
            self._port_combo.addItems(ports)
        else:
            self._port_combo.addItem("(no ports found)")

    def _on_mode_changed(self) -> None:
        self._port_section.setVisible(self._mode_combo.currentIndex() == _MIDI_INDEX)

    def _current_mode_idx(self) -> int:
        return self._mode_combo.currentIndex()

    # ── GUI update (main thread only) ─────────────────────────────────────────

    def _update_labels(self, r: TwinGateResult) -> None:
        self._tempo_lbl.setText(f'{r.tempo_bpm:.2f} BPM')

        if r.gcd_tempo is not None:
            if r.gate_accepted:
                self._gate_lbl.setText('✓ ACCEPT')
                self._gate_lbl.setStyleSheet(f'color: {ACCENT_GREEN};')
            else:
                reason = r.reject_reason or '?'
                self._gate_lbl.setText(f'✗ REJECT [{reason}]')
                self._gate_lbl.setStyleSheet(f'color: {ACCENT_RED};')
            self._gcd_tempo_lbl.setText(f'{r.gcd_tempo:.2f} BPM')
            self._gcd_conf_lbl.setText(f'conf: {r.gcd_confidence:.2f}')
        else:
            self._gate_lbl.setText('(no GCD)')
            self._gate_lbl.setStyleSheet(f'color: {FG_DIM};')
            self._gcd_tempo_lbl.setText('---')
            self._gcd_conf_lbl.setText('---')

        var = r.kalman_variance
        if var < 2.0:
            var_color = FG
        elif var < 10.0:
            var_color = COLOR_YELLOW
        else:
            var_color = ACCENT_RED
        self._kalman_var_lbl.setText(f'var:   {var:.2f}')
        self._kalman_var_lbl.setStyleSheet(f'color: {var_color};')

        pred_str  = f'{r.predicted_tempo:.2f}' if r.predicted_tempo is not None else '---'
        self._kalman_pred_lbl.setText(f'pred:  {pred_str}')

        if r.innovation is not None:
            sign = '+' if r.innovation >= 0 else ''
            self._kalman_innov_lbl.setText(f'innov: {sign}{r.innovation:.2f}')
        else:
            self._kalman_innov_lbl.setText('innov: ---')

        self._counts_lbl.setText(
            f'K:{r.kick_count:3d} S:{r.snare_count:3d}\n'
            f'H:{r.hihat_count:3d} O:{r.others_count:3d}'
        )

        col = _CAT_COLOR.get(r.category, FG_DIM)
        self._last_hit_lbl.setText(r.category.name)
        self._last_hit_lbl.setStyleSheet(f'color: {col};')

    def _update_graph(self, r: TwinGateResult) -> None:
        self._history.append(r)
        xs = [h.event_count for h in self._history]

        # GCD tempo line (skip entries with no GCD)
        xs_g = [h.event_count for h in self._history if h.gcd_tempo is not None]
        ys_g = [h.gcd_tempo   for h in self._history if h.gcd_tempo is not None]
        self._line_gcd.set_xdata(xs_g)
        self._line_gcd.set_ydata(ys_g)

        # Kalman tempo line
        self._line_kalman.set_xdata(xs)
        self._line_kalman.set_ydata([h.tempo_bpm for h in self._history])

        # CI fill: mean ± 2σ
        if self._ci_fill is not None:
            self._ci_fill.remove()
            self._ci_fill = None
        ys_lo = [h.tempo_bpm - 2 * np.sqrt(max(0, h.kalman_variance)) for h in self._history]
        ys_hi = [h.tempo_bpm + 2 * np.sqrt(max(0, h.kalman_variance)) for h in self._history]
        self._ci_fill = self.ax.fill_between(
            xs, ys_lo, ys_hi, alpha=0.12, color=ACCENT_BLUE,
        )

        # REJECT markers (orange triangle at gcd_tempo position, excluding no_gcd)
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

        # Debug graph
        ys_conf  = [h.gcd_confidence for h in self._history]
        ys_var   = [h.kalman_variance for h in self._history]
        ys_mahal = [h.mahal_distance if h.mahal_distance is not None else 0
                    for h in self._history]
        thr = r.mahal_threshold

        self._line_conf.set_xdata(xs)
        self._line_conf.set_ydata(ys_conf)
        self._line_var.set_xdata(xs)
        self._line_var.set_ydata(ys_var)
        self._line_mahal.set_xdata(xs)
        self._line_mahal.set_ydata(ys_mahal)
        self._line_thr.set_xdata(xs)
        self._line_thr.set_ydata([thr] * len(xs))

        if xs:
            self.ax_debug.set_xlim(xs[0], max(xs[-1], xs[0] + 1))
            self.ax_debug2.set_xlim(xs[0], max(xs[-1], xs[0] + 1))

        # Auto-scale right y-axis
        combined = [v for v in ys_var + ys_mahal if v is not None] + [thr]
        if combined:
            top = max(combined) * 1.15
            self.ax_debug2.set_ylim(0, max(top, 0.1))

        self._update_gcd_hist(r)

        self.canvas.draw_idle()

    def _update_gcd_hist(self, r: TwinGateResult) -> None:
        """GCDバッファの内容をIOI積み上げ棒グラフで表示する。

        各バー = そのイベント時点のgcd_timestampsバッファのスナップショット。
        各セグメント = 連続タイムスタンプ間のIOI、カテゴリ色で色分け。
        """
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
                ioi = buf[i][0] - buf[i - 1][0]
                cat = buf[i][1]
                color = _CAT_COLOR.get(cat, FG_DIM)
                ax.bar(ev_idx, ioi, bottom=bottom, color=color, width=0.8, alpha=0.75)
                bottom += ioi
            max_total = max(max_total, bottom)

        # Reference line: current beat period (60 / gcd_tempo)
        if r.gcd_period is not None and 0 < r.gcd_period <= 2.0:
            ax.axhline(y=r.gcd_period, color=ACCENT_BLUE, linestyle='--',
                       linewidth=1.0, alpha=0.85, label=f'beat {r.gcd_period:.3f}s')
            ax.legend(loc='upper left', fontsize=7, framealpha=0.3,
                      facecolor=BG_WIDGET, edgecolor=BORDER, labelcolor=FG)

        if max_total > 0:
            ax.set_ylim(0, max_total * 1.15)

        ev_indices = [ev for ev, _ in self._gcd_buf_history]
        ax.set_xlim(min(ev_indices) - 0.5, max(ev_indices) + 0.5)

        # 最新バッファのIOI値をグラフ上部に小さなテキストで並べて表示
        _, last_buf = self._gcd_buf_history[-1]
        if len(last_buf) >= 2:
            n = len(last_buf) - 1
            for i in range(1, len(last_buf)):
                ioi = last_buf[i][0] - last_buf[i - 1][0]
                cat = last_buf[i][1]
                color = _CAT_COLOR.get(cat, FG_DIM)
                x_pos = (i - 0.5) / n
                ax.text(x_pos, 0.98, f'{ioi:.3f}\n{cat.name[0]}',
                        transform=ax.transAxes,
                        color=color, fontsize=6.5, ha='center', va='top',
                        fontfamily='monospace')

    # ── Queue polling ─────────────────────────────────────────────────────────

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
        port_index = self._port_combo.currentIndex()
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
            self._run_status_lbl.setText('▶  RUNNING')
            self._run_status_lbl.setStyleSheet(f'color: {ACCENT_GREEN};')
        except RuntimeError as e:
            print(f"[midi] ERROR: {e}", flush=True)

    def _start_mock(self, mode_idx: int) -> None:
        self.is_running = True
        self._run_status_lbl.setText('▶  RUNNING')
        self._run_status_lbl.setStyleSheet(f'color: {ACCENT_GREEN};')
        threading.Thread(target=self._mock_thread, args=(mode_idx,), daemon=True).start()

    def _stop(self) -> None:
        self.is_running = False
        if self._midi_listener is not None:
            self._midi_listener.stop()
            self._midi_listener = None
        self._run_status_lbl.setText('■  STOPPED')
        self._run_status_lbl.setStyleSheet(f'color: {FG_DIM};')

    def _stop_and_summarize(self) -> None:
        self._stop()
        print("\n=== TwinGate Summary ===", flush=True)
        print(f"Total events    : {self._total_events}", flush=True)
        gcd_avail = self._accept_count + self._reject_mahal + self._reject_octave + self._reject_conf
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

        # Reset left panel
        self._tempo_lbl.setText('---')
        self._gate_lbl.setText('---')
        self._gate_lbl.setStyleSheet(f'color: {FG_DIM};')
        self._gcd_tempo_lbl.setText('---')
        self._gcd_conf_lbl.setText('---')
        self._kalman_var_lbl.setText('var:   ---')
        self._kalman_pred_lbl.setText('pred:  ---')
        self._kalman_innov_lbl.setText('innov: ---')
        self._counts_lbl.setText('K:  0 S:  0\nH:  0 O:  0')
        self._last_hit_lbl.setText('---')
        self._last_hit_lbl.setStyleSheet(f'color: {FG_DIM};')

        # Reset graph lines
        for line in (self._line_gcd, self._line_kalman, self._line_reject,
                     self._line_conf, self._line_var, self._line_mahal, self._line_thr):
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

    def closeEvent(self, event) -> None:
        self._stop()
        self._timer.stop()
        super().closeEvent(event)

    # ── Mock worker thread ────────────────────────────────────────────────────

    def _mock_thread(self, mode_idx: int) -> None:
        tempo = self.args.tempo
        humanize = self.args.humanize

        if mode_idx == 0:          # Mock (Beat)
            timestamps = generate_mock_events(tempo, n_beats=256, humanize_ms=humanize)
            event_list = [(ts, 36, 9) for ts in timestamps]
            print(f"[mock] beat: {len(event_list)} events at {tempo} BPM", flush=True)
        elif mode_idx == 1:        # Mock (Rock)
            raw = generate_drum_pattern_events(tempo, n_bars=32, pattern='basic_rock',
                                               humanize_ms=humanize)
            event_list = list(raw)
            print(f"[mock] basic_rock: {len(event_list)} events at {tempo} BPM", flush=True)
        else:                      # Mock (16th)
            raw = generate_drum_pattern_events(tempo, n_bars=32, pattern='hihat_16th',
                                               humanize_ms=humanize)
            event_list = list(raw)
            print(f"[mock] hihat_16th: {len(event_list)} events at {tempo} BPM", flush=True)

        prev_ts   = None
        prev_wall = time.perf_counter()

        for i, (ts, note, channel) in enumerate(event_list):
            if not self.is_running:
                print("[mock] stopped.", flush=True)
                break

            # Real-time replay: sleep to match timestamp differences
            if prev_ts is not None:
                ideal_dt    = ts - prev_ts
                actual_dt   = time.perf_counter() - prev_wall
                sleep_sec   = ideal_dt - actual_dt
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
    p = argparse.ArgumentParser(description='MIDI Tempo Estimator — TwinGate GUI (PyQt6)')
    p.add_argument('--tempo',    type=float, default=120.0,
                   help='Mock tempo in BPM (default: 120.0)')
    p.add_argument('--humanize', type=float, default=10.0,
                   help='Timing jitter std-dev in ms (default: 10.0)')
    p.add_argument('--mode',     type=str,   default='midi',
                   choices=['midi', 'mock'],
                   help='デフォルト入力モード: midi (default) / mock')
    p.add_argument('--drum-pattern', type=str, default='none',
                   choices=['none', 'basic_rock', 'hihat_16th'],
                   help='Drum pattern for mock mode (default: none = beat events)')
    p.add_argument('--autostart', action='store_true',
                   help='起動時に自動でSTARTする')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = TempoEstimatorApp(args)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
