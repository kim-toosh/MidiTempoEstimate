"""GUI front-end for the MIDI tempo estimator (PyQt6 + matplotlib).

Modes
-----
MIDI  : リアルタイム MIDI 入力（デフォルト）。START でポートを開き、
        NOTE_ON を受け取るたびに particle filter を更新する。
Mock  : テスト用。指定テンポの仮想イベントを自動生成する。

Threading model
---------------
MIDI mode  : rtmidi コールバック（MIDIスレッド）が result_queue に put
Mock mode  : worker thread が result_queue に put
Main thread: QTimer が 50ms ごとに _poll_results() でキューをドレイン
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections import deque

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
from midi_tempo_hmm.core.particle_filter import ParticleFilter
from midi_tempo_hmm.interface.midi_input import MidiInputHandler
from midi_tempo_hmm.interface.mock_input import (
    generate_drum_pattern_events,
    generate_mock_events,
)
from midi_tempo_hmm.output.estimator_result import EstimatorResult

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
    InstrumentCategory.KICK:   '#E74C3C',
    InstrumentCategory.SNARE:  '#3498DB',
    InstrumentCategory.HIHAT:  '#2ECC71',
    InstrumentCategory.OTHERS: '#6C7086',
}


def _section_hdr(layout: QVBoxLayout, title: str) -> None:
    lbl = QLabel(f'─ {title} ─')
    lbl.setFont(QFont('Helvetica', 8))
    lbl.setStyleSheet(f'color: {ACCENT_BLUE}; margin-top: 10px; margin-bottom: 2px;')
    layout.addWidget(lbl)


class TempoEstimatorApp(QMainWindow):
    """Main application window (PyQt6 + matplotlib)."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args

        self.setWindowTitle("MIDI Tempo Estimator")
        self.resize(960, 540)
        self.setStyleSheet(_QSS)

        # ── Filter state ──────────────────────────────────────────────────────
        self.particle_filter = ParticleFilter(config)
        self.tempo_hist:  deque[float] = deque(maxlen=32)
        self.event_count: int  = 0
        self.is_running:  bool = False
        self.lock = threading.Lock()
        self.result_queue: queue.Queue = queue.Queue()

        # Kalman gate history (max 256 results for visualisation)
        self.result_history: deque[EstimatorResult] = deque(maxlen=256)
        self._pred_hist: deque[float] = deque(maxlen=32)
        self._var_hist:  deque[float] = deque(maxlen=32)
        self._rej_x: list[int]   = []
        self._rej_y: list[float] = []
        self._fill = None  # fill_between PolyCollection

        # Per-category event positions for graph markers
        self._cat_x: dict[InstrumentCategory, list[int]]   = {c: [] for c in InstrumentCategory}
        self._cat_y: dict[InstrumentCategory, list[float]] = {c: [] for c in InstrumentCategory}

        # IOI per note number (for the lower debug graph)
        self._ioi_by_note:      dict[int, float]               = {}
        self._category_by_note: dict[int, InstrumentCategory]  = {}
        self._kalman_gated_tempo: float | None = None

        # MIDI handler (used only in midi mode)
        self._midi_handler: MidiInputHandler | None = None

        self._build_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_results)
        self._timer.start(50)

        # 起動時に最初の MIDI ポートへ自動接続
        QTimer.singleShot(0, self._auto_start)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        self.setCentralWidget(splitter)

        left  = QWidget(); right = QWidget(); gcd_panel = QWidget()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.addWidget(gcd_panel)
        splitter.setSizes([260, 600, 240])

        self._build_left(left)
        self._build_right(right)
        self._build_gcd_panel(gcd_panel)

    def _build_left(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(0)

        font_dim       = QFont('Helvetica', 10)
        font_value     = QFont('Helvetica', 26); font_value.setBold(True)
        font_small_val = QFont('Helvetica', 16); font_small_val.setBold(True)
        font_small     = QFont('Helvetica', 9)
        font_combo     = QFont('Helvetica', 11)

        # ════════════════════════════════════════════════════════════════════
        # グループ 1: MidiInputGate
        # ════════════════════════════════════════════════════════════════════
        _section_hdr(layout, 'MidiInputGate')

        last_hdr = QLabel('LAST HIT')
        last_hdr.setFont(font_dim)
        last_hdr.setStyleSheet(f'color: {FG_DIM}; margin-top: 4px;')
        layout.addWidget(last_hdr)
        self._last_hit_lbl = QLabel('---')
        self._last_hit_lbl.setFont(font_small)
        self._last_hit_lbl.setStyleSheet(f'color: {FG_DIM};')
        layout.addWidget(self._last_hit_lbl)

        counts_hdr = QLabel('EVENT COUNTS')
        counts_hdr.setFont(font_dim)
        counts_hdr.setStyleSheet(f'color: {FG_DIM}; margin-top: 4px;')
        layout.addWidget(counts_hdr)
        self._counts_lbl = QLabel('K:0  S:0  H:0  O:0')
        self._counts_lbl.setFont(QFont('Helvetica', 10))
        self._counts_lbl.setStyleSheet(f'color: {FG_DIM};')
        layout.addWidget(self._counts_lbl)

        gcd_hdr = QLabel('GCD INTERVAL')
        gcd_hdr.setFont(font_dim)
        gcd_hdr.setStyleSheet(f'color: {FG_DIM}; margin-top: 4px;')
        layout.addWidget(gcd_hdr)
        self._gcd_lbl = QLabel('---')
        self._gcd_lbl.setFont(font_small_val)
        self._gcd_lbl.setStyleSheet(f'color: {FG_DIM};')
        layout.addWidget(self._gcd_lbl)

        # ════════════════════════════════════════════════════════════════════
        # グループ 2: Particle Filter
        # ════════════════════════════════════════════════════════════════════
        _section_hdr(layout, 'Particle Filter')

        self._val_labels: dict[str, QLabel] = {}
        for key, title in [
            ('tempo',  'Particle TEMPO'),
            ('beat',   'Particle BEAT'),
            ('m_beat', 'Particle MEASURE'),
            ('conf',   'CONFIDENCE'),
        ]:
            hdr = QLabel(title)
            hdr.setFont(font_dim)
            hdr.setStyleSheet(f'color: {FG_DIM}; margin-top: 8px;')
            layout.addWidget(hdr)

            val = QLabel('---')
            val.setFont(font_small_val if key in ('tempo', 'beat', 'm_beat') else font_value)
            layout.addWidget(val)
            self._val_labels[key] = val

        conv_hdr = QLabel('CONVERGED')
        conv_hdr.setFont(font_dim)
        conv_hdr.setStyleSheet(f'color: {FG_DIM}; margin-top: 8px;')
        layout.addWidget(conv_hdr)
        self._conv_lbl = QLabel('---')
        self._conv_lbl.setFont(font_value)
        layout.addWidget(self._conv_lbl)

        acorr_hdr = QLabel('AUTOCORR')
        acorr_hdr.setFont(font_dim)
        acorr_hdr.setStyleSheet(f'color: {FG_DIM}; margin-top: 4px;')
        layout.addWidget(acorr_hdr)
        self._acorr_lbl = QLabel('---')
        self._acorr_lbl.setFont(font_small)
        self._acorr_lbl.setStyleSheet(f'color: {FG_DIM};')
        layout.addWidget(self._acorr_lbl)

        self._proc_lbl = QLabel('')
        self._proc_lbl.setFont(font_small)
        self._proc_lbl.setStyleSheet(f'color: {FG_DIM}; margin-top: 4px;')
        layout.addWidget(self._proc_lbl)

        # ════════════════════════════════════════════════════════════════════
        # グループ 3: Kalman Filter
        # ════════════════════════════════════════════════════════════════════
        _section_hdr(layout, 'Kalman Filter')

        kalman_top_hdr = QLabel('KALMAN TEMPO')
        kalman_top_hdr.setFont(font_dim)
        kalman_top_hdr.setStyleSheet(f'color: {FG_DIM}; margin-top: 4px;')
        layout.addWidget(kalman_top_hdr)
        self._kalman_top_lbl = QLabel('---')
        self._kalman_top_lbl.setFont(font_value)
        self._kalman_top_lbl.setStyleSheet(f'color: {ACCENT_BLUE};')
        layout.addWidget(self._kalman_top_lbl)

        gate_hdr = QLabel('GATE STATUS')
        gate_hdr.setFont(font_dim)
        gate_hdr.setStyleSheet(f'color: {FG_DIM}; margin-top: 4px;')
        layout.addWidget(gate_hdr)
        self._gate_lbl = QLabel('---')
        self._gate_lbl.setFont(font_small)
        self._gate_lbl.setStyleSheet(f'color: {FG_DIM};')
        layout.addWidget(self._gate_lbl)

        var_hdr = QLabel('KALMAN VAR')
        var_hdr.setFont(font_dim)
        var_hdr.setStyleSheet(f'color: {FG_DIM}; margin-top: 4px;')
        layout.addWidget(var_hdr)
        self._kalman_var_lbl = QLabel('---')
        self._kalman_var_lbl.setFont(font_small)
        self._kalman_var_lbl.setStyleSheet(f'color: {FG_DIM};')
        layout.addWidget(self._kalman_var_lbl)

        layout.addStretch(1)

        # ── モード選択 ────────────────────────────────────────────────────────
        mode_hdr = QLabel('INPUT MODE')
        mode_hdr.setFont(font_dim)
        mode_hdr.setStyleSheet(f'color: {FG_DIM}; margin-top: 10px;')
        layout.addWidget(mode_hdr)

        self._mode_combo = QComboBox()
        self._mode_combo.setFont(font_combo)
        self._mode_combo.addItems(['MIDI Input', 'Mock (test)'])
        if self.args.mode == 'mock':
            self._mode_combo.setCurrentIndex(1)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        layout.addWidget(self._mode_combo)
        layout.addSpacing(6)

        # ── MIDIポート選択 ────────────────────────────────────────────────────
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

        # ── ボタン ────────────────────────────────────────────────────────────
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

    def _build_right(self, parent: QWidget) -> None:
        self.fig = Figure(facecolor=BG, tight_layout=False)
        self.fig.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.07, hspace=0.50)
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1])

        # ── 上 3/4: テンポ履歴グラフ ─────────────────────────────────────────
        self.ax = self.fig.add_subplot(gs[0])
        self.ax.set_facecolor(BG_GRAPH)
        self.ax.tick_params(colors=FG, labelcolor=FG, which='both')
        self.ax.xaxis.label.set_color(FG)
        self.ax.yaxis.label.set_color(FG)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(BORDER)
        self.ax.set_xlabel('Event #',    color=FG)
        self.ax.set_ylabel('Tempo (BPM)', color=FG)
        self.ax.set_ylim(config.TEMPO_MIN, config.TEMPO_MAX)
        self.ax.grid(True, color=BG_WIDGET, linestyle='--', linewidth=0.5)

        self.tempo_line, = self.ax.plot([], [], color=ACCENT_RED,  linewidth=2, label='tempo', zorder=3)
        self._pred_line, = self.ax.plot([], [], color=ACCENT_BLUE, linewidth=1,
                                        linestyle='--', alpha=0.7, label='Kalman pred', zorder=2)
        self._rej_line,  = self.ax.plot([], [], color='#FF6B6B', linestyle='none',
                                        marker='x', markersize=8, label='rejected', zorder=4)

        _CAT_STYLE = {
            InstrumentCategory.KICK:   ('#E74C3C', 'v', 8, 'kick'),
            InstrumentCategory.SNARE:  ('#3498DB', '^', 8, 'snare'),
            InstrumentCategory.HIHAT:  ('#2ECC71', '.', 6, 'hihat'),
            InstrumentCategory.OTHERS: ('#6C7086', 'D', 5, 'others'),
        }
        self._cat_lines: dict[InstrumentCategory, object] = {}
        for cat, (col, mk, ms, lbl) in _CAT_STYLE.items():
            line, = self.ax.plot([], [], linestyle='none', color=col,
                                 marker=mk, markersize=ms, alpha=0.6,
                                 label=lbl, zorder=5)
            self._cat_lines[cat] = line

        # ── 下 1/4: IOI per Note Number ──────────────────────────────────────
        self.ax_ioi = self.fig.add_subplot(gs[1])
        self._style_ioi_ax()

        self.canvas = FigureCanvasQTAgg(self.fig)
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def _style_ioi_ax(self) -> None:
        ax = self.ax_ioi
        ax.set_facecolor(BG_GRAPH)
        ax.tick_params(colors=FG, labelcolor=FG, which='both')
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.set_xlabel('IOI (s)', color=FG, fontsize=9)
        ax.set_ylabel('Note #',  color=FG, fontsize=9)
        ax.set_title('IOI per Note Number', color=FG, fontsize=9, pad=4)
        ax.set_xlim(0, 2.0)
        ax.grid(True, axis='x', color=BG_WIDGET, linestyle='--', linewidth=0.5)

    def _build_gcd_panel(self, parent: QWidget) -> None:
        self.fig_gcd = Figure(facecolor=BG, tight_layout=False)
        self.fig_gcd.subplots_adjust(left=0.18, right=0.95, top=0.90, bottom=0.10)
        self.ax_gcd = self.fig_gcd.add_subplot(111)
        self._style_gcd_ax()

        self.canvas_gcd = FigureCanvasQTAgg(self.fig_gcd)
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas_gcd)

    def _style_gcd_ax(self) -> None:
        ax = self.ax_gcd
        ax.set_facecolor(BG_GRAPH)
        ax.tick_params(colors=FG, labelcolor=FG, which='both')
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.set_xlabel('IOI index', color=FG, fontsize=9)
        ax.set_ylabel('IOI (s)',   color=FG, fontsize=9)
        ax.set_title('GCD: ---', color=FG, fontsize=9, pad=6)
        ax.grid(True, axis='y', color=BG_WIDGET, linestyle='--', linewidth=0.5)

    # ── Port / mode helpers ───────────────────────────────────────────────────

    def _refresh_ports(self) -> None:
        self._port_combo.clear()
        ports = MidiInputHandler.list_ports()
        if ports:
            self._port_combo.addItems(ports)
            print(f"[midi] available ports: {ports}", flush=True)
        else:
            self._port_combo.addItem("(no ports found)")
            print("[midi] no MIDI ports found", flush=True)

    def _on_mode_changed(self) -> None:
        is_midi = self._mode_combo.currentIndex() == 0
        self._port_section.setVisible(is_midi)

    def _current_mode(self) -> str:
        return 'midi' if self._mode_combo.currentIndex() == 0 else 'mock'

    # ── GUI update helpers (main thread only) ─────────────────────────────────

    def _update_labels(self, result: EstimatorResult) -> None:
        self._val_labels['tempo'].setText(f'{result.tempo_bpm:.2f} BPM')
        self._val_labels['beat'].setText(f'{result.beat_position:.3f}')
        self._val_labels['m_beat'].setText(f'{result.measure_beat}')
        self._val_labels['conf'].setText(f'{result.confidence:.2f}')

        if result.is_converged:
            self._conv_lbl.setText('YES')
            self._conv_lbl.setStyleSheet(f'color: {ACCENT_GREEN};')
            self._acorr_lbl.setText('---')
        else:
            self._conv_lbl.setText('no')
            self._conv_lbl.setStyleSheet(f'color: #F9E2AF;')
            if result.autocorr_tempo is not None:
                self._acorr_lbl.setText(f'{result.autocorr_tempo:.1f} BPM')
            else:
                self._acorr_lbl.setText('---')

        if result.gate_result is not None:
            gr = result.gate_result
            self._kalman_gated_tempo = gr.gated_tempo
            self._kalman_top_lbl.setText(f'{gr.gated_tempo:.2f} BPM')
            if gr.accepted:
                self._gate_lbl.setText('ACCEPT')
                self._gate_lbl.setStyleSheet(f'color: {ACCENT_GREEN};')
            else:
                self._gate_lbl.setText(f'REJECT [{gr.reject_reason}]')
                self._gate_lbl.setStyleSheet(f'color: {ACCENT_RED};')
            self._kalman_var_lbl.setText(f'{gr.current_var:.4f}')

        if result.last_category is not None:
            col = _CAT_COLOR.get(result.last_category, FG_DIM)
            self._last_hit_lbl.setText(result.last_category.name)
            self._last_hit_lbl.setStyleSheet(f'color: {col};')

        if result.category_counts is not None:
            cc = result.category_counts
            k = cc.get(InstrumentCategory.KICK,   0)
            s = cc.get(InstrumentCategory.SNARE,  0)
            h = cc.get(InstrumentCategory.HIHAT,  0)
            o = cc.get(InstrumentCategory.OTHERS, 0)
            self._counts_lbl.setText(f'K:{k}  S:{s}  H:{h}  O:{o}')

        if result.gcd_tempo is not None and result.gcd_tempo > 0:
            gcd_interval = 60.0 / result.gcd_tempo
            self._gcd_lbl.setText(
                f'{gcd_interval:.3f} s  (conf:{result.gcd_confidence:.2f})'
            )
        else:
            self._gcd_lbl.setText('---')

        self._proc_lbl.setText(f'proc: {result.processing_time_ms:.1f} ms')

    def _update_graph(self, result: EstimatorResult) -> None:
        self.event_count += 1
        self.tempo_hist.append(result.tempo_bpm)

        n = len(self.tempo_hist)
        x = list(range(self.event_count - n + 1, self.event_count + 1))

        self.tempo_line.set_xdata(x)
        self.tempo_line.set_ydata(list(self.tempo_hist))

        # Kalman gate overlays
        if result.gate_result is not None:
            gr = result.gate_result
            self._pred_hist.append(gr.predicted_tempo)
            self._var_hist.append(gr.current_var)
            if not gr.accepted:
                self._rej_x.append(self.event_count)
                self._rej_y.append(gr.raw_candidate)

            pred_n = len(self._pred_hist)
            pred_x = list(range(self.event_count - pred_n + 1, self.event_count + 1))
            self._pred_line.set_xdata(pred_x)
            self._pred_line.set_ydata(list(self._pred_hist))

            self._rej_line.set_xdata(self._rej_x)
            self._rej_line.set_ydata(self._rej_y)

            if self._fill is not None:
                self._fill.remove()
                self._fill = None
            sigma2 = np.sqrt(np.array(list(self._var_hist)))
            pred_a = np.array(list(self._pred_hist))
            self._fill = self.ax.fill_between(
                pred_x, pred_a - 2 * sigma2, pred_a + 2 * sigma2,
                alpha=0.12, color=ACCENT_BLUE,
            )

        # Category event markers
        if result.last_category is not None:
            gated_y = (result.gate_result.gated_tempo
                       if result.gate_result is not None else result.tempo_bpm)
            self._cat_x[result.last_category].append(self.event_count)
            self._cat_y[result.last_category].append(gated_y)
            line = self._cat_lines[result.last_category]
            line.set_xdata(self._cat_x[result.last_category])
            line.set_ydata(self._cat_y[result.last_category])

        self.ax.set_xlim(x[0], max(x[-1], x[0] + 1))

        # IOI per note number の更新
        if result.note_number is not None and result.ioi_sec is not None:
            self._ioi_by_note[result.note_number] = result.ioi_sec
            if result.last_category is not None:
                self._category_by_note[result.note_number] = result.last_category
            self._redraw_ioi()

        self.canvas.draw_idle()

    def _redraw_ioi(self) -> None:
        ax = self.ax_ioi
        ax.cla()
        self._style_ioi_ax()

        # Kalman beat period の縦線
        if self._kalman_gated_tempo is not None and self._kalman_gated_tempo > 0:
            beat_period = 60.0 / self._kalman_gated_tempo
            if 0 < beat_period <= 2.0:
                ax.axvline(x=beat_period, color=ACCENT_BLUE, linestyle='--',
                           linewidth=1.0, alpha=0.8, zorder=6)

        if not self._ioi_by_note:
            return

        notes  = sorted(self._ioi_by_note.keys())
        iois   = [self._ioi_by_note[n] for n in notes]
        colors = [
            _CAT_COLOR.get(self._category_by_note.get(n), FG_DIM)
            for n in notes
        ]
        y_pos = list(range(len(notes)))
        ax.barh(y_pos, iois, color=colors, alpha=0.85, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([str(n) for n in notes], fontsize=8)
        ax.set_xlim(0, 2.0)

    def _update_gcd_graph(self, result: EstimatorResult) -> None:
        ax = self.ax_gcd
        ax.cla()
        self._style_gcd_ax()

        gcd_period = 60.0 / result.gcd_tempo if (result.gcd_tempo is not None and result.gcd_tempo > 0) else None
        if gcd_period is not None:
            title = f'GCD: {gcd_period:.3f} s (conf: {result.gcd_confidence:.2f})'
        else:
            title = 'GCD: ---'
        ax.set_title(title, color=FG, fontsize=9, pad=6)

        iois = result.gcd_iois or []
        if iois:
            x_pos = list(range(len(iois)))
            ax.bar(x_pos, iois, color=ACCENT_GREEN, alpha=0.85, width=0.6)
            ax.set_xlim(-0.5, len(iois) - 0.5)
            y_max = max(iois)
            if gcd_period is not None:
                y_max = max(y_max, gcd_period)
            ax.set_ylim(0, y_max * 1.15)
        else:
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(0, 2.0)

        if gcd_period is not None:
            ax.axhline(y=gcd_period, color=ACCENT_BLUE, linestyle='--',
                       linewidth=1.0, alpha=0.8, zorder=6)

        self.canvas_gcd.draw_idle()

    # ── Queue polling ─────────────────────────────────────────────────────────

    def _poll_results(self) -> None:
        processed = 0
        try:
            while processed < 10:
                result = self.result_queue.get_nowait()
                print(
                    f"[gui ] event {self.event_count + 1:3d}  "
                    f"tempo={result.tempo_bpm:.2f}  "
                    f"beat={result.beat_position:.3f}",
                    flush=True,
                )
                self._update_labels(result)
                self._update_graph(result)
                self._update_gcd_graph(result)
                self.result_history.append(result)
                processed += 1
        except queue.Empty:
            pass

    # ── Control ───────────────────────────────────────────────────────────────

    def _start(self) -> None:
        if self.is_running:
            return
        if self._current_mode() == 'midi':
            self._start_midi()
        else:
            self._start_mock()

    def _start_midi(self) -> None:
        port_index = self._port_combo.currentIndex()
        if port_index < 0:
            print("[midi] no port selected", flush=True)
            return
        try:
            self._midi_handler = MidiInputHandler(
                self.particle_filter,
                port_index=port_index,
                on_result=self.result_queue.put,
            )
            self._midi_handler.start()
            self.is_running = True
            print(f"[midi] started on port {port_index}", flush=True)
        except RuntimeError as e:
            print(f"[midi] ERROR: {e}", flush=True)

    def _start_mock(self) -> None:
        self.is_running = True
        threading.Thread(target=self._mock_thread, daemon=True).start()

    def _stop(self) -> None:
        self.is_running = False
        if self._midi_handler is not None:
            self._midi_handler.stop()
            self._midi_handler = None

    def _stop_and_summarize(self) -> None:
        self._stop()
        if self.result_history:
            from midi_tempo_hmm.tools.gate_visualizer import plot_gate_debug, print_gate_summary
            print_gate_summary(list(self.result_history))
            plot_gate_debug(list(self.result_history), block=False)

    def _reset(self) -> None:
        self._stop()
        with self.lock:
            self.particle_filter.reset()
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        self.tempo_hist.clear()
        self.event_count = 0
        self.result_history.clear()
        self._pred_hist.clear()
        self._var_hist.clear()
        self._rej_x.clear()
        self._rej_y.clear()
        for c in InstrumentCategory:
            self._cat_x[c].clear()
            self._cat_y[c].clear()
            self._cat_lines[c].set_xdata([])
            self._cat_lines[c].set_ydata([])
        for lbl in self._val_labels.values():
            lbl.setText('---')
        self._proc_lbl.setText('')
        self._gcd_lbl.setText('---')
        self._kalman_top_lbl.setText('---')
        self._kalman_top_lbl.setStyleSheet(f'color: {ACCENT_BLUE};')
        self._gate_lbl.setText('---')
        self._gate_lbl.setStyleSheet(f'color: {FG_DIM};')
        self._kalman_var_lbl.setText('---')
        self._kalman_gated_tempo = None
        self.tempo_line.set_xdata([])
        self.tempo_line.set_ydata([])
        self._pred_line.set_xdata([])
        self._pred_line.set_ydata([])
        self._rej_line.set_xdata([])
        self._rej_line.set_ydata([])
        if self._fill is not None:
            self._fill.remove()
            self._fill = None
        # IOI グラフをクリア
        self._ioi_by_note.clear()
        self._category_by_note.clear()
        self.ax_ioi.cla()
        self._style_ioi_ax()
        self.canvas.draw_idle()

        # GCDパネルをクリア
        self.ax_gcd.cla()
        self._style_gcd_ax()
        self.canvas_gcd.draw_idle()

    def _auto_start(self) -> None:
        if self._current_mode() == 'midi' and self._port_combo.count() > 0:
            first = self._port_combo.itemText(0)
            if first != "(no ports found)":
                self._port_combo.setCurrentIndex(0)
                self._start()

    def closeEvent(self, event) -> None:
        self._stop()
        self._timer.stop()
        super().closeEvent(event)

    # ── Mock worker thread ────────────────────────────────────────────────────

    def _mock_thread(self) -> None:
        sleep_sec = 60.0 / self.args.tempo

        if self.args.drum_pattern != 'none':
            raw = generate_drum_pattern_events(
                self.args.tempo, n_bars=32,
                pattern=self.args.drum_pattern,
                humanize_ms=self.args.humanize,
            )
            event_list = [(ts, note, ch) for ts, note, ch in raw]
            print(
                f"[mock] drum pattern '{self.args.drum_pattern}': "
                f"{len(event_list)} events at {self.args.tempo} BPM",
                flush=True,
            )
        else:
            timestamps = generate_mock_events(
                self.args.tempo, n_beats=256, humanize_ms=self.args.humanize
            )
            event_list = [(ts, None, None) for ts in timestamps]
            print(f"[mock] starting: {len(event_list)} events at {self.args.tempo} BPM",
                  flush=True)

        for i, (ts, note, channel) in enumerate(event_list):
            if not self.is_running:
                print("[mock] stopped.", flush=True)
                break
            with self.lock:
                result = self.particle_filter.update(ts, note_number=note, channel=channel)
            if result is None:
                print(f"[mock] event {i+1:3d}  (first — skipped)", flush=True)
            else:
                self.result_queue.put(result)
                conv = "CONV" if result.is_converged else "    "
                print(
                    f"[mock] event {i+1:3d}  ts={ts:.4f}s  "
                    f"tempo={result.tempo_bpm:.2f}  {conv}",
                    flush=True,
                )
            time.sleep(sleep_sec / max(1, len(event_list) // 256))

        self.is_running = False
        print("[mock] done.", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='MIDI Tempo Estimator — GUI (PyQt6)')
    p.add_argument('--tempo',    type=float, default=120.0,
                   help='Mock tempo in BPM (default: 120.0)')
    p.add_argument('--humanize', type=float, default=10.0,
                   help='Timing jitter std-dev in ms (default: 10.0)')
    p.add_argument('--mode',     type=str,   default='midi',
                   choices=['midi', 'mock'],
                   help='デフォルトの入力モード: midi (default) / mock')
    p.add_argument('--drum-pattern', type=str, default='none',
                   choices=['none', 'basic_rock', 'hihat_16th', 'sparse'],
                   help='Drum pattern for mock mode (default: none)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(name)s %(levelname)s %(message)s')
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = TempoEstimatorApp(args)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
