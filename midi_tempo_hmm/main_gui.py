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
from midi_tempo_hmm.core.particle_filter import ParticleFilter
from midi_tempo_hmm.interface.midi_input import MidiInputHandler
from midi_tempo_hmm.interface.mock_input import generate_mock_events
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

        left  = QWidget(); right = QWidget()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([260, 700])

        self._build_left(left)
        self._build_right(right)

    def _build_left(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(0)

        font_dim   = QFont('Helvetica', 10)
        font_value = QFont('Helvetica', 26); font_value.setBold(True)
        font_small = QFont('Helvetica', 9)
        font_combo = QFont('Helvetica', 11)

        # ── 現在値ラベル ─────────────────────────────────────────────────────
        self._val_labels: dict[str, QLabel] = {}
        for key, title in [
            ('tempo',  'TEMPO'),
            ('beat',   'BEAT'),
            ('m_beat', 'MEASURE'),
            ('conf',   'CONFIDENCE'),
        ]:
            hdr = QLabel(title)
            hdr.setFont(font_dim)
            hdr.setStyleSheet(f'color: {FG_DIM}; margin-top: 16px;')
            layout.addWidget(hdr)

            val = QLabel('---')
            val.setFont(font_value)
            layout.addWidget(val)
            self._val_labels[key] = val

        self._proc_lbl = QLabel('')
        self._proc_lbl.setFont(font_small)
        self._proc_lbl.setStyleSheet(f'color: {FG_DIM}; margin-top: 4px;')
        layout.addWidget(self._proc_lbl)

        layout.addStretch(1)

        # ── モード選択 ────────────────────────────────────────────────────────
        mode_hdr = QLabel('INPUT MODE')
        mode_hdr.setFont(font_dim)
        mode_hdr.setStyleSheet(f'color: {FG_DIM}; margin-top: 10px;')
        layout.addWidget(mode_hdr)

        self._mode_combo = QComboBox()
        self._mode_combo.setFont(font_combo)
        self._mode_combo.addItems(['MIDI Input', 'Mock (test)'])
        # 引数で --mode mock が指定されていたらデフォルトを Mock に
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

        self._refresh_ports()  # initial population
        self._on_mode_changed()  # initial visibility

        # ── ボタン ────────────────────────────────────────────────────────────
        for text, slot in [
            ('▶  START', self._start),
            ('■  STOP',  self._stop),
            ('↺  RESET', self._reset),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            layout.addWidget(btn)
            layout.addSpacing(4)

    def _build_right(self, parent: QWidget) -> None:
        self.fig = Figure(facecolor=BG, tight_layout=True)
        self.ax  = self.fig.add_subplot(111)

        self.ax.set_facecolor(BG_GRAPH)
        self.ax.tick_params(colors=FG, labelcolor=FG, which='both')
        self.ax.xaxis.label.set_color(FG)
        self.ax.yaxis.label.set_color(FG)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(BORDER)
        self.ax.set_xlabel('Event #',    color=FG)
        self.ax.set_ylabel('Tempo (BPM)', color=FG)
        self.ax.set_ylim(25, 300)
        self.ax.grid(True, color=BG_WIDGET, linestyle='--', linewidth=0.5)

        self.tempo_line, = self.ax.plot([], [], color=ACCENT_RED, linewidth=2)

        self.canvas = FigureCanvasQTAgg(self.fig)
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

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
        self._proc_lbl.setText(f'proc: {result.processing_time_ms:.1f} ms')

    def _update_graph(self, result: EstimatorResult) -> None:
        self.event_count += 1
        self.tempo_hist.append(result.tempo_bpm)

        n = len(self.tempo_hist)
        x = list(range(self.event_count - n + 1, self.event_count + 1))

        self.tempo_line.set_xdata(x)
        self.tempo_line.set_ydata(list(self.tempo_hist))

        self.ax.set_xlim(x[0], max(x[-1], x[0] + 1))
        self.canvas.draw_idle()

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
                on_result=self.result_queue.put,   # ← GUIキューに直接繋ぐ
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
        for lbl in self._val_labels.values():
            lbl.setText('---')
        self._proc_lbl.setText('')
        self.tempo_line.set_xdata([])
        self.tempo_line.set_ydata([])
        self.canvas.draw_idle()

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
        timestamps = generate_mock_events(
            self.args.tempo, n_beats=256, humanize_ms=self.args.humanize
        )
        sleep_sec = 60.0 / self.args.tempo
        print(f"[mock] starting: {len(timestamps)} events at {self.args.tempo} BPM",
              flush=True)

        for i, ts in enumerate(timestamps):
            if not self.is_running:
                print("[mock] stopped.", flush=True)
                break
            with self.lock:
                result = self.particle_filter.update(ts)
            if result is None:
                print(f"[mock] event {i+1:3d}  (first — skipped)", flush=True)
            else:
                self.result_queue.put(result)
                print(
                    f"[mock] event {i+1:3d}  ts={ts:.4f}s  "
                    f"tempo={result.tempo_bpm:.2f}",
                    flush=True,
                )
            time.sleep(sleep_sec)

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
