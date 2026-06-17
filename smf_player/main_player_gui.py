"""SMF Player GUI (PyQt6)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mido
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from midi_event import BeatMap, PlaybackEvent, build_beat_map, load_midi_events
from player_engine import PlayerEngine

# ── Colours ────────────────────────────────────────────────────────────────────
BG           = '#1E1E2E'
BG_LIST      = '#181825'
BG_WIDGET    = '#313244'
BG_HOVER     = '#45475A'
FG           = '#CDD6F4'
FG_DIM       = '#6C7086'
FG_HI        = '#1E1E2E'
BG_HI        = '#89B4FA'
BORDER       = '#45475A'
ACCENT_GREEN = '#A6E3A1'
ACCENT_RED   = '#F38BA8'

SETTINGS_PATH = Path.home() / '.smf_player.json'

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BG};
    color: {FG};
}}
QListWidget {{
    background-color: {BG_LIST};
    color: {FG};
    border: 1px solid {BORDER};
    font-family: "Courier New";
    font-size: 10pt;
}}
QListWidget::item:selected,
QListWidget::item:selected:!active {{
    background-color: {BG_HI};
    color: {FG_HI};
}}
QComboBox {{
    background-color: {BG_WIDGET};
    color: {FG};
    border: 1px solid {BORDER};
    padding: 3px 6px;
    min-width: 180px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_WIDGET};
    color: {FG};
    selection-background-color: {BG_HI};
    selection-color: {FG_HI};
}}
QComboBox::drop-down {{
    border: none;
}}
QPushButton {{
    background-color: {BG_WIDGET};
    color: {FG};
    border: none;
    padding: 6px 10px;
    font-size: 11pt;
    font-family: "Helvetica";
}}
QPushButton:hover {{
    background-color: {BG_HOVER};
}}
QPushButton:pressed {{
    background-color: {BORDER};
}}
QCheckBox {{
    color: {FG};
    font-family: "Courier New";
    font-size: 9pt;
    spacing: 4px;
}}
QCheckBox::indicator {{
    width: 12px;
    height: 12px;
    background-color: {BG_WIDGET};
    border: 1px solid {BORDER};
}}
QCheckBox::indicator:checked {{
    background-color: {BG_HI};
}}
QLabel {{
    color: {FG_DIM};
    font-family: "Helvetica";
    font-size: 10pt;
}}
QFrame[frameShape="4"],
QFrame[frameShape="5"] {{
    color: {BORDER};
}}
QSplitter::handle {{
    background-color: {BORDER};
}}
"""


class SmfPlayerWindow(QMainWindow):
    # Signals for cross-thread communication from playback thread → GUI thread
    _sig_event_played     = pyqtSignal(object)  # PlaybackEvent
    _sig_playback_finished = pyqtSignal()

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        self.engine: PlayerEngine | None = None
        self.events: list[PlaybackEvent] = []
        self._filepath: str | None = None
        self._beat_map: BeatMap | None = None

        self.channel_mute_vars: dict[int, QCheckBox] = {}

        self.setWindowTitle('SMF Player')
        self.setMinimumSize(860, 520)
        self.setStyleSheet(STYLESHEET)

        self._build_ui()
        self._sig_event_played.connect(self._apply_event_played)
        self._sig_playback_finished.connect(self._apply_playback_finished)
        self._apply_settings(self._load_settings())

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)

        # ── Left: event list ─────────────────────────────────────────────────
        self.list_widget = QListWidget()
        self.list_widget.setFont(QFont('Courier New', 10))
        self.list_widget.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.list_widget.clicked.connect(self._on_list_click)
        self.list_widget.doubleClicked.connect(self._on_list_click)
        splitter.addWidget(self.list_widget)

        # ── Right: control panel ─────────────────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(4)
        right_widget.setMaximumWidth(320)
        right_widget.setMinimumWidth(260)

        # MIDI output port
        right_layout.addWidget(self._dim_label('MIDI Output Port'))
        port_row = QHBoxLayout()
        self._port_combo = QComboBox()
        self._refresh_ports()
        port_row.addWidget(self._port_combo, stretch=1)
        refresh_btn = QPushButton('⟳')
        refresh_btn.setFixedWidth(32)
        refresh_btn.clicked.connect(self._refresh_ports)
        port_row.addWidget(refresh_btn)
        right_layout.addLayout(port_row)

        right_layout.addWidget(self._separator())

        # Status row: state indicator + BPM + measure/beat
        status_row = QHBoxLayout()
        self._status_label = QLabel('■  STOPPED')
        self._status_label.setFont(QFont('Courier New', 11))
        self._status_label.setStyleSheet(f'color: {FG_DIM};')
        self._bpm_label = QLabel('')
        self._bpm_label.setFont(QFont('Courier New', 11))
        self._bpm_label.setStyleSheet(f'color: {FG_DIM};')
        self._bpm_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._beat_label = QLabel('')
        self._beat_label.setFont(QFont('Courier New', 11))
        self._beat_label.setStyleSheet(f'color: {FG_DIM};')
        self._beat_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        status_row.addWidget(self._status_label)
        status_row.addWidget(self._bpm_label, stretch=1)
        status_row.addWidget(self._beat_label)
        right_layout.addLayout(status_row)

        right_layout.addWidget(self._separator())

        # Playback buttons
        for text, slot in [
            ('▶  START', self._on_start),
            ('⏸  STOP',  self._on_stop),
            ('⏮  RESET', self._on_reset),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            right_layout.addWidget(btn)

        right_layout.addWidget(self._separator())

        # Channel mute
        right_layout.addWidget(self._dim_label('Channel Mute'))
        mute_grid = QGridLayout()
        mute_grid.setHorizontalSpacing(4)
        mute_grid.setVerticalSpacing(2)
        for ch in range(16):
            row, col = divmod(ch, 4)
            label = f'Ch{ch + 1:02d}'
            if ch == 9:
                label += '(D)'
            cb = QCheckBox(label)
            cb.stateChanged.connect(
                lambda state, c=ch: self._on_channel_mute_toggle(c))
            self.channel_mute_vars[ch] = cb
            mute_grid.addWidget(cb, row, col)
        right_layout.addLayout(mute_grid)

        right_layout.addWidget(self._separator())

        # Open file button
        open_btn = QPushButton('Open MIDI File...')
        open_btn.clicked.connect(self._on_open_file)
        right_layout.addWidget(open_btn)

        right_layout.addWidget(self._separator())

        # File name display
        self._file_label = QLabel('(no file)')
        self._file_label.setFont(QFont('Courier New', 9))
        self._file_label.setStyleSheet(f'color: {FG_DIM};')
        self._file_label.setWordWrap(True)
        right_layout.addWidget(self._file_label)

        # Position display
        self._pos_label  = QLabel('Event: --- / ---')
        self._time_label = QLabel('Time:  --- / ---')
        self._pos_label.setFont(QFont('Courier New', 10))
        self._time_label.setFont(QFont('Courier New', 10))
        right_layout.addWidget(self._pos_label)
        right_layout.addWidget(self._time_label)

        right_layout.addStretch()

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        self.setCentralWidget(splitter)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _dim_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f'color: {FG_DIM};')
        return lbl

    def _separator(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet(f'color: {BORDER}; margin: 4px 0;')
        return line

    def _refresh_ports(self) -> None:
        current = self._port_combo.currentText()
        self._port_combo.clear()
        ports = mido.get_output_names()
        for p in ports:
            self._port_combo.addItem(p)
        if current in ports:
            self._port_combo.setCurrentText(current)
        elif ports:
            self._port_combo.setCurrentIndex(0)

    def _current_port(self) -> str | None:
        val = self._port_combo.currentText()
        return val if val else None

    def _ensure_engine(self) -> bool:
        port = self._current_port()
        if not port:
            return False
        if self.engine is None:
            self.engine = PlayerEngine(port)
            self._wire_callbacks()
            for ch, cb in self.channel_mute_vars.items():
                if cb.isChecked():
                    self.engine.set_channel_mute(ch, True)
        return True

    def _wire_callbacks(self) -> None:
        if self.engine is None:
            return
        self.engine.on_event_played      = self._on_event_played
        self.engine.on_playback_finished = self._on_playback_finished

    def _update_position_display(self, index: int | None = None) -> None:
        total = len(self.events)
        if index is None:
            idx = self.engine.current_index if self.engine else 0
        else:
            idx = index

        if total == 0:
            self._pos_label.setText('Event: --- / ---')
            self._time_label.setText('Time:  --- / ---')
            return

        idx_disp   = min(idx, total - 1)
        ev         = self.events[idx_disp]
        total_time = self.events[-1].abs_time_sec

        self._pos_label.setText(f'Event: {idx_disp + 1:04d} / {total:04d}')
        self._time_label.setText(
            f'Time:  {ev.abs_time_sec:7.3f}s / {total_time:.3f}s')

    def _set_status(self, text: str, color: str) -> None:
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f'color: {color};')

    def _update_beat_display(self, sec: float) -> None:
        if self._beat_map is None:
            return
        bpm     = self._beat_map.sec_to_bpm(sec)
        m, b    = self._beat_map.sec_to_pos(sec)
        self._bpm_label.setText(f'{bpm:.1f} BPM')
        self._beat_label.setText(f'M:{m:03d} B:{b}')

    def _highlight_row(self, index: int) -> None:
        self.list_widget.blockSignals(True)
        self.list_widget.setCurrentRow(index)
        self.list_widget.blockSignals(False)
        item = self.list_widget.item(index)
        if item is not None:
            self.list_widget.scrollToItem(
                item, QListWidget.ScrollHint.EnsureVisible)

    # ── Settings persistence ───────────────────────────────────────────────────

    def _load_settings(self) -> dict:
        try:
            return json.loads(SETTINGS_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _save_settings(self) -> None:
        settings: dict = {
            'last_file': self._filepath,
            'channel_mutes': {
                str(ch): cb.isChecked()
                for ch, cb in self.channel_mute_vars.items()
            },
        }
        try:
            SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
        except OSError:
            pass

    def _apply_settings(self, settings: dict) -> None:
        for ch_str, muted in settings.get('channel_mutes', {}).items():
            ch = int(ch_str)
            if ch in self.channel_mute_vars:
                self.channel_mute_vars[ch].setChecked(bool(muted))

        last_file = settings.get('last_file')
        if last_file and Path(last_file).exists():
            QTimer.singleShot(200, lambda: self._load_file(last_file))

    # ── Event handlers ─────────────────────────────────────────────────────────

    def _on_open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open MIDI File', '',
            'MIDI files (*.mid *.midi);;All files (*.*)',
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str) -> None:
        self.events    = load_midi_events(path)
        self._filepath = path
        self._beat_map = build_beat_map(path)

        self._file_label.setText(Path(path).name)

        self.list_widget.clear()
        for ev in self.events:
            self.list_widget.addItem(QListWidgetItem(ev.display_text))

        self.list_widget.setCurrentRow(-1)
        self._bpm_label.setText('')
        self._beat_label.setText('')
        self._update_position_display(0)
        self._set_status('■  STOPPED', FG_DIM)

        if self.engine is not None:
            self.engine.load(path)

    def _on_start(self) -> None:
        if not self.events:
            return
        if not self._ensure_engine():
            return
        assert self.engine is not None
        if not self.engine.events and self._filepath:
            self.engine.load(self._filepath)
        self.engine.play()
        self._set_status('▶  PLAYING', ACCENT_GREEN)

    def _on_stop(self) -> None:
        if self.engine:
            self.engine.stop()
        self._set_status('■  STOPPED', FG_DIM)

    def _on_reset(self) -> None:
        if self.engine:
            self.engine.stop()
            self.engine.seek(0)

        self.list_widget.setCurrentRow(-1)
        self._bpm_label.setText('')
        self._beat_label.setText('')
        self._update_position_display(0)
        self._set_status('■  STOPPED', FG_DIM)

    def _on_list_click(self) -> None:
        if not self.engine or not self.events:
            return
        index = self.list_widget.currentRow()
        if index < 0:
            return
        self.engine.seek(index)
        self._update_position_display(index)
        if self._beat_map is not None:
            self._update_beat_display(self.events[index].abs_time_sec)

    def _on_event_played(self, ev: PlaybackEvent) -> None:
        # Called from playback thread — emit to GUI thread via queued signal
        self._sig_event_played.emit(ev)

    def _apply_event_played(self, ev: PlaybackEvent) -> None:
        # Runs in GUI thread
        self._highlight_row(ev.index)
        self._update_position_display(ev.index)
        self._update_beat_display(ev.abs_time_sec)

    def _on_playback_finished(self) -> None:
        self._sig_playback_finished.emit()

    def _apply_playback_finished(self) -> None:
        self._pos_label.setText('再生終了 — RESET で先頭に戻る')
        self._set_status('■  FINISHED', ACCENT_RED)
        self._bpm_label.setText('')
        self._beat_label.setText('')

    def _on_channel_mute_toggle(self, channel: int) -> None:
        if self.engine:
            self.engine.set_channel_mute(
                channel, self.channel_mute_vars[channel].isChecked())

    # ── Public helpers (CLI) ───────────────────────────────────────────────────

    def set_port(self, port_name: str) -> None:
        ports = mido.get_output_names()
        if port_name in ports:
            self._port_combo.setCurrentText(port_name)

    def load_file(self, filepath: str) -> None:
        self._load_file(filepath)

    # ── Window close ──────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self.engine:
            self.engine.stop()
            self.engine.close()
        self._save_settings()
        event.accept()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='SMF Player')
    parser.add_argument('--port', type=str, default=None,
                        help='初期選択する MIDI 出力ポート名')
    parser.add_argument('--file', type=str, default=None,
                        help='起動時に読み込む MIDI ファイルパス')
    args = parser.parse_args()

    app    = QApplication(sys.argv)
    window = SmfPlayerWindow(args)

    if args.port:
        window.set_port(args.port)

    if args.file:
        QTimer.singleShot(100, lambda: window.load_file(args.file))

    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
