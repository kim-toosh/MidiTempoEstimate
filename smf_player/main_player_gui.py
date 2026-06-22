"""SMF Player GUI (Tkinter)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

# Ensure smf_player/ is on sys.path when run as `python -m smf_player.main_player_gui`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mido

from midi_event import BeatMap, PlaybackEvent, build_beat_map, load_midi_events
from player_engine import PlayerEngine

# ── Colours ───────────────────────────────────────────────────────────────────
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

# (mido type string, display label) — channel messages only
_MSG_TYPES: list[tuple[str, str]] = [
    ('note_on',        'Note On'),
    ('note_off',       'Note Off'),
    ('control_change', 'Ctrl Chg'),
    ('program_change', 'Prog Chg'),
    ('polytouch',      'Poly Press'),
    ('aftertouch',     'Ch Press'),
    ('pitchwheel',     'Pitch Bend'),
]


class SmfPlayerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root   = root
        self.engine : PlayerEngine | None = None
        self.events : list[PlaybackEvent] = []
        self._filepath  : str | None = None
        self._beat_map  : BeatMap | None = None
        self._programmatic_select = False  # guard against seek during playback highlight

        self.channel_mute_vars: dict[int, tk.BooleanVar] = {
            ch: tk.BooleanVar(value=False) for ch in range(16)
        }
        self.msg_type_show_vars: dict[str, tk.BooleanVar] = {
            mtype: tk.BooleanVar(value=False) for mtype, _ in _MSG_TYPES
        }
        self._listbox_to_event: list[int] = []   # listbox row → events index
        self._event_to_listbox: dict[int, int] = {}  # events index → listbox row

        root.title('SMF Player')
        root.configure(bg=BG)
        root.minsize(860, 620)
        root.geometry('860x620')

        self._configure_ttk_style()
        self._build_ui()
        root.protocol('WM_DELETE_WINDOW', self._on_close)
        self._apply_settings(self._load_settings())

    # ── ttk theme ─────────────────────────────────────────────────────────────

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
        style.configure('Vertical.TScrollbar',
                        background=BG_WIDGET, troughcolor=BG,
                        arrowcolor=FG_DIM, bordercolor=BG)
        style.configure('TSeparator', background=BORDER)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = self.root

        pane = tk.PanedWindow(root, orient=tk.HORIZONTAL,
                              bg=BORDER, sashwidth=4, sashrelief=tk.FLAT)
        pane.pack(fill=tk.BOTH, expand=True)

        # ── Left: event list ─────────────────────────────────────────────────
        left = tk.Frame(pane, bg=BG)
        pane.add(left, minsize=450, stretch='always')

        list_frame = tk.Frame(left, bg=BG)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL,
                                  style='Vertical.TScrollbar')
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(
            list_frame,
            font=('Courier New', 10),
            bg=BG_LIST, fg=FG,
            selectbackground=BG_HI, selectforeground=FG_HI,
            activestyle='none',
            selectmode=tk.SINGLE,
            yscrollcommand=scrollbar.set,
            borderwidth=0, highlightthickness=0,
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)
        self.listbox.bind('<<ListboxSelect>>', self._on_listbox_select)
        self.listbox.bind('<Up>',    lambda e: self._on_listbox_key_move(-1) or 'break')
        self.listbox.bind('<Down>',  lambda e: self._on_listbox_key_move(+1) or 'break')
        self.listbox.bind('<space>', lambda e: self._on_space() or 'break')
        root.bind('<space>', lambda e: self._on_space())

        # ── Right: control panel ─────────────────────────────────────────────
        right = tk.Frame(pane, bg=BG, width=300)
        pane.add(right, minsize=260, stretch='never')

        pad = dict(padx=10, pady=3)

        def _sep() -> None:
            ttk.Separator(right, orient=tk.HORIZONTAL).pack(
                fill=tk.X, padx=8, pady=5)

        def _dim_lbl(text: str) -> tk.Label:
            return tk.Label(right, text=text, bg=BG, fg=FG_DIM,
                            font=('Helvetica', 9))

        # MIDI Output Port
        _dim_lbl('MIDI Output Port').pack(anchor='w', **pad)
        port_row = tk.Frame(right, bg=BG)
        port_row.pack(fill=tk.X, padx=10, pady=2)
        self._port_var   = tk.StringVar()
        self._port_combo = ttk.Combobox(
            port_row, textvariable=self._port_var, state='readonly', width=22)
        self._port_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._mk_btn(port_row, '⟳', self._refresh_ports,
                     width=2).pack(side=tk.LEFT, padx=(4, 0))
        self._refresh_ports()

        _sep()

        # Status row: state + BPM + MEAS/BEAT
        status_row = tk.Frame(right, bg=BG)
        status_row.pack(fill=tk.X, **pad)
        self._status_var = tk.StringVar(value='■  STOPPED')
        self._status_lbl = tk.Label(status_row, textvariable=self._status_var,
                                    bg=BG, fg=FG_DIM,
                                    font=('Courier New', 11))
        self._status_lbl.pack(side=tk.LEFT)
        self._bpm_var = tk.StringVar(value='')
        tk.Label(status_row, textvariable=self._bpm_var,
                 bg=BG, fg=FG_DIM,
                 font=('Courier New', 11)).pack(side=tk.LEFT, padx=(8, 0))
        self._beat_var = tk.StringVar(value='')
        tk.Label(status_row, textvariable=self._beat_var,
                 bg=BG, fg=FG_DIM,
                 font=('Courier New', 11)).pack(side=tk.RIGHT)

        _sep()

        # Playback buttons
        for text, cmd in [
            ('▶  START', self._on_start),
            ('⏸  STOP',  self._on_stop),
            ('⏮  RESET', self._on_reset),
        ]:
            self._mk_btn(right, text, cmd).pack(fill=tk.X, padx=10, pady=2)

        _sep()

        # Channel Mute
        _dim_lbl('Channel Mute').pack(anchor='w', **pad)
        mute_frame = tk.Frame(right, bg=BG)
        mute_frame.pack(anchor='w', padx=10, pady=2)
        for ch in range(16):
            row, col = divmod(ch, 4)
            label = f'Ch{ch + 1:02d}'
            if ch == 9:
                label += '(D)'
            tk.Checkbutton(
                mute_frame, text=label,
                variable=self.channel_mute_vars[ch],
                command=lambda c=ch: self._on_mute_toggle(c),
                bg=BG, fg=FG, selectcolor=BG_WIDGET,
                activebackground=BG, activeforeground=FG,
                font=('Courier New', 9),
            ).grid(row=row, column=col, sticky='w', padx=2, pady=1)

        _sep()

        # Event Type Filter
        _dim_lbl('Event Type').pack(anchor='w', **pad)
        etype_frame = tk.Frame(right, bg=BG)
        etype_frame.pack(anchor='w', padx=10, pady=2)
        for i, (mtype, label) in enumerate(_MSG_TYPES):
            row, col = divmod(i, 2)
            tk.Checkbutton(
                etype_frame, text=label,
                variable=self.msg_type_show_vars[mtype],
                command=self._on_msg_type_toggle,
                bg=BG, fg=FG, selectcolor=BG_WIDGET,
                activebackground=BG, activeforeground=FG,
                font=('Courier New', 9),
            ).grid(row=row, column=col, sticky='w', padx=2, pady=1)

        _sep()

        # Open file button
        self._mk_btn(right, 'Open MIDI File...', self._on_open_file).pack(
            fill=tk.X, padx=10, pady=2)

        _sep()

        # File name display
        self._filename_var = tk.StringVar(value='(no file)')
        tk.Label(right, textvariable=self._filename_var,
                 bg=BG, fg=FG_DIM,
                 font=('Courier New', 9),
                 wraplength=220, justify='left').pack(anchor='w', padx=10, pady=2)

        # Position display
        self._pos_var  = tk.StringVar(value='Event: --- / ---')
        self._time_var = tk.StringVar(value='Time:  --- / ---')
        tk.Label(right, textvariable=self._pos_var,
                 bg=BG, fg=FG_DIM,
                 font=('Courier New', 10)).pack(anchor='w', padx=10, pady=1)
        tk.Label(right, textvariable=self._time_var,
                 bg=BG, fg=FG_DIM,
                 font=('Courier New', 10)).pack(anchor='w', padx=10, pady=1)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _mk_btn(self, parent: tk.Widget, text: str, cmd,
                width: int | None = None) -> tk.Button:
        kw: dict = {}
        if width is not None:
            kw['width'] = width
        return tk.Button(
            parent, text=text, command=cmd,
            bg=BG_WIDGET, fg=FG,
            activebackground=BG_HOVER, activeforeground=FG,
            relief=tk.FLAT, padx=6, pady=5,
            font=('Helvetica', 11), cursor='hand2',
            **kw,
        )

    def _refresh_ports(self) -> None:
        ports = mido.get_output_names()
        self._port_combo['values'] = ports
        if self._port_var.get() not in ports and ports:
            self._port_combo.current(0)

    def _current_port(self) -> str | None:
        val = self._port_var.get()
        return val if val else None

    def _ensure_engine(self) -> bool:
        port = self._current_port()
        if not port:
            return False
        if self.engine is None:
            self.engine = PlayerEngine(port)
            self.engine.on_event_played      = self._cb_event_played
            self.engine.on_playback_finished = self._cb_playback_finished
            for ch, var in self.channel_mute_vars.items():
                if var.get():
                    self.engine.set_channel_mute(ch, True)
        return True

    def _set_status(self, text: str, color: str) -> None:
        self._status_var.set(text)
        self._status_lbl.config(fg=color)

    def _update_beat_display(self, sec: float) -> None:
        if self._beat_map is None:
            return
        bpm  = self._beat_map.sec_to_bpm(sec)
        m, b = self._beat_map.sec_to_pos(sec)
        self._bpm_var.set(f'{bpm:.1f} BPM')
        self._beat_var.set(f'M:{m:03d} B:{b}')

    def _clear_beat_display(self) -> None:
        self._bpm_var.set('')
        self._beat_var.set('')

    def _rebuild_listbox(self) -> None:
        """Rebuild listbox applying channel-mute and event-type-show filters.

        Channel mute : checked   = hidden  (blacklist)
        Event type   : checked   = visible (whitelist); none checked = show all
        Non-channel messages (meta, sysex, …) are always visible.
        """
        show_types = {m for m, v in self.msg_type_show_vars.items() if v.get()}

        self.listbox.delete(0, tk.END)
        self._listbox_to_event = []
        self._event_to_listbox = {}
        prev_vis_sec = 0.0
        for ev in self.events:
            if ev.channel is not None and self.channel_mute_vars[ev.channel].get():
                continue
            # whitelist active + known channel-message type + not in whitelist → hide
            if show_types and ev.msg_type in self.msg_type_show_vars and ev.msg_type not in show_types:
                continue
            lb_idx = len(self._listbox_to_event)
            # Replace stored diff (from previous event in full list) with
            # the gap from the previous *visible* event.  diff_str occupies
            # chars [29:37] in the fixed-width display format.
            vis_diff = (ev.abs_time_sec - prev_vis_sec) * 1000
            disp = ev.display_text[:29] + f'd{vis_diff:5.0f}ms' + ev.display_text[37:]
            self.listbox.insert(tk.END, disp)
            self._listbox_to_event.append(ev.index)
            self._event_to_listbox[ev.index] = lb_idx
            prev_vis_sec = ev.abs_time_sec

    def _highlight_row(self, event_index: int) -> None:
        lb_idx = self._event_to_listbox.get(event_index)
        if lb_idx is None:
            return  # event is on a muted channel — skip highlight
        self._programmatic_select = True
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(lb_idx)
        self.listbox.see(lb_idx)
        self._programmatic_select = False

    def _update_position_display(self, index: int | None = None) -> None:
        total = len(self.events)
        if total == 0:
            self._pos_var.set('Event: --- / ---')
            self._time_var.set('Time:  --- / ---')
            return
        if index is None:
            idx = self.engine.current_index if self.engine else 0
        else:
            idx = index
        idx_disp   = min(idx, total - 1)
        ev         = self.events[idx_disp]
        total_time = self.events[-1].abs_time_sec
        self._pos_var.set(f'Event: {idx_disp + 1:04d} / {total:04d}')
        self._time_var.set(f'Time:  {ev.abs_time_sec:7.3f}s / {total_time:.3f}s')

    # ── Settings ──────────────────────────────────────────────────────────────

    def _load_settings(self) -> dict:
        try:
            return json.loads(SETTINGS_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _save_settings(self) -> None:
        settings = {
            'last_file': self._filepath,
            'channel_mutes': {str(ch): var.get()
                               for ch, var in self.channel_mute_vars.items()},
            'msg_type_shows': {mtype: var.get()
                                for mtype, var in self.msg_type_show_vars.items()},
        }
        try:
            SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
        except OSError:
            pass

    def _apply_settings(self, settings: dict) -> None:
        for ch_str, muted in settings.get('channel_mutes', {}).items():
            ch = int(ch_str)
            if ch in self.channel_mute_vars:
                self.channel_mute_vars[ch].set(bool(muted))
        for mtype, shown in settings.get('msg_type_shows', {}).items():
            if mtype in self.msg_type_show_vars:
                self.msg_type_show_vars[mtype].set(bool(shown))
        last_file = settings.get('last_file')
        if last_file and Path(last_file).exists():
            self.root.after(200, lambda: self._load_file(last_file))

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_open_file(self) -> None:
        path = filedialog.askopenfilename(
            title='Open MIDI File',
            filetypes=[('MIDI files', '*.mid *.midi'), ('All files', '*.*')],
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str) -> None:
        self.events    = load_midi_events(path)
        self._filepath = path
        self._beat_map = build_beat_map(path)

        self._filename_var.set(Path(path).name)

        self._rebuild_listbox()

        self._clear_beat_display()
        self._update_position_display(0)
        self._set_status('■  STOPPED', FG_DIM)

        if self.engine is not None:
            self.engine.load(path)

    def _on_space(self) -> None:
        if self.engine and self.engine.is_playing:
            self._on_stop()
        else:
            self._on_start()

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
        self.listbox.selection_clear(0, tk.END)
        self._clear_beat_display()
        self._update_position_display(0)
        self._set_status('■  STOPPED', FG_DIM)

    def _on_listbox_key_move(self, delta: int) -> None:
        sel = self.listbox.curselection()
        cur = sel[0] if sel else -1
        nxt = max(0, min(cur + delta, self.listbox.size() - 1))
        if nxt == cur:
            return
        self._programmatic_select = True
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(nxt)
        self.listbox.see(nxt)
        self._programmatic_select = False
        if self._listbox_to_event:
            event_index = self._listbox_to_event[nxt]
            if self.engine:
                self.engine.seek(event_index)
            self._update_position_display(event_index)
            if self._beat_map is not None:
                self._update_beat_display(self.events[event_index].abs_time_sec)

    def _on_listbox_select(self, _event: tk.Event) -> None:
        if self._programmatic_select:
            return
        sel = self.listbox.curselection()
        if not sel or not self.events:
            return
        event_index = self._listbox_to_event[sel[0]]
        if self.engine:
            self.engine.seek(event_index)
        self._update_position_display(event_index)
        if self._beat_map is not None:
            self._update_beat_display(self.events[event_index].abs_time_sec)

    def _on_mute_toggle(self, channel: int) -> None:
        if self.engine:
            self.engine.set_channel_mute(channel, self.channel_mute_vars[channel].get())
        self._rebuild_listbox()

    def _on_msg_type_toggle(self) -> None:
        self._rebuild_listbox()
        self._save_settings()

    # ── Playback callbacks (playback thread → main thread) ───────────────────

    def _cb_event_played(self, ev: PlaybackEvent) -> None:
        self.root.after(0, lambda: self._apply_event_played(ev))

    def _apply_event_played(self, ev: PlaybackEvent) -> None:
        self._highlight_row(ev.index)
        self._update_position_display(ev.index)
        self._update_beat_display(ev.abs_time_sec)

    def _cb_playback_finished(self) -> None:
        self.root.after(0, self._apply_playback_finished)

    def _apply_playback_finished(self) -> None:
        self._pos_var.set('再生終了 — RESET で先頭に戻る')
        self._set_status('■  FINISHED', ACCENT_RED)
        self._clear_beat_display()

    # ── Public helpers (CLI) ──────────────────────────────────────────────────

    def set_port(self, port_name: str) -> None:
        if port_name in mido.get_output_names():
            self._port_var.set(port_name)

    def load_file(self, filepath: str) -> None:
        self._load_file(filepath)

    # ── Close ─────────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        if self.engine:
            self.engine.stop()
            self.engine.close()
        self._save_settings()
        self.root.destroy()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='SMF Player')
    parser.add_argument('--port', type=str, default=None)
    parser.add_argument('--file', type=str, default=None)
    args = parser.parse_args()

    root = tk.Tk()
    app  = SmfPlayerApp(root)

    if args.port:
        app.set_port(args.port)
    if args.file:
        root.after(100, lambda: app.load_file(args.file))

    root.mainloop()


if __name__ == '__main__':
    main()
