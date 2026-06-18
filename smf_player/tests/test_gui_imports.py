"""Tests: no PyQt dependency, Tkinter importable."""

from __future__ import annotations

import os


def test_no_pyqt_dependency():
    this_file = os.path.abspath(__file__)
    smf_dir   = os.path.join(os.path.dirname(__file__), '..')
    needle    = 'Py' + 'Qt'  # split so this file doesn't trigger itself
    for dirpath, _dirs, files in os.walk(smf_dir):
        for fname in files:
            if not fname.endswith('.py'):
                continue
            path = os.path.abspath(os.path.join(dirpath, fname))
            if path == this_file:
                continue
            with open(path, encoding='utf-8') as fp:
                content = fp.read()
            assert needle not in content, f'{path} still references PyQt'


def test_tkinter_importable():
    import tkinter as tk
    root = tk.Tk()
    root.destroy()
