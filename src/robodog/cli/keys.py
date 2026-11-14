"""Keyboard e-stop: SPACE (or e) = emergency stop, q = quit. Works in a normal terminal."""

from __future__ import annotations

import select
import sys
import termios
import threading
import tty


class KeyWatcher:
    def __init__(self, on_estop, on_quit):
        self.on_estop = on_estop
        self.on_quit = on_quit
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._old = None

    def __enter__(self):
        if not sys.stdin.isatty():
            print("(stdin is not a terminal: keyboard e-stop disabled, use Ctrl+C)")
            return self
        fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._thread = threading.Thread(target=self._run, name="keywatcher", daemon=True)
        self._thread.start()
        print("keys: [SPACE] e-stop   [q] quit")
        return self

    def _run(self):
        while not self._stop.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not ready:
                continue
            ch = sys.stdin.read(1)
            if ch in (" ", "e", "E"):
                self.on_estop()
            elif ch in ("q", "Q"):
                self.on_quit()

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if self._old is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old)
        return False
