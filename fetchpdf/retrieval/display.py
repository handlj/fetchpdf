"""Where fetchpdf's headed browsers put their windows: hidden (default) or visible.

Several routes need a *headed* Chrome -- Cloudflare, the Atypon SI pages and
repository WAFs refuse headless ones -- yet a batch run must not open windows
on the user's desktop. In ``hidden`` mode every headed browser runs on one
private Xvfb display per process; in ``visible`` mode it uses the user's real
display (to watch or debug a route).

Hiding takes more than pointing ``DISPLAY`` at Xvfb. On a Wayland desktop,
current Chrome picks Wayland whenever ``WAYLAND_DISPLAY`` is set and ignores
``DISPLAY`` -- measured 2026-11-17 with Chrome 157: ``chrome://gpu`` showed
"Ozone platform: wayland" for a launch that had DISPLAY=:120, i.e. every
"hidden" window opened on the desktop. So the hidden environment also drops
``WAYLAND_DISPLAY`` and the launch forces ``--ozone-platform=x11``.

The mode is process state, like institutional access: set once from the CLI
(``--browser-windows``) or the environment (``FETCHPDF_BROWSER_WINDOWS``;
the older ``FETCHPDF_INSTITUTIONAL_VISIBLE=1`` means ``visible``). Nothing here
mutates ``os.environ``: batch workers are threads, and a display swapped
under one worker would move another's browser.

Interactive sign-in (``fetchpdf fetchcookies``, ``cookies window``) does not go
through this module: a person has to see those windows.
"""
from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import sys
import threading
import time
from typing import List, Optional

HIDDEN, VISIBLE = "hidden", "visible"
MODES = (HIDDEN, VISIBLE)

_mode_lock = threading.Lock()
_mode: Optional[str] = None            # None: not decided yet, read the environment

_xvfb_lock = threading.Lock()
_xvfb_display: Optional[str] = None
_xvfb_proc = None
_warned_no_xvfb = False


def _is_desktop_os() -> bool:
    """macOS / Windows: headed windows need no display and cannot be hidden here."""
    return os.name == "nt" or sys.platform == "darwin"


def set_browser_windows(mode: Optional[str]) -> None:
    """Choose ``hidden`` or ``visible`` for the rest of the process (``--browser-windows``)."""
    global _mode
    if mode is not None and mode not in MODES:
        raise ValueError(f"browser windows must be one of {MODES}, not {mode!r}")
    with _mode_lock:
        _mode = mode


def browser_windows() -> str:
    """The current mode: CLI choice, else FETCHPDF_BROWSER_WINDOWS, else hidden."""
    with _mode_lock:
        if _mode:
            return _mode
    env = (os.environ.get("FETCHPDF_BROWSER_WINDOWS") or "").strip().lower()
    if env in MODES:
        return env
    if os.environ.get("FETCHPDF_INSTITUTIONAL_VISIBLE"):
        return VISIBLE
    return HIDDEN


def hidden() -> bool:
    return browser_windows() == HIDDEN


def _private_xvfb() -> Optional[str]:
    """One shared private Xvfb display for this process, started on first use."""
    global _xvfb_display, _xvfb_proc
    with _xvfb_lock:
        if _xvfb_display and _xvfb_proc and _xvfb_proc.poll() is None:
            return _xvfb_display
        xvfb = shutil.which("Xvfb")
        if not xvfb:
            return None
        for n in range(120, 140):
            if os.path.exists(f"/tmp/.X{n}-lock"):
                continue
            try:
                proc = subprocess.Popen(
                    [xvfb, f":{n}", "-screen", "0", "1400x900x24", "-nolisten", "tcp"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                return None
            time.sleep(1.0)
            if proc.poll() is None:
                _xvfb_proc, _xvfb_display = proc, f":{n}"
                atexit.register(proc.terminate)
                return _xvfb_display
    return None


def _warn_no_xvfb() -> None:
    global _warned_no_xvfb
    with _xvfb_lock:
        if _warned_no_xvfb:
            return
        _warned_no_xvfb = True
    print("⚠️  Browser windows are hidden but Xvfb is not installed: the routes that need "
          "a browser are skipped to stay silent. Install it (sudo apt install xvfb) or "
          "run with --browser-windows visible.")


def browser_display() -> Optional[str]:
    """The X display a headed browser should use, ``""`` when none is needed, or None.

    None means no headed browser may run: hidden mode without Xvfb, or visible
    mode without any display.
    """
    if _is_desktop_os():
        return ""
    if hidden():
        display = _private_xvfb()
        if display is None:
            _warn_no_xvfb()
        return display
    # Visible: the user's own desktop, X11 or Wayland.
    if os.environ.get("DISPLAY"):
        return os.environ["DISPLAY"]
    if os.environ.get("WAYLAND_DISPLAY"):
        return ""
    return None


def browser_launch_env() -> Optional[dict]:
    """The environment for a headed launch, or None when none may run.

    Hidden: a copy of the environment on the private Xvfb, with every pointer
    to the user's real desktop removed. Visible: the environment as it is.
    """
    display = browser_display()
    if display is None:
        return None
    env = dict(os.environ)
    if _is_desktop_os():
        return env
    if hidden():
        env["DISPLAY"] = display
        env.pop("WAYLAND_DISPLAY", None)
        env["XDG_SESSION_TYPE"] = "x11"
    return env


def browser_launch_args(base_args: Optional[List[str]] = None) -> List[str]:
    """Chromium arguments for a headed launch; hidden mode pins it to X11."""
    args = list(base_args or [])
    if hidden() and not _is_desktop_os() and not any(a.startswith("--ozone-platform")
                                                     for a in args):
        args.append("--ozone-platform=x11")
    return args


def headed_browser_available() -> bool:
    """Whether a headed browser can run in the current mode (starts Xvfb if hidden)."""
    return browser_display() is not None
