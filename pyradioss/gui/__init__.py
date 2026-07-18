"""
pyradioss.gui — a run-and-monitor desktop GUI for the pyradioss port.

There is **no Fortran counterpart** to this package: OpenRadioss ships no
solver GUI (the reference ``OpenRadioss/openradioss_gui`` used for
inspiration is Altair's separate job launcher — a queue front-end around
the native ``starter``/``engine`` executables). This package is the
port-side equivalent: a small Tkinter front-end that wraps the two
console entry points ``pyradioss-starter`` / ``pyradioss-engine`` as
subprocesses, streams and parses their listings, monitors the run against
the ``/RUN`` end time, plots the T01 time-history and inspects a deck with
the port's own read-only deck reader.

Design rules (see the module docstrings):

* **stdlib only** in the base install — the GUI is built on ``tkinter``
  (Python standard library). ``matplotlib`` is *optional*: when it is
  importable the Results tab embeds a live figure, otherwise it degrades
  to a textual channel table. The base ``pyradioss`` install therefore
  grows no hard GUI dependency.
* the subprocess-facing logic (``runner.py``) is kept free of any
  ``tkinter`` import so it is fully unit-testable head-less, and so a
  scripted caller can drive a run without opening a window.

The console entry point is ``pyradioss-gui`` (``python -m pyradioss.gui``).
"""

from __future__ import annotations

__all__ = ["__version__"]

# Tracks the pyradioss package version this GUI ships with (informational).
__version__ = "0.1.0"
