"""
Command-line entry point:  pyradioss-gui [deck_0000.rad]

No Fortran counterpart (see the package docstring). Launches the Tkinter
run-and-monitor GUI; an optional positional argument pre-selects a starter
deck. ``--help`` works without a display (argparse handles it before any Tk
window is created), and a genuine "no display / no Tk" situation is reported
as a clean error rather than a traceback.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="pyradioss-gui",
        description="pyradioss run-and-monitor GUI - pick a starter deck, run "
                    "the Starter then the Engine, watch the listing, plot the "
                    "T01 time-history and inspect a deck.")
    ap.add_argument("deck", nargs="?", default=None,
                    help="optional starter deck to pre-select "
                         "(RunName_0000.rad)")
    ap.add_argument("--web", action="store_true", default=False,
                    help="launch the modern web UI instead of Tkinter")
    args = ap.parse_args(argv)

    if args.web:
        import webbrowser
        from .server import run_server
        # Open in default browser
        webbrowser.open('http://localhost:8080')
        run_server(port=8080)
        return 0

    # Import Tk-facing code only after argparse (so --help needs no display).
    try:
        from .app import launch
    except ImportError as exc:  # tkinter missing from this Python build
        print(f"pyradioss-gui: Tkinter is unavailable ({exc}); a GUI cannot "
              f"be started.", file=sys.stderr)
        return 1

    try:
        launch(deck=args.deck)
    except Exception as exc:  # e.g. TclError: no display available
        print(f"pyradioss-gui: could not open a window ({exc}).",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
