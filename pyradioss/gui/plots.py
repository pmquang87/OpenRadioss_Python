"""
Results plotting for the pyradioss GUI — optional matplotlib, text fallback.

No Fortran counterpart (see the package docstring). ``matplotlib`` is an
*optional* dependency: when it imports, :class:`ResultsView` embeds a live
:class:`~matplotlib.backends.backend_tkagg.FigureCanvasTkAgg` figure of the
selected T01 channels; when it is absent the same view degrades to a scrolled
textual channel table built by :func:`format_channel_table` (which is pure
text and needs neither matplotlib nor tkinter, so it is unit-testable).
"""

from __future__ import annotations

from typing import List, Optional

from .runner import T01Data

# Optional matplotlib — probed once at import. The base install stays
# stdlib-only; this simply flips the GUI between the embedded-figure and
# the textual-table presentation.
try:
    import matplotlib
    matplotlib.use("TkAgg")  # embed in Tk; harmless if a canvas is never made
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MPL = True
except Exception:  # ImportError, or a backend that cannot initialise
    HAS_MPL = False


# Channels the Results tab offers by default (present in every T01 global
# header: TIME,IE,KE,HE,CE,EN,DE,EW,ERR%,...).
DEFAULT_CHANNELS = ["IE", "KE", "EW", "ERR%"]


def available_channels(t01: T01Data) -> List[str]:
    """The plottable channels of a loaded T01 (every column except TIME)."""
    if not t01.columns:
        return []
    return [c for c in t01.columns[1:]]


def format_channel_table(t01: T01Data, channels: List[str],
                         max_rows: int = 200) -> str:
    """Render selected channels vs TIME as a fixed-width text table.

    The fallback for when matplotlib is unavailable. Down-samples to at most
    ``max_rows`` evenly-spaced rows so a long run stays readable.
    """
    if not t01.columns:
        return "(no data)"
    chans = [c for c in channels if t01.has(c)]
    if not chans:
        return "(no matching channels in T01)"
    header = ["TIME"] + chans
    widths = [max(12, len(h)) for h in header]
    lines = ["  ".join(h.rjust(w) for h, w in zip(header, widths))]
    lines.append("  ".join("-" * w for w in widths))

    n = t01.nrows
    if n == 0:
        lines.append("(no rows yet)")
        return "\n".join(lines)
    step = max(1, n // max_rows)
    time_series = t01.time
    series = [t01.column(c) for c in chans]
    for i in range(0, n, step):
        cells = [f"{time_series[i]:.5E}"]
        cells += [f"{s[i]:.5E}" for s in series]
        lines.append("  ".join(c.rjust(w) for c, w in zip(cells, widths)))
    return "\n".join(lines)


class ResultsView:
    """Owns the Results-tab plot area inside a parent Tk frame.

    Builds an embedded matplotlib canvas when :data:`HAS_MPL`, else a
    read-only scrolled text table. :meth:`update_plot` re-renders the given
    channels of a :class:`T01Data`.
    """

    def __init__(self, parent):
        # tkinter is imported lazily so this module stays importable head-less.
        import tkinter as tk
        from tkinter import scrolledtext

        self.parent = parent
        self.has_mpl = HAS_MPL
        self._t01: Optional[T01Data] = None

        if HAS_MPL:
            self.figure = Figure(figsize=(6.0, 4.0), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            self.text = scrolledtext.ScrolledText(
                parent, wrap=tk.NONE, font=("Courier New", 9))
            self.text.pack(fill=tk.BOTH, expand=True)
            self.text.insert(
                tk.END,
                "matplotlib not installed — showing a textual channel "
                "table.\nInstall matplotlib for embedded plots.\n")
            self.text.configure(state=tk.DISABLED)

    def update_plot(self, t01: T01Data, channels: List[str],
                    t_end: float = 0.0) -> None:
        self._t01 = t01
        if self.has_mpl:
            self._draw_figure(t01, channels, t_end)
        else:
            self._draw_table(t01, channels)

    # -- matplotlib path -----------------------------------------------
    def _draw_figure(self, t01: T01Data, channels: List[str],
                     t_end: float) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        time_series = t01.time
        plotted = 0
        for chan in channels:
            if not t01.has(chan):
                continue
            ax.plot(time_series, t01.column(chan), label=chan, linewidth=1.2)
            plotted += 1
        ax.set_xlabel("time")
        ax.set_ylabel("value")
        ax.set_title("Time history (T01)")
        if t_end > 0:
            ax.set_xlim(left=0.0, right=t_end)
        if plotted:
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "no channels selected",
                    ha="center", va="center", transform=ax.transAxes)
        self.figure.tight_layout()
        self.canvas.draw()

    # -- text fallback --------------------------------------------------
    def _draw_table(self, t01: T01Data, channels: List[str]) -> None:
        import tkinter as tk
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, format_channel_table(t01, channels))
        self.text.configure(state=tk.DISABLED)
