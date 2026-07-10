"""
Piecewise-linear function tables — the /FUNCT keyword.

Fortran origin: ``common_source/tools/curve/`` (function storage) and the
``FINTER`` interpolation routine used by every material law and load that
takes a function ID (e.g. ``engine/source/tools/curve/finter.F``).

In a Radioss deck, `/FUNCT/<id>` defines a curve as (X, Y) pairs. Loads
(`/CLOAD`, `/IMPVEL`, `/GRAV`) reference the curve by ID and evaluate it at
the current time; material laws may use curves for hardening or rate effects.

Behaviour outside the definition interval matches Radioss: the curve is
extrapolated with the slope of the last (or first) segment. This matters:
e.g. an /IMPVEL curve defined only up to t=1 keeps its final slope, it does
NOT clamp — decks rely on defining a flat last segment when they want a
constant tail.
"""

from __future__ import annotations

import numpy as np


class FunctTable:
    """One /FUNCT curve: piecewise-linear y(x) with slope extrapolation."""

    def __init__(self, fct_id: int, x, y, title: str = ""):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size < 2:
            raise ValueError(f"/FUNCT/{fct_id}: needs at least 2 points")
        if np.any(np.diff(x) <= 0):
            raise ValueError(f"/FUNCT/{fct_id}: abscissae must be strictly increasing")
        self.id = fct_id
        self.title = title
        self.x = x
        self.y = y
        # Pre-computed segment slopes (dy/dx); used both for interpolation
        # and for the extrapolation beyond the ends.
        self.slope = np.diff(y) / np.diff(x)

    def eval(self, t):
        """Evaluate the curve at scalar or array abscissa ``t``.

        Equivalent of the Fortran FINTER function: linear interpolation
        inside [x0, xn], linear *extrapolation* with the end-segment slope
        outside.
        """
        t = np.asarray(t, dtype=float)
        # np.interp clamps outside the range; fix up the two ends by
        # extending with the first/last slopes afterwards.
        out = np.interp(t, self.x, self.y)
        below = t < self.x[0]
        above = t > self.x[-1]
        if np.any(below):
            out = np.where(below, self.y[0] + self.slope[0] * (t - self.x[0]), out)
        if np.any(above):
            out = np.where(above, self.y[-1] + self.slope[-1] * (t - self.x[-1]), out)
        return float(out) if out.ndim == 0 else out

    def __repr__(self):  # pragma: no cover - debug helper
        return f"FunctTable(id={self.id}, npoints={self.x.size}, title={self.title!r})"
