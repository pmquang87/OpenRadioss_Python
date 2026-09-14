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
        if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
            raise ValueError(f"/FUNCT/{fct_id}: x and y must be 1D arrays of identical length")
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

    def transform(self, scx: float, scy: float, shx: float, shy: float) -> None:
        """Apply a /MOVE_FUNCT scale and shift to the curve in place.
        If scx < 0, reverses the point order so abscissae stay strictly
        increasing (matching hm_read_move_funct.F)."""
        if abs(scx) < 1.0e-20:
            raise ValueError(f"/MOVE_FUNCT/{self.id}: X scale factor cannot be zero")
        if scx < 0.0:
            self.x = self.x[::-1] * scx + shx
            self.y = self.y[::-1] * scy + shy
        else:
            self.x = self.x * scx + shx
            self.y = self.y * scy + shy
        self.slope = np.diff(self.y) / np.diff(self.x)

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

    def __call__(self, t):
        return self.eval(t)

    def __repr__(self):  # pragma: no cover - debug helper
        return f"FunctTable(id={self.id}, npoints={self.x.size}, title={self.title!r})"


class SmoothFunctTable(FunctTable):
    """One /FUNCT_SMOOTH curve (M37).

    Fortran origin: ``starter/source/tools/curve/hm_read_funct.F``
    (ISMOOTH = 1 branch) + ``engine/source/tools/curve/finter_smooth.F``.
    The points are already scale/shift-transformed at READ time
    (x*Ascalex + Ashiftx, y*Fscaley + Fshifty — the reference transforms
    them before storage, so the stored table needs no further scaling).

    Evaluation differs from the linear /FUNCT in two ways, both matching
    FINTER_SMOOTH exactly:

    * inside each segment the quintic *smoothstep* Hermite polynomial
      interpolates:  y = y1 + (y2 - y1) * s^3 (10 - 15 s + 6 s^2) with
      s = (t - x1)/(x2 - x1)  — C2-continuous (zero first AND second
      derivative at each data point), which is why the option exists:
      ramps defined with few points drive loads without acceleration
      jumps;
    * outside [x0, xn] the curve is CLAMPED to the end ordinates (the
      linear /FUNCT extrapolates with the end slope instead).
    """

    def eval(self, t):
        t = np.asarray(t, dtype=float)
        # segment index: i such that x[i] <= t < x[i+1]
        i = np.clip(np.searchsorted(self.x, t, side="right") - 1,
                    0, self.x.size - 2)
        x1, x2 = self.x[i], self.x[i + 1]
        y1, y2 = self.y[i], self.y[i + 1]
        s = np.clip((t - x1) / (x2 - x1), 0.0, 1.0)
        out = y1 + (y2 - y1) * s ** 3 * (10.0 - 15.0 * s + 6.0 * s * s)
        # FINTER_SMOOTH clamps outside the definition interval
        out = np.where(t <= self.x[0], self.y[0], out)
        out = np.where(t >= self.x[-1], self.y[-1], out)
        return float(out) if out.ndim == 0 else out

    def __repr__(self):  # pragma: no cover - debug helper
        return (f"SmoothFunctTable(id={self.id}, npoints={self.x.size}, "
                f"title={self.title!r})")
