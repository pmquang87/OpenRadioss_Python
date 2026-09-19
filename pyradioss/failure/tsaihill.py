"""
Tsai-Hill orthotropic failure criterion (/FAIL/TSAIHILL).

Fortran origin: ``starter/source/materials/fail/tsaihill/hm_read_fail_tsaihill.F``,
``engine/source/materials/fail/tsaihill/fail_tsaihill_s.F`` (solids) and
``engine/source/materials/fail/tsaihill/fail_tsaihill_c.F`` (shells).
Failure model IRUPT = 45.

Theory:
-------
Tsai-Hill criterion for orthotropic materials:

Parameters:
  X11     : Critical strength in direction 1 (X)
  X22     : Critical strength in direction 2 (Y)
  S12     : Critical shear strength (S)
  TAU_MAX : Relaxation time (default inf)
  FCUT    : Stress filter cutoff frequency

Shell formula:
  FINDEX = (sxx/X11)**2 - (sxx*syy)/(X11**2) + (syy/X22)**2 + (sxy/S12)**2

Solid formula (transversely isotropic in direction 1):
  FINDEX = (sxx/X11)**2 - (sxx*syy)/(X11**2) + (syy/X22)**2 + (sxy/S12)**2
         - (sxx*szz)/(X11**2) + (szz/X22)**2 + (szx/S12)**2

Reserve factor:
  R = 1 / sqrt(max(FINDEX, 1e-20))

Failure condition:
  D = min(1.0, max(D, FINDEX))
  Fails when D >= 1.0.
"""

from __future__ import annotations

import numpy as np

_INF = 1e30
_TINY = 1e-20


def _get_param(params: dict, keys: list[str], default: float = _INF) -> float:
    for k in keys:
        if k in params:
            val = params[k]
            if val is not None:
                return float(val)
    return default


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance Tsai-Hill failure for a solid element slice; returns broken mask."""
    p = fail.params
    x11 = max(_get_param(p, ["x11", "X11", "x", "X", "sigma_11"], _INF), _TINY)
    x22 = max(_get_param(p, ["x22", "X22", "y", "Y", "sigma_22"], _INF), _TINY)
    s12 = max(_get_param(p, ["s12", "S12", "s", "S", "sigma_12"], _INF), _TINY)

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    szz = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)
    sxy = sig_arr[:, 3] if sig_arr.shape[1] > 3 else np.zeros_like(sxx)
    szx = sig_arr[:, 5] if sig_arr.shape[1] > 5 else np.zeros_like(sxx)

    findex = ((sxx / x11) ** 2 - (sxx * syy) / (x11 ** 2) + (syy / x22) ** 2 + (sxy / s12) ** 2
              - (sxx * szz) / (x11 ** 2) + (szz / x22) ** 2 + (szx / s12) ** 2)
    findex = np.maximum(0.0, findex)

    dama[:] = np.minimum(1.0, np.maximum(dama, findex))
    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance Tsai-Hill failure for a shell layer; returns broken mask."""
    p = fail.params
    x11 = max(_get_param(p, ["x11", "X11", "x", "X", "sigma_11"], _INF), _TINY)
    x22 = max(_get_param(p, ["x22", "X22", "y", "Y", "sigma_22"], _INF), _TINY)
    s12 = max(_get_param(p, ["s12", "S12", "s", "S", "sigma_12"], _INF), _TINY)

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)

    findex = (sxx / x11) ** 2 - (sxx * syy) / (x11 ** 2) + (syy / x22) ** 2 + (sxy / s12) ** 2
    findex = np.maximum(0.0, findex)

    dama[:] = np.minimum(1.0, np.maximum(dama, findex))
    return dama >= 1.0
