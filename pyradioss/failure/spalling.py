"""
Spalling Dynamic Failure Model (/FAIL/SPALLING).

Fortran origin: ``starter/source/materials/fail/spalling/hm_read_fail_spalling.F90``,
``engine/source/materials/fail/spalling/fail_spalling_s.F90`` (solids).
Failure model IRUPT = 8.

Theory:
-------
Tensile cutoff pressure combined with Johnson-Cook ductile fracture for solids.

Parameters:
  D1, D2, D3, D4, D5 : Johnson-Cook parameters
  P_MIN              : Cutoff tensile pressure (negative in tension)
  EPS_DOT_0          : Reference strain rate
  IFAIL_SO           : Solid failure behavior option (1..6)

Spalling condition:
  p_spall = -1/3 * tr(sigma) <= P_min
  D_spall = min(1.0, max(D_spall, min(p_spall, 0) / P_min))

Failure condition:
  D = max(D_jc, D_spall) >= 1.0.
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-20


def _get_param(params: dict, keys: list[str], default: float = 0.0) -> float:
    for k in keys:
        if k in params:
            val = params[k]
            if val is not None:
                return float(val)
    return default


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance Spalling failure for a solid element slice; returns broken mask."""
    p = fail.params
    d1 = _get_param(p, ["d1", "D1"], 0.0)
    d2 = _get_param(p, ["d2", "D2"], 0.0)
    d3 = _get_param(p, ["d3", "D3"], 0.0)
    d4 = _get_param(p, ["d4", "D4"], 0.0)
    d5 = _get_param(p, ["d5", "D5"], 0.0)
    p_min = _get_param(p, ["p_min", "P_min", "pmin", "PMIN"], -1e20)
    if p_min > 0:
        p_min = -p_min  # convention: negative in tension

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    szz = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)
    sxy = sig_arr[:, 3] if sig_arr.shape[1] > 3 else np.zeros_like(sxx)
    syz = sig_arr[:, 4] if sig_arr.shape[1] > 4 else np.zeros_like(sxx)
    szx = sig_arr[:, 5] if sig_arr.shape[1] > 5 else np.zeros_like(sxx)

    # Pressure positive in compression, p_spall negative in tension
    p_hydro = (sxx + syy + szz) / 3.0
    p_spall = -p_hydro

    # Spalling damage
    d_spall = np.where(p_spall <= p_min, 1.0, np.where(p_spall < 0, p_spall / p_min, 0.0))

    # JC damage
    s_dev_xx = sxx - p_hydro
    s_dev_yy = syy - p_hydro
    s_dev_zz = szz - p_hydro
    von_mises = np.sqrt(1.5 * (s_dev_xx**2 + s_dev_yy**2 + s_dev_zz**2
                               + 2.0 * (sxy**2 + syz**2 + szx**2)))
    triax = p_hydro / np.maximum(von_mises, _TINY)

    eps_f = (d1 + d2 * np.exp(np.clip(d3 * triax, -30.0, 30.0)))
    if d1 == 0 and d2 == 0:
        d_jc = np.zeros_like(dama)
    else:
        d_jc = np.asarray(d_epsp, dtype=float) / np.maximum(eps_f, _TINY)

    dama[:] = np.minimum(1.0, np.maximum(dama + d_jc, d_spall))
    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Spalling is primarily for solids, but provide shell compatibility."""
    sig_arr = np.asarray(sig, dtype=float)
    n = len(sig_arr)
    sig_3d = np.zeros((n, 6), dtype=float)
    sig_3d[:, 0] = sig_arr[:, 0]
    sig_3d[:, 1] = sig_arr[:, 1]
    if sig_arr.shape[1] > 2:
        sig_3d[:, 3] = sig_arr[:, 2]
    return solid_step(fail, sig_3d, d_epsp, deps, dt, dama, tstar=tstar)
