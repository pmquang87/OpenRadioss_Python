"""
Classic Tabulated Failure Model (/FAIL/TAB).

Fortran origin: ``starter/source/materials/fail/tabulated/hm_read_fail_tab_old.F``,
``engine/source/materials/fail/tabulated/fail_tab_old_s.F`` (solids) and
``engine/source/materials/fail/tabulated/fail_tab_old_c.F`` (shells).
Failure model IRUPT = 37.

Theory:
-------
Classic tabulated failure model where failure strain is defined as a function of
triaxiality eta and strain rate:
  dD = d_epsp / eps_f(eta, eps_dot)
  D += dD
Point fails when D >= D_crit.
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
    """Advance classic Tab failure for a solid element slice; returns broken mask."""
    p = fail.params
    eps_f = _get_param(p, ["eps_f", "EPS_F", "fscale", "fcrit"], 0.2)
    dcrit = _get_param(p, ["dcrit", "DCRIT", "d_crit"], 1.0)

    d_dama = np.asarray(d_epsp, dtype=float) / max(eps_f, _TINY)
    dama[:] = np.minimum(dcrit, dama + d_dama)
    return dama >= dcrit


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance classic Tab failure for a shell layer; returns broken mask."""
    return solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
