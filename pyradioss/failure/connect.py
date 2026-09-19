"""
Connector Failure Model (/FAIL/CONNECT).

Fortran origin: ``starter/source/materials/fail/connect/hm_read_fail_connect.F``,
``engine/source/materials/fail/connect/fail_connect.F``, and ``suser43.F``.
Failure model IRUPT = 20.

Theory:
-------
Evaluates deformation and internal energy limits for connection elements:
  Normal strain:     eps_N = eps_zz
  Tangential strain: eps_T = sqrt(eps_zx**2 + eps_yz**2)

Damage criteria:
  C_strain = max(alpha_N * eps_N / eps_maxN, alpha_T * eps_T / eps_maxT)
  or multidirectional: |C_N|**exp_N + |C_T|**exp_T

Point fails when accumulated damage exceeds T_max (or immediate if T_max=0).
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
    """Advance Connect failure for a solid element slice; returns broken mask."""
    p = fail.params
    eps_max_n = _get_param(p, ["eps_maxn", "Epsilon_maxN", "maxdn", "MAXDN"], _INF)
    eps_max_t = _get_param(p, ["eps_maxt", "Epsilon_maxT", "maxdt", "MAXDT"], _INF)
    alpha_n = _get_param(p, ["alpha_n", "Alpha_N", "facn"], 1.0)
    alpha_t = _get_param(p, ["alpha_t", "Alpha_T", "fact"], 1.0)
    exp_n = _get_param(p, ["exp_n", "Exponent_N", "expn"], 1.0)
    exp_t = _get_param(p, ["exp_t", "Exponent_T", "expt"], 1.0)
    ifail = int(_get_param(p, ["ifail", "Ifail"], 0))

    deps_arr = np.asarray(deps, dtype=float) if deps is not None else np.zeros((len(dama), 6))
    ezz = deps_arr[:, 2] if deps_arr.shape[1] > 2 else np.zeros(len(dama))
    eyz = deps_arr[:, 4] if deps_arr.shape[1] > 4 else np.zeros(len(dama))
    ezx = deps_arr[:, 5] if deps_arr.shape[1] > 5 else np.zeros(len(dama))

    eps_n = np.abs(ezz)
    eps_t = np.sqrt(eyz**2 + ezx**2)

    c_n = alpha_n * (eps_n / max(eps_max_n, _TINY))
    c_t = alpha_t * (eps_t / max(eps_max_t, _TINY))

    if ifail == 1:
        c_tot = (c_n ** exp_n) + (c_t ** exp_t)
    else:
        c_tot = np.maximum(c_n, c_t)

    dama[:] = np.minimum(1.0, np.maximum(dama, c_tot))
    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Connect shell step compatibility."""
    deps_arr = np.asarray(deps, dtype=float) if deps is not None else np.zeros((len(dama), 3))
    n = len(deps_arr)
    deps_3d = np.zeros((n, 6), dtype=float)
    deps_3d[:, 0] = deps_arr[:, 0]
    deps_3d[:, 1] = deps_arr[:, 1]
    if deps_arr.shape[1] > 2:
        deps_3d[:, 3] = deps_arr[:, 2]
    return solid_step(fail, sig, d_epsp, deps_3d, dt, dama, tstar=tstar)
