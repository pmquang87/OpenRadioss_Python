"""
Maximum strain failure criterion (/FAIL/MAXSTRAIN).

Fortran origin: ``starter/source/materials/fail/max_strain/hm_read_fail_maxstrain.F``,
``engine/source/materials/fail/max_strain/fail_maxstrain_s.F`` (solids) and
``engine/source/materials/fail/max_strain/fail_maxstrain_c.F`` (shells).
Failure model IRUPT = 47.

Theory:
-------
Evaluates strain components against directional strain limits:

Parameters:
  EPS1_MAX   : Critical strain in direction 1 (eps1_max)
  EPS2_MAX   : Critical strain in direction 2 (eps2_max)
  GAM12_MAX  : Critical shear strain (gam12_max)
  TAU_MAX    : Relaxation time (default inf)
  FCUT       : Filter frequency

Shell:
  FINDEX = max(|eps_xx|/EPS1_MAX, |eps_yy|/EPS2_MAX, |eps_xy|/GAM12_MAX)

Solid (transversely isotropic in direction 1):
  FINDEX = max(|eps_xx|/EPS1_MAX, |eps_yy|/EPS2_MAX, |eps_xy|/GAM12_MAX,
               |eps_zz|/EPS2_MAX, |eps_zx|/GAM12_MAX)

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
    """Advance MaxStrain failure for a solid element slice; returns broken mask."""
    p = fail.params
    eps1_max = max(_get_param(p, ["eps1_max", "EPS1_MAX", "e1_max", "eps1"], _INF), _TINY)
    eps2_max = max(_get_param(p, ["eps2_max", "EPS2_MAX", "e2_max", "eps2"], _INF), _TINY)
    gam12_max = max(_get_param(p, ["gam12_max", "GAM12_MAX", "gamma12", "gam12"], _INF), _TINY)

    deps_arr = np.asarray(deps, dtype=float) if deps is not None else np.zeros((len(dama), 6))
    exx = np.abs(deps_arr[:, 0]) if deps_arr.shape[1] > 0 else np.zeros(len(dama))
    eyy = np.abs(deps_arr[:, 1]) if deps_arr.shape[1] > 1 else np.zeros(len(dama))
    ezz = np.abs(deps_arr[:, 2]) if deps_arr.shape[1] > 2 else np.zeros(len(dama))
    exy = np.abs(deps_arr[:, 3]) if deps_arr.shape[1] > 3 else np.zeros(len(dama))
    ezx = np.abs(deps_arr[:, 5]) if deps_arr.shape[1] > 5 else np.zeros(len(dama))

    findex = np.maximum.reduce([
        exx / eps1_max,
        eyy / eps2_max,
        ezz / eps2_max,
        exy / gam12_max,
        ezx / gam12_max,
    ])

    dama[:] = np.minimum(1.0, np.maximum(dama, findex))
    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance MaxStrain failure for a shell layer; returns broken mask."""
    p = fail.params
    eps1_max = max(_get_param(p, ["eps1_max", "EPS1_MAX", "e1_max", "eps1"], _INF), _TINY)
    eps2_max = max(_get_param(p, ["eps2_max", "EPS2_MAX", "e2_max", "eps2"], _INF), _TINY)
    gam12_max = max(_get_param(p, ["gam12_max", "GAM12_MAX", "gamma12", "gam12"], _INF), _TINY)

    # If total strain tensor provided, use it; otherwise use deps
    strain_arr = eps_tot if eps_tot is not None else deps
    strain_arr = np.asarray(strain_arr, dtype=float) if strain_arr is not None else np.zeros((len(dama), 3))

    exx = np.abs(strain_arr[:, 0]) if strain_arr.shape[1] > 0 else np.zeros(len(dama))
    eyy = np.abs(strain_arr[:, 1]) if strain_arr.shape[1] > 1 else np.zeros(len(dama))
    exy = np.abs(strain_arr[:, 2]) if strain_arr.shape[1] > 2 else np.zeros(len(dama))

    findex = np.maximum.reduce([
        exx / eps1_max,
        eyy / eps2_max,
        exy / gam12_max,
    ])

    dama[:] = np.minimum(1.0, np.maximum(dama, findex))
    return dama >= 1.0
