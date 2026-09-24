"""
Fabric Orthotropic Failure Model (/FAIL/FABRIC).

Fortran origin: ``starter/source/materials/fail/fabric/hm_read_fail_fabric.F``,
``engine/source/materials/fail/fabric/fail_fabric_c.F`` (shells).
Failure model IRUPT = 31.

Theory:
-------
Orthotropic fabric rupture model evaluated in yarn directions 1 and 2:
  eps_f1, eps_r1 : Initiation and rupture strains in direction 1
  eps_f2, eps_r2 : Initiation and rupture strains in direction 2
  NDIR           : 1 = either fiber breaks, 2 = both fibers break (default 2)

Directional damage:
  D_i = (eps_i - eps_fi) / (eps_ri - eps_fi)   for eps_i > eps_fi
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


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance Fabric failure for a shell layer; returns broken mask."""
    p = fail.params
    eps_f1 = _get_param(p, ["eps_f1", "Epsilon_f1", "epsf1"], _INF)
    eps_r1 = _get_param(p, ["eps_r1", "Epsilon_r1", "epsr1"], eps_f1 * 1.5 if eps_f1 < _INF else _INF)
    eps_f2 = _get_param(p, ["eps_f2", "Epsilon_f2", "epsf2"], _INF)
    eps_r2 = _get_param(p, ["eps_r2", "Epsilon_r2", "epsr2"], eps_f2 * 1.5 if eps_f2 < _INF else _INF)
    ndir = int(_get_param(p, ["ndir", "NDIR"], 2))

    strain_arr = eps_tot if eps_tot is not None else deps
    deps_arr = np.asarray(strain_arr, dtype=float) if strain_arr is not None else np.zeros((len(dama), 3))

    e1 = deps_arr[:, 0]
    e2 = deps_arr[:, 1]

    d1 = np.where(e1 <= eps_f1, 0.0,
                  np.where(e1 >= eps_r1, 1.0, (e1 - eps_f1) / max(eps_r1 - eps_f1, _TINY)))
    d2 = np.where(e2 <= eps_f2, 0.0,
                  np.where(e2 >= eps_r2, 1.0, (e2 - eps_f2) / max(eps_r2 - eps_f2, _TINY)))

    if ndir == 1:
        broken = (d1 >= 1.0) | (d2 >= 1.0)
        d_eff = np.maximum(d1, d2)
    else:
        broken = (d1 >= 1.0) & (d2 >= 1.0)
        d_eff = np.minimum(d1, d2)

    dama[:] = np.minimum(1.0, np.maximum(dama, d_eff))
    return broken


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Fabric solid step compatibility."""
    deps_arr = np.asarray(deps, dtype=float) if deps is not None else np.zeros((len(dama), 6))
    return shell_step(fail, sig, d_epsp, deps_arr[:, :3], dt, dama, tstar=tstar)
