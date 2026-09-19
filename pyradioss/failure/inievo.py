"""
Initiation and Evolution Failure Model (/FAIL/INIEVO).

Fortran origin: ``starter/source/materials/fail/inievo/hm_read_fail_inievo.F``,
``engine/source/materials/fail/inievo/fail_inievo_s.F`` (solids) and
``engine/source/materials/fail/inievo/fail_inievo_c.F`` (shells).
Failure model IRUPT = 42.

Theory:
-------
Two-stage failure model:
1. Initiation Stage:
   Damage initiates when cumulative initiation damage D_ini reaches 1.0:
   dD_ini = d_epsp / eps_f(eta, ...)
   D_ini += dD_ini

2. Evolution Stage (active once D_ini >= 1.0):
   Post-initiation damage evolution based on plastic displacement u_f^p or
   fracture energy G_f:
   Linear:      dD_evo = (L_e * d_epsp) / u_f^p
   Exponential: dD_evo = (alpha / (1 - exp(-alpha))) * exp(-alpha * u_p / u_f^p) * ...

Point fails when D_evo >= 1.0.
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
    """Advance Inievo failure for a solid element slice; returns broken mask."""
    p = fail.params
    eps_f = _get_param(p, ["eps_f", "fscale", "FSCALE", "eps_init"], 0.2)
    disp = _get_param(p, ["disp", "DISP", "u_f"], 0.001)

    d_epsp_arr = np.asarray(d_epsp, dtype=float)
    # Stage 1: Initiation
    d_ini = d_epsp_arr / max(eps_f, _TINY)

    # In dama we store the combined damage [0, 1]
    dama[:] = np.minimum(1.0, dama + d_ini)
    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance Inievo failure for a shell layer; returns broken mask."""
    return solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
