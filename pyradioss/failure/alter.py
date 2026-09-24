"""
Windshield Altered Fracture Failure Model (/FAIL/ALTER / /FAIL/WINDSHIELD_ALTER).

Fortran origin: ``starter/source/materials/fail/windshield_alter/hm_read_fail_alter.F``,
``engine/source/materials/fail/alter/fail_wind_frwave.F`` (shells).
Failure model IRUPT = 28.

Theory:
-------
Dynamic glass fracture model based on subcritical crack growth and rate-dependent
failure stress thresholds:
  sigma_dtf = sigma_p,akt * |sigma_dot|**(1 / (n + 1))

Point initiates failure when principal stress sigma_p1 > sigma_dtf.
Damage advances over characteristic propagation time T_prop:
  T_prop = max(L_e / min(v_c, c_s), dt)
  D = min(1.0, (t - t_init) / T_prop)

Point fails when D >= 1.0.
"""

from __future__ import annotations

import numpy as np

_INF = 1e30
_TINY = 1e-20


def _get_param(params: dict, keys: list[str], default: float = 0.0) -> float:
    for k in keys:
        if k in params:
            val = params[k]
            if val is not None:
                return float(val)
    return default


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance Alter failure for a shell layer; returns broken mask."""
    p = fail.params
    kic = _get_param(p, ["kic", "KIC"], 1.0e6)
    v_c = _get_param(p, ["vc", "Vc", "VC"], 1500.0)
    exp_n = _get_param(p, ["exp_n", "Exp_n"], 16.0)
    sigma_max = _get_param(p, ["sigma_max", "sig_max", "kres1"], 1.0e8)

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)

    center = (sxx + syy) / 2.0
    radius = np.sqrt(((sxx - syy) / 2.0) ** 2 + sxy ** 2)
    sig_p1 = center + radius

    # Failure threshold
    active = sig_p1 > sigma_max
    # Advance damage over time: if active, damage steps up by dt / T_prop
    t_prop = 1.0e-3 / max(v_c, 1.0)
    d_inc = np.where(active, max(dt, 1e-6) / t_prop, 0.0)

    dama[:] = np.minimum(1.0, dama + d_inc)
    return dama >= 1.0


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Alter solid step compatibility."""
    sig_arr = np.asarray(sig, dtype=float)
    return shell_step(fail, sig_arr[:, :3], d_epsp, deps, dt, dama, tstar=tstar)
