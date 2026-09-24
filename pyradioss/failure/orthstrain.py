"""
Orthotropic Strain Failure Model (/FAIL/ORTHSTRAIN).

Fortran origin: ``starter/source/materials/fail/orthstrain/hm_read_fail_orthstrain.F``,
``engine/source/materials/fail/orthstrain/fail_orthstrain_s.F`` (solids) and
``engine/source/materials/fail/orthstrain/fail_orthstrain_c.F`` (shells).
Failure model IRUPT = 24.

Theory:
-------
Evaluates up to 12 directional tensile/compressive/shear strain components against
strain-rate-dependent thresholds:
  11t, 22t, 12t, 11c, 22c, 12c, 33t, 33c, 23t, 23c, 31t, 31c

For each mode, damage evolves linearly between initiation strain eps_f and rupture strain eps_m:
  D_i = eps_m / |eps| * (|eps| - eps_f) / (eps_m - eps_f)   for |eps| > eps_f
  D_mode = max(D_mode, D_i) <= 1.0

Point fails when D_mode >= 1.0.
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


def _mode_damage(eps, eps_f, eps_m):
    abs_eps = np.abs(eps)
    active = abs_eps > eps_f
    d = np.zeros_like(abs_eps)
    denom = np.maximum(eps_m - eps_f, _TINY)
    d[active] = (eps_m / np.maximum(abs_eps[active], _TINY)) * ((abs_eps[active] - eps_f) / denom)
    return np.clip(d, 0.0, 1.0)


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance OrthStrain failure for a solid element slice; returns broken mask."""
    p = fail.params
    e11tf = _get_param(p, ["eps_11tf", "EPSILON_11TF", "e11t"], _INF)
    e11tm = _get_param(p, ["eps_11tm", "EPSILON_11TM", "e11tm"], e11tf * 1.2 if e11tf < _INF else _INF)
    e22tf = _get_param(p, ["eps_22tf", "EPSILON_22TF", "e22t"], _INF)
    e22tm = _get_param(p, ["eps_22tm", "EPSILON_22TM", "e22tm"], e22tf * 1.2 if e22tf < _INF else _INF)
    e33tf = _get_param(p, ["eps_33tf", "EPSILON_33TF", "e33t"], _INF)
    e33tm = _get_param(p, ["eps_33tm", "EPSILON_33TM", "e33tm"], e33tf * 1.2 if e33tf < _INF else _INF)

    e11cf = _get_param(p, ["eps_11cf", "EPSILON_11CF", "e11c"], e11tf)
    e11cm = _get_param(p, ["eps_11cm", "EPSILON_11CM", "e11cm"], e11cf * 1.2 if e11cf < _INF else _INF)
    e22cf = _get_param(p, ["eps_22cf", "EPSILON_22CF", "e22c"], e22tf)
    e22cm = _get_param(p, ["eps_22cm", "EPSILON_22CM", "e22cm"], e22cf * 1.2 if e22cf < _INF else _INF)
    e33cf = _get_param(p, ["eps_33cf", "EPSILON_33CF", "e33c"], e33tf)
    e33cm = _get_param(p, ["eps_33cm", "EPSILON_33CM", "e33cm"], e33cf * 1.2 if e33cf < _INF else _INF)

    e12tf = _get_param(p, ["eps_12tf", "EPSILON_12TF", "e12t"], _INF)
    e12tm = _get_param(p, ["eps_12tm", "EPSILON_12TM", "e12tm"], e12tf * 1.2 if e12tf < _INF else _INF)
    e23tf = _get_param(p, ["eps_23tf", "EPSILON_23TF", "e23t"], _INF)
    e23tm = _get_param(p, ["eps_23tm", "EPSILON_23TM", "e23tm"], e23tf * 1.2 if e23tf < _INF else _INF)
    e31tf = _get_param(p, ["eps_31tf", "EPSILON_31TF", "e31t"], _INF)
    e31tm = _get_param(p, ["eps_31tm", "EPSILON_31TM", "e31tm"], e31tf * 1.2 if e31tf < _INF else _INF)

    deps_arr = np.asarray(deps, dtype=float) if deps is not None else np.zeros((len(dama), 6))
    e11 = deps_arr[:, 0] if deps_arr.shape[1] > 0 else np.zeros(len(dama))
    e22 = deps_arr[:, 1] if deps_arr.shape[1] > 1 else np.zeros(len(dama))
    e33 = deps_arr[:, 2] if deps_arr.shape[1] > 2 else np.zeros(len(dama))
    e12 = deps_arr[:, 3] if deps_arr.shape[1] > 3 else np.zeros(len(dama))
    e23 = deps_arr[:, 4] if deps_arr.shape[1] > 4 else np.zeros(len(dama))
    e31 = deps_arr[:, 5] if deps_arr.shape[1] > 5 else np.zeros(len(dama))

    d11t = np.where(e11 >= 0, _mode_damage(e11, e11tf, e11tm), 0.0)
    d11c = np.where(e11 < 0, _mode_damage(e11, e11cf, e11cm), 0.0)
    d22t = np.where(e22 >= 0, _mode_damage(e22, e22tf, e22tm), 0.0)
    d22c = np.where(e22 < 0, _mode_damage(e22, e22cf, e22cm), 0.0)
    d33t = np.where(e33 >= 0, _mode_damage(e33, e33tf, e33tm), 0.0)
    d33c = np.where(e33 < 0, _mode_damage(e33, e33cf, e33cm), 0.0)

    d12 = _mode_damage(e12, e12tf, e12tm)
    d23 = _mode_damage(e23, e23tf, e23tm)
    d31 = _mode_damage(e31, e31tf, e31tm)

    findex = np.maximum.reduce([d11t, d11c, d22t, d22c, d33t, d33c, d12, d23, d31])
    dama[:] = np.minimum(1.0, np.maximum(dama, findex))
    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance OrthStrain failure for a shell layer; returns broken mask."""
    p = fail.params
    e11tf = _get_param(p, ["eps_11tf", "EPSILON_11TF", "e11t"], _INF)
    e11tm = _get_param(p, ["eps_11tm", "EPSILON_11TM", "e11tm"], e11tf * 1.2 if e11tf < _INF else _INF)
    e22tf = _get_param(p, ["eps_22tf", "EPSILON_22TF", "e22t"], _INF)
    e22tm = _get_param(p, ["eps_22tm", "EPSILON_22TM", "e22tm"], e22tf * 1.2 if e22tf < _INF else _INF)

    e11cf = _get_param(p, ["eps_11cf", "EPSILON_11CF", "e11c"], e11tf)
    e11cm = _get_param(p, ["eps_11cm", "EPSILON_11CM", "e11cm"], e11cf * 1.2 if e11cf < _INF else _INF)
    e22cf = _get_param(p, ["eps_22cf", "EPSILON_22CF", "e22c"], e22tf)
    e22cm = _get_param(p, ["eps_22cm", "EPSILON_22CM", "e22cm"], e22cf * 1.2 if e22cf < _INF else _INF)

    e12tf = _get_param(p, ["eps_12tf", "EPSILON_12TF", "e12t"], _INF)
    e12tm = _get_param(p, ["eps_12tm", "EPSILON_12TM", "e12tm"], e12tf * 1.2 if e12tf < _INF else _INF)

    strain_arr = eps_tot if eps_tot is not None else deps
    strain_arr = np.asarray(strain_arr, dtype=float) if strain_arr is not None else np.zeros((len(dama), 3))

    e11 = strain_arr[:, 0] if strain_arr.shape[1] > 0 else np.zeros(len(dama))
    e22 = strain_arr[:, 1] if strain_arr.shape[1] > 1 else np.zeros(len(dama))
    e12 = strain_arr[:, 2] if strain_arr.shape[1] > 2 else np.zeros(len(dama))

    d11t = np.where(e11 >= 0, _mode_damage(e11, e11tf, e11tm), 0.0)
    d11c = np.where(e11 < 0, _mode_damage(e11, e11cf, e11cm), 0.0)
    d22t = np.where(e22 >= 0, _mode_damage(e22, e22tf, e22tm), 0.0)
    d22c = np.where(e22 < 0, _mode_damage(e22, e22cf, e22cm), 0.0)
    d12 = _mode_damage(e12, e12tf, e12tm)

    findex = np.maximum.reduce([d11t, d11c, d22t, d22c, d12])
    dama[:] = np.minimum(1.0, np.maximum(dama, findex))
    return dama >= 1.0
