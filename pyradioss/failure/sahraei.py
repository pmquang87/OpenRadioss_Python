"""
Sahraei Battery Separator Failure Model (/FAIL/SAHRAEI).

Fortran origin: ``starter/source/materials/fail/sahraei/hm_read_fail_sahraei.F``,
``engine/source/materials/fail/sahraei/fail_sahraei_s.F`` (solids).
Failure model IRUPT = 29.

Theory (Sahraei & Wierzbicki, MIT):
-----------------------------------
Failure criterion combining volumetric strain activation trigger with
strain-ratio-dependent envelope and in-plane compressive strain limits:

Trigger:
  |e_vol| >= Vol_strain

Ratio:
  r = |numerator / denominator|
  eps_lim = f_ratio(r)

Failure condition:
  eps_ordi >= eps_lim  or  compressive strain exceeded.
  D = 1.0.
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


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance Sahraei failure for a solid element slice; returns broken mask."""
    p = fail.params
    vol_strain_lim = _get_param(p, ["vol_strain", "Vol_strain", "vol_limit"], 0.05)
    eps_lim = _get_param(p, ["eps_lim", "f_ratio", "eps_max"], 0.2)
    max_comp_strain = _get_param(p, ["max_comp_strain", "Max_Comp_Strain"], _INF)

    deps_arr = np.asarray(deps, dtype=float) if deps is not None else np.zeros((len(dama), 6))
    exx = deps_arr[:, 0]
    eyy = deps_arr[:, 1]
    ezz = deps_arr[:, 2] if deps_arr.shape[1] > 2 else np.zeros_like(exx)

    # Volumetric strain
    e_vol = np.abs(exx + eyy + ezz)
    active = e_vol >= vol_strain_lim

    max_tens_strain = np.maximum.reduce([exx, eyy, ezz])
    tens_break = active & (max_tens_strain >= max(eps_lim, _TINY))
    comp_break = (exx < -max_comp_strain) | (eyy < -max_comp_strain) | (ezz < -max_comp_strain)

    broken = tens_break | comp_break
    dama[:] = np.where(broken, 1.0, np.maximum(dama, max_tens_strain / max(eps_lim, _TINY)))
    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Sahraei shell step compatibility."""
    strain_arr = eps_tot if eps_tot is not None else deps
    deps_arr = np.asarray(strain_arr, dtype=float) if strain_arr is not None else np.zeros((len(dama), 3))
    n = len(deps_arr)
    deps_3d = np.zeros((n, 6), dtype=float)
    deps_3d[:, 0] = deps_arr[:, 0]
    deps_3d[:, 1] = deps_arr[:, 1]
    return solid_step(fail, sig, d_epsp, deps_3d, dt, dama, tstar=tstar)
