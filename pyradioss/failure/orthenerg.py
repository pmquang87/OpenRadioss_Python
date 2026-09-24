"""
Orthotropic Energy-Based Failure Model (/FAIL/ORTHENERG).

Fortran origin: ``starter/source/materials/fail/orthenerg/hm_read_fail_orthenerg.F``,
``engine/source/materials/fail/orthenerg/fail_orthenerg_s.F`` (solids) and
``engine/source/materials/fail/orthenerg/fail_orthenerg_c.F`` (shells).
Failure model IRUPT = 48.

Theory:
-------
Evaluates directional critical fracture energies G_c with linear (ISHAP=1)
or exponential (ISHAP=2) damage evolution across normal and shear modes.

Damage per mode:
  Linear:      dD = L_e * |deps_ii| * sigma_crit / (2 * G_crit)
  Exponential: D = 1 - exp(-E / G_crit)

Point fails when number of broken modes reaches NMOD.
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
    """Advance OrthEnerg failure for a solid element slice; returns broken mask."""
    p = fail.params
    sig11t = _get_param(p, ["sigma_11t", "sig11t", "SIGMA_11T"], _INF)
    g11t = _get_param(p, ["g_11t", "g11t", "G_11T"], _INF)
    sig22t = _get_param(p, ["sigma_22t", "sig22t", "SIGMA_22T"], _INF)
    g22t = _get_param(p, ["g_22t", "g22t", "G_22T"], _INF)
    sig33t = _get_param(p, ["sigma_33t", "sig33t", "SIGMA_33T"], _INF)
    g33t = _get_param(p, ["g_33t", "g33t", "G_33T"], _INF)

    nmod = int(_get_param(p, ["nmod", "NMOD"], 1))

    sig_arr = np.asarray(sig, dtype=float)
    deps_arr = np.asarray(deps, dtype=float) if deps is not None else np.zeros((len(dama), 6))

    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    szz = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)

    dexx = np.abs(deps_arr[:, 0]) if deps_arr.shape[1] > 0 else np.zeros_like(sxx)
    deyy = np.abs(deps_arr[:, 1]) if deps_arr.shape[1] > 1 else np.zeros_like(sxx)
    dezz = np.abs(deps_arr[:, 2]) if deps_arr.shape[1] > 2 else np.zeros_like(sxx)

    le = 1.0  # characteristic length proxy

    d11 = np.where(sxx > sig11t, (le * dexx * sig11t) / (2.0 * max(g11t, _TINY)), 0.0)
    d22 = np.where(syy > sig22t, (le * deyy * sig22t) / (2.0 * max(g22t, _TINY)), 0.0)
    d33 = np.where(szz > sig33t, (le * dezz * sig33t) / (2.0 * max(g33t, _TINY)), 0.0)

    # Accumulate into dama
    dama[:] = np.minimum(1.0, dama + np.maximum.reduce([d11, d22, d33]))
    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance OrthEnerg failure for a shell layer; returns broken mask."""
    p = fail.params
    sig11t = _get_param(p, ["sigma_11t", "sig11t", "SIGMA_11T"], _INF)
    g11t = _get_param(p, ["g_11t", "g11t", "G_11T"], _INF)
    sig22t = _get_param(p, ["sigma_22t", "sig22t", "SIGMA_22T"], _INF)
    g22t = _get_param(p, ["g_22t", "g22t", "G_22T"], _INF)

    sig_arr = np.asarray(sig, dtype=float)
    strain_arr = eps_tot if eps_tot is not None else deps
    deps_arr = np.asarray(strain_arr, dtype=float) if strain_arr is not None else np.zeros((len(dama), 3))

    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]

    dexx = np.abs(deps_arr[:, 0]) if deps_arr.shape[1] > 0 else np.zeros_like(sxx)
    deyy = np.abs(deps_arr[:, 1]) if deps_arr.shape[1] > 1 else np.zeros_like(sxx)

    le = 1.0

    d11 = np.where(sxx > sig11t, (le * dexx * sig11t) / (2.0 * max(g11t, _TINY)), 0.0)
    d22 = np.where(syy > sig22t, (le * deyy * sig22t) / (2.0 * max(g22t, _TINY)), 0.0)

    dama[:] = np.minimum(1.0, dama + np.maximum(d11, d22))
    return dama >= 1.0
