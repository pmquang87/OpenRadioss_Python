"""
Lemaitre continuum damage mechanics model (/FAIL/LEMAITRE).

Fortran origin: ``starter/source/materials/fail/lemaitre/hm_read_fail_lemaitre.F90``,
``engine/source/materials/fail/lemaitre/fail_lemaitre_s.F90`` (solids) and
``engine/source/materials/fail/lemaitre/fail_lemaitre_c.F90`` (shells).
Failure model IRUPT = 50.

Theory (J. Lemaitre, A Course on Damage Mechanics, Springer, 1996):
------------------------------------------------------------------
Damage begins when plastic strain exceeds threshold eps_d:
  eps_p >= eps_d

Strain energy release rate Y:
  Y = (sigma_vm**2 * R_v) / (2 * E)
where:
  R_v = (2/3)*(1 + nu) + 3*(1 - 2*nu)*eta**2
  eta = p / sigma_vm (triaxiality)

Damage increment:
  dD = (Y / S) * d_epsp    (for sigma_1 > 0)
  D += dD, clamped to D <= D_c

Failure condition:
  D >= D_c  (default D_c = 1.0)
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
    """Advance Lemaitre damage for a solid element slice; returns broken mask."""
    p = fail.params
    eps_d = _get_param(p, ["fail_epsd", "eps_d", "epsd", "EPSD"], 0.0)
    s_param = _get_param(p, ["fail_s", "s", "S"], 1.0e9)
    d_c = _get_param(p, ["fail_dc", "d_c", "dc", "DC"], 1.0)
    e_young = _get_param(p, ["young", "E", "e"], 2.1e11)
    nu = _get_param(p, ["nu", "NU", "poisson"], 0.3)

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    szz = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)
    sxy = sig_arr[:, 3] if sig_arr.shape[1] > 3 else np.zeros_like(sxx)
    syz = sig_arr[:, 4] if sig_arr.shape[1] > 4 else np.zeros_like(sxx)
    szx = sig_arr[:, 5] if sig_arr.shape[1] > 5 else np.zeros_like(sxx)

    pressure = (sxx + syy + szz) / 3.0
    s_dev_xx = sxx - pressure
    s_dev_yy = syy - pressure
    s_dev_zz = szz - pressure
    von_mises = np.sqrt(1.5 * (s_dev_xx**2 + s_dev_yy**2 + s_dev_zz**2
                               + 2.0 * (sxy**2 + syz**2 + szx**2)))

    eta = pressure / np.maximum(von_mises, _TINY)
    r_v = (2.0 / 3.0) * (1.0 + nu) + 3.0 * (1.0 - 2.0 * nu) * (eta ** 2)
    y_energy = (von_mises ** 2 * r_v) / (2.0 * max(e_young, _TINY))

    d_epsp_arr = np.asarray(d_epsp, dtype=float)
    d_dama = np.where(d_epsp_arr > 0, (y_energy / max(s_param, _TINY)) * d_epsp_arr, 0.0)

    dama[:] = np.minimum(d_c, dama + d_dama)
    return dama >= d_c


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance Lemaitre damage for a shell layer; returns broken mask."""
    p = fail.params
    eps_d = _get_param(p, ["fail_epsd", "eps_d", "epsd", "EPSD"], 0.0)
    s_param = _get_param(p, ["fail_s", "s", "S"], 1.0e9)
    d_c = _get_param(p, ["fail_dc", "d_c", "dc", "DC"], 1.0)
    e_young = _get_param(p, ["young", "E", "e"], 2.1e11)
    nu = _get_param(p, ["nu", "NU", "poisson"], 0.3)

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)

    pressure = (sxx + syy) / 3.0
    von_mises = np.sqrt(sxx**2 + syy**2 - sxx * syy + 3.0 * sxy**2)

    eta = pressure / np.maximum(von_mises, _TINY)
    r_v = (2.0 / 3.0) * (1.0 + nu) + 3.0 * (1.0 - 2.0 * nu) * (eta ** 2)
    y_energy = (von_mises ** 2 * r_v) / (2.0 * max(e_young, _TINY))

    d_epsp_arr = np.asarray(d_epsp, dtype=float)
    d_dama = np.where(d_epsp_arr > 0, (y_energy / max(s_param, _TINY)) * d_epsp_arr, 0.0)

    dama[:] = np.minimum(d_c, dama + d_dama)
    return dama >= d_c
