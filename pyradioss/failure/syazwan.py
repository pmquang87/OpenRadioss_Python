"""
Syazwan Ductile Fracture Model (/FAIL/SYAZWAN).

Fortran origin: ``starter/source/materials/fail/syazwan/hm_read_fail_syazwan.F``,
``engine/source/materials/fail/syazwan/fail_syazwan_s.F`` (solids) and
``engine/source/materials/fail/syazwan/fail_syazwan_c.F`` (shells).
Failure model IRUPT = 43.

Theory:
-------
Quadratic failure plastic strain envelope in stress triaxiality eta and
normalized Lode angle parameter theta_bar:
  eps_f(eta, theta_bar) = max(epfmin, C1 + C2*eta + C3*theta_bar + C4*eta**2 + C5*theta_bar**2 + C6*eta*theta_bar)

Damage accumulation:
  dD = d_epsp / eps_f
  D += dD
Point fails when D >= 1.0.
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
    """Advance Syazwan failure for a solid element slice; returns broken mask."""
    p = fail.params
    c1 = _get_param(p, ["c1", "C1"], 0.3)
    c2 = _get_param(p, ["c2", "C2"], -0.2)
    c3 = _get_param(p, ["c3", "C3"], 0.0)
    c4 = _get_param(p, ["c4", "C4"], 0.1)
    c5 = _get_param(p, ["c5", "C5"], 0.0)
    c6 = _get_param(p, ["c6", "C6"], 0.0)
    epfmin = _get_param(p, ["epfmin", "EPFMIN"], 0.01)

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
    eta = np.clip(pressure / np.maximum(von_mises, _TINY), -2.0 / 3.0, 2.0 / 3.0)

    det_s = (s_dev_xx * s_dev_yy * s_dev_zz + 2.0 * sxy * syz * szx
             - s_dev_xx * (syz**2) - s_dev_yy * (szx**2) - s_dev_zz * (sxy**2))
    cos3t = np.clip(13.5 * det_s / np.maximum(von_mises**3, _TINY), -1.0, 1.0)
    theta_bar = 1.0 - (2.0 / np.pi) * np.arccos(cos3t)

    eps_f = c1 + c2 * eta + c3 * theta_bar + c4 * (eta**2) + c5 * (theta_bar**2) + c6 * eta * theta_bar
    eps_f = np.maximum(eps_f, max(epfmin, _TINY))

    d_dama = np.asarray(d_epsp, dtype=float) / eps_f
    dama[:] = np.minimum(1.0, dama + d_dama)
    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance Syazwan failure for a shell layer; returns broken mask."""
    p = fail.params
    c1 = _get_param(p, ["c1", "C1"], 0.3)
    c2 = _get_param(p, ["c2", "C2"], -0.2)
    c3 = _get_param(p, ["c3", "C3"], 0.0)
    c4 = _get_param(p, ["c4", "C4"], 0.1)
    c5 = _get_param(p, ["c5", "C5"], 0.0)
    c6 = _get_param(p, ["c6", "C6"], 0.0)
    epfmin = _get_param(p, ["epfmin", "EPFMIN"], 0.01)

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)

    pressure = (sxx + syy) / 3.0
    von_mises = np.sqrt(sxx**2 + syy**2 - sxx * syy + 3.0 * sxy**2)
    eta = np.clip(pressure / np.maximum(von_mises, _TINY), -2.0 / 3.0, 2.0 / 3.0)

    # In plane stress, Lode angle directly depends on eta:
    cos3t = np.clip(-13.5 * eta * (eta**2 - 1.0 / 3.0), -1.0, 1.0)
    theta_bar = 1.0 - (2.0 / np.pi) * np.arccos(cos3t)

    eps_f = c1 + c2 * eta + c3 * theta_bar + c4 * (eta**2) + c5 * (theta_bar**2) + c6 * eta * theta_bar
    eps_f = np.maximum(eps_f, max(epfmin, _TINY))

    d_dama = np.asarray(d_epsp, dtype=float) / eps_f
    dama[:] = np.minimum(1.0, dama + d_dama)
    return dama >= 1.0
