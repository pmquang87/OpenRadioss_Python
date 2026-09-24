"""
Orthotropic Biquadratic Failure Model (/FAIL/ORTHBIQUAD).

Fortran origin: ``starter/source/materials/fail/orthbiquad/hm_read_fail_orthbiquad.F``,
``engine/source/materials/fail/orthbiquad/fail_orthbiquad_s.F`` (solids) and
``engine/source/materials/fail/orthbiquad/fail_orthbiquad_c.F`` (shells).
Failure model IRUPT = 38.

Theory:
-------
Biquadratic failure curve parameterized by stress triaxiality eta and in-plane loading
angle theta.

Parabolas:
  Low triaxiality (eta <= 1/3): through (-1/3, c1), (0, c2), (1/3, c3)
  High triaxiality (eta > 1/3): through (1/3, c3), (1/sqrt(3), c4), (2/3, c5)

Damage increment:
  dD = d_epsp / eps_fail(eta, theta)
  D += dD
Point fails when D >= 1.0.
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-20


def _fit_parabola(x1, y1, x2, y2, x3, y3):
    """Fit parabola y = a*x^2 + b*x + c through 3 points."""
    A = np.array([
        [x1**2, x1, 1.0],
        [x2**2, x2, 1.0],
        [x3**2, x3, 1.0]
    ], dtype=float)
    y = np.array([y1, y2, y3], dtype=float)
    try:
        return np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        return np.array([0.0, 0.0, max(y1, y2, y3)])


def _get_eps_fail(c1, c2, c3, c4, c5, eta):
    """Compute failure plastic strain for given triaxiality."""
    eta = np.clip(eta, -2.0 / 3.0, 2.0 / 3.0)
    p1 = _fit_parabola(-1.0 / 3.0, c1, 0.0, c2, 1.0 / 3.0, c3)
    p2 = _fit_parabola(1.0 / 3.0, c3, 1.0 / np.sqrt(3.0), c4, 2.0 / 3.0, c5)

    val_low = p1[0] * (eta ** 2) + p1[1] * eta + p1[2]
    val_high = p2[0] * (eta ** 2) + p2[1] * eta + p2[2]

    res = np.where(eta <= 1.0 / 3.0, val_low, val_high)
    return np.maximum(res, _TINY)


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance OrthBiquad failure for a solid element slice; returns broken mask."""
    p = fail.params
    c1 = float(p.get("c1", p.get("C1", 0.5)))
    c2 = float(p.get("c2", p.get("C2", 0.4)))
    c3 = float(p.get("c3", p.get("C3", 0.3)))
    c4 = float(p.get("c4", p.get("C4", 0.2)))
    c5 = float(p.get("c5", p.get("C5", 0.25)))

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

    eps_f = _get_eps_fail(c1, c2, c3, c4, c5, eta)
    d_dama = np.asarray(d_epsp, dtype=float) / eps_f
    dama[:] += d_dama

    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance OrthBiquad failure for a shell layer; returns broken mask."""
    p = fail.params
    c1 = float(p.get("c1", p.get("C1", 0.5)))
    c2 = float(p.get("c2", p.get("C2", 0.4)))
    c3 = float(p.get("c3", p.get("C3", 0.3)))
    c4 = float(p.get("c4", p.get("C4", 0.2)))
    c5 = float(p.get("c5", p.get("C5", 0.25)))

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)

    pressure = (sxx + syy) / 3.0
    von_mises = np.sqrt(sxx**2 + syy**2 - sxx * syy + 3.0 * sxy**2)
    eta = pressure / np.maximum(von_mises, _TINY)

    eps_f = _get_eps_fail(c1, c2, c3, c4, c5, eta)
    d_dama = np.asarray(d_epsp, dtype=float) / eps_f
    dama[:] += d_dama

    return dama >= 1.0
