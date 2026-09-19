"""
Extended Mohr-Coulomb Ductile Fracture Criterion (/FAIL/EMC).

Fortran origin: ``starter/source/materials/fail/emc/hm_read_fail_emc.F``,
``engine/source/materials/fail/emc/fail_emc.F`` (solids).
Failure model IRUPT = 27.

Theory (Bai and Wierzbicki, IJSS 2010):
---------------------------------------
Stress-state dependent fracture strain formulated using Hosford yield surface
and Mohr-Coulomb friction criterion:

Parameters:
  a_EMC     : Hosford exponent a (default 1.0)
  b0        : Fracture strain under uniaxial tension
  c         : Friction coefficient
  n_EMC     : Sensitivity exponent n
  Gamma     : Strain rate sensitivity
  EPS_DOT_0 : Reference strain rate

Normalized Lode angle parameter:
  theta_bar = 1 - (2/pi) * arccos(xi)

Fracture strain:
  eps_f = b(eps_dot) * (1 + c)**(1/n) * [ (1/2 * (df12**a + df23**a + df13**a))**(1/a) + c*(2*eta + f1 + f3) ]**(-1/n)

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
    """Advance EMC failure for a solid element slice; returns broken mask."""
    p = fail.params
    a = _get_param(p, ["a_emc", "a_EMC", "a", "A"], 1.0)
    b0 = _get_param(p, ["b0", "B0"], 1.0)
    c = _get_param(p, ["c", "C"], 0.0)
    n = _get_param(p, ["n_emc", "n_EMC", "n", "N"], 1.0)
    gamma = _get_param(p, ["gamma", "Gamma"], 0.0)
    eps_dot_0 = _get_param(p, ["eps_dot_0", "Epsilon_Dot_0", "eps0"], 1e-5)

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

    # Determinant of deviatoric stress
    det_s = (s_dev_xx * s_dev_yy * s_dev_zz + 2.0 * sxy * syz * szx
             - s_dev_xx * (syz**2) - s_dev_yy * (szx**2) - s_dev_zz * (sxy**2))
    xi = np.clip(13.5 * det_s / np.maximum(von_mises**3, _TINY), -1.0, 1.0)
    theta_bar = 1.0 - (2.0 / np.pi) * np.arccos(xi)

    f1 = (2.0 / 3.0) * np.cos((1.0 - theta_bar) * (np.pi / 6.0))
    f2 = (2.0 / 3.0) * np.cos((3.0 + theta_bar) * (np.pi / 6.0))
    f3 = -(2.0 / 3.0) * np.cos((1.0 + theta_bar) * (np.pi / 6.0))

    df12 = np.maximum(f1 - f2, _TINY)
    df23 = np.maximum(f2 - f3, _TINY)
    df13 = np.maximum(f1 - f3, _TINY)

    bracket = (0.5 * (df12**a + df23**a + df13**a)) ** (1.0 / a) + c * (2.0 * eta + f1 + f3)
    bracket = np.maximum(bracket, _TINY)

    rate = np.maximum(np.asarray(d_epsp, dtype=float) / max(dt, _TINY), eps_dot_0)
    b_rate = b0 * (1.0 + gamma * np.log(rate / eps_dot_0))

    eps_f = np.where(eta < -1.0 / 3.0, 100.0, b_rate * ((1.0 + c) ** (1.0 / n)) * (bracket ** (-1.0 / n)))
    eps_f = np.maximum(eps_f, _TINY)

    d_dama = np.asarray(d_epsp, dtype=float) / eps_f
    dama[:] = np.minimum(1.0, dama + d_dama)
    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """EMC shell step compatibility."""
    sig_arr = np.asarray(sig, dtype=float)
    n = len(sig_arr)
    sig_3d = np.zeros((n, 6), dtype=float)
    sig_3d[:, 0] = sig_arr[:, 0]
    sig_3d[:, 1] = sig_arr[:, 1]
    if sig_arr.shape[1] > 2:
        sig_3d[:, 3] = sig_arr[:, 2]
    return solid_step(fail, sig_3d, d_epsp, deps, dt, dama, tstar=tstar)
