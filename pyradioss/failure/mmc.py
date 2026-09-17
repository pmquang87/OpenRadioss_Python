"""
Modified Mohr-Coulomb ductile fracture criterion (/FAIL/MMC / /FAIL/WIERZBICKI).

Fortran origin: ``engine/source/materials/fail/wierzbicki/fail_wierzbicki_s.F`` (solids)
and ``fail_wierzbicki_c.F`` (shells); reader
``starter/source/materials/fail/wierzbicki/hm_read_fail_wierzbicki.F``.

Theory (Bai & Wierzbicki, Int. J. Solids Struct. 47 (2010) 1953-1970)
---------------------------------------------------------------------
Fracture strain as a function of both stress triaxiality (eta) and normalized
Lode parameter (xi):

    eta = P / sigma_vm
    xi  = (27 / 2) * J_3 / sigma_vm^3      (with -1 <= xi <= 1)

Fracture strain formula:
    Delta = (c_1 * exp(-c_2 * eta))^n - (c_3 * exp(-c_4 * eta))^n
    term  = Delta * [ max(0, 1 - |xi|^m) ]^(1 / m)
    eps_f = [ (c_1 * exp(-c_2 * eta))^n - term ]^(1 / n)

Damage accumulation with plastic strain:
    If eps_f > 0:
        dD = d_eps_p / eps_f
        D += dD

Failure condition:
    Point breaks when D >= 1.0.
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-20


def _invariants_3d(sxx, syy, szz, sxy, syz, szx):
    """Compute mean stress, von Mises stress, and normalized Lode xi."""
    p_mean = (sxx + syy + szz) / 3.0
    dev_xx = sxx - p_mean
    dev_yy = syy - p_mean
    dev_zz = szz - p_mean

    # J2 and von Mises
    j2 = 0.5 * (dev_xx**2 + dev_yy**2 + dev_zz**2) + sxy**2 + syz**2 + szx**2
    svm = np.sqrt(np.maximum(_TINY, 3.0 * j2))

    # J3 = det(s)
    j3 = (
        dev_xx * dev_yy * dev_zz
        + 2.0 * sxy * syz * szx
        - dev_yy * (szx**2)
        - dev_xx * (syz**2)
        - dev_zz * (sxy**2)
    )

    eta = p_mean / svm
    # Lode parameter xi = 13.5 * J3 / svm^3, clamped to [-1, 1]
    xi = np.clip(13.5 * j3 / np.maximum(_TINY, svm**3), -1.0, 1.0)

    return p_mean, svm, eta, xi


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance MMC / Wierzbicki damage for a 3D solid slice.

    Parameters
    ----------
    fail : Failure object
        Parameters dictionary:
        - c1, c2, c3, c4 : MMC envelope constants
        - m, cm          : Lode exponent (default 1.0)
        - n, cn          : hardening exponent (default 1.0)
    sig : ndarray of shape (m, 6)
        [s_xx, s_yy, s_zz, s_xy, s_yz, s_zx].
    d_epsp : ndarray of shape (m,)
        Equivalent plastic strain increment.
    deps : ndarray
        Total strain increment.
    dt : float
        Time step size.
    dama : ndarray of shape (m,)
        Accumulated damage (in-place).
    tstar : optional
        Homologous temperature.

    Returns
    -------
    ndarray of bool
        Broken element mask (dama >= 1.0).
    """
    p = fail.params
    c1 = float(p.get("c1", p.get("C1_WIERZBICKI", p.get("C1", 0.0))))
    c2 = float(p.get("c2", p.get("C2_WIERZBICKI", p.get("C2", 0.0))))
    c3 = float(p.get("c3", p.get("C3_WIERZBICKI", p.get("C3", 0.0))))
    c4 = float(p.get("c4", p.get("C4_WIERZBICKI", p.get("C4", 0.0))))
    cm = max(float(p.get("m", p.get("cm", 1.0))), 1e-4)
    cn = max(float(p.get("n", p.get("cn", p.get("n_WIERZBICKI", 1.0)))), 1e-4)

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    szz = sig_arr[:, 2]
    sxy = sig_arr[:, 3]
    syz = sig_arr[:, 4] if sig_arr.shape[1] > 4 else np.zeros_like(sxx)
    szx = sig_arr[:, 5] if sig_arr.shape[1] > 5 else np.zeros_like(sxx)

    _, _, eta, xi = _invariants_3d(sxx, syy, szz, sxy, syz, szx)

    # MMC fracture strain formula
    term1 = (c1 * np.exp(np.clip(-c2 * eta, -80.0, 80.0))) ** cn
    term2 = (c3 * np.exp(np.clip(-c4 * eta, -80.0, 80.0))) ** cn
    delta = term1 - term2

    lode_factor = np.maximum(0.0, 1.0 - np.abs(xi) ** cm) ** (1.0 / cm)
    inner = term1 - delta * lode_factor
    eps_f = np.where(inner > 0.0, inner ** (1.0 / cn), 0.0)

    d_epsp_arr = np.asarray(d_epsp, dtype=float) if d_epsp is not None else np.zeros_like(sxx)
    active = (eps_f > 0.0) & (d_epsp_arr > 0.0)
    dama += np.where(active, d_epsp_arr / np.maximum(_TINY, eps_f), 0.0)
    return dama >= 1.0





def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance MMC / Wierzbicki damage for a plane-stress shell layer.

    Parameters
    ----------
    fail : Failure object
    sig : ndarray of shape (m, 3) or (m, 5)
        Plane stress tensor [s_xx, s_yy, s_xy, (s_yz, s_zx)].
    d_epsp : ndarray of shape (m,)
    deps : ndarray
    dt : float
    dama : ndarray of shape (m,)
    tstar : optional
    eps_tot : optional

    Returns
    -------
    ndarray of bool
        Broken mask.
    """
    sig_arr = np.asarray(sig, dtype=float)
    n = len(sig_arr)
    sig_3d = np.zeros((n, 6), dtype=float)
    sig_3d[:, 0] = sig_arr[:, 0]
    sig_3d[:, 1] = sig_arr[:, 1]
    # szz = 0 in plane stress
    if sig_arr.shape[1] > 2:
        sig_3d[:, 3] = sig_arr[:, 2]  # sxy
    if sig_arr.shape[1] > 3:
        sig_3d[:, 4] = sig_arr[:, 3]  # syz
    if sig_arr.shape[1] > 4:
        sig_3d[:, 5] = sig_arr[:, 4]  # szx

    return solid_step(fail, sig_3d, d_epsp, deps, dt, dama, tstar=tstar)
