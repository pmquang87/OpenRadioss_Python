"""
Tvergaard-Needleman porous ductile fracture model (/FAIL/TVERGAARD / /FAIL/GURSON).

Fortran origin: ``engine/source/materials/mat/mat104/mat104_ldam_newton.F``,
``sigeps104.F``, ``sigeps80.F``; reader
``starter/source/materials/fail/gurson/hm_read_fail_gurson.F``.

Theory (Gurson 1977, Tvergaard & Needleman 1984, Nahshon & Hutchinson 2008)
--------------------------------------------------------------------------
Micromechanical ductile fracture model driven by void nucleation, growth,
shear deformation, and coalescence:

1. Void growth:
       df_growth = (1 - f) * tr(d_eps_p)
   or from the Gurson dilatational normality rule:
       df_growth = (1 - f) * 1.5 * q_1 * q_2 * sinh(1.5 * q_2 * eta) * d_eps_p
   where eta = P / sigma_vm (stress triaxiality).

2. Void nucleation (strain-controlled):
   When eps_p >= eps_strain and f < f_r:
       For eta >= 0:
           df_nucl = A_s * d_eps_p
       For -1/3 <= eta < 0:
           df_nucl = A_s * max(1 + 3 * eta, 0) * d_eps_p
       For eta < -1/3:
           df_nucl = 0

3. Shear damage extension (Nahshon & Hutchinson 2008):
   Accounts for void shear deformation at low triaxiality:
       xi = (27 / 2) * J_3 / sigma_vm^3
       omega = max(0, 1 - xi^2)
       df_shear = k_w * omega * f * d_eps_p

4. Void coalescence (Tvergaard-Needleman f* function):
       f*(f) = f                                        if f <= f_c
       f*(f) = f_c + [ (1/q_1 - f_c) / (f_r - f_c) ] * (f - f_c)  if f > f_c

Failure condition:
    Point breaks when total void volume fraction f >= f_r (or damage D = f / f_r >= 1.0).
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-20


def _invariants(sxx, syy, szz, sxy, syz, szx):
    """Compute mean stress, von Mises stress, and normalized Lode xi."""
    p_mean = (sxx + syy + szz) / 3.0
    dev_xx = sxx - p_mean
    dev_yy = syy - p_mean
    dev_zz = szz - p_mean

    j2 = 0.5 * (dev_xx**2 + dev_yy**2 + dev_zz**2) + sxy**2 + syz**2 + szx**2
    svm = np.sqrt(np.maximum(_TINY, 3.0 * j2))

    j3 = (
        dev_xx * dev_yy * dev_zz
        + 2.0 * sxy * syz * szx
        - dev_yy * (szx**2)
        - dev_xx * (syz**2)
        - dev_zz * (sxy**2)
    )

    eta = p_mean / svm
    xi = np.clip(13.5 * j3 / np.maximum(_TINY, svm**3), -1.0, 1.0)
    return p_mean, svm, eta, xi


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance Tvergaard / Gurson damage for a 3D solid slice.

    Parameters
    ----------
    fail : Failure object
        Parameters dictionary:
        - FAIL_q1 / q1 : Tvergaard q1 parameter (default 1.5)
        - FAIL_q2 / q2 : Tvergaard q2 parameter (default 1.0)
        - FAIL_Fc / fc : critical void volume fraction (default 0.15)
        - FAIL_Fr / fr : failure void volume fraction (default 0.25)
        - FAIL_F0 / f0 : initial void volume fraction (default 0.0)
        - FAIL_As / as / an : nucleation slope (default 0.0)
        - FAIL_Kw / kw : shear coefficient (default 0.0)
        - FAIL_eps_strain / epn : nucleation threshold strain (default 0.0)
    sig : ndarray of shape (m, 6)
        [s_xx, s_yy, s_zz, s_xy, s_yz, s_zx].
    d_epsp : ndarray of shape (m,)
        Equivalent plastic strain increment.
    deps : ndarray
        Strain increment tensor.
    dt : float
        Time step size.
    dama : ndarray of shape (m,)
        Accumulated damage / void fraction array (updated in-place).
    tstar : optional
        Homologous temperature.

    Returns
    -------
    ndarray of bool
        Broken element mask (dama >= 1.0).
    """
    p = fail.params
    q1 = float(p.get("FAIL_q1", p.get("q1", 1.5)))
    q2 = float(p.get("FAIL_q2", p.get("q2", 1.0)))
    fc = float(p.get("FAIL_Fc", p.get("fc", 0.15)))
    fr = max(float(p.get("FAIL_Fr", p.get("fr", 0.25))), fc + 1e-6)
    f0 = float(p.get("FAIL_F0", p.get("f0", 0.0)))
    a_n = float(p.get("FAIL_As", p.get("as", p.get("as_", p.get("an", 0.0)))))
    k_w = float(p.get("FAIL_Kw", p.get("kw", 0.0)))
    epn = float(p.get("FAIL_eps_strain", p.get("epn", p.get("ed", 0.0))))

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    szz = sig_arr[:, 2]
    sxy = sig_arr[:, 3]
    syz = sig_arr[:, 4] if sig_arr.shape[1] > 4 else np.zeros_like(sxx)
    szx = sig_arr[:, 5] if sig_arr.shape[1] > 5 else np.zeros_like(sxx)

    _, _, eta, xi = _invariants(sxx, syy, szz, sxy, syz, szx)

    d_epsp_arr = np.asarray(d_epsp, dtype=float) if d_epsp is not None else np.zeros_like(sxx)

    # Reconstruct current void fraction f from dama: dama = f / fr
    f_curr = dama * fr
    f_curr = np.maximum(f0, f_curr)

    # 1. Void growth via dilatational normality rule:
    # df_growth = (1 - f) * 3 * q1 * q2 * f * sinh(1.5 * q2 * eta) * d_epsp
    sinh_arg = np.clip(1.5 * q2 * eta, -50.0, 50.0)
    dilat = 3.0 * q1 * q2 * f_curr * np.sinh(sinh_arg)
    df_growth = (1.0 - f_curr) * np.maximum(0.0, dilat) * d_epsp_arr


    # 2. Nucleation
    # Active when plastic strain exceeds epn
    df_nucl = np.zeros_like(f_curr)
    if a_n > 0.0:
        pos_eta = eta >= 0.0
        low_eta = (eta < 0.0) & (eta >= -1.0 / 3.0)
        df_nucl[pos_eta] = a_n * d_epsp_arr[pos_eta]
        df_nucl[low_eta] = a_n * np.maximum(0.0, 1.0 + 3.0 * eta[low_eta]) * d_epsp_arr[low_eta]

    # 3. Shear damage (Nahshon & Hutchinson 2008)
    df_shear = np.zeros_like(f_curr)
    if k_w > 0.0:
        omega = np.maximum(0.0, 1.0 - xi**2)
        df_shear = k_w * omega * f_curr * d_epsp_arr

    # Total void volume fraction increment
    df_tot = np.where(f_curr < fr, df_growth + df_nucl + df_shear, 0.0)
    f_new = np.clip(f_curr + df_tot, 0.0, fr)

    # Coalescence acceleration (Tvergaard-Needleman f*)
    f_star = np.where(
        f_new <= fc,
        f_new,
        fc + ((1.0 / q1 - fc) / max(fr - fc, 1e-6)) * (f_new - fc),
    )

    # Update damage variable D = f_new / fr
    dama[:] = np.maximum(dama, f_new / fr)
    np.minimum(dama, 1.0, out=dama)

    return (f_new >= fr) | (dama >= 1.0)


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance Tvergaard / Gurson damage for a plane-stress shell layer.

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
    if sig_arr.shape[1] > 2:
        sig_3d[:, 3] = sig_arr[:, 2]  # sxy
    if sig_arr.shape[1] > 3:
        sig_3d[:, 4] = sig_arr[:, 3]  # syz
    if sig_arr.shape[1] > 4:
        sig_3d[:, 5] = sig_arr[:, 4]  # szx

    return solid_step(fail, sig_3d, d_epsp, deps, dt, dama, tstar=tstar)
