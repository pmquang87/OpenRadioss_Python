"""Gurson porous metal ductile failure model (/FAIL/GURSON).

Fortran origin:
- ``engine/source/materials/mat/mat104/mat104_ldam_newton.F``
- ``sigeps104.F``, ``sigeps80.F``, ``sigeps52.F``
- ``starter/source/materials/fail/gurson/hm_read_fail_gurson.F``
- CFG: ``radioss2022/FAIL/fail_gurson.cfg``

Theory (Gurson 1977, Chu & Needleman 1980, Tvergaard & Needleman 1984)
----------------------------------------------------------------------
Micromechanical ductile fracture criterion based on the evolution of void
volume fraction f through void nucleation, void growth, shear deformation,
and void coalescence:

1. Void growth rate:
       df_growth = (1 - f) * tr(d_eps_p)
   From Gurson dilatational normality rule:
       tr(d_eps_p) = 3 * q_1 * q_2 * f * sinh(1.5 * q_2 * eta) * d_epsp
   where eta = P / sigma_vm is the stress triaxiality (P = hydrostatic mean
   stress, sigma_vm = von Mises equivalent stress). Under tension (eta > 0),
   voids expand; under compression (eta <= 0), void expansion is inhibited.

2. Void nucleation:
   - Strain-controlled Gaussian distribution (Chu & Needleman 1980):
       A_N = [ f_N / (s_N * sqrt(2 * pi)) ] * exp(-0.5 * ((eps_p - eps_N) / s_N)^2)
       df_nucl = A_N * d_epsp
     where eps_N is mean nucleation strain, s_N is standard deviation, and
     f_N is volume fraction of nucleating void particles.
   - Strain-controlled linear nucleation (if A_s > 0 is provided):
       When eps_p >= eps_N:
           df_nucl = A_s * d_epsp                          (for eta >= 0)
           df_nucl = A_s * max(0, 1 + 3 * eta) * d_epsp     (for -1/3 <= eta < 0)

3. Shear damage extension (Nahshon & Hutchinson 2008):
   Accounts for void distortion and shearing at low triaxiality:
       xi = (27 / 2) * J_3 / sigma_vm^3   (normalized Lode parameter)
       omega = max(0, 1 - xi^2)
       df_shear = k_w * omega * f * d_epsp

4. Void coalescence (Tvergaard-Needleman accelerated function):
   Past the critical void volume fraction f_c, void interaction and ligament
   necking accelerate effective void growth up to failure fraction f_F (or f_r):
       f*(f) = f                                              if f <= f_c
       f*(f) = f_c + [ (1/q_1 - f_c) / (f_F - f_c) ] * (f - f_c)   if f > f_c
   where f_u* = 1 / q_1 represents complete loss of stress carrying capacity.

5. Failure condition and element deletion:
   Element / integration point breaks when void volume fraction f reaches
   the rupture fraction f_F (or f* >= 1/q_1, or damage D = f / f_F >= 1.0).
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-20


def _invariants_3d(sxx, syy, szz, sxy, syz, szx):
    """Compute hydrostatic pressure, von Mises stress, triaxiality, and Lode parameter xi."""
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

    eta = p_mean / np.maximum(_TINY, svm)
    # Normalized Lode parameter xi in [-1, 1]
    svm_cubed = np.maximum(_TINY, svm**3)
    xi = np.clip(13.5 * j3 / svm_cubed, -1.0, 1.0)
    return p_mean, svm, eta, xi


def _invariants_shell(sxx, syy, sxy):
    """Compute invariants for plane-stress shell layer."""
    p_mean = (sxx + syy) / 3.0
    dev_xx = sxx - p_mean
    dev_yy = syy - p_mean
    dev_zz = -p_mean

    j2 = 0.5 * (dev_xx**2 + dev_yy**2 + dev_zz**2) + sxy**2
    svm = np.sqrt(np.maximum(_TINY, 3.0 * j2))

    j3 = dev_xx * dev_yy * dev_zz - dev_zz * (sxy**2)
    eta = p_mean / np.maximum(_TINY, svm)
    svm_cubed = np.maximum(_TINY, svm**3)
    xi = np.clip(13.5 * j3 / svm_cubed, -1.0, 1.0)
    return p_mean, svm, eta, xi


def gaussian_void_nucleation(
    epsp: np.ndarray,
    d_epsp: np.ndarray,
    eps_n: float,
    s_n: float,
    f_n: float,
) -> np.ndarray:
    """Compute Gaussian strain-controlled void nucleation increment (Chu & Needleman 1980).

    df_nucl = [ f_N / (s_N * sqrt(2 * pi)) ] * exp(-0.5 * ((eps_p - eps_N) / s_N)^2) * d_epsp

    Parameters
    ----------
    epsp : ndarray
        Current accumulated equivalent plastic strain.
    d_epsp : ndarray
        Increment of equivalent plastic strain.
    eps_n : float
        Mean nucleation plastic strain.
    s_n : float
        Standard deviation of nucleation plastic strain.
    f_n : float
        Nucleation void volume fraction amplitude.

    Returns
    -------
    ndarray
        Increment of void volume fraction due to nucleation.
    """
    if f_n <= 0.0 or s_n <= 0.0:
        return np.zeros_like(epsp)

    prefactor = f_n / (s_n * np.sqrt(2.0 * np.pi))
    arg = np.clip(-0.5 * (((epsp - eps_n) / s_n) ** 2), -50.0, 0.0)
    a_n = prefactor * np.exp(arg)
    return a_n * d_epsp


def tvergaard_coalescence(f: np.ndarray, fc: float, fr: float, q1: float = 1.5) -> np.ndarray:
    """Compute Tvergaard-Needleman accelerated void volume fraction function f*(f).

    f*(f) = f                                              if f <= fc
    f*(f) = fc + [ (1/q_1 - fc) / (fr - fc) ] * (f - fc)   if f > fc

    Parameters
    ----------
    f : ndarray
        Void volume fraction.
    fc : float
        Critical void volume fraction.
    fr : float
        Failure / rupture void volume fraction.
    q1 : float, optional
        First Gurson parameter (default 1.5).

    Returns
    -------
    ndarray
        Effective void fraction f*.
    """
    f_star = np.copy(f)
    fu_star = 1.0 / max(0.1, q1)
    coalesce_mask = f > fc
    if np.any(coalesce_mask):
        slope = (fu_star - fc) / max(1e-6, fr - fc)
        f_star[coalesce_mask] = fc + slope * (f[coalesce_mask] - fc)
    return f_star


def _extract_params(fail) -> dict[str, float | int]:
    """Extract and sanitize Gurson parameters from fail card."""
    p = fail.params if hasattr(fail, "params") else {}
    q1 = float(p.get("q1", p.get("FAIL_q1", 1.5)))
    q2 = float(p.get("q2", p.get("FAIL_q2", 1.0)))
    fc = float(p.get("fc", p.get("f_c", p.get("FAIL_Fc", 0.15))))
    fr = float(p.get("fr", p.get("f_r", p.get("f_u", p.get("f_f", p.get("FAIL_Fr", 0.25))))))
    if fr <= fc:
        fr = fc + 0.10
    f0 = float(p.get("f0", p.get("f_0", p.get("FAIL_F0", 0.0))))

    # Nucleation parameters
    eps_n = float(p.get("eps_n", p.get("epn", p.get("FAIL_eps_strain", 0.0))))
    s_n = float(p.get("s_n", p.get("sn", 0.1)))
    f_n = float(p.get("f_n", p.get("fn", 0.0)))
    a_s = float(p.get("a_s", p.get("as_", p.get("as", p.get("FAIL_As", 0.0)))))
    k_w = float(p.get("k_w", p.get("kw", p.get("FAIL_Kw", 0.0))))

    return {
        "q1": q1,
        "q2": q2,
        "fc": fc,
        "fr": fr,
        "f0": f0,
        "eps_n": eps_n,
        "s_n": s_n,
        "f_n": f_n,
        "a_s": a_s,
        "k_w": k_w,
    }


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance Gurson void damage for a 3D solid slice.

    Parameters
    ----------
    fail : Failure object or FailureModel
    sig : ndarray of shape (m, 6)
        Stress tensor [s_xx, s_yy, s_zz, s_xy, s_yz, s_zx].
    d_epsp : ndarray of shape (m,)
        Equivalent plastic strain increment.
    deps : ndarray of shape (m, 6) or None
        Total strain increment tensor.
    dt : float
        Time step size.
    dama : ndarray of shape (m,)
        Persistent accumulated damage array (D = f / f_r, updated in-place).
    tstar : optional

    Returns
    -------
    ndarray of bool
        Broken element mask (dama >= 1.0 or f >= fr).
    """
    p = _extract_params(fail)
    q1, q2 = p["q1"], p["q2"]
    fc, fr, f0 = p["fc"], p["fr"], p["f0"]
    eps_n, s_n, f_n = p["eps_n"], p["s_n"], p["f_n"]
    a_s, k_w = p["a_s"], p["k_w"]

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    szz = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)
    sxy = sig_arr[:, 3] if sig_arr.shape[1] > 3 else np.zeros_like(sxx)
    syz = sig_arr[:, 4] if sig_arr.shape[1] > 4 else np.zeros_like(sxx)
    szx = sig_arr[:, 5] if sig_arr.shape[1] > 5 else np.zeros_like(sxx)

    d_epsp_arr = np.asarray(d_epsp, dtype=float) if d_epsp is not None else np.zeros_like(sxx)

    # Invariants
    _, _, eta, xi = _invariants_3d(sxx, syy, szz, sxy, syz, szx)

    # Recover current void volume fraction f from dama: dama = f / fr
    f_curr = np.clip(dama * fr, f0, fr)

    # 1. Void growth rate: df_growth = (1 - f) * tr(d_eps_p)
    # Via Gurson dilatational normality rule:
    # tr(d_eps_p) = 3 * q1 * q2 * f * sinh(1.5 * q2 * eta) * d_epsp
    # Under hydrostatic tension (eta > 0), voids grow. Under compression (eta <= 0),
    # void growth is 0.
    sinh_arg = np.clip(1.5 * q2 * eta, -50.0, 50.0)
    # Ensure baseline growth even if initial f is small
    f_eff = np.maximum(f_curr, max(f0, 0.01))
    dilat = 3.0 * q1 * q2 * f_eff * np.sinh(sinh_arg)
    df_growth = (1.0 - f_curr) * np.maximum(0.0, dilat) * d_epsp_arr

    # If volumetric plastic strain increment is directly provided in deps
    if deps is not None and getattr(fail, "use_trace_deps", False):
        deps_arr = np.asarray(deps, dtype=float)
        tr_deps = deps_arr[:, 0] + deps_arr[:, 1] + (deps_arr[:, 2] if deps_arr.shape[1] > 2 else 0.0)
        df_growth = np.maximum(df_growth, (1.0 - f_curr) * np.maximum(0.0, tr_deps))

    # 2. Void nucleation
    df_nucl = np.zeros_like(f_curr)
    # Gaussian nucleation (Chu & Needleman 1980)
    if f_n > 0.0 and s_n > 0.0:
        # Approximate current accumulated plastic strain from d_epsp and dama
        epsp_est = d_epsp_arr  # or cumulative
        df_nucl += gaussian_void_nucleation(epsp_est, d_epsp_arr, eps_n, s_n, f_n)
    # Linear nucleation (A_s)
    if a_s > 0.0:
        pos_eta = eta >= 0.0
        mid_eta = (eta < 0.0) & (eta >= -1.0 / 3.0)
        df_nucl[pos_eta] += a_s * d_epsp_arr[pos_eta]
        df_nucl[mid_eta] += a_s * np.maximum(0.0, 1.0 + 3.0 * eta[mid_eta]) * d_epsp_arr[mid_eta]

    # 3. Shear damage extension (Nahshon & Hutchinson 2008)
    df_shear = np.zeros_like(f_curr)
    if k_w > 0.0:
        omega = np.maximum(0.0, 1.0 - xi**2)
        df_shear = k_w * omega * f_curr * d_epsp_arr

    # Total void volume fraction increment
    df_tot = np.where(f_curr < fr, df_growth + df_nucl + df_shear, 0.0)
    f_new = np.clip(f_curr + df_tot, 0.0, fr)

    # 4. Void coalescence acceleration (Tvergaard & Needleman 1984)
    f_star = tvergaard_coalescence(f_new, fc, fr, q1=q1)
    fu_star = 1.0 / max(0.1, q1)

    # Update damage variable D = f_new / fr (or accelerated coalescence damage)
    d_target = np.maximum(f_new / fr, np.where(f_new > fc, (f_star - fc) / max(1e-6, fu_star - fc), f_new / fr))
    dama[:] = np.clip(np.maximum(dama, d_target), 0.0, 1.0)

    # Broken mask: critical void coalescence reaching fracture limit
    broken = (f_new >= fr) | (f_star >= fu_star) | (dama >= 1.0)
    return broken


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance Gurson void damage for a plane-stress shell layer.

    Parameters
    ----------
    fail : Failure object or FailureModel
    sig : ndarray of shape (m, 3) or (m, 5)
        Plane stress tensor [s_xx, s_yy, s_xy, (s_yz, s_zx)].
    d_epsp : ndarray of shape (m,)
    deps : ndarray
    dt : float
    dama : ndarray of shape (m,)
        Persistent accumulated damage array (updated in-place).
    tstar : optional
    eps_tot : optional

    Returns
    -------
    ndarray of bool
        Broken point mask in this layer.
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
