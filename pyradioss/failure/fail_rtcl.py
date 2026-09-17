"""Rice-Tracey and Cockcroft-Latham ductile failure model (/FAIL/RTCL).

Fortran origin:
- ``engine/source/materials/fail/rtcl/fail_rtcl_s.F`` (solids)
- ``engine/source/materials/fail/rtcl/fail_rtcl_c.F`` (shells)
- ``starter/source/materials/fail/rtcl/hm_read_fail_rtcl.F`` (starter reader)
- CFG: ``radioss2022/FAIL/fail_rtcl.cfg``

Theory
------
Dual-regime ductile fracture criterion combining void growth and shear failure:
1. High stress triaxiality regime (eta >= 1/3):
   Rice-Tracey void growth criterion dominates:
       F_RTCL(eta) = exp(-0.5) * exp(1.5 * eta)
   where eta = P / sigma_vm is stress triaxiality (P = hydrostatic mean stress,
   sigma_vm = von Mises equivalent stress). At eta = 1/3 (uniaxial tension),
   F_RTCL(1/3) = exp(-0.5 + 0.5) = 1.0.

2. Low stress triaxiality regime (-1/3 <= eta < 1/3):
   Cockcroft-Latham shear fracture criterion dominates:
       F_RTCL(eta) = 2 * [ (1 + eta * sqrt(12 - 27 * eta^2)) /
                           (3 * eta + sqrt(12 - 27 * eta^2)) ]
   In plane stress, this analytical term exactly equals sigma_1 / sigma_vm
   (ratio of maximum principal stress to von Mises stress), normalized so that
   at uniaxial tension (eta = 1/3), F_RTCL = 1.0, matching the Rice-Tracey
   regime continuously.

3. Compressive cut-off regime (eta < -1/3):
   Under triaxial compression below -1/3, void growth is suppressed and
   shear cracking is closed:
       F_RTCL(eta) = 0.0
   At eta = -1/3, F_RTCL continuously approaches 0.0:
       2 * (1 - (1/3) * 3) / (-1 + 3) = 0.0.

4. Mesh sensitivity regularization for shells (Inst = 2):
   EPS_CR = N + (EPSCAL - N) * (thk0 / l_elem)
   where l_elem = sqrt(Area). If Inst != 2: EPS_CR = EPSCAL.

5. Damage accumulation:
   dD = F_RTCL * d_epsp / max(EPS_CR, 1e-6)
   D = min(1.0, D + dD)
   Element / point is broken when D >= 1.0.
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-20


def rtcl_transition_function(eta: np.ndarray | float) -> np.ndarray:
    """Evaluate the smooth RTCL dual-regime transition function F_RTCL(eta).

    Parameters
    ----------
    eta : ndarray or float
        Stress triaxiality P / sigma_vm.

    Returns
    -------
    ndarray
        Damage weighting factor F_RTCL >= 0.
    """
    eta_arr = np.asarray(eta, dtype=float)
    scalar_input = eta_arr.ndim == 0
    if scalar_input:
        eta_arr = np.atleast_1d(eta_arr)

    f_rtcl = np.zeros_like(eta_arr)

    # 1. High triaxiality: Rice-Tracey (eta >= 1/3)
    high_mask = eta_arr >= (1.0 / 3.0)
    if np.any(high_mask):
        f_rtcl[high_mask] = np.exp(-0.5) * np.exp(1.5 * eta_arr[high_mask])

    # 2. Low triaxiality: Cockcroft-Latham (-1/3 <= eta < 1/3)
    mid_mask = (eta_arr >= -1.0 / 3.0) & (eta_arr < 1.0 / 3.0)
    if np.any(mid_mask):
        e_mid = eta_arr[mid_mask]
        radicand = np.maximum(0.0, 12.0 - 27.0 * (e_mid**2))
        sqrt_term = np.sqrt(radicand)
        numer = 1.0 + e_mid * sqrt_term
        denom = np.maximum(_TINY, 3.0 * e_mid + sqrt_term)
        f_rtcl[mid_mask] = 2.0 * (numer / denom)

    # 3. Cut-off: eta < -1/3 -> f_rtcl remains 0.0

    if scalar_input:
        return f_rtcl[0]
    return f_rtcl


def _triaxiality_3d(sxx, syy, szz, sxy, syz, szx):
    """Compute hydrostatic pressure, von Mises stress, and triaxiality for 3D state."""
    p_mean = (sxx + syy + szz) / 3.0
    dev_xx = sxx - p_mean
    dev_yy = syy - p_mean
    dev_zz = szz - p_mean

    j2 = 0.5 * (dev_xx**2 + dev_yy**2 + dev_zz**2) + sxy**2 + syz**2 + szx**2
    svm = np.sqrt(np.maximum(_TINY, 3.0 * j2))

    triaxs = p_mean / np.maximum(_TINY, svm)
    # Fortran fail_rtcl_s.F lines 90-91: triaxiality clipped to [-1.0, 1.0]
    triaxs = np.clip(triaxs, -1.0, 1.0)
    return p_mean, svm, triaxs


def _triaxiality_shell(sxx, syy, sxy):
    """Compute hydrostatic pressure, von Mises stress, and triaxiality for plane stress."""
    p_mean = (sxx + syy) / 3.0
    svm = np.sqrt(np.maximum(_TINY, sxx**2 + syy**2 - sxx * syy + 3.0 * (sxy**2)))
    triaxs = p_mean / np.maximum(_TINY, svm)
    # Fortran fail_rtcl_c.F lines 87-88: triaxiality clipped to [-2/3, 2/3]
    triaxs = np.clip(triaxs, -2.0 / 3.0, 2.0 / 3.0)
    return p_mean, svm, triaxs


def _extract_params(fail) -> dict[str, float | int]:
    """Extract RTCL parameters from fail card."""
    p = fail.params if hasattr(fail, "params") else {}
    epscal = float(p.get("epscal", p.get("MAT_EPSCAL", p.get("eps_cr", 0.3))))
    if epscal == 0.0:
        epscal = 0.3
    inst = int(p.get("inst", p.get("Inst", 2)))
    if inst == 0:
        inst = 2
    n_exp = float(p.get("n", p.get("MAT_N", p.get("n_exp", 0.0))))
    return {"epscal": epscal, "inst": inst, "n": n_exp}


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance RTCL damage for a 3D solid slice.

    Parameters
    ----------
    fail : Failure object or FailureModel
    sig : ndarray of shape (m, 6)
        Stress tensor [s_xx, s_yy, s_zz, s_xy, s_yz, s_zx].
    d_epsp : ndarray of shape (m,)
        Equivalent plastic strain increment.
    deps : ndarray of shape (m, 6) or None
    dt : float
        Time step size.
    dama : ndarray of shape (m,)
        Persistent accumulated damage array (updated in-place).
    tstar : optional

    Returns
    -------
    ndarray of bool
        Broken element mask (dama >= 1.0).
    """
    p = _extract_params(fail)
    eps_cr = max(1e-6, p["epscal"])

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    szz = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)
    sxy = sig_arr[:, 3] if sig_arr.shape[1] > 3 else np.zeros_like(sxx)
    syz = sig_arr[:, 4] if sig_arr.shape[1] > 4 else np.zeros_like(sxx)
    szx = sig_arr[:, 5] if sig_arr.shape[1] > 5 else np.zeros_like(sxx)

    d_epsp_arr = np.asarray(d_epsp, dtype=float) if d_epsp is not None else np.zeros_like(sxx)

    # Compute stress triaxiality
    _, _, triaxs = _triaxiality_3d(sxx, syy, szz, sxy, syz, szx)

    # Evaluate RTCL transition function
    f_rtcl = rtcl_transition_function(triaxs)

    # Damage accumulation: dD = F_RTCL * d_epsp / eps_cr
    d_damage = f_rtcl * d_epsp_arr / eps_cr
    dama[:] = np.clip(dama + d_damage, 0.0, 1.0)

    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None, area=None, thk=None):
    """Advance RTCL damage for a plane-stress shell layer.

    Parameters
    ----------
    fail : Failure object or FailureModel
    sig : ndarray of shape (m, 3) or (m, 5)
        Plane stress tensor [s_xx, s_yy, s_xy, (s_yz, s_zx)].
    d_epsp : ndarray of shape (m,)
        Equivalent plastic strain increment.
    deps : ndarray
    dt : float
    dama : ndarray of shape (m,)
        Persistent accumulated damage array (updated in-place).
    tstar : optional
    eps_tot : optional
    area : ndarray or float, optional
        Element surface area for mesh regularization (Inst = 2).
    thk : ndarray or float, optional
        Element initial thickness for mesh regularization (Inst = 2).

    Returns
    -------
    ndarray of bool
        Broken point mask (dama >= 1.0).
    """
    p = _extract_params(fail)
    epscal = p["epscal"]
    inst = p["inst"]
    n_exp = p["n"]

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)

    d_epsp_arr = np.asarray(d_epsp, dtype=float) if d_epsp is not None else np.zeros_like(sxx)

    # Compute stress triaxiality
    _, _, triaxs = _triaxiality_shell(sxx, syy, sxy)

    # Evaluate RTCL transition function
    f_rtcl = rtcl_transition_function(triaxs)

    # Mesh regularization for shells (Inst = 2)
    if inst == 2 and area is not None and thk is not None:
        area_arr = np.asarray(area, dtype=float)
        thk_arr = np.asarray(thk, dtype=float)
        l_elem = np.sqrt(np.maximum(_TINY, area_arr))
        scale = np.clip(thk_arr / l_elem, 0.05, 5.0)
        eps_cr = np.maximum(1e-6, n_exp + (epscal - n_exp) * scale)
    else:
        eps_cr = max(1e-6, epscal)

    # Damage accumulation: dD = F_RTCL * d_epsp / eps_cr
    d_damage = f_rtcl * d_epsp_arr / eps_cr
    dama[:] = np.clip(dama + d_damage, 0.0, 1.0)

    return dama >= 1.0
