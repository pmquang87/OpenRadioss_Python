"""Puck 3D composite failure criterion (/FAIL/PUCK).

Fortran origin:
- ``engine/source/materials/fail/puck/fail_puck_s.F90`` (solids)
- ``engine/source/materials/fail/puck/fail_puck_c.F90`` (shells)
- ``starter/source/materials/fail/puck/hm_read_fail_puck.F90`` (reader)
- CFG: ``radioss2022/FAIL/fail_puck.cfg``

Theory (A. Puck, J. Kopp, Compos. Sci. Technol. 58 (1998) 1045-1067;
        A. Puck, H. Schürmann, Compos. Sci. Technol. 62 (2002) 1633-1662)
-------------------------------------------------------------------------
Physically based phenomenological 3D failure criteria distinguishing:
1. Fiber Failure (FF):
   - Longitudinal tension (sigma_11 >= 0):
       f_1 = sigma_11 / Sigma_1t
   - Longitudinal compression (sigma_11 < 0):
       f_1 = -sigma_11 / Sigma_1c

2. Inter-Fiber Failure (IFF) on fracture action plane:
   Transverse direction 2 (sigma_22, sigma_12):
   - Mode A (tensile transverse, sigma_22 >= 0):
       fac = (1 - p_12+ * Sigma_2t / Sigma_12) * (sigma_22 / Sigma_2t)
       f_a = sqrt((sigma_12 / Sigma_12)^2 + fac^2) + p_12+ * (sigma_22 / Sigma_12)
   - Mode B (moderate compressive transverse, sigma_22 < 0):
       f_b = [sqrt(sigma_12^2 + (p_12- * sigma_22)^2) + p_12- * sigma_22] / Sigma_12
       f_b = max(0, f_b)
   - Mode C (high compressive transverse shear, sigma_22 < 0):
       fac = 0.5 / ((1 + p_22-) * Sigma_12)
       f_c = (sigma_12 * fac)^2 + (sigma_22 / Sigma_2c)^2
       f_c = -f_c * Sigma_2c / min(sigma_22, -0.01 * Sigma_2c)

   Transverse direction 3 (solids, through-thickness sigma_33, sigma_31):
   - Mode A (sigma_33 >= 0):
       fac = (1 - p_12+ * Sigma_2t / Sigma_12) * (sigma_33 / Sigma_2t)
       f_a3 = sqrt((sigma_31 / Sigma_12)^2 + fac^2) + p_12+ * (sigma_33 / Sigma_12)
   - Mode B (sigma_33 < 0):
       f_b3 = [sqrt(sigma_31^2 + (p_12- * sigma_33)^2) + p_12- * sigma_33] / Sigma_12
       f_b3 = max(0, f_b3)
   - Mode C (sigma_33 < 0):
       fac = 0.5 / ((1 + p_22-) * Sigma_12)
       f_c3 = (sigma_31 * fac)^2 + (sigma_33 / Sigma_2c)^2
       f_c3 = -f_c3 * Sigma_2c / min(sigma_33, -0.01 * Sigma_2c)

Damage update:
    D = max(f_1, f_a, f_b, f_c [, f_a3, f_b3, f_c3])
    D_accum = max(D_prev, min(1.0, D))
    Failure when D_accum >= 1.0.
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-20
_INF = 1e20


def compute_puck_modes(sig: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute individual Puck failure modes (f1, fa, fb, fc) for in-plane stress states.

    Parameters
    ----------
    sig : ndarray of shape (m, 3) or (m, 6)
        Stress components [s_xx, s_yy, s_zz, s_xy, s_yz, s_zx] or [s_xx, s_yy, s_xy].
    params : dict
        Model parameters.

    Returns
    -------
    f1, fa, fb, fc : ndarray of shape (m,)
        Mode values for longitudinal fiber failure (f1), IFF Mode A (fa),
        IFF Mode B (fb), and IFF Mode C (fc).
    """
    sig_arr = np.asarray(sig, dtype=float)
    if sig_arr.ndim == 1:
        sig_arr = sig_arr.reshape(1, -1)

    s1t = max(float(params.get("sigma_1t", params.get("Sigma_1t", params.get("sigt1", _INF)))), _TINY)
    s2t = max(float(params.get("sigma_2t", params.get("Sigma_2t", params.get("sigt2", _INF)))), _TINY)
    s12 = max(float(params.get("sigma_12", params.get("Sigma_12", params.get("fsig12", params.get("sigt12", _INF))))), _TINY)
    s1c = max(float(params.get("sigma_1c", params.get("Sigma_1c", params.get("sigc1", _INF)))), _TINY)
    s2c = max(float(params.get("sigma_2c", params.get("Sigma_2c", params.get("sigc2", _INF)))), _TINY)

    pp12 = float(params.get("p12_pos", params.get("p12_Positive", params.get("pp12", 0.0))))
    pn12 = float(params.get("p12_neg", params.get("p12_Negative", params.get("pn12", 0.0))))
    pn22 = float(params.get("p22_neg", params.get("p22_Negative", params.get("pn22", 0.0))))

    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 3] if sig_arr.shape[1] > 3 else (sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx))

    # 1. Fiber failure (Longitudinal)
    f1 = np.where(sxx >= 0.0, sxx / s1t, -sxx / s1c)

    # 2. Inter-fiber failure (IFF)
    # Mode A (tensile transverse, syy >= 0)
    fac_a = (1.0 - pp12 * s2t / s12) * (syy / s2t)
    fa = np.where(syy >= 0.0, np.sqrt((sxy / s12) ** 2 + fac_a ** 2) + pp12 * (syy / s12), 0.0)

    # Mode B (moderate compressive transverse, syy < 0)
    fb_val = (np.sqrt(sxy ** 2 + (pn12 * syy) ** 2) + pn12 * syy) / s12
    fb = np.where(syy < 0.0, np.maximum(0.0, fb_val), 0.0)

    # Mode C (high compressive transverse shear, syy < 0)
    fac_c = 0.5 / ((1.0 + pn22) * s12)
    fc_bracket = (sxy * fac_c) ** 2 + (syy / s2c) ** 2
    denom = np.minimum(syy, -0.01 * s2c)
    fc_val = -fc_bracket * s2c / denom
    fc = np.where(syy < 0.0, fc_val, 0.0)

    return f1, fa, fb, fc


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance Puck composite failure criterion for a 3D solid slice.

    Parameters
    ----------
    fail : Failure object
        Must have `.params` dictionary with Puck parameters:
        - sigma_1t, sigma_2t, sigma_12, sigma_1c, sigma_2c
        - p12_pos, p12_neg, p22_neg
        - tau_max, fcut, ifail_so, ifail_sh
    sig : ndarray of shape (m, 6)
        Stress components [s_xx, s_yy, s_zz, s_xy, s_yz, s_zx].
    d_epsp : ndarray of shape (m,)
        Incremental plastic strain.
    deps : ndarray
        Strain increment.
    dt : float
        Time step.
    dama : ndarray of shape (m,)
        Persistent accumulated damage array.
    tstar : optional
        Homologous temperature (unused).

    Returns
    -------
    ndarray of bool
        Broken element mask (dama >= 1.0).
    """
    p = fail.params
    s1t = max(float(p.get("sigma_1t", p.get("Sigma_1t", p.get("sigt1", _INF)))), _TINY)
    s2t = max(float(p.get("sigma_2t", p.get("Sigma_2t", p.get("sigt2", _INF)))), _TINY)
    s12 = max(float(p.get("sigma_12", p.get("Sigma_12", p.get("fsig12", p.get("sigt12", _INF))))), _TINY)
    s1c = max(float(p.get("sigma_1c", p.get("Sigma_1c", p.get("sigc1", _INF)))), _TINY)
    s2c = max(float(p.get("sigma_2c", p.get("Sigma_2c", p.get("sigc2", _INF)))), _TINY)

    pp12 = float(p.get("p12_pos", p.get("p12_Positive", p.get("pp12", 0.0))))
    pn12 = float(p.get("p12_neg", p.get("p12_Negative", p.get("pn12", 0.0))))
    pn22 = float(p.get("p22_neg", p.get("p22_Negative", p.get("pn22", 0.0))))

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    szz = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)
    sxy = sig_arr[:, 3] if sig_arr.shape[1] > 3 else np.zeros_like(sxx)
    szx = sig_arr[:, 5] if sig_arr.shape[1] > 5 else np.zeros_like(sxx)

    # 1. Fiber failure (Longitudinal x)
    f1 = np.where(sxx >= 0.0, sxx / s1t, -sxx / s1c)

    # 2. Transverse direction 2 (syy, sxy)
    fac_a2 = (1.0 - pp12 * s2t / s12) * (syy / s2t)
    fa2 = np.where(syy >= 0.0, np.sqrt((sxy / s12) ** 2 + fac_a2 ** 2) + pp12 * (syy / s12), 0.0)

    fb2_val = (np.sqrt(sxy ** 2 + (pn12 * syy) ** 2) + pn12 * syy) / s12
    fb2 = np.where(syy < 0.0, np.maximum(0.0, fb2_val), 0.0)

    fac_c2 = 0.5 / ((1.0 + pn22) * s12)
    fc2_bracket = (sxy * fac_c2) ** 2 + (syy / s2c) ** 2
    denom2 = np.minimum(syy, -0.01 * s2c)
    fc2 = np.where(syy < 0.0, -fc2_bracket * s2c / denom2, 0.0)

    # 3. Transverse direction 3 (szz, szx) for 3D solids (fail_puck_s.F90 lines 206-225)
    fac_a3 = (1.0 - pp12 * s2t / s12) * (szz / s2t)
    fa3 = np.where(szz >= 0.0, np.sqrt((szx / s12) ** 2 + fac_a3 ** 2) + pp12 * (szz / s12), 0.0)

    fb3_val = (np.sqrt(szx ** 2 + (pn12 * szz) ** 2) + pn12 * szz) / s12
    fb3 = np.where(szz < 0.0, np.maximum(0.0, fb3_val), 0.0)

    fac_c3 = 0.5 / ((1.0 + pn22) * s12)
    fc3_bracket = (szx * fac_c3) ** 2 + (szz / s2c) ** 2
    denom3 = np.minimum(szz, -0.01 * s2c)
    fc3 = np.where(szz < 0.0, -fc3_bracket * s2c / denom3, 0.0)

    d_instant = np.maximum.reduce([f1, fa2, fb2, fc2, fa3, fb3, fc3])
    dama[:] = np.maximum(dama, np.minimum(1.0, d_instant))
    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance Puck composite failure criterion for a plane-stress shell layer.

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
    sig_3d[:, 0] = sig_arr[:, 0]  # sxx
    sig_3d[:, 1] = sig_arr[:, 1]  # syy
    # szz = 0 in plane stress
    if sig_arr.shape[1] > 2:
        sig_3d[:, 3] = sig_arr[:, 2]  # sxy
    if sig_arr.shape[1] > 3:
        sig_3d[:, 4] = sig_arr[:, 3]  # syz
    if sig_arr.shape[1] > 4:
        sig_3d[:, 5] = sig_arr[:, 4]  # szx

    return solid_step(fail, sig_3d, d_epsp, deps, dt, dama, tstar=tstar)
