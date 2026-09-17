"""
Chang-Chang composite failure criterion (/FAIL/CHANG / /FAIL/CHANGCHANG).

Fortran origin: ``engine/source/materials/fail/changchang/fail_changchang_s.F90`` (solids)
and ``fail_changchang_c.F90`` (shells); reader
``starter/source/materials/fail/changchang/hm_read_fail_chang.F90``.

Theory (F.K. Chang and K.Y. Chang, J. Composite Materials 21 (1987) 834-855)
---------------------------------------------------------------------------
Progressive failure analysis of laminated composites with 4 distinct failure modes:
1. Fiber tensile failure (sigma_11 >= 0):
       F_ft = (sigma_11 / X_t)^2 + beta * (sigma_12 / S_12)^2 + beta * (sigma_31 / S_12)^2

2. Fiber compressive failure (sigma_11 < 0):
       F_fc = (sigma_11 / X_c)^2

3. Matrix tensile failure (sigma_22 >= 0):
       F_mt = (sigma_22 / Y_t)^2 + (sigma_12 / S_12)^2

4. Matrix compressive failure (sigma_22 < 0):
       F_mc = (sigma_22 / (2 * S_12))^2 + (sigma_12 / S_12)^2
            + (sigma_22 / Y_c) * [ (Y_c / (2 * S_12))^2 - 1 ]

For 3D solids, matrix failure is also evaluated in direction 3 (sigma_33) with
shear stress sigma_31.

Global damage variable:
    D = max(F_ft, F_fc, F_mt, F_mc)
    D_max = max(D_prev, min(1.0, D))

Failure condition:
    Point breaks when D_max >= 1.0.
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-20
_INF = 1e30


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance Chang-Chang damage for a 3D solid slice.

    Parameters
    ----------
    fail : Failure object
        Parameters dictionary:
        - Sigma_1t : longitudinal tensile strength (X_t)
        - Sigma_2t : transversal tensile strength (Y_t)
        - Sigma_12 : shear strength (S_12)
        - Sigma_1c : longitudinal compressive strength (X_c)
        - Sigma_2c : transversal compressive strength (Y_c)
        - Beta     : shear coupling factor (beta, default 0.0)
    sig : ndarray of shape (m, 6)
        Stress tensor components [s_xx, s_yy, s_zz, s_xy, s_yz, s_zx].
    d_epsp : ndarray of shape (m,)
        Plastic strain increment (unused for stress-based failure).
    deps : ndarray
        Strain increment.
    dt : float
        Time step.
    dama : ndarray of shape (m,)
        Persistent damage variable array (updated in-place).
    tstar : optional
        Homologous temperature (unused).

    Returns
    -------
    ndarray of bool
        Mask of broken integration points.
    """
    p = fail.params
    sigt1 = max(float(p.get("sigma_1t", p.get("Sigma_1t", p.get("sigt1", p.get("xt", p.get("Xt", _INF)))))), _TINY)
    sigt2 = max(float(p.get("sigma_2t", p.get("Sigma_2t", p.get("sigt2", p.get("yt", p.get("Yt", _INF)))))), _TINY)
    sigt12 = max(float(p.get("sigma_12", p.get("Sigma_12", p.get("sigt12", p.get("s", p.get("S", p.get("s12", _INF))))))), _TINY)
    sigc1 = max(float(p.get("sigma_1c", p.get("Sigma_1c", p.get("sigc1", p.get("xc", p.get("Xc", _INF)))))), _TINY)
    sigc2 = max(float(p.get("sigma_2c", p.get("Sigma_2c", p.get("sigc2", p.get("yc", p.get("Yc", _INF)))))), _TINY)
    beta = float(p.get("Beta", p.get("beta", 0.0)))

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    szz = sig_arr[:, 2]
    sxy = sig_arr[:, 3]
    szx = sig_arr[:, 5] if sig_arr.shape[1] > 5 else np.zeros_like(sxx)

    # 1. Fiber criterion (direction 1)
    tens_fiber = sxx >= 0.0
    damft = np.where(
        tens_fiber,
        (sxx / sigt1) ** 2 + beta * (sxy / sigt12) ** 2 + beta * (szx / sigt12) ** 2,
        0.0,
    )
    damfc = np.where(~tens_fiber, (sxx / sigc1) ** 2, 0.0)

    # 2. Matrix failure criterion (direction 2)
    tens_mat2 = syy >= 0.0
    dammt1 = np.where(tens_mat2, (syy / sigt2) ** 2 + (sxy / sigt12) ** 2, 0.0)
    dammc1 = np.where(
        ~tens_mat2,
        (syy / (2.0 * sigt12)) ** 2
        + (sxy / sigt12) ** 2
        + syy * ((sigc2 / (2.0 * sigt12)) ** 2 - 1.0) / sigc2,
        0.0,
    )

    # 3. Matrix failure criterion (direction 3 for solids)
    tens_mat3 = szz >= 0.0
    dammt2 = np.where(tens_mat3, (szz / sigt2) ** 2 + (szx / sigt12) ** 2, 0.0)
    dammc2 = np.where(
        ~tens_mat3,
        (szz / (2.0 * sigt12)) ** 2
        + (szx / sigt12) ** 2
        + szz * ((sigc2 / (2.0 * sigt12)) ** 2 - 1.0) / sigc2,
        0.0,
    )

    # Global damage
    d_instant = np.maximum.reduce([damft, damfc, dammt1, dammc1, dammt2, dammc2])
    dama[:] = np.maximum(dama, np.minimum(1.0, d_instant))

    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance Chang-Chang damage for a plane-stress shell layer.

    Parameters
    ----------
    fail : Failure object
    sig : ndarray of shape (m, 3) or (m, 5)
        Stress tensor components [s_xx, s_yy, s_xy, (s_yz, s_zx)].
    d_epsp : ndarray of shape (m,)
    deps : ndarray
    dt : float
    dama : ndarray of shape (m,)
    tstar : optional
    eps_tot : optional

    Returns
    -------
    ndarray of bool
        Mask of broken integration points.
    """
    p = fail.params
    sigt1 = max(float(p.get("sigma_1t", p.get("Sigma_1t", p.get("sigt1", p.get("xt", p.get("Xt", _INF)))))), _TINY)
    sigt2 = max(float(p.get("sigma_2t", p.get("Sigma_2t", p.get("sigt2", p.get("yt", p.get("Yt", _INF)))))), _TINY)
    sigt12 = max(float(p.get("sigma_12", p.get("Sigma_12", p.get("sigt12", p.get("s", p.get("S", p.get("s12", _INF))))))), _TINY)
    sigc1 = max(float(p.get("sigma_1c", p.get("Sigma_1c", p.get("sigc1", p.get("xc", p.get("Xc", _INF)))))), _TINY)
    sigc2 = max(float(p.get("sigma_2c", p.get("Sigma_2c", p.get("sigc2", p.get("yc", p.get("Yc", _INF)))))), _TINY)
    beta = float(p.get("Beta", p.get("beta", 0.0)))

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)

    # 1. Fiber mode
    tens_fiber = sxx >= 0.0
    damft = np.where(tens_fiber, (sxx / sigt1) ** 2 + beta * (sxy / sigt12) ** 2, 0.0)
    damfc = np.where(~tens_fiber, (sxx / sigc1) ** 2, 0.0)

    # 2. Matrix mode
    tens_mat = syy >= 0.0
    dammt = np.where(tens_mat, (syy / sigt2) ** 2 + (sxy / sigt12) ** 2, 0.0)
    dammc = np.where(
        ~tens_mat,
        (syy / (2.0 * sigt12)) ** 2
        + (sxy / sigt12) ** 2
        + syy * ((sigc2 / (2.0 * sigt12)) ** 2 - 1.0) / sigc2,
        0.0,
    )

    d_instant = np.maximum.reduce([damft, damfc, dammt, dammc])
    dama[:] = np.maximum(dama, np.minimum(1.0, d_instant))

    return dama >= 1.0
