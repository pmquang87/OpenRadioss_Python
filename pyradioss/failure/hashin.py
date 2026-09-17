"""
Hashin 3D composite failure criteria (/FAIL/HASHIN).

Fortran origin: ``engine/source/materials/fail/hashin/fail_hashin_s.F90`` (solids)
and ``fail_hashin_c.F90`` (shells); reader
``starter/source/materials/fail/hashin/hm_read_fail_hashin.F90``.

Theory (Z. Hashin, J. Appl. Mech. 47 (1980) 329-334)
----------------------------------------------------
3D failure criteria for unidirectional (Imodel=1) and woven/fabric (Imodel=2)
composite laminae.

Unidirectional lamina (Imodel=1):
1. Tensile/shear fiber mode:
       f_1 = (max(sigma_11, 0) / X_t)^2 + (sigma_12^2 + sigma_31^2) / S_12f^2

2. Compressive fiber mode:
       sigma* = -sigma_11 + max(-0.5 * (sigma_22 + sigma_33), 0)
       f_2 = (sigma* / X_c)^2   if sigma* > 0 else 0

3. Crush mode (hydrostatic compression):
       P = -1/3 (sigma_11 + sigma_22 + sigma_33)
       f_3 = (P / sigma_c)^2   if P > 0 else 0

4. Matrix failure mode:
       S_12* = S_12m - min(sigma_22, 0) * tan(phi)
       S_23* = S_23m - min(sigma_22, 0) * tan(phi)
       f_4 = (max(sigma_22, 0) / Y_t)^2 + (sigma_23 / S_23*)^2 + (sigma_12 / S_12*)^2

5. Delamination mode:
       S_13* = S_13m - min(sigma_33, 0) * tan(phi)
       S_23* = S_23m - min(sigma_33, 0) * tan(phi)
       f_5 = S_del^2 * [ (max(sigma_33, 0) / Z_t)^2 + (sigma_23 / S_23*)^2 + (sigma_31 / S_13*)^2 ]

Global damage variable:
    D = max(f_1, f_2, f_3, f_4, f_5)
    D_max = max(D_prev, min(1.0, D))

Failure condition:
    Point breaks when D_max >= 1.0.
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-20
_INF = 1e30


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance Hashin failure criteria for a 3D solid slice.

    Parameters
    ----------
    fail : Failure object
        Parameters dictionary:
        - Sigma_1t : X_t (tensile fiber strength)
        - Sigma_2t : Y_t (transverse tensile strength)
        - Sigma_3t : Z_t (through-thickness tensile strength)
        - Sigma_1c : X_c (compressive fiber strength)
        - Sigma_2c : Y_c (transverse compressive strength)
        - C_sig    : sigma_c (crush strength)
        - Sigma_12 : S_12f (fiber shear strength)
        - msig12   : S_12m (matrix shear strength 1-2)
        - msig13   : S_13m (matrix shear strength 1-3)
        - msig23   : S_23m (matrix shear strength 2-3)
        - angle    : phi (friction angle in radians or degrees)
        - sdel     : S_del (delamination factor)
        - Imodel   : 1 (unidirectional) or 2 (fabric)
    sig : ndarray of shape (m, 6)
        [s_xx, s_yy, s_zz, s_xy, s_yz, s_zx].
    d_epsp : ndarray of shape (m,)
    deps : ndarray
    dt : float
    dama : ndarray of shape (m,)
    tstar : optional

    Returns
    -------
    ndarray of bool
        Mask of broken integration points.
    """
    p = fail.params
    sigt1 = max(float(p.get("sigma_1t", p.get("Sigma_1t", p.get("sigt1", p.get("xt", p.get("Xt", p.get("XT", _INF))))))), _TINY)
    sigt2 = max(float(p.get("sigma_2t", p.get("Sigma_2t", p.get("sigt2", p.get("yt", p.get("Yt", p.get("YT", _INF))))))), _TINY)
    sigt3 = max(float(p.get("sigma_3t", p.get("Sigma_3t", p.get("sigt3", p.get("zt", p.get("Zt", p.get("ZT", _INF))))))), _TINY)
    sigc1 = max(float(p.get("sigma_1c", p.get("Sigma_1c", p.get("sigc1", p.get("xc", p.get("Xc", p.get("XC", _INF))))))), _TINY)
    sigc2 = max(float(p.get("sigma_2c", p.get("Sigma_2c", p.get("sigc2", p.get("yc", p.get("Yc", p.get("YC", _INF))))))), _TINY)
    csig = max(float(p.get("sigma_c", p.get("sigma_3c", p.get("Sigma_c", p.get("C_sig", p.get("csig", p.get("Csig", _INF))))))), _TINY)
    fsig12 = max(float(p.get("sigma_12f", p.get("sigma_12", p.get("Sigma_12", p.get("fsig12", p.get("Fsig12", p.get("s12", p.get("S12", _INF)))))))), _TINY)
    msig12 = max(float(p.get("sigma_12m", p.get("msig12", p.get("Msig12", fsig12)))), _TINY)
    msig13 = max(float(p.get("sigma_13m", p.get("msig13", p.get("Msig13", p.get("sigma_31", fsig12))))), _TINY)
    msig23 = max(float(p.get("sigma_23m", p.get("msig23", p.get("Msig23", p.get("sigma_23", fsig12))))), _TINY)


    angle = float(p.get("angle", p.get("alpha", 0.0)))
    # If angle given in degrees (> 1.57), convert to radians
    if abs(angle) > 1.5707963:
        angle = np.radians(angle)
    tan_phi = np.tan(angle)

    sdel = float(p.get("sdel", 1.0))
    imodel = int(p.get("Imodel", p.get("imodel", 1)))

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    szz = sig_arr[:, 2]
    sxy = sig_arr[:, 3]
    syz = sig_arr[:, 4] if sig_arr.shape[1] > 4 else np.zeros_like(sxx)
    szx = sig_arr[:, 5] if sig_arr.shape[1] > 5 else np.zeros_like(sxx)

    if imodel == 2:
        # Fabric lamina model
        f1 = (np.maximum(sxx, 0.0) / sigt1) ** 2 + (sxy**2 + szx**2) / (fsig12**2)
        tsig12 = max(fsig12 * sigt2 / sigt1, _TINY)
        f2 = (np.maximum(syy, 0.0) / sigt2) ** 2 + (sxy**2 + syz**2) / (tsig12**2)

        sig_c1 = -sxx + np.maximum(-szz, 0.0)
        f3 = np.where(sig_c1 > 0.0, (sig_c1 / sigc1) ** 2, 0.0)

        sig_c2 = -syy + np.maximum(-szz, 0.0)
        f4 = np.where(sig_c2 > 0.0, (sig_c2 / sigc2) ** 2, 0.0)

        p_hyd = -(sxx + syy + szz) / 3.0
        f5 = np.where(p_hyd > 0.0, (p_hyd / csig) ** 2, 0.0)

        f6 = (sxy / msig12) ** 2

        xsig13 = np.where(szz < 0.0, msig13 - szz * tan_phi, msig13)
        xsig23 = np.where(szz < 0.0, msig23 - szz * tan_phi, msig23)
        f7 = (np.maximum(szz, 0.0) / sigt3) ** 2 + (syz / xsig23) ** 2 + (szx / xsig13) ** 2

        d_instant = np.maximum.reduce([f1, f2, f3, f4, f5, f6, f7])
    else:
        # Unidirectional lamina model (imodel=1 default)
        f1 = (np.maximum(sxx, 0.0) / sigt1) ** 2 + (sxy**2 + szx**2) / (fsig12**2)

        sig_comp = -sxx + np.maximum(-0.5 * (syy + szz), 0.0)
        f2 = np.where(sig_comp > 0.0, (sig_comp / sigc1) ** 2, 0.0)

        p_hyd = -(sxx + syy + szz) / 3.0
        f3 = np.where(p_hyd > 0.0, (p_hyd / csig) ** 2, 0.0)

        xsig12 = np.where(syy < 0.0, msig12 - syy * tan_phi, msig12)
        xsig23_y = np.where(syy < 0.0, msig23 - syy * tan_phi, msig23)
        f4 = (np.maximum(syy, 0.0) / sigt2) ** 2 + (syz / xsig23_y) ** 2 + (sxy / xsig12) ** 2

        xsig13_z = np.where(szz < 0.0, msig13 - szz * tan_phi, msig13)
        xsig23_z = np.where(szz < 0.0, msig23 - szz * tan_phi, msig23)
        f5 = (sdel**2) * (
            (np.maximum(szz, 0.0) / sigt3) ** 2
            + (syz / xsig23_z) ** 2
            + (szx / xsig13_z) ** 2
        )

        d_instant = np.maximum.reduce([f1, f2, f3, f4, f5])

    dama[:] = np.maximum(dama, np.minimum(1.0, d_instant))
    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance Hashin failure criteria for a plane-stress shell layer.

    Parameters
    ----------
    fail : Failure object
    sig : ndarray of shape (m, 3) or (m, 5)
        [s_xx, s_yy, s_xy, (s_yz, s_zx)].
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
