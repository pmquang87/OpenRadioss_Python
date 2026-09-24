"""
Wilkins cumulative damage failure criterion (/FAIL/WILKINS).

Fortran origin: ``engine/source/materials/fail/wilkins/fail_wilkins_s.F`` (solids)
and ``fail_wilkins_c.F`` (shells); reader
``starter/source/materials/fail/wilkins/hm_read_fail_wilkins.F``.

Theory (M.L. Wilkins, J. Mech. Phys. Solids 28 (1980) 907-918)
--------------------------------------------------------------
Cumulative damage fracture model accounting for both hydrostatic tension
and asymmetric deviatoric shear state:

    D = integral W_1(P) * W_2(s) * d(eps_p)

where:
1. Hydrostatic pressure weighting:
       W_1 = ( max(1e-20, 1 / (1 - P / P_c)) )^alpha
   with P = 1/3 tr(sigma) = mean stress, and P_c = hydrostatic tensile limit (Plim).

2. Asymmetric shear weighting:
   From the deviatoric stress tensor s = sigma - P * I, sorted principal
   deviatoric stresses s1 >= s2 >= s3 (with s1 + s2 + s3 = 0):
       A = max(s2 / s3, s2 / s1)    (for s1 != 0 and s3 != 0)
       W_2 = ( max(1e-20, 2 - A) )^beta

Damage accumulates with plastic strain:
    dD = W_1 * W_2 * d(eps_p)
    D += dD

Failure condition:
    Point breaks when D >= D_c (critical damage threshold, default 1.0).
    - If ifail_so == 1: solid element is eroded/deleted.
    - If ifail_so == 2: deviatoric stress vanishes (sigma = P * I).
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-20


def _principal_deviatoric(s_xx, s_yy, s_zz, s_xy, s_yz, s_zx):
    """Compute sorted principal deviatoric stresses s1 >= s2 >= s3 using analytical Cardano/Newton.
    All inputs 1D numpy arrays per element.
    """
    n = len(s_xx)
    s1 = np.zeros(n, dtype=float)
    s2 = np.zeros(n, dtype=float)
    s3 = np.zeros(n, dtype=float)

    for i in range(n):
        # Deviatoric components
        e1, e2, e3 = s_xx[i], s_yy[i], s_zz[i]
        e4, e5, e6 = s_xy[i], s_yz[i], s_zx[i]
        e42, e52, e62 = e4 * e4, e5 * e5, e6 * e6

        # Invariants of deviator: e1 + e2 + e3 = 0
        c = -0.5 * (e1 * e1 + e2 * e2 + e3 * e3) - e42 - e52 - e62
        d = -e1 * e2 * e3 + e1 * e52 + e2 * e62 + e3 * e42 - 2.0 * e4 * e5 * e6

        cc1 = c / 3.0
        val = max(0.0, -cc1)
        r1 = np.sqrt(val)
        epst2 = r1 * r1
        y = (epst2 + c) * r1 + d

        if abs(y) > 1e-8:
            r1 = 1.75 * r1
            for _ in range(4):
                epst2 = r1 * r1
                y = (epst2 + c) * r1 + d
                yp = 3.0 * epst2 + c
                if abs(yp) > 1e-15:
                    r1 = r1 - y / yp

        c_quad = c + r1 * r1
        disc = max(0.0, r1 * r1 - 4.0 * c_quad)
        sqrt_disc = np.sqrt(disc)
        r2 = 0.5 * (-r1 + sqrt_disc)
        r3 = 0.5 * (-r1 - sqrt_disc)

        # Sort descending: s1 >= s2 >= s3
        roots = sorted([r1, r2, r3], reverse=True)
        s1[i], s2[i], s3[i] = roots[0], roots[1], roots[2]

    return s1, s2, s3


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance Wilkins damage for a solid element slice; returns broken mask."""
    p = fail.params
    alpha = float(p.get("alpha", p.get("Alpha", 1.0)))
    beta = float(p.get("beta", p.get("Beta", p.get("Beta_WILKINS", 1.0))))
    pc = float(p.get("plim", p.get("p_lim", p.get("Plim", p.get("P_c", p.get("Pc", p.get("pc", 1e20)))))))
    dc = float(p.get("df", p.get("Df", p.get("D_c", p.get("Dc", p.get("dc", 1.0))))))

    sig_arr = np.asarray(sig, dtype=float)
    d_epsp_arr = np.asarray(d_epsp, dtype=float)

    # Mean hydrostatic pressure P = 1/3 tr(sigma)
    p_mean = (sig_arr[:, 0] + sig_arr[:, 1] + sig_arr[:, 2]) / 3.0

    # Deviatoric stresses
    s_xx = sig_arr[:, 0] - p_mean
    s_yy = sig_arr[:, 1] - p_mean
    s_zz = sig_arr[:, 2] - p_mean
    s_xy = sig_arr[:, 3]
    s_yz = sig_arr[:, 4] if sig_arr.shape[1] > 4 else np.zeros_like(s_xx)
    s_zx = sig_arr[:, 5] if sig_arr.shape[1] > 5 else np.zeros_like(s_xx)

    # Principal deviatoric stresses
    s1, s2, s3 = _principal_deviatoric(s_xx, s_yy, s_zz, s_xy, s_yz, s_zx)

    # Asymmetry ratio A = max(s2/s3, s2/s1)
    a_ratio = np.ones_like(s1)
    mask_s1 = np.abs(s1) > 1e-15
    mask_s3 = np.abs(s3) > 1e-15
    both = mask_s1 & mask_s3
    only_s1 = mask_s1 & (~mask_s3)
    only_s3 = (~mask_s1) & mask_s3

    a_ratio[both] = np.maximum(s2[both] / s3[both], s2[both] / s1[both])
    a_ratio[only_s1] = s2[only_s1] / s1[only_s1]
    a_ratio[only_s3] = s2[only_s3] / s3[only_s3]

    # W1 = (max(1e-20, 1 / (1 - P / Pc)))^alpha
    w1_term = np.maximum(_TINY, 1.0 / np.maximum(1e-12, 1.0 - p_mean / pc))
    w1 = w1_term ** alpha

    # W2 = (max(1e-20, 2 - A))^beta
    w2_term = np.maximum(_TINY, 2.0 - a_ratio)
    w2 = w2_term ** beta

    # Accumulate damage only with active plastic strain
    d_dama = np.where(d_epsp_arr > 0.0, w1 * w2 * d_epsp_arr, 0.0)
    dama[:] += d_dama

    return dama >= dc


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance Wilkins damage for a shell layer; returns broken mask."""
    sig_arr = np.asarray(sig, dtype=float)
    n = len(sig_arr)
    # Reconstruct plane stress 3D tensor: sig_zz = 0, sig_yz = 0, sig_zx = 0
    sig_3d = np.zeros((n, 6), dtype=float)
    sig_3d[:, 0] = sig_arr[:, 0]  # sxx
    sig_3d[:, 1] = sig_arr[:, 1]  # syy
    if sig_arr.shape[1] > 2:
        sig_3d[:, 3] = sig_arr[:, 2]  # sxy
    if sig_arr.shape[1] > 3:
        sig_3d[:, 4] = sig_arr[:, 3]  # syz
    if sig_arr.shape[1] > 4:
        sig_3d[:, 5] = sig_arr[:, 4]  # szx

    return solid_step(fail, sig_3d, d_epsp, deps, dt, dama, tstar=tstar)
