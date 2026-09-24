"""Hosford-Coulomb fracture locus with DSSE (/FAIL/HC_DSSE).

Fortran origin:
- ``engine/source/materials/fail/hc_dsse/fail_hc_dsse_c.F`` (shells)
- ``starter/source/materials/fail/hc_dsse/hm_read_fail_hc_dsse.F`` (reader & fitting)
- CFG: ``radioss2018/FAIL/fail_hc_dsse.cfg``

Theory (D. Mohr, F. Pack, J. Mech. Phys. Solids 67 (2014) 40-54;
        Y. Bao, T. Wierzbicki, Int. J. Mech. Sci. 46 (2004) 81-98)
-----------------------------------------------------------------
Hosford-Coulomb fracture locus expresses the equivalent plastic strain to fracture
as a function of stress triaxiality eta and normalized Lode angle parameter theta_bar:

    eps_f(eta, theta_bar) = b * [ (1 + c) / g_HC(eta, theta_bar) ]^(1 / n_f)

where:
    eta = sigma_m / sigma_vm
    xi = -27/2 * eta * (eta^2 - 1/3)   (for plane stress)
    theta_bar = 1 - (2 / pi) * arccos(clip(xi, -1, 1))

Transformation parameters:
    f_1 =  2/3 * cos((1 - theta_bar) * pi / 6)
    f_2 =  2/3 * cos((3 + theta_bar) * pi / 6)
    f_3 = -2/3 * cos((1 + theta_bar) * pi / 6)

Hosford-Coulomb envelope:
    g_HC = { 1/2 [ <f_1 - f_2>^a + <f_2 - f_3>^a + <f_1 - f_3>^a ] }^(1/a)
           + c * (2*eta + f_1 + f_3)

Regions:
- eta < -1/3: No fracture (eps_f = 100.0)
- -1/3 <= eta < 2/3: Standard HC evaluation
- eta >= 2/3: Equibiaxial tension plateau (eta = 2/3, f1=f2=1/3, f3=-2/3)

Domain of Shell Stress States Extrapolation (DSSE) for localized necking (eta > 1/3):
    g_1 = 3/2 * eta + sqrt(max(0, 1/3 - 3/4 * eta^2))
    g_2 = 3/2 * eta - sqrt(max(0, 1/3 - 3/4 * eta^2))
    g_3 = { 1/2 [ <g_1 - g_2>^d + g_1^d + <g_2>^d ] }^(1/d)
    eps_DSSE = b * g_3^(-100)

Damage accumulation:
    dD_1 = d(eps_p) / max(1e-6, eps_f)
    dD_2 = d(eps_p) / max(1e-6, eps_DSSE)   (for eta > 1/3)
    Element failure when D_1 >= 1.0 or (D_2 >= 1.0 through thickness).
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-20
_INF = 1e20


def fit_physical_parameters(a_shear: float, b_ut: float, c_pst: float,
                            inst: float, n_f: float = 1.0) -> tuple[float, float, float, float, float]:
    """Fit physical failure strain data (I_flag=1) to HC-DSSE model parameters (a, b, c, d, n_f).

    Faithfully replicates ``hm_read_fail_hc_dsse.F`` lines 2553-2580.

    Parameters
    ----------
    a_shear : float
        Failure strain in shear.
    b_ut : float
        Failure strain in uniaxial tension.
    c_pst : float
        Failure strain in plane strain tension.
    inst : float
        Necking / instability strain.
    n_f : float
        Exponent parameter (default 1.0).

    Returns
    -------
    a, b, c, d, n_f : tuple of float
        Fitted model parameters.
    """
    b_frac2 = float(b_ut)
    b16 = (float(a_shear) / b_frac2) ** 0.1
    b17 = (float(c_pst) / b_frac2) ** 0.1
    b18 = 1.0 / b16 + 2.0 / np.sqrt(3.0) - 1.0 / b17
    b19 = -1.0 / b16 + 1.0 / b17
    c_frac2 = b19 / b18
    b21 = (1.0 + c_frac2) / b16

    # 1201-point lookup table from hm_read_fail_hc_dsse.F
    a_grid = np.linspace(2.0, 0.8, 1201)
    lx_grid = ((1.0 + 2.0 ** (a_grid - 1.0)) ** (1.0 / a_grid)) / np.sqrt(3.0)
    # lx_grid is monotonically increasing as a_grid decreases from 2.0 to 0.8
    a_frac2 = float(np.interp(b21, lx_grid, a_grid))

    # Newton-Raphson iteration for inst2 (instability exponent d)
    f05 = np.sqrt(3.0) * (float(inst) / b_frac2) ** (-0.01)
    inst2 = 1.0
    small = 0.00001
    for _ in range(20):
        g12 = ((1.0 + 2.0 ** (inst2 - 1.0)) ** (1.0 / inst2) - f05) ** 2
        f_plus = ((1.0 + 2.0 ** (inst2 + small - 1.0)) ** (1.0 / (inst2 + small)) - f05) ** 2
        f_minus = ((1.0 + 2.0 ** (inst2 - small - 1.0)) ** (1.0 / (inst2 - small)) - f05) ** 2
        i12 = (f_plus - f_minus) / (2.0 * small)
        if abs(i12) > 1e-15:
            inst2 = inst2 - g12 / i12

    return a_frac2, b_frac2, c_frac2, float(inst2), float(n_f)


def compute_hc_fracture_strain(eta: np.ndarray | float,
                               lode_param: np.ndarray | float,
                               a: float, b: float, c: float,
                               n_f: float = 1.0) -> np.ndarray | float:
    """Compute Hosford-Coulomb fracture strain eps_f(eta, theta_bar).

    Parameters
    ----------
    eta : ndarray or float
        Stress triaxiality eta = sigma_m / sigma_vm.
    lode_param : ndarray or float
        Normalized Lode angle parameter theta_bar in [-1, 1].
    a : float
        HC exponent a.
    b : float
        HC coefficient b (uniaxial tensile fracture strain).
    c : float
        HC coefficient c.
    n_f : float
        Damage exponent n_f (default 1.0).

    Returns
    -------
    eps_f : ndarray or float
        Fracture strain.
    """
    is_scalar = np.isscalar(eta)
    eta_arr = np.atleast_1d(np.asarray(eta, dtype=float))
    lode_arr = np.atleast_1d(np.asarray(lode_param, dtype=float))
    lode_arr = np.clip(lode_arr, -1.0, 1.0)

    pi = np.pi
    f1 = (2.0 / 3.0) * np.cos((1.0 - lode_arr) * pi / 6.0)
    f2 = (2.0 / 3.0) * np.cos((3.0 + lode_arr) * pi / 6.0)
    f3 = -(2.0 / 3.0) * np.cos((1.0 + lode_arr) * pi / 6.0)

    eps_f = np.zeros_like(eta_arr)

    # 1. Cutoff: eta < -1/3
    mask_comp = eta_arr < -1.0 / 3.0
    eps_f[mask_comp] = 100.0

    # 2. Intermediate range: -1/3 <= eta < 2/3
    mask_mid = (~mask_comp) & (eta_arr < 2.0 / 3.0)
    if np.any(mask_mid):
        f1_m, f2_m, f3_m = f1[mask_mid], f2[mask_mid], f3[mask_mid]
        eta_m = eta_arr[mask_mid]
        diff1 = np.maximum(0.0, f1_m - f2_m) ** a
        diff2 = np.maximum(0.0, f2_m - f3_m) ** a
        diff3 = np.maximum(0.0, f1_m - f3_m) ** a
        ghc = (0.5 * (diff1 + diff2 + diff3)) ** (1.0 / a) + c * (2.0 * eta_m + f1_m + f3_m)
        eps_f[mask_mid] = b * np.maximum(0.0, (1.0 + c) / np.maximum(_TINY, ghc)) ** (1.0 / n_f)

    # 3. Equibiaxial plateau: eta >= 2/3
    mask_high = eta_arr >= 2.0 / 3.0
    if np.any(mask_high):
        f1_h = 1.0 / 3.0
        f2_h = 1.0 / 3.0
        f3_h = -2.0 / 3.0
        eta_h = 2.0 / 3.0
        diff1 = 0.0
        diff2 = (f2_h - f3_h) ** a
        diff3 = (f1_h - f3_h) ** a
        ghc = (0.5 * (diff1 + diff2 + diff3)) ** (1.0 / a) + c * (2.0 * eta_h + f1_h + f3_h)
        eps_f[mask_high] = b * np.maximum(0.0, (1.0 + c) / np.maximum(_TINY, ghc)) ** (1.0 / n_f)

    if is_scalar:
        return float(eps_f[0])
    return eps_f


def compute_dsse_strain(eta: np.ndarray | float, b: float, d: float) -> np.ndarray | float:
    """Compute DSSE instability strain for localized necking (eta > 1/3).

    Matches fail_hc_dsse_c.F lines 168-172:
        g1 = 3/2*eta + sqrt(1/3 - 3/4*eta^2)
        g2 = 3/2*eta - sqrt(1/3 - 3/4*eta^2)
        g3 = (1/2 * (<g1 - g2>^d + g1^d + <g2>^d))^(1/d)
        eps_dsse = b * g3^(-100)
    """
    is_scalar = np.isscalar(eta)
    eta_arr = np.atleast_1d(np.asarray(eta, dtype=float))

    dsse = np.full_like(eta_arr, _INF)
    mask = eta_arr > 1.0 / 3.0
    if np.any(mask):
        e_m = eta_arr[mask]
        rad = np.maximum(0.0, 1.0 / 3.0 - 0.75 * (e_m ** 2))
        sqrt_rad = np.sqrt(rad)
        g1 = 1.5 * e_m + sqrt_rad
        g2 = 1.5 * e_m - sqrt_rad
        term1 = np.maximum(0.0, g1 - g2) ** d
        term2 = np.maximum(0.0, g1) ** d
        term3 = np.maximum(0.0, g2) ** d
        g3 = (0.5 * (term1 + term2 + term3)) ** (1.0 / d)
        dsse[mask] = b * (np.maximum(1e-20, g3) ** (-100.0))

    if is_scalar:
        return float(dsse[0])
    return dsse


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance HC-DSSE failure model for a plane-stress shell layer.

    Parameters
    ----------
    fail : Failure object
        Model parameters:
        - a_hc_dsse / a
        - b_hc_dsse / b
        - c_hc_dsse / c
        - d_hc_dsse / d
        - n_f
        - iflag (0 = direct parameters, 1 = physical fitting)
    sig : ndarray of shape (m, 3) or (m, 5)
        Plane stress tensor [s_xx, s_yy, s_xy, (s_yz, s_zx)].
    d_epsp : ndarray of shape (m,)
        Equivalent plastic strain increment.
    deps : ndarray
    dt : float
    dama : ndarray of shape (m,)
        Accumulated damage (in-place).
    tstar : optional
    eps_tot : optional

    Returns
    -------
    ndarray of bool
        Broken element mask (dama >= 1.0).
    """
    p = fail.params
    iflag = int(p.get("iflag", p.get("I_Flag", p.get("I_flag", 0))))
    a_in = float(p.get("a_hc_dsse", p.get("a", p.get("A", 1.0))))
    b_in = float(p.get("b_hc_dsse", p.get("b", p.get("B", 0.5))))
    c_in = float(p.get("c_hc_dsse", p.get("c", p.get("C", 0.1))))
    d_in = float(p.get("d_hc_dsse", p.get("d", p.get("D", 1.0))))
    nf_in = float(p.get("n_f", p.get("n", 1.0)))

    if iflag == 1:
        a, b, c, d, n_f = fit_physical_parameters(a_in, b_in, c_in, d_in, nf_in)
    else:
        a, b, c, d, n_f = a_in, b_in, c_in, d_in, nf_in

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)

    sig_m = (sxx + syy) / 3.0
    sig_vm = np.sqrt(np.maximum(0.0, sxx ** 2 + syy ** 2 - sxx * syy + 3.0 * (sxy ** 2)))
    eta = sig_m / np.maximum(1e-10, sig_vm)

    xi = -13.5 * eta * (eta ** 2 - 1.0 / 3.0)
    xi = np.clip(xi, -1.0, 1.0)
    lode = 1.0 - (2.0 / np.pi) * np.arccos(xi)

    eps_f = compute_hc_fracture_strain(eta, lode, a, b, c, n_f)

    d_epsp_arr = np.asarray(d_epsp, dtype=float) if d_epsp is not None else np.zeros_like(sxx)
    dama[:] = np.minimum(1.0, dama + d_epsp_arr / np.maximum(1e-6, eps_f))
    return dama >= 1.0


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance HC-DSSE failure model for a 3D solid slice.

    Parameters
    ----------
    fail : Failure object
    sig : ndarray of shape (m, 6)
        Stress components [s_xx, s_yy, s_zz, s_xy, s_yz, s_zx].
    d_epsp : ndarray of shape (m,)
        Equivalent plastic strain increment.
    deps : ndarray
    dt : float
    dama : ndarray of shape (m,)
        Accumulated damage (in-place).
    tstar : optional

    Returns
    -------
    ndarray of bool
        Broken element mask (dama >= 1.0).
    """
    p = fail.params
    iflag = int(p.get("iflag", p.get("I_Flag", p.get("I_flag", 0))))
    a_in = float(p.get("a_hc_dsse", p.get("a", p.get("A", 1.0))))
    b_in = float(p.get("b_hc_dsse", p.get("b", p.get("B", 0.5))))
    c_in = float(p.get("c_hc_dsse", p.get("c", p.get("C", 0.1))))
    d_in = float(p.get("d_hc_dsse", p.get("d", p.get("D", 1.0))))
    nf_in = float(p.get("n_f", p.get("n", 1.0)))

    if iflag == 1:
        a, b, c, d, n_f = fit_physical_parameters(a_in, b_in, c_in, d_in, nf_in)
    else:
        a, b, c, d, n_f = a_in, b_in, c_in, d_in, nf_in

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    szz = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)
    sxy = sig_arr[:, 3] if sig_arr.shape[1] > 3 else np.zeros_like(sxx)
    syz = sig_arr[:, 4] if sig_arr.shape[1] > 4 else np.zeros_like(sxx)
    szx = sig_arr[:, 5] if sig_arr.shape[1] > 5 else np.zeros_like(sxx)

    sig_m = (sxx + syy + szz) / 3.0
    svm_sq = 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2
                    + 6.0 * (sxy ** 2 + syz ** 2 + szx ** 2))
    sig_vm = np.sqrt(np.maximum(0.0, svm_sq))
    eta = sig_m / np.maximum(1e-10, sig_vm)

    # 3D third invariant of deviatoric stress
    dev_xx = sxx - sig_m
    dev_yy = syy - sig_m
    dev_zz = szz - sig_m
    j3 = (dev_xx * dev_yy * dev_zz + 2.0 * sxy * syz * szx
          - dev_xx * (syz ** 2) - dev_yy * (szx ** 2) - dev_zz * (sxy ** 2))
    xi = 13.5 * j3 / np.maximum(1e-20, sig_vm ** 3)
    xi = np.clip(xi, -1.0, 1.0)
    lode = 1.0 - (2.0 / np.pi) * np.arccos(xi)

    eps_f = compute_hc_fracture_strain(eta, lode, a, b, c, n_f)

    d_epsp_arr = np.asarray(d_epsp, dtype=float) if d_epsp is not None else np.zeros_like(sxx)
    dama[:] = np.minimum(1.0, dama + d_epsp_arr / np.maximum(1e-6, eps_f))
    return dama >= 1.0
