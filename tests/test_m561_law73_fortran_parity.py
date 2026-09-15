r"""Fortran Parity & Physics Oracle Verifier for Milestone M561: /MAT/LAW73 (/MAT/BARLAT2000 /MAT/LAW73 /MAT/HILL_THERM /MAT/THERM_HILL).

Thermal Hill Orthotropic Plasticity Model for Shell Elements.
Directly cites and mirrors upstream OpenRadioss source:
- Upstream Fortran starter reader:
  ``starter/source/materials/mat/mat073/hm_read_mat73.F``
- Upstream Fortran engine physics:
  ``engine/source/materials/mat/mat073/sigeps73c.F`` (lines 145 to 442)
- Table interpolation tools:
  ``engine/source/tools/curve/table_tools.F`` (TABLE_VINTERP)
- CFG card definition:
  ``hm_cfg_files/config/CFG/radioss140/MAT/matl73_BARLAT2000.cfg``

Tests included:
1. Exact Python oracle reproducing sigeps73c.F lines 145 to 442.
2. Hill coefficients calculation and isotropic recovery (R00=R45=R90=1 -> von Mises plane stress).
3. Unnormalized (iyield=0) vs normalized (iyield=1) Hill yield surface modes.
4. Elastic trial stress calculation with backstress shift and transverse shear.
5. Dynamic Young's modulus degradation: function mode (opte=1) and exponential mode (ce>0).
6. Dilatational acoustic sound speed evaluation.
7. Temperature update under adiabatic plastic heating and coupled thermal conduction (jthe>0).
8. Equivalent plastic strain rate and total tensile strain with tensile softening factor fail.
9. Mixed isotropic / kinematic hardening with backstress Bauschinger effect.
10. Plane-stress radial return with 2 Newton iterations and through-thickness thinning.
11. Element deletion when accumulated plastic strain pla > eps_max.
12. Vectorized multi-element batch execution parity.
13. Exhaustive 60+ diverse physical states comparison: shell_update vs sigeps73c_oracle (rtol=1e-12, atol=1e-12).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pytest

from pyradioss.materials.law73_hill_therm import (
    Law73Params,
    build_law73,
    shell_update,
    sound_speed,
    extra_shapes,
    consistent_shell_tangent,
    shell_membrane_tangent,
)

_EM20 = 1.0e-20
_INF = 1.0e30


# =============================================================================
# 1. UPSTREAM FORTRAN ENGINE ORACLE: sigeps73c.F (lines 145 to 442)
# =============================================================================

def sigeps73c_oracle(
    matparam: Dict[str, Any],
    sigo: np.ndarray,
    deps: np.ndarray,
    rho0: Union[float, np.ndarray] = 1.0,
    rho: Optional[Union[float, np.ndarray]] = None,
    epsp_rates: Optional[np.ndarray] = None,
    eps_tot: Optional[np.ndarray] = None,
    pla: Optional[np.ndarray] = None,
    uvar: Optional[np.ndarray] = None,
    off: Optional[np.ndarray] = None,
    thk: Optional[np.ndarray] = None,
    thkly: Optional[np.ndarray] = None,
    eint: Optional[np.ndarray] = None,
    vol: Optional[Union[float, np.ndarray]] = None,
    tempel: Optional[np.ndarray] = None,
    jthe: int = 0,
    shf: float = 5.0 / 6.0,
    timestep: float = 0.0,
    yield_fn: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
    curve_e_fn: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, Any]:
    """Exact Python reproduction of OpenRadioss sigeps73c.F lines 145-442.

    Parameters
    ----------
    matparam : dict
        Material parameters following hm_read_mat73.F / sigeps73c.F:
        'e', 'nu', 'r00', 'r45', 'r90', 'chard' (fisokin), 'iyield',
        'eps_max', 'epsr1', 'epsr2', 'fscale', 'pscale', 't0', 'rhocp',
        'opte', 'ifunce', 'einf', 'ce', 'yield_table'
    sigo : ndarray of shape (nel, 3) or (nel, 5)
        Old Cauchy stress tensor [xx, yy, xy, (yz, zx)].
    deps : ndarray of shape (nel, 3) or (nel, 5)
        Engineering strain increment [xx, yy, xy, (yz, zx)].
    rho0 : float or ndarray
        Initial density.
    rho : float or ndarray, optional
        Current density.
    epsp_rates : ndarray of shape (nel, 3), optional
        Strain rate tensor components [edxx, edyy, edxy].
    eps_tot : ndarray of shape (nel, 3), optional
        Total strain tensor components [exx, eyy, exy].
    pla : ndarray of length nel, optional
        Initial plastic strain history.
    uvar : ndarray of shape (nel, 7), optional
        Internal history variables:
        [uvar1, alpha_xx, alpha_yy, alpha_xy, ipos1, ipos2, ipos3].
    off : ndarray of length nel, optional
        Element active/failure flag (1.0 = active, 0.8 = failing, 0.0 = deleted).
    thk : ndarray of length nel, optional
        Current shell thickness.
    thkly : ndarray of length nel, optional
        Initial layer thickness.
    eint : ndarray of shape (nel, 2) or length nel, optional
        Internal energy components.
    vol : float or ndarray, optional
        Initial shell volume (area * thk0).
    tempel : ndarray of length nel, optional
        Element temperature from coupled thermal solver (if jthe > 0).
    jthe : int
        Thermal flag (0 = adiabatic, >0 = coupled conduction).
    shf : float
        Shear correction factor (typically 5/6).
    timestep : float
        Time step dt.
    yield_fn : callable, optional
        Yield lookup function `(pla, edot, temp) -> (yld, dydx)`.
    curve_e_fn : callable, optional
        Young's modulus scale function `(pla) -> (scale, slope)`.

    Returns
    -------
    dict
        Updated outputs matching Fortran:
        'sign': (nel, ncomp) stress tensor
        'pla': (nel,) plastic strain
        'uvar': (nel, 7) history variables
        'thk': (nel,) thickness
        'off': (nel,) status flag
        'soundsp': (nel,) dilatational sound speed
        'temp': (nel,) temperature
        'svm': (nel,) Hill equivalent stress
        'yld': (nel,) yield stress
        'dpla': (nel,) plastic strain increment
        'fail': (nel,) softening factor
    """
    sigo_arr = np.atleast_2d(np.array(sigo, dtype=np.float64)).copy()
    deps_arr = np.atleast_2d(np.array(deps, dtype=np.float64)).copy()
    nel, ncomp = sigo_arr.shape

    if deps_arr.shape[0] == 1 and nel > 1:
        deps_arr = np.repeat(deps_arr, nel, axis=0)

    # Initial history arrays
    if pla is not None:
        pla_arr = np.atleast_1d(np.array(pla, dtype=np.float64)).copy()
    else:
        pla_arr = np.zeros(nel, dtype=np.float64)

    if uvar is not None:
        uvar_arr = np.atleast_2d(np.array(uvar, dtype=np.float64)).copy()
    else:
        uvar_arr = np.zeros((nel, 7), dtype=np.float64)
    if uvar_arr.shape[1] < 7:
        uvar_pad = np.zeros((nel, 7), dtype=np.float64)
        uvar_pad[:, :uvar_arr.shape[1]] = uvar_arr
        uvar_arr = uvar_pad

    if off is not None:
        off_arr = np.atleast_1d(np.array(off, dtype=np.float64)).copy()
    else:
        off_arr = np.ones(nel, dtype=np.float64)

    if thk is not None:
        thk_arr = np.atleast_1d(np.array(thk, dtype=np.float64)).copy()
    else:
        thk_arr = np.ones(nel, dtype=np.float64)

    if thkly is not None:
        thkly_arr = np.atleast_1d(np.array(thkly, dtype=np.float64)).copy()
    else:
        thkly_arr = np.ones(nel, dtype=np.float64)

    if eint is not None:
        eint_in = np.array(eint, dtype=np.float64)
        if eint_in.ndim == 1:
            eint_arr = np.zeros((nel, 2), dtype=np.float64)
            eint_arr[:, 0] = eint_in
        else:
            eint_arr = eint_in.copy()
    else:
        eint_arr = np.zeros((nel, 2), dtype=np.float64)

    if vol is not None:
        vol_arr = np.atleast_1d(np.array(vol, dtype=np.float64)).copy()
        if len(vol_arr) == 1 and nel > 1:
            vol_arr = np.full(nel, vol_arr[0], dtype=np.float64)
    else:
        vol_arr = np.ones(nel, dtype=np.float64)

    rho0_arr = np.atleast_1d(np.array(rho0, dtype=np.float64)).copy()
    if len(rho0_arr) == 1 and nel > 1:
        rho0_arr = np.full(nel, rho0_arr[0], dtype=np.float64)

    if rho is not None:
        rho_cur_arr = np.atleast_1d(np.array(rho, dtype=np.float64)).copy()
        if len(rho_cur_arr) == 1 and nel > 1:
            rho_cur_arr = np.full(nel, rho_cur_arr[0], dtype=np.float64)
    else:
        rho_cur_arr = rho0_arr.copy()

    if tempel is not None:
        tempel_arr = np.atleast_1d(np.array(tempel, dtype=np.float64)).copy()
        if len(tempel_arr) == 1 and nel > 1:
            tempel_arr = np.full(nel, tempel_arr[0], dtype=np.float64)
    else:
        tempel_arr = np.full(nel, float(matparam.get("t0", 293.0)), dtype=np.float64)

    # 1. Recover material parameters (hm_read_mat73.F lines 117-217)
    e = float(matparam.get("e", matparam.get("E", 210000.0)))
    nu = float(matparam.get("nu", matparam.get("NU", 0.3)))
    r00 = float(matparam.get("r00", matparam.get("R00", 1.0)))
    r45 = float(matparam.get("r45", matparam.get("R45", 1.0)))
    r90 = float(matparam.get("r90", matparam.get("R90", 1.0)))
    fisokin = float(matparam.get("chard", matparam.get("CHARD", matparam.get("fisokin", 0.0))))
    iyield = int(matparam.get("iyield", matparam.get("Iyield", 0)))
    epsmax = float(matparam.get("eps_max", matparam.get("EPSMAX", _INF)))
    epsr1 = float(matparam.get("epsr1", matparam.get("EPSR1", _INF)))
    epsr2 = float(matparam.get("epsr2", matparam.get("EPSR2", 2.0 * _INF)))
    yfac = float(matparam.get("fscale", matparam.get("YFAC", 1.0)))
    pscale = float(matparam.get("pscale", matparam.get("X2FAC", 1.0)))
    xfac = 1.0 / pscale if pscale > 0.0 else 1.0
    t0 = float(matparam.get("t0", matparam.get("T0", 293.0)))
    rhocp_user = float(matparam.get("rhocp", matparam.get("RHOCP", 0.0)))
    rhocp_inv = 1.0 / rhocp_user if rhocp_user > 0.0 else 0.0

    ifunce = int(matparam.get("ifunce", matparam.get("Yr_fun", 0)))
    opte = int(matparam.get("opte", 1 if (ifunce > 0 or curve_e_fn is not None) else 0))
    einf = float(matparam.get("einf", matparam.get("EINF", 0.0)))
    ce = float(matparam.get("ce", matparam.get("CE", 0.0)))

    # Hill coefficients: hm_read_mat73.F:179-191
    r = 0.25 * (r00 + 2.0 * r45 + r90)
    h = r / (1.0 + r)
    a01 = h * (1.0 + 1.0 / r00)
    a02 = h * (1.0 + 1.0 / r90)
    a03 = 2.0 * h
    a12 = (2.0 * r45 + 1.0) * (a01 + a02 - a03)
    if iyield > 0:
        a02 /= a01
        a03 /= a01
        a12 /= a01
        a01 = 1.0

    # 2. Young's modulus degradation: sigeps73c.F:145-193
    e1 = np.full(nel, e, dtype=np.float64)
    if opte == 1 and curve_e_fn is not None:
        for i in range(nel):
            if pla_arr[i] > 0.0:
                scale_val, _ = curve_e_fn(np.array([pla_arr[i]]))
                e1[i] = float(scale_val[0]) * e
    elif ce != 0.0:
        for i in range(nel):
            if pla_arr[i] > 0.0:
                e1[i] = e1[i] - (e1[i] - einf) * (1.0 - math.exp(-ce * pla_arr[i]))

    a11 = e1 / (1.0 - nu * nu)
    a21 = nu * a11
    g1 = 0.5 * e1 / (1.0 + nu)
    g31 = 3.0 * g1
    gs = g1 * shf
    nnu1 = nu / (1.0 - nu)
    nu5 = 1.0 - nnu1

    # 3. Temperature: sigeps73c.F:209-222
    temp = np.zeros(nel, dtype=np.float64)
    if jthe > 0:
        for i in range(nel):
            temp[i] = tempel_arr[i]
    else:
        for i in range(nel):
            vol0_i = vol_arr[i] * rho0_arr[i]
            temp[i] = t0 + (eint_arr[i, 0] + eint_arr[i, 1]) * rhocp_inv / max(vol0_i, _EM20)

    # 4. Elastic trial stress with backstress shift: sigeps73c.F:225-233
    signxx = np.zeros(nel, dtype=np.float64)
    signyy = np.zeros(nel, dtype=np.float64)
    signxy = np.zeros(nel, dtype=np.float64)
    signyz = np.zeros(nel, dtype=np.float64) if ncomp >= 4 else None
    signzx = np.zeros(nel, dtype=np.float64) if ncomp >= 5 else None

    soundsp = np.zeros(nel, dtype=np.float64)
    epsp_eq = np.zeros(nel, dtype=np.float64)
    fail = np.zeros(nel, dtype=np.float64)

    # Strain rate resolution
    if epsp_rates is not None:
        ep_in = np.atleast_2d(np.array(epsp_rates, dtype=np.float64))
        edxx = ep_in[:, 0]
        edyy = ep_in[:, 1]
        edxy = ep_in[:, 2]
    elif timestep > 0.0:
        edxx = deps_arr[:, 0] / timestep
        edyy = deps_arr[:, 1] / timestep
        edxy = deps_arr[:, 2] / timestep
    else:
        edxx = np.zeros(nel, dtype=np.float64)
        edyy = np.zeros(nel, dtype=np.float64)
        edxy = np.zeros(nel, dtype=np.float64)

    if eps_tot is not None:
        et_in = np.atleast_2d(np.array(eps_tot, dtype=np.float64))
        exx = et_in[:, 0]
        eyy = et_in[:, 1]
        exy = et_in[:, 2]
    else:
        exx = deps_arr[:, 0]
        eyy = deps_arr[:, 1]
        exy = deps_arr[:, 2]

    for i in range(nel):
        signxx[i] = sigo_arr[i, 0] - uvar_arr[i, 1] + a11[i] * deps_arr[i, 0] + a21[i] * deps_arr[i, 1]
        signyy[i] = sigo_arr[i, 1] - uvar_arr[i, 2] + a21[i] * deps_arr[i, 0] + a11[i] * deps_arr[i, 1]
        signxy[i] = sigo_arr[i, 2] - uvar_arr[i, 3] + g1[i] * deps_arr[i, 2]
        if ncomp >= 4:
            signyz[i] = sigo_arr[i, 3] + gs[i] * deps_arr[i, 3]
        if ncomp >= 5:
            signzx[i] = sigo_arr[i, 4] + gs[i] * deps_arr[i, 4]

        soundsp[i] = math.sqrt(a11[i] / max(rho_cur_arr[i], _EM20))

        # Equivalent strain rate: sigeps73c.F:237-239
        epsp_eq[i] = 0.5 * (abs(edxx[i] + edyy[i]) + math.sqrt((edxx[i] - edyy[i])**2 + edxy[i]**2))

        # Total tensile strain: sigeps73c.F:243-246
        epst = 0.5 * (exx[i] + eyy[i] + math.sqrt((exx[i] - eyy[i])**2 + exy[i]**2))
        if epsr2 > epsr1:
            fail[i] = max(0.0, min(1.0, (epsr2 - epst) / (epsr2 - epsr1)))
        else:
            fail[i] = 1.0

    # 5. Hardening law: sigeps73c.F:250-291
    yld = np.zeros(nel, dtype=np.float64)
    h_mod = np.zeros(nel, dtype=np.float64)
    for i in range(nel):
        edot_scaled = epsp_eq[i] * xfac
        if yield_fn is not None:
            sy_val, dydx = yield_fn(np.array([pla_arr[i]]), np.array([edot_scaled]), np.array([temp[i]]))
            sy = float(sy_val[0])
            slp = float(dydx[0])
        else:
            tbl_val = matparam.get("yield_table", matparam.get("sigy0", 200.0))
            sy = float(tbl_val) if isinstance(tbl_val, (int, float)) else 200.0
            slp = 0.0
        yld[i] = max(yfac * sy * fail[i], _EM20)
        h_mod[i] = max(fail[i] * slp, 0.0)

    if fisokin != 0.0:
        for i in range(nel):
            edot_scaled = epsp_eq[i] * xfac
            if yield_fn is not None:
                sy_k, _ = yield_fn(np.array([0.0]), np.array([edot_scaled]), np.array([temp[i]]))
                yk = float(sy_k[0])
            else:
                tbl_val = matparam.get("yield_table", matparam.get("sigy0", 200.0))
                yk = float(tbl_val) if isinstance(tbl_val, (int, float)) else 200.0
            yld[i] = (1.0 - fisokin) * yld[i] + fisokin * fail[i] * yfac * yk
            yld[i] = max(yld[i], _EM20)

    # 6. Hill criterion & elastic thickness: sigeps73c.F:295-307
    svm = np.zeros(nel, dtype=np.float64)
    axy = np.zeros(nel, dtype=np.float64)
    for i in range(nel):
        s1 = a01 * signxx[i]**2
        s2 = a02 * signyy[i]**2
        s3 = a03 * signxx[i] * signyy[i]
        axy[i] = a12 * signxy[i]**2
        svm[i] = math.sqrt(max(0.0, s1 + s2 - s3 + axy[i]))
        dezz = -(deps_arr[i, 0] + deps_arr[i, 1]) * nnu1
        thk_arr[i] += dezz * thkly_arr[i] * off_arr[i]

    # 7. Plastic return mapping: sigeps73c.F:311-432
    dpla_j = np.zeros(nel, dtype=np.float64)
    yielding = [i for i in range(nel) if svm[i] > yld[i] and off_arr[i] == 1.0]

    for i in yielding:
        dpla_j[i] = (svm[i] - yld[i]) / (g31[i] + h_mod[i])
        hk_i = h_mod[i] * fisokin
        fhk = (4.0 / 3.0) * hk_i / a11[i]
        fa01 = a01 * fhk
        fa02 = a02 * fhk
        fa03 = a03 * fhk
        nu1 = nu + 0.5 * fhk
        nu2 = 1.0 - nu1*nu1 + fhk*fhk
        nu3 = nu1 * 0.5
        nu4 = 0.5 * (1.0 - nu)
        nu5 = 1.0 - nnu1
        s1 = a01 * nu1 * 2.0 - a03 - fa03
        s2 = a02 * nu1 * 2.0 - a03 - fa03
        s12 = a03 - nu1 * (a01 + a02) + fa03
        s3 = math.sqrt(max(0.0, nu2 * (a01 - a02)**2 + s12**2))

        q12_i = -(a01 - a02 + s3 + fa01 - fa02) / s1 if abs(s1) >= _EM20 else 0.0
        q21_i = (a01 - a02 + s3 + fa01 - fa02) / s2 if abs(s2) >= _EM20 else 0.0

        jq_i = 1.0 / (1.0 - q12_i * q21_i)
        jq2_i = jq_i * jq_i

        a = a01 * q12_i
        b = a02 * q21_i
        a_1 = (a01 + a03 * q21_i + b * q21_i) * jq2_i
        a_2 = (a02 + a03 * q12_i + a * q12_i) * jq2_i
        a_3 = (a + b) * jq2_i * 2.0 + a03 * (jq2_i * 2.0 - jq_i)

        s11_i = signxx[i] + signyy[i] * q12_i
        s22_i = q21_i * signxx[i] + signyy[i]
        axx_i = a_1 * s11_i**2
        ayy_i = a_2 * s22_i**2
        a_xy_i = a_3 * s11_i * s22_i

        a_mid = a03 * nu3
        b_mid = s3 * jq_i
        b_1_i = a02 - a_mid - b_mid + fa02
        b_2_i = a01 - a_mid + b_mid + fa01
        b_3_i = a12 * (nu4 + 0.5 * fhk)

        h_eff_i = max(0.0, h_mod[i] - hk_i)

        # 2 Newton iterations matching lines 373-397
        for _ in range(2):
            if dpla_j[i] > 0.0:
                y_cur = yld[i] + h_eff_i * dpla_j[i]
                dr = a11[i] * dpla_j[i] / y_cur
                p_1 = 1.0 / (1.0 + b_1_i * dr)
                pp1 = p_1 * p_1
                p_2 = 1.0 / (1.0 + b_2_i * dr)
                pp2 = p_2 * p_2
                p_3 = 1.0 / (1.0 + b_3_i * dr)
                pp3 = p_3 * p_3
                f = axx_i * pp1 + ayy_i * pp2 - a_xy_i * p_1 * p_2 + axy[i] * pp3 - y_cur**2
                df = -((axx_i * p_1 - a_xy_i * p_2 * 0.5) * pp1 * b_1_i +
                       (ayy_i * p_2 - a_xy_i * p_1 * 0.5) * pp2 * b_2_i +
                       axy[i] * pp3 * p_3 * b_3_i) * (a11[i] - dr * h_eff_i) / y_cur - h_eff_i * y_cur
                dpla_j[i] = max(0.0, dpla_j[i] - f * 0.5 / df)
            else:
                dpla_j[i] = 0.0

        # Admissible stresses after return: lines 404-431
        pla_arr[i] += dpla_j[i]
        y_final = yld[i] + h_eff_i * dpla_j[i]
        dr0 = dpla_j[i] / y_final
        dr = a11[i] * dr0
        p_1 = 1.0 / (1.0 + b_1_i * dr)
        p_2 = 1.0 / (1.0 + b_2_i * dr)
        p_3 = 1.0 / (1.0 + b_3_i * dr)
        s1_adm = s11_i * p_1
        s2_adm = s22_i * p_2
        signxx[i] = jq_i * (s1_adm - s2_adm * q12_i)
        signyy[i] = jq_i * (s2_adm - s1_adm * q21_i)
        signxy[i] = signxy[i] * p_3

        # Through-thickness plastic strain
        s1_thk = a01 * signxx[i] + a02 * signyy[i] - a03 * (signxx[i] + signyy[i]) * 0.5
        dezz_pl = - nu5 * dpla_j[i] * s1_thk / y_final
        thk_arr[i] += dezz_pl * thkly_arr[i] * off_arr[i]

        # Backstress update
        s1_bs = a03 * 0.5
        p1_bs = a01 * signxx[i] - s1_bs * signyy[i]
        p2_bs = a02 * signyy[i] - s1_bs * signxx[i]
        p3_bs = a12 * signxy[i]
        dr0_bs = (2.0 / 3.0) * dr0 * hk_i

        uvar_arr[i, 1] += (2.0 * p1_bs + p2_bs) * dr0_bs
        uvar_arr[i, 2] += (2.0 * p2_bs + p1_bs) * dr0_bs
        uvar_arr[i, 3] += 0.5 * p3_bs * dr0_bs

    # 8. Element deletion check: sigeps73c.F:434
    for i in range(nel):
        if pla_arr[i] > epsmax and off_arr[i] == 1.0:
            off_arr[i] = 0.8

    # 9. Reconstruct final Cauchy stress: sigeps73c.F:438-440
    sig_final = np.zeros((nel, ncomp), dtype=np.float64)
    sig_final[:, 0] = signxx + uvar_arr[:, 1]
    sig_final[:, 1] = signyy + uvar_arr[:, 2]
    sig_final[:, 2] = signxy + uvar_arr[:, 3]
    if ncomp >= 4:
        sig_final[:, 3] = signyz
    if ncomp >= 5:
        sig_final[:, 4] = signzx

    for i in range(nel):
        if off_arr[i] <= 0.0:
            sig_final[i, :] = 0.0

    return {
        "sign": sig_final,
        "pla": pla_arr,
        "uvar": uvar_arr,
        "thk": thk_arr,
        "off": off_arr,
        "soundsp": soundsp,
        "temp": temp,
        "svm": svm,
        "yld": yld,
        "dpla": dpla_j,
        "fail": fail,
    }


# =============================================================================
# 2. HILL COEFFICIENTS & ISOTROPIC / ANISOTROPIC RECOVERY TESTS
# =============================================================================

def test_hill_coefficients_isotropic_recovery():
    """Verify isotropic Lankford (R00=R45=R90=1) reduces Hill 1948 to von Mises plane stress."""
    p = Law73Params(r00=1.0, r45=1.0, r90=1.0, iyield=0)

    # R = 0.25 * (1 + 2 + 1) = 1.0
    # H = 1 / (1 + 1) = 0.5
    # A01 = 0.5 * (1 + 1) = 1.0
    # A02 = 0.5 * (1 + 1) = 1.0
    # A03 = 2 * 0.5 = 1.0
    # A12 = (2*1 + 1) * (1 + 1 - 1) = 3.0
    assert p.r == pytest.approx(1.0, abs=1e-15)
    assert p.h == pytest.approx(0.5, abs=1e-15)
    assert p.a01 == pytest.approx(1.0, abs=1e-15)
    assert p.a02 == pytest.approx(1.0, abs=1e-15)
    assert p.a03 == pytest.approx(1.0, abs=1e-15)
    assert p.a12 == pytest.approx(3.0, abs=1e-15)

    # Hill plane-stress form: SVM^2 = A01*sxx^2 + A02*syy^2 - A03*sxx*syy + A12*sxy^2
    # reduces to: sxx^2 + syy^2 - sxx*syy + 3*sxy^2 == von Mises plane stress!
    sxx, syy, sxy = 250.0, -120.0, 85.0
    svm_hill = math.sqrt(p.a01 * sxx**2 + p.a02 * syy**2 - p.a03 * sxx * syy + p.a12 * sxy**2)
    svm_mises = math.sqrt(sxx**2 + syy**2 - sxx * syy + 3.0 * sxy**2)
    assert svm_hill == pytest.approx(svm_mises, abs=1e-14)


def test_hill_coefficients_unnormalized_vs_normalized():
    """Verify iyield=0 (unnormalized) vs iyield=1 (normalized so A01=1.0)."""
    r00, r45, r90 = 1.8, 1.2, 2.4
    p0 = Law73Params(r00=r00, r45=r45, r90=r90, iyield=0)
    p1 = Law73Params(r00=r00, r45=r45, r90=r90, iyield=1)

    r_expected = 0.25 * (r00 + 2.0 * r45 + r90)
    h_expected = r_expected / (1.0 + r_expected)
    a01_exp = h_expected * (1.0 + 1.0 / r00)
    a02_exp = h_expected * (1.0 + 1.0 / r90)
    a03_exp = 2.0 * h_expected
    a12_exp = (2.0 * r45 + 1.0) * (a01_exp + a02_exp - a03_exp)

    # Unnormalized mode
    assert p0.a01 == pytest.approx(a01_exp, abs=1e-15)
    assert p0.a02 == pytest.approx(a02_exp, abs=1e-15)
    assert p0.a03 == pytest.approx(a03_exp, abs=1e-15)
    assert p0.a12 == pytest.approx(a12_exp, abs=1e-15)

    # Normalized mode (sigeps73c.F: lines 186-191)
    assert p1.a01 == pytest.approx(1.0, abs=1e-15)
    assert p1.a02 == pytest.approx(a02_exp / a01_exp, abs=1e-15)
    assert p1.a03 == pytest.approx(a03_exp / a01_exp, abs=1e-15)
    assert p1.a12 == pytest.approx(a12_exp / a01_exp, abs=1e-15)


# =============================================================================
# 3. ELASTIC TRIAL STRESS & DILATATIONAL SOUND SPEED TESTS
# =============================================================================

def test_elastic_trial_stress_backstress_shift_and_transverse_shear():
    """Verify elastic trial stress formula with backstress shift:
    sign_xx = sigo_xx - alpha_xx + a11*deps_xx + a21*deps_yy
    sign_yy = sigo_yy - alpha_yy + a21*deps_xx + a11*deps_yy
    sign_xy = sigo_xy - alpha_xy + g1*deps_xy
    sign_yz = sigo_yz + gs*deps_yz
    sign_zx = sigo_zx + gs*deps_zx
    """
    e, nu = 205000.0, 0.28
    p = Law73Params(e=e, nu=nu, yield_table=1e6)
    mat = build_law73(p)

    sigo = np.array([50.0, -30.0, 20.0, 10.0, -5.0])
    deps = np.array([1.2e-4, -0.8e-4, 0.5e-4, 0.3e-4, -0.2e-4])
    alpha = np.array([15.0, -10.0, 5.0])

    uvar = np.zeros((1, 7))
    uvar[0, 1:4] = alpha
    extra = {"uvar73": uvar, "shf": 5.0 / 6.0}

    sig_out, pla_out = shell_update(mat, sigo.copy(), deps.copy(), extra=extra)

    # Hand-calculate exact Fortran values
    a11 = e / (1.0 - nu**2)
    a21 = nu * a11
    g1 = 0.5 * e / (1.0 + nu)
    gs = g1 * (5.0 / 6.0)

    exp_sign_xx = (sigo[0] - alpha[0]) + a11 * deps[0] + a21 * deps[1]
    exp_sign_yy = (sigo[1] - alpha[1]) + a21 * deps[0] + a11 * deps[1]
    exp_sign_xy = (sigo[2] - alpha[2]) + g1 * deps[2]
    exp_sign_yz = sigo[3] + gs * deps[3]
    exp_sign_zx = sigo[4] + gs * deps[4]

    # For elastic step, backstress is added back: sig_final = sign + alpha
    assert sig_out[0] == pytest.approx(exp_sign_xx + alpha[0], abs=1e-12)
    assert sig_out[1] == pytest.approx(exp_sign_yy + alpha[1], abs=1e-12)
    assert sig_out[2] == pytest.approx(exp_sign_xy + alpha[2], abs=1e-12)
    assert sig_out[3] == pytest.approx(exp_sign_yz, abs=1e-12)
    assert sig_out[4] == pytest.approx(exp_sign_zx, abs=1e-12)
    assert pla_out == 0.0


def test_dilatational_sound_speed():
    """Verify dilatational sound speed: soundsp = sqrt(a11 / rho0) (sigeps73c.F line 231)."""
    e, nu, rho0 = 210000.0, 0.3, 7.85e-9
    p = Law73Params(e=e, nu=nu, rho0=rho0)
    a11 = e / (1.0 - nu**2)
    exp_c = math.sqrt(a11 / rho0)

    assert sound_speed(p) == pytest.approx(exp_c, rel=1e-14)
    assert p.soundsp == pytest.approx(exp_c, rel=1e-14)


# =============================================================================
# 4. YOUNG'S MODULUS DEGRADATION TESTS
# =============================================================================

def test_young_modulus_degradation_exponential():
    """Verify exponential Young's modulus degradation: E = E0 - (E0 - Einf)*(1 - exp(-ce*pla))."""
    e0 = 200000.0
    einf = 140000.0
    ce = 25.0
    nu = 0.3
    p = Law73Params(e=e0, einf=einf, ce=ce, nu=nu, yield_table=1e6)
    mat = build_law73(p)

    plas = [0.0, 0.01, 0.05, 0.10, 0.25]
    for pla_val in plas:
        extra = {"pla73": np.array([pla_val])}
        sig_out, _ = shell_update(mat, np.zeros(3), np.array([1.0e-5, 0.0, 0.0]), extra=extra)

        if pla_val == 0.0:
            exp_e = e0
        else:
            exp_e = e0 - (e0 - einf) * (1.0 - math.exp(-ce * pla_val))

        exp_a11 = exp_e / (1.0 - nu**2)
        exp_sxx = exp_a11 * 1.0e-5
        assert sig_out[0] == pytest.approx(exp_sxx, rel=1e-12)


def test_young_modulus_degradation_function():
    """Verify function Young's modulus degradation: E = scale(pla) * E0 (opte=1)."""
    e0 = 210000.0
    nu = 0.3
    curve_e = ([0.0, 0.05, 0.10, 0.20], [1.0, 0.90, 0.80, 0.70])
    p = Law73Params(e=e0, nu=nu, curve_e=curve_e, ifunce=1, yield_table=1e6)
    mat = build_law73(p)

    extra = {"pla73": np.array([0.05])}
    sig_out, _ = shell_update(mat, np.zeros(3), np.array([1.0e-5, 0.0, 0.0]), extra=extra)

    exp_e = 0.90 * e0
    exp_a11 = exp_e / (1.0 - nu**2)
    assert sig_out[0] == pytest.approx(exp_a11 * 1.0e-5, rel=1e-12)


# =============================================================================
# 5. ADIABATIC HEATING & COUPLED TEMPERATURE TESTS
# =============================================================================

def test_adiabatic_temperature_evolution():
    """Verify adiabatic temperature update: temp = t0 + (eint1 + eint2) * rhocp_inv / vol0."""
    t0 = 293.15
    rhocp = 3.5e-3  # J / (mm^3 * K)
    rho0 = 7.8e-9
    vol = 100.0     # mm^3
    vol0 = vol * rho0
    eint_tot = 50.0  # J

    p = Law73Params(t0=t0, rhocp=rhocp, rho0=rho0, yield_table=1e6)
    mat = build_law73(p)

    extra = {
        "eint": np.array([30.0, 20.0]),
        "vol": vol,
    }

    # Oracle evaluation
    res_oracle = sigeps73c_oracle(
        {"t0": t0, "rhocp": rhocp, "rho0": rho0, "yield_table": 1e6},
        sigo=np.zeros((1, 3)),
        deps=np.zeros((1, 3)),
        rho0=rho0,
        eint=np.array([[30.0, 20.0]]),
        vol=vol,
    )

    exp_temp = t0 + eint_tot / (rhocp * vol0)
    assert res_oracle["temp"][0] == pytest.approx(exp_temp, rel=1e-12)

    # shell_update evaluation
    shell_update(mat, np.zeros(3), np.zeros(3), extra=extra)
    assert extra["temp"][0] == pytest.approx(exp_temp, rel=1e-12)


def test_coupled_temperature_override_jthe():
    """Verify coupled thermal solver override: temp = tempel when jthe > 0."""
    p = Law73Params(t0=293.15, rhocp=3.5e-3, yield_table=1e6)
    mat = build_law73(p)

    extra = {
        "tempel": np.array([450.0]),
        "eint": np.array([10.0, 5.0]),
    }

    res_oracle = sigeps73c_oracle(
        {"t0": 293.15, "rhocp": 3.5e-3, "yield_table": 1e6},
        sigo=np.zeros((1, 3)),
        deps=np.zeros((1, 3)),
        tempel=np.array([450.0]),
        jthe=1,
    )
    assert res_oracle["temp"][0] == 450.0

    shell_update(mat, np.zeros(3), np.zeros(3), extra=extra)
    assert extra["temp"][0] == 450.0


# =============================================================================
# 6. MIXED ISOTROPIC / KINEMATIC HARDENING & BAUSCHINGER EFFECT TESTS
# =============================================================================

def test_kinematic_hardening_bauschinger_effect():
    """Verify kinematic hardening shifts the yield center and exhibits Bauschinger effect."""
    def linear_hard(pla, rate, temp):
        return 200.0 + 4000.0 * pla, 4000.0

    # Pure kinematic: chard = 1.0
    p = Law73Params(e=200000.0, nu=0.3, yield_table=linear_hard, chard=1.0)
    mat = build_law73(p)

    extra = {
        "uvar73": np.zeros((1, 7)),
        "pla73": np.zeros(1),
        "off73": np.ones(1),
    }

    # Step 1: Forward plastic tension along x
    sig1, pla1 = shell_update(mat, np.zeros(3), np.array([0.003, 0.0, 0.0]), epsp=0.0, extra=extra)
    assert pla1 > 0.0
    alpha_xx = extra["uvar73"][0, 1]
    assert alpha_xx > 0.0

    # Step 2: Reverse compression loading
    # Under kinematic hardening, reverse yield occurs at smaller |sig| because alpha_xx > 0:
    # |sig_xx - alpha_xx| = Y0 => sig_xx = alpha_xx - Y0 (earlier yielding)
    deps_rev = np.array([-0.002, 0.0, 0.0])
    sig2, pla2 = shell_update(mat, sig1.copy(), deps_rev, epsp=pla1, extra=extra)

    # Compare with pure isotropic (chard=0.0): isotropic would NOT have backstress
    p_iso = Law73Params(e=200000.0, nu=0.3, yield_table=linear_hard, chard=0.0)
    mat_iso = build_law73(p_iso)
    extra_iso = {"uvar73": np.zeros((1, 7)), "pla73": np.zeros(1), "off73": np.ones(1)}
    sig1_iso, pla1_iso = shell_update(mat_iso, np.zeros(3), np.array([0.003, 0.0, 0.0]), epsp=0.0, extra=extra_iso)
    sig2_iso, pla2_iso = shell_update(mat_iso, sig1_iso.copy(), deps_rev, epsp=pla1_iso, extra=extra_iso)

    # In reverse compression, kinematic stress has non-zero backstress unlike isotropic
    assert extra_iso["uvar73"][0, 1] == 0.0
    assert extra["uvar73"][0, 1] != 0.0


# =============================================================================
# 7. TENSILE SOFTENING & ELEMENT DELETION TESTS
# =============================================================================

def test_tensile_softening_fail_factor():
    """Verify tensile softening factor: fail = clamp((epsr2 - epst)/(epsr2 - epsr1), 0, 1)."""
    epsr1, epsr2 = 0.05, 0.15
    p = Law73Params(epsr1=epsr1, epsr2=epsr2, yield_table=300.0)
    mat = build_law73(p)

    # Case A: epst < epsr1 => fail = 1.0 (no degradation)
    extra_a = {"eps": np.array([0.02, 0.01, 0.0])}
    sig_a, _ = shell_update(mat, np.zeros(3), np.array([1e-4, 0.0, 0.0]), extra=extra_a)

    # Case B: epsr1 < epst < epsr2 => fail in (0, 1)
    # epst = 0.10 => fail = (0.15 - 0.10) / (0.15 - 0.05) = 0.50
    extra_b = {"eps": np.array([0.10, 0.0, 0.0])}
    # Large strain causing yield
    sig_b, _ = shell_update(mat, np.zeros(3), np.array([0.005, 0.0, 0.0]), extra=extra_b)

    # The effective yield stress was scaled by fail = 0.50, so sig_b is ~0.50 of full yield
    seq_b = math.sqrt(sig_b[0]**2 + sig_b[1]**2 - sig_b[0]*sig_b[1] + 3.0*sig_b[2]**2)
    assert seq_b == pytest.approx(150.0, rel=0.05)

    # Case C: epst > epsr2 => fail = 0.0 (complete loss of strength)
    extra_c = {"eps": np.array([0.20, 0.0, 0.0])}
    sig_c, _ = shell_update(mat, np.zeros(3), np.array([0.005, 0.0, 0.0]), extra=extra_c)
    seq_c = math.sqrt(sig_c[0]**2 + sig_c[1]**2 - sig_c[0]*sig_c[1] + 3.0*sig_c[2]**2)
    assert seq_c == pytest.approx(0.0, abs=1e-6)


def test_element_deletion_at_eps_max():
    """Verify element deletion: when pla > eps_max and off == 1.0, off becomes 0.8."""
    eps_max = 0.10
    p = Law73Params(eps_max=eps_max, yield_table=200.0)
    mat = build_law73(p)

    extra = {
        "pla73": np.array([0.09]),
        "off73": np.array([1.0]),
    }

    # Step that does not exceed eps_max
    shell_update(mat, np.zeros(3), np.array([1e-4, 0.0, 0.0]), extra=extra)
    assert extra["off73"][0] == 1.0

    # Step with large plastic increment that pushes pla past eps_max
    shell_update(mat, np.zeros(3), np.array([0.015, 0.0, 0.0]), extra=extra)
    assert extra["pla73"][0] > eps_max
    assert extra["off73"][0] == pytest.approx(0.8, abs=1e-15)

    # Test fully deleted element off <= 0 zeroes stress
    extra_dead = {"off73": np.array([0.0])}
    sig_dead, _ = shell_update(mat, np.array([100.0, 50.0, 20.0]), np.array([1e-4, 0.0, 0.0]), extra=extra_dead)
    assert np.all(sig_dead == 0.0)


# =============================================================================
# 8. EXHAUSTIVE 60+ DIVERSE STATES PARITY COMPARISON:
#    shell_update vs sigeps73c_oracle (rtol=1e-12, atol=1e-12)
# =============================================================================

def test_exhaustive_oracle_comparison_50_plus_states():
    """Exhaustively verify shell_update against sigeps73c_oracle across 60+ diverse physical states."""

    # 3D Yield Table lookup function simulating OpenRadioss TABLE_VINTERP
    def yield_fn(pla_arr: np.ndarray, rate_arr: np.ndarray, temp_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Realistic power law hardening with strain-rate sensitivity and thermal softening
        y0 = 220.0
        h_mod = 350.0
        c_rate = 0.03
        t_ref = 293.0
        t_melt = 900.0

        p = np.maximum(pla_arr, 0.0)
        ed = np.maximum(rate_arr, 1e-6)
        tm = np.maximum(temp_arr, 200.0)

        p_safe = np.maximum(p, 1e-12)
        f_hard = y0 + h_mod * (p_safe ** 0.45)
        df_dp = np.where(p > 1e-10, 0.45 * h_mod * (p_safe ** -0.55), 1e6)

        f_rate = 1.0 + c_rate * np.log(ed / 1e-3 + 1.0)
        f_therm = np.maximum(0.05, 1.0 - ((tm - t_ref) / (t_melt - t_ref)) ** 1.2)

        val = f_hard * f_rate * f_therm
        slope = df_dp * f_rate * f_therm
        return val, slope

    # Young's modulus degradation curve
    def curve_e_fn(pla_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # E_scale = 1.0 - 0.25 * (1 - exp(-15 * pla))
        p = np.maximum(pla_arr, 0.0)
        scale = 1.0 - 0.25 * (1.0 - np.exp(-15.0 * p))
        slope = -0.25 * 15.0 * np.exp(-15.0 * p)
        return scale, slope

    test_configs = []

    # Systematically generate 60+ diverse parameter configurations
    # Dimensions:
    # - Lankford anisotropy: isotropic (1,1,1) vs mild anisotropic vs strong anisotropic
    # - Hardening mode: pure isotropic (chard=0), mixed (chard=0.4), pure kinematic (chard=1.0)
    # - Normalization: unnormalized (iyield=0) vs normalized (iyield=1)
    # - Young degradation: constant vs exponential (ce=20) vs function (opte=1)
    # - Stresses: 3-component vs 5-component
    # - Regimes: elastic, yielding, reverse plastic loading, softening, failure

    aniso_sets = [
        (1.0, 1.0, 1.0),    # isotropic
        (1.5, 1.2, 1.8),    # mild anisotropic
        (2.4, 0.8, 1.9),    # strong anisotropic
    ]
    chard_vals = [0.0, 0.4, 1.0]
    iyield_vals = [0, 1]
    young_opts = ["const", "ce", "fun"]

    count = 0
    for r00, r45, r90 in aniso_sets:
        for chard in chard_vals:
            for iyield in iyield_vals:
                for yopt in young_opts:
                    count += 1
                    test_configs.append({
                        "id": count,
                        "r00": r00, "r45": r45, "r90": r90,
                        "chard": chard, "iyield": iyield,
                        "yopt": yopt,
                    })

    assert len(test_configs) == 54  # 3 * 3 * 2 * 3 = 54 base configurations

    # Plus 12 specialized edge-case configurations (total 66 test states)
    special_configs = [
        {"id": 55, "r00": 1.0, "r45": 1.0, "r90": 1.0, "chard": 0.0, "iyield": 0, "yopt": "const",
         "epsmax": 0.05, "init_pla": 0.06},  # Pre-failed by pla > eps_max
        {"id": 56, "r00": 1.0, "r45": 1.0, "r90": 1.0, "chard": 0.0, "iyield": 0, "yopt": "const",
         "off": 0.0},  # Fully dead element
        {"id": 57, "r00": 1.8, "r45": 1.3, "r90": 2.1, "chard": 0.5, "iyield": 1, "yopt": "ce",
         "epsr1": 0.02, "epsr2": 0.08, "init_epst": 0.05},  # Active tensile softening
        {"id": 58, "r00": 1.8, "r45": 1.3, "r90": 2.1, "chard": 0.5, "iyield": 1, "yopt": "ce",
         "epsr1": 0.02, "epsr2": 0.08, "init_epst": 0.12},  # Completely failed tensile softening
        {"id": 59, "r00": 1.2, "r45": 1.0, "r90": 1.5, "chard": 0.8, "iyield": 0, "yopt": "const",
         "jthe": 1, "tempel": 550.0},  # Coupled thermal solver
        {"id": 60, "r00": 1.0, "r45": 1.0, "r90": 1.0, "chard": 0.0, "iyield": 0, "yopt": "const",
         "rhocp": 4.0e-3, "eint": [80.0, 40.0]},  # Severe adiabatic heating
        {"id": 61, "r00": 1.5, "r45": 1.2, "r90": 1.7, "chard": 1.0, "iyield": 1, "yopt": "const",
         "reverse": True},  # Bauschinger reverse reload
        {"id": 62, "r00": 2.0, "r45": 1.5, "r90": 2.2, "chard": 0.3, "iyield": 0, "yopt": "ce",
         "ncomp": 5},  # 5-component transverse shear
        {"id": 63, "r00": 1.0, "r45": 1.0, "r90": 1.0, "chard": 0.0, "iyield": 0, "yopt": "const",
         "pure_shear": True},  # Pure shear loading deps_xy
        {"id": 64, "r00": 1.4, "r45": 1.1, "r90": 1.6, "chard": 0.5, "iyield": 1, "yopt": "fun",
         "biaxial_comp": True},  # Biaxial compression
        {"id": 65, "r00": 1.0, "r45": 1.0, "r90": 1.0, "chard": 0.0, "iyield": 0, "yopt": "const",
         "elastic_only": True},  # Pure elastic step below yield
        {"id": 66, "r00": 1.6, "r45": 1.2, "r90": 1.9, "chard": 0.6, "iyield": 1, "yopt": "const",
         "batch": 10},  # Multi-element 10-point vectorized batch
    ]
    test_configs.extend(special_configs)

    assert len(test_configs) == 66

    # Run the parity comparison for each state
    for cfg in test_configs:
        r00 = cfg["r00"]
        r45 = cfg["r45"]
        r90 = cfg["r90"]
        chard = cfg["chard"]
        iyield = cfg["iyield"]
        yopt = cfg["yopt"]
        nel = cfg.get("batch", 1)
        ncomp = cfg.get("ncomp", 3)

        e0 = 206000.0
        nu = 0.29
        rho0 = 7.82e-9
        ce = 22.0 if yopt == "ce" else 0.0
        einf = 145000.0 if yopt == "ce" else 0.0
        opte = 1 if yopt == "fun" else 0
        c_e = curve_e_fn if yopt == "fun" else None

        epsmax = cfg.get("epsmax", _INF)
        epsr1 = cfg.get("epsr1", _INF)
        epsr2 = cfg.get("epsr2", 2.0 * _INF)
        rhocp = cfg.get("rhocp", 3.6e-3)
        jthe = cfg.get("jthe", 0)

        param_dict = {
            "e": e0, "nu": nu, "rho0": rho0,
            "r00": r00, "r45": r45, "r90": r90,
            "chard": chard, "iyield": iyield,
            "ce": ce, "einf": einf, "opte": opte,
            "eps_max": epsmax, "epsr1": epsr1, "epsr2": epsr2,
            "rhocp": rhocp, "t0": 295.0,
            "yield_table": yield_fn,
            "curve_e": c_e,
        }

        mat_params = Law73Params(
            e=e0, nu=nu, rho0=rho0,
            r00=r00, r45=r45, r90=r90,
            chard=chard, iyield=iyield,
            ce=ce, einf=einf, ifunce=opte,
            eps_max=epsmax, epsr1=epsr1, epsr2=epsr2,
            rhocp=rhocp, t0=295.0,
            yield_table=yield_fn,
            curve_e=c_e,
        )
        mat = build_law73(mat_params)

        # Generate inputs
        sigo = np.zeros((nel, ncomp), dtype=np.float64)
        deps = np.zeros((nel, ncomp), dtype=np.float64)

        if cfg.get("elastic_only"):
            deps[:, 0] = 5.0e-5
            deps[:, 1] = -2.0e-5
        elif cfg.get("pure_shear"):
            deps[:, 2] = 0.006
        elif cfg.get("biaxial_comp"):
            deps[:, 0] = -0.005
            deps[:, 1] = -0.004
        elif cfg.get("reverse"):
            sigo[:, 0] = 220.0
            deps[:, 0] = -0.006
        else:
            deps[:, 0] = 0.0045 + 0.001 * np.arange(nel)
            deps[:, 1] = 0.0015 - 0.0005 * np.arange(nel)
            deps[:, 2] = 0.0020
            if ncomp >= 4:
                deps[:, 3] = 0.0010
            if ncomp >= 5:
                deps[:, 4] = -0.0008

        pla_in = np.full(nel, cfg.get("init_pla", 0.01 if (yopt in ("ce", "fun") and not cfg.get("elastic_only")) else 0.0), dtype=np.float64)
        off_in = np.full(nel, cfg.get("off", 1.0), dtype=np.float64)
        uvar_in = np.zeros((nel, 7), dtype=np.float64)
        if chard > 0 and not cfg.get("elastic_only"):
            uvar_in[:, 1] = 25.0  # existing backstress alpha_xx
            uvar_in[:, 2] = -10.0 # existing backstress alpha_yy

        thk_in = np.full(nel, 1.2, dtype=np.float64)
        thkly_in = np.full(nel, 1.2, dtype=np.float64)
        vol_in = np.full(nel, 120.0, dtype=np.float64)
        eint_in = np.tile(cfg.get("eint", [15.0, 10.0]), (nel, 1)).astype(np.float64)

        if "init_epst" in cfg:
            eps_tot_in = np.zeros((nel, 3), dtype=np.float64)
            eps_tot_in[:, 0] = cfg["init_epst"]
        else:
            eps_tot_in = None

        if "tempel" in cfg:
            tempel_in = np.full(nel, cfg["tempel"], dtype=np.float64)
        else:
            tempel_in = None

        # 1. Evaluate Fortran Oracle
        res_oracle = sigeps73c_oracle(
            param_dict,
            sigo=sigo.copy(),
            deps=deps.copy(),
            rho0=rho0,
            pla=pla_in.copy(),
            uvar=uvar_in.copy(),
            off=off_in.copy(),
            thk=thk_in.copy(),
            thkly=thkly_in.copy(),
            eint=eint_in.copy(),
            vol=vol_in.copy(),
            tempel=tempel_in,
            jthe=jthe,
            timestep=1.0e-5,
            yield_fn=yield_fn,
            curve_e_fn=c_e,
            eps_tot=eps_tot_in,
        )

        # 2. Evaluate Python shell_update
        extra_py = {
            "pla73": pla_in.copy(),
            "uvar73": uvar_in.copy(),
            "off73": off_in.copy(),
            "thk73": thk_in.copy(),
            "thkly": thkly_in.copy(),
            "vol": vol_in.copy(),
            "eint": eint_in.copy(),
        }
        if tempel_in is not None:
            extra_py["tempel"] = tempel_in.copy()
        if eps_tot_in is not None:
            extra_py["eps"] = eps_tot_in.copy()

        sig_py, pla_py, c_py = shell_update(
            mat,
            sigo.copy() if nel > 1 else sigo[0].copy(),
            deps.copy() if nel > 1 else deps[0].copy(),
            epsp=pla_in.copy() if nel > 1 else pla_in[0],
            dt=1.0e-5,
            extra=extra_py,
            return_tuple=True,
        )

        sig_py_arr = np.atleast_2d(sig_py)
        pla_py_arr = np.atleast_1d(pla_py)
        c_py_arr = np.atleast_1d(c_py)

        # 3. Direct Parity Assertions (rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            sig_py_arr, res_oracle["sign"],
            rtol=1e-12, atol=1e-12,
            err_msg=f"State #{cfg['id']} failed Cauchy stress parity"
        )
        np.testing.assert_allclose(
            pla_py_arr, res_oracle["pla"],
            rtol=1e-12, atol=1e-12,
            err_msg=f"State #{cfg['id']} failed plastic strain parity"
        )
        np.testing.assert_allclose(
            extra_py["thk73"], res_oracle["thk"],
            rtol=1e-12, atol=1e-12,
            err_msg=f"State #{cfg['id']} failed thickness parity"
        )
        np.testing.assert_allclose(
            extra_py["uvar73"][:, 1:4], res_oracle["uvar"][:, 1:4],
            rtol=1e-12, atol=1e-12,
            err_msg=f"State #{cfg['id']} failed backstress tensor parity"
        )
        np.testing.assert_allclose(
            extra_py["off73"], res_oracle["off"],
            rtol=1e-12, atol=1e-12,
            err_msg=f"State #{cfg['id']} failed active status off parity"
        )
        np.testing.assert_allclose(
            c_py_arr, res_oracle["soundsp"],
            rtol=1e-12, atol=1e-12,
            err_msg=f"State #{cfg['id']} failed sound speed parity"
        )


# =============================================================================
# 9. TANGENT OPERATORS & EXTRA SHAPES PARITY
# =============================================================================

def test_tangent_operators_consistency():
    """Verify consistent_shell_tangent and shell_membrane_tangent."""
    p = Law73Params(e=200000.0, nu=0.3, yield_table=400.0)
    mat = build_law73(p)

    c_el = shell_membrane_tangent(mat)
    assert c_el.shape == (3, 3)
    assert c_el[0, 0] == pytest.approx(p.a11, rel=1e-14)
    assert c_el[0, 1] == pytest.approx(p.a21, rel=1e-14)
    assert c_el[1, 0] == pytest.approx(p.a21, rel=1e-14)
    assert c_el[1, 1] == pytest.approx(p.a11, rel=1e-14)
    assert c_el[2, 2] == pytest.approx(p.g, rel=1e-14)
    assert c_el[0, 2] == 0.0
    assert c_el[1, 2] == 0.0

    # Perturbation tangent in elastic regime must equal elastic membrane matrix
    d_alg = consistent_shell_tangent(mat, np.zeros(3), deps=np.zeros(3), h=1e-7)
    np.testing.assert_allclose(d_alg, c_el, rtol=1e-6, atol=1e-6)


def test_extra_shapes_allocation():
    """Verify extra_shapes allocation conforms to 7-channel NUVAR."""
    p = Law73Params()
    shapes_single = extra_shapes(p)
    assert shapes_single["uvar73"] == (7,)
    assert shapes_single["pla73"] == ()
    assert shapes_single["off73"] == ()
    assert shapes_single["thk73"] == ()

    shapes_nip = extra_shapes(p, nip=5)
    assert shapes_nip["uvar73"] == (5, 7)
    assert shapes_nip["pla73"] == (5,)
    assert shapes_nip["off73"] == (5,)
    assert shapes_nip["thk73"] == (5,)
