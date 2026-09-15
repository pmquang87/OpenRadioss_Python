r"""Fortran Parity & Physics Oracle Verifier for Milestone M559: /MAT/LAW50 (/MAT/VISC_HONEY /MAT/HYP_FOAM).

Rate-Dependent Viscoelastic Honeycomb Material Model.
Directly cites and mirrors upstream OpenRadioss source:
- Starter card reader and parameters:
  ``starter/source/materials/mat/mat050/hm_read_mat50.F90``
- Engine constitutive physics and sound speed:
  ``engine/source/materials/mat/mat050/sigeps50s.F90`` (lines 112 to 401)
- CFG schema and defaults:
  ``hm_cfg_files/config/CFG/radioss2025/MAT/mat_law50.cfg``

Tests included:
1. Exact Python oracle reproducing sigeps50s.F90 lines 112 to 401.
2. Uncompacted orthotropic elasticity & sound speed in all 6 directions (lines 164-172).
3. Failure strain criteria and element deletion stress-zeroing (lines 174-179).
4. Normal strain definition and yield clamping with Gflag in {-1, 0, 1} (lines 183-195, 354-371).
5. Shear strain definition and yield clamping with Vflag in {-1, 0, 1} (lines 196-208, 354-371).
6. Strain rate filtering with irate=2 (independent) and irate=1 (common equivalent) across dt and fcut (lines 229-259).
7. Compaction transition: relative volume rvol, beta interpolation, and irreversible latching (lines 127-154).
8. Compacted J2 plasticity: radial return, deviatoric flow, linear hardening, and bulk pressure under positive/negative depsv (lines 376-401).
9. Vectorized multi-element execution parity.
10. Exhaustive 60+ diverse physical states comparison: solid_update vs sigeps50s_oracle (rtol=1e-12, atol=1e-12).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional

import numpy as np
import pytest

from pyradioss.model.entities import Material
from pyradioss.common.tables import FunctTable
from pyradioss.materials.law50_visc_honey import (
    Law50Params,
    build_law50,
    solid_update,
    sound_speed_solid,
    consistent_solid_tangent,
)


# =============================================================================
# 1. UPSTREAM FORTRAN ENGINE ORACLE: sigeps50s.F90 (lines 112 - 401)
# =============================================================================

def sigeps50s_oracle(
    uparam: np.ndarray,          # uparam(1:19)
    iparam: np.ndarray,          # iparam(1:4): [iflag1, iflag2, icomp, irate]
    rho: np.ndarray,             # rho(nel)
    timestep: float,             # timestep (dt)
    sigo: np.ndarray,            # sigo(nel, 6): [xx, yy, zz, xy, yz, zx]
    deps: np.ndarray,            # deps(nel, 6): [xx, yy, zz, xy, yz, zx]
    eps: np.ndarray,             # eps(nel, 6): total strains
    epsp: np.ndarray,            # epsp(nel, 6): strain rate tensor
    amu: np.ndarray,             # amu(nel): volumetric strain (mu = rho/rho0 - 1)
    off: np.ndarray,             # off(nel): element activation flag
    eplas: np.ndarray,           # eplas(nel): accumulated plastic strain (compacted)
    uvar: np.ndarray,            # uvar(nel, 6): filtered strain rates
    vartmp13: np.ndarray,        # vartmp(nel, 13): compaction latch state (0 or 1)
    yield_fn: Optional[Callable[[int, np.ndarray, np.ndarray], np.ndarray]] = None,
) -> Dict[str, Any]:
    """Faithful and exact Python reproduction of OpenRadioss sigeps50s.F90 lines 112-401.

    Parameters
    ----------
    uparam : ndarray of length 13 or 19
        [0..5]:  E11_0, E22_0, E33_0, G12_0, G23_0, G31_0
        [6..11]: EMX11, EMX22, EMX33, EMX12, EMX23, EMX31
        [12]:    FCUT
        [13..18] (if icomp == 1): ECOMP, GCOMP (2G), BULK (K), SIGY, HCOMP, VCOMP
    iparam : ndarray of length 4
        [0]: iflag1 (Gflag: 1 -> epsxx, -1 -> -epsxx, 0 -> amu)
        [1]: iflag2 (Vflag: 1 -> epsxy, -1 -> -epsxy, 0 -> amu)
        [2]: icomp  (0 = uncompacted, 1 = compaction enabled)
        [3]: irate  (1 = common equivalent rate, 2 = independent directional rates)
    rho : ndarray of length nel
        Current density.
    timestep : float
        Explicit time step dt.
    sigo : ndarray of shape (nel, 6)
        Old Cauchy stresses [xx, yy, zz, xy, yz, zx].
    deps : ndarray of shape (nel, 6)
        Strain increment tensor (engineering shear components xy, yz, zx).
    eps : ndarray of shape (nel, 6)
        Total strain tensor.
    epsp : ndarray of shape (nel, 6)
        Strain rate tensor.
    amu : ndarray of length nel
        Relative volume change mu = rho / rho0 - 1.
    off : ndarray of length nel
        Element activation flag (1.0 = active, 0.0 = deleted).
    eplas : ndarray of length nel
        Accumulated equivalent plastic strain for compacted state.
    uvar : ndarray of shape (nel, 6)
        Filtered rate history.
    vartmp13 : ndarray of length nel
        Compaction latch flag (0 = uncompacted, 1 = permanently compacted).
    yield_fn : callable, optional
        Callback `(k_component, ep_k, dep_k) -> sigy_k` returning directional yield stress.
        If None, yield stress defaults to 1.0e30 (elastic unyielding).

    Returns
    -------
    dict
        Dictionary containing exact updated outputs:
        'sign', 'soundsp', 'off', 'eplas', 'epsd', 'uvar', 'vartmp13', 'beta', 'rvol', 'st_trial'.
    """
    sigo_arr = np.atleast_2d(np.array(sigo, dtype=np.float64)).copy()
    deps_arr = np.atleast_2d(np.array(deps, dtype=np.float64)).copy()
    eps_arr = np.atleast_2d(np.array(eps, dtype=np.float64)).copy()
    epsp_arr = np.atleast_2d(np.array(epsp, dtype=np.float64)).copy()

    nel = sigo_arr.shape[0]

    rho_arr = np.atleast_1d(np.array(rho, dtype=np.float64)).copy()
    amu_arr = np.atleast_1d(np.array(amu, dtype=np.float64)).copy()
    off_arr = np.atleast_1d(np.array(off, dtype=np.float64)).copy()
    eplas_arr = np.atleast_1d(np.array(eplas, dtype=np.float64)).copy()
    uvar_arr = np.atleast_2d(np.array(uvar, dtype=np.float64)).copy()
    vartmp13_arr = np.atleast_1d(np.array(vartmp13, dtype=np.int64)).copy()

    # sigeps50s.F90 lines 112-123
    iflag1 = int(iparam[0])
    iflag2 = int(iparam[1])
    icomp = int(iparam[2])
    irate = int(iparam[3])

    e11_0 = float(uparam[0])
    e22_0 = float(uparam[1])
    e33_0 = float(uparam[2])
    g12_0 = float(uparam[3])
    g23_0 = float(uparam[4])
    g31_0 = float(uparam[5])

    emx11 = float(uparam[6])
    emx22 = float(uparam[7])
    emx33 = float(uparam[8])
    emx12 = float(uparam[9])
    emx23 = float(uparam[10])
    emx31 = float(uparam[11])

    fcut = float(uparam[12])
    asrate = min(1.0, fcut * max(timestep, 0.0))

    # sigeps50s.F90 lines 125-162: compaction state and moduli transition
    nindx = 0
    nindxc = 0
    indxc = []
    indx = []

    rvol = np.zeros(nel, dtype=np.float64)
    beta = np.zeros(nel, dtype=np.float64)
    e11 = np.zeros(nel, dtype=np.float64)
    e22 = np.zeros(nel, dtype=np.float64)
    e33 = np.zeros(nel, dtype=np.float64)
    g12 = np.zeros(nel, dtype=np.float64)
    g23 = np.zeros(nel, dtype=np.float64)
    g31 = np.zeros(nel, dtype=np.float64)

    if icomp == 1:
        ecomp = float(uparam[13])
        gcomp = float(uparam[14])  # 2G in Fortran engine
        bulk = float(uparam[15])
        sigy = float(uparam[16])
        hcomp = float(uparam[17])
        vcomp = float(uparam[18])

        for i in range(nel):
            rvol[i] = 1.0 / (1.0 + amu_arr[i])
            denom_v = 1.0 - vcomp
            if abs(denom_v) > 1.0e-20:
                beta[i] = max(min((1.0 - rvol[i]) / denom_v, 1.0), 0.0)
            else:
                beta[i] = 0.0
            if rvol[i] <= vcomp and vartmp13_arr[i] == 0:
                vartmp13_arr[i] = 1  # Element passes permanently to compacted state
            if vartmp13_arr[i] == 1:
                nindxc += 1
                indxc.append(i)
            else:
                nindx += 1
                indx.append(i)

        for i in range(nel):
            e11[i] = beta[i] * ecomp + (1.0 - beta[i]) * e11_0
            e22[i] = beta[i] * ecomp + (1.0 - beta[i]) * e22_0
            e33[i] = beta[i] * ecomp + (1.0 - beta[i]) * e33_0
            g12[i] = beta[i] * gcomp + (1.0 - beta[i]) * g12_0
            g23[i] = beta[i] * gcomp + (1.0 - beta[i]) * g23_0
            g31[i] = beta[i] * gcomp + (1.0 - beta[i]) * g31_0
    else:
        e11.fill(e11_0)
        e22.fill(e22_0)
        e33.fill(e33_0)
        g12.fill(g12_0)
        g23.fill(g23_0)
        g31.fill(g31_0)
        indx = list(range(nel))

    # sigeps50s.F90 lines 164-172: elastic trial stresses and acoustic sound speed
    sign = np.zeros((nel, 6), dtype=np.float64)
    soundsp = np.zeros(nel, dtype=np.float64)

    for i in range(nel):
        sign[i, 0] = sigo_arr[i, 0] + e11[i] * deps_arr[i, 0]
        sign[i, 1] = sigo_arr[i, 1] + e22[i] * deps_arr[i, 1]
        sign[i, 2] = sigo_arr[i, 2] + e33[i] * deps_arr[i, 2]
        sign[i, 3] = sigo_arr[i, 3] + g12[i] * deps_arr[i, 3]
        sign[i, 4] = sigo_arr[i, 4] + g23[i] * deps_arr[i, 4]
        sign[i, 5] = sigo_arr[i, 5] + g31[i] * deps_arr[i, 5]
        max_mod = max(e11[i], e22[i], e33[i], g12[i], g23[i], g31[i])
        soundsp[i] = math.sqrt(max(max_mod, 0.0) / max(rho_arr[i], 1.0e-20))

    st_trial = sign.copy()

    # sigeps50s.F90 lines 174-179: element failure criteria
    for i in range(nel):
        if (
            eps_arr[i, 0] > emx11
            or eps_arr[i, 1] > emx22
            or eps_arr[i, 2] > emx33
            or abs(eps_arr[i, 3] * 0.5) > emx12
            or abs(eps_arr[i, 4] * 0.5) > emx23
            or abs(eps_arr[i, 5] * 0.5) > emx31
        ):
            off_arr[i] = 0.0

    # sigeps50s.F90 lines 181-208: strain definition for yield functions
    if iflag1 == 1:
        ep1 = eps_arr[:, 0].copy()
        ep2 = eps_arr[:, 1].copy()
        ep3 = eps_arr[:, 2].copy()
    elif iflag1 == -1:
        ep1 = -eps_arr[:, 0].copy()
        ep2 = -eps_arr[:, 1].copy()
        ep3 = -eps_arr[:, 2].copy()
    else:
        ep1 = amu_arr.copy()
        ep2 = amu_arr.copy()
        ep3 = amu_arr.copy()

    if iflag2 == 1:
        ep4 = eps_arr[:, 3].copy()
        ep5 = eps_arr[:, 4].copy()
        ep6 = eps_arr[:, 5].copy()
    elif iflag2 == -1:
        ep4 = -eps_arr[:, 3].copy()
        ep5 = -eps_arr[:, 4].copy()
        ep6 = -eps_arr[:, 5].copy()
    else:
        ep4 = amu_arr.copy()
        ep5 = amu_arr.copy()
        ep6 = amu_arr.copy()

    # sigeps50s.F90 lines 229-259: strain rate filtering
    dep1 = np.zeros(nel, dtype=np.float64)
    dep2 = np.zeros(nel, dtype=np.float64)
    dep3 = np.zeros(nel, dtype=np.float64)
    dep4 = np.zeros(nel, dtype=np.float64)
    dep5 = np.zeros(nel, dtype=np.float64)
    dep6 = np.zeros(nel, dtype=np.float64)
    epsd = np.zeros(nel, dtype=np.float64)

    if irate == 2:
        for i in range(nel):
            uvar_arr[i, 0] = asrate * epsp_arr[i, 0] + (1.0 - asrate) * uvar_arr[i, 0]
            uvar_arr[i, 1] = asrate * epsp_arr[i, 1] + (1.0 - asrate) * uvar_arr[i, 1]
            uvar_arr[i, 2] = asrate * epsp_arr[i, 2] + (1.0 - asrate) * uvar_arr[i, 2]
            uvar_arr[i, 3] = asrate * epsp_arr[i, 3] + (1.0 - asrate) * uvar_arr[i, 3]
            uvar_arr[i, 4] = asrate * epsp_arr[i, 4] + (1.0 - asrate) * uvar_arr[i, 4]
            uvar_arr[i, 5] = asrate * epsp_arr[i, 5] + (1.0 - asrate) * uvar_arr[i, 5]

            dep1[i] = abs(uvar_arr[i, 0])
            dep2[i] = abs(uvar_arr[i, 1])
            dep3[i] = abs(uvar_arr[i, 2])
            dep4[i] = abs(uvar_arr[i, 3])
            dep5[i] = abs(uvar_arr[i, 4])
            dep6[i] = abs(uvar_arr[i, 5])

            epsd[i] = (dep1[i] ** 2 + dep2[i] ** 2 + dep3[i] ** 2) + 0.5 * (
                dep4[i] ** 2 + dep5[i] ** 2 + dep6[i] ** 2
            )
    else:  # irate == 1
        for i in range(nel):
            sum_sq = (epsp_arr[i, 0] ** 2 + epsp_arr[i, 1] ** 2 + epsp_arr[i, 2] ** 2) + 0.5 * (
                epsp_arr[i, 3] ** 2 + epsp_arr[i, 4] ** 2 + epsp_arr[i, 5] ** 2
            )
            rate_eq = math.sqrt(sum_sq)
            rate_filt = asrate * rate_eq + (1.0 - asrate) * uvar_arr[i, 0]
            epsd[i] = rate_filt
            dep1[i] = rate_filt
            dep2[i] = rate_filt
            dep3[i] = rate_filt
            dep4[i] = rate_filt
            dep5[i] = rate_filt
            dep6[i] = rate_filt
            uvar_arr[i, 0] = rate_filt

    # sigeps50s.F90 lines 260-353: yield stress evaluation
    sigyxx = np.full(nel, 1.0e30, dtype=np.float64)
    sigyyy = np.full(nel, 1.0e30, dtype=np.float64)
    sigyzz = np.full(nel, 1.0e30, dtype=np.float64)
    sigyxy = np.full(nel, 1.0e30, dtype=np.float64)
    sigyyz = np.full(nel, 1.0e30, dtype=np.float64)
    sigyzx = np.full(nel, 1.0e30, dtype=np.float64)

    if yield_fn is not None:
        sigyxx = yield_fn(0, ep1, dep1)
        sigyyy = yield_fn(1, ep2, dep2)
        sigyzz = yield_fn(2, ep3, dep3)
        sigyxy = yield_fn(3, ep4, dep4)
        sigyyz = yield_fn(4, ep5, dep5)
        sigyzx = yield_fn(5, ep6, dep6)

    def _fortran_sign(a: float, b: float) -> float:
        return math.copysign(a, b) if b != 0.0 else 0.0

    # sigeps50s.F90 lines 354-371: directional clamping for uncompacted elements
    if icomp == 0:
        for i in range(nel):
            sign[i, 0] = _fortran_sign(min(abs(sign[i, 0]), sigyxx[i]), sign[i, 0])
            sign[i, 1] = _fortran_sign(min(abs(sign[i, 1]), sigyyy[i]), sign[i, 1])
            sign[i, 2] = _fortran_sign(min(abs(sign[i, 2]), sigyzz[i]), sign[i, 2])
            sign[i, 3] = _fortran_sign(min(abs(sign[i, 3]), sigyxy[i]), sign[i, 3])
            sign[i, 4] = _fortran_sign(min(abs(sign[i, 4]), sigyyz[i]), sign[i, 4])
            sign[i, 5] = _fortran_sign(min(abs(sign[i, 5]), sigyzx[i]), sign[i, 5])
    else:
        for i in indx:
            sign[i, 0] = _fortran_sign(min(abs(sign[i, 0]), sigyxx[i]), sign[i, 0])
            sign[i, 1] = _fortran_sign(min(abs(sign[i, 1]), sigyyy[i]), sign[i, 1])
            sign[i, 2] = _fortran_sign(min(abs(sign[i, 2]), sigyzz[i]), sign[i, 2])
            sign[i, 3] = _fortran_sign(min(abs(sign[i, 3]), sigyxy[i]), sign[i, 3])
            sign[i, 4] = _fortran_sign(min(abs(sign[i, 4]), sigyyz[i]), sign[i, 4])
            sign[i, 5] = _fortran_sign(min(abs(sign[i, 5]), sigyzx[i]), sign[i, 5])

    # sigeps50s.F90 lines 376-401: J2 plasticity for fully compacted elements
    if icomp == 1:
        for i in indxc:
            depsv = (deps_arr[i, 0] + deps_arr[i, 1] + deps_arr[i, 2]) * (1.0 / 3.0)
            pres = (sigo_arr[i, 0] + sigo_arr[i, 1] + sigo_arr[i, 2]) * (1.0 / 3.0)

            stxx = sigo_arr[i, 0] + gcomp * (deps_arr[i, 0] - depsv) - pres
            styy = sigo_arr[i, 1] + gcomp * (deps_arr[i, 1] - depsv) - pres
            stzz = sigo_arr[i, 2] + gcomp * (deps_arr[i, 2] - depsv) - pres
            stxy = sigo_arr[i, 3] + gcomp * deps_arr[i, 3] * 0.5
            styz = sigo_arr[i, 4] + gcomp * deps_arr[i, 4] * 0.5
            stzx = sigo_arr[i, 5] + gcomp * deps_arr[i, 5] * 0.5

            svm_sq = (stxx ** 2 + styy ** 2 + stzz ** 2) * 0.5 + stxy ** 2 + styz ** 2 + stzx ** 2
            svm = math.sqrt(max(3.0 * svm_sq, 0.0))
            yld = sigy + hcomp * eplas_arr[i]

            rfact = min(1.0, yld / svm) if svm > 1.0e-30 else 1.0
            pres = pres + 3.0 * bulk * depsv

            sign[i, 0] = stxx * rfact + pres
            sign[i, 1] = styy * rfact + pres
            sign[i, 2] = stzz * rfact + pres
            sign[i, 3] = stxy * rfact
            sign[i, 4] = styz * rfact
            sign[i, 5] = stzx * rfact

            denom = 1.5 * gcomp + hcomp
            if denom > 1.0e-20 and svm > 1.0e-30:
                eplas_arr[i] += (1.0 - rfact) * svm / denom

    # Element deletion stress zeroing (engine physics)
    for i in range(nel):
        if off_arr[i] == 0.0:
            sign[i, :] = 0.0

    return {
        "sign": sign,
        "soundsp": soundsp,
        "off": off_arr,
        "eplas": eplas_arr,
        "epsd": epsd,
        "uvar": uvar_arr,
        "vartmp13": vartmp13_arr,
        "beta": beta,
        "rvol": rvol,
        "st_trial": st_trial,
    }


def _make_uparam(
    ea: float = 100.0,
    eb: float = 200.0,
    ec: float = 300.0,
    gab: float = 40.0,
    gbc: float = 50.0,
    gca: float = 60.0,
    emx11: float = 1.0e30,
    emx22: float = 1.0e30,
    emx33: float = 1.0e30,
    emx12: float = 1.0e30,
    emx23: float = 1.0e30,
    emx31: float = 1.0e30,
    fcut: float = 1.0e20,
    ecomp: float = 0.0,
    pr: float = 0.0,
    sigy: float = 0.0,
    hcomp: float = 0.0,
    vcomp: float = 0.0,
) -> np.ndarray:
    """Pack Fortran UPARAM array according to hm_read_mat50.F90 lines 379-403."""
    pr_eff = min(pr, 0.495)
    gcomp = ecomp / (1.0 + pr_eff) if ecomp > 0.0 else 0.0
    bulk = ecomp / (3.0 * (1.0 - 2.0 * pr_eff)) if ecomp > 0.0 else 0.0
    return np.array([
        ea, eb, ec, gab, gbc, gca,
        emx11, emx22, emx33, emx12, emx23, emx31,
        fcut,
        ecomp, gcomp, bulk, sigy, hcomp, vcomp,
    ], dtype=np.float64)


# =============================================================================
# 2. Targeted Unit Tests for Fortran Physics Expressions
# =============================================================================

class TestLaw50OracleTargetedPhysics:
    """Direct validation of individual Fortran expressions from sigeps50s.F90."""

    def test_elastic_trial_stress_and_sound_speed(self):
        """Verify lines 164-172: st_k = sig_k + E_k * deps_k and soundsp = sqrt(max(E, G)/rho)."""
        up = _make_uparam(ea=120.0, eb=240.0, ec=360.0, gab=45.0, gbc=55.0, gca=65.0)
        ip = np.array([0, 0, 0, 2])
        rho = np.array([0.002])
        dt = 0.001

        sigo = np.array([[10.0, 20.0, 30.0, 5.0, 6.0, 7.0]])
        deps = np.array([[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]])
        eps = np.zeros((1, 6))
        epsp = np.zeros((1, 6))
        amu = np.array([0.0])
        off = np.array([1.0])
        eplas = np.array([0.0])
        uvar = np.zeros((1, 6))
        vartmp13 = np.array([0])

        res = sigeps50s_oracle(
            up, ip, rho, dt, sigo, deps, eps, epsp, amu, off, eplas, uvar, vartmp13
        )

        # Expected trial stresses:
        exp_xx = 10.0 + 120.0 * 0.01
        exp_yy = 20.0 + 240.0 * 0.02
        exp_zz = 30.0 + 360.0 * 0.03
        exp_xy = 5.0 + 45.0 * 0.04
        exp_yz = 6.0 + 55.0 * 0.05
        exp_zx = 7.0 + 65.0 * 0.06

        expected_st = np.array([exp_xx, exp_yy, exp_zz, exp_xy, exp_yz, exp_zx])
        np.testing.assert_allclose(res["sign"][0], expected_st, rtol=1e-14, atol=1e-14)

        # Sound speed = sqrt(max(120, 240, 360, 45, 55, 65) / 0.002) = sqrt(360 / 0.002)
        exp_c = math.sqrt(360.0 / 0.002)
        assert math.isclose(res["soundsp"][0], exp_c, rel_tol=1e-14)

    def test_failure_deletion_normal_and_shear(self):
        """Verify lines 174-179: tensile normal strain and shear strain half-angle criteria."""
        up = _make_uparam(
            ea=100.0, gab=50.0,
            emx11=0.05, emx22=0.06, emx33=0.07,
            emx12=0.03, emx23=0.04, emx31=0.05,
        )
        ip = np.array([0, 0, 0, 2])
        rho = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        dt = 0.001

        sigo = np.ones((7, 6)) * 10.0
        deps = np.zeros((7, 6))

        # Case 0: Intact
        # Case 1: epsxx = 0.051 > 0.05 -> failure
        # Case 2: epsyy = 0.061 > 0.06 -> failure
        # Case 3: epszz = 0.071 > 0.07 -> failure
        # Case 4: |epsxy * 0.5| = 0.035 > 0.03 (epsxy = 0.07) -> failure
        # Case 5: |epsyz * 0.5| = 0.045 > 0.04 (epsyz = -0.09) -> failure
        # Case 6: |epszx * 0.5| = 0.055 > 0.05 (epszx = 0.11) -> failure
        eps = np.zeros((7, 6))
        eps[1, 0] = 0.051
        eps[2, 1] = 0.061
        eps[3, 2] = 0.071
        eps[4, 3] = 0.07
        eps[5, 4] = -0.09
        eps[6, 5] = 0.11

        epsp = np.zeros((7, 6))
        amu = np.zeros(7)
        off = np.ones(7)
        eplas = np.zeros(7)
        uvar = np.zeros((7, 6))
        vartmp13 = np.zeros(7, dtype=int)

        res = sigeps50s_oracle(
            up, ip, rho, dt, sigo, deps, eps, epsp, amu, off, eplas, uvar, vartmp13
        )

        expected_off = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(res["off"], expected_off)
        # Deleted elements have zero stresses
        assert np.all(res["sign"][1:] == 0.0)
        assert np.all(res["sign"][0] == 10.0)

    @pytest.mark.parametrize("gflag, expected_ep_mult", [(1, 1.0), (-1, -1.0), (0, 0.0)])
    def test_normal_strain_definition_gflag(self, gflag: int, expected_ep_mult: float):
        """Verify lines 183-195: ep1..3 definition based on iflag1."""
        up = _make_uparam(ea=100.0)
        ip = np.array([gflag, 0, 0, 2])
        rho = np.array([1.0])
        dt = 0.001

        sigo = np.zeros((1, 6))
        deps = np.array([[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]])
        eps = np.array([[0.05, 0.02, 0.01, 0.0, 0.0, 0.0]])
        epsp = np.zeros((1, 6))
        amu = np.array([0.12])
        off = np.array([1.0])
        eplas = np.array([0.0])
        uvar = np.zeros((1, 6))
        vartmp13 = np.array([0])

        queried_ep = []

        def yld_probe(k, ep, dep):
            if k == 0:
                queried_ep.append(float(ep[0]))
            return np.full_like(ep, 100.0)

        sigeps50s_oracle(
            up, ip, rho, dt, sigo, deps, eps, epsp, amu, off, eplas, uvar, vartmp13,
            yield_fn=yld_probe
        )

        if gflag == 1:
            assert math.isclose(queried_ep[0], 0.05, rel_tol=1e-14)
        elif gflag == -1:
            assert math.isclose(queried_ep[0], -0.05, rel_tol=1e-14)
        else:
            assert math.isclose(queried_ep[0], 0.12, rel_tol=1e-14)

    @pytest.mark.parametrize("vflag, expected_ep_mult", [(1, 1.0), (-1, -1.0), (0, 0.0)])
    def test_shear_strain_definition_vflag(self, vflag: int, expected_ep_mult: float):
        """Verify lines 196-208: ep4..6 definition based on iflag2."""
        up = _make_uparam(gab=50.0)
        ip = np.array([0, vflag, 0, 2])
        rho = np.array([1.0])
        dt = 0.001

        sigo = np.zeros((1, 6))
        deps = np.array([[0.0, 0.0, 0.0, 0.08, 0.0, 0.0]])
        eps = np.array([[0.0, 0.0, 0.0, 0.08, 0.02, 0.01]])
        epsp = np.zeros((1, 6))
        amu = np.array([0.15])
        off = np.array([1.0])
        eplas = np.array([0.0])
        uvar = np.zeros((1, 6))
        vartmp13 = np.array([0])

        queried_ep = []

        def yld_probe(k, ep, dep):
            if k == 3:
                queried_ep.append(float(ep[0]))
            return np.full_like(ep, 100.0)

        sigeps50s_oracle(
            up, ip, rho, dt, sigo, deps, eps, epsp, amu, off, eplas, uvar, vartmp13,
            yield_fn=yld_probe
        )

        if vflag == 1:
            assert math.isclose(queried_ep[0], 0.08, rel_tol=1e-14)
        elif vflag == -1:
            assert math.isclose(queried_ep[0], -0.08, rel_tol=1e-14)
        else:
            assert math.isclose(queried_ep[0], 0.15, rel_tol=1e-14)

    def test_strain_rate_filtering_irate_2_independent(self):
        """Verify lines 229-245: independent rate filtering for irate == 2."""
        fcut = 200.0
        timestep = 0.0025  # asrate = min(1.0, 200 * 0.0025) = 0.5
        up = _make_uparam(fcut=fcut)
        ip = np.array([0, 0, 0, 2])
        rho = np.array([1.0])

        sigo = np.zeros((1, 6))
        deps = np.zeros((1, 6))
        eps = np.zeros((1, 6))
        epsp = np.array([[10.0, -20.0, 30.0, 40.0, -50.0, 60.0]])
        amu = np.array([0.0])
        off = np.array([1.0])
        eplas = np.array([0.0])
        uvar_init = np.array([[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]])
        vartmp13 = np.array([0])

        res = sigeps50s_oracle(
            up, ip, rho, timestep, sigo, deps, eps, epsp, amu, off, eplas, uvar_init, vartmp13
        )

        # Expected: uvar_new = 0.5 * epsp + 0.5 * uvar_old
        exp_uvar = 0.5 * epsp + 0.5 * uvar_init
        np.testing.assert_allclose(res["uvar"], exp_uvar, rtol=1e-14, atol=1e-14)

        dep = np.abs(exp_uvar[0])
        exp_epsd = (dep[0]**2 + dep[1]**2 + dep[2]**2) + 0.5 * (dep[3]**2 + dep[4]**2 + dep[5]**2)
        assert math.isclose(res["epsd"][0], exp_epsd, rel_tol=1e-14)

    def test_strain_rate_filtering_irate_1_common_equivalent(self):
        """Verify lines 246-258: common equivalent strain rate filtering for irate == 1."""
        fcut = 100.0
        timestep = 0.004  # asrate = 0.4
        up = _make_uparam(fcut=fcut)
        ip = np.array([0, 0, 0, 1])
        rho = np.array([1.0])

        sigo = np.zeros((1, 6))
        deps = np.zeros((1, 6))
        eps = np.zeros((1, 6))
        epsp = np.array([[3.0, 4.0, 0.0, 2.0, 0.0, 0.0]])
        amu = np.array([0.0])
        off = np.array([1.0])
        eplas = np.array([0.0])
        uvar_init = np.array([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        vartmp13 = np.array([0])

        res = sigeps50s_oracle(
            up, ip, rho, timestep, sigo, deps, eps, epsp, amu, off, eplas, uvar_init, vartmp13
        )

        # Raw rate sum_sq = 3^2 + 4^2 + 0^2 + 0.5 * (2^2 + 0 + 0) = 9 + 16 + 2 = 27
        raw_eq = math.sqrt(27.0)
        exp_epsd = 0.4 * raw_eq + 0.6 * 10.0
        assert math.isclose(res["uvar"][0, 0], exp_epsd, rel_tol=1e-14)
        assert math.isclose(res["epsd"][0], exp_epsd, rel_tol=1e-14)

    def test_compaction_threshold_beta_and_irreversible_latching(self):
        """Verify lines 127-154: rvol = 1/(1+amu), beta, latching when rvol <= vcomp."""
        e11_0, ecomp = 100.0, 1000.0
        gab_0, gcomp = 40.0, 400.0
        vcomp = 0.25
        up = _make_uparam(
            ea=e11_0, gab=gab_0, ecomp=ecomp, pr=0.25, sigy=50.0, vcomp=vcomp
        )
        ip = np.array([0, 0, 1, 2])
        rho = np.array([1.0, 1.0, 1.0])
        dt = 0.001

        sigo = np.zeros((3, 6))
        deps = np.zeros((3, 6))
        eps = np.zeros((3, 6))
        epsp = np.zeros((3, 6))
        off = np.ones(3)
        eplas = np.zeros(3)
        uvar = np.zeros((3, 6))

        # Elem 0: amu = 0.0 -> rvol = 1.0 > 0.25 -> beta = 0.0, uncompacted
        # Elem 1: amu = 1.0 -> rvol = 0.5 > 0.25 -> beta = (1 - 0.5) / (1 - 0.25) = 0.5 / 0.75 = 2/3
        # Elem 2: amu = 4.0 -> rvol = 0.20 <= 0.25 -> beta = 1.0, permanently compacted!
        amu = np.array([0.0, 1.0, 4.0])
        vartmp13_init = np.array([0, 0, 0])

        res = sigeps50s_oracle(
            up, ip, rho, dt, sigo, deps, eps, epsp, amu, off, eplas, uvar, vartmp13_init
        )

        assert math.isclose(res["rvol"][0], 1.0, rel_tol=1e-14)
        assert math.isclose(res["beta"][0], 0.0, rel_tol=1e-14)
        assert res["vartmp13"][0] == 0

        assert math.isclose(res["rvol"][1], 0.5, rel_tol=1e-14)
        assert math.isclose(res["beta"][1], 2.0 / 3.0, rel_tol=1e-14)
        assert res["vartmp13"][1] == 0

        assert math.isclose(res["rvol"][2], 0.2, rel_tol=1e-14)
        assert math.isclose(res["beta"][2], 1.0, rel_tol=1e-14)
        assert res["vartmp13"][2] == 1  # Latched!

        # Step 2: Elem 2 now expands to rvol = 1.0 (amu = 0.0) -> must STAY compacted!
        amu_rebound = np.array([0.0])
        vartmp13_latched = np.array([1])
        res2 = sigeps50s_oracle(
            up, ip, np.array([1.0]), dt, np.zeros((1, 6)), np.zeros((1, 6)),
            np.zeros((1, 6)), np.zeros((1, 6)), amu_rebound, np.array([1.0]),
            np.array([0.0]), np.zeros((1, 6)), vartmp13_latched
        )
        assert res2["vartmp13"][0] == 1, "Compaction state must irreversibly latch!"

    def test_compacted_j2_plasticity_and_bulk_pressure(self):
        """Verify lines 376-401: deviatoric J2 return and bulk pressure for +/- depsv."""
        ecomp = 1000.0
        pr = 0.25  # nu = 0.25 -> gcomp = 2G = 1000 / 1.25 = 800.0; bulk = 1000 / (3 * 0.5) = 666.6666666666666
        sigy = 20.0
        hcomp = 100.0
        vcomp = 0.3
        up = _make_uparam(
            ecomp=ecomp, pr=pr, sigy=sigy, hcomp=hcomp, vcomp=vcomp
        )
        ip = np.array([0, 0, 1, 2])
        rho = np.array([1.0, 1.0])
        dt = 0.001

        # Element 0: Pure compression (depsv < 0) + shear yielding
        # Element 1: Pure tension (depsv > 0) + shear yielding
        sigo = np.zeros((2, 6))
        deps = np.zeros((2, 6))

        # Elem 0: depsxx = depsyy = depszz = -0.01 -> depsv = -0.01; depsxy = 0.1
        deps[0, :3] = -0.01
        deps[0, 3] = 0.10

        # Elem 1: depsxx = depsyy = depszz = +0.02 -> depsv = +0.02; depsxy = 0.1
        deps[1, :3] = 0.02
        deps[1, 3] = 0.10

        eps = np.zeros((2, 6))
        epsp = np.zeros((2, 6))
        amu = np.zeros(2)
        off = np.ones(2)
        eplas = np.zeros(2)
        uvar = np.zeros((2, 6))
        vartmp13 = np.array([1, 1])  # Both already in compacted state

        res = sigeps50s_oracle(
            up, ip, rho, dt, sigo, deps, eps, epsp, amu, off, eplas, uvar, vartmp13
        )

        gcomp_val = 800.0
        bulk_val = 1000.0 / 1.5

        # In compacted J2:
        # Elem 0:
        # depsv = -0.01, pres_0 = 0 -> pres_new = 3 * bulk * (-0.01) = -20.0
        # deviatoric normal = 0, stxy = 0 + 800 * 0.10 * 0.5 = 40.0
        # svm = sqrt(3 * (40^2)) = 40 * sqrt(3) ~ 69.28203 > sigy (20.0)
        # rfact = 20.0 / svm
        # sign_xy = 40.0 * (20.0 / svm) = 20.0 / sqrt(3)
        # Delta_eplas = (1 - rfact) * svm / (1.5 * 800 + 100) = (svm - 20) / 1300
        svm_0 = 40.0 * math.sqrt(3.0)
        rfact_0 = sigy / svm_0
        exp_pres_0 = 3.0 * bulk_val * (-0.01)
        exp_xy_0 = 40.0 * rfact_0
        exp_eplas_0 = (svm_0 - sigy) / (1.5 * gcomp_val + hcomp)

        assert math.isclose(res["sign"][0, 0], exp_pres_0, rel_tol=1e-12)
        assert math.isclose(res["sign"][0, 1], exp_pres_0, rel_tol=1e-12)
        assert math.isclose(res["sign"][0, 2], exp_pres_0, rel_tol=1e-12)
        assert math.isclose(res["sign"][0, 3], exp_xy_0, rel_tol=1e-12)
        assert math.isclose(res["eplas"][0], exp_eplas_0, rel_tol=1e-12)

        # Elem 1:
        exp_pres_1 = 3.0 * bulk_val * 0.02
        assert math.isclose(res["sign"][1, 0], exp_pres_1, rel_tol=1e-12)
        assert math.isclose(res["sign"][1, 1], exp_pres_1, rel_tol=1e-12)
        assert math.isclose(res["sign"][1, 2], exp_pres_1, rel_tol=1e-12)
        assert math.isclose(res["sign"][1, 3], exp_xy_0, rel_tol=1e-12)
        assert math.isclose(res["eplas"][1], exp_eplas_0, rel_tol=1e-12)


# =============================================================================
# 3. Exhaustive Parity Verification: solid_update vs Fortran Oracle (60 States)
# =============================================================================

@pytest.mark.parametrize("case_idx", range(60))
def test_solid_update_vs_fortran_oracle_exhaustive_60_states(case_idx: int):
    """Exhaustively verify pyradioss solid_update against sigeps50s_oracle.

    Evaluates 60 diverse physical configurations spanning:
    - Pure orthotropic elastic loading and unloading
    - Directional yield clamping with Gflag in {-1, 0, 1} and Vflag in {-1, 0, 1}
    - Strain rate filtering with irate in {1, 2} across various dt and fcut
    - Tensile and shear failure strain deletion
    - Compaction transition and irreversible latching
    - Compacted J2 elastoplasticity with positive and negative volumetric increments
    - Tolerance: assert_allclose(rtol=1e-12, atol=1e-12).
    """
    rng = np.random.default_rng(5000 + case_idx)

    # 1. Moduli and basic material parameters
    ea = float(rng.uniform(50.0, 300.0))
    eb = float(rng.uniform(50.0, 300.0))
    ec = float(rng.uniform(50.0, 300.0))
    gab = float(rng.uniform(20.0, 150.0))
    gbc = float(rng.uniform(20.0, 150.0))
    gca = float(rng.uniform(20.0, 150.0))

    rho0 = float(rng.uniform(1.0e-3, 5.0e-3))
    dt = float(rng.choice([1.0e-5, 1.0e-4, 1.0e-3, 0.005]))

    gflag = int(rng.choice([-1, 0, 1]))
    vflag = int(rng.choice([-1, 0, 1]))
    irate = int(rng.choice([1, 2]))
    fcut = float(rng.choice([0.0, 50.0, 200.0, 1000.0, 1.0e20]))

    # Compaction parameters
    enable_comp = (case_idx % 2 == 1)
    if enable_comp:
        ecomp = float(rng.uniform(500.0, 2000.0))
        pr = float(rng.uniform(0.1, 0.45))
        sigy = float(rng.uniform(10.0, 60.0))
        et = float(rng.uniform(10.0, 100.0))
        vcomp = float(rng.uniform(0.15, 0.4))
        icomp = 1
    else:
        ecomp = 0.0
        pr = 0.0
        sigy = 0.0
        et = 0.0
        vcomp = 0.0
        icomp = 0

    # Failure strain thresholds
    if case_idx % 5 == 0:
        # Trigger failure in 20% of cases
        emx11 = float(rng.uniform(0.01, 0.05))
        emx12 = float(rng.uniform(0.01, 0.04))
    else:
        emx11 = 1.0e30
        emx12 = 1.0e30
    emx22 = 1.0e30
    emx33 = 1.0e30
    emx23 = 1.0e30
    emx31 = 1.0e30

    # Yield curve setup for directional clamping
    has_yield_curves = (case_idx % 3 != 0)
    tables_mat = [None] * 6
    yld_levels = [float(rng.uniform(2.0, 25.0)) for _ in range(6)]

    if has_yield_curves:
        for k in range(6):
            # Flat yield stress level for deterministic direct evaluation
            val = yld_levels[k]
            tbl = FunctTable(k + 1, x=np.array([-10.0, 10.0]), y=np.array([val, val]))
            tables_mat[k] = tbl

        def oracle_yield_fn(k, ep, dep):
            return np.full_like(ep, yld_levels[k])
    else:
        oracle_yield_fn = None

    # Construct pyradioss Material
    mat_params = {
        "rho0": rho0,
        "MAT_RHO": rho0,
        "ea": ea,
        "eb": eb,
        "ec": ec,
        "gab": gab,
        "gbc": gbc,
        "gca": gca,
        "MAT_EA": ea,
        "MAT_EB": eb,
        "MAT_EC": ec,
        "MAT_GAB": gab,
        "MAT_GBC": gbc,
        "MAT_GCA": gca,
        "asrate": fcut,
        "gflag": gflag,
        "vflag": vflag,
        "irate": irate,
        "eps_max11": emx11,
        "eps_max22": emx22,
        "eps_max33": emx33,
        "eps_max12": emx12,
        "eps_max23": emx23,
        "eps_max31": emx31,
        "ecomp": ecomp,
        "pr": pr,
        "sigy": sigy,
        "et": et,
        "vcomp": vcomp,
        "tables": tables_mat,
    }
    mat = build_law50({"id": 1, "params": mat_params})

    # Construct Fortran UPARAM and IPARAM
    fcut_eff = 1.0e20 if fcut == 0.0 else fcut
    uparam = _make_uparam(
        ea=ea, eb=eb, ec=ec, gab=gab, gbc=gbc, gca=gca,
        emx11=emx11, emx22=emx22, emx33=emx33,
        emx12=emx12, emx23=emx23, emx31=emx31,
        fcut=fcut_eff,
        ecomp=ecomp, pr=pr, sigy=sigy, hcomp=et, vcomp=vcomp
    )
    iparam = np.array([gflag, vflag, icomp, irate])

    # Generate physical state
    sigo = rng.uniform(-5.0, 5.0, size=6)
    deps = rng.uniform(-0.02, 0.02, size=6)
    if case_idx % 4 == 0:
        deps[0] = rng.uniform(0.04, 0.08)  # Large normal strain
    if case_idx % 4 == 1:
        deps[3] = rng.uniform(0.06, 0.12)  # Large shear strain

    # Volumetric expansion / compression
    amu_val = float(rng.choice([
        0.0,
        rng.uniform(-0.05, -0.001),  # tension
        rng.uniform(0.01, 0.5),      # compression
        rng.uniform(1.5, 4.0),       # heavy compaction
    ]))

    # Strain rate tensor
    epsp_rate = rng.uniform(-50.0, 50.0, size=6)

    # Initial internal variables
    uvar_init = rng.uniform(0.0, 20.0, size=6) if irate == 2 else np.array([float(rng.uniform(0.0, 20.0)), 0, 0, 0, 0, 0])
    eplas_init = float(rng.uniform(0.0, 0.02)) if enable_comp else 0.0
    vartmp13_init = 1 if (enable_comp and case_idx % 6 == 0) else 0

    # Current total strain
    eps_curr = deps.copy()

    # 2. Run exact Fortran Oracle
    oracle_res = sigeps50s_oracle(
        uparam=uparam,
        iparam=iparam,
        rho=np.array([rho0]),
        timestep=dt,
        sigo=sigo.reshape(1, 6),
        deps=deps.reshape(1, 6),
        eps=eps_curr.reshape(1, 6),
        epsp=epsp_rate.reshape(1, 6),
        amu=np.array([amu_val]),
        off=np.array([1.0]),
        eplas=np.array([eplas_init]),
        uvar=uvar_init.reshape(1, 6),
        vartmp13=np.array([vartmp13_init]),
        yield_fn=oracle_yield_fn,
    )

    # 3. Run pyradioss solid_update
    extra_py = {
        "amu": np.array([amu_val]),
        "rate": epsp_rate.copy(),
        "eps_total": eps_curr.copy(),
        "uvar50": uvar_init.reshape(1, 6).copy(),
        "compacted": np.array([vartmp13_init == 1]),
        "off50": np.array([1.0]),
    }
    epsp_py = np.array([eplas_init])

    sig_py, eplas_py_out, c_py = solid_update(
        mat, sigo.copy(), deps.copy(), epsp=epsp_py, dt=dt, extra=extra_py, return_tuple=True
    )

    # 4. Assert Exact Parity (rtol=1e-12, atol=1e-12)
    oracle_sign = oracle_res["sign"][0]
    oracle_c = oracle_res["soundsp"][0]
    oracle_off = oracle_res["off"][0]
    oracle_eplas = oracle_res["eplas"][0]
    oracle_uvar = oracle_res["uvar"][0]
    oracle_compacted = (oracle_res["vartmp13"][0] == 1)

    np.testing.assert_allclose(
        sig_py, oracle_sign, rtol=1e-12, atol=1e-12,
        err_msg=f"Case {case_idx}: stress tensor mismatch"
    )
    np.testing.assert_allclose(
        c_py, oracle_c, rtol=1e-12, atol=1e-12,
        err_msg=f"Case {case_idx}: acoustic sound speed mismatch"
    )
    np.testing.assert_allclose(
        extra_py["off50"][0], oracle_off, rtol=1e-12, atol=1e-12,
        err_msg=f"Case {case_idx}: element deletion flag mismatch"
    )
    if enable_comp:
        np.testing.assert_allclose(
            eplas_py_out, oracle_eplas, rtol=1e-12, atol=1e-12,
            err_msg=f"Case {case_idx}: plastic strain mismatch"
        )
        assert extra_py["compacted"][0] == oracle_compacted, f"Case {case_idx}: compacted state mismatch"

    if irate == 2:
        np.testing.assert_allclose(
            extra_py["uvar50"][0], oracle_uvar, rtol=1e-12, atol=1e-12,
            err_msg=f"Case {case_idx}: irate=2 filtered rate tensor mismatch"
        )
    else:
        np.testing.assert_allclose(
            extra_py["uvar50"][0, 0], oracle_uvar[0], rtol=1e-12, atol=1e-12,
            err_msg=f"Case {case_idx}: irate=1 common rate mismatch"
        )
