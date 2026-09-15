r"""Fortran Parity & Physics Oracle Verifier for Milestone M560: /MAT/LAW163 (/MAT/CRUSHABLE_FOAM /MAT/CRUSH_FOAM).

Crushable Foam Material Model.
Directly cites and mirrors upstream OpenRadioss source:
- Upstream Fortran starter reader:
  ``starter/source/materials/mat/mat163/hm_read_mat163.F90``
- Upstream Fortran engine physics:
  ``engine/source/materials/mat/mat163/sigeps163.F90`` (lines 78 to 307)
- CFG schema and defaults:
  ``hm_cfg_files/config/CFG/radioss2025/MAT/matl163_crushable_foam.cfg``

Tests included:
1. Exact Python oracle reproducing sigeps163.F90 lines 78 to 307.
2. Elastic trial stress calculation with cii, cij, and g.
3. Spectral decomposition (eigenvalues and eigenvectors) and Cauchy stress reconstruction.
4. Pure volumetric compression and tension across various densities.
5. Principal stress clamping with differing compressive yield and tensile cutoff values.
6. Volumetric strain rate filtering under nrs=0/1 and consecutive cycles.
7. Rate increment limit clipping with srclmt * dt.
8. Viscous damping stress tensor evaluation with varied damp and characteristic lengths le.
9. Acoustic sound speed evaluation with and without dt, and density fallback.
10. Algorithmic consistent solid tangent tensor.
11. Vectorized multi-element execution parity.
12. Exhaustive 60+ diverse physical states comparison: solid_update vs sigeps163_oracle (rtol=1e-12, atol=1e-12).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pytest

from pyradioss.common.tables import FunctTable
from pyradioss.model.entities import Material, MatLaw163
from pyradioss.materials.law163_crush_foam import (
    Law163Params,
    build_law163,
    solid_update,
    sound_speed_solid,
    consistent_solid_tangent,
)

_EM20 = 1.0e-20
_TWO_PI = 2.0 * math.pi
_FOUR_THIRD = 4.0 / 3.0
_TWO_THIRD = 2.0 / 3.0


# =============================================================================
# 1. UPSTREAM FORTRAN ENGINE ORACLE: sigeps163.F90 (lines 78 to 307)
# =============================================================================

def sigeps163_oracle(
    matparam: Dict[str, Any],
    rho0: np.ndarray,
    rho: np.ndarray,
    timestep: float,
    aldt: np.ndarray,
    sigo: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    uvar: Optional[np.ndarray] = None,
    epsd: Optional[np.ndarray] = None,
    yield_fn: Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, Any]:
    """Exact Python reproduction of OpenRadioss sigeps163.F90 lines 78-307.

    Parameters
    ----------
    matparam : dict
        Material parameters dictionary:
        'young', 'nu', 'g', 'bulk', 'cii', 'cij', 'tsc', 'damp',
        'ncycle', 'srclmt', 'nrs', 'fscale', 'table'
    rho0 : ndarray of length nel
        Reference / initial density.
    rho : ndarray of length nel
        Current density.
    timestep : float
        Explicit time step dt.
    aldt : ndarray of length nel
        Element characteristic length.
    sigo : ndarray of shape (nel, 6)
        Old Cauchy stress tensor [xx, yy, zz, xy, yz, zx].
    deps : ndarray of shape (nel, 6)
        Strain increment tensor (engineering shear components xy, yz, zx).
    epsp : ndarray of shape (nel, 6), optional
        Strain rate tensor. If None, computed as deps / max(dt, 1e-20).
    uvar : ndarray of shape (nel, 2), optional
        Internal history variables: [uvar1 = gama_old, uvar2 = initial_length].
    epsd : ndarray of length nel, optional
        Old filtered volumetric strain rate.
    yield_fn : callable, optional
        Yield lookup function `(gama, dgamdt) -> (sigy_val, dsdgam)`.

    Returns
    -------
    dict
        Dictionary containing exact updated outputs:
        'sign', 'sigv', 'sig_tot', 'ssp', 'uvar', 'epsd', 'plas',
        'sigy', 'dsdgam', 'sigp', 'sigp_scaled', 'dirp', 'a', 'dgamdt', 'gama'
    """
    sigo_arr = np.atleast_2d(np.array(sigo, dtype=np.float64)).copy()
    deps_arr = np.atleast_2d(np.array(deps, dtype=np.float64)).copy()
    nel = sigo_arr.shape[0]

    rho0_arr = np.atleast_1d(np.array(rho0, dtype=np.float64)).copy()
    rho_arr = np.atleast_1d(np.array(rho, dtype=np.float64)).copy()
    aldt_arr = np.atleast_1d(np.array(aldt, dtype=np.float64)).copy()

    if epsp is not None:
        epsp_arr = np.atleast_2d(np.array(epsp, dtype=np.float64)).copy()
    else:
        dt_eff = max(float(timestep), _EM20)
        epsp_arr = deps_arr / dt_eff

    if uvar is not None:
        uvar_arr = np.atleast_2d(np.array(uvar, dtype=np.float64)).copy()
    else:
        uvar_arr = np.zeros((nel, 2), dtype=np.float64)

    if epsd is not None:
        epsd_arr = np.atleast_1d(np.array(epsd, dtype=np.float64)).copy()
    else:
        epsd_arr = np.zeros(nel, dtype=np.float64)

    # Recover model parameters (sigeps163.F90 lines 140-153)
    ncycle = int(matparam.get("ncycle", 12))
    nrs = int(matparam.get("nrs", 0))
    young = float(matparam.get("young", matparam.get("e", 0.0)))
    nu = float(matparam.get("nu", 0.0))
    g = float(matparam.get("g", matparam.get("shear", young / (2.0 * (1.0 + nu)) if (young > 0 and nu >= 0) else 0.0)))
    bulk = float(matparam.get("bulk", young / (3.0 * (1.0 - 2.0 * nu)) if (young > 0 and nu < 0.5) else 0.0))
    cii = float(matparam.get("cii", bulk + _FOUR_THIRD * g))
    cij = float(matparam.get("cij", bulk - _TWO_THIRD * g))
    tsc = float(matparam.get("tsc", 0.0))
    damp = float(matparam.get("damp", 0.10))
    srclmt = float(matparam.get("srclmt", 1.0e20))
    fscale = float(matparam.get("fscale", 1.0))
    alpha = _TWO_PI / (_TWO_PI + float(ncycle))

    # Element initial length (sigeps163.F90 lines 156-161)
    if uvar_arr[0, 1] == 0.0:
        for i in range(nel):
            uvar_arr[i, 1] = aldt_arr[i]
    le = uvar_arr[:, 1].copy()

    # Step 1: Trial stress tensor (sigeps163.F90 lines 169-176)
    signxx = np.zeros(nel, dtype=np.float64)
    signyy = np.zeros(nel, dtype=np.float64)
    signzz = np.zeros(nel, dtype=np.float64)
    signxy = np.zeros(nel, dtype=np.float64)
    signyz = np.zeros(nel, dtype=np.float64)
    signzx = np.zeros(nel, dtype=np.float64)

    for i in range(nel):
        signxx[i] = sigo_arr[i, 0] + cii * deps_arr[i, 0] + cij * deps_arr[i, 1] + cij * deps_arr[i, 2]
        signyy[i] = sigo_arr[i, 1] + cij * deps_arr[i, 0] + cii * deps_arr[i, 1] + cij * deps_arr[i, 2]
        signzz[i] = sigo_arr[i, 2] + cij * deps_arr[i, 0] + cij * deps_arr[i, 1] + cii * deps_arr[i, 2]
        signxy[i] = sigo_arr[i, 3] + g * deps_arr[i, 3]
        signyz[i] = sigo_arr[i, 4] + g * deps_arr[i, 4]
        signzx[i] = sigo_arr[i, 5] + g * deps_arr[i, 5]

    # Step 2: Principal stresses and directions (sigeps163.F90 lines 178-188)
    sigp = np.zeros((nel, 3), dtype=np.float64)
    dirp = np.zeros((nel, 3, 3), dtype=np.float64)

    for i in range(nel):
        S = np.array([
            [signxx[i], signxy[i], signzx[i]],
            [signxy[i], signyy[i], signyz[i]],
            [signzx[i], signyz[i], signzz[i]],
        ], dtype=np.float64)
        w, v = np.linalg.eigh(S)
        sigp[i, :] = w
        dirp[i, :, :] = v  # dirp(i, row, col) where col k is eigenvector k

    # Step 3: Volumetric strain & strain rate filtering (sigeps163.F90 lines 193-214)
    gama = np.zeros(nel, dtype=np.float64)
    dgamdt = np.zeros(nel, dtype=np.float64)
    plas = np.zeros(nel, dtype=np.float64)
    dt_safe = max(float(timestep), _EM20)

    for i in range(nel):
        gama[i] = 1.0 - rho0_arr[i] / max(rho_arr[i], _EM20)
        if nrs == 1:
            dgamdt[i] = (gama[i] - uvar_arr[i, 0]) / dt_safe
        else:
            dgamdt[i] = -(epsp_arr[i, 0] + epsp_arr[i, 1] + epsp_arr[i, 2])

        # Filtering of volumetric strain rate
        dgamdt[i] = alpha * dgamdt[i] + (1.0 - alpha) * epsd_arr[i]

        # Cap change of volumetric strain rate
        diff = dgamdt[i] - epsd_arr[i]
        if abs(diff) > srclmt * timestep:
            dgamdt[i] = epsd_arr[i] + math.copysign(1.0, diff) * srclmt * timestep

        plas[i] = math.log(max(rho0_arr[i] / max(rho_arr[i], _EM20), _EM20))
        epsd_arr[i] = dgamdt[i]

    # Step 4: Yield stress computation (sigeps163.F90 lines 219-228)
    sigy = np.zeros(nel, dtype=np.float64)
    dsdgam = np.zeros(nel, dtype=np.float64)

    for i in range(nel):
        g_pos = max(gama[i], 0.0)
        dg_pos = max(dgamdt[i], 0.0)
        if yield_fn is not None:
            sy_val, ds_val = yield_fn(np.array([g_pos]), np.array([dg_pos]))
            sigy[i] = -abs(float(sy_val[0])) * fscale
            dsdgam[i] = max(float(ds_val[0]) * fscale, 0.0)
        else:
            tbl = matparam.get("table", None)
            if tbl is not None and isinstance(tbl, FunctTable):
                # FunctTable lookup
                xs = np.asarray(tbl.x, dtype=float)
                ys = np.asarray(tbl.y, dtype=float)
                slopes = getattr(tbl, "slope", np.diff(ys) / np.maximum(np.diff(xs), _EM20))
                idx = np.clip(np.searchsorted(xs, g_pos, side="right"), 1, len(xs) - 1)
                ds_val = slopes[idx - 1]
                sy_val = ys[idx - 1] + ds_val * (g_pos - xs[idx - 1])
                sigy[i] = -abs(float(sy_val)) * fscale
                dsdgam[i] = max(float(ds_val) * fscale, 0.0)
            elif tbl is not None and callable(tbl):
                res = tbl(g_pos, dg_pos)
                if isinstance(res, tuple):
                    sigy[i] = -abs(float(res[0])) * fscale
                    dsdgam[i] = max(float(res[1]) * fscale, 0.0)
                else:
                    sigy[i] = -abs(float(res)) * fscale
                    dsdgam[i] = 0.0
            elif isinstance(tbl, (int, float)):
                sigy[i] = -abs(float(tbl)) * fscale
                dsdgam[i] = 0.0
            else:
                sigy[i] = -1.0e20 * fscale
                dsdgam[i] = 0.0

    # Step 5: Stress scaling procedure (sigeps163.F90 lines 232-260)
    sigp_scaled = np.zeros((nel, 3), dtype=np.float64)
    for i in range(nel):
        for j in range(3):
            if sigp[i, j] < sigy[i]:
                sigp_scaled[i, j] = sigy[i]
            elif sigp[i, j] > tsc:
                sigp_scaled[i, j] = tsc
            else:
                sigp_scaled[i, j] = sigp[i, j]

        # Reconstruct Cauchy stress tensor (sigeps163.F90 lines 242-260)
        signxx[i] = (
            dirp[i, 0, 0] * dirp[i, 0, 0] * sigp_scaled[i, 0]
            + dirp[i, 0, 1] * dirp[i, 0, 1] * sigp_scaled[i, 1]
            + dirp[i, 0, 2] * dirp[i, 0, 2] * sigp_scaled[i, 2]
        )
        signyy[i] = (
            dirp[i, 1, 1] * dirp[i, 1, 1] * sigp_scaled[i, 1]
            + dirp[i, 1, 2] * dirp[i, 1, 2] * sigp_scaled[i, 2]
            + dirp[i, 1, 0] * dirp[i, 1, 0] * sigp_scaled[i, 0]
        )
        signzz[i] = (
            dirp[i, 2, 2] * dirp[i, 2, 2] * sigp_scaled[i, 2]
            + dirp[i, 2, 0] * dirp[i, 2, 0] * sigp_scaled[i, 0]
            + dirp[i, 2, 1] * dirp[i, 2, 1] * sigp_scaled[i, 1]
        )
        signxy[i] = (
            dirp[i, 0, 0] * dirp[i, 1, 0] * sigp_scaled[i, 0]
            + dirp[i, 0, 1] * dirp[i, 1, 1] * sigp_scaled[i, 1]
            + dirp[i, 0, 2] * dirp[i, 1, 2] * sigp_scaled[i, 2]
        )
        signyz[i] = (
            dirp[i, 1, 1] * dirp[i, 2, 1] * sigp_scaled[i, 1]
            + dirp[i, 1, 2] * dirp[i, 2, 2] * sigp_scaled[i, 2]
            + dirp[i, 1, 0] * dirp[i, 2, 0] * sigp_scaled[i, 0]
        )
        signzx[i] = (
            dirp[i, 2, 2] * dirp[i, 0, 2] * sigp_scaled[i, 2]
            + dirp[i, 2, 0] * dirp[i, 0, 0] * sigp_scaled[i, 0]
            + dirp[i, 2, 1] * dirp[i, 0, 1] * sigp_scaled[i, 1]
        )

    # Step 6: User variables and sound speed (sigeps163.F90 lines 266-274)
    ssp = np.zeros(nel, dtype=np.float64)
    for i in range(nel):
        rho_den = rho_arr[i] if rho_arr[i] > _EM20 else rho0_arr[i]
        ssp[i] = math.sqrt((max(bulk, dsdgam[i]) + _FOUR_THIRD * g) / rho_den)
        uvar_arr[i, 0] = gama[i]

    # Step 7: Viscous damping (sigeps163.F90 lines 279-307)
    sigvxx = np.zeros(nel, dtype=np.float64)
    sigvyy = np.zeros(nel, dtype=np.float64)
    sigvzz = np.zeros(nel, dtype=np.float64)
    sigvxy = np.zeros(nel, dtype=np.float64)
    sigvyz = np.zeros(nel, dtype=np.float64)
    sigvzx = np.zeros(nel, dtype=np.float64)
    a = np.zeros(nel, dtype=np.float64)

    for i in range(nel):
        denom = math.copysign(max(abs(1.0 + gama[i]), _EM20), 1.0 + gama[i])
        a[i] = ssp[i] * max(rho_arr[i], _EM20) * damp * le[i] / denom
        ldav = (epsp_arr[i, 0] + epsp_arr[i, 1] + epsp_arr[i, 2]) / 3.0

        one_p_nu = 1.0 + nu
        one_m_2nu = 1.0 - 2.0 * nu
        sigvxx[i] = a[i] * ((epsp_arr[i, 0] - ldav) / one_p_nu + ldav / one_m_2nu)
        sigvyy[i] = a[i] * ((epsp_arr[i, 1] - ldav) / one_p_nu + ldav / one_m_2nu)
        sigvzz[i] = a[i] * ((epsp_arr[i, 2] - ldav) / one_p_nu + ldav / one_m_2nu)
        sigvxy[i] = a[i] * epsp_arr[i, 3] / (2.0 * one_p_nu)
        sigvyz[i] = a[i] * epsp_arr[i, 4] / (2.0 * one_p_nu)
        sigvzx[i] = a[i] * epsp_arr[i, 5] / (2.0 * one_p_nu)

        # Update sound speed with damping stiffness (lines 292-306)
        rho_den = rho_arr[i] if rho_arr[i] > _EM20 else rho0_arr[i]
        if timestep > 0.0:
            ssp[i] = math.sqrt(
                (max(bulk, dsdgam[i]) + _FOUR_THIRD * g + abs(a[i]) / max(timestep, _EM20)) / rho_den
            )
        else:
            ssp[i] = math.sqrt((max(bulk, dsdgam[i]) + _FOUR_THIRD * g) / rho_den)

    sign = np.column_stack([signxx, signyy, signzz, signxy, signyz, signzx])
    sigv = np.column_stack([sigvxx, sigvyy, sigvzz, sigvxy, sigvyz, sigvzx])
    sig_tot = sign + sigv

    return {
        "sign": sign,
        "sigv": sigv,
        "sig_tot": sig_tot,
        "ssp": ssp,
        "uvar": uvar_arr,
        "epsd": epsd_arr,
        "plas": plas,
        "sigy": sigy,
        "dsdgam": dsdgam,
        "sigp": sigp,
        "sigp_scaled": sigp_scaled,
        "dirp": dirp,
        "a": a,
        "dgamdt": dgamdt,
        "gama": gama,
    }


# =============================================================================
# 2. TARGETED ORACLE PHYSICS TESTS
# =============================================================================

def test_oracle_elastic_trial_stress_formulas():
    """Verify trial stress calculation lines 169-176 across normal, hydrostatic, and shear increments."""
    young = 100.0
    nu = 0.25
    g = young / (2.0 * (1.0 + nu))          # 40.0
    bulk = young / (3.0 * (1.0 - 2.0 * nu)) # 66.6666667
    cii = bulk + _FOUR_THIRD * g           # 120.0
    cij = bulk - _TWO_THIRD * g            # 40.0

    matparam = {
        "young": young,
        "nu": nu,
        "g": g,
        "bulk": bulk,
        "cii": cii,
        "cij": cij,
        "tsc": 1.0e10,
        "damp": 0.0,
        "table": 1.0e10,
    }

    sigo = np.array([[10.0, 5.0, -2.0, 3.0, -1.0, 4.0]])
    deps = np.array([[0.01, -0.005, 0.002, 0.008, -0.004, 0.006]])

    res = sigeps163_oracle(
        matparam=matparam,
        rho0=np.array([1.0]),
        rho=np.array([1.0]),
        timestep=1.0e-5,
        aldt=np.array([1.0]),
        sigo=sigo,
        deps=deps,
    )

    expected_st = np.array([11.08, 4.88, -1.56, 3.32, -1.16, 4.24])
    np.testing.assert_allclose(res["sign"][0], expected_st, rtol=1e-12, atol=1e-12)


def test_oracle_spectral_decomposition_and_reconstruction():
    """Verify principal stress projection and Cauchy reconstruction lines 178-188 & 242-260."""
    matparam = {
        "young": 100.0,
        "nu": 0.0,
        "tsc": 1.0e10,
        "damp": 0.0,
        "table": 1.0e10,
    }
    sigo = np.zeros((1, 6))
    deps = np.array([[0.02, -0.01, 0.005, 0.015, -0.008, 0.012]])

    res = sigeps163_oracle(
        matparam=matparam,
        rho0=np.array([1.0]),
        rho=np.array([1.0]),
        timestep=1.0e-4,
        aldt=np.array([1.0]),
        sigo=sigo,
        deps=deps,
    )

    cii = matparam["young"]
    g = 50.0
    trial = np.array([
        cii * deps[0, 0],
        cii * deps[0, 1],
        cii * deps[0, 2],
        g * deps[0, 3],
        g * deps[0, 4],
        g * deps[0, 5],
    ])
    np.testing.assert_allclose(res["sign"][0], trial, rtol=1e-12, atol=1e-12)

    # Check that eigenvalues of reconstructed matrix match sigp
    S_rec = np.array([
        [res["sign"][0, 0], res["sign"][0, 3], res["sign"][0, 5]],
        [res["sign"][0, 3], res["sign"][0, 1], res["sign"][0, 4]],
        [res["sign"][0, 5], res["sign"][0, 4], res["sign"][0, 2]],
    ])
    eigs = np.linalg.eigvalsh(S_rec)
    np.testing.assert_allclose(np.sort(eigs), np.sort(res["sigp"][0]), rtol=1e-12, atol=1e-12)


def test_pure_volumetric_compression_and_tension():
    """Verify volumetric strain gama and true plastic strain plas across compression and expansion."""
    rho0 = 1.2
    matparam = {"young": 10.0, "nu": 0.0, "tsc": 10.0, "damp": 0.0, "table": 50.0}

    # Compression sweep: rho > rho0 -> gama in (0, 1), plas < 0
    rhos_comp = np.array([1.25, 1.5, 2.0, 3.0, 6.0])
    for r in rhos_comp:
        res = sigeps163_oracle(
            matparam=matparam,
            rho0=np.array([rho0]),
            rho=np.array([r]),
            timestep=1e-3,
            aldt=np.array([1.0]),
            sigo=np.zeros((1, 6)),
            deps=np.zeros((1, 6)),
        )
        expected_gama = 1.0 - rho0 / r
        expected_plas = math.log(rho0 / r)
        assert res["gama"][0] > 0.0
        assert res["plas"][0] < 0.0
        np.testing.assert_allclose(res["gama"][0], expected_gama, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(res["plas"][0], expected_plas, rtol=1e-12, atol=1e-12)

    # Tension sweep: rho < rho0 -> gama < 0, plas > 0
    rhos_tens = np.array([1.15, 1.0, 0.8, 0.5, 0.2])
    for r in rhos_tens:
        res = sigeps163_oracle(
            matparam=matparam,
            rho0=np.array([rho0]),
            rho=np.array([r]),
            timestep=1e-3,
            aldt=np.array([1.0]),
            sigo=np.zeros((1, 6)),
            deps=np.zeros((1, 6)),
        )
        expected_gama = 1.0 - rho0 / r
        expected_plas = math.log(rho0 / r)
        np.testing.assert_allclose(res["gama"][0], expected_gama, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(res["plas"][0], expected_plas, rtol=1e-12, atol=1e-12)


def test_principal_stress_clamping_yield_and_tensile_cutoff():
    """Verify lines 234-240: sigp clamped below sigy (compressive) and above tsc (tensile)."""
    young = 1000.0
    nu = 0.0
    sigy_target = 35.0  # sigy in Fortran is -abs(sigy) = -35.0
    tsc = 12.0

    matparam = {
        "young": young,
        "nu": nu,
        "tsc": tsc,
        "damp": 0.0,
        "table": sigy_target,
    }

    # Element 0: trial principal stresses [-60, 5, 25]
    sigo = np.array([[-60.0, 5.0, 25.0, 0.0, 0.0, 0.0]])
    deps = np.zeros((1, 6))

    res = sigeps163_oracle(
        matparam=matparam,
        rho0=np.array([1.0]),
        rho=np.array([1.0]),
        timestep=1e-4,
        aldt=np.array([1.0]),
        sigo=sigo,
        deps=deps,
    )

    np.testing.assert_allclose(res["sigy"][0], -35.0, rtol=1e-12, atol=1e-12)
    expected_scaled = np.array([-35.0, 5.0, 12.0])
    np.testing.assert_allclose(np.sort(res["sigp_scaled"][0]), np.sort(expected_scaled), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res["sign"][0, 0:3], expected_scaled, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res["sign"][0, 3:6], 0.0, atol=1e-12)


def test_rate_filtering_nrs_flag_and_srclmt_limits():
    """Verify lines 198-209: engineering vs true rate, alpha exponential filter, and srclmt rate limit."""
    ncycle = 12
    alpha = _TWO_PI / (_TWO_PI + float(ncycle))

    # Test 1: nrs = 1 (engineering rate = (gama - uvar1) / dt)
    matparam_nrs1 = {
        "young": 100.0,
        "nu": 0.0,
        "ncycle": ncycle,
        "nrs": 1,
        "srclmt": 1e20,
        "table": 100.0,
    }
    uvar = np.array([[0.05, 1.0]])
    epsd_old = 10.0
    rho0 = 1.0
    rho_curr = 1.25  # gama = 1 - 1/1.25 = 0.20
    dt = 1e-3
    raw_eng = (0.20 - 0.05) / dt  # 150.0

    res1 = sigeps163_oracle(
        matparam=matparam_nrs1,
        rho0=np.array([rho0]),
        rho=np.array([rho_curr]),
        timestep=dt,
        aldt=np.array([1.0]),
        sigo=np.zeros((1, 6)),
        deps=np.zeros((1, 6)),
        uvar=uvar,
        epsd=np.array([epsd_old]),
    )
    expected_rate1 = alpha * raw_eng + (1.0 - alpha) * epsd_old
    np.testing.assert_allclose(res1["epsd"][0], expected_rate1, rtol=1e-12, atol=1e-12)

    # Test 2: nrs = 0 (true rate = -(epsp_xx + epsp_yy + epsp_zz))
    matparam_nrs0 = {
        "young": 100.0,
        "nu": 0.0,
        "ncycle": ncycle,
        "nrs": 0,
        "srclmt": 1e20,
        "table": 100.0,
    }
    deps = np.array([[-0.01, -0.02, -0.015, 0.0, 0.0, 0.0]])
    raw_true = -((-0.01 - 0.02 - 0.015) / dt)  # 45.0
    res2 = sigeps163_oracle(
        matparam=matparam_nrs0,
        rho0=np.array([rho0]),
        rho=np.array([rho_curr]),
        timestep=dt,
        aldt=np.array([1.0]),
        sigo=np.zeros((1, 6)),
        deps=deps,
        epsd=np.array([epsd_old]),
    )
    expected_rate2 = alpha * raw_true + (1.0 - alpha) * epsd_old
    np.testing.assert_allclose(res2["epsd"][0], expected_rate2, rtol=1e-12, atol=1e-12)

    # Test 3: srclmt rate change limit clipping
    srclmt = 5000.0
    matparam_clip = {
        "young": 100.0,
        "nu": 0.0,
        "ncycle": ncycle,
        "nrs": 0,
        "srclmt": srclmt,
        "table": 100.0,
    }
    deps_huge = np.array([[-100.0 * dt, 0.0, 0.0, 0.0, 0.0, 0.0]])
    res3 = sigeps163_oracle(
        matparam=matparam_clip,
        rho0=np.array([rho0]),
        rho=np.array([rho_curr]),
        timestep=dt,
        aldt=np.array([1.0]),
        sigo=np.zeros((1, 6)),
        deps=deps_huge,
        epsd=np.array([0.0]),
    )
    np.testing.assert_allclose(res3["epsd"][0], 5.0, rtol=1e-12, atol=1e-12)


def test_rate_filtering_consecutive_cycles():
    """Verify multi-step convergence of volumetric strain rate under constant excitation."""
    ncycle = 8
    alpha = _TWO_PI / (_TWO_PI + float(ncycle))
    matparam = {"young": 10.0, "nu": 0.0, "ncycle": ncycle, "nrs": 0, "srclmt": 1e20, "table": 10.0}
    dt = 1e-4
    steady_rate = 250.0
    deps = np.array([[-steady_rate * dt, 0.0, 0.0, 0.0, 0.0, 0.0]])

    epsd_curr = 0.0
    uvar_curr = np.zeros((1, 2))

    for step in range(15):
        res = sigeps163_oracle(
            matparam=matparam,
            rho0=np.array([1.0]),
            rho=np.array([1.0]),
            timestep=dt,
            aldt=np.array([1.0]),
            sigo=np.zeros((1, 6)),
            deps=deps,
            uvar=uvar_curr,
            epsd=np.array([epsd_curr]),
        )
        expected = alpha * steady_rate + (1.0 - alpha) * epsd_curr
        np.testing.assert_allclose(res["epsd"][0], expected, rtol=1e-12, atol=1e-12)
        epsd_curr = res["epsd"][0]
        uvar_curr = res["uvar"]

    analytical_15 = steady_rate * (1.0 - (1.0 - alpha) ** 15)
    np.testing.assert_allclose(epsd_curr, analytical_15, rtol=1e-12, atol=1e-12)


def test_viscous_damping_stress_and_parameters():
    """Verify lines 280-290: damping coefficient a, deviatoric / volumetric split, and sigv addition."""
    young = 60.0
    nu = 0.2
    g = young / (2.0 * (1.0 + nu))          # 25.0
    bulk = young / (3.0 * (1.0 - 2.0 * nu)) # 33.3333333
    damp = 0.15
    le_val = 2.5
    rho_val = 0.8
    dt = 1e-3

    matparam = {
        "young": young,
        "nu": nu,
        "g": g,
        "bulk": bulk,
        "damp": damp,
        "tsc": 1e10,
        "table": 1e10,
    }

    epsp = np.array([[10.0, -5.0, 2.0, 8.0, -4.0, 6.0]])
    deps = epsp * dt
    uvar = np.array([[0.0, le_val]])

    res = sigeps163_oracle(
        matparam=matparam,
        rho0=np.array([1.0]),
        rho=np.array([rho_val]),
        timestep=dt,
        aldt=np.array([le_val]),
        sigo=np.zeros((1, 6)),
        deps=deps,
        epsp=epsp,
        uvar=uvar,
    )

    gama = 1.0 - 1.0 / rho_val  # -0.25
    denom = math.copysign(max(abs(1.0 + gama), _EM20), 1.0 + gama)  # 0.75
    ssp0 = math.sqrt((bulk + _FOUR_THIRD * g) / rho_val)
    expected_a = ssp0 * rho_val * damp * le_val / denom
    np.testing.assert_allclose(res["a"][0], expected_a, rtol=1e-12, atol=1e-12)

    ldav = (10.0 - 5.0 + 2.0) / 3.0
    one_p_nu = 1.2
    one_m_2nu = 0.6
    sigv_xx = expected_a * ((10.0 - ldav) / one_p_nu + ldav / one_m_2nu)
    sigv_yy = expected_a * ((-5.0 - ldav) / one_p_nu + ldav / one_m_2nu)
    sigv_zz = expected_a * ((2.0 - ldav) / one_p_nu + ldav / one_m_2nu)
    sigv_xy = expected_a * 8.0 / (2.0 * one_p_nu)
    sigv_yz = expected_a * (-4.0) / (2.0 * one_p_nu)
    sigv_zx = expected_a * 6.0 / (2.0 * one_p_nu)

    expected_sigv = np.array([sigv_xx, sigv_yy, sigv_zz, sigv_xy, sigv_yz, sigv_zx])
    np.testing.assert_allclose(res["sigv"][0], expected_sigv, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res["sig_tot"][0], res["sign"][0] + expected_sigv, rtol=1e-12, atol=1e-12)


def test_acoustic_sound_speed_evaluation_with_and_without_dt():
    """Verify sound speed formulas lines 267-271 and 292-306, table slope dsdgam, and damping contribution."""
    young = 100.0
    nu = 0.25
    g = 40.0
    bulk = 100.0 / 1.5
    damp = 0.10
    le_val = 1.0
    rho0 = 1.0

    # 1. Without dt (dt = 0): c = sqrt((max(bulk, dsdgam) + 4/3*g) / rho)
    matparam = {
        "young": young,
        "nu": nu,
        "damp": damp,
        "tsc": 1e10,
        "table": 1e10,
    }
    res_dt0 = sigeps163_oracle(
        matparam=matparam,
        rho0=np.array([rho0]),
        rho=np.array([1.25]),
        timestep=0.0,
        aldt=np.array([le_val]),
        sigo=np.zeros((1, 6)),
        deps=np.zeros((1, 6)),
    )
    expected_c0 = math.sqrt((bulk + _FOUR_THIRD * g) / 1.25)
    np.testing.assert_allclose(res_dt0["ssp"][0], expected_c0, rtol=1e-12, atol=1e-12)

    # 2. With dt > 0: damping stiffness abs(a) / dt added
    dt = 1e-4
    res_dt = sigeps163_oracle(
        matparam=matparam,
        rho0=np.array([rho0]),
        rho=np.array([1.25]),
        timestep=dt,
        aldt=np.array([le_val]),
        sigo=np.zeros((1, 6)),
        deps=np.zeros((1, 6)),
    )
    a_val = res_dt["a"][0]
    expected_c_dt = math.sqrt((bulk + _FOUR_THIRD * g + abs(a_val) / dt) / 1.25)
    np.testing.assert_allclose(res_dt["ssp"][0], expected_c_dt, rtol=1e-12, atol=1e-12)

    # 3. Table slope dsdgam > bulk stiffening sound speed
    dsdgam_custom = 250.0
    def custom_yield(gama_in, rate_in):
        return np.full_like(gama_in, 50.0), np.full_like(gama_in, dsdgam_custom)

    res_slope = sigeps163_oracle(
        matparam=matparam,
        rho0=np.array([rho0]),
        rho=np.array([1.25]),
        timestep=0.0,
        aldt=np.array([le_val]),
        sigo=np.zeros((1, 6)),
        deps=np.zeros((1, 6)),
        yield_fn=custom_yield,
    )
    expected_c_slope = math.sqrt((dsdgam_custom + _FOUR_THIRD * g) / 1.25)
    np.testing.assert_allclose(res_slope["ssp"][0], expected_c_slope, rtol=1e-12, atol=1e-12)

    # 4. Density fallback when rho <= 1e-20: denominator falls back to rho0
    res_fallback = sigeps163_oracle(
        matparam=matparam,
        rho0=np.array([rho0]),
        rho=np.array([1.0e-22]),
        timestep=0.0,
        aldt=np.array([le_val]),
        sigo=np.zeros((1, 6)),
        deps=np.zeros((1, 6)),
    )
    expected_c_fallback = math.sqrt((bulk + _FOUR_THIRD * g) / rho0)
    np.testing.assert_allclose(res_fallback["ssp"][0], expected_c_fallback, rtol=1e-12, atol=1e-12)


def test_sound_speed_solid_standalone_parity():
    """Verify sound_speed_solid standalone function matches Fortran sound speed."""
    p = Law163Params(rho0=1.2, e=120.0, nu=0.2, damp=0.12)
    rho = 1.5
    dt = 1e-4
    le = 2.0

    # Call sigeps163_oracle to get reference sound speed
    res = sigeps163_oracle(
        matparam={"young": p.e, "nu": p.nu, "damp": p.damp, "tsc": p.tsc, "table": 1e10},
        rho0=np.array([p.rho0]),
        rho=np.array([rho]),
        timestep=dt,
        aldt=np.array([le]),
        sigo=np.zeros((1, 6)),
        deps=np.zeros((1, 6)),
        uvar=np.array([[0.20, le]]),
    )
    c_oracle = res["ssp"][0]

    # Call standalone sound_speed_solid with extra
    extra_test = {"a": res["a"][0], "le": le, "gama": 0.20}
    c_func = sound_speed_solid(p, rho=rho, extra=extra_test, dt=dt)
    np.testing.assert_allclose(c_func, c_oracle, rtol=1e-12, atol=1e-12)


# =============================================================================
# 3. EXHAUSTIVE 60+ PHYSICAL STATES PARITY: solid_update vs sigeps163_oracle
# =============================================================================

@pytest.fixture
def physical_states_suite():
    """Generate 64 diverse physical states spanning all foam regimes."""
    states = []

    # Yield curve with non-trivial slopes
    gammas_k = np.array([0.0, 0.05, 0.15, 0.35, 0.60, 0.85, 1.0])
    sigy_k = np.array([2.0, 4.5, 8.0, 15.0, 30.0, 75.0, 200.0])
    table = FunctTable(fct_id=1, x=gammas_k, y=sigy_k)

    configs = [
        {"young": 50.0, "nu": 0.0, "tsc": 5.0, "damp": 0.0, "ncycle": 12, "nrs": 0, "srclmt": 1e20, "fscale": 1.0},
        {"young": 100.0, "nu": 0.2, "tsc": 10.0, "damp": 0.10, "ncycle": 10, "nrs": 0, "srclmt": 1e20, "fscale": 1.0},
        {"young": 80.0, "nu": 0.3, "tsc": 8.0, "damp": 0.15, "ncycle": 16, "nrs": 1, "srclmt": 1e20, "fscale": 1.2},
        {"young": 120.0, "nu": 0.15, "tsc": 15.0, "damp": 0.08, "ncycle": 12, "nrs": 1, "srclmt": 5000.0, "fscale": 0.8},
    ]

    densities = [0.6, 0.9, 1.0, 1.25, 1.8, 3.5]
    dts = [1e-5, 5e-5, 1e-4]
    les = [0.5, 1.0, 3.0]

    rng = np.random.default_rng(seed=1630560)

    for i in range(64):
        cfg = configs[i % len(configs)]
        rho_val = densities[i % len(densities)]
        dt_val = dts[i % len(dts)]
        le_val = les[i % len(les)]

        regime = i % 8
        if regime == 0:
            # Pure uniaxial compression (yielding expected)
            deps_val = np.array([-0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
            sigo_val = np.array([-10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif regime == 1:
            # Pure hydrostatic compression
            deps_val = np.array([-0.02, -0.02, -0.02, 0.0, 0.0, 0.0])
            sigo_val = np.array([-15.0, -15.0, -15.0, 0.0, 0.0, 0.0])
        elif regime == 2:
            # Pure tension exceeding cutoff (tsc clamping)
            deps_val = np.array([0.04, 0.03, 0.02, 0.0, 0.0, 0.0])
            sigo_val = np.array([5.0, 4.0, 3.0, 0.0, 0.0, 0.0])
        elif regime == 3:
            # Pure shear state
            deps_val = np.array([0.0, 0.0, 0.0, 0.03, -0.02, 0.025])
            sigo_val = np.array([0.0, 0.0, 0.0, 2.0, -1.5, 1.0])
        elif regime == 4:
            # Mixed: one principal axis compressive yielding, one tensile cutoff, one elastic
            deps_val = np.array([-0.04, 0.05, 0.001, 0.01, -0.01, 0.005])
            sigo_val = np.array([-8.0, 6.0, 0.5, 1.0, -0.5, 0.8])
        elif regime == 5:
            # Elastic regime (well inside yield and tensile cutoff bounds)
            deps_val = np.array([-0.001, 0.001, -0.0005, 0.002, 0.001, -0.001])
            sigo_val = np.array([-0.5, 0.5, -0.2, 0.1, 0.05, -0.05])
        elif regime == 6:
            # High rate jump (triggering srclmt if active)
            deps_val = np.array([-0.10, -0.08, -0.06, 0.04, -0.03, 0.05])
            sigo_val = rng.uniform(-20.0, 5.0, size=6)
        else:
            # Randomized realistic multi-axial dynamic state
            deps_val = rng.normal(0.0, 0.015, size=6)
            sigo_val = rng.uniform(-15.0, 10.0, size=6)

        uvar1_old = rng.uniform(-0.1, 0.4)
        epsd_old = rng.uniform(0.0, 150.0)

        states.append({
            "name": f"state_{i:02d}_regime_{regime}",
            "cfg": cfg,
            "table": table,
            "rho0": 1.0,
            "rho": rho_val,
            "dt": dt_val,
            "le": le_val,
            "sigo": sigo_val,
            "deps": deps_val,
            "uvar": np.array([[uvar1_old, le_val]]),
            "epsd": np.array([epsd_old]),
        })

    return states


def test_exhaustive_60_plus_physical_states_parity(physical_states_suite):
    """Exhaustive differential parity check: solid_update vs sigeps163_oracle across 64 states."""
    passed_count = 0

    for state in physical_states_suite:
        cfg = state["cfg"]
        tbl = state["table"]
        rho0 = state["rho0"]
        rho = state["rho"]
        dt = state["dt"]
        le = state["le"]
        sigo = state["sigo"]
        deps = state["deps"]
        uvar_oracle = state["uvar"].copy()
        epsd_oracle = state["epsd"].copy()

        # Build pyradioss Material model
        params_obj = Law163Params(
            rho0=rho0,
            e=cfg["young"],
            nu=cfg["nu"],
            tsc=cfg["tsc"],
            damp=cfg["damp"],
            ncycle=cfg["ncycle"],
            srclmt=cfg["srclmt"],
            fscale=cfg["fscale"],
            nrs=cfg["nrs"],
            table=tbl,
        )
        mat = build_law163(params_obj)

        # 1. Execute Fortran Oracle
        matparam_oracle = {
            "young": cfg["young"],
            "nu": cfg["nu"],
            "g": params_obj.g,
            "bulk": params_obj.bulk,
            "cii": params_obj.cii,
            "cij": params_obj.cij,
            "tsc": cfg["tsc"],
            "damp": cfg["damp"],
            "ncycle": cfg["ncycle"],
            "srclmt": cfg["srclmt"],
            "nrs": cfg["nrs"],
            "fscale": cfg["fscale"],
            "table": tbl,
        }
        res_oracle = sigeps163_oracle(
            matparam=matparam_oracle,
            rho0=np.array([rho0]),
            rho=np.array([rho]),
            timestep=dt,
            aldt=np.array([le]),
            sigo=sigo.reshape(1, 6),
            deps=deps.reshape(1, 6),
            uvar=uvar_oracle,
            epsd=epsd_oracle,
        )

        # 2. Execute pyradioss solid_update
        extra_py = {
            "rho": rho,
            "le": le,
            "aldt": le,
            "uvar163": state["uvar"].copy(),
            "epsd163": state["epsd"].copy(),
            "table": tbl,
        }
        sig_py, epsp_py, c_py = solid_update(
            mat=mat,
            sig=sigo.copy(),
            deps=deps.copy(),
            dt=dt,
            extra=extra_py,
            return_tuple=True,
        )

        # 3. Assert exact mathematical parity (rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            sig_py,
            res_oracle["sig_tot"][0],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Total stress mismatch on {state['name']}",
        )
        np.testing.assert_allclose(
            epsp_py,
            res_oracle["plas"][0],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Plastic volumetric strain mismatch on {state['name']}",
        )
        np.testing.assert_allclose(
            c_py,
            res_oracle["ssp"][0],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Sound speed mismatch on {state['name']}",
        )

        # History variables parity
        np.testing.assert_allclose(
            extra_py["uvar163"][0, 0],
            res_oracle["uvar"][0, 0],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"uvar1 (gama) mismatch on {state['name']}",
        )
        np.testing.assert_allclose(
            extra_py["epsd163"][0],
            res_oracle["epsd"][0],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"epsd (strain rate) mismatch on {state['name']}",
        )

        # Viscous stresses parity
        np.testing.assert_allclose(
            extra_py["sigv"][0],
            res_oracle["sigv"][0],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Viscous stress sigv mismatch on {state['name']}",
        )

        passed_count += 1

    assert passed_count == 64


def test_vectorized_multi_element_group_parity():
    """Verify batch multi-element execution (nel=8) exactly matches vectorized oracle."""
    nel = 8
    p = Law163Params(rho0=1.1, e=80.0, nu=0.1, tsc=6.0, damp=0.08, ncycle=12, table=25.0)
    mat = build_law163(p)

    rng = np.random.default_rng(seed=42)
    sigo = rng.uniform(-10.0, 5.0, size=(nel, 6))
    deps = rng.uniform(-0.02, 0.02, size=(nel, 6))
    rho = rng.uniform(0.8, 1.8, size=nel)
    le = rng.uniform(0.5, 3.0, size=nel)
    uvar = np.column_stack([rng.uniform(-0.1, 0.2, size=nel), le])
    epsd = rng.uniform(0.0, 50.0, size=nel)
    dt = 2e-4

    # Oracle
    res_oracle = sigeps163_oracle(
        matparam={"young": p.e, "nu": p.nu, "tsc": p.tsc, "damp": p.damp, "table": 25.0},
        rho0=np.full(nel, p.rho0),
        rho=rho,
        timestep=dt,
        aldt=le,
        sigo=sigo,
        deps=deps,
        uvar=uvar.copy(),
        epsd=epsd.copy(),
    )

    # solid_update
    extra = {
        "rho": rho,
        "le": le,
        "uvar163": uvar.copy(),
        "epsd163": epsd.copy(),
        "table": 25.0,
    }
    sig_py, epsp_py, c_py = solid_update(
        mat=mat,
        sig=sigo.copy(),
        deps=deps.copy(),
        dt=dt,
        extra=extra,
        return_tuple=True,
    )

    np.testing.assert_allclose(sig_py, res_oracle["sig_tot"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(epsp_py, res_oracle["plas"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(c_py, res_oracle["ssp"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(extra["sigv"], res_oracle["sigv"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(extra["epsd163"], res_oracle["epsd"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(extra["uvar163"][:, 0], res_oracle["uvar"][:, 0], rtol=1e-12, atol=1e-12)


def test_consistent_solid_tangent_structure():
    """Verify algorithmic consistent solid tangent matches elasticity matrix (cii, cij, g)."""
    p = Law163Params(e=120.0, nu=0.2)
    D = consistent_solid_tangent(p)
    assert D.shape == (1, 6, 6)

    assert math.isclose(D[0, 0, 0], p.cii, rel_tol=1e-12)
    assert math.isclose(D[0, 1, 1], p.cii, rel_tol=1e-12)
    assert math.isclose(D[0, 2, 2], p.cii, rel_tol=1e-12)

    assert math.isclose(D[0, 0, 1], p.cij, rel_tol=1e-12)
    assert math.isclose(D[0, 1, 0], p.cij, rel_tol=1e-12)
    assert math.isclose(D[0, 0, 2], p.cij, rel_tol=1e-12)
    assert math.isclose(D[0, 2, 0], p.cij, rel_tol=1e-12)
    assert math.isclose(D[0, 1, 2], p.cij, rel_tol=1e-12)
    assert math.isclose(D[0, 2, 1], p.cij, rel_tol=1e-12)

    assert math.isclose(D[0, 3, 3], p.g, rel_tol=1e-12)
    assert math.isclose(D[0, 4, 4], p.g, rel_tol=1e-12)
    assert math.isclose(D[0, 5, 5], p.g, rel_tol=1e-12)
