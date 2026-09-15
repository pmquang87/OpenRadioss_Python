r"""Fortran Parity & Physics Oracle Verifier for Milestone M563: /MAT/LAW74 (/MAT/HILL_3D /MAT/ORTH_PLAS /MAT/THERM_HILL).

3D Tabulated Hill Orthotropic Plasticity Model for Solids.
Directly cites and mirrors upstream OpenRadioss source:
- Upstream Fortran starter reader:
  ``starter/source/materials/mat/mat074/hm_read_mat74.F``
- Upstream Fortran engine physics:
  ``engine/source/materials/mat/mat074/sigeps74.F`` (lines 188 to 956)
- Table interpolation tools:
  ``engine/source/tools/curve/table_tools.F`` (TABLE_VINTERP)
- Function interpolation tools:
  ``engine/source/tools/curve/finter.F`` (FINTER)
- CFG card definition:
  ``hm_cfg_files/config/CFG/radioss140/MAT/matl74_74.cfg``

Tests included:
1. Exact Python oracle reproducing sigeps74.F lines 188 to 956.
2. Exact Hill 1948 anisotropic yield coefficients (FF, GG, HH, LL, MM, NN) and isotropic von Mises recovery.
3. Sound speed formula for longitudinal dilatational waves in solids: sqrt((C11 + 4/3*G)/rho0).
4. Maximum tensile principal strain cubic solver with 4 Newton iterations matching exact analytical eigenvalues.
5. Softening factor FAIL calculation and tensile failure regime.
6. Pure elastic 3D states in tension, compression, and shear.
7. Hill directional yield parameters (S11y, S22y, S33y) governing directional yielding.
8. Hill shear yield parameters (S12y, S23y, S31y) governing shear yielding.
9. Triaxial stress and hydrostatic pressure states with AMU volumetric strain.
10. Dynamic Young's modulus degradation: exponential mode (CE > 0, Einf > 0) and curve mode (OPTE = 1, IFUNCE > 0).
11. Cyclic loading with Bauschinger effect under kinematic hardening (CHARD = 0.0, 0.5, 1.0).
12. Element rupture and deletion when accumulated plastic strain pla > eps_max.
13. Adiabatic plastic heating Delta T = yld * dpla / (rho * Cp).
14. Algorithmic tangent operator (6, 6) matching numerical perturbation (1e-8).
15. Vectorized multi-element batch execution.
16. Exhaustive 64 diverse physical states comparison: solid_update vs sigeps74_oracle (rtol=1e-12, atol=1e-12).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytest

from pyradioss.materials.law74_hill_3d import (
    Law74Params,
    build_law74,
    solid_update,
    sound_speed,
    sound_speed_solid,
    consistent_solid_tangent,
    extra_shapes,
)

_EM20 = 1.0e-20
_INF = 1.0e30


# =============================================================================
# 1. UPSTREAM FORTRAN ENGINE ORACLE: sigeps74.F (lines 188 to 956)
# =============================================================================

def eval_curve_1d_oracle(curve: Any, x: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear 1D curve interpolation matching OpenRadioss FINTER."""
    x_arr = np.asarray(x, dtype=np.float64)
    if callable(curve):
        res = curve(x_arr)
        if isinstance(res, tuple):
            return np.asarray(res[0], dtype=np.float64), np.asarray(res[1], dtype=np.float64)
        return np.asarray(res, dtype=np.float64), np.zeros_like(x_arr)

    if isinstance(curve, (list, tuple)) and len(curve) == 2:
        cx = np.asarray(curve[0], dtype=np.float64)
        cy = np.asarray(curve[1], dtype=np.float64)
    elif hasattr(curve, "x") and hasattr(curve, "y"):
        cx = np.asarray(curve.x, dtype=np.float64)
        cy = np.asarray(curve.y, dtype=np.float64)
    elif isinstance(curve, dict) and "x" in curve and "y" in curve:
        cx = np.asarray(curve["x"], dtype=np.float64)
        cy = np.asarray(curve["y"], dtype=np.float64)
    else:
        return np.ones_like(x_arr), np.zeros_like(x_arr)

    if len(cx) == 0:
        return np.ones_like(x_arr), np.zeros_like(x_arr)
    if len(cx) == 1:
        return np.full_like(x_arr, cy[0]), np.zeros_like(x_arr)

    cs = np.diff(cy) / np.maximum(np.diff(cx), _EM20)
    idx = np.clip(np.searchsorted(cx, x_arr, side="right") - 1, 0, len(cx) - 2)
    val = cy[idx] + cs[idx] * (x_arr - cx[idx])
    return val, cs[idx]


def eval_yield_table_oracle(
    table: Any,
    pla: np.ndarray,
    rate: np.ndarray,
    temp: np.ndarray,
    sigy0: float = 1.0,
    t0: float = 293.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate yield stress and hardening slope matching OpenRadioss TABLE_VINTERP."""
    pla_arr = np.asarray(pla, dtype=np.float64)
    rate_arr = np.asarray(rate, dtype=np.float64)
    temp_arr = np.asarray(temp, dtype=np.float64)
    n = len(pla_arr)

    if table is None or table == 0:
        return np.full(n, float(sigy0), dtype=np.float64), np.zeros(n, dtype=np.float64)

    if callable(table):
        try:
            res = table(pla_arr, rate_arr, temp_arr)
        except TypeError:
            try:
                res = table(pla_arr, rate_arr)
            except TypeError:
                res = table(pla_arr)
        if isinstance(res, tuple):
            return np.asarray(res[0], dtype=np.float64), np.asarray(res[1], dtype=np.float64)
        return np.asarray(res, dtype=np.float64), np.zeros(n, dtype=np.float64)

    if isinstance(table, (list, tuple)) and len(table) == 2:
        return eval_curve_1d_oracle(table, pla_arr)

    if hasattr(table, "x") and hasattr(table, "y") and not hasattr(table, "ndim"):
        return eval_curve_1d_oracle(table, pla_arr)

    if isinstance(table, dict):
        if "x1" in table and "y" in table:
            x1 = np.asarray(table.get("x1", [0.0, 1.0]), dtype=np.float64)
            x2 = np.asarray(table.get("x2", [0.0]), dtype=np.float64)
            x3 = np.asarray(table.get("x3", [t0]), dtype=np.float64)
            y = np.asarray(table.get("y", [1.0]), dtype=np.float64)
        elif "x" in table and "y" in table:
            return eval_curve_1d_oracle(table, pla_arr)
        else:
            return np.full(n, float(sigy0), dtype=np.float64), np.zeros(n, dtype=np.float64)
    elif hasattr(table, "x1") and hasattr(table, "y"):
        x1 = np.asarray(table.x1, dtype=np.float64)
        x2 = np.asarray(getattr(table, "x2", [0.0]), dtype=np.float64)
        x3 = np.asarray(getattr(table, "x3", [t0]), dtype=np.float64)
        y = np.asarray(table.y, dtype=np.float64)
    else:
        return np.full(n, float(sigy0), dtype=np.float64), np.zeros(n, dtype=np.float64)

    nx1 = len(x1)
    nx2 = len(x2)
    nx3 = len(x3)

    if nx1 == 0:
        return np.ones(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
    if nx1 == 1:
        return np.full(n, y.flat[0] if y.size > 0 else 1.0, dtype=np.float64), np.zeros(n, dtype=np.float64)

    if y.ndim == 3:
        if y.shape == (nx1, nx2, nx3):
            y_grid = np.transpose(y, (2, 1, 0))
        elif y.shape == (nx3, nx2, nx1):
            y_grid = y
        else:
            y_grid = y.reshape((nx3, nx2, nx1))
    elif y.ndim == 2:
        y_grid = y[None, :, :]
    elif y.ndim == 1:
        if len(y) == nx1:
            return eval_curve_1d_oracle({"x": x1, "y": y}, pla_arr)
        y_grid = y.reshape((nx3, nx2, nx1))
    else:
        return eval_curve_1d_oracle({"x": x1, "y": y.flatten()[:nx1]}, pla_arr)

    i1 = np.clip(np.searchsorted(x1, pla_arr, side="right") - 1, 0, nx1 - 2)
    dx1 = np.maximum(x1[i1 + 1] - x1[i1], _EM20)
    w1 = (pla_arr - x1[i1]) / dx1

    if nx2 > 1:
        i2 = np.clip(np.searchsorted(x2, rate_arr, side="right") - 1, 0, nx2 - 2)
        dx2 = np.maximum(x2[i2 + 1] - x2[i2], _EM20)
        w2 = np.clip((rate_arr - x2[i2]) / dx2, 0.0, 1.0)
    else:
        i2 = np.zeros(n, dtype=int)
        w2 = np.zeros(n, dtype=np.float64)

    if nx3 > 1:
        i3 = np.clip(np.searchsorted(x3, temp_arr, side="right") - 1, 0, nx3 - 2)
        dx3 = np.maximum(x3[i3 + 1] - x3[i3], _EM20)
        w3 = np.clip((temp_arr - x3[i3]) / dx3, 0.0, 1.0)
    else:
        i3 = np.zeros(n, dtype=int)
        w3 = np.zeros(n, dtype=np.float64)

    v000 = y_grid[i3, i2, i1]
    v001 = y_grid[i3, i2, i1 + 1]
    v010 = y_grid[i3, np.minimum(i2 + 1, nx2 - 1), i1]
    v011 = y_grid[i3, np.minimum(i2 + 1, nx2 - 1), i1 + 1]
    v100 = y_grid[np.minimum(i3 + 1, nx3 - 1), i2, i1]
    v101 = y_grid[np.minimum(i3 + 1, nx3 - 1), i2, i1 + 1]
    v110 = y_grid[np.minimum(i3 + 1, nx3 - 1), np.minimum(i2 + 1, nx2 - 1), i1]
    v111 = y_grid[np.minimum(i3 + 1, nx3 - 1), np.minimum(i2 + 1, nx2 - 1), i1 + 1]

    c00 = (1.0 - w2) * v000 + w2 * v010
    c01 = (1.0 - w2) * v001 + w2 * v011
    c10 = (1.0 - w2) * v100 + w2 * v110
    c11 = (1.0 - w2) * v101 + w2 * v111

    c0 = (1.0 - w3) * c00 + w3 * c10
    c1 = (1.0 - w3) * c01 + w3 * c11

    val = c0 + w1 * (c1 - c0)
    slp = (c1 - c0) / dx1
    return val, slp


def solve_max_tensile_strain_oracle(eps_tot: np.ndarray) -> np.ndarray:
    """Exact reproduction of sigeps74.F lines 317-359.

    Solve cubic characteristic equation for maximum tensile principal strain:
      y(x) = x^3 + C*x + D = 0
    with 4 Newton-Raphson iterations.
    """
    eps = np.atleast_2d(np.asarray(eps_tot, dtype=np.float64))
    dav = (eps[:, 0] + eps[:, 1] + eps[:, 2]) / 3.0
    e1 = eps[:, 0] - dav
    e2 = eps[:, 1] - dav
    e3 = eps[:, 2] - dav
    e4 = 0.5 * eps[:, 3]
    e5 = 0.5 * eps[:, 4]
    e6 = 0.5 * eps[:, 5]

    e42 = e4 * e4
    e52 = e5 * e5
    e62 = e6 * e6

    c = -0.5 * (e1 * e1 + e2 * e2 + e3 * e3) - e42 - e52 - e62
    d = -e1 * e2 * e3 + e1 * e52 + e2 * e62 + e3 * e42 - 2.0 * e4 * e5 * e6
    cc = c / 3.0
    epst = np.sqrt(np.maximum(0.0, -cc))

    epst2 = epst * epst
    y = (epst2 + c) * epst + d
    mask = np.abs(y) > 1.0e-8

    if np.any(mask):
        epst_m = 1.75 * epst[mask]
        c_m = c[mask]
        d_m = d[mask]
        for _ in range(4):
            epst2_m = epst_m * epst_m
            y_m = (epst2_m + c_m) * epst_m + d_m
            yp_m = 3.0 * epst2_m + c_m
            nonzero = yp_m != 0.0
            epst_m[nonzero] -= y_m[nonzero] / yp_m[nonzero]
        epst[mask] = epst_m

    return epst + dav


def sigeps74_oracle(
    matparam: Dict[str, Any],
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    uvar: Optional[np.ndarray] = None,
    temp: Optional[Union[float, np.ndarray]] = None,
    off: Optional[Union[float, np.ndarray]] = None,
    eps_tot: Optional[np.ndarray] = None,
    amu: Optional[Union[float, np.ndarray]] = None,
    rho: Optional[Union[float, np.ndarray]] = None,
    rate: Optional[Union[float, np.ndarray]] = None,
    yield_fn: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
    curve_e_fn: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, Any]:
    """Exact Python reproduction of OpenRadioss sigeps74.F lines 188-956.

    Parameters
    ----------
    matparam : dict
        Material parameters:
        'e', 'nu', 's11y', 's22y', 's33y', 's12y', 's23y', 's31y',
        'chard' (fisokin), 'rho0', 'fscale', 'pscale', 'sigy0',
        'eps_max', 'epsr1', 'epsr2', 't0', 'rhocp', 'ifunce', 'einf', 'ce',
        'fsmooth', 'fcut', 'ipla', 'yield_table', 'curve_e'
    sig : ndarray
        Old Cauchy stress tensor [xx, yy, zz, xy, yz, zx] of shape (nel, 6) or (6,).
    deps : ndarray
        Engineering strain increment [xx, yy, zz, xy, yz, zx] of shape (nel, 6) or (6,).
    epsp : float or ndarray, optional
        Accumulated equivalent plastic strain history.
    dt : float, optional
        Time step increment.
    uvar : ndarray, optional
        10 internal history variables [pla, ipos1, ipos2, ipos3, alpha_xx..zx].
    temp : float or ndarray, optional
        Temperature array.
    off : float or ndarray, optional
        Element active/deletion flag.
    eps_tot : ndarray, optional
        Total strain tensor for principal tensile strain solver.
    amu : float or ndarray, optional
        Volumetric strain amu = rho/rho0 - 1.
    rho : float or ndarray, optional
        Current density.
    rate : float or ndarray, optional
        Effective strain rate.
    yield_fn : callable, optional
        Direct yield stress lookup function.
    curve_e_fn : callable, optional
        Direct Young's modulus scale function.

    Returns
    -------
    dict
        Updated outputs matching Fortran:
        'sig': Cauchy stress tensor
        'epsp': accumulated plastic strain
        'alpha': backstress tensor (6,) or (nel, 6)
        'temp': updated temperature
        'off': updated element deletion flag
        'soundsp': longitudinal dilatational sound speed
        'cri': Hill equivalent stress
        'yld': yield stress
        'dpla': plastic strain increment
        'fail': tensile softening factor
        'r': radial return factor
        'p_hydro': hydrostatic pressure
        'uvar': full 10-channel history array
    """
    sig_arr = np.asarray(sig, dtype=np.float64)
    deps_arr = np.asarray(deps, dtype=np.float64)
    is_1d = (sig_arr.ndim == 1)

    if is_1d:
        sig_arr = sig_arr[None, :]
        deps_arr = deps_arr[None, :]

    nel = sig_arr.shape[0]
    if deps_arr.shape[0] == 1 and nel > 1:
        deps_arr = np.repeat(deps_arr, nel, axis=0)

    # 1. Recover material parameters (hm_read_mat74.F lines 121-267)
    e = float(matparam.get("e", matparam.get("E", 210000.0)))
    nu = float(matparam.get("nu", matparam.get("NU", 0.3)))
    rho0 = float(matparam.get("rho0", matparam.get("MAT_RHO", 1.0)))
    s11y = float(matparam.get("s11y", matparam.get("S11Y", 1.0)))
    s22y = float(matparam.get("s22y", matparam.get("S22Y", 1.0)))
    s33y = float(matparam.get("s33y", matparam.get("S33Y", 1.0)))
    s12y = float(matparam.get("s12y", matparam.get("S12Y", 1.0)))
    s23y = float(matparam.get("s23y", matparam.get("S23Y", 1.0)))
    s31y = float(matparam.get("s31y", matparam.get("S31Y", 1.0)))
    fisokin = float(matparam.get("chard", matparam.get("CHARD", matparam.get("fisokin", 0.0))))
    fscale = float(matparam.get("fscale", matparam.get("FScale", 1.0)))
    pscale = float(matparam.get("pscale", matparam.get("PScale", 1.0)))
    xfac = 1.0 / pscale if pscale != 0.0 else 1.0
    yfac = fscale
    sigy0 = float(matparam.get("sigy0", matparam.get("SIGY0", 1.0)))
    eps_max = float(matparam.get("eps_max", matparam.get("EPSMAX", _INF)))
    epsr1 = float(matparam.get("epsr1", matparam.get("EPSR1", _INF)))
    epsr2 = float(matparam.get("epsr2", matparam.get("EPSR2", 2.0 * _INF)))
    t0 = float(matparam.get("t0", matparam.get("T0", 293.0)))
    rhocp = float(matparam.get("rhocp", matparam.get("RHOCP", 0.0)))
    ifunce = int(matparam.get("ifunce", matparam.get("Yr_fun", 0)))
    curve_e_obj = curve_e_fn if curve_e_fn is not None else matparam.get("curve_e")
    opte = int(matparam.get("opte", 1 if (ifunce > 0 or curve_e_obj is not None) else 0))
    einf = float(matparam.get("einf", matparam.get("EINF", 0.0)))
    ce = float(matparam.get("ce", matparam.get("CE", 0.0)))
    fsmooth = int(matparam.get("fsmooth", matparam.get("Fsmooth", 0)))
    fcut = float(matparam.get("fcut", matparam.get("Fcut", 0.0)))
    ipla = int(matparam.get("ipla", matparam.get("IPLA", 0)))

    # Hill 1948 anisotropic constants (hm_read_mat74.F lines 229-234)
    ff = 0.5 * (1.0 / (s22y**2) + 1.0 / (s33y**2) - 1.0 / (s11y**2))
    gg = 0.5 * (1.0 / (s11y**2) + 1.0 / (s33y**2) - 1.0 / (s22y**2))
    hh = 0.5 * (1.0 / (s11y**2) + 1.0 / (s22y**2) - 1.0 / (s33y**2))
    ll = 0.5 / (s23y**2)
    mm = 0.5 / (s31y**2)
    nn = 0.5 / (s12y**2)

    # State history allocations
    uvar_arr = np.zeros((nel, 10), dtype=np.float64)
    if uvar is not None:
        uvar_in = np.asarray(uvar, dtype=np.float64)
        if uvar_in.ndim == 1:
            uvar_arr[0, :min(10, len(uvar_in))] = uvar_in[:min(10, len(uvar_in))]
        else:
            uvar_arr[:, :min(10, uvar_in.shape[1])] = uvar_in[:, :min(10, uvar_in.shape[1])]

    if epsp is not None:
        pla = np.atleast_1d(np.asarray(epsp, dtype=np.float64)).copy()
        if len(pla) == 1 and nel > 1:
            pla = np.full(nel, pla[0], dtype=np.float64)
    else:
        pla = uvar_arr[:, 0].copy()

    if temp is not None:
        temp_arr = np.atleast_1d(np.asarray(temp, dtype=np.float64)).copy()
        if len(temp_arr) == 1 and nel > 1:
            temp_arr = np.full(nel, temp_arr[0], dtype=np.float64)
    else:
        temp_arr = np.full(nel, t0, dtype=np.float64)

    if off is not None:
        off_arr = np.atleast_1d(np.asarray(off, dtype=np.float64)).copy()
        if len(off_arr) == 1 and nel > 1:
            off_arr = np.full(nel, off_arr[0], dtype=np.float64)
    else:
        off_arr = np.ones(nel, dtype=np.float64)

    alpha = uvar_arr[:, 4:10].copy()

    # 2. Dynamic Young's modulus degradation (sigeps74.F lines 220-243)
    e_curr = np.full(nel, e, dtype=np.float64)
    if opte == 1 and curve_e_obj is not None:
        mask_pla = pla > 0.0
        if np.any(mask_pla):
            esc, _ = eval_curve_1d_oracle(curve_e_obj, pla[mask_pla])
            e_curr[mask_pla] = esc * e
    elif ce > 0.0:
        mask_pla = pla > 0.0
        if np.any(mask_pla):
            e_curr[mask_pla] = e - (e - einf) * (1.0 - np.exp(-ce * pla[mask_pla]))

    g1 = 0.5 * e_curr / (1.0 + nu)
    g21 = 2.0 * g1
    g31 = 3.0 * g1
    c11 = e_curr / (3.0 * (1.0 - 2.0 * nu))
    soundsp = np.sqrt((c11 + 4.0 / 3.0 * g1) / max(rho0, _EM20))

    # 3. Backstress shift & Elastic trial stress (sigeps74.F lines 276-304)
    sigo = sig_arr.copy()
    if fisokin != 0.0:
        sigo -= alpha

    p0 = -(sigo[:, 0] + sigo[:, 1] + sigo[:, 2]) / 3.0
    dav = (deps_arr[:, 0] + deps_arr[:, 1] + deps_arr[:, 2]) / 3.0

    st_trial = np.empty_like(sig_arr)
    st_trial[:, 0] = sigo[:, 0] + p0 + g21 * (deps_arr[:, 0] - dav)
    st_trial[:, 1] = sigo[:, 1] + p0 + g21 * (deps_arr[:, 1] - dav)
    st_trial[:, 2] = sigo[:, 2] + p0 + g21 * (deps_arr[:, 2] - dav)
    st_trial[:, 3] = sigo[:, 3] + g1 * deps_arr[:, 3]
    st_trial[:, 4] = sigo[:, 4] + g1 * deps_arr[:, 4]
    st_trial[:, 5] = sigo[:, 5] + g1 * deps_arr[:, 5]

    # 4. Maximum tensile principal strain & softening factor (sigeps74.F lines 317-365)
    if eps_tot is not None:
        etot = np.asarray(eps_tot, dtype=np.float64)
        if etot.ndim == 1:
            etot = etot[None, :]
        if etot.shape[0] == 1 and nel > 1:
            etot = np.repeat(etot, nel, axis=0)
    else:
        etot = deps_arr.copy()

    epst = solve_max_tensile_strain_oracle(etot)
    if epsr2 > epsr1:
        fail = np.clip((epsr2 - epst) / (epsr2 - epsr1), 0.0, 1.0)
    else:
        fail = np.ones(nel, dtype=np.float64)

    # 5. Strain rate evaluation
    if rate is not None:
        rate_arr = np.atleast_1d(np.asarray(rate, dtype=np.float64)).copy()
        if len(rate_arr) == 1 and nel > 1:
            rate_arr = np.full(nel, rate_arr[0], dtype=np.float64)
    elif dt > 0.0:
        exx_d = deps_arr[:, 0] - dav
        eyy_d = deps_arr[:, 1] - dav
        ezz_d = deps_arr[:, 2] - dav
        ee = (exx_d**2 + eyy_d**2 + ezz_d**2
              + 0.5 * (deps_arr[:, 3]**2 + deps_arr[:, 4]**2 + deps_arr[:, 5]**2))
        rate_arr = np.sqrt(np.maximum((2.0 / 3.0) * ee, 0.0)) / dt
    else:
        rate_arr = np.zeros(nel, dtype=np.float64)

    if fsmooth > 0 and fcut > 0.0 and dt > 0.0:
        omega = 2.0 * math.pi * fcut
        alpha_f = min(1.0, omega * dt)
        rate_arr = alpha_f * rate_arr

    # 6. Yield stress lookup (sigeps74.F lines 367-444)
    ytable = yield_fn if yield_fn is not None else matparam.get("yield_table", matparam.get("table_id"))
    yld_val, dydx = eval_yield_table_oracle(ytable, pla, rate_arr * xfac, temp_arr, sigy0=sigy0, t0=t0)
    yld = yfac * yld_val * fail
    h = fail * dydx

    if fisokin != 0.0:
        yk, _ = eval_yield_table_oracle(ytable, np.zeros_like(pla), rate_arr * xfac, temp_arr, sigy0=sigy0, t0=t0)
        yld = (1.0 - fisokin) * yld + fisokin * fail * yfac * yk
    yld = np.maximum(yld, _EM20)

    # 7. Hill 3D equivalent stress (sigeps74.F lines 450-454)
    cri2 = (ff * (st_trial[:, 1] - st_trial[:, 2])**2
            + gg * (st_trial[:, 2] - st_trial[:, 0])**2
            + hh * (st_trial[:, 0] - st_trial[:, 1])**2
            + 2.0 * ll * st_trial[:, 4]**2
            + 2.0 * mm * st_trial[:, 5]**2
            + 2.0 * nn * st_trial[:, 3]**2)
    cri = np.sqrt(np.maximum(cri2, 0.0))

    # 8. Radial return projection (sigeps74.F lines 448-513)
    r = np.minimum(1.0, yld / np.maximum(cri, _EM20))

    if amu is not None:
        amu_arr = np.atleast_1d(np.asarray(amu, dtype=np.float64)).copy()
        if len(amu_arr) == 1 and nel > 1:
            amu_arr = np.full(nel, amu_arr[0], dtype=np.float64)
        p_hydro = c11 * amu_arr
    elif rho is not None:
        rho_arr = np.atleast_1d(np.asarray(rho, dtype=np.float64)).copy()
        if len(rho_arr) == 1 and nel > 1:
            rho_arr = np.full(nel, rho_arr[0], dtype=np.float64)
        amu_arr = rho_arr / max(rho0, _EM20) - 1.0
        p_hydro = c11 * amu_arr
    else:
        p_hydro = p0 - c11 * 3.0 * dav

    if ipla == 1:
        dpla = (1.0 - r) * cri / np.maximum(g31 + h, _EM20)
        yld = np.maximum(yld + (1.0 - fisokin) * dpla * h, 0.0)
        r = np.minimum(1.0, yld / np.maximum(cri, _EM20))
        pla += dpla
    elif ipla == 2:
        dpla = (1.0 - r) * cri / np.maximum(g31, _EM20)
        pla += dpla
    else:  # IPLA == 0
        dpla = (1.0 - r) * cri / np.maximum(g31 + h, _EM20)
        pla += dpla

    s_dev = st_trial * r[:, None]

    # 9. Kinematic hardening update (sigeps74.F lines 517-551)
    if fisokin != 0.0:
        ds = st_trial - s_dev
        hkin = (2.0 / 3.0) * fisokin * h
        denom_kin = g21 + hkin
        alpha_fac = np.where(denom_kin > _EM20, hkin / denom_kin, 0.0)
        delta_alpha = alpha_fac[:, None] * ds
        alpha += delta_alpha
        uvar_arr[:, 4:10] = alpha

    # 10. Total stress tensor
    sig_new = s_dev.copy()
    sig_new[:, 0] -= p_hydro
    sig_new[:, 1] -= p_hydro
    sig_new[:, 2] -= p_hydro

    if fisokin != 0.0:
        sig_new += alpha

    # 11. Adiabatic plastic heating (sigeps74.F lines 908-910)
    if rhocp > 0.0:
        dtemp = (yld * dpla) / rhocp
        temp_arr += dtemp

    # 12. Element deletion when pla > eps_max (sigeps74.F lines 914-926)
    off_arr[off_arr < 0.1] = 0.0
    off_arr[off_arr < 1.0] *= 0.8
    rupture = (pla > eps_max) & (off_arr == 1.0)
    off_arr[rupture] = 0.8
    deleted = (off_arr == 0.0)
    if np.any(deleted):
        sig_new[deleted] = 0.0

    uvar_arr[:, 0] = pla

    return {
        "sig": sig_new[0] if is_1d else sig_new,
        "epsp": float(pla[0]) if is_1d else pla,
        "alpha": alpha[0] if is_1d else alpha,
        "temp": float(temp_arr[0]) if is_1d else temp_arr,
        "off": float(off_arr[0]) if is_1d else off_arr,
        "soundsp": float(soundsp[0]) if is_1d else soundsp,
        "cri": float(cri[0]) if is_1d else cri,
        "yld": float(yld[0]) if is_1d else yld,
        "dpla": float(dpla[0]) if is_1d else dpla,
        "fail": float(fail[0]) if is_1d else fail,
        "r": float(r[0]) if is_1d else r,
        "p_hydro": float(p_hydro[0]) if is_1d else p_hydro,
        "uvar": uvar_arr[0] if is_1d else uvar_arr,
    }


# =============================================================================
# 2. UNIT TESTS FOR FOUNDATIONAL FORTRAN PHYSICS COMPONENTS
# =============================================================================

def test_hill_1948_coefficients_and_isotropic_recovery():
    """Verify exact formulas for FF, GG, HH, LL, MM, NN and isotropic von Mises recovery."""
    # 1. Von Mises isotropic recovery: S_ii = 1.0, S_ij = 1 / sqrt(3)
    p_iso = Law74Params(
        s11y=1.0, s22y=1.0, s33y=1.0,
        s12y=1.0 / math.sqrt(3.0),
        s23y=1.0 / math.sqrt(3.0),
        s31y=1.0 / math.sqrt(3.0),
    )
    assert p_iso.ff == pytest.approx(0.5, rel=1e-12)
    assert p_iso.gg == pytest.approx(0.5, rel=1e-12)
    assert p_iso.hh == pytest.approx(0.5, rel=1e-12)
    assert p_iso.ll == pytest.approx(1.5, rel=1e-12)
    assert p_iso.mm == pytest.approx(1.5, rel=1e-12)
    assert p_iso.nn == pytest.approx(1.5, rel=1e-12)

    # CRI^2 = 0.5*(syy-szz)^2 + 0.5*(szz-sxx)^2 + 0.5*(sxx-syy)^2 + 3*(syz^2 + szx^2 + sxy^2) = sigma_vm^2
    s_test = np.array([400.0, 100.0, -200.0, 50.0, -30.0, 20.0])
    sxx, syy, szz, sxy, syz, szx = s_test
    cri2 = (p_iso.ff * (syy - szz)**2 + p_iso.gg * (szz - sxx)**2 + p_iso.hh * (sxx - syy)**2
            + 2.0 * p_iso.ll * syz**2 + 2.0 * p_iso.mm * szx**2 + 2.0 * p_iso.nn * sxy**2)
    vm2 = 0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2) + 3.0 * (sxy**2 + syz**2 + szx**2)
    assert cri2 == pytest.approx(vm2, rel=1e-12)

    # 2. General anisotropic parameters (hm_read_mat74.F lines 229-234)
    s11, s22, s33 = 450.0, 380.0, 320.0
    s12, s23, s31 = 200.0, 170.0, 190.0
    p_aniso = Law74Params(s11y=s11, s22y=s22, s33y=s33, s12y=s12, s23y=s23, s31y=s31)

    exp_ff = 0.5 * (1.0 / s22**2 + 1.0 / s33**2 - 1.0 / s11**2)
    exp_gg = 0.5 * (1.0 / s11**2 + 1.0 / s33**2 - 1.0 / s22**2)
    exp_hh = 0.5 * (1.0 / s11**2 + 1.0 / s22**2 - 1.0 / s33**2)
    exp_ll = 0.5 / s23**2
    exp_mm = 0.5 / s31**2
    exp_nn = 0.5 / s12**2

    assert p_aniso.ff == pytest.approx(exp_ff, rel=1e-14)
    assert p_aniso.gg == pytest.approx(exp_gg, rel=1e-14)
    assert p_aniso.hh == pytest.approx(exp_hh, rel=1e-14)
    assert p_aniso.ll == pytest.approx(exp_ll, rel=1e-14)
    assert p_aniso.mm == pytest.approx(exp_mm, rel=1e-14)
    assert p_aniso.nn == pytest.approx(exp_nn, rel=1e-14)


def test_sound_speed_exact_formula():
    """Verify longitudinal acoustic sound speed: sqrt((C11 + 4/3*G)/rho0)."""
    e = 210000.0
    nu = 0.28
    rho0 = 7.8e-6

    p = Law74Params(e=e, nu=nu, rho0=rho0)
    mat = build_law74(p)

    g = 0.5 * e / (1.0 + nu)
    c1 = e / (3.0 * (1.0 - 2.0 * nu))
    expected_c = math.sqrt((c1 + 4.0 / 3.0 * g) / rho0)

    assert p.soundsp == pytest.approx(expected_c, rel=1e-14)
    assert sound_speed(mat) == pytest.approx(expected_c, rel=1e-14)
    assert sound_speed_solid(mat) == pytest.approx(expected_c, rel=1e-14)


def test_cubic_solver_maximum_tensile_strain_4_newton_steps():
    """Verify cubic solver for maximum tensile principal strain against analytical eigenvalues."""
    # State A: Pure hydrostatic strain
    eps_hydro = np.array([0.012, 0.012, 0.012, 0.0, 0.0, 0.0])
    epst_hydro = solve_max_tensile_strain_oracle(eps_hydro)[0]
    assert epst_hydro == pytest.approx(0.012, rel=1e-12)

    # State B: Uniaxial tension
    eps_uni = np.array([0.025, -0.0075, -0.0075, 0.0, 0.0, 0.0])
    epst_uni = solve_max_tensile_strain_oracle(eps_uni)[0]
    assert epst_uni == pytest.approx(0.025, rel=1e-12)

    # State C: Pure shear (eigenvalues: +gamma/2, 0, -gamma/2)
    gamma = 0.016
    eps_shear = np.array([0.0, 0.0, 0.0, gamma, 0.0, 0.0])
    epst_shear = solve_max_tensile_strain_oracle(eps_shear)[0]
    assert epst_shear == pytest.approx(0.5 * gamma, rel=1e-12)

    # State D: Arbitrary 3D strain states compared with np.linalg.eigvalsh
    np.random.seed(42)
    for _ in range(10):
        eps_rand = np.random.uniform(-0.02, 0.03, size=6)
        epst_calc = solve_max_tensile_strain_oracle(eps_rand)[0]

        # Construct 3x3 symmetric tensor
        e_mat = np.array([
            [eps_rand[0], 0.5 * eps_rand[3], 0.5 * eps_rand[5]],
            [0.5 * eps_rand[3], eps_rand[1], 0.5 * eps_rand[4]],
            [0.5 * eps_rand[5], 0.5 * eps_rand[4], eps_rand[2]],
        ])
        eigs = np.linalg.eigvalsh(e_mat)
        max_eig = eigs[-1]

        assert epst_calc == pytest.approx(max_eig, rel=1e-8, abs=1e-10)
        # Verify exact parity between oracle and pyradioss solver
        from pyradioss.materials.law74_hill_3d import _solve_max_principal_strain
        epst_py = _solve_max_principal_strain(eps_rand[None, :])[0]
        assert epst_calc == pytest.approx(epst_py, rel=1e-15, abs=1e-15)


def test_tensile_failure_softening_factor():
    """Verify softening factor FAIL = max(0, min(1, (epsr2 - epst)/(epsr2 - epsr1)))."""
    p_dict = {
        "e": 200000.0, "nu": 0.3, "sigy0": 300.0,
        "epsr1": 0.02, "epsr2": 0.04,
    }

    # Case 1: Below threshold epst <= epsr1 => FAIL = 1.0
    res1 = sigeps74_oracle(p_dict, np.zeros(6), np.array([0.015, -0.005, -0.005, 0, 0, 0]))
    assert res1["fail"] == pytest.approx(1.0, rel=1e-14)

    # Case 2: Linear softening epsr1 < epst < epsr2 => FAIL = (0.04 - 0.03) / (0.04 - 0.02) = 0.5
    res2 = sigeps74_oracle(p_dict, np.zeros(6), np.array([0.030, -0.01, -0.01, 0, 0, 0]))
    assert res2["fail"] == pytest.approx(0.5, rel=1e-10)

    # Case 3: Above threshold epst >= epsr2 => FAIL = 0.0
    res3 = sigeps74_oracle(p_dict, np.zeros(6), np.array([0.045, -0.015, -0.015, 0, 0, 0]))
    assert res3["fail"] == pytest.approx(0.0, rel=1e-14)


# =============================================================================
# 3. TARGETED PHYSICS VERIFICATION TESTS
# =============================================================================

def test_pure_elastic_3d_states():
    """Verify 3D Hookean elastic response in tension, compression, and shear."""
    params = {
        "e": 200000.0,
        "nu": 0.3,
        "rho0": 7.8e-6,
        "sigy0": 500.0,
    }
    p = Law74Params(**params)
    mat = build_law74(p)

    c11 = p.c1 + 4.0 / 3.0 * p.g
    c12 = p.c1 - 2.0 / 3.0 * p.g
    g = p.g

    # 1. Uniaxial tension X below yield: deps = [1e-4, -3e-5, -3e-5, 0, 0, 0]
    deps_x = np.array([1.0e-4, -3.0e-5, -3.0e-5, 0.0, 0.0, 0.0])
    sig_out, epsp_out, c_out = solid_update(mat, np.zeros(6), deps_x, return_tuple=True)
    oracle_res = sigeps74_oracle(params, np.zeros(6), deps_x)

    np.testing.assert_allclose(sig_out, oracle_res["sig"], rtol=1e-12, atol=1e-12)
    assert epsp_out == 0.0
    assert sig_out[0] == pytest.approx(200000.0 * 1.0e-4, rel=1e-12)
    assert abs(sig_out[1]) < 1e-10
    assert abs(sig_out[2]) < 1e-10

    # 2. Pure shear XY below yield: deps = [0, 0, 0, 2e-4, 0, 0]
    deps_xy = np.array([0.0, 0.0, 0.0, 2.0e-4, 0.0, 0.0])
    sig_xy, epsp_xy = solid_update(mat, np.zeros(6), deps_xy)
    oracle_xy = sigeps74_oracle(params, np.zeros(6), deps_xy)

    np.testing.assert_allclose(sig_xy, oracle_xy["sig"], rtol=1e-12, atol=1e-12)
    assert epsp_xy == 0.0
    assert sig_xy[3] == pytest.approx(g * 2.0e-4, rel=1e-12)

    # 3. Triaxial hydrostatic strain: deps = [1e-4, 1e-4, 1e-4, 0, 0, 0]
    deps_hyd = np.array([1.0e-4, 1.0e-4, 1.0e-4, 0.0, 0.0, 0.0])
    sig_hyd, epsp_hyd = solid_update(mat, np.zeros(6), deps_hyd)
    oracle_hyd = sigeps74_oracle(params, np.zeros(6), deps_hyd)

    np.testing.assert_allclose(sig_hyd, oracle_hyd["sig"], rtol=1e-12, atol=1e-12)
    expected_p = 3.0 * p.c1 * 1.0e-4
    assert sig_hyd[0] == pytest.approx(expected_p, rel=1e-12)
    assert sig_hyd[1] == pytest.approx(expected_p, rel=1e-12)
    assert sig_hyd[2] == pytest.approx(expected_p, rel=1e-12)


def test_hill_directional_yield_stresses_xyz():
    """Verify anisotropic directional yield stresses along X, Y, Z axes."""
    s11, s22, s33 = 400.0, 320.0, 260.0
    params = {
        "e": 200000.0, "nu": 0.3,
        "s11y": s11, "s22y": s22, "s33y": s33,
        "s12y": 200.0, "s23y": 200.0, "s31y": 200.0,
        "sigy0": 1.0,
    }
    p = Law74Params(**params)
    mat = build_law74(p)

    # Large strain to drive plastic deformation
    # Tension X
    deps_x = np.array([0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0])
    sig_x, epsp_x = solid_update(mat, np.zeros(6), deps_x)
    ora_x = sigeps74_oracle(params, np.zeros(6), deps_x)
    np.testing.assert_allclose(sig_x, ora_x["sig"], rtol=1e-12, atol=1e-12)
    assert epsp_x > 0.0

    # Tension Y
    deps_y = np.array([-0.0015, 0.005, -0.0015, 0.0, 0.0, 0.0])
    sig_y, epsp_y = solid_update(mat, np.zeros(6), deps_y)
    ora_y = sigeps74_oracle(params, np.zeros(6), deps_y)
    np.testing.assert_allclose(sig_y, ora_y["sig"], rtol=1e-12, atol=1e-12)
    assert epsp_y > 0.0

    # Tension Z
    deps_z = np.array([-0.0015, -0.0015, 0.005, 0.0, 0.0, 0.0])
    sig_z, epsp_z = solid_update(mat, np.zeros(6), deps_z)
    ora_z = sigeps74_oracle(params, np.zeros(6), deps_z)
    np.testing.assert_allclose(sig_z, ora_z["sig"], rtol=1e-12, atol=1e-12)
    assert epsp_z > 0.0


def test_hill_shear_yield_stresses():
    """Verify shear yield parameters along XY, YZ, ZX planes."""
    s12, s23, s31 = 230.0, 180.0, 210.0
    params = {
        "e": 200000.0, "nu": 0.3,
        "s11y": 1.0, "s22y": 1.0, "s33y": 1.0,
        "s12y": s12, "s23y": s23, "s31y": s31,
        "sigy0": 1.0,
    }
    p = Law74Params(**params)
    mat = build_law74(p)

    # Pure shear XY
    deps_xy = np.array([0.0, 0.0, 0.0, 0.010, 0.0, 0.0])
    sig_xy, epsp_xy = solid_update(mat, np.zeros(6), deps_xy)
    ora_xy = sigeps74_oracle(params, np.zeros(6), deps_xy)
    np.testing.assert_allclose(sig_xy, ora_xy["sig"], rtol=1e-12, atol=1e-12)
    assert epsp_xy > 0.0

    # Pure shear YZ
    deps_yz = np.array([0.0, 0.0, 0.0, 0.0, 0.010, 0.0])
    sig_yz, epsp_yz = solid_update(mat, np.zeros(6), deps_yz)
    ora_yz = sigeps74_oracle(params, np.zeros(6), deps_yz)
    np.testing.assert_allclose(sig_yz, ora_yz["sig"], rtol=1e-12, atol=1e-12)
    assert epsp_yz > 0.0

    # Pure shear ZX
    deps_zx = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.010])
    sig_zx, epsp_zx = solid_update(mat, np.zeros(6), deps_zx)
    ora_zx = sigeps74_oracle(params, np.zeros(6), deps_zx)
    np.testing.assert_allclose(sig_zx, ora_zx["sig"], rtol=1e-12, atol=1e-12)
    assert epsp_zx > 0.0


def test_triaxial_pressure_states():
    """Verify hydrostatic pressure handling via AMU and its decoupling from Hill deviator."""
    params = {
        "e": 200000.0, "nu": 0.3, "rho0": 7.8e-6,
        "s11y": 1.0, "s22y": 1.0, "s33y": 1.0,
        "s12y": 1.0 / math.sqrt(3.0),
        "s23y": 1.0 / math.sqrt(3.0),
        "s31y": 1.0 / math.sqrt(3.0),
        "sigy0": 350.0,
    }
    p = Law74Params(**params)
    mat = build_law74(p)

    c11 = p.c1
    amu_val = 0.015  # positive compression: P = C11 * 0.015
    deps = np.array([0.006, -0.002, -0.002, 0.003, 0.0, 0.0])

    extra = {"uvar74": np.zeros(10), "amu": np.array([amu_val])}
    sig_out, epsp_out = solid_update(mat, np.zeros(6), deps, extra=extra)
    ora = sigeps74_oracle(params, np.zeros(6), deps, amu=amu_val)

    np.testing.assert_allclose(sig_out, ora["sig"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(epsp_out, ora["epsp"], rtol=1e-12, atol=1e-12)
    assert ora["p_hydro"] == pytest.approx(c11 * amu_val, rel=1e-12)


def test_dynamic_youngs_modulus_degradation_exponential_and_curve():
    """Verify dynamic Young's modulus degradation and resulting sound speed change."""
    # 1. Exponential degradation: E(pla) = E0 - (E0 - Einf) * (1 - exp(-CE * pla))
    params_exp = {
        "e": 200000.0, "nu": 0.3, "einf": 120000.0, "ce": 15.0,
        "s11y": 1.0, "s22y": 1.0, "s33y": 1.0,
        "s12y": 1.0 / math.sqrt(3.0), "s23y": 1.0 / math.sqrt(3.0), "s31y": 1.0 / math.sqrt(3.0),
        "sigy0": 350.0,
    }
    p_exp = Law74Params(**params_exp)
    mat_exp = build_law74(p_exp)

    uvar = np.zeros(10)
    extra = {"uvar74": uvar}
    deps = np.array([0.006, -0.0018, -0.0018, 0, 0, 0])

    # Step 1: Initial plastic deformation
    s1, ep1, snd1 = solid_update(mat_exp, np.zeros(6), deps, extra=extra, return_tuple=True)
    ora1 = sigeps74_oracle(params_exp, np.zeros(6), deps, uvar=np.zeros(10))
    np.testing.assert_allclose(s1, ora1["sig"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ep1, ora1["epsp"], rtol=1e-12, atol=1e-12)

    # Step 2: Second increment with pla > 0 causing modulus degradation
    s1_copy = s1.copy()
    uvar_copy = uvar.copy()
    ora2 = sigeps74_oracle(params_exp, s1_copy, deps, epsp=ep1, uvar=uvar_copy)
    s2, ep2, snd2 = solid_update(mat_exp, s1, deps, epsp=ep1, extra=extra, return_tuple=True)
    np.testing.assert_allclose(s2, ora2["sig"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ep2, ora2["epsp"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(snd2, ora2["soundsp"], rtol=1e-12, atol=1e-12)
    assert snd2 < snd1

    # 2. Curve mode: OPTE = 1
    curve_e = ([0.0, 0.05, 0.10], [1.0, 0.85, 0.70])
    params_crv = {
        "e": 200000.0, "nu": 0.3, "ifunce": 1, "curve_e": curve_e,
        "s11y": 1.0, "s22y": 1.0, "s33y": 1.0,
        "s12y": 1.0 / math.sqrt(3.0), "s23y": 1.0 / math.sqrt(3.0), "s31y": 1.0 / math.sqrt(3.0),
        "sigy0": 350.0,
    }
    p_crv = Law74Params(**params_crv)
    mat_crv = build_law74(p_crv)

    uvar_crv = np.zeros(10)
    extra_crv = {"uvar74": uvar_crv}
    s_c1, ep_c1, snd_c1 = solid_update(mat_crv, np.zeros(6), deps, extra=extra_crv, return_tuple=True)
    s_c1_copy = s_c1.copy()
    uvar_crv_copy = uvar_crv.copy()
    ora_c2 = sigeps74_oracle(params_crv, s_c1_copy, deps, epsp=ep_c1, uvar=uvar_crv_copy, curve_e_fn=curve_e)
    s_c2, ep_c2, snd_c2 = solid_update(mat_crv, s_c1, deps, epsp=ep_c1, extra=extra_crv, return_tuple=True)

    np.testing.assert_allclose(s_c2, ora_c2["sig"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ep_c2, ora_c2["epsp"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(snd_c2, ora_c2["soundsp"], rtol=1e-12, atol=1e-12)


def test_cyclic_loading_and_bauschinger_effect():
    """Verify mixed isotropic/kinematic hardening and Bauschinger effect across CHARD = 0, 0.5, 1.0."""
    curve = ([0.0, 0.05, 0.10], [300.0, 400.0, 480.0])

    for chard in [0.0, 0.5, 1.0]:
        params = {
            "e": 200000.0, "nu": 0.3, "chard": chard,
            "s11y": 1.0, "s22y": 1.0, "s33y": 1.0,
            "s12y": 1.0 / math.sqrt(3.0), "s23y": 1.0 / math.sqrt(3.0), "s31y": 1.0 / math.sqrt(3.0),
            "yield_table": curve,
        }
        p = Law74Params(**params)
        mat = build_law74(p)

        uvar = np.zeros(10)
        extra = {"uvar74": uvar}

        # Step 1: Forward tension loading
        deps_fwd = np.array([0.005, -0.0015, -0.0015, 0, 0, 0])
        s1, ep1 = solid_update(mat, np.zeros(6), deps_fwd, extra=extra)
        ora1 = sigeps74_oracle(params, np.zeros(6), deps_fwd, uvar=np.zeros(10), yield_fn=curve)

        np.testing.assert_allclose(s1, ora1["sig"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ep1, ora1["epsp"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(uvar[4:10], ora1["alpha"], rtol=1e-12, atol=1e-12)

        if chard == 0.0:
            assert np.all(uvar[4:10] == 0.0)
        else:
            assert uvar[4] > 0.0  # alpha_xx > 0

        # Step 2: Reverse compression loading
        deps_rev = np.array([-0.006, 0.0018, 0.0018, 0, 0, 0])
        s1_copy = s1.copy()
        uvar_copy = uvar.copy()
        ora2 = sigeps74_oracle(params, s1_copy, deps_rev, epsp=ep1, uvar=uvar_copy, yield_fn=curve)
        s2, ep2 = solid_update(mat, s1, deps_rev, epsp=ep1, extra=extra)

        np.testing.assert_allclose(s2, ora2["sig"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ep2, ora2["epsp"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(uvar[4:10], ora2["alpha"], rtol=1e-12, atol=1e-12)


def test_failure_softening_regime_epst():
    """Verify tensile softening factor degradation between epsr1 and epsr2."""
    params = {
        "e": 200000.0, "nu": 0.3, "sigy0": 400.0,
        "epsr1": 0.002, "epsr2": 0.006,
        "s11y": 1.0, "s22y": 1.0, "s33y": 1.0,
        "s12y": 1.0 / math.sqrt(3.0), "s23y": 1.0 / math.sqrt(3.0), "s31y": 1.0 / math.sqrt(3.0),
    }
    p = Law74Params(**params)
    mat = build_law74(p)

    # Apply strain such that epst ~ 0.004 (midway between 0.002 and 0.006)
    deps = np.array([0.004, -0.0012, -0.0012, 0, 0, 0])
    sig_out, epsp_out = solid_update(mat, np.zeros(6), deps)
    ora = sigeps74_oracle(params, np.zeros(6), deps)

    np.testing.assert_allclose(sig_out, ora["sig"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(epsp_out, ora["epsp"], rtol=1e-12, atol=1e-12)
    assert 0.0 < ora["fail"] < 1.0


def test_element_rupture_and_deletion_epsmax():
    """Verify progressive damage decay and final deletion when pla > eps_max."""
    params = {
        "e": 200000.0, "nu": 0.3, "sigy0": 350.0, "eps_max": 0.002,
        "s11y": 1.0, "s22y": 1.0, "s33y": 1.0,
        "s12y": 1.0 / math.sqrt(3.0), "s23y": 1.0 / math.sqrt(3.0), "s31y": 1.0 / math.sqrt(3.0),
    }
    p = Law74Params(**params)
    mat = build_law74(p)

    deps = np.array([0.005, -0.0015, -0.0015, 0, 0, 0])
    off = np.array([1.0])
    uvar = np.zeros(10)
    extra = {"uvar74": uvar, "off": off}

    # Step 1: Rupture triggered (off: 1.0 -> 0.8)
    s1, ep1 = solid_update(mat, np.zeros(6), deps, extra=extra)
    ora1 = sigeps74_oracle(params, np.zeros(6), deps, off=1.0)
    assert off[0] == pytest.approx(0.8, rel=1e-12)
    np.testing.assert_allclose(s1, ora1["sig"], rtol=1e-12, atol=1e-12)

    # Step 2: Decay (off: 0.8 -> 0.64)
    s1_copy = s1.copy()
    uvar_copy = uvar.copy()
    off_copy = off.copy()
    ora2 = sigeps74_oracle(params, s1_copy, deps, epsp=ep1, off=off_copy, uvar=uvar_copy)
    s2, ep2 = solid_update(mat, s1, deps, epsp=ep1, extra=extra)
    assert off[0] == pytest.approx(0.64, rel=1e-12)
    np.testing.assert_allclose(s2, ora2["sig"], rtol=1e-12, atol=1e-12)

    # Simulate deletion: off < 0.1 -> 0.0 and sig -> 0
    off[0] = 0.08
    s3, ep3 = solid_update(mat, s2, deps, epsp=ep2, extra=extra)
    assert off[0] == 0.0
    assert np.all(s3 == 0.0)


def test_adiabatic_plastic_heating():
    """Verify adiabatic plastic heating Delta T = yld * dpla / (rho * Cp)."""
    rhocp_val = 3.5e6
    t0_val = 295.0
    params = {
        "e": 200000.0, "nu": 0.3, "sigy0": 400.0,
        "rhocp": rhocp_val, "t0": t0_val,
        "s11y": 1.0, "s22y": 1.0, "s33y": 1.0,
        "s12y": 1.0 / math.sqrt(3.0), "s23y": 1.0 / math.sqrt(3.0), "s31y": 1.0 / math.sqrt(3.0),
    }
    p = Law74Params(**params)
    mat = build_law74(p)

    temp_arr = np.array([t0_val])
    extra = {"uvar74": np.zeros(10), "temp": temp_arr}
    deps = np.array([0.005, -0.0015, -0.0015, 0, 0, 0])

    s_out, ep_out = solid_update(mat, np.zeros(6), deps, extra=extra)
    ora = sigeps74_oracle(params, np.zeros(6), deps, temp=t0_val)

    np.testing.assert_allclose(s_out, ora["sig"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(temp_arr[0], ora["temp"], rtol=1e-12, atol=1e-12)
    assert temp_arr[0] > t0_val


def test_algorithmic_tangent_consistency():
    """Verify consistent solid tangent matches elasticity matrix in elastic range and finite differences in plastic range."""
    params = {
        "e": 210000.0, "nu": 0.3, "sigy0": 350.0,
        "s11y": 1.0, "s22y": 1.0, "s33y": 1.0,
        "s12y": 1.0 / math.sqrt(3.0), "s23y": 1.0 / math.sqrt(3.0), "s31y": 1.0 / math.sqrt(3.0),
    }
    p = Law74Params(**params)
    mat = build_law74(p)

    # 1. Pure elastic state tangent
    c_el_num = consistent_solid_tangent(mat, np.zeros(6))
    c11_exp = p.c1 + 4.0 / 3.0 * p.g
    c12_exp = p.c1 - 2.0 / 3.0 * p.g
    g_exp = p.g

    assert c_el_num[0, 0] == pytest.approx(c11_exp, rel=1e-12)
    assert c_el_num[1, 1] == pytest.approx(c11_exp, rel=1e-12)
    assert c_el_num[2, 2] == pytest.approx(c11_exp, rel=1e-12)
    assert c_el_num[0, 1] == pytest.approx(c12_exp, rel=1e-12)
    assert c_el_num[0, 2] == pytest.approx(c12_exp, rel=1e-12)
    assert c_el_num[3, 3] == pytest.approx(g_exp, rel=1e-12)
    assert c_el_num[4, 4] == pytest.approx(g_exp, rel=1e-12)
    assert c_el_num[5, 5] == pytest.approx(g_exp, rel=1e-12)

    # 2. Plastic tangent matching numerical perturbation (1e-8)
    deps_plas = np.array([0.004, -0.0012, -0.0012, 0.002, 0.0, 0.0])
    c_plas = consistent_solid_tangent(mat, np.zeros(6), deps=deps_plas, h=1.0e-8)

    # Check finite difference approximation independently
    h = 1.0e-8
    d_fd = np.zeros((6, 6))
    for j in range(6):
        dp = deps_plas.copy()
        dm = deps_plas.copy()
        dp[j] += h
        dm[j] -= h
        sp, _ = solid_update(mat, np.zeros(6), dp)
        sm, _ = solid_update(mat, np.zeros(6), dm)
        d_fd[:, j] = (sp - sm) / (2.0 * h)

    np.testing.assert_allclose(c_plas, d_fd, rtol=1e-7, atol=1e-5)


def test_vectorized_multi_element_batch_parity():
    """Verify vectorized multi-element batch execution produces identical results to single-element evaluations."""
    nel = 5
    params = {
        "e": 200000.0, "nu": 0.28, "rho0": 7.8e-6, "sigy0": 380.0,
        "s11y": 1.2, "s22y": 1.0, "s33y": 0.85,
        "s12y": 0.7, "s23y": 0.65, "s31y": 0.75,
        "chard": 0.4,
    }
    p = Law74Params(**params)
    mat = build_law74(p)

    sig_batch = np.zeros((nel, 6))
    deps_batch = np.zeros((nel, 6))
    for i in range(nel):
        deps_batch[i, 0] = 0.001 * (i + 1)
        deps_batch[i, 1] = -0.0003 * (i + 1)
        deps_batch[i, 2] = -0.0003 * (i + 1)
        deps_batch[i, 3] = 0.0005 * i

    uvar_batch = np.zeros((nel, 10))
    extra_batch = {"uvar74": uvar_batch}

    sig_batch_in = sig_batch.copy()
    sig_out_batch, epsp_out_batch, snd_out_batch = solid_update(
        mat, sig_batch_in, deps_batch, extra=extra_batch, return_tuple=True
    )

    # Compare each element independently against single-element solid_update and oracle
    for i in range(nel):
        uvar_single = np.zeros(10)
        extra_single = {"uvar74": uvar_single}
        s_s, ep_s, snd_s = solid_update(mat, np.zeros(6), deps_batch[i].copy(), extra=extra_single, return_tuple=True)
        ora_s = sigeps74_oracle(params, np.zeros(6), deps_batch[i].copy())

        np.testing.assert_allclose(sig_out_batch[i], s_s, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(sig_out_batch[i], ora_s["sig"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(epsp_out_batch[i], ep_s, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(epsp_out_batch[i], ora_s["epsp"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(snd_out_batch[i], snd_s, rtol=1e-12, atol=1e-12)


# =============================================================================
# 4. EXHAUSTIVE PARAMETERIZED COMPARISON SUITE (64 DIVERSE PHYSICAL STATES)
# =============================================================================

@pytest.mark.parametrize("state_idx", range(64))
def test_exhaustive_solid_oracle_comparison_64_states(state_idx: int):
    """Exhaustively verify solid_update against sigeps74_oracle across 64 distinct physical configurations.

    Tests full matrix of:
    - Pure elastic 3D states in tension, compression, and shear
    - Uniaxial tension along X, Y, Z testing directional yield parameters
    - Pure shear states testing shear yield parameters
    - Triaxial tension/compression with hydrostatic pressure AMU
    - Dynamic Young's modulus degradation (exponential CE and curve IFUNCE)
    - Cyclic loading and Bauschinger effect with CHARD = 0.0, 0.5, 1.0
    - Failure softening regime (epst between epsr1 and epsr2, and epst > epsr2)
    - Element deletion at eps_max
    - Adiabatic plastic heating
    - Single and multi-element batches
    """
    # Base yield curves and tables
    curve_hard = ([0.0, 0.02, 0.08, 0.20], [350.0, 420.0, 520.0, 620.0])
    curve_e = ([0.0, 0.05, 0.15], [1.0, 0.88, 0.72])
    table_rate_temp = {
        "x1": [0.0, 0.05, 0.15],
        "x2": [0.0, 100.0],
        "x3": [293.0, 500.0],
        "y": np.array([
            [[350.0, 420.0, 500.0], [380.0, 460.0, 550.0]],
            [[310.0, 370.0, 440.0], [340.0, 410.0, 480.0]],
        ]),
    }

    # Configuration builder based on state_idx
    # 1. Moduli
    e_list = [210000.0, 195000.0, 180000.0, 72000.0]
    nu_list = [0.30, 0.28, 0.33, 0.22]
    e_val = e_list[state_idx % 4]
    nu_val = nu_list[(state_idx // 4) % 4]
    rho0_val = 7.85e-6 if state_idx % 2 == 0 else 2.7e-6

    # 2. Anisotropy directional ratios
    if state_idx < 16:
        # Isotropic or standard ratios
        s11, s22, s33 = 1.0, 1.0, 1.0
        s12, s23, s31 = 1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)
    elif state_idx < 32:
        # Mild orthotropy
        s11, s22, s33 = 1.15, 0.95, 0.90
        s12, s23, s31 = 0.65, 0.60, 0.70
    elif state_idx < 48:
        # Strong orthotropy
        s11, s22, s33 = 1.40, 1.10, 0.80
        s12, s23, s31 = 0.75, 0.55, 0.65
    else:
        # Anisotropic with direct stress scaling
        s11, s22, s33 = 450.0, 380.0, 320.0
        s12, s23, s31 = 230.0, 190.0, 210.0

    # 3. Hardening parameter CHARD
    chard_vals = [0.0, 0.25, 0.50, 0.75, 1.0]
    chard_val = chard_vals[(state_idx // 3) % 5]

    # 4. Modulus degradation & failure parameters
    ce_val = 12.0 if (state_idx % 8 == 3 or state_idx % 8 == 7) else 0.0
    einf_val = 130000.0 if ce_val > 0.0 else 0.0
    ifunce_val = 1 if (state_idx % 8 == 4) else 0
    crv_e_arg = curve_e if ifunce_val == 1 else None

    # Failure parameters
    if state_idx % 8 == 5:
        epsr1_val = 0.003
        epsr2_val = 0.008
        eps_max_val = _INF
    elif state_idx % 8 == 6:
        epsr1_val = _INF
        epsr2_val = 2.0 * _INF
        eps_max_val = 0.003
    else:
        epsr1_val = _INF
        epsr2_val = 2.0 * _INF
        eps_max_val = _INF

    # Thermal parameters
    rhocp_val = 3.5e6 if state_idx % 4 == 2 else 0.0
    t0_val = 293.0 + (state_idx % 5) * 5.0

    # Yield table selection
    if state_idx >= 48:
        ytable_arg = None
        sigy0_val = 1.0
    elif state_idx % 4 == 0:
        ytable_arg = None
        sigy0_val = 350.0
    elif state_idx % 4 == 1:
        ytable_arg = curve_hard
        sigy0_val = 350.0
    else:
        ytable_arg = table_rate_temp
        sigy0_val = 350.0

    # Batch size
    nel = 3 if (state_idx == 63) else 1

    p_dict = {
        "e": e_val,
        "nu": nu_val,
        "rho0": rho0_val,
        "s11y": s11,
        "s22y": s22,
        "s33y": s33,
        "s12y": s12,
        "s23y": s23,
        "s31y": s31,
        "chard": chard_val,
        "ce": ce_val,
        "einf": einf_val,
        "ifunce": ifunce_val,
        "curve_e": crv_e_arg,
        "epsr1": epsr1_val,
        "epsr2": epsr2_val,
        "eps_max": eps_max_val,
        "rhocp": rhocp_val,
        "t0": t0_val,
        "yield_table": ytable_arg,
        "sigy0": sigy0_val,
    }

    mat = build_law74(Law74Params(**p_dict))

    # Initial states
    sig_init = np.zeros((nel, 6), dtype=np.float64)
    deps = np.zeros((nel, 6), dtype=np.float64)
    epsp_init = np.zeros(nel, dtype=np.float64)
    alpha_init = np.zeros((nel, 6), dtype=np.float64)
    amu_val = None

    # Load configuration
    category = state_idx % 8
    if category == 0:
        # Elastic only
        deps[:, 0] = 0.0002
        deps[:, 1] = -0.00006
        deps[:, 2] = -0.00006
        deps[:, 3] = 0.0001
    elif category == 1:
        # Uniaxial tension X plastic
        deps[:, 0] = 0.005
        deps[:, 1] = -0.0015
        deps[:, 2] = -0.0015
    elif category == 2:
        # Pure shear XY plastic with thermal heating
        deps[:, 3] = 0.009
    elif category == 3:
        # Triaxial state with AMU hydrostatic pressure
        amu_val = 0.008
        deps[:, 0] = 0.004
        deps[:, 1] = 0.002
        deps[:, 2] = -0.001
        deps[:, 4] = 0.003
    elif category == 4:
        # Modulus degradation with established plastic strain
        epsp_init[:] = 0.02
        deps[:, 1] = 0.006
        deps[:, 0] = -0.0018
        deps[:, 2] = -0.0018
    elif category == 5:
        # Softening regime
        deps[:, 0] = 0.005
        deps[:, 1] = -0.0015
        deps[:, 2] = -0.0015
    elif category == 6:
        # Element deletion regime
        epsp_init[:] = 0.0028
        deps[:, 2] = 0.006
        deps[:, 0] = -0.0018
        deps[:, 1] = -0.0018
    else:
        # Reverse cyclic loading
        sig_init[:, 0] = 250.0
        sig_init[:, 1] = -80.0
        alpha_init[:, 0] = 40.0
        alpha_init[:, 1] = -20.0
        epsp_init[:] = 0.015
        deps[:, 0] = -0.006
        deps[:, 1] = 0.0018
        deps[:, 2] = 0.0018

    # Multi-element differentiation
    if nel > 1:
        for k in range(nel):
            deps[k] *= (k + 1) * 0.7

    uvar_init = np.zeros((nel, 10), dtype=np.float64)
    uvar_init[:, 0] = epsp_init
    uvar_init[:, 4:10] = alpha_init

    temp_init = np.full(nel, t0_val, dtype=np.float64)
    off_init = np.ones(nel, dtype=np.float64)

    extra = {
        "uvar74": uvar_init.copy(),
        "temp": temp_init.copy(),
        "off": off_init.copy(),
    }
    if amu_val is not None:
        extra["amu"] = np.full(nel, amu_val, dtype=np.float64)

    dt_val = 1.0e-6

    sig_in = sig_init[0].copy() if nel == 1 else sig_init.copy()
    deps_in = deps[0].copy() if nel == 1 else deps.copy()
    epsp_in = float(epsp_init[0]) if nel == 1 else epsp_init.copy()
    uvar_in = uvar_init[0].copy() if nel == 1 else uvar_init.copy()
    temp_in = float(temp_init[0]) if nel == 1 else temp_init.copy()
    off_in = float(off_init[0]) if nel == 1 else off_init.copy()

    # Execute sigeps74_oracle before solid_update modifies extra in-place
    oracle_res = sigeps74_oracle(
        p_dict,
        sig_in.copy(),
        deps_in.copy(),
        epsp=epsp_in,
        dt=dt_val,
        uvar=uvar_in.copy(),
        temp=temp_in,
        off=off_in,
        amu=amu_val,
        yield_fn=ytable_arg,
        curve_e_fn=crv_e_arg,
    )

    # Execute solid_update
    sig_out, epsp_out, soundsp_out = solid_update(
        mat, sig_in, deps_in, epsp=epsp_in, dt=dt_val, extra=extra, return_tuple=True
    )

    # Strict double-precision comparison: rtol=1e-12, atol=1e-12
    if nel == 1:
        np.testing.assert_allclose(sig_out, oracle_res["sig"], rtol=1e-12, atol=1e-12,
                                   err_msg=f"State {state_idx} stress mismatch")
        assert math.isclose(epsp_out, oracle_res["epsp"], rel_tol=1e-12, abs_tol=1e-12), \
            f"State {state_idx} epsp mismatch"
        assert math.isclose(soundsp_out, oracle_res["soundsp"], rel_tol=1e-12, abs_tol=1e-12), \
            f"State {state_idx} soundsp mismatch"
        np.testing.assert_allclose(extra["uvar74"][0, 4:10], oracle_res["alpha"], rtol=1e-12, atol=1e-12,
                                   err_msg=f"State {state_idx} alpha mismatch")
        assert math.isclose(extra["temp"][0], oracle_res["temp"], rel_tol=1e-12, abs_tol=1e-12), \
            f"State {state_idx} temp mismatch"
        assert math.isclose(extra["off"][0], oracle_res["off"], rel_tol=1e-12, abs_tol=1e-12), \
            f"State {state_idx} off mismatch"
    else:
        np.testing.assert_allclose(sig_out, oracle_res["sig"], rtol=1e-12, atol=1e-12,
                                   err_msg=f"State {state_idx} batch stress mismatch")
        np.testing.assert_allclose(epsp_out, oracle_res["epsp"], rtol=1e-12, atol=1e-12,
                                   err_msg=f"State {state_idx} batch epsp mismatch")
        np.testing.assert_allclose(soundsp_out, oracle_res["soundsp"], rtol=1e-12, atol=1e-12,
                                   err_msg=f"State {state_idx} batch soundsp mismatch")
        np.testing.assert_allclose(extra["uvar74"][:, 4:10], oracle_res["alpha"], rtol=1e-12, atol=1e-12,
                                   err_msg=f"State {state_idx} batch alpha mismatch")
        np.testing.assert_allclose(extra["temp"], oracle_res["temp"], rtol=1e-12, atol=1e-12,
                                   err_msg=f"State {state_idx} batch temp mismatch")
        np.testing.assert_allclose(extra["off"], oracle_res["off"], rtol=1e-12, atol=1e-12,
                                   err_msg=f"State {state_idx} batch off mismatch")
