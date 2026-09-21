"""
LAW73 — Thermal Hill Orthotropic Plasticity for Shell Elements (/MAT/LAW73, /MAT/BARLAT2000, /MAT/HILL_THERM).

Upstream Fortran reference:
  - Starter reader: starter/source/materials/mat/mat073/hm_read_mat73.F
  - Engine physics: engine/source/materials/mat/mat073/sigeps73c.F
  - Table interpolation: engine/source/tools/curve/table_tools.F (TABLE_VINTERP)
  - CFG card definition: hm_cfg_files/config/CFG/radioss140/MAT/matl73_BARLAT2000.cfg

Theory & Algorithm
------------------
Hill 1948 quadratic orthotropic yield criterion in plane stress:
    SVM = sqrt(A01 * sign_xx^2 + A02 * sign_yy^2 - A03 * sign_xx * sign_yy + A12 * sign_xy^2)

Hill Anisotropy Constants computed from Lankford coefficients (R00, R45, R90):
    R = 0.25 * (R00 + 2.0 * R45 + R90)
    H = R / (1.0 + R)
    A01 = H * (1.0 + 1.0 / R00)
    A02 = H * (1.0 + 1.0 / R90)
    A03 = 2.0 * H
    A12 = (2.0 * R45 + 1.0) * (A01 + A02 - A03)
    if Iyield > 0:
        A02 /= A01
        A03 /= A01
        A12 /= A01
        A01 = 1.0

Mixed Isotropic-Kinematic Hardening:
    F_isokin = CHARD in [0, 1]
    YLD = (1.0 - F_isokin) * FAIL * Y(pla, epsd*xfac, temp) + F_isokin * FAIL * Y(0, epsd*xfac, temp)
    H_slope = FAIL * dY/dpla

Plastic Return Mapping:
    2-iteration Newton-Raphson scheme in plane stress (sigeps73c.F:322-397)
    solving for plastic strain increment dpla.
    Back-stress tensor alpha evolved along plastic normal:
        alpha_xx += (2 * P1 + P2) * dr0
        alpha_yy += (2 * P2 + P1) * dr0
        alpha_xy += 0.5 * P3 * dr0

Dynamic Young's Modulus Degradation:
    if OPTE == 1 (ifunce > 0): E = E0 * f_E(pla)
    elif CE > 0: E = E0 - (E0 - Einf) * (1.0 - exp(-CE * pla))

Adiabatic Plastic Heating:
    temp = T0 + (eint1 + eint2) * rhocp_inv / vol0

Failure & Deletion:
    FAIL = clamp((EPSR2 - epst) / (EPSR2 - EPSR1), 0.0, 1.0)
    if pla > EPSMAX: off = 0.8 * off
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20
_INF = 1.0e30


# ============================================================================
# 1D & 3D Analytical Hill 1948 Relations & Flow Stress
# ============================================================================

def hill48_yield_criterion_3d(
    sig: Union[Sequence[float], np.ndarray],
    F: float,
    G: float,
    H: float,
    L: float,
    M: float,
    N: float,
) -> float | np.ndarray:
    r"""Evaluate Hill 1948 3D quadratic orthotropic equivalent yield stress:
        \sigma_y = \sqrt{ F(\sigma_{22} - \sigma_{33})^2 + G(\sigma_{33} - \sigma_{11})^2
                        + H(\sigma_{11} - \sigma_{22})^2 + 2 L \sigma_{23}^2
                        + 2 M \sigma_{31}^2 + 2 N \sigma_{12}^2 }

    Cited from:
      - engine/source/materials/mat/mat073/sigeps73c.F (Hill 1948 yield formulation)
      - starter/source/materials/mat/mat073/hm_read_mat73.F (lines 179-191)

    Parameters:
        sig: Stress tensor: (6,) Voigt [xx, yy, zz, xy, yz, zx], (3, 3) matrix,
             or (N, 6) array.
        F, G, H, L, M, N: Hill anisotropy constants
    """
    sig_arr = np.asarray(sig, dtype=float)
    is_1d = (sig_arr.ndim == 1 and sig_arr.size == 6)
    is_tensor = (sig_arr.ndim == 2 and sig_arr.shape == (3, 3))

    if is_tensor:
        s11, s22, s33 = sig_arr[0, 0], sig_arr[1, 1], sig_arr[2, 2]
        s12, s23, s31 = sig_arr[0, 1], sig_arr[1, 2], sig_arr[2, 0]
    elif is_1d:
        s11, s22, s33 = sig_arr[0], sig_arr[1], sig_arr[2]
        s12, s23, s31 = sig_arr[3], sig_arr[4], sig_arr[5]
    elif sig_arr.ndim == 2 and sig_arr.shape[1] == 6:
        s11, s22, s33 = sig_arr[:, 0], sig_arr[:, 1], sig_arr[:, 2]
        s12, s23, s31 = sig_arr[:, 3], sig_arr[:, 4], sig_arr[:, 5]
    else:
        raise ValueError(f"Expected stress with 6 components or (3, 3) matrix, got shape {sig_arr.shape}")

    val = (
        float(F) * (s22 - s33) ** 2
        + float(G) * (s33 - s11) ** 2
        + float(H) * (s11 - s22) ** 2
        + 2.0 * float(L) * s23 ** 2
        + 2.0 * float(M) * s31 ** 2
        + 2.0 * float(N) * s12 ** 2
    )
    res = np.sqrt(np.maximum(val, 0.0))
    return float(res) if (is_1d or is_tensor) else res


def hill48_yield_criterion_plane_stress(
    sig: Union[Sequence[float], np.ndarray],
    A01: float,
    A02: float,
    A03: float,
    A12: float,
) -> float | np.ndarray:
    r"""Evaluate Hill 1948 plane-stress orthotropic equivalent yield stress:
        \sigma_{eq} = \sqrt{ A_{01} \sigma_{xx}^2 + A_{02} \sigma_{yy}^2
                           - A_{03} \sigma_{xx} \sigma_{yy} + A_{12} \sigma_{xy}^2 }

    Cited from:
      - engine/source/materials/mat/mat073/sigeps73c.F (lines 298-302, SVM calculation)
      - starter/source/materials/mat/mat073/hm_read_mat73.F (lines 179-191)

    Parameters:
        sig: Plane stress [xx, yy, xy] (or (N, 3) array)
        A01, A02, A03, A12: Plane-stress Hill sheet anisotropy coefficients
    """
    sig_arr = np.asarray(sig, dtype=float)
    is_1d = (sig_arr.ndim == 1 and sig_arr.size >= 3)
    if is_1d:
        sxx, syy, sxy = sig_arr[0], sig_arr[1], sig_arr[2]
    elif sig_arr.ndim == 2 and sig_arr.shape[1] >= 3:
        sxx, syy, sxy = sig_arr[:, 0], sig_arr[:, 1], sig_arr[:, 2]
    else:
        raise ValueError(f"Expected at least 3 stress components, got shape {sig_arr.shape}")

    val = (
        float(A01) * sxx ** 2
        + float(A02) * syy ** 2
        - float(A03) * sxx * syy
        + float(A12) * sxy ** 2
    )
    res = np.sqrt(np.maximum(val, 0.0))
    return float(res) if is_1d else res


def hill48_lankford_to_anisotropy(
    r00: float,
    r45: float,
    r90: float,
    iyield: int = 0,
) -> Tuple[float, float, float, float, float, float]:
    r"""Compute plane-stress Hill 1948 coefficients from Lankford parameters R00, R45, R90:
        R = 0.25 * (R00 + 2 * R45 + R90)
        H = R / (1 + R)
        A01 = H * (1 + 1 / R00)
        A02 = H * (1 + 1 / R90)
        A03 = 2 * H
        A12 = (2 * R45 + 1) * (A01 + A02 - A03)
        if iyield > 0: normalize so A01 = 1.0

    Cited from:
      - starter/source/materials/mat/mat073/hm_read_mat73.F (lines 179-191)

    Returns:
        (A01, A02, A03, A12, R, H)
    """
    r0 = max(float(r00), 1e-6)
    r4 = max(float(r45), 1e-6)
    r9 = max(float(r90), 1e-6)

    r = 0.25 * (r0 + 2.0 * r4 + r9)
    h = r / (1.0 + r)
    a01 = h * (1.0 + 1.0 / r0)
    a02 = h * (1.0 + 1.0 / r9)
    a03 = 2.0 * h
    a12 = (2.0 * r4 + 1.0) * (a01 + a02 - a03)

    if int(iyield) > 0 and a01 > 0.0:
        a02 /= a01
        a03 /= a01
        a12 /= a01
        a01 = 1.0

    return a01, a02, a03, a12, r, h


def hill48_lankford_to_3d_coefficients(
    r00: float,
    r45: float,
    r90: float,
) -> Tuple[float, float, float, float, float, float]:
    r"""Convert Lankford parameters to 3D Hill 1948 constants (F, G, H, L, M, N):
        H = R00 / (1 + R00)
        F = R00 / (R90 * (1 + R00))
        G = 1 / (1 + R00)
        N = (R00 + R90) * (2 * R45 + 1) / (2 * R90 * (1 + R00))
        L = M = 1.5  (standard isotropic transverse shear)

    Cited from:
      - engine/source/materials/mat/mat073/sigeps73c.F
      - starter/source/materials/mat/mat073/hm_read_mat73.F
    """
    r0 = max(float(r00), 1e-6)
    r4 = max(float(r45), 1e-6)
    r9 = max(float(r90), 1e-6)

    denom = 1.0 + r0
    h = r0 / denom
    f = r0 / (r9 * denom)
    g = 1.0 / denom
    n = (r0 + r9) * (2.0 * r4 + 1.0) / (2.0 * r9 * denom)
    l = 1.5
    m = 1.5
    return f, g, h, l, m, n


def hill48_thermal_yield_stress(
    pla: float | np.ndarray,
    rate: float | np.ndarray = 0.0,
    temp: float | np.ndarray = 293.0,
    yield_table: Any = None,
    fscale: float = 1.0,
    pscale: float = 1.0,
    chard: float = 0.0,
) -> Tuple[float | np.ndarray, float | np.ndarray]:
    r"""Evaluate temperature- and strain-rate-dependent flow stress and hardening slope.

    Supports callable table(pla, rate, temp), numeric constant, or piecewise table.

    Cited from:
      - engine/source/materials/mat/mat073/sigeps73c.F (lines 252-291)
      - engine/source/tools/curve/table_tools.F (TABLE_VINTERP)
    """
    is_scalar = np.isscalar(pla) and np.isscalar(rate) and np.isscalar(temp)
    pla_arr = np.atleast_1d(np.asarray(pla, dtype=float))
    rate_arr = np.atleast_1d(np.asarray(rate, dtype=float))
    temp_arr = np.atleast_1d(np.asarray(temp, dtype=float))

    if yield_table is None:
        yld = np.full_like(pla_arr, 1.0 * float(fscale))
        slope = np.zeros_like(pla_arr)
    elif isinstance(yield_table, (int, float, np.floating, np.integer)):
        yld = np.full_like(pla_arr, float(yield_table) * float(fscale))
        slope = np.zeros_like(pla_arr)
    elif callable(yield_table):
        vals = []
        slopes = []
        for p_i, r_i, t_i in zip(pla_arr, rate_arr, temp_arr):
            res = yield_table(p_i, r_i * (1.0 / max(float(pscale), 1e-20)), t_i)
            if isinstance(res, (tuple, list)) and len(res) >= 2:
                vals.append(float(res[0]))
                slopes.append(float(res[1]))
            else:
                vals.append(float(res))
                slopes.append(0.0)
        yld = np.array(vals, dtype=float) * float(fscale)
        slope = np.array(slopes, dtype=float) * float(fscale)
    else:
        mock_p = Law73Params(fscale=fscale, pscale=pscale, yield_table=yield_table, chard=chard)
        yld, slope, _ = _eval_yield_table(mock_p, pla_arr, rate_arr, temp_arr)

    if is_scalar:
        return float(yld[0]), float(slope[0])
    return yld, slope


# ============================================================================
# Parameter Dataclass: Law73Params
# ============================================================================

@dataclass
class Law73Params:
    """Strongly-typed parameters for /MAT/LAW73 (/MAT/BARLAT2000, /MAT/HILL_THERM).

    Follows starter/source/materials/mat/mat073/hm_read_mat73.F.
    """
    # Density
    rho0: float = 1.0
    refer_rho: float = 1.0

    # Elasticity
    e: float = 210000.0
    nu: float = 0.3

    # Dynamic Young's modulus evolution
    ifunce: int = 0
    einf: float = 0.0
    ce: float = 0.0

    # Lankford anisotropy coefficients
    r00: float = 1.0
    r45: float = 1.0
    r90: float = 1.0

    # Hardening
    chard: float = 0.0      # fisokin: iso-kinematic hardening factor [0..1]
    iyield: int = 0        # 1: normalized in dir 1
    eps_max: float = _INF  # maximum plastic strain
    epsr1: float = _INF    # tensile strain start of failure
    epsr2: float = 2.0 * _INF  # tensile strain end of failure

    # Yield table & scaling
    table_id: int = 0      # table/curve ID or object
    fscale: float = 1.0    # yield stress scale factor
    pscale: float = 1.0    # strain rate scale factor

    # Thermal properties
    t0: float = 293.0      # initial temperature
    rhocp: float = 0.0     # heat capacity per unit volume (rho * Cp)

    # Identifiers & extra objects
    id: int = 1
    title: str = "LAW73_HILL_THERM"
    curve_e: Any = None
    yield_table: Any = None

    # Derived coefficients (computed in __post_init__)
    g: float = field(init=False, default=0.0)
    a11: float = field(init=False, default=0.0)
    a21: float = field(init=False, default=0.0)
    c1: float = field(init=False, default=0.0)
    r: float = field(init=False, default=1.0)
    h: float = field(init=False, default=0.5)
    a01: float = field(init=False, default=1.0)
    a02: float = field(init=False, default=1.0)
    a03: float = field(init=False, default=1.0)
    a12: float = field(init=False, default=3.0)
    xfac: float = field(init=False, default=1.0)
    yfac: float = field(init=False, default=1.0)
    opte: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.refer_rho <= 0.0:
            self.refer_rho = self.rho0
        if self.rho0 <= 0.0:
            self.rho0 = self.refer_rho if self.refer_rho > 0.0 else 1.0

        if self.nu >= 0.5:
            self.nu = 0.499
        if self.nu < 0.0:
            self.nu = 0.0

        # Lankford defaults (hm_read_mat73.F:156-158)
        if self.r00 <= 0.0:
            self.r00 = 1.0
        if self.r45 <= 0.0:
            self.r45 = 1.0
        if self.r90 <= 0.0:
            self.r90 = 1.0

        if self.eps_max <= 0.0:
            self.eps_max = _INF
        if self.epsr1 <= 0.0:
            self.epsr1 = _INF
        if self.epsr2 <= 0.0:
            self.epsr2 = 2.0 * _INF
        if self.t0 <= 0.0:
            self.t0 = 293.0

        if self.fscale <= 0.0:
            self.fscale = 1.0
        if self.pscale <= 0.0:
            self.pscale = 1.0

        # Elastic derived constants (hm_read_mat73.F:172-178)
        self.a11 = self.e / max(1.0 - self.nu ** 2, 1.0e-15)
        self.a21 = self.nu * self.a11
        self.g = 0.5 * self.e / max(1.0 + self.nu, 1.0e-15)
        self.c1 = self.e / (3.0 * max(1.0 - 2.0 * self.nu, 1.0e-15))

        # Lankford Hill parameters (hm_read_mat73.F:179-191)
        self.r = 0.25 * (self.r00 + 2.0 * self.r45 + self.r90)
        self.h = self.r / (1.0 + self.r)
        self.a01 = self.h * (1.0 + 1.0 / self.r00)
        self.a02 = self.h * (1.0 + 1.0 / self.r90)
        self.a03 = 2.0 * self.h
        self.a12 = (2.0 * self.r45 + 1.0) * (self.a01 + self.a02 - self.a03)

        if self.iyield > 0:
            self.a02 /= self.a01
            self.a03 /= self.a01
            self.a12 /= self.a01
            self.a01 = 1.0

        self.xfac = 1.0 / self.pscale
        self.yfac = self.fscale
        self.opte = 1 if (self.ifunce > 0 or self.curve_e is not None) else 0

    @property
    def E(self) -> float:
        return self.e

    @property
    def nu0(self) -> float:
        return self.nu

    @property
    def G(self) -> float:
        return self.g

    @property
    def fisokin(self) -> float:
        return self.chard

    @property
    def soundsp(self) -> float:
        return math.sqrt(self.a11 / max(self.rho0, _EM20))

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)


# ============================================================================
# Curve & Table Evaluation Utilities
# ============================================================================

def _eval_curve_1d(curve: Any, x: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear evaluation of 1D curve returning (value, slope).

    Extrapolates with end-segment slope matching OpenRadioss FINTER.
    """
    x_arr = np.asarray(x, dtype=float)
    if hasattr(curve, "eval"):
        val = np.asarray(curve.eval(x_arr), dtype=float)
        slope = getattr(curve, "slope", np.zeros_like(val))
        if isinstance(slope, np.ndarray) and slope.size > 0:
            slp = np.full_like(val, slope[-1] if len(slope) > 0 else 0.0)
        else:
            slp = np.zeros_like(val)
        return val, slp

    if hasattr(curve, "x") and hasattr(curve, "y"):
        cx = np.asarray(curve.x, dtype=float)
        cy = np.asarray(curve.y, dtype=float)
    elif hasattr(curve, "data"):
        data = np.asarray(curve.data, dtype=float)
        if data.ndim == 2 and data.shape[1] >= 2:
            cx, cy = data[:, 0], data[:, 1]
        else:
            cx, cy = np.array([]), np.array([])
    elif isinstance(curve, (list, tuple)):
        if len(curve) == 2 and isinstance(curve[0], (list, tuple, np.ndarray)) and isinstance(curve[1], (list, tuple, np.ndarray)):
            arr0 = np.asarray(curve[0], dtype=float)
            arr1 = np.asarray(curve[1], dtype=float)
            if len(arr0) != 2 or len(arr1) != 2:
                cx, cy = arr0, arr1
            elif isinstance(curve[0], tuple) and isinstance(curve, list):
                pts = np.asarray(curve, dtype=float)
                cx, cy = pts[:, 0], pts[:, 1]
            elif arr0[1] > arr0[0] and (arr1[0] > arr0[1] or arr1[1] > arr0[1]):
                cx, cy = arr0, arr1
            else:
                pts = np.asarray(curve, dtype=float)
                cx, cy = pts[:, 0], pts[:, 1]
        else:
            try:
                data = np.asarray(curve, dtype=float)
                if data.ndim == 2 and data.shape[1] >= 2:
                    cx, cy = data[:, 0], data[:, 1]
                else:
                    cx, cy = np.array([]), np.array([])
            except Exception:
                cx, cy = np.array([]), np.array([])
    elif isinstance(curve, dict) and "x" in curve and "y" in curve:
        cx = np.asarray(curve["x"], dtype=float)
        cy = np.asarray(curve["y"], dtype=float)
    elif callable(curve):
        res = curve(x_arr)
        if isinstance(res, tuple):
            return np.asarray(res[0], dtype=float), np.asarray(res[1], dtype=float)
        return np.asarray(res, dtype=float), np.zeros_like(x_arr)
    else:
        return np.ones_like(x_arr), np.zeros_like(x_arr)

    if len(cx) == 0:
        return np.ones_like(x_arr), np.zeros_like(x_arr)
    if len(cx) == 1:
        return np.full_like(x_arr, cy[0]), np.zeros_like(x_arr)

    cs = np.diff(cy) / np.maximum(np.diff(cx), _EM20)
    idx = np.minimum(np.maximum(np.searchsorted(cx, x_arr, side="right") - 1, 0), len(cx) - 2)
    val = cy[idx] + cs[idx] * (x_arr - cx[idx])
    return val, cs[idx]


def _eval_yield_table(p: Law73Params, pla: np.ndarray, rate: np.ndarray,
                      temp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate yield stress and hardening slope dY/dpla from table/curve/callable.

    Upstream reference: engine/source/tools/curve/table_tools.F (TABLE_VINTERP).
    Coordinates:
      x1 = pla (plastic strain)
      x2 = rate * xfac (effective strain rate)
      x3 = temp (temperature)

    Returns:
      (yld, dydx1) where dydx1 is the partial derivative w.r.t. plastic strain.
    """
    table = p.yield_table if p.yield_table is not None else p.table_id
    pla_arr = np.asarray(pla, dtype=float)
    rate_arr = np.asarray(rate, dtype=float)
    temp_arr = np.asarray(temp, dtype=float)
    n = len(pla_arr)

    if table is None or table == 0:
        sigy0 = float(p.get("sigy0", 1.0))
        return np.full(n, sigy0, dtype=float), np.zeros(n, dtype=float)

    if callable(table):
        try:
            res = table(pla_arr, rate_arr, temp_arr)
        except TypeError:
            try:
                res = table(pla_arr, rate_arr)
            except TypeError:
                res = table(pla_arr)
        if isinstance(res, tuple):
            return np.asarray(res[0], dtype=float), np.asarray(res[1], dtype=float)
        return np.asarray(res, dtype=float), np.zeros(n, dtype=float)

    if hasattr(table, "x") and hasattr(table, "y") and not hasattr(table, "ndim"):
        return _eval_curve_1d(table, pla_arr)
    if hasattr(table, "data"):
        return _eval_curve_1d(table, pla_arr)
    if isinstance(table, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(table, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return _eval_curve_1d(table, pla_arr)
        except Exception:
            pass
        if len(table) == 2 and isinstance(table[0], (np.ndarray, list)):
            return _eval_curve_1d(table, pla_arr)

    if isinstance(table, dict):
        x1 = np.asarray(table.get("x1", table.get("x", [0.0, 1.0])), dtype=float)
        x2 = np.asarray(table.get("x2", [0.0]), dtype=float)
        x3 = np.asarray(table.get("x3", [p.t0]), dtype=float)
        y = np.asarray(table.get("y", table.get("values", [1.0])), dtype=float)
    elif hasattr(table, "x1") and hasattr(table, "y"):
        x1 = np.asarray(table.x1, dtype=float)
        x2 = np.asarray(getattr(table, "x2", [0.0]), dtype=float)
        x3 = np.asarray(getattr(table, "x3", [p.t0]), dtype=float)
        y = np.asarray(table.y, dtype=float)
    elif hasattr(table, "eval"):
        try:
            val = np.asarray(table.eval(pla_arr, rate_arr, temp_arr), dtype=float)
            slp = getattr(table, "dydx", np.zeros_like(val))
            return val, np.asarray(slp, dtype=float)
        except Exception:
            return _eval_curve_1d(table, pla_arr)
    else:
        sigy0 = float(table) if isinstance(table, (int, float)) else 1.0
        return np.full(n, sigy0, dtype=float), np.zeros(n, dtype=float)

    nx1 = len(x1)
    nx2 = len(x2)
    nx3 = len(x3)

    if nx1 == 0:
        return np.ones(n, dtype=float), np.zeros(n, dtype=float)
    if nx1 == 1:
        return np.full(n, y[0] if y.size > 0 else 1.0), np.zeros(n, dtype=float)

    if y.ndim == 3:
        if y.shape == (nx1, nx2, nx3):
            y_grid = np.transpose(y, (2, 1, 0))  # (nx3, nx2, nx1)
        elif y.shape == (nx3, nx2, nx1):
            y_grid = y
        else:
            y_grid = y.reshape((nx3, nx2, nx1))
    elif y.ndim == 1:
        if y.size == nx1 * nx2 * nx3:
            y_grid = y.reshape((nx3, nx2, nx1))
        elif y.size == nx1:
            return _eval_curve_1d((x1, y), pla_arr)
        else:
            return np.full(n, float(y[0])), np.zeros(n, dtype=float)
    else:
        y_grid = y.reshape((nx3, nx2, nx1))

    i1 = np.clip(np.searchsorted(x1, pla_arr, side="right") - 1, 0, nx1 - 2)
    dx1 = np.maximum(x1[i1 + 1] - x1[i1], _EM20)
    w1 = (pla_arr - x1[i1]) / dx1
    unw1 = 1.0 - w1

    if nx2 > 1:
        i2 = np.clip(np.searchsorted(x2, rate_arr, side="right") - 1, 0, nx2 - 2)
        dx2 = np.maximum(x2[i2 + 1] - x2[i2], _EM20)
        w2 = np.clip((rate_arr - x2[i2]) / dx2, 0.0, 1.0)
        unw2 = 1.0 - w2
    else:
        i2 = np.zeros(n, dtype=int)
        w2 = np.zeros(n, dtype=float)
        unw2 = np.ones(n, dtype=float)

    if nx3 > 1:
        i3 = np.clip(np.searchsorted(x3, temp_arr, side="right") - 1, 0, nx3 - 2)
        dx3 = np.maximum(x3[i3 + 1] - x3[i3], _EM20)
        w3 = np.clip((temp_arr - x3[i3]) / dx3, 0.0, 1.0)
        unw3 = 1.0 - w3
    else:
        i3 = np.zeros(n, dtype=int)
        w3 = np.zeros(n, dtype=float)
        unw3 = np.ones(n, dtype=float)

    i2_p1 = np.minimum(i2 + 1, nx2 - 1)
    i3_p1 = np.minimum(i3 + 1, nx3 - 1)

    y000 = y_grid[i3, i2, i1]
    y001 = y_grid[i3, i2, i1 + 1]
    y010 = y_grid[i3, i2_p1, i1]
    y011 = y_grid[i3, i2_p1, i1 + 1]
    y100 = y_grid[i3_p1, i2, i1]
    y101 = y_grid[i3_p1, i2, i1 + 1]
    y110 = y_grid[i3_p1, i2_p1, i1]
    y111 = y_grid[i3_p1, i2_p1, i1 + 1]

    y00 = unw1 * y000 + w1 * y001
    y01 = unw1 * y010 + w1 * y011
    y10 = unw1 * y100 + w1 * y101
    y11 = unw1 * y110 + w1 * y111

    y0 = unw2 * y00 + w2 * y01
    y1 = unw2 * y10 + w2 * y11

    yy = unw3 * y0 + w3 * y1

    dy00 = (y001 - y000) / dx1
    dy01 = (y011 - y010) / dx1
    dy10 = (y101 - y100) / dx1
    dy11 = (y111 - y110) / dx1

    dy0 = unw2 * dy00 + w2 * dy01
    dy1 = unw2 * dy10 + w2 * dy11
    dydx1 = unw3 * dy0 + w3 * dy1

    return yy, dydx1


# ============================================================================
# Constructor: build_law73
# ============================================================================

def _extract_param(p: Dict[str, Any], keys: Sequence[str], default: Any) -> Any:
    for k in keys:
        if k in p and p[k] is not None:
            return p[k]
    if "params" in p and isinstance(p["params"], dict):
        for k in keys:
            if k in p["params"] and p["params"][k] is not None:
                return p["params"][k]
    return default


def build_law73(rec: Any = None, **kwargs: Any) -> Material:
    """Physics constructor for /MAT/LAW73 (/MAT/BARLAT2000, /MAT/HILL_THERM).

    Extracts parameters per starter/source/materials/mat/mat073/hm_read_mat73.F.
    """
    if isinstance(rec, Law73Params):
        p_obj = rec
        mid = rec.id
        rho0 = rec.rho0
        title = rec.title
    elif rec is None:
        p_dict = dict(kwargs)
        if "params" in p_dict and isinstance(p_dict["params"], dict):
            p_dict = {**p_dict["params"], **p_dict}
        mid = int(p_dict.get("id", 1))
        rho0 = float(p_dict.get("rho0", p_dict.get("MAT_RHO", p_dict.get("density", 1.0))))
        title = str(p_dict.get("title", "LAW73_HILL_THERM"))
        p_obj = None
    elif isinstance(rec, dict):
        base = rec.get("params", rec)
        p_dict = {**base, **kwargs}
        mid = int(rec.get("id", p_dict.get("id", 1)))
        rho0 = float(p_dict.get("rho0", p_dict.get("MAT_RHO", p_dict.get("density", 1.0))))
        title = str(rec.get("title", p_dict.get("title", "LAW73_HILL_THERM")))
        p_obj = None
    elif hasattr(rec, "params") and isinstance(rec.params, dict):
        p_dict = {**rec.params, **kwargs}
        mid = int(getattr(rec, "id", p_dict.get("id", 1)))
        rho0 = float(getattr(rec, "rho0", p_dict.get("rho0", 1.0)))
        title = str(getattr(rec, "title", p_dict.get("title", "LAW73_HILL_THERM")))
        p_obj = None
    elif hasattr(rec, "__dict__"):
        p_dict = {**rec.__dict__, **kwargs}
        mid = int(getattr(rec, "id", p_dict.get("id", 1)))
        rho0 = float(getattr(rec, "rho0", p_dict.get("rho0", 1.0)))
        title = str(getattr(rec, "title", p_dict.get("title", "LAW73_HILL_THERM")))
        p_obj = None
    else:
        p_dict = dict(kwargs)
        mid = int(p_dict.get("id", 1))
        rho0 = float(p_dict.get("rho0", 1.0))
        title = str(p_dict.get("title", "LAW73_HILL_THERM"))
        p_obj = None

    if p_obj is None:
        def _get_f(keys: Sequence[str], default: float) -> float:
            val = _extract_param(p_dict, keys, default)
            try:
                return float(val)
            except (ValueError, TypeError):
                return float(default)

        def _get_i(keys: Sequence[str], default: int) -> int:
            val = _extract_param(p_dict, keys, default)
            try:
                return int(val)
            except (ValueError, TypeError):
                return int(default)

        rho0_val = _get_f(["MAT_RHO", "Refer_Rho", "rho0", "density", "rho"], rho0)
        rhor_val = _get_f(["Refer_Rho", "rhor", "MAT_REFRHO"], rho0_val)
        e_val = _get_f(["MAT_E", "E", "e", "young", "Young"], 210000.0)
        nu_val = _get_f(["MAT_NU", "NU", "nu", "anu"], 0.3)

        ifunce_val = _get_i(["Yr_fun", "IFUNCE", "ifunce", "fct_e"], 0)
        einf_val = _get_f(["MAT_EFIB", "EINF", "Einf", "einf"], 0.0)
        ce_val = _get_f(["MAT_C", "CE", "ce", "c_e"], 0.0)

        r00_val = _get_f(["MAT_R00", "R00", "r00", "r_00"], 1.0)
        r45_val = _get_f(["MAT_R45", "R45", "r45", "r_45"], 1.0)
        r90_val = _get_f(["MAT_R90", "R90", "r90", "r_90"], 1.0)

        chard_val = _get_f(["MAT_CHARD", "CHARD", "chard", "C_hard", "FISOKIN", "fisokin"], 0.0)
        iyield_val = _get_i(["MAT_Iyield", "Iyield", "iyield", "ir0", "IR0"], 0)

        eps_max_val = _get_f(["MAT_EPS", "EPS", "eps", "eps_max", "EPSMAX"], _INF)
        epsr1_val = _get_f(["MAT_EPST1", "EPST1", "epsr1", "EPSR1"], _INF)
        epsr2_val = _get_f(["MAT_EPST2", "EPST2", "epsr2", "EPSR2"], 2.0 * _INF)

        table_id_val = _extract_param(p_dict, ["FUN_A1", "ITABLE", "table_id", "TABLE", "table", "curve"], 0)
        fscale_val = _get_f(["MAT_FScale", "FScale", "fscale", "YFAC", "yfac"], 1.0)
        pscale_val = _get_f(["MAT_PScale", "PScale", "pscale", "X2FAC", "xfac"], 1.0)

        t0_val = _get_f(["T_Initial", "T0", "t0", "temp_initial", "temp0"], 293.0)
        rhocp_val = _get_f(["MAT_SPHEAT", "RHOCP", "rhocp", "spheat"], 0.0)

        curve_e_val = _extract_param(p_dict, ["curve_e", "E_curve", "fct_e_obj"], None)
        yield_table_val = _extract_param(p_dict, ["yield_table", "table_obj"], None)

        p_obj = Law73Params(
            rho0=rho0_val,
            refer_rho=rhor_val,
            e=e_val,
            nu=nu_val,
            ifunce=ifunce_val,
            einf=einf_val,
            ce=ce_val,
            r00=r00_val,
            r45=r45_val,
            r90=r90_val,
            chard=chard_val,
            iyield=iyield_val,
            eps_max=eps_max_val,
            epsr1=epsr1_val,
            epsr2=epsr2_val,
            table_id=table_id_val,
            fscale=fscale_val,
            pscale=pscale_val,
            t0=t0_val,
            rhocp=rhocp_val,
            id=mid,
            title=title,
            curve_e=curve_e_val,
            yield_table=yield_table_val,
        )

    params_dict = {
        "rho0": p_obj.rho0,
        "refer_rho": p_obj.refer_rho,
        "e": p_obj.e,
        "E": p_obj.e,
        "nu": p_obj.nu,
        "ifunce": p_obj.ifunce,
        "einf": p_obj.einf,
        "ce": p_obj.ce,
        "r00": p_obj.r00,
        "r45": p_obj.r45,
        "r90": p_obj.r90,
        "chard": p_obj.chard,
        "fisokin": p_obj.chard,
        "iyield": p_obj.iyield,
        "eps_max": p_obj.eps_max,
        "epsr1": p_obj.epsr1,
        "epsr2": p_obj.epsr2,
        "table_id": p_obj.table_id,
        "fscale": p_obj.fscale,
        "pscale": p_obj.pscale,
        "t0": p_obj.t0,
        "rhocp": p_obj.rhocp,
        "G": p_obj.g,
        "A11": p_obj.a11,
        "A21": p_obj.a21,
        "C1": p_obj.c1,
        "A01": p_obj.a01,
        "A02": p_obj.a02,
        "A03": p_obj.a03,
        "A12": p_obj.a12,
        "_obj": p_obj,
    }

    mat = Material(
        id=p_obj.id,
        law=73,
        rho0=p_obj.rho0,
        title=p_obj.title,
        law_name="LAW73",
        params=params_dict,
    )
    return mat


def _get_params(mat: Any) -> Law73Params:
    if isinstance(mat, Law73Params):
        return mat
    if hasattr(mat, "params") and isinstance(mat.params, dict) and "_obj" in mat.params:
        return mat.params["_obj"]
    if isinstance(mat, dict) and "_obj" in mat:
        return mat["_obj"]
    m = build_law73(mat)
    return m.params["_obj"]


# ============================================================================
# Sound Speed & Allocations
# ============================================================================

def sound_speed(mat: Any, rho: Optional[float] = None, extra: Optional[Dict[str, Any]] = None) -> float:
    """Acoustic sound speed in shell plane stress: c = sqrt(A11 / rho).

    Upstream reference: sigeps73c.F line 231 (SOUNDSP = SQRT(A11/RHO0)).
    """
    p = _get_params(mat)
    rho_val = float(rho) if rho is not None and float(rho) > 0.0 else p.rho0
    return math.sqrt(p.a11 / max(rho_val, _EM20))


def extra_shapes(mat: Any, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Extra allocations required in element layer state for LAW73.

    NUVAR = 7 channels:
      - uvar[:, 1]: alpha_xx (back-stress)
      - uvar[:, 2]: alpha_yy (back-stress)
      - uvar[:, 3]: alpha_xy (back-stress)
      - uvar[:, 4:7]: table interpolation position cache
    """
    if nip:
        return {
            "uvar73": (nip, 7),
            "pla73": (nip,),
            "off73": (nip,),
            "thk73": (nip,),
            "temp": (nip,),
        }
    return {
        "uvar73": (7,),
        "pla73": (),
        "off73": (),
        "thk73": (),
        "temp": (),
    }


# ============================================================================
# Shell Constitutive Update: shell_update (sigeps73c.F)
# ============================================================================

def shell_update(mat: Any, sig: np.ndarray, deps: np.ndarray,
                 epsp: Optional[Union[float, np.ndarray]] = None,
                 dt: float = 0.0,
                 extra: Optional[Dict[str, Any]] = None,
                 return_tuple: bool = False,
                 **kwargs: Any) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Plane-stress constitutive update for /MAT/LAW73 (/MAT/HILL_THERM).

    Follows engine/source/materials/mat/mat073/sigeps73c.F line by line.

    Parameters
    ----------
    mat : Material, Law73Params, or dict
    sig : (3,) or (n, 3) or (n, 5) ndarray
        Old Cauchy stress tensor [xx, yy, xy, (yz, zx)].
    deps : (3,) or (n, 3) or (n, 5) ndarray
        Engineering strain increment [xx, yy, xy, (yz, zx)].
    epsp : (n,) or float, optional
        Accumulated equivalent plastic strain history.
    dt : float
        Time step increment.
    extra : dict, optional
        Layer state dict containing 'uvar73', 'pla73', 'off73', 'thk', 'temp', 'eint', etc.
    return_tuple : bool, default False
        If True, returns (sig, epsp, soundsp). Otherwise returns (sig, epsp).
    """
    p = _get_params(mat)

    sig_arr = np.asarray(sig, dtype=float)
    deps_arr = np.asarray(deps, dtype=float)
    is_1d = (sig_arr.ndim == 1)

    if is_1d:
        sig_arr = sig_arr[None, :]
        deps_arr = deps_arr[None, :]

    n = sig_arr.shape[0]
    ncomp = max(sig_arr.shape[1], deps_arr.shape[1], 3)

    if deps_arr.shape[0] == 1 and n > 1:
        deps_arr = np.repeat(deps_arr, n, axis=0)

    if extra is None:
        extra = {}

    # Plastic strain (pla)
    if "pla73" in extra and extra["pla73"] is not None:
        pla = np.asarray(extra["pla73"], dtype=float).copy().flatten()
        if len(pla) == 1 and n > 1:
            pla = np.full(n, pla[0], dtype=float)
    elif epsp is not None:
        pla = np.asarray(epsp, dtype=float).copy().flatten()
        if len(pla) == 1 and n > 1:
            pla = np.full(n, pla[0], dtype=float)
    else:
        pla = np.zeros(n, dtype=float)

    # History variables (uvar: 7 channels)
    if "uvar73" in extra and extra["uvar73"] is not None:
        uvar = np.asarray(extra["uvar73"], dtype=float).copy()
        if uvar.ndim == 1:
            uvar = uvar[None, :] if n == 1 else np.tile(uvar, (n, 1))
    elif "uvar" in extra and extra["uvar"] is not None:
        uvar = np.asarray(extra["uvar"], dtype=float).copy()
        if uvar.ndim == 1:
            uvar = uvar[None, :] if n == 1 else np.tile(uvar, (n, 1))
    else:
        uvar = np.zeros((n, 7), dtype=float)

    if uvar.shape[1] < 7:
        uvar_pad = np.zeros((n, 7), dtype=float)
        uvar_pad[:, :uvar.shape[1]] = uvar
        uvar = uvar_pad

    # Element deletion / active flag (off)
    if "off73" in extra and extra["off73"] is not None:
        off = np.asarray(extra["off73"], dtype=float).copy().flatten()
    elif "off" in extra and extra["off"] is not None:
        off = np.asarray(extra["off"], dtype=float).copy().flatten()
    elif "layfail" in extra and extra["layfail"] is not None:
        off = np.asarray(extra["layfail"], dtype=float).copy().flatten()
    else:
        off = np.ones(n, dtype=float)
    if len(off) == 1 and n > 1:
        off = np.full(n, off[0], dtype=float)

    thk = None
    if "thk73" in extra and extra["thk73"] is not None:
        thk_arr = np.asarray(extra["thk73"], dtype=float).flatten().copy()
        if not np.all(thk_arr == 0.0):
            thk = thk_arr
    if thk is None:
        thk = np.asarray(extra.get("thk", extra.get("thkn", 1.0)), dtype=float).flatten().copy()
    if len(thk) == 1 and n > 1:
        thk = np.full(n, thk[0], dtype=float)

    thkly = np.asarray(extra.get("thkly", extra.get("thklyl", extra.get("thk0", 1.0))), dtype=float).flatten()
    if len(thkly) == 1 and n > 1:
        thkly = np.full(n, thkly[0], dtype=float)

    # 2. Dynamic Young's Modulus Degradation (sigeps73c.F:145-193)
    e1 = np.full(n, p.e, dtype=float)
    if p.opte == 1 and (p.curve_e is not None or p.ifunce > 0):
        pos = pla > 0.0
        if np.any(pos):
            c_e = p.curve_e if p.curve_e is not None else p.ifunce
            scale, _ = _eval_curve_1d(c_e, pla[pos])
            e1[pos] = scale * p.e
    elif p.ce > 0.0:
        pos = pla > 0.0
        if np.any(pos):
            e1[pos] = p.e - (p.e - p.einf) * (1.0 - np.exp(-p.ce * pla[pos]))

    a11 = e1 / max(1.0 - p.nu ** 2, 1.0e-15)
    a21 = p.nu * a11
    g1 = 0.5 * e1 / max(1.0 + p.nu, 1.0e-15)
    g31 = 3.0 * g1
    shf = float(extra.get("shf", 5.0 / 6.0))
    gs = g1 * shf

    # 3. Temperature calculation (sigeps73c.F:209-222)
    rhocp = p.rhocp
    rhocp_inv = 1.0 / rhocp if rhocp > 0.0 else 0.0
    if "tempel" in extra and extra["tempel"] is not None:
        temp = np.asarray(extra["tempel"], dtype=float).flatten().copy()
        if np.all(temp == 0.0):
            temp = np.full(n, p.t0, dtype=float)
        elif len(temp) == 1 and n > 1:
            temp = np.full(n, temp[0], dtype=float)
    elif rhocp > 0.0 and "eint" in extra and extra["eint"] is not None:
        eint_raw = np.asarray(extra["eint"], dtype=float)
        if eint_raw.size == n:
            eint_tot = eint_raw.flatten()
        elif eint_raw.size >= 2:
            eint_tot = np.full(n, float(eint_raw.flatten()[0] + eint_raw.flatten()[1]))
        else:
            eint_tot = np.full(n, float(np.sum(eint_raw)))
        vol = float(np.asarray(extra.get("vol", 1.0)).ravel()[0])
        vol0 = vol * p.rho0
        temp = p.t0 + eint_tot * rhocp_inv / max(vol0, _EM20)
    elif "temp" in extra and extra["temp"] is not None:
        temp = np.asarray(extra["temp"], dtype=float).flatten().copy()
        if np.all(temp == 0.0):
            temp = np.full(n, p.t0, dtype=float)
        elif len(temp) == 1 and n > 1:
            temp = np.full(n, temp[0], dtype=float)
    else:
        temp = np.full(n, p.t0, dtype=float)

    # 4. Back-stress shifted trial stress (sigeps73c.F:225-229)
    alpha_xx = uvar[:, 1]
    alpha_yy = uvar[:, 2]
    alpha_xy = uvar[:, 3]

    sign_xx = (sig_arr[:, 0] - alpha_xx) + a11 * deps_arr[:, 0] + a21 * deps_arr[:, 1]
    sign_yy = (sig_arr[:, 1] - alpha_yy) + a21 * deps_arr[:, 0] + a11 * deps_arr[:, 1]
    sign_xy = (sig_arr[:, 2] - alpha_xy) + g1 * deps_arr[:, 2]

    if ncomp >= 4:
        sign_yz = sig_arr[:, 3] + gs * deps_arr[:, 3]
    else:
        sign_yz = None
    if ncomp >= 5:
        sign_zx = sig_arr[:, 4] + gs * deps_arr[:, 4]
    else:
        sign_zx = None

    # 5. Sound speed (sigeps73c.F:231)
    rho_cur = np.asarray(extra.get("rho", p.rho0), dtype=float).flatten()
    if len(rho_cur) == 1 and n > 1:
        rho_cur = np.full(n, rho_cur[0], dtype=float)
    soundsp = np.sqrt(a11 / np.maximum(rho_cur, _EM20))

    # 6. Equivalent plastic strain rate & tensile strain (sigeps73c.F:237-247)
    if "epsd_pg" in extra and extra["epsd_pg"] is not None:
        epsd = np.asarray(extra["epsd_pg"], dtype=float).flatten()
        if len(epsd) == 1 and n > 1:
            epsd = np.full(n, epsd[0], dtype=float)
    elif "epsd" in extra and extra["epsd"] is not None:
        epsd = np.asarray(extra["epsd"], dtype=float).flatten()
        if len(epsd) == 1 and n > 1:
            epsd = np.full(n, epsd[0], dtype=float)
    elif "epsp_tensor" in extra and extra["epsp_tensor"] is not None:
        ep_arr = np.asarray(extra["epsp_tensor"], dtype=float)
        if ep_arr.ndim == 1:
            ep_arr = ep_arr[None, :]
        edxx = ep_arr[:, 0]
        edyy = ep_arr[:, 1]
        edxy = ep_arr[:, 2]
        epsd = 0.5 * (np.abs(edxx + edyy) + np.sqrt((edxx - edyy) ** 2 + edxy ** 2))
    elif dt > 0.0:
        edxx = deps_arr[:, 0] / dt
        edyy = deps_arr[:, 1] / dt
        edxy = deps_arr[:, 2] / dt
        epsd = 0.5 * (np.abs(edxx + edyy) + np.sqrt((edxx - edyy) ** 2 + edxy ** 2))
    else:
        epsd = np.zeros(n, dtype=float)

    if "eps" in extra and extra["eps"] is not None:
        eps_tot = np.asarray(extra["eps"], dtype=float)
        if eps_tot.ndim == 1:
            eps_tot = eps_tot[None, :]
        exx = eps_tot[:, 0]
        eyy = eps_tot[:, 1]
        exy = eps_tot[:, 2]
        epst = 0.5 * (exx + eyy + np.sqrt((exx - eyy) ** 2 + exy ** 2))
    else:
        exx = deps_arr[:, 0]
        eyy = deps_arr[:, 1]
        exy = deps_arr[:, 2]
        epst = 0.5 * (exx + eyy + np.sqrt((exx - eyy) ** 2 + exy ** 2))

    if p.epsr2 > p.epsr1:
        fail = np.clip((p.epsr2 - epst) / (p.epsr2 - p.epsr1), 0.0, 1.0)
    else:
        fail = np.ones(n, dtype=float)

    # 7. Yield stress evaluation from table/function (sigeps73c.F:250-291)
    xfac = p.xfac
    yfac = p.yfac
    edot_scaled = epsd * xfac

    yld_val, dydx = _eval_yield_table(p, pla, edot_scaled, temp)
    yld = yfac * yld_val * fail
    yld = np.maximum(yld, _EM20)
    h_slope = np.maximum(fail * dydx, 0.0)

    if p.chard != 0.0:  # FISOKIN
        yk, _ = _eval_yield_table(p, np.zeros(n, dtype=float), edot_scaled, temp)
        yld = (1.0 - p.chard) * yld + p.chard * fail * yfac * yk
        yld = np.maximum(yld, _EM20)

    # 8. Hill equivalent stress (sigeps73c.F:295-307)
    s1_hill = p.a01 * sign_xx ** 2
    s2_hill = p.a02 * sign_yy ** 2
    s3_hill = p.a03 * sign_xx * sign_yy
    axy_hill = p.a12 * sign_xy ** 2
    svm = np.sqrt(np.maximum(0.0, s1_hill + s2_hill - s3_hill + axy_hill))

    # Thickness strain (elastic part)
    nnu1 = p.nu / max(1.0 - p.nu, _EM20)
    dezz_el = -(deps_arr[:, 0] + deps_arr[:, 1]) * nnu1
    thk += dezz_el * thkly * off

    # 9. Plastic return mapping (sigeps73c.F:311-432)
    yielding = (svm > yld) & (off == 1.0)
    dpla_arr = np.zeros(n, dtype=float)

    if np.any(yielding):
        y_idx = np.where(yielding)[0]
        for i in y_idx:
            svm_i = svm[i]
            yld_i0 = yld[i]
            h_i = h_slope[i]
            a11_i = a11[i]
            g31_i = g31[i]

            dpla_j = (svm_i - yld_i0) / max(g31_i + h_i, _EM20)

            # Hill coupling coefficients
            hk_i = h_i * p.chard
            fhk = (4.0 / 3.0) * hk_i / a11_i
            fa01 = p.a01 * fhk
            fa02 = p.a02 * fhk
            fa03 = p.a03 * fhk
            nu1 = p.nu + 0.5 * fhk

            nu2 = 1.0 - nu1 * nu1 + fhk * fhk
            nu3 = nu1 * 0.5
            nu4 = 0.5 * (1.0 - p.nu)
            nu5 = 1.0 - nnu1

            s1 = 2.0 * p.a01 * nu1 - p.a03 - fa03
            s2 = 2.0 * p.a02 * nu1 - p.a03 - fa03
            s12 = p.a03 - nu1 * (p.a01 + p.a02) + fa03
            s3 = math.sqrt(max(0.0, nu2 * (p.a01 - p.a02) ** 2 + s12 ** 2))

            q12 = -(p.a01 - p.a02 + s3 + fa01 - fa02) / s1 if abs(s1) >= _EM20 else 0.0
            q21 = (p.a01 - p.a02 + s3 + fa01 - fa02) / s2 if abs(s2) >= _EM20 else 0.0

            jq = 1.0 / max(1.0 - q12 * q21, _EM20)
            jq2 = jq * jq

            a = p.a01 * q12
            b = p.a02 * q21
            a_1 = (p.a01 + p.a03 * q21 + b * q21) * jq2
            a_2 = (p.a02 + p.a03 * q12 + a * q12) * jq2
            a_3 = (a + b) * jq2 * 2.0 + p.a03 * (jq2 * 2.0 - jq)

            s11_i = sign_xx[i] + sign_yy[i] * q12
            s22_i = q21 * sign_xx[i] + sign_yy[i]
            axx_i = a_1 * s11_i * s11_i
            ayy_i = a_2 * s22_i * s22_i
            a_xy_i = a_3 * s11_i * s22_i

            a_mid = p.a03 * nu3
            b_mid = s3 * jq
            b_1 = p.a02 - a_mid - b_mid + fa02
            b_2 = p.a01 - a_mid + b_mid + fa01
            b_3 = p.a12 * (nu4 + 0.5 * fhk)

            h_eff = max(0.0, h_i - hk_i)

            # 2 Newton iterations
            for _ in range(2):
                if dpla_j > 0.0:
                    y_cur = yld_i0 + h_eff * dpla_j
                    dr = a11_i * dpla_j / max(y_cur, _EM20)

                    p_1 = 1.0 / (1.0 + b_1 * dr)
                    pp1 = p_1 * p_1
                    p_2 = 1.0 / (1.0 + b_2 * dr)
                    pp2 = p_2 * p_2
                    p_3 = 1.0 / (1.0 + b_3 * dr)
                    pp3 = p_3 * p_3

                    f = axx_i * pp1 + ayy_i * pp2 - a_xy_i * p_1 * p_2 + axy_hill[i] * pp3 - y_cur * y_cur
                    df = -((axx_i * p_1 - a_xy_i * p_2 * 0.5) * pp1 * b_1 +
                           (ayy_i * p_2 - a_xy_i * p_1 * 0.5) * pp2 * b_2 +
                           axy_hill[i] * pp3 * p_3 * b_3) * (a11_i - dr * h_eff) / max(y_cur, _EM20) - h_eff * y_cur

                    if df != 0.0:
                        dpla_j = max(0.0, dpla_j - f * 0.5 / df)
                else:
                    dpla_j = 0.0

            dpla_arr[i] = dpla_j
            pla[i] += dpla_j

            # Admissible stresses
            y_final = yld_i0 + h_eff * dpla_j
            dr0 = dpla_j / max(y_final, _EM20)
            dr = a11_i * dr0

            p_1 = 1.0 / (1.0 + b_1 * dr)
            p_2 = 1.0 / (1.0 + b_2 * dr)
            p_3 = 1.0 / (1.0 + b_3 * dr)

            s1_adm = s11_i * p_1
            s2_adm = s22_i * p_2
            sign_xx[i] = jq * (s1_adm - s2_adm * q12)
            sign_yy[i] = jq * (s2_adm - s1_adm * q21)
            sign_xy[i] = sign_xy[i] * p_3

            # Through-thickness plastic strain
            s1_thk = p.a01 * sign_xx[i] + p.a02 * sign_yy[i] - p.a03 * (sign_xx[i] + sign_yy[i]) * 0.5
            dezz_pl = - nu5 * dpla_j * s1_thk / max(y_final, _EM20)
            thk[i] += dezz_pl * thkly[i] * off[i]

            # Back-stress update
            s1_bs = p.a03 * 0.5
            p1_bs = p.a01 * sign_xx[i] - s1_bs * sign_yy[i]
            p2_bs = p.a02 * sign_yy[i] - s1_bs * sign_xx[i]
            p3_bs = p.a12 * sign_xy[i]
            dr0_bs = (2.0 / 3.0) * dr0 * hk_i

            uvar[i, 1] += (2.0 * p1_bs + p2_bs) * dr0_bs
            uvar[i, 2] += (2.0 * p2_bs + p1_bs) * dr0_bs
            uvar[i, 3] += 0.5 * p3_bs * dr0_bs

        # Plastic dissipation & adiabatic heating increment (sigeps73c.F:220)
        if rhocp > 0.0 and np.any(dpla_arr > 0.0):
            vol = float(np.asarray(extra.get("vol", 1.0)).ravel()[0])
            vol0 = max(vol * p.rho0, _EM20)
            dwp = np.sum(dpla_arr * svm) * vol
            dtemp = dpla_arr * svm * rhocp_inv / max(p.rho0, _EM20)
            temp += dtemp

    # Element deletion check (sigeps73c.F:434)
    deleted = (pla > p.eps_max) & (off == 1.0)
    if np.any(deleted):
        off[deleted] = 0.8 * off[deleted]

    # 10. Reconstruct final Cauchy stress (sigeps73c.F:438-440)
    sig_out = np.zeros((n, ncomp), dtype=float)
    sig_out[:, 0] = sign_xx + uvar[:, 1]
    sig_out[:, 1] = sign_yy + uvar[:, 2]
    sig_out[:, 2] = sign_xy + uvar[:, 3]

    if sign_yz is not None:
        sig_out[:, 3] = sign_yz
    if sign_zx is not None:
        sig_out[:, 4] = sign_zx

    # Zero out stress if fully failed
    fully_failed = off <= 0.0
    if np.any(fully_failed):
        sig_out[fully_failed] = 0.0

    # Final equivalent stress for output
    s1_out = p.a01 * sig_out[:, 0] ** 2
    s2_out = p.a02 * sig_out[:, 1] ** 2
    s3_out = p.a03 * sig_out[:, 0] * sig_out[:, 1]
    axy_out = p.a12 * sig_out[:, 2] ** 2
    seq_final = np.sqrt(np.maximum(0.0, s1_out + s2_out - s3_out + axy_out))

    # Write state back to extra dict
    if "uvar73" in extra and isinstance(extra["uvar73"], np.ndarray):
        extra["uvar73"][:] = uvar
    elif "uvar" in extra and isinstance(extra["uvar"], np.ndarray):
        extra["uvar"][:] = uvar
    else:
        extra["uvar73"] = uvar

    if "pla73" in extra and isinstance(extra["pla73"], np.ndarray):
        extra["pla73"][:] = pla
    if "off73" in extra and isinstance(extra["off73"], np.ndarray):
        extra["off73"][:] = off
    if "off" in extra and isinstance(extra["off"], np.ndarray):
        extra["off"][:] = off
    if "layfail" in extra and isinstance(extra["layfail"], np.ndarray):
        extra["layfail"][:] = off
    if "thk73" in extra and isinstance(extra["thk73"], np.ndarray):
        extra["thk73"][:] = thk
    if "thk" in extra and isinstance(extra["thk"], np.ndarray):
        extra["thk"][:] = thk
    if "temp" in extra and isinstance(extra["temp"], np.ndarray):
        extra["temp"][:] = temp
    else:
        extra["temp"] = temp
    if "tempel" in extra and isinstance(extra["tempel"], np.ndarray):
        extra["tempel"][:] = temp
    if "eint" in extra and isinstance(extra["eint"], np.ndarray) and rhocp > 0.0:
        if np.any(dpla_arr > 0.0):
            extra["eint"][:] += dwp
    if "seq" in extra:
        extra["seq"] = seq_final[0] if is_1d else seq_final
    if "seq73" in extra:
        extra["seq73"] = seq_final[0] if is_1d else seq_final

    if epsp is not None and isinstance(epsp, np.ndarray):
        try:
            epsp[:] = pla
        except Exception:
            pass

    out_s = sig_out[0] if is_1d else sig_out
    out_p = pla[0] if is_1d else pla
    out_c = soundsp[0] if is_1d else soundsp

    if return_tuple:
        return out_s, out_p, out_c
    return out_s, out_p


shell_update_law73 = shell_update


# ============================================================================
# Solid Constitutive Update: solid_update
# ============================================================================

def solid_update(*args: Any, **kwargs: Any) -> Any:
    """Constitutive update for solids is not supported in LAW73."""
    raise NotImplementedError("LAW73 is implemented for shell elements only.")


solid_update_law73 = solid_update


# ============================================================================
# Tangent Operators: shell_membrane_tangent & consistent_shell_tangent
# ============================================================================

def shell_membrane_tangent(mat: Any, **kwargs: Any) -> np.ndarray:
    """Constant elastic plane-stress (3, 3) membrane matrix."""
    p = _get_params(mat)
    return np.array([
        [p.a11, p.a21, 0.0],
        [p.a21, p.a11, 0.0],
        [0.0, 0.0, p.g],
    ], dtype=float)


def consistent_shell_tangent(mat: Any, sig: np.ndarray,
                             epsp: Optional[Union[float, np.ndarray]] = None,
                             epsp_incr: Optional[Union[float, np.ndarray]] = None,
                             extra: Optional[Dict[str, Any]] = None,
                             deps: Optional[np.ndarray] = None,
                             dt: float = 0.0,
                             symmetric: bool = False,
                             h: float = 1.0e-7,
                             **kwargs: Any) -> np.ndarray:
    """Consistent (algorithmic) plane-stress tangent operator (n, 3, 3) or (3, 3)."""
    p = _get_params(mat)
    sig_arr = np.asarray(sig, dtype=float)
    is_1d = (sig_arr.ndim == 1)

    if is_1d:
        sig_arr = sig_arr[None, :]

    n = sig_arr.shape[0]
    c_el = shell_membrane_tangent(mat)
    d_tangent = np.tile(c_el, (n, 1, 1))

    if deps is not None:
        deps_arr = np.asarray(deps, dtype=float)
        if deps_arr.ndim == 1:
            deps_arr = deps_arr[None, :]
        h_val = float(h)

        for i in range(n):
            sig_i = sig_arr[i].copy()
            deps_i = deps_arr[i].copy()
            ext_i = dict(extra) if extra else {}

            d_num = np.zeros((3, 3), dtype=float)
            for j in range(3):
                dp = deps_i.copy()
                dm = deps_i.copy()
                dp[j] += h_val
                dm[j] -= h_val

                sp, _ = shell_update(mat, sig_i.copy(), dp, epsp=epsp, dt=dt, extra=dict(ext_i))
                sm, _ = shell_update(mat, sig_i.copy(), dm, epsp=epsp, dt=dt, extra=dict(ext_i))

                d_num[:, j] = (sp[:3] - sm[:3]) / (2.0 * h_val)

            if symmetric:
                d_num = 0.5 * (d_num + d_num.T)
            d_tangent[i] = d_num

        return d_tangent[0] if is_1d else d_tangent

    if epsp_incr is not None:
        incr = np.asarray(epsp_incr, dtype=float).flatten()
        if len(incr) == 1 and n > 1:
            incr = np.full(n, incr[0], dtype=float)
        plastic = incr > 0.0
        if np.any(plastic):
            for i in np.where(plastic)[0]:
                h_slope = max(p.chard, 0.0)
                denom = p.g * 3.0 + h_slope
                reduction = (p.g * 3.0) / max(denom, _EM20)
                d_tangent[i] *= max(0.01, min(1.0, 1.0 - 0.5 * (1.0 - reduction)))

    if symmetric:
        d_tangent = 0.5 * (d_tangent + np.swapaxes(d_tangent, -1, -2))

    return d_tangent[0] if is_1d else d_tangent


tangent_law73_shell = consistent_shell_tangent


def resolve(mat: Material, model: Any, log: Any = None) -> None:
    """Resolve /FUNCT and /TABLE references into curve_e and yield_table in mat.params."""
    p = _get_params(mat)
    if hasattr(model, "functions"):
        functions = model.functions
    elif isinstance(model, dict):
        functions = model.get("functions", model)
    else:
        functions = {}

    if hasattr(model, "tables"):
        tables = model.tables
    elif isinstance(model, dict):
        tables = model.get("tables", model)
    else:
        tables = {}

    if p.ifunce > 0 and p.curve_e is None:
        if p.ifunce in functions:
            p.curve_e = functions[p.ifunce]
            p.opte = 1
            if hasattr(mat, "params") and isinstance(mat.params, dict):
                mat.params["curve_e"] = p.curve_e

    if p.table_id > 0 and p.yield_table is None:
        if p.table_id in tables:
            p.yield_table = tables[p.table_id]
        elif p.table_id in functions:
            p.yield_table = functions[p.table_id]
        if hasattr(mat, "params") and isinstance(mat.params, dict):
            mat.params["yield_table"] = p.yield_table


# ============================================================================
# Registration Helper
# ============================================================================

def _register() -> None:
    """Register LAW73 in pyradioss MAT_PHYSICS_REGISTRY."""
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        for k in (73, "73", "LAW73", "HILL_THERM",
                  "MAT_LAW73", "MAT_HILL_THERM", "LAW73_HILL_THERM"):
            MAT_PHYSICS_REGISTRY[k] = build_law73
    except Exception:
        pass


_register()

