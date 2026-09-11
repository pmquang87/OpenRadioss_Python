"""
LAW74 — 3D Tabulated Hill Orthotropic Plasticity Model for Solids
(/MAT/LAW74, /MAT/HILL_3D, /MAT/ORTH_PLAS, /MAT/THERM_HILL).

Upstream Fortran reference:
  - Starter reader: starter/source/materials/mat/mat074/hm_read_mat74.F
  - Engine physics: engine/source/materials/mat/mat074/sigeps74.F
  - Table interpolation: engine/source/tools/curve/table_tools.F (TABLE_VINTERP)
  - Function interpolation: engine/source/tools/curve/finter.F (FINTER)

Theory & Algorithm
------------------
Hill 1948 3D quadratic orthotropic yield criterion:
    CRI = sqrt(FF*(s_yy - s_zz)^2 + GG*(s_zz - s_xx)^2 + HH*(s_xx - s_yy)^2
               + 2*LL*s_yz^2 + 2*MM*s_zx^2 + 2*NN*s_xy^2)

Hill Anisotropy Constants computed from directional yield stresses:
    FF = 0.5 * (1/s22y^2 + 1/s33y^2 - 1/s11y^2)
    GG = 0.5 * (1/s11y^2 + 1/s33y^2 - 1/s22y^2)
    HH = 0.5 * (1/s11y^2 + 1/s22y^2 - 1/s33y^2)
    LL = 0.5 / s23y^2
    MM = 0.5 / s31y^2
    NN = 0.5 / s12y^2

Mixed Isotropic-Kinematic Hardening:
    F_isokin = CHARD in [0, 1]
    YLD = (1 - F_isokin) * FAIL * Y(pla, epsd*xfac, temp) + F_isokin * FAIL * Y(0, epsd*xfac, temp)
    H_slope = FAIL * dY/dpla

Back-stress Evolution:
    ds = s_trial - s_dev
    hkin = (2/3) * F_isokin * H_slope
    alpha_fac = hkin / (2*G + hkin)
    delta_alpha = alpha_fac * ds
    alpha += delta_alpha

Dynamic Young's Modulus Degradation:
    if OPTE == 1 (ifunce > 0): E = E0 * f_E(pla)
    elif CE > 0: E = E0 - (E0 - Einf) * (1 - exp(-CE * pla))

Adiabatic Plastic Heating:
    if rhocp > 0: temp += yld * dpla / rhocp

Failure & Deletion:
    FAIL = clamp((EPSR2 - epst) / (EPSR2 - EPSR1), 0.0, 1.0)
    where epst is the maximum principal tensile strain from cubic eigenvalue solve.
    if pla > EPSMAX: off = 0.8 * off; if off < 0.1: off = 0.0
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
        if (len(curve) == 2 and isinstance(curve[0], (list, tuple, np.ndarray))
                and isinstance(curve[1], (list, tuple, np.ndarray))):
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


def _eval_yield_table(p: Law74Params, pla: np.ndarray, rate: np.ndarray,
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
        sigy0 = float(p.sigy0) if hasattr(p, "sigy0") and p.sigy0 is not None else float(p.get("sigy0", 1.0))
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
        sigy0 = float(table) if isinstance(table, (int, float)) else (float(p.sigy0) if hasattr(p, "sigy0") and p.sigy0 is not None else 1.0)
        return np.full(n, sigy0, dtype=float), np.zeros(n, dtype=float)

    nx1 = len(x1)
    nx2 = len(x2)
    nx3 = len(x3)

    if nx1 == 0:
        return np.ones(n, dtype=float), np.zeros(n, dtype=float)
    if nx1 == 1:
        return np.full(n, y.flat[0] if y.size > 0 else 1.0), np.zeros(n, dtype=float)

    if y.ndim == 3:
        if y.shape == (nx1, nx2, nx3):
            y_grid = np.transpose(y, (2, 1, 0))  # (nx3, nx2, nx1)
        elif y.shape == (nx3, nx2, nx1):
            y_grid = y
        else:
            y_grid = y.reshape((nx3, nx2, nx1))
    elif y.ndim == 2:
        if y.shape == (nx1, nx2):
            y_grid = y.T[None, :, :]  # (1, nx2, nx1)
        else:
            y_grid = y[None, :, :]
    elif y.ndim == 1:
        if len(y) == nx1:
            return _eval_curve_1d({"x": x1, "y": y}, pla_arr)
        y_grid = y.reshape((nx3, nx2, nx1))
    else:
        return _eval_curve_1d({"x": x1, "y": y.flatten()[:nx1]}, pla_arr)

    # Multi-dimensional table interpolation matching TABLE_VINTERP
    i1 = np.clip(np.searchsorted(x1, pla_arr, side="right") - 1, 0, nx1 - 2)
    dx1 = np.maximum(x1[i1 + 1] - x1[i1], _EM20)
    w1 = (pla_arr - x1[i1]) / dx1

    if nx2 > 1:
        i2 = np.clip(np.searchsorted(x2, rate_arr, side="right") - 1, 0, nx2 - 2)
        dx2 = np.maximum(x2[i2 + 1] - x2[i2], _EM20)
        w2 = np.clip((rate_arr - x2[i2]) / dx2, 0.0, 1.0)
    else:
        i2 = np.zeros(n, dtype=int)
        w2 = np.zeros(n, dtype=float)

    if nx3 > 1:
        i3 = np.clip(np.searchsorted(x3, temp_arr, side="right") - 1, 0, nx3 - 2)
        dx3 = np.maximum(x3[i3 + 1] - x3[i3], _EM20)
        w3 = np.clip((temp_arr - x3[i3]) / dx3, 0.0, 1.0)
    else:
        i3 = np.zeros(n, dtype=int)
        w3 = np.zeros(n, dtype=float)

    v000 = y_grid[i3, i2, i1]
    v001 = y_grid[i3, i2, i1 + 1]
    v010 = y_grid[i3, np.minimum(i2 + 1, nx2 - 1), i1]
    v011 = y_grid[i3, np.minimum(i2 + 1, nx2 - 1), i1 + 1]
    v100 = y_grid[np.minimum(i3 + 1, nx3 - 1), i2, i1]
    v101 = y_grid[np.minimum(i3 + 1, nx3 - 1), i2, i1 + 1]
    v110 = y_grid[np.minimum(i3 + 1, nx3 - 1), np.minimum(i2 + 1, nx2 - 1), i1]
    v111 = y_grid[np.minimum(i3 + 1, nx3 - 1), np.minimum(i2 + 1, nx2 - 1), i1 + 1]

    # Interpolate along x2 and x3
    c00 = (1.0 - w2) * v000 + w2 * v010
    c01 = (1.0 - w2) * v001 + w2 * v011
    c10 = (1.0 - w2) * v100 + w2 * v110
    c11 = (1.0 - w2) * v101 + w2 * v111

    c0 = (1.0 - w3) * c00 + w3 * c10
    c1 = (1.0 - w3) * c01 + w3 * c11

    val = c0 + w1 * (c1 - c0)
    slp = (c1 - c0) / dx1
    return val, slp


def _solve_max_principal_strain(eps_tot: np.ndarray) -> np.ndarray:
    """Solve cubic characteristic equation for maximum tensile principal strain.

    Upstream Fortran reference: sigeps74.F lines 317-360.
    4 Newton-Raphson iterations.
    """
    n = eps_tot.shape[0]
    dav = (eps_tot[:, 0] + eps_tot[:, 1] + eps_tot[:, 2]) / 3.0
    e1 = eps_tot[:, 0] - dav
    e2 = eps_tot[:, 1] - dav
    e3 = eps_tot[:, 2] - dav
    e4 = 0.5 * eps_tot[:, 3]
    e5 = 0.5 * eps_tot[:, 4]
    e6 = 0.5 * eps_tot[:, 5]

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


# ============================================================================
# Parameter Dataclass: Law74Params
# ============================================================================

@dataclass
class Law74Params:
    """Strongly-typed parameters for /MAT/LAW74 (/MAT/HILL_3D, /MAT/ORTH_PLAS, /MAT/THERM_HILL).

    Follows starter/source/materials/mat/mat074/hm_read_mat74.F.
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

    # Hardening and failure
    fsmooth: int = 0
    chard: float = 0.0      # fisokin: iso-kinematic hardening factor [0..1]
    fcut: float = 0.0
    eps_max: float = _INF   # maximum plastic strain
    epsr1: float = _INF     # tensile strain start of failure
    epsr2: float = 2.0 * _INF  # tensile strain end of failure

    # Hill 3D yield parameters
    s11y: float = 1.0
    s22y: float = 1.0
    s33y: float = 1.0
    s12y: float = 1.0
    s23y: float = 1.0
    s31y: float = 1.0

    # Yield table & scaling
    table_id: int = 0       # table/curve ID or object
    fscale: float = 1.0     # yield stress scale factor
    pscale: float = 1.0     # strain rate scale factor
    sigy0: float = 1.0      # reference/initial yield stress when table_id is 0

    # Thermal properties
    t0: float = 293.0       # initial temperature
    rhocp: float = 0.0      # heat capacity per unit volume (rho * Cp)

    # Identifiers & extra objects
    id: int = 1
    title: str = "LAW74_HILL_3D"
    curve_e: Any = None
    yield_table: Any = None
    ipla: int = 0
    opte: int = 0

    # Derived coefficients (computed in __post_init__)
    g: float = field(init=False, default=0.0)
    c1: float = field(init=False, default=0.0)
    ff: float = field(init=False, default=0.0)
    gg: float = field(init=False, default=0.0)
    hh: float = field(init=False, default=0.0)
    ll: float = field(init=False, default=0.0)
    mm: float = field(init=False, default=0.0)
    nn: float = field(init=False, default=0.0)
    xfac: float = field(init=False, default=1.0)
    yfac: float = field(init=False, default=1.0)

    def __post_init__(self) -> None:
        if self.e <= 0.0:
            raise ValueError(f"/MAT/LAW74/{self.id}: Young's modulus E must be > 0.")
        if self.nu >= 0.5:
            self.nu = 0.499
        if self.nu < 0.0:
            raise ValueError(f"/MAT/LAW74/{self.id}: Poisson's ratio nu must be >= 0.")
        if self.rho0 <= 0.0:
            self.rho0 = 1.0
        if self.refer_rho <= 0.0:
            self.refer_rho = self.rho0

        if self.eps_max <= 0.0:
            self.eps_max = _INF
        if self.epsr1 <= 0.0:
            self.epsr1 = _INF
        if self.epsr2 <= 0.0:
            self.epsr2 = 2.0 * _INF
        if self.t0 <= 0.0:
            self.t0 = 293.0

        for name, val in [("s11y", self.s11y), ("s22y", self.s22y), ("s33y", self.s33y),
                          ("s12y", self.s12y), ("s23y", self.s23y), ("s31y", self.s31y)]:
            if val <= 0.0:
                raise ValueError(f"/MAT/LAW74/{self.id}: Yield parameter {name} must be > 0 (got {val}).")

        # Derived coefficients (hm_read_mat74.F lines 229-256)
        self.g = 0.5 * self.e / (1.0 + self.nu)
        self.c1 = self.e / (3.0 * (1.0 - 2.0 * self.nu))
        self.ff = 0.5 * (1.0 / (self.s22y ** 2) + 1.0 / (self.s33y ** 2) - 1.0 / (self.s11y ** 2))
        self.gg = 0.5 * (1.0 / (self.s11y ** 2) + 1.0 / (self.s33y ** 2) - 1.0 / (self.s22y ** 2))
        self.hh = 0.5 * (1.0 / (self.s11y ** 2) + 1.0 / (self.s22y ** 2) - 1.0 / (self.s33y ** 2))
        self.ll = 0.5 / (self.s23y ** 2)
        self.mm = 0.5 / (self.s31y ** 2)
        self.nn = 0.5 / (self.s12y ** 2)
        self.xfac = 1.0 / self.pscale if self.pscale != 0.0 else 1.0
        self.yfac = self.fscale

        if self.ifunce > 0 or self.curve_e is not None:
            self.opte = 1

    @property
    def fisokin(self) -> float:
        return self.chard

    @property
    def soundsp(self) -> float:
        return math.sqrt((self.c1 + 4.0 / 3.0 * self.g) / max(self.rho0, _EM20))

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)


# ============================================================================
# Material Constructor: build_law74
# ============================================================================

def _extract_param(p: Dict[str, Any], keys: Sequence[str], default: Any) -> Any:
    for k in keys:
        if k in p and p[k] is not None:
            return p[k]
    return default


def build_law74(rec: Any = None, **kwargs: Any) -> Material:
    """Construct a Material entity for /MAT/LAW74 (/MAT/HILL_3D).

    Follows starter/source/materials/mat/mat074/hm_read_mat74.F.
    """
    if isinstance(rec, Law74Params):
        p_obj = rec
    else:
        p_dict: Dict[str, Any] = {}
        if rec is not None:
            if isinstance(rec, dict):
                p_dict.update(rec.get("params", rec))
                for attr in ("id", "title", "rho0", "density"):
                    if attr in rec:
                        p_dict.setdefault(attr, rec[attr])
            elif hasattr(rec, "params") and isinstance(rec.params, dict):
                p_dict.update(rec.params)
                for attr in ("id", "title", "rho0"):
                    if hasattr(rec, attr):
                        p_dict.setdefault(attr, getattr(rec, attr))
        p_dict.update(kwargs)

        mid = int(_extract_param(p_dict, ["id", "mat_id", "mid", "MAT_ID"], 1))
        title = str(_extract_param(p_dict, ["title", "name", "TITR"], "LAW74_HILL_3D"))

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

        rho0_val = _get_f(["MAT_RHO", "Refer_Rho", "rho0", "density", "rho"], 1.0)
        rhor_val = _get_f(["Refer_Rho", "rhor", "MAT_REFRHO"], rho0_val)
        e_val = _get_f(["MAT_E", "E", "e", "young", "Young"], 210000.0)
        nu_val = _get_f(["MAT_NU", "NU", "nu", "anu"], 0.3)

        ifunce_val = _get_i(["Yr_fun", "IFUNCE", "ifunce", "fct_e"], 0)
        einf_val = _get_f(["MAT_EFIB", "EINF", "Einf", "einf"], 0.0)
        ce_val = _get_f(["MAT_C", "CE", "ce", "c_e"], 0.0)

        fsmooth_val = _get_i(["Fsmooth", "fsmooth", "israte", "ISRATE"], 0)
        chard_val = _get_f(["MAT_HARD", "CHARD", "chard", "C_hard", "FISOKIN", "fisokin"], 0.0)
        fcut_val = _get_f(["Fcut", "fcut", "FCUT"], 0.0)

        s11y_val = _get_f(["MAT_SIGT1", "SIGT1", "s11y", "S11Y", "sig11y"], 1.0)
        s22y_val = _get_f(["MAT_SIGT2", "SIGT2", "s22y", "S22Y", "sig22y"], 1.0)
        s33y_val = _get_f(["MAT_SIGT3", "SIGT3", "s33y", "S33Y", "sig33y"], 1.0)
        s12y_val = _get_f(["MAT_SIGYT1", "SIGYT1", "s12y", "S12Y", "sig12y"], 1.0)
        s23y_val = _get_f(["MAT_SIGYT2", "SIGYT2", "s23y", "S23Y", "sig23y"], 1.0)
        s31y_val = _get_f(["MAT_SIGYT3", "SIGYT3", "s31y", "S31Y", "sig31y"], 1.0)

        table_id_val = _extract_param(p_dict, ["FUN_A1", "ITABLE", "table_id", "TABLE", "table", "curve"], 0)
        fscale_val = _get_f(["MAT_FScale", "FScale", "fscale", "YFAC", "yfac"], 1.0)
        pscale_val = _get_f(["MAT_PScale", "PScale", "pscale", "X2FAC", "xfac"], 1.0)
        sigy0_val = _get_f(["sigy0", "SIGY0", "sigy", "SIGY", "sig_y", "SIG_Y", "MAT_SIGY", "MAT_YLD", "yld0", "yield_stress"], 1.0)

        eps_max_val = _get_f(["MAT_EPS", "EPS", "eps", "eps_max", "EPSMAX"], _INF)
        epsr1_val = _get_f(["MAT_EPST1", "EPST1", "epsr1", "EPSR1"], _INF)
        epsr2_val = _get_f(["MAT_EPST2", "EPST2", "epsr2", "EPSR2"], 2.0 * _INF)

        t0_val = _get_f(["T_Initial", "T0", "t0", "temp_initial", "temp0"], 293.0)
        rhocp_val = _get_f(["MAT_SPHEAT", "RHOCP", "rhocp", "spheat"], 0.0)

        ipla_val = _get_i(["IPLA", "ipla"], 0)
        curve_e_val = _extract_param(p_dict, ["curve_e", "E_curve", "fct_e_obj"], None)
        yield_table_val = _extract_param(p_dict, ["yield_table", "table_obj"], None)

        p_obj = Law74Params(
            rho0=rho0_val,
            refer_rho=rhor_val,
            e=e_val,
            nu=nu_val,
            ifunce=ifunce_val,
            einf=einf_val,
            ce=ce_val,
            fsmooth=fsmooth_val,
            chard=chard_val,
            fcut=fcut_val,
            s11y=s11y_val,
            s22y=s22y_val,
            s33y=s33y_val,
            s12y=s12y_val,
            s23y=s23y_val,
            s31y=s31y_val,
            table_id=table_id_val,
            fscale=fscale_val,
            pscale=pscale_val,
            sigy0=sigy0_val,
            eps_max=eps_max_val,
            epsr1=epsr1_val,
            epsr2=epsr2_val,
            t0=t0_val,
            rhocp=rhocp_val,
            id=mid,
            title=title,
            curve_e=curve_e_val,
            yield_table=yield_table_val,
            ipla=ipla_val,
        )

    params_dict = {
        "rho0": p_obj.rho0,
        "refer_rho": p_obj.refer_rho,
        "e": p_obj.e,
        "E": p_obj.e,
        "nu": p_obj.nu,
        "sigy0": p_obj.sigy0,
        "ifunce": p_obj.ifunce,
        "einf": p_obj.einf,
        "ce": p_obj.ce,
        "fsmooth": p_obj.fsmooth,
        "chard": p_obj.chard,
        "fisokin": p_obj.chard,
        "fcut": p_obj.fcut,
        "s11y": p_obj.s11y,
        "s22y": p_obj.s22y,
        "s33y": p_obj.s33y,
        "s12y": p_obj.s12y,
        "s23y": p_obj.s23y,
        "s31y": p_obj.s31y,
        "table_id": p_obj.table_id,
        "fscale": p_obj.fscale,
        "pscale": p_obj.pscale,
        "eps_max": p_obj.eps_max,
        "epsr1": p_obj.epsr1,
        "epsr2": p_obj.epsr2,
        "t0": p_obj.t0,
        "rhocp": p_obj.rhocp,
        "G": p_obj.g,
        "C1": p_obj.c1,
        "FF": p_obj.ff,
        "GG": p_obj.gg,
        "HH": p_obj.hh,
        "LL": p_obj.ll,
        "MM": p_obj.mm,
        "NN": p_obj.nn,
        "_obj": p_obj,
    }

    mat = Material(
        id=p_obj.id,
        law=74,
        rho0=p_obj.rho0,
        title=p_obj.title,
        law_name="LAW74",
        params=params_dict,
    )
    return mat


def _get_params(mat: Any) -> Law74Params:
    if isinstance(mat, Law74Params):
        return mat
    if hasattr(mat, "params") and isinstance(mat.params, dict) and "_obj" in mat.params:
        return mat.params["_obj"]
    if isinstance(mat, dict) and "_obj" in mat:
        return mat["_obj"]
    m = build_law74(mat)
    return m.params["_obj"]


# ============================================================================
# Sound Speed & Allocations
# ============================================================================

def sound_speed_solid(mat: Any, rho: Optional[float] = None, extra: Optional[Dict[str, Any]] = None) -> float:
    """Acoustic longitudinal sound speed for solids: c = sqrt((c11 + 4/3*g) / rho).

    Upstream reference: sigeps74.F line 305 (SOUNDSP = SQRT((C11+FOUR*G1/THREE)/RHO0)).
    """
    p = _get_params(mat)
    rho_val = float(rho) if rho is not None and float(rho) > 0.0 else p.rho0

    e_curr = p.e
    if extra is not None and "uvar74" in extra and extra["uvar74"] is not None:
        uvar = np.asarray(extra["uvar74"])
        if uvar.ndim >= 1 and uvar.shape[-1] >= 1:
            pla_val = float(np.max(uvar[..., 0]))
            if pla_val > 0.0:
                if p.opte == 1 and p.curve_e is not None:
                    escale, _ = _eval_curve_1d(p.curve_e, pla_val)
                    e_curr = float(escale * p.e)
                elif p.ce > 0.0:
                    e_curr = float(p.e - (p.e - p.einf) * (1.0 - math.exp(-p.ce * pla_val)))

    g = 0.5 * e_curr / (1.0 + p.nu)
    c1 = e_curr / (3.0 * (1.0 - 2.0 * p.nu))
    return math.sqrt((c1 + 4.0 / 3.0 * g) / max(rho_val, _EM20))


sound_speed = sound_speed_solid


def extra_shapes(mat: Any, nip: Optional[int] = 1) -> Dict[str, Tuple[int, ...]]:
    """Extra allocations required in element state for LAW74.

    10 channels:
      - uvar[:, 0]: plastic strain pla
      - uvar[:, 1:4]: table interpolation coordinate cache
      - uvar[:, 4:10]: back-stress tensor alpha (xx, yy, zz, xy, yz, zx)
    """
    if nip is not None and nip > 1:
        return {
            "uvar74": (nip, 10),
            "temp": (nip,),
            "off": (nip,),
        }
    return {
        "uvar74": (10,),
        "temp": (),
        "off": (),
    }


# ============================================================================
# Shell update (unsupported)
# ============================================================================

def shell_update(mat: Any, *args: Any, **kwargs: Any) -> Any:
    """Plane-stress update is not supported for LAW74 (solids only)."""
    raise NotImplementedError("/MAT/LAW74 is for solid elements only.")


shell_update_law74 = shell_update


# ============================================================================
# Solid Constitutive Update: solid_update (sigeps74.F)
# ============================================================================

def solid_update(mat: Any, sig: np.ndarray, deps: np.ndarray,
                 epsp: Optional[Union[float, np.ndarray]] = None,
                 dt: float = 0.0,
                 extra: Optional[Dict[str, Any]] = None,
                 return_tuple: bool = False,
                 **kwargs: Any) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """3D constitutive update for /MAT/LAW74 (/MAT/HILL_3D, /MAT/THERM_HILL).

    Follows engine/source/materials/mat/mat074/sigeps74.F line by line.

    Parameters
    ----------
    mat : Material, Law74Params, or dict
    sig : (6,) or (n, 6) ndarray
        Old Cauchy stress tensor [xx, yy, zz, xy, yz, zx].
    deps : (6,) or (n, 6) ndarray
        Engineering strain increment [xx, yy, zz, xy, yz, zx].
    epsp : (n,) or float, optional
        Accumulated equivalent plastic strain history.
    dt : float
        Time step increment.
    extra : dict, optional
        Element state views: 'uvar74', 'temp', 'off', 'rho', 'amu', 'eps', 'rate', etc.
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
    if deps_arr.shape[0] == 1 and n > 1:
        deps_arr = np.repeat(deps_arr, n, axis=0)

    if n == 0:
        c_empty = np.empty(0, dtype=float)
        return (sig, epsp, c_empty) if return_tuple else (sig, epsp)

    # 1. State extraction: uvar (10 channels)
    uvar = None
    if extra is not None and "uvar74" in extra and extra["uvar74"] is not None:
        uvar_raw = extra["uvar74"]
        if isinstance(uvar_raw, np.ndarray):
            if uvar_raw.ndim == 1 and n == 1 and uvar_raw.shape[0] == 10:
                uvar = uvar_raw[None, :]
            elif uvar_raw.ndim == 2 and uvar_raw.shape[0] == n and uvar_raw.shape[1] >= 10:
                uvar = uvar_raw
    if uvar is None:
        uvar = np.zeros((n, 10), dtype=float)

    # Plastic strain history
    if epsp is not None:
        pla = np.asarray(epsp, dtype=float).flatten().copy()
        if len(pla) == 1 and n > 1:
            pla = np.full(n, pla[0], dtype=float)
    else:
        pla = uvar[:, 0].copy()

    # Temperature history
    if extra is not None and "temp" in extra and extra["temp"] is not None:
        temp_arr = np.asarray(extra["temp"], dtype=float).flatten().copy()
        if len(temp_arr) == 1 and n > 1:
            temp_arr = np.full(n, temp_arr[0], dtype=float)
    else:
        temp_arr = np.full(n, p.t0, dtype=float)

    # Deletion flag
    off = None
    if extra is not None and "off" in extra and extra["off"] is not None:
        off = np.asarray(extra["off"], dtype=float).flatten()
        if len(off) == 1 and n > 1:
            off = np.full(n, off[0], dtype=float)
    else:
        off = np.ones(n, dtype=float)

    # 2. Dynamic Young's modulus degradation (sigeps74.F lines 220-243)
    e_curr = np.full(n, p.e, dtype=float)
    if p.opte == 1 and p.curve_e is not None:
        mask_pla = pla > 0.0
        if np.any(mask_pla):
            esc, _ = _eval_curve_1d(p.curve_e, pla[mask_pla])
            e_curr[mask_pla] = esc * p.e
    elif p.ce > 0.0:
        mask_pla = pla > 0.0
        if np.any(mask_pla):
            e_curr[mask_pla] = p.e - (p.e - p.einf) * (1.0 - np.exp(-p.ce * pla[mask_pla]))

    g1 = 0.5 * e_curr / (1.0 + p.nu)
    g21 = 2.0 * g1
    g31 = 3.0 * g1
    c11 = e_curr / (3.0 * (1.0 - 2.0 * p.nu))

    soundsp = np.sqrt((c11 + 4.0 / 3.0 * g1) / max(p.rho0, _EM20))

    # 3. Backstress shift & Elastic trial stress (sigeps74.F lines 276-304)
    alpha = uvar[:, 4:10].copy()  # [xx, yy, zz, xy, yz, zx]
    sig_eff = sig_arr.copy()
    if p.fisokin != 0.0:
        sig_eff -= alpha

    p0 = -(sig_eff[:, 0] + sig_eff[:, 1] + sig_eff[:, 2]) / 3.0
    dav = (deps_arr[:, 0] + deps_arr[:, 1] + deps_arr[:, 2]) / 3.0

    st = np.empty_like(sig_arr)
    st[:, 0] = sig_eff[:, 0] + p0 + g21 * (deps_arr[:, 0] - dav)
    st[:, 1] = sig_eff[:, 1] + p0 + g21 * (deps_arr[:, 1] - dav)
    st[:, 2] = sig_eff[:, 2] + p0 + g21 * (deps_arr[:, 2] - dav)
    st[:, 3] = sig_eff[:, 3] + g1 * deps_arr[:, 3]
    st[:, 4] = sig_eff[:, 4] + g1 * deps_arr[:, 4]
    st[:, 5] = sig_eff[:, 5] + g1 * deps_arr[:, 5]

    st_trial = st.copy()  # SIGE in Fortran

    # 4. Maximum principal tensile strain & Tensile failure factor (sigeps74.F lines 317-365)
    eps_tot = None
    if extra is not None and "eps" in extra and extra["eps"] is not None:
        eps_tot = np.asarray(extra["eps"], dtype=float)
        if eps_tot.ndim == 1:
            eps_tot = eps_tot[None, :]
        if eps_tot.shape[0] == 1 and n > 1:
            eps_tot = np.repeat(eps_tot, n, axis=0)
    elif extra is not None and "strain" in extra and extra["strain"] is not None:
        eps_tot = np.asarray(extra["strain"], dtype=float)
        if eps_tot.ndim == 1:
            eps_tot = eps_tot[None, :]
        if eps_tot.shape[0] == 1 and n > 1:
            eps_tot = np.repeat(eps_tot, n, axis=0)
    else:
        eps_tot = deps_arr.copy()

    epst = _solve_max_principal_strain(eps_tot)
    if p.epsr2 > p.epsr1:
        fail = np.clip((p.epsr2 - epst) / (p.epsr2 - p.epsr1), 0.0, 1.0)
    else:
        fail = np.ones(n, dtype=float)

    # 5. Strain rate evaluation
    if extra is not None and "rate" in extra and extra["rate"] is not None:
        rate = np.asarray(extra["rate"], dtype=float).flatten()
        if len(rate) == 1 and n > 1:
            rate = np.full(n, rate[0], dtype=float)
    elif extra is not None and "epsp_rate" in extra and extra["epsp_rate"] is not None:
        rate = np.asarray(extra["epsp_rate"], dtype=float).flatten()
        if len(rate) == 1 and n > 1:
            rate = np.full(n, rate[0], dtype=float)
    elif dt > 0.0:
        exx_d = deps_arr[:, 0] - dav
        eyy_d = deps_arr[:, 1] - dav
        ezz_d = deps_arr[:, 2] - dav
        ee = (exx_d ** 2 + eyy_d ** 2 + ezz_d ** 2
              + 0.5 * (deps_arr[:, 3] ** 2 + deps_arr[:, 4] ** 2 + deps_arr[:, 5] ** 2))
        rate = np.sqrt(np.maximum((2.0 / 3.0) * ee, 0.0)) / dt
    else:
        rate = np.zeros(n, dtype=float)

    if p.fsmooth > 0 and p.fcut > 0.0 and dt > 0.0:
        omega = 2.0 * math.pi * p.fcut
        alpha_f = min(1.0, omega * dt)
        rate = alpha_f * rate

    # 6. Yield stress lookup (sigeps74.F lines 367-444)
    yld_val, dydx = _eval_yield_table(p, pla, rate * p.xfac, temp_arr)
    yld = p.yfac * yld_val * fail
    h = fail * dydx

    if p.fisokin != 0.0:
        yk, _ = _eval_yield_table(p, np.zeros_like(pla), rate * p.xfac, temp_arr)
        yld = (1.0 - p.fisokin) * yld + p.fisokin * fail * p.yfac * yk
    yld = np.maximum(yld, _EM20)

    # 7. Hill 3D equivalent stress (sigeps74.F lines 450-454)
    cri2 = (p.ff * (st[:, 1] - st[:, 2]) ** 2
            + p.gg * (st[:, 2] - st[:, 0]) ** 2
            + p.hh * (st[:, 0] - st[:, 1]) ** 2
            + 2.0 * p.ll * st[:, 4] ** 2
            + 2.0 * p.mm * st[:, 5] ** 2
            + 2.0 * p.nn * st[:, 3] ** 2)
    cri = np.sqrt(np.maximum(cri2, 0.0))

    # 8. Radial return projection (sigeps74.F lines 448-513)
    r = np.minimum(1.0, yld / np.maximum(cri, _EM20))

    # Hydrostatic pressure P = C11 * AMU
    if extra is not None and "amu" in extra and extra["amu"] is not None:
        amu = np.asarray(extra["amu"], dtype=float).flatten()
        p_hydro = c11 * amu
    elif extra is not None and "rho" in extra and extra["rho"] is not None:
        rho_curr = np.asarray(extra["rho"], dtype=float).flatten()
        amu = rho_curr / max(p.rho0, _EM20) - 1.0
        p_hydro = c11 * amu
    else:
        p_hydro = -p0 - c11 * 3.0 * dav

    if p.ipla == 1:
        dpla = (1.0 - r) * cri / np.maximum(g31 + h, _EM20)
        yld = np.maximum(yld + (1.0 - p.fisokin) * dpla * h, 0.0)
        r = np.minimum(1.0, yld / np.maximum(cri, _EM20))
        pla += dpla
    elif p.ipla == 2:
        dpla = (1.0 - r) * cri / np.maximum(g31, _EM20)
        pla += dpla
    else:  # IPLA == 0 (default)
        dpla = (1.0 - r) * cri / np.maximum(g31 + h, _EM20)
        pla += dpla

    s_dev = st * r[:, None]

    # 9. Kinematic hardening update (sigeps74.F lines 517-551)
    if p.fisokin != 0.0:
        ds = st_trial - s_dev
        hkin = (2.0 / 3.0) * p.fisokin * h
        denom_kin = g21 + hkin
        alpha_fac = np.where(denom_kin > _EM20, hkin / denom_kin, 0.0)
        delta_alpha = alpha_fac[:, None] * ds
        alpha += delta_alpha
        uvar[:, 4:10] = alpha

    # 10. Total stress tensor
    sig_new = s_dev.copy()
    sig_new[:, 0] -= p_hydro
    sig_new[:, 1] -= p_hydro
    sig_new[:, 2] -= p_hydro

    if p.fisokin != 0.0:
        sig_new += alpha

    # 11. Temperature update from adiabatic plastic heating (sigeps74.F lines 908-910)
    if p.rhocp > 0.0:
        dtemp = (yld * dpla) / p.rhocp
        temp_arr += dtemp
        if extra is not None and "temp" in extra and extra["temp"] is not None:
            if hasattr(extra["temp"], "__setitem__"):
                extra["temp"][:] = temp_arr[0] if is_1d else temp_arr

    # 12. Element deletion when pla > eps_max (sigeps74.F lines 914-926)
    if off is not None:
        off[off < 0.1] = 0.0
        off[off < 1.0] *= 0.8
        rupture = (pla > p.eps_max) & (off == 1.0)
        off[rupture] = 0.8
        deleted = (off == 0.0)
        if np.any(deleted):
            sig_new[deleted] = 0.0
        if extra is not None and "off" in extra and extra["off"] is not None:
            if hasattr(extra["off"], "__setitem__"):
                extra["off"][:] = off[0] if is_1d else off

    # Store uvar history
    uvar[:, 0] = pla
    if extra is not None and "uvar74" in extra and extra["uvar74"] is not None:
        if hasattr(extra["uvar74"], "__setitem__"):
            extra["uvar74"][:] = uvar[0] if is_1d else uvar

    if epsp is not None and hasattr(epsp, "__setitem__"):
        try:
            epsp[:] = pla[0] if is_1d else pla
        except Exception:
            pass

    if hasattr(sig, "__setitem__"):
        try:
            sig[:] = sig_new[0] if is_1d else sig_new
        except Exception:
            pass

    if is_1d:
        sig_out = sig_new[0]
        epsp_out = float(pla[0])
        c_out = float(soundsp[0])
    else:
        sig_out = sig_new
        epsp_out = pla
        c_out = soundsp

    return (sig_out, epsp_out, c_out) if return_tuple else (sig_out, epsp_out)


solid_update_law74 = solid_update


# ============================================================================
# Consistent Algorithmic Tangent: consistent_solid_tangent
# ============================================================================

def consistent_solid_tangent(mat: Any, sig: np.ndarray,
                             epsp: Optional[Union[float, np.ndarray]] = None,
                             dt: Any = 0.0,
                             extra: Optional[Dict[str, Any]] = None,
                             epsp_incr: Optional[Union[float, np.ndarray]] = None,
                             deps: Optional[np.ndarray] = None,
                             symmetric: bool = False,
                             h: float = 1.0e-7,
                             **kwargs: Any) -> np.ndarray:
    """Consistent algorithmic solid tangent operator (n, 6, 6) or (6, 6)."""
    if isinstance(dt, (np.ndarray, list)):
        epsp_incr = dt
        dt = 0.0
    elif isinstance(dt, dict) and extra is None:
        extra = dt
        dt = 0.0
    elif isinstance(extra, (int, float, np.ndarray, list)) and epsp_incr is None:
        epsp_incr = extra
        extra = None

    if "deps" in kwargs and deps is None:
        deps = kwargs["deps"]
    if "epsp_incr" in kwargs and epsp_incr is None:
        epsp_incr = kwargs["epsp_incr"]
    if "extra" in kwargs and extra is None:
        extra = kwargs["extra"]
    if "dt" in kwargs:
        dt = float(kwargs["dt"])
    if "symmetric" in kwargs:
        symmetric = bool(kwargs["symmetric"])
    if "h" in kwargs:
        h = float(kwargs["h"])

    sig_arr = np.asarray(sig, dtype=float)
    is_1d = (sig_arr.ndim == 1)
    if is_1d:
        sig_arr = sig_arr[None, :]

    n = sig_arr.shape[0]
    if n == 0:
        return np.empty((0, 6, 6), dtype=float) if not is_1d else np.empty((6, 6), dtype=float)

    p = _get_params(mat)

    # Elastic tangent fallback
    c_el = np.zeros((6, 6), dtype=float)
    c11 = p.c1 + 4.0 / 3.0 * p.g
    c12 = p.c1 - 2.0 / 3.0 * p.g
    c_el[0, 0] = c_el[1, 1] = c_el[2, 2] = c11
    c_el[0, 1] = c_el[0, 2] = c_el[1, 0] = c_el[1, 2] = c_el[2, 0] = c_el[2, 1] = c12
    c_el[3, 3] = c_el[4, 4] = c_el[5, 5] = p.g

    d_tangent = np.zeros((n, 6, 6), dtype=float)
    for i in range(n):
        d_tangent[i] = c_el.copy()

    if deps is not None:
        deps_arr = np.asarray(deps, dtype=float)
        if deps_arr.ndim == 1:
            deps_arr = deps_arr[None, :]
        if deps_arr.shape[0] == 1 and n > 1:
            deps_arr = np.repeat(deps_arr, n, axis=0)

        h_val = float(h)
        for i in range(n):
            sig_i = sig_arr[i].copy()
            deps_i = deps_arr[i].copy()
            epsp_i = float(epsp[i]) if (epsp is not None and hasattr(epsp, "__len__")) else (float(epsp) if epsp is not None else 0.0)

            ext_i: Dict[str, Any] = {}
            if extra is not None:
                for k, v in extra.items():
                    if isinstance(v, np.ndarray):
                        if v.shape[0] == n:
                            ext_i[k] = v[i].copy()
                        else:
                            ext_i[k] = v.copy()
                    else:
                        ext_i[k] = v

            d_num = np.zeros((6, 6), dtype=float)
            for j in range(6):
                deps_p = deps_i.copy()
                deps_m = deps_i.copy()
                deps_p[j] += h_val
                deps_m[j] -= h_val

                ext_p = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ext_i.items()}
                ext_m = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ext_i.items()}

                sig_p, _ = solid_update(mat, sig_i.copy(), deps_p, epsp=epsp_i, dt=dt, extra=ext_p, return_tuple=False)
                sig_m, _ = solid_update(mat, sig_i.copy(), deps_m, epsp=epsp_i, dt=dt, extra=ext_m, return_tuple=False)
                d_num[:, j] = (sig_p[:6] - sig_m[:6]) / (2.0 * h_val)

            if symmetric:
                d_num = 0.5 * (d_num + d_num.T)
            d_tangent[i] = d_num

    if symmetric:
        d_tangent = 0.5 * (d_tangent + np.swapaxes(d_tangent, -1, -2))

    return d_tangent[0] if is_1d else d_tangent


# ============================================================================
# Resolution Helper
# ============================================================================

def resolve(mat: Material, model: Any, log: Any = None) -> None:
    """Resolve /FUNCT and /TABLE references into curve_e and yield_table in mat.params."""
    p = _get_params(mat)
    functions = getattr(model, "functions", model.get("functions", {}) if isinstance(model, dict) else {})
    tables = getattr(model, "tables", model.get("tables", {}) if isinstance(model, dict) else {})

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
    """Register LAW74 in pyradioss MAT_PHYSICS_REGISTRY."""
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        for k in (74, "74", "LAW74", "HILL_3D", "ORTH_PLAS", "THERM_HILL",
                  "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS", "MAT_THERM_HILL",
                  "LAW74_HILL_3D"):
            MAT_PHYSICS_REGISTRY[k] = build_law74
    except Exception:
        pass


_register()
