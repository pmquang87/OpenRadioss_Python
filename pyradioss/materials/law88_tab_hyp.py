"""
LAW88 — Tabulated Hyperelastic Material Model for Solids and Shells (/MAT/LAW88, /MAT/HYP_TAB).

Implements the tabulated hyperelastic constitutive law for 3D solid continuum elements
and 2D shell / membrane elements in pure Python / NumPy.

Upstream Reference Files (OpenRadioss Fortran & C++):
------------------------------------------------------
- starter/source/materials/mat/mat088/cpp_table_mat_spline_fit.cpp
- starter/source/materials/mat/mat088/table_mat_spline_fit_mod.F90
- starter/source/materials/mat/mat088/hm_read_mat88.F90
- engine/source/materials/tools/table_mat_vinterp.F
- engine/source/materials/mat/mat088/sigeps88.F90 (3D continuum solid kernel)
- engine/source/materials/mat/mat088/sigeps88c.F90 (2D shell / membrane plane-stress kernel)

Theory and Formulation:
-----------------------
1. Curve Preprocessing & Smoothing:
   - Isotone regression using Pool Adjacent Violators (PAV) algorithm enforcing monotonic
     non-decreasing behavior (cpp_table_mat_spline_fit.cpp:34-85).
   - 1D Laplacian smoothing combined with PAV projection and origin pinning (0, 0)
     (cpp_table_mat_spline_fit.cpp:88-126).
   - Fritsch-Carlson PCHIP slope calculation preserving monotonicity (lines 129-151).
   - Cubic Hermite evaluation with flat extrapolation (lines 154-173).
   - Uniform grid reconstruction with exact zero placement (lines 196-227).

2. Principal Stretch Representation:
   - Strains are rotated to principal directions via 3x3 eigen-decomposition (solids, valpvec_v)
     or 2x2 eigen-decomposition (shells).
   - Principal stretches computed for logarithmic (exp), Green-Lagrange (sqrt(1+2E)),
     or engineering (1+eps) strain measures.
   - Isochoric stretches bar{lambda}_i = lambda_i * J^(-1/3) for incompressible materials (nu >= 0.49).

3. Recursive Series Expansion:
   - Uniaxial loading function g(lambda, eps_dot) evaluated from tabulated data.
   - 6-iteration recursive stretch series:
       f(lambda) = lambda * g(lambda) + sum_{n=1}^6 lambda^{(-nu)^n} * g(lambda^{(-nu)^n})
     with analytical derivatives df/dlambda.

4. Hydrostatic Pressure & Stress:
   - Incompressible: P = K*(J - 1), t_i = [(2/3)*f_i - (1/3)*(f_j + f_k) + P] / J.
   - Compressible foam (0 < nu < 0.49): x_foam = J^{-nu/(1-2nu)}, f_J from recursive series,
     effective bulk modulus K_eff = [-nu/(1-2nu)] * x_foam * df_J/dx, t_i = (f_i - f_J) / J.
   - Viscous pressure (beta > 0): P = P_old * exp(-beta*dt) + K * ldav * (1 - exp(-beta*dt)) / beta.

5. Loading / Unloading and Hysteresis:
   - Energy-based loading/unloading detection.
   - iunl_for = 1: Tabulated unloading curve with dominant direction selection, normalized
     amplitude x_hat in [-1, 1], loop closure detection, ratio R = g_unl / g_load, and cubic blending.
   - iunl_for = 2: Hysteretic energy damage ratio R = 1 - (1 - hys)*(1 - (E/E_max)^shape).

6. Shell Plane Stress:
   - 3-iteration Newton-Raphson loop solving t_3(lambda_3) = 0 for out-of-plane stretch lambda_3.
   - Shell thickness update: h_n = h_n + h_layer * h_0 * lambda_3 / J^(1/3).
   - Acoustic plane-stress wave speed: c = sqrt(A_11 / min(rho, rho_0)).

7. Damping, Softening and Tangents:
   - Frictional deviatoric damping stresses with cutoff sigma_f.
   - Cosine damage softening based on stretch invariants I_1, I_2.
   - Consistent algorithmic tangents for solid (6x6) and shell (3x3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Numerical guards matching OpenRadioss constant_mod / implicit_f.inc
_EM01 = 0.1
_EM03 = 1e-3
_EM06 = 1e-6
_EM07 = 1e-7
_EM08 = 1e-8
_EM10 = 1e-10
_EM12 = 1e-12
_EM20 = 1e-20
_FOUR_OVER_3 = 4.0 / 3.0
_FOUR_OVER_5 = 0.8
_TWO_THIRD = 2.0 / 3.0
_THIRD = 1.0 / 3.0


# ============================================================================
# 1. Curve Preprocessing, Smoothing, and Spline Fitting
#    Reference: cpp_table_mat_spline_fit.cpp:34-273
# ============================================================================

def isotone_project_pav(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Pool Adjacent Violators (PAV) isotonic regression enforcing monotonic non-decreasing behavior.

    Matches C++ implementation in cpp_table_mat_spline_fit.cpp:34-85.

    Parameters
    ----------
    y : np.ndarray
        Input 1D sequence of ordinates.
    w : np.ndarray, optional
        Weights for each point. If None, default weight is 1.0.

    Returns
    -------
    z : np.ndarray
        Monotonic non-decreasing projected values.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    n = len(y_arr)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return y_arr.copy()

    if w is None:
        w_in = np.ones(n, dtype=np.float64)
    else:
        w_in = np.asarray(w, dtype=np.float64)

    z = np.zeros(n, dtype=np.float64)
    ymin = float(np.min(y_arr))
    ymax = float(np.max(y_arr))
    range_y = max(1.0, ymax - ymin)
    abs_tol = 1e-12 * range_y
    rel_tol = 1e-12
    eps_w = 1e-15

    # Each block in S: [L, R, sw, sy]
    S: list[list[float]] = []

    for i in range(n):
        yi = float(y_arr[i])
        wi = float(w_in[i]) if np.isfinite(w_in[i]) else 1.0
        if not np.isfinite(yi):
            yi = float(z[i - 1]) if i > 0 else 0.0
        wi = max(wi, 0.0)
        if wi == 0.0:
            wi = eps_w

        # S.push_back({i, i, wi, wi * yi})
        S.append([float(i), float(i), wi, wi * yi])

        while len(S) >= 2:
            B2 = S[-1]
            B1 = S[-2]
            m1 = B1[3] / B1[2]
            m2 = B2[3] / B2[2]

            eps_pav = abs_tol + rel_tol * max(abs(m1), abs(m2))
            if m1 <= m2 + eps_pav:
                break

            B1[2] += B2[2]  # sw
            B1[3] += B2[3]  # sy
            B1[1] = B2[1]   # R
            S.pop()

        z[i] = yi

    for B in S:
        m = B[3] / B[2]
        L = int(B[0])
        R = int(B[1])
        z[L:R + 1] = m

    return z


def smooth_isotone(
    x: np.ndarray,
    y: np.ndarray,
    mu: float = 0.01,
    maxit: int = 500,
    tol: float = 1e-9,
) -> np.ndarray:
    """1D Laplacian smoothing combined with PAV projection and origin pinning (0, 0).

    Matches C++ implementation in cpp_table_mat_spline_fit.cpp:88-126.

    Parameters
    ----------
    x : np.ndarray
        Abscissas.
    y : np.ndarray
        Input ordinates.
    mu : float, optional
        Smoothing regularization parameter (lambda, default 0.01).
    maxit : int, optional
        Maximum iterations (default 500).
    tol : float, optional
        Convergence tolerance on squared change (default 1e-9).

    Returns
    -------
    z : np.ndarray
        Smoothed monotonic ordinates with z(0) = 0 if x contains 0.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_in = np.asarray(y, dtype=np.float64)
    n = len(x_arr)
    if n <= 1:
        return y_in.copy()

    w = np.ones(n, dtype=np.float64)
    z = y_in.copy()

    # Find index i0 where abs(x[i]) < 1e-15
    i0 = -1
    for i in range(n):
        if abs(x_arr[i]) < 1e-15:
            i0 = i
            break
    if i0 >= 0:
        z[i0] = 0.0

    eta = 1.0 / (1.0 + 4.0 * mu)

    for _ in range(maxit):
        zprev = z.copy()
        g = z - y_in

        # 1D Laplacian update:
        g[0] += mu * (z[0] - z[1])
        if n > 2:
            g[1:-1] += mu * (2.0 * z[1:-1] - z[:-2] - z[2:])
        g[-1] += mu * (z[-1] - z[-2])

        z -= eta * g

        if i0 >= 0:
            z[i0] = 0.0

        z = isotone_project_pav(z, w)

        diff = float(np.sum((z - zprev) ** 2))
        if diff < tol:
            break

    return z


def pchip_slopes(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson PCHIP slopes calculation preserving monotonicity.

    Matches C++ implementation in cpp_table_mat_spline_fit.cpp:129-151.

    Parameters
    ----------
    x : np.ndarray
        Monotonically increasing abscissas.
    z : np.ndarray
        Ordinates.

    Returns
    -------
    m : np.ndarray
        PCHIP derivatives at each point.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64)
    n = len(x_arr)
    m = np.zeros(n, dtype=np.float64)
    if n <= 1:
        return m

    h = x_arr[1:] - x_arr[:-1]
    h_safe = np.where(np.abs(h) < 1e-15, 1e-15, h)
    d = (z_arr[1:] - z_arr[:-1]) / h_safe

    m[0] = d[0]
    m[-1] = d[-1]

    for i in range(1, n - 1):
        if d[i - 1] * d[i] <= 0.0:
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            denom = (w1 / d[i - 1]) + (w2 / d[i])
            if abs(denom) > 1e-15:
                m[i] = (w1 + w2) / denom
            else:
                m[i] = 0.0

    return m


def pchip_eval(
    x: np.ndarray,
    z: np.ndarray,
    m: np.ndarray,
    xi: float | np.ndarray,
) -> float | np.ndarray:
    """Hermite cubic evaluation of PCHIP interpolant at xi with flat boundary extrapolation.

    Matches C++ implementation in cpp_table_mat_spline_fit.cpp:154-173.

    Parameters
    ----------
    x : np.ndarray
        Grid abscissas.
    z : np.ndarray
        Grid values.
    m : np.ndarray
        Grid slopes.
    xi : float or np.ndarray
        Evaluation points.

    Returns
    -------
    yi : float or np.ndarray
        Interpolated values.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64)
    m_arr = np.asarray(m, dtype=np.float64)
    is_scalar = np.isscalar(xi)
    xi_arr = np.atleast_1d(np.asarray(xi, dtype=np.float64))

    n = len(x_arr)
    res = np.zeros_like(xi_arr)

    for idx, val in enumerate(xi_arr):
        if val <= x_arr[0]:
            res[idx] = z_arr[0]
            continue
        if val >= x_arr[-1]:
            res[idx] = z_arr[-1]
            continue

        k = int(np.searchsorted(x_arr, val, side="right")) - 1
        k = max(0, min(n - 2, k))
        h = x_arr[k + 1] - x_arr[k]
        t = (val - x_arr[k]) / h
        t2 = t * t
        t3 = t2 * t

        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = (t3 - 2.0 * t2 + t) * h
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = (t3 - t2) * h

        res[idx] = h00 * z_arr[k] + h10 * m_arr[k] + h01 * z_arr[k + 1] + h11 * m_arr[k + 1]

    if is_scalar:
        return float(res[0])
    return res


def _ensure_origin_point(
    x: np.ndarray,
    y: np.ndarray,
    xtol: float = 1e-15,
) -> tuple[np.ndarray, np.ndarray]:
    """Ensure (0, 0) point is present in sorted x, y data matching cpp_table_mat_spline_fit.cpp:175-194."""
    x_list = list(x)
    y_list = list(y)
    n = len(x_list)
    if n == 0:
        return np.array([0.0]), np.array([0.0])

    for i in range(n):
        if abs(x_list[i]) < xtol:
            x_list[i] = 0.0
            y_list[i] = 0.0
            return np.array(x_list, dtype=np.float64), np.array(y_list, dtype=np.float64)

    pos = int(np.searchsorted(x_list, 0.0, side="left"))
    x_list.insert(pos, 0.0)
    y_list.insert(pos, 0.0)
    return np.array(x_list, dtype=np.float64), np.array(y_list, dtype=np.float64)


def _build_uniform_grid_with_zero(
    xmin: float,
    xmax: float,
    nout: int,
) -> tuple[float, float, int]:
    """Construct uniform grid containing 0.0 matching cpp_table_mat_spline_fit.cpp:196-227."""
    if nout < 2:
        nout = 2

    if xmin >= 0.0:
        return 0.0, xmax, 0
    if xmax <= 0.0:
        return xmin, 0.0, nout - 1

    ratio = (-xmin) / (xmax - xmin)
    i0 = int(round(ratio * float(nout - 1)))
    i0 = max(1, min(nout - 2, i0))

    h1 = (-xmin) / float(i0)
    h2 = xmax / float(nout - 1 - i0)
    h = max(h1, h2)

    xmin_new = -float(i0) * h
    xmax_new = float(nout - 1 - i0) * h
    return xmin_new, xmax_new, i0


def table_mat_spline_fit(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    nout: int = 300,
    lam: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Full curve smoothing, monotonicity enforcement and spline resample pipeline.

    Matches C++ interface in cpp_table_mat_spline_fit.cpp:233-273.

    Parameters
    ----------
    x_raw : np.ndarray
        Raw input strain / stretch abscissas.
    y_raw : np.ndarray
        Raw input stress ordinates.
    nout : int, optional
        Number of output grid points (default 300).
    lam : float, optional
        Smoothing regularization parameter (lambda, default 1e-3).

    Returns
    -------
    x_out : np.ndarray
        Resampled uniform grid including exact 0.0.
    y_out : np.ndarray
        Monotonic smoothed stress values evaluated at x_out.
    """
    x_arr = np.asarray(x_raw, dtype=np.float64)
    y_arr = np.asarray(y_raw, dtype=np.float64)
    n = len(x_arr)
    if n == 0:
        return np.array([0.0]), np.array([0.0])
    if n == 1:
        return np.full(nout, x_arr[0]), np.full(nout, y_arr[0])

    x_orig, y_orig = _ensure_origin_point(x_arr, y_arr)
    z = smooth_isotone(x_orig, y_orig, mu=max(0.0, float(lam)))
    m = pchip_slopes(x_orig, z)

    xmin = float(x_orig[0])
    xmax = float(x_orig[-1])
    if nout < 2:
        nout = 2

    xmin_new, xmax_new, i0 = _build_uniform_grid_with_zero(xmin, xmax, nout)
    h = (xmax_new - xmin_new) / float(nout - 1)

    x_out = np.zeros(nout, dtype=np.float64)
    for i in range(nout):
        xi = xmin_new + h * float(i)
        if i == i0:
            xi = 0.0
        x_out[i] = xi

    y_out = pchip_eval(x_orig, z, m, x_out)
    return x_out, y_out


# ============================================================================
# 2. 1D / 2D Table Interpolation
#    Reference: table_mat_vinterp.F:144-430
# ============================================================================

@dataclass
class TableData:
    """Structure representing 1D or 2D OpenRadioss material table (TABLE_4D_)."""
    ndim: int = 1
    x1: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    x2: Optional[np.ndarray] = None
    y1d: Optional[np.ndarray] = None
    y2d: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.ndim == 1 and self.y1d is None and self.y2d is not None:
            self.y1d = self.y2d[:, 0]
        elif self.ndim == 2 and self.y2d is None and self.y1d is not None:
            self.y2d = self.y1d[:, None]
        elif self.y1d is None and self.y2d is None:
            self.y1d = np.array([0.0, 100.0])


def table_mat_vinterp_eval(
    table_data: Any,
    stretch: Union[float, np.ndarray],
    rate: Union[float, np.ndarray] = 0.0,
    opt_extrapolate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform vectorized 1D or 2D table interpolation with stretch slopes.

    Matches Fortran implementation in table_mat_vinterp.F:398-427.

    Parameters
    ----------
    table_data : TableData, tuple, list, or dict
        Table containing abscissas and values.
    stretch : float or np.ndarray
        Stretch abscissa values (NEL,).
    rate : float or np.ndarray, optional
        Strain rate abscissa values (NEL,) for 2D table (default 0.0).
    opt_extrapolate : bool, optional
        Whether to linearly extrapolate outside bounds (default True).

    Returns
    -------
    yy : np.ndarray
        Interpolated stress values of shape (NEL,).
    dydx : np.ndarray
        Derivative of stress with respect to stretch of shape (NEL,).
    """
    if hasattr(table_data, "ndim"):
        ndim = table_data.ndim
        x1 = np.asarray(table_data.x1, dtype=np.float64)
        x2 = np.asarray(table_data.x2, dtype=np.float64) if table_data.x2 is not None else None
        y1d = np.asarray(table_data.y1d, dtype=np.float64) if table_data.y1d is not None else None
        y2d = np.asarray(table_data.y2d, dtype=np.float64) if table_data.y2d is not None else None
    elif isinstance(table_data, dict):
        ndim = table_data.get("ndim", 1)
        x1 = np.asarray(table_data["x1"], dtype=np.float64)
        x2 = np.asarray(table_data["x2"], dtype=np.float64) if "x2" in table_data else None
        y1d = np.asarray(table_data["y1d"], dtype=np.float64) if "y1d" in table_data else None
        y2d = np.asarray(table_data["y2d"], dtype=np.float64) if "y2d" in table_data else None
    elif isinstance(table_data, (tuple, list)):
        x1 = np.asarray(table_data[0], dtype=np.float64)
        y_raw = np.asarray(table_data[1], dtype=np.float64)
        if y_raw.ndim == 1:
            ndim = 1
            x2 = None
            y1d = y_raw
            y2d = None
        else:
            ndim = 2
            x2 = np.asarray(table_data[2], dtype=np.float64) if len(table_data) > 2 else np.array([0.0])
            y1d = None
            y2d = y_raw
    else:
        raise ValueError(f"Unsupported table_data type: {type(table_data)}")

    stretch_arr = np.atleast_1d(np.asarray(stretch, dtype=np.float64))
    nel = len(stretch_arr)
    rate_arr = np.atleast_1d(np.asarray(rate, dtype=np.float64))
    if len(rate_arr) == 1 and nel > 1:
        rate_arr = np.full(nel, rate_arr[0], dtype=np.float64)

    # 1D Table Interpolation (table_mat_vinterp.F:416-425)
    if ndim == 1 or y2d is None:
        if y1d is None:
            raise ValueError("y1d is required for 1D table interpolation")
        npt = len(x1)
        i1 = np.searchsorted(x1, stretch_arr, side="right") - 1
        i1 = np.clip(i1, 0, npt - 2)
        i2 = i1 + 1

        dx = x1[i2] - x1[i1]
        dx_safe = np.where(np.abs(dx) < 1e-15, 1e-15, dx)
        fac = (x1[i2] - stretch_arr) / dx_safe
        if not opt_extrapolate:
            fac = np.clip(fac, 0.0, 1.0)

        alpha = fac
        alphai = 1.0 - alpha
        yy = alpha * y1d[i1] + alphai * y1d[i2]
        dydx = (y1d[i2] - y1d[i1]) / dx_safe

        if not opt_extrapolate:
            out_mask = (stretch_arr < x1[0]) | (stretch_arr > x1[-1])
            dydx = np.where(out_mask, 0.0, dydx)

        return yy, dydx

    # 2D Table Interpolation (table_mat_vinterp.F:398-414)
    npt1 = len(x1)
    if x2 is None:
        x2 = np.array([0.0], dtype=np.float64)
    npt2 = len(x2)

    i1 = np.searchsorted(x1, stretch_arr, side="right") - 1
    i1 = np.clip(i1, 0, npt1 - 2)
    i2 = i1 + 1
    dx1 = x1[i2] - x1[i1]
    dx1_safe = np.where(np.abs(dx1) < 1e-15, 1e-15, dx1)
    fac1 = (x1[i2] - stretch_arr) / dx1_safe
    if not opt_extrapolate:
        fac1 = np.clip(fac1, 0.0, 1.0)
    alpha = fac1
    alphai = 1.0 - alpha

    if npt2 > 1:
        j1 = np.searchsorted(x2, rate_arr, side="right") - 1
        j1 = np.clip(j1, 0, npt2 - 2)
        j2 = j1 + 1
        dx2 = x2[j2] - x2[j1]
        dx2_safe = np.where(np.abs(dx2) < 1e-15, 1e-15, dx2)
        fac2 = (x2[j2] - rate_arr) / dx2_safe
        if not opt_extrapolate:
            fac2 = np.clip(fac2, 0.0, 1.0)
        beta = fac2
        betai = 1.0 - beta
    else:
        j1 = np.zeros(nel, dtype=int)
        j2 = np.zeros(nel, dtype=int)
        beta = np.ones(nel, dtype=np.float64)
        betai = np.zeros(nel, dtype=np.float64)

    yy = beta * (alpha * y2d[i1, j1] + alphai * y2d[i2, j1]) + \
         betai * (alpha * y2d[i1, j2] + alphai * y2d[i2, j2])

    dydx = (beta * (y2d[i2, j1] - y2d[i1, j1]) + \
            betai * (y2d[i2, j2] - y2d[i1, j2])) / dx1_safe

    if not opt_extrapolate:
        out_mask = (stretch_arr < x1[0]) | (stretch_arr > x1[-1])
        dydx = np.where(out_mask, 0.0, dydx)

    return yy, dydx


# ============================================================================
# Material Parameter Container for LAW88
# ============================================================================

@dataclass
class MatparamLaw88:
    """Material parameters container matching matparam_struct_ for LAW88."""
    bulk: float = 1e3
    shear: float = 1e2
    nu: float = 0.495
    young: float = 3e2
    rho0: float = 1e-9
    rho: float = 1e-9
    iparam: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=int))
    uparam: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=np.float64))
    table: list[TableData] = field(default_factory=list)

    @property
    def E(self) -> float:
        return self.young

    @property
    def G(self) -> float:
        return self.shear

    @property
    def K(self) -> float:
        return self.bulk

    @classmethod
    def from_dict_or_obj(cls, obj: Any) -> MatparamLaw88:
        if isinstance(obj, MatparamLaw88):
            return obj
        p: dict[str, Any] = {}
        if isinstance(obj, dict):
            p.update(obj)
        elif hasattr(obj, "params") and isinstance(obj.params, dict):
            p.update(obj.params)

        def _get(keys: tuple[str, ...], default: Any) -> Any:
            for k in keys:
                if k in p and p[k] is not None:
                    return p[k]
                if hasattr(obj, k):
                    val = getattr(obj, k)
                    if val is not None and not (isinstance(val, float) and val == 0.0 and default != 0.0):
                        return val
            return default

        bulk = float(_get(("bulk", "K", "LAW88_K"), 1e3))
        shear = float(_get(("shear", "G", "LAW88_G"), 1e2))
        nu = float(_get(("nu", "LAW88_Nu"), 0.495))
        young = float(_get(("young", "E", "MAT_E", "LAW88_E"), 3e2))
        rho0 = float(_get(("rho0", "rho", "MAT_RHO"), 1e-9))
        rho = float(_get(("rho", "rho0", "MAT_RHO"), rho0))

        iparam_raw = _get(("iparam",), None)
        if iparam_raw is not None:
            iparam = np.asarray(iparam_raw, dtype=int)
        else:
            iparam = np.zeros(6, dtype=int)

        uparam_raw = _get(("uparam",), None)
        if uparam_raw is not None:
            uparam = np.asarray(uparam_raw, dtype=np.float64)
        else:
            uparam = np.zeros(9, dtype=np.float64)

        if np.all(iparam == 0):
            itens = int(_get(("tension", "LAW88_Tension", "itens"), 0))
            ifunc_unload = int(_get(("ifunc_unload", "LAW88_fct_IDunL"), 0))
            hys = float(_get(("hys", "LAW88_Hys"), 0.0))
            iunl_for = int(_get(("iunl_for",), 1 if ifunc_unload > 0 else (2 if hys != 0.0 else 0)))
            nl = int(_get(("nl", "LAW88_NL", "nload"), 0))
            if nl == 0:
                func_load_list = _get(("func_load_list", "LAW88_arr1"), None)
                if func_load_list:
                    nl = len(func_load_list)
            rtype = int(_get(("rtype", "LAW88_RTYPE"), 0))
            failip = int(_get(("failip", "LAW88_FAILIP"), 0))
            nv_base = int(_get(("nv_base",), 12))
            iparam = np.array([itens, iunl_for, nl, rtype, failip, nv_base], dtype=int)

        if np.all(uparam == 0):
            hys = float(_get(("hys", "LAW88_Hys"), 0.0))
            shape = float(_get(("shape", "LAW88_Shape"), 1.0))
            gdamp = float(_get(("gdamp", "LAW88_GDAMP"), 0.0))
            sigf = float(_get(("sigf", "LAW88_SIGF"), 0.0))
            if gdamp == 0.0 and sigf > 0.0:
                gdamp = float(_get(("LAW88_G", "g"), 0.0))
            kfail = float(_get(("kfail", "LAW88_KFAIL"), 0.0))
            gam1 = float(_get(("gam1", "LAW88_GAM1"), 0.0))
            gam2 = float(_get(("gam2", "LAW88_GAM2"), 0.0))
            eh = float(_get(("eh", "LAW88_EH"), 0.0))
            beta = float(_get(("beta", "LAW88_BETA"), 0.0))
            uparam = np.array([hys, shape, gdamp, sigf, kfail, gam1, gam2, eh, beta], dtype=np.float64)

        table = _get(("table",), [])
        if not table and hasattr(obj, "table"):
            table = obj.table

        return cls(
            bulk=bulk,
            shear=shear,
            nu=nu,
            young=young,
            rho0=rho0,
            rho=rho,
            iparam=iparam,
            uparam=uparam,
            table=table,
        )


# ============================================================================
# 3. 3D Solid Continuum Material Kernel (sigeps88_solid)
#    Reference: sigeps88.F90:46-1006
# ============================================================================

def sigeps88_solid(
    nel: int,
    matparam: Any,
    uvar: np.ndarray,
    tstep: float,
    tt: float,
    rho0: np.ndarray,
    rho: np.ndarray,
    soundsp: np.ndarray,
    off: np.ndarray,
    ismstr: int,
    israte: int,
    epsxx: np.ndarray,
    epsyy: np.ndarray,
    epszz: np.ndarray,
    epsxy: np.ndarray,
    epsyz: np.ndarray,
    epszx: np.ndarray,
    depsxx: np.ndarray,
    depsyy: np.ndarray,
    depszz: np.ndarray,
    depsxy: np.ndarray,
    depsyz: np.ndarray,
    depszx: np.ndarray,
    epspxx: np.ndarray,
    epspyy: np.ndarray,
    epspzz: np.ndarray,
    epspxy: np.ndarray,
    epspyz: np.ndarray,
    epspzx: np.ndarray,
    sigoxx: np.ndarray,
    sigoyy: np.ndarray,
    sigozz: np.ndarray,
    sigoxy: np.ndarray,
    sigoyz: np.ndarray,
    sigozx: np.ndarray,
    signxx: np.ndarray,
    signyy: np.ndarray,
    signzz: np.ndarray,
    signxy: np.ndarray,
    signyz: np.ndarray,
    signzx: np.ndarray,
    asrate: float,
    et: np.ndarray,
    offg: np.ndarray,
    epsd: np.ndarray,
    iresp: int,
    nvartmp: int,
    vartmp: np.ndarray,
    dmg: np.ndarray,
    ngl: np.ndarray,
    npg: int,
) -> None:
    """Constitutive stress update for 3D continuum solid elements under /MAT/LAW88.

    Matches sigeps88.F90:46-1006. Mutates signxx..zz..zx, soundsp, et, uvar, off, dmg in-place.
    """
    mp = MatparamLaw88.from_dict_or_obj(matparam)

    itens = int(mp.iparam[0])
    iunl_for = int(mp.iparam[1])
    nload = int(mp.iparam[2])
    rtype = int(mp.iparam[3])
    failip = min(int(mp.iparam[4]), npg)
    nv_base = int(mp.iparam[5]) if len(mp.iparam) > 5 and mp.iparam[5] > 0 else 12

    rbulk = np.full(nel, mp.bulk, dtype=np.float64)
    nu = mp.nu
    if nu >= 0.49:
        nu = 0.5
    gs = mp.shear
    hys = float(mp.uparam[0])
    shape = float(mp.uparam[1]) if mp.uparam[1] != 0.0 else 1.0
    gdamp = float(mp.uparam[2])
    sigf = float(mp.uparam[3])
    kfail = float(mp.uparam[4])
    gam1 = float(mp.uparam[5])
    gam2 = float(mp.uparam[6])
    eh = float(mp.uparam[7])
    beta = float(mp.uparam[8])

    emax = uvar[:, 0].copy()
    ecurent = uvar[:, 1].copy()
    loadflg = uvar[:, 4].copy()
    loadflg_old = loadflg.copy()

    off[:] = np.where(off < 1.0, off * _FOUR_OVER_5, off)
    off[:] = np.where(off < _EM01, 0.0, off)

    sigdxx = uvar[:, 5].copy()
    sigdyy = uvar[:, 6].copy()
    sigdzz = uvar[:, 7].copy()
    sigdxy = uvar[:, 8].copy()
    sigdyz = uvar[:, 9].copy()
    sigdzx = uvar[:, 10].copy()

    dmg_fac = 0.5 * (1.0 / np.maximum(1.0 - dmg, _EM20))
    deint0 = dmg_fac * (
        (sigoxx - sigdxx) * depsxx +
        (sigoyy - sigdyy) * depsyy +
        (sigozz - sigdzz) * depszz +
        (sigoxy - sigdxy) * depsxy +
        (sigoyz - sigdyz) * depsyz +
        (sigozx - sigdzx) * depszx
    )

    if iunl_for == 1:
        for j in range(3):
            mask_init = uvar[:, nv_base + 9 + j] <= 0.0
            uvar[mask_init, nv_base + 9 + j] = 1.0
        mask_r = uvar[:, nv_base + 12] <= 0.0
        uvar[mask_r, nv_base + 12] = 1.0

    eps_mat = np.zeros((nel, 3, 3), dtype=np.float64)
    eps_mat[:, 0, 0] = epsxx
    eps_mat[:, 1, 1] = epsyy
    eps_mat[:, 2, 2] = epszz
    eps_mat[:, 0, 1] = eps_mat[:, 1, 0] = 0.5 * epsxy
    eps_mat[:, 1, 2] = eps_mat[:, 2, 1] = 0.5 * epsyz
    eps_mat[:, 0, 2] = eps_mat[:, 2, 0] = 0.5 * epszx

    evv, dirprv = np.linalg.eigh(eps_mat)
    sort_idx = np.argsort(evv, axis=-1)[:, ::-1]
    evv = np.take_along_axis(evv, sort_idx, axis=-1)
    dirprv = np.take_along_axis(dirprv, sort_idx[:, None, :], axis=-1)

    epsp_mat = np.zeros((nel, 3, 3), dtype=np.float64)
    epsp_mat[:, 0, 0] = epspxx
    epsp_mat[:, 1, 1] = epspyy
    epsp_mat[:, 2, 2] = epspzz
    epsp_mat[:, 0, 1] = epsp_mat[:, 1, 0] = 0.5 * epspxy
    epsp_mat[:, 1, 2] = epsp_mat[:, 2, 1] = 0.5 * epspyz
    epsp_mat[:, 0, 2] = epsp_mat[:, 2, 0] = 0.5 * epspzx

    epsp_rot = np.einsum("nia,nab,nbj->nij", dirprv, epsp_mat, dirprv)
    evvp = np.zeros((nel, 3), dtype=np.float64)
    evvp[:, 0] = epsp_rot[:, 0, 0]
    evvp[:, 1] = epsp_rot[:, 1, 1]
    evvp[:, 2] = epsp_rot[:, 2, 2]

    lam = np.zeros((nel, 3), dtype=np.float64)
    if ismstr in (0, 2, 4):
        lam = np.exp(evv)
    elif ismstr in (10, 12):
        for i in range(nel):
            if offg[i] <= 1.0:
                lam[i] = np.sqrt(np.maximum(evv[i] + 1.0, _EM20))
            else:
                lam[i] = evv[i] + 1.0
    else:
        lam = evv + 1.0

    ee = lam - 1.0

    eep = np.zeros((nel, 3), dtype=np.float64)
    if rtype == 1:
        if ismstr in (0, 2, 4):
            eep = np.exp(evv) * evvp
        elif ismstr in (10, 12):
            for i in range(nel):
                if offg[i] <= 1.0:
                    eep[i] = 0.5 * (1.0 / np.sqrt(np.maximum(evv[i] + 1.0, _EM20))) * evvp[i]
                else:
                    eep[i] = evvp[i]
        else:
            eep = evvp.copy()
        erate = np.sqrt(np.sum(eep ** 2, axis=1))
    else:
        erate = np.sqrt(np.sum(evvp ** 2, axis=1))

    if israte > 0:
        epsd[:] = asrate * erate + (1.0 - asrate) * epsd
    else:
        epsd[:] = erate

    rv = lam[:, 0] * lam[:, 1] * lam[:, 2]
    rv_safe = np.maximum(rv, _EM20)
    rv_mth = np.exp(-_THIRD * np.log(rv_safe))
    if (nu > 0.0) and (nu < 0.49):
        rv_mth[:] = 1.0

    lam[:, 0] *= rv_mth
    lam[:, 1] *= rv_mth
    lam[:, 2] *= rv_mth

    table_load = mp.table[0] if len(mp.table) > 0 else TableData(x1=np.array([0.0, 1.0]), y1d=np.array([0.0, max(float(mp.young), float(mp.shear), 1.0)]))

    f = np.zeros((nel, 3), dtype=np.float64)
    dfdlam = np.zeros((nel, 3), dtype=np.float64)
    dgdlam_saved = np.zeros((nel, 3), dtype=np.float64)

    for j in range(3):
        lam_j = lam[:, j]
        rate_j = epsd.copy()
        if itens == 0:
            rate_j = np.where(rv > 1.0, 0.0, rate_j)

        g_j, dg_j = table_mat_vinterp_eval(table_load, lam_j, rate_j, opt_extrapolate=True)
        dgdlam_saved[:, j] = dg_j

        f[:, j] = lam_j * g_j
        dfdlam[:, j] = g_j + lam_j * dg_j

        for n in range(1, 7):
            power_n = (-nu) ** n
            lam_p = lam_j ** power_n
            g_sqr, dg_sqr = table_mat_vinterp_eval(table_load, lam_p, rate_j, opt_extrapolate=True)

            f[:, j] += lam_p * g_sqr
            dfdlam[:, j] += power_n * (lam_j ** (power_n - 1.0)) * (g_sqr + lam_p * dg_sqr)

    p = np.zeros(nel, dtype=np.float64)
    fJ = np.zeros(nel, dtype=np.float64)
    dfJdx = np.zeros(nel, dtype=np.float64)

    if beta > 0.0:
        ldav = epspxx + epspyy + epspzz
        exp_beta = math.exp(-beta * tstep)
        p = uvar[:, 11] * exp_beta + rbulk * ldav * ((1.0 - exp_beta) / beta)
        uvar[:, 11] = p
    elif (nu > 0.0) and (nu < 0.49):
        xfoam = rv_safe ** (-nu / (1.0 - 2.0 * nu))
        rate_foam = epsd.copy()
        gJ, dgJ = table_mat_vinterp_eval(table_load, xfoam, rate_foam, opt_extrapolate=True)
        fJ = gJ * xfoam
        dfJdx = gJ + xfoam * dgJ

        for n in range(1, 7):
            power_n = (-nu) ** n
            xf_p = xfoam ** power_n
            gJsqr, dgJsqr = table_mat_vinterp_eval(table_load, xf_p, rate_foam, opt_extrapolate=True)
            fJ += xf_p * gJsqr
            dfJdx += power_n * (xfoam ** (power_n - 1.0)) * (gJsqr + xf_p * dgJsqr)

        rbulk = (-nu / (1.0 - 2.0 * nu)) * xfoam * dfJdx
        rbulk = np.maximum(rbulk, _EM12)
    else:
        p = rbulk * (rv - 1.0)

    t = np.zeros((nel, 3), dtype=np.float64)
    if (nu <= 0.0) or (nu >= 0.49):
        t[:, 0] = (_TWO_THIRD * f[:, 0] - _THIRD * (f[:, 1] + f[:, 2]) + p) / rv_safe
        t[:, 1] = (_TWO_THIRD * f[:, 1] - _THIRD * (f[:, 0] + f[:, 2]) + p) / rv_safe
        t[:, 2] = (_TWO_THIRD * f[:, 2] - _THIRD * (f[:, 0] + f[:, 1]) + p) / rv_safe
    else:
        t[:, 0] = (f[:, 0] - fJ) / rv_safe
        t[:, 1] = (f[:, 1] - fJ) / rv_safe
        t[:, 2] = (f[:, 2] - fJ) / rv_safe

    epseq = np.sqrt(np.sum(ee ** 2, axis=1))
    deint = deint0 + 0.5 * (
        t[:, 0] * evvp[:, 0] +
        t[:, 1] * evvp[:, 1] +
        t[:, 2] * evvp[:, 2]
    ) * tstep

    ecurent = np.maximum(_EM20, ecurent + deint)
    emax = np.maximum(emax, ecurent)

    for i in range(nel):
        if off[i] == 1.0:
            if loadflg[i] == -1.0:
                if deint[i] / max(ecurent[i], _EM20) >= _EM07:
                    loadflg[i] = 1.0
                    emax[i] = ecurent[i]
                else:
                    loadflg[i] = -1.0
            else:
                if deint[i] / max(emax[i], _EM20) >= 0.0:
                    loadflg[i] = 1.0
                else:
                    loadflg[i] = -1.0

    if iunl_for == 1:
        for i in range(nel):
            if loadflg_old[i] == 1.0 and loadflg[i] == -1.0:
                for j in range(3):
                    uvar[i, nv_base + j] = lam[i, j]
                    if abs(uvar[i, nv_base + 3 + j]) < _EM10:
                        uvar[i, nv_base + 3 + j] = 1.0
                    uvar[i, nv_base + 6 + j] = f[i, j]
                    uvar[i, nv_base + 9 + j] = 1.0
                uvar[i, nv_base + 12] = 1.0
            elif loadflg_old[i] == -1.0 and loadflg[i] == 1.0:
                for j in range(3):
                    uvar[i, nv_base + 3 + j] = uvar[i, nv_base + j]
                    uvar[i, nv_base + j] = lam[i, j]
                    uvar[i, nv_base + 6 + j] = f[i, j]
                    uvar[i, nv_base + 9 + j] = 1.0
                uvar[i, nv_base + 12] = 1.0

        for i in range(nel):
            if loadflg[i] == -1.0:
                amax = 0.0
                for j in range(3):
                    lam_r = uvar[i, nv_base + j]
                    lam_tg = uvar[i, nv_base + 3 + j]
                    denom = max(abs(lam_r - lam_tg), _EM20)
                    xhat = np.clip((lam[i, j] - lam_tg) / denom, -1.0, 1.0)
                    prev = uvar[i, nv_base + 9 + j]
                    if prev <= 0.0:
                        prev = 1.0
                    xmag = min(abs(xhat), prev)
                    uvar[i, nv_base + 9 + j] = xmag
                    amax = max(amax, xmag)

                if amax <= _EM03:
                    loadflg[i] = 1.0
                    for j in range(3):
                        uvar[i, nv_base + 3 + j] = uvar[i, nv_base + j]
                        uvar[i, nv_base + j] = lam[i, j]
                        uvar[i, nv_base + 6 + j] = f[i, j]
                        uvar[i, nv_base + 9 + j] = 1.0
                    uvar[i, nv_base + 12] = 1.0

        unl_indices = [i for i in range(nel) if off[i] == 1.0 and loadflg[i] == -1.0]
        if unl_indices and len(mp.table) >= 3:
            table_unl = mp.table[1]
            table_norm = mp.table[2]

            for i in unl_indices:
                amax = -1.0
                xhat_dom = 0.0
                jdom = 0

                for j in range(3):
                    lam_r = uvar[i, nv_base + j]
                    lam_tg = uvar[i, nv_base + 3 + j]
                    denom = max(abs(lam_r - lam_tg), _EM20)
                    xhat = np.clip((lam[i, j] - lam_tg) / denom, -1.0, 1.0)
                    prev = uvar[i, nv_base + 9 + j]
                    if prev <= 0.0:
                        prev = 1.0
                    xmag = min(abs(xhat), prev)
                    uvar[i, nv_base + 9 + j] = xmag
                    if xmag > amax:
                        amax = xmag
                        jdom = j
                        xhat_dom = math.copysign(xmag, xhat)

                xvec_1 = math.copysign(min(abs(xhat_dom), 1.0 - _EM08), xhat_dom)
                xvec_2 = 0.0
                if itens == 0 and rv[i] < 1.0:
                    xvec_2 = epsd[i]
                elif itens == 1:
                    xvec_2 = epsd[i]

                g_norm, _ = table_mat_vinterp_eval(table_norm, xvec_1, xvec_2, opt_extrapolate=True)
                gunl, _ = table_mat_vinterp_eval(table_unl, xvec_1, xvec_2, opt_extrapolate=True)

                ratioR_val = 0.0
                if abs(g_norm[0]) > _EM20:
                    ratioR_val = np.clip(gunl[0] / g_norm[0], 0.0, 1.0)

                amax_val = np.clip(uvar[i, nv_base + 9 + jdom], 0.0, 1.0)
                Rblend = (1.0 - amax_val ** 3) * uvar[i, nv_base + 12] + (amax_val ** 3) * ratioR_val
                ratioR_val = float(np.clip(Rblend, 0.0, 1.0))
                uvar[i, nv_base + 12] = ratioR_val

                f[i, :] *= ratioR_val
                dfdlam[i, :] *= ratioR_val

                if (nu <= 0.0) or (nu >= 0.49):
                    t[i, 0] = (_TWO_THIRD * f[i, 0] - _THIRD * (f[i, 1] + f[i, 2]) + p[i]) / rv_safe[i]
                    t[i, 1] = (_TWO_THIRD * f[i, 1] - _THIRD * (f[i, 0] + f[i, 2]) + p[i]) / rv_safe[i]
                    t[i, 2] = (_TWO_THIRD * f[i, 2] - _THIRD * (f[i, 0] + f[i, 1]) + p[i]) / rv_safe[i]
                else:
                    t[i, 0] = (f[i, 0] - fJ[i] * ratioR_val) / rv_safe[i]
                    t[i, 1] = (f[i, 1] - fJ[i] * ratioR_val) / rv_safe[i]
                    t[i, 2] = (f[i, 2] - fJ[i] * ratioR_val) / rv_safe[i]

    elif iunl_for == 2:
        for i in range(nel):
            if off[i] == 1.0 and loadflg[i] == -1.0:
                e_ratio = ecurent[i] / max(emax[i], _EM20)
                ratioR = 1.0 - (1.0 - hys) * (1.0 - (e_ratio ** shape))
                t[i, :] *= ratioR

    signxx[:] = dirprv[:, 0, 0] ** 2 * t[:, 0] + dirprv[:, 0, 1] ** 2 * t[:, 1] + dirprv[:, 0, 2] ** 2 * t[:, 2]
    signyy[:] = dirprv[:, 1, 0] ** 2 * t[:, 0] + dirprv[:, 1, 1] ** 2 * t[:, 1] + dirprv[:, 1, 2] ** 2 * t[:, 2]
    signzz[:] = dirprv[:, 2, 0] ** 2 * t[:, 0] + dirprv[:, 2, 1] ** 2 * t[:, 1] + dirprv[:, 2, 2] ** 2 * t[:, 2]
    signxy[:] = dirprv[:, 0, 0] * dirprv[:, 1, 0] * t[:, 0] + dirprv[:, 0, 1] * dirprv[:, 1, 1] * t[:, 1] + dirprv[:, 0, 2] * dirprv[:, 1, 2] * t[:, 2]
    signyz[:] = dirprv[:, 1, 0] * dirprv[:, 2, 0] * t[:, 0] + dirprv[:, 1, 1] * dirprv[:, 2, 1] * t[:, 1] + dirprv[:, 1, 2] * dirprv[:, 2, 2] * t[:, 2]
    signzx[:] = dirprv[:, 2, 0] * dirprv[:, 0, 0] * t[:, 0] + dirprv[:, 2, 1] * dirprv[:, 0, 1] * t[:, 1] + dirprv[:, 2, 2] * dirprv[:, 0, 2] * t[:, 2]

    gmax = np.full(nel, gs, dtype=np.float64)
    for i in range(nel):
        for j in range(3):
            if iunl_for == 1:
                dlam_eff = max(dfdlam[i, j], dgdlam_saved[i, j]) if loadflg[i] == -1.0 else dgdlam_saved[i, j]
            else:
                dlam_eff = max(dgdlam_saved[i, j], 0.0)
            dlam_eff = max(dlam_eff, 0.0)

            lamj = max(lam[i, j], _EM20)
            denom = 9.0 * rbulk[i] - dlam_eff * lamj
            if denom < _EM12:
                dlam_eff = min(dlam_eff, (0.98 * 9.0 * rbulk[i]) / lamj)
                denom = 9.0 * rbulk[i] - dlam_eff * lamj
            if denom > 0.0:
                gdir = 3.0 * rbulk[i] * dlam_eff * lamj / denom
                gmax[i] = max(gmax[i], gdir)

        soundsp[i] = math.sqrt((_FOUR_OVER_3 * (gmax[i] + gdamp) + rbulk[i]) / max(min(rho[i], rho0[i]), _EM20))
        et[i] = (gmax[i] + gdamp) / max(gs, _EM20)

    deint_new = deint0 + 0.5 * (
        signxx * depsxx + signyy * depsyy + signzz * depszz +
        signxy * depsxy + signyz * depsyz + signzx * depszx
    )
    ecurent = np.maximum(_EM20, uvar[:, 1] + deint_new)
    emax = np.maximum(uvar[:, 0], ecurent)

    uvar[:, 0] = emax
    uvar[:, 1] = ecurent
    uvar[:, 2] = epseq
    uvar[:, 4] = loadflg

    ldav_d = depsxx + depsyy + depszz
    sigdxx_new = uvar[:, 5] + 2.0 * gdamp * (depsxx - _THIRD * ldav_d)
    sigdyy_new = uvar[:, 6] + 2.0 * gdamp * (depsyy - _THIRD * ldav_d)
    sigdzz_new = uvar[:, 7] + 2.0 * gdamp * (depszz - _THIRD * ldav_d)
    sigdxy_new = uvar[:, 8] + gdamp * depsxy
    sigdyz_new = uvar[:, 9] + gdamp * depsyz
    sigdzx_new = uvar[:, 10] + gdamp * depszx

    sigdeff = np.sqrt(3.0 * (
        0.5 * (sigdxx_new ** 2 + sigdyy_new ** 2 + sigdzz_new ** 2) +
        sigdxy_new ** 2 + sigdyz_new ** 2 + sigdzx_new ** 2
    ))

    scale_d = np.minimum(sigf / np.maximum(sigdeff, _EM20), 1.0)
    sigdxx_new *= scale_d
    sigdyy_new *= scale_d
    sigdzz_new *= scale_d
    sigdxy_new *= scale_d
    sigdyz_new *= scale_d
    sigdzx_new *= scale_d

    signxx[:] += sigdxx_new
    signyy[:] += sigdyy_new
    signzz[:] += sigdzz_new
    signxy[:] += sigdxy_new
    signyz[:] += sigdyz_new
    signzx[:] += sigdzx_new

    uvar[:, 5] = sigdxx_new
    uvar[:, 6] = sigdyy_new
    uvar[:, 7] = sigdzz_new
    uvar[:, 8] = sigdxy_new
    uvar[:, 9] = sigdyz_new
    uvar[:, 10] = sigdzx_new

    if kfail > 0.0:
        lam_unnorm = lam / rv_mth[:, None]
        i1_inv = np.sum(lam_unnorm ** 2, axis=1)
        i2_inv = (
            lam_unnorm[:, 0] ** 2 * lam_unnorm[:, 1] ** 2 +
            lam_unnorm[:, 1] ** 2 * lam_unnorm[:, 2] ** 2 +
            lam_unnorm[:, 2] ** 2 * lam_unnorm[:, 0] ** 2
        )

        fcrit = (i1_inv - 3.0) + gam1 * ((i1_inv - 3.0) ** 2) + gam2 * (i2_inv - 3.0)

        for i in range(nel):
            if dmg[i] < 1.0 and off[i] == 1.0:
                if fcrit[i] <= (1.0 - eh) * kfail:
                    dmg[i] = 0.0
                elif ((1.0 - eh) * kfail < fcrit[i]) and (fcrit[i] < kfail):
                    dmg[i] = 0.5 * (1.0 + math.cos(math.pi * (fcrit[i] - kfail) / (eh * kfail)))
                elif fcrit[i] >= kfail:
                    dmg[i] = 1.0
                    uvar[i, 3] += 1.0
                    if (failip > 0) and (int(uvar[i, 3]) == failip):
                        off[i] = _FOUR_OVER_5

            dmg[i] = max(0.0, min(1.0, dmg[i]))

        signxx[:] *= (1.0 - dmg)
        signyy[:] *= (1.0 - dmg)
        signzz[:] *= (1.0 - dmg)
        signxy[:] *= (1.0 - dmg)
        signyz[:] *= (1.0 - dmg)
        signzx[:] *= (1.0 - dmg)


# ============================================================================
# 4. 2D Shell Membrane Material Kernel (sigeps88_shell)
#    Reference: sigeps88c.F90:44-1259
# ============================================================================

def sigeps88_shell(
    nel: int,
    matparam: Any,
    uvar: np.ndarray,
    tstep: float,
    tt: float,
    rho: np.ndarray,
    soundsp: np.ndarray,
    off: np.ndarray,
    ismstr: int,
    israte: int,
    ngl: np.ndarray,
    epsxx: np.ndarray,
    epsyy: np.ndarray,
    epsxy: np.ndarray,
    epspxx: np.ndarray,
    epspyy: np.ndarray,
    epspxy: np.ndarray,
    depsxx: np.ndarray,
    depsyy: np.ndarray,
    depsxy: np.ndarray,
    depsyz: np.ndarray,
    depszx: np.ndarray,
    sigoxx: np.ndarray,
    sigoyy: np.ndarray,
    sigoxy: np.ndarray,
    sigoyz: np.ndarray,
    sigozx: np.ndarray,
    signxx: np.ndarray,
    signyy: np.ndarray,
    signxy: np.ndarray,
    signyz: np.ndarray,
    signzx: np.ndarray,
    asrate: float,
    et: np.ndarray,
    epsd: np.ndarray,
    nvartmp: int,
    vartmp: np.ndarray,
    dmg: np.ndarray,
    thkly: np.ndarray,
    thk0: np.ndarray,
    thkn: np.ndarray,
    shf: np.ndarray,
    ipg: int,
    npg: int,
) -> None:
    """Constitutive stress update for 2D shell / membrane elements under /MAT/LAW88.

    Matches sigeps88c.F90:44-1259. Mutates signxx..zx, soundsp, et, uvar, off, thkn in-place.
    """
    mp = MatparamLaw88.from_dict_or_obj(matparam)

    itens = int(mp.iparam[0])
    iunl_for = int(mp.iparam[1])
    nload = int(mp.iparam[2])
    rtype = int(mp.iparam[3])
    failip = min(int(mp.iparam[4]), npg)
    nv_base = int(mp.iparam[5]) if len(mp.iparam) > 5 and mp.iparam[5] > 0 else 12

    rho0 = mp.rho0
    rbulk = np.full(nel, mp.bulk, dtype=np.float64)
    nu = mp.nu
    if nu >= 0.49:
        nu = 0.5
    gs = mp.shear
    hys = float(mp.uparam[0])
    shape = float(mp.uparam[1]) if mp.uparam[1] != 0.0 else 1.0
    gdamp = float(mp.uparam[2])
    sigf = float(mp.uparam[3])
    kfail = float(mp.uparam[4])
    gam1 = float(mp.uparam[5])
    gam2 = float(mp.uparam[6])
    eh = float(mp.uparam[7])
    beta = float(mp.uparam[8])

    emax = uvar[:, 0].copy()
    ecurent = uvar[:, 1].copy()
    loadflg = uvar[:, 4].copy()
    lam3_0 = uvar[:, 7].copy()
    loadflg_old = loadflg.copy()

    off[:] = np.where(off < 1.0, off * _FOUR_OVER_5, off)
    off[:] = np.where(off < _EM01, 0.0, off)

    sigdxx = uvar[:, 5].copy()
    sigdyy = uvar[:, 6].copy()
    sigdxy = uvar[:, 8].copy()

    dmg_fac = 0.5 * (1.0 / np.maximum(1.0 - dmg, _EM20))
    deint0 = dmg_fac * (
        (sigoxx - sigdxx) * depsxx +
        (sigoyy - sigdyy) * depsyy +
        (sigoxy - sigdxy) * depsxy
    )

    if ipg == 1:
        thkn[:] = 0.0

    if iunl_for == 1:
        for j in range(3):
            mask_init = uvar[:, nv_base + 9 + j] <= 0.0
            uvar[mask_init, nv_base + 9 + j] = 1.0
        mask_r = uvar[:, nv_base + 12] <= 0.0
        uvar[mask_r, nv_base + 12] = 1.0

    trav = epsxx + epsyy
    rootv = np.sqrt((epsxx - epsyy) ** 2 + epsxy ** 2)
    evv = np.zeros((nel, 3), dtype=np.float64)
    evv[:, 0] = 0.5 * (trav + rootv)
    evv[:, 1] = 0.5 * (trav - rootv)
    evv[:, 2] = 0.0

    eigv = np.zeros((nel, 3, 2), dtype=np.float64)
    for i in range(nel):
        vx = 0.5 * epsxy[i]
        vy = evv[i, 0] - epsxx[i]
        alt1 = abs(vx) + abs(vy)
        if alt1 < _EM06:
            vx = evv[i, 0] - epsyy[i]
            vy = 0.5 * epsxy[i]

        nrm = math.sqrt(vx * vx + vy * vy)
        if nrm > _EM06:
            vx /= nrm
            vy /= nrm
        else:
            if abs(uvar[i, 9]) + abs(uvar[i, 10]) > 0.0:
                vx = uvar[i, 9]
                vy = uvar[i, 10]
            else:
                vx = 1.0
                vy = 0.0

        if vx * uvar[i, 9] + vy * uvar[i, 10] < 0.0:
            vx = -vx
            vy = -vy

        eigv[i, 0, 0] = vx
        eigv[i, 1, 0] = vy
        eigv[i, 2, 0] = 0.0

        eigv[i, 0, 1] = -vy
        eigv[i, 1, 1] = vx
        eigv[i, 2, 1] = 0.0

        uvar[i, 9] = vx
        uvar[i, 10] = vy

    evvp = np.zeros((nel, 3), dtype=np.float64)
    evvp[:, 0] = (
        eigv[:, 0, 0] ** 2 * epspxx +
        eigv[:, 0, 0] * eigv[:, 1, 0] * epspxy +
        eigv[:, 1, 0] ** 2 * epspyy
    )
    evvp[:, 1] = (
        eigv[:, 0, 1] ** 2 * epspxx +
        eigv[:, 0, 1] * eigv[:, 1, 1] * epspxy +
        eigv[:, 1, 1] ** 2 * epspyy
    )
    if nu > 0.49:
        evvp[:, 2] = -(evvp[:, 0] + evvp[:, 1])

    lam = np.zeros((nel, 3), dtype=np.float64)
    if ismstr in (0, 2, 4):
        lam[:, 0] = np.exp(evv[:, 0])
        lam[:, 1] = np.exp(evv[:, 1])
    elif ismstr in (10, 12):
        lam[:, 0] = np.sqrt(np.maximum(evv[:, 0] + 1.0, _EM20))
        lam[:, 1] = np.sqrt(np.maximum(evv[:, 1] + 1.0, _EM20))
    else:
        lam[:, 0] = evv[:, 0] + 1.0
        lam[:, 1] = evv[:, 1] + 1.0

    for i in range(nel):
        if lam3_0[i] > 0.0:
            lam[i, 2] = lam3_0[i]
        else:
            lam[i, 2] = 1.0 / max(lam[i, 0] * lam[i, 1], _EM20)

    ee = lam - 1.0

    if rtype == 1:
        if ismstr in (0, 2, 4):
            eep1 = np.exp(evv[:, 0]) * evvp[:, 0]
            eep2 = np.exp(evv[:, 1]) * evvp[:, 1]
        elif ismstr in (10, 12):
            eep1 = 0.5 * (1.0 / np.sqrt(np.maximum(evv[:, 0] + 1.0, _EM20))) * evvp[:, 0]
            eep2 = 0.5 * (1.0 / np.sqrt(np.maximum(evv[:, 1] + 1.0, _EM20))) * evvp[:, 1]
        else:
            eep1 = evvp[:, 0].copy()
            eep2 = evvp[:, 1].copy()
        erate = np.sqrt(eep1 ** 2 + eep2 ** 2)
    else:
        erate = np.sqrt(evvp[:, 0] ** 2 + evvp[:, 1] ** 2)

    if israte > 0:
        epsd[:] = asrate * erate + (1.0 - asrate) * epsd
    else:
        epsd[:] = erate

    table_load = mp.table[0] if len(mp.table) > 0 else TableData(x1=np.array([0.0, 1.0]), y1d=np.array([0.0, max(float(mp.young), float(mp.shear), 1.0)]))

    f = np.zeros((nel, 3), dtype=np.float64)
    dfdlam = np.zeros((nel, 3), dtype=np.float64)
    p = np.zeros(nel, dtype=np.float64)
    dpdrv = np.zeros(nel, dtype=np.float64)
    fJ = np.zeros(nel, dtype=np.float64)
    dfJdx = np.zeros(nel, dtype=np.float64)
    t = np.zeros((nel, 3), dtype=np.float64)
    rv = np.zeros(nel, dtype=np.float64)
    rv_mth = np.zeros(nel, dtype=np.float64)
    dgdlam_saved = np.zeros((nel, 3), dtype=np.float64)

    for _ in range(3):
        rv = lam[:, 0] * lam[:, 1] * lam[:, 2]
        rv_safe = np.maximum(rv, _EM20)
        rv_mth = np.exp(-_THIRD * np.log(rv_safe))
        if (nu > 0.0) and (nu < 0.49):
            rv_mth[:] = 1.0

        lam_iso = lam * rv_mth[:, None]

        for j in range(3):
            lam_j = lam_iso[:, j]
            rate_j = epsd.copy()
            if itens == 0:
                rate_j = np.where(rv > 1.0, 0.0, rate_j)

            g_j, dg_j = table_mat_vinterp_eval(table_load, lam_j, rate_j, opt_extrapolate=True)
            if j < 2:
                dgdlam_saved[:, j] = dg_j

            f[:, j] = lam_j * g_j
            dfdlam[:, j] = g_j + lam_j * dg_j

            for n in range(1, 7):
                power_n = (-nu) ** n
                lam_p = lam_j ** power_n
                g_sqr, dg_sqr = table_mat_vinterp_eval(table_load, lam_p, rate_j, opt_extrapolate=True)
                f[:, j] += lam_p * g_sqr
                dfdlam[:, j] += power_n * (lam_j ** (power_n - 1.0)) * (g_sqr + lam_p * dg_sqr)

        if beta > 0.0:
            ldav = epspxx + epspyy
            exp_beta = math.exp(-beta * tstep)
            p = uvar[:, 11] * exp_beta + rbulk * ldav * ((1.0 - exp_beta) / beta)
            dpdrv[:] = 0.0
        elif (nu > 0.0) and (nu < 0.49):
            xfoam = rv_safe ** (-nu / (1.0 - 2.0 * nu))
            rate_foam = epsd.copy()
            gJ, dgJ = table_mat_vinterp_eval(table_load, xfoam, rate_foam, opt_extrapolate=True)
            fJ = gJ * xfoam
            dfJdx = gJ + xfoam * dgJ

            for n in range(1, 7):
                power_n = (-nu) ** n
                xf_p = xfoam ** power_n
                gJsqr, dgJsqr = table_mat_vinterp_eval(table_load, xf_p, rate_foam, opt_extrapolate=True)
                fJ += xf_p * gJsqr
                dfJdx += power_n * (xfoam ** (power_n - 1.0)) * (gJsqr + xf_p * dgJsqr)

            rbulk = (-nu / (1.0 - 2.0 * nu)) * xfoam * dfJdx
            rbulk = np.maximum(rbulk, _EM12)
        else:
            p = rbulk * (rv - 1.0)
            dpdrv[:] = rbulk

        if (nu <= 0.0) or (nu >= 0.49):
            t[:, 2] = (_TWO_THIRD * f[:, 2] - _THIRD * (f[:, 0] + f[:, 1]) + p) / rv_safe
            dt3_dlam3 = (1.0 / rv_safe) * (
                (4.0 / 9.0) * dfdlam[:, 2] * lam_iso[:, 2] / np.maximum(lam[:, 2], _EM20) +
                (1.0 / 9.0) * dfdlam[:, 0] * lam_iso[:, 0] / np.maximum(lam[:, 2], _EM20) +
                (1.0 / 9.0) * dfdlam[:, 1] * lam_iso[:, 1] / np.maximum(lam[:, 2], _EM20) +
                lam[:, 0] * lam[:, 1] * (dpdrv - t[:, 2])
            )
        else:
            t[:, 2] = (f[:, 2] - fJ) / rv_safe
            dt3_dlam3 = (
                (1.0 / rv_safe) * dfdlam[:, 2] +
                (1.0 / np.maximum(lam[:, 2], _EM20)) * dfJdx * (nu / (1.0 - 2.0 * nu)) * (rv_safe ** ((nu - 1.0) / (1.0 - 2.0 * nu))) -
                (1.0 / np.maximum(lam[:, 2], _EM20)) * t[:, 2]
            )

        dt3_dlam3_safe = np.where(np.abs(dt3_dlam3) < _EM20, np.copysign(_EM20, dt3_dlam3), dt3_dlam3)
        lam[:, 2] = lam[:, 2] - t[:, 2] / dt3_dlam3_safe

    rv = lam[:, 0] * lam[:, 1] * lam[:, 2]
    rv_safe = np.maximum(rv, _EM20)
    rv_mth = np.exp(-_THIRD * np.log(rv_safe))
    if (nu > 0.0) and (nu < 0.49):
        rv_mth[:] = 1.0

    if (nu <= 0.0) or (nu >= 0.49):
        t[:, 0] = (_TWO_THIRD * f[:, 0] - _THIRD * (f[:, 1] + f[:, 2]) + p) / rv_safe
        t[:, 1] = (_TWO_THIRD * f[:, 1] - _THIRD * (f[:, 0] + f[:, 2]) + p) / rv_safe
        t[:, 2] = (_TWO_THIRD * f[:, 2] - _THIRD * (f[:, 0] + f[:, 1]) + p) / rv_safe
    else:
        t[:, 0] = (f[:, 0] - fJ) / rv_safe
        t[:, 1] = (f[:, 1] - fJ) / rv_safe
        t[:, 2] = (f[:, 2] - fJ) / rv_safe

    epseq = np.sqrt(ee[:, 0] ** 2 + ee[:, 1] ** 2)
    deint = deint0 + 0.5 * (t[:, 0] * evvp[:, 0] + t[:, 1] * evvp[:, 1]) * tstep
    ecurent = np.maximum(_EM20, ecurent + deint)
    emax = np.maximum(emax, ecurent)

    for i in range(nel):
        if off[i] == 1.0:
            if loadflg[i] == -1.0:
                if deint[i] / max(ecurent[i], _EM20) >= _EM07:
                    loadflg[i] = 1.0
                    emax[i] = ecurent[i]
                else:
                    loadflg[i] = -1.0
            else:
                if deint[i] / max(emax[i], _EM20) >= 0.0:
                    loadflg[i] = 1.0
                else:
                    loadflg[i] = -1.0

    if iunl_for == 1:
        for i in range(nel):
            if loadflg_old[i] == 1.0 and loadflg[i] == -1.0:
                for j in range(3):
                    uvar[i, nv_base + j] = lam[i, j]
                    if abs(uvar[i, nv_base + 3 + j]) < _EM10:
                        uvar[i, nv_base + 3 + j] = 1.0
                    uvar[i, nv_base + 6 + j] = f[i, j]
                    uvar[i, nv_base + 9 + j] = 1.0
                uvar[i, nv_base + 12] = 1.0
            elif loadflg_old[i] == -1.0 and loadflg[i] == 1.0:
                for j in range(3):
                    uvar[i, nv_base + 3 + j] = uvar[i, nv_base + j]
                    uvar[i, nv_base + j] = lam[i, j]
                    uvar[i, nv_base + 6 + j] = f[i, j]
                    uvar[i, nv_base + 9 + j] = 1.0
                uvar[i, nv_base + 12] = 1.0

        for i in range(nel):
            if loadflg[i] == -1.0:
                amax = 0.0
                for j in range(3):
                    lam_r = uvar[i, nv_base + j]
                    lam_tg = uvar[i, nv_base + 3 + j]
                    denom = max(abs(lam_r - lam_tg), _EM20)
                    xhat = np.clip((lam[i, j] - lam_tg) / denom, -1.0, 1.0)
                    prev = uvar[i, nv_base + 9 + j]
                    if prev <= 0.0:
                        prev = 1.0
                    xmag = min(abs(xhat), prev)
                    uvar[i, nv_base + 9 + j] = xmag
                    amax = max(amax, xmag)

                if amax <= _EM03:
                    loadflg[i] = 1.0
                    for j in range(3):
                        uvar[i, nv_base + 3 + j] = uvar[i, nv_base + j]
                        uvar[i, nv_base + j] = lam[i, j]
                        uvar[i, nv_base + 6 + j] = f[i, j]
                        uvar[i, nv_base + 9 + j] = 1.0
                    uvar[i, nv_base + 12] = 1.0

        unl_indices = [i for i in range(nel) if off[i] == 1.0 and loadflg[i] == -1.0]
        if unl_indices and len(mp.table) >= 3:
            table_unl = mp.table[1]
            table_norm = mp.table[2]

            for i in unl_indices:
                amax = -1.0
                xhat_dom = 0.0
                jdom = 0

                for j in range(2):
                    lam_r = uvar[i, nv_base + j]
                    lam_tg = uvar[i, nv_base + 3 + j]
                    denom = max(abs(lam_r - lam_tg), _EM20)
                    xhat = np.clip((lam[i, j] - lam_tg) / denom, -1.0, 1.0)
                    prev = uvar[i, nv_base + 9 + j]
                    if prev <= 0.0:
                        prev = 1.0
                    xmag = min(abs(xhat), prev)
                    uvar[i, nv_base + 9 + j] = xmag
                    if xmag > amax:
                        amax = xmag
                        jdom = j
                        xhat_dom = math.copysign(xmag, xhat)

                xvec_1 = math.copysign(min(abs(xhat_dom), 1.0 - _EM08), xhat_dom)
                xvec_2 = 0.0
                if itens == 0 and rv[i] < 1.0:
                    xvec_2 = epsd[i]
                elif itens == 1:
                    xvec_2 = epsd[i]

                g_norm, _ = table_mat_vinterp_eval(table_norm, xvec_1, xvec_2, opt_extrapolate=True)
                gunl, _ = table_mat_vinterp_eval(table_unl, xvec_1, xvec_2, opt_extrapolate=True)

                ratioR_val = 0.0
                if abs(g_norm[0]) > _EM20:
                    ratioR_val = np.clip(gunl[0] / g_norm[0], 0.0, 1.0)

                amax_val = np.clip(uvar[i, nv_base + 9 + jdom], 0.0, 1.0)
                Rblend = (1.0 - amax_val ** 3) * uvar[i, nv_base + 12] + (amax_val ** 3) * ratioR_val
                ratioR_val = float(np.clip(Rblend, 0.0, 1.0))
                uvar[i, nv_base + 12] = ratioR_val

                f[i, :] *= ratioR_val
                dfdlam[i, :] *= ratioR_val

                if (nu <= 0.0) or (nu >= 0.49):
                    t[i, 0] = (_TWO_THIRD * f[i, 0] - _THIRD * (f[i, 1] + f[i, 2]) + p[i]) / rv_safe[i]
                    t[i, 1] = (_TWO_THIRD * f[i, 1] - _THIRD * (f[i, 0] + f[i, 2]) + p[i]) / rv_safe[i]
                else:
                    t[i, 0] = (f[i, 0] - fJ[i] * ratioR_val) / rv_safe[i]
                    t[i, 1] = (f[i, 1] - fJ[i] * ratioR_val) / rv_safe[i]

    elif iunl_for == 2:
        for i in range(nel):
            if off[i] == 1.0 and loadflg[i] == -1.0:
                e_ratio = ecurent[i] / max(emax[i], _EM20)
                ratioR = 1.0 - (1.0 - hys) * (1.0 - (e_ratio ** shape))
                t[i, 0] *= ratioR
                t[i, 1] *= ratioR

    signxx[:] = t[:, 0] * eigv[:, 0, 0] ** 2 + t[:, 1] * eigv[:, 0, 1] ** 2
    signyy[:] = t[:, 0] * eigv[:, 1, 0] ** 2 + t[:, 1] * eigv[:, 1, 1] ** 2
    signxy[:] = t[:, 0] * eigv[:, 0, 0] * eigv[:, 1, 0] + t[:, 1] * eigv[:, 0, 1] * eigv[:, 1, 1]
    signyz[:] = sigoyz + gs * shf * depsyz
    signzx[:] = sigozx + gs * shf * depszx

    gmax = np.full(nel, gs, dtype=np.float64)
    for i in range(nel):
        for j in range(2):
            if iunl_for == 1:
                dlam_eff = max(dfdlam[i, j], dgdlam_saved[i, j]) if loadflg[i] == -1.0 else dgdlam_saved[i, j]
            else:
                dlam_eff = max(dgdlam_saved[i, j], 0.0)
            dlam_eff = max(dlam_eff, 0.0)

            lamj = max(lam[i, j], _EM20)
            denom = 9.0 * rbulk[i] - dlam_eff * lamj
            if denom < _EM12:
                dlam_eff = min(dlam_eff, (0.98 * 9.0 * rbulk[i]) / lamj)
                denom = 9.0 * rbulk[i] - dlam_eff * lamj
            if denom > 0.0:
                gdir = 3.0 * rbulk[i] * dlam_eff * lamj / denom
                gmax[i] = max(gmax[i], gdir)

        a11 = gmax[i] + gdamp
        a11 = 4.0 * a11 * (a11 + 3.0 * rbulk[i]) / (4.0 * a11 + 3.0 * rbulk[i])
        soundsp[i] = math.sqrt(a11 / max(min(rho[i], rho0), _EM20))
        et[i] = (gmax[i] + gdamp) / max(gs, _EM20)

    deint_new = deint0 + 0.5 * (signxx * depsxx + signyy * depsyy + signxy * depsxy)
    ecurent = np.maximum(_EM20, uvar[:, 1] + deint_new)
    emax = np.maximum(uvar[:, 0], ecurent)

    uvar[:, 0] = emax
    uvar[:, 1] = ecurent
    uvar[:, 2] = epseq
    uvar[:, 4] = loadflg
    uvar[:, 7] = lam[:, 2]
    if beta > 0.0:
        uvar[:, 11] = p

    thkn[:] += thkly * thk0 * lam[:, 2]

    ldav_d = depsxx + depsyy
    sigdxx_new = uvar[:, 5] + 2.0 * gdamp * (depsxx - _THIRD * ldav_d)
    sigdyy_new = uvar[:, 6] + 2.0 * gdamp * (depsyy - _THIRD * ldav_d)
    sigdxy_new = uvar[:, 8] + gdamp * depsxy

    sigdeff = np.sqrt(3.0 * (0.5 * (sigdxx_new ** 2 + sigdyy_new ** 2) + sigdxy_new ** 2))
    scale_d = np.minimum(sigf / np.maximum(sigdeff, _EM20), 1.0)
    sigdxx_new *= scale_d
    sigdyy_new *= scale_d
    sigdxy_new *= scale_d

    signxx[:] += sigdxx_new
    signyy[:] += sigdyy_new
    signxy[:] += sigdxy_new

    uvar[:, 5] = sigdxx_new
    uvar[:, 6] = sigdyy_new
    uvar[:, 8] = sigdxy_new

    if kfail > 0.0:
        lam_unnorm = lam / rv_mth[:, None]
        i1_inv = np.sum(lam_unnorm ** 2, axis=1)
        i2_inv = (
            lam_unnorm[:, 0] ** 2 * lam_unnorm[:, 1] ** 2 +
            lam_unnorm[:, 1] ** 2 * lam_unnorm[:, 2] ** 2 +
            lam_unnorm[:, 2] ** 2 * lam_unnorm[:, 0] ** 2
        )
        fcrit = (i1_inv - 3.0) + gam1 * ((i1_inv - 3.0) ** 2) + gam2 * (i2_inv - 3.0)

        for i in range(nel):
            if dmg[i] < 1.0 and off[i] == 1.0:
                if fcrit[i] <= (1.0 - eh) * kfail:
                    dmg[i] = 0.0
                elif ((1.0 - eh) * kfail < fcrit[i]) and (fcrit[i] < kfail):
                    dmg[i] = 0.5 * (1.0 + math.cos(math.pi * (fcrit[i] - kfail) / (eh * kfail)))
                elif fcrit[i] >= kfail:
                    dmg[i] = 1.0
                    uvar[i, 3] += 1.0
                    if (failip > 0) and (int(uvar[i, 3]) == failip):
                        off[i] = _FOUR_OVER_5

            dmg[i] = max(0.0, min(1.0, dmg[i]))

        signxx[:] *= (1.0 - dmg)
        signyy[:] *= (1.0 - dmg)
        signxy[:] *= (1.0 - dmg)


# ============================================================================
# 5. Algorithmic Consistent Tangent Tensors
# ============================================================================

def tangent_law88_solid(
    eps: np.ndarray,
    matparam: Any,
    dt: float = 1e-6,
    h: float = 1e-7,
    ismstr: int = 0,
    israte: int = 0,
    **kwargs: Any,
) -> np.ndarray:
    """Compute (NEL, 6, 6) consistent spatial algorithmic tangent tensor for 3D solid elements.

    Uses central finite-difference perturbations of the constitutive kernel.

    Parameters
    ----------
    eps : np.ndarray
        Strain tensor array of shape (NEL, 6) or (6,) [epsxx, epsyy, epszz, epsxy, epsyz, epszx].
    matparam : MatparamLaw88 or dict or object
        Material parameter structure.
    dt : float, optional
        Time step (default 1e-6).
    h : float, optional
        Finite difference perturbation magnitude (default 1e-7).
    ismstr : int, optional
        Strain formulation flag (default 0).
    israte : int, optional
        Strain rate flag (default 0).

    Returns
    -------
    C : np.ndarray
        Algorithmic tangent tensor of shape (NEL, 6, 6) or (6, 6).
    """
    eps_in = np.asarray(eps, dtype=np.float64)
    is_1d = (eps_in.ndim == 1)
    if is_1d:
        eps_arr = eps_in.reshape(1, -1)
    else:
        eps_arr = eps_in

    nel = eps_arr.shape[0]
    mp = MatparamLaw88.from_dict_or_obj(matparam)
    C = np.zeros((nel, 6, 6), dtype=np.float64)

    def _eval_sig(eps_trial: np.ndarray) -> np.ndarray:
        uvar_trial = np.zeros((nel, 30), dtype=np.float64)
        rho_arr = np.full(nel, mp.rho, dtype=np.float64)
        rho0_arr = np.full(nel, mp.rho0, dtype=np.float64)
        soundsp = np.zeros(nel, dtype=np.float64)
        off = np.ones(nel, dtype=np.float64)
        et = np.zeros(nel, dtype=np.float64)
        offg = np.zeros(nel, dtype=np.float64)
        epsd = np.zeros(nel, dtype=np.float64)
        vartmp = np.zeros((nel, 6), dtype=int)
        dmg = np.zeros(nel, dtype=np.float64)
        ngl = np.arange(1, nel + 1, dtype=int)

        sig_out = np.zeros((nel, 6), dtype=np.float64)

        sigeps88_solid(
            nel=nel,
            matparam=mp,
            uvar=uvar_trial,
            tstep=dt,
            tt=0.0,
            rho0=rho0_arr,
            rho=rho_arr,
            soundsp=soundsp,
            off=off,
            ismstr=ismstr,
            israte=israte,
            epsxx=eps_trial[:, 0],
            epsyy=eps_trial[:, 1],
            epszz=eps_trial[:, 2],
            epsxy=eps_trial[:, 3],
            epsyz=eps_trial[:, 4],
            epszx=eps_trial[:, 5],
            depsxx=eps_trial[:, 0],
            depsyy=eps_trial[:, 1],
            depszz=eps_trial[:, 2],
            depsxy=eps_trial[:, 3],
            depsyz=eps_trial[:, 4],
            depszx=eps_trial[:, 5],
            epspxx=np.zeros(nel),
            epspyy=np.zeros(nel),
            epspzz=np.zeros(nel),
            epspxy=np.zeros(nel),
            epspyz=np.zeros(nel),
            epspzx=np.zeros(nel),
            sigoxx=np.zeros(nel),
            sigoyy=np.zeros(nel),
            sigozz=np.zeros(nel),
            sigoxy=np.zeros(nel),
            sigoyz=np.zeros(nel),
            sigozx=np.zeros(nel),
            signxx=sig_out[:, 0],
            signyy=sig_out[:, 1],
            signzz=sig_out[:, 2],
            signxy=sig_out[:, 3],
            signyz=sig_out[:, 4],
            signzx=sig_out[:, 5],
            asrate=0.0,
            et=et,
            offg=offg,
            epsd=epsd,
            iresp=0,
            nvartmp=6,
            vartmp=vartmp,
            dmg=dmg,
            ngl=ngl,
            npg=1,
        )
        return sig_out

    for j in range(6):
        eps_p = eps_arr.copy()
        eps_p[:, j] += h
        eps_m = eps_arr.copy()
        eps_m[:, j] -= h

        sp = _eval_sig(eps_p)
        sm = _eval_sig(eps_m)
        C[:, :, j] = (sp - sm) / (2.0 * h)

    return C[0] if is_1d else C


def tangent_law88_shell(
    eps: np.ndarray,
    matparam: Any,
    dt: float = 1e-6,
    h: float = 1e-7,
    ismstr: int = 0,
    israte: int = 0,
    **kwargs: Any,
) -> np.ndarray:
    """Compute (NEL, 3, 3) consistent in-plane algorithmic tangent tensor for 2D shell elements.

    Parameters
    ----------
    eps : np.ndarray
        In-plane strain tensor array of shape (NEL, 3) or (3,) [epsxx, epsyy, epsxy].
    matparam : MatparamLaw88 or dict or object
        Material parameter structure.
    dt : float, optional
        Time step (default 1e-6).
    h : float, optional
        Finite difference perturbation magnitude (default 1e-7).
    ismstr : int, optional
        Strain formulation flag (default 0).
    israte : int, optional
        Strain rate flag (default 0).

    Returns
    -------
    C : np.ndarray
        Condensed in-plane tangent tensor of shape (NEL, 3, 3) or (3, 3).
    """
    eps_in = np.asarray(eps, dtype=np.float64)
    is_1d = (eps_in.ndim == 1)
    if is_1d:
        eps_arr = eps_in.reshape(1, -1)
    else:
        eps_arr = eps_in

    nel = eps_arr.shape[0]
    mp = MatparamLaw88.from_dict_or_obj(matparam)
    C = np.zeros((nel, 3, 3), dtype=np.float64)

    def _eval_sig(eps_trial: np.ndarray) -> np.ndarray:
        uvar_trial = np.zeros((nel, 30), dtype=np.float64)
        rho_arr = np.full(nel, mp.rho, dtype=np.float64)
        soundsp = np.zeros(nel, dtype=np.float64)
        off = np.ones(nel, dtype=np.float64)
        et = np.zeros(nel, dtype=np.float64)
        epsd = np.zeros(nel, dtype=np.float64)
        vartmp = np.zeros((nel, 6), dtype=int)
        dmg = np.zeros(nel, dtype=np.float64)
        ngl = np.arange(1, nel + 1, dtype=int)
        thkly = np.ones(nel, dtype=np.float64)
        thk0 = np.ones(nel, dtype=np.float64)
        thkn = np.ones(nel, dtype=np.float64)
        shf = np.ones(nel, dtype=np.float64)

        sig_out = np.zeros((nel, 5), dtype=np.float64)

        sigeps88_shell(
            nel=nel,
            matparam=mp,
            uvar=uvar_trial,
            tstep=dt,
            tt=0.0,
            rho=rho_arr,
            soundsp=soundsp,
            off=off,
            ismstr=ismstr,
            israte=israte,
            ngl=ngl,
            epsxx=eps_trial[:, 0],
            epsyy=eps_trial[:, 1],
            epsxy=eps_trial[:, 2],
            epspxx=np.zeros(nel),
            epspyy=np.zeros(nel),
            epspxy=np.zeros(nel),
            depsxx=eps_trial[:, 0],
            depsyy=eps_trial[:, 1],
            depsxy=eps_trial[:, 2],
            depsyz=np.zeros(nel),
            depszx=np.zeros(nel),
            sigoxx=np.zeros(nel),
            sigoyy=np.zeros(nel),
            sigoxy=np.zeros(nel),
            sigoyz=np.zeros(nel),
            sigozx=np.zeros(nel),
            signxx=sig_out[:, 0],
            signyy=sig_out[:, 1],
            signxy=sig_out[:, 2],
            signyz=sig_out[:, 3],
            signzx=sig_out[:, 4],
            asrate=0.0,
            et=et,
            epsd=epsd,
            nvartmp=6,
            vartmp=vartmp,
            dmg=dmg,
            thkly=thkly,
            thk0=thk0,
            thkn=thkn,
            shf=shf,
            ipg=1,
            npg=1,
        )
        return sig_out[:, :3]

    for j in range(3):
        eps_p = eps_arr.copy()
        eps_p[:, j] += h
        eps_m = eps_arr.copy()
        eps_m[:, j] -= h

        sp = _eval_sig(eps_p)
        sm = _eval_sig(eps_m)
        C[:, :, j] = (sp - sm) / (2.0 * h)

    return C[0] if is_1d else C


# ============================================================================
# High-level Solver Wrappers and Registration
# ============================================================================

@dataclass
class Law88Params:
    """Strongly-typed container for /MAT/LAW88 material parameters."""
    id: int = 1
    law: int = 88
    title: str = "LAW88_TAB_HYP"
    rho0: float = 1.0
    ref_rho: float = 1.0
    nu: float = 0.495
    bulk: float = 1000.0
    fcut: float = 0.0
    fsmooth: int = 0
    nl: int = 1
    ifunc_unload: int = 0
    fscale_unload: float = 1.0
    hys: float = 0.0
    shape: float = 1.0
    tension: int = 0
    rtype: int = 0
    func_load_list: list = field(default_factory=list)
    fscale_load_list: list = field(default_factory=list)
    rate_load_list: list = field(default_factory=list)
    lamfit_list: list = field(default_factory=list)
    sgl: float = 0.0
    sw: float = 0.0
    st: float = 0.0
    g: float = 0.0
    sigf: float = 0.0
    kfail: float = 0.0
    gam1: float = 0.0
    gam2: float = 0.0
    eh: float = 0.0
    failip: int = 0
    young: float = 300.0
    shear: float = 100.0
    table: list[TableData] = field(default_factory=list)


def build_law88(rec: Any = None, **kwargs: Any) -> Any:
    """Build a Material or MatparamLaw88 from a parsed LAW88 record or kwargs."""
    if isinstance(rec, MatparamLaw88):
        return rec
    if isinstance(rec, Law88Params):
        return rec

    p: dict[str, Any] = {}
    if rec is not None:
        if isinstance(rec, dict):
            p.update(rec)
        elif hasattr(rec, "params") and isinstance(rec.params, dict):
            p.update(rec.params)
        elif hasattr(rec, "__dict__"):
            p.update(rec.__dict__)
    p.update(kwargs)

    mid = int(p.get("id", getattr(rec, "id", 1)))
    title = str(p.get("title", getattr(rec, "title", "LAW88")))
    rho0 = float(p.get("rho0", p.get("MAT_RHO", p.get("rho", 1.0))))
    nu = float(p.get("nu", p.get("LAW88_Nu", 0.495)))
    bulk = float(p.get("bulk", p.get("LAW88_K", p.get("K", 1000.0))))
    young = float(p.get("young", p.get("E", p.get("MAT_E", 300.0))))
    shear = float(p.get("shear", p.get("G", p.get("MAT_G", 100.0))))

    p["rho0"] = rho0
    p["nu"] = nu
    p["bulk"] = bulk
    p["young"] = young
    p["shear"] = shear
    p["E"] = young
    p["G"] = shear
    p["K"] = bulk

    try:
        from ..model.entities import Material
        mat_inst = Material(id=mid, law=88, rho0=rho0, title=title, law_name="LAW88", params=p)
        return mat_inst
    except Exception:
        return MatparamLaw88.from_dict_or_obj(p)


def sound_speed_solid(mat: Any, rho: Optional[Union[float, np.ndarray]] = None, extra: Any = None, **kwargs: Any) -> Union[float, np.ndarray]:
    """Bulk acoustic sound speed c = sqrt((K + 4/3 G) / rho)."""
    mp = MatparamLaw88.from_dict_or_obj(mat)
    k = mp.bulk
    g = mp.shear
    rho_val = rho if rho is not None else mp.rho0
    rho_arr = np.asarray(rho_val, dtype=np.float64)
    rho_safe = np.maximum(rho_arr, 1e-20)
    c2 = np.maximum(0.0, k + (4.0 / 3.0) * g) / rho_safe
    res = np.sqrt(c2)
    return float(res) if res.ndim == 0 else res


def sound_speed_shell(mat: Any, rho: Optional[Union[float, np.ndarray]] = None, extra: Any = None, **kwargs: Any) -> Union[float, np.ndarray]:
    """Plane-stress acoustic sound speed c = sqrt(E / ((1 - nu^2) * rho))."""
    mp = MatparamLaw88.from_dict_or_obj(mat)
    e = mp.young
    nu = min(0.499, max(0.0, mp.nu))
    rho_val = rho if rho is not None else mp.rho0
    rho_arr = np.asarray(rho_val, dtype=np.float64)
    rho_safe = np.maximum(rho_arr, 1e-20)
    denom = max(1e-4, 1.0 - nu * nu)
    c2 = np.maximum(0.0, e / (denom * rho_safe))
    res = np.sqrt(c2)
    return float(res) if res.ndim == 0 else res


sound_speed = sound_speed_solid


def extra_shapes(mat: Any = None, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Allocate uvar88, off88, and eps88 state variables for solid and shell elements."""
    nuvar = 30
    shapes: Dict[str, Tuple[int, ...]] = {}
    if nip is None:
        shapes["uvar88"] = (nuvar,)
        shapes["off88"] = ()
        shapes["eps88"] = (6,)
    else:
        shapes["uvar88"] = (nip, nuvar)
        shapes["off88"] = (nip,)
        shapes["eps88"] = (nip, 3)
        shapes["thk88"] = (nip,)
    return shapes


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_tuple: bool = True,
    **kwargs: Any,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Standard pyradioss solid stress update for LAW88."""
    sig_arr = np.asarray(sig, dtype=np.float64)
    deps_arr = np.asarray(deps, dtype=np.float64)
    is_1d = sig_arr.ndim == 1
    if is_1d:
        sig_arr = sig_arr.reshape((1, 6))
        deps_arr = deps_arr.reshape((1, 6))
    nel = sig_arr.shape[0]

    mp = MatparamLaw88.from_dict_or_obj(mat)

    # State variables
    uvar = None
    if extra is not None:
        for k in ("uvar88", "uvar"):
            if k in extra and isinstance(extra[k], np.ndarray):
                uvar = extra[k]
                break
    if uvar is None:
        uvar = np.zeros((nel, 30), dtype=np.float64)
    elif uvar.ndim == 1:
        uvar = uvar.reshape((nel, -1))

    soundsp = np.zeros(nel, dtype=np.float64)
    off = np.ones(nel, dtype=np.float64)
    if extra is not None:
        if "off" in extra:
            off[:] = extra["off"]
        elif "off88" in extra:
            off[:] = extra["off88"]
    et = np.zeros(nel, dtype=np.float64)
    offg = np.zeros(nel, dtype=np.float64)
    epsd = np.zeros(nel, dtype=np.float64)
    vartmp = np.zeros((nel, 6), dtype=int)
    dmg = np.zeros(nel, dtype=np.float64)
    ngl = np.arange(1, nel + 1, dtype=int)

    rho0 = np.full(nel, mp.rho0, dtype=np.float64)
    rho = np.full(nel, mp.rho, dtype=np.float64)
    if extra is not None and "rho" in extra:
        rho[:] = extra["rho"]

    # Total strains
    if extra is not None and "eps88" in extra:
        extra["eps88"] += deps_arr
        eps_tot = extra["eps88"]
    elif extra is not None and "eps" in extra:
        eps_tot = np.asarray(extra["eps"], dtype=np.float64)
        if eps_tot.ndim == 1:
            eps_tot = eps_tot.reshape((1, 6))
    else:
        eps_tot = deps_arr

    sign = np.zeros_like(sig_arr)
    tstep = dt if dt > 0.0 else 1e-5
    epsp_in = (deps_arr / tstep)

    sigeps88_solid(
        nel=nel,
        matparam=mp,
        uvar=uvar,
        tstep=tstep,
        tt=0.0,
        rho0=rho0,
        rho=rho,
        soundsp=soundsp,
        off=off,
        ismstr=0,
        israte=0,
        epsxx=eps_tot[:, 0],
        epsyy=eps_tot[:, 1],
        epszz=eps_tot[:, 2],
        epsxy=eps_tot[:, 3],
        epsyz=eps_tot[:, 4],
        epszx=eps_tot[:, 5],
        depsxx=deps_arr[:, 0],
        depsyy=deps_arr[:, 1],
        depszz=deps_arr[:, 2],
        depsxy=deps_arr[:, 3],
        depsyz=deps_arr[:, 4],
        depszx=deps_arr[:, 5],
        epspxx=epsp_in[:, 0],
        epspyy=epsp_in[:, 1],
        epspzz=epsp_in[:, 2],
        epspxy=epsp_in[:, 3],
        epspyz=epsp_in[:, 4],
        epspzx=epsp_in[:, 5],
        sigoxx=sig_arr[:, 0],
        sigoyy=sig_arr[:, 1],
        sigozz=sig_arr[:, 2],
        sigoxy=sig_arr[:, 3],
        sigoyz=sig_arr[:, 4],
        sigozx=sig_arr[:, 5],
        signxx=sign[:, 0],
        signyy=sign[:, 1],
        signzz=sign[:, 2],
        signxy=sign[:, 3],
        signyz=sign[:, 4],
        signzx=sign[:, 5],
        asrate=0.0,
        et=et,
        offg=offg,
        epsd=epsd,
        iresp=0,
        nvartmp=6,
        vartmp=vartmp,
        dmg=dmg,
        ngl=ngl,
        npg=1,
    )

    if extra is not None:
        if "off" in extra:
            extra["off"][:] = off
        if "off88" in extra:
            extra["off88"][:] = off

    if is_1d:
        sign = sign[0]
        soundsp = soundsp[0]

    epsp_out = epsp if epsp is not None else np.zeros(nel)
    if return_tuple:
        return sign, epsp_out, soundsp
    return sign, epsp_out


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Standard pyradioss plane-stress shell stress update for LAW88."""
    sig_arr = np.asarray(sig, dtype=np.float64)
    deps_arr = np.asarray(deps, dtype=np.float64)
    is_1d = sig_arr.ndim == 1
    ncomp = sig_arr.shape[-1]
    if is_1d:
        sig_arr = sig_arr.reshape((1, ncomp))
        deps_arr = deps_arr.reshape((1, ncomp))
    nel = sig_arr.shape[0]

    mp = MatparamLaw88.from_dict_or_obj(mat)

    uvar = None
    if extra is not None:
        for k in ("uvar88", "uvar"):
            if k in extra and isinstance(extra[k], np.ndarray):
                uvar = extra[k]
                break
    if uvar is None:
        uvar = np.zeros((nel, 30), dtype=np.float64)
    elif uvar.ndim == 1:
        uvar = uvar.reshape((nel, -1))

    soundsp = np.zeros(nel, dtype=np.float64)
    off = np.ones(nel, dtype=np.float64)
    if extra is not None:
        if "off" in extra:
            off[:] = extra["off"]
        elif "off88" in extra:
            off[:] = extra["off88"]
    et = np.zeros(nel, dtype=np.float64)
    epsd = np.zeros(nel, dtype=np.float64)
    vartmp = np.zeros((nel, 6), dtype=int)
    dmg = np.zeros(nel, dtype=np.float64)
    ngl = np.arange(1, nel + 1, dtype=int)

    thk0 = np.ones(nel, dtype=np.float64)
    thkly = np.ones(nel, dtype=np.float64)
    if extra is not None and "thklyl" in extra:
        thk0[:] = extra["thklyl"]
    elif extra is not None and "thk0" in extra:
        thk0[:] = extra["thk0"]
    elif extra is not None and "thk" in extra:
        thk0[:] = extra["thk"]
    if extra is not None and "thkly" in extra:
        thkly[:] = extra["thkly"]
    else:
        thkly[:] = 1.0
    thkn = thk0.copy()
    if extra is not None and "thk88" in extra:
        thkn[:] = extra["thk88"]
    shf = np.ones(nel, dtype=np.float64)

    rho = np.full(nel, mp.rho, dtype=np.float64)
    if extra is not None and "rho" in extra:
        rho[:] = extra["rho"]

    if extra is not None and "eps88" in extra:
        extra["eps88"] += deps_arr[:, :3]
        eps_tot = extra["eps88"]
    elif extra is not None and "eps" in extra:
        eps_tot = np.asarray(extra["eps"], dtype=np.float64)
        if eps_tot.ndim == 1:
            eps_tot = eps_tot.reshape((1, ncomp))
    else:
        eps_tot = deps_arr

    sign = np.zeros((nel, 5), dtype=np.float64)
    sigo = np.zeros((nel, 5), dtype=np.float64)
    sigo[:, :min(ncomp, 3)] = sig_arr[:, :min(ncomp, 3)]
    if ncomp >= 5:
        sigo[:, 3:5] = sig_arr[:, 3:5]

    tstep = dt if dt > 0.0 else 1e-5
    depsyz = deps_arr[:, 3] if ncomp >= 4 else np.zeros(nel)
    depszx = deps_arr[:, 4] if ncomp >= 5 else np.zeros(nel)

    sigeps88_shell(
        nel=nel,
        matparam=mp,
        uvar=uvar,
        tstep=tstep,
        tt=0.0,
        rho=rho,
        soundsp=soundsp,
        off=off,
        ismstr=0,
        israte=0,
        ngl=ngl,
        epsxx=eps_tot[:, 0],
        epsyy=eps_tot[:, 1],
        epsxy=eps_tot[:, 2] if ncomp >= 3 else np.zeros(nel),
        epspxx=np.zeros(nel),
        epspyy=np.zeros(nel),
        epspxy=np.zeros(nel),
        depsxx=deps_arr[:, 0],
        depsyy=deps_arr[:, 1],
        depsxy=deps_arr[:, 2] if ncomp >= 3 else np.zeros(nel),
        depsyz=depsyz,
        depszx=depszx,
        sigoxx=sigo[:, 0],
        sigoyy=sigo[:, 1],
        sigoxy=sigo[:, 2],
        sigoyz=sigo[:, 3],
        sigozx=sigo[:, 4],
        signxx=sign[:, 0],
        signyy=sign[:, 1],
        signxy=sign[:, 2],
        signyz=sign[:, 3],
        signzx=sign[:, 4],
        asrate=0.0,
        et=et,
        epsd=epsd,
        nvartmp=6,
        vartmp=vartmp,
        dmg=dmg,
        thkly=thkly,
        thk0=thk0,
        thkn=thkn,
        shf=shf,
        ipg=1,
        npg=1,
    )

    if extra is not None:
        if "off" in extra:
            extra["off"][:] = off
        if "off88" in extra:
            extra["off88"][:] = off
        if "thk88" in extra:
            extra["thk88"][:] = thkn

    res_sig = sign[:, :ncomp]
    if is_1d:
        res_sig = res_sig[0]
    epsp_out = epsp if epsp is not None else np.zeros(nel)
    return res_sig, epsp_out


consistent_solid_tangent = tangent_law88_solid
consistent_shell_tangent = tangent_law88_shell
shell_membrane_tangent = tangent_law88_shell
solid_update_law88 = solid_update
shell_update_law88 = shell_update
sound_speed_solid_law88 = sound_speed_solid
sound_speed_shell_law88 = sound_speed_shell


def resolve(mat: Any, model: Any, log: Any = None) -> None:
    """Resolve /FUNCT references into TableData objects for /MAT/LAW88.

    Reference: starter/source/materials/mat/mat088/hm_read_mat88.F90:255-469
    """
    p: dict[str, Any] = {}
    if hasattr(mat, "params") and isinstance(mat.params, dict):
        p = mat.params
    elif isinstance(mat, dict):
        p = mat
    elif hasattr(mat, "__dict__"):
        p = mat.__dict__

    def _resolve_one(fid: Any) -> Any:
        if fid == 0 or fid is None:
            return None
        if hasattr(model, "functions") and fid in model.functions:
            return model.functions[fid]
        if hasattr(model, "curves") and fid in model.curves:
            return model.curves[fid]
        if isinstance(model, dict):
            if "functions" in model and fid in model["functions"]:
                return model["functions"][fid]
            if "curves" in model and fid in model["curves"]:
                return model["curves"][fid]
            if fid in model:
                return model[fid]
        return None

    func_load_list = p.get("func_load_list", getattr(mat, "func_load_list", p.get("LAW88_arr1", [])))
    ifunc_unload = p.get("ifunc_unload", getattr(mat, "ifunc_unload", p.get("LAW88_fct_IDunL", 0)))
    fscale_load_list = p.get("fscale_load_list", getattr(mat, "fscale_load_list", p.get("LAW88_arr2", [])))
    rate_load_list = p.get("rate_load_list", getattr(mat, "rate_load_list", p.get("LAW88_arr3", [])))
    lamfit_list = p.get("lamfit_list", getattr(mat, "lamfit_list", p.get("LAW88_LAMFIT", [])))
    sw = float(p.get("sw", getattr(mat, "sw", p.get("LAW88_SW", 0.0))))
    st = float(p.get("st", getattr(mat, "st", p.get("LAW88_ST", 0.0))))
    sgl = float(p.get("sgl", getattr(mat, "sgl", p.get("LAW88_SGL", 0.0))))
    areafac = 1.0 / (sw * st) if (sw > 0.0 and st > 0.0) else 1.0
    lenfac = 1.0 / sgl if sgl > 0.0 else 1.0

    tables: list[TableData] = []

    # 1. Loading curves -> TableData
    nl = len(func_load_list) if func_load_list else 0
    if nl > 0:
        resolved_funcs = [_resolve_one(fid) for fid in func_load_list]
        if any(rf is not None for rf in resolved_funcs):
            rf = resolved_funcs[0]
            if rf is not None:
                if hasattr(rf, "x") and hasattr(rf, "y"):
                    x_raw = np.asarray(rf.x, dtype=np.float64) * lenfac
                    y_raw = np.asarray(rf.y, dtype=np.float64) * (fscale_load_list[0] if len(fscale_load_list) > 0 else 1.0) * areafac
                elif hasattr(rf, "points"):
                    pts = np.asarray(rf.points, dtype=np.float64)
                    x_raw = pts[:, 0] * lenfac
                    y_raw = pts[:, 1] * (fscale_load_list[0] if len(fscale_load_list) > 0 else 1.0) * areafac
                else:
                    x_raw = np.array([0.0, 1.0])
                    y_raw = np.array([0.0, 100.0])

                lam = lamfit_list[0] if len(lamfit_list) > 0 and lamfit_list[0] > 0.0 else 1e-3
                x_sm, y_sm = table_mat_spline_fit(x_raw, y_raw, nout=300, lam=lam)
                stretch_vals = 1.0 + x_sm
                tbl1 = TableData(ndim=1, x1=stretch_vals, y1d=y_sm)
                tables.append(tbl1)

    # 2. Unloading curve if present
    if ifunc_unload > 0:
        rf_unl = _resolve_one(ifunc_unload)
        if rf_unl is not None:
            if hasattr(rf_unl, "x") and hasattr(rf_unl, "y"):
                x_raw = np.asarray(rf_unl.x, dtype=np.float64) * lenfac
                y_raw = np.asarray(rf_unl.y, dtype=np.float64) * float(p.get("fscale_unload", 1.0)) * areafac
            elif hasattr(rf_unl, "points"):
                pts = np.asarray(rf_unl.points, dtype=np.float64)
                x_raw = pts[:, 0] * lenfac
                y_raw = pts[:, 1] * float(p.get("fscale_unload", 1.0)) * areafac
            else:
                x_raw = np.array([0.0, 1.0])
                y_raw = np.array([0.0, 50.0])
            lam = 1e-3
            x_sm, y_sm = table_mat_spline_fit(x_raw, y_raw, nout=300, lam=lam)
            stretch_vals = 1.0 + x_sm
            tbl2 = TableData(ndim=1, x1=stretch_vals, y1d=y_sm)
            tables.append(tbl2)

            if len(tables) > 0:
                tbl3 = TableData(ndim=1, x1=tables[0].x1.copy(), y1d=tables[0].y1d.copy())
                tables.append(tbl3)

    if tables:
        if hasattr(mat, "table"):
            mat.table = tables
        if hasattr(mat, "params") and isinstance(mat.params, dict):
            mat.params["table"] = tables


