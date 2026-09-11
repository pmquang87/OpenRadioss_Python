"""
LAW60 — Tabulated elasto-plastic material model with strain-rate sensitivity,
dynamic modulus degradation, pressure-dependent yield scaling, and tensile failure
(/MAT/LAW60, /MAT/PLAS_T3, /MAT/FABRIC).

Fortran references:
- ``engine/source/materials/mat/mat060/sigeps60.F`` (3D solid continuum kernel)
- ``engine/source/materials/mat/mat060/sigeps60c.F`` (2D shell plane-stress kernel)
- ``engine/source/materials/mat/mat060/sigeps60g.F`` (thick shells / 3D)
- ``starter/source/materials/mat/mat060/hm_read_mat60.F`` (parameter initialization)
- ``config/CFG/radioss110/MAT/matl60_PLAS_T3.cfg`` (keyword format)

Physics overview:
-----------------
1. Deviatoric and volumetric trial stress:
   P0 = -tr(sig_old) / 3 (hydrostatic pressure, compression > 0)
   dav = tr(deps) / 3
   sig_trial = dev(sig_old) + 2*G*(deps - dav*I)
2. Dynamic modulus degradation:
   If ifunce > 0: E_cur = finter(ifunce, epsp_old) * E0
   Else if ce > 0: E_cur = E0 - (E0 - einf) * (1 - exp(-ce * epsp_old))
   Else: E_cur = E0
   G = E_cur / (2 * (1 + nu)), C1 = E_cur / (3 * (1 - 2*nu)), A1 = E_cur / (1 - nu^2)
3. Pressure-dependent yield scaling:
   If ipfun > 0: PFAC = finter(ipfun, P0 * pscale)
   Else: PFAC = 1.0
4. Rate-dependent yield stress and hardening:
   sigma_y(epsp, rate) interpolated across the tabulated curve family
   using INTER_RAT rational interpolation (>= 4 curves) or linear rate interpolation.
5. Tensile failure scaling:
   epst = max principal total strain
   FAIL = max(0, min(1, (eps_t2 - epst) / (eps_t2 - eps_t1)))
   YLD = FAIL * PFAC * sigma_y
   H = FAIL * H_raw
6. Yield check and radial return:
   sigma_vm = sqrt(3 * J2)
   f = sigma_vm - YLD
   If f > 0:
     delta_epsp = f / (3*G + H_iso)
     s_new = s_trial * (YLD / sigma_vm)
     epsp_new = epsp_old + delta_epsp
   If epsp_new >= eps_max:
     off = 0.0, sig_new = 0.0 (element deletion)
7. Layer thinning (shells):
   Delta_eps_zz = Delta_eps_zz_el + Delta_eps_zz_pl
8. Instantaneous sound speeds:
   Solids: c = sqrt((C1 + 4/3*G) / rho0)
   Shells: c = sqrt(A1 / rho0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..common.tables import FunctTable, SmoothFunctTable
from ..model.entities import Material

_EM20 = 1e-20
_EM30 = 1e-30


@dataclass
class Law60Params:
    """Parameters for /MAT/LAW60 (/MAT/PLAS_T3 /MAT/FABRIC).

    Attributes
    ----------
    e0 : float
        Initial Young's modulus E > 0.
    nu : float
        Poisson's ratio 0 <= nu < 0.5.
    eps_max : float
        Maximum failure plastic strain (default 1e30).
    eps_t1 : float
        Tensile failure start strain (default 1e30).
    eps_t2 : float
        Tensile failure rupture strain (default 2e30).
    nfunc : int
        Number of strain-rate functions (1..10).
    fsmooth : int
        Strain rate smoothing flag (0: none, 1: log, 2: exp).
    fisokin : float
        Mixed hardening factor (MAT_HARD in [0, 1], 0: isotropic, 1: kinematic).
    fcut : float
        Strain rate cutoff frequency (ASRATE, default 1e30).
    ipfun : int
        Pressure-yield factor function ID (Xr_fun).
    ifunce : int
        Dynamic modulus function ID (fct_ID_k).
    pscale : float
        Scale factor for pressure (stored as 1/pscale if pscale > 0, else 1.0 or 0.0).
    einf : float
        Asymptotic degraded Young's modulus (E_R).
    ce : float
        Exponential modulus degradation rate (MAT_C1).
    funcs : list
        List of yield stress function IDs or FunctTable/SmoothFunctTable objects.
    fscale_arr : list
        List of scale factors for yield stress functions (MAT_ALPHA1..10).
    rate_arr : list
        List of strain rates (MAT_EPSR1..10).
    """

    e0: float = 210000.0
    nu: float = 0.3
    eps_max: float = 1e30
    eps_t1: float = 1e30
    eps_t2: float = 2e30
    nfunc: int = 1
    fsmooth: int = 0
    fisokin: float = 0.0
    fcut: float = 1e30
    ipfun: int = 0
    ifunce: int = 0
    pscale: float = 0.0
    einf: float = 0.0
    ce: float = 0.0
    funcs: list = field(default_factory=list)
    fscale_arr: list = field(default_factory=list)
    rate_arr: list = field(default_factory=list)

    # Identifiers and reference density
    id: int = 1
    law: int = 60
    law_name: str = "LAW60"
    rho0: float = 1.0
    rhor: float = 0.0
    title: str = ""

    # Pre-processed curve arrays for high-performance evaluation
    curve_x: list = field(default_factory=list)
    curve_y: list = field(default_factory=list)
    curve_s: list = field(default_factory=list)
    rates: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))

    # Optional pressure function curve arrays
    p_curve_x: Optional[np.ndarray] = None
    p_curve_y: Optional[np.ndarray] = None
    p_curve_s: Optional[np.ndarray] = None

    # Optional modulus function curve arrays
    e_curve_x: Optional[np.ndarray] = None
    e_curve_y: Optional[np.ndarray] = None
    e_curve_s: Optional[np.ndarray] = None

    # Helper elastic constants
    G0: float = field(init=False)
    C1_0: float = field(init=False)

    def __post_init__(self) -> None:
        if self.e0 <= 0.0:
            raise ValueError(f"/MAT/LAW60: Young's modulus E must be > 0 (got {self.e0})")
        if not (0.0 <= self.nu < 0.5):
            raise ValueError(f"/MAT/LAW60: Poisson's ratio must satisfy 0 <= nu < 0.5 (got {self.nu})")

        self.G0 = self.e0 / (2.0 * (1.0 + self.nu))
        self.C1_0 = self.e0 / (3.0 * (1.0 - 2.0 * self.nu))

        if not isinstance(self.rates, np.ndarray):
            self.rates = np.asarray(self.rates, dtype=float)

    @property
    def E(self) -> float:
        return self.e0

    @property
    def G(self) -> float:
        return self.G0

    @property
    def K(self) -> float:
        return self.C1_0

    @property
    def C1(self) -> float:
        return self.C1_0

    @property
    def params(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rho0": self.rho0,
            "rhor": self.rhor,
            "title": self.title,
            "E": self.e0,
            "e0": self.e0,
            "nu": self.nu,
            "G": self.G0,
            "K": self.C1_0,
            "C1": self.C1_0,
            "eps_max": self.eps_max,
            "eps_p_max": self.eps_max,
            "eps_t1": self.eps_t1,
            "eps_t2": self.eps_t2,
            "nfunc": self.nfunc,
            "fsmooth": self.fsmooth,
            "fisokin": self.fisokin,
            "fcut": self.fcut,
            "ipfun": self.ipfun,
            "ifunce": self.ifunce,
            "pscale": self.pscale,
            "einf": self.einf,
            "ce": self.ce,
            "funcs": self.funcs,
            "fscale_arr": self.fscale_arr,
            "rate_arr": self.rate_arr,
            "curve_x": self.curve_x,
            "curve_y": self.curve_y,
            "curve_s": self.curve_s,
            "rates": self.rates,
        }

    def sound_speed_solid(self, rho: float | None = None, epsp: float = 0.0) -> float:
        rho_val = float(rho) if rho is not None and rho > 0.0 else self.rho0
        _, g_cur, c1_cur, _ = _current_moduli(self, np.array([epsp]))
        return float(np.sqrt((c1_cur[0] + (4.0 / 3.0) * g_cur[0]) / max(rho_val, _EM20)))

    def sound_speed_shell(self, rho: float | None = None, epsp: float = 0.0) -> float:
        rho_val = float(rho) if rho is not None and rho > 0.0 else self.rho0
        _, _, _, a1_cur = _current_moduli(self, np.array([epsp]))
        return float(np.sqrt(a1_cur[0] / max(rho_val, _EM20)))


# ----------------------------------------------------------------------------
# Curve Evaluation & Rational Interpolation Helpers
# ----------------------------------------------------------------------------

def _curve_eval(cx: np.ndarray, cy: np.ndarray, cs: np.ndarray,
                e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear evaluation with end-slope extrapolation.

    Returns (value, slope).
    """
    e_arr = np.asarray(e, dtype=float)
    if len(cx) < 2:
        val = np.full_like(e_arr, cy[0] if len(cy) > 0 else 0.0)
        slp = np.zeros_like(e_arr)
        return val, slp

    i = np.minimum(np.maximum(np.searchsorted(cx, e_arr, side="right") - 1, 0), len(cx) - 2)
    val = cy[i] + cs[i] * (e_arr - cx[i])
    slp = cs[i]
    return val, slp


def _inter_rat(x0: float, x1: float, x2: float, x3: float,
               y0: float, y1: float, y2: float, y3: float,
               x: float, i: int, n: int) -> tuple[float, float]:
    """Port of INTER_RAT from engine/source/materials/mat/mat060/sigeps60c.F.

    Evaluates rational function interpolation across four points (x0..x3, y0..y3)
    at query point x.
    i is the 1-based active interval index (1..n-1), where n is total curves.
    Returns (y, yp) where y is interpolated value and yp is derivative wrt x.
    """
    q = x - x1
    d = x2 - x1
    r = d - q
    s = (y2 - y1) / d if abs(d) > _EM30 else 0.0

    d_x3_x2 = x3 - x2
    sp = (y3 - y2) / d_x3_x2 if abs(d_x3_x2) > _EM30 else 0.0

    d_x3_x1 = x3 - x1
    c2 = (sp - s) / d_x3_x1 if abs(d_x3_x1) > _EM30 else 0.0

    dm = x1 - x0
    dm_sign = 1.0 if dm >= 0.0 else -1.0
    dm = dm_sign * max(_EM30, abs(dm))
    sm = (y1 - y0) / dm

    d_plus_dm = d + dm
    c1 = (s - sm) / d_plus_dm if abs(d_plus_dm) > _EM30 else 0.0

    if i == 1:
        if x <= x0:
            c1 = 0.0
            c2 = 0.0
            sm = 0.0
        else:
            c2 = c1
            c1 = sm / (x1 - x0) if abs(x1 - x0) > _EM30 else 0.0
        r = x1 - x
        q = x - x0
        d = x1 - x0
        c3 = abs(c2 * r)
        c3d = c3 + abs(c1 * q)
        c5 = (c3 / c3d) * (c1 - c2) if c3d > 0.0 else 0.0
        c4 = c2 + c5
        c6 = d * c5 * (1.0 - (c3 / c3d if c3d > 0.0 else 0.0))
        y = y0 + q * (sm - r * c4)
        yp = sm + (q - r) * c4 + c6
    elif i == n - 1:
        if sp == 0.0 or x > x3:
            c1 = 0.0
            c2 = 0.0
        else:
            c1 = (sp - s) / (x3 - x1) if abs(x3 - x1) > _EM30 else 0.0
            c2 = 0.0
        r = x3 - x
        q = x - x2
        d = x3 - x2
        c3 = abs(c2 * r)
        c3d = c3 + abs(c1 * q)
        c5 = (c3 / c3d) * (c1 - c2) if c3d > 0.0 else 0.0
        c4 = c2 + c5
        c6 = d * c5 * (1.0 - (c3 / c3d if c3d > 0.0 else 0.0))
        y = y2 + (x - x2) * (sp - (x3 - x) * c4)
        yp = sp + (q - r) * c4 + c6
    else:
        if i == 2 and sm * (sm - dm * c1) <= 0.0:
            c1 = (s - sm - sm) / d if abs(d) > _EM30 else 0.0
        c3 = abs(c2 * r)
        c3d = c3 + abs(c1 * q)
        c5 = (c3 / c3d) * (c1 - c2) if c3d > 0.0 else 0.0
        c4 = c2 + c5
        c6 = d * c5 * (1.0 - (c3 / c3d if c3d > 0.0 else 0.0))
        y = y1 + q * (s - r * c4)
        yp = s + (q - r) * c4 + c6

    return y, yp


def _current_moduli(params: Law60Params,
                    epsp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate current degraded Young's modulus E_cur, shear modulus G_cur,
    bulk modulus C1_cur, and plane-stress modulus A1_cur.

    Matches sigeps60.F:236-258 and sigeps60c.F:226-254.
    """
    e0 = params.e0
    nu = params.nu
    n = len(epsp)

    if params.ifunce > 0 and params.e_curve_x is not None and len(params.e_curve_x) > 0:
        scale, _ = _curve_eval(params.e_curve_x, params.e_curve_y, params.e_curve_s, epsp)
        e_cur = np.maximum(scale * e0, _EM20)
    elif params.ce > 0.0:
        deg = (e0 - params.einf) * (1.0 - np.exp(-params.ce * np.maximum(epsp, 0.0)))
        e_cur = np.maximum(e0 - deg, _EM20)
    else:
        e_cur = np.full(n, e0, dtype=float)

    g_cur = e_cur / (2.0 * (1.0 + nu))
    c1_cur = e_cur / (3.0 * (1.0 - 2.0 * nu))
    a1_cur = e_cur / (1.0 - nu * nu)
    return e_cur, g_cur, c1_cur, a1_cur


def _pressure_factor(params: Law60Params, p0: np.ndarray) -> np.ndarray:
    """Calculate pressure-dependent yield scaling factor PFAC.
    P0 is hydrostatic pressure (positive in compression).

    Matches sigeps60.F:444-468, 520.
    """
    if params.ipfun > 0 and params.p_curve_x is not None and len(params.p_curve_x) > 0:
        x_val = p0 * params.pscale
        pfac, _ = _curve_eval(params.p_curve_x, params.p_curve_y, params.p_curve_s, x_val)
        return np.maximum(pfac, 0.0)
    return np.ones_like(p0, dtype=float)


def _tensile_failure_factor(params: Law60Params, epst: np.ndarray) -> np.ndarray:
    """Calculate tensile damage failure factor FAIL in [0, 1].

    Matches sigeps60.F:351-352:
    FAIL = MAX(0, MIN(1, (eps_t2 - epst) / (eps_t2 - eps_t1)))
    """
    t1 = params.eps_t1
    t2 = params.eps_t2
    if t2 <= t1 or t1 >= 1e29:
        return np.ones_like(epst, dtype=float)
    return np.clip((t2 - epst) / (t2 - t1), 0.0, 1.0)


def _principal_strain(eps: np.ndarray) -> np.ndarray:
    """Max principal strain from total 3D strain tensor via 4-iteration Newton solve.

    Matches sigeps60.F:305-348 and law44_cowper._principal_strain.
    """
    dav = (eps[:, 0] + eps[:, 1] + eps[:, 2]) / 3.0
    e1 = eps[:, 0] - dav
    e2 = eps[:, 1] - dav
    e3 = eps[:, 2] - dav
    e4 = 0.5 * eps[:, 3]
    e5 = 0.5 * eps[:, 4]
    e6 = 0.5 * eps[:, 5]
    c = -(e1 ** 2 + e2 ** 2 + e3 ** 2 + e4 ** 2 + e5 ** 2 + e6 ** 2)
    d = (
        -(e1 * e2 * e3)
        + e1 * e5 ** 2
        + e2 * e6 ** 2
        + e3 * e4 ** 2
        - 2.0 * e4 * e5 * e6
    )
    epst = np.sqrt(np.maximum(-c / 3.0, 0.0))
    y = (epst ** 2 + c) * epst + d
    active = np.abs(y) > 1e-8
    x = np.where(active, 1.75 * epst, epst)
    for _ in range(4):
        y = (x ** 2 + c) * x + d
        yp = 3.0 * x ** 2 + c
        denom = np.where(yp == 0.0, 1.0, yp)
        x = np.where(active & (yp != 0.0), x - y / denom, x)
    return np.where(active, x + dav, epst)


def _principal_strain_2d(eps: np.ndarray) -> np.ndarray:
    """Max in-plane principal strain for 2D shells.

    Matches sigeps60c.F:330-333:
    EPST = 0.5 * (eps_xx + eps_yy + sqrt((eps_xx - eps_yy)^2 + eps_xy^2))
    """
    exx = eps[:, 0]
    eyy = eps[:, 1]
    exy = eps[:, 2]
    return 0.5 * (exx + eyy + np.sqrt((exx - eyy) ** 2 + exy ** 2))


def _eval_yield_and_hardening(params: Law60Params, epsp: np.ndarray,
                              rate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate yield stress sigma_y and hardening slope H at (epsp, rate).

    Interpolates across the tabulated curve family per sigeps60.F:406-525.
    """
    cxs = params.curve_x
    cys = params.curve_y
    css = params.curve_s
    nfun = len(cxs)
    n = len(epsp)

    if nfun == 0:
        # Fallback if no curves loaded
        return np.full(n, 1e9, dtype=float), np.zeros(n, dtype=float)

    if nfun == 1:
        return _curve_eval(cxs[0], cys[0], css[0], epsp)

    rates = params.rates
    vals = np.empty((nfun, n), dtype=float)
    slps = np.empty((nfun, n), dtype=float)
    for k in range(nfun):
        vals[k], slps[k] = _curve_eval(cxs[k], cys[k], css[k], epsp)

    if nfun < 4 or params.fsmooth != 0:
        # Piecewise linear or log interpolation across rates
        j = np.clip(np.searchsorted(rates, rate, side="right") - 1, 0, nfun - 2)
        if params.fsmooth == 1:
            # Log smoothing
            r_clamp = np.maximum(rate, 1e-10)
            r0 = np.maximum(rates[j], 1e-10)
            r1 = np.maximum(rates[j + 1], 1e-10)
            denom = np.maximum(np.log(r1 / r0), 1e-20)
            w = np.clip(np.log(r_clamp / r0) / denom, 0.0, 1.0)
        else:
            denom = np.maximum(rates[j + 1] - rates[j], 1e-20)
            w = np.clip((rate - rates[j]) / denom, 0.0, 1.0)

        cols = np.arange(n)
        sy = (1.0 - w) * vals[j, cols] + w * vals[j + 1, cols]
        h = (1.0 - w) * slps[j, cols] + w * slps[j + 1, cols]
        return sy, h

    # 4 or more curves with fsmooth == 0: use Fortran INTER_RAT rational interpolation
    sy = np.empty(n, dtype=float)
    h = np.empty(n, dtype=float)

    for i in range(n):
        r_val = rate[i]
        # Find JJ such that rates[JJ] <= r_val
        jj = 0
        for j in range(1, nfun - 1):
            if r_val >= rates[j]:
                jj = j

        if jj == 0:
            j1, j2, j3, j4 = 0, 1, 2, 3
            irat = 1
            fac = (r_val - rates[0]) / max(rates[1] - rates[0], _EM20)
            fac = max(0.0, min(1.0, fac))
            h[i] = slps[0, i] + fac * (slps[1, i] - slps[0, i])
        elif jj == nfun - 2:
            j1, j2, j3, j4 = nfun - 4, nfun - 3, nfun - 2, nfun - 1
            irat = nfun - 1
            fac = (r_val - rates[nfun - 2]) / max(rates[nfun - 1] - rates[nfun - 2], _EM20)
            fac = max(0.0, min(1.0, fac))
            h[i] = slps[nfun - 2, i] + fac * (slps[nfun - 1, i] - slps[nfun - 2, i])
        else:
            j1, j2, j3, j4 = jj - 1, jj, jj + 1, jj + 2
            irat = jj + 1
            fac = (r_val - rates[jj]) / max(rates[jj + 1] - rates[jj], _EM20)
            fac = max(0.0, min(1.0, fac))
            h[i] = slps[jj, i] + fac * (slps[jj + 1, i] - slps[jj, i])

        x1, x2, x3, x4 = rates[j1], rates[j2], rates[j3], rates[j4]
        y1, y2, y3, y4 = vals[j1, i], vals[j2, i], vals[j3, i], vals[j4, i]
        y_val, _ = _inter_rat(x1, x2, x3, x4, y1, y2, y3, y4, r_val, irat, nfun)
        sy[i] = y_val

    return sy, h


def _parse_update_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, Any, float, dict[str, Any] | None]:
    """Parse flexible calling patterns for solid_update and shell_update."""
    eps = kwargs.pop("eps", None)
    deps = kwargs.pop("deps", None)
    sig_old = kwargs.pop("sig_old", kwargs.pop("sig", None))
    epsp_old = kwargs.pop("epsp_old", kwargs.pop("epsp", None))
    dt = float(kwargs.pop("dt", 0.0))
    extra = kwargs.pop("extra", None)

    if len(args) > 0:
        if args[0] is None:
            # Called as (mat, None, deps, sig, epsp, dt, extra)
            if len(args) > 1 and deps is None:
                deps = args[1]
            if len(args) > 2 and sig_old is None:
                sig_old = args[2]
            if len(args) > 3 and epsp_old is None:
                epsp_old = args[3]
            if len(args) > 4:
                dt = float(args[4])
            if len(args) > 5 and extra is None:
                extra = args[5]
        else:
            # Called as (mat, sig, deps, epsp, dt, extra)
            if sig_old is None:
                sig_old = args[0]
            if len(args) > 1 and deps is None:
                deps = args[1]
            if len(args) > 2 and epsp_old is None:
                epsp_old = args[2]
            if len(args) > 3:
                dt = float(args[3])
            if len(args) > 4 and extra is None:
                extra = args[4]

    return eps, deps, sig_old, epsp_old, dt, extra


# ----------------------------------------------------------------------------
# 3D Solid Continuum Kernel
# ----------------------------------------------------------------------------

def solid_update(
    mat_params: Any,
    *args: Any,
    eps: np.ndarray | None = None,
    deps: np.ndarray | None = None,
    sig_old: np.ndarray | None = None,
    epsp_old: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3D solid continuum radial-return stress update for /MAT/LAW60.

    Parameters
    ----------
    mat_params : Law60Params or Material or dict
        Constitutive model parameters.
    sig_old : np.ndarray
        Previous Cauchy stress tensor (NEL, 6).
    deps : np.ndarray
        Strain increment tensor (NEL, 6) in Voigt engineering shear [xx, yy, zz, xy, yz, zx].
    epsp_old : np.ndarray, optional
        Previous equivalent plastic strain (NEL,).
    dt : float, optional
        Time step increment.
    extra : dict, optional
        State variables ('rho', 'off', 'epsd60', etc.).

    Returns
    -------
    sig_new : np.ndarray (NEL, 6)
        Updated Cauchy stress tensor.
    epsp_new : np.ndarray (NEL,)
        Updated equivalent plastic strain.
    sound_speed : np.ndarray (NEL,)
        Instantaneous longitudinal sound speed.
    """
    params = mat_params if (isinstance(mat_params, Law60Params) and len(mat_params.curve_x) > 0) else build_law60(mat_params)

    kw = dict(kwargs)
    if eps is not None:
        kw["eps"] = eps
    if deps is not None:
        kw["deps"] = deps
    if sig_old is not None:
        kw["sig_old"] = sig_old
    if epsp_old is not None:
        kw["epsp_old"] = epsp_old
    if dt != 0.0:
        kw["dt"] = dt
    if extra is not None:
        kw["extra"] = extra

    eps, deps, sig_old, epsp_old, dt, extra = _parse_update_args(args, kw)

    if sig_old is None:
        raise ValueError("solid_update requires stress tensor sig_old")
    if deps is None:
        raise ValueError("solid_update requires strain increment deps")

    sig_arr = np.asarray(sig_old, dtype=float).copy()
    single_element = (sig_arr.ndim == 1)
    if single_element:
        sig_arr = sig_arr.reshape(1, 6)

    nel = sig_arr.shape[0]
    deps_arr = np.asarray(deps, dtype=float)
    if deps_arr.ndim == 1:
        deps_arr = deps_arr.reshape(1, 6)

    if epsp_old is None:
        epsp_arr = np.zeros(nel, dtype=float)
    else:
        epsp_arr = np.asarray(epsp_old, dtype=float).copy().reshape(nel)

    if nel == 0:
        return sig_arr, epsp_arr, np.empty(0, dtype=float)

    # 1. Dynamic modulus degradation
    e_cur, g_cur, c1_cur, _ = _current_moduli(params, epsp_arr)

    # 2. Deviatoric and volumetric trial stress (sigeps60.F:286-293)
    p0 = -(sig_arr[:, 0] + sig_arr[:, 1] + sig_arr[:, 2]) / 3.0  # compression > 0
    dav = (deps_arr[:, 0] + deps_arr[:, 1] + deps_arr[:, 2]) / 3.0

    s_trial = np.empty_like(sig_arr)
    s_trial[:, 0] = sig_arr[:, 0] + p0 + 2.0 * g_cur * (deps_arr[:, 0] - dav)
    s_trial[:, 1] = sig_arr[:, 1] + p0 + 2.0 * g_cur * (deps_arr[:, 1] - dav)
    s_trial[:, 2] = sig_arr[:, 2] + p0 + 2.0 * g_cur * (deps_arr[:, 2] - dav)
    s_trial[:, 3] = sig_arr[:, 3] + g_cur * deps_arr[:, 3]
    s_trial[:, 4] = sig_arr[:, 4] + g_cur * deps_arr[:, 4]
    s_trial[:, 5] = sig_arr[:, 5] + g_cur * deps_arr[:, 5]

    # 3. Equivalent von Mises trial stress
    j2 = (
        0.5 * (s_trial[:, 0] ** 2 + s_trial[:, 1] ** 2 + s_trial[:, 2] ** 2)
        + s_trial[:, 3] ** 2
        + s_trial[:, 4] ** 2
        + s_trial[:, 5] ** 2
    )
    sig_vm = np.sqrt(3.0 * j2)

    # 4. Equivalent deviatoric strain rate of increment
    exx = deps_arr[:, 0] - dav
    eyy = deps_arr[:, 1] - dav
    ezz = deps_arr[:, 2] - dav
    ee = (
        exx ** 2
        + eyy ** 2
        + ezz ** 2
        + 0.5 * (deps_arr[:, 3] ** 2 + deps_arr[:, 4] ** 2 + deps_arr[:, 5] ** 2)
    )
    raw_rate = np.sqrt((2.0 / 3.0) * ee) / dt if dt > 0.0 else np.zeros(nel, dtype=float)

    if params.fcut < 1e20 and dt > 0.0:
        asrate = 2.0 * np.pi * params.fcut
        alpha = min(1.0, asrate * dt)
        if extra is not None and "epsd60" in extra:
            extra["epsd60"][:] = alpha * raw_rate + (1.0 - alpha) * extra["epsd60"]
            rate = extra["epsd60"].copy()
        else:
            rate = raw_rate
    else:
        rate = raw_rate

    # 5. Pressure factor PFAC
    pfac = _pressure_factor(params, p0)

    # 6. Tabulated rate-dependent yield stress and hardening
    sy_rate, h_rate = _eval_yield_and_hardening(params, epsp_arr, rate)

    # 7. Tensile failure factor FAIL
    if eps is not None:
        eps_tot = np.asarray(eps, dtype=float) + deps_arr
        if eps_tot.ndim == 1:
            eps_tot = eps_tot.reshape(1, 6)
    else:
        eps_tot = deps_arr
    epst = _principal_strain(eps_tot)
    fail = _tensile_failure_factor(params, epst)

    yld = fail * pfac * sy_rate
    h_eff = fail * h_rate

    # 8. Yield check & radial return (sigeps60.F:530-546)
    h_iso = (1.0 - params.fisokin) * h_eff
    denom = np.maximum(3.0 * g_cur + h_iso, _EM20)
    f = sig_vm - yld
    plastic = f > 0.0

    delta_epsp = np.where(plastic, f / denom, 0.0)
    epsp_new = epsp_arr + delta_epsp

    scale = np.where(plastic & (sig_vm > _EM20), yld / np.maximum(sig_vm, _EM20), 1.0)
    s_new = s_trial * scale[:, None]

    # Pressure update
    if extra is not None and "rho" in extra:
        rho = extra["rho"]
        rho_val = rho if isinstance(rho, np.ndarray) else np.full(nel, float(rho))
        p_new = -c1_cur * (rho_val / max(params.rho0, _EM20) - 1.0)
    elif extra is not None and "amu" in extra:
        p_new = -c1_cur * extra["amu"]
    else:
        # Hypoelastic trace update
        p_new = -p0 + c1_cur * 3.0 * dav

    sig_new = s_new.copy()
    sig_new[:, 0] += p_new
    sig_new[:, 1] += p_new
    sig_new[:, 2] += p_new

    # 9. Element deletion
    deleted = epsp_new >= params.eps_max
    if np.any(deleted):
        sig_new[deleted] = 0.0
        if extra is not None:
            if "off" in extra:
                extra["off"][deleted] = 0.0
            if "off60" in extra:
                extra["off60"][deleted] = 0.0

    # 10. Instantaneous sound speed
    sound_speed = np.sqrt((c1_cur + (4.0 / 3.0) * g_cur) / max(params.rho0, _EM20))

    if single_element:
        return sig_new[0], epsp_new[0], sound_speed[0]
    return sig_new, epsp_new, sound_speed


# ----------------------------------------------------------------------------
# 2D Shell Plane-Stress Kernel
# ----------------------------------------------------------------------------

def shell_update(
    mat_params: Any,
    *args: Any,
    eps: np.ndarray | None = None,
    deps: np.ndarray | None = None,
    sig_old: np.ndarray | None = None,
    epsp_old: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D shell plane-stress radial-projection stress update for /MAT/LAW60.

    Parameters
    ----------
    mat_params : Law60Params or Material or dict
        Constitutive model parameters.
    sig_old : np.ndarray
        Previous plane-stress tensor (NEL, 3) = [xx, yy, xy].
    deps : np.ndarray
        Strain increment tensor (NEL, 3) in Voigt engineering shear [xx, yy, xy].
    epsp_old : np.ndarray, optional
        Previous equivalent plastic strain (NEL,).
    dt : float, optional
        Time step increment.
    extra : dict, optional
        State variables ('off', 'thk', etc.).

    Returns
    -------
    sig_new : np.ndarray (NEL, 3)
        Updated plane-stress tensor.
    epsp_new : np.ndarray (NEL,)
        Updated equivalent plastic strain.
    sound_speed : np.ndarray (NEL,)
        Instantaneous shell membrane sound speed.
    """
    params = mat_params if (isinstance(mat_params, Law60Params) and len(mat_params.curve_x) > 0) else build_law60(mat_params)

    kw = dict(kwargs)
    if eps is not None:
        kw["eps"] = eps
    if deps is not None:
        kw["deps"] = deps
    if sig_old is not None:
        kw["sig_old"] = sig_old
    if epsp_old is not None:
        kw["epsp_old"] = epsp_old
    if dt != 0.0:
        kw["dt"] = dt
    if extra is not None:
        kw["extra"] = extra

    eps, deps, sig_old, epsp_old, dt, extra = _parse_update_args(args, kw)

    if sig_old is None:
        raise ValueError("shell_update requires stress tensor sig_old")
    if deps is None:
        raise ValueError("shell_update requires strain increment deps")

    sig_arr = np.asarray(sig_old, dtype=float).copy()
    single_element = (sig_arr.ndim == 1)
    if single_element:
        sig_arr = sig_arr.reshape(1, 3)

    nel = sig_arr.shape[0]
    deps_arr = np.asarray(deps, dtype=float)
    if deps_arr.ndim == 1:
        deps_arr = deps_arr.reshape(1, 3)

    if epsp_old is None:
        epsp_arr = np.zeros(nel, dtype=float)
    else:
        epsp_arr = np.asarray(epsp_old, dtype=float).copy().reshape(nel)

    if nel == 0:
        return sig_arr, epsp_arr, np.empty(0, dtype=float)

    # 1. Dynamic modulus degradation
    e_cur, g_cur, _, a1_cur = _current_moduli(params, epsp_arr)
    a2_cur = params.nu * a1_cur

    # 2. Elastic trial update (sigeps60c.F:303-305)
    sig_tr = np.empty_like(sig_arr)
    sig_tr[:, 0] = sig_arr[:, 0] + a1_cur * deps_arr[:, 0] + a2_cur * deps_arr[:, 1]
    sig_tr[:, 1] = sig_arr[:, 1] + a2_cur * deps_arr[:, 0] + a1_cur * deps_arr[:, 1]
    sig_tr[:, 2] = sig_arr[:, 2] + g_cur * deps_arr[:, 2]

    # 3. Plane-stress von Mises stress (sigeps60c.F:593-596)
    sxx = sig_tr[:, 0]
    syy = sig_tr[:, 1]
    sxy = sig_tr[:, 2]
    svm2 = sxx ** 2 + syy ** 2 - sxx * syy + 3.0 * sxy ** 2
    sig_vm = np.sqrt(np.maximum(svm2, 0.0))

    # 4. In-plane strain rate
    dxx = deps_arr[:, 0]
    dyy = deps_arr[:, 1]
    dxy = deps_arr[:, 2]
    dzz_est = -0.5 * (dxx + dyy)
    tr3 = (dxx + dyy + dzz_est) / 3.0
    ee = (dxx - tr3) ** 2 + (dyy - tr3) ** 2 + (dzz_est - tr3) ** 2 + 0.5 * dxy ** 2
    raw_rate = np.sqrt((2.0 / 3.0) * ee) / dt if dt > 0.0 else np.zeros(nel, dtype=float)

    if params.fcut < 1e20 and dt > 0.0:
        asrate = 2.0 * np.pi * params.fcut
        alpha = min(1.0, asrate * dt)
        if extra is not None and "epsd60" in extra:
            extra["epsd60"][:] = alpha * raw_rate + (1.0 - alpha) * extra["epsd60"]
            rate = extra["epsd60"].copy()
        else:
            rate = raw_rate
    else:
        rate = raw_rate

    # 5. Tabulated rate-dependent yield stress and hardening
    sy_rate, h_rate = _eval_yield_and_hardening(params, epsp_arr, rate)

    # 6. In-plane tensile failure factor FAIL
    if eps is not None:
        eps_tot = np.asarray(eps, dtype=float) + deps_arr
        if eps_tot.ndim == 1:
            eps_tot = eps_tot.reshape(1, 3)
    else:
        eps_tot = deps_arr
    epst = _principal_strain_2d(eps_tot)
    fail = _tensile_failure_factor(params, epst)

    yld = fail * sy_rate
    h_eff = fail * h_rate

    # 7. Yield check and radial projection (sigeps60c.F:597-609)
    h_iso = (1.0 - params.fisokin) * h_eff
    denom = np.maximum(3.0 * g_cur + h_iso, _EM20)
    f = sig_vm - yld
    plastic = f > 0.0

    delta_epsp = np.where(plastic, f / denom, 0.0)
    epsp_new = epsp_arr + delta_epsp

    scale = np.where(plastic & (sig_vm > _EM20), yld / np.maximum(sig_vm, _EM20), 1.0)
    sig_new = sig_tr * scale[:, None]

    # 8. Layer thinning update Delta_eps_zz
    nnu11 = params.nu / (1.0 - params.nu)
    nu31 = (1.0 - 2.0 * params.nu) / (1.0 - params.nu)
    dezz_el = -(dxx + dyy) * nnu11
    s_mean = 0.5 * (sig_new[:, 0] + sig_new[:, 1])
    dezz_pl = -delta_epsp * s_mean / np.maximum(yld, _EM20)
    dezz = dezz_el + nu31 * dezz_pl

    if extra is not None and "thk" in extra:
        extra["thk"] += dezz * extra["thk"]

    # 9. Element deletion
    deleted = epsp_new >= params.eps_max
    if np.any(deleted):
        sig_new[deleted] = 0.0
        if extra is not None:
            if "off" in extra:
                extra["off"][deleted] = 0.0
            if "off60" in extra:
                extra["off60"][deleted] = 0.0
            if "layf" in extra:
                extra["layf"][deleted] = 0.0
            if "layfail" in extra:
                extra["layfail"][deleted] = 0.0

    # 10. Shell sound speed
    sound_speed = np.sqrt(a1_cur / max(params.rho0, _EM20))

    if single_element:
        return sig_new[0], epsp_new[0], sound_speed[0]
    return sig_new, epsp_new, sound_speed


# ----------------------------------------------------------------------------
# Algorithmic Consistent Tangent Stiffness Tensors
# ----------------------------------------------------------------------------

def consistent_solid_tangent(
    mat_params: Any,
    eps: np.ndarray | None = None,
    deps: np.ndarray | None = None,
    sig: np.ndarray | None = None,
    epsp: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict[str, Any] | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Algorithmic consistent elastoplastic solid tangent tensor (NEL, 6, 6).

    Follows Simo & Hughes Box 7.3 J2 radial-return tangent algebra.
    """
    params = mat_params if isinstance(mat_params, Law60Params) else build_law60(mat_params)

    # Disambiguate arguments: (mat, sig, epsp, epsp_incr) vs (mat, eps, deps, sig, epsp, dt, extra)
    epsp_incr = kwargs.get("epsp_incr", None)
    if eps is not None and eps.shape[-1] == 6 and sig is None:
        sig = eps
        if deps is not None and deps.ndim == 1:
            epsp = deps
        if epsp_old := kwargs.get("epsp_old", None):
            epsp = epsp_old

    if sig is None:
        raise ValueError("consistent_solid_tangent requires stress tensor sig")

    sig_arr = np.asarray(sig, dtype=float)
    single_element = (sig_arr.ndim == 1)
    if single_element:
        sig_arr = sig_arr.reshape(1, 6)

    nel = sig_arr.shape[0]
    if nel == 0:
        return np.empty((0, 6, 6), dtype=float)

    if epsp is None:
        epsp_arr = np.zeros(nel, dtype=float)
    else:
        epsp_arr = np.asarray(epsp, dtype=float).reshape(nel)

    e_cur, g_cur, c1_cur, _ = _current_moduli(params, epsp_arr)

    # Build per-element elastic tangent C_el
    D = np.zeros((nel, 6, 6), dtype=float)
    for i in range(nel):
        g = g_cur[i]
        k = c1_cur[i]
        c11 = k + (4.0 / 3.0) * g
        c12 = k - (2.0 / 3.0) * g
        D[i, 0:3, 0:3] = c12
        np.fill_diagonal(D[i, 0:3, 0:3], c11)
        D[i, 3, 3] = g
        D[i, 4, 4] = g
        D[i, 5, 5] = g

    # Static rate 0 curve evaluation
    _, h_raw = _eval_yield_and_hardening(params, epsp_arr, np.zeros(nel))

    for i in range(nel):
        s = sig_arr[i].copy()
        p = (s[0] + s[1] + s[2]) / 3.0
        s[0] -= p
        s[1] -= p
        s[2] -= p
        snorm = np.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2 + 2.0 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2))
        q = np.sqrt(1.5) * snorm
        if epsp_incr is not None:
            dep = float(epsp_incr[i]) if hasattr(epsp_incr, "__getitem__") else float(epsp_incr)
        elif deps is not None and deps.ndim == 1:
            dep = float(deps[i])
        else:
            dep = 0.0
        if dep <= 0.0:
            continue

        g = g_cur[i]
        k = c1_cur[i]
        nv = s / max(snorm, _EM30)
        q_tr = q + 3.0 * g * dep
        h = max(h_raw[i], -2.97 * g)
        hd = max(3.0 * g + h, 0.03 * g)

        a = 3.0 * g * dep / max(q_tr, _EM30)
        b = 6.0 * g * g * (dep / max(q_tr, _EM30) - 1.0 / hd)

        # 2G I_dev in Voigt engineering shear
        c_dev = np.zeros((6, 6), dtype=float)
        c_dev[0:3, 0:3] = -2.0 * g / 3.0
        np.fill_diagonal(c_dev[0:3, 0:3], 4.0 * g / 3.0)
        c_dev[3, 3] = g
        c_dev[4, 4] = g
        c_dev[5, 5] = g

        nn = np.outer(nv, nv)
        D[i] -= a * c_dev - b * nn

    if single_element:
        return D[0]
    return D


def consistent_shell_tangent(
    mat_params: Any,
    eps: np.ndarray | None = None,
    deps: np.ndarray | None = None,
    sig: np.ndarray | None = None,
    epsp: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict[str, Any] | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Algorithmic consistent plane-stress tangent tensor (NEL, 3, 3)."""
    params = mat_params if isinstance(mat_params, Law60Params) else build_law60(mat_params)

    epsp_incr = kwargs.get("epsp_incr", None)
    if eps is not None and eps.shape[-1] == 3 and sig is None:
        sig = eps
        if deps is not None and deps.ndim == 1:
            epsp = deps

    if sig is None:
        raise ValueError("consistent_shell_tangent requires stress tensor sig")

    sig_arr = np.asarray(sig, dtype=float)
    single_element = (sig_arr.ndim == 1)
    if single_element:
        sig_arr = sig_arr.reshape(1, 3)

    nel = sig_arr.shape[0]
    if nel == 0:
        return np.empty((0, 3, 3), dtype=float)

    if epsp is None:
        epsp_arr = np.zeros(nel, dtype=float)
    else:
        epsp_arr = np.asarray(epsp, dtype=float).reshape(nel)

    e_cur, g_cur, _, a1_cur = _current_moduli(params, epsp_arr)
    a2_cur = params.nu * a1_cur

    p_plane = np.array([[1.0, -0.5, 0.0],
                        [-0.5, 1.0, 0.0],
                        [0.0, 0.0, 3.0]], dtype=float)

    D = np.zeros((nel, 3, 3), dtype=float)
    _, h_raw = _eval_yield_and_hardening(params, epsp_arr, np.zeros(nel))

    for i in range(nel):
        a1 = a1_cur[i]
        a2 = a2_cur[i]
        g = g_cur[i]
        c_el = np.array([[a1, a2, 0.0],
                         [a2, a1, 0.0],
                         [0.0, 0.0, g]], dtype=float)
        D[i] = c_el.copy()

        if epsp_incr is not None:
            dep = float(epsp_incr[i]) if hasattr(epsp_incr, "__getitem__") else float(epsp_incr)
        elif deps is not None and deps.ndim == 1:
            dep = float(deps[i])
        else:
            dep = 0.0
        if dep <= 0.0:
            continue

        s = sig_arr[i]
        sy = np.sqrt(max(float(s @ p_plane @ s), 0.0))
        sy = max(sy, _EM30)
        q_tr = sy + 3.0 * g * dep
        sfac = sy / q_tr
        sig_tr = s / sfac

        h = max(h_raw[i], -2.97 * g)
        hd = max(3.0 * g + h, 0.03 * g)
        hfrac = (hd - 3.0 * g) / hd

        cp = c_el @ p_plane
        gvec = cp @ sig_tr
        coef = (hfrac - sfac) / (q_tr * q_tr)
        D[i] = sfac * c_el + coef * np.outer(sig_tr, gvec)

    if single_element:
        return D[0]
    return D


# ----------------------------------------------------------------------------
# Builder Function
# ----------------------------------------------------------------------------

def build_law60(
    mat_record: Any = None,
    functs: dict[int, Any] | list[Any] | None = None,
    **kwargs: Any,
) -> Law60Params:
    """Build and initialize Law60Params from a Material, dictionary, or card parameters.

    Extracts parameters per starter/source/materials/mat/mat060/hm_read_mat60.F.
    Maps /FUNCT curves from `functs` or from embedded table objects.
    """
    if hasattr(mat_record, "law60_params") and functs is None and not kwargs:
        return mat_record.law60_params
    if isinstance(mat_record, Law60Params) and len(mat_record.curve_x) > 0 and functs is None and not kwargs:
        return mat_record

    if mat_record is None:
        p: dict[str, Any] = dict(kwargs)
        _id = int(kwargs.get("id", 1))
        rho0_in = kwargs.get("rho0", kwargs.get("density", kwargs.get("MAT_RHO", 1.0)))
        _title = str(kwargs.get("title", ""))
    elif isinstance(mat_record, Law60Params):
        p = mat_record.params
        p.update(kwargs)
        _id = mat_record.id
        rho0_in = mat_record.rho0
        _title = mat_record.title
    elif isinstance(mat_record, dict):
        base_params = mat_record.get("params", mat_record)
        p = {**base_params, **kwargs}
        _id = int(mat_record.get("id", kwargs.get("id", 1)))
        rho0_in = mat_record.get("rho0", mat_record.get("density", kwargs.get("rho0", 1.0)))
        _title = str(mat_record.get("title", kwargs.get("title", "")))
    elif hasattr(mat_record, "params"):
        base_params = mat_record.params if isinstance(mat_record.params, dict) else {}
        p = {**base_params, **kwargs}
        _id = int(getattr(mat_record, "id", kwargs.get("id", 1)))
        rho0_in = getattr(mat_record, "rho0", getattr(mat_record, "density", kwargs.get("rho0", 1.0)))
        _title = str(getattr(mat_record, "title", kwargs.get("title", "")))
    else:
        p = dict(kwargs)
        _id = int(kwargs.get("id", 1))
        rho0_in = kwargs.get("rho0", 1.0)
        _title = str(kwargs.get("title", ""))

    def _get(keys: Sequence[str], default: float = 0.0) -> float:
        for k in keys:
            if k in p and p[k] is not None:
                try:
                    return float(p[k])
                except (ValueError, TypeError):
                    pass
        return float(default)

    def _geti(keys: Sequence[str], default: int = 0) -> int:
        for k in keys:
            if k in p and p[k] is not None:
                try:
                    return int(p[k])
                except (ValueError, TypeError):
                    pass
        return int(default)

    rho0 = _get(["MAT_RHO", "rho0", "density", "rho"], float(rho0_in) if rho0_in else 1.0)
    rhor = _get(["Refer_Rho", "rhor", "MAT_REFRHO"], rho0)
    if rhor == 0.0:
        rhor = rho0

    e0 = _get(["MAT_E", "E", "e0", "e", "young", "E0"], 210000.0)
    nu = _get(["MAT_NU", "NU", "nu", "poisson"], 0.3)
    eps_max = _get(["MAT_EPS", "EPS_MAX", "eps_max", "eps_p_max"], 1e30)
    eps_t1 = _get(["MAT_EPST1", "EPS_T1", "eps_t1"], 1e30)
    eps_t2 = _get(["MAT_EPST2", "EPS_T2", "eps_t2"], 2e30)
    nfunc = _geti(["NFUNC", "nfunc", "NRATE", "nrate"], 1)
    fsmooth = _geti(["Fsmooth", "fsmooth", "israte"], 0)
    fisokin = _get(["MAT_HARD", "Chard", "chard", "fisokin"], 0.0)
    fcut = _get(["Fcut", "fcut", "asrate"], 1e30)
    ipfun = _geti(["Xr_fun", "Ipfun", "ipfun", "pfun"], 0)
    ifunce = _geti(["fct_ID_k", "ifunce", "IFUNCE"], 0)
    pscale_raw = _get(["MAT_FScale", "Fpscale", "pscale"], 0.0)
    einf = _get(["E_R", "einf", "EINF"], 0.0)
    ce = _get(["MAT_C1", "ce", "CE"], 0.0)

    # In Fortran hm_read_mat60.F:
    # IF (IPFUN == 0) THEN PSCALE = 0 ELSEIF (PSCALE == 0) THEN PSCALE = 1 ELSE PSCALE = 1/PSCALE
    if ipfun == 0:
        pscale = 0.0
    elif pscale_raw == 0.0:
        pscale = 1.0
    else:
        pscale = 1.0 / pscale_raw

    # Collect yield stress function IDs or objects
    raw_funcs = p.get("funcs", None)
    if raw_funcs is None:
        raw_funcs = []
        for name in [
            "FUN_A1", "FUN_B1", "FUN_A2", "FUN_B2", "FUN_A3",
            "FUN_B3", "FUN_A4", "FUN_B4", "FUN_A5", "FUN_B5",
        ]:
            if name in p and p[name] is not None and int(p[name]) != 0:
                raw_funcs.append(int(p[name]))

    raw_fscale = p.get("fscale_arr", None)
    if raw_fscale is None:
        raw_fscale = []
        for name in [
            "MAT_ALPHA1", "MAT_ALPHA2", "MAT_ALPHA3", "MAT_ALPHA4", "MAT_ALPHA5",
            "MAT_ALPHA6", "MAT_ALPHA7", "MAT_ALPHA8", "MAT_ALPHA9", "MAT_ALPHA0",
        ]:
            if name in p and p[name] is not None:
                raw_fscale.append(float(p[name]))
            else:
                raw_fscale.append(1.0)

    raw_rates = p.get("rate_arr", None)
    if raw_rates is None:
        raw_rates = []
        for name in [
            "MAT_EPSR1", "MAT_EPSR2", "MAT_EPSR3", "MAT_EPSR4", "MAT_EPSR5",
            "MAT_EPSR6", "MAT_EPSR7", "MAT_EPSR8", "MAT_EPSR9", "MAT_EPSR10",
        ]:
            if name in p and p[name] is not None:
                raw_rates.append(float(p[name]))
            else:
                raw_rates.append(0.0)

    # Map /FUNCT curves
    funct_dict: dict[int, Any] = {}
    if functs is not None:
        if isinstance(functs, dict):
            funct_dict = functs
        elif isinstance(functs, (list, tuple)):
            for f in functs:
                fid = getattr(f, "id", None)
                if fid is not None:
                    funct_dict[int(fid)] = f

    curve_x: list[np.ndarray] = []
    curve_y: list[np.ndarray] = []
    curve_s: list[np.ndarray] = []
    rates_list: list[float] = []

    # Check if curve arrays were already pre-extracted
    if "curve_x" in p and len(p["curve_x"]) > 0:
        curve_x = [np.asarray(x, dtype=float) for x in p["curve_x"]]
        curve_y = [np.asarray(y, dtype=float) for y in p["curve_y"]]
        curve_s = [np.asarray(s, dtype=float) for s in p["curve_s"]]
        rates_list = list(p.get("rates", raw_rates[:len(curve_x)]))
    else:
        for idx, item in enumerate(raw_funcs):
            f_obj = None
            if isinstance(item, (FunctTable, SmoothFunctTable)) or hasattr(item, "x"):
                f_obj = item
            elif isinstance(item, (int, np.integer)):
                f_obj = funct_dict.get(int(item))

            scale_y = float(raw_fscale[idx]) if idx < len(raw_fscale) else 1.0
            r_val = float(raw_rates[idx]) if idx < len(raw_rates) else 0.0

            if f_obj is not None:
                cx = np.asarray(f_obj.x, dtype=float).copy()
                cy = np.asarray(f_obj.y, dtype=float).copy() * scale_y
                if hasattr(f_obj, "slope"):
                    cs = np.asarray(f_obj.slope, dtype=float).copy() * scale_y
                elif len(cx) > 1:
                    cs = np.diff(cy) / np.maximum(np.diff(cx), _EM20)
                else:
                    cs = np.zeros(0, dtype=float)
                curve_x.append(cx)
                curve_y.append(cy)
                curve_s.append(cs)
                rates_list.append(r_val)

    # If only 1 curve, duplicate at rate 1.0 per hm_read_mat60.F:249-254
    if len(curve_x) == 1 and len(rates_list) == 1 and rates_list[0] == 0.0:
        curve_x.append(curve_x[0].copy())
        curve_y.append(curve_y[0].copy())
        curve_s.append(curve_s[0].copy())
        rates_list.append(1.0)

    rates_arr = np.asarray(rates_list, dtype=float)

    # Pressure curve mapping
    p_cx, p_cy, p_cs = None, None, None
    if ipfun > 0:
        pf_obj = funct_dict.get(ipfun) if funct_dict else None
        if pf_obj is not None:
            p_cx = np.asarray(pf_obj.x, dtype=float)
            p_cy = np.asarray(pf_obj.y, dtype=float)
            p_cs = getattr(pf_obj, "slope", np.diff(p_cy) / np.maximum(np.diff(p_cx), _EM20))
        elif "p_curve_x" in p and p["p_curve_x"] is not None:
            p_cx = np.asarray(p["p_curve_x"], dtype=float)
            p_cy = np.asarray(p["p_curve_y"], dtype=float)
            p_cs = np.asarray(p["p_curve_s"], dtype=float)

    # Dynamic modulus curve mapping
    e_cx, e_cy, e_cs = None, None, None
    if ifunce > 0:
        ef_obj = funct_dict.get(ifunce) if funct_dict else None
        if ef_obj is not None:
            e_cx = np.asarray(ef_obj.x, dtype=float)
            e_cy = np.asarray(ef_obj.y, dtype=float)
            e_cs = getattr(ef_obj, "slope", np.diff(e_cy) / np.maximum(np.diff(e_cx), _EM20))
        elif "e_curve_x" in p and p["e_curve_x"] is not None:
            e_cx = np.asarray(p["e_curve_x"], dtype=float)
            e_cy = np.asarray(p["e_curve_y"], dtype=float)
            e_cs = np.asarray(p["e_curve_s"], dtype=float)

    return Law60Params(
        id=_id,
        rho0=rho0,
        rhor=rhor,
        title=_title,
        e0=e0,
        nu=nu,
        eps_max=eps_max,
        eps_t1=eps_t1,
        eps_t2=eps_t2,
        nfunc=max(len(curve_x), nfunc),
        fsmooth=fsmooth,
        fisokin=fisokin,
        fcut=fcut,
        ipfun=ipfun,
        ifunce=ifunce,
        pscale=pscale,
        einf=einf,
        ce=ce,
        funcs=raw_funcs,
        fscale_arr=raw_fscale,
        rate_arr=raw_rates,
        curve_x=curve_x,
        curve_y=curve_y,
        curve_s=curve_s,
        rates=rates_arr,
        p_curve_x=p_cx,
        p_curve_y=p_cy,
        p_curve_s=p_cs,
        e_curve_x=e_cx,
        e_curve_y=e_cy,
        e_curve_s=e_cs,
    )


def sound_speed(
    mat: Any,
    rho: float | None = None,
    extra: dict[str, Any] | None = None,
) -> float:
    """Compute instantaneous sound speed for LAW60 solids or shells."""
    params = mat if isinstance(mat, Law60Params) else build_law60(mat)
    is_solid = True
    if extra is not None:
        if extra.get("is_shell", False) or extra.get("element_type") == "shell":
            is_solid = False

    if is_solid:
        return params.sound_speed_solid(rho)
    return params.sound_speed_shell(rho)

def extra_shapes(mat: Any, nip: int | None = None) -> dict[str, tuple[int, ...]]:
    """Return dictionary of extra state variable shapes for /MAT/LAW60.

    Allocates uvar array of size (5 + nfunc,) per integration point,
    and off60 deletion flags.
    """
    nfunc = 5
    if hasattr(mat, "params") and isinstance(mat.params, dict):
        nfunc = int(mat.params.get("nfunc", mat.params.get("NFUNC", 5)))
    elif hasattr(mat, "nfunc"):
        nfunc = int(mat.nfunc)
    uvar_dim = (5 + nfunc,)
    return {
        "uvar": (nip, *uvar_dim) if nip else uvar_dim,
        "off60": (nip,) if nip else (),
    }


def resolve(mat: Any, model: Any, log: Any = None) -> None:
    """Resolve /FUNCT references into stored numpy arrays in mat.params and build Law60Params."""
    functs = getattr(model, "functions", {})
    params = build_law60(mat, functs=functs)
    mat.law60_params = params
    if hasattr(mat, "params") and isinstance(mat.params, dict):
        mat.params["curve_x"] = params.curve_x
        mat.params["curve_y"] = params.curve_y
        mat.params["curve_s"] = params.curve_s
        mat.params["rates"] = params.rates
        mat.params["p_curve_x"] = params.p_curve_x
        mat.params["p_curve_y"] = params.p_curve_y
        mat.params["p_curve_s"] = params.p_curve_s
        mat.params["e_curve_x"] = params.e_curve_x
        mat.params["e_curve_y"] = params.e_curve_y
        mat.params["e_curve_s"] = params.e_curve_s


def _register():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        for k in (60, "60", "LAW60", "PLAS_T3", "MAT_LAW60", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC"):
            MAT_PHYSICS_REGISTRY[k] = build_law60
    except Exception:
        pass

