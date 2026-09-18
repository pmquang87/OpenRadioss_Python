"""
LAW55 — Shell orthotropic elastic-plastic rate-dependent formulation (/MAT/LAW55).

Fortran origins:
- engine: ``engine/source/materials/mat/mat055/sigeps55c.F``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pyradioss.model.entities import Material

_EM20 = 1e-20
_INF = 1e30


def _eval_curve(curve: Any, x: float) -> float:
    if curve is None:
        return 0.0
    if isinstance(curve, (int, float)):
        return float(curve)
    if callable(curve):
        return float(curve(x))
    xs = getattr(curve, "x", None)
    ys = getattr(curve, "y", None)
    if xs is not None and ys is not None:
        return float(np.interp(x, xs, ys))
    if hasattr(curve, "data"):
        d = np.asarray(curve.data, dtype=float)
        if d.ndim == 2 and d.shape[1] >= 2:
            return float(np.interp(x, d[:, 0], d[:, 1]))
    return 0.0


@dataclass
class Law55Params:
    """Parameters for /MAT/LAW55."""

    rho0: float = 1.0
    refer_rho: float = 1.0
    e: float = 1000.0
    nu: float = 0.3
    g: float = 0.0
    visc: float = 0.0
    viscv: float = 0.0
    nratep: int = 0
    nraten: int = 0
    epsmax: float = _INF

    ratep: List[float] = field(default_factory=list)
    raten: List[float] = field(default_factory=list)
    fun_pos: List[Any] = field(default_factory=list)
    fun_neg: List[Any] = field(default_factory=list)
    fun_elas: Any = None
    sigy0: float = 100.0

    title: str = ""

    # Derived
    a1: float = field(init=False)
    a2: float = field(init=False)
    k: float = field(init=False)
    c_shell: float = field(init=False)
    c_solid: float = field(init=False)

    def __post_init__(self) -> None:
        if self.refer_rho == 0.0:
            self.refer_rho = self.rho0
        if self.g == 0.0:
            self.g = self.e / (2.0 * (1.0 + self.nu))
        denom_k = 3.0 * (1.0 - 2.0 * self.nu)
        self.k = self.e / denom_k if abs(denom_k) > 1e-12 else self.e
        denom_shell = 1.0 - self.nu * self.nu
        if abs(denom_shell) > 1e-12:
            self.a1 = self.e / denom_shell
            self.a2 = self.nu * self.a1
        else:
            self.a1 = self.e
            self.a2 = 0.0
        if self.epsmax <= 0.0:
            self.epsmax = _INF

        self.c_shell = math.sqrt(max(self.a1 / max(self.rho0, _EM20), _EM20))
        c14g3 = self.k + (4.0 / 3.0) * self.g
        self.c_solid = math.sqrt(max(c14g3 / max(self.rho0, _EM20), _EM20))

    @property
    def young(self) -> float:
        return self.e

    @property
    def shear(self) -> float:
        return self.g

    @property
    def bulk(self) -> float:
        return self.k

    def eval_yield_limits(self, epst: float, epsdot: float) -> Tuple[float, float, float]:
        """Compute (yld_elas, yld_max, yld_min)."""
        yld_elas = _eval_curve(self.fun_elas, epst) if self.fun_elas is not None else self.sigy0
        if yld_elas <= 0.0:
            yld_elas = self.sigy0

        if not self.fun_pos:
            yld_max = yld_elas
        else:
            # Interpolate in rate space
            if len(self.fun_pos) == 1 or not self.ratep:
                yld_max = _eval_curve(self.fun_pos[0], epst)
            else:
                rates = np.asarray(self.ratep, dtype=float)
                idx = int(np.clip(np.searchsorted(rates, abs(epsdot)) - 1, 0, len(rates) - 2))
                r1, r2 = rates[idx], rates[idx + 1]
                fac = (abs(epsdot) - r1) / max(r2 - r1, _EM20)
                fac = min(1.0, max(0.0, fac))
                y1 = _eval_curve(self.fun_pos[idx], epst)
                y2 = _eval_curve(self.fun_pos[idx + 1], epst)
                yld_max = y1 + fac * (y2 - y1)

        if not self.fun_neg:
            yld_min = yld_elas * 0.5
        else:
            if len(self.fun_neg) == 1 or not self.raten:
                yld_min = _eval_curve(self.fun_neg[0], epst)
            else:
                rates = np.asarray(self.raten, dtype=float)
                idx = int(np.clip(np.searchsorted(rates, abs(epsdot)) - 1, 0, len(rates) - 2))
                r1, r2 = rates[idx], rates[idx + 1]
                fac = (abs(epsdot) - r1) / max(r2 - r1, _EM20)
                fac = min(1.0, max(0.0, fac))
                y1 = _eval_curve(self.fun_neg[idx], epst)
                y2 = _eval_curve(self.fun_neg[idx + 1], epst)
                yld_min = y1 + fac * (y2 - y1)

        return max(yld_elas, _EM20), max(yld_max, _EM20), max(yld_min, _EM20)


def _extract_param(d: Dict[str, Any], keys: Tuple[str, ...], default: Any = 0.0) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def resolve(mat: Any) -> Law55Params:
    """Resolve Law55Params from Material entity, dict, or Law55Params."""
    if isinstance(mat, Law55Params):
        return mat

    p: Dict[str, Any] = {}
    title = ""
    rho0_val = 1.0

    if isinstance(mat, Material):
        p = dict(mat.params) if mat.params is not None else {}
        title = mat.title
        rho0_val = getattr(mat, "rho0", None)
    elif isinstance(mat, dict):
        p = dict(mat.get("params", mat))
        title = mat.get("title", "")
        rho0_val = mat.get("rho0") or mat.get("rho") or mat.get("density")
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        p = dict(mat.params)
        title = getattr(mat, "title", "")
        rho0_val = getattr(mat, "rho0", getattr(mat, "rho", None))
    elif hasattr(mat, "__dict__"):
        p = dict(mat.__dict__)
        title = getattr(mat, "title", "")
        rho0_val = getattr(mat, "rho0", getattr(mat, "rho", None))

    if rho0_val is None or float(rho0_val) == 0.0:
        rho0_val = _extract_param(p, ("rho0", "rho", "density", "MAT_RHO", "Refer_Rho"), 1.0)
    rho0 = float(rho0_val)

    refer_rho_val = _extract_param(p, ("refer_rho", "Refer_Rho", "rho_ref"), rho0)
    refer_rho = float(refer_rho_val) if refer_rho_val is not None and float(refer_rho_val) != 0.0 else rho0

    e = float(_extract_param(p, ("e", "E", "MAT_E", "young"), 1000.0))
    nu = float(_extract_param(p, ("nu", "NU", "MAT_NU", "pr"), 0.3))
    g = float(_extract_param(p, ("g", "G", "MAT_G"), 0.0))

    visc = float(_extract_param(p, ("visc", "VISC", "MAT_VISC"), 0.0))
    viscv = float(_extract_param(p, ("viscv", "VISCV", "MAT_VISCV"), 0.0))
    epsmax = float(_extract_param(p, ("epsmax", "EPSMAX", "MAT_EPS", "eps_max"), _INF))
    sigy0 = float(_extract_param(p, ("sigy0", "SIGY0", "MAT_SIGY", "sig_y"), 100.0))

    nratep = int(_extract_param(p, ("nratep", "NRATEP"), 0))
    nraten = int(_extract_param(p, ("nraten", "NRATEN"), 0))
    ratep = list(_extract_param(p, ("ratep", "RATEP"), []))
    raten = list(_extract_param(p, ("raten", "RATEN"), []))
    fun_pos = list(_extract_param(p, ("fun_pos", "FUN_POS"), []))
    fun_neg = list(_extract_param(p, ("fun_neg", "FUN_NEG"), []))
    fun_elas = _extract_param(p, ("fun_elas", "FUN_ELAS"), None)

    return Law55Params(
        rho0=rho0,
        refer_rho=refer_rho,
        e=e,
        nu=nu,
        g=g,
        visc=visc,
        viscv=viscv,
        nratep=nratep,
        nraten=nraten,
        epsmax=epsmax,
        ratep=ratep,
        raten=raten,
        fun_pos=fun_pos,
        fun_neg=fun_neg,
        fun_elas=fun_elas,
        sigy0=sigy0,
        title=title,
    )


def build_law55(mat: Any) -> Law55Params:
    """Build Law55Params from Material or dictionary."""
    return resolve(mat)


def extra_shapes(mat: Any, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """State variables: uvar (5,) and off scalar."""
    if nip > 1:
        return {"uvar55": (nip, 5), "off55": (nip,)}
    return {"uvar55": (5,), "off55": ()}


def needs_defgrad(mat: Any) -> bool:
    return False


def sound_speed(mat: Any, eps: Optional[Any] = None, extra: Optional[Dict[str, Any]] = None) -> float:
    p = resolve(mat)
    return p.c_shell


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Shell plane-stress constitutive cycle for LAW55 (sigeps55c.F)."""
    p = resolve(mat)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).copy()
    deps_arr = np.atleast_2d(deps).copy()
    nel = sig_arr.shape[0]

    if epsp is None:
        epsp_arr = np.zeros(nel, dtype=float)
    else:
        epsp_arr = np.atleast_1d(epsp).astype(float).copy()

    if extra is None:
        extra = {}
    off_arr = extra.get("off55", extra.get("off"))
    if off_arr is None or len(np.atleast_1d(off_arr)) != nel:
        off_arr = np.ones(nel, dtype=float)
        extra["off55"] = off_arr
    else:
        off_arr = np.atleast_1d(off_arr).astype(float).copy()

    gs = (5.0 / 6.0) * p.g
    sign = np.zeros_like(sig_arr)
    c_arr = np.full(nel, p.c_shell, dtype=float)

    for i in range(nel):
        if off_arr[i] <= 0.0:
            continue

        de_xx = deps_arr[i, 0]
        de_yy = deps_arr[i, 1]
        de_xy = deps_arr[i, 2] if deps_arr.shape[1] > 2 else 0.0

        s_xx = sig_arr[i, 0] + p.a1 * de_xx + p.a2 * de_yy
        s_yy = sig_arr[i, 1] + p.a2 * de_xx + p.a1 * de_yy
        s_xy = sig_arr[i, 2] + p.g * de_xy if sig_arr.shape[1] > 2 else 0.0
        s_yz = sig_arr[i, 3] + gs * deps_arr[i, 3] if sig_arr.shape[1] > 3 else 0.0
        s_zx = sig_arr[i, 4] + gs * deps_arr[i, 4] if sig_arr.shape[1] > 4 else 0.0

        # Total effective strain and rate
        epst = 0.5 * (de_xx + de_yy + math.sqrt((de_xx - de_yy)**2 + de_xy**2))
        epsdot = epst / dt if dt > 0.0 else 0.0

        if epsp_arr[i] + abs(epst) > p.epsmax:
            off_arr[i] = 0.0
            sign[i, :] = 0.0
            continue

        yld_elas, yld_max, yld_min = p.eval_yield_limits(epsp_arr[i], epsdot)
        svm = math.sqrt(max(s_xx**2 + s_yy**2 - s_xx * s_yy + 3.0 * s_xy**2, 0.0))

        if svm > yld_max and svm > _EM20:
            scale = yld_max / svm
            s_xx *= scale
            s_yy *= scale
            s_xy *= scale
            dpla = (1.0 - scale) * svm / p.e
            epsp_arr[i] += dpla

        sign[i, 0] = s_xx
        sign[i, 1] = s_yy
        if sign.shape[1] > 2:
            sign[i, 2] = s_xy
        if sign.shape[1] > 3:
            sign[i, 3] = s_yz
        if sign.shape[1] > 4:
            sign[i, 4] = s_zx

    extra["off55"] = off_arr

    out_sig = sign[0] if is_1d else sign
    out_epsp = epsp_arr[0] if is_1d else epsp_arr
    out_c = c_arr[0] if is_1d else c_arr

    if return_sound_speed:
        return out_sig, out_epsp, out_c
    return out_sig, out_epsp, None


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Solid update for LAW55."""
    p = resolve(mat)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).copy()
    deps_arr = np.atleast_2d(deps).copy()
    nel = sig_arr.shape[0]

    if epsp is None:
        epsp_arr = np.zeros(nel, dtype=float)
    else:
        epsp_arr = np.atleast_1d(epsp).astype(float).copy()

    if extra is None:
        extra = {}
    off_arr = extra.get("off55", extra.get("off"))
    if off_arr is None or len(np.atleast_1d(off_arr)) != nel:
        off_arr = np.ones(nel, dtype=float)
        extra["off55"] = off_arr
    else:
        off_arr = np.atleast_1d(off_arr).astype(float).copy()

    sign = np.zeros_like(sig_arr)
    c_arr = np.full(nel, p.c_solid, dtype=float)

    for i in range(nel):
        if off_arr[i] <= 0.0:
            continue

        dtrace = deps_arr[i, 0] + deps_arr[i, 1] + deps_arr[i, 2]
        dvol = dtrace / 3.0
        pres_old = (sig_arr[i, 0] + sig_arr[i, 1] + sig_arr[i, 2]) / 3.0
        pres_new = pres_old + p.k * dtrace

        s_xx = sig_arr[i, 0] - pres_old + 2.0 * p.g * (deps_arr[i, 0] - dvol)
        s_yy = sig_arr[i, 1] - pres_old + 2.0 * p.g * (deps_arr[i, 1] - dvol)
        s_zz = sig_arr[i, 2] - pres_old + 2.0 * p.g * (deps_arr[i, 2] - dvol)
        s_xy = sig_arr[i, 3] + p.g * deps_arr[i, 3] if sig_arr.shape[1] > 3 else 0.0
        s_yz = sig_arr[i, 4] + p.g * deps_arr[i, 4] if sig_arr.shape[1] > 4 else 0.0
        s_zx = sig_arr[i, 5] + p.g * deps_arr[i, 5] if sig_arr.shape[1] > 5 else 0.0

        j2 = 0.5 * (s_xx**2 + s_yy**2 + s_zz**2) + s_xy**2 + s_yz**2 + s_zx**2
        svm = math.sqrt(max(3.0 * j2, 0.0))

        epsdot = math.sqrt(max((2.0 / 3.0) * ((deps_arr[i, 0] - dvol)**2 + (deps_arr[i, 1] - dvol)**2 + (deps_arr[i, 2] - dvol)**2), 0.0)) / dt if dt > 0.0 else 0.0
        yld_elas, yld_max, yld_min = p.eval_yield_limits(epsp_arr[i], epsdot)

        if svm > yld_max and svm > _EM20:
            scale = yld_max / svm
            s_xx *= scale
            s_yy *= scale
            s_zz *= scale
            s_xy *= scale
            s_yz *= scale
            s_zx *= scale
            dpla = (1.0 - scale) * svm / (3.0 * p.g)
            epsp_arr[i] += dpla

        sign[i, 0] = s_xx + pres_new
        sign[i, 1] = s_yy + pres_new
        sign[i, 2] = s_zz + pres_new
        if sign.shape[1] > 3:
            sign[i, 3] = s_xy
        if sign.shape[1] > 4:
            sign[i, 4] = s_yz
        if sign.shape[1] > 5:
            sign[i, 5] = s_zx

    extra["off55"] = off_arr

    out_sig = sign[0] if is_1d else sign
    out_epsp = epsp_arr[0] if is_1d else epsp_arr
    out_c = c_arr[0] if is_1d else c_arr

    if return_sound_speed:
        return out_sig, out_epsp, out_c
    return out_sig, out_epsp, None


def shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Consistent shell tangent (3, 3)."""
    p = resolve(mat)
    c = np.array([
        [p.a1, p.a2, 0.0],
        [p.a2, p.a1, 0.0],
        [0.0, 0.0, p.g],
    ], dtype=float)
    if sig is not None and sig.ndim == 2:
        return np.broadcast_to(c, (sig.shape[0], 3, 3)).copy()
    return c


def consistent_shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    return shell_tangent(mat, sig=sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra)


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Consistent solid tangent (6, 6)."""
    p = resolve(mat)
    lam = p.k - (2.0 / 3.0) * p.g
    c = np.zeros((6, 6), dtype=float)
    c[0, 0] = c[1, 1] = c[2, 2] = lam + 2.0 * p.g
    c[0, 1] = c[0, 2] = c[1, 0] = c[1, 2] = c[2, 0] = c[2, 1] = lam
    c[3, 3] = c[4, 4] = c[5, 5] = p.g
    if sig is not None and sig.ndim == 2:
        return np.broadcast_to(c, (sig.shape[0], 6, 6)).copy()
    return c


def consistent_solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    return solid_tangent(mat, sig=sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
