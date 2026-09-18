"""
LAW56 — Orthotropic elasto-plastic with failure and rate dependency (/MAT/LAW56).

Fortran origins:
- engine: ``engine/source/materials/mat/mat056/sigeps56.F`` (solids)
- engine: ``engine/source/materials/mat/mat056/sigeps56c.F`` (shells)
- engine: ``engine/source/materials/mat/mat056/sigeps56g.F``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pyradioss.model.entities import Material

_EM20 = 1e-20
_INF = 1e30


def _eval_curve(curve: Any, x: float) -> Tuple[float, float]:
    if curve is None:
        return 0.0, 0.0
    if isinstance(curve, (int, float)):
        return float(curve), 0.0
    if callable(curve):
        y = float(curve(x))
        h = max(abs(x) * 1e-6, 1e-8)
        yp = float(curve(x + h))
        ym = float(curve(x - h))
        return y, (yp - ym) / (2.0 * h)
    xs = getattr(curve, "x", None)
    ys = getattr(curve, "y", None)
    if xs is not None and ys is not None:
        y = float(np.interp(x, xs, ys))
        h = max(abs(x) * 1e-6, 1e-8)
        yp = float(np.interp(x + h, xs, ys))
        ym = float(np.interp(x - h, xs, ys))
        return y, (yp - ym) / (2.0 * h)
    return 0.0, 0.0


@dataclass
class Law56Params:
    """Parameters for /MAT/LAW56."""

    rho0: float = 1.0
    refer_rho: float = 1.0
    e: float = 1000.0
    nu: float = 0.3
    g: float = 0.0
    nrate: int = 0
    epsmax: float = _INF
    epsr1: float = _INF
    epsr2: float = _INF
    fisokin: float = 0.0  # 0 isotropic, 1 kinematic

    rate: List[float] = field(default_factory=list)
    fun_rate: List[Any] = field(default_factory=list)
    sigy0: float = 100.0
    h_iso: float = 0.0

    title: str = ""

    # Derived
    k: float = field(init=False)
    a11: float = field(init=False)
    a21: float = field(init=False)
    c_solid: float = field(init=False)
    c_shell: float = field(init=False)

    def __post_init__(self) -> None:
        if self.refer_rho == 0.0:
            self.refer_rho = self.rho0
        if self.g == 0.0:
            self.g = self.e / (2.0 * (1.0 + self.nu))
        denom_k = 3.0 * (1.0 - 2.0 * self.nu)
        self.k = self.e / denom_k if abs(denom_k) > 1e-12 else self.e
        denom_shell = 1.0 - self.nu * self.nu
        if abs(denom_shell) > 1e-12:
            self.a11 = self.e / denom_shell
            self.a21 = self.nu * self.a11
        else:
            self.a11 = self.e
            self.a21 = 0.0
        if self.epsmax <= 0.0:
            self.epsmax = _INF
        if self.epsr1 <= 0.0:
            self.epsr1 = _INF
        if self.epsr2 <= 0.0:
            self.epsr2 = 2.0 * self.epsr1

        c14g3 = self.k + (4.0 / 3.0) * self.g
        self.c_solid = math.sqrt(max(c14g3 / max(self.rho0, _EM20), _EM20))
        self.c_shell = math.sqrt(max(self.a11 / max(self.rho0, _EM20), _EM20))

    @property
    def young(self) -> float:
        return self.e

    @property
    def shear(self) -> float:
        return self.g

    @property
    def bulk(self) -> float:
        return self.k

    def eval_yield(self, pla: float, epsdot: float) -> Tuple[float, float]:
        """Evaluate (sigy, slope)."""
        if not self.fun_rate:
            return self.sigy0 + self.h_iso * pla, self.h_iso

        if len(self.fun_rate) == 1 or not self.rate:
            return _eval_curve(self.fun_rate[0], pla)

        rates = np.asarray(self.rate, dtype=float)
        idx = int(np.clip(np.searchsorted(rates, abs(epsdot)) - 1, 0, len(rates) - 2))
        r1, r2 = rates[idx], rates[idx + 1]
        fac = (abs(epsdot) - r1) / max(r2 - r1, _EM20)
        fac = min(1.0, max(0.0, fac))
        y1, s1 = _eval_curve(self.fun_rate[idx], pla)
        y2, s2 = _eval_curve(self.fun_rate[idx + 1], pla)
        return y1 + fac * (y2 - y1), s1 + fac * (s2 - s1)

    def eval_fail(self, epst: float) -> float:
        """Evaluate tension softening / failure factor FAIL in [0, 1]."""
        if self.epsr1 >= _INF:
            return 1.0
        if epst <= self.epsr1:
            return 1.0
        if epst >= self.epsr2:
            return 0.0
        return max(0.0, min(1.0, (self.epsr2 - epst) / max(self.epsr2 - self.epsr1, _EM20)))


def _extract_param(d: Dict[str, Any], keys: Tuple[str, ...], default: Any = 0.0) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def resolve(mat: Any) -> Law56Params:
    """Resolve Law56Params from Material entity, dict, or Law56Params."""
    if isinstance(mat, Law56Params):
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

    nrate = int(_extract_param(p, ("nrate", "NRATE"), 0))
    epsmax = float(_extract_param(p, ("epsmax", "EPSMAX", "MAT_EPS", "eps_max"), _INF))
    epsr1 = float(_extract_param(p, ("epsr1", "EPSR1", "MAT_EPSR1"), _INF))
    epsr2 = float(_extract_param(p, ("epsr2", "EPSR2", "MAT_EPSR2"), _INF))
    fisokin = float(_extract_param(p, ("fisokin", "FISOKIN", "MAT_HARD"), 0.0))

    rate = list(_extract_param(p, ("rate", "RATE"), []))
    fun_rate = list(_extract_param(p, ("fun_rate", "FUN_RATE"), []))
    sigy0 = float(_extract_param(p, ("sigy0", "SIGY0", "MAT_SIGY", "sig_y"), 100.0))
    h_iso = float(_extract_param(p, ("h_iso", "H_ISO", "MAT_B", "h"), 0.0))

    return Law56Params(
        rho0=rho0,
        refer_rho=refer_rho,
        e=e,
        nu=nu,
        g=g,
        nrate=nrate,
        epsmax=epsmax,
        epsr1=epsr1,
        epsr2=epsr2,
        fisokin=fisokin,
        rate=rate,
        fun_rate=fun_rate,
        sigy0=sigy0,
        h_iso=h_iso,
        title=title,
    )


def build_law56(mat: Any) -> Law56Params:
    """Build Law56Params from Material or dictionary."""
    return resolve(mat)


def extra_shapes(mat: Any, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """State variables: uvar (8,) [epsp, alpha_xx, alpha_yy, alpha_xy, ...], off scalar."""
    if nip > 1:
        return {"uvar56": (nip, 8), "off56": (nip,)}
    return {"uvar56": (8,), "off56": ()}


def needs_defgrad(mat: Any) -> bool:
    return False


def sound_speed(mat: Any, eps: Optional[Any] = None, extra: Optional[Dict[str, Any]] = None) -> float:
    p = resolve(mat)
    return p.c_solid


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Shell update for LAW56 (sigeps56c.F)."""
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
    uvar = extra.get("uvar56", extra.get("uvar"))
    if uvar is None or uvar.shape[0] != nel:
        uvar = np.zeros((nel, 8), dtype=float)
        extra["uvar56"] = uvar

    off_arr = extra.get("off56", extra.get("off"))
    if off_arr is None or len(np.atleast_1d(off_arr)) != nel:
        off_arr = np.ones(nel, dtype=float)
        extra["off56"] = off_arr
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

        # Back stresses from uvar
        alpha_xx = uvar[i, 1]
        alpha_yy = uvar[i, 2]
        alpha_xy = uvar[i, 3]

        s_xx = sig_arr[i, 0] - alpha_xx + p.a11 * de_xx + p.a21 * de_yy
        s_yy = sig_arr[i, 1] - alpha_yy + p.a21 * de_xx + p.a11 * de_yy
        s_xy = sig_arr[i, 2] - alpha_xy + p.g * de_xy if sig_arr.shape[1] > 2 else 0.0

        # Max principal strain for failure
        epst = 0.5 * (de_xx + de_yy + math.sqrt((de_xx - de_yy)**2 + de_xy**2))
        epsdot = epst / dt if dt > 0.0 else 0.0
        fail_fac = p.eval_fail(epsp_arr[i] + abs(epst))

        if epsp_arr[i] + abs(epst) > p.epsmax or fail_fac <= 0.0:
            off_arr[i] = 0.0
            sign[i, :] = 0.0
            continue

        sigy, h_slope = p.eval_yield(epsp_arr[i], epsdot)
        sigy_eff = sigy * fail_fac

        svm = math.sqrt(max(s_xx**2 + s_yy**2 - s_xx * s_yy + 3.0 * s_xy**2, 0.0))
        if svm > sigy_eff and svm > _EM20:
            scale = sigy_eff / svm
            s_xx *= scale
            s_yy *= scale
            s_xy *= scale
            dpla = (1.0 - scale) * svm / max(p.e + h_slope, _EM20)
            epsp_arr[i] += dpla

            # Kinematic hardening backstress update
            if p.fisokin > 0.0:
                uvar[i, 1] += p.fisokin * (2.0 / 3.0) * h_slope * de_xx
                uvar[i, 2] += p.fisokin * (2.0 / 3.0) * h_slope * de_yy
                uvar[i, 3] += p.fisokin * (1.0 / 3.0) * h_slope * de_xy

        sign[i, 0] = s_xx + uvar[i, 1]
        sign[i, 1] = s_yy + uvar[i, 2]
        if sign.shape[1] > 2:
            sign[i, 2] = s_xy + uvar[i, 3]
        if sign.shape[1] > 3:
            sign[i, 3] = sig_arr[i, 3] + gs * deps_arr[i, 3]
        if sign.shape[1] > 4:
            sign[i, 4] = sig_arr[i, 4] + gs * deps_arr[i, 4]

    extra["off56"] = off_arr
    extra["uvar56"] = uvar

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
    """Solid update for LAW56 (sigeps56.F)."""
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
    off_arr = extra.get("off56", extra.get("off"))
    if off_arr is None or len(np.atleast_1d(off_arr)) != nel:
        off_arr = np.ones(nel, dtype=float)
        extra["off56"] = off_arr
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

        epst = math.sqrt(max((2.0 / 3.0) * ((deps_arr[i, 0] - dvol)**2 + (deps_arr[i, 1] - dvol)**2 + (deps_arr[i, 2] - dvol)**2), 0.0))
        epsdot = epst / dt if dt > 0.0 else 0.0
        fail_fac = p.eval_fail(epsp_arr[i] + epst)

        if epsp_arr[i] + epst > p.epsmax or fail_fac <= 0.0:
            off_arr[i] = 0.0
            sign[i, :] = 0.0
            continue

        sigy, h_slope = p.eval_yield(epsp_arr[i], epsdot)
        sigy_eff = sigy * fail_fac

        if svm > sigy_eff and svm > _EM20:
            scale = sigy_eff / svm
            s_xx *= scale
            s_yy *= scale
            s_zz *= scale
            s_xy *= scale
            s_yz *= scale
            s_zx *= scale
            dpla = (1.0 - scale) * svm / max(3.0 * p.g + h_slope, _EM20)
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

    extra["off56"] = off_arr

    out_sig = sign[0] if is_1d else sign
    out_epsp = epsp_arr[0] if is_1d else epsp_arr
    out_c = c_arr[0] if is_1d else c_arr

    if return_sound_speed:
        return out_sig, out_epsp, out_c
    return out_sig, out_epsp, None


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
        [p.a11, p.a21, 0.0],
        [p.a21, p.a11, 0.0],
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
