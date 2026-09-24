"""
LAW65 — Tabulated Thermo-Viscoelastic Elastomer / Plastic Law (/MAT/LAW65).

Fortran origins:
- engine: ``engine/source/materials/mat/mat065/sigeps65.F`` (solids)
- engine: ``engine/source/materials/mat/mat065/sigeps65c.F`` (shells)
- starter: ``starter/source/materials/mat/mat065/hm_read_mat65.F``
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
    """Evaluate curve value and slope at x."""
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
    if hasattr(curve, "data"):
        d = np.asarray(curve.data, dtype=float)
        if d.ndim == 2 and d.shape[1] >= 2:
            y = float(np.interp(x, d[:, 0], d[:, 1]))
            h = max(abs(x) * 1e-6, 1e-8)
            yp = float(np.interp(x + h, d[:, 0], d[:, 1]))
            ym = float(np.interp(x - h, d[:, 0], d[:, 1]))
            return y, (yp - ym) / (2.0 * h)
    return 0.0, 0.0


@dataclass
class Law65Params:
    """Parameters for /MAT/LAW65."""

    rho0: float = 1.0
    refer_rho: float = 1.0
    e: float = 1000.0
    nu: float = 0.45
    epsmax: float = _INF
    israte: int = 0
    fcut: float = 0.0
    nrate: int = 1

    rates: List[float] = field(default_factory=list)
    yfac: List[float] = field(default_factory=list)
    fun_load: List[Any] = field(default_factory=list)
    fun_unload: List[Any] = field(default_factory=list)

    sigy_load_default: float = 20.0
    sigy_unload_default: float = 10.0

    title: str = ""

    # Derived
    g: float = field(init=False)
    k: float = field(init=False)
    a1: float = field(init=False)
    a2: float = field(init=False)
    c_solid: float = field(init=False)
    c_shell: float = field(init=False)

    def __post_init__(self) -> None:
        if self.refer_rho == 0.0:
            self.refer_rho = self.rho0
        self.g = self.e / (2.0 * (1.0 + self.nu))
        denom_k = 3.0 * (1.0 - 2.0 * self.nu)
        self.k = self.e / denom_k if abs(denom_k) > 1e-12 else self.e * 10.0
        denom_shell = 1.0 - self.nu * self.nu
        if abs(denom_shell) > 1e-12:
            self.a1 = self.e / denom_shell
            self.a2 = self.nu * self.a1
        else:
            self.a1 = self.e
            self.a2 = 0.0

        if self.epsmax <= 0.0:
            self.epsmax = _INF

        c14g3 = self.k + (4.0 / 3.0) * self.g
        self.c_solid = math.sqrt(max(c14g3 / max(self.rho0, _EM20), _EM20))
        self.c_shell = math.sqrt(max(self.a1 / max(self.rho0, _EM20), _EM20))

    @property
    def young(self) -> float:
        return self.e

    @property
    def shear(self) -> float:
        return self.g

    @property
    def bulk(self) -> float:
        return self.k

    def eval_curves(self, eps_tot: float, eps_ela: float, rate: float) -> Tuple[float, float, float, float]:
        """Return (sig_load, h_load, sig_unload, h_unload) interpolated over rate."""
        if not self.fun_load and not self.fun_unload:
            return self.sigy_load_default, 0.0, self.sigy_unload_default, 0.0

        nr = max(len(self.rates), 1)
        if nr == 1 or not self.rates:
            c_load = self.fun_load[0] if self.fun_load else None
            c_unload = self.fun_unload[0] if self.fun_unload else None
            fac = self.yfac[0] if self.yfac else 1.0

            s_l, h_l = _eval_curve(c_load, eps_tot)
            s_u, h_u = _eval_curve(c_unload, eps_ela)
            s_l = s_l * fac if s_l > 0 else self.sigy_load_default * fac
            s_u = s_u * fac if s_u > 0 else self.sigy_unload_default * fac
            return max(s_l, _EM20), max(h_l * fac, 0.0), max(s_u, _EM20), max(h_u * fac, 0.0)

        # Multi-rate interpolation
        idx = 0
        while idx < nr - 1 and rate > self.rates[idx + 1]:
            idx += 1
        idx2 = min(idx + 1, nr - 1)
        r1, r2 = self.rates[idx], self.rates[idx2]
        w2 = (rate - r1) / (r2 - r1) if abs(r2 - r1) > 1e-12 else 0.0
        w2 = max(0.0, min(1.0, w2))
        w1 = 1.0 - w2

        fac1 = self.yfac[idx] if idx < len(self.yfac) else 1.0
        fac2 = self.yfac[idx2] if idx2 < len(self.yfac) else 1.0

        cl1 = self.fun_load[idx] if idx < len(self.fun_load) else None
        cl2 = self.fun_load[idx2] if idx2 < len(self.fun_load) else None
        cu1 = self.fun_unload[idx] if idx < len(self.fun_unload) else None
        cu2 = self.fun_unload[idx2] if idx2 < len(self.fun_unload) else None

        sl1, hl1 = _eval_curve(cl1, eps_tot)
        sl2, hl2 = _eval_curve(cl2, eps_tot)
        su1, hu1 = _eval_curve(cu1, eps_ela)
        su2, hu2 = _eval_curve(cu2, eps_ela)

        sl = w1 * (sl1 * fac1 if sl1 > 0 else self.sigy_load_default * fac1) + w2 * (sl2 * fac2 if sl2 > 0 else self.sigy_load_default * fac2)
        hl = w1 * hl1 * fac1 + w2 * hl2 * fac2
        su = w1 * (su1 * fac1 if su1 > 0 else self.sigy_unload_default * fac1) + w2 * (su2 * fac2 if su2 > 0 else self.sigy_unload_default * fac2)
        hu = w1 * hu1 * fac1 + w2 * hu2 * fac2

        return max(sl, _EM20), max(hl, 0.0), max(su, _EM20), max(hu, 0.0)


def _extract_param(d: Dict[str, Any], keys: Tuple[str, ...], default: Any = 0.0) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def resolve(mat: Any) -> Law65Params:
    """Resolve Law65Params from Material, dict, or Law65Params."""
    if isinstance(mat, Law65Params):
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
    nu = float(_extract_param(p, ("nu", "NU", "MAT_NU", "pr"), 0.45))
    epsmax_val = float(_extract_param(p, ("epsmax", "EPSMAX", "MAT_EPS", "eps_max"), _INF))
    epsmax = epsmax_val if epsmax_val > 0.0 else _INF

    israte = int(_extract_param(p, ("israte", "ISRATE", "Fsmooth"), 0))
    fcut = float(_extract_param(p, ("fcut", "FCUT", "Fcut"), 0.0))
    nrate = int(_extract_param(p, ("nrate", "NRATE", "NRATEP"), 1))

    rates_raw = _extract_param(p, ("rates", "RATES", "STRAINRATE_LOAD"), [])
    rates = [float(x) for x in rates_raw] if isinstance(rates_raw, (list, tuple)) else []

    yfac_raw = _extract_param(p, ("yfac", "YFAC", "SCALE_LOAD"), [])
    yfac = [float(x) for x in yfac_raw] if isinstance(yfac_raw, (list, tuple)) else []

    fun_load = _extract_param(p, ("fun_load", "FUN_LOAD", "load_curves"), [])
    if not isinstance(fun_load, list):
        fun_load = [fun_load] if fun_load is not None else []

    fun_unload = _extract_param(p, ("fun_unload", "FUN_UNLOAD", "unload_curves"), [])
    if not isinstance(fun_unload, list):
        fun_unload = [fun_unload] if fun_unload is not None else []

    sigy_load_default = float(_extract_param(p, ("sigy_load_default", "SIGY_LOAD"), 20.0))
    sigy_unload_default = float(_extract_param(p, ("sigy_unload_default", "SIGY_UNLOAD"), 10.0))

    return Law65Params(
        rho0=rho0,
        refer_rho=refer_rho,
        e=e,
        nu=nu,
        epsmax=epsmax,
        israte=israte,
        fcut=fcut,
        nrate=nrate,
        rates=rates,
        yfac=yfac,
        fun_load=fun_load,
        fun_unload=fun_unload,
        sigy_load_default=sigy_load_default,
        sigy_unload_default=sigy_unload_default,
        title=title,
    )


def build_law65(mat: Any) -> Law65Params:
    """Build Law65Params from Material or dictionary."""
    return resolve(mat)


def extra_shapes(mat: Any, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """State variables: uvar (8,) [epss, epse, svm, pla, epsp, flag, spare1, spare2]."""
    if nip > 1:
        return {"uvar65": (nip, 8)}
    return {"uvar65": (8,)}


def needs_defgrad(mat: Any) -> bool:
    return False


def sound_speed(mat: Any, eps: Optional[Any] = None, extra: Optional[Dict[str, Any]] = None) -> float:
    p = resolve(mat)
    return p.c_solid


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Solid update for LAW65 tabulated thermo-viscoelastic model."""
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
    uvar = extra.get("uvar65", extra.get("uvar"))
    if uvar is None or uvar.shape[0] != nel:
        uvar = np.zeros((nel, 8), dtype=float)
        extra["uvar65"] = uvar

    sign = np.zeros_like(sig_arr)
    c_arr = np.full(nel, p.c_solid, dtype=float)

    for i in range(nel):
        # Rupture check
        if epsp_arr[i] >= p.epsmax:
            sign[i] = 0.0
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

        epss = uvar[i, 0] + math.sqrt(max(2.0 / 3.0 * (deps_arr[i, 0]**2 + deps_arr[i, 1]**2 + deps_arr[i, 2]**2), 0.0))
        epse = max(epss - epsp_arr[i], 0.0)
        rate = math.sqrt(max(deps_arr[i, 0]**2 + deps_arr[i, 1]**2 + deps_arr[i, 2]**2, 0.0)) / max(dt, 1e-12) if dt > 0.0 else 0.0

        sig_l, h_l, sig_u, h_u = p.eval_curves(epss, epse, rate)

        dpla = 0.0
        if svm >= sig_l and svm > _EM20:
            scale = sig_l / svm
            s_xx *= scale
            s_yy *= scale
            s_zz *= scale
            s_xy *= scale
            s_yz *= scale
            s_zx *= scale
            dpla = (svm - sig_l) / max(3.0 * p.g + h_l, _EM20)
            epsp_arr[i] += dpla
            uvar[i, 5] = 1.0  # loading
        elif svm < sig_u and svm > _EM20:
            # Unloading path
            uvar[i, 5] = -1.0  # unloading
        else:
            uvar[i, 5] = 0.0  # elastic

        uvar[i, 0] = epss
        uvar[i, 1] = epse
        uvar[i, 2] = svm
        uvar[i, 3] = epsp_arr[i]
        uvar[i, 4] = rate

        if epsp_arr[i] >= p.epsmax:
            sign[i] = 0.0
        else:
            sign[i, 0] = s_xx + pres_new
            sign[i, 1] = s_yy + pres_new
            sign[i, 2] = s_zz + pres_new
            if sign.shape[1] > 3:
                sign[i, 3] = s_xy
            if sign.shape[1] > 4:
                sign[i, 4] = s_yz
            if sign.shape[1] > 5:
                sign[i, 5] = s_zx

    extra["uvar65"] = uvar

    out_sig = sign[0] if is_1d else sign
    out_epsp = epsp_arr[0] if is_1d else epsp_arr
    out_c = c_arr[0] if is_1d else c_arr

    if return_sound_speed:
        return out_sig, out_epsp, out_c
    return out_sig, out_epsp, None


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Shell update for LAW65 (sigeps65c.F)."""
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
    uvar = extra.get("uvar65", extra.get("uvar"))
    if uvar is None or uvar.shape[0] != nel:
        uvar = np.zeros((nel, 8), dtype=float)
        extra["uvar65"] = uvar

    gs = (5.0 / 6.0) * p.g
    sign = np.zeros_like(sig_arr)
    c_arr = np.full(nel, p.c_shell, dtype=float)

    for i in range(nel):
        if epsp_arr[i] >= p.epsmax:
            sign[i] = 0.0
            continue

        de_xx = deps_arr[i, 0]
        de_yy = deps_arr[i, 1]
        de_xy = deps_arr[i, 2] if deps_arr.shape[1] > 2 else 0.0

        s_xx = sig_arr[i, 0] + p.a1 * de_xx + p.a2 * de_yy
        s_yy = sig_arr[i, 1] + p.a2 * de_xx + p.a1 * de_yy
        s_xy = sig_arr[i, 2] + p.g * de_xy if sig_arr.shape[1] > 2 else 0.0
        s_yz = sig_arr[i, 3] + gs * deps_arr[i, 3] if sig_arr.shape[1] > 3 else 0.0
        s_zx = sig_arr[i, 4] + gs * deps_arr[i, 4] if sig_arr.shape[1] > 4 else 0.0

        svm = math.sqrt(max(s_xx**2 + s_yy**2 - s_xx * s_yy + 3.0 * s_xy**2, 0.0))
        epss = uvar[i, 0] + math.sqrt(max(de_xx**2 + de_yy**2 + 2.0 * de_xy**2, 0.0))
        epse = max(epss - epsp_arr[i], 0.0)
        rate = math.sqrt(max(de_xx**2 + de_yy**2 + 2.0 * de_xy**2, 0.0)) / max(dt, 1e-12) if dt > 0.0 else 0.0

        sig_l, h_l, sig_u, h_u = p.eval_curves(epss, epse, rate)

        dpla = 0.0
        if svm >= sig_l and svm > _EM20:
            scale = sig_l / svm
            s_xx *= scale
            s_yy *= scale
            s_xy *= scale
            dpla = (svm - sig_l) / max(p.e + h_l, _EM20)
            epsp_arr[i] += dpla
            uvar[i, 5] = 1.0
        elif svm < sig_u and svm > _EM20:
            uvar[i, 5] = -1.0
        else:
            uvar[i, 5] = 0.0

        uvar[i, 0] = epss
        uvar[i, 1] = epse
        uvar[i, 2] = svm
        uvar[i, 3] = epsp_arr[i]
        uvar[i, 4] = rate

        if epsp_arr[i] >= p.epsmax:
            sign[i] = 0.0
        else:
            sign[i, 0] = s_xx
            sign[i, 1] = s_yy
            if sign.shape[1] > 2:
                sign[i, 2] = s_xy
            if sign.shape[1] > 3:
                sign[i, 3] = s_yz
            if sign.shape[1] > 4:
                sign[i, 4] = s_zx

    extra["uvar65"] = uvar

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
