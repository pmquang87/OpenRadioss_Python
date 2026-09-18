"""
LAW59 — Rate-dependent plasticity for structural connectors and springs (/MAT/LAW59, /MAT/CONNECT).

Fortran origins:
- engine: ``engine/source/materials/mat/mat059/sigeps59.F``
- starter: ``starter/source/materials/mat/mat059/hm_read_mat59.F``
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
class Law59Params:
    """Parameters for /MAT/LAW59 (/MAT/CONNECT)."""

    rho0: float = 1.0
    refer_rho: float = 1.0
    e: float = 1000.0
    ecomp: float = 0.0
    g: float = 400.0
    nrate: int = 0
    fcut: float = 0.0

    rate: List[float] = field(default_factory=list)
    yfac: List[float] = field(default_factory=list)
    fun_norm: List[Any] = field(default_factory=list)
    fun_tang: List[Any] = field(default_factory=list)
    sigy_n: float = 100.0
    sigy_t: float = 50.0

    title: str = ""

    # Derived
    sound_speed: float = field(init=False)

    def __post_init__(self) -> None:
        if self.refer_rho == 0.0:
            self.refer_rho = self.rho0
        if self.ecomp <= 0.0:
            self.ecomp = self.e
        mod_max = max(self.e, self.ecomp, self.g, _EM20)
        self.sound_speed = math.sqrt(mod_max / max(self.rho0, _EM20))

    @property
    def young(self) -> float:
        return self.e

    @property
    def shear(self) -> float:
        return self.g

    @property
    def c_sound(self) -> float:
        return self.sound_speed

    @property
    def bulk(self) -> float:
        return self.e / 3.0

    def eval_normal_yield(self, eplas: float, epsdot: float) -> Tuple[float, float]:
        if not self.fun_norm:
            return self.sigy_n, 0.0
        if len(self.fun_norm) == 1 or not self.rate:
            y, s = _eval_curve(self.fun_norm[0], eplas)
            fac = self.yfac[0] if self.yfac else 1.0
            return max(0.0, y * fac), s * fac

        rates = np.asarray(self.rate, dtype=float)
        idx = int(np.clip(np.searchsorted(rates, abs(epsdot)) - 1, 0, len(rates) - 2))
        r1, r2 = rates[idx], rates[idx + 1]
        fac_r = (abs(epsdot) - r1) / max(r2 - r1, _EM20)
        fac_r = min(1.0, max(0.0, fac_r))

        sc1 = self.yfac[idx] if idx < len(self.yfac) else 1.0
        sc2 = self.yfac[idx + 1] if (idx + 1) < len(self.yfac) else 1.0
        y1, s1 = _eval_curve(self.fun_norm[idx], eplas)
        y2, s2 = _eval_curve(self.fun_norm[idx + 1], eplas)
        y1, y2 = y1 * sc1, y2 * sc2
        s1, s2 = s1 * sc1, s2 * sc2
        return max(0.0, y1 + fac_r * (y2 - y1)), s1 + fac_r * (s2 - s1)

    def eval_tangent_yield(self, eplas: float, epsdot: float) -> Tuple[float, float]:
        if not self.fun_tang:
            return self.sigy_t, 0.0
        if len(self.fun_tang) == 1 or not self.rate:
            y, s = _eval_curve(self.fun_tang[0], eplas)
            fac = self.yfac[0] if self.yfac else 1.0
            return max(0.0, y * fac), s * fac

        rates = np.asarray(self.rate, dtype=float)
        idx = int(np.clip(np.searchsorted(rates, abs(epsdot)) - 1, 0, len(rates) - 2))
        r1, r2 = rates[idx], rates[idx + 1]
        fac_r = (abs(epsdot) - r1) / max(r2 - r1, _EM20)
        fac_r = min(1.0, max(0.0, fac_r))

        sc1 = self.yfac[idx] if idx < len(self.yfac) else 1.0
        sc2 = self.yfac[idx + 1] if (idx + 1) < len(self.yfac) else 1.0
        y1, s1 = _eval_curve(self.fun_tang[idx], eplas)
        y2, s2 = _eval_curve(self.fun_tang[idx + 1], eplas)
        y1, y2 = y1 * sc1, y2 * sc2
        s1, s2 = s1 * sc1, s2 * sc2
        return max(0.0, y1 + fac_r * (y2 - y1)), s1 + fac_r * (s2 - s1)


def _extract_param(d: Dict[str, Any], keys: Tuple[str, ...], default: Any = 0.0) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def resolve(mat: Any) -> Law59Params:
    """Resolve Law59Params from Material entity, dict, or Law59Params."""
    if isinstance(mat, Law59Params):
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
    ecomp = float(_extract_param(p, ("ecomp", "Ecomp", "ECOMP"), e))
    g = float(_extract_param(p, ("g", "G", "MAT_G", "MAT_G0", "shear"), 400.0))
    nrate = int(_extract_param(p, ("nrate", "NRATE", "NFUNC"), 0))
    fcut = float(_extract_param(p, ("fcut", "Fcut", "FCUT"), 0.0))

    rate = list(_extract_param(p, ("rate", "RATE", "Fp1"), []))
    yfac = list(_extract_param(p, ("yfac", "YFAC", "Fp2"), []))
    fun_norm = list(_extract_param(p, ("fun_norm", "FUN_NORM", "ABG_IPt"), []))
    fun_tang = list(_extract_param(p, ("fun_tang", "FUN_TANG", "ABG_IPdel"), []))
    sigy_n = float(_extract_param(p, ("sigy_n", "SIGY_N", "sigy0"), 100.0))
    sigy_t = float(_extract_param(p, ("sigy_t", "SIGY_T"), 50.0))

    return Law59Params(
        rho0=rho0,
        refer_rho=refer_rho,
        e=e,
        ecomp=ecomp,
        g=g,
        nrate=nrate,
        fcut=fcut,
        rate=rate,
        yfac=yfac,
        fun_norm=fun_norm,
        fun_tang=fun_tang,
        sigy_n=sigy_n,
        sigy_t=sigy_t,
        title=title,
    )


def build_law59(mat: Any) -> Law59Params:
    """Build Law59Params from Material or dictionary."""
    return resolve(mat)


def extra_shapes(mat: Any, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """State variables: eplas_norm, eplas_tang, off."""
    if nip > 1:
        return {"eplas_n": (nip,), "eplas_t": (nip,), "off59": (nip,)}
    return {"eplas_n": (), "eplas_t": (), "off59": ()}


def needs_defgrad(mat: Any) -> bool:
    return False


def sound_speed(mat: Any, eps: Optional[Any] = None, extra: Optional[Dict[str, Any]] = None) -> float:
    p = resolve(mat)
    return p.sound_speed


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Solid update for LAW59 structural connectors/springs (sigeps59.F)."""
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
    eplas_n = extra.get("eplas_n")
    if eplas_n is None or len(np.atleast_1d(eplas_n)) != nel:
        eplas_n = np.zeros(nel, dtype=float)
        extra["eplas_n"] = eplas_n
    else:
        eplas_n = np.atleast_1d(eplas_n).astype(float).copy()

    eplas_t = extra.get("eplas_t")
    if eplas_t is None or len(np.atleast_1d(eplas_t)) != nel:
        eplas_t = np.zeros(nel, dtype=float)
        extra["eplas_t"] = eplas_t
    else:
        eplas_t = np.atleast_1d(eplas_t).astype(float).copy()

    off_arr = extra.get("off59", extra.get("off"))
    if off_arr is None or len(np.atleast_1d(off_arr)) != nel:
        off_arr = np.ones(nel, dtype=float)
        extra["off59"] = off_arr
    else:
        off_arr = np.atleast_1d(off_arr).astype(float).copy()

    sign = np.zeros_like(sig_arr)
    c_arr = np.full(nel, p.sound_speed, dtype=float)

    for i in range(nel):
        if off_arr[i] <= 0.0:
            continue

        de_zz = deps_arr[i, 2] if deps_arr.shape[1] > 2 else deps_arr[i, 0]
        de_yz = deps_arr[i, 4] if deps_arr.shape[1] > 4 else 0.0
        de_zx = deps_arr[i, 5] if deps_arr.shape[1] > 5 else 0.0

        # Normal modulus
        e_mod = p.e if de_zz >= 0.0 else p.ecomp
        s_zz = (sig_arr[i, 2] if sig_arr.shape[1] > 2 else sig_arr[i, 0]) + e_mod * de_zz
        s_yz = (sig_arr[i, 4] if sig_arr.shape[1] > 4 else 0.0) + p.g * de_yz
        s_zx = (sig_arr[i, 5] if sig_arr.shape[1] > 5 else 0.0) + p.g * de_zx

        epsdot = math.sqrt(de_zz**2 + de_yz**2 + de_zx**2) / dt if dt > 0.0 else 0.0

        # 1. Normal plasticity
        yld_n, h_n = p.eval_normal_yield(eplas_n[i], epsdot)
        svm_n = abs(s_zz)
        if svm_n > yld_n and svm_n > _EM20:
            dpla_n = (svm_n - yld_n) / max(e_mod + h_n, _EM20)
            r_n = (yld_n + dpla_n * h_n) / svm_n
            s_zz *= r_n
            eplas_n[i] += dpla_n

        # 2. Tangent plasticity
        yld_t, h_t = p.eval_tangent_yield(eplas_t[i], epsdot)
        svm_t = math.sqrt(s_yz**2 + s_zx**2)
        if svm_t > yld_t and svm_t > _EM20:
            dpla_t = (svm_t - yld_t) / max(p.g + h_t, _EM20)
            r_t = (yld_t + dpla_t * h_t) / svm_t
            s_yz *= r_t
            s_zx *= r_t
            eplas_t[i] += dpla_t

        epsp_arr[i] = eplas_n[i] + eplas_t[i]

        sign[i, 0] = s_zz * 0.1  # lateral small constraint
        sign[i, 1] = s_zz * 0.1
        sign[i, 2] = s_zz
        if sign.shape[1] > 3:
            sign[i, 3] = 0.0
        if sign.shape[1] > 4:
            sign[i, 4] = s_yz
        if sign.shape[1] > 5:
            sign[i, 5] = s_zx

    extra["eplas_n"] = eplas_n
    extra["eplas_t"] = eplas_t
    extra["off59"] = off_arr

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
    """Shell representation of connector law."""
    p = resolve(mat)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).copy()
    deps_arr = np.atleast_2d(deps).copy()
    nel = sig_arr.shape[0]

    if epsp is None:
        epsp_arr = np.zeros(nel, dtype=float)
    else:
        epsp_arr = np.atleast_1d(epsp).astype(float).copy()

    sign = np.zeros_like(sig_arr)
    c_arr = np.full(nel, p.sound_speed, dtype=float)

    for i in range(nel):
        de_xx = deps_arr[i, 0]
        de_xy = deps_arr[i, 2] if deps_arr.shape[1] > 2 else 0.0
        e_mod = p.e if de_xx >= 0.0 else p.ecomp
        s_xx = sig_arr[i, 0] + e_mod * de_xx
        s_xy = (sig_arr[i, 2] if sig_arr.shape[1] > 2 else 0.0) + p.g * de_xy

        yld_n, h_n = p.eval_normal_yield(epsp_arr[i], 0.0)
        if abs(s_xx) > yld_n and abs(s_xx) > _EM20:
            dpla = (abs(s_xx) - yld_n) / max(e_mod + h_n, _EM20)
            s_xx = math.copysign(yld_n + dpla * h_n, s_xx)
            epsp_arr[i] += dpla

        sign[i, 0] = s_xx
        sign[i, 1] = 0.0
        if sign.shape[1] > 2:
            sign[i, 2] = s_xy

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
    c = np.zeros((6, 6), dtype=float)
    c[0, 0] = p.e * 0.1
    c[1, 1] = p.e * 0.1
    c[2, 2] = p.e
    c[3, 3] = p.g * 0.1
    c[4, 4] = p.g
    c[5, 5] = p.g
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
        [p.e, 0.0, 0.0],
        [0.0, p.e * 0.1, 0.0],
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
