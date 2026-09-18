"""
LAW64 — Rate-dependent TRIP Steel Plasticity Model (/MAT/LAW64).

Fortran origins:
- engine: ``engine/source/materials/mat/mat064/sigeps64c.F``
- starter: ``starter/source/materials/mat/mat064/hm_read_mat64.F``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pyradioss.model.entities import Material

_EM20 = 1e-20
_INF = 1e30


def _eval_curve(curve: Any, x: float) -> Tuple[float, float]:
    """Evaluate curve f(x) and its derivative df/dx."""
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
class Law64Params:
    """Parameters for /MAT/LAW64 (TRIP steel plastic law)."""

    rho0: float = 1.0
    refer_rho: float = 1.0
    e: float = 210000.0
    nu: float = 0.3
    cp: float = 450.0

    d: float = 1.0
    n: float = 1.0
    md: float = 350.0  # Limit martensite temperature
    v0: float = 0.05
    vmc: float = 1.0   # Content martensite volume fraction
    temp0: float = 293.15
    yfac1: float = 1.0
    yfac2: float = 1.0
    hl: float = 0.0

    fun1: Any = None
    fun2: Any = None
    sigy1_default: float = 300.0
    sigy2_default: float = 800.0

    title: str = ""

    # Derived
    g: float = field(init=False)
    k: float = field(init=False)
    a1: float = field(init=False)
    a2: float = field(init=False)
    c_shell: float = field(init=False)
    c_solid: float = field(init=False)

    def __post_init__(self) -> None:
        if self.refer_rho == 0.0:
            self.refer_rho = self.rho0
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

        if self.vmc <= 0.0 or self.vmc > 1.0:
            self.vmc = 1.0
        if self.v0 <= 0.0 or self.v0 >= 1.0:
            self.v0 = _EM20
        if self.yfac1 == 0.0:
            self.yfac1 = 1.0
        if self.yfac2 == 0.0:
            self.yfac2 = 1.0

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

    def eval_yield(self, pla: float, vm: float) -> Tuple[float, float]:
        """Evaluate (yield_stress, hardening_modulus) matching sigeps64c.F."""
        if self.fun1 is not None:
            y1, h1 = _eval_curve(self.fun1, pla)
            y1 *= self.yfac1
            h1 *= self.yfac1
        else:
            y1 = self.sigy1_default * self.yfac1
            h1 = 0.0

        if self.fun2 is not None:
            y2, h2 = _eval_curve(self.fun2, pla)
            y2 *= self.yfac2
            h2 *= self.yfac2
        else:
            y2 = self.sigy2_default * self.yfac2
            h2 = 0.0

        fac = vm / max(self.vmc, _EM20)
        yld = y1 + fac * (y2 - y1)
        h = h1 + fac * (h2 - h1)
        return max(yld, _EM20), max(h, 0.0)

    def eval_dvm(self, pla: float, temp: float) -> float:
        """Compute rate of martensite volume fraction dVM/d(epsp)."""
        if temp >= self.md:
            return 0.0
        cd_pla = max(self.d * pla, _EM20)
        term1 = self.n * self.d * (cd_pla ** (self.n - 1.0))
        term2 = math.exp(-(cd_pla ** self.n))
        term3 = self.v0 * math.log(max(self.md - temp + 1.0, 1.0))
        return max(term1 * term2 * term3, 0.0)


def _extract_param(d: Dict[str, Any], keys: Tuple[str, ...], default: Any = 0.0) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def resolve(mat: Any) -> Law64Params:
    """Resolve Law64Params from Material entity, dict, or Law64Params."""
    if isinstance(mat, Law64Params):
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

    e = float(_extract_param(p, ("e", "E", "MAT_E", "young"), 210000.0))
    nu = float(_extract_param(p, ("nu", "NU", "MAT_NU", "pr"), 0.3))
    cp = float(_extract_param(p, ("cp", "CP", "MAT_CP"), 450.0))

    d_val = float(_extract_param(p, ("d", "D", "MAT_D"), 1.0))
    n_val = float(_extract_param(p, ("n", "N", "MAT_N"), 1.0))
    md = float(_extract_param(p, ("md", "MD", "M64_Md", "LIMIT_MARTENSITE_TEMP"), 350.0))
    v0 = float(_extract_param(p, ("v0", "V0", "M64_Vo"), 0.05))
    vmc = float(_extract_param(p, ("vmc", "VMC", "M64_Vm"), 1.0))

    temp0 = float(_extract_param(p, ("temp0", "TEMP", "M64_INI_TEMP", "T0"), 293.15))
    yfac1 = float(_extract_param(p, ("yfac1", "YFAC1", "M64_SCALE_0", "scale1"), 1.0))
    yfac2 = float(_extract_param(p, ("yfac2", "YFAC2", "M64_SCALE_1", "scale2"), 1.0))
    hl = float(_extract_param(p, ("hl", "HL", "MAT_HL"), 0.0))

    fun1 = _extract_param(p, ("fun1", "FUN1", "M64_FUNCT_ID_0", "func1"), None)
    fun2 = _extract_param(p, ("fun2", "FUN2", "M64_FUNCT_ID_1", "func2"), None)
    sigy1_default = float(_extract_param(p, ("sigy1_default", "SIGY1", "sigy0"), 300.0))
    sigy2_default = float(_extract_param(p, ("sigy2_default", "SIGY2"), 800.0))

    return Law64Params(
        rho0=rho0,
        refer_rho=refer_rho,
        e=e,
        nu=nu,
        cp=cp,
        d=d_val,
        n=n_val,
        md=md,
        v0=v0,
        vmc=vmc,
        temp0=temp0,
        yfac1=yfac1,
        yfac2=yfac2,
        hl=hl,
        fun1=fun1,
        fun2=fun2,
        sigy1_default=sigy1_default,
        sigy2_default=sigy2_default,
        title=title,
    )


def build_law64(mat: Any) -> Law64Params:
    """Build Law64Params from Material or dictionary."""
    return resolve(mat)


def extra_shapes(mat: Any, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """State variables: uvar (7,) [pla, vm, ipos1, ipos2, temp, dvm, spare]."""
    if nip > 1:
        return {"uvar64": (nip, 7)}
    return {"uvar64": (7,)}


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
    """Shell update for TRIP steel LAW64 (sigeps64c.F)."""
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
    uvar = extra.get("uvar64", extra.get("uvar"))
    if uvar is None or uvar.shape[0] != nel:
        uvar = np.zeros((nel, 7), dtype=float)
        uvar[:, 1] = 0.0  # VM
        uvar[:, 4] = p.temp0
        extra["uvar64"] = uvar

    gs = (5.0 / 6.0) * p.g
    sign = np.zeros_like(sig_arr)
    c_arr = np.full(nel, p.c_shell, dtype=float)

    for i in range(nel):
        de_xx = deps_arr[i, 0]
        de_yy = deps_arr[i, 1]
        de_xy = deps_arr[i, 2] if deps_arr.shape[1] > 2 else 0.0

        s_xx = sig_arr[i, 0] + p.a1 * de_xx + p.a2 * de_yy
        s_yy = sig_arr[i, 1] + p.a2 * de_xx + p.a1 * de_yy
        s_xy = sig_arr[i, 2] + p.g * de_xy if sig_arr.shape[1] > 2 else 0.0
        s_yz = sig_arr[i, 3] + gs * deps_arr[i, 3] if sig_arr.shape[1] > 3 else 0.0
        s_zx = sig_arr[i, 4] + gs * deps_arr[i, 4] if sig_arr.shape[1] > 4 else 0.0

        vm = uvar[i, 1]
        temp = uvar[i, 4]
        sigy, h_slope = p.eval_yield(epsp_arr[i], vm)

        svm = math.sqrt(max(s_xx**2 + s_yy**2 - s_xx * s_yy + 3.0 * s_xy**2, 0.0))
        dpla = 0.0
        if svm > sigy and svm > _EM20:
            scale = sigy / svm
            s_xx *= scale
            s_yy *= scale
            s_xy *= scale
            dpla = (1.0 - scale) * svm / max(p.e + h_slope, _EM20)
            epsp_arr[i] += dpla

            # Martensite kinetics
            dvm = p.eval_dvm(epsp_arr[i], temp)
            vm = min(1.0, vm + max(dvm * dpla, 0.0))

            # Temperature rise (adiabatic)
            d_work = (s_xx * de_xx + s_yy * de_yy + s_xy * de_xy)
            dtemp = (d_work + (vm - uvar[i, 1]) * p.hl) / max(p.rho0 * p.cp, _EM20)
            temp += dtemp

        uvar[i, 0] = epsp_arr[i]
        uvar[i, 1] = vm
        uvar[i, 4] = temp

        sign[i, 0] = s_xx
        sign[i, 1] = s_yy
        if sign.shape[1] > 2:
            sign[i, 2] = s_xy
        if sign.shape[1] > 3:
            sign[i, 3] = s_yz
        if sign.shape[1] > 4:
            sign[i, 4] = s_zx

    extra["uvar64"] = uvar

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
    """Solid update for LAW64."""
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
    uvar = extra.get("uvar64", extra.get("uvar"))
    if uvar is None or uvar.shape[0] != nel:
        uvar = np.zeros((nel, 7), dtype=float)
        uvar[:, 1] = 0.0
        uvar[:, 4] = p.temp0
        extra["uvar64"] = uvar

    sign = np.zeros_like(sig_arr)
    c_arr = np.full(nel, p.c_solid, dtype=float)

    for i in range(nel):
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

        vm = uvar[i, 1]
        temp = uvar[i, 4]
        sigy, h_slope = p.eval_yield(epsp_arr[i], vm)

        if svm > sigy and svm > _EM20:
            scale = sigy / svm
            s_xx *= scale
            s_yy *= scale
            s_zz *= scale
            s_xy *= scale
            s_yz *= scale
            s_zx *= scale
            dpla = (1.0 - scale) * svm / max(3.0 * p.g + h_slope, _EM20)
            epsp_arr[i] += dpla

            dvm = p.eval_dvm(epsp_arr[i], temp)
            vm = min(1.0, vm + max(dvm * dpla, 0.0))
            uvar[i, 1] = vm

            d_work = sigy * dpla
            dtemp = (d_work + (vm - uvar[i, 1]) * p.hl) / max(p.rho0 * p.cp, _EM20)
            temp += dtemp
            uvar[i, 4] = temp

        uvar[i, 0] = epsp_arr[i]

        sign[i, 0] = s_xx + pres_new
        sign[i, 1] = s_yy + pres_new
        sign[i, 2] = s_zz + pres_new
        if sign.shape[1] > 3:
            sign[i, 3] = s_xy
        if sign.shape[1] > 4:
            sign[i, 4] = s_yz
        if sign.shape[1] > 5:
            sign[i, 5] = s_zx

    extra["uvar64"] = uvar

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
