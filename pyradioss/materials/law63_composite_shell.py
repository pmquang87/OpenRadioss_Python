"""
LAW63 — TRIP Steel / Transformation Induced Plasticity Shell Model (/MAT/LAW63).

Fortran origins:
- engine: ``engine/source/materials/mat/mat063/sigeps63c.F``
- starter: ``starter/source/materials/mat/mat063/hm_read_mat63.F``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pyradioss.model.entities import Material

_EM20 = 1e-20
_INF = 1e30


@dataclass
class Law63Params:
    """Parameters for /MAT/LAW63 (TRIP steel plastic law)."""

    rho0: float = 1.0
    refer_rho: float = 1.0
    e: float = 210000.0
    nu: float = 0.3
    cp: float = 450.0

    a: float = 1.0
    b: float = -1.0
    q: float = 0.0
    c: float = 0.0
    d: float = 0.0
    p: float = 1.0

    ahs: float = 300.0
    bhs: float = 800.0
    cm: float = 10.0
    cn: float = 0.2
    k1: float = 1.0
    k2: float = 0.0
    dh: float = 100.0
    vm0: float = 0.0
    eps0: float = 0.001

    temp0: float = 293.15
    hl: float = 0.0
    coef: float = 0.9

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

        if self.coef <= 0.0 or self.coef > 1.0:
            self.coef = 0.9
        if self.eps0 <= 0.0:
            self.eps0 = 0.001

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

    def eval_yield(self, pla: float, temp: float, vm: float) -> Tuple[float, float]:
        """Evaluate (sigma_y, H) matching sigeps63c.F."""
        aux0 = max(pla + self.eps0, _EM20)
        aux2 = aux0 ** self.cn
        aux3 = (self.bhs - self.ahs) * math.exp(-self.cm * aux2)
        kt = self.k1 + self.k2 * temp

        yld = (self.bhs - aux3) * kt + self.dh * vm
        h = self.cm * self.cn * (aux0 ** (self.cn - 1.0)) * aux3 * kt
        return max(yld, _EM20), max(h, 0.0)


def _extract_param(d: Dict[str, Any], keys: Tuple[str, ...], default: Any = 0.0) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def resolve(mat: Any) -> Law63Params:
    """Resolve Law63Params from Material entity, dict, or Law63Params."""
    if isinstance(mat, Law63Params):
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

    a = float(_extract_param(p, ("a", "A", "MAT_A"), 1.0))
    b = float(_extract_param(p, ("b", "B", "MAT_B"), -1.0))
    q = float(_extract_param(p, ("q", "Q", "MAT_Q"), 0.0))
    c = float(_extract_param(p, ("c", "C", "MAT_C"), 0.0))
    d = float(_extract_param(p, ("d", "D", "MAT_D"), 0.0))
    p_val = float(_extract_param(p, ("p", "P", "MAT_P0"), 1.0))

    ahs = float(_extract_param(p, ("ahs", "AHS", "A_HS"), 300.0))
    bhs = float(_extract_param(p, ("bhs", "BHS", "B_HS"), 800.0))
    cm = float(_extract_param(p, ("cm", "CM", "MAT_M"), 10.0))
    cn = float(_extract_param(p, ("cn", "CN", "MAT_N"), 0.2))
    k1 = float(_extract_param(p, ("k1", "K1"), 1.0))
    k2 = float(_extract_param(p, ("k2", "K2"), 0.0))
    dh = float(_extract_param(p, ("dh", "DH", "delta_H"), 100.0))
    vm0 = float(_extract_param(p, ("vm0", "VM0", "Kvm"), 0.0))
    eps0 = float(_extract_param(p, ("eps0", "EPS0", "Epsilon_0"), 0.001))

    temp0 = float(_extract_param(p, ("temp0", "TEMP", "MAT_T0"), 293.15))
    hl = float(_extract_param(p, ("hl", "HL", "MAT_HL"), 0.0))
    coef = float(_extract_param(p, ("coef", "COEF", "MAT_ETA"), 0.9))

    return Law63Params(
        rho0=rho0,
        refer_rho=refer_rho,
        e=e,
        nu=nu,
        cp=cp,
        a=a,
        b=b,
        q=q,
        c=c,
        d=d,
        p=p_val,
        ahs=ahs,
        bhs=bhs,
        cm=cm,
        cn=cn,
        k1=k1,
        k2=k2,
        dh=dh,
        vm0=vm0,
        eps0=eps0,
        temp0=temp0,
        hl=hl,
        coef=coef,
        title=title,
    )


def build_law63(mat: Any) -> Law63Params:
    """Build Law63Params from Material or dictionary."""
    return resolve(mat)


def extra_shapes(mat: Any, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """State variables: uvar (4,) [pla, vm, dvm, temp]."""
    if nip > 1:
        return {"uvar63": (nip, 4)}
    return {"uvar63": (4,)}


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
    """Shell update for TRIP steel LAW63 (sigeps63c.F)."""
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
    uvar = extra.get("uvar63", extra.get("uvar"))
    if uvar is None or uvar.shape[0] != nel:
        uvar = np.zeros((nel, 4), dtype=float)
        uvar[:, 1] = p.vm0
        uvar[:, 3] = p.temp0
        extra["uvar63"] = uvar

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
        temp = uvar[i, 3]
        sigy, h_slope = p.eval_yield(epsp_arr[i], temp, vm)

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
            dvm = 0.5 * (1.0 - math.tanh(p.c + p.d * temp))
            if p.b != 0.0 and vm > 0.0 and vm < 1.0:
                arg = p.q / max(temp, _EM20) + ((p.b + 1.0) / p.b) * math.log((1.0 - vm) / vm) + p.p * math.log(vm)
                dvm = abs(p.b) * math.exp(min(arg, 50.0)) * dvm / max(p.a, _EM20)
            vm = min(1.0, vm + max(dvm * dpla, 0.0))

            # Temperature rise from plastic work and latent heat
            d_work = (s_xx * de_xx + s_yy * de_yy + s_xy * de_xy)
            dtemp = (p.coef * d_work + (vm - uvar[i, 1]) * p.hl) / max(p.rho0 * p.cp, _EM20)
            temp += dtemp

        uvar[i, 0] = epsp_arr[i]
        uvar[i, 1] = vm
        uvar[i, 3] = temp

        sign[i, 0] = s_xx
        sign[i, 1] = s_yy
        if sign.shape[1] > 2:
            sign[i, 2] = s_xy
        if sign.shape[1] > 3:
            sign[i, 3] = s_yz
        if sign.shape[1] > 4:
            sign[i, 4] = s_zx

    extra["uvar63"] = uvar

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
    """Solid update for LAW63."""
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
    uvar = extra.get("uvar63", extra.get("uvar"))
    if uvar is None or uvar.shape[0] != nel:
        uvar = np.zeros((nel, 4), dtype=float)
        uvar[:, 1] = p.vm0
        uvar[:, 3] = p.temp0
        extra["uvar63"] = uvar

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
        temp = uvar[i, 3]
        sigy, h_slope = p.eval_yield(epsp_arr[i], temp, vm)

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

            dvm = 0.5 * (1.0 - math.tanh(p.c + p.d * temp))
            vm = min(1.0, vm + max(dvm * dpla, 0.0))
            uvar[i, 1] = vm

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

    extra["uvar63"] = uvar

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
