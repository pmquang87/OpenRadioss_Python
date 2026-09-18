"""
LAW45 — Rate-dependent elasto-plastic Zhao model / viscoelastic orthotropic fabric (/MAT/LAW45).

Fortran origins:
- engine: ``engine/source/materials/mat/mat045/sigeps45.F`` (solids)
- engine: ``engine/source/materials/mat/mat045/sigeps45c.F`` (shells)
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
class Law45Params:
    """Parameters for /MAT/LAW45 (Zhao rate-dependent model)."""

    rho0: float = 1.0
    refer_rho: float = 1.0
    e: float = 1000.0
    nu: float = 0.3
    g: float = 0.0
    ca: float = 100.0
    cb: float = 0.0
    cn: float = 1.0
    epsm: float = _INF
    sigm: float = _INF
    cc: float = 0.0
    cd: float = 0.0
    cm: float = 1.0
    eps0: float = 1.0
    ce: float = 0.0
    ck: float = 1.0
    cutfre: float = 0.0
    title: str = ""

    # Derived
    k: float = field(init=False)
    a1: float = field(init=False)
    a2: float = field(init=False)
    c14g3: float = field(init=False)

    def __post_init__(self) -> None:
        if self.refer_rho == 0.0:
            self.refer_rho = self.rho0
        if self.g == 0.0:
            self.g = self.e / (2.0 * (1.0 + self.nu))
        denom_k = 3.0 * (1.0 - 2.0 * self.nu)
        self.k = self.e / denom_k if abs(denom_k) > 1e-12 else self.e
        self.c14g3 = self.k + (4.0 / 3.0) * self.g
        denom_shell = 1.0 - self.nu * self.nu
        if abs(denom_shell) > 1e-12:
            self.a1 = self.e / denom_shell
            self.a2 = self.nu * self.a1
        else:
            self.a1 = self.e
            self.a2 = 0.0
        if self.eps0 <= 0.0:
            self.eps0 = 1.0
        if self.epsm <= 0.0:
            self.epsm = _INF
        if self.sigm <= 0.0:
            self.sigm = _INF

    @property
    def young(self) -> float:
        return self.e

    @property
    def bulk(self) -> float:
        return self.k

    @property
    def shear(self) -> float:
        return self.g

    def eval_yield(self, epsp: float, epsdot: float) -> Tuple[float, float]:
        """Compute (sigma_y, H) matching sigeps45c.F."""
        if epsp <= 0.0:
            ch1 = self.ca
        elif epsp > self.epsm:
            ch1 = self.ca + self.cb * (self.epsm ** self.cn)
        else:
            ch1 = self.ca + self.cb * (epsp ** self.cn)

        if epsdot <= self.eps0:
            ch2 = 0.0
        elif epsp <= 0.0:
            ch2 = self.cc * math.log(epsdot / self.eps0)
        else:
            ch2 = (self.cc - self.cd * (epsp ** self.cm)) * math.log(epsdot / self.eps0)

        if epsdot <= 0.0:
            ch3 = 0.0
        else:
            ch3 = self.ce * (epsdot ** self.ck)

        sigy = min(self.sigm + ch3, ch1 + ch2 + ch3)

        # Hardening modulus H = d(sigy)/d(epsp)
        if epsp > 0.0 and self.cn >= 1.0:
            qh1 = self.cb * self.cn * (epsp ** (self.cn - 1.0))
        elif epsp > 0.0 and self.cn < 1.0:
            qh1 = self.cb * self.cn * (epsp ** (1.0 - self.cn))
        else:
            qh1 = 0.0

        if epsp <= 0.0 or epsdot <= self.eps0:
            qh2 = 0.0
        elif self.cm >= 1.0:
            qh2 = -self.cd * self.cm * (epsp ** (self.cm - 1.0)) * math.log(epsdot / self.eps0)
        else:
            qh2 = -self.cd * self.cm * (epsp ** (1.0 - self.cm)) * math.log(epsdot / self.eps0)

        h = max(0.0, qh1 + qh2)
        return sigy, h


def _extract_param(d: Dict[str, Any], keys: Tuple[str, ...], default: Any = 0.0) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def resolve(mat: Any) -> Law45Params:
    """Resolve Law45Params from Material entity, dict, or Law45Params."""
    if isinstance(mat, Law45Params):
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

    e = float(_extract_param(p, ("E", "e", "MAT_E", "young"), 1000.0))
    nu = float(_extract_param(p, ("nu", "NU", "MAT_NU", "pr"), 0.3))
    g = float(_extract_param(p, ("G", "g", "MAT_G"), 0.0))

    ca = float(_extract_param(p, ("ca", "CA", "MAT_A", "a", "sig_y"), 100.0))
    cb = float(_extract_param(p, ("cb", "CB", "MAT_B", "b"), 0.0))
    cn = float(_extract_param(p, ("cn", "CN", "MAT_N", "n"), 1.0))
    epsm = float(_extract_param(p, ("epsm", "EPSM", "MAT_EPS", "eps_max"), _INF))
    sigm = float(_extract_param(p, ("sigm", "SIGM", "MAT_SIG", "sig_max"), _INF))

    cc = float(_extract_param(p, ("cc", "CC", "MAT_C", "c"), 0.0))
    cd = float(_extract_param(p, ("cd", "CD", "MAT_D", "d"), 0.0))
    cm = float(_extract_param(p, ("cm", "CM", "MAT_M", "m"), 1.0))
    eps0 = float(_extract_param(p, ("eps0", "EPS0", "MAT_EPS0"), 1.0))

    ce = float(_extract_param(p, ("ce", "CE", "MAT_E_VISC"), 0.0))
    ck = float(_extract_param(p, ("ck", "CK", "MAT_K_VISC"), 1.0))
    cutfre = float(_extract_param(p, ("cutfre", "CUTFRE", "fcut", "Fcut"), 0.0))

    return Law45Params(
        rho0=rho0,
        refer_rho=refer_rho,
        e=e,
        nu=nu,
        g=g,
        ca=ca,
        cb=cb,
        cn=cn,
        epsm=epsm,
        sigm=sigm,
        cc=cc,
        cd=cd,
        cm=cm,
        eps0=eps0,
        ce=ce,
        ck=ck,
        cutfre=cutfre,
        title=title,
    )


def build_law45(mat: Any) -> Law45Params:
    """Build Law45Params from Material, GenericMaterialRecord, or dict."""
    return resolve(mat)


def extra_shapes(mat: Any, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """State variables: uvar (5,) [epsp, epsdot_filtered, sigy, h, epsdot_old]."""
    if nip > 1:
        return {"uvar45": (nip, 5)}
    return {"uvar45": (5,)}


def needs_defgrad(mat: Any) -> bool:
    return False


def sound_speed(mat: Any, eps: Optional[Any] = None, extra: Optional[Dict[str, Any]] = None) -> float:
    """Sound speed c = sqrt(A1 / rho0) for shells or sqrt(c14g3 / rho0) for solids."""
    p = resolve(mat)
    return math.sqrt(max(p.c14g3 / p.rho0, _EM20))


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Plane-stress update for shells (sigeps45c.F)."""
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
    uvar = extra.get("uvar45", extra.get("uvar"))
    if uvar is None or uvar.shape[0] != nel:
        uvar = np.zeros((nel, 5), dtype=float)
        extra["uvar45"] = uvar

    beta = 1.0
    if p.cutfre > 0.0 and dt > 0.0:
        beta = min(1.0, dt * 2.0 * math.pi * p.cutfre)

    gs = (5.0 / 6.0) * p.g
    sign = np.zeros_like(sig_arr)
    c_arr = np.zeros(nel, dtype=float)
    c_val = math.sqrt(max(p.a1 / p.rho0, _EM20))

    for i in range(nel):
        de_xx = deps_arr[i, 0]
        de_yy = deps_arr[i, 1]
        de_xy = deps_arr[i, 2] if deps_arr.shape[1] > 2 else 0.0
        de_yz = deps_arr[i, 3] if deps_arr.shape[1] > 3 else 0.0
        de_zx = deps_arr[i, 4] if deps_arr.shape[1] > 4 else 0.0

        # Elastic trial
        s_xx = sig_arr[i, 0] + p.a1 * de_xx + p.a2 * de_yy
        s_yy = sig_arr[i, 1] + p.a2 * de_xx + p.a1 * de_yy
        s_xy = sig_arr[i, 2] + p.g * de_xy if sig_arr.shape[1] > 2 else 0.0
        s_yz = sig_arr[i, 3] + gs * de_yz if sig_arr.shape[1] > 3 else 0.0
        s_zx = sig_arr[i, 4] + gs * de_zx if sig_arr.shape[1] > 4 else 0.0

        # Strain rate calculation
        epsdot_raw = 0.0
        if dt > 0.0:
            epsdot_raw = math.sqrt((de_xx / dt)**2 + (de_yy / dt)**2 + 0.5 * (de_xy / dt)**2)
        epsdot_filt = beta * epsdot_raw + (1.0 - beta) * uvar[i, 4]
        uvar[i, 4] = epsdot_filt

        # Yield stress & hardening
        sigy, h_slope = p.eval_yield(epsp_arr[i], epsdot_filt)
        svm = math.sqrt(max(s_xx**2 + s_yy**2 - s_xx * s_yy + 3.0 * s_xy**2, 0.0))

        if svm > sigy and svm > _EM20:
            scale = sigy / svm
            s_xx *= scale
            s_yy *= scale
            s_xy *= scale
            dpla = (1.0 - scale) * svm / max(p.e + h_slope, _EM20)
            epsp_arr[i] += dpla

        sign[i, 0] = s_xx
        sign[i, 1] = s_yy
        if sign.shape[1] > 2:
            sign[i, 2] = s_xy
        if sign.shape[1] > 3:
            sign[i, 3] = s_yz
        if sign.shape[1] > 4:
            sign[i, 4] = s_zx
        c_arr[i] = c_val

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
    """3D solid update (sigeps45.F)."""
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
    uvar = extra.get("uvar45", extra.get("uvar"))
    if uvar is None or uvar.shape[0] != nel:
        uvar = np.zeros((nel, 5), dtype=float)
        extra["uvar45"] = uvar

    beta = 1.0
    if p.cutfre > 0.0 and dt > 0.0:
        beta = min(1.0, dt * 2.0 * math.pi * p.cutfre)

    sign = np.zeros_like(sig_arr)
    c_arr = np.zeros(nel, dtype=float)
    c_val = math.sqrt(max(p.c14g3 / p.rho0, _EM20))

    for i in range(nel):
        dtrace = deps_arr[i, 0] + deps_arr[i, 1] + deps_arr[i, 2]
        dvol = dtrace / 3.0
        pres_old = (sig_arr[i, 0] + sig_arr[i, 1] + sig_arr[i, 2]) / 3.0
        pres_new = pres_old + p.k * dtrace

        # Deviatoric trial
        s_xx = sig_arr[i, 0] - pres_old + 2.0 * p.g * (deps_arr[i, 0] - dvol)
        s_yy = sig_arr[i, 1] - pres_old + 2.0 * p.g * (deps_arr[i, 1] - dvol)
        s_zz = sig_arr[i, 2] - pres_old + 2.0 * p.g * (deps_arr[i, 2] - dvol)
        s_xy = sig_arr[i, 3] + p.g * deps_arr[i, 3] if sig_arr.shape[1] > 3 else 0.0
        s_yz = sig_arr[i, 4] + p.g * deps_arr[i, 4] if sig_arr.shape[1] > 4 else 0.0
        s_zx = sig_arr[i, 5] + p.g * deps_arr[i, 5] if sig_arr.shape[1] > 5 else 0.0

        j2 = 0.5 * (s_xx**2 + s_yy**2 + s_zz**2) + s_xy**2 + s_yz**2 + s_zx**2
        svm = math.sqrt(max(3.0 * j2, 0.0))

        epsdot_raw = 0.0
        if dt > 0.0:
            epsdot_raw = math.sqrt(max((2.0 / 3.0) * (
                (deps_arr[i, 0] - dvol)**2 + (deps_arr[i, 1] - dvol)**2 + (deps_arr[i, 2] - dvol)**2 +
                0.5 * (deps_arr[i, 3]**2 + deps_arr[i, 4]**2 + deps_arr[i, 5]**2)
            ), 0.0)) / dt
        epsdot_filt = beta * epsdot_raw + (1.0 - beta) * uvar[i, 4]
        uvar[i, 4] = epsdot_filt

        sigy, h_slope = p.eval_yield(epsp_arr[i], epsdot_filt)
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

        sign[i, 0] = s_xx + pres_new
        sign[i, 1] = s_yy + pres_new
        sign[i, 2] = s_zz + pres_new
        if sign.shape[1] > 3:
            sign[i, 3] = s_xy
        if sign.shape[1] > 4:
            sign[i, 4] = s_yz
        if sign.shape[1] > 5:
            sign[i, 5] = s_zx
        c_arr[i] = c_val

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
    """Consistent shell tangent stiffness matrix C (3, 3)."""
    p = resolve(mat)
    c_plane = np.array([
        [p.a1, p.a2, 0.0],
        [p.a2, p.a1, 0.0],
        [0.0, 0.0, p.g],
    ], dtype=float)
    if sig is not None and sig.ndim == 2:
        return np.broadcast_to(c_plane, (sig.shape[0], 3, 3)).copy()
    return c_plane


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
    """Consistent solid tangent stiffness matrix C (6, 6)."""
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
