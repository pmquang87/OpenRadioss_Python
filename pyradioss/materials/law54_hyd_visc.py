"""
LAW54 — Hydrodynamic viscous fluid / component elastoplastic damage model (/MAT/LAW54).

Fortran origins:
- starter: ``starter/source/materials/mat/mat054/hm_read_mat54.F``
- cfg: ``hm_cfg_files/config/CFG/radioss110/MAT/matl54_54.cfg``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pyradioss.model.entities import Material

_EM20 = 1e-20
_INF = 1e30


def _eval_curve(curve: Any, x: float, scale: float = 1.0) -> Tuple[float, float]:
    """Evaluate curve f(x)*scale and slope df/dx*scale."""
    if curve is None:
        return 0.0, 0.0
    if callable(curve):
        y = float(curve(x)) * scale
        h = max(abs(x) * 1e-6, 1e-8)
        yp = float(curve(x + h)) * scale
        ym = float(curve(x - h)) * scale
        return y, (yp - ym) / (2.0 * h)
    xs = getattr(curve, "x", None)
    ys = getattr(curve, "y", None)
    if xs is not None and ys is not None:
        y = float(np.interp(x, xs, ys)) * scale
        h = max(abs(x) * 1e-6, 1e-8)
        yp = float(np.interp(x + h, xs, ys)) * scale
        ym = float(np.interp(x - h, xs, ys)) * scale
        return y, (yp - ym) / (2.0 * h)
    return 0.0, 0.0


@dataclass
class Law54Params:
    """Parameters for /MAT/LAW54."""

    rho0: float = 1.0
    refer_rho: float = 1.0
    e: float = 1000.0
    nu: float = 0.3
    func: Any = None
    sig0: float = 100.0
    h: float = 0.0
    m: float = 1.0
    sfac: float = 1.0

    ay: float = 1.0
    az: float = 1.0
    by: float = 1.0
    bz: float = 1.0
    cx: float = 1.0

    dc: float = 0.99999
    pr: float = _INF  # rupture strain
    ps: float = 0.0  # threshold strain

    title: str = ""

    # Derived
    k: float = field(init=False)
    g: float = field(init=False)
    a1: float = field(init=False)
    a2: float = field(init=False)
    c_solid: float = field(init=False)
    c_shell: float = field(init=False)

    def __post_init__(self) -> None:
        if self.refer_rho == 0.0:
            self.refer_rho = self.rho0
        if self.sfac == 0.0:
            self.sfac = 1.0
        if self.dc <= 0.0 or self.dc >= 1.0:
            self.dc = 0.99999
        if self.pr == 0.0:
            self.pr = _INF

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

        c14g3 = self.k + (4.0 / 3.0) * self.g
        self.c_solid = math.sqrt(max(c14g3 / max(self.rho0, _EM20), _EM20))
        self.c_shell = math.sqrt(max(self.a1 / max(self.rho0, _EM20), _EM20))

    @property
    def young(self) -> float:
        return self.e

    @property
    def bulk(self) -> float:
        return self.k

    @property
    def shear(self) -> float:
        return self.g

    @property
    def c_sound(self) -> float:
        return self.c_solid

    def eval_yield(self, epsp: float) -> Tuple[float, float]:
        """Evaluate (sigma_y, H)."""
        if self.func is not None:
            f_val, f_slope = _eval_curve(self.func, epsp, self.sfac)
            if f_val > 0.0:
                return f_val, f_slope
        if epsp <= 0.0:
            return self.sig0, self.e
        yld = self.sig0 + self.h * (epsp ** self.m)
        slope = self.h * self.m * (epsp ** (self.m - 1.0)) if self.m >= 1.0 else 0.0
        return yld, slope

    def eval_damage(self, epsp: float) -> float:
        """Evaluate damage D in [0, 1]."""
        if epsp <= self.ps:
            return 0.0
        if self.pr <= self.ps:
            return 1.0 if epsp >= self.pr else 0.0
        dmg = self.dc * (epsp - self.ps) / (self.pr - self.ps)
        return min(1.0, max(0.0, dmg))


def _extract_param(d: Dict[str, Any], keys: Tuple[str, ...], default: Any = 0.0) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def resolve(mat: Any) -> Law54Params:
    """Resolve Law54Params from Material entity, dict, or Law54Params."""
    if isinstance(mat, Law54Params):
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
    nu = float(_extract_param(p, ("nu", "NU", "MAT_NU", "poisson"), 0.3))
    func = _extract_param(p, ("func", "FUNC", "fct"), None)
    sig0 = float(_extract_param(p, ("sig0", "SIG0", "MAT_A", "a", "sig_y"), 100.0))
    h = float(_extract_param(p, ("h", "H", "MAT_B", "b"), 0.0))
    m = float(_extract_param(p, ("m", "M", "MAT_N", "n"), 1.0))
    sfac = float(_extract_param(p, ("sfac", "SFAC", "MAT_Sfac_Yield"), 1.0))

    ay = float(_extract_param(p, ("ay", "AY", "MAT_Ay"), 1.0))
    az = float(_extract_param(p, ("az", "AZ", "MAT_Az"), 1.0))
    by = float(_extract_param(p, ("by", "BY", "MAT_By"), 1.0))
    bz = float(_extract_param(p, ("bz", "BZ", "MAT_Bz"), 1.0))
    cx = float(_extract_param(p, ("cx", "CX", "MAT_Cx"), 1.0))

    dc = float(_extract_param(p, ("dc", "DC", "MAT_Dc"), 0.99999))
    pr = float(_extract_param(p, ("pr", "PR", "MAT_Rc", "eps_max"), _INF))
    ps = float(_extract_param(p, ("ps", "PS", "MAT_EPS", "eps_th"), 0.0))

    return Law54Params(
        rho0=rho0,
        refer_rho=refer_rho,
        e=e,
        nu=nu,
        func=func,
        sig0=sig0,
        h=h,
        m=m,
        sfac=sfac,
        ay=ay,
        az=az,
        by=by,
        bz=bz,
        cx=cx,
        dc=dc,
        pr=pr,
        ps=ps,
        title=title,
    )


def build_law54(mat: Any) -> Law54Params:
    """Build Law54Params from Material or dictionary."""
    return resolve(mat)


def extra_shapes(mat: Any, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """Persistent damage and deletion status."""
    if nip > 1:
        return {"dmg54": (nip,), "off54": (nip,)}
    return {"dmg54": (), "off54": ()}


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
    """Solid update for LAW54."""
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
    dmg_arr = extra.get("dmg54", extra.get("dmg"))
    if dmg_arr is None or len(np.atleast_1d(dmg_arr)) != nel:
        dmg_arr = np.zeros(nel, dtype=float)
        extra["dmg54"] = dmg_arr
    else:
        dmg_arr = np.atleast_1d(dmg_arr).astype(float).copy()

    off_arr = extra.get("off54", extra.get("off"))
    if off_arr is None or len(np.atleast_1d(off_arr)) != nel:
        off_arr = np.ones(nel, dtype=float)
        extra["off54"] = off_arr
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

        sigy, h_slope = p.eval_yield(epsp_arr[i])
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

        # Damage update
        dmg = p.eval_damage(epsp_arr[i])
        dmg_arr[i] = dmg
        if dmg >= p.dc or epsp_arr[i] >= p.pr:
            off_arr[i] = 0.0
            sign[i, :] = 0.0
        else:
            deg = 1.0 - dmg
            sign[i, 0] = (s_xx + pres_new) * deg
            sign[i, 1] = (s_yy + pres_new) * deg
            sign[i, 2] = (s_zz + pres_new) * deg
            if sign.shape[1] > 3:
                sign[i, 3] = s_xy * deg
            if sign.shape[1] > 4:
                sign[i, 4] = s_yz * deg
            if sign.shape[1] > 5:
                sign[i, 5] = s_zx * deg

    extra["dmg54"] = dmg_arr
    extra["off54"] = off_arr

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
    """Shell plane-stress update for LAW54."""
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
    dmg_arr = extra.get("dmg54", extra.get("dmg"))
    if dmg_arr is None or len(np.atleast_1d(dmg_arr)) != nel:
        dmg_arr = np.zeros(nel, dtype=float)
        extra["dmg54"] = dmg_arr
    else:
        dmg_arr = np.atleast_1d(dmg_arr).astype(float).copy()

    off_arr = extra.get("off54", extra.get("off"))
    if off_arr is None or len(np.atleast_1d(off_arr)) != nel:
        off_arr = np.ones(nel, dtype=float)
        extra["off54"] = off_arr
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

        svm = math.sqrt(max(s_xx**2 + s_yy**2 - s_xx * s_yy + 3.0 * s_xy**2, 0.0))
        sigy, h_slope = p.eval_yield(epsp_arr[i])
        if svm > sigy and svm > _EM20:
            scale = sigy / svm
            s_xx *= scale
            s_yy *= scale
            s_xy *= scale
            dpla = (1.0 - scale) * svm / max(p.e + h_slope, _EM20)
            epsp_arr[i] += dpla

        dmg = p.eval_damage(epsp_arr[i])
        dmg_arr[i] = dmg
        if dmg >= p.dc or epsp_arr[i] >= p.pr:
            off_arr[i] = 0.0
            sign[i, :] = 0.0
        else:
            deg = 1.0 - dmg
            sign[i, 0] = s_xx * deg
            sign[i, 1] = s_yy * deg
            if sign.shape[1] > 2:
                sign[i, 2] = s_xy * deg
            if sign.shape[1] > 3:
                sign[i, 3] = s_yz * deg
            if sign.shape[1] > 4:
                sign[i, 4] = s_zx * deg

    extra["dmg54"] = dmg_arr
    extra["off54"] = off_arr

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
    """Consistent solid tangent matrix (6, 6) for LAW54."""
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
    """Consistent shell tangent matrix (3, 3) for LAW54."""
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
