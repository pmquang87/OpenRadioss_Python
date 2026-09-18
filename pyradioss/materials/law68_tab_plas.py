"""
LAW68 — Honeycomb Orthotropic Tabulated Plasticity Law (/MAT/LAW68).

Fortran origins:
- engine: ``engine/source/materials/mat/mat068/sigeps68.F`` (solids)
- starter: ``starter/source/materials/mat/mat068/hm_read_mat68.F``
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
    """Evaluate curve f(x) and df/dx."""
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
class Law68Params:
    """Parameters for /MAT/LAW68."""

    rho0: float = 1.0
    refer_rho: float = 1.0

    e11: float = 1000.0
    e22: float = 1000.0
    e33: float = 1000.0
    g12: float = 400.0
    g23: float = 400.0
    g31: float = 400.0

    gflag: int = 0
    vflag: int = 0

    emx11: float = _INF
    emx22: float = _INF
    emx33: float = _INF
    emx12: float = _INF
    emx23: float = _INF
    emx31: float = _INF

    emf11: float = _INF
    emf22: float = _INF
    emf33: float = _INF
    emf12: float = _INF
    emf23: float = _INF
    emf31: float = _INF

    fac: List[float] = field(default_factory=lambda: [1.0] * 18)
    functions: List[Any] = field(default_factory=lambda: [None] * 18)

    # Defaults if curves are absent
    sigy_default: float = 50.0

    title: str = ""

    # Derived
    c_sound: float = field(init=False)

    def __post_init__(self) -> None:
        if self.refer_rho == 0.0:
            self.refer_rho = self.rho0

        if len(self.fac) < 18:
            self.fac = self.fac + [1.0] * (18 - len(self.fac))
        if len(self.functions) < 18:
            self.functions = self.functions + [None] * (18 - len(self.functions))

        max_mod = max(self.e11, self.e22, self.e33, self.g12, self.g23, self.g31)
        self.c_sound = math.sqrt(max(max_mod / max(self.rho0, _EM20), _EM20))

    @property
    def young(self) -> float:
        return max(self.e11, self.e22, self.e33)

    @property
    def shear(self) -> float:
        return max(self.g12, self.g23, self.g31)

    @property
    def bulk(self) -> float:
        return (self.e11 + self.e22 + self.e33) / 3.0

    def eval_limit(self, comp_idx: int, strain_val: float, is_failed: bool = False) -> float:
        """Evaluate yield clamping stress limit for component comp_idx (0..8)."""
        idx = comp_idx if not is_failed else comp_idx + 9
        fn = self.functions[idx]
        scale = self.fac[idx]
        if fn is not None:
            y, _ = _eval_curve(fn, strain_val)
            if y > 0.0:
                return y * scale
        return self.sigy_default * scale


def _extract_param(d: Dict[str, Any], keys: Tuple[str, ...], default: Any = 0.0) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def resolve(mat: Any) -> Law68Params:
    """Resolve Law68Params from Material, dict, or Law68Params."""
    if isinstance(mat, Law68Params):
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

    e11 = float(_extract_param(p, ("e11", "E11", "MAT_EA"), 1000.0))
    e22 = float(_extract_param(p, ("e22", "E22", "MAT_EB"), 1000.0))
    e33 = float(_extract_param(p, ("e33", "E33", "MAT_EC"), 1000.0))
    g12 = float(_extract_param(p, ("g12", "G12", "MAT_GAB"), 400.0))
    g23 = float(_extract_param(p, ("g23", "G23", "MAT_GBC"), 400.0))
    g31 = float(_extract_param(p, ("g31", "G31", "MAT_GCA"), 400.0))

    gflag = int(_extract_param(p, ("gflag", "Gflag", "IF1"), 0))
    vflag = int(_extract_param(p, ("vflag", "Vflag", "IF2"), 0))

    def _inf_strain(k_list: Tuple[str, ...]) -> float:
        v = float(_extract_param(p, k_list, _INF))
        return v if v > 0.0 else _INF

    emx11 = _inf_strain(("emx11", "EMX11", "MAT_EPSR1"))
    emx22 = _inf_strain(("emx22", "EMX22", "MAT_EPSR2"))
    emx33 = _inf_strain(("emx33", "EMX33", "MAT_EPSR3"))
    emx12 = _inf_strain(("emx12", "EMX12", "MAT_EPSR4"))
    emx23 = _inf_strain(("emx23", "EMX23", "MAT_EPSR5"))
    emx31 = _inf_strain(("emx31", "EMX31", "MAT_EPSR6"))

    emf11 = _inf_strain(("emf11", "EMF11", "MAT_EPS11_2"))
    emf22 = _inf_strain(("emf22", "EMF22", "MAT_EPS22_2"))
    emf33 = _inf_strain(("emf33", "EMF33", "MAT_EPS33_2"))
    emf12 = _inf_strain(("emf12", "EMF12", "MAT_EPS12_2"))
    emf23 = _inf_strain(("emf23", "EMF23", "MAT_EPS23_2"))
    emf31 = _inf_strain(("emf31", "EMF31", "MAT_EPS31_2"))

    fac = [1.0] * 18
    for i in range(18):
        f_val = _extract_param(p, (f"fac{i+1}", f"FAC{i+1}", f"FScale{i+1}"), 1.0)
        fac[i] = float(f_val) if f_val is not None and float(f_val) != 0.0 else 1.0

    functions = [None] * 18
    fn_keys = [
        ("fun_a1", "FUN_A1", "I11"),
        ("fun_b1", "FUN_B1", "I22"),
        ("fun_a2", "FUN_A2", "I33"),
        ("fun_a3", "FUN_A3", "I12"),
        ("fun_b3", "FUN_B3", "I23"),
        ("fun_a4", "FUN_A4", "I31"),
        ("fun_b4", "FUN_B4", "I21"),
        ("fun_b5", "FUN_B5", "I32"),
        ("fun_b6", "FUN_B6", "I13"),
        ("mat_yfun11_2", "MAT_YFUN11_2", "J11"),
        ("mat_yfun22_2", "MAT_YFUN22_2", "J22"),
        ("mat_yfun33_2", "MAT_YFUN33_2", "J33"),
        ("mat_yfun12_2", "MAT_YFUN12_2", "J12"),
        ("mat_yfun23_2", "MAT_YFUN23_2", "J23"),
        ("mat_yfun31_2", "MAT_YFUN31_2", "J31"),
        ("mat_yfun21_2", "MAT_YFUN21_2", "J21"),
        ("mat_yfun32_2", "MAT_YFUN32_2", "J32"),
        ("mat_yfun13_2", "MAT_YFUN13_2", "J13"),
    ]
    for i, keys in enumerate(fn_keys):
        fn = _extract_param(p, keys, None)
        if fn is not None and fn != 0:
            functions[i] = fn

    sigy_default = float(_extract_param(p, ("sigy_default", "SIGY", "sigy0"), 50.0))

    return Law68Params(
        rho0=rho0,
        refer_rho=refer_rho,
        e11=e11,
        e22=e22,
        e33=e33,
        g12=g12,
        g23=g23,
        g31=g31,
        gflag=gflag,
        vflag=vflag,
        emx11=emx11,
        emx22=emx22,
        emx33=emx33,
        emx12=emx12,
        emx23=emx23,
        emx31=emx31,
        emf11=emf11,
        emf22=emf22,
        emf33=emf33,
        emf12=emf12,
        emf23=emf23,
        emf31=emf31,
        fac=fac,
        functions=functions,
        sigy_default=sigy_default,
        title=title,
    )


def build_law68(mat: Any) -> Law68Params:
    """Build Law68Params from Material or dictionary."""
    return resolve(mat)


def extra_shapes(mat: Any, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """State variables: uvar (18,) for function interpolation positions & failure flags."""
    if nip > 1:
        return {"uvar68": (nip, 18)}
    return {"uvar68": (18,)}


def needs_defgrad(mat: Any) -> bool:
    return False


def sound_speed(mat: Any, eps: Optional[Any] = None, extra: Optional[Dict[str, Any]] = None) -> float:
    p = resolve(mat)
    return p.c_sound


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Solid update for LAW68 orthotropic tabulated plasticity (sigeps68.F)."""
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
    uvar = extra.get("uvar68", extra.get("uvar"))
    if uvar is None or uvar.shape[0] != nel:
        uvar = np.zeros((nel, 18), dtype=float)
        extra["uvar68"] = uvar

    sign = np.zeros_like(sig_arr)
    c_arr = np.full(nel, p.c_sound, dtype=float)

    for i in range(nel):
        # 1. Uncoupled trial stress update
        s11 = sig_arr[i, 0] + p.e11 * deps_arr[i, 0]
        s22 = sig_arr[i, 1] + p.e22 * deps_arr[i, 1]
        s33 = sig_arr[i, 2] + p.e33 * deps_arr[i, 2]
        s12 = sig_arr[i, 3] + p.g12 * deps_arr[i, 3] if sig_arr.shape[1] > 3 else 0.0
        s23 = sig_arr[i, 4] + p.g23 * deps_arr[i, 4] if sig_arr.shape[1] > 4 else 0.0
        s31 = sig_arr[i, 5] + p.g31 * deps_arr[i, 5] if sig_arr.shape[1] > 5 else 0.0

        # Accumulated strains (stored in uvar or estimated from deps)
        exx = deps_arr[i, 0]
        eyy = deps_arr[i, 1]
        ezz = deps_arr[i, 2]
        exy = deps_arr[i, 3] if deps_arr.shape[1] > 3 else 0.0
        eyz = deps_arr[i, 4] if deps_arr.shape[1] > 4 else 0.0
        ezx = deps_arr[i, 5] if deps_arr.shape[1] > 5 else 0.0

        # Rupture / Failure check
        failed = (
            exx > p.emx11 or eyy > p.emx22 or ezz > p.emx33 or
            abs(exy * 0.5) > p.emx12 or abs(eyz * 0.5) > p.emx23 or abs(ezx * 0.5) > p.emx31
        )
        if failed:
            sign[i] = 0.0
            continue

        # Secondary failure
        secondary = (
            -exx > p.emf11 or -eyy > p.emf22 or -ezz > p.emf33 or
            abs(exy * 0.5) > p.emf12 or abs(eyz * 0.5) > p.emf23 or abs(ezx * 0.5) > p.emf31
        )

        # 2. Clamping against yield curves
        # Arguments for curves: either direct strain or volumetric strain
        dvol = deps_arr[i, 0] + deps_arr[i, 1] + deps_arr[i, 2]
        arg_dir = exx if p.gflag == 1 else (-exx if p.gflag == -1 else dvol)
        arg_shr = exy if p.vflag == 1 else (-exy if p.vflag == -1 else dvol)

        lim11 = p.eval_limit(0, arg_dir, secondary)
        lim22 = p.eval_limit(1, arg_dir, secondary)
        lim33 = p.eval_limit(2, arg_dir, secondary)
        lim12 = p.eval_limit(3, arg_shr, secondary)
        lim23 = p.eval_limit(4, arg_shr, secondary)
        lim31 = p.eval_limit(5, arg_shr, secondary)

        dpla = 0.0
        if abs(s11) > lim11:
            dpla += (abs(s11) - lim11) / p.e11
            s11 = math.copysign(lim11, s11)
        if abs(s22) > lim22:
            dpla += (abs(s22) - lim22) / p.e22
            s22 = math.copysign(lim22, s22)
        if abs(s33) > lim33:
            dpla += (abs(s33) - lim33) / p.e33
            s33 = math.copysign(lim33, s33)
        if abs(s12) > lim12:
            dpla += (abs(s12) - lim12) / p.g12
            s12 = math.copysign(lim12, s12)
        if abs(s23) > lim23:
            dpla += (abs(s23) - lim23) / p.g23
            s23 = math.copysign(lim23, s23)
        if abs(s31) > lim31:
            dpla += (abs(s31) - lim31) / p.g31
            s31 = math.copysign(lim31, s31)

        epsp_arr[i] += dpla

        sign[i, 0] = s11
        sign[i, 1] = s22
        sign[i, 2] = s33
        if sign.shape[1] > 3:
            sign[i, 3] = s12
        if sign.shape[1] > 4:
            sign[i, 4] = s23
        if sign.shape[1] > 5:
            sign[i, 5] = s31

    extra["uvar68"] = uvar

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
    """Shell update for LAW68."""
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
    uvar = extra.get("uvar68", extra.get("uvar"))
    if uvar is None or uvar.shape[0] != nel:
        uvar = np.zeros((nel, 18), dtype=float)
        extra["uvar68"] = uvar

    sign = np.zeros_like(sig_arr)
    c_arr = np.full(nel, p.c_sound, dtype=float)

    for i in range(nel):
        s11 = sig_arr[i, 0] + p.e11 * deps_arr[i, 0]
        s22 = sig_arr[i, 1] + p.e22 * deps_arr[i, 1]
        s12 = sig_arr[i, 2] + p.g12 * deps_arr[i, 2] if sig_arr.shape[1] > 2 else 0.0

        exx = deps_arr[i, 0]
        eyy = deps_arr[i, 1]
        exy = deps_arr[i, 2] if deps_arr.shape[1] > 2 else 0.0

        failed = exx > p.emx11 or eyy > p.emx22 or abs(exy * 0.5) > p.emx12
        if failed:
            sign[i] = 0.0
            continue

        secondary = -exx > p.emf11 or -eyy > p.emf22 or abs(exy * 0.5) > p.emf12

        dvol = deps_arr[i, 0] + deps_arr[i, 1]
        arg_dir = exx if p.gflag == 1 else (-exx if p.gflag == -1 else dvol)
        arg_shr = exy if p.vflag == 1 else (-exy if p.vflag == -1 else dvol)

        lim11 = p.eval_limit(0, arg_dir, secondary)
        lim22 = p.eval_limit(1, arg_dir, secondary)
        lim12 = p.eval_limit(3, arg_shr, secondary)

        dpla = 0.0
        if abs(s11) > lim11:
            dpla += (abs(s11) - lim11) / p.e11
            s11 = math.copysign(lim11, s11)
        if abs(s22) > lim22:
            dpla += (abs(s22) - lim22) / p.e22
            s22 = math.copysign(lim22, s22)
        if abs(s12) > lim12:
            dpla += (abs(s12) - lim12) / p.g12
            s12 = math.copysign(lim12, s12)

        epsp_arr[i] += dpla

        sign[i, 0] = s11
        sign[i, 1] = s22
        if sign.shape[1] > 2:
            sign[i, 2] = s12

    extra["uvar68"] = uvar

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
    """Orthotropic solid tangent (6, 6)."""
    p = resolve(mat)
    c = np.diag([p.e11, p.e22, p.e33, p.g12, p.g23, p.g31]).astype(float)
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
    """Orthotropic shell tangent (3, 3)."""
    p = resolve(mat)
    c = np.diag([p.e11, p.e22, p.g12]).astype(float)
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
