"""
LAW53 — Tabulated Low-Density Crushable Foam Material Model (/MAT/LAW53).

Fortran origins:
- engine: ``engine/source/materials/mat/mat053/sigeps53.F``
- starter: ``starter/source/materials/mat/mat053/hm_read_mat53.F``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pyradioss.model.entities import Material

_EM20 = 1e-20
_INF = 1e30


def _eval_func(func: Any, x: float, scale: float = 1.0) -> float:
    """Evaluate tabulated curve or constant."""
    if func is None:
        return _INF
    if isinstance(func, (int, float)):
        return float(func) * scale
    if callable(func):
        return float(func(x)) * scale
    xs = getattr(func, "x", None)
    ys = getattr(func, "y", None)
    if xs is not None and ys is not None:
        return float(np.interp(x, xs, ys)) * scale
    if hasattr(func, "data"):
        d = np.asarray(func.data, dtype=float)
        if d.ndim == 2 and d.shape[1] >= 2:
            return float(np.interp(x, d[:, 0], d[:, 1])) * scale
    return _INF


@dataclass
class Law53Params:
    """Parameters for /MAT/LAW53 (Tabulated Crushable Foam)."""

    rho0: float = 1.0
    refer_rho: float = 1.0
    e11: float = 100.0
    e22: float = 100.0
    g12: float = 40.0
    g23: float = 40.0

    fun_a1: Any = None
    fun_b1: Any = None
    fun_a3: Any = None
    fun_a5: Any = None
    fun_a6: Any = None

    sfac11: float = 1.0
    sfac22: float = 1.0
    sfac12: float = 1.0
    sfac23: float = 1.0
    sfac45: float = 1.0

    title: str = ""

    # Derived
    sound_speed: float = field(init=False)

    def __post_init__(self) -> None:
        if self.refer_rho == 0.0:
            self.refer_rho = self.rho0
        if self.sfac11 == 0.0:
            self.sfac11 = 1.0
        if self.sfac22 == 0.0:
            self.sfac22 = 1.0
        if self.sfac12 == 0.0:
            self.sfac12 = 1.0
        if self.sfac23 == 0.0:
            self.sfac23 = 1.0
        if self.sfac45 == 0.0:
            self.sfac45 = 1.0

        mod_max = max(self.e11, self.e22, self.g12, self.g23, _EM20)
        self.sound_speed = math.sqrt(mod_max / max(self.rho0, _EM20))

    @property
    def young(self) -> float:
        return max(self.e11, self.e22)

    @property
    def shear(self) -> float:
        return max(self.g12, self.g23)

    @property
    def bulk(self) -> float:
        return self.young / 3.0


def _extract_param(d: Dict[str, Any], keys: Tuple[str, ...], default: Any = 0.0) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def resolve(mat: Any) -> Law53Params:
    """Resolve Law53Params from Material entity, dict, or Law53Params."""
    if isinstance(mat, Law53Params):
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

    e11 = float(_extract_param(p, ("e11", "E11", "MAT_E1", "E_a", "EA", "E"), 100.0))
    e22 = float(_extract_param(p, ("e22", "E22", "MAT_E2", "E_b", "EB"), e11))
    g12 = float(_extract_param(p, ("g12", "G12", "MAT_G12", "MAT_GAB", "G_ab", "GAB", "G"), 40.0))
    g23 = float(_extract_param(p, ("g23", "G23", "MAT_G23", "MAT_GBC", "G_bc", "GBC"), g12))

    fun_a1 = _extract_param(p, ("fun_a1", "FUN_A1", "fct_1"), None)
    fun_b1 = _extract_param(p, ("fun_b1", "FUN_B1", "fct_2"), None)
    fun_a3 = _extract_param(p, ("fun_a3", "FUN_A3", "fct_3"), None)
    fun_a5 = _extract_param(p, ("fun_a5", "FUN_A5", "fct_4"), None)
    fun_a6 = _extract_param(p, ("fun_a6", "FUN_A6", "fct_5"), None)

    sfac11 = float(_extract_param(p, ("sfac11", "MAT_SFAC11", "scale11"), 1.0))
    sfac22 = float(_extract_param(p, ("sfac22", "MAT_SFAC22", "scale22"), 1.0))
    sfac12 = float(_extract_param(p, ("sfac12", "MAT_SFAC12", "scale12"), 1.0))
    sfac23 = float(_extract_param(p, ("sfac23", "MAT_SFAC23", "scale23"), 1.0))
    sfac45 = float(_extract_param(p, ("sfac45", "MAT_SFAC45", "scale45"), 1.0))

    return Law53Params(
        rho0=rho0,
        refer_rho=refer_rho,
        e11=e11,
        e22=e22,
        g12=g12,
        g23=g23,
        fun_a1=fun_a1,
        fun_b1=fun_b1,
        fun_a3=fun_a3,
        fun_a5=fun_a5,
        fun_a6=fun_a6,
        sfac11=sfac11,
        sfac22=sfac22,
        sfac12=sfac12,
        sfac23=sfac23,
        sfac45=sfac45,
        title=title,
    )


def build_law53(mat: Any) -> Law53Params:
    """Build Law53Params from Material or dictionary."""
    return resolve(mat)


def extra_shapes(mat: Any, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """State variables: uvar (6,) [vol_strain, ipos1, ipos2, ipos3, ipos4, ipos5]."""
    if nip > 1:
        return {"uvar53": (nip, 6)}
    return {"uvar53": (6,)}


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
    """Solid constitutive update for LAW53 (sigeps53.F)."""
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
    uvar = extra.get("uvar53", extra.get("uvar"))
    if uvar is None or uvar.shape[0] != nel:
        uvar = np.zeros((nel, 6), dtype=float)
        extra["uvar53"] = uvar

    sign = np.zeros_like(sig_arr)
    c_arr = np.full(nel, p.sound_speed, dtype=float)

    for i in range(nel):
        uvar[i, 0] += deps_arr[i, 0] + deps_arr[i, 1] + deps_arr[i, 2]
        evol = 1.0 - math.exp(uvar[i, 0])

        sxx = sig_arr[i, 0] + p.e11 * deps_arr[i, 0]
        syy = sig_arr[i, 1] + p.e22 * deps_arr[i, 1]
        szz = sig_arr[i, 2] + p.e22 * deps_arr[i, 2]
        sxy = sig_arr[i, 3] + p.g12 * deps_arr[i, 3] if sig_arr.shape[1] > 3 else 0.0
        syz = sig_arr[i, 4] + p.g23 * deps_arr[i, 4] if sig_arr.shape[1] > 4 else 0.0
        szx = sig_arr[i, 5] + p.g12 * deps_arr[i, 5] if sig_arr.shape[1] > 5 else 0.0

        # Tabulated strength limits vs volumetric strain
        s1c = max(_eval_func(p.fun_a1, evol, p.sfac11), 1e-6)
        s1t = max(_eval_func(p.fun_a1, 0.0, p.sfac11), 1e-6)
        s2c = max(_eval_func(p.fun_b1, evol, p.sfac22), 1e-6)
        s2t = max(_eval_func(p.fun_b1, 0.0, p.sfac22), 1e-6)
        s3c = max(_eval_func(p.fun_a3, evol, p.sfac12), 1e-6)
        s3t = max(_eval_func(p.fun_a3, 0.0, p.sfac12), 1e-6)
        s4c = max(_eval_func(p.fun_a5, evol, p.sfac23), 1e-6)
        s4t = max(_eval_func(p.fun_a5, 0.0, p.sfac23), 1e-6)

        # Tsai-Wu coefficients
        f1 = -1.0 / s1c + 1.0 / s1t
        f2 = -1.0 / s2c + 1.0 / s2t
        f11 = 1.0 / (s1c * s1t)
        f22 = 1.0 / (s2c * s2t)
        f44 = 1.0 / (s3c * s3t)
        f55 = 1.0 / (s4c * s4t)
        f12 = -0.5 * math.sqrt(f11 * f22)
        f23 = -0.5 * f22

        if p.fun_a6 is not None:
            s45c = max(_eval_func(p.fun_a6, evol, p.sfac45), 1e-6)
            f12 = 2.0 / (s45c**2) - 0.5 * (f11 + f22 + f44) + (f1 + f2) / s45c

        f_val = (
            f1 * sxx + f2 * syy + f2 * szz
            + f11 * (sxx**2) + f22 * (syy**2) + f22 * (szz**2)
            + f44 * (sxy**2) + f55 * (syz**2) + f44 * (szx**2)
            + 2.0 * f12 * sxx * syy + 2.0 * f23 * syy * szz + 2.0 * f12 * szz * sxx
        )

        scale = 1.0
        if f_val > 1.0:
            bb = f1 * sxx + f2 * syy + f2 * szz
            aa = (
                f11 * (sxx**2) + f22 * (syy**2) + f22 * (szz**2)
                + f44 * (sxy**2) + f55 * (syz**2) + f44 * (szx**2)
                + 2.0 * f12 * sxx * syy + 2.0 * f23 * syy * szz + 2.0 * f12 * szz * sxx
            )
            cc = -1.0
            dd = bb**2 - 4.0 * aa * cc
            if dd >= 0.0 and aa > _EM20:
                ss1 = (-bb + math.sqrt(dd)) / (2.0 * aa)
                ss2 = (-bb - math.sqrt(dd)) / (2.0 * aa)
                if ss1 > 0.0 and ss2 > 0.0:
                    scale = min(ss1, ss2)
                elif ss1 > 0.0:
                    scale = ss1
                elif ss2 > 0.0:
                    scale = ss2
                scale = min(1.0, max(0.0, scale))

        sign[i, 0] = sxx * scale
        sign[i, 1] = syy * scale
        sign[i, 2] = szz * scale
        if sign.shape[1] > 3:
            sign[i, 3] = sxy * scale
        if sign.shape[1] > 4:
            sign[i, 4] = syz * scale
        if sign.shape[1] > 5:
            sign[i, 5] = szx * scale

        dpla = (1.0 - scale) * math.sqrt(sxx**2 + syy**2 + szz**2 + 2.0 * (sxy**2 + syz**2 + szx**2)) / p.young
        epsp_arr[i] += dpla

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
    """Shell plane-stress update for LAW53."""
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
        sxx = sig_arr[i, 0] + p.e11 * deps_arr[i, 0]
        syy = sig_arr[i, 1] + p.e22 * deps_arr[i, 1]
        sxy = sig_arr[i, 2] + p.g12 * deps_arr[i, 2] if sig_arr.shape[1] > 2 else 0.0

        evol = abs(deps_arr[i, 0] + deps_arr[i, 1])
        s1c = max(_eval_func(p.fun_a1, evol, p.sfac11), 1e-6)
        s1t = max(_eval_func(p.fun_a1, 0.0, p.sfac11), 1e-6)
        s2c = max(_eval_func(p.fun_b1, evol, p.sfac22), 1e-6)
        s2t = max(_eval_func(p.fun_b1, 0.0, p.sfac22), 1e-6)
        s3c = max(_eval_func(p.fun_a3, evol, p.sfac12), 1e-6)
        s3t = max(_eval_func(p.fun_a3, 0.0, p.sfac12), 1e-6)

        f1 = -1.0 / s1c + 1.0 / s1t
        f2 = -1.0 / s2c + 1.0 / s2t
        f11 = 1.0 / (s1c * s1t)
        f22 = 1.0 / (s2c * s2t)
        f44 = 1.0 / (s3c * s3t)
        f12 = -0.5 * math.sqrt(f11 * f22)

        f_val = f1 * sxx + f2 * syy + f11 * (sxx**2) + f22 * (syy**2) + f44 * (sxy**2) + 2.0 * f12 * sxx * syy
        scale = 1.0
        if f_val > 1.0:
            bb = f1 * sxx + f2 * syy
            aa = f11 * (sxx**2) + f22 * (syy**2) + f44 * (sxy**2) + 2.0 * f12 * sxx * syy
            cc = -1.0
            dd = bb**2 - 4.0 * aa * cc
            if dd >= 0.0 and aa > _EM20:
                ss1 = (-bb + math.sqrt(dd)) / (2.0 * aa)
                ss2 = (-bb - math.sqrt(dd)) / (2.0 * aa)
                if ss1 > 0.0 and ss2 > 0.0:
                    scale = min(ss1, ss2)
                elif ss1 > 0.0:
                    scale = ss1
                elif ss2 > 0.0:
                    scale = ss2
                scale = min(1.0, max(0.0, scale))

        sign[i, 0] = sxx * scale
        sign[i, 1] = syy * scale
        if sign.shape[1] > 2:
            sign[i, 2] = sxy * scale
        if sign.shape[1] > 3:
            sign[i, 3] = sig_arr[i, 3] + p.g23 * deps_arr[i, 3]
        if sign.shape[1] > 4:
            sign[i, 4] = sig_arr[i, 4] + p.g12 * deps_arr[i, 4]

        dpla = (1.0 - scale) * math.sqrt(sxx**2 + syy**2 + 2.0 * sxy**2) / p.young
        epsp_arr[i] += dpla

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
    """Consistent solid tangent matrix (6, 6) for LAW53."""
    p = resolve(mat)
    c = np.zeros((6, 6), dtype=float)
    c[0, 0] = p.e11
    c[1, 1] = p.e22
    c[2, 2] = p.e22
    c[3, 3] = p.g12
    c[4, 4] = p.g23
    c[5, 5] = p.g12
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
    """Consistent shell tangent matrix (3, 3) for LAW53."""
    p = resolve(mat)
    c = np.array([
        [p.e11, 0.0, 0.0],
        [0.0, p.e22, 0.0],
        [0.0, 0.0, p.g12],
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
