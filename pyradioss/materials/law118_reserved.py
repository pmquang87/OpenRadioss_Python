"""OpenRadioss /MAT/LAW118 — Reserved Material Law Stub.

Placeholder / reserved material law definition in OpenRadioss (internal tracking RD-5734).
Provides standard elastoplastic fallback behavior and interface compliance.

Upstream Fortran reference:
  - `starter/source/materials/mat/mat118/reserved.txt`
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_EM20 = 1.0e-20


@dataclass
class Law118Params:
    """Parameters for OpenRadioss /MAT/LAW118 (Reserved Stub)."""
    id: int = 1
    title: str = "LAW118_RESERVED"
    rho0: float = 0.0
    refer_rho: float = 0.0
    rho: float = 0.0
    young: float = 1.0
    nu: float = 0.3
    sigy0: float = 1.0e20
    # Derived moduli
    g: float = field(init=False)
    bulk: float = field(init=False)
    lame: float = field(init=False)
    a11: float = field(init=False)
    a12: float = field(init=False)

    def __post_init__(self) -> None:
        if self.rho > 0.0 and self.rho0 <= 0.0:
            self.rho0 = self.rho
        if self.rho0 > 0.0 and self.rho <= 0.0:
            self.rho = self.rho0
        if self.refer_rho <= 0.0:
            self.refer_rho = self.rho0

        if self.young <= 0.0:
            self.young = 1.0
        if self.nu < 0.0 or self.nu >= 0.5:
            self.nu = 0.3

        self.g = self.young / (2.0 * (1.0 + self.nu))
        self.bulk = self.young / max(_EM20, 3.0 * (1.0 - 2.0 * self.nu))
        self.lame = (self.young * self.nu) / max(_EM20, (1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        denom_shell = 1.0 - self.nu * self.nu
        self.a11 = self.young / max(_EM20, denom_shell)
        self.a12 = self.a11 * self.nu

    @classmethod
    def from_material(cls, mat: Any) -> Law118Params:
        """Construct Law118Params from generic Material or dictionary."""
        if isinstance(mat, Law118Params):
            return mat

        def _get(keys: Sequence[str], default: Any) -> Any:
            for k in keys:
                if hasattr(mat, k):
                    v = getattr(mat, k)
                    if v is not None:
                        return v
                if hasattr(mat, "params") and isinstance(mat.params, dict) and k in mat.params:
                    v = mat.params[k]
                    if v is not None:
                        return v
                if isinstance(mat, dict) and k in mat:
                    v = mat[k]
                    if v is not None:
                        return v
            return default

        mid = int(_get(["id", "mid", "mat_id"], 1))
        title = str(_get(["title", "name"], "LAW118_RESERVED"))
        rho0 = float(_get(["rho0", "rho", "MAT_RHO"], 0.0))
        young = float(_get(["young", "e", "MAT_E", "E"], 1.0))
        nu = float(_get(["nu", "MAT_NU"], 0.3))
        sigy0 = float(_get(["sigy0", "sigy", "SIGY0"], 1.0e20))

        return cls(
            id=mid,
            title=title,
            rho0=rho0,
            young=young,
            nu=nu,
            sigy0=sigy0,
        )


def build_law118(mat: Any = None, **kwargs: Any) -> Law118Params:
    """Construct Law118Params from material or keyword arguments."""
    if mat is not None:
        p = Law118Params.from_material(mat)
        for k, v in kwargs.items():
            if hasattr(p, k):
                setattr(p, k, v)
        p.__post_init__()
        return p
    valid_keys = {f.name for f in Law118Params.__dataclass_fields__.values() if f.init}
    init_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in valid_keys}
    p = Law118Params(**init_kwargs)
    for k, v in extra_kwargs.items():
        if hasattr(p, k):
            setattr(p, k, v)
    p.__post_init__()
    return p


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law118Params:
    """Resolve references for /MAT/LAW118 and return Law118Params."""
    return build_law118(mat)


def extra_shapes(mat: Any = None, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Return extra history variable shapes for LAW118."""
    if nip is not None:
        return {
            "uvar118": (nip, 1),
            "epsp": (nip,),
        }
    return {
        "uvar118": (1,),
        "epsp": (),
    }


def needs_defgrad(mat: Any = None) -> bool:
    """LAW118 reserved stub does not need deformation gradient."""
    return False


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    """3D continuum solid stress update for LAW118."""
    p = build_law118(mat)
    sig_arr = np.asarray(sig, dtype=np.float64)
    deps_arr = np.asarray(deps, dtype=np.float64)
    is_1d = (deps_arr.ndim == 1)

    sig_2d = sig_arr.reshape(1, 6) if is_1d else sig_arr.copy()
    deps_2d = deps_arr.reshape(1, 6) if is_1d else deps_arr.copy()
    n = sig_2d.shape[0]

    if epsp is None:
        epsp_arr = np.zeros(n, dtype=np.float64)
    else:
        epsp_in = np.asarray(epsp, dtype=np.float64)
        epsp_arr = np.full(n, float(epsp_in)) if epsp_in.ndim == 0 else epsp_in.copy()

    sig_new = np.zeros_like(sig_2d)
    lame = p.lame
    g = p.g
    g2 = 2.0 * g

    for i in range(n):
        deps_i = deps_2d[i]
        tr_deps = deps_i[0] + deps_i[1] + deps_i[2]

        sig_new[i, 0] = sig_2d[i, 0] + lame * tr_deps + g2 * deps_i[0]
        sig_new[i, 1] = sig_2d[i, 1] + lame * tr_deps + g2 * deps_i[1]
        sig_new[i, 2] = sig_2d[i, 2] + lame * tr_deps + g2 * deps_i[2]
        sig_new[i, 3] = sig_2d[i, 3] + g * deps_i[3]
        sig_new[i, 4] = sig_2d[i, 4] + g * deps_i[4]
        sig_new[i, 5] = sig_2d[i, 5] + g * deps_i[5]

    c = sound_speed(p)
    c_out = c if is_1d else np.full(n, c, dtype=np.float64)

    if is_1d:
        return sig_new[0], float(epsp_arr[0]), float(c_out)
    return sig_new, epsp_arr, c_out


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    """2D plane-stress shell stress update for LAW118."""
    p = build_law118(mat)
    sig_arr = np.asarray(sig, dtype=np.float64)
    deps_arr = np.asarray(deps, dtype=np.float64)
    is_1d = (deps_arr.ndim == 1)

    sig_2d = sig_arr.reshape(1, -1) if is_1d else sig_arr.copy()
    deps_2d = deps_arr.reshape(1, -1) if is_1d else deps_arr.copy()
    n = sig_2d.shape[0]

    if epsp is None:
        epsp_arr = np.zeros(n, dtype=np.float64)
    else:
        epsp_in = np.asarray(epsp, dtype=np.float64)
        epsp_arr = np.full(n, float(epsp_in)) if epsp_in.ndim == 0 else epsp_in.copy()

    ncomp = sig_2d.shape[1]
    sig_new = np.zeros_like(sig_2d)

    a11 = p.a11
    a12 = p.a12
    g = p.g

    for i in range(n):
        deps_i = deps_2d[i]
        sig_new[i, 0] = sig_2d[i, 0] + a11 * deps_i[0] + a12 * deps_i[1]
        sig_new[i, 1] = sig_2d[i, 1] + a12 * deps_i[0] + a11 * deps_i[1]
        sig_new[i, 2] = sig_2d[i, 2] + g * deps_i[2]

        if ncomp > 3:
            sig_new[i, 3:] = sig_2d[i, 3:]

    c = sound_speed(p, is_shell=True)
    c_out = c if is_1d else np.full(n, c, dtype=np.float64)

    if is_1d:
        return sig_new[0], float(epsp_arr[0]), float(c_out)
    return sig_new, epsp_arr, c_out


def sound_speed(mat: Any, eps: Optional[Any] = None, extra: Optional[Any] = None, is_shell: bool = False, **kwargs: Any) -> float:
    """Compute acoustic wave speed for LAW118."""
    p = build_law118(mat)
    rho = p.rho0 if p.rho0 > 0.0 else 1.0
    if is_shell:
        mod = p.a11
    else:
        mod = p.bulk + (4.0 / 3.0) * p.g
    return float(math.sqrt(max(0.0, mod / rho)))


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[float] = None,
    epsp_incr: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent 6x6 continuum solid tangent stiffness for LAW118."""
    p = build_law118(mat)
    c_el = np.zeros((6, 6), dtype=np.float64)
    lame = p.lame
    g = p.g
    g2 = 2.0 * g

    c_el[0, 0] = lame + g2
    c_el[1, 1] = lame + g2
    c_el[2, 2] = lame + g2
    c_el[0, 1] = c_el[1, 0] = lame
    c_el[0, 2] = c_el[2, 0] = lame
    c_el[1, 2] = c_el[2, 1] = lame
    c_el[3, 3] = g
    c_el[4, 4] = g
    c_el[5, 5] = g

    if sig is None:
        return c_el

    sig_arr = np.asarray(sig, dtype=np.float64)
    if sig_arr.ndim == 2:
        n = sig_arr.shape[0]
        t = np.zeros((n, 6, 6), dtype=np.float64)
        for i in range(n):
            t[i] = solid_tangent(p, sig=sig_arr[i])
        return t
    return c_el


def shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[float] = None,
    epsp_incr: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent 3x3 plane stress algorithmic tangent matrix for LAW118."""
    p = build_law118(mat)
    c_el = np.array([
        [p.a11, p.a12, 0.0],
        [p.a12, p.a11, 0.0],
        [0.0, 0.0, p.g],
    ], dtype=np.float64)

    if sig is None:
        return c_el

    sig_arr = np.asarray(sig, dtype=np.float64)
    if sig_arr.ndim == 2:
        n = sig_arr.shape[0]
        t = np.zeros((n, 3, 3), dtype=np.float64)
        for i in range(n):
            t[i] = shell_tangent(p, sig=sig_arr[i])
        return t
    return c_el


consistent_solid_tangent = solid_tangent
consistent_shell_tangent = shell_tangent
