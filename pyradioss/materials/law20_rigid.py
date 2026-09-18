"""
LAW20 — Rigid Material Handler (/MAT/LAW20, /MAT/RIGID).

Fortran origins:
- ``starter/source/materials/mat/mat020/hm_read_mat20.F`` (starter reader)

Theory
------
In OpenRadioss, LAW20 defines rigid bodies or rigid material domains. Elements assigned
a rigid material do not undergo plastic deformation. In kinematic solvers, their degrees
of freedom are tied to the center of mass of the rigid body. In continuum constitutive updates,
they provide elastic baseline stiffness and sound speed without plastic dissipation.

Sound speed:
    c = sqrt((K + 4/3 * G) / max(rho0, 1e-20))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20


@dataclass
class Law20Params:
    """Parameters for /MAT/LAW20 (Rigid Material Handler)."""

    id: int = 1
    title: str = ""
    rho0: float = 1.0
    refer_rho: float = 0.0
    young: float = 210000.0
    nu: float = 0.3
    mat1: int = 0
    mat2: int = 0
    alpha1: float = 1.0
    alpha2: float = 0.0

    def __post_init__(self) -> None:
        if self.refer_rho <= 0.0 and self.rho0 > 0.0:
            self.refer_rho = self.rho0
        if self.nu >= 0.5:
            self.nu = 0.499

    @property
    def E(self) -> float:
        return self.young

    @property
    def G(self) -> float:
        return self.young / (2.0 * (1.0 + self.nu))

    @property
    def K(self) -> float:
        denom = 3.0 * (1.0 - 2.0 * self.nu)
        return self.young / denom if denom > 0.0 else self.young

    @property
    def bulk(self) -> float:
        return self.K

    @property
    def sound_speed_val(self) -> float:
        r = self.refer_rho if self.refer_rho > 0.0 else (self.rho0 if self.rho0 > 0.0 else 1.0)
        dpdm = self.K + (4.0 / 3.0) * self.G
        return math.sqrt(max(0.0, dpdm) / max(r, _EM20))

    @property
    def sound_speed_shell_val(self) -> float:
        r = self.refer_rho if self.refer_rho > 0.0 else (self.rho0 if self.rho0 > 0.0 else 1.0)
        denom = 1.0 - self.nu * self.nu
        mod = self.young / denom if denom > 0.0 else self.young
        return math.sqrt(max(0.0, mod) / max(r, _EM20))


def _get_params(mat: Any, **kwargs: Any) -> Law20Params:
    if isinstance(mat, Law20Params):
        return mat
    if hasattr(mat, "law20_params") and isinstance(mat.law20_params, Law20Params):
        return mat.law20_params

    p: Dict[str, Any] = {}
    if isinstance(mat, Material):
        p = dict(mat.params) if mat.params is not None else {}
        rho = getattr(mat, "rho0", getattr(mat, "rho", 1.0))
        p.setdefault("rho0", rho)
    elif isinstance(mat, dict):
        p = dict(mat.get("params", mat))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        p = dict(mat.params)
    p.update(kwargs)

    rho0 = float(p.get("MAT_RHO", p.get("rho0", p.get("rho", p.get("density", 1.0)))))
    refer_rho = float(p.get("Refer_Rho", p.get("refer_rho", rho0)))
    young = float(p.get("MAT_E", p.get("young", p.get("E", p.get("e", 210000.0)))))
    nu = float(p.get("MAT_NU", p.get("nu", p.get("Nu", 0.3))))
    mat1 = int(p.get("MAT1", p.get("mat1", 0)))
    mat2 = int(p.get("MAT2", p.get("mat2", 0)))
    alpha1 = float(p.get("MAT_ALPHA1", p.get("alpha1", 1.0)))
    alpha2 = float(p.get("MAT_ALPHA2", p.get("alpha2", 0.0)))

    return Law20Params(
        id=int(p.get("id", getattr(mat, "id", 1))),
        title=str(p.get("title", getattr(mat, "title", ""))),
        rho0=rho0,
        refer_rho=refer_rho,
        young=young,
        nu=nu,
        mat1=mat1,
        mat2=mat2,
        alpha1=alpha1,
        alpha2=alpha2,
    )


def build_law20(mat: Any = None, **kwargs: Any) -> Law20Params:
    """Build Law20Params from Material, dict, or arguments."""
    return _get_params(mat, **kwargs)


def resolve(mat: Any = None, model: Any = None, log: Any = None, **kwargs: Any) -> Law20Params:
    """Resolve material parameters for LAW20."""
    return _get_params(mat, **kwargs)


def extra_shapes(mat: Any = None, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """Extra history shapes for LAW20 (none needed)."""
    return {}


def needs_defgrad(mat: Any = None) -> bool:
    """LAW20 does not require deformation gradient tensor."""
    return False


def sound_speed(
    mat: Any = None,
    eps: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
    is_shell: bool = False,
    **kwargs: Any,
) -> float:
    """Acoustic sound speed for LAW20."""
    p = _get_params(mat, **kwargs)
    return p.sound_speed_shell_val if is_shell else p.sound_speed_val


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
    **kwargs: Any,
) -> Union[Tuple[np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray]]:
    """Solid update for LAW20: rigid material with zero plastic deformation."""
    p = _get_params(mat, **kwargs)
    is_1d = sig.ndim == 1
    sig_arr = np.atleast_2d(sig).copy()
    deps_arr = np.atleast_2d(deps).copy()
    nel = sig_arr.shape[0]

    if epsp is None:
        out_epsp = np.zeros(nel, dtype=float)
    else:
        out_epsp = np.atleast_1d(epsp).astype(float).copy()

    k = p.bulk
    g = p.G
    lam = k - (2.0 / 3.0) * g

    tr_deps = deps_arr[:, 0] + deps_arr[:, 1] + deps_arr[:, 2]

    out_sig = sig_arr.copy()
    out_sig[:, 0] += lam * tr_deps + 2.0 * g * deps_arr[:, 0]
    out_sig[:, 1] += lam * tr_deps + 2.0 * g * deps_arr[:, 1]
    out_sig[:, 2] += lam * tr_deps + 2.0 * g * deps_arr[:, 2]
    out_sig[:, 3] += g * deps_arr[:, 3]
    out_sig[:, 4] += g * deps_arr[:, 4]
    out_sig[:, 5] += g * deps_arr[:, 5]

    c = p.sound_speed_val

    res_sig = out_sig[0] if is_1d else out_sig
    res_epsp = float(out_epsp[0]) if (is_1d and out_epsp.size == 1) else out_epsp

    if return_sound_speed:
        return res_sig, res_epsp, c
    return res_sig, res_epsp


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
    **kwargs: Any,
) -> Union[Tuple[np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray]]:
    """Plane-stress shell update for LAW20."""
    p = _get_params(mat, **kwargs)
    is_1d = sig.ndim == 1
    sig_arr = np.atleast_2d(sig).copy()
    deps_arr = np.atleast_2d(deps).copy()
    nel = sig_arr.shape[0]

    if epsp is None:
        out_epsp = np.zeros(nel, dtype=float)
    else:
        out_epsp = np.atleast_1d(epsp).astype(float).copy()

    denom = 1.0 - p.nu * p.nu
    q11 = p.young / denom if denom > 0.0 else p.young
    q12 = p.nu * q11
    q33 = p.G

    out_sig = sig_arr.copy()
    out_sig[:, 0] += q11 * deps_arr[:, 0] + q12 * deps_arr[:, 1]
    out_sig[:, 1] += q12 * deps_arr[:, 0] + q11 * deps_arr[:, 1]
    out_sig[:, 2] += q33 * deps_arr[:, 2]

    c = p.sound_speed_shell_val

    res_sig = out_sig[0] if is_1d else out_sig
    res_epsp = float(out_epsp[0]) if (is_1d and out_epsp.size == 1) else out_epsp

    if return_sound_speed:
        return res_sig, res_epsp, c
    return res_sig, res_epsp


def solid_tangent(
    mat: Any = None,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """3D solid elastic stiffness tensor (6x6)."""
    p = _get_params(mat, **kwargs)
    k = p.bulk
    g = p.G
    lam = k - (2.0 / 3.0) * g

    c11 = lam + 2.0 * g
    c12 = lam

    c_mat = np.array([
        [c11, c12, c12, 0.0, 0.0, 0.0],
        [c12, c11, c12, 0.0, 0.0, 0.0],
        [c12, c12, c11, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, g,   0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, g,   0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, g  ],
    ], dtype=np.float64)

    if sig is not None and np.ndim(sig) > 1:
        n = np.shape(sig)[0]
        return np.broadcast_to(c_mat, (n, 6, 6)).copy()
    return c_mat


consistent_solid_tangent = solid_tangent


def shell_tangent(
    mat: Any = None,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Plane-stress membrane tangent matrix (3x3)."""
    p = _get_params(mat, **kwargs)
    denom = 1.0 - p.nu * p.nu
    q11 = p.young / denom if denom > 0.0 else p.young
    q12 = p.nu * q11
    q33 = p.G

    c_mat = np.array([
        [q11, q12, 0.0],
        [q12, q11, 0.0],
        [0.0, 0.0, q33],
    ], dtype=np.float64)

    if sig is not None and np.ndim(sig) > 1:
        n = np.shape(sig)[0]
        return np.broadcast_to(c_mat, (n, 3, 3)).copy()
    return c_mat


consistent_shell_tangent = shell_tangent


def _register() -> None:
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        for k in (20, "20", "LAW20", "RIGID", "MAT_LAW20"):
            MAT_PHYSICS_REGISTRY[k] = build_law20
    except Exception:
        pass


_register()
