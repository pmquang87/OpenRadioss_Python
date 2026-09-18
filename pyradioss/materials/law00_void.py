"""
LAW00 — Void / dummy / null material (/MAT/LAW0, /MAT/VOID).

Fortran origins:
- ``starter/source/materials/mat/mat000/hm_read_mat00.F`` (starter reader)

Theory
------
Void material represents empty, non-resisting space or dummy elements.
In structural mechanics, void elements carry zero stress and offer zero stiffness,
while retaining density and elastic sound speed for time step scaling and dummy parts.

Sound speed:
    c = sqrt(E / max(rho0, 1e-20)) (solids)
    c = sqrt((E / (1 - nu^2)) / max(rho0, 1e-20)) (shells)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20


@dataclass
class Law00Params:
    """Parameters for /MAT/LAW0 (Void / Null Material)."""

    id: int = 1
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    young: float = 0.0
    nu: float = 0.0

    def __post_init__(self) -> None:
        if self.refer_rho <= 0.0 and self.rho0 > 0.0:
            self.refer_rho = self.rho0

    @property
    def E(self) -> float:
        return self.young

    @property
    def G(self) -> float:
        return self.young / (2.0 * (1.0 + self.nu)) if (1.0 + self.nu) != 0.0 else 0.0

    @property
    def K(self) -> float:
        denom = 3.0 * (1.0 - 2.0 * self.nu)
        return self.young / denom if denom != 0.0 else 0.0

    @property
    def bulk(self) -> float:
        return self.K


def _get_params(mat: Any, **kwargs: Any) -> Law00Params:
    if isinstance(mat, Law00Params):
        return mat
    if hasattr(mat, "law00_params") and isinstance(mat.law00_params, Law00Params):
        return mat.law00_params

    p: Dict[str, Any] = {}
    if isinstance(mat, Material):
        p = dict(mat.params) if mat.params is not None else {}
        rho = getattr(mat, "rho0", getattr(mat, "rho", 0.0))
        p.setdefault("rho0", rho)
    elif isinstance(mat, dict):
        p = dict(mat.get("params", mat))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        p = dict(mat.params)
    p.update(kwargs)

    rho0 = float(p.get("MAT_RHO", p.get("rho0", p.get("rho", p.get("density", 0.0)))))
    refer_rho = float(p.get("Refer_Rho", p.get("refer_rho", rho0)))
    young = float(p.get("MAT_E", p.get("young", p.get("E", p.get("e", 0.0)))))
    nu = float(p.get("nu", p.get("MAT_NU", p.get("Nu", 0.0))))

    return Law00Params(
        id=int(p.get("id", getattr(mat, "id", 1))),
        title=str(p.get("title", getattr(mat, "title", ""))),
        rho0=rho0,
        refer_rho=refer_rho,
        young=young,
        nu=nu,
    )


def build_law00(mat: Any = None, **kwargs: Any) -> Law00Params:
    """Build Law00Params from Material, dict, or arguments."""
    return _get_params(mat, **kwargs)


def resolve(mat: Any = None, model: Any = None, log: Any = None, **kwargs: Any) -> Law00Params:
    """Resolve material parameters for LAW00."""
    return _get_params(mat, **kwargs)


def extra_shapes(mat: Any = None, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """Extra history shapes for LAW00 (none needed)."""
    return {}


def needs_defgrad(mat: Any = None) -> bool:
    """LAW00 does not require deformation gradient."""
    return False


def sound_speed(
    mat: Any = None,
    eps: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
    is_shell: bool = False,
    **kwargs: Any,
) -> float:
    """Acoustic sound speed for LAW00 void material."""
    p = _get_params(mat, **kwargs)
    r = p.refer_rho if p.refer_rho > 0.0 else (p.rho0 if p.rho0 > 0.0 else 1.0)
    if is_shell:
        denom = 1.0 - p.nu * p.nu
        mod = p.young / denom if denom > 0.0 else p.young
        return math.sqrt(max(0.0, mod) / max(r, _EM20))
    dpdm = p.K + (4.0 / 3.0) * p.G if p.young > 0.0 else 0.0
    if dpdm > 0.0:
        return math.sqrt(dpdm / max(r, _EM20))
    return math.sqrt(max(0.0, p.young) / max(r, _EM20))


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
    """3D solid update for LAW00: void material carries zero stress."""
    is_1d = sig.ndim == 1
    sig_arr = np.atleast_2d(sig)
    nel = sig_arr.shape[0]

    out_sig = np.zeros_like(sig_arr)
    if epsp is None:
        out_epsp = np.zeros(nel, dtype=float)
    else:
        out_epsp = np.atleast_1d(epsp).astype(float).copy()

    c = sound_speed(mat, extra=extra, is_shell=False, **kwargs)

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
    """Plane-stress shell update for LAW00: void material carries zero stress."""
    is_1d = sig.ndim == 1
    sig_arr = np.atleast_2d(sig)
    nel = sig_arr.shape[0]

    out_sig = np.zeros_like(sig_arr)
    if epsp is None:
        out_epsp = np.zeros(nel, dtype=float)
    else:
        out_epsp = np.atleast_1d(epsp).astype(float).copy()

    c = sound_speed(mat, extra=extra, is_shell=True, **kwargs)

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
    """Tangent stiffness matrix for LAW00 (zero resistance)."""
    c_mat = np.zeros((6, 6), dtype=np.float64)
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
    """Plane-stress membrane tangent matrix for LAW00 (zero resistance)."""
    c_mat = np.zeros((3, 3), dtype=np.float64)
    if sig is not None and np.ndim(sig) > 1:
        n = np.shape(sig)[0]
        return np.broadcast_to(c_mat, (n, 3, 3)).copy()
    return c_mat


consistent_shell_tangent = shell_tangent
