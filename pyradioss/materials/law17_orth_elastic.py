"""
LAW17 — 3D Orthotropic Elastic Material (/MAT/LAW17, /MAT/ORTH_ELAS).

Fortran origins:
- ``engine/source/materials/mat/mat017/m17law.F`` (3D orthotropic elastic material)

Theory
------
LAW17 models 3D linear elastic orthotropic materials with 9 independent elastic constants:
- E1, E2, E3: Young's moduli in principal material directions
- nu12, nu23, nu31: Poisson's ratios (with reciprocal relation nu_ji / Ej = nu_ij / Ei)
- G12, G23, G31: Shear moduli

1. 3D Compliance matrix S:
   [ eps_11 ]   [  1/E1     -nu21/E2   -nu31/E3      0         0         0    ] [ sig_11 ]
   [ eps_22 ]   [ -nu12/E1    1/E2     -nu32/E3      0         0         0    ] [ sig_22 ]
   [ eps_33 ] = [ -nu13/E1   -nu23/E2    1/E3        0         0         0    ] [ sig_33 ]
   [ gam_12 ]   [    0          0          0       1/G12       0         0    ] [ sig_12 ]
   [ gam_23 ]   [    0          0          0         0       1/G23       0    ] [ sig_23 ]
   [ gam_31 ]   [    0          0          0         0         0       1/G31  ] [ sig_31 ]

2. 3D Elastic Stiffness C = S^(-1):
   Delta_sig = C : Delta_eps

3. Plane-stress 2D shells:
   Q_11 = E1 / (1 - nu12 * nu21)
   Q_22 = E2 / (1 - nu12 * nu21)
   Q_12 = nu12 * E2 / (1 - nu12 * nu21)
   Q_33 = G12

Sound speed:
    c = sqrt(max(C11, C22, C33) / max(rho0, 1e-20))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20


@dataclass
class Law17Params:
    """Parameters for /MAT/LAW17 (3D Orthotropic Elastic Material)."""

    id: int = 1
    title: str = ""
    rho0: float = 1.0
    refer_rho: float = 0.0
    e1: float = 210000.0
    e2: float = 210000.0
    e3: float = 210000.0
    nu12: float = 0.3
    nu23: float = 0.3
    nu31: float = 0.3
    g12: float = 0.0
    g23: float = 0.0
    g31: float = 0.0

    def __post_init__(self) -> None:
        if self.refer_rho <= 0.0 and self.rho0 > 0.0:
            self.refer_rho = self.rho0
        if self.e2 <= 0.0:
            self.e2 = self.e1
        if self.e3 <= 0.0:
            self.e3 = self.e1
        if self.g12 <= 0.0:
            self.g12 = self.e1 / (2.0 * (1.0 + self.nu12)) if (1.0 + self.nu12) != 0.0 else 0.5 * self.e1
        if self.g23 <= 0.0:
            self.g23 = self.g12
        if self.g31 <= 0.0:
            self.g31 = self.g12

    @property
    def E(self) -> float:
        return self.e1

    @property
    def nu(self) -> float:
        return self.nu12

    @property
    def G(self) -> float:
        return self.g12

    @property
    def stiffness_matrix(self) -> np.ndarray:
        """Construct (6, 6) orthotropic stiffness tensor C = S^(-1)."""
        s = np.zeros((6, 6), dtype=np.float64)
        e1 = max(_EM20, self.e1)
        e2 = max(_EM20, self.e2)
        e3 = max(_EM20, self.e3)

        s[0, 0] = 1.0 / e1
        s[1, 1] = 1.0 / e2
        s[2, 2] = 1.0 / e3

        s[0, 1] = s[1, 0] = -self.nu12 / e1
        s[1, 2] = s[2, 1] = -self.nu23 / e2
        s[2, 0] = s[0, 2] = -self.nu31 / e3

        s[3, 3] = 1.0 / max(_EM20, self.g12)
        s[4, 4] = 1.0 / max(_EM20, self.g23)
        s[5, 5] = 1.0 / max(_EM20, self.g31)

        try:
            return np.linalg.inv(s)
        except np.linalg.LinAlgError:
            # Fallback to diagonal if ill-conditioned
            c = np.zeros((6, 6), dtype=np.float64)
            c[0, 0] = e1
            c[1, 1] = e2
            c[2, 2] = e3
            c[3, 3] = self.g12
            c[4, 4] = self.g23
            c[5, 5] = self.g31
            return c

    @property
    def shell_stiffness_matrix(self) -> np.ndarray:
        """Plane-stress (3, 3) membrane reduced stiffness matrix Q."""
        e1 = max(_EM20, self.e1)
        e2 = max(_EM20, self.e2)
        nu12 = self.nu12
        nu21 = nu12 * e2 / e1
        denom = 1.0 - nu12 * nu21
        if denom <= 0.0 or abs(denom) < 1.0e-5:
            denom = 1.0e-5

        q11 = e1 / denom
        q22 = e2 / denom
        q12 = nu12 * e2 / denom
        q33 = max(_EM20, self.g12)

        return np.array([
            [q11, q12, 0.0],
            [q12, q22, 0.0],
            [0.0, 0.0, q33],
        ], dtype=np.float64)

    @property
    def sound_speed_val(self) -> float:
        r = self.refer_rho if self.refer_rho > 0.0 else (self.rho0 if self.rho0 > 0.0 else 1.0)
        c_mat = self.stiffness_matrix
        max_diag = max(c_mat[0, 0], c_mat[1, 1], c_mat[2, 2])
        return math.sqrt(max(0.0, max_diag) / max(r, _EM20))

    @property
    def sound_speed_shell_val(self) -> float:
        r = self.refer_rho if self.refer_rho > 0.0 else (self.rho0 if self.rho0 > 0.0 else 1.0)
        q_mat = self.shell_stiffness_matrix
        max_diag = max(q_mat[0, 0], q_mat[1, 1])
        return math.sqrt(max(0.0, max_diag) / max(r, _EM20))


def _get_params(mat: Any, **kwargs: Any) -> Law17Params:
    if isinstance(mat, Law17Params):
        return mat
    if hasattr(mat, "law17_params") and isinstance(mat.law17_params, Law17Params):
        return mat.law17_params

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

    # Direction 1
    e1 = float(p.get("MAT_E1", p.get("e1", p.get("E1", p.get("MAT_E", p.get("young", p.get("E", 210000.0)))))))
    # Direction 2
    e2 = float(p.get("MAT_E2", p.get("e2", p.get("E2", e1))))
    # Direction 3
    e3 = float(p.get("MAT_E3", p.get("e3", p.get("E3", e1))))

    # Poissons
    nu12 = float(p.get("MAT_NU12", p.get("nu12", p.get("NU12", p.get("MAT_NU", p.get("nu", 0.3))))))
    nu23 = float(p.get("MAT_NU23", p.get("nu23", p.get("NU23", nu12))))
    nu31 = float(p.get("MAT_NU31", p.get("nu31", p.get("NU31", nu12))))

    # Shear
    g12 = float(p.get("MAT_G12", p.get("g12", p.get("G12", p.get("G", 0.0)))))
    g23 = float(p.get("MAT_G23", p.get("g23", p.get("G23", g12))))
    g31 = float(p.get("MAT_G31", p.get("g31", p.get("G31", g12))))

    return Law17Params(
        id=int(p.get("id", getattr(mat, "id", 1))),
        title=str(p.get("title", getattr(mat, "title", ""))),
        rho0=rho0,
        refer_rho=refer_rho,
        e1=e1,
        e2=e2,
        e3=e3,
        nu12=nu12,
        nu23=nu23,
        nu31=nu31,
        g12=g12,
        g23=g23,
        g31=g31,
    )


def build_law17(mat: Any = None, **kwargs: Any) -> Law17Params:
    """Build Law17Params from Material, dict, or arguments."""
    return _get_params(mat, **kwargs)


def resolve(mat: Any = None, model: Any = None, log: Any = None, **kwargs: Any) -> Law17Params:
    """Resolve material parameters for LAW17."""
    return _get_params(mat, **kwargs)


def extra_shapes(mat: Any = None, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """Extra history shapes for LAW17 (none needed)."""
    return {}


def needs_defgrad(mat: Any = None) -> bool:
    """LAW17 does not require deformation gradient tensor."""
    return False


def sound_speed(
    mat: Any = None,
    eps: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
    is_shell: bool = False,
    **kwargs: Any,
) -> float:
    """Acoustic sound speed for LAW17."""
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
    """3D orthotropic solid stress update."""
    p = _get_params(mat, **kwargs)
    is_1d = sig.ndim == 1
    sig_arr = np.atleast_2d(sig).copy()
    deps_arr = np.atleast_2d(deps).copy()
    nel = sig_arr.shape[0]

    if epsp is None:
        out_epsp = np.zeros(nel, dtype=float)
    else:
        out_epsp = np.atleast_1d(epsp).astype(float).copy()

    c_mat = p.stiffness_matrix
    dsig = deps_arr @ c_mat.T
    out_sig = sig_arr + dsig

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
    """Plane-stress orthotropic shell update."""
    p = _get_params(mat, **kwargs)
    is_1d = sig.ndim == 1
    sig_arr = np.atleast_2d(sig).copy()
    deps_arr = np.atleast_2d(deps).copy()
    nel = sig_arr.shape[0]

    if epsp is None:
        out_epsp = np.zeros(nel, dtype=float)
    else:
        out_epsp = np.atleast_1d(epsp).astype(float).copy()

    q_mat = p.shell_stiffness_matrix
    out_sig = sig_arr.copy()
    dsig = deps_arr[:, :3] @ q_mat.T
    out_sig[:, :3] += dsig

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
    """Orthotropic 3D stiffness matrix (6x6)."""
    p = _get_params(mat, **kwargs)
    c_mat = p.stiffness_matrix
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
    q_mat = p.shell_stiffness_matrix
    if sig is not None and np.ndim(sig) > 1:
        n = np.shape(sig)[0]
        return np.broadcast_to(q_mat, (n, 3, 3)).copy()
    return q_mat


consistent_shell_tangent = shell_tangent
