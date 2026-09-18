"""
LAW11 — Viscoelastic Navier-Stokes Fluid / ALE (/MAT/LAW11, /MAT/FLUID_VISC).

Fortran origins:
- ``engine/source/materials/mat/mat011/m11law.F`` (stagnation / non-reflecting ALE fluid solver)
- ``engine/source/materials/mat/mat011/m11vs2.F`` (2D Navier-Stokes viscous deviatoric stress)
- ``engine/source/materials/mat/mat011/m11vs3.F`` (3D Navier-Stokes viscous deviatoric stress)
- ``starter/source/materials/mat/mat011/hm_read_mat11.F`` (starter reader)

Theory
------
LAW11 models compressible or slightly compressible viscous fluids under ALE / Eulerian
or Lagrangian formulation with Navier-Stokes viscous deviatoric shear stress:
    sigma_ij = -P * delta_ij + tau'_ij

1. Hydrostatic pressure P:
   - Bulk modulus K = C1 (or rho0 * c0^2 for acoustic medium)
   - Delta P = -K * tr(deps) (compression gives positive P)
   - P = P_old + Delta P

2. Viscous deviatoric stresses tau'_ij (Newton-Stokes):
   - Deviatoric strain rate: epsdot'_ij = (deps_ij - 1/3 * tr(deps) * delta_ij) / dt
   - tau'_ii = 2 * mu_vis * epsdot'_ii
   - tau'_ij = mu_vis * gammadot_ij = 2 * mu_vis * epsdot'_ij (engineering shear)

Sound speed:
    c = sqrt(K / max(rho0, 1e-20))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20


@dataclass
class Law11Params:
    """Parameters for /MAT/LAW11 (Viscoelastic Navier-Stokes Fluid)."""

    id: int = 1
    title: str = ""
    rho0: float = 1.0
    refer_rho: float = 0.0
    c0: float = 0.0
    c1: float = 0.0
    gamma: float = 1.4
    mu_vis: float = 0.0
    p0: float = 0.0
    psh: float = 0.0
    ityp: int = 0
    k_cdi: float = 0.0

    def __post_init__(self) -> None:
        if self.refer_rho <= 0.0 and self.rho0 > 0.0:
            self.refer_rho = self.rho0
        if self.c1 <= 0.0 and self.c0 > 0.0 and self.rho0 > 0.0:
            self.c1 = self.rho0 * (self.c0 ** 2)
        elif self.c1 <= 0.0 and self.c0 <= 0.0:
            self.c1 = 2.1e9  # default bulk modulus (water ~ 2.1 GPa)

    @property
    def bulk(self) -> float:
        return self.c1

    @property
    def K(self) -> float:
        return self.c1

    @property
    def G(self) -> float:
        return self.mu_vis

    @property
    def sound_speed_val(self) -> float:
        r = self.refer_rho if self.refer_rho > 0.0 else (self.rho0 if self.rho0 > 0.0 else 1.0)
        return math.sqrt(max(0.0, self.c1) / max(r, _EM20))


def _get_params(mat: Any, **kwargs: Any) -> Law11Params:
    if isinstance(mat, Law11Params):
        return mat
    if hasattr(mat, "law11_params") and isinstance(mat.law11_params, Law11Params):
        return mat.law11_params

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
    c0 = float(p.get("MAT_C0", p.get("c0", p.get("sound_speed", 0.0))))
    c1 = float(p.get("MAT_C1", p.get("c1", p.get("bulk", p.get("K", 0.0)))))
    gamma = float(p.get("GAMMA", p.get("gamma", 1.4)))
    mu_vis = float(p.get("mu_vis", p.get("vis", p.get("VIS", p.get("K_cdi", p.get("viscosity", 0.0))))))
    p0 = float(p.get("MAT_PScale", p.get("p0", p.get("P0", 0.0))))
    psh = float(p.get("MAT_PSH", p.get("psh", 0.0)))
    ityp = int(p.get("Itype", p.get("ityp", 0)))
    k_cdi = float(p.get("K_cdi", p.get("k_cdi", 0.0)))

    return Law11Params(
        id=int(p.get("id", getattr(mat, "id", 1))),
        title=str(p.get("title", getattr(mat, "title", ""))),
        rho0=rho0,
        refer_rho=refer_rho,
        c0=c0,
        c1=c1,
        gamma=gamma,
        mu_vis=mu_vis,
        p0=p0,
        psh=psh,
        ityp=ityp,
        k_cdi=k_cdi,
    )


def build_law11(mat: Any = None, **kwargs: Any) -> Law11Params:
    """Build Law11Params from Material, dict, or arguments."""
    return _get_params(mat, **kwargs)


def resolve(mat: Any = None, model: Any = None, log: Any = None, **kwargs: Any) -> Law11Params:
    """Resolve material parameters for LAW11."""
    return _get_params(mat, **kwargs)


def extra_shapes(mat: Any = None, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """Extra history shapes for LAW11 (pressure and vorticity tracking)."""
    return {"uvar11": (nip, 2) if nip else (2,)}


def needs_defgrad(mat: Any = None) -> bool:
    """LAW11 does not require full deformation gradient tensor."""
    return False


def sound_speed(
    mat: Any = None,
    eps: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> float:
    """Acoustic sound speed c = sqrt(K / rho0)."""
    p = _get_params(mat, **kwargs)
    return p.sound_speed_val


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
    """3D Navier-Stokes fluid stress update: hydrostatic pressure + viscous shear."""
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
    mu = p.mu_vis

    # Hydrostatic trace and volumetric strain increment
    tr_deps = deps_arr[:, 0] + deps_arr[:, 1] + deps_arr[:, 2]
    # Current pressure from old stress: P_old = -1/3 * (sig_xx + sig_yy + sig_zz)
    p_old = - (sig_arr[:, 0] + sig_arr[:, 1] + sig_arr[:, 2]) / 3.0
    # Update pressure: Delta P = -K * tr(deps)
    p_new = p_old - k * tr_deps

    # Deviatoric strain increments
    e_xx = deps_arr[:, 0] - tr_deps / 3.0
    e_yy = deps_arr[:, 1] - tr_deps / 3.0
    e_zz = deps_arr[:, 2] - tr_deps / 3.0

    # Viscous deviatoric stresses
    if dt > 0.0 and mu > 0.0:
        inv_dt = 1.0 / dt
        s_xx = 2.0 * mu * e_xx * inv_dt
        s_yy = 2.0 * mu * e_yy * inv_dt
        s_zz = 2.0 * mu * e_zz * inv_dt
        s_xy = mu * deps_arr[:, 3] * inv_dt
        s_yz = mu * deps_arr[:, 4] * inv_dt
        s_zx = mu * deps_arr[:, 5] * inv_dt
    else:
        s_xx = np.zeros(nel, dtype=float)
        s_yy = np.zeros(nel, dtype=float)
        s_zz = np.zeros(nel, dtype=float)
        s_xy = np.zeros(nel, dtype=float)
        s_yz = np.zeros(nel, dtype=float)
        s_zx = np.zeros(nel, dtype=float)

    out_sig = np.zeros_like(sig_arr)
    out_sig[:, 0] = -p_new + s_xx
    out_sig[:, 1] = -p_new + s_yy
    out_sig[:, 2] = -p_new + s_zz
    out_sig[:, 3] = s_xy
    out_sig[:, 4] = s_yz
    out_sig[:, 5] = s_zx

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
    """Plane-stress fluid membrane update."""
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
    mu = p.mu_vis

    tr_deps = deps_arr[:, 0] + deps_arr[:, 1]
    p_old = - (sig_arr[:, 0] + sig_arr[:, 1]) / 2.0
    p_new = p_old - k * tr_deps

    if dt > 0.0 and mu > 0.0:
        inv_dt = 1.0 / dt
        e_xx = deps_arr[:, 0] - tr_deps / 2.0
        e_yy = deps_arr[:, 1] - tr_deps / 2.0
        s_xx = 2.0 * mu * e_xx * inv_dt
        s_yy = 2.0 * mu * e_yy * inv_dt
        s_xy = mu * deps_arr[:, 2] * inv_dt
    else:
        s_xx = np.zeros(nel, dtype=float)
        s_yy = np.zeros(nel, dtype=float)
        s_xy = np.zeros(nel, dtype=float)

    out_sig = np.zeros_like(sig_arr)
    out_sig[:, 0] = -p_new + s_xx
    out_sig[:, 1] = -p_new + s_yy
    out_sig[:, 2] = s_xy

    c = p.sound_speed_val

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
    dt: float = 0.0,
    **kwargs: Any,
) -> np.ndarray:
    """Algorithmic tangent stiffness matrix (6x6) for fluid medium."""
    p = _get_params(mat, **kwargs)
    k = p.bulk
    mu_eff = p.mu_vis / max(dt, 1.0e-5) if dt > 0.0 else 0.0

    c_mat = np.zeros((6, 6), dtype=np.float64)
    # Bulk hydrostatic response
    c_mat[0:3, 0:3] = k
    # Deviatoric viscous additions
    if mu_eff > 0.0:
        c_mat[0, 0] += (4.0 / 3.0) * mu_eff
        c_mat[1, 1] += (4.0 / 3.0) * mu_eff
        c_mat[2, 2] += (4.0 / 3.0) * mu_eff
        c_mat[0, 1] -= (2.0 / 3.0) * mu_eff
        c_mat[1, 0] -= (2.0 / 3.0) * mu_eff
        c_mat[0, 2] -= (2.0 / 3.0) * mu_eff
        c_mat[2, 0] -= (2.0 / 3.0) * mu_eff
        c_mat[1, 2] -= (2.0 / 3.0) * mu_eff
        c_mat[2, 1] -= (2.0 / 3.0) * mu_eff
        c_mat[3, 3] = mu_eff
        c_mat[4, 4] = mu_eff
        c_mat[5, 5] = mu_eff

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
    dt: float = 0.0,
    **kwargs: Any,
) -> np.ndarray:
    """Plane-stress membrane tangent matrix (3x3)."""
    p = _get_params(mat, **kwargs)
    k = p.bulk
    mu_eff = p.mu_vis / max(dt, 1.0e-5) if dt > 0.0 else 0.0

    c_mat = np.zeros((3, 3), dtype=np.float64)
    c_mat[0:2, 0:2] = k
    if mu_eff > 0.0:
        c_mat[0, 0] += mu_eff
        c_mat[1, 1] += mu_eff
        c_mat[0, 1] -= mu_eff
        c_mat[1, 0] -= mu_eff
        c_mat[2, 2] = mu_eff

    if sig is not None and np.ndim(sig) > 1:
        n = np.shape(sig)[0]
        return np.broadcast_to(c_mat, (n, 3, 3)).copy()
    return c_mat


consistent_shell_tangent = shell_tangent


def _register() -> None:
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        for k in (11, "11", "LAW11", "FLUID_VISC", "MAT_LAW11"):
            MAT_PHYSICS_REGISTRY[k] = build_law11
    except Exception:
        pass


_register()
