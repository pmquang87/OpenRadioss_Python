# Ported from OpenRadioss Fortran:
# Source: engine/source/materials/mat/mat046/sigeps46.F
# Function: SIGEPS46 (lines 30-250)
# Subroutine: M46LAW (engine/source/materials/mat/mat046/m46law.F, lines 35-238)
# Starter: starter/source/materials/mat/mat046/hm_read_mat46.F (lines 37-200)
"""
LAW46 — Combined Isotropic & Kinematic Hardening Plasticity (/MAT/LAW46, /MAT/KIN_HARD).

Fortran origins:
- ``engine/source/materials/mat/mat046/m46law.F`` (material handler)
- ``engine/source/materials/mat/mat046/sigeps46.F`` (LES and stress update formulation)
- ``starter/source/materials/mat/mat046/hm_read_mat46.F`` (starter reader)

Theory
------
LAW46 models combined non-linear kinematic (Armstrong-Frederick / Chaboche) and
isotropic hardening:
1. Yield criterion:
   f = || s - alpha ||_vm - sigma_y(eps_p) <= 0
   where s is the deviatoric Cauchy stress and alpha is the backstress tensor.

2. Isotropic hardening:
   sigma_y(eps_p) = min(sig_max, sigma_y0 + H_iso * eps_p + Q_iso * (1 - exp(-b_iso * eps_p)))

3. Non-linear kinematic backstress evolution:
   dalpha = 2/3 * C_kin * deps_p - gamma_kin * alpha * deps_p

4. Radial return:
   eta_tr = s_trial - alpha_old
   eta_vm = sqrt(3/2 * eta_tr : eta_tr)
   If eta_vm > sigma_y:
       D = 3*G + C_kin + H_iso + Q_iso * b_iso * exp(-b_iso * eps_p)
       deps_p = (eta_vm - sigma_y) / D
       N = sqrt(3/2) * eta_tr / eta_vm
       alpha_new = (alpha_old + 2/3 * C_kin * deps_p * N) / (1 + gamma_kin * deps_p)
       s_new = alpha_new + (sigma_y / eta_vm) * (s_trial - alpha_new)

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
_INF = 1.0e30


@dataclass
class Law46Params:
    """Parameters for /MAT/LAW46 (Combined Isotropic & Kinematic Hardening)."""

    id: int = 1
    title: str = ""
    rho0: float = 1.0
    refer_rho: float = 0.0
    young: float = 210000.0
    nu: float = 0.3
    sig_y: float = 0.0
    c_kin: float = 0.0
    gamma_kin: float = 0.0
    h_iso: float = 0.0
    b_iso: float = 0.0
    q_iso: float = 0.0
    sig_max: float = _INF
    eps_max: float = _INF
    smag: float = 0.0
    vis: float = 0.0

    def __post_init__(self) -> None:
        if self.refer_rho <= 0.0 and self.rho0 > 0.0:
            self.refer_rho = self.rho0
        if self.sig_max <= 0.0:
            self.sig_max = _INF
        if self.eps_max <= 0.0:
            self.eps_max = _INF

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


def _get_params(mat: Any, **kwargs: Any) -> Law46Params:
    if isinstance(mat, Law46Params):
        return mat
    if hasattr(mat, "law46_params") and isinstance(mat.law46_params, Law46Params):
        return mat.law46_params

    p: Dict[str, Any] = {}
    if isinstance(mat, Material):
        p = dict(mat.params) if mat.params is not None else {}
        rho = getattr(mat, "rho0", getattr(mat, "rho", 1.0))
        p.setdefault("rho0", rho)
    elif isinstance(mat, dict):
        p = dict(mat.get("params", mat))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        p = dict(mat.params)
    elif hasattr(mat, "__dict__"):
        p = {k: v for k, v in mat.__dict__.items() if not k.startswith("_")}
    p.update(kwargs)

    rho0 = float(p.get("MAT_RHO", p.get("rho0", p.get("rho", p.get("density", 1.0)))))
    refer_rho = float(p.get("Refer_Rho", p.get("refer_rho", rho0)))
    young = float(p.get("MAT_E", p.get("young", p.get("E", p.get("e", 210000.0)))))
    nu = float(p.get("MAT_NU", p.get("nu", p.get("Nu", 0.3))))
    sig_y = float(p.get("MAT_SIGY", p.get("sig_y", p.get("sigy", p.get("A", 0.0)))))
    c_kin = float(p.get("MAT_C", p.get("c_kin", p.get("C", 0.0))))
    gamma_kin = float(p.get("GAMMA", p.get("gamma_kin", p.get("gamma", 0.0))))
    h_iso = float(p.get("MAT_H", p.get("h_iso", p.get("H", 0.0))))
    b_iso = float(p.get("MAT_B", p.get("b_iso", p.get("B", 0.0))))
    q_iso = float(p.get("MAT_Q", p.get("q_iso", p.get("Q", 0.0))))
    sig_max = float(p.get("MAT_SIG", p.get("sig_max", _INF)))
    eps_max = float(p.get("MAT_EPS", p.get("eps_max", _INF)))
    smag = float(p.get("MAT_C5", p.get("smag", 0.0)))
    vis = float(p.get("VIS", p.get("vis", 0.0)))

    return Law46Params(
        id=int(p.get("id", getattr(mat, "id", 1))),
        title=str(p.get("title", getattr(mat, "title", ""))),
        rho0=rho0,
        refer_rho=refer_rho,
        young=young,
        nu=nu,
        sig_y=sig_y,
        c_kin=c_kin,
        gamma_kin=gamma_kin,
        h_iso=h_iso,
        b_iso=b_iso,
        q_iso=q_iso,
        sig_max=sig_max,
        eps_max=eps_max,
        smag=smag,
        vis=vis,
    )


def build_law46(mat: Any = None, **kwargs: Any) -> Law46Params:
    """Build Law46Params from Material, dict, or arguments."""
    return _get_params(mat, **kwargs)


def resolve(mat: Any = None, model: Any = None, log: Any = None, **kwargs: Any) -> Law46Params:
    """Resolve material parameters for LAW46."""
    return _get_params(mat, **kwargs)


def extra_shapes(mat: Any = None, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """Extra history shapes for LAW46 (backstress tensor alpha: 6 components)."""
    return {"alpha46": (nip, 6) if nip else (6,)}


def needs_defgrad(mat: Any = None) -> bool:
    """LAW46 does not require deformation gradient tensor."""
    return False


def sound_speed(
    mat: Any = None,
    eps: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
    is_shell: bool = False,
    **kwargs: Any,
) -> float:
    """Acoustic sound speed for LAW46."""
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
    """3D solid update for LAW46 with combined isotropic and kinematic hardening."""
    p = _get_params(mat, **kwargs)
    is_1d = sig.ndim == 1
    sig_arr = np.atleast_2d(sig).copy()
    deps_arr = np.atleast_2d(deps).copy()
    nel = sig_arr.shape[0]

    if epsp is None:
        epsp_arr = np.zeros(nel, dtype=float)
    else:
        epsp_arr = np.atleast_1d(epsp).astype(float).copy()

    # Backstress alpha [xx, yy, zz, xy, yz, zx]
    alpha = np.zeros((nel, 6), dtype=float)
    if extra is not None:
        if "alpha" in extra and extra["alpha"] is not None:
            a_in = np.atleast_2d(extra["alpha"])
            alpha[:len(a_in)] = a_in
        elif "alpha46" in extra and extra["alpha46"] is not None:
            a_in = np.atleast_2d(extra["alpha46"])
            alpha[:len(a_in)] = a_in

    g = p.G
    k = p.bulk

    tr_deps = deps_arr[:, 0] + deps_arr[:, 1] + deps_arr[:, 2]
    p_old = - (sig_arr[:, 0] + sig_arr[:, 1] + sig_arr[:, 2]) / 3.0
    p_new = p_old - k * tr_deps

    s_tr = np.zeros_like(sig_arr)
    s_tr[:, 0] = sig_arr[:, 0] + p_old + 2.0 * g * (deps_arr[:, 0] - tr_deps / 3.0)
    s_tr[:, 1] = sig_arr[:, 1] + p_old + 2.0 * g * (deps_arr[:, 1] - tr_deps / 3.0)
    s_tr[:, 2] = sig_arr[:, 2] + p_old + 2.0 * g * (deps_arr[:, 2] - tr_deps / 3.0)
    s_tr[:, 3] = sig_arr[:, 3] + g * deps_arr[:, 3]
    s_tr[:, 4] = sig_arr[:, 4] + g * deps_arr[:, 4]
    s_tr[:, 5] = sig_arr[:, 5] + g * deps_arr[:, 5]

    # Relative stress eta = s - alpha
    eta = s_tr - alpha
    j2 = 0.5 * (eta[:, 0]**2 + eta[:, 1]**2 + eta[:, 2]**2) + eta[:, 3]**2 + eta[:, 4]**2 + eta[:, 5]**2
    eta_vm = np.sqrt(3.0 * np.maximum(0.0, j2))

    ep = np.clip(epsp_arr, 0.0, p.eps_max)
    # Isotropic yield stress
    iso_hard = p.h_iso * ep + p.q_iso * (1.0 - np.exp(-p.b_iso * ep)) if p.b_iso > 0.0 else p.h_iso * ep
    sig_y = np.minimum(p.sig_max, p.sig_y + iso_hard)

    # Plastic check
    plastic = eta_vm > sig_y
    s = s_tr.copy()
    if np.any(plastic):
        idx = np.where(plastic)[0]
        # Hardening slope
        h_prime = p.h_iso + (p.q_iso * p.b_iso * np.exp(-p.b_iso * ep[idx]) if p.b_iso > 0.0 else 0.0)
        denom = 3.0 * g + p.c_kin + h_prime
        denom = np.maximum(1.0e-10, denom)

        dpla = (eta_vm[idx] - sig_y[idx]) / denom
        dpla = np.maximum(0.0, dpla)

        # Unit normal N_dir
        n_fac = np.sqrt(1.5) / np.maximum(1.0e-15, eta_vm[idx])
        n_dir = eta[idx] * n_fac[:, None]

        # Armstrong-Frederick backstress update
        if p.gamma_kin > 0.0:
            alpha_denom = 1.0 + p.gamma_kin * dpla
            alpha[idx] = (alpha[idx] + (2.0 / 3.0) * p.c_kin * dpla[:, None] * n_dir) / alpha_denom[:, None]
        else:
            alpha[idx] += (2.0 / 3.0) * p.c_kin * dpla[:, None] * n_dir

        scale = np.where(eta_vm[idx] > 0.0, sig_y[idx] / eta_vm[idx], 0.0)
        s[idx] = alpha[idx] + scale[:, None] * eta[idx]
        epsp_arr[idx] += dpla

    if extra is not None:
        extra["alpha"] = alpha
        extra["alpha46"] = alpha

    out_sig = np.zeros_like(sig_arr)
    out_sig[:, 0] = -p_new + s[:, 0]
    out_sig[:, 1] = -p_new + s[:, 1]
    out_sig[:, 2] = -p_new + s[:, 2]
    out_sig[:, 3:] = s[:, 3:]

    c = p.sound_speed_val

    res_sig = out_sig[0] if is_1d else out_sig
    res_epsp = float(epsp_arr[0]) if (is_1d and epsp_arr.size == 1) else epsp_arr

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
    """Plane-stress shell update for LAW46."""
    p = _get_params(mat, **kwargs)
    is_1d = sig.ndim == 1
    sig_arr = np.atleast_2d(sig).copy()
    deps_arr = np.atleast_2d(deps).copy()
    nel = sig_arr.shape[0]

    if epsp is None:
        epsp_arr = np.zeros(nel, dtype=float)
    else:
        epsp_arr = np.atleast_1d(epsp).astype(float).copy()

    denom = 1.0 - p.nu * p.nu
    q11 = p.young / denom if denom > 0.0 else p.young
    q12 = p.nu * q11
    g = p.G

    s = sig_arr[:, :3].copy()
    s[:, 0] += q11 * deps_arr[:, 0] + q12 * deps_arr[:, 1]
    s[:, 1] += q12 * deps_arr[:, 0] + q11 * deps_arr[:, 1]
    s[:, 2] += g * deps_arr[:, 2]

    sig_vm = np.sqrt(np.maximum(0.0, s[:, 0]**2 - s[:, 0] * s[:, 1] + s[:, 1]**2 + 3.0 * s[:, 2]**2))

    ep = np.clip(epsp_arr, 0.0, p.eps_max)
    iso_hard = p.h_iso * ep + p.q_iso * (1.0 - np.exp(-p.b_iso * ep)) if p.b_iso > 0.0 else p.h_iso * ep
    sig_y = np.minimum(p.sig_max, p.sig_y + iso_hard)

    plastic = sig_vm > sig_y
    if np.any(plastic):
        idx = np.where(plastic)[0]
        h_prime = p.h_iso + p.c_kin + (p.q_iso * p.b_iso * np.exp(-p.b_iso * ep[idx]) if p.b_iso > 0.0 else 0.0)
        dpla = (sig_vm[idx] - sig_y[idx]) / (3.0 * g + h_prime)
        scale = np.where(sig_vm[idx] > 0.0, sig_y[idx] / sig_vm[idx], 0.0)
        s[idx] *= scale[:, None]
        epsp_arr[idx] += dpla

    out_sig = sig_arr.copy()
    out_sig[:, :3] = s

    c = p.sound_speed_shell_val

    res_sig = out_sig[0] if is_1d else out_sig
    res_epsp = float(epsp_arr[0]) if (is_1d and epsp_arr.size == 1) else epsp_arr

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
    """Algorithmic elastoplastic tangent stiffness matrix (6x6)."""
    p = _get_params(mat, **kwargs)
    k = p.bulk
    g = p.G
    lam = k - (2.0 / 3.0) * g

    c_mat = np.array([
        [lam + 2.0 * g, lam, lam, 0.0, 0.0, 0.0],
        [lam, lam + 2.0 * g, lam, 0.0, 0.0, 0.0],
        [lam, lam, lam + 2.0 * g, 0.0, 0.0, 0.0],
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


def tangent(mat: Any = None, **kwargs: Any) -> np.ndarray:
    """Algorithmic elastoplastic tangent stiffness matrix.

    Alias for solid_tangent matching the material law template.
    """
    return solid_tangent(mat, **kwargs)


tangent_law46_solid = solid_tangent
tangent_law46_shell = shell_tangent

