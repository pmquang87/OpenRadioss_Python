"""
LAW41 — Lee-Tarver JWL Reactive Burn / High Explosive Ignition & Growth (/MAT/LAW41, /MAT/JWL_BURN).

Fortran origins:
- ``engine/source/materials/mat/mat041/sigeps41.F`` (Lee-Tarver ignition & growth solver)
- ``starter/source/materials/mat/mat041/hm_read_mat41.F`` (starter reader)

Theory
------
LAW41 simulates the shock initiation and detonation propagation of solid high explosives
using the Lee-Tarver Ignition and Growth reaction rate model coupled with two JWL equations
of state (for unreacted explosive and reacted gaseous detonation products):

1. Mixture pressure P:
   P = (1 - F) * P_u + F * P_r
   where:
   P_u(eta, E) = A_u*(1 - w_u/(R1_u*eta))*exp(-R1_u*eta) + B_u*(1 - w_u/(R2_u*eta))*exp(-R2_u*eta) + w_u*E/eta
   P_r(eta, E) = A_r*(1 - w_r/(R1_r*eta))*exp(-R1_r*eta) + B_r*(1 - w_r/(R2_r*eta))*exp(-R2_r*eta) + w_r*(E + E0)/eta
   eta = rho / rho0 = 1 / V_rel

2. Reaction rate dF/dt (IREAC = 1: 2-term; IREAC = 2: 3-term):
   - Ignition term:
     rate_i = I * (1 - F)^b * max(0, eta - 1 - ccrit)^x
   - Growth term 1:
     rate_g1 = G1 * (1 - F)^c * F^d * max(0, P)^y
   - Growth term 2:
     rate_g2 = G2 * (1 - F)^e * F^g * max(0, P)^z

3. Hydrodynamic fluid stress:
   sigma_ij = -P * delta_ij

Sound speed:
    c = sqrt(max(dP/drho, 0))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20


@dataclass
class Law41Params:
    """Parameters for /MAT/LAW41 (JWL Reactive Burn / Ignition & Growth)."""

    id: int = 1
    title: str = ""
    rho0: float = 1.8  # Initial explosive density (g/cm^3 or SI)
    refer_rho: float = 0.0

    # Unreacted JWL EOS
    a_u: float = 0.0
    b_u: float = 0.0
    r1_u: float = 4.5
    r2_u: float = 1.5
    omega_u: float = 0.35

    # Reacted JWL EOS
    a_r: float = 500.0e9
    b_r: float = 10.0e9
    r1_r: float = 4.5
    r2_r: float = 1.5
    omega_r: float = 0.35
    e0: float = 5.0e9

    # Reaction kinetics
    ireac: int = 1
    i_ign: float = 0.0
    b_ign: float = 0.667
    x_ign: float = 4.0
    g1: float = 0.0
    c_g1: float = 0.667
    d_g1: float = 0.2
    y_g1: float = 1.0
    g2: float = 0.0
    e_g2: float = 0.667
    g_g2: float = 0.667
    z_g2: float = 2.0
    f_igmax: float = 1.0
    f_g1max: float = 1.0
    f_g2min: float = 0.0
    ccrit: float = 0.0

    def __post_init__(self) -> None:
        if self.refer_rho <= 0.0 and self.rho0 > 0.0:
            self.refer_rho = self.rho0
        if self.r1_u <= 0.0:
            self.r1_u = 4.5
        if self.r2_u <= 0.0:
            self.r2_u = 1.5
        if self.r1_r <= 0.0:
            self.r1_r = 4.5
        if self.r2_r <= 0.0:
            self.r2_r = 1.5

    @property
    def bulk(self) -> float:
        # Approximate bulk modulus from reacted/unreacted EOS
        return max(1.0e9, self.a_r * self.omega_r + self.b_r * self.omega_r)

    @property
    def sound_speed_val(self) -> float:
        r = self.refer_rho if self.refer_rho > 0.0 else (self.rho0 if self.rho0 > 0.0 else 1.0)
        c2 = (self.a_r * self.r1_r + self.b_r * self.r2_r) / max(r, _EM20)
        return math.sqrt(max(1000.0, c2)) if c2 > 0.0 else 3000.0


def _get_params(mat: Any, **kwargs: Any) -> Law41Params:
    if isinstance(mat, Law41Params):
        return mat
    if hasattr(mat, "law41_params") and isinstance(mat.law41_params, Law41Params):
        return mat.law41_params

    p: Dict[str, Any] = {}
    if isinstance(mat, Material):
        p = dict(mat.params) if mat.params is not None else {}
        rho = getattr(mat, "rho0", getattr(mat, "rho", 1.8))
        p.setdefault("rho0", rho)
    elif isinstance(mat, dict):
        p = dict(mat.get("params", mat))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        p = dict(mat.params)
    p.update(kwargs)

    rho0 = float(p.get("MAT_RHO", p.get("rho0", p.get("rho", p.get("density", 1.8)))))
    refer_rho = float(p.get("Refer_Rho", p.get("refer_rho", rho0)))

    return Law41Params(
        id=int(p.get("id", getattr(mat, "id", 1))),
        title=str(p.get("title", getattr(mat, "title", ""))),
        rho0=rho0,
        refer_rho=refer_rho,
        a_u=float(p.get("a_u", p.get("A_u", p.get("AU", 0.0)))),
        b_u=float(p.get("b_u", p.get("B_u", p.get("BU", 0.0)))),
        r1_u=float(p.get("r1_u", p.get("R1_u", 4.5))),
        r2_u=float(p.get("r2_u", p.get("R2_u", 1.5))),
        omega_u=float(p.get("omega_u", p.get("w_u", 0.35))),
        a_r=float(p.get("a_r", p.get("A_r", p.get("A", 500.0e9)))),
        b_r=float(p.get("b_r", p.get("B_r", p.get("B", 10.0e9)))),
        r1_r=float(p.get("r1_r", p.get("R1_r", p.get("R1", 4.5)))),
        r2_r=float(p.get("r2_r", p.get("R2_r", p.get("R2", 1.5)))),
        omega_r=float(p.get("omega_r", p.get("w_r", p.get("omega", 0.35)))),
        e0=float(p.get("e0", p.get("E0", 5.0e9))),
        ireac=int(p.get("ireac", p.get("IREAC", 1))),
        i_ign=float(p.get("i_ign", p.get("I", 0.0))),
        b_ign=float(p.get("b_ign", p.get("b", 0.667))),
        x_ign=float(p.get("x_ign", p.get("x", 4.0))),
        g1=float(p.get("g1", p.get("G1", 0.0))),
        c_g1=float(p.get("c_g1", p.get("c", 0.667))),
        d_g1=float(p.get("d_g1", p.get("d", 0.2))),
        y_g1=float(p.get("y_g1", p.get("y", 1.0))),
        g2=float(p.get("g2", p.get("G2", 0.0))),
        e_g2=float(p.get("e_g2", p.get("e", 0.667))),
        g_g2=float(p.get("g_g2", p.get("g", 0.667))),
        z_g2=float(p.get("z_g2", p.get("z", 2.0))),
        f_igmax=float(p.get("f_igmax", 1.0)),
        f_g1max=float(p.get("f_g1max", 1.0)),
        f_g2min=float(p.get("f_g2min", 0.0)),
        ccrit=float(p.get("ccrit", 0.0)),
    )


def build_law41(mat: Any = None, **kwargs: Any) -> Law41Params:
    """Build Law41Params from Material, dict, or arguments."""
    return _get_params(mat, **kwargs)


def resolve(mat: Any = None, model: Any = None, log: Any = None, **kwargs: Any) -> Law41Params:
    """Resolve material parameters for LAW41."""
    return _get_params(mat, **kwargs)


def extra_shapes(mat: Any = None, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """Extra history shapes for LAW41 (burn fraction F)."""
    return {"burn41": (nip,) if nip else ()}


def needs_defgrad(mat: Any = None) -> bool:
    """LAW41 does not require deformation gradient tensor."""
    return False


def sound_speed(
    mat: Any = None,
    eps: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> float:
    """Acoustic / detonation sound speed for LAW41."""
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
    """Solid update for LAW41 JWL reactive burn model."""
    p = _get_params(mat, **kwargs)
    is_1d = sig.ndim == 1
    sig_arr = np.atleast_2d(sig).copy()
    deps_arr = np.atleast_2d(deps).copy()
    nel = sig_arr.shape[0]

    # History variable: burn fraction F in [0, 1]
    burn = np.zeros(nel, dtype=float)
    if extra is not None:
        if "burn" in extra and extra["burn"] is not None:
            b_in = np.atleast_1d(extra["burn"])
            burn[:len(b_in)] = b_in
        elif "burn41" in extra and extra["burn41"] is not None:
            b_in = np.atleast_1d(extra["burn41"])
            burn[:len(b_in)] = b_in

    # Volumetric trace: negative tr(deps) is compression
    tr_deps = deps_arr[:, 0] + deps_arr[:, 1] + deps_arr[:, 2]
    # Compression eta = rho / rho0 approximately 1 - tr(deps)
    eta = np.maximum(0.5, 1.0 - tr_deps)

    # Evaluate unreacted JWL pressure
    e_unreact = np.exp(-p.r1_u * eta)
    term1_u = p.a_u * (1.0 - p.omega_u / (p.r1_u * eta)) * e_unreact
    term2_u = p.b_u * (1.0 - p.omega_u / (p.r2_u * eta)) * np.exp(-p.r2_u * eta)
    p_u = np.maximum(0.0, term1_u + term2_u)

    # Evaluate reacted JWL pressure
    e_react = np.exp(-p.r1_r * eta)
    term1_r = p.a_r * (1.0 - p.omega_r / (p.r1_r * eta)) * e_react
    term2_r = p.b_r * (1.0 - p.omega_r / (p.r2_r * eta)) * np.exp(-p.r2_r * eta)
    term3_r = (p.omega_r * p.e0) / eta
    p_r = np.maximum(0.0, term1_r + term2_r + term3_r)

    # Rate of burn fraction dF/dt
    p_mix = (1.0 - burn) * p_u + burn * p_r
    p_mix = np.maximum(0.0, p_mix)

    if dt > 0.0:
        # Ignition rate
        comp = np.maximum(0.0, eta - 1.0 - p.ccrit)
        rate_i = p.i_ign * ((1.0 - burn) ** p.b_ign) * (comp ** p.x_ign)
        rate_i = np.where(burn < p.f_igmax, rate_i, 0.0)

        # Growth rate 1
        rate_g1 = p.g1 * ((1.0 - burn) ** p.c_g1) * (burn ** p.d_g1) * (p_mix ** p.y_g1)
        rate_g1 = np.where(burn < p.f_g1max, rate_g1, 0.0)

        # Growth rate 2
        rate_g2 = p.g2 * ((1.0 - burn) ** p.e_g2) * (burn ** p.g_g2) * (p_mix ** p.z_g2)
        rate_g2 = np.where(burn >= p.f_g2min, rate_g2, 0.0)

        df_dt = rate_i + rate_g1 + rate_g2
        burn = np.clip(burn + df_dt * dt, 0.0, 1.0)
        p_mix = (1.0 - burn) * p_u + burn * p_r

    if extra is not None:
        extra["burn"] = burn
        extra["burn41"] = burn

    out_sig = np.zeros_like(sig_arr)
    out_sig[:, 0] = -p_mix
    out_sig[:, 1] = -p_mix
    out_sig[:, 2] = -p_mix
    # Fluid hydrodynamic: shear stresses are zero

    c = p.sound_speed_val

    res_sig = out_sig[0] if is_1d else out_sig
    res_epsp = float(burn[0]) if (is_1d and burn.size == 1) else burn

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
    """Plane-stress membrane update for LAW41."""
    p = _get_params(mat, **kwargs)
    is_1d = sig.ndim == 1
    sig_arr = np.atleast_2d(sig).copy()
    deps_arr = np.atleast_2d(deps).copy()
    nel = sig_arr.shape[0]

    burn = np.zeros(nel, dtype=float)
    if extra is not None:
        if "burn" in extra and extra["burn"] is not None:
            b_in = np.atleast_1d(extra["burn"])
            burn[:len(b_in)] = b_in
        elif "burn41" in extra and extra["burn41"] is not None:
            b_in = np.atleast_1d(extra["burn41"])
            burn[:len(b_in)] = b_in

    tr_deps = deps_arr[:, 0] + deps_arr[:, 1]
    eta = np.maximum(0.5, 1.0 - tr_deps)

    term1_r = p.a_r * (1.0 - p.omega_r / (p.r1_r * eta)) * np.exp(-p.r1_r * eta)
    term2_r = p.b_r * (1.0 - p.omega_r / (p.r2_r * eta)) * np.exp(-p.r2_r * eta)
    p_mix = np.maximum(0.0, term1_r + term2_r)

    out_sig = np.zeros_like(sig_arr)
    out_sig[:, 0] = -p_mix
    out_sig[:, 1] = -p_mix

    c = p.sound_speed_val

    res_sig = out_sig[0] if is_1d else out_sig
    res_epsp = float(burn[0]) if (is_1d and burn.size == 1) else burn

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
    """Hydrodynamic fluid tangent matrix (6x6)."""
    p = _get_params(mat, **kwargs)
    k = p.bulk
    c_mat = np.zeros((6, 6), dtype=np.float64)
    c_mat[0:3, 0:3] = k

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
    k = p.bulk
    c_mat = np.zeros((3, 3), dtype=np.float64)
    c_mat[0:2, 0:2] = k

    if sig is not None and np.ndim(sig) > 1:
        n = np.shape(sig)[0]
        return np.broadcast_to(c_mat, (n, 3, 3)).copy()
    return c_mat


consistent_shell_tangent = shell_tangent


def _register() -> None:
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        for k in (41, "41", "LAW41", "JWL_BURN", "MAT_LAW41"):
            MAT_PHYSICS_REGISTRY[k] = build_law41
    except Exception:
        pass


_register()
