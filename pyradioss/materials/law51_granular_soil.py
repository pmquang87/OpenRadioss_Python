"""
LAW51 (/MAT/LAW51, /MAT/GRANULAR, /MAT/SOIL_PORE) — Multiphase Granular / Soil Model with Pore Pressure & EOS.

OpenRadioss Fortran reference:
- engine/source/materials/mat/mat051/sigeps51.F90
- engine/source/materials/mat/mat051/granular51.F90
- engine/source/materials/mat/mat051/dprag51.F
- engine/source/materials/mat/mat051/jcook51.F90
- starter/source/materials/mat/mat051/hm_read_mat51.F
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20
_EM14 = 1.0e-14
_THIRD = 1.0 / 3.0
_TWO_THIRDS = 2.0 / 3.0
_FOUR_THIRDS = 4.0 / 3.0


@dataclass
class Law51Params:
    """Parameters for /MAT/LAW51 (Granular soil with pore pressure)."""
    E: float = 1.0e8            # Young's modulus
    nu: float = 0.3            # Poisson's ratio
    rho0: float = 2000.0       # Initial mass density
    pfrac: float = 0.0         # Fracture pressure / tensile cutoff (<= 0)
    a0: float = 1.0e6          # Cohesion / intercept parameter a0
    a1: float = 0.5            # Pressure dependency slope a1
    a2: float = 0.0            # Quadratic pressure term a2
    ymax: float = 1.0e9        # Maximum yield stress limit
    k_pore: float = 2.0e9      # Pore fluid bulk modulus
    porosity: float = 0.3      # Initial porosity (0 <= n < 1)
    skempton_b: float = 0.0    # Skempton pore pressure coefficient B (0 to 1)

    # Derived quantities
    G: float = field(init=False, default=0.0)
    K: float = field(init=False, default=0.0)
    bulk: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.nu < 0.0 or self.nu >= 0.5:
            self.nu = 0.3
        if self.E <= 0.0:
            self.E = 1.0e8
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))
        self.bulk = self.K


def resolve(mat: Any) -> Law51Params:
    """Resolve Law51Params from Material, dict, or Law51Params instance."""
    if isinstance(mat, Law51Params):
        return mat
    if isinstance(mat, Material):
        p = mat.params or {}
        return Law51Params(
            E=float(p.get("E", p.get("MAT_E", 1.0e8))),
            nu=float(p.get("nu", p.get("MAT_NU", 0.3))),
            rho0=float(getattr(mat, "rho0", p.get("rho", p.get("RHO", 2000.0)))),
            pfrac=float(p.get("pfrac", p.get("PFRAC", 0.0))),
            a0=float(p.get("a0", p.get("MAT_A0", 1.0e6))),
            a1=float(p.get("a1", p.get("MAT_A1", 0.5))),
            a2=float(p.get("a2", p.get("MAT_A2", 0.0))),
            ymax=float(p.get("ymax", p.get("YMAX", 1.0e9))),
            k_pore=float(p.get("k_pore", p.get("K_PORE", 2.0e9))),
            porosity=float(p.get("porosity", p.get("PORO", 0.3))),
            skempton_b=float(p.get("skempton_b", p.get("SKEMPTON_B", 0.0))),
        )
    if isinstance(mat, dict):
        return Law51Params(
            E=float(mat.get("E", mat.get("MAT_E", 1.0e8))),
            nu=float(mat.get("nu", mat.get("MAT_NU", 0.3))),
            rho0=float(mat.get("rho0", mat.get("rho", mat.get("RHO", 2000.0)))),
            pfrac=float(mat.get("pfrac", mat.get("PFRAC", 0.0))),
            a0=float(mat.get("a0", mat.get("MAT_A0", 1.0e6))),
            a1=float(mat.get("a1", mat.get("MAT_A1", 0.5))),
            a2=float(mat.get("a2", mat.get("MAT_A2", 0.0))),
            ymax=float(mat.get("ymax", mat.get("YMAX", 1.0e9))),
            k_pore=float(mat.get("k_pore", mat.get("K_PORE", 2.0e9))),
            porosity=float(mat.get("porosity", mat.get("PORO", 0.3))),
            skempton_b=float(mat.get("skempton_b", mat.get("SKEMPTON_B", 0.0))),
        )
    return Law51Params()


def build_law51(mat: Any) -> Law51Params:
    """Build Law51Params from Material entity or configuration record."""
    return resolve(mat)


def needs_defgrad(mat: Any = None) -> bool:
    """LAW51 uses small-strain incremental formulation and does not need F."""
    return False


def extra_shapes(mat: Any = None, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """Define persistent state arrays for LAW51 (plastic strain, pore pressure, etc.)."""
    return {
        "uvar": (nip, 20) if nip > 1 else (20,),
        "p_pore": (nip,) if nip > 1 else (1,),
    }


def sound_speed(
    mat: Any,
    eps: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    rho: Optional[float] = None,
    is_shell: bool = False,
    **kwargs: Any,
) -> float:
    """Calculate acoustic sound speed c = sqrt((K + 4/3 G)/rho) for LAW51."""
    p = resolve(mat)
    eff_rho = float(rho) if (rho is not None and rho > 0.0) else p.rho0
    if eff_rho <= 0.0:
        eff_rho = 2000.0
    if is_shell:
        return math.sqrt(max(0.0, p.E / (eff_rho * max(1.0 - p.nu ** 2, 1e-6))))
    return math.sqrt(max(0.0, (p.K + _FOUR_THIRDS * p.G) / eff_rho))


def _solid_step_single(
    p: Law51Params,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp_in: float,
    extra_i: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, float, float]:
    """Single-integration-point solid constitutive update for LAW51."""
    # Volumetric strain increment
    de_vol = deps[0] + deps[1] + deps[2]
    de_dev = np.copy(deps)
    de_dev[0] -= _THIRD * de_vol
    de_dev[1] -= _THIRD * de_vol
    de_dev[2] -= _THIRD * de_vol

    # Old pressure and deviatoric stress
    p_old = -_THIRD * (sig[0] + sig[1] + sig[2])
    s_old = np.copy(sig)
    s_old[0] += p_old
    s_old[1] += p_old
    s_old[2] += p_old

    # Trial pressure and deviators
    p_trial = p_old - p.K * de_vol
    s_trial = np.copy(s_old)
    s_trial[0] += 2.0 * p.G * de_dev[0]
    s_trial[1] += 2.0 * p.G * de_dev[1]
    s_trial[2] += 2.0 * p.G * de_dev[2]
    s_trial[3] += p.G * de_dev[3]
    s_trial[4] += p.G * de_dev[4]
    s_trial[5] += p.G * de_dev[5]

    # Effective pressure considering pore pressure & fracture cutoff
    p_pore = 0.0
    if extra_i is not None and "p_pore" in extra_i:
        p_pore = float(extra_i["p_pore"])
        p_pore += p.skempton_b * (-p.k_pore * de_vol)
        extra_i["p_pore"] = p_pore

    p_eff = p_trial - p_pore
    if p_eff < p.pfrac:
        # Tensile failure cutoff
        sig_y = 0.0
        p_final = max(p_trial, p.pfrac)
    else:
        sig_y = p.a0 + p.a1 * p_eff + p.a2 * (p_eff ** 2)
        sig_y = min(max(0.0, sig_y), p.ymax)
        p_final = p_trial

    # J2 and equivalent von Mises stress
    j2 = 0.5 * (s_trial[0]**2 + s_trial[1]**2 + s_trial[2]**2) + (s_trial[3]**2 + s_trial[4]**2 + s_trial[5]**2)
    sig_vm = math.sqrt(3.0 * max(0.0, j2))

    dpla = 0.0
    s_final = np.copy(s_trial)
    if sig_vm > sig_y:
        ratio = sig_y / max(sig_vm, _EM14)
        s_final = s_trial * ratio
        dpla = (1.0 - ratio) * sig_vm / max(3.0 * p.G, _EM20)

    epsp_out = epsp_in + dpla
    sig_out = np.copy(s_final)
    sig_out[0] -= p_final
    sig_out[1] -= p_final
    sig_out[2] -= p_final

    c = math.sqrt(max(0.0, (p.K + _FOUR_THIRDS * p.G) / p.rho0))
    return sig_out, epsp_out, c


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Vectorized 3D solid update for /MAT/LAW51."""
    p = resolve(mat)
    sig_arr = np.asarray(sig, dtype=np.float64)
    deps_arr = np.asarray(deps, dtype=np.float64)
    is_1d = (sig_arr.ndim == 1)

    s = np.atleast_2d(sig_arr).copy()
    d = np.atleast_2d(deps_arr).copy()
    n = s.shape[0]

    if epsp is None:
        ep = np.zeros(n, dtype=np.float64)
    elif np.isscalar(epsp):
        ep = np.full(n, float(epsp), dtype=np.float64)
    else:
        ep = np.asarray(epsp, dtype=np.float64).copy()

    s_out = np.zeros_like(s)
    ep_out = np.zeros(n, dtype=np.float64)
    c_out = np.zeros(n, dtype=np.float64)

    for i in range(n):
        ex_i = None
        if extra is not None:
            ex_i = {"p_pore": extra.get("p_pore", [0.0])[i] if hasattr(extra.get("p_pore"), "__len__") else extra.get("p_pore", 0.0)}
        si, epi, ci = _solid_step_single(p, s[i], d[i], ep[i], ex_i)
        s_out[i] = si
        ep_out[i] = epi
        c_out[i] = ci

    if is_1d:
        res_sig = s_out[0]
        res_ep = float(ep_out[0])
        res_c = float(c_out[0])
    else:
        res_sig = s_out
        res_ep = ep_out
        res_c = c_out

    if return_sound_speed:
        return res_sig, res_ep, res_c
    return res_sig, res_ep


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Vectorized plane-stress shell update for /MAT/LAW51."""
    p = resolve(mat)
    sig_arr = np.asarray(sig, dtype=np.float64)
    deps_arr = np.asarray(deps, dtype=np.float64)
    is_1d = (sig_arr.ndim == 1)

    s = np.atleast_2d(sig_arr).copy()
    d = np.atleast_2d(deps_arr).copy()
    n = s.shape[0]

    if epsp is None:
        ep = np.zeros(n, dtype=np.float64)
    elif np.isscalar(epsp):
        ep = np.full(n, float(epsp), dtype=np.float64)
    else:
        ep = np.asarray(epsp, dtype=np.float64).copy()

    # Plane-stress elastic moduli
    denom = max(1.0 - p.nu ** 2, 1e-8)
    q11 = p.E / denom
    q12 = p.nu * q11
    q33 = p.G

    s_out = np.zeros_like(s)
    ep_out = np.zeros(n, dtype=np.float64)
    c_shell = math.sqrt(max(0.0, p.E / (p.rho0 * denom)))
    c_out = np.full(n, c_shell, dtype=np.float64)

    for i in range(n):
        s_trial = np.copy(s[i])
        s_trial[0] += q11 * d[i, 0] + q12 * d[i, 1]
        s_trial[1] += q12 * d[i, 0] + q11 * d[i, 1]
        if s_trial.shape[0] > 2 and d.shape[1] > 2:
            s_trial[2] += q33 * d[i, 2]

        p_eff = -_THIRD * (s_trial[0] + s_trial[1])
        sig_y = min(max(0.0, p.a0 + p.a1 * max(p_eff, p.pfrac)), p.ymax)

        sig_vm = math.sqrt(max(0.0, s_trial[0]**2 + s_trial[1]**2 - s_trial[0]*s_trial[1] + 3.0 * (s_trial[2]**2 if s_trial.shape[0] > 2 else 0.0)))
        dpla = 0.0
        if sig_vm > sig_y:
            ratio = sig_y / max(sig_vm, _EM14)
            s_trial[:3] *= ratio
            dpla = (1.0 - ratio) * sig_vm / max(3.0 * p.G, _EM20)

        s_out[i] = s_trial
        ep_out[i] = ep[i] + dpla

    if is_1d:
        res_sig = s_out[0]
        res_ep = float(ep_out[0])
        res_c = float(c_out[0])
    else:
        res_sig = s_out
        res_ep = ep_out
        res_c = c_out

    if return_sound_speed:
        return res_sig, res_ep, res_c
    return res_sig, res_ep


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Return 6x6 initial isotropic elastic tangent stiffness matrix."""
    p = resolve(mat)
    k = p.K
    g = p.G
    c11 = k + _FOUR_THIRDS * g
    c12 = k - _TWO_THIRDS * g

    c = np.zeros((6, 6), dtype=np.float64)
    c[0, 0] = c[1, 1] = c[2, 2] = c11
    c[0, 1] = c[1, 0] = c[0, 2] = c[2, 0] = c[1, 2] = c[2, 1] = c12
    c[3, 3] = c[4, 4] = c[5, 5] = g

    if sig is not None and np.ndim(sig) > 1:
        n = np.shape(sig)[0]
        return np.broadcast_to(c, (n, 6, 6)).copy()
    return c


consistent_solid_tangent = solid_tangent


def shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Return 3x3 plane-stress membrane elastic tangent matrix."""
    p = resolve(mat)
    denom = max(1.0 - p.nu ** 2, 1e-8)
    q11 = p.E / denom
    q12 = p.nu * q11
    q33 = p.G

    c = np.array([
        [q11, q12, 0.0],
        [q12, q11, 0.0],
        [0.0, 0.0, q33],
    ], dtype=np.float64)

    if sig is not None and np.ndim(sig) > 1:
        n = np.shape(sig)[0]
        return np.broadcast_to(c, (n, 3, 3)).copy()
    return c


consistent_shell_tangent = shell_tangent


def _register() -> None:
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        for key in (51, "51", "LAW51", "GRANULAR", "SOIL_PORE", "MAT_LAW51"):
            MAT_PHYSICS_REGISTRY[key] = build_law51
    except Exception:
        pass


_register()
