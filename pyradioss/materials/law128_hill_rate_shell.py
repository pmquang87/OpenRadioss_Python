"""
LAW128 (/MAT/LAW128, /MAT/HILL_RATE) — Orthotropic Hill Plasticity with Strain-Rate Dependency for Shells and Solids.

OpenRadioss Fortran reference:
- engine/source/materials/mat/mat128/sigeps128c.F90
- engine/source/materials/mat/mat128/sigeps128s.F90
- starter/source/materials/mat/mat128/
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
class Law128Params:
    """Parameters for /MAT/LAW128 (Hill orthotropic plasticity with strain rate dependency)."""
    E: float = 2.1e11           # Young's modulus
    nu: float = 0.3            # Poisson's ratio
    rho0: float = 7850.0       # Initial mass density
    sig0: float = 2.5e8        # Initial yield stress
    # Hill 1948 yield coefficients (normalized so F=G=H=0.5, L=M=N=1.5 for von Mises)
    F: float = 0.5
    G_hill: float = 0.5
    H: float = 0.5
    L: float = 1.5
    M: float = 1.5
    N: float = 1.5
    # Optional Lankford r-values
    r00: Optional[float] = None
    r45: Optional[float] = None
    r90: Optional[float] = None
    # Hardening parameters (Swift A*(eps0 + epsp)**n + linear hardening)
    A_swift: float = 0.0
    eps0_swift: float = 0.001
    n_swift: float = 0.2
    hp: float = 0.0            # Linear hardening modulus
    # Cowper-Symonds strain rate parameters: (1 + (eps_dot / C)**(1/p))
    c_rate: float = 0.0        # Cowper-Symonds C (> 0 for rate effects)
    p_rate: float = 1.0        # Cowper-Symonds exponent p

    # Derived elastic moduli
    G: float = field(init=False, default=0.0)
    K: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.nu < 0.0 or self.nu >= 0.5:
            self.nu = 0.3
        if self.E <= 0.0:
            self.E = 2.1e11
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))

        # Convert Lankford r-values to Hill parameters if supplied
        if self.r00 is not None and self.r45 is not None and self.r90 is not None:
            r0 = max(0.01, self.r00)
            r45 = max(0.01, self.r45)
            r90 = max(0.01, self.r90)
            denom = 1.0 + r0
            self.H = r0 / denom
            self.G_hill = 1.0 / denom
            self.F = r0 / (r90 * denom)
            self.N = (r0 + r90) * (1.0 + 2.0 * r45) / (2.0 * r90 * denom)
            self.L = 1.5
            self.M = 1.5


def resolve(mat: Any) -> Law128Params:
    """Resolve Law128Params from Material, dict, or existing Law128Params."""
    if isinstance(mat, Law128Params):
        return mat
    if isinstance(mat, Material):
        p = mat.params or {}
        return Law128Params(
            E=float(p.get("E", p.get("MAT_E", 2.1e11))),
            nu=float(p.get("nu", p.get("MAT_NU", 0.3))),
            rho0=float(getattr(mat, "rho0", p.get("rho", p.get("MAT_RHO", 7850.0)))),
            sig0=float(p.get("sig0", p.get("SIG0", p.get("MAT_SIGY", 2.5e8)))),
            F=float(p.get("F", p.get("HILL_F", 0.5))),
            G_hill=float(p.get("G_hill", p.get("HILL_G", 0.5))),
            H=float(p.get("H", p.get("HILL_H", 0.5))),
            L=float(p.get("L", p.get("HILL_L", 1.5))),
            M=float(p.get("M", p.get("HILL_M", 1.5))),
            N=float(p.get("N", p.get("HILL_N", 1.5))),
            r00=float(p["r00"]) if "r00" in p else (float(p["R00"]) if "R00" in p else None),
            r45=float(p["r45"]) if "r45" in p else (float(p["R45"]) if "R45" in p else None),
            r90=float(p["r90"]) if "r90" in p else (float(p["R90"]) if "R90" in p else None),
            A_swift=float(p.get("A_swift", p.get("SWIFT_A", 0.0))),
            eps0_swift=float(p.get("eps0_swift", p.get("SWIFT_EPS0", 0.001))),
            n_swift=float(p.get("n_swift", p.get("SWIFT_N", 0.2))),
            hp=float(p.get("hp", p.get("HP", 0.0))),
            c_rate=float(p.get("c_rate", p.get("C_RATE", p.get("MAT_C", 0.0)))),
            p_rate=float(p.get("p_rate", p.get("P_RATE", p.get("MAT_P", 1.0)))),
        )
    if isinstance(mat, dict):
        return Law128Params(
            E=float(mat.get("E", mat.get("MAT_E", 2.1e11))),
            nu=float(mat.get("nu", mat.get("MAT_NU", 0.3))),
            rho0=float(mat.get("rho0", mat.get("rho", mat.get("MAT_RHO", 7850.0)))),
            sig0=float(mat.get("sig0", mat.get("SIG0", mat.get("MAT_SIGY", 2.5e8)))),
            F=float(mat.get("F", mat.get("HILL_F", 0.5))),
            G_hill=float(mat.get("G_hill", mat.get("HILL_G", 0.5))),
            H=float(mat.get("H", mat.get("HILL_H", 0.5))),
            L=float(mat.get("L", mat.get("HILL_L", 1.5))),
            M=float(mat.get("M", mat.get("HILL_M", 1.5))),
            N=float(mat.get("N", mat.get("HILL_N", 1.5))),
            r00=float(mat["r00"]) if "r00" in mat else (float(mat["R00"]) if "R00" in mat else None),
            r45=float(mat["r45"]) if "r45" in mat else (float(mat["R45"]) if "R45" in mat else None),
            r90=float(mat["r90"]) if "r90" in mat else (float(mat["R90"]) if "R90" in mat else None),
            A_swift=float(mat.get("A_swift", mat.get("SWIFT_A", 0.0))),
            eps0_swift=float(mat.get("eps0_swift", mat.get("SWIFT_EPS0", 0.001))),
            n_swift=float(mat.get("n_swift", mat.get("SWIFT_N", 0.2))),
            hp=float(mat.get("hp", mat.get("HP", 0.0))),
            c_rate=float(mat.get("c_rate", mat.get("C_RATE", mat.get("MAT_C", 0.0)))),
            p_rate=float(mat.get("p_rate", mat.get("P_RATE", mat.get("MAT_P", 1.0)))),
        )
    return Law128Params()


def build_law128(mat: Any) -> Law128Params:
    """Build Law128Params from Material entity or dictionary."""
    return resolve(mat)


def needs_defgrad(mat: Any = None) -> bool:
    """LAW128 uses incremental rate formulation and does not need F."""
    return False


def extra_shapes(mat: Any = None, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """Persistent history variables for LAW128."""
    return {
        "uvar": (nip, 12) if nip > 1 else (12,),
    }


def sound_speed(
    mat: Any,
    eps: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    rho: Optional[float] = None,
    is_shell: bool = False,
    **kwargs: Any,
) -> float:
    """Dilatational sound speed for LAW128."""
    p = resolve(mat)
    eff_rho = float(rho) if (rho is not None and rho > 0.0) else p.rho0
    if eff_rho <= 0.0:
        eff_rho = 7850.0
    if is_shell:
        return math.sqrt(max(0.0, p.E / (eff_rho * max(1.0 - p.nu ** 2, 1e-6))))
    return math.sqrt(max(0.0, (p.K + _FOUR_THIRDS * p.G) / eff_rho))


def _hill_equivalent_stress_3d(p: Law128Params, s: np.ndarray) -> float:
    """Calculate 3D Hill 1948 equivalent stress."""
    sxx, syy, szz, sxy, syz, szx = s[0], s[1], s[2], s[3], s[4], s[5]
    val = (
        p.F * (syy - szz) ** 2
        + p.G_hill * (szz - sxx) ** 2
        + p.H * (sxx - syy) ** 2
        + 2.0 * p.L * (syz ** 2)
        + 2.0 * p.M * (szx ** 2)
        + 2.0 * p.N * (sxy ** 2)
    )
    return math.sqrt(max(0.0, val))


def _yield_stress(p: Law128Params, epsp: float, eps_dot: float) -> float:
    """Calculate current yield stress accounting for work hardening and strain rate."""
    base_sig = p.sig0 + p.hp * epsp
    if p.A_swift > 0.0:
        base_sig += p.A_swift * ((p.eps0_swift + epsp) ** p.n_swift)

    rate_factor = 1.0
    if p.c_rate > 0.0 and eps_dot > 0.0:
        rate_factor = 1.0 + (eps_dot / p.c_rate) ** (1.0 / max(0.01, p.p_rate))

    return base_sig * rate_factor


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Solid constitutive update for /MAT/LAW128."""
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
    soundsp = math.sqrt(max(0.0, (p.K + _FOUR_THIRDS * p.G) / p.rho0))

    lam = p.K - _TWO_THIRDS * p.G
    g2 = 2.0 * p.G

    for i in range(n):
        # Elastic trial stress
        de_vol = d[i, 0] + d[i, 1] + d[i, 2]
        s_trial = np.copy(s[i])
        s_trial[0] += lam * de_vol + g2 * d[i, 0]
        s_trial[1] += lam * de_vol + g2 * d[i, 1]
        s_trial[2] += lam * de_vol + g2 * d[i, 2]
        s_trial[3] += p.G * d[i, 3]
        s_trial[4] += p.G * d[i, 4]
        s_trial[5] += p.G * d[i, 5]

        # Deviatoric Hill equivalent stress
        sig_hill = _hill_equivalent_stress_3d(p, s_trial)

        eps_dot = 0.0
        if dt > 0.0:
            eps_dot = math.sqrt(d[i, 0]**2 + d[i, 1]**2 + d[i, 2]**2 + 0.5 * (d[i, 3]**2 + d[i, 4]**2 + d[i, 5]**2)) / dt

        yld = _yield_stress(p, ep[i], eps_dot)

        dpla = 0.0
        if sig_hill > yld:
            ratio = yld / max(sig_hill, _EM14)
            # Radial scaling back to Hill surface
            p_mean = _THIRD * (s_trial[0] + s_trial[1] + s_trial[2])
            s_trial[0] = p_mean + ratio * (s_trial[0] - p_mean)
            s_trial[1] = p_mean + ratio * (s_trial[1] - p_mean)
            s_trial[2] = p_mean + ratio * (s_trial[2] - p_mean)
            s_trial[3] *= ratio
            s_trial[4] *= ratio
            s_trial[5] *= ratio
            dpla = (1.0 - ratio) * sig_hill / max(3.0 * p.G, _EM20)

        s_out[i] = s_trial
        ep_out[i] = ep[i] + dpla
        c_out[i] = soundsp

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
    """Plane-stress shell update for /MAT/LAW128."""
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

        sxx = s_trial[0]
        syy = s_trial[1]
        sxy = s_trial[2] if s_trial.shape[0] > 2 else 0.0

        # Plane-stress Hill equivalent stress
        sig_hill_sq = (p.G_hill + p.H) * (sxx**2) + (p.F + p.H) * (syy**2) - 2.0 * p.H * sxx * syy + 2.0 * p.N * (sxy**2)
        sig_hill = math.sqrt(max(0.0, sig_hill_sq))

        eps_dot = 0.0
        if dt > 0.0:
            eps_dot = math.sqrt(d[i, 0]**2 + d[i, 1]**2 + 0.5 * (d[i, 2]**2 if d.shape[1] > 2 else 0.0)) / dt

        yld = _yield_stress(p, ep[i], eps_dot)

        dpla = 0.0
        if sig_hill > yld:
            ratio = yld / max(sig_hill, _EM14)
            s_trial[:3] *= ratio
            dpla = (1.0 - ratio) * sig_hill / max(3.0 * p.G, _EM20)

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
    """Return 6x6 elastic solid tangent matrix."""
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
        for key in (128, "128", "LAW128", "HILL_RATE", "MAT_LAW128"):
            MAT_PHYSICS_REGISTRY[key] = build_law128
    except Exception:
        pass


_register()
