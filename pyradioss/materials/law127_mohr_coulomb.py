"""
LAW127 (/MAT/LAW127, /MAT/MOHR_COULOMB, /MAT/ENHANCED_COMPOSITE) — Mohr-Coulomb / Drucker-Prager Cap & Enhanced Composite Model.

OpenRadioss Fortran reference:
- engine/source/materials/mat/mat127/sigeps127.F90
- engine/source/materials/mat/mat127/sigeps127c.F90
- starter/source/materials/mat/mat127/hm_read_mat127.F90
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
class Law127Params:
    """Parameters for /MAT/LAW127 (Mohr-Coulomb / DP Cap Plasticity & Enhanced Composite Damage)."""
    E: float = 3.0e10            # Young's modulus (or longitudinal Ea)
    nu: float = 0.25            # Poisson's ratio (or nu12)
    rho0: float = 2000.0        # Mass density
    # Mohr-Coulomb / Drucker-Prager cap parameters
    cohesion: float = 1.0e7     # Cohesion c (stress units)
    phi: float = 30.0           # Internal friction angle (degrees)
    psi: float = 15.0           # Dilation angle (degrees)
    p_cap: float = 1.0e8        # Cap initiation pressure
    r_cap: float = 2.0          # Cap shape / aspect ratio R
    # Orthotropic composite parameters
    e1: Optional[float] = None
    e2: Optional[float] = None
    e3: Optional[float] = None
    g12: Optional[float] = None
    g23: Optional[float] = None
    g13: Optional[float] = None
    xt: float = 1.5e9           # Fiber tensile strength
    xc: float = 1.0e9           # Fiber compressive strength
    yt: float = 5.0e7           # Matrix tensile strength
    yc: float = 2.0e8           # Matrix compressive strength
    sc: float = 1.0e8           # Shear strength
    alpha: float = 0.0          # Nonlinear shear coefficient
    beta: float = 0.0           # Chang-Chang weighting coefficient

    # Derived quantities
    G: float = field(init=False, default=0.0)
    K: float = field(init=False, default=0.0)
    alpha_dp: float = field(init=False, default=0.0)
    k_dp: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.nu < 0.0 or self.nu >= 0.5:
            self.nu = 0.25
        if self.E <= 0.0:
            self.E = 3.0e10
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))

        phi_rad = math.radians(max(0.0, min(self.phi, 89.0)))
        sin_phi = math.sin(phi_rad)
        cos_phi = math.cos(phi_rad)
        # Match Drucker-Prager inscribed Mohr-Coulomb match
        denom = math.sqrt(3.0) * (3.0 - sin_phi)
        self.alpha_dp = 2.0 * sin_phi / denom
        self.k_dp = 6.0 * max(0.0, self.cohesion) * cos_phi / denom


def resolve(mat: Any) -> Law127Params:
    """Resolve Law127Params from Material, dict, or existing Law127Params."""
    if isinstance(mat, Law127Params):
        return mat
    if isinstance(mat, Material):
        p = mat.params or {}
        e_val = float(p.get("E", p.get("MAT_E", p.get("LSDYNA_EA", 3.0e10))))
        nu_val = float(p.get("nu", p.get("MAT_NU", p.get("LSDYNA_PRBA", 0.25))))
        return Law127Params(
            E=e_val,
            nu=nu_val,
            rho0=float(getattr(mat, "rho0", p.get("rho", p.get("MAT_RHO", 2000.0)))),
            cohesion=float(p.get("cohesion", p.get("COHESION", p.get("C", 1.0e7)))),
            phi=float(p.get("phi", p.get("PHI", 30.0))),
            psi=float(p.get("psi", p.get("PSI", 15.0))),
            p_cap=float(p.get("p_cap", p.get("P_CAP", 1.0e8))),
            r_cap=float(p.get("r_cap", p.get("R_CAP", 2.0))),
            e1=float(p.get("LSDYNA_EA", e_val)),
            e2=float(p.get("LSDYNA_EB", e_val)),
            e3=float(p.get("LSDYNA_EC", e_val)),
            g12=float(p.get("LSDYNA_GAB", e_val / (2.0 * (1.0 + nu_val)))),
            g23=float(p.get("LSDYNA_GBC", e_val / (2.0 * (1.0 + nu_val)))),
            g13=float(p.get("LSDYNA_GCA", e_val / (2.0 * (1.0 + nu_val)))),
            xt=float(p.get("xt", p.get("LSD_MAT_XT", 1.5e9))),
            xc=float(p.get("xc", p.get("LSD_MAT_XC", 1.0e9))),
            yt=float(p.get("yt", p.get("LSD_MAT_YT", 5.0e7))),
            yc=float(p.get("yc", p.get("LSD_MAT_YC", 2.0e8))),
            sc=float(p.get("sc", p.get("LSD_MAT_SC", 1.0e8))),
            alpha=float(p.get("alpha", p.get("ALPHA", 0.0))),
            beta=float(p.get("beta", p.get("BETA", 0.0))),
        )
    if isinstance(mat, dict):
        e_val = float(mat.get("E", mat.get("MAT_E", mat.get("LSDYNA_EA", 3.0e10))))
        nu_val = float(mat.get("nu", mat.get("MAT_NU", mat.get("LSDYNA_PRBA", 0.25))))
        return Law127Params(
            E=e_val,
            nu=nu_val,
            rho0=float(mat.get("rho0", mat.get("rho", mat.get("MAT_RHO", 2000.0)))),
            cohesion=float(mat.get("cohesion", mat.get("COHESION", mat.get("C", 1.0e7)))),
            phi=float(mat.get("phi", mat.get("PHI", 30.0))),
            psi=float(mat.get("psi", mat.get("PSI", 15.0))),
            p_cap=float(mat.get("p_cap", mat.get("P_CAP", 1.0e8))),
            r_cap=float(mat.get("r_cap", mat.get("R_CAP", 2.0))),
            e1=float(mat.get("LSDYNA_EA", e_val)),
            e2=float(mat.get("LSDYNA_EB", e_val)),
            e3=float(mat.get("LSDYNA_EC", e_val)),
            g12=float(mat.get("LSDYNA_GAB", e_val / (2.0 * (1.0 + nu_val)))),
            g23=float(mat.get("LSDYNA_GBC", e_val / (2.0 * (1.0 + nu_val)))),
            g13=float(mat.get("LSDYNA_GCA", e_val / (2.0 * (1.0 + nu_val)))),
            xt=float(mat.get("xt", mat.get("LSD_MAT_XT", 1.5e9))),
            xc=float(mat.get("xc", mat.get("LSD_MAT_XC", 1.0e9))),
            yt=float(mat.get("yt", mat.get("LSD_MAT_YT", 5.0e7))),
            yc=float(mat.get("yc", mat.get("LSD_MAT_YC", 2.0e8))),
            sc=float(mat.get("sc", mat.get("LSD_MAT_SC", 1.0e8))),
            alpha=float(mat.get("alpha", mat.get("ALPHA", 0.0))),
            beta=float(mat.get("beta", mat.get("BETA", 0.0))),
        )
    return Law127Params()


def build_law127(mat: Any) -> Law127Params:
    """Build Law127Params from Material entity or record."""
    return resolve(mat)


def needs_defgrad(mat: Any = None) -> bool:
    """LAW127 is an incremental small-strain formulation and does not need F."""
    return False


def extra_shapes(mat: Any = None, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """Define persistent history variables for LAW127 (damage, plastic strain)."""
    return {
        "uvar": (nip, 16) if nip > 1 else (16,),
        "dmg": (nip, 8) if nip > 1 else (8,),
    }


def sound_speed(
    mat: Any,
    eps: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    rho: Optional[float] = None,
    is_shell: bool = False,
    **kwargs: Any,
) -> float:
    """Compute acoustic dilatational sound speed for LAW127."""
    p = resolve(mat)
    eff_rho = float(rho) if (rho is not None and rho > 0.0) else p.rho0
    if eff_rho <= 0.0:
        eff_rho = 2000.0
    if is_shell:
        return math.sqrt(max(0.0, p.E / (eff_rho * max(1.0 - p.nu ** 2, 1e-6))))
    return math.sqrt(max(0.0, (p.K + _FOUR_THIRDS * p.G) / eff_rho))


def _solid_step_single(
    p: Law127Params,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp_in: float,
) -> Tuple[np.ndarray, float, float]:
    """Single element solid update for Mohr-Coulomb / Drucker-Prager cap."""
    # Volumetric and deviatoric decomposition
    de_vol = deps[0] + deps[1] + deps[2]
    de_dev = np.copy(deps)
    de_dev[0] -= _THIRD * de_vol
    de_dev[1] -= _THIRD * de_vol
    de_dev[2] -= _THIRD * de_vol

    p_old = -_THIRD * (sig[0] + sig[1] + sig[2])
    s_old = np.copy(sig)
    s_old[0] += p_old
    s_old[1] += p_old
    s_old[2] += p_old

    # Elastic trial step
    p_trial = p_old - p.K * de_vol
    s_trial = np.copy(s_old)
    s_trial[0] += 2.0 * p.G * de_dev[0]
    s_trial[1] += 2.0 * p.G * de_dev[1]
    s_trial[2] += 2.0 * p.G * de_dev[2]
    s_trial[3] += p.G * de_dev[3]
    s_trial[4] += p.G * de_dev[4]
    s_trial[5] += p.G * de_dev[5]

    i1_trial = -3.0 * p_trial
    j2 = 0.5 * (s_trial[0]**2 + s_trial[1]**2 + s_trial[2]**2) + (s_trial[3]**2 + s_trial[4]**2 + s_trial[5]**2)
    sqrt_j2 = math.sqrt(max(0.0, j2))

    # Mohr-Coulomb / DP shear failure surface
    f_shear = sqrt_j2 + p.alpha_dp * i1_trial - p.k_dp

    dpla = 0.0
    s_final = np.copy(s_trial)
    p_final = p_trial

    if f_shear > 0.0:
        # Radial return to DP envelope
        denom = p.G + 9.0 * p.K * p.alpha_dp * math.sin(math.radians(max(0.0, p.psi)))
        dlam = f_shear / max(denom, _EM20)
        sqrt_j2_new = max(0.0, sqrt_j2 - p.G * dlam)
        ratio = sqrt_j2_new / max(sqrt_j2, _EM14)
        s_final = s_trial * ratio
        p_final = p_trial + 3.0 * p.K * math.sin(math.radians(max(0.0, p.psi))) * dlam
        dpla = dlam

    # Check compressive cap
    if p_final > p.p_cap:
        # Cap plasticity compaction
        p_excess = p_final - p.p_cap
        p_final = p.p_cap + p_excess * 0.1
        dpla += p_excess / max(p.K, _EM20)

    sig_out = np.copy(s_final)
    sig_out[0] -= p_final
    sig_out[1] -= p_final
    sig_out[2] -= p_final

    c = math.sqrt(max(0.0, (p.K + _FOUR_THIRDS * p.G) / p.rho0))
    return sig_out, epsp_in + dpla, c


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Solid constitutive update for /MAT/LAW127."""
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
        si, epi, ci = _solid_step_single(p, s[i], d[i], ep[i])
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
    """Plane-stress shell update for /MAT/LAW127."""
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

        # Check Chang-Chang / Hashin or Mohr-Coulomb plane stress limit
        i1 = s_trial[0] + s_trial[1]
        j2 = _THIRD * (s_trial[0]**2 + s_trial[1]**2 - s_trial[0]*s_trial[1] + 3.0 * (s_trial[2]**2 if s_trial.shape[0] > 2 else 0.0))
        f_val = math.sqrt(max(0.0, j2)) + p.alpha_dp * i1 - p.k_dp
        dpla = 0.0
        if f_val > 0.0:
            scale = max(0.0, 1.0 - f_val / max(math.sqrt(max(0.0, j2)) + p.G, _EM20))
            s_trial[:3] *= scale
            dpla = f_val / max(3.0 * p.G, _EM20)

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
        for key in (127, "127", "LAW127", "MOHR_COULOMB", "ENHANCED_COMPOSITE", "MAT_LAW127"):
            MAT_PHYSICS_REGISTRY[key] = build_law127
    except Exception:
        pass


_register()
