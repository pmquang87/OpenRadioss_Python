"""
LAW187 (/MAT/LAW187, /MAT/SAMP1_POLYMER, /MAT/SAMP-1) — SAMP-1 Semi-Analytical Model for Polymers with Crazing & Dilatancy.

OpenRadioss Fortran reference:
- engine/source/materials/mat/mat187/sigeps187.F
- hm_cfg_files/config/CFG/radioss2021/MAT/mat_law187.cfg
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
class Law187Params:
    """Parameters for /MAT/LAW187 (SAMP-1 polymer model with crazing and dilatancy)."""
    E: float = 2.0e9            # Young's modulus
    nu: float = 0.35           # Elastic Poisson's ratio
    rho0: float = 1000.0       # Mass density
    sig0: float = 4.0e7        # Initial tensile yield stress
    # Dilatancy and crazing
    nu_p: float = 0.35         # Plastic Poisson's ratio (governs plastic dilatancy)
    sig_craze: float = 6.0e7   # Crazing / cavitation stress threshold
    d_max: float = 0.99        # Maximum damage
    # Parabolic / pressure-dependent yield parameters (Phi = sig_vm^2 - A0 - A1*P - A2*P^2)
    a0: float = 1.6e15         # A0 = 3 * sig_shear^2 (or sig0^2)
    a1: float = 0.0            # Linear pressure term
    a2: float = 0.0            # Quadratic pressure term
    # Swift-Voce hardening
    alpha_sv: float = 0.5      # Swift-Voce weighting (1=Swift, 0=Voce)
    a_swift: float = 8.0e7     # Swift A
    eps0_swift: float = 0.005  # Swift eps0
    n_swift: float = 0.15      # Swift n
    q_voce: float = 2.0e7      # Voce Q
    b_voce: float = 20.0       # Voce b
    hp: float = 0.0            # Additional linear hardening

    # Derived elastic moduli
    G: float = field(init=False, default=0.0)
    K: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.nu < 0.0 or self.nu >= 0.5:
            self.nu = 0.35
        if self.E <= 0.0:
            self.E = 2.0e9
        if self.a0 <= 0.0:
            self.a0 = self.sig0 ** 2
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))


def resolve(mat: Any) -> Law187Params:
    """Resolve Law187Params from Material, dict, or Law187Params instance."""
    if isinstance(mat, Law187Params):
        return mat
    if isinstance(mat, Material):
        p = mat.params or {}
        e_val = float(p.get("E", p.get("MAT_E", 2.0e9)))
        sig0_val = float(p.get("sig0", p.get("SIG0", p.get("MAT_SIGY", 4.0e7))))
        return Law187Params(
            E=e_val,
            nu=float(p.get("nu", p.get("MAT_NU", 0.35))),
            rho0=float(getattr(mat, "rho0", p.get("rho", p.get("MAT_RHO", 1000.0)))),
            sig0=sig0_val,
            nu_p=float(p.get("nu_p", p.get("NU_P", 0.35))),
            sig_craze=float(p.get("sig_craze", p.get("SIG_CRAZE", 6.0e7))),
            d_max=float(p.get("d_max", p.get("D_MAX", 0.99))),
            a0=float(p.get("a0", p.get("A0", sig0_val ** 2))),
            a1=float(p.get("a1", p.get("A1", 0.0))),
            a2=float(p.get("a2", p.get("A2", 0.0))),
            alpha_sv=float(p.get("alpha_sv", p.get("ALPHA", 0.5))),
            a_swift=float(p.get("a_swift", p.get("ASWIFT", 8.0e7))),
            eps0_swift=float(p.get("eps0_swift", p.get("EPSO", 0.005))),
            n_swift=float(p.get("n_swift", p.get("NEXP", 0.15))),
            q_voce=float(p.get("q_voce", p.get("QVOCE", 2.0e7))),
            b_voce=float(p.get("b_voce", p.get("BETA", 20.0))),
            hp=float(p.get("hp", p.get("HP", 0.0))),
        )
    if isinstance(mat, dict):
        e_val = float(mat.get("E", mat.get("MAT_E", 2.0e9)))
        sig0_val = float(mat.get("sig0", mat.get("SIG0", mat.get("MAT_SIGY", 4.0e7))))
        return Law187Params(
            E=e_val,
            nu=float(mat.get("nu", mat.get("MAT_NU", 0.35))),
            rho0=float(mat.get("rho0", mat.get("rho", mat.get("MAT_RHO", 1000.0)))),
            sig0=sig0_val,
            nu_p=float(mat.get("nu_p", mat.get("NU_P", 0.35))),
            sig_craze=float(mat.get("sig_craze", mat.get("SIG_CRAZE", 6.0e7))),
            d_max=float(mat.get("d_max", mat.get("D_MAX", 0.99))),
            a0=float(mat.get("a0", mat.get("A0", sig0_val ** 2))),
            a1=float(mat.get("a1", mat.get("A1", 0.0))),
            a2=float(mat.get("a2", mat.get("A2", 0.0))),
            alpha_sv=float(mat.get("alpha_sv", mat.get("ALPHA", 0.5))),
            a_swift=float(mat.get("a_swift", mat.get("ASWIFT", 8.0e7))),
            eps0_swift=float(mat.get("eps0_swift", mat.get("EPSO", 0.005))),
            n_swift=float(mat.get("n_swift", mat.get("NEXP", 0.15))),
            q_voce=float(mat.get("q_voce", mat.get("QVOCE", 2.0e7))),
            b_voce=float(mat.get("b_voce", mat.get("BETA", 20.0))),
            hp=float(mat.get("hp", mat.get("HP", 0.0))),
        )
    return Law187Params()


def build_law187(mat: Any) -> Law187Params:
    """Build Law187Params from Material entity or dictionary."""
    return resolve(mat)


def needs_defgrad(mat: Any = None) -> bool:
    """LAW187 uses incremental formulation and does not need F."""
    return False


def extra_shapes(mat: Any = None, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """Define persistent history variables for LAW187 (plastic strain, crazing damage)."""
    return {
        "uvar": (nip, 16) if nip > 1 else (16,),
        "dmg": (nip, 2) if nip > 1 else (2,),
    }


def sound_speed(
    mat: Any,
    eps: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    rho: Optional[float] = None,
    is_shell: bool = False,
    **kwargs: Any,
) -> float:
    """Dilatational sound speed for LAW187."""
    p = resolve(mat)
    eff_rho = float(rho) if (rho is not None and rho > 0.0) else p.rho0
    if eff_rho <= 0.0:
        eff_rho = 1000.0
    if is_shell:
        return math.sqrt(max(0.0, p.E / (eff_rho * max(1.0 - p.nu ** 2, 1e-6))))
    return math.sqrt(max(0.0, (p.K + _FOUR_THIRDS * p.G) / eff_rho))


def _yield_stress_samp1(p: Law187Params, epsp: float) -> float:
    """Compute current yield stress via Swift-Voce model."""
    k_swift = p.a_swift * ((p.eps0_swift + epsp) ** p.n_swift)
    k_voce = p.q_voce * (1.0 - math.exp(-p.b_voce * epsp)) + p.sig0
    return p.alpha_sv * k_swift + (1.0 - p.alpha_sv) * k_voce + p.hp * epsp


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Vectorized solid update for SAMP-1 polymer law."""
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

    dmg_craze = np.zeros(n, dtype=np.float64)
    dmg_all = np.zeros((n, 2), dtype=np.float64)
    if extra is not None and "dmg" in extra:
        dmg_arr = np.asarray(extra["dmg"], dtype=np.float64)
        if dmg_arr.ndim == 1 and n == 1:
            dmg_all[0, :min(2, dmg_arr.shape[0])] = dmg_arr[:2]
            dmg_craze[0] = dmg_all[0, 0]
        elif dmg_arr.ndim == 2:
            dmg_all[:min(n, dmg_arr.shape[0]), :min(2, dmg_arr.shape[1])] = dmg_arr[:min(n, dmg_arr.shape[0]), :min(2, dmg_arr.shape[1])]
            dmg_craze[:min(n, dmg_arr.shape[0])] = dmg_all[:min(n, dmg_arr.shape[0]), 0]

    lam = p.K - _TWO_THIRDS * p.G
    g2 = 2.0 * p.G
    soundsp = math.sqrt(max(0.0, (p.K + _FOUR_THIRDS * p.G) / p.rho0))
    c_out = np.full(n, soundsp, dtype=np.float64)

    s_out = np.zeros_like(s)
    ep_out = np.zeros(n, dtype=np.float64)

    for i in range(n):
        de_vol = d[i, 0] + d[i, 1] + d[i, 2]
        s_trial = np.copy(s[i])
        s_trial[0] += lam * de_vol + g2 * d[i, 0]
        s_trial[1] += lam * de_vol + g2 * d[i, 1]
        s_trial[2] += lam * de_vol + g2 * d[i, 2]
        s_trial[3] += p.G * d[i, 3]
        s_trial[4] += p.G * d[i, 4]
        s_trial[5] += p.G * d[i, 5]

        p_hyd = -_THIRD * (s_trial[0] + s_trial[1] + s_trial[2])
        j2 = 0.5 * ((s_trial[0] + p_hyd)**2 + (s_trial[1] + p_hyd)**2 + (s_trial[2] + p_hyd)**2) + s_trial[3]**2 + s_trial[4]**2 + s_trial[5]**2
        sig_vm = math.sqrt(3.0 * max(0.0, j2))

        # Check crazing under high hydrostatic tension
        if -p_hyd > p.sig_craze:
            d_inc = (-p_hyd - p.sig_craze) / max(p.E, 1.0)
            dmg_craze[i] = min(p.d_max, dmg_craze[i] + d_inc)

        # Pressure-dependent yield
        yld_base = _yield_stress_samp1(p, ep[i])
        scale_p = max(0.0, 1.0 + (p.a1 * p_hyd + p.a2 * (p_hyd**2)) / max(p.a0, 1.0))
        yld = yld_base * math.sqrt(scale_p)

        dpla = 0.0
        if sig_vm > yld:
            ratio = yld / max(sig_vm, _EM14)
            s_trial[0] = -p_hyd + ratio * (s_trial[0] + p_hyd)
            s_trial[1] = -p_hyd + ratio * (s_trial[1] + p_hyd)
            s_trial[2] = -p_hyd + ratio * (s_trial[2] + p_hyd)
            s_trial[3] *= ratio
            s_trial[4] *= ratio
            s_trial[5] *= ratio

            dpla = (1.0 - ratio) * sig_vm / max(3.0 * p.G, _EM20)

        # Apply crazing damage
        if dmg_craze[i] > 0.0:
            s_trial *= (1.0 - dmg_craze[i])

        s_out[i] = s_trial
        ep_out[i] = ep[i] + dpla

    if extra is not None:
        dmg_all[:, 0] = dmg_craze
        extra["dmg"] = dmg_all[0] if is_1d else dmg_all

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
    """Plane-stress shell update for SAMP-1 polymer law."""
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

        sig_vm = math.sqrt(max(0.0, sxx**2 + syy**2 - sxx * syy + 3.0 * (sxy**2)))
        yld = _yield_stress_samp1(p, ep[i])

        dpla = 0.0
        if sig_vm > yld:
            ratio = yld / max(sig_vm, _EM14)
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
        for key in (187, "187", "LAW187", "SAMP1", "SAMP1_POLYMER", "MAT_LAW187"):
            MAT_PHYSICS_REGISTRY[key] = build_law187
    except Exception:
        pass


_register()
