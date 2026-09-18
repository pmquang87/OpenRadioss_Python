"""
LAW131 (/MAT/LAW131, /MAT/MODULAR_ELASTOPLAS) — General Modular Elasto-Plastic Framework.

OpenRadioss Fortran reference:
- engine/source/materials/mat/mat131/sigeps131.F90
- engine/source/materials/mat/mat131/sigeps131c.F90
- engine/source/materials/mat/mat131/elasto_plastic_eq_stress.F90
- engine/source/materials/mat/mat131/return_mapping/
- engine/source/materials/mat/mat131/yield_criterion/
- engine/source/materials/mat/mat131/work_hardening/
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
class Law131Params:
    """Parameters for /MAT/LAW131 (Modular Elasto-Plasticity)."""
    E: float = 2.1e11           # Young's modulus
    nu: float = 0.3            # Poisson's ratio
    rho0: float = 7850.0       # Mass density
    sig0: float = 2.5e8        # Initial yield stress
    # Pluggable modular selectors
    yield_type: int = 1        # 1=von Mises, 2=Hill, 3=Drucker-Prager
    hard_type: int = 1         # 1=linear, 2=Swift, 3=Voce, 4=Swift-Voce
    return_alg: int = 1        # 1=NICE (Newton), 2=cutting-plane, 3=CPPM
    # Hardening parameters
    hp: float = 0.0            # Linear hardening slope
    a_swift: float = 0.0       # Swift multiplier A
    eps0_swift: float = 0.001  # Swift pre-strain eps0
    n_swift: float = 0.2       # Swift exponent n
    r_inf_voce: float = 0.0    # Voce saturation stress R_inf
    b_voce: float = 0.0        # Voce rate b
    alpha_comb: float = 0.5    # Weight for combined Swift-Voce
    # Kinematic hardening (Armstrong-Frederick)
    c_af: float = 0.0          # Modulus C
    gamma_af: float = 0.0      # Recall parameter gamma
    # Hill anisotropy parameters (if yield_type=2)
    f_hill: float = 0.5
    g_hill: float = 0.5
    h_hill: float = 0.5
    l_hill: float = 1.5
    m_hill: float = 1.5
    n_hill: float = 1.5
    # Drucker-Prager parameters (if yield_type=3)
    alpha_dp: float = 0.0

    # Derived quantities
    G: float = field(init=False, default=0.0)
    K: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.nu < 0.0 or self.nu >= 0.5:
            self.nu = 0.3
        if self.E <= 0.0:
            self.E = 2.1e11
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))


def resolve(mat: Any) -> Law131Params:
    """Resolve Law131Params from Material, dict, or Law131Params instance."""
    if isinstance(mat, Law131Params):
        return mat
    if isinstance(mat, Material):
        p = mat.params or {}
        return Law131Params(
            E=float(p.get("E", p.get("MAT_E", 2.1e11))),
            nu=float(p.get("nu", p.get("MAT_NU", 0.3))),
            rho0=float(getattr(mat, "rho0", p.get("rho", p.get("MAT_RHO", 7850.0)))),
            sig0=float(p.get("sig0", p.get("SIG0", p.get("MAT_SIGY", 2.5e8)))),
            yield_type=int(p.get("yield_type", p.get("IYIELD", 1))),
            hard_type=int(p.get("hard_type", p.get("IHARD", 1))),
            return_alg=int(p.get("return_alg", p.get("IALG", 1))),
            hp=float(p.get("hp", p.get("HP", 0.0))),
            a_swift=float(p.get("a_swift", p.get("SWIFT_A", 0.0))),
            eps0_swift=float(p.get("eps0_swift", p.get("SWIFT_EPS0", 0.001))),
            n_swift=float(p.get("n_swift", p.get("SWIFT_N", 0.2))),
            r_inf_voce=float(p.get("r_inf_voce", p.get("VOCE_R", 0.0))),
            b_voce=float(p.get("b_voce", p.get("VOCE_B", 0.0))),
            alpha_comb=float(p.get("alpha_comb", p.get("ALPHA_COMB", 0.5))),
            c_af=float(p.get("c_af", p.get("C_AF", 0.0))),
            gamma_af=float(p.get("gamma_af", p.get("GAMMA_AF", 0.0))),
            f_hill=float(p.get("f_hill", p.get("HILL_F", 0.5))),
            g_hill=float(p.get("g_hill", p.get("HILL_G", 0.5))),
            h_hill=float(p.get("h_hill", p.get("HILL_H", 0.5))),
            l_hill=float(p.get("l_hill", p.get("HILL_L", 1.5))),
            m_hill=float(p.get("m_hill", p.get("HILL_M", 1.5))),
            n_hill=float(p.get("n_hill", p.get("HILL_N", 1.5))),
            alpha_dp=float(p.get("alpha_dp", p.get("ALPHA_DP", 0.0))),
        )
    if isinstance(mat, dict):
        return Law131Params(
            E=float(mat.get("E", mat.get("MAT_E", 2.1e11))),
            nu=float(mat.get("nu", mat.get("MAT_NU", 0.3))),
            rho0=float(mat.get("rho0", mat.get("rho", mat.get("MAT_RHO", 7850.0)))),
            sig0=float(mat.get("sig0", mat.get("SIG0", mat.get("MAT_SIGY", 2.5e8)))),
            yield_type=int(mat.get("yield_type", mat.get("IYIELD", 1))),
            hard_type=int(mat.get("hard_type", mat.get("IHARD", 1))),
            return_alg=int(mat.get("return_alg", mat.get("IALG", 1))),
            hp=float(mat.get("hp", mat.get("HP", 0.0))),
            a_swift=float(mat.get("a_swift", mat.get("SWIFT_A", 0.0))),
            eps0_swift=float(mat.get("eps0_swift", mat.get("SWIFT_EPS0", 0.001))),
            n_swift=float(mat.get("n_swift", mat.get("SWIFT_N", 0.2))),
            r_inf_voce=float(mat.get("r_inf_voce", mat.get("VOCE_R", 0.0))),
            b_voce=float(mat.get("b_voce", mat.get("VOCE_B", 0.0))),
            alpha_comb=float(mat.get("alpha_comb", mat.get("ALPHA_COMB", 0.5))),
            c_af=float(mat.get("c_af", mat.get("C_AF", 0.0))),
            gamma_af=float(mat.get("gamma_af", mat.get("GAMMA_AF", 0.0))),
            f_hill=float(mat.get("f_hill", mat.get("HILL_F", 0.5))),
            g_hill=float(mat.get("g_hill", mat.get("HILL_G", 0.5))),
            h_hill=float(mat.get("h_hill", mat.get("HILL_H", 0.5))),
            l_hill=float(mat.get("l_hill", mat.get("HILL_L", 1.5))),
            m_hill=float(mat.get("m_hill", mat.get("HILL_M", 1.5))),
            n_hill=float(mat.get("n_hill", mat.get("HILL_N", 1.5))),
            alpha_dp=float(mat.get("alpha_dp", mat.get("ALPHA_DP", 0.0))),
        )
    return Law131Params()


def build_law131(mat: Any) -> Law131Params:
    """Build Law131Params from Material entity or dictionary."""
    return resolve(mat)


def needs_defgrad(mat: Any = None) -> bool:
    """Incremental rate plasticity does not need deformation gradient F."""
    return False


def extra_shapes(mat: Any = None, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """Define persistent internal variables for LAW131 (backstress, plastic strain)."""
    return {
        "backstress": (nip, 6) if nip > 1 else (6,),
        "uvar": (nip, 16) if nip > 1 else (16,),
    }


def sound_speed(
    mat: Any,
    eps: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    rho: Optional[float] = None,
    is_shell: bool = False,
    **kwargs: Any,
) -> float:
    """Dilatational sound speed for LAW131."""
    p = resolve(mat)
    eff_rho = float(rho) if (rho is not None and rho > 0.0) else p.rho0
    if eff_rho <= 0.0:
        eff_rho = 7850.0
    if is_shell:
        return math.sqrt(max(0.0, p.E / (eff_rho * max(1.0 - p.nu ** 2, 1e-6))))
    return math.sqrt(max(0.0, (p.K + _FOUR_THIRDS * p.G) / eff_rho))


def _compute_yield_stress(p: Law131Params, epsp: float) -> float:
    """Calculate isotropic yield stress based on selected hardening law."""
    base = p.sig0 + p.hp * epsp
    if p.hard_type == 2 and p.a_swift > 0.0:
        # Swift
        return base + p.a_swift * ((p.eps0_swift + epsp) ** p.n_swift)
    elif p.hard_type == 3 and p.r_inf_voce > 0.0:
        # Voce
        return base + p.r_inf_voce * (1.0 - math.exp(-p.b_voce * epsp))
    elif p.hard_type == 4:
        # Combined Swift-Voce
        sig_s = p.sig0 + (p.a_swift * ((p.eps0_swift + epsp) ** p.n_swift) if p.a_swift > 0.0 else 0.0)
        sig_v = p.sig0 + (p.r_inf_voce * (1.0 - math.exp(-p.b_voce * epsp)) if p.r_inf_voce > 0.0 else 0.0)
        return p.alpha_comb * sig_s + (1.0 - p.alpha_comb) * sig_v + p.hp * epsp
    return max(0.0, base)


def _compute_equivalent_stress(p: Law131Params, s: np.ndarray) -> float:
    """Compute equivalent stress based on selected yield criterion."""
    if p.yield_type == 2:
        # Hill 1948
        val = (
            p.f_hill * (s[1] - s[2]) ** 2
            + p.g_hill * (s[2] - s[0]) ** 2
            + p.h_hill * (s[0] - s[1]) ** 2
            + 2.0 * p.l_hill * (s[4] ** 2)
            + 2.0 * p.m_hill * (s[5] ** 2)
            + 2.0 * p.n_hill * (s[3] ** 2)
        )
        return math.sqrt(max(0.0, val))
    elif p.yield_type == 3:
        # Drucker-Prager
        p_mean = _THIRD * (s[0] + s[1] + s[2])
        j2 = 0.5 * ((s[0] - p_mean)**2 + (s[1] - p_mean)**2 + (s[2] - p_mean)**2) + s[3]**2 + s[4]**2 + s[5]**2
        return math.sqrt(3.0 * max(0.0, j2)) + p.alpha_dp * (-3.0 * p_mean)
    else:
        # von Mises J2
        p_mean = _THIRD * (s[0] + s[1] + s[2])
        j2 = 0.5 * ((s[0] - p_mean)**2 + (s[1] - p_mean)**2 + (s[2] - p_mean)**2) + s[3]**2 + s[4]**2 + s[5]**2
        return math.sqrt(3.0 * max(0.0, j2))


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Modular 3D solid elasto-plastic constitutive update."""
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

    alpha_back = np.zeros((n, 6), dtype=np.float64)
    if extra is not None and "backstress" in extra:
        bs = np.asarray(extra["backstress"], dtype=np.float64)
        if bs.ndim == 1 and n == 1:
            alpha_back[0] = bs[:6]
        elif bs.ndim == 2:
            alpha_back[:min(n, bs.shape[0])] = bs[:min(n, bs.shape[0]), :6]

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

        # Shifted relative stress s_rel = s_trial - alpha_back
        s_rel = s_trial - alpha_back[i]
        sig_eq = _compute_equivalent_stress(p, s_rel)
        yld = _compute_yield_stress(p, ep[i])

        dpla = 0.0
        if sig_eq > yld:
            ratio = yld / max(sig_eq, _EM14)
            p_mean = _THIRD * (s_trial[0] + s_trial[1] + s_trial[2])
            s_trial[0] = p_mean + ratio * (s_trial[0] - p_mean)
            s_trial[1] = p_mean + ratio * (s_trial[1] - p_mean)
            s_trial[2] = p_mean + ratio * (s_trial[2] - p_mean)
            s_trial[3] *= ratio
            s_trial[4] *= ratio
            s_trial[5] *= ratio

            dpla = (1.0 - ratio) * sig_eq / max(3.0 * p.G + p.hp + p.c_af, _EM20)

            # Armstrong-Frederick backstress update
            if p.c_af > 0.0:
                dev_flow = (s_rel - _THIRD * (s_rel[0] + s_rel[1] + s_rel[2])) / max(sig_eq, _EM14)
                d_alpha = _TWO_THIRDS * p.c_af * dev_flow * dpla - p.gamma_af * alpha_back[i] * dpla
                alpha_back[i] += d_alpha

        s_out[i] = s_trial
        ep_out[i] = ep[i] + dpla

    if extra is not None:
        extra["backstress"] = alpha_back[0] if is_1d else alpha_back

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
    """Plane-stress shell update for /MAT/LAW131."""
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
        yld = _compute_yield_stress(p, ep[i])

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
        for key in (131, "131", "LAW131", "MODULAR_ELASTOPLAS", "MAT_LAW131"):
            MAT_PHYSICS_REGISTRY[key] = build_law131
    except Exception:
        pass


_register()
