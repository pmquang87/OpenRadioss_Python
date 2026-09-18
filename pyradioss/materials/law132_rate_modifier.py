"""
LAW132 (/MAT/LAW132, /MAT/RATE_MODIFIER, /MAT/DAIMLER_CAMANHO) — Generalized Strain-Rate Dependency Modifier for Composites.

OpenRadioss Fortran reference:
- engine/source/materials/mat/mat132/sigeps132c.F90
- engine/source/materials/mat/mat132/rate_dependency_parameters.F90
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
class Law132Params:
    """Parameters for /MAT/LAW132 (Daimler-Camanho rate-dependent composite model)."""
    E: float = 1.4e11            # Longitudinal modulus Ea
    eb: float = 9.0e9           # Transverse modulus Eb
    nu: float = 0.3             # In-plane Poisson's ratio nu_ba
    rho0: float = 1600.0        # Mass density
    gab: float = 4.5e9          # In-plane shear modulus Gab
    gbc: float = 3.2e9          # Transverse shear modulus Gbc
    gca: float = 4.5e9          # Transverse shear modulus Gca
    # Static strengths
    xt: float = 2.0e9           # Longitudinal tensile strength
    xc: float = 1.2e9           # Longitudinal compressive strength
    yt: float = 5.0e7           # Transverse tensile strength
    yc: float = 2.0e8           # Transverse compressive strength
    sl: float = 8.0e7           # In-plane shear strength
    # Static fracture energies (N/m or J/m2)
    gxt: float = 8.0e4          # Longitudinal tensile fracture energy
    gxc: float = 4.0e4          # Longitudinal compressive fracture energy
    gyt: float = 3.0e2          # Transverse tensile fracture energy
    gyc: float = 1.5e3          # Transverse compressive fracture energy
    gsl: float = 1.0e3          # Shear fracture energy
    # Dynamic rate multiplier coefficients
    eps0_rate: float = 1.0      # Reference strain rate (s^-1)
    c_xt: float = 0.05          # Rate multiplier on Xt
    c_xc: float = 0.05          # Rate multiplier on Xc
    c_yt: float = 0.08          # Rate multiplier on Yt
    c_yc: float = 0.08          # Rate multiplier on Yc
    c_sl: float = 0.06          # Rate multiplier on Sl

    # Derived elastic moduli
    G: float = field(init=False, default=0.0)
    K: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.nu < 0.0 or self.nu >= 0.5:
            self.nu = 0.3
        if self.E <= 0.0:
            self.E = 1.4e11
        if self.eb <= 0.0:
            self.eb = 9.0e9
        if self.gab <= 0.0:
            self.gab = 4.5e9
        self.G = self.gab
        self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))


def resolve(mat: Any) -> Law132Params:
    """Resolve Law132Params from Material, dict, or Law132Params instance."""
    if isinstance(mat, Law132Params):
        return mat
    if isinstance(mat, Material):
        p = mat.params or {}
        ea = float(p.get("E", p.get("MAT_E", p.get("EA", 1.4e11))))
        eb = float(p.get("eb", p.get("EB", 9.0e9)))
        nu = float(p.get("nu", p.get("MAT_NU", p.get("PRBA", 0.3))))
        return Law132Params(
            E=ea,
            eb=eb,
            nu=nu,
            rho0=float(getattr(mat, "rho0", p.get("rho", p.get("MAT_RHO", 1600.0)))),
            gab=float(p.get("gab", p.get("GAB", 4.5e9))),
            gbc=float(p.get("gbc", p.get("GBC", 3.2e9))),
            gca=float(p.get("gca", p.get("GCA", 4.5e9))),
            xt=float(p.get("xt", p.get("XT", 2.0e9))),
            xc=float(p.get("xc", p.get("XC", 1.2e9))),
            yt=float(p.get("yt", p.get("YT", 5.0e7))),
            yc=float(p.get("yc", p.get("YC", 2.0e8))),
            sl=float(p.get("sl", p.get("SL", 8.0e7))),
            gxt=float(p.get("gxt", p.get("GXT", 8.0e4))),
            gxc=float(p.get("gxc", p.get("GXC", 4.0e4))),
            gyt=float(p.get("gyt", p.get("GYT", 3.0e2))),
            gyc=float(p.get("gyc", p.get("GYC", 1.5e3))),
            gsl=float(p.get("gsl", p.get("GSL", 1.0e3))),
            eps0_rate=float(p.get("eps0_rate", p.get("EPS0_RATE", 1.0))),
            c_xt=float(p.get("c_xt", p.get("C_XT", 0.05))),
            c_xc=float(p.get("c_xc", p.get("C_XC", 0.05))),
            c_yt=float(p.get("c_yt", p.get("C_YT", 0.08))),
            c_yc=float(p.get("c_yc", p.get("C_YC", 0.08))),
            c_sl=float(p.get("c_sl", p.get("C_SL", 0.06))),
        )
    if isinstance(mat, dict):
        ea = float(mat.get("E", mat.get("MAT_E", mat.get("EA", 1.4e11))))
        eb = float(mat.get("eb", mat.get("EB", 9.0e9)))
        nu = float(mat.get("nu", mat.get("MAT_NU", mat.get("PRBA", 0.3))))
        return Law132Params(
            E=ea,
            eb=eb,
            nu=nu,
            rho0=float(mat.get("rho0", mat.get("rho", mat.get("MAT_RHO", 1600.0)))),
            gab=float(mat.get("gab", mat.get("GAB", 4.5e9))),
            gbc=float(mat.get("gbc", mat.get("GBC", 3.2e9))),
            gca=float(mat.get("gca", mat.get("GCA", 4.5e9))),
            xt=float(mat.get("xt", mat.get("XT", 2.0e9))),
            xc=float(mat.get("xc", mat.get("XC", 1.2e9))),
            yt=float(mat.get("yt", mat.get("YT", 5.0e7))),
            yc=float(mat.get("yc", mat.get("YC", 2.0e8))),
            sl=float(mat.get("sl", mat.get("SL", 8.0e7))),
            gxt=float(mat.get("gxt", mat.get("GXT", 8.0e4))),
            gxc=float(mat.get("gxc", mat.get("GXC", 4.0e4))),
            gyt=float(mat.get("gyt", mat.get("GYT", 3.0e2))),
            gyc=float(mat.get("gyc", mat.get("GYC", 1.5e3))),
            gsl=float(mat.get("gsl", mat.get("GSL", 1.0e3))),
            eps0_rate=float(mat.get("eps0_rate", mat.get("EPS0_RATE", 1.0))),
            c_xt=float(mat.get("c_xt", mat.get("C_XT", 0.05))),
            c_xc=float(mat.get("c_xc", mat.get("C_XC", 0.05))),
            c_yt=float(mat.get("c_yt", mat.get("C_YT", 0.08))),
            c_yc=float(mat.get("c_yc", mat.get("C_YC", 0.08))),
            c_sl=float(mat.get("c_sl", mat.get("C_SL", 0.06))),
        )
    return Law132Params()


def build_law132(mat: Any) -> Law132Params:
    """Build Law132Params from Material entity or dictionary."""
    return resolve(mat)


def needs_defgrad(mat: Any = None) -> bool:
    """LAW132 uses rate-dependent incremental formulation and does not need F."""
    return False


def extra_shapes(mat: Any = None, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """Define persistent history variables for LAW132 (damage, strain rate)."""
    return {
        "dmg": (nip, 6) if nip > 1 else (6,),
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
    """Dilatational sound speed for LAW132."""
    p = resolve(mat)
    eff_rho = float(rho) if (rho is not None and rho > 0.0) else p.rho0
    if eff_rho <= 0.0:
        eff_rho = 1600.0
    if is_shell:
        det = max(1.0 - (p.nu ** 2) * (p.eb / p.E), 1e-6)
        return math.sqrt(max(p.E, p.eb) / (eff_rho * det))
    return math.sqrt(max(0.0, (p.K + _FOUR_THIRDS * p.G) / eff_rho))


def _rate_multiplier(eps_dot: float, eps0: float, coeff: float) -> float:
    """Logarithmic dynamic magnification factor."""
    if coeff <= 0.0 or eps_dot <= 0.0:
        return 1.0
    ratio = max(1.0, eps_dot / max(eps0, 1e-12))
    return 1.0 + coeff * math.log(ratio)


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Vectorized plane-stress shell update for LAW132."""
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

    nu_ba = p.nu
    nu_ab = nu_ba * (p.E / p.eb)
    det = max(1.0 - nu_ba * nu_ab, 1e-8)
    q11 = p.E / det
    q22 = p.eb / det
    q12 = nu_ba * p.E / det
    q33 = p.gab

    c_shell = math.sqrt(max(p.E, p.eb) / (p.rho0 * det))
    c_out = np.full(n, c_shell, dtype=np.float64)
    s_out = np.zeros_like(s)
    ep_out = np.zeros(n, dtype=np.float64)

    for i in range(n):
        s_trial = np.copy(s[i])
        s_trial[0] += q11 * d[i, 0] + q12 * d[i, 1]
        s_trial[1] += q12 * d[i, 0] + q22 * d[i, 1]
        if s_trial.shape[0] > 2 and d.shape[1] > 2:
            s_trial[2] += q33 * d[i, 2]

        eps_dot = 0.0
        if dt > 0.0:
            eps_dot = math.sqrt(d[i, 0]**2 + d[i, 1]**2 + 0.5 * (d[i, 2]**2 if d.shape[1] > 2 else 0.0)) / dt

        # Strain-rate modified strengths
        xt_dyn = p.xt * _rate_multiplier(eps_dot, p.eps0_rate, p.c_xt)
        xc_dyn = p.xc * _rate_multiplier(eps_dot, p.eps0_rate, p.c_xc)
        yt_dyn = p.yt * _rate_multiplier(eps_dot, p.eps0_rate, p.c_yt)
        yc_dyn = p.yc * _rate_multiplier(eps_dot, p.eps0_rate, p.c_yc)
        sl_dyn = p.sl * _rate_multiplier(eps_dot, p.eps0_rate, p.c_sl)

        # Check tensile/compressive fiber and matrix criteria
        dpla = 0.0
        s11, s22 = s_trial[0], s_trial[1]
        s12 = s_trial[2] if s_trial.shape[0] > 2 else 0.0

        f_fiber = (s11 / xt_dyn if s11 >= 0.0 else -s11 / xc_dyn)
        f_matrix = (s22 / yt_dyn if s22 >= 0.0 else -s22 / yc_dyn) + (s12 / sl_dyn) ** 2

        f_max = max(f_fiber, math.sqrt(max(0.0, f_matrix)))
        if f_max > 1.0:
            scale = 1.0 / f_max
            s_trial[:3] *= scale
            dpla = (f_max - 1.0) / max(p.E, 1.0)

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


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Vectorized 3D solid update for LAW132."""
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
        s_trial[3] += p.gab * d[i, 3]
        s_trial[4] += p.gbc * d[i, 4]
        s_trial[5] += p.gca * d[i, 5]

        eps_dot = 0.0
        if dt > 0.0:
            eps_dot = math.sqrt(d[i, 0]**2 + d[i, 1]**2 + d[i, 2]**2 + 0.5 * (d[i, 3]**2 + d[i, 4]**2 + d[i, 5]**2)) / dt

        xt_dyn = p.xt * _rate_multiplier(eps_dot, p.eps0_rate, p.c_xt)
        xc_dyn = p.xc * _rate_multiplier(eps_dot, p.eps0_rate, p.c_xc)

        f_max = max(s_trial[0] / xt_dyn if s_trial[0] >= 0.0 else -s_trial[0] / xc_dyn, 0.0)
        dpla = 0.0
        if f_max > 1.0:
            scale = 1.0 / f_max
            s_trial *= scale
            dpla = (f_max - 1.0) / max(p.E, 1.0)

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
    c[3, 3] = p.gab
    c[4, 4] = p.gbc
    c[5, 5] = p.gca

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
    nu_ba = p.nu
    nu_ab = nu_ba * (p.E / p.eb)
    det = max(1.0 - nu_ba * nu_ab, 1e-8)
    q11 = p.E / det
    q22 = p.eb / det
    q12 = nu_ba * p.E / det
    q33 = p.gab

    c = np.array([
        [q11, q12, 0.0],
        [q12, q22, 0.0],
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
        for key in (132, "132", "LAW132", "RATE_MODIFIER", "MAT_LAW132"):
            MAT_PHYSICS_REGISTRY[key] = build_law132
    except Exception:
        pass


_register()
