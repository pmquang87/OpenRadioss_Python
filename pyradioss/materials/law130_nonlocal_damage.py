"""
LAW130 (/MAT/LAW130, /MAT/NONLOCAL_DAMAGE) — Nonlocal Continuum Damage Regularization Model.

OpenRadioss Fortran reference:
- engine/source/materials/mat/mat130/sigeps130.F90
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20
_ZEP999 = 0.999
_THIRD = 1.0 / 3.0
_TWO_THIRDS = 2.0 / 3.0
_FOUR_THIRDS = 4.0 / 3.0


@dataclass
class Law130Params:
    """Parameters for /MAT/LAW130 (Nonlocal continuum damage regularization)."""
    E: float = 2.0e10            # Initial elastic Young's modulus
    nu: float = 0.2             # Poisson's ratio
    rho0: float = 1500.0        # Mass density
    lc: float = 0.01            # Nonlocal characteristic length scale Lc
    eps_d0: float = 0.001       # Damage initiation threshold strain kappa0
    eps_df: float = 0.05        # Critical / ultimate failure strain
    alpha_d: float = 0.95       # Residual stress asymptote parameter alpha
    beta_d: float = 50.0        # Softening exponent beta
    d_type: int = 1             # Softening formulation: 1=exponential, 2=linear

    # Derived elastic moduli
    G: float = field(init=False, default=0.0)
    K: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.nu < 0.0 or self.nu >= 0.5:
            self.nu = 0.2
        if self.E <= 0.0:
            self.E = 2.0e10
        if self.eps_d0 <= 0.0:
            self.eps_d0 = 0.001
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))


def resolve(mat: Any) -> Law130Params:
    """Resolve Law130Params from Material, dict, or Law130Params instance."""
    if isinstance(mat, Law130Params):
        return mat
    if isinstance(mat, Material):
        p = mat.params or {}
        return Law130Params(
            E=float(p.get("E", p.get("MAT_E", 2.0e10))),
            nu=float(p.get("nu", p.get("MAT_NU", 0.2))),
            rho0=float(getattr(mat, "rho0", p.get("rho", p.get("MAT_RHO", 1500.0)))),
            lc=float(p.get("lc", p.get("LC", p.get("L_CHAR", 0.01)))),
            eps_d0=float(p.get("eps_d0", p.get("EPS_D0", p.get("KAPPA0", 0.001)))),
            eps_df=float(p.get("eps_df", p.get("EPS_DF", 0.05))),
            alpha_d=float(p.get("alpha_d", p.get("ALPHA_D", 0.95))),
            beta_d=float(p.get("beta_d", p.get("BETA_D", 50.0))),
            d_type=int(p.get("d_type", p.get("D_TYPE", 1))),
        )
    if isinstance(mat, dict):
        return Law130Params(
            E=float(mat.get("E", mat.get("MAT_E", 2.0e10))),
            nu=float(mat.get("nu", mat.get("MAT_NU", 0.2))),
            rho0=float(mat.get("rho0", mat.get("rho", mat.get("MAT_RHO", 1500.0)))),
            lc=float(mat.get("lc", mat.get("LC", mat.get("L_CHAR", 0.01)))),
            eps_d0=float(mat.get("eps_d0", mat.get("EPS_D0", mat.get("KAPPA0", 0.001)))),
            eps_df=float(mat.get("eps_df", mat.get("EPS_DF", 0.05))),
            alpha_d=float(mat.get("alpha_d", mat.get("ALPHA_D", 0.95))),
            beta_d=float(mat.get("beta_d", mat.get("BETA_D", 50.0))),
            d_type=int(mat.get("d_type", mat.get("D_TYPE", 1))),
        )
    return Law130Params()


def build_law130(mat: Any) -> Law130Params:
    """Build Law130Params from Material entity or dictionary."""
    return resolve(mat)


def needs_defgrad(mat: Any = None) -> bool:
    """LAW130 is small-strain incremental continuum damage and does not need F."""
    return False


def extra_shapes(mat: Any = None, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """State arrays for LAW130 (damage, historical maximum equivalent strain)."""
    return {
        "dmg": (nip, 1) if nip > 1 else (1,),
        "kappa": (nip, 1) if nip > 1 else (1,),
        "uvar": (nip, 8) if nip > 1 else (8,),
    }


def sound_speed(
    mat: Any,
    eps: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    rho: Optional[float] = None,
    is_shell: bool = False,
    **kwargs: Any,
) -> float:
    """Dilatational sound speed for LAW130."""
    p = resolve(mat)
    eff_rho = float(rho) if (rho is not None and rho > 0.0) else p.rho0
    if eff_rho <= 0.0:
        eff_rho = 1500.0
    if is_shell:
        return math.sqrt(max(0.0, p.E / (eff_rho * max(1.0 - p.nu ** 2, 1e-6))))
    return math.sqrt(max(0.0, (p.K + _FOUR_THIRDS * p.G) / eff_rho))


def _compute_damage(p: Law130Params, kappa: float) -> float:
    """Compute scalar damage D in [0, 0.999]."""
    if kappa <= p.eps_d0:
        return 0.0
    if p.d_type == 2:
        # Linear softening
        d = (kappa - p.eps_d0) / max(p.eps_df - p.eps_d0, _EM20)
    else:
        # Exponential softening
        ratio = p.eps_d0 / kappa
        d = 1.0 - ratio * (1.0 - p.alpha_d + p.alpha_d * math.exp(-p.beta_d * (kappa - p.eps_d0)))
    return min(max(0.0, d), _ZEP999)


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Vectorized solid constitutive update with nonlocal continuum damage."""
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

    dmg_arr = np.zeros(n, dtype=np.float64)
    kappa_arr = np.full(n, p.eps_d0, dtype=np.float64)
    if extra is not None:
        if "dmg" in extra:
            dmg_arr = np.asarray(extra["dmg"], dtype=np.float64).flatten()[:n]
        if "kappa" in extra:
            kappa_arr = np.asarray(extra["kappa"], dtype=np.float64).flatten()[:n]

    lam = p.K - _TWO_THIRDS * p.G
    g2 = 2.0 * p.G
    soundsp = math.sqrt(max(0.0, (p.K + _FOUR_THIRDS * p.G) / p.rho0))
    c_out = np.full(n, soundsp, dtype=np.float64)

    s_out = np.zeros_like(s)

    for i in range(n):
        # Elastic strain increment
        de_vol = d[i, 0] + d[i, 1] + d[i, 2]
        # Undamaged stress increment
        ds_undamaged = np.array([
            lam * de_vol + g2 * d[i, 0],
            lam * de_vol + g2 * d[i, 1],
            lam * de_vol + g2 * d[i, 2],
            p.G * d[i, 3],
            p.G * d[i, 4],
            p.G * d[i, 5],
        ], dtype=np.float64)

        # Equivalent strain measure (positive tensile strain)
        e_pos_sq = max(0.0, d[i, 0])**2 + max(0.0, d[i, 1])**2 + max(0.0, d[i, 2])**2 + 0.5 * (d[i, 3]**2 + d[i, 4]**2 + d[i, 5]**2)
        eps_eq = math.sqrt(max(0.0, e_pos_sq))

        # Update historical threshold kappa
        kappa_new = max(kappa_arr[i], eps_eq + ep[i])
        kappa_arr[i] = kappa_new

        # Compute damage
        d_val = _compute_damage(p, kappa_new)
        dmg_arr[i] = max(dmg_arr[i], d_val)

        # Degraded stress
        s_eff = (s[i] + ds_undamaged) * (1.0 - dmg_arr[i])
        s_out[i] = s_eff
        ep[i] += max(0.0, eps_eq - p.eps_d0)

    if extra is not None:
        extra["dmg"] = dmg_arr[0] if is_1d else dmg_arr
        extra["kappa"] = kappa_arr[0] if is_1d else kappa_arr

    if is_1d:
        res_sig = s_out[0]
        res_ep = float(ep[0])
        res_c = float(c_out[0])
    else:
        res_sig = s_out
        res_ep = ep
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
    """Plane-stress shell update for /MAT/LAW130."""
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
    c_shell = math.sqrt(max(0.0, p.E / (p.rho0 * denom)))
    c_out = np.full(n, c_shell, dtype=np.float64)

    dmg_arr = np.zeros(n, dtype=np.float64)
    kappa_arr = np.full(n, p.eps_d0, dtype=np.float64)
    if extra is not None:
        if "dmg" in extra:
            dmg_arr = np.asarray(extra["dmg"], dtype=np.float64).flatten()[:n]
        if "kappa" in extra:
            kappa_arr = np.asarray(extra["kappa"], dtype=np.float64).flatten()[:n]

    for i in range(n):
        s_trial = np.copy(s[i])
        s_trial[0] += q11 * d[i, 0] + q12 * d[i, 1]
        s_trial[1] += q12 * d[i, 0] + q11 * d[i, 1]
        if s_trial.shape[0] > 2 and d.shape[1] > 2:
            s_trial[2] += q33 * d[i, 2]

        eps_eq = math.sqrt(max(0.0, max(0.0, d[i, 0])**2 + max(0.0, d[i, 1])**2 + 0.5 * (d[i, 2]**2 if d.shape[1] > 2 else 0.0)))
        kappa_new = max(kappa_arr[i], eps_eq + ep[i])
        kappa_arr[i] = kappa_new

        d_val = _compute_damage(p, kappa_new)
        dmg_arr[i] = max(dmg_arr[i], d_val)

        s_out[i] = s_trial * (1.0 - dmg_arr[i])
        ep[i] += max(0.0, eps_eq - p.eps_d0)

    if extra is not None:
        extra["dmg"] = dmg_arr[0] if is_1d else dmg_arr
        extra["kappa"] = kappa_arr[0] if is_1d else kappa_arr

    if is_1d:
        res_sig = s_out[0]
        res_ep = float(ep[0])
        res_c = float(c_out[0])
    else:
        res_sig = s_out
        res_ep = ep
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
        for key in (130, "130", "LAW130", "NONLOCAL_DAMAGE", "MAT_LAW130"):
            MAT_PHYSICS_REGISTRY[key] = build_law130
    except Exception:
        pass


_register()
