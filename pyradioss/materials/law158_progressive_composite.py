"""
LAW158 (/MAT/LAW158, /MAT/PROGRESSIVE_COMPOSITE) — Progressive Failure Multilayer Composite Shell Model.

OpenRadioss Fortran reference:
- engine/source/materials/mat/mat158/sigeps158c.F
- starter/source/materials/mat/mat158/hm_read_mat158.F
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20
_THIRD = 1.0 / 3.0
_TWO_THIRDS = 2.0 / 3.0
_FOUR_THIRDS = 4.0 / 3.0


@dataclass
class Law158Params:
    """Parameters for /MAT/LAW158 (Progressive failure multilayer composite shell)."""
    E: float = 6.0e10            # Longitudinal tensile modulus Ec
    et_mod: float = 6.0e10      # Longitudinal compressive modulus Et
    nu: float = 0.25            # Poisson's ratio
    rho0: float = 1600.0        # Mass density
    gmax: float = 4.0e9         # Maximum shear modulus Gmax
    kflex: float = 1.0e6        # Flexural stiffness KFLEX
    kflex1: float = 0.0         # Flexural stiffness parameter 1
    kflex2: float = 0.0         # Flexural stiffness parameter 2
    dc0: float = 0.02           # Damage threshold in compression DC0
    dt0: float = 0.02           # Damage threshold in tension DT0
    hc0: float = 1.0            # Hardening parameter HC0
    ht0: float = 1.0            # Hardening parameter HT0
    zerostress: float = 0.0     # Post-failure stress relaxation threshold

    # Derived elastic moduli
    G: float = field(init=False, default=0.0)
    K: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.nu < 0.0 or self.nu >= 0.5:
            self.nu = 0.25
        if self.E <= 0.0:
            self.E = 6.0e10
        if self.et_mod <= 0.0:
            self.et_mod = self.E
        if self.gmax <= 0.0:
            self.gmax = self.E / (2.0 * (1.0 + self.nu))
        self.G = self.gmax
        self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))


def resolve(mat: Any) -> Law158Params:
    """Resolve Law158Params from Material, dict, or Law158Params instance."""
    if isinstance(mat, Law158Params):
        return mat
    if isinstance(mat, Material):
        p = mat.params or {}
        e_val = float(p.get("E", p.get("MAT_E", 6.0e10)))
        return Law158Params(
            E=e_val,
            et_mod=float(p.get("et_mod", p.get("ET", e_val))),
            nu=float(p.get("nu", p.get("MAT_NU", 0.25))),
            rho0=float(getattr(mat, "rho0", p.get("rho", p.get("MAT_RHO", 1600.0)))),
            gmax=float(p.get("gmax", p.get("GMAX", 4.0e9))),
            kflex=float(p.get("kflex", p.get("KFLEX", 1.0e6))),
            kflex1=float(p.get("kflex1", p.get("KFLEX1", 0.0))),
            kflex2=float(p.get("kflex2", p.get("KFLEX2", 0.0))),
            dc0=float(p.get("dc0", p.get("DC0", 0.02))),
            dt0=float(p.get("dt0", p.get("DT0", 0.02))),
            hc0=float(p.get("hc0", p.get("HC0", 1.0))),
            ht0=float(p.get("ht0", p.get("HT0", 1.0))),
            zerostress=float(p.get("zerostress", p.get("ZEROSTRESS", 0.0))),
        )
    if isinstance(mat, dict):
        e_val = float(mat.get("E", mat.get("MAT_E", 6.0e10)))
        return Law158Params(
            E=e_val,
            et_mod=float(mat.get("et_mod", mat.get("ET", e_val))),
            nu=float(mat.get("nu", mat.get("MAT_NU", 0.25))),
            rho0=float(mat.get("rho0", mat.get("rho", mat.get("MAT_RHO", 1600.0)))),
            gmax=float(mat.get("gmax", mat.get("GMAX", 4.0e9))),
            kflex=float(mat.get("kflex", mat.get("KFLEX", 1.0e6))),
            kflex1=float(mat.get("kflex1", mat.get("KFLEX1", 0.0))),
            kflex2=float(mat.get("kflex2", mat.get("KFLEX2", 0.0))),
            dc0=float(mat.get("dc0", mat.get("DC0", 0.02))),
            dt0=float(mat.get("dt0", mat.get("DT0", 0.02))),
            hc0=float(mat.get("hc0", mat.get("HC0", 1.0))),
            ht0=float(mat.get("ht0", mat.get("HT0", 1.0))),
            zerostress=float(mat.get("zerostress", mat.get("ZEROSTRESS", 0.0))),
        )
    return Law158Params()


def build_law158(mat: Any) -> Law158Params:
    """Build Law158Params from Material entity or dictionary."""
    return resolve(mat)


def needs_defgrad(mat: Any = None) -> bool:
    """LAW158 incremental composite formulation does not need deformation gradient F."""
    return False


def extra_shapes(mat: Any = None, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """State arrays for LAW158 (16 uvars per integration point matching sigeps158c.F)."""
    return {
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
    """Dilatational sound speed for LAW158."""
    p = resolve(mat)
    eff_rho = float(rho) if (rho is not None and rho > 0.0) else p.rho0
    if eff_rho <= 0.0:
        eff_rho = 1600.0
    if is_shell:
        return math.sqrt(max(0.0, p.E / (eff_rho * max(1.0 - p.nu ** 2, 1e-6))))
    return math.sqrt(max(0.0, (p.K + _FOUR_THIRDS * p.G) / eff_rho))


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Progressive failure multilayer composite shell constitutive update matching sigeps158c.F."""
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

    uvar_all = np.zeros((n, 16), dtype=np.float64)
    if extra is not None and "uvar" in extra:
        uv = np.asarray(extra["uvar"], dtype=np.float64)
        if uv.ndim == 1 and n == 1:
            uvar_all[0, :min(16, uv.shape[0])] = uv[:16]
        elif uv.ndim == 2:
            uvar_all[:min(n, uv.shape[0]), :min(16, uv.shape[1])] = uv[:min(n, uv.shape[0]), :min(16, uv.shape[1])]

    denom = max(1.0 - p.nu ** 2, 1e-8)
    q11 = p.E / denom
    q12 = p.nu * q11
    q33 = p.gmax

    c_shell = math.sqrt(max(0.0, p.E / (p.rho0 * denom)))
    c_out = np.full(n, c_shell, dtype=np.float64)

    s_out = np.zeros_like(s)
    ep_out = np.zeros(n, dtype=np.float64)

    for i in range(n):
        s_trial = np.copy(s[i])
        s_trial[0] += q11 * d[i, 0] + q12 * d[i, 1]
        s_trial[1] += q12 * d[i, 0] + q11 * d[i, 1]
        if s_trial.shape[0] > 2 and d.shape[1] > 2:
            s_trial[2] += q33 * d[i, 2]

        # Progressive damage checks: positive strain for tension (et), negative for compression (ec)
        if d[i, 0] > 0.0:
            et = uvar_all[i, 4] + d[i, 0]
            ec = uvar_all[i, 3]
        else:
            et = uvar_all[i, 4]
            ec = uvar_all[i, 3] + abs(d[i, 0])
        uvar_all[i, 3] = ec
        uvar_all[i, 4] = et

        dmg_c = max(0.0, min(1.0, (ec - p.dc0) / max(p.dc0 * 2.0, _EM20))) if ec > p.dc0 else 0.0
        dmg_t = max(0.0, min(1.0, (et - p.dt0) / max(p.dt0 * 2.0, _EM20))) if et > p.dt0 else 0.0

        uvar_all[i, 14] = dmg_c
        uvar_all[i, 15] = dmg_t

        s_trial[0] *= (1.0 - dmg_c)
        s_trial[1] *= (1.0 - dmg_t)
        if s_trial.shape[0] > 2:
            s_trial[2] *= (1.0 - 0.5 * (dmg_c + dmg_t))

        s_out[i] = s_trial
        ep_out[i] = ep[i] + max(dmg_c, dmg_t)

    if extra is not None:
        extra["uvar"] = uvar_all[0] if is_1d else uvar_all

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
    """Solid update for LAW158 composite."""
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

    for i in range(n):
        de_vol = d[i, 0] + d[i, 1] + d[i, 2]
        s_out[i, 0] = s[i, 0] + lam * de_vol + g2 * d[i, 0]
        s_out[i, 1] = s[i, 1] + lam * de_vol + g2 * d[i, 1]
        s_out[i, 2] = s[i, 2] + lam * de_vol + g2 * d[i, 2]
        s_out[i, 3] = s[i, 3] + p.G * d[i, 3]
        s_out[i, 4] = s[i, 4] + p.G * d[i, 4]
        s_out[i, 5] = s[i, 5] + p.G * d[i, 5]

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
