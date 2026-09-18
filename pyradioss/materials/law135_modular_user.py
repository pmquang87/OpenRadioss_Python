"""
LAW135 (/MAT/LAW135, /MAT/MODULAR_USER, /MAT/WTM_STM) — Modular User Plasticity Framework.

OpenRadioss Fortran reference:
- starter/source/materials/mat/mat135/hm_read_mat135.F90
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20
_EM14 = 1.0e-14
_THIRD = 1.0 / 3.0
_TWO_THIRDS = 2.0 / 3.0
_FOUR_THIRDS = 4.0 / 3.0


@dataclass
class Law135Params:
    """Parameters for /MAT/LAW135 (Modular user plasticity framework)."""
    E: float = 2.0e11            # Young's modulus
    nu: float = 0.3             # Poisson's ratio
    rho0: float = 7850.0        # Mass density
    sig0: float = 3.0e8         # Initial yield stress
    hp: float = 0.0             # Linear hardening modulus
    tkr: float = 1.0            # Translational stiffness scale factor TKR
    rkr: float = 1.0            # Rotational stiffness scale factor RKR
    # 6-DOF constraint/yield flags or thresholds
    tr: int = 0
    ts: int = 0
    tt: int = 0
    rr: int = 0
    rs: int = 0
    rt: int = 0

    # Derived elastic moduli
    G: float = field(init=False, default=0.0)
    K: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.nu < 0.0 or self.nu >= 0.5:
            self.nu = 0.3
        if self.E <= 0.0:
            self.E = 2.0e11
        if self.tkr <= 0.0:
            self.tkr = 1.0
        if self.rkr <= 0.0:
            self.rkr = 1.0
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))


def resolve(mat: Any) -> Law135Params:
    """Resolve Law135Params from Material, dict, or Law135Params instance."""
    if isinstance(mat, Law135Params):
        return mat
    if isinstance(mat, Material):
        p = mat.params or {}
        e_val = float(p.get("E", p.get("MAT_E", 2.0e11)))
        return Law135Params(
            E=e_val,
            nu=float(p.get("nu", p.get("MAT_NU", 0.3))),
            rho0=float(getattr(mat, "rho0", p.get("rho", p.get("RHO", 7850.0)))),
            sig0=float(p.get("sig0", p.get("SIG0", p.get("MAT_SIGY", 3.0e8)))),
            hp=float(p.get("hp", p.get("HP", 0.0))),
            tkr=float(p.get("tkr", p.get("LSDYNA_TKR", 1.0))),
            rkr=float(p.get("rkr", p.get("LSDYNA_RKR", 1.0))),
            tr=int(p.get("tr", p.get("LSD_TR", 0))),
            ts=int(p.get("ts", p.get("LSD_TS", 0))),
            tt=int(p.get("tt", p.get("LSD_TT", 0))),
            rr=int(p.get("rr", p.get("LSD_RR", 0))),
            rs=int(p.get("rs", p.get("LSD_RS", 0))),
            rt=int(p.get("rt", p.get("LSD_RT", 0))),
        )
    if isinstance(mat, dict):
        e_val = float(mat.get("E", mat.get("MAT_E", 2.0e11)))
        return Law135Params(
            E=e_val,
            nu=float(mat.get("nu", mat.get("MAT_NU", 0.3))),
            rho0=float(mat.get("rho0", mat.get("rho", mat.get("RHO", 7850.0)))),
            sig0=float(mat.get("sig0", mat.get("SIG0", mat.get("MAT_SIGY", 3.0e8)))),
            hp=float(mat.get("hp", mat.get("HP", 0.0))),
            tkr=float(mat.get("tkr", mat.get("LSDYNA_TKR", 1.0))),
            rkr=float(mat.get("rkr", mat.get("LSDYNA_RKR", 1.0))),
            tr=int(mat.get("tr", mat.get("LSD_TR", 0))),
            ts=int(mat.get("ts", mat.get("LSD_TS", 0))),
            tt=int(mat.get("tt", mat.get("LSD_TT", 0))),
            rr=int(mat.get("rr", mat.get("LSD_RR", 0))),
            rs=int(mat.get("rs", mat.get("LSD_RS", 0))),
            rt=int(mat.get("rt", mat.get("LSD_RT", 0))),
        )
    return Law135Params()


def build_law135(mat: Any) -> Law135Params:
    """Build Law135Params from Material entity or dictionary."""
    return resolve(mat)


def needs_defgrad(mat: Any = None) -> bool:
    """LAW135 incremental formulation does not require deformation gradient F."""
    return False


def extra_shapes(mat: Any = None, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """Define persistent internal variables for LAW135 (39 uvars matching hm_read_mat135.F90)."""
    return {
        "uvar": (nip, 39) if nip > 1 else (39,),
    }


def sound_speed(
    mat: Any,
    eps: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    rho: Optional[float] = None,
    is_shell: bool = False,
    **kwargs: Any,
) -> float:
    """Dilatational sound speed for LAW135."""
    p = resolve(mat)
    eff_rho = float(rho) if (rho is not None and rho > 0.0) else p.rho0
    if eff_rho <= 0.0:
        eff_rho = 7850.0
    if is_shell:
        return math.sqrt(max(0.0, p.E / (eff_rho * max(1.0 - p.nu ** 2, 1e-6))))
    return math.sqrt(max(0.0, (p.K + _FOUR_THIRDS * p.G) / eff_rho))


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """3D solid update for modular user plasticity model LAW135."""
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

    uvar_all = np.zeros((n, 39), dtype=np.float64)
    if extra is not None and "uvar" in extra:
        uv = np.asarray(extra["uvar"], dtype=np.float64)
        if uv.ndim == 1 and n == 1:
            uvar_all[0, :min(39, uv.shape[0])] = uv[:39]
        elif uv.ndim == 2:
            uvar_all[:min(n, uv.shape[0]), :min(39, uv.shape[1])] = uv[:min(n, uv.shape[0]), :min(39, uv.shape[1])]

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

        p_mean = _THIRD * (s_trial[0] + s_trial[1] + s_trial[2])
        j2 = 0.5 * ((s_trial[0] - p_mean)**2 + (s_trial[1] - p_mean)**2 + (s_trial[2] - p_mean)**2) + s_trial[3]**2 + s_trial[4]**2 + s_trial[5]**2
        sig_vm = math.sqrt(3.0 * max(0.0, j2))

        yld = p.sig0 + p.hp * ep[i]
        dpla = 0.0
        if sig_vm > yld:
            ratio = yld / max(sig_vm, _EM14)
            s_trial[0] = p_mean + ratio * (s_trial[0] - p_mean)
            s_trial[1] = p_mean + ratio * (s_trial[1] - p_mean)
            s_trial[2] = p_mean + ratio * (s_trial[2] - p_mean)
            s_trial[3] *= ratio
            s_trial[4] *= ratio
            s_trial[5] *= ratio
            dpla = (1.0 - ratio) * sig_vm / max(3.0 * p.G + p.hp, _EM20)

        s_out[i] = s_trial
        ep_out[i] = ep[i] + dpla

        # Store deformation history in uvars
        uvar_all[i, 0] = ep_out[i]
        uvar_all[i, 1] = sig_vm

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


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Plane-stress shell update for LAW135."""
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
        yld = p.sig0 + p.hp * ep[i]

        dpla = 0.0
        if sig_vm > yld:
            ratio = yld / max(sig_vm, _EM14)
            s_trial[:3] *= ratio
            dpla = (1.0 - ratio) * sig_vm / max(3.0 * p.G + p.hp, _EM20)

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
        for key in (135, "135", "LAW135", "MODULAR_USER", "WTM_STM", "MAT_LAW135"):
            MAT_PHYSICS_REGISTRY[key] = build_law135
    except Exception:
        pass


_register()
