"""
LAW133 (/MAT/LAW133, /MAT/MICROPLANE_CONCRETE) — Microplane Formulation for Concrete and Quasi-Brittle Materials.

OpenRadioss Fortran reference:
- engine/source/materials/mat/mat133/sigeps133.F90
- starter/source/materials/mat/mat133/hm_read_mat133.F90
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
class Law133Params:
    """Parameters for /MAT/LAW133 (Microplane concrete and quasi-brittle model)."""
    E: float = 3.0e10            # Young's modulus
    nu: float = 0.2             # Poisson's ratio
    rho0: float = 2400.0        # Mass density
    pmin: float = -1.0e6        # Minimum / fracture pressure cutoff (p <= pmin => sigy = 0)
    fc: float = 3.0e7           # Compressive strength
    ft: float = 3.0e6           # Tensile strength
    a0: float = 3.0e6           # Yield stress intercept (at P=0)
    a1: float = 1.2             # Linear pressure sensitivity
    a2: float = 0.0             # Quadratic pressure sensitivity

    # Derived elastic moduli
    G: float = field(init=False, default=0.0)
    K: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.nu < 0.0 or self.nu >= 0.5:
            self.nu = 0.2
        if self.E <= 0.0:
            self.E = 3.0e10
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))


def resolve(mat: Any) -> Law133Params:
    """Resolve Law133Params from Material, dict, or Law133Params instance."""
    if isinstance(mat, Law133Params):
        return mat
    if isinstance(mat, Material):
        p = mat.params or {}
        return Law133Params(
            E=float(p.get("E", p.get("MAT_E", 3.0e10))),
            nu=float(p.get("nu", p.get("MAT_NU", 0.2))),
            rho0=float(getattr(mat, "rho0", p.get("rho", p.get("MAT_RHO", 2400.0)))),
            pmin=float(p.get("pmin", p.get("PMIN", -1.0e6))),
            fc=float(p.get("fc", p.get("FC", 3.0e7))),
            ft=float(p.get("ft", p.get("FT", 3.0e6))),
            a0=float(p.get("a0", p.get("A0", 3.0e6))),
            a1=float(p.get("a1", p.get("A1", 1.2))),
            a2=float(p.get("a2", p.get("A2", 0.0))),
        )
    if isinstance(mat, dict):
        return Law133Params(
            E=float(mat.get("E", mat.get("MAT_E", 3.0e10))),
            nu=float(mat.get("nu", mat.get("MAT_NU", 0.2))),
            rho0=float(mat.get("rho0", mat.get("rho", mat.get("MAT_RHO", 2400.0)))),
            pmin=float(mat.get("pmin", mat.get("PMIN", -1.0e6))),
            fc=float(mat.get("fc", mat.get("FC", 3.0e7))),
            ft=float(mat.get("ft", mat.get("FT", 3.0e6))),
            a0=float(mat.get("a0", mat.get("A0", 3.0e6))),
            a1=float(mat.get("a1", mat.get("A1", 1.2))),
            a2=float(mat.get("a2", mat.get("A2", 0.0))),
        )
    return Law133Params()


def build_law133(mat: Any) -> Law133Params:
    """Build Law133Params from Material entity or dictionary."""
    return resolve(mat)


def needs_defgrad(mat: Any = None) -> bool:
    """LAW133 uses strain increments and does not need F."""
    return False


def extra_shapes(mat: Any = None, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """Define persistent history variables for LAW133."""
    return {
        "uvar": (nip, 10) if nip > 1 else (10,),
    }


def sound_speed(
    mat: Any,
    eps: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    rho: Optional[float] = None,
    is_shell: bool = False,
    **kwargs: Any,
) -> float:
    """Dilatational sound speed for LAW133."""
    p = resolve(mat)
    eff_rho = float(rho) if (rho is not None and rho > 0.0) else p.rho0
    if eff_rho <= 0.0:
        eff_rho = 2400.0
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
    """Solid constitutive update matching upstream sigeps133.F90."""
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
    soundsp = math.sqrt(max(0.0, (p.K + _FOUR_THIRDS * p.G) / p.rho0))
    c_out = np.full(n, soundsp, dtype=np.float64)

    for i in range(n):
        # Hydrostatic and deviatoric decomposition
        dav = (d[i, 0] + d[i, 1] + d[i, 2]) * _THIRD
        p_old = -_THIRD * (s[i, 0] + s[i, 1] + s[i, 2])
        p_new = p_old - p.K * (d[i, 0] + d[i, 1] + d[i, 2])

        # Deviatoric stresses
        s_dev = np.array([
            s[i, 0] + p_old + 2.0 * p.G * (d[i, 0] - dav),
            s[i, 1] + p_old + 2.0 * p.G * (d[i, 1] - dav),
            s[i, 2] + p_old + 2.0 * p.G * (d[i, 2] - dav),
            s[i, 3] + p.G * d[i, 3],
            s[i, 4] + p.G * d[i, 4],
            s[i, 5] + p.G * d[i, 5],
        ], dtype=np.float64)

        j2 = 0.5 * (s_dev[0]**2 + s_dev[1]**2 + s_dev[2]**2) + s_dev[3]**2 + s_dev[4]**2 + s_dev[5]**2
        sig_vm = math.sqrt(3.0 * max(0.0, j2))

        # Pressure-dependent yield stress Y(P)
        if p_new <= p.pmin:
            sig_y = 0.0
        else:
            sig_y = max(0.0, p.a0 + p.a1 * p_new + p.a2 * (p_new ** 2))

        dpla = 0.0
        if sig_y == 0.0:
            s_dev[:] = 0.0
        elif sig_vm > sig_y:
            ratio = sig_y / max(sig_vm, _EM14)
            s_dev *= ratio
            dpla = (1.0 - ratio) * sig_vm / max(3.0 * p.G, _EM20)

        # Recombine deviatoric and hydrostatic components
        s_final = np.copy(s_dev)
        s_final[0] -= p_new
        s_final[1] -= p_new
        s_final[2] -= p_new

        s_out[i] = s_final
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


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Plane-stress shell update for /MAT/LAW133."""
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

        p_eff = -_THIRD * (s_trial[0] + s_trial[1])
        if p_eff <= p.pmin:
            sig_y = 0.0
        else:
            sig_y = max(0.0, p.a0 + p.a1 * p_eff + p.a2 * (p_eff ** 2))

        sxy = s_trial[2] if s_trial.shape[0] > 2 else 0.0
        sig_vm = math.sqrt(max(0.0, s_trial[0]**2 + s_trial[1]**2 - s_trial[0]*s_trial[1] + 3.0 * (sxy**2)))

        dpla = 0.0
        if sig_y == 0.0:
            s_trial[:3] = 0.0
        elif sig_vm > sig_y:
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
    """Elastic 6x6 tangent matrix for LAW133."""
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
