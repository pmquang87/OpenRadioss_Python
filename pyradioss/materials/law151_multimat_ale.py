"""
LAW151 (/MAT/LAW151, /MAT/MULTIMAT_ALE, /MAT/MULTI_FVM) — Multi-Material ALE / FVM Mixture Law.

OpenRadioss Fortran reference:
- starter/source/materials/mat/mat151/hm_read_mat151.F
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20
_THIRD = 1.0 / 3.0
_TWO_THIRDS = 2.0 / 3.0
_FOUR_THIRDS = 4.0 / 3.0


@dataclass
class Law151Params:
    """Parameters for /MAT/LAW151 (Multi-material ALE/FVM material)."""
    E: float = 1.0e9            # Homogenized mixture Young's modulus
    nu: float = 0.3            # Homogenized mixture Poisson's ratio
    rho0: float = 1000.0       # Initial mixture density
    nbmat: int = 2             # Number of phases in the multi-material (up to 20)
    vfrac: List[float] = field(default_factory=lambda: [0.5, 0.5])  # Nominal phase volume fractions
    rho_phases: Optional[List[float]] = None # Phase densities
    bulk_phases: Optional[List[float]] = None # Phase bulk moduli

    # Derived mixture moduli
    G: float = field(init=False, default=0.0)
    K: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.nu < 0.0 or self.nu >= 0.5:
            self.nu = 0.3
        if self.nbmat <= 0:
            self.nbmat = len(self.vfrac) if self.vfrac else 1

        # Normalize volume fractions
        total_v = sum(self.vfrac) if self.vfrac else 1.0
        if total_v > 0.0:
            self.vfrac = [v / total_v for v in self.vfrac]
        else:
            self.vfrac = [1.0 / max(1, self.nbmat)] * self.nbmat

        # Homogenized mixture density
        if self.rho_phases is not None and len(self.rho_phases) == len(self.vfrac):
            self.rho0 = sum(vf * r for vf, r in zip(self.vfrac, self.rho_phases))
        elif self.rho0 <= 0.0:
            self.rho0 = 1000.0

        # Homogenized mixture bulk modulus (isobaric Reuss harmonic mean)
        if self.bulk_phases is not None and len(self.bulk_phases) == len(self.vfrac):
            inv_k = sum(vf / max(k, 1e-6) for vf, k in zip(self.vfrac, self.bulk_phases))
            self.K = 1.0 / max(inv_k, 1e-12)
            self.G = (3.0 * self.K * (1.0 - 2.0 * self.nu)) / (2.0 * (1.0 + self.nu))
            self.E = 9.0 * self.K * self.G / (3.0 * self.K + self.G)
        else:
            self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))
            self.G = self.E / (2.0 * (1.0 + self.nu))


def resolve(mat: Any) -> Law151Params:
    """Resolve Law151Params from Material, dict, or Law151Params instance."""
    if isinstance(mat, Law151Params):
        return mat
    if isinstance(mat, Material):
        p = mat.params or {}
        nb = int(p.get("nbmat", p.get("NBMAT", p.get("NIP", 2))))
        vf = p.get("vfrac", p.get("VFRAC", [0.5, 0.5]))
        if not isinstance(vf, list):
            vf = list(vf) if hasattr(vf, "__iter__") else [float(vf)]
        rhos = p.get("rho_phases", p.get("RHO_PHASES", None))
        if rhos is not None and not isinstance(rhos, list):
            rhos = list(rhos) if hasattr(rhos, "__iter__") else [float(rhos)]
        bulks = p.get("bulk_phases", p.get("BULK_PHASES", None))
        if bulks is not None and not isinstance(bulks, list):
            bulks = list(bulks) if hasattr(bulks, "__iter__") else [float(bulks)]

        return Law151Params(
            E=float(p.get("E", p.get("MAT_E", 1.0e9))),
            nu=float(p.get("nu", p.get("MAT_NU", 0.3))),
            rho0=float(getattr(mat, "rho0", p.get("rho", p.get("MAT_RHO", 1000.0)))),
            nbmat=nb,
            vfrac=vf,
            rho_phases=rhos,
            bulk_phases=bulks,
        )
    if isinstance(mat, dict):
        nb = int(mat.get("nbmat", mat.get("NBMAT", mat.get("NIP", 2))))
        vf = mat.get("vfrac", mat.get("VFRAC", [0.5, 0.5]))
        if not isinstance(vf, list):
            vf = list(vf) if hasattr(vf, "__iter__") else [float(vf)]
        rhos = mat.get("rho_phases", mat.get("RHO_PHASES", None))
        if rhos is not None and not isinstance(rhos, list):
            rhos = list(rhos) if hasattr(rhos, "__iter__") else [float(rhos)]
        bulks = mat.get("bulk_phases", mat.get("BULK_PHASES", None))
        if bulks is not None and not isinstance(bulks, list):
            bulks = list(bulks) if hasattr(bulks, "__iter__") else [float(bulks)]

        return Law151Params(
            E=float(mat.get("E", mat.get("MAT_E", 1.0e9))),
            nu=float(mat.get("nu", mat.get("MAT_NU", 0.3))),
            rho0=float(mat.get("rho0", mat.get("rho", mat.get("MAT_RHO", 1000.0)))),
            nbmat=nb,
            vfrac=vf,
            rho_phases=rhos,
            bulk_phases=bulks,
        )
    return Law151Params()


def build_law151(mat: Any) -> Law151Params:
    """Build Law151Params from Material entity or dictionary."""
    return resolve(mat)


def needs_defgrad(mat: Any = None) -> bool:
    """Multi-material ALE uses Eulerian/ALE volume fraction advection and does not need F."""
    return False


def extra_shapes(mat: Any = None, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """Persistent phase volume fraction state arrays for LAW151."""
    p = resolve(mat) if mat else Law151Params()
    return {
        "vfrac": (nip, p.nbmat) if nip > 1 else (p.nbmat,),
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
    """Dilatational sound speed for multi-material mixture LAW151."""
    p = resolve(mat)
    eff_rho = float(rho) if (rho is not None and rho > 0.0) else p.rho0
    if eff_rho <= 0.0:
        eff_rho = 1000.0
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
    """Multi-material ALE homogenized solid constitutive update."""
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


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Multi-material homogenized plane-stress shell update."""
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

    for i in range(n):
        s_out[i, 0] = s[i, 0] + q11 * d[i, 0] + q12 * d[i, 1]
        s_out[i, 1] = s[i, 1] + q12 * d[i, 0] + q11 * d[i, 1]
        if s.shape[1] > 2 and d.shape[1] > 2:
            s_out[i, 2] = s[i, 2] + q33 * d[i, 2]

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
