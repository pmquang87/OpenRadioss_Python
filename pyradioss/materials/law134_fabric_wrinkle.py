"""
LAW134 (/MAT/LAW134, /MAT/FABRIC_WRINKLE) — Orthotropic Fabric with Wrinkling Kinematics and Viscoelastic Damping.

OpenRadioss Fortran reference:
- engine/source/materials/mat/mat134/sigeps134s.F90
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20
_EM10 = 1.0e-10
_THIRD = 1.0 / 3.0
_TWO_THIRDS = 2.0 / 3.0
_FOUR_THIRDS = 4.0 / 3.0


@dataclass
class Law134Params:
    """Parameters for /MAT/LAW134 (Orthotropic fabric with wrinkling kinematics)."""
    E: float = 1.0e9            # Reference elastic modulus
    nu: float = 0.3            # Poisson's ratio
    rho0: float = 1000.0       # Mass density
    e1: float = 1.0e9          # Modulus parameter e1 (E1 = e1 * (rho0/rho)^(-n1))
    n1: float = 0.0            # Modulus density exponent n1
    e2: float = 1.0e8          # Maxwell viscoelastic modulus e2
    v2: float = 1.0e7          # Viscosity coefficient v2
    n2: float = 0.0            # Viscosity strain exponent n2
    wrinkle_flag: int = 1      # 1=apply wrinkling (zero compression), 0=standard membrane

    # Derived elastic moduli
    G: float = field(init=False, default=0.0)
    K: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.nu < 0.0 or self.nu >= 0.5:
            self.nu = 0.3
        if self.E <= 0.0:
            self.E = 1.0e9
        if self.e1 <= 0.0:
            self.e1 = self.E
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))


def resolve(mat: Any) -> Law134Params:
    """Resolve Law134Params from Material, dict, or Law134Params instance."""
    if isinstance(mat, Law134Params):
        return mat
    if isinstance(mat, Material):
        p = mat.params or {}
        e_val = float(p.get("E", p.get("MAT_E", 1.0e9)))
        return Law134Params(
            E=e_val,
            nu=float(p.get("nu", p.get("MAT_NU", 0.3))),
            rho0=float(getattr(mat, "rho0", p.get("rho", p.get("MAT_RHO", 1000.0)))),
            e1=float(p.get("e1", p.get("E1", e_val))),
            n1=float(p.get("n1", p.get("N1", 0.0))),
            e2=float(p.get("e2", p.get("E2", 1.0e8))),
            v2=float(p.get("v2", p.get("V2", 1.0e7))),
            n2=float(p.get("n2", p.get("N2", 0.0))),
            wrinkle_flag=int(p.get("wrinkle_flag", p.get("IWRINKLE", 1))),
        )
    if isinstance(mat, dict):
        e_val = float(mat.get("E", mat.get("MAT_E", 1.0e9)))
        return Law134Params(
            E=e_val,
            nu=float(mat.get("nu", mat.get("MAT_NU", 0.3))),
            rho0=float(mat.get("rho0", mat.get("rho", mat.get("MAT_RHO", 1000.0)))),
            e1=float(mat.get("e1", mat.get("E1", e_val))),
            n1=float(mat.get("n1", mat.get("N1", 0.0))),
            e2=float(mat.get("e2", mat.get("E2", 1.0e8))),
            v2=float(mat.get("v2", mat.get("V2", 1.0e7))),
            n2=float(mat.get("n2", mat.get("N2", 0.0))),
            wrinkle_flag=int(mat.get("wrinkle_flag", mat.get("IWRINKLE", 1))),
        )
    return Law134Params()


def build_law134(mat: Any) -> Law134Params:
    """Build Law134Params from Material entity or dictionary."""
    return resolve(mat)


def needs_defgrad(mat: Any = None) -> bool:
    """LAW134 incremental fabric formulation does not need deformation gradient F."""
    return False


def extra_shapes(mat: Any = None, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """State arrays for LAW134 (viscous stress history uvar)."""
    return {
        "uvar": (nip, 6) if nip > 1 else (6,),
    }


def sound_speed(
    mat: Any,
    eps: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    rho: Optional[float] = None,
    is_shell: bool = False,
    **kwargs: Any,
) -> float:
    """Dilatational sound speed for LAW134."""
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
    """Vectorized 3D solid update matching sigeps134s.F90."""
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

    uvar_all = np.zeros((n, 6), dtype=np.float64)
    if extra is not None and "uvar" in extra:
        uv = np.asarray(extra["uvar"], dtype=np.float64)
        if uv.ndim == 1 and n == 1:
            uvar_all[0] = uv[:6]
        elif uv.ndim == 2:
            uvar_all[:min(n, uv.shape[0])] = uv[:min(n, uv.shape[0]), :6]

    dtime = max(dt, _EM20)
    nu = p.nu
    nu_c = 1.0 / ((1.0 + nu) * (1.0 - 2.0 * nu))
    nu_shear = 0.5 / (1.0 + nu)

    soundsp = math.sqrt(max(0.0, (p.K + _FOUR_THIRDS * p.G) / p.rho0))
    c_out = np.full(n, soundsp, dtype=np.float64)
    s_out = np.zeros_like(s)

    for i in range(n):
        # Relative volume r = rho0 / rho
        de_vol = d[i, 0] + d[i, 1] + d[i, 2]
        r = max(_EM20, 1.0 + de_vol)
        e1t = p.e1 * (r ** (-p.n1))
        cc = abs(1.0 - r)
        v2t = 2.0 * p.v2 * (cc ** p.n2) if cc >= _EM10 else 0.0
        beta = p.e2 / max(_EM20, v2t)

        aa = math.exp(-beta * dtime)
        bb = p.e2 * math.exp(-beta * 0.5 * dtime)

        # Incremental elastic and viscous stress
        dsige = e1t * d[i]
        sigv_old = uvar_all[i].copy()
        sigv = aa * sigv_old + bb * d[i]
        uvar_all[i] = sigv

        dsig = dsige + sigv - sigv_old

        s_new = np.copy(s[i])
        s_new[0] += nu_c * ((1.0 - nu) * dsig[0] + nu * (dsig[1] + dsig[2]))
        s_new[1] += nu_c * ((1.0 - nu) * dsig[1] + nu * (dsig[0] + dsig[2]))
        s_new[2] += nu_c * ((1.0 - nu) * dsig[2] + nu * (dsig[0] + dsig[1]))
        s_new[3] += dsig[3] * nu_shear
        s_new[4] += dsig[4] * nu_shear
        s_new[5] += dsig[5] * nu_shear

        # Wrinkling condition (no compressive resistance)
        if p.wrinkle_flag:
            for k in range(3):
                if s_new[k] < 0.0:
                    s_new[k] = 0.0

        s_out[i] = s_new

    if extra is not None:
        extra["uvar"] = uvar_all[0] if is_1d else uvar_all

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
    """Plane-stress membrane shell update with fabric wrinkling kinematics."""
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
        s_trial = np.copy(s[i])
        s_trial[0] += q11 * d[i, 0] + q12 * d[i, 1]
        s_trial[1] += q12 * d[i, 0] + q11 * d[i, 1]
        if s_trial.shape[0] > 2 and d.shape[1] > 2:
            s_trial[2] += q33 * d[i, 2]

        if p.wrinkle_flag:
            # Principal stress transformation for plane stress wrinkling
            sxx = s_trial[0]
            syy = s_trial[1]
            sxy = s_trial[2] if s_trial.shape[0] > 2 else 0.0

            s_avg = 0.5 * (sxx + syy)
            r = math.sqrt(max(0.0, (0.5 * (sxx - syy))**2 + sxy**2))
            s1 = s_avg + r
            s2 = s_avg - r

            # Wrinkling states:
            # Taut: s1 > 0, s2 > 0 -> retain stress
            # Wrinkled: s1 > 0, s2 <= 0 -> uniaxial tension along direction 1
            # Slack: s1 <= 0, s2 <= 0 -> completely zero stress
            if s1 <= 0.0 and s2 <= 0.0:
                s_trial[:3] = 0.0
            elif s1 > 0.0 and s2 <= 0.0:
                # Uniaxial tension along principal direction 1
                theta = 0.5 * math.atan2(2.0 * sxy, max(sxx - syy, 1e-12))
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                s_trial[0] = s1 * (cos_t ** 2)
                s_trial[1] = s1 * (sin_t ** 2)
                if s_trial.shape[0] > 2:
                    s_trial[2] = s1 * cos_t * sin_t

        s_out[i] = s_trial

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
        for key in (134, "134", "LAW134", "FABRIC_WRINKLE", "MAT_LAW134"):
            MAT_PHYSICS_REGISTRY[key] = build_law134
    except Exception:
        pass


_register()
