"""LAW91 — Transverse Pinching Shell Formulation with Elastoplasticity (/MAT/LAW91, /MAT/PINCH_SHELL).

Upstream OpenRadioss Fortran reference:
- Transverse Pinching Shell Global Formulation:
  `engine/source/materials/mat/mat091/sigeps91gpinch.F`
- Shell Bending & Transverse Shear Shell Interface:
  `engine/source/elements/shell/coqueba/mulawglcpinch.F`
- HyperMesh CFG Schema:
  `hm_cfg_files/config/CFG/Keyword971_R6.1/MAT/mat_091_092.cfg`

Theory:
-------
1. Transverse Pinching & Bending Global Formulation:
   Coupled Ilyushin-type global yield surface incorporating in-plane stresses,
   transverse shear stresses, through-thickness pinching normal stress sigma_zz,
   and pinching/bending moments:
     M_s = M_xx + M_yy
     S_eq = sqrt(16 * (M_s^2 + 3*(M_xy^2 - M_xx*M_yy)) +
                 3/2 * (s_dev_xx^2 + s_dev_yy^2 + s_dev_zz^2 + 2*sigma_xy^2 + 2*sigma_yz^2 + 2*sigma_zx^2))

2. Yield condition:
     sigma_y = sigma_y0 + H_m * eps_p
     RR = min(1.0, sigma_y / max(S_eq, 1.0e-20))

3. Stress update:
     sigma_dev_new = sigma_dev_trial * RR
     sigma_xx = sigma_dev_xx_new - P_new
     sigma_yy = sigma_dev_yy_new - P_new
     sigma_zz = sigma_dev_zz_new - P_new
     sigma_xy = sigma_xy_trial * RR
     sigma_yz = sigma_yz_trial * RR
     sigma_zx = sigma_zx_trial * RR

4. Transverse normal / pinching acoustic wave speed:
     SSP = sqrt(PA1 / rho0)
     where PA1 = E * (1 - nu) / ((1 + nu) * (1 - 2*nu))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20
_EM10 = 1.0e-10


@dataclass
class Law91Params:
    """Parameters for OpenRadioss /MAT/LAW91 (Transverse Pinching Shell with Elastoplasticity)."""
    id: int = 1
    title: str = ""
    law: int = 91
    law_name: str = "LAW91"

    # Density
    rho0: float = 1.0
    rhor: float = 1.0

    # Elastic Moduli
    young: float = 1000.0
    nu: float = 0.3

    # Plasticity & Hardening
    yield_stress: float = 20.0
    hardening: float = 50.0

    # Derived constants
    g: float = field(init=False)
    gs: float = field(init=False)
    bulk: float = field(init=False)
    pa1: float = field(init=False)
    pa2: float = field(init=False)
    pa3: float = field(init=False)
    pa4: float = field(init=False)
    pa5: float = field(init=False)
    c1: float = field(init=False)
    c2: float = field(init=False)
    c3: float = field(init=False)
    c_solid: float = field(init=False)
    c_shell: float = field(init=False)

    def __post_init__(self) -> None:
        if self.rho0 <= 0.0:
            self.rho0 = 1.0
        if self.rhor <= 0.0:
            self.rhor = self.rho0
        if self.young <= 0.0:
            self.young = 1000.0

        self.nu = max(0.0, min(0.49999, self.nu))
        self.g = 0.5 * self.young / max(1.0 + self.nu, _EM20)
        self.gs = (5.0 / 6.0) * self.g
        self.bulk = self.young / max(3.0 * (1.0 - 2.0 * self.nu), _EM20)

        # 3D Lamé parameters (PA1, PA2)
        denom_3d = max((1.0 + self.nu) * (1.0 - 2.0 * self.nu), _EM20)
        self.pa1 = self.young * (1.0 - self.nu) / denom_3d
        self.pa2 = self.young * self.nu / denom_3d
        self.pa3 = self.g

        # Shell plane stress parameters (PA4, PA5)
        denom_2d = max(1.0 - self.nu ** 2, _EM20)
        self.pa4 = self.young / denom_2d
        self.pa5 = self.nu * self.young / denom_2d

        self.c1 = 1.0 / self.young
        self.c2 = -self.nu * self.c1
        self.c3 = 1.0 / self.g

        # Sound speed: matching sigeps91gpinch.F: SSP = SQRT(PA1 / RHO0)
        self.c_solid = math.sqrt(max(0.0, self.pa1 / self.rho0))
        self.c_shell = math.sqrt(max(0.0, self.pa4 / self.rho0))


def _extract_val(data: Dict[str, Any], keys: Sequence[str], default: float) -> float:
    for k in keys:
        if k in data and data[k] is not None:
            try:
                return float(data[k])
            except (ValueError, TypeError):
                pass
    return default


def build_law91(mat_def: Any = None, **kwargs: Any) -> Law91Params:
    """Construct Law91Params from a Material entity, dictionary, or keyword arguments."""
    if isinstance(mat_def, Law91Params):
        return mat_def

    data: Dict[str, Any] = {}
    if isinstance(mat_def, dict):
        data.update(mat_def)
    elif hasattr(mat_def, "__dict__"):
        data.update(mat_def.__dict__)
        if hasattr(mat_def, "params") and isinstance(mat_def.params, dict):
            data.update(mat_def.params)

    data.update(kwargs)

    mat_id = int(_extract_val(data, ["id", "mat_id", "user_id"], 1))
    title = str(data.get("title", f"LAW91_{mat_id}"))

    rho0 = _extract_val(data, ["MAT_RHO", "rho0", "rho", "density"], 1.0)
    rhor = _extract_val(data, ["Refer_Rho", "rhor", "ref_rho"], rho0)

    young = _extract_val(data, ["MAT_E", "young", "e", "E", "UPARAM_1"], 1000.0)
    nu = _extract_val(data, ["MAT_NU", "nu", "poisson", "nux", "UPARAM_2"], 0.3)

    sigy0 = _extract_val(data, ["SIGMA_r", "yield_stress", "sigy0", "sigy", "sig0", "UPARAM_3"], 20.0)
    hm = _extract_val(data, ["HM", "hardening", "h_iso", "h", "UPARAM_4"], 50.0)

    return Law91Params(
        id=mat_id,
        title=title,
        rho0=rho0,
        rhor=rhor,
        young=young,
        nu=nu,
        yield_stress=sigy0,
        hardening=hm,
    )


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law91Params:
    """Resolve and cache Law91Params from a Material or dict."""
    if isinstance(mat, Law91Params):
        return mat
    cached = getattr(mat, "_cached_law91", None)
    if cached is None:
        cached = build_law91(mat)
        try:
            setattr(mat, "_cached_law91", cached)
        except Exception:
            pass
    return cached


def needs_defgrad(mat: Any = None) -> bool:
    """Return False: LAW91 is an incremental rate formulation."""
    return False


def extra_shapes(mat: Any = None, nip: Optional[int] = 1) -> Dict[str, Tuple[int, ...]]:
    """Persistent history variables for LAW91.
    Indices:
    [0]: plastic strain eps_p
    [1]: thickness h
    [2]: initial area*thickness
    [3..7]: moments (m_xx, m_yy, m_xy, m_pxz, m_pyz)
    [8..9]: reserved / plastic work
    """
    if nip is not None and nip > 1:
        return {"uvar91": (nip, 10), "uvar": (nip, 10)}
    return {"uvar91": (10,), "uvar": (10,)}


def sound_speed(
    mat: Any,
    eps: Any = None,
    extra: Any = None,
    is_shell: bool = False,
) -> float | np.ndarray:
    """Acoustic sound speed for LAW91."""
    p = resolve(mat)
    c_val = p.c_shell if is_shell else p.c_solid
    if eps is not None and isinstance(eps, np.ndarray) and eps.ndim > 1:
        return np.full(len(eps), c_val, dtype=float)
    return c_val


def _solid_update_single(
    p: Law91Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    uvar0: np.ndarray,
    off: float = 1.0,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """3D solid continuum update for LAW91."""
    if off < 0.1:
        return np.zeros(6, dtype=float), float(uvar0[0]), uvar0.copy(), 0.0

    uvar = uvar0.copy()
    epsp = uvar[0]

    # Volumetric and deviatoric strain increments
    deps_v = deps[0] + deps[1] + deps[2]
    deps_dev = deps[:3] - (deps_v / 3.0)

    # Elastic trial stresses
    p_old = -(sig0[0] + sig0[1] + sig0[2]) / 3.0
    s_dev_old = sig0[:3] + p_old

    s_dev_trial = np.zeros(6, dtype=float)
    s_dev_trial[:3] = s_dev_old + 2.0 * p.g * deps_dev
    s_dev_trial[3] = sig0[3] + p.g * deps[3]
    s_dev_trial[4] = sig0[4] + p.g * deps[4]
    s_dev_trial[5] = sig0[5] + p.g * deps[5]

    p_new = p_old - p.bulk * deps_v

    # von Mises equivalent stress
    j2 = 0.5 * (s_dev_trial[0] ** 2 + s_dev_trial[1] ** 2 + s_dev_trial[2] ** 2) + \
        (s_dev_trial[3] ** 2 + s_dev_trial[4] ** 2 + s_dev_trial[5] ** 2)
    s_eq = math.sqrt(max(0.0, 3.0 * j2))

    sigy = p.yield_stress + p.hardening * epsp
    rr = min(1.0, sigy / max(s_eq, _EM20))

    if rr < 1.0:
        dlam = (s_eq - sigy) / max(3.0 * p.g + p.hardening, _EM20)
        epsp += max(0.0, dlam)

    sign = np.empty(6, dtype=float)
    sign[:3] = s_dev_trial[:3] * rr - p_new
    sign[3:] = s_dev_trial[3:] * rr

    uvar[0] = epsp
    return sign, epsp, uvar, p.c_solid


def solid_update(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    """3D solid constitutive update for /MAT/LAW91."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(6, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(float)
    nel = len(sig_arr)

    uvar_arr = np.zeros((nel, 10), dtype=float)
    if extra is not None and isinstance(extra, dict):
        for k in ("uvar91", "uvar"):
            if k in extra and extra[k] is not None:
                u = np.asarray(extra[k], dtype=float)
                if u.ndim == 1:
                    uvar_arr[0, :min(10, len(u))] = u[:min(10, len(u))]
                elif u.ndim == 2:
                    uvar_arr[:min(nel, len(u)), :min(10, u.shape[1])] = u[:min(nel, len(u)), :min(10, u.shape[1])]
                break

    if epsp is not None:
        ep_in = np.atleast_1d(epsp).astype(float)
        uvar_arr[:min(nel, len(ep_in)), 0] = ep_in[:nel]

    off_arr = np.ones(nel, dtype=float)
    if extra is not None and isinstance(extra, dict) and "off" in extra:
        o = np.atleast_1d(extra["off"]).astype(float)
        off_arr[:min(nel, len(o))] = o[:nel]

    sig_out = np.zeros_like(sig_arr)
    epsp_out = np.zeros(nel, dtype=float)
    c_out = np.zeros(nel, dtype=float)

    for i in range(nel):
        s_i, ep_i, u_i, c_i = _solid_update_single(p, sig_arr[i], deps_arr[i], uvar_arr[i], off=off_arr[i])
        sig_out[i] = s_i
        epsp_out[i] = ep_i
        uvar_arr[i] = u_i
        c_out[i] = c_i

    if extra is not None and isinstance(extra, dict):
        extra["uvar91"] = uvar_arr
        extra["uvar"] = uvar_arr

    res_sig = sig_out[0] if is_1d else sig_out
    res_epsp = epsp_out[0] if is_1d else epsp_out
    res_c = float(c_out[0]) if is_1d else c_out

    if hasattr(sig, "__setitem__"):
        try:
            sig[:] = res_sig
        except Exception:
            pass
    if epsp is not None and hasattr(epsp, "__setitem__"):
        try:
            epsp[:] = res_epsp
        except Exception:
            pass

    return res_sig, res_epsp, res_c


def _shell_update_single(
    p: Law91Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    uvar0: np.ndarray,
    off: float = 1.0,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """2D pinching shell constitutive update matching sigeps91gpinch.F."""
    if off < 0.1:
        return np.zeros(len(sig0), dtype=float), float(uvar0[0]), uvar0.copy(), 0.0

    uvar = uvar0.copy()
    epsp = uvar[0]
    moms = uvar[3:8]  # [m_xx, m_yy, m_xy, m_pxz, m_pyz]

    n_comp = len(sig0)
    # Recover strains
    deps_xx = deps[0]
    deps_yy = deps[1]
    deps_xy = deps[2] if len(deps) > 2 else 0.0

    # Through-thickness strain: deps_zz
    if len(deps) >= 6:
        deps_zz = deps[2]
        deps_xy = deps[3]
        deps_yz = deps[4]
        deps_zx = deps[5]
    elif len(deps) == 5:
        deps_zz = -(p.nu / max(1.0 - p.nu, _EM20)) * (deps_xx + deps_yy)
        deps_yz = deps[3]
        deps_zx = deps[4]
    else:
        deps_zz = -(p.nu / max(1.0 - p.nu, _EM20)) * (deps_xx + deps_yy)
        deps_yz = 0.0
        deps_zx = 0.0

    # Old stresses
    sig_xx = sig0[0]
    sig_yy = sig0[1]
    sig_xy = sig0[2] if n_comp <= 5 else sig0[3]
    sig_zz = sig0[2] if n_comp == 6 else 0.0
    sig_yz = sig0[3] if n_comp == 5 else (sig0[4] if n_comp == 6 else 0.0)
    sig_zx = sig0[4] if n_comp == 5 else (sig0[5] if n_comp == 6 else 0.0)

    p_old = - (sig_xx + sig_yy + sig_zz) / 3.0
    s_dev_xx = sig_xx + p_old
    s_dev_yy = sig_yy + p_old
    s_dev_zz = sig_zz + p_old

    # Spherical strain change
    dd = (deps_xx + deps_yy + deps_zz) / 3.0
    s_dev_xx_trial = s_dev_xx + 2.0 * p.g * (deps_xx - dd)
    s_dev_yy_trial = s_dev_yy + 2.0 * p.g * (deps_yy - dd)
    s_dev_zz_trial = s_dev_zz + 2.0 * p.g * (deps_zz - dd)
    s_xy_trial = sig_xy + p.g * deps_xy
    s_yz_trial = sig_yz + p.gs * deps_yz
    s_zx_trial = sig_zx + p.gs * deps_zx

    # Moments (Ilyushin global surface coupling)
    m_s = moms[0] + moms[1]
    m_term = 16.0 * (m_s ** 2 + 3.0 * (moms[2] ** 2 - moms[0] * moms[1]))
    s_term = 1.5 * (s_dev_xx_trial ** 2 + s_dev_yy_trial ** 2 + s_dev_zz_trial ** 2 +
                     2.0 * (s_xy_trial ** 2 + s_yz_trial ** 2 + s_zx_trial ** 2))
    s_eq = math.sqrt(max(0.0, m_term + s_term))

    sigy = p.yield_stress + p.hardening * epsp
    rr = min(1.0, sigy / max(s_eq, _EM20))

    if rr < 1.0:
        dlam = (s_eq - sigy) / max(3.0 * p.g + p.hardening, _EM20)
        epsp += max(0.0, dlam)

    # Volumetric pressure
    deps_v = deps_xx + deps_yy + deps_zz
    p_new = p_old - p.bulk * deps_v

    sign_xx = s_dev_xx_trial * rr - p_new
    sign_yy = s_dev_yy_trial * rr - p_new
    sign_zz = s_dev_zz_trial * rr - p_new
    sign_xy = s_xy_trial * rr
    sign_yz = s_yz_trial * rr
    sign_zx = s_zx_trial * rr

    # Scale moments
    moms[0] *= rr
    moms[1] *= rr
    moms[2] *= rr

    uvar[0] = epsp
    uvar[3:8] = moms

    if n_comp == 3:
        sign = np.array([sign_xx, sign_yy, sign_xy], dtype=float)
    elif n_comp == 5:
        sign = np.array([sign_xx, sign_yy, sign_xy, sign_yz, sign_zx], dtype=float)
    elif n_comp == 6:
        sign = np.array([sign_xx, sign_yy, sign_zz, sign_xy, sign_yz, sign_zx], dtype=float)
    else:
        sign = np.array([sign_xx, sign_yy, sign_xy], dtype=float)

    return sign, epsp, uvar, p.c_shell


def shell_update(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    """2D pinching shell constitutive update for /MAT/LAW91."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(3, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(float)
    nel = len(sig_arr)

    uvar_arr = np.zeros((nel, 10), dtype=float)
    if extra is not None and isinstance(extra, dict):
        for k in ("uvar91", "uvar"):
            if k in extra and extra[k] is not None:
                u = np.asarray(extra[k], dtype=float)
                if u.ndim == 1:
                    uvar_arr[0, :min(10, len(u))] = u[:min(10, len(u))]
                elif u.ndim == 2:
                    uvar_arr[:min(nel, len(u)), :min(10, u.shape[1])] = u[:min(nel, len(u)), :min(10, u.shape[1])]
                break

    if epsp is not None:
        ep_in = np.atleast_1d(epsp).astype(float)
        uvar_arr[:min(nel, len(ep_in)), 0] = ep_in[:nel]

    off_arr = np.ones(nel, dtype=float)
    if extra is not None and isinstance(extra, dict) and "off" in extra:
        o = np.atleast_1d(extra["off"]).astype(float)
        off_arr[:min(nel, len(o))] = o[:nel]

    sig_out = np.zeros_like(sig_arr)
    epsp_out = np.zeros(nel, dtype=float)
    c_out = np.zeros(nel, dtype=float)

    for i in range(nel):
        s_i, ep_i, u_i, c_i = _shell_update_single(p, sig_arr[i], deps_arr[i], uvar_arr[i], off=off_arr[i])
        sig_out[i] = s_i
        epsp_out[i] = ep_i
        uvar_arr[i] = u_i
        c_out[i] = c_i

    if extra is not None and isinstance(extra, dict):
        extra["uvar91"] = uvar_arr
        extra["uvar"] = uvar_arr

    res_sig = sig_out[0] if is_1d else sig_out
    res_epsp = epsp_out[0] if is_1d else epsp_out
    res_c = float(c_out[0]) if is_1d else c_out

    if hasattr(sig, "__setitem__"):
        try:
            sig[:] = res_sig
        except Exception:
            pass
    if epsp is not None and hasattr(epsp, "__setitem__"):
        try:
            epsp[:] = res_epsp
        except Exception:
            pass

    return res_sig, res_epsp, res_c


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent solid tangent stiffness operator (n, 6, 6)."""
    p = resolve(mat)
    c_el = np.zeros((6, 6), dtype=float)
    c11 = p.bulk + 4.0 / 3.0 * p.g
    c12 = p.bulk - 2.0 / 3.0 * p.g
    c_el[0, 0] = c_el[1, 1] = c_el[2, 2] = c11
    c_el[0, 1] = c_el[0, 2] = c_el[1, 0] = c_el[1, 2] = c_el[2, 0] = c_el[2, 1] = c12
    c_el[3, 3] = c_el[4, 4] = c_el[5, 5] = p.g

    n = 1
    if sig is not None and np.ndim(sig) >= 2:
        n = len(sig)
    return np.broadcast_to(c_el, (n, 6, 6)).copy()


consistent_solid_tangent = solid_tangent


def shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent shell plane-stress tangent operator (n, 3, 3)."""
    p = resolve(mat)
    c_el = np.array([
        [p.pa4, p.pa5, 0.0],
        [p.pa5, p.pa4, 0.0],
        [0.0, 0.0, p.g],
    ], dtype=float)

    n = 1
    if sig is not None and np.ndim(sig) >= 2:
        n = len(sig)
    return np.broadcast_to(c_el, (n, 3, 3)).copy()


consistent_shell_tangent = shell_tangent
