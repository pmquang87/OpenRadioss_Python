"""LAW97 — JWL-Baker (JWLB) Explosive EOS & Orthotropic Non-linear Formulation (/MAT/LAW97, /MAT/JWLB).

Upstream OpenRadioss Fortran reference:
- Starter Card Reader:
  `starter/source/materials/mat/mat097/hm_read_mat97.F`
- Engine Constitutive Update:
  `engine/source/materials/mat/mat097/sigeps97.F`
- HyperMesh CFG Schema:
  `hm_cfg_files/config/CFG/Keyword971/MAT/mat_097.cfg`

Theory:
-------
1. JWL-Baker Multi-term Equation of State for high explosives and composite detonation products:
   Pressure:
     P(v, E) = sum_{i=1}^5 A_i * (1 - lambda / (R_i * v)) * exp(-R_i * v)
               + lambda * (E_int / (v * V_0))
               + C * (1 - lambda / omega) * v^(-omega - 1)
   where:
     v = V / V_0 is the relative expansion volume.
     lambda(v) = sum_{i=1}^5 (A_{Li} * v + B_{Li}) * exp(-R_{Li} * v) + omega

2. Sound speed calculation:
     rho * c^2 = sum_{i=1}^5 A_i * [ (v * dlambda/dv - lambda) / R_i + R_i * v^2 - lambda * v ] * exp(-R_i * v)
                 + C * [ (omega + 1) * (1 - lambda/omega) + v * (dlambda/dv) / omega ] * v^(-omega)
                 + (E_int / V_0) * lambda + lambda * v * (P + P_sh) - (E_int / V_0) * v * dlambda/dv
     SSP = sqrt(max(rho * c^2 / rho0, 1.0e-20))

3. Burn fraction BFRAC:
     Controlled by detonation velocity D, detonation arrival time TBURN,
     and compression shock factor BHE:
     BFRAC in [0, 1].
     P_effective = (1 - BFRAC) * P_0 + BFRAC * P_JWLB.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20
_EM10 = 1.0e-10


@dataclass
class Law97Params:
    """Parameters for OpenRadioss /MAT/LAW97 (JWLB Explosive & Orthotropic Non-linear)."""
    id: int = 1
    title: str = ""
    law: int = 97
    law_name: str = "LAW97"

    # Density
    rho0: float = 1.63
    rhor: float = 1.63

    # Explosive detonation & JWLB EOS parameters
    p0: float = 0.0           # Initial pressure
    psh: float = 0.0          # Pressure shift / tension cutoff
    ibfrac: int = 0           # Burn fraction formulation flag
    d: float = 8000.0         # Detonation velocity
    pcj: float = 2.8e10       # Chapman-Jouguet pressure
    e0: float = 8.5e9         # Initial energy per unit volume
    w: float = 0.35           # Omega parameter
    c: float = 1.0e9          # Constant C

    # Multi-term JWLB coefficients (5 terms)
    a: Tuple[float, ...] = (5.0e11, 1.0e11, 2.0e10, 0.0, 0.0)
    r: Tuple[float, ...] = (4.5, 1.5, 1.0, 1.0, 1.0)
    al: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    bl: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    rl: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)

    # Elastic Moduli (for structural solid / shear stiffness)
    young: float = 10000.0
    nu: float = 0.3

    # Derived constants
    bhe: float = field(init=False)
    vcj: float = field(init=False)
    g: float = field(init=False)
    bulk: float = field(init=False)
    a11_2d: float = field(init=False)
    a21_2d: float = field(init=False)
    c_solid: float = field(init=False)
    c_shell: float = field(init=False)

    def __post_init__(self) -> None:
        if self.rho0 <= 0.0:
            self.rho0 = 1.63
        if self.rhor <= 0.0:
            self.rhor = self.rho0
        if self.d <= 0.0:
            self.d = 8000.0
        if self.pcj <= 0.0:
            self.pcj = 2.8e10
        if self.w <= 0.0:
            self.w = 0.35

        # BHE = rho0 * D^2 / PCJ
        denom_bhe = max(self.pcj, _EM20)
        self.bhe = self.rho0 * (self.d ** 2) / denom_bhe
        self.vcj = 1.0 - (1.0 / max(self.bhe, _EM20))

        if self.young <= 0.0:
            self.young = 10000.0
        self.nu = max(0.0, min(0.49999, self.nu))
        self.g = 0.5 * self.young / max(1.0 + self.nu, _EM20)
        self.bulk = self.young / max(3.0 * (1.0 - 2.0 * self.nu), _EM20)

        denom_2d = max(1.0 - self.nu ** 2, _EM20)
        self.a11_2d = self.young / denom_2d
        self.a21_2d = self.nu * self.young / denom_2d

        # Acoustic wave speeds (at unburned state, reference detonation velocity D is sound speed)
        c_elastic = math.sqrt(max(0.0, (self.bulk + 4.0 / 3.0 * self.g) / self.rho0))
        self.c_solid = max(self.d, c_elastic)
        self.c_shell = max(self.d, math.sqrt(max(0.0, self.a11_2d / self.rho0)))


def _extract_val(data: Dict[str, Any], keys: Sequence[str], default: float) -> float:
    for k in keys:
        if k in data and data[k] is not None:
            try:
                return float(data[k])
            except (ValueError, TypeError):
                pass
    return default


def build_law97(mat_def: Any = None, **kwargs: Any) -> Law97Params:
    """Construct Law97Params from a Material entity, dictionary, or keyword arguments."""
    if isinstance(mat_def, Law97Params):
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
    title = str(data.get("title", f"LAW97_{mat_id}"))

    rho0 = _extract_val(data, ["MAT_RHO", "rho0", "rho", "density"], 1.63)
    rhor = _extract_val(data, ["Refer_Rho", "rhor", "ref_rho"], rho0)

    p0 = _extract_val(data, ["MAT_P0", "p0", "P0"], 0.0)
    psh = _extract_val(data, ["MAT_PSH", "psh", "PSH"], 0.0)
    ibfrac = int(_extract_val(data, ["MAT_IBFRAC", "ibfrac", "IBFRAC"], 0))

    d = _extract_val(data, ["MLAW97_D", "d", "D", "det_vel"], 8000.0)
    pcj = _extract_val(data, ["MLAW97_PCJ", "pcj", "PCJ"], 2.8e10)
    e0 = _extract_val(data, ["MLAW97_E0", "e0", "E0"], 8.5e9)
    w = _extract_val(data, ["Omega", "w", "W", "omega"], 0.35)
    c = _extract_val(data, ["MLAW97_C", "c", "C"], 1.0e9)

    a_list: List[float] = []
    r_list: List[float] = []
    al_list: List[float] = []
    bl_list: List[float] = []
    rl_list: List[float] = []

    for i in range(1, 6):
        a_val = _extract_val(data, [f"MLAW97_A{i}", f"a{i}", f"A{i}"], 5.0e11 if i == 1 else (1.0e11 if i == 2 else 0.0))
        r_val = _extract_val(data, [f"MLAW97_R{i}", f"r{i}", f"R{i}"], 4.5 if i == 1 else (1.5 if i == 2 else 1.0))
        al_val = _extract_val(data, [f"MLAW97_AL{i}", f"al{i}", f"AL{i}"], 0.0)
        bl_val = _extract_val(data, [f"MLAW97_BL{i}", f"bl{i}", f"BL{i}"], 0.0)
        rl_val = _extract_val(data, [f"MLAW97_RL{i}", f"rl{i}", f"RL{i}"], 1.0)
        a_list.append(a_val)
        r_list.append(max(r_val, 1.0e-4))
        al_list.append(al_val)
        bl_list.append(bl_val)
        rl_list.append(max(rl_val, 1.0e-4))

    young = _extract_val(data, ["MAT_E", "young", "e", "E"], 10000.0)
    nu = _extract_val(data, ["MAT_NU", "nu", "poisson", "nux"], 0.3)

    return Law97Params(
        id=mat_id,
        title=title,
        rho0=rho0,
        rhor=rhor,
        p0=p0,
        psh=psh,
        ibfrac=ibfrac,
        d=d,
        pcj=pcj,
        e0=e0,
        w=w,
        c=c,
        a=tuple(a_list),
        r=tuple(r_list),
        al=tuple(al_list),
        bl=tuple(bl_list),
        rl=tuple(rl_list),
        young=young,
        nu=nu,
    )


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law97Params:
    """Resolve and cache Law97Params from a Material or dict."""
    if isinstance(mat, Law97Params):
        return mat
    cached = getattr(mat, "_cached_law97", None)
    if cached is None:
        cached = build_law97(mat)
        try:
            setattr(mat, "_cached_law97", cached)
        except Exception:
            pass
    return cached


def needs_defgrad(mat: Any = None) -> bool:
    """Return False: LAW97 is an incremental rate formulation."""
    return False


def extra_shapes(mat: Any = None, nip: Optional[int] = 1) -> Dict[str, Tuple[int, ...]]:
    """Persistent history variables for LAW97.
    Cols:
    [0]: burn fraction bfrac
    [1]: current internal energy eint
    [2]: relative volume v
    [3]: pressure p_new
    [4]: plastic strain / work
    """
    if nip is not None and nip > 1:
        return {"uvar97": (nip, 5), "uvar": (nip, 5)}
    return {"uvar97": (5,), "uvar": (5,)}


def sound_speed(
    mat: Any,
    eps: Any = None,
    extra: Any = None,
    is_shell: bool = False,
) -> float | np.ndarray:
    """Acoustic sound speed for LAW97."""
    p = resolve(mat)
    c_val = p.c_shell if is_shell else p.c_solid
    if eps is not None and isinstance(eps, np.ndarray) and eps.ndim > 1:
        return np.full(len(eps), c_val, dtype=float)
    return c_val


def _calc_jwlb_pressure(p: Law97Params, v: float, eint: float) -> Tuple[float, float]:
    """Evaluate JWL-Baker EOS pressure and acoustic sound speed squared matching sigeps97.F."""
    v_eff = max(0.01, min(100.0, v))

    erlv = [math.exp(-min(50.0, max(0.0, p.rl[i] * v_eff))) for i in range(5)]
    lambdas = [(p.al[i] * v_eff + p.bl[i]) * erlv[i] for i in range(5)]
    lambda_tot = sum(lambdas) + p.w

    dldv = sum([p.al[i] * erlv[i] - (p.al[i] * v_eff + p.bl[i]) * p.rl[i] * erlv[i] for i in range(5)])

    rv = [min(50.0, max(0.0, p.r[i] * v_eff)) for i in range(5)]
    p_terms = [p.a[i] * (1.0 - lambda_tot / max(rv[i], _EM20)) * math.exp(-rv[i]) for i in range(5)]

    # C * (1 - lambda/w) * v^(-w - 1)
    c_term = p.c * (1.0 - lambda_tot / max(p.w, _EM20)) * (v_eff ** (-p.w - 1.0))
    p_jwlb = sum(p_terms) + (lambda_tot * eint / v_eff) + c_term

    # Sound speed terms (rhoc2)
    rhoc2_terms = [
        p.a[i] * (((v_eff * dldv - lambda_tot) / max(p.r[i], _EM20)) + p.r[i] * v_eff * v_eff - lambda_tot * v_eff) * math.exp(-rv[i])
        for i in range(5)
    ]
    rhoc2 = sum(rhoc2_terms)
    rhoc2 += p.c * ((p.w + 1.0) * (1.0 - lambda_tot / max(p.w, _EM20)) + v_eff * dldv / max(p.w, _EM20)) * (v_eff ** (-p.w))
    rhoc2 += eint * lambda_tot + lambda_tot * v_eff * (p_jwlb + p.psh) - eint * v_eff * dldv

    c_snd = math.sqrt(max(_EM20, rhoc2 / p.rho0))
    return p_jwlb, c_snd


def _solid_update_single(
    p: Law97Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    uvar0: np.ndarray,
    off: float = 1.0,
    time: float = 0.0,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Single 3D solid continuum update for LAW97."""
    if off < 0.1:
        return np.zeros(6, dtype=float), float(uvar0[4]), uvar0.copy(), 0.0

    uvar = uvar0.copy()
    bfrac = uvar[0]
    eint = uvar[1] if uvar[1] != 0.0 else p.e0
    v_old = uvar[2] if uvar[2] > 0.0 else 1.0

    deps_v = deps[0] + deps[1] + deps[2]
    v_new = max(0.01, v_old * (1.0 + deps_v))

    # Burn fraction calculation
    if bfrac < 1.0:
        if p.ibfrac == 0:
            bfrac = min(1.0, max(0.0, p.bhe * max(0.0, 1.0 - v_new) + (p.d * time * 0.01)))
        elif p.ibfrac == 1:
            bfrac = min(1.0, max(0.0, p.bhe * max(0.0, 1.0 - v_new)))
        elif p.ibfrac == 2:
            bfrac = min(1.0, max(0.0, p.d * time * 0.01))

    p_jwlb, c_snd = _calc_jwlb_pressure(p, v_new, eint)
    p_eff = (1.0 - bfrac) * p.p0 + bfrac * p_jwlb
    p_eff = max(-p.psh, p_eff - p.psh)

    # Deviatoric stresses (fluid / hydrodynamic explosive response retains shear if unburned)
    deps_dev = deps[:3] - (deps_v / 3.0)
    g_eff = (1.0 - bfrac) * p.g
    s_dev = sig0[:3] + 2.0 * g_eff * deps_dev
    s_shear = sig0[3:] + g_eff * deps[3:]

    sign = np.empty(6, dtype=float)
    sign[:3] = s_dev - p_eff
    sign[3:] = s_shear

    uvar[0] = bfrac
    uvar[1] = eint
    uvar[2] = v_new
    uvar[3] = p_eff
    return sign, float(uvar[4]), uvar, max(c_snd, p.c_solid * (1.0 - bfrac))


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
    """3D solid constitutive update for /MAT/LAW97."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(6, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(float)
    nel = len(sig_arr)

    uvar_arr = np.zeros((nel, 5), dtype=float)
    if extra is not None and isinstance(extra, dict):
        for k in ("uvar97", "uvar"):
            if k in extra and extra[k] is not None:
                u = np.asarray(extra[k], dtype=float)
                if u.ndim == 1:
                    uvar_arr[0, :min(5, len(u))] = u[:min(5, len(u))]
                elif u.ndim == 2:
                    uvar_arr[:min(nel, len(u)), :min(5, u.shape[1])] = u[:min(nel, len(u)), :min(5, u.shape[1])]
                break

    if epsp is not None:
        ep_in = np.atleast_1d(epsp).astype(float)
        uvar_arr[:min(nel, len(ep_in)), 4] = ep_in[:nel]

    off_arr = np.ones(nel, dtype=float)
    if extra is not None and isinstance(extra, dict) and "off" in extra:
        o = np.atleast_1d(extra["off"]).astype(float)
        off_arr[:min(nel, len(o))] = o[:nel]

    time = float(extra.get("time", 0.0)) if (extra is not None and isinstance(extra, dict)) else 0.0

    sig_out = np.zeros_like(sig_arr)
    epsp_out = np.zeros(nel, dtype=float)
    c_out = np.zeros(nel, dtype=float)

    for i in range(nel):
        s_i, ep_i, u_i, c_i = _solid_update_single(p, sig_arr[i], deps_arr[i], uvar_arr[i], off=off_arr[i], time=time)
        sig_out[i] = s_i
        epsp_out[i] = ep_i
        uvar_arr[i] = u_i
        c_out[i] = c_i

    if extra is not None and isinstance(extra, dict):
        extra["uvar97"] = uvar_arr
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
    p: Law97Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    uvar0: np.ndarray,
    off: float = 1.0,
    time: float = 0.0,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """2D plane-stress shell update for LAW97."""
    if off < 0.1:
        return np.zeros(len(sig0), dtype=float), float(uvar0[4]), uvar0.copy(), 0.0

    uvar = uvar0.copy()
    bfrac = uvar[0]
    eint = uvar[1] if uvar[1] != 0.0 else p.e0
    v_old = uvar[2] if uvar[2] > 0.0 else 1.0

    deps_xx = deps[0]
    deps_yy = deps[1]
    deps_zz = -(p.nu / max(1.0 - p.nu, _EM20)) * (deps_xx + deps_yy)
    deps_v = deps_xx + deps_yy + deps_zz
    v_new = max(0.01, v_old * (1.0 + deps_v))

    if bfrac < 1.0:
        if p.ibfrac == 0:
            bfrac = min(1.0, max(0.0, p.bhe * max(0.0, 1.0 - v_new) + (p.d * time * 0.01)))
        elif p.ibfrac == 1:
            bfrac = min(1.0, max(0.0, p.bhe * max(0.0, 1.0 - v_new)))
        elif p.ibfrac == 2:
            bfrac = min(1.0, max(0.0, p.d * time * 0.01))

    p_jwlb, c_snd = _calc_jwlb_pressure(p, v_new, eint)
    p_eff = (1.0 - bfrac) * p.p0 + bfrac * p_jwlb
    p_eff = max(-p.psh, p_eff - p.psh)

    g_eff = (1.0 - bfrac) * p.g
    s0_xx = sig0[0] + (1.0 - bfrac) * (p.a11_2d * deps_xx + p.a21_2d * deps_yy) - bfrac * p_eff
    s0_yy = sig0[1] + (1.0 - bfrac) * (p.a21_2d * deps_xx + p.a11_2d * deps_yy) - bfrac * p_eff
    s0_xy = sig0[2] + g_eff * deps[2]

    sign = np.empty_like(sig0, dtype=float)
    sign[0] = s0_xx
    sign[1] = s0_yy
    sign[2] = s0_xy
    if len(sig0) >= 5:
        sign[3] = sig0[3] + g_eff * deps[3]
        sign[4] = sig0[4] + g_eff * deps[4]

    uvar[0] = bfrac
    uvar[1] = eint
    uvar[2] = v_new
    uvar[3] = p_eff
    return sign, float(uvar[4]), uvar, max(c_snd, p.c_shell * (1.0 - bfrac))


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
    """2D plane-stress shell constitutive update for /MAT/LAW97."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(3, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(float)
    nel = len(sig_arr)

    uvar_arr = np.zeros((nel, 5), dtype=float)
    if extra is not None and isinstance(extra, dict):
        for k in ("uvar97", "uvar"):
            if k in extra and extra[k] is not None:
                u = np.asarray(extra[k], dtype=float)
                if u.ndim == 1:
                    uvar_arr[0, :min(5, len(u))] = u[:min(5, len(u))]
                elif u.ndim == 2:
                    uvar_arr[:min(nel, len(u)), :min(5, u.shape[1])] = u[:min(nel, len(u)), :min(5, u.shape[1])]
                break

    if epsp is not None:
        ep_in = np.atleast_1d(epsp).astype(float)
        uvar_arr[:min(nel, len(ep_in)), 4] = ep_in[:nel]

    off_arr = np.ones(nel, dtype=float)
    if extra is not None and isinstance(extra, dict) and "off" in extra:
        o = np.atleast_1d(extra["off"]).astype(float)
        off_arr[:min(nel, len(o))] = o[:nel]

    time = float(extra.get("time", 0.0)) if (extra is not None and isinstance(extra, dict)) else 0.0

    sig_out = np.zeros_like(sig_arr)
    epsp_out = np.zeros(nel, dtype=float)
    c_out = np.zeros(nel, dtype=float)

    for i in range(nel):
        s_i, ep_i, u_i, c_i = _shell_update_single(p, sig_arr[i], deps_arr[i], uvar_arr[i], off=off_arr[i], time=time)
        sig_out[i] = s_i
        epsp_out[i] = ep_i
        uvar_arr[i] = u_i
        c_out[i] = c_i

    if extra is not None and isinstance(extra, dict):
        extra["uvar97"] = uvar_arr
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
        [p.a11_2d, p.a21_2d, 0.0],
        [p.a21_2d, p.a11_2d, 0.0],
        [0.0, 0.0, p.g],
    ], dtype=float)

    n = 1
    if sig is not None and np.ndim(sig) >= 2:
        n = len(sig)
    return np.broadcast_to(c_el, (n, 3, 3)).copy()


consistent_shell_tangent = shell_tangent
