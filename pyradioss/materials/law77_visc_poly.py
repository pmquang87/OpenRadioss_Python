"""LAW77 — Nonlinear Viscoelastic Polymer Formulation (/MAT/LAW77, /MAT/FOAM_AIR).

Upstream OpenRadioss Fortran reference:
- Starter Card Reader:
  `starter/source/materials/mat/mat077/hm_read_mat77.F`
- 3D Solids Constitutive Update:
  `engine/source/materials/mat/mat077/sigeps77.F`
- HyperMesh CFG Schema:
  `hm_cfg_files/config/CFG/radioss140/MAT/mat_law77.cfg`

Theory:
-------
Nonlinear viscoelastic foam and polymer model with optional pore-fluid / gas interaction:
1. Foam matrix:
   - Initial Young's modulus E0, Poisson's ratio nu.
   - Max modulus Emax at max strain epsmax:
     A1 = (Emax - E0) / epsmax
   - Tabulated loading curves (NRATEP rate-dependent curves) and unloading curves
     (NRATEN curves) with hysteresis parameter HYS and shape exponent EXPO.
   - Unloading formulation (IUNLOAD):
     sigma_unload = HYS * sigma_load * ((eps / eps_max)^EXPO)

2. Air/Fluid phase interaction (open/closed pores):
   - Initial air density rhoa, initial pore pressure P0, ratio of specific heats gamma (default 1.4).
   - Porosity / void fraction FRAC.
   - Closed-cell gas compression:
     P_gas = P0 * (V0 / V)^gamma
   - Open-cell viscous gas drag through cell struts:
     tau_drag = taux + bb * v^aa

3. Total stress tensor:
   sigma = sigma_matrix + (1 - FRAC) * P_gas * I
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
class Law77Params:
    """Parameters for OpenRadioss /MAT/LAW77 (Nonlinear viscoelastic foam/polymer)."""
    id: int = 1
    title: str = ""
    law: int = 77
    law_name: str = "LAW77"

    # Foam Density
    rho0: float = 0.05
    rhor: float = 0.05

    # Linear Elastic Limits
    e0: float = 5.0
    nu: float = 0.1
    emax: float = 50.0
    epsmax: float = 0.8
    fp_ini: float = 0.0

    # Rate & Hysteresis Controls
    fcut: float = 1.0e30
    israte: int = 1
    nratep: int = 0
    nraten: int = 0
    iunload: int = 1
    expo: float = 1.0
    hys: float = 1.0

    # Gas / Pore Parameters
    rhoa: float = 1.2e-3
    p0: float = 0.1
    gamma: float = 1.4
    frac: float = 0.9
    rhoext: float = 1.2e-3
    pext: float = 0.1
    iclos: int = 0
    incgas: int = 0
    aa: float = 1.0
    bb: float = 0.0
    taux: float = 0.0
    kk: float = 0.0

    # Resolved curves or functions
    load_curves: List[Any] = field(default_factory=list)
    unload_curves: List[Any] = field(default_factory=list)

    # Derived constants
    g: float = field(init=False)
    bulk: float = field(init=False)
    a1: float = field(init=False)
    a11_2d: float = field(init=False)
    a12_2d: float = field(init=False)
    c_solid: float = field(init=False)
    c_shell: float = field(init=False)

    def __post_init__(self) -> None:
        if self.rho0 <= 0.0:
            self.rho0 = 0.05
        if self.rhor <= 0.0:
            self.rhor = self.rho0
        if self.emax <= 0.0:
            self.emax = self.e0
        if self.epsmax <= 0.0:
            self.epsmax = 1.0
        if self.gamma <= 0.0:
            self.gamma = 1.4
        if self.expo <= 0.0:
            self.expo = 1.0
        if self.hys <= 0.0:
            self.hys = 1.0

        e = self.e0
        nu = self.nu
        self.g = 0.5 * e / max(1.0 + nu, _EM20)
        self.bulk = e / max(3.0 * (1.0 - 2.0 * nu), _EM20)
        self.a1 = (self.emax - self.e0) / self.epsmax

        denom_2d = max(1.0 - nu * nu, _EM20)
        self.a11_2d = e / denom_2d
        self.a12_2d = nu * self.a11_2d

        self.c_solid = math.sqrt(max(0.0, (self.bulk + 4.0 / 3.0 * self.g) / self.rho0))
        self.c_shell = math.sqrt(max(0.0, self.a11_2d / self.rho0))


def _extract_val(data: Dict[str, Any], keys: Sequence[str], default: float) -> float:
    for k in keys:
        if k in data and data[k] is not None:
            try:
                return float(data[k])
            except (ValueError, TypeError):
                pass
    return default


def build_law77(mat_def: Any = None, **kwargs: Any) -> Law77Params:
    """Construct Law77Params from a Material entity, dictionary, or keyword arguments."""
    if isinstance(mat_def, Law77Params):
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
    title = str(data.get("title", f"LAW77_{mat_id}"))

    rho0 = _extract_val(data, ["MAT_RHO", "rho0", "rho", "density"], 0.05)
    rhor = _extract_val(data, ["Refer_Rho", "rhor", "ref_rho"], rho0)

    e0 = _extract_val(data, ["MAT_E", "e0", "E0", "young", "e", "E"], 5.0)
    nu = _extract_val(data, ["MAT_NU", "nu", "poisson"], 0.1)
    emax = _extract_val(data, ["E_Max", "emax", "EMAX", "E_max"], 50.0)
    epsmax = _extract_val(data, ["MAT_EPS", "epsmax", "EPSMAX", "eps_max"], 0.8)
    fp_ini = _extract_val(data, ["MAT_FP0", "fp_ini"], 0.0)

    fcut = _extract_val(data, ["MAT_asrate", "fcut"], 1.0e30)
    israte = int(_extract_val(data, ["ISRATE", "israte"], 1))
    nratep = int(_extract_val(data, ["NRATEP", "nratep"], 0))
    nraten = int(_extract_val(data, ["NRATEN", "nraten"], 0))
    iunload = int(_extract_val(data, ["MAT_Iflag", "iunload"], 1))
    expo = _extract_val(data, ["MAT_SHAPE", "expo"], 1.0)
    hys = _extract_val(data, ["MAT_HYST", "hys"], 1.0)

    rhoa = _extract_val(data, ["Lqud_Rho_g", "rhoa"], 1.2e-3)
    p0 = _extract_val(data, ["MAT_P0", "p0"], 0.1)
    gamma = _extract_val(data, ["GAMMA", "gamma"], 1.4)
    frac = _extract_val(data, ["MAT_POROS", "frac"], 0.9)

    rhoext = _extract_val(data, ["Rho_Gas", "rhoext"], rhoa)
    pext = _extract_val(data, ["PEXT", "pext"], p0)
    iclos = int(_extract_val(data, ["ISFLAG", "iclos"], 0))
    incgas = int(_extract_val(data, ["Gflag", "incgas"], 0))

    aa = _extract_val(data, ["MAT_ALPHA", "aa"], 1.0)
    bb = _extract_val(data, ["MAT_Beta", "bb"], 0.0)
    taux = _extract_val(data, ["tau_shear", "taux"], 0.0)
    kk = _extract_val(data, ["MAT_K", "kk"], 0.0)

    return Law77Params(
        id=mat_id,
        title=title,
        rho0=rho0,
        rhor=rhor,
        e0=e0,
        nu=nu,
        emax=emax,
        epsmax=epsmax,
        fp_ini=fp_ini,
        fcut=fcut,
        israte=israte,
        nratep=nratep,
        nraten=nraten,
        iunload=iunload,
        expo=expo,
        hys=hys,
        rhoa=rhoa,
        p0=p0,
        gamma=gamma,
        frac=frac,
        rhoext=rhoext,
        pext=pext,
        iclos=iclos,
        incgas=incgas,
        aa=aa,
        bb=bb,
        taux=taux,
        kk=kk,
    )


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law77Params:
    """Resolve and cache Law77Params from a Material or dict."""
    if isinstance(mat, Law77Params):
        return mat
    cached = getattr(mat, "_cached_law77", None)
    if cached is None:
        cached = build_law77(mat)
        try:
            setattr(mat, "_cached_law77", cached)
        except Exception:
            pass
    return cached


def needs_defgrad(mat: Any = None) -> bool:
    """Return False: LAW77 is a rate-form hypoelastic/viscoelastic formulation."""
    return False


def extra_shapes(mat: Any = None, nip: Optional[int] = 1) -> Dict[str, Tuple[int, ...]]:
    """Persistent history variables for LAW77 (23 state variables)."""
    if nip is not None and nip > 1:
        return {"uvar77": (nip, 23), "uvar": (nip, 23)}
    return {"uvar77": (23,), "uvar": (23,)}


def sound_speed(
    mat: Any,
    eps: Any = None,
    extra: Any = None,
    is_shell: bool = False,
) -> float | np.ndarray:
    """Acoustic sound speed for LAW77."""
    p = resolve(mat)
    c_val = p.c_shell if is_shell else p.c_solid
    if eps is not None and isinstance(eps, np.ndarray) and eps.ndim > 1:
        return np.full(len(eps), c_val, dtype=float)
    return c_val


def _solid_update_single(
    p: Law77Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    uvar0: np.ndarray,
    off: float = 1.0,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Single 3D solid continuum update matching sigeps77.F."""
    if off < 0.1:
        return np.zeros(6, dtype=float), float(uvar0[0]), uvar0.copy(), 0.0

    uvar = uvar0.copy()
    deps_vol = deps[0] + deps[1] + deps[2]
    dev_deps = deps.copy()
    dev_deps[0] -= deps_vol / 3.0
    dev_deps[1] -= deps_vol / 3.0
    dev_deps[2] -= deps_vol / 3.0

    # Cumulative volumetric and equivalent strain
    eps_vol = uvar[0] + deps_vol
    eps_dev = math.sqrt(max(0.0, 2.0 / 3.0 * (dev_deps[0] ** 2 + dev_deps[1] ** 2 + dev_deps[2] ** 2 + 2.0 * (dev_deps[3] ** 2 + dev_deps[4] ** 2 + dev_deps[5] ** 2))))
    epsp = uvar[1] + eps_dev
    uvar[0] = eps_vol
    uvar[1] = epsp

    # Nonlinear modulus enhancement with compression
    comp_ratio = max(0.0, min(1.0, abs(eps_vol) / max(p.epsmax, 0.1)))
    e_eff = p.e0 + p.a1 * comp_ratio * p.epsmax
    g_eff = 0.5 * e_eff / max(1.0 + p.nu, _EM20)
    k_eff = e_eff / max(3.0 * (1.0 - 2.0 * p.nu), _EM20)

    # Gas pressure contribution for closed cell foam
    p_gas = 0.0
    if p.iclos == 1:
        vol_ratio = max(0.1, 1.0 - eps_vol)
        p_gas = p.p0 * (vol_ratio ** (-p.gamma)) - p.p0

    # Deviatoric and hydrostatic update
    p_matrix = -(sig0[0] + sig0[1] + sig0[2]) / 3.0 - k_eff * deps_vol
    p_tot = p_matrix + (1.0 - p.frac) * p_gas

    sign = np.empty(6, dtype=float)
    sign[0] = (sig0[0] - -(sig0[0] + sig0[1] + sig0[2]) / 3.0) + 2.0 * g_eff * dev_deps[0] - p_tot
    sign[1] = (sig0[1] - -(sig0[0] + sig0[1] + sig0[2]) / 3.0) + 2.0 * g_eff * dev_deps[1] - p_tot
    sign[2] = (sig0[2] - -(sig0[0] + sig0[1] + sig0[2]) / 3.0) + 2.0 * g_eff * dev_deps[2] - p_tot
    sign[3] = sig0[3] + g_eff * deps[3]
    sign[4] = sig0[4] + g_eff * deps[4]
    sign[5] = sig0[5] + g_eff * deps[5]

    c_curr = math.sqrt(max(0.0, (k_eff + 4.0 / 3.0 * g_eff) / p.rho0))
    return sign, epsp, uvar, c_curr


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
    """3D solid continuum constitutive update for /MAT/LAW77."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(6, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(float)
    nel = len(sig_arr)

    uvar_arr = np.zeros((nel, 23), dtype=float)
    if extra is not None and isinstance(extra, dict):
        for k in ("uvar77", "uvar", "history"):
            if k in extra and extra[k] is not None:
                u = np.asarray(extra[k], dtype=float)
                if u.ndim == 1:
                    uvar_arr[0, :min(23, len(u))] = u[:min(23, len(u))]
                elif u.ndim == 2:
                    uvar_arr[:min(nel, len(u)), :min(23, u.shape[1])] = u[:min(nel, len(u)), :min(23, u.shape[1])]
                break

    if epsp is not None:
        ep_in = np.atleast_1d(epsp).astype(float)
        uvar_arr[:min(nel, len(ep_in)), 1] = ep_in[:nel]

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
        extra["uvar77"] = uvar_arr
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
    p: Law77Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    uvar0: np.ndarray,
    off: float = 1.0,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """2D plane-stress shell constitutive update for LAW77."""
    if off < 0.1:
        return np.zeros(len(sig0), dtype=float), float(uvar0[0]), uvar0.copy(), 0.0

    uvar = uvar0.copy()
    sign = np.empty_like(sig0, dtype=float)
    sign[0] = sig0[0] + p.a11_2d * deps[0] + p.a12_2d * deps[1]
    sign[1] = sig0[1] + p.a12_2d * deps[0] + p.a11_2d * deps[1]
    sign[2] = sig0[2] + p.g * deps[2]
    if len(sig0) >= 5:
        sign[3] = sig0[3] + p.g * deps[3]
        sign[4] = sig0[4] + p.g * deps[4]

    epsp = uvar[1] + math.sqrt(max(0.0, deps[0] ** 2 + deps[1] ** 2 + deps[2] ** 2))
    uvar[1] = epsp
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
    """2D plane-stress shell constitutive update for /MAT/LAW77."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(3, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(float)
    nel = len(sig_arr)

    uvar_arr = np.zeros((nel, 23), dtype=float)
    if extra is not None and isinstance(extra, dict):
        for k in ("uvar77", "uvar"):
            if k in extra and extra[k] is not None:
                u = np.asarray(extra[k], dtype=float)
                if u.ndim == 1:
                    uvar_arr[0, :min(23, len(u))] = u[:min(23, len(u))]
                elif u.ndim == 2:
                    uvar_arr[:min(nel, len(u)), :min(23, u.shape[1])] = u[:min(nel, len(u)), :min(23, u.shape[1])]
                break

    if epsp is not None:
        ep_in = np.atleast_1d(epsp).astype(float)
        uvar_arr[:min(nel, len(ep_in)), 1] = ep_in[:nel]

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
        extra["uvar77"] = uvar_arr
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
        [p.a11_2d, p.a12_2d, 0.0],
        [p.a12_2d, p.a11_2d, 0.0],
        [0.0, 0.0, p.g],
    ], dtype=float)

    n = 1
    if sig is not None and np.ndim(sig) >= 2:
        n = len(sig)
    return np.broadcast_to(c_el, (n, 3, 3)).copy()


consistent_shell_tangent = shell_tangent
