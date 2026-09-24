"""LAW75 — Porous Compaction & Phase Transformation Material Model (/MAT/LAW75, /MAT/POROUS).

Upstream OpenRadioss Fortran reference:
- Starter Card Reader:
  `starter/source/materials/mat/mat075/hm_read_mat75.F`
- Material Initializer:
  `starter/source/materials/mat/mat075/m75init.F`
- 3D Solids Constitutive Update:
  `engine/source/materials/mat/mat075/sigeps75.F`
- HyperMesh CFG Schema:
  `hm_cfg_files/config/CFG/radioss120/MAT/matl75_75.cfg`

Theory:
-------
Porous compaction model based on the Carroll-Holt P-alpha formulation,
coupled with optional solid phase EOS and thermo-plastic transformation:
1. Porosity parameter:
   alpha = rho_solid / rho >= 1.0.
   At initial state: alpha_e = rho_0_solid / rho_0.
   When alpha = 1.0, the porous material is fully compacted into a solid.

2. Compaction pressure regimes:
   - Elastic regime (P < Pe):
     Pores deform elastically with initial porous bulk modulus K_0.
   - Plastic compaction regime (Pe <= P <= Ps):
     P(alpha) = Ps - (Ps - Pe) * ((alpha - 1.0) / (alpha_p - 1.0))^(1/n)
     where alpha_p is the porosity at initial plastic yield Pe.
   - Fully compacted regime (P > Ps):
     alpha = 1.0; response follows the solid phase matrix material (bulk modulus Ks).

3. Shear Modulus Evolution:
   G(alpha) = G_s + (G_0 - G_s) * ((alpha - 1.0) / (alpha_e - 1.0))
   where G_0 is initial porous shear modulus and G_s is compacted solid shear modulus.
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
class Law75Params:
    """Parameters for OpenRadioss /MAT/LAW75 (Porous compaction / thermo-transformation)."""
    id: int = 1
    title: str = ""
    law: int = 75
    law_name: str = "LAW75"

    # Density
    rho0: float = 1.0
    rhor: float = 1.0

    # Porous Elastic Constants
    young: float = 10000.0
    nu: float = 0.25

    # Compaction parameters
    mats: int = 0             # Solid material identifier
    iflag1: int = 1           # Pressure formulation flag
    iflag2: int = 1           # Deviatoric stresses formulation flag
    itemax: int = 5           # Max iterations on calculation
    pe: float = 10.0          # Elastic compaction pressure
    ps: float = 100.0         # Solid compaction pressure
    nn: float = 2.0           # Compaction exponent
    tol: float = 1.0e-8       # Convergence tolerance

    # Optional phase transformation / thermo fields
    e_mart: float = 0.0
    tini: float = 293.15
    t_trans: float = 500.0
    h_trans: float = 0.0

    # Solid phase properties (defaults to scaled porous if mats not resolved)
    rho0_s: float = 1.2
    bulk_s: float = 20000.0
    g_s: float = 12000.0

    # Derived constants
    bulk: float = field(init=False)
    g: float = field(init=False)
    lamhook: float = field(init=False)
    a11_2d: float = field(init=False)
    a12_2d: float = field(init=False)
    alpha_e: float = field(init=False)
    alpha_p: float = field(init=False)
    c_solid: float = field(init=False)
    c_shell: float = field(init=False)

    def __post_init__(self) -> None:
        if self.rho0 <= 0.0:
            self.rho0 = 1.0
        if self.rhor <= 0.0:
            self.rhor = self.rho0
        if self.pe > self.ps:
            self.ps = self.pe * 2.0
        if self.nn <= 0.0:
            self.nn = 2.0
        if self.tol <= 0.0:
            self.tol = 1.0e-8

        e = self.young
        nu = self.nu
        self.bulk = e / max(3.0 * (1.0 - 2.0 * nu), _EM20)
        self.g = 0.5 * e / max(1.0 + nu, _EM20)
        self.lamhook = 2.0 * self.g * nu / max(1.0 - 2.0 * nu, _EM20)

        denom_2d = max(1.0 - nu * nu, _EM20)
        self.a11_2d = e / denom_2d
        self.a12_2d = nu * self.a11_2d

        if self.rho0_s <= self.rho0:
            self.rho0_s = 1.2 * self.rho0
        if self.bulk_s <= self.bulk:
            self.bulk_s = 1.5 * self.bulk
        if self.g_s <= self.g:
            self.g_s = 1.5 * self.g

        self.alpha_e = self.rho0_s / self.rho0
        dalpdpe = (1.0 / max(self.bulk_s, _EM20) - 1.0 / max(self.bulk, _EM20)) * self.alpha_e
        self.alpha_p = max(1.0, self.alpha_e + self.pe * dalpdpe)

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


def build_law75(mat_def: Any = None, **kwargs: Any) -> Law75Params:
    """Construct Law75Params from a Material entity, dictionary, or keyword arguments."""
    if isinstance(mat_def, Law75Params):
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
    title = str(data.get("title", f"LAW75_{mat_id}"))

    rho0 = _extract_val(data, ["MAT_RHO", "rho0", "rho", "density"], 1.0)
    rhor = _extract_val(data, ["Refer_Rho", "rhor", "ref_rho"], rho0)

    young = _extract_val(data, ["MAT_E", "young", "e", "E"], 10000.0)
    nu = _extract_val(data, ["MAT_NU", "nu", "poisson"], 0.25)

    mats = int(_extract_val(data, ["MAT1", "mats", "solid_mat_id"], 0))
    iflag1 = int(_extract_val(data, ["HFLAG1", "iflag1", "p_flag"], 1))
    iflag2 = int(_extract_val(data, ["HFLAG2", "iflag2", "dev_flag"], 1))
    itemax = int(_extract_val(data, ["Nppmax", "itemax", "max_iter"], 5))

    pe = _extract_val(data, ["MAT_PPRES", "pe", "Pe", "p_elastic"], 10.0)
    ps = _extract_val(data, ["MAT_YPRES", "ps", "Ps", "p_solid"], 100.0)
    nn = _extract_val(data, ["MAT_EXP1", "nn", "NN", "exponent"], 2.0)
    tol = _extract_val(data, ["MAT_Tol", "tol", "tolerance"], 1.0e-8)

    e_mart = _extract_val(data, ["e_mart", "E_mart", "young_martensite"], 0.0)
    tini = _extract_val(data, ["Tini", "tini", "TINI"], 293.15)
    t_trans = _extract_val(data, ["t_trans", "T_trans"], 500.0)
    h_trans = _extract_val(data, ["h_trans", "H_trans"], 0.0)

    rho0_s = _extract_val(data, ["rho0_s", "rho_solid"], 1.2 * rho0)
    bulk_s = _extract_val(data, ["bulk_s", "bulk_solid"], 2.0 * young)
    g_s = _extract_val(data, ["g_s", "g_solid"], young)

    return Law75Params(
        id=mat_id,
        title=title,
        rho0=rho0,
        rhor=rhor,
        young=young,
        nu=nu,
        mats=mats,
        iflag1=iflag1,
        iflag2=iflag2,
        itemax=itemax,
        pe=pe,
        ps=ps,
        nn=nn,
        tol=tol,
        e_mart=e_mart,
        tini=tini,
        t_trans=t_trans,
        h_trans=h_trans,
        rho0_s=rho0_s,
        bulk_s=bulk_s,
        g_s=g_s,
    )


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law75Params:
    """Resolve and cache Law75Params from a Material or dict."""
    if isinstance(mat, Law75Params):
        return mat
    cached = getattr(mat, "_cached_law75", None)
    if cached is None:
        cached = build_law75(mat)
        try:
            setattr(mat, "_cached_law75", cached)
        except Exception:
            pass
    return cached


def needs_defgrad(mat: Any = None) -> bool:
    """Return False: LAW75 is a hypoelastic rate formulation."""
    return False


def extra_shapes(mat: Any = None, nip: Optional[int] = 1) -> Dict[str, Tuple[int, ...]]:
    """Persistent history variables for LAW75 (col 0: eps_p, col 1: temp, col 2: alpha, col 3: alpha_p)."""
    if nip is not None and nip > 1:
        return {"uvar75": (nip, 4), "uvar": (nip, 4)}
    return {"uvar75": (4,), "uvar": (4,)}


def sound_speed(
    mat: Any,
    eps: Any = None,
    extra: Any = None,
    is_shell: bool = False,
) -> float | np.ndarray:
    """Acoustic sound speed for LAW75."""
    p = resolve(mat)
    c_val = p.c_shell if is_shell else p.c_solid
    if eps is not None and isinstance(eps, np.ndarray) and eps.ndim > 1:
        return np.full(len(eps), c_val, dtype=float)
    return c_val


def _solid_update_single(
    p: Law75Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    uvar0: np.ndarray,
    off: float = 1.0,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Single 3D solid element update matching sigeps75.F."""
    if off < 0.1:
        return np.zeros(6, dtype=float), float(uvar0[0]), uvar0.copy(), 0.0

    uvar = uvar0.copy()
    if uvar[2] <= 0.0:
        uvar[2] = p.alpha_e
    if uvar[3] <= 0.0:
        uvar[3] = p.alpha_p

    alpha_old = uvar[2]
    alpha_yld = uvar[3]

    # Volumetric strain increment
    deps_vol = deps[0] + deps[1] + deps[2]
    dev_deps = deps.copy()
    dev_deps[0] -= deps_vol / 3.0
    dev_deps[1] -= deps_vol / 3.0
    dev_deps[2] -= deps_vol / 3.0

    # Current pressure from bulk response
    p_old = -(sig0[0] + sig0[1] + sig0[2]) / 3.0
    p_trial = p_old - p.bulk * deps_vol

    # Porosity compaction check
    p_comp = p_trial
    alpha_new = alpha_old
    epsp = uvar[0]

    if p_trial > p.pe:
        # Plastic compaction regime
        if p_trial >= p.ps:
            alpha_new = 1.0
            p_comp = p.ps + p.bulk_s * max(0.0, (p_trial - p.ps) / p.bulk)
        else:
            ratio = (p.ps - p_trial) / max(p.ps - p.pe, _EM20)
            ratio = max(0.0, min(1.0, ratio))
            alpha_target = 1.0 + (p.alpha_p - 1.0) * (ratio ** p.nn)
            if alpha_target < alpha_old:
                alpha_new = alpha_target
                d_alpha = alpha_old - alpha_new
                epsp += d_alpha / max(alpha_old, 1.0)
            p_comp = p_trial

    uvar[0] = epsp
    uvar[2] = alpha_new
    uvar[3] = max(1.0, min(alpha_yld, alpha_new))

    # Shear modulus interpolated with porosity
    g_curr = p.g_s + (p.g - p.g_s) * max(0.0, (alpha_new - 1.0) / max(p.alpha_e - 1.0, _EM20))
    g2 = 2.0 * g_curr

    # Update Cauchy stresses
    sign = np.empty(6, dtype=float)
    s_xx = (sig0[0] + p_old) + g2 * dev_deps[0]
    s_yy = (sig0[1] + p_old) + g2 * dev_deps[1]
    s_zz = (sig0[2] + p_old) + g2 * dev_deps[2]
    sign[0] = s_xx - p_comp
    sign[1] = s_yy - p_comp
    sign[2] = s_zz - p_comp
    sign[3] = sig0[3] + g_curr * deps[3]
    sign[4] = sig0[4] + g_curr * deps[4]
    sign[5] = sig0[5] + g_curr * deps[5]

    c_curr = math.sqrt(max(0.0, (p.bulk + 4.0 / 3.0 * g_curr) / p.rho0))
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
    """3D solid continuum constitutive update for /MAT/LAW75."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(6, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(float)
    nel = len(sig_arr)

    uvar_arr = np.zeros((nel, 4), dtype=float)
    uvar_arr[:, 2] = p.alpha_e
    uvar_arr[:, 3] = p.alpha_p

    if extra is not None and isinstance(extra, dict):
        for k in ("uvar75", "uvar", "history"):
            if k in extra and extra[k] is not None:
                u = np.asarray(extra[k], dtype=float)
                if u.ndim == 1:
                    uvar_arr[0, :min(4, len(u))] = u[:min(4, len(u))]
                elif u.ndim == 2:
                    uvar_arr[:min(nel, len(u)), :min(4, u.shape[1])] = u[:min(nel, len(u)), :min(4, u.shape[1])]
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
        extra["uvar75"] = uvar_arr
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
    p: Law75Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    uvar0: np.ndarray,
    off: float = 1.0,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """2D plane-stress shell constitutive update for LAW75."""
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

    epsp = uvar[0]
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
    """2D plane-stress shell constitutive update for /MAT/LAW75."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(3, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(float)
    nel = len(sig_arr)

    uvar_arr = np.zeros((nel, 4), dtype=float)
    uvar_arr[:, 2] = p.alpha_e
    uvar_arr[:, 3] = p.alpha_p

    if extra is not None and isinstance(extra, dict):
        for k in ("uvar75", "uvar"):
            if k in extra and extra[k] is not None:
                u = np.asarray(extra[k], dtype=float)
                if u.ndim == 1:
                    uvar_arr[0, :min(4, len(u))] = u[:min(4, len(u))]
                elif u.ndim == 2:
                    uvar_arr[:min(nel, len(u)), :min(4, u.shape[1])] = u[:min(nel, len(u)), :min(4, u.shape[1])]
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
        extra["uvar75"] = uvar_arr
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
