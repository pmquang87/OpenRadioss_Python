"""LAW78 — Yoshida-Uemori Two-Surface Cyclic Plasticity (/MAT/LAW78, /MAT/YOSHIDA).

Upstream OpenRadioss Fortran reference:
- Starter Card Reader:
  `starter/source/materials/mat/mat078/hm_read_mat78.F`
- 3D Solids Constitutive Update:
  `engine/source/materials/mat/mat078/sigeps78.F`
- 2D Shells Constitutive Update:
  `engine/source/materials/mat/mat078/sigeps78c.F`
- HyperMesh CFG Schema:
  `hm_cfg_files/config/CFG/radioss140/MAT/matl78_78.cfg`

Theory:
-------
The Yoshida-Uemori (2002) model describes large-strain cyclic plasticity,
Bauschinger effect, work hardening stagnation, and cyclic elastic modulus degradation:
1. Two-surface representation:
   - Yield surface:
     f(sigma - alpha) - Y = 0
     where alpha is the backstress center and Y is the yield surface radius (constant size).
   - Bounding surface:
     F(sigma - beta) - (B + R) = 0
     where beta is the center of the bounding surface, B is initial size, and R is isotropic hardening.
   - Non-isotropic hardening stagnation surface:
     g_sigma(beta - q) - r = 0
     where q is the center and r is the radius of the stagnation boundary.

2. Kinematic hardening evolution:
   Relative backstress alpha_* = alpha - beta:
   d alpha_* = C * [ (a / Y) * (sigma - alpha) - sqrt(a / alpha_*_bar) * alpha_* ] * d eps_p
   d beta = m * [ (2/3) * b * n - beta ] * d eps_p
   where a = B + R - Y.

3. Isotropic hardening of bounding surface:
   d R = m * (R_sat - R) * d eps_p

4. Cyclic elastic modulus degradation:
   E(eps_p) = E_0 - (E_0 - E_inf) * (1 - exp(-xi * eps_p))

5. Anisotropy:
   Hill 1948 (IPLAS = 1) or Barlat 1989 (IPLAS = 2).
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
class Law78Params:
    """Parameters for OpenRadioss /MAT/LAW78 (Yoshida-Uemori two-surface model)."""
    id: int = 1
    title: str = ""
    law: int = 78
    law_name: str = "LAW78"

    # Density
    rho0: float = 7.8e-3
    rhor: float = 7.8e-3

    # Elastic Constants
    young: float = 210000.0
    nu: float = 0.3
    einf: float = 160000.0
    coe: float = 20.0          # xi: modulus degradation rate
    opte: int = 0              # 1 if modulus degradation active

    # Yield & Bounding Surface Parameters
    yield_stress: float = 200.0  # Y: yield surface size
    byu: float = 150.0          # B: initial bounding surface size
    cyu: float = 1000.0         # C: kinematic hardening rate (inner surface)
    hyu: float = 0.5            # h: work hardening stagnation parameter [0, 1]
    bsat: float = 300.0         # B_sat: saturated bounding surface size
    myu: float = 20.0           # m: kinematic hardening rate (bounding surface)
    rsat: float = 100.0         # R_sat: isotropic hardening saturation

    # Additional kinematic hardening controls
    c1_kh: float = 1000.0
    optr: int = 0
    cst: float = 0.0
    cstt: float = 0.0

    # Anisotropy
    r00: float = 1.0
    r45: float = 1.0
    r90: float = 1.0
    mexp: float = 0.0
    iplas: int = 1             # 1: Hill48, 2: Barlat89

    # Derived constants
    bulk: float = field(init=False)
    g: float = field(init=False)
    lamhook: float = field(init=False)
    a11_2d: float = field(init=False)
    a12_2d: float = field(init=False)
    c_solid: float = field(init=False)
    c_shell: float = field(init=False)

    def __post_init__(self) -> None:
        if self.rho0 <= 0.0:
            self.rho0 = 7.8e-3
        if self.rhor <= 0.0:
            self.rhor = self.rho0
        if self.bsat < self.yield_stress:
            self.bsat = self.yield_stress
        if self.c1_kh <= self.cyu:
            self.c1_kh = self.cyu

        e = self.young
        nu = self.nu
        self.bulk = e / max(3.0 * (1.0 - 2.0 * nu), _EM20)
        self.g = 0.5 * e / max(1.0 + nu, _EM20)
        self.lamhook = 2.0 * self.g * nu / max(1.0 - 2.0 * nu, _EM20)

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


def build_law78(mat_def: Any = None, **kwargs: Any) -> Law78Params:
    """Construct Law78Params from a Material entity, dictionary, or keyword arguments."""
    if isinstance(mat_def, Law78Params):
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
    title = str(data.get("title", f"LAW78_{mat_id}"))

    rho0 = _extract_val(data, ["MAT_RHO", "rho0", "rho", "density"], 7.8e-3)
    rhor = _extract_val(data, ["Refer_Rho", "rhor", "ref_rho"], rho0)

    young = _extract_val(data, ["MAT_E", "young", "e", "E"], 210000.0)
    nu = _extract_val(data, ["MAT_NU", "nu", "poisson"], 0.3)
    einf = _extract_val(data, ["MAT_EA", "einf", "E_inf"], 160000.0)
    coe = _extract_val(data, ["MAT_CE", "coe", "xi"], 20.0)
    opte = int(_extract_val(data, ["MAT_fct_IDE", "opte"], 0))

    sigy = _extract_val(data, ["MAT_SIGY", "yield_stress", "sigy", "Y"], 200.0)
    byu = _extract_val(data, ["MAT_BSAT", "byu", "B"], 150.0)
    cyu = _extract_val(data, ["MAT_HARD", "cyu", "C"], 1000.0)
    hyu = _extract_val(data, ["MAT_HYST", "hyu", "h"], 0.5)
    bsat = _extract_val(data, ["MAT_B", "bsat", "B_sat"], 300.0)
    myu = _extract_val(data, ["MAT_M", "myu", "m"], 20.0)
    rsat = _extract_val(data, ["MAT_RSAT", "rsat", "R_SAT", "R_sat"], 100.0)

    c1_kh = _extract_val(data, ["MAT_C1KH", "c1_kh"], cyu)
    optr = int(_extract_val(data, ["MAT_OptR", "optr"], 0))
    cst = _extract_val(data, ["C1", "cst"], 0.0)
    cstt = _extract_val(data, ["C2", "cstt"], 0.0)

    r00 = _extract_val(data, ["MAT_R00", "r00"], 1.0)
    r45 = _extract_val(data, ["MAT_R45", "r45"], 1.0)
    r90 = _extract_val(data, ["MAT_R90", "r90"], 1.0)
    mexp = _extract_val(data, ["MAT_MEXP", "mexp"], 0.0)
    iplas = int(_extract_val(data, ["MAT_IPLAS", "iplas"], 1))

    return Law78Params(
        id=mat_id,
        title=title,
        rho0=rho0,
        rhor=rhor,
        young=young,
        nu=nu,
        einf=einf,
        coe=coe,
        opte=opte,
        yield_stress=sigy,
        byu=byu,
        cyu=cyu,
        hyu=hyu,
        bsat=bsat,
        myu=myu,
        rsat=rsat,
        c1_kh=c1_kh,
        optr=optr,
        cst=cst,
        cstt=cstt,
        r00=r00,
        r45=r45,
        r90=r90,
        mexp=mexp,
        iplas=iplas,
    )


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law78Params:
    """Resolve and cache Law78Params from a Material or dict."""
    if isinstance(mat, Law78Params):
        return mat
    cached = getattr(mat, "_cached_law78", None)
    if cached is None:
        cached = build_law78(mat)
        try:
            setattr(mat, "_cached_law78", cached)
        except Exception:
            pass
    return cached


def needs_defgrad(mat: Any = None) -> bool:
    """Return False: LAW78 is an incremental rate formulation."""
    return False


def extra_shapes(mat: Any = None, nip: Optional[int] = 1) -> Dict[str, Tuple[int, ...]]:
    """Persistent history variables for LAW78 (6 state variables + backstresses)."""
    if nip is not None and nip > 1:
        return {"uvar78": (nip, 6), "uvar": (nip, 6)}
    return {"uvar78": (6,), "uvar": (6,)}


def sound_speed(
    mat: Any,
    eps: Any = None,
    extra: Any = None,
    is_shell: bool = False,
) -> float | np.ndarray:
    """Acoustic sound speed for LAW78."""
    p = resolve(mat)
    c_val = p.c_shell if is_shell else p.c_solid
    if eps is not None and isinstance(eps, np.ndarray) and eps.ndim > 1:
        return np.full(len(eps), c_val, dtype=float)
    return c_val


def _solid_update_single(
    p: Law78Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    uvar0: np.ndarray,
    alpha_in: Optional[np.ndarray] = None,
    off: float = 1.0,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, float]:
    """Single 3D solid element update matching sigeps78.F."""
    if off < 0.1:
        return np.zeros(6, dtype=float), float(uvar0[4]), uvar0.copy(), np.zeros(6, dtype=float), 0.0

    uvar = uvar0.copy()
    alpha = np.zeros(6, dtype=float) if alpha_in is None else alpha_in.copy()

    # Dynamic elastic modulus degradation
    epsp_tot = uvar[4]
    if p.opte == 1 or p.coe > 0.0:
        e_curr = p.young - (p.young - p.einf) * (1.0 - math.exp(-p.coe * epsp_tot))
    else:
        e_curr = p.young
    g_curr = 0.5 * e_curr / max(1.0 + p.nu, _EM20)
    bulk_curr = e_curr / max(3.0 * (1.0 - 2.0 * p.nu), _EM20)
    lam_curr = 2.0 * g_curr * p.nu / max(1.0 - 2.0 * p.nu, _EM20)

    # Elastic trial stress
    deps_vol = deps[0] + deps[1] + deps[2]
    dav = deps_vol * lam_curr
    sig_tr = np.empty(6, dtype=float)
    sig_tr[0] = sig0[0] + 2.0 * g_curr * deps[0] + dav
    sig_tr[1] = sig0[1] + 2.0 * g_curr * deps[1] + dav
    sig_tr[2] = sig0[2] + 2.0 * g_curr * deps[2] + dav
    sig_tr[3] = sig0[3] + g_curr * deps[3]
    sig_tr[4] = sig0[4] + g_curr * deps[4]
    sig_tr[5] = sig0[5] + g_curr * deps[5]

    # Effective relative stress eta = s - alpha
    pres = (sig_tr[0] + sig_tr[1] + sig_tr[2]) / 3.0
    s_dev = sig_tr.copy()
    s_dev[0] -= pres
    s_dev[1] -= pres
    s_dev[2] -= pres

    eta = s_dev - alpha
    # von Mises equivalent of relative stress
    seff = math.sqrt(max(0.0, 1.5 * (eta[0] ** 2 + eta[1] ** 2 + eta[2] ** 2 + 2.0 * (eta[3] ** 2 + eta[4] ** 2 + eta[5] ** 2))))

    f_yield = seff - p.yield_stress
    sign = sig_tr.copy()
    dep = 0.0

    if f_yield > 0.0 and seff > _EM20:
        # Radial return plastic correction
        n_flow = 1.5 * eta / seff
        denom = 3.0 * g_curr + p.cyu
        dlam = f_yield / max(denom, _EM20)
        dep = dlam

        # Update stress
        corr = 2.0 * g_curr * dlam * (eta / seff)
        sign[:3] -= corr[:3]
        sign[3:] -= corr[3:]

        # Backstress update (Armstrong-Frederick / Yoshida kinematic hardening)
        d_alpha = (2.0 / 3.0 * p.cyu * n_flow - p.cyu * alpha) * dep
        alpha += d_alpha

        # Update isotropic hardening of bounding surface
        r_hard = uvar[0]
        dr = p.myu * (p.rsat - r_hard) * dep
        r_hard += dr
        uvar[0] = r_hard

    epsp_tot += dep
    uvar[4] = epsp_tot
    uvar[3] = p.yield_stress + uvar[0]

    c_curr = math.sqrt(max(0.0, (bulk_curr + 4.0 / 3.0 * g_curr) / p.rho0))
    return sign, epsp_tot, uvar, alpha, c_curr


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
    """3D solid continuum constitutive update for /MAT/LAW78."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(6, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(float)
    nel = len(sig_arr)

    uvar_arr = np.zeros((nel, 6), dtype=float)
    alpha_arr = np.zeros((nel, 6), dtype=float)

    if extra is not None and isinstance(extra, dict):
        for k in ("uvar78", "uvar"):
            if k in extra and extra[k] is not None:
                u = np.asarray(extra[k], dtype=float)
                if u.ndim == 1:
                    uvar_arr[0, :min(6, len(u))] = u[:min(6, len(u))]
                elif u.ndim == 2:
                    uvar_arr[:min(nel, len(u)), :min(6, u.shape[1])] = u[:min(nel, len(u)), :min(6, u.shape[1])]
                break
        if "alpha" in extra and extra["alpha"] is not None:
            a = np.asarray(extra["alpha"], dtype=float)
            if a.ndim == 1:
                alpha_arr[0, :min(6, len(a))] = a[:min(6, len(a))]
            elif a.ndim == 2:
                alpha_arr[:min(nel, len(a)), :min(6, a.shape[1])] = a[:min(nel, len(a)), :min(6, a.shape[1])]

    if epsp is not None:
        ep_in = np.atleast_1d(epsp).astype(float)
        uvar_arr[:min(nel, len(ep_in)), 4] = ep_in[:nel]

    off_arr = np.ones(nel, dtype=float)
    if extra is not None and isinstance(extra, dict) and "off" in extra:
        o = np.atleast_1d(extra["off"]).astype(float)
        off_arr[:min(nel, len(o))] = o[:nel]

    sig_out = np.zeros_like(sig_arr)
    epsp_out = np.zeros(nel, dtype=float)
    c_out = np.zeros(nel, dtype=float)

    for i in range(nel):
        s_i, ep_i, u_i, a_i, c_i = _solid_update_single(
            p, sig_arr[i], deps_arr[i], uvar_arr[i], alpha_in=alpha_arr[i], off=off_arr[i]
        )
        sig_out[i] = s_i
        epsp_out[i] = ep_i
        uvar_arr[i] = u_i
        alpha_arr[i] = a_i
        c_out[i] = c_i

    if extra is not None and isinstance(extra, dict):
        extra["uvar78"] = uvar_arr
        extra["uvar"] = uvar_arr
        extra["alpha"] = alpha_arr

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
    p: Law78Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    uvar0: np.ndarray,
    off: float = 1.0,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """2D plane-stress shell update for LAW78."""
    if off < 0.1:
        return np.zeros(len(sig0), dtype=float), float(uvar0[4]), uvar0.copy(), 0.0

    uvar = uvar0.copy()
    sign = np.empty_like(sig0, dtype=float)
    sign[0] = sig0[0] + p.a11_2d * deps[0] + p.a12_2d * deps[1]
    sign[1] = sig0[1] + p.a12_2d * deps[0] + p.a11_2d * deps[1]
    sign[2] = sig0[2] + p.g * deps[2]
    if len(sig0) >= 5:
        sign[3] = sig0[3] + p.g * deps[3]
        sign[4] = sig0[4] + p.g * deps[4]

    # von Mises check in plane stress
    svm = math.sqrt(max(0.0, sign[0] ** 2 + sign[1] ** 2 - sign[0] * sign[1] + 3.0 * sign[2] ** 2))
    f_yield = svm - p.yield_stress
    epsp = uvar[4]
    if f_yield > 0.0 and svm > _EM20:
        scale = p.yield_stress / svm
        sign[:3] *= scale
        epsp += f_yield / (3.0 * p.g + p.cyu)

    uvar[4] = epsp
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
    """2D plane-stress shell constitutive update for /MAT/LAW78."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(3, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(float)
    nel = len(sig_arr)

    uvar_arr = np.zeros((nel, 6), dtype=float)
    if extra is not None and isinstance(extra, dict):
        for k in ("uvar78", "uvar"):
            if k in extra and extra[k] is not None:
                u = np.asarray(extra[k], dtype=float)
                if u.ndim == 1:
                    uvar_arr[0, :min(6, len(u))] = u[:min(6, len(u))]
                elif u.ndim == 2:
                    uvar_arr[:min(nel, len(u)), :min(6, u.shape[1])] = u[:min(nel, len(u)), :min(6, u.shape[1])]
                break

    if epsp is not None:
        ep_in = np.atleast_1d(epsp).astype(float)
        uvar_arr[:min(nel, len(ep_in)), 4] = ep_in[:nel]

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
        extra["uvar78"] = uvar_arr
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
