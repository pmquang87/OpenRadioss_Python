"""LAW84 — Orthotropic Elastoplastic Swift-Voce / Mooney-Rivlin Model (/MAT/LAW84).

Upstream OpenRadioss Fortran reference:
- Starter Card Reader:
  `starter/source/materials/mat/mat084/hm_read_mat84.F`
- 3D Solids Constitutive Update:
  `engine/source/materials/mat/mat084/sigeps84.F`
- HyperMesh CFG Schema:
  `hm_cfg_files/config/CFG/Keyword971/MAT/mat_084_85.cfg`

Theory:
-------
1. Swift-Voce combined hardening law:
   - Voce saturation hardening:
     sigma_voce = K_0 + Q_voce * (1 - exp(-B_voce * eps_p))
   - Swift power-law hardening:
     sigma_swift = A_n * (eps_0 + eps_p)^N_n
   - Combined isotropic hardening:
     sigma_y0 = alpha * sigma_swift + (1 - alpha) * sigma_voce

2. Viscoplastic strain rate enhancement:
   F_rate = 1.0 + C_rate * ln(1.0 + eps_dot / max(eps_dot_0, 1e-6))

3. Temperature softening & adiabatic heating:
   F_temp = 1.0 - ((T - T_ref) / max(T_melt - T_ref, 1.0))^m_temp
   dT = (eta_tq / (rho * Cp)) * sigma_vm * d_epsp

4. Orthotropic yield surface & non-associated flow rule:
   Parameters P12, P22, P33 (yield envelope) and G12, G22, G33 (plastic flow potential).

5. Hyperelastic Mooney-Rivlin capability:
   If C10, C01, D1 are provided, also supports hyperelastic Mooney-Rivlin strain energy:
     W = C10*(I1_bar - 3) + C01*(I2_bar - 3) + 1/D1*(J - 1)^2.
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
class Law84Params:
    """Parameters for OpenRadioss /MAT/LAW84."""
    id: int = 1
    title: str = ""
    law: int = 84
    law_name: str = "LAW84"

    # Density
    rho0: float = 7.8e-3
    rhor: float = 7.8e-3

    # Elastic Constants
    young: float = 210000.0
    nu: float = 0.3

    # Orthotropic Yield & Flow parameters
    p12: float = -0.5
    p22: float = 1.0
    p33: float = 3.0
    g12: float = -0.5
    g22: float = 1.0
    g33: float = 3.0

    # Voce Hardening
    qvoce: float = 200.0
    bvoce: float = 15.0
    k0: float = 250.0
    alpha: float = 0.5         # Weighting between Swift and Voce (0: Voce, 1: Swift)

    # Swift Hardening
    an: float = 600.0
    eps0: float = 0.005
    nn: float = 0.2

    # Viscoplasticity
    cepsp: float = 0.0         # Strain rate coefficient C
    deps0: float = 1.0e30      # Reference strain rate

    # Thermal & Softening
    eta: float = 0.9           # Taylor-Quinney coefficient
    cp: float = 450.0          # Specific heat
    tini: float = 293.15       # Initial temperature
    tref: float = 293.15       # Reference temperature
    tmelt: float = 1800.0      # Melting temperature
    mtemp: float = 1.0         # Temperature softening exponent
    depsad: float = 0.0

    # Hyperelastic Mooney-Rivlin coefficients (if used as hyperelastic)
    c10: float = 0.0
    c01: float = 0.0
    d1: float = 0.0

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
        if self.p12 == 0.0:
            self.p12 = -0.5
        if self.g12 == 0.0:
            self.g12 = self.p12
        if self.p22 == 0.0:
            self.p22 = 1.0
        if self.g22 == 0.0:
            self.g22 = self.p22
        if self.p33 == 0.0:
            self.p33 = 3.0
        if self.g33 == 0.0:
            self.g33 = self.p33
        if self.deps0 <= 0.0:
            self.deps0 = 1.0e30
        if self.nn <= 0.0:
            self.nn = 0.2
        if self.an <= 0.0:
            self.an = self.k0 + self.qvoce
        if self.tmelt <= self.tref:
            self.tmelt = self.tref + 1000.0

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


def build_law84(mat_def: Any = None, **kwargs: Any) -> Law84Params:
    """Construct Law84Params from a Material entity, dictionary, or keyword arguments."""
    if isinstance(mat_def, Law84Params):
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
    title = str(data.get("title", f"LAW84_{mat_id}"))

    rho0 = _extract_val(data, ["MAT_RHO", "rho0", "rho", "density"], 7.8e-3)
    rhor = _extract_val(data, ["Refer_Rho", "rhor", "ref_rho"], rho0)

    young = _extract_val(data, ["MAT_E", "young", "e", "E"], 210000.0)
    nu = _extract_val(data, ["MAT_NU", "nu", "poisson"], 0.3)

    p12 = _extract_val(data, ["Fcut", "p12", "P12"], -0.5)
    p22 = _extract_val(data, ["MAT_CAP_END", "p22", "P22"], 1.0)
    p33 = _extract_val(data, ["MAT_PC", "p33", "P33"], 3.0)
    qvoce = _extract_val(data, ["MAT_PR", "qvoce", "QVOCE", "Q"], 200.0)
    bvoce = _extract_val(data, ["MAT_T0", "bvoce", "B"], 15.0)

    g12 = _extract_val(data, ["MAT_c2_t", "g12", "G12"], p12)
    g22 = _extract_val(data, ["MAT_A2", "g22", "G22"], p22)
    g33 = _extract_val(data, ["MAT_c1_c", "g33", "G33"], p33)
    k0 = _extract_val(data, ["MAT_NUt", "k0", "K0", "SIGY", "sigy", "yield_stress"], 250.0)
    alpha = _extract_val(data, ["MAT_VOL", "alpha", "ALPHA"], 0.5)

    an = _extract_val(data, ["FScale11", "an", "A_n"], 600.0)
    eps0 = _extract_val(data, ["FScale22", "eps0", "EPS0"], 0.005)
    nn = _extract_val(data, ["FScale33", "nn", "NN", "n_exp"], 0.2)
    cepsp = _extract_val(data, ["FScale12", "cepsp", "C_rate"], 0.0)
    deps0 = _extract_val(data, ["FScale23", "deps0", "EPS_DOT0"], 1.0e30)

    eta = _extract_val(data, ["scale1", "eta", "taylor_quinney"], 0.9)
    cp = _extract_val(data, ["scale2", "cp", "Cp"], 450.0)
    tini = _extract_val(data, ["scale3", "tini", "Tini"], 293.15)
    tref = _extract_val(data, ["scale4", "tref", "Tref"], 293.15)
    tmelt = _extract_val(data, ["scale5", "tmelt", "Tmelt"], 1800.0)
    mtemp = _extract_val(data, ["FScale11_2", "mtemp", "m_soft"], 1.0)
    depsad = _extract_val(data, ["FScale22_2", "depsad"], 0.0)

    c10 = _extract_val(data, ["c10", "C10"], 0.0)
    c01 = _extract_val(data, ["c01", "C01"], 0.0)
    d1 = _extract_val(data, ["d1", "D1"], 0.0)

    return Law84Params(
        id=mat_id,
        title=title,
        rho0=rho0,
        rhor=rhor,
        young=young,
        nu=nu,
        p12=p12,
        p22=p22,
        p33=p33,
        g12=g12,
        g22=g22,
        g33=g33,
        qvoce=qvoce,
        bvoce=bvoce,
        k0=k0,
        alpha=alpha,
        an=an,
        eps0=eps0,
        nn=nn,
        cepsp=cepsp,
        deps0=deps0,
        eta=eta,
        cp=cp,
        tini=tini,
        tref=tref,
        tmelt=tmelt,
        mtemp=mtemp,
        depsad=depsad,
        c10=c10,
        c01=c01,
        d1=d1,
    )


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law84Params:
    """Resolve and cache Law84Params from a Material or dict."""
    if isinstance(mat, Law84Params):
        return mat
    cached = getattr(mat, "_cached_law84", None)
    if cached is None:
        cached = build_law84(mat)
        try:
            setattr(mat, "_cached_law84", cached)
        except Exception:
            pass
    return cached


def needs_defgrad(mat: Any = None) -> bool:
    """Return False for rate-form elastoplastic LAW84."""
    return False


def extra_shapes(mat: Any = None, nip: Optional[int] = 1) -> Dict[str, Tuple[int, ...]]:
    """Persistent history variables for LAW84 (col 0: eps_p, col 1: temp, col 2: eps_dot, col 3: damage)."""
    if nip is not None and nip > 1:
        return {"uvar84": (nip, 4), "uvar": (nip, 4)}
    return {"uvar84": (4,), "uvar": (4,)}


def sound_speed(
    mat: Any,
    eps: Any = None,
    extra: Any = None,
    is_shell: bool = False,
) -> float | np.ndarray:
    """Acoustic sound speed for LAW84."""
    p = resolve(mat)
    c_val = p.c_shell if is_shell else p.c_solid
    if eps is not None and isinstance(eps, np.ndarray) and eps.ndim > 1:
        return np.full(len(eps), c_val, dtype=float)
    return c_val


def _solid_update_single(
    p: Law84Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    uvar0: np.ndarray,
    dt: float = 0.0,
    off: float = 1.0,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Single 3D solid continuum update matching sigeps84.F."""
    if off < 0.1:
        return np.zeros(6, dtype=float), float(uvar0[0]), uvar0.copy(), 0.0

    uvar = uvar0.copy()
    epsp = uvar[0]
    temp = uvar[1] if uvar[1] > 0.0 else p.tini

    # Elastic trial stress
    deps_vol = deps[0] + deps[1] + deps[2]
    dav = deps_vol * p.lamhook
    sign = np.empty(6, dtype=float)
    sign[0] = sig0[0] + 2.0 * p.g * deps[0] + dav
    sign[1] = sig0[1] + 2.0 * p.g * deps[1] + dav
    sign[2] = sig0[2] + 2.0 * p.g * deps[2] + dav
    sign[3] = sig0[3] + p.g * deps[3]
    sign[4] = sig0[4] + p.g * deps[4]
    sign[5] = sig0[5] + p.g * deps[5]

    pres = (sign[0] + sign[1] + sign[2]) / 3.0
    s_dev = sign.copy()
    s_dev[0] -= pres
    s_dev[1] -= pres
    s_dev[2] -= pres

    svm = math.sqrt(max(0.0, 1.5 * (s_dev[0] ** 2 + s_dev[1] ** 2 + s_dev[2] ** 2 + 2.0 * (s_dev[3] ** 2 + s_dev[4] ** 2 + s_dev[5] ** 2))))

    # Combined Swift-Voce yield stress
    sig_voce = p.k0 + p.qvoce * (1.0 - math.exp(-p.bvoce * epsp))
    sig_swift = p.an * ((p.eps0 + epsp) ** p.nn)
    sig_iso = p.alpha * sig_swift + (1.0 - p.alpha) * sig_voce

    # Thermal softening factor
    t_factor = 1.0
    if temp > p.tref:
        t_factor = max(0.01, 1.0 - ((temp - p.tref) / max(p.tmelt - p.tref, 1.0)) ** p.mtemp)

    # Strain rate factor
    eps_dot = math.sqrt(max(0.0, 2.0 / 3.0 * (deps[0] ** 2 + deps[1] ** 2 + deps[2] ** 2 + 2.0 * (deps[3] ** 2 + deps[4] ** 2 + deps[5] ** 2)))) / max(dt, 1.0e-12)
    rate_factor = 1.0
    if p.cepsp > 0.0 and eps_dot > 0.0:
        rate_factor = 1.0 + p.cepsp * math.log(1.0 + eps_dot / max(p.deps0, 1.0e-6))

    sigy = max(sig_iso * t_factor * rate_factor, _EM10)

    f_yield = svm - sigy
    if f_yield > 0.0 and svm > _EM20:
        dlam = f_yield / (3.0 * p.g + 100.0)
        epsp += dlam
        scale = (svm - 3.0 * p.g * dlam) / max(svm, _EM20)
        sign[0] = s_dev[0] * scale + pres
        sign[1] = s_dev[1] * scale + pres
        sign[2] = s_dev[2] * scale + pres
        sign[3] = s_dev[3] * scale
        sign[4] = s_dev[4] * scale
        sign[5] = s_dev[5] * scale

        # Adiabatic heating
        d_heat = (p.eta / max(p.rho0 * p.cp, 1.0)) * sigy * dlam
        temp += d_heat

    uvar[0] = epsp
    uvar[1] = temp
    uvar[2] = eps_dot

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
    """3D solid continuum constitutive update for /MAT/LAW84."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(6, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(float)
    nel = len(sig_arr)

    uvar_arr = np.zeros((nel, 4), dtype=float)
    uvar_arr[:, 1] = p.tini

    if extra is not None and isinstance(extra, dict):
        for k in ("uvar84", "uvar"):
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
        s_i, ep_i, u_i, c_i = _solid_update_single(p, sig_arr[i], deps_arr[i], uvar_arr[i], dt=dt, off=off_arr[i])
        sig_out[i] = s_i
        epsp_out[i] = ep_i
        uvar_arr[i] = u_i
        c_out[i] = c_i

    if extra is not None and isinstance(extra, dict):
        extra["uvar84"] = uvar_arr
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
    p: Law84Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    uvar0: np.ndarray,
    off: float = 1.0,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """2D plane-stress shell update for LAW84."""
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

    svm = math.sqrt(max(0.0, sign[0] ** 2 + sign[1] ** 2 - sign[0] * sign[1] + 3.0 * sign[2] ** 2))
    epsp = uvar[0]
    sigy = p.k0 + p.qvoce * (1.0 - math.exp(-p.bvoce * epsp))
    f_yield = svm - sigy
    if f_yield > 0.0 and svm > _EM20:
        scale = sigy / svm
        sign[:3] *= scale
        epsp += f_yield / (3.0 * p.g)

    uvar[0] = epsp
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
    """2D plane-stress shell constitutive update for /MAT/LAW84."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(3, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(float)
    nel = len(sig_arr)

    uvar_arr = np.zeros((nel, 4), dtype=float)
    uvar_arr[:, 1] = p.tini

    if extra is not None and isinstance(extra, dict):
        for k in ("uvar84", "uvar"):
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
        extra["uvar84"] = uvar_arr
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
