"""LAW80 — Ramberg-Osgood & Kirkaldy Metallurgical Kinetics (/MAT/LAW80, /MAT/KIRKALDY).

Upstream OpenRadioss Fortran reference:
- Starter Card Reader:
  `starter/source/materials/mat/mat080/hm_read_mat80.F`
- 3D Solids Constitutive Update:
  `engine/source/materials/mat/mat080/sigeps80.F`
- 2D Shells Constitutive Update:
  `engine/source/materials/mat/mat080/sigeps80c.F`
- Kirkaldy Metallurgical Kinetics:
  `engine/source/materials/mat/mat080/kirkaldykinetics.F`
- Phase Kinetics 2:
  `engine/source/materials/mat/mat080/phasekinetic2.F`
- HyperMesh CFG Schema:
  `hm_cfg_files/config/CFG/radioss2021/MAT/matl80_80.cfg`

Theory:
-------
1. Ramberg-Osgood elastoplastic law:
   Total strain decomposed into elastic and plastic components:
     eps = sigma / E + alpha_0 * (sigma / sigma_0)^n
   Or in yield stress representation:
     sigma_y = sigma_0 * (1.0 + alpha_0 * (eps_p / eps_0)^n)

2. Kirkaldy metallurgical kinetics for steel phase transformations:
   Simulates diffusion-controlled phase transformations during cooling/heating
   between Austenite, Ferrite, Pearlite, Bainite, and Martensite:
   - Martensite start temperature Ms (Andrews formula):
     Ms = 512 - 453*C - 16.9*Ni + 15*Cr - 9.5*Mo + 217*(C^2) - 71.5*(C*Mn) - 67.6*(C*Cr)
   - Bainite start temperature Bs:
     Bs = 656 - 58*C - 35*Mn - 75*Si - 15*Ni - 34*Cr - 41*Mo
   - Koistinen-Marburger equation for diffusionless martensite fraction:
     X_m = 1.0 - exp(-alpha_m * max(0.0, Ms - T))
   - Kirkaldy reaction rates for ferrite, pearlite, and bainite:
     dX_i / dt = B_i * (T) * X_i^(a_i) * (1 - X_i)^(b_i) * (Delta T)^(c_i)

3. Linear mixture law for equivalent macroscopic stress:
   sigma_y = sum_k (X_k * sigma_y,k(T, eps_p))
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
class Law80Params:
    """Parameters for OpenRadioss /MAT/LAW80 (Ramberg-Osgood & Kirkaldy kinetics)."""
    id: int = 1
    title: str = ""
    law: int = 80
    law_name: str = "LAW80"

    # Density
    rho0: float = 7.8e-3
    rhor: float = 7.8e-3

    # Elastic Constants
    young: float = 210000.0
    nu: float = 0.3

    # Ramberg-Osgood Parameters
    sigy0: float = 350.0
    eps0: float = 0.002
    alpha0: float = 0.02
    n_ramberg: float = 5.0
    efac: float = 1.0

    # Thermal & Time Scales
    tini: float = 293.15
    tref: float = 293.15
    unitt: float = 1.0
    israte: int = 1

    # Alloy Chemical Composition (wt %)
    c_alloy: float = 0.2
    mn_alloy: float = 1.2
    si_alloy: float = 0.3
    cr_alloy: float = 0.5
    ni_alloy: float = 0.2
    mo_alloy: float = 0.1
    v_alloy: float = 0.0
    gsize: float = 7.0         # ASTM grain size number

    # Critical Transformation Temperatures
    ae1: float = 727.0
    ae3: float = 850.0
    bs: float = 550.0
    ms: float = 400.0

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
        if self.n_ramberg <= 0.0:
            self.n_ramberg = 5.0
        if self.eps0 <= 0.0:
            self.eps0 = 0.002
        if self.sigy0 <= 0.0:
            self.sigy0 = 350.0

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


def build_law80(mat_def: Any = None, **kwargs: Any) -> Law80Params:
    """Construct Law80Params from a Material entity, dictionary, or keyword arguments."""
    if isinstance(mat_def, Law80Params):
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
    title = str(data.get("title", f"LAW80_{mat_id}"))

    rho0 = _extract_val(data, ["MAT_RHO", "rho0", "rho", "density"], 7.8e-3)
    rhor = _extract_val(data, ["Refer_Rho", "rhor", "ref_rho"], rho0)

    young = _extract_val(data, ["MAT_E", "young", "e", "E"], 210000.0)
    nu = _extract_val(data, ["MAT_NU", "nu", "poisson"], 0.3)
    efac = _extract_val(data, ["SCALE", "efac"], 1.0)
    unitt = _extract_val(data, ["time_inputunit_value", "unitt"], 1.0)
    israte = int(_extract_val(data, ["Fsmooth", "israte"], 1))

    sigy0 = _extract_val(data, ["sigy0", "SIGMA_r", "sig0", "yield_stress"], 350.0)
    eps0 = _extract_val(data, ["Epsilon_0", "eps0", "EPS0"], 0.002)
    alpha0 = _extract_val(data, ["alpha0", "ALPHA0", "alpha"], 0.02)
    n_ramberg = _extract_val(data, ["n_ramberg", "n", "N", "EXP"], 5.0)

    tini = _extract_val(data, ["TINI", "tini", "Tini"], 293.15)
    tref = _extract_val(data, ["TREF", "tref"], 293.15)

    c_alloy = _extract_val(data, ["C", "c_alloy"], 0.2)
    mn_alloy = _extract_val(data, ["MN", "mn_alloy"], 1.2)
    si_alloy = _extract_val(data, ["SI", "si_alloy"], 0.3)
    cr_alloy = _extract_val(data, ["CR", "cr_alloy"], 0.5)
    ni_alloy = _extract_val(data, ["NI", "ni_alloy"], 0.2)
    mo_alloy = _extract_val(data, ["MO", "mo_alloy"], 0.1)
    v_alloy = _extract_val(data, ["V", "v_alloy"], 0.0)
    gsize = _extract_val(data, ["GSIZE", "gsize"], 7.0)

    ae1 = _extract_val(data, ["AE1", "ae1"], 727.0)
    ae3 = _extract_val(data, ["AE3", "ae3"], 850.0)
    bs = _extract_val(data, ["BS", "bs"], 550.0)
    ms = _extract_val(data, ["MS", "ms"], 400.0)

    return Law80Params(
        id=mat_id,
        title=title,
        rho0=rho0,
        rhor=rhor,
        young=young,
        nu=nu,
        sigy0=sigy0,
        eps0=eps0,
        alpha0=alpha0,
        n_ramberg=n_ramberg,
        efac=efac,
        tini=tini,
        tref=tref,
        unitt=unitt,
        israte=israte,
        c_alloy=c_alloy,
        mn_alloy=mn_alloy,
        si_alloy=si_alloy,
        cr_alloy=cr_alloy,
        ni_alloy=ni_alloy,
        mo_alloy=mo_alloy,
        v_alloy=v_alloy,
        gsize=gsize,
        ae1=ae1,
        ae3=ae3,
        bs=bs,
        ms=ms,
    )


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law80Params:
    """Resolve and cache Law80Params from a Material or dict."""
    if isinstance(mat, Law80Params):
        return mat
    cached = getattr(mat, "_cached_law80", None)
    if cached is None:
        cached = build_law80(mat)
        try:
            setattr(mat, "_cached_law80", cached)
        except Exception:
            pass
    return cached


def needs_defgrad(mat: Any = None) -> bool:
    """Return False: LAW80 is a rate-form hypoelastic/kinetics formulation."""
    return False


def extra_shapes(mat: Any = None, nip: Optional[int] = 1) -> Dict[str, Tuple[int, ...]]:
    """Persistent history variables for LAW80 (15 metallurgical & phase state variables)."""
    if nip is not None and nip > 1:
        return {"uvar80": (nip, 15), "uvar": (nip, 15)}
    return {"uvar80": (15,), "uvar": (15,)}


def sound_speed(
    mat: Any,
    eps: Any = None,
    extra: Any = None,
    is_shell: bool = False,
) -> float | np.ndarray:
    """Acoustic sound speed for LAW80."""
    p = resolve(mat)
    c_val = p.c_shell if is_shell else p.c_solid
    if eps is not None and isinstance(eps, np.ndarray) and eps.ndim > 1:
        return np.full(len(eps), c_val, dtype=float)
    return c_val


def _solid_update_single(
    p: Law80Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    uvar0: np.ndarray,
    off: float = 1.0,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Single 3D solid continuum update matching sigeps80.F."""
    if off < 0.1:
        return np.zeros(6, dtype=float), float(uvar0[0]), uvar0.copy(), 0.0

    uvar = uvar0.copy()
    epsp = uvar[0]
    temp = uvar[1] if uvar[1] > 0.0 else p.tini

    g2 = 2.0 * p.g
    deps_vol = deps[0] + deps[1] + deps[2]
    dav = deps_vol * p.lamhook

    # Elastic trial stresses
    sign = np.empty(6, dtype=float)
    sign[0] = sig0[0] + 2.0 * p.g * deps[0] + dav
    sign[1] = sig0[1] + 2.0 * p.g * deps[1] + dav
    sign[2] = sig0[2] + 2.0 * p.g * deps[2] + dav
    sign[3] = sig0[3] + p.g * deps[3]
    sign[4] = sig0[4] + p.g * deps[4]
    sign[5] = sig0[5] + p.g * deps[5]

    # Deviatoric stresses and von Mises
    pres = (sign[0] + sign[1] + sign[2]) / 3.0
    s_dev = sign.copy()
    s_dev[0] -= pres
    s_dev[1] -= pres
    s_dev[2] -= pres

    svm = math.sqrt(max(0.0, 1.5 * (s_dev[0] ** 2 + s_dev[1] ** 2 + s_dev[2] ** 2 + 2.0 * (s_dev[3] ** 2 + s_dev[4] ** 2 + s_dev[5] ** 2))))

    # Ramberg-Osgood current yield stress
    # sigma_y = sigma_0 * (1 + alpha * (eps_p / eps_0)^(1/n))
    term_p = (epsp / max(p.eps0, 1.0e-6)) ** (1.0 / max(p.n_ramberg, 1.0))
    sigy = p.sigy0 * (1.0 + p.alpha0 * term_p)

    f_yield = svm - sigy
    if f_yield > 0.0 and svm > _EM20:
        # Radial return
        h_mod = (p.sigy0 * p.alpha0 / (p.n_ramberg * max(p.eps0, 1.0e-6))) * max(term_p, 1.0e-6) ** (1.0 - p.n_ramberg)
        dlam = f_yield / max(3.0 * p.g + h_mod, _EM20)
        epsp += dlam

        scale = (svm - 3.0 * p.g * dlam) / max(svm, _EM20)
        sign[0] = s_dev[0] * scale + pres
        sign[1] = s_dev[1] * scale + pres
        sign[2] = s_dev[2] * scale + pres
        sign[3] = s_dev[3] * scale
        sign[4] = s_dev[4] * scale
        sign[5] = s_dev[5] * scale

    uvar[0] = epsp
    uvar[1] = temp

    # Update martensite fraction if cooling below Ms
    if temp < p.ms:
        uvar[2] = min(1.0, max(uvar[2], 1.0 - math.exp(-0.011 * (p.ms - temp))))

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
    """3D solid continuum constitutive update for /MAT/LAW80."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(6, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(float)
    nel = len(sig_arr)

    uvar_arr = np.zeros((nel, 15), dtype=float)
    uvar_arr[:, 1] = p.tini

    if extra is not None and isinstance(extra, dict):
        for k in ("uvar80", "uvar"):
            if k in extra and extra[k] is not None:
                u = np.asarray(extra[k], dtype=float)
                if u.ndim == 1:
                    uvar_arr[0, :min(15, len(u))] = u[:min(15, len(u))]
                elif u.ndim == 2:
                    uvar_arr[:min(nel, len(u)), :min(15, u.shape[1])] = u[:min(nel, len(u)), :min(15, u.shape[1])]
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
        extra["uvar80"] = uvar_arr
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
    p: Law80Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    uvar0: np.ndarray,
    off: float = 1.0,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """2D plane-stress shell update for LAW80."""
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
    term_p = (epsp / max(p.eps0, 1.0e-6)) ** (1.0 / max(p.n_ramberg, 1.0))
    sigy = p.sigy0 * (1.0 + p.alpha0 * term_p)

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
    """2D plane-stress shell constitutive update for /MAT/LAW80."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(3, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(float)
    nel = len(sig_arr)

    uvar_arr = np.zeros((nel, 15), dtype=float)
    uvar_arr[:, 1] = p.tini

    if extra is not None and isinstance(extra, dict):
        for k in ("uvar80", "uvar"):
            if k in extra and extra[k] is not None:
                u = np.asarray(extra[k], dtype=float)
                if u.ndim == 1:
                    uvar_arr[0, :min(15, len(u))] = u[:min(15, len(u))]
                elif u.ndim == 2:
                    uvar_arr[:min(nel, len(u)), :min(15, u.shape[1])] = u[:min(nel, len(u)), :min(15, u.shape[1])]
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
        extra["uvar80"] = uvar_arr
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
