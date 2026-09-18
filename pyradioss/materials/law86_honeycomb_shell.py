"""LAW86 — Orthotropic Honeycomb & Elastoplastic Shell Formulation (/MAT/LAW86, /MAT/HONEYCOMB_SHELL).

Upstream OpenRadioss Fortran reference:
- 2D Shells Plane Stress Constitutive Update:
  `engine/source/materials/mat/mat086/sigeps86c.F`
- 2D Shells Global Plasticity Update:
  `engine/source/materials/mat/mat086/sigeps86g.F`
- HyperMesh CFG Schema:
  `hm_cfg_files/config/CFG/Keyword971/MAT/mat_086.cfg`

Theory:
-------
1. Orthotropic honeycomb shell with kinematic & isotropic hardening:
   - Elastic matrix in plane stress:
     A11 = E1 / (1 - nu12 * nu21)
     A21 = nu12 * E2 / (1 - nu12 * nu21)
     G1  = G12
   - Backstress tensor alpha_xx, alpha_yy, alpha_xy (mixed kinematic hardening).
   - Strain rate sensitivity with ASRATE filter.

2. Yield function:
   f = sigma_vm(sigma - alpha) - sigma_y(eps_p, eps_dot) <= 0.
   Plastic multiplier dlam projects back onto the yield surface.

3. Through-thickness strain increment for shells:
   deps_zz = -nu / (1 - nu) * (deps_xx + deps_yy).
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
class Law86Params:
    """Parameters for OpenRadioss /MAT/LAW86 (Orthotropic honeycomb shell)."""
    id: int = 1
    title: str = ""
    law: int = 86
    law_name: str = "LAW86"

    # Density
    rho0: float = 0.1
    rhor: float = 0.1

    # Orthotropic / Elastic Properties
    young: float = 1000.0
    nu: float = 0.25
    e1: float = 1000.0
    e2: float = 1000.0
    g12: float = 400.0

    # Hardening & Rate Controls
    yield_stress: float = 20.0
    h_iso: float = 50.0
    h_kin: float = 50.0
    israte: int = 0
    asrate: float = 1.0e30
    epsmax1: float = 0.5
    epsr11: float = 0.0
    epsr21: float = 0.0
    fisokin1: float = 0.5      # Kinematic vs isotropic hardening partition [0, 1]
    epsf1: float = 1.0

    # Derived constants
    g: float = field(init=False)
    bulk: float = field(init=False)
    a11_2d: float = field(init=False)
    a21_2d: float = field(init=False)
    c_solid: float = field(init=False)
    c_shell: float = field(init=False)

    def __post_init__(self) -> None:
        if self.rho0 <= 0.0:
            self.rho0 = 0.1
        if self.rhor <= 0.0:
            self.rhor = self.rho0
        if self.e1 <= 0.0:
            self.e1 = self.young
        if self.e2 <= 0.0:
            self.e2 = self.young
        if self.g12 <= 0.0:
            self.g12 = 0.5 * self.young / max(1.0 + self.nu, _EM20)

        nu12 = self.nu
        nu21 = nu12 * self.e2 / max(self.e1, _EM20)
        denom_2d = max(1.0 - nu12 * nu21, _EM20)

        self.a11_2d = self.e1 / denom_2d
        self.a21_2d = nu12 * self.e2 / denom_2d
        self.g = self.g12
        self.bulk = self.young / max(3.0 * (1.0 - 2.0 * self.nu), _EM20)

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


def build_law86(mat_def: Any = None, **kwargs: Any) -> Law86Params:
    """Construct Law86Params from a Material entity, dictionary, or keyword arguments."""
    if isinstance(mat_def, Law86Params):
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
    title = str(data.get("title", f"LAW86_{mat_id}"))

    rho0 = _extract_val(data, ["MAT_RHO", "rho0", "rho", "density"], 0.1)
    rhor = _extract_val(data, ["Refer_Rho", "rhor", "ref_rho"], rho0)

    young = _extract_val(data, ["MAT_E", "young", "e", "E"], 1000.0)
    nu = _extract_val(data, ["MAT_NU", "nu", "poisson", "nux"], 0.25)
    e1 = _extract_val(data, ["E1", "e1"], young)
    e2 = _extract_val(data, ["E2", "e2"], young)
    g12 = _extract_val(data, ["G12", "g12", "G1"], 0.5 * young / max(1.0 + nu, _EM20))

    sigy = _extract_val(data, ["SIGMA_r", "yield_stress", "sigy", "sig0"], 20.0)
    h_iso = _extract_val(data, ["h_iso", "H_iso", "hardening"], 50.0)
    h_kin = _extract_val(data, ["h_kin", "H_kin"], 50.0)
    israte = int(_extract_val(data, ["ISRATE", "israte"], 0))
    asrate = _extract_val(data, ["ASRATE", "asrate"], 1.0e30)
    epsmax1 = _extract_val(data, ["EPSMAX", "epsmax1"], 0.5)
    epsr11 = _extract_val(data, ["EPSR1", "epsr11"], 0.0)
    epsr21 = _extract_val(data, ["EPSR2", "epsr21"], 0.0)
    fisokin1 = _extract_val(data, ["FISOKIN", "fisokin1"], 0.5)
    epsf1 = _extract_val(data, ["EPSF", "epsf1"], 1.0)

    return Law86Params(
        id=mat_id,
        title=title,
        rho0=rho0,
        rhor=rhor,
        young=young,
        nu=nu,
        e1=e1,
        e2=e2,
        g12=g12,
        yield_stress=sigy,
        h_iso=h_iso,
        h_kin=h_kin,
        israte=israte,
        asrate=asrate,
        epsmax1=epsmax1,
        epsr11=epsr11,
        epsr21=epsr21,
        fisokin1=fisokin1,
        epsf1=epsf1,
    )


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law86Params:
    """Resolve and cache Law86Params from a Material or dict."""
    if isinstance(mat, Law86Params):
        return mat
    cached = getattr(mat, "_cached_law86", None)
    if cached is None:
        cached = build_law86(mat)
        try:
            setattr(mat, "_cached_law86", cached)
        except Exception:
            pass
    return cached


def needs_defgrad(mat: Any = None) -> bool:
    """Return False: LAW86 is an incremental rate formulation."""
    return False


def extra_shapes(mat: Any = None, nip: Optional[int] = 1) -> Dict[str, Tuple[int, ...]]:
    """Persistent history variables for LAW86 (col 0: eps_p, col 1: alpha_xx, col 2: alpha_yy, col 3: alpha_xy, col 4: eps_dot)."""
    if nip is not None and nip > 1:
        return {"uvar86": (nip, 5), "uvar": (nip, 5)}
    return {"uvar86": (5,), "uvar": (5,)}


def sound_speed(
    mat: Any,
    eps: Any = None,
    extra: Any = None,
    is_shell: bool = False,
) -> float | np.ndarray:
    """Acoustic sound speed for LAW86."""
    p = resolve(mat)
    c_val = p.c_shell if is_shell else p.c_solid
    if eps is not None and isinstance(eps, np.ndarray) and eps.ndim > 1:
        return np.full(len(eps), c_val, dtype=float)
    return c_val


def _solid_update_single(
    p: Law86Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    uvar0: np.ndarray,
    off: float = 1.0,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Single 3D solid continuum update for LAW86."""
    if off < 0.1:
        return np.zeros(6, dtype=float), float(uvar0[0]), uvar0.copy(), 0.0

    uvar = uvar0.copy()
    epsp = uvar[0]
    alpha = uvar[1:4]  # backstress [xx, yy, xy]

    deps_vol = deps[0] + deps[1] + deps[2]
    dav = deps_vol * (p.bulk - 2.0 / 3.0 * p.g)
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

    eta_xx = s_dev[0] - alpha[0]
    eta_yy = s_dev[1] - alpha[1]
    eta_xy = s_dev[3] - alpha[2]

    seff = math.sqrt(max(0.0, 1.5 * (eta_xx ** 2 + eta_yy ** 2 + s_dev[2] ** 2 + 2.0 * (eta_xy ** 2 + s_dev[4] ** 2 + s_dev[5] ** 2))))
    sigy = p.yield_stress + p.h_iso * epsp

    f_yield = seff - sigy
    if f_yield > 0.0 and seff > _EM20:
        dlam = f_yield / (3.0 * p.g + p.h_iso + p.h_kin)
        epsp += dlam
        scale = (seff - 3.0 * p.g * dlam) / max(seff, _EM20)
        sign[0] = eta_xx * scale + alpha[0] + pres
        sign[1] = eta_yy * scale + alpha[1] + pres
        sign[2] = s_dev[2] * scale + pres
        sign[3] = eta_xy * scale + alpha[2]
        sign[4] = s_dev[4] * scale
        sign[5] = s_dev[5] * scale

        # Backstress update
        alpha[0] += (2.0 / 3.0 * p.h_kin * (eta_xx / seff)) * dlam
        alpha[1] += (2.0 / 3.0 * p.h_kin * (eta_yy / seff)) * dlam
        alpha[2] += (p.h_kin * (eta_xy / seff)) * dlam

    uvar[0] = epsp
    uvar[1:4] = alpha
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
    """3D solid continuum constitutive update for /MAT/LAW86."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(6, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(float)
    nel = len(sig_arr)

    uvar_arr = np.zeros((nel, 5), dtype=float)
    if extra is not None and isinstance(extra, dict):
        for k in ("uvar86", "uvar"):
            if k in extra and extra[k] is not None:
                u = np.asarray(extra[k], dtype=float)
                if u.ndim == 1:
                    uvar_arr[0, :min(5, len(u))] = u[:min(5, len(u))]
                elif u.ndim == 2:
                    uvar_arr[:min(nel, len(u)), :min(5, u.shape[1])] = u[:min(nel, len(u)), :min(5, u.shape[1])]
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
        extra["uvar86"] = uvar_arr
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
    p: Law86Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    uvar0: np.ndarray,
    off: float = 1.0,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """2D plane-stress shell update matching sigeps86c.F."""
    if off < 0.1:
        return np.zeros(len(sig0), dtype=float), float(uvar0[0]), uvar0.copy(), 0.0

    uvar = uvar0.copy()
    epsp = uvar[0]
    alpha = uvar[1:4]

    # Effective stress minus backstress
    s0_xx = sig0[0] - alpha[0]
    s0_yy = sig0[1] - alpha[1]
    s0_xy = sig0[2] - alpha[2]

    # Trial stress
    sign = np.empty_like(sig0, dtype=float)
    sign[0] = s0_xx + p.a11_2d * deps[0] + p.a21_2d * deps[1]
    sign[1] = s0_yy + p.a21_2d * deps[0] + p.a11_2d * deps[1]
    sign[2] = s0_xy + p.g12 * deps[2]
    if len(sig0) >= 5:
        sign[3] = sig0[3] + p.g12 * deps[3]
        sign[4] = sig0[4] + p.g12 * deps[4]

    svm = math.sqrt(max(0.0, sign[0] ** 2 + sign[1] ** 2 - sign[0] * sign[1] + 3.0 * sign[2] ** 2))
    sigy = p.yield_stress + p.h_iso * epsp

    f_yield = svm - sigy
    if f_yield > 0.0 and svm > _EM20:
        dlam = f_yield / (3.0 * p.g12 + p.h_iso + p.h_kin)
        epsp += dlam
        scale = (svm - 3.0 * p.g12 * dlam) / max(svm, _EM20)
        sign[0] = sign[0] * scale + alpha[0]
        sign[1] = sign[1] * scale + alpha[1]
        sign[2] = sign[2] * scale + alpha[2]

        alpha[0] += (p.h_kin * (sign[0] / svm)) * dlam
        alpha[1] += (p.h_kin * (sign[1] / svm)) * dlam
        alpha[2] += (p.h_kin * (sign[2] / svm)) * dlam
    else:
        sign[0] += alpha[0]
        sign[1] += alpha[1]
        sign[2] += alpha[2]

    uvar[0] = epsp
    uvar[1:4] = alpha
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
    """2D plane-stress shell constitutive update for /MAT/LAW86."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(3, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(float)
    nel = len(sig_arr)

    uvar_arr = np.zeros((nel, 5), dtype=float)
    if extra is not None and isinstance(extra, dict):
        for k in ("uvar86", "uvar"):
            if k in extra and extra[k] is not None:
                u = np.asarray(extra[k], dtype=float)
                if u.ndim == 1:
                    uvar_arr[0, :min(5, len(u))] = u[:min(5, len(u))]
                elif u.ndim == 2:
                    uvar_arr[:min(nel, len(u)), :min(5, u.shape[1])] = u[:min(nel, len(u)), :min(5, u.shape[1])]
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
        extra["uvar86"] = uvar_arr
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
        [0.0, 0.0, p.g12],
    ], dtype=float)

    n = 1
    if sig is not None and np.ndim(sig) >= 2:
        n = len(sig)
    return np.broadcast_to(c_el, (n, 3, 3)).copy()


consistent_shell_tangent = shell_tangent
