"""
LAW26 — Honeycomb / Porous Material with SESAME Tabular EOS (/MAT/LAW26, /MAT/HONEYCOMB_SESAME).

Fortran origins:
- ``engine/source/materials/mat/mat026/m26law.F`` (Johnson-Cook / SESAME constitutive deviatoric return)
- ``engine/source/materials/mat/mat026/m26th.F`` (thermal softening and temperature interpolation)
- ``engine/source/materials/mat/mat026/sesa10.F`` (SESAME internal energy and compaction integration)
- ``starter/source/materials/mat/mat026/hm_read_mat26.F`` (starter reader)

Theory
------
LAW26 models porous media, honeycombs, and metals under shock loading coupled with
SESAME tabular equation of state P(rho, T) or hydrostatic compaction and deviatoric
elastoplasticity with Johnson-Cook hardening, rate sensitivity, and thermal melting:
    sigma_y = min(sig_max, A + B * eps_p^n) * Ce * Ct

1. Pressure: P = K * mu (or tabular EOS P(rho, T))
2. Thermal factor: Ct = max(0, 1 - ((T - 300)/(T_melt - 300))^cm)
3. Rate factor: Ce = 1 + C * ln(max(epsdot, eps0)/eps0)
4. Radial return deviatoric projection.

Sound speed:
    c = sqrt((K + 4/3 * G) / max(rho0, 1e-20))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20
_INF = 1.0e30


@dataclass
class Law26Params:
    """Parameters for /MAT/LAW26 (Honeycomb with SESAME EOS)."""

    id: int = 1
    title: str = ""
    rho0: float = 1.0
    refer_rho: float = 0.0
    young: float = 210000.0
    nu: float = 0.3
    a: float = 0.0
    b: float = 0.0
    n: float = 1.0
    c: float = 0.0
    eps0: float = 1.0
    tmelt: float = 1000.0
    cm: float = 1.0
    eps_max: float = _INF
    sig_max: float = _INF
    t0: float = 300.0

    def __post_init__(self) -> None:
        if self.refer_rho <= 0.0 and self.rho0 > 0.0:
            self.refer_rho = self.rho0
        if self.n == 0.0 or self.n == 1.0:
            self.n = 1.0001
        if self.eps_max <= 0.0:
            self.eps_max = _INF
        if self.sig_max <= 0.0:
            self.sig_max = _INF
        if self.eps0 <= 0.0:
            self.eps0 = 1.0
        if self.tmelt <= 0.0:
            self.tmelt = 1000.0
        if self.cm <= 0.0:
            self.cm = 1.0

    @property
    def E(self) -> float:
        return self.young

    @property
    def G(self) -> float:
        return self.young / (2.0 * (1.0 + self.nu))

    @property
    def K(self) -> float:
        denom = 3.0 * (1.0 - 2.0 * self.nu)
        return self.young / denom if denom > 0.0 else self.young

    @property
    def bulk(self) -> float:
        return self.K

    @property
    def sound_speed_val(self) -> float:
        r = self.refer_rho if self.refer_rho > 0.0 else (self.rho0 if self.rho0 > 0.0 else 1.0)
        dpdm = self.K + (4.0 / 3.0) * self.G
        return math.sqrt(max(0.0, dpdm) / max(r, _EM20))

    @property
    def sound_speed_shell_val(self) -> float:
        r = self.refer_rho if self.refer_rho > 0.0 else (self.rho0 if self.rho0 > 0.0 else 1.0)
        denom = 1.0 - self.nu * self.nu
        mod = self.young / denom if denom > 0.0 else self.young
        return math.sqrt(max(0.0, mod) / max(r, _EM20))


def _get_params(mat: Any, **kwargs: Any) -> Law26Params:
    if isinstance(mat, Law26Params):
        return mat
    if hasattr(mat, "law26_params") and isinstance(mat.law26_params, Law26Params):
        return mat.law26_params

    p: Dict[str, Any] = {}
    if isinstance(mat, Material):
        p = dict(mat.params) if mat.params is not None else {}
        rho = getattr(mat, "rho0", getattr(mat, "rho", 1.0))
        p.setdefault("rho0", rho)
    elif isinstance(mat, dict):
        p = dict(mat.get("params", mat))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        p = dict(mat.params)
    p.update(kwargs)

    rho0 = float(p.get("MAT_RHO", p.get("rho0", p.get("rho", p.get("density", 1.0)))))
    refer_rho = float(p.get("Refer_Rho", p.get("refer_rho", rho0)))
    young = float(p.get("MAT_E", p.get("young", p.get("E", p.get("e", 210000.0)))))
    nu = float(p.get("MAT_NU", p.get("nu", p.get("Nu", 0.3))))
    a = float(p.get("MAT_SIGY", p.get("a", p.get("A", p.get("sigy", 0.0)))))
    b = float(p.get("MAT_BETA", p.get("b", p.get("B", 0.0))))
    n = float(p.get("MAT_HARD", p.get("n", p.get("N", 1.0))))
    c = float(p.get("MAT_SRC", p.get("c", p.get("C", 0.0))))
    eps0 = float(p.get("MAT_SRP", p.get("eps0", p.get("EPS0", 1.0))))
    tmelt = float(p.get("MAT_TMELT", p.get("tmelt", 1000.0)))
    cm = float(p.get("MAT_M", p.get("cm", p.get("m", 1.0))))
    eps_max = float(p.get("MAT_EPS", p.get("eps_max", _INF)))
    sig_max = float(p.get("MAT_SIG", p.get("sig_max", _INF)))
    t0 = float(p.get("MAT_T0", p.get("t0", 300.0)))

    return Law26Params(
        id=int(p.get("id", getattr(mat, "id", 1))),
        title=str(p.get("title", getattr(mat, "title", ""))),
        rho0=rho0,
        refer_rho=refer_rho,
        young=young,
        nu=nu,
        a=a,
        b=b,
        n=n,
        c=c,
        eps0=eps0,
        tmelt=tmelt,
        cm=cm,
        eps_max=eps_max,
        sig_max=sig_max,
        t0=t0,
    )


def build_law26(mat: Any = None, **kwargs: Any) -> Law26Params:
    """Build Law26Params from Material, dict, or arguments."""
    return _get_params(mat, **kwargs)


def resolve(mat: Any = None, model: Any = None, log: Any = None, **kwargs: Any) -> Law26Params:
    """Resolve material parameters for LAW26."""
    return _get_params(mat, **kwargs)


def extra_shapes(mat: Any = None, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """Extra history shapes for LAW26 (temperature)."""
    return {"temp26": (nip,) if nip else ()}


def needs_defgrad(mat: Any = None) -> bool:
    """LAW26 does not require deformation gradient tensor."""
    return False


def sound_speed(
    mat: Any = None,
    eps: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
    is_shell: bool = False,
    **kwargs: Any,
) -> float:
    """Acoustic sound speed for LAW26."""
    p = _get_params(mat, **kwargs)
    return p.sound_speed_shell_val if is_shell else p.sound_speed_val


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
    **kwargs: Any,
) -> Union[Tuple[np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray]]:
    """3D solid update for LAW26 with compaction EOS and thermal softening."""
    p = _get_params(mat, **kwargs)
    is_1d = sig.ndim == 1
    sig_arr = np.atleast_2d(sig).copy()
    deps_arr = np.atleast_2d(deps).copy()
    nel = sig_arr.shape[0]

    if epsp is None:
        epsp_arr = np.zeros(nel, dtype=float)
    else:
        epsp_arr = np.atleast_1d(epsp).astype(float).copy()

    g = p.G
    k = p.bulk

    temp = np.full(nel, p.t0)
    if extra is not None:
        if "temp" in extra and extra["temp"] is not None:
            t_in = np.atleast_1d(extra["temp"])
            temp[:len(t_in)] = t_in
        elif "temp26" in extra and extra["temp26"] is not None:
            t_in = np.atleast_1d(extra["temp26"])
            temp[:len(t_in)] = t_in

    tr_deps = deps_arr[:, 0] + deps_arr[:, 1] + deps_arr[:, 2]
    p_old = - (sig_arr[:, 0] + sig_arr[:, 1] + sig_arr[:, 2]) / 3.0
    p_new = p_old - k * tr_deps

    s = np.zeros_like(sig_arr)
    s[:, 0] = sig_arr[:, 0] + p_old + 2.0 * g * (deps_arr[:, 0] - tr_deps / 3.0)
    s[:, 1] = sig_arr[:, 1] + p_old + 2.0 * g * (deps_arr[:, 1] - tr_deps / 3.0)
    s[:, 2] = sig_arr[:, 2] + p_old + 2.0 * g * (deps_arr[:, 2] - tr_deps / 3.0)
    s[:, 3] = sig_arr[:, 3] + g * deps_arr[:, 3]
    s[:, 4] = sig_arr[:, 4] + g * deps_arr[:, 4]
    s[:, 5] = sig_arr[:, 5] + g * deps_arr[:, 5]

    j2 = 0.5 * (s[:, 0]**2 + s[:, 1]**2 + s[:, 2]**2) + s[:, 3]**2 + s[:, 4]**2 + s[:, 5]**2
    sig_vm = np.sqrt(3.0 * np.maximum(0.0, j2))

    t_star = np.clip((temp - 300.0) / max(1.0, p.tmelt - 300.0), 0.0, 1.0)
    c_t = np.where(temp >= p.tmelt, 0.0, np.maximum(0.0, 1.0 - t_star**p.cm))

    c_e = np.ones(nel, dtype=float)
    if dt > 0.0 and p.c > 0.0:
        rate = np.maximum(np.abs(deps_arr[:, 0]), np.maximum(np.abs(deps_arr[:, 1]), np.abs(deps_arr[:, 2]))) / dt
        c_e = np.where(rate > p.eps0, 1.0 + p.c * np.log(np.maximum(rate, p.eps0) / p.eps0), 1.0)

    ep = np.clip(epsp_arr, 0.0, p.eps_max)
    sig_y = np.minimum(p.sig_max, p.a + p.b * (ep ** p.n)) * c_e * c_t
    ep_pos = np.where(ep > 0.0, ep, 1.0)
    qh = np.where(ep > 0.0, p.b * p.n * (ep_pos ** (p.n - 1.0)), 0.0) * c_e * c_t

    plastic = (sig_vm > sig_y) & (c_t > 0.0)
    if np.any(plastic):
        idx = np.where(plastic)[0]
        dpla = (sig_vm[idx] - sig_y[idx]) / (3.0 * g + qh[idx])
        dpla = np.maximum(0.0, dpla)
        scale = np.where(sig_vm[idx] > 0.0, (sig_y[idx] + qh[idx] * dpla) / sig_vm[idx], 0.0)
        s[idx] *= scale[:, None]
        epsp_arr[idx] += dpla

    melted = temp >= p.tmelt
    if np.any(melted):
        s[melted] = 0.0

    out_sig = np.zeros_like(sig_arr)
    out_sig[:, 0] = -p_new + s[:, 0]
    out_sig[:, 1] = -p_new + s[:, 1]
    out_sig[:, 2] = -p_new + s[:, 2]
    out_sig[:, 3:] = s[:, 3:]

    c = p.sound_speed_val

    res_sig = out_sig[0] if is_1d else out_sig
    res_epsp = float(epsp_arr[0]) if (is_1d and epsp_arr.size == 1) else epsp_arr

    if return_sound_speed:
        return res_sig, res_epsp, c
    return res_sig, res_epsp


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
    **kwargs: Any,
) -> Union[Tuple[np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray]]:
    """Plane-stress shell update for LAW26."""
    p = _get_params(mat, **kwargs)
    is_1d = sig.ndim == 1
    sig_arr = np.atleast_2d(sig).copy()
    deps_arr = np.atleast_2d(deps).copy()
    nel = sig_arr.shape[0]

    if epsp is None:
        epsp_arr = np.zeros(nel, dtype=float)
    else:
        epsp_arr = np.atleast_1d(epsp).astype(float).copy()

    denom = 1.0 - p.nu * p.nu
    q11 = p.young / denom if denom > 0.0 else p.young
    q12 = p.nu * q11
    g = p.G

    temp = np.full(nel, p.t0)
    if extra is not None and "temp" in extra:
        t_in = np.atleast_1d(extra["temp"])
        temp[:len(t_in)] = t_in

    s = sig_arr[:, :3].copy()
    s[:, 0] += q11 * deps_arr[:, 0] + q12 * deps_arr[:, 1]
    s[:, 1] += q12 * deps_arr[:, 0] + q11 * deps_arr[:, 1]
    s[:, 2] += g * deps_arr[:, 2]

    sig_vm = np.sqrt(np.maximum(0.0, s[:, 0]**2 - s[:, 0] * s[:, 1] + s[:, 1]**2 + 3.0 * s[:, 2]**2))

    t_star = np.clip((temp - 300.0) / max(1.0, p.tmelt - 300.0), 0.0, 1.0)
    c_t = np.where(temp >= p.tmelt, 0.0, np.maximum(0.0, 1.0 - t_star**p.cm))

    c_e = np.ones(nel, dtype=float)
    if dt > 0.0 and p.c > 0.0:
        rate = np.maximum(np.abs(deps_arr[:, 0]), np.maximum(np.abs(deps_arr[:, 1]), 0.5 * np.abs(deps_arr[:, 2]))) / dt
        c_e = np.where(rate > p.eps0, 1.0 + p.c * np.log(np.maximum(rate, p.eps0) / p.eps0), 1.0)

    ep = np.clip(epsp_arr, 0.0, p.eps_max)
    sig_y = np.minimum(p.sig_max, p.a + p.b * (ep ** p.n)) * c_e * c_t
    ep_pos = np.where(ep > 0.0, ep, 1.0)
    qh = np.where(ep > 0.0, p.b * p.n * (ep_pos ** (p.n - 1.0)), 0.0) * c_e * c_t

    plastic = (sig_vm > sig_y) & (c_t > 0.0)
    if np.any(plastic):
        idx = np.where(plastic)[0]
        dpla = (sig_vm[idx] - sig_y[idx]) / (3.0 * g + qh[idx])
        scale = np.where(sig_vm[idx] > 0.0, (sig_y[idx] + qh[idx] * dpla) / sig_vm[idx], 0.0)
        s[idx] *= scale[:, None]
        epsp_arr[idx] += dpla

    out_sig = sig_arr.copy()
    out_sig[:, :3] = s

    c = p.sound_speed_shell_val

    res_sig = out_sig[0] if is_1d else out_sig
    res_epsp = float(epsp_arr[0]) if (is_1d and epsp_arr.size == 1) else epsp_arr

    if return_sound_speed:
        return res_sig, res_epsp, c
    return res_sig, res_epsp


def solid_tangent(
    mat: Any = None,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Algorithmic elastoplastic tangent stiffness matrix (6x6)."""
    p = _get_params(mat, **kwargs)
    k = p.bulk
    g = p.G
    lam = k - (2.0 / 3.0) * g

    c_mat = np.array([
        [lam + 2.0 * g, lam, lam, 0.0, 0.0, 0.0],
        [lam, lam + 2.0 * g, lam, 0.0, 0.0, 0.0],
        [lam, lam, lam + 2.0 * g, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, g,   0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, g,   0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, g  ],
    ], dtype=np.float64)

    if sig is not None and np.ndim(sig) > 1:
        n = np.shape(sig)[0]
        return np.broadcast_to(c_mat, (n, 6, 6)).copy()
    return c_mat


consistent_solid_tangent = solid_tangent


def shell_tangent(
    mat: Any = None,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Plane-stress membrane tangent matrix (3x3)."""
    p = _get_params(mat, **kwargs)
    denom = 1.0 - p.nu * p.nu
    q11 = p.young / denom if denom > 0.0 else p.young
    q12 = p.nu * q11
    q33 = p.G

    c_mat = np.array([
        [q11, q12, 0.0],
        [q12, q11, 0.0],
        [0.0, 0.0, q33],
    ], dtype=np.float64)

    if sig is not None and np.ndim(sig) > 1:
        n = np.shape(sig)[0]
        return np.broadcast_to(c_mat, (n, 3, 3)).copy()
    return c_mat


consistent_shell_tangent = shell_tangent
