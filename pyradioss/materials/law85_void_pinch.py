"""LAW85 — Void Material with Shell Transverse Pinching (/MAT/LAW85, /MAT/VOID).

Upstream OpenRadioss Fortran reference:
- Starter Card Reader:
  `starter/source/materials/mat/mat085/lecm85_void.F`
- 2D Shells (Plane Stress & Transverse Pinching) Constitutive Update:
  `engine/source/materials/mat/mat085/sigeps85c_void.F`
- HyperMesh CFG Schema:
  `hm_cfg_files/config/CFG/Keyword971/MAT/mat_084_85.cfg`

Theory:
-------
Elastic void material formulation designed for dummy layers, gap elements,
transverse pinching, and non-structural contact surfaces:
1. Linear plane-stress elastic matrix:
   A_1 = E / (1 - nu^2)
   A_2 = nu * E / (1 - nu^2)
   G   = 1/2 * E / (1 + nu)

2. Stress incremental update:
   sigma_xx = sigma_xx_old + A_1 * deps_xx + A_2 * deps_yy
   sigma_yy = sigma_yy_old + A_2 * deps_xx + A_1 * deps_yy
   sigma_xy = sigma_xy_old + G   * deps_xy
   sigma_yz = sigma_yz_old + G   * deps_yz
   sigma_zx = sigma_zx_old + G   * deps_zx

3. Transverse pinch support:
   Supports through-thickness stretch deps_zz and transverse contact normal stress.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20


@dataclass
class Law85Params:
    """Parameters for OpenRadioss /MAT/LAW85 (Void material with shell transverse pinching)."""
    id: int = 1
    title: str = ""
    law: int = 85
    law_name: str = "LAW85"

    # Density
    rho0: float = 1.0
    rhor: float = 1.0

    # Elastic Moduli
    young: float = 1000.0
    nu: float = 0.3

    # Derived constants
    g: float = field(init=False)
    bulk: float = field(init=False)
    a1: float = field(init=False)
    a2: float = field(init=False)
    a11_2d: float = field(init=False)
    a12_2d: float = field(init=False)
    c_solid: float = field(init=False)
    c_shell: float = field(init=False)

    def __post_init__(self) -> None:
        if self.rho0 <= 0.0:
            self.rho0 = 1.0
        if self.rhor <= 0.0:
            self.rhor = self.rho0

        e = self.young
        nu = self.nu
        self.g = 0.5 * e / max(1.0 + nu, _EM20)
        self.bulk = e / max(3.0 * (1.0 - 2.0 * nu), _EM20)
        denom_3d = (1.0 + nu) * (1.0 - 2.0 * nu)
        self.a1 = (e * (1.0 - nu) / denom_3d) if abs(denom_3d) > _EM20 else (e + 4.0 / 3.0 * self.g)
        self.a2 = (self.a1 * nu / max(1.0 - nu, _EM20))

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


def build_law85(mat_def: Any = None, **kwargs: Any) -> Law85Params:
    """Construct Law85Params from a Material entity, dictionary, or keyword arguments."""
    if isinstance(mat_def, Law85Params):
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
    title = str(data.get("title", f"LAW85_{mat_id}"))

    rho0 = _extract_val(data, ["MAT_RHO", "rho0", "rho", "density"], 1.0)
    rhor = _extract_val(data, ["Refer_Rho", "rhor", "ref_rho"], rho0)

    young = _extract_val(data, ["MAT_E", "young", "e", "E"], 1000.0)
    nu = _extract_val(data, ["MAT_NU", "nu", "poisson"], 0.3)

    return Law85Params(
        id=mat_id,
        title=title,
        rho0=rho0,
        rhor=rhor,
        young=young,
        nu=nu,
    )


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law85Params:
    """Resolve and cache Law85Params from a Material or dict."""
    if isinstance(mat, Law85Params):
        return mat
    cached = getattr(mat, "_cached_law85", None)
    if cached is None:
        cached = build_law85(mat)
        try:
            setattr(mat, "_cached_law85", cached)
        except Exception:
            pass
    return cached


def needs_defgrad(mat: Any = None) -> bool:
    """Return False: LAW85 is an incremental elastic formulation."""
    return False


def extra_shapes(mat: Any = None, nip: Optional[int] = 1) -> Dict[str, Tuple[int, ...]]:
    """Persistent history variables for LAW85."""
    if nip is not None and nip > 1:
        return {"uvar85": (nip, 2), "uvar": (nip, 2)}
    return {"uvar85": (2,), "uvar": (2,)}


def sound_speed(
    mat: Any,
    eps: Any = None,
    extra: Any = None,
    is_shell: bool = False,
) -> float | np.ndarray:
    """Acoustic sound speed for LAW85."""
    p = resolve(mat)
    c_val = p.c_shell if is_shell else p.c_solid
    if eps is not None and isinstance(eps, np.ndarray) and eps.ndim > 1:
        return np.full(len(eps), c_val, dtype=float)
    return c_val


def _solid_update_single(
    p: Law85Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    off: float = 1.0,
) -> Tuple[np.ndarray, float, float]:
    """Single 3D solid continuum update for LAW85."""
    if off < 0.1:
        return np.zeros(6, dtype=float), 0.0, 0.0

    g2 = 2.0 * p.g
    dav = (deps[0] + deps[1] + deps[2]) * (p.bulk - (2.0 / 3.0) * p.g)

    sign = np.empty(6, dtype=float)
    sign[0] = sig0[0] + g2 * deps[0] + dav
    sign[1] = sig0[1] + g2 * deps[1] + dav
    sign[2] = sig0[2] + g2 * deps[2] + dav
    sign[3] = sig0[3] + p.g * deps[3]
    sign[4] = sig0[4] + p.g * deps[4]
    sign[5] = sig0[5] + p.g * deps[5]

    return sign, 0.0, p.c_solid


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
    """3D solid continuum constitutive update for /MAT/LAW85."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(6, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(float)
    nel = len(sig_arr)

    off_arr = np.ones(nel, dtype=float)
    if extra is not None and isinstance(extra, dict) and "off" in extra:
        o = np.atleast_1d(extra["off"]).astype(float)
        off_arr[:min(nel, len(o))] = o[:nel]

    sig_out = np.zeros_like(sig_arr)
    epsp_out = np.zeros(nel, dtype=float)
    c_out = np.zeros(nel, dtype=float)

    for i in range(nel):
        s_i, ep_i, c_i = _solid_update_single(p, sig_arr[i], deps_arr[i], off=off_arr[i])
        sig_out[i] = s_i
        epsp_out[i] = ep_i
        c_out[i] = c_i

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
    p: Law85Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    off: float = 1.0,
) -> Tuple[np.ndarray, float, float]:
    """Single 2D plane-stress shell update matching sigeps85c_void.F."""
    if off < 0.1:
        return np.zeros(len(sig0), dtype=float), 0.0, 0.0

    sign = np.empty_like(sig0, dtype=float)
    sign[0] = sig0[0] + p.a11_2d * deps[0] + p.a12_2d * deps[1]
    sign[1] = sig0[1] + p.a12_2d * deps[0] + p.a11_2d * deps[1]
    sign[2] = sig0[2] + p.g * deps[2]
    if len(sig0) >= 5:
        sign[3] = sig0[3] + p.g * deps[3]
        sign[4] = sig0[4] + p.g * deps[4]
    if len(sig0) >= 6:
        # Transverse normal pinching stress
        sign[5] = sig0[5] + p.young * deps[5]

    return sign, 0.0, p.c_shell


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
    """2D plane-stress shell constitutive update for /MAT/LAW85."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(3, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(float)
    nel = len(sig_arr)

    off_arr = np.ones(nel, dtype=float)
    if extra is not None and isinstance(extra, dict) and "off" in extra:
        o = np.atleast_1d(extra["off"]).astype(float)
        off_arr[:min(nel, len(o))] = o[:nel]

    sig_out = np.zeros_like(sig_arr)
    epsp_out = np.zeros(nel, dtype=float)
    c_out = np.zeros(nel, dtype=float)

    for i in range(nel):
        s_i, ep_i, c_i = _shell_update_single(p, sig_arr[i], deps_arr[i], off=off_arr[i])
        sig_out[i] = s_i
        epsp_out[i] = ep_i
        c_out[i] = c_i

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
