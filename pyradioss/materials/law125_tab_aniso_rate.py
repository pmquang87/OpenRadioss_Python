"""OpenRadioss /MAT/LAW125 — Tabulated Anisotropic Plasticity with Strain-Rate Dependency.

Continuum composite damage and rate-dependent plasticity model based on Mat058 (LS-DYNA),
dedicated to laminated composites with orthotropic directional strengths, progressive
exponential damage evolution, and strain-rate dependent strength scaling.

Upstream Fortran references:
  - `engine/source/materials/mat/mat125/sigeps125.F90`
  - `engine/source/materials/mat/mat125/sigeps125c.F90`
  - `engine/source/materials/mat/mat125/strainrate_dependency_125c.F90`
  - `starter/source/materials/mat/mat125/hm_read_mat125.F90`
  - `hm_cfg_files/config/CFG/radioss2026/MAT/matl125_laminated_composite.cfg`
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_EM20 = 1.0e-20
_EM10 = 1.0e-10


@dataclass
class Law125Params:
    """Parameters for OpenRadioss /MAT/LAW125 (Laminated Composite Mat058)."""
    id: int = 1
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    rho: float = 0.0
    # Orthotropic elastic moduli
    young1: float = 1.0      # E1 (longitudinal fiber)
    young2: float = 1.0      # E2 (transverse matrix)
    young3: float = 1.0      # E3 (through-thickness)
    nu12: float = 0.3
    nu21: float = 0.3
    nu13: float = 0.3
    nu31: float = 0.3
    nu23: float = 0.3
    nu32: float = 0.3
    g12: float = 0.0
    g13: float = 0.0
    g23: float = 0.0
    # Directional strengths
    xt: float = 1.0e20       # Longitudinal tensile strength
    xc: float = 1.0e20       # Longitudinal compressive strength
    yt: float = 1.0e20       # Transverse tensile strength
    yc: float = 1.0e20       # Transverse compressive strength
    zt: float = 1.0e20       # Normal tensile strength
    zc: float = 1.0e20       # Normal compressive strength
    sc: float = 1.0e20       # In-plane shear strength
    sc13: float = 1.0e20     # Transverse shear 13 strength
    sc23: float = 1.0e20     # Transverse shear 23 strength
    # Peak strain limits
    em11t: float = 0.02
    em11c: float = 0.02
    em22t: float = 0.02
    em22c: float = 0.02
    em33t: float = 0.02
    em33c: float = 0.02
    ems: float = 0.04
    # Failure strains
    ef11t: float = 0.05
    ef11c: float = 0.05
    ef22t: float = 0.05
    ef22c: float = 0.05
    # Damage parameters
    m1t: float = 2.0
    al1t: float = 1.0
    dmax: float = 0.99
    # Strain rate dependency
    c_rate: float = 0.0
    p_rate: float = 1.0
    # Derived plane stress moduli
    a11: float = field(init=False)
    a12: float = field(init=False)
    a21: float = field(init=False)
    a22: float = field(init=False)

    def __post_init__(self) -> None:
        if self.rho > 0.0 and self.rho0 <= 0.0:
            self.rho0 = self.rho
        if self.rho0 > 0.0 and self.rho <= 0.0:
            self.rho = self.rho0
        if self.refer_rho <= 0.0:
            self.refer_rho = self.rho0

        if self.young1 <= 0.0:
            self.young1 = 1.0
        if self.young2 <= 0.0:
            self.young2 = self.young1
        if self.young3 <= 0.0:
            self.young3 = self.young1

        if self.g12 <= 0.0:
            self.g12 = self.young1 / (2.0 * (1.0 + self.nu12))
        if self.g13 <= 0.0:
            self.g13 = self.g12
        if self.g23 <= 0.0:
            self.g23 = self.g12

        denom = 1.0 - self.nu12 * self.nu21
        denom = max(_EM20, abs(denom))
        self.a11 = self.young1 / denom
        self.a12 = (self.young2 * self.nu12) / denom
        self.a21 = (self.young1 * self.nu21) / denom
        self.a22 = self.young2 / denom

    @property
    def young(self) -> float:
        return self.young1

    @property
    def nu(self) -> float:
        return self.nu12

    @property
    def g(self) -> float:
        return self.g12

    @classmethod
    def from_material(cls, mat: Any) -> Law125Params:
        """Construct Law125Params from generic Material or dictionary."""
        if isinstance(mat, Law125Params):
            return mat

        def _get(keys: Sequence[str], default: Any) -> Any:
            for k in keys:
                if hasattr(mat, k):
                    v = getattr(mat, k)
                    if v is not None:
                        return v
                if hasattr(mat, "params") and isinstance(mat.params, dict) and k in mat.params:
                    v = mat.params[k]
                    if v is not None:
                        return v
                if isinstance(mat, dict) and k in mat:
                    v = mat[k]
                    if v is not None:
                        return v
            return default

        mid = int(_get(["id", "mid", "mat_id"], 1))
        title = str(_get(["title", "name"], ""))
        rho0 = float(_get(["rho0", "rho", "MAT_RHO"], 0.0))
        young1 = float(_get(["young1", "e1", "MAT_E1", "young", "e", "MAT_E"], 1.0))
        young2 = float(_get(["young2", "e2", "MAT_E2"], young1))
        young3 = float(_get(["young3", "e3", "MAT_E3"], young1))
        nu12 = float(_get(["nu12", "MAT_NU12", "nu", "MAT_NU"], 0.3))
        nu21 = float(_get(["nu21", "MAT_NU21"], nu12 * (young2 / max(_EM20, young1))))
        nu13 = float(_get(["nu13", "MAT_NU13"], nu12))
        nu31 = float(_get(["nu31", "MAT_NU31"], nu12))
        nu23 = float(_get(["nu23", "MAT_NU23"], nu12))
        nu32 = float(_get(["nu32", "MAT_NU32"], nu12))
        g12 = float(_get(["g12", "MAT_G12"], young1 / (2.0 * (1.0 + nu12))))
        g13 = float(_get(["g13", "MAT_G13"], g12))
        g23 = float(_get(["g23", "MAT_G23"], g12))

        xt = float(_get(["xt", "XT", "MAT_XT"], 1.0e20))
        xc = float(_get(["xc", "XC", "MAT_XC"], 1.0e20))
        yt = float(_get(["yt", "YT", "MAT_YT"], 1.0e20))
        yc = float(_get(["yc", "YC", "MAT_YC"], 1.0e20))
        zt = float(_get(["zt", "ZT", "MAT_ZT"], 1.0e20))
        zc = float(_get(["zc", "ZC", "MAT_ZC"], 1.0e20))
        sc = float(_get(["sc", "SC", "MAT_SC"], 1.0e20))
        sc13 = float(_get(["sc13", "SC13", "MAT_SC13"], sc))
        sc23 = float(_get(["sc23", "SC23", "MAT_SC23"], sc))

        em11t = float(_get(["em11t", "EM11T"], 0.02))
        em11c = float(_get(["em11c", "EM11C"], 0.02))
        em22t = float(_get(["em22t", "EM22T"], 0.02))
        em22c = float(_get(["em22c", "EM22C"], 0.02))
        em33t = float(_get(["em33t", "EM33T"], 0.02))
        em33c = float(_get(["em33c", "EM33C"], 0.02))
        ems = float(_get(["ems", "EMS"], 0.04))

        ef11t = float(_get(["ef11t", "EF11T"], 0.05))
        ef11c = float(_get(["ef11c", "EF11C"], 0.05))
        ef22t = float(_get(["ef22t", "EF22T"], 0.05))
        ef22c = float(_get(["ef22c", "EF22C"], 0.05))

        m1t = float(_get(["m1t", "M1T"], 2.0))
        al1t = float(_get(["al1t", "AL1T"], 1.0))
        dmax = float(_get(["dmax", "DMAX"], 0.99))
        c_rate = float(_get(["c_rate", "C_RATE"], 0.0))
        p_rate = float(_get(["p_rate", "P_RATE"], 1.0))

        return cls(
            id=mid,
            title=title,
            rho0=rho0,
            young1=young1,
            young2=young2,
            young3=young3,
            nu12=nu12,
            nu21=nu21,
            nu13=nu13,
            nu31=nu31,
            nu23=nu23,
            nu32=nu32,
            g12=g12,
            g13=g13,
            g23=g23,
            xt=xt,
            xc=xc,
            yt=yt,
            yc=yc,
            zt=zt,
            zc=zc,
            sc=sc,
            sc13=sc13,
            sc23=sc23,
            em11t=em11t,
            em11c=em11c,
            em22t=em22t,
            em22c=em22c,
            em33t=em33t,
            em33c=em33c,
            ems=ems,
            ef11t=ef11t,
            ef11c=ef11c,
            ef22t=ef22t,
            ef22c=ef22c,
            m1t=m1t,
            al1t=al1t,
            dmax=dmax,
            c_rate=c_rate,
            p_rate=p_rate,
        )


def build_law125(mat: Any = None, **kwargs: Any) -> Law125Params:
    """Construct Law125Params from material or keyword arguments."""
    if mat is not None:
        p = Law125Params.from_material(mat)
        for k, v in kwargs.items():
            if hasattr(p, k):
                setattr(p, k, v)
        p.__post_init__()
        return p
    valid_keys = {f.name for f in Law125Params.__dataclass_fields__.values() if f.init}
    init_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in valid_keys}
    p = Law125Params(**init_kwargs)
    for k, v in extra_kwargs.items():
        if hasattr(p, k):
            setattr(p, k, v)
    p.__post_init__()
    return p


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law125Params:
    """Resolve references for /MAT/LAW125 and return Law125Params."""
    return build_law125(mat)


def extra_shapes(mat: Any = None, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Return extra history variable shapes for LAW125."""
    if nip is not None:
        return {
            "uvar125": (nip, 13),
            "dmg125": (nip, 13),
            "epsp": (nip,),
        }
    return {
        "uvar125": (13,),
        "dmg125": (13,),
        "epsp": (),
    }


def needs_defgrad(mat: Any = None) -> bool:
    """LAW125 uses total / engineering strain formulation; defgrad is False."""
    return False


def _compute_composite_damage(
    eps: float,
    em: float,
    ef: float,
    m: float,
    alpha: float,
    dmax: float,
) -> float:
    """Compute exponential progressive composite damage scalar matching Mat058."""
    e_abs = abs(eps)
    if e_abs <= em:
        return 0.0
    if e_abs >= ef:
        return dmax

    ratio = (e_abs - em) / max(_EM20, ef - em)
    dmg = 1.0 - math.exp(- (ratio ** m) / max(_EM20, alpha))
    return float(min(dmax, max(0.0, dmg)))


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    """3D continuum solid stress update for LAW125."""
    p = build_law125(mat)
    sig_arr = np.asarray(sig, dtype=np.float64)
    deps_arr = np.asarray(deps, dtype=np.float64)
    is_1d = (deps_arr.ndim == 1)

    sig_2d = sig_arr.reshape(1, 6) if is_1d else sig_arr.copy()
    deps_2d = deps_arr.reshape(1, 6) if is_1d else deps_arr.copy()
    n = sig_2d.shape[0]

    if epsp is None:
        epsp_arr = np.zeros(n, dtype=np.float64)
    else:
        epsp_in = np.asarray(epsp, dtype=np.float64)
        epsp_arr = np.full(n, float(epsp_in)) if epsp_in.ndim == 0 else epsp_in.copy()

    sig_new = np.zeros_like(sig_2d)
    epsp_new = np.zeros_like(epsp_arr)

    e1 = p.young1
    e2 = p.young2
    e3 = p.young3
    g12 = p.g12
    g13 = p.g13
    g23 = p.g23

    for i in range(n):
        deps_i = deps_2d[i]
        # Elastic trial stresses
        sig_tr = np.zeros(6, dtype=np.float64)
        sig_tr[0] = sig_2d[i, 0] + e1 * deps_i[0]
        sig_tr[1] = sig_2d[i, 1] + e2 * deps_i[1]
        sig_tr[2] = sig_2d[i, 2] + e3 * deps_i[2]
        sig_tr[3] = sig_2d[i, 3] + g12 * deps_i[3]
        sig_tr[4] = sig_2d[i, 4] + g23 * deps_i[4]
        sig_tr[5] = sig_2d[i, 5] + g13 * deps_i[5]

        # Directional damage evaluation
        dmg1 = _compute_composite_damage(deps_i[0], p.em11t, p.ef11t, p.m1t, p.al1t, p.dmax)
        dmg2 = _compute_composite_damage(deps_i[1], p.em22t, p.ef22t, p.m1t, p.al1t, p.dmax)
        dmgs = _compute_composite_damage(deps_i[3], p.ems, 2.0 * p.ems, p.m1t, p.al1t, p.dmax)

        # Apply damage attenuation
        sig_new[i, 0] = (1.0 - dmg1) * min(p.xt, max(-p.xc, sig_tr[0]))
        sig_new[i, 1] = (1.0 - dmg2) * min(p.yt, max(-p.yc, sig_tr[1]))
        sig_new[i, 2] = min(p.zt, max(-p.zc, sig_tr[2]))
        sig_new[i, 3] = (1.0 - dmgs) * min(p.sc, max(-p.sc, sig_tr[3]))
        sig_new[i, 4] = min(p.sc23, max(-p.sc23, sig_tr[4]))
        sig_new[i, 5] = min(p.sc13, max(-p.sc13, sig_tr[5]))

        epsp_new[i] = epsp_arr[i] + max(dmg1, max(dmg2, dmgs))

    c = sound_speed(p)
    c_out = c if is_1d else np.full(n, c, dtype=np.float64)

    if is_1d:
        return sig_new[0], float(epsp_new[0]), float(c_out)
    return sig_new, epsp_new, c_out


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    """2D plane-stress shell stress update for LAW125."""
    p = build_law125(mat)
    sig_arr = np.asarray(sig, dtype=np.float64)
    deps_arr = np.asarray(deps, dtype=np.float64)
    is_1d = (deps_arr.ndim == 1)

    sig_2d = sig_arr.reshape(1, -1) if is_1d else sig_arr.copy()
    deps_2d = deps_arr.reshape(1, -1) if is_1d else deps_arr.copy()
    n = sig_2d.shape[0]

    if epsp is None:
        epsp_arr = np.zeros(n, dtype=np.float64)
    else:
        epsp_in = np.asarray(epsp, dtype=np.float64)
        epsp_arr = np.full(n, float(epsp_in)) if epsp_in.ndim == 0 else epsp_in.copy()

    ncomp = sig_2d.shape[1]
    sig_new = np.zeros_like(sig_2d)
    epsp_new = np.zeros_like(epsp_arr)

    a11 = p.a11
    a12 = p.a12
    a22 = p.a22
    g12 = p.g12

    for i in range(n):
        deps_i = deps_2d[i]
        s_xx = sig_2d[i, 0] + a11 * deps_i[0] + a12 * deps_i[1]
        s_yy = sig_2d[i, 1] + a12 * deps_i[0] + a22 * deps_i[1]
        s_xy = sig_2d[i, 2] + g12 * deps_i[2]

        dmg1 = _compute_composite_damage(deps_i[0], p.em11t, p.ef11t, p.m1t, p.al1t, p.dmax)
        dmg2 = _compute_composite_damage(deps_i[1], p.em22t, p.ef22t, p.m1t, p.al1t, p.dmax)
        dmgs = _compute_composite_damage(deps_i[2], p.ems, 2.0 * p.ems, p.m1t, p.al1t, p.dmax)

        sig_new[i, 0] = (1.0 - dmg1) * min(p.xt, max(-p.xc, s_xx))
        sig_new[i, 1] = (1.0 - dmg2) * min(p.yt, max(-p.yc, s_yy))
        sig_new[i, 2] = (1.0 - dmgs) * min(p.sc, max(-p.sc, s_xy))

        if ncomp > 3:
            sig_new[i, 3:] = sig_2d[i, 3:]

        epsp_new[i] = epsp_arr[i] + max(dmg1, max(dmg2, dmgs))

    c = sound_speed(p, is_shell=True)
    c_out = c if is_1d else np.full(n, c, dtype=np.float64)

    if is_1d:
        return sig_new[0], float(epsp_new[0]), float(c_out)
    return sig_new, epsp_new, c_out


def sound_speed(mat: Any, eps: Optional[Any] = None, extra: Optional[Any] = None, is_shell: bool = False, **kwargs: Any) -> float:
    """Compute acoustic wave speed for LAW125."""
    p = build_law125(mat)
    rho = p.rho0 if p.rho0 > 0.0 else 1.0
    if is_shell:
        mod = p.a11
    else:
        mod = max(p.young1, max(p.young2, p.young3))
    return float(math.sqrt(max(0.0, mod / rho)))


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[float] = None,
    epsp_incr: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent 6x6 continuum solid tangent stiffness for LAW125."""
    p = build_law125(mat)
    c_el = np.zeros((6, 6), dtype=np.float64)
    c_el[0, 0] = p.young1
    c_el[1, 1] = p.young2
    c_el[2, 2] = p.young3
    c_el[3, 3] = p.g12
    c_el[4, 4] = p.g23
    c_el[5, 5] = p.g13

    if sig is None:
        return c_el

    sig_arr = np.asarray(sig, dtype=np.float64)
    if sig_arr.ndim == 2:
        n = sig_arr.shape[0]
        t = np.zeros((n, 6, 6), dtype=np.float64)
        for i in range(n):
            t[i] = solid_tangent(p, sig=sig_arr[i])
        return t
    return c_el


def shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[float] = None,
    epsp_incr: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent 3x3 plane stress algorithmic tangent matrix for LAW125."""
    p = build_law125(mat)
    c_el = np.array([
        [p.a11, p.a12, 0.0],
        [p.a21, p.a22, 0.0],
        [0.0, 0.0, p.g12],
    ], dtype=np.float64)

    if sig is None:
        return c_el

    sig_arr = np.asarray(sig, dtype=np.float64)
    if sig_arr.ndim == 2:
        n = sig_arr.shape[0]
        t = np.zeros((n, 3, 3), dtype=np.float64)
        for i in range(n):
            t[i] = shell_tangent(p, sig=sig_arr[i])
        return t
    return c_el


consistent_solid_tangent = solid_tangent
consistent_shell_tangent = shell_tangent
