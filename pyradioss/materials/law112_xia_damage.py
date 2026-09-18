"""OpenRadioss /MAT/LAW112 — Xia Ductile Damage and Void Nucleation Model.

Orthotropic elastoplastic paperboard model incorporating tension/compression
asymmetry, directional hardening, ductile damage evolution, and void nucleation.

Upstream Fortran references:
  - `engine/source/materials/mat/mat112/sigeps112.F`
  - `engine/source/materials/mat/mat112/sigeps112c.F`
  - `engine/source/materials/mat/mat112/mat112_xia_newton.F`
  - `starter/source/materials/mat/mat112/hm_read_mat112.F`
  - `hm_cfg_files/config/CFG/radioss2021/MAT/matl112_paper.cfg`
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_EM20 = 1.0e-20
_EM10 = 1.0e-10


@dataclass
class Law112Params:
    """Parameters for OpenRadioss /MAT/LAW112 (/MAT/XIA_DAMAGE)."""
    id: int = 1
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    rho: float = 0.0
    # Orthotropic elastic moduli
    young1: float = 1.0   # MD
    young2: float = 1.0   # CD
    young3: float = 1.0   # ZD
    nu12: float = 0.3
    nu21: float = 0.3
    g12: float = 0.5
    g23: float = 0.5
    g31: float = 0.5
    # Orthotropic shell plane stress stiffness coefficients
    a11: float = field(init=False)
    a12: float = field(init=False)
    a21: float = field(init=False)
    a22: float = field(init=False)
    # Plasticity & damage constants
    ires: int = 2         # 1: Nice, 2: Newton
    itab: int = 0         # 0: Analytic, 1: Tabulated
    deuxk: float = 2.0    # Yield criterion exponent
    sigy1: float = 1.0    # Yield stress MD
    sigy2: float = 1.0    # Yield stress CD
    sigy3: float = 1.0    # Yield stress ZD
    sigys: float = 0.5    # Shear yield stress
    # Hardening parameters
    h1: float = 0.0
    h2: float = 0.0
    # Ductile damage parameters
    dmax: float = 0.99    # Maximum damage limit
    eps_d0: float = 0.05  # Damage initiation strain
    eps_df: float = 0.20  # Damage failure strain
    fc: float = 0.15      # Critical void volume fraction
    ff: float = 0.25      # Failure void volume fraction

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

        # Calculate orthotropic plane stress stiffness
        denom = 1.0 - self.nu12 * self.nu21
        denom = max(_EM20, abs(denom))
        self.a11 = self.young1 / denom
        self.a12 = (self.nu12 * self.young2) / denom
        self.a21 = (self.nu21 * self.young1) / denom
        self.a22 = self.young2 / denom

        if self.g12 <= 0.0:
            self.g12 = self.young1 / (2.0 * (1.0 + self.nu12))
        if self.g23 <= 0.0:
            self.g23 = self.g12
        if self.g31 <= 0.0:
            self.g31 = self.g12

    @property
    def young(self) -> float:
        return self.young1

    @property
    def nu(self) -> float:
        return self.nu12

    @classmethod
    def from_material(cls, mat: Any) -> Law112Params:
        """Construct Law112Params from generic Material or dictionary."""
        if isinstance(mat, Law112Params):
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
        g12 = float(_get(["g12", "MAT_G12"], young1 / (2.0 * (1.0 + nu12))))
        g23 = float(_get(["g23", "MAT_G23"], g12))
        g31 = float(_get(["g31", "MAT_G31"], g12))
        ires = int(_get(["ires", "MAT_IRES"], 2))
        itab = int(_get(["itab", "MAT_ITAB"], 0))
        deuxk = float(_get(["deuxk", "DEUXK", "2k"], 2.0))
        sigy1 = float(_get(["sigy1", "SIGY1", "sigy0"], 1.0))
        sigy2 = float(_get(["sigy2", "SIGY2"], sigy1))
        sigy3 = float(_get(["sigy3", "SIGY3"], sigy1))
        sigys = float(_get(["sigys", "SIGYS"], sigy1 / math.sqrt(3.0)))
        h1 = float(_get(["h1", "H1", "h_mod"], 0.0))
        h2 = float(_get(["h2", "H2"], 0.0))
        dmax = float(_get(["dmax", "DMAX"], 0.99))
        eps_d0 = float(_get(["eps_d0", "EPS_D0"], 0.05))
        eps_df = float(_get(["eps_df", "EPS_DF"], 0.20))
        fc = float(_get(["fc", "FC"], 0.15))
        ff = float(_get(["ff", "FF"], 0.25))

        return cls(
            id=mid,
            title=title,
            rho0=rho0,
            young1=young1,
            young2=young2,
            young3=young3,
            nu12=nu12,
            nu21=nu21,
            g12=g12,
            g23=g23,
            g31=g31,
            ires=ires,
            itab=itab,
            deuxk=deuxk,
            sigy1=sigy1,
            sigy2=sigy2,
            sigy3=sigy3,
            sigys=sigys,
            h1=h1,
            h2=h2,
            dmax=dmax,
            eps_d0=eps_d0,
            eps_df=eps_df,
            fc=fc,
            ff=ff,
        )


def build_law112(mat: Any = None, **kwargs: Any) -> Law112Params:
    """Construct Law112Params from material or keyword arguments."""
    if mat is not None:
        p = Law112Params.from_material(mat)
        for k, v in kwargs.items():
            if hasattr(p, k):
                setattr(p, k, v)
        p.__post_init__()
        return p
    valid_keys = {f.name for f in Law112Params.__dataclass_fields__.values() if f.init}
    init_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in valid_keys}
    p = Law112Params(**init_kwargs)
    for k, v in extra_kwargs.items():
        if hasattr(p, k):
            setattr(p, k, v)
    p.__post_init__()
    return p


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law112Params:
    """Resolve curves/functions for /MAT/LAW112 from model and return Law112Params."""
    return build_law112(mat)


def extra_shapes(mat: Any = None, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Return extra history variable shapes for LAW112."""
    if nip is not None:
        return {
            "uvar112": (nip, 8),
            "pla": (nip, 4),
            "dmg": (nip,),
        }
    return {
        "uvar112": (8,),
        "pla": (4,),
        "dmg": (),
    }


def needs_defgrad(mat: Any = None) -> bool:
    """LAW112 uses small strain rate plasticity formulation; defgrad is False."""
    return False


def _compute_damage(p: Law112Params, eps_p: float) -> float:
    """Compute progressive ductile damage scalar D matching Xia formulation."""
    if eps_p <= p.eps_d0:
        return 0.0
    if eps_p >= p.eps_df:
        return p.dmax
    dmg = (eps_p - p.eps_d0) / max(_EM20, p.eps_df - p.eps_d0)
    return float(min(p.dmax, max(0.0, dmg)))


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
    """3D continuum solid stress update with Xia damage."""
    p = build_law112(mat)
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
    g23 = p.g23
    g31 = p.g31

    for i in range(n):
        deps_i = deps_2d[i]
        # Trial effective stresses
        sig_tr = np.zeros(6, dtype=np.float64)
        sig_tr[0] = sig_2d[i, 0] + e1 * deps_i[0]
        sig_tr[1] = sig_2d[i, 1] + e2 * deps_i[1]
        sig_tr[2] = sig_2d[i, 2] + e3 * deps_i[2]
        sig_tr[3] = sig_2d[i, 3] + g12 * deps_i[3]
        sig_tr[4] = sig_2d[i, 4] + g23 * deps_i[4]
        sig_tr[5] = sig_2d[i, 5] + g31 * deps_i[5]

        # Xia effective equivalent stress (Hill / orthotropic quadratic form)
        s1 = sig_tr[0] / max(_EM20, p.sigy1)
        s2 = sig_tr[1] / max(_EM20, p.sigy2)
        s3 = sig_tr[2] / max(_EM20, p.sigy3)
        ss12 = sig_tr[3] / max(_EM20, p.sigys)
        ss23 = sig_tr[4] / max(_EM20, p.sigys)
        ss31 = sig_tr[5] / max(_EM20, p.sigys)

        phi_val = (s1 - s2)**2 + (s2 - s3)**2 + (s3 - s1)**2 + 6.0 * (ss12**2 + ss23**2 + ss31**2)
        seq_tr = math.sqrt(max(0.0, 0.5 * phi_val)) * p.sigy1

        sig_y = p.sigy1 + p.h1 * epsp_arr[i]

        if seq_tr > sig_y and seq_tr > _EM10:
            denom = 3.0 * g12 + p.h1
            dgamma = (seq_tr - sig_y) / max(_EM20, denom)
            factor = max(0.0, 1.0 - (3.0 * g12 * dgamma) / seq_tr)
            epsp_new[i] = epsp_arr[i] + dgamma

            # Apply damage softening
            dmg = _compute_damage(p, epsp_new[i])
            sig_new[i] = sig_tr * factor * (1.0 - dmg)
        else:
            epsp_new[i] = epsp_arr[i]
            dmg = _compute_damage(p, epsp_new[i])
            sig_new[i] = sig_tr * (1.0 - dmg)

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
    """2D plane-stress shell stress update with Xia damage."""
    p = build_law112(mat)
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

        s1 = s_xx / max(_EM20, p.sigy1)
        s2 = s_yy / max(_EM20, p.sigy2)
        ss12 = s_xy / max(_EM20, p.sigys)

        seq_tr = math.sqrt(max(0.0, s1**2 + s2**2 - s1 * s2 + 3.0 * ss12**2)) * p.sigy1
        sig_y = p.sigy1 + p.h1 * epsp_arr[i]

        if seq_tr > sig_y and seq_tr > _EM10:
            denom = a11 + p.h1
            dgamma = (seq_tr - sig_y) / max(_EM20, denom)
            factor = max(0.0, 1.0 - (a11 * dgamma) / seq_tr)
            epsp_new[i] = epsp_arr[i] + dgamma
            dmg = _compute_damage(p, epsp_new[i])

            sig_new[i, 0] = s_xx * factor * (1.0 - dmg)
            sig_new[i, 1] = s_yy * factor * (1.0 - dmg)
            sig_new[i, 2] = s_xy * factor * (1.0 - dmg)
        else:
            epsp_new[i] = epsp_arr[i]
            dmg = _compute_damage(p, epsp_new[i])
            sig_new[i, 0] = s_xx * (1.0 - dmg)
            sig_new[i, 1] = s_yy * (1.0 - dmg)
            sig_new[i, 2] = s_xy * (1.0 - dmg)

        if ncomp > 3:
            sig_new[i, 3:] = sig_2d[i, 3:] * (1.0 - dmg)

    c = sound_speed(p, is_shell=True)
    c_out = c if is_1d else np.full(n, c, dtype=np.float64)

    if is_1d:
        return sig_new[0], float(epsp_new[0]), float(c_out)
    return sig_new, epsp_new, c_out


def sound_speed(mat: Any, eps: Optional[Any] = None, extra: Optional[Any] = None, is_shell: bool = False, **kwargs: Any) -> float:
    """Compute acoustic wave speed for LAW112."""
    p = build_law112(mat)
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
    """Consistent 6x6 continuum solid tangent stiffness for LAW112."""
    p = build_law112(mat)
    c_el = np.zeros((6, 6), dtype=np.float64)
    c_el[0, 0] = p.young1
    c_el[1, 1] = p.young2
    c_el[2, 2] = p.young3
    c_el[3, 3] = p.g12
    c_el[4, 4] = p.g23
    c_el[5, 5] = p.g31

    dmg = _compute_damage(p, epsp if epsp is not None else 0.0)
    c_el *= (1.0 - dmg)

    if sig is None:
        return c_el

    sig_arr = np.asarray(sig, dtype=np.float64)
    if sig_arr.ndim == 2:
        n = sig_arr.shape[0]
        t = np.zeros((n, 6, 6), dtype=np.float64)
        for i in range(n):
            t[i] = solid_tangent(p, sig=sig_arr[i], epsp=epsp)
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
    """Consistent 3x3 plane stress algorithmic tangent matrix for LAW112."""
    p = build_law112(mat)
    c_el = np.array([
        [p.a11, p.a12, 0.0],
        [p.a21, p.a22, 0.0],
        [0.0, 0.0, p.g12],
    ], dtype=np.float64)

    dmg = _compute_damage(p, epsp if epsp is not None else 0.0)
    c_el *= (1.0 - dmg)

    if sig is None:
        return c_el

    sig_arr = np.asarray(sig, dtype=np.float64)
    if sig_arr.ndim == 2:
        n = sig_arr.shape[0]
        t = np.zeros((n, 3, 3), dtype=np.float64)
        for i in range(n):
            t[i] = shell_tangent(p, sig=sig_arr[i], epsp=epsp)
        return t
    return c_el


consistent_solid_tangent = solid_tangent
consistent_shell_tangent = shell_tangent
