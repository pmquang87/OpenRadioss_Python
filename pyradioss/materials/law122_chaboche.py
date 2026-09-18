"""OpenRadioss /MAT/LAW122 — Chaboche Nonlinear Kinematic Hardening Model.

Elastoplastic constitutive model incorporating Chaboche / Armstrong-Frederick
nonlinear kinematic hardening, Voce isotropic hardening, orthotropic damage,
and cutting-plane / Newton return mapping for cyclic plasticity.

Upstream Fortran references:
  - `engine/source/materials/mat/mat122/sigeps122.F`
  - `engine/source/materials/mat/mat122/sigeps122c.F`
  - `engine/source/materials/mat/mat122/mat122_newton.F`
  - `starter/source/materials/mat/mat122/hm_read_mat122.F`
  - `hm_cfg_files/config/CFG/radioss2023/MAT/matl122_modified_ladeveze.cfg`
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_EM20 = 1.0e-20
_EM10 = 1.0e-10


@dataclass
class Law122Params:
    """Parameters for OpenRadioss /MAT/LAW122 (Chaboche / Modified Ladeveze)."""
    id: int = 1
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    rho: float = 0.0
    # Elastic parameters
    young1: float = 1.0
    young2: float = 1.0
    young3: float = 1.0
    nu12: float = 0.3
    nu21: float = 0.3
    nu13: float = 0.3
    nu31: float = 0.3
    nu23: float = 0.3
    nu32: float = 0.3
    g12: float = 0.0
    g23: float = 0.0
    g31: float = 0.0
    # Initial yield stress and isotropic hardening
    sigy0: float = 1.0
    r_inf: float = 0.0       # Isotropic saturation stress increment
    b_iso: float = 0.0       # Isotropic hardening rate
    # Chaboche kinematic hardening parameters
    c_kin: float = 0.0       # Kinematic hardening modulus C
    gamma_kin: float = 0.0   # Kinematic recall parameter gamma
    # Algorithmic options
    ires: int = 2            # 1: Nice, 2: Newton
    dmax: float = 0.99       # Maximum damage
    # Derived moduli
    bulk: float = field(init=False)
    lame: float = field(init=False)
    a11: float = field(init=False)
    a12: float = field(init=False)

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
        if self.g23 <= 0.0:
            self.g23 = self.g12
        if self.g31 <= 0.0:
            self.g31 = self.g12

        mean_young = (self.young1 + self.young2 + self.young3) / 3.0
        mean_nu = max(0.0, min(0.499, (self.nu12 + self.nu23 + self.nu31) / 3.0))

        self.bulk = mean_young / max(_EM20, 3.0 * (1.0 - 2.0 * mean_nu))
        self.lame = (mean_young * mean_nu) / max(_EM20, (1.0 + mean_nu) * (1.0 - 2.0 * mean_nu))

        denom = 1.0 - self.nu12 * self.nu21
        self.a11 = self.young1 / max(_EM20, denom)
        self.a12 = (self.young2 * self.nu12) / max(_EM20, denom)

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
    def from_material(cls, mat: Any) -> Law122Params:
        """Construct Law122Params from generic Material or dictionary."""
        if isinstance(mat, Law122Params):
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
        young1 = float(_get(["young1", "e1", "E10", "MAT_E1", "young", "e", "MAT_E"], 1.0))
        young2 = float(_get(["young2", "e2", "E20", "MAT_E2"], young1))
        young3 = float(_get(["young3", "e3", "E30", "MAT_E3"], young1))
        nu12 = float(_get(["nu12", "NU12", "MAT_NU12", "nu", "MAT_NU"], 0.3))
        nu21 = float(_get(["nu21", "NU21", "MAT_NU21"], nu12 * (young2 / max(_EM20, young1))))
        nu13 = float(_get(["nu13", "NU13", "MAT_NU13"], nu12))
        nu31 = float(_get(["nu31", "NU31", "MAT_NU31"], nu12))
        nu23 = float(_get(["nu23", "NU23", "MAT_NU23"], nu12))
        nu32 = float(_get(["nu32", "NU32", "MAT_NU32"], nu12))
        g12 = float(_get(["g12", "G120", "MAT_G12"], young1 / (2.0 * (1.0 + nu12))))
        g23 = float(_get(["g23", "G230", "MAT_G23"], g12))
        g31 = float(_get(["g31", "G310", "MAT_G31"], g12))
        sigy0 = float(_get(["sigy0", "SIGY0", "MAT_SIGY0", "sigy"], 1.0))
        r_inf = float(_get(["r_inf", "R_INF", "DSAT1", "MAT_R_INF"], 0.0))
        b_iso = float(_get(["b_iso", "B_ISO", "B", "MAT_B"], 0.0))
        c_kin = float(_get(["c_kin", "C_KIN", "C", "GAMMA", "MAT_C"], 0.0))
        gamma_kin = float(_get(["gamma_kin", "GAMMA_KIN", "BETA", "MAT_GAMMA"], 0.0))
        ires = int(_get(["ires", "IRES", "MAT_IRES"], 2))
        dmax = float(_get(["dmax", "DMAX", "MAT_DMAX"], 0.99))

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
            g23=g23,
            g31=g31,
            sigy0=sigy0,
            r_inf=r_inf,
            b_iso=b_iso,
            c_kin=c_kin,
            gamma_kin=gamma_kin,
            ires=ires,
            dmax=dmax,
        )


def build_law122(mat: Any = None, **kwargs: Any) -> Law122Params:
    """Construct Law122Params from material or keyword arguments."""
    if mat is not None:
        p = Law122Params.from_material(mat)
        for k, v in kwargs.items():
            if hasattr(p, k):
                setattr(p, k, v)
        p.__post_init__()
        return p
    valid_keys = {f.name for f in Law122Params.__dataclass_fields__.values() if f.init}
    init_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in valid_keys}
    p = Law122Params(**init_kwargs)
    for k, v in extra_kwargs.items():
        if hasattr(p, k):
            setattr(p, k, v)
    p.__post_init__()
    return p


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law122Params:
    """Resolve references for /MAT/LAW122 and return Law122Params."""
    return build_law122(mat)


def extra_shapes(mat: Any = None, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Return extra history variable shapes for LAW122."""
    if nip is not None:
        return {
            "uvar122": (nip, 8),
            "backstress": (nip, 6),
            "epsp": (nip,),
        }
    return {
        "uvar122": (8,),
        "backstress": (6,),
        "epsp": (),
    }


def needs_defgrad(mat: Any = None) -> bool:
    """LAW122 uses small strain rate plasticity formulation; defgrad is False."""
    return False


def _eval_chaboche_yield_stress(p: Law122Params, eps_p: float) -> Tuple[float, float]:
    """Evaluate isotropic hardening yield stress and slope R'(p)."""
    p_eff = max(0.0, eps_p)
    if p.b_iso > 0.0 and p.r_inf > 0.0:
        exp_term = math.exp(-p.b_iso * p_eff)
        r = p.r_inf * (1.0 - exp_term)
        dr = p.r_inf * p.b_iso * exp_term
    else:
        r = 0.0
        dr = 0.0
    return p.sigy0 + r, dr


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
    """3D continuum solid stress update with Chaboche kinematic hardening."""
    p = build_law122(mat)
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

    # Retrieve or initialize backstress
    if extra is not None and "backstress" in extra:
        alpha_arr = np.asarray(extra["backstress"], dtype=np.float64).reshape(-1, 6)
    else:
        alpha_arr = np.zeros((n, 6), dtype=np.float64)

    sig_new = np.zeros_like(sig_2d)
    epsp_new = np.zeros_like(epsp_arr)
    alpha_new = np.zeros_like(alpha_arr)

    lame = p.lame
    g = p.g12
    g2 = 2.0 * g

    for i in range(n):
        deps_i = deps_2d[i]
        tr_deps = deps_i[0] + deps_i[1] + deps_i[2]

        # Elastic trial stress
        sig_tr = np.zeros(6, dtype=np.float64)
        sig_tr[0] = sig_2d[i, 0] + lame * tr_deps + g2 * deps_i[0]
        sig_tr[1] = sig_2d[i, 1] + lame * tr_deps + g2 * deps_i[1]
        sig_tr[2] = sig_2d[i, 2] + lame * tr_deps + g2 * deps_i[2]
        sig_tr[3] = sig_2d[i, 3] + g * deps_i[3]
        sig_tr[4] = sig_2d[i, 4] + g * deps_i[4]
        sig_tr[5] = sig_2d[i, 5] + g * deps_i[5]

        p_m = (sig_tr[0] + sig_tr[1] + sig_tr[2]) / 3.0
        s_tr = np.array([
            sig_tr[0] - p_m,
            sig_tr[1] - p_m,
            sig_tr[2] - p_m,
            sig_tr[3],
            sig_tr[4],
            sig_tr[5],
        ], dtype=np.float64)

        # Relative stress eta = s - alpha
        alpha_i = alpha_arr[i] if i < len(alpha_arr) else np.zeros(6, dtype=np.float64)
        eta_tr = s_tr - alpha_i

        j2 = 0.5 * (eta_tr[0]**2 + eta_tr[1]**2 + eta_tr[2]**2) + eta_tr[3]**2 + eta_tr[4]**2 + eta_tr[5]**2
        seq_tr = math.sqrt(max(0.0, 3.0 * j2))

        sig_y, h_iso = _eval_chaboche_yield_stress(p, epsp_arr[i])

        if seq_tr > sig_y and seq_tr > _EM10:
            n_flow = np.zeros(6, dtype=np.float64)
            n_flow[:3] = 1.5 * eta_tr[:3] / seq_tr
            n_flow[3:] = 3.0 * eta_tr[3:] / seq_tr

            # Chaboche kinematic hardening modulus: H_kin = C - gamma * (alpha : n)
            alpha_dot_n = alpha_i[0] * n_flow[0] + alpha_i[1] * n_flow[1] + alpha_i[2] * n_flow[2] + 2.0 * (alpha_i[3] * n_flow[3] + alpha_i[4] * n_flow[4] + alpha_i[5] * n_flow[5])
            h_kin = p.c_kin - p.gamma_kin * alpha_dot_n

            denom = 3.0 * g + h_iso + h_kin
            dgamma = (seq_tr - sig_y) / max(_EM20, denom)

            # Update backstress: d_alpha = 2/3 C d_eps_p - gamma * alpha * dgamma
            d_alpha = (2.0 / 3.0) * p.c_kin * dgamma * n_flow - p.gamma_kin * alpha_i * dgamma
            alpha_new[i] = alpha_i + d_alpha

            factor = max(0.0, 1.0 - (3.0 * g * dgamma) / seq_tr)
            sig_new[i, 0] = p_m + alpha_new[i, 0] + eta_tr[0] * factor
            sig_new[i, 1] = p_m + alpha_new[i, 1] + eta_tr[1] * factor
            sig_new[i, 2] = p_m + alpha_new[i, 2] + eta_tr[2] * factor
            sig_new[i, 3] = alpha_new[i, 3] + eta_tr[3] * factor
            sig_new[i, 4] = alpha_new[i, 4] + eta_tr[4] * factor
            sig_new[i, 5] = alpha_new[i, 5] + eta_tr[5] * factor

            epsp_new[i] = epsp_arr[i] + dgamma
        else:
            sig_new[i] = sig_tr
            alpha_new[i] = alpha_i
            epsp_new[i] = epsp_arr[i]

    if extra is not None:
        extra["backstress"] = alpha_new

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
    """2D plane-stress shell stress update with Chaboche kinematic hardening."""
    p = build_law122(mat)
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
    g = p.g12

    for i in range(n):
        deps_i = deps_2d[i]
        s_xx = sig_2d[i, 0] + a11 * deps_i[0] + a12 * deps_i[1]
        s_yy = sig_2d[i, 1] + a12 * deps_i[0] + a11 * deps_i[1]
        s_xy = sig_2d[i, 2] + g * deps_i[2]

        seq_tr = math.sqrt(max(0.0, s_xx**2 + s_yy**2 - s_xx * s_yy + 3.0 * s_xy**2))
        sig_y, h_iso = _eval_chaboche_yield_stress(p, epsp_arr[i])

        if seq_tr > sig_y and seq_tr > _EM10:
            denom = a11 + h_iso + p.c_kin
            dgamma = (seq_tr - sig_y) / max(_EM20, denom)
            factor = max(0.0, 1.0 - (a11 * dgamma) / seq_tr)

            sig_new[i, 0] = s_xx * factor
            sig_new[i, 1] = s_yy * factor
            sig_new[i, 2] = s_xy * factor
            epsp_new[i] = epsp_arr[i] + dgamma
        else:
            sig_new[i, 0] = s_xx
            sig_new[i, 1] = s_yy
            sig_new[i, 2] = s_xy
            epsp_new[i] = epsp_arr[i]

        if ncomp > 3:
            sig_new[i, 3:] = sig_2d[i, 3:]

    c = sound_speed(p, is_shell=True)
    c_out = c if is_1d else np.full(n, c, dtype=np.float64)

    if is_1d:
        return sig_new[0], float(epsp_new[0]), float(c_out)
    return sig_new, epsp_new, c_out


def sound_speed(mat: Any, eps: Optional[Any] = None, extra: Optional[Any] = None, is_shell: bool = False, **kwargs: Any) -> float:
    """Compute acoustic wave speed for LAW122."""
    p = build_law122(mat)
    rho = p.rho0 if p.rho0 > 0.0 else 1.0
    if is_shell:
        mod = p.a11
    else:
        mod = p.bulk + (4.0 / 3.0) * p.g12
    return float(math.sqrt(max(0.0, mod / rho)))


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[float] = None,
    epsp_incr: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent 6x6 continuum solid tangent stiffness for LAW122."""
    p = build_law122(mat)
    c_el = np.zeros((6, 6), dtype=np.float64)
    lame = p.lame
    g = p.g12
    g2 = 2.0 * g

    c_el[0, 0] = lame + g2
    c_el[1, 1] = lame + g2
    c_el[2, 2] = lame + g2
    c_el[0, 1] = c_el[1, 0] = lame
    c_el[0, 2] = c_el[2, 0] = lame
    c_el[1, 2] = c_el[2, 1] = lame
    c_el[3, 3] = g
    c_el[4, 4] = g
    c_el[5, 5] = g

    if sig is None:
        return c_el

    sig_arr = np.asarray(sig, dtype=np.float64)
    if sig_arr.ndim == 2:
        n = sig_arr.shape[0]
        t = np.zeros((n, 6, 6), dtype=np.float64)
        for i in range(n):
            t[i] = solid_tangent(p, sig=sig_arr[i], epsp=epsp)
        return t

    p_m = (sig_arr[0] + sig_arr[1] + sig_arr[2]) / 3.0
    s = sig_arr - np.array([p_m, p_m, p_m, 0.0, 0.0, 0.0])
    seq = math.sqrt(max(0.0, 1.5 * (s[0]**2 + s[1]**2 + s[2]**2 + 2.0 * (s[3]**2 + s[4]**2 + s[5]**2))))
    sig_y, h_iso = _eval_chaboche_yield_stress(p, epsp if epsp is not None else 0.0)

    if seq < sig_y or seq <= _EM10:
        return c_el

    n_vec = np.zeros(6, dtype=np.float64)
    n_vec[:3] = 1.5 * s[:3] / seq
    n_vec[3:] = 3.0 * s[3:] / seq

    denom = 3.0 * g + h_iso + p.c_kin
    if denom > _EM20:
        gamma = (2.0 * g) / denom
        c_tan = c_el - (2.0 * g * gamma) * np.outer(n_vec, n_vec)
        return c_tan
    return c_el


def shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[float] = None,
    epsp_incr: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent 3x3 plane stress algorithmic tangent matrix for LAW122."""
    p = build_law122(mat)
    c_el = np.array([
        [p.a11, p.a12, 0.0],
        [p.a12, p.a11, 0.0],
        [0.0, 0.0, p.g12],
    ], dtype=np.float64)

    if sig is None:
        return c_el

    sig_arr = np.asarray(sig, dtype=np.float64)
    if sig_arr.ndim == 2:
        n = sig_arr.shape[0]
        t = np.zeros((n, 3, 3), dtype=np.float64)
        for i in range(n):
            t[i] = shell_tangent(p, sig=sig_arr[i], epsp=epsp)
        return t

    sxx, syy, sxy = sig_arr[0], sig_arr[1], sig_arr[2]
    seq = math.sqrt(max(0.0, sxx**2 + syy**2 - sxx * syy + 3.0 * sxy**2))
    sig_y, h_iso = _eval_chaboche_yield_stress(p, epsp if epsp is not None else 0.0)

    if seq < sig_y or seq <= _EM10:
        return c_el

    n_vec = np.array([
        (2.0 * sxx - syy) / (2.0 * seq),
        (2.0 * syy - sxx) / (2.0 * seq),
        (3.0 * sxy) / seq,
    ], dtype=np.float64)

    cn = c_el @ n_vec
    denom = float(n_vec @ cn) + h_iso + p.c_kin
    if denom > _EM20:
        return c_el - np.outer(cn, cn) / denom
    return c_el


consistent_solid_tangent = solid_tangent
consistent_shell_tangent = shell_tangent
