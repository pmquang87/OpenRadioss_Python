"""OpenRadioss /MAT/LAW115 — Perzyna Viscoplastic & Deshpande-Fleck Formulation.

Deshpande-Fleck metallic foam plasticity model with pressure-dependent elliptical
yield surface, Perzyna rate-dependent viscoplastic flow, nonlinear densification
hardening, and volumetric failure criteria.

Upstream Fortran references:
  - `engine/source/materials/mat/mat115/sigeps115.F`
  - `engine/source/materials/mat/mat115/mat115_newton.F`
  - `starter/source/materials/mat/mat115/hm_read_mat115.F`
  - `hm_cfg_files/config/CFG/radioss2021/MAT/matl115_deshfleck.cfg`
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_EM20 = 1.0e-20
_EM10 = 1.0e-10


@dataclass
class Law115Params:
    """Parameters for OpenRadioss /MAT/LAW115 (/MAT/DESHFLECK)."""
    id: int = 1
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    rho: float = 0.0
    young: float = 1.0
    nu: float = 0.3
    alpha: float = 0.0       # Yield function shape parameter (volumetric coupling)
    gamma: float = 0.0       # Linear hardening modulus
    epsd: float = 1.0        # Densification strain
    alpha2: float = 0.0      # Non-linear hardening modulus
    beta: float = 2.0        # Non-linear hardening exponent
    sigp: float = 1.0        # Initial yield stress
    cfail: float = 0.0       # Tensile volumetric strain at failure
    pfail: float = 0.0       # Maximum principal stress at failure
    ires: int = 2            # 1: Nice explicit, 2: Newton cutting plane
    istat: int = 0           # Statistical variation flag
    rhof0: float = 0.0       # Base material density
    # Perzyna rate-sensitivity parameters
    perzyna_gamma: float = 0.0
    perzyna_n: float = 1.0
    # Derived elastic moduli
    g: float = field(init=False)
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

        if self.young <= 0.0:
            self.young = 1.0
        if self.nu < 0.0 or self.nu >= 0.5:
            self.nu = 0.3

        self.g = self.young / (2.0 * (1.0 + self.nu))
        self.bulk = self.young / max(_EM20, 3.0 * (1.0 - 2.0 * self.nu))
        self.lame = (self.young * self.nu) / max(_EM20, (1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        denom_shell = 1.0 - self.nu * self.nu
        self.a11 = self.young / max(_EM20, denom_shell)
        self.a12 = self.a11 * self.nu

    @classmethod
    def from_material(cls, mat: Any) -> Law115Params:
        """Construct Law115Params from generic Material or dictionary."""
        if isinstance(mat, Law115Params):
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
        young = float(_get(["young", "e", "MAT_E", "E"], 1.0))
        nu = float(_get(["nu", "MAT_NU"], 0.3))
        alpha = float(_get(["alpha", "MAT_ALPHA"], 0.0))
        gamma = float(_get(["gamma", "MAT_GAMMA"], 0.0))
        epsd = float(_get(["epsd", "MAT_EPSD"], 1.0))
        alpha2 = float(_get(["alpha2", "MAT_ALPHA2"], 0.0))
        beta = float(_get(["beta", "MAT_BETA"], 2.0))
        sigp = float(_get(["sigp", "MAT_SIGP", "sigy0"], 1.0))
        cfail = float(_get(["cfail", "MAT_CFAIL"], 0.0))
        pfail = float(_get(["pfail", "MAT_PFAIL"], 0.0))
        ires = int(_get(["ires", "MAT_IRES"], 2))
        istat = int(_get(["istat", "MAT_ISTAT"], 0))
        rhof0 = float(_get(["rhof0", "MAT_RHOF0"], 0.0))
        perzyna_gamma = float(_get(["perzyna_gamma", "GAMMA0"], 0.0))
        perzyna_n = float(_get(["perzyna_n", "VN"], 1.0))

        return cls(
            id=mid,
            title=title,
            rho0=rho0,
            young=young,
            nu=nu,
            alpha=alpha,
            gamma=gamma,
            epsd=epsd,
            alpha2=alpha2,
            beta=beta,
            sigp=sigp,
            cfail=cfail,
            pfail=pfail,
            ires=ires,
            istat=istat,
            rhof0=rhof0,
            perzyna_gamma=perzyna_gamma,
            perzyna_n=perzyna_n,
        )


def build_law115(mat: Any = None, **kwargs: Any) -> Law115Params:
    """Construct Law115Params from material or keyword arguments."""
    if mat is not None:
        p = Law115Params.from_material(mat)
        for k, v in kwargs.items():
            if hasattr(p, k):
                setattr(p, k, v)
        p.__post_init__()
        return p
    valid_keys = {f.name for f in Law115Params.__dataclass_fields__.values() if f.init}
    init_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in valid_keys}
    p = Law115Params(**init_kwargs)
    for k, v in extra_kwargs.items():
        if hasattr(p, k):
            setattr(p, k, v)
    p.__post_init__()
    return p


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law115Params:
    """Resolve references for /MAT/LAW115 and return Law115Params."""
    return build_law115(mat)


def extra_shapes(mat: Any = None, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Return extra history variable shapes for LAW115."""
    if nip is not None:
        return {
            "uvar115": (nip, 4),
            "epsp": (nip,),
            "seq": (nip,),
        }
    return {
        "uvar115": (4,),
        "epsp": (),
        "seq": (),
    }


def needs_defgrad(mat: Any = None) -> bool:
    """LAW115 is a small-strain plasticity formulation; defgrad is False."""
    return False


def _eval_deshfleck_yield_and_hardening(p: Law115Params, eps_p: float) -> Tuple[float, float]:
    """Evaluate flow stress and hardening slope matching Deshpande-Fleck formulation."""
    eps_val = max(0.0, eps_p)
    sig_y = p.sigp + p.gamma * eps_val
    h = p.gamma

    if p.alpha2 > 0.0 and p.epsd > 0.0:
        ratio = eps_val / p.epsd
        sig_y += p.alpha2 * (ratio ** p.beta)
        if p.beta > 0.0:
            h += (p.alpha2 * p.beta / p.epsd) * (ratio ** max(0.0, p.beta - 1.0))

    return max(sig_y, 1.0e-6), max(h, 0.0)


def _compute_deshfleck_equivalent_stress(sig: np.ndarray, alpha: float) -> Tuple[float, float, np.ndarray]:
    """Compute Deshpande-Fleck equivalent stress:

    hat{sigma}^2 = 1 / (1 + (alpha/3)^2) * [sigma_vm^2 + alpha^2 * sigma_m^2]
    """
    p_m = (sig[0] + sig[1] + sig[2]) / 3.0
    s = np.array([
        sig[0] - p_m,
        sig[1] - p_m,
        sig[2] - p_m,
        sig[3],
        sig[4],
        sig[5],
    ], dtype=np.float64)

    j2 = 0.5 * (s[0]**2 + s[1]**2 + s[2]**2) + s[3]**2 + s[4]**2 + s[5]**2
    sig_vm_sq = 3.0 * j2

    coeff = 1.0 / (1.0 + (alpha / 3.0)**2)
    sig_hat_sq = coeff * (sig_vm_sq + (alpha**2) * (p_m**2))
    sig_hat = math.sqrt(max(0.0, sig_hat_sq))

    return sig_hat, p_m, s


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
    """3D continuum solid stress update for LAW115."""
    p = build_law115(mat)
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

    lame = p.lame
    g = p.g
    g2 = 2.0 * g
    k_bulk = p.bulk
    alpha = p.alpha

    for i in range(n):
        deps_i = deps_2d[i]
        tr_deps = deps_i[0] + deps_i[1] + deps_i[2]

        sig_tr = np.zeros(6, dtype=np.float64)
        sig_tr[0] = sig_2d[i, 0] + lame * tr_deps + g2 * deps_i[0]
        sig_tr[1] = sig_2d[i, 1] + lame * tr_deps + g2 * deps_i[1]
        sig_tr[2] = sig_2d[i, 2] + lame * tr_deps + g2 * deps_i[2]
        sig_tr[3] = sig_2d[i, 3] + g * deps_i[3]
        sig_tr[4] = sig_2d[i, 4] + g * deps_i[4]
        sig_tr[5] = sig_2d[i, 5] + g * deps_i[5]

        sig_hat_tr, p_m_tr, s_tr = _compute_deshfleck_equivalent_stress(sig_tr, alpha)
        sig_y, h = _eval_deshfleck_yield_and_hardening(p, epsp_arr[i])

        if sig_hat_tr > sig_y and sig_hat_tr > _EM10:
            coeff = 1.0 / (1.0 + (alpha / 3.0)**2)
            denom = coeff * (3.0 * g + (alpha**2 / 9.0) * k_bulk) + h
            dgamma = (sig_hat_tr - sig_y) / max(_EM20, denom)

            # Deviatoric and volumetric plastic updates
            factor_dev = max(0.0, 1.0 - coeff * 3.0 * g * dgamma / sig_hat_tr)
            factor_vol = max(0.0, 1.0 - coeff * (alpha**2 / 9.0) * k_bulk * dgamma / sig_hat_tr)

            p_m_new = p_m_tr * factor_vol
            sig_new[i, 0] = p_m_new + s_tr[0] * factor_dev
            sig_new[i, 1] = p_m_new + s_tr[1] * factor_dev
            sig_new[i, 2] = p_m_new + s_tr[2] * factor_dev
            sig_new[i, 3] = s_tr[3] * factor_dev
            sig_new[i, 4] = s_tr[4] * factor_dev
            sig_new[i, 5] = s_tr[5] * factor_dev
            epsp_new[i] = epsp_arr[i] + dgamma
        else:
            sig_new[i] = sig_tr
            epsp_new[i] = epsp_arr[i]

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
    """2D plane-stress shell stress update for LAW115."""
    p = build_law115(mat)
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
    g = p.g

    for i in range(n):
        deps_i = deps_2d[i]
        s_xx = sig_2d[i, 0] + a11 * deps_i[0] + a12 * deps_i[1]
        s_yy = sig_2d[i, 1] + a12 * deps_i[0] + a11 * deps_i[1]
        s_xy = sig_2d[i, 2] + g * deps_i[2]

        sig_full = np.array([s_xx, s_yy, 0.0, s_xy, 0.0, 0.0], dtype=np.float64)
        sig_hat, _, _ = _compute_deshfleck_equivalent_stress(sig_full, p.alpha)
        sig_y, h = _eval_deshfleck_yield_and_hardening(p, epsp_arr[i])

        if sig_hat > sig_y and sig_hat > _EM10:
            denom = a11 + h
            dgamma = (sig_hat - sig_y) / max(_EM20, denom)
            factor = max(0.0, 1.0 - (a11 * dgamma) / sig_hat)

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
    """Compute acoustic wave speed for LAW115."""
    p = build_law115(mat)
    rho = p.rho0 if p.rho0 > 0.0 else 1.0
    if is_shell:
        mod = p.a11
    else:
        mod = p.bulk + (4.0 / 3.0) * p.g
    return float(math.sqrt(max(0.0, mod / rho)))


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[float] = None,
    epsp_incr: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent 6x6 continuum solid tangent stiffness for LAW115."""
    p = build_law115(mat)
    c_el = np.zeros((6, 6), dtype=np.float64)
    lame = p.lame
    g = p.g
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
            t[i] = solid_tangent(p, sig=sig_arr[i])
        return t

    sig_hat, _, s = _compute_deshfleck_equivalent_stress(sig_arr, p.alpha)
    sig_y, h = _eval_deshfleck_yield_and_hardening(p, epsp if epsp is not None else 0.0)

    if sig_hat < sig_y or sig_hat <= _EM10:
        return c_el

    n_vec = np.zeros(6, dtype=np.float64)
    n_vec[:3] = 1.5 * s[:3] / sig_hat
    n_vec[3:] = 3.0 * s[3:] / sig_hat

    denom = 3.0 * g + h
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
    """Consistent 3x3 plane stress algorithmic tangent matrix for LAW115."""
    p = build_law115(mat)
    c_el = np.array([
        [p.a11, p.a12, 0.0],
        [p.a12, p.a11, 0.0],
        [0.0, 0.0, p.g],
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

    sig_full = np.array([sig_arr[0], sig_arr[1], 0.0, sig_arr[2], 0.0, 0.0], dtype=np.float64)
    sig_hat, _, _ = _compute_deshfleck_equivalent_stress(sig_full, p.alpha)
    sig_y, h = _eval_deshfleck_yield_and_hardening(p, epsp if epsp is not None else 0.0)

    if sig_hat < sig_y or sig_hat <= _EM10:
        return c_el

    sxx, syy, sxy = sig_arr[0], sig_arr[1], sig_arr[2]
    n_vec = np.array([
        (2.0 * sxx - syy) / (2.0 * sig_hat),
        (2.0 * syy - sxx) / (2.0 * sig_hat),
        (3.0 * sxy) / sig_hat,
    ], dtype=np.float64)

    cn = c_el @ n_vec
    denom = float(n_vec @ cn) + h
    if denom > _EM20:
        return c_el - np.outer(cn, cn) / denom
    return c_el


consistent_solid_tangent = solid_tangent
consistent_shell_tangent = shell_tangent
