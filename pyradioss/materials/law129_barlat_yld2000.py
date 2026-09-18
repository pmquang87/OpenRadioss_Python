"""
LAW129 (/MAT/LAW129, /MAT/BARLAT_YLD2000) — Barlat Yld2000-2d Anisotropic Plasticity with Asymmetric Tension-Compression.

OpenRadioss Fortran reference:
- engine/source/materials/mat/mat129/sigeps129s.F90
- starter/source/materials/mat/mat129/hm_read_mat129.F90
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20
_EM14 = 1.0e-14
_THIRD = 1.0 / 3.0
_TWO_THIRDS = 2.0 / 3.0
_FOUR_THIRDS = 4.0 / 3.0


@dataclass
class Law129Params:
    """Parameters for /MAT/LAW129 (Barlat Yld2000-2d with asymmetric tension-compression)."""
    E: float = 7.0e10            # Young's modulus (aluminum baseline)
    nu: float = 0.33            # Poisson's ratio
    rho0: float = 2700.0        # Mass density
    sig0: float = 2.0e8         # Initial reference yield stress
    m_exp: float = 8.0          # Barlat exponent (8 for FCC aluminum, 6 for BCC steel)
    # Barlat Yld2000 8 anisotropy parameters (1.0 for isotropic)
    alpha1: float = 1.0
    alpha2: float = 1.0
    alpha3: float = 1.0
    alpha4: float = 1.0
    alpha5: float = 1.0
    alpha6: float = 1.0
    alpha7: float = 1.0
    alpha8: float = 1.0
    # Tension-compression asymmetry parameter (k_asym = 0 for symmetric)
    k_asym: float = 0.0
    # Hardening parameters (Swift / Voce / Chaboche)
    hp: float = 0.0             # Linear hardening modulus
    qr1: float = 0.0            # Voce / Chaboche saturation QR1
    cr1: float = 0.0            # Voce / Chaboche rate CR1
    qr2: float = 0.0            # Saturation QR2
    cr2: float = 0.0            # Rate CR2

    # Derived quantities
    G: float = field(init=False, default=0.0)
    K: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.nu < 0.0 or self.nu >= 0.5:
            self.nu = 0.33
        if self.E <= 0.0:
            self.E = 7.0e10
        if self.m_exp <= 1.0:
            self.m_exp = 8.0
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))


def resolve(mat: Any) -> Law129Params:
    """Resolve Law129Params from Material, dict, or Law129Params instance."""
    if isinstance(mat, Law129Params):
        return mat
    if isinstance(mat, Material):
        p = mat.params or {}
        return Law129Params(
            E=float(p.get("E", p.get("MAT_E", 7.0e10))),
            nu=float(p.get("nu", p.get("MAT_NU", 0.33))),
            rho0=float(getattr(mat, "rho0", p.get("rho", p.get("MAT_RHO", 2700.0)))),
            sig0=float(p.get("sig0", p.get("SIG0", p.get("MAT_SIGY", 2.0e8)))),
            m_exp=float(p.get("m_exp", p.get("M_EXP", p.get("M", 8.0)))),
            alpha1=float(p.get("alpha1", p.get("ALPHA1", 1.0))),
            alpha2=float(p.get("alpha2", p.get("ALPHA2", 1.0))),
            alpha3=float(p.get("alpha3", p.get("ALPHA3", 1.0))),
            alpha4=float(p.get("alpha4", p.get("ALPHA4", 1.0))),
            alpha5=float(p.get("alpha5", p.get("ALPHA5", 1.0))),
            alpha6=float(p.get("alpha6", p.get("ALPHA6", 1.0))),
            alpha7=float(p.get("alpha7", p.get("ALPHA7", 1.0))),
            alpha8=float(p.get("alpha8", p.get("ALPHA8", 1.0))),
            k_asym=float(p.get("k_asym", p.get("K_ASYM", 0.0))),
            hp=float(p.get("hp", p.get("HP", 0.0))),
            qr1=float(p.get("qr1", p.get("MAT_QR1", 0.0))),
            cr1=float(p.get("cr1", p.get("MAT_CR1", 0.0))),
            qr2=float(p.get("qr2", p.get("MAT_QR2", 0.0))),
            cr2=float(p.get("cr2", p.get("MAT_CR2", 0.0))),
        )
    if isinstance(mat, dict):
        return Law129Params(
            E=float(mat.get("E", mat.get("MAT_E", 7.0e10))),
            nu=float(mat.get("nu", mat.get("MAT_NU", 0.33))),
            rho0=float(mat.get("rho0", mat.get("rho", mat.get("MAT_RHO", 2700.0)))),
            sig0=float(mat.get("sig0", mat.get("SIG0", mat.get("MAT_SIGY", 2.0e8)))),
            m_exp=float(mat.get("m_exp", mat.get("M_EXP", mat.get("M", 8.0)))),
            alpha1=float(mat.get("alpha1", mat.get("ALPHA1", 1.0))),
            alpha2=float(mat.get("alpha2", mat.get("ALPHA2", 1.0))),
            alpha3=float(mat.get("alpha3", mat.get("ALPHA3", 1.0))),
            alpha4=float(mat.get("alpha4", mat.get("ALPHA4", 1.0))),
            alpha5=float(mat.get("alpha5", mat.get("ALPHA5", 1.0))),
            alpha6=float(mat.get("alpha6", mat.get("ALPHA6", 1.0))),
            alpha7=float(mat.get("alpha7", mat.get("ALPHA7", 1.0))),
            alpha8=float(mat.get("alpha8", mat.get("ALPHA8", 1.0))),
            k_asym=float(mat.get("k_asym", mat.get("K_ASYM", 0.0))),
            hp=float(mat.get("hp", mat.get("HP", 0.0))),
            qr1=float(mat.get("qr1", mat.get("MAT_QR1", 0.0))),
            cr1=float(mat.get("cr1", mat.get("MAT_CR1", 0.0))),
            qr2=float(mat.get("qr2", mat.get("MAT_QR2", 0.0))),
            cr2=float(mat.get("cr2", mat.get("MAT_CR2", 0.0))),
        )
    return Law129Params()


def build_law129(mat: Any) -> Law129Params:
    """Build Law129Params from Material entity or dict."""
    return resolve(mat)


def needs_defgrad(mat: Any = None) -> bool:
    """LAW129 uses incremental strain rate formulation and does not need F."""
    return False


def extra_shapes(mat: Any = None, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """Persistent history variables for LAW129 (plastic strain, backstress)."""
    return {
        "uvar": (nip, 16) if nip > 1 else (16,),
    }


def sound_speed(
    mat: Any,
    eps: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    rho: Optional[float] = None,
    is_shell: bool = False,
    **kwargs: Any,
) -> float:
    """Dilatational sound speed for LAW129."""
    p = resolve(mat)
    eff_rho = float(rho) if (rho is not None and rho > 0.0) else p.rho0
    if eff_rho <= 0.0:
        eff_rho = 2700.0
    if is_shell:
        return math.sqrt(max(0.0, p.E / (eff_rho * max(1.0 - p.nu ** 2, 1e-6))))
    return math.sqrt(max(0.0, (p.K + _FOUR_THIRDS * p.G) / eff_rho))


def _barlat_yld2000_eq_stress(p: Law129Params, sxx: float, syy: float, sxy: float) -> float:
    """Compute Barlat Yld2000-2d equivalent stress for plane stress."""
    # Linear transformation L'
    xp_xx = _TWO_THIRDS * p.alpha1 * sxx - _THIRD * p.alpha1 * syy
    xp_yy = -_THIRD * p.alpha2 * sxx + _TWO_THIRDS * p.alpha2 * syy
    xp_xy = p.alpha7 * sxy

    # Principal values of X'
    disc_p = math.sqrt(max(0.0, ((xp_xx - xp_yy) * 0.5) ** 2 + xp_xy ** 2))
    xp_1 = (xp_xx + xp_yy) * 0.5 + disc_p
    xp_2 = (xp_xx + xp_yy) * 0.5 - disc_p

    # Linear transformation L''
    xpp_xx = (1.0 / 9.0) * (
        (-2.0 * p.alpha3 + 2.0 * p.alpha4 + 8.0 * p.alpha5 - 2.0 * p.alpha6) * sxx
        + (p.alpha3 - 4.0 * p.alpha4 - 4.0 * p.alpha5 + 4.0 * p.alpha6) * syy
    )
    xpp_yy = (1.0 / 9.0) * (
        (4.0 * p.alpha3 - 4.0 * p.alpha4 - 4.0 * p.alpha5 + p.alpha6) * sxx
        + (-2.0 * p.alpha3 + 8.0 * p.alpha4 + 2.0 * p.alpha5 - 2.0 * p.alpha6) * syy
    )
    xpp_xy = p.alpha8 * sxy

    # Principal values of X''
    disc_pp = math.sqrt(max(0.0, ((xpp_xx - xpp_yy) * 0.5) ** 2 + xpp_xy ** 2))
    xpp_1 = (xpp_xx + xpp_yy) * 0.5 + disc_pp
    xpp_2 = (xpp_xx + xpp_yy) * 0.5 - disc_pp

    m = p.m_exp
    phi = abs(xp_1 - xp_2) ** m + abs(2.0 * xpp_2 + xpp_1) ** m + abs(2.0 * xpp_1 + xpp_2) ** m
    sig_barlat = (0.5 * phi) ** (1.0 / m)

    # Asymmetry correction (tension vs compression)
    if p.k_asym != 0.0:
        i1 = sxx + syy
        sig_barlat += p.k_asym * i1

    return max(0.0, sig_barlat)


def _current_yield(p: Law129Params, epsp: float) -> float:
    """Calculate isotropic hardening stress."""
    y = p.sig0 + p.hp * epsp
    if p.qr1 != 0.0 and p.cr1 != 0.0:
        y += p.qr1 * (1.0 - math.exp(-p.cr1 * epsp))
    if p.qr2 != 0.0 and p.cr2 != 0.0:
        y += p.qr2 * (1.0 - math.exp(-p.cr2 * epsp))
    return max(0.0, y)


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Plane-stress shell update for /MAT/LAW129."""
    p = resolve(mat)
    sig_arr = np.asarray(sig, dtype=np.float64)
    deps_arr = np.asarray(deps, dtype=np.float64)
    is_1d = (sig_arr.ndim == 1)

    s = np.atleast_2d(sig_arr).copy()
    d = np.atleast_2d(deps_arr).copy()
    n = s.shape[0]

    if epsp is None:
        ep = np.zeros(n, dtype=np.float64)
    elif np.isscalar(epsp):
        ep = np.full(n, float(epsp), dtype=np.float64)
    else:
        ep = np.asarray(epsp, dtype=np.float64).copy()

    denom = max(1.0 - p.nu ** 2, 1e-8)
    q11 = p.E / denom
    q12 = p.nu * q11
    q33 = p.G

    s_out = np.zeros_like(s)
    ep_out = np.zeros(n, dtype=np.float64)
    c_shell = math.sqrt(max(0.0, p.E / (p.rho0 * denom)))
    c_out = np.full(n, c_shell, dtype=np.float64)

    for i in range(n):
        s_trial = np.copy(s[i])
        s_trial[0] += q11 * d[i, 0] + q12 * d[i, 1]
        s_trial[1] += q12 * d[i, 0] + q11 * d[i, 1]
        if s_trial.shape[0] > 2 and d.shape[1] > 2:
            s_trial[2] += q33 * d[i, 2]

        sxy = s_trial[2] if s_trial.shape[0] > 2 else 0.0
        sig_eq = _barlat_yld2000_eq_stress(p, s_trial[0], s_trial[1], sxy)
        yld = _current_yield(p, ep[i])

        dpla = 0.0
        if sig_eq > yld:
            ratio = yld / max(sig_eq, _EM14)
            s_trial[:3] *= ratio
            dpla = (1.0 - ratio) * sig_eq / max(3.0 * p.G, _EM20)

        s_out[i] = s_trial
        ep_out[i] = ep[i] + dpla

    if is_1d:
        res_sig = s_out[0]
        res_ep = float(ep_out[0])
        res_c = float(c_out[0])
    else:
        res_sig = s_out
        res_ep = ep_out
        res_c = c_out

    if return_sound_speed:
        return res_sig, res_ep, res_c
    return res_sig, res_ep


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Solid update for /MAT/LAW129."""
    p = resolve(mat)
    sig_arr = np.asarray(sig, dtype=np.float64)
    deps_arr = np.asarray(deps, dtype=np.float64)
    is_1d = (sig_arr.ndim == 1)

    s = np.atleast_2d(sig_arr).copy()
    d = np.atleast_2d(deps_arr).copy()
    n = s.shape[0]

    if epsp is None:
        ep = np.zeros(n, dtype=np.float64)
    elif np.isscalar(epsp):
        ep = np.full(n, float(epsp), dtype=np.float64)
    else:
        ep = np.asarray(epsp, dtype=np.float64).copy()

    lam = p.K - _TWO_THIRDS * p.G
    g2 = 2.0 * p.G

    s_out = np.zeros_like(s)
    ep_out = np.zeros(n, dtype=np.float64)
    soundsp = math.sqrt(max(0.0, (p.K + _FOUR_THIRDS * p.G) / p.rho0))
    c_out = np.full(n, soundsp, dtype=np.float64)

    for i in range(n):
        de_vol = d[i, 0] + d[i, 1] + d[i, 2]
        s_trial = np.copy(s[i])
        s_trial[0] += lam * de_vol + g2 * d[i, 0]
        s_trial[1] += lam * de_vol + g2 * d[i, 1]
        s_trial[2] += lam * de_vol + g2 * d[i, 2]
        s_trial[3] += p.G * d[i, 3]
        s_trial[4] += p.G * d[i, 4]
        s_trial[5] += p.G * d[i, 5]

        # Deviatoric stress
        p_mean = _THIRD * (s_trial[0] + s_trial[1] + s_trial[2])
        dev_xx = s_trial[0] - p_mean
        dev_yy = s_trial[1] - p_mean

        sig_eq = _barlat_yld2000_eq_stress(p, dev_xx, dev_yy, s_trial[3])
        yld = _current_yield(p, ep[i])

        dpla = 0.0
        if sig_eq > yld:
            ratio = yld / max(sig_eq, _EM14)
            s_trial[0] = p_mean + ratio * dev_xx
            s_trial[1] = p_mean + ratio * dev_yy
            s_trial[2] = p_mean + ratio * (s_trial[2] - p_mean)
            s_trial[3] *= ratio
            s_trial[4] *= ratio
            s_trial[5] *= ratio
            dpla = (1.0 - ratio) * sig_eq / max(3.0 * p.G, _EM20)

        s_out[i] = s_trial
        ep_out[i] = ep[i] + dpla

    if is_1d:
        res_sig = s_out[0]
        res_ep = float(ep_out[0])
        res_c = float(c_out[0])
    else:
        res_sig = s_out
        res_ep = ep_out
        res_c = c_out

    if return_sound_speed:
        return res_sig, res_ep, res_c
    return res_sig, res_ep


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Elastic 6x6 tangent matrix for LAW129."""
    p = resolve(mat)
    k = p.K
    g = p.G
    c11 = k + _FOUR_THIRDS * g
    c12 = k - _TWO_THIRDS * g

    c = np.zeros((6, 6), dtype=np.float64)
    c[0, 0] = c[1, 1] = c[2, 2] = c11
    c[0, 1] = c[1, 0] = c[0, 2] = c[2, 0] = c[1, 2] = c[2, 1] = c12
    c[3, 3] = c[4, 4] = c[5, 5] = g

    if sig is not None and np.ndim(sig) > 1:
        n = np.shape(sig)[0]
        return np.broadcast_to(c, (n, 6, 6)).copy()
    return c


consistent_solid_tangent = solid_tangent


def shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Return 3x3 plane-stress membrane elastic tangent matrix."""
    p = resolve(mat)
    denom = max(1.0 - p.nu ** 2, 1e-8)
    q11 = p.E / denom
    q12 = p.nu * q11
    q33 = p.G

    c = np.array([
        [q11, q12, 0.0],
        [q12, q11, 0.0],
        [0.0, 0.0, q33],
    ], dtype=np.float64)

    if sig is not None and np.ndim(sig) > 1:
        n = np.shape(sig)[0]
        return np.broadcast_to(c, (n, 3, 3)).copy()
    return c


consistent_shell_tangent = shell_tangent
