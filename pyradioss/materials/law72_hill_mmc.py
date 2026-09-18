"""LAW72 — Hill Anisotropic Yield with Modified Mohr-Coulomb (MMC) Ductile Fracture.

Upstream OpenRadioss Fortran reference:
- Starter Card Reader:
  `starter/source/materials/mat/mat072/hm_read_mat72.F`
- 3D Solids Constitutive Update:
  `engine/source/materials/mat/mat072/sigeps72.F`
- 2D Shells (Plane Stress) Constitutive Update:
  `engine/source/materials/mat/mat072/sigeps72c.F`
- HyperMesh CFG Schema:
  `hm_cfg_files/config/CFG/radioss140/MAT/matl72_72.cfg`

Theory:
-------
1. Hill 1948 Anisotropic Yield Criterion:
   3D:
     F_hill = F*(sigma_yy - sigma_zz)^2 + G*(sigma_zz - sigma_xx)^2 + H*(sigma_xx - sigma_yy)^2
            + 2*L*sigma_yz^2 + 2*M*sigma_zx^2 + 2*N*sigma_xy^2
     sigma_hill = sqrt(max(0, F_hill))
   2D Plane Stress (sigma_zz = sigma_yz = sigma_zx = 0):
     F_hill = (F + H)*sigma_yy^2 + (G + H)*sigma_xx^2 - 2*H*sigma_xx*sigma_yy + 2*N*sigma_xy^2
     sigma_hill = sqrt(max(0, F_hill))

2. Hardening Law:
   sigma_yld = beta * SIGY * (eps_p + EPS0)^NEXP
   where EPS0 > 0, NEXP is hardening exponent.

3. Modified Mohr-Coulomb (MMC) Ductile Fracture Criterion (Bai-Wierzbicki 2008):
   Hydrostatic pressure: P = 1/3 * (sigma_xx + sigma_yy + sigma_zz)
   Deviatoric stress: S_ij = sigma_ij - P * delta_ij
   von Mises stress: sigma_vm = sqrt(3/2 * S_ij * S_ij)
   Stress triaxiality: eta = P / max(sigma_vm, 1e-20), clamped to [-1, 1] (or [-2/3, 2/3] in shells)
   Lode angle parameter:
     cos(3*theta) = (27/2) * det(S) / max(sigma_vm^3, 1e-20)
     theta_bar = 1 - (2 / pi) * arccos(cos(3*theta))
   Failure functions:
     f1 = cos(theta_bar * pi / 6)
     f2 = sin(theta_bar * pi / 6)
     f3 = c3 + (sqrt(3)/(2 - sqrt(3))) * (1 - c3) * (1/max(f1, 1e-20) - 1)
   Equivalent plastic strain at fracture:
     eps_f = [ (SIGY / c2) * f3 * (f1 * sqrt((1 + c1^2)/3) + c1 * (eta + f2/3)) ]^(-1/NEXP)
   Damage accumulation:
     dD = deps_p / max(eps_f, 1e-20)
     D = D + dD
   Damage softening beta factor:
     If 1.0 <= D <= Dc:
       beta = ((Dc - D) / (Dc - 1.0))^MEXP
     If D >= Dc:
       element deleted / ruptured (OFF = 0.8 -> 0).
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
class Law72Params:
    """Parameters for OpenRadioss /MAT/LAW72 (Hill + MMC ductile fracture)."""
    id: int = 1
    title: str = ""
    law: int = 72
    law_name: str = "LAW72"

    # Density
    rho0: float = 1.0
    rhor: float = 1.0

    # Isotropic Elasticity
    young: float = 210000.0
    nu: float = 0.3

    # Hardening
    sigy: float = 300.0
    eps0: float = 0.001
    nexp: float = 0.2

    # Hill 1948 Anisotropy Coefficients
    f_hill: float = 0.5
    g_hill: float = 0.5
    h_hill: float = 0.5
    n_hill: float = 1.5
    l_hill: float = 1.5
    m_hill: float = 1.5

    # Modified Mohr-Coulomb (MMC) Failure Parameters
    c1: float = 0.05
    c2: float = 300.0
    c3: float = 1.0
    mexp: float = 1.0
    dc: float = 1.0

    # Derived constants
    g: float = field(init=False)
    bulk: float = field(init=False)
    lamhook: float = field(init=False)
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
        if self.eps0 <= 0.0:
            self.eps0 = _EM20
        if self.nexp == 0.0:
            self.nexp = 1.0
        if self.mexp == 0.0:
            self.mexp = 1.0
        if self.c2 == 0.0:
            self.c2 = self.sigy if self.sigy > 0.0 else 300.0
        if self.dc <= 0.0:
            self.dc = 1.0

        e = self.young
        nu = self.nu
        denom = (1.0 + nu) * (1.0 - 2.0 * nu)
        self.g = 0.5 * e / max(1.0 + nu, _EM20)
        self.bulk = e / max(3.0 * (1.0 - 2.0 * nu), _EM20)
        self.a1 = (e * (1.0 - nu) / denom) if abs(denom) > _EM20 else (e + 4.0 / 3.0 * self.g)
        self.a2 = (self.a1 * nu / max(1.0 - nu, _EM20))
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


def build_law72(mat_def: Any = None, **kwargs: Any) -> Law72Params:
    """Construct Law72Params from a Material entity, dictionary, or keyword arguments."""
    if isinstance(mat_def, Law72Params):
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
    title = str(data.get("title", f"LAW72_{mat_id}"))

    rho0 = _extract_val(data, ["MAT_RHO", "rho0", "rho", "density"], 1.0)
    rhor = _extract_val(data, ["Refer_Rho", "rhor", "ref_rho"], rho0)

    young = _extract_val(data, ["MAT_E", "young", "e", "E"], 210000.0)
    nu = _extract_val(data, ["MAT_NU", "nu", "poisson"], 0.3)

    sigy = _extract_val(data, ["SIGMA_r", "MAT_SIGY", "sigy", "sig0", "yield_stress"], 300.0)
    eps0 = _extract_val(data, ["Epsilon_0", "eps0", "EPS0", "MAT_EPS0"], 0.001)
    nexp = _extract_val(data, ["MAT_n1_t", "MAT_NEXP", "nexp", "n", "EXP"], 0.2)

    ff = _extract_val(data, ["MAT_F", "f_hill", "ff", "F"], 0.5)
    gg = _extract_val(data, ["MAT_G0", "MAT_G", "g_hill", "gg", "G_hill"], 0.5)
    hh = _extract_val(data, ["MAT_HARD", "MAT_H", "h_hill", "hh", "H_hill"], 0.5)
    nn = _extract_val(data, ["MAT_N", "n_hill", "nn", "N_hill"], 1.5)
    ll = _extract_val(data, ["MAT_Lamda", "MAT_L", "l_hill", "ll", "L_hill"], 1.5)
    mm = _extract_val(data, ["MAT_M", "m_hill", "mm", "M_hill"], 1.5)

    c1 = _extract_val(data, ["MAT_C1", "c1", "C1"], 0.05)
    c2 = _extract_val(data, ["MAT_C2", "c2", "C2"], sigy)
    c3 = _extract_val(data, ["MAT_C3", "c3", "C3"], 1.0)
    mexp = _extract_val(data, ["MAT_MUE1", "mexp", "MEXP", "m_damage"], 1.0)
    dc = _extract_val(data, ["MAT_Dc", "dc", "Dc", "critical_damage"], 1.0)

    return Law72Params(
        id=mat_id,
        title=title,
        rho0=rho0,
        rhor=rhor,
        young=young,
        nu=nu,
        sigy=sigy,
        eps0=eps0,
        nexp=nexp,
        f_hill=ff,
        g_hill=gg,
        h_hill=hh,
        n_hill=nn,
        l_hill=ll,
        m_hill=mm,
        c1=c1,
        c2=c2,
        c3=c3,
        mexp=mexp,
        dc=dc,
    )


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law72Params:
    """Resolve and cache Law72Params from a Material or dict."""
    if isinstance(mat, Law72Params):
        return mat
    cached = getattr(mat, "_cached_law72", None)
    if cached is None:
        cached = build_law72(mat)
        try:
            setattr(mat, "_cached_law72", cached)
        except Exception:
            pass
    return cached


def needs_defgrad(mat: Any = None) -> bool:
    """Return False: LAW72 is a hypoelastic-plastic rate formulation."""
    return False


def extra_shapes(mat: Any = None, nip: Optional[int] = 1) -> Dict[str, Tuple[int, ...]]:
    """Persistent history variables for LAW72 (col 0: damage D)."""
    if nip is not None and nip > 1:
        return {"uvar72": (nip, 1), "uvar": (nip, 1)}
    return {"uvar72": (1,), "uvar": (1,)}


def sound_speed(
    mat: Any,
    eps: Any = None,
    extra: Any = None,
    is_shell: bool = False,
) -> float | np.ndarray:
    """Acoustic sound speed for LAW72."""
    p = resolve(mat)
    c_val = p.c_shell if is_shell else p.c_solid
    if eps is not None and isinstance(eps, np.ndarray) and eps.ndim > 1:
        return np.full(len(eps), c_val, dtype=float)
    return c_val


def _solid_update_single(
    p: Law72Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    epsp0: float,
    dam0: float,
    off: float = 1.0,
) -> Tuple[np.ndarray, float, float, float]:
    """Single 3D solid continuum update matching sigeps72.F."""
    if off < 0.1:
        return np.zeros(6, dtype=float), epsp0, dam0, 0.0

    g2 = 2.0 * p.g
    dav = (deps[0] + deps[1] + deps[2]) * p.lamhook

    # Trial stress
    sign = np.empty(6, dtype=float)
    sign[0] = sig0[0] + deps[0] * g2 + dav
    sign[1] = sig0[1] + deps[1] * g2 + dav
    sign[2] = sig0[2] + deps[2] * g2 + dav
    sign[3] = sig0[3] + deps[3] * p.g
    sign[4] = sig0[4] + deps[4] * p.g
    sign[5] = sig0[5] + deps[5] * p.g

    dam = dam0
    beta = 1.0
    if 1.0 <= dam <= p.dc:
        beta = max((p.dc - dam) / max(p.dc - 1.0, _EM20), 0.0) ** p.mexp

    cc = epsp0 + p.eps0
    yld = beta * p.sigy * (cc ** p.nexp)
    yld = max(yld, _EM10)

    # Hill equivalent stress
    f_hill = (
        p.f_hill * (sign[1] - sign[2]) ** 2
        + p.g_hill * (sign[2] - sign[0]) ** 2
        + p.h_hill * (sign[0] - sign[1]) ** 2
        + 2.0 * p.l_hill * (sign[4] ** 2)
        + 2.0 * p.m_hill * (sign[5] ** 2)
        + 2.0 * p.n_hill * (sign[3] ** 2)
    )
    sighl = math.sqrt(max(0.0, f_hill))
    phi = sighl - yld

    pla = epsp0
    dpla = 0.0

    # Cutting plane iterations if yielding
    if phi > 0.0 and sighl > _EM20:
        for _ in range(5):
            inv_sighl = 1.0 / max(sighl, _EM20)
            normxx = (p.g_hill * (sign[0] - sign[2]) + p.h_hill * (sign[0] - sign[1])) * inv_sighl
            normyy = (p.f_hill * (sign[1] - sign[2]) + p.h_hill * (sign[1] - sign[0])) * inv_sighl
            normzz = (p.f_hill * (sign[2] - sign[1]) + p.g_hill * (sign[2] - sign[0])) * inv_sighl
            normxy = 2.0 * p.n_hill * sign[3] * inv_sighl
            normyz = 2.0 * p.l_hill * sign[4] * inv_sighl
            normzx = 2.0 * p.m_hill * sign[5] * inv_sighl

            dfdsig2 = (
                (normxx ** 2 + normyy ** 2 + normzz ** 2) * g2
                + (normxy ** 2 + normyz ** 2 + normzx ** 2) * p.g
            )

            h_slope = beta * p.nexp * p.sigy * (cc ** (p.nexp - 1.0))
            sig_dfdsig = (
                sign[0] * normxx + sign[1] * normyy + sign[2] * normzz
                + sign[3] * normxy + sign[4] * normyz + sign[5] * normzx
            )
            dpla_dlam = sig_dfdsig / max(yld, _EM20)
            dphi_dlam = -dfdsig2 - h_slope * dpla_dlam
            if abs(dphi_dlam) < _EM20:
                break

            dlam = -phi / dphi_dlam
            if dlam <= 0.0:
                break

            sign[0] -= dlam * normxx * g2
            sign[1] -= dlam * normyy * g2
            sign[2] -= dlam * normzz * g2
            sign[3] -= dlam * normxy * p.g
            sign[4] -= dlam * normyz * p.g
            sign[5] -= dlam * normzx * p.g

            ddep = dlam * dpla_dlam
            dpla += max(0.0, ddep)
            pla += max(0.0, ddep)

            cc = pla + p.eps0
            yld = max(beta * p.sigy * (cc ** p.nexp), _EM10)

            f_hill = (
                p.f_hill * (sign[1] - sign[2]) ** 2
                + p.g_hill * (sign[2] - sign[0]) ** 2
                + p.h_hill * (sign[0] - sign[1]) ** 2
                + 2.0 * p.l_hill * (sign[4] ** 2)
                + 2.0 * p.m_hill * (sign[5] ** 2)
                + 2.0 * p.n_hill * (sign[3] ** 2)
            )
            sighl = math.sqrt(max(0.0, f_hill))
            phi = sighl - yld
            if abs(phi) < 1.0e-5 * yld:
                break

    # MMC Failure Evaluation
    pres = (sign[0] + sign[1] + sign[2]) / 3.0
    sd11 = sign[0] - pres
    sd22 = sign[1] - pres
    sd33 = sign[2] - pres
    svm = math.sqrt(max(0.0, 1.5 * (sd11 ** 2 + sd22 ** 2 + sd33 ** 2) + 3.0 * (sign[3] ** 2 + sign[4] ** 2 + sign[5] ** 2)))

    det_s = (
        sd11 * sd22 * sd33 + 2.0 * sign[3] * sign[4] * sign[5]
        - sd11 * (sign[4] ** 2) - sd33 * (sign[3] ** 2) - sd22 * (sign[5] ** 2)
    )

    eta = np.clip(pres / max(svm, _EM20), -1.0, 1.0)
    cos3theta = np.clip(13.5 * det_s / max(svm ** 3, _EM20), -1.0, 1.0)
    theta = 1.0 - (2.0 / math.pi) * math.acos(cos3theta)

    f1 = math.cos(theta * math.pi / 6.0)
    f2 = math.sin(theta * math.pi / 6.0)
    f3 = p.c3 + (math.sqrt(3.0) / (2.0 - math.sqrt(3.0))) * (1.0 - p.c3) * (1.0 / max(f1, _EM20) - 1.0)

    term_epsf = (p.sigy / max(p.c2, _EM20)) * f3 * (f1 * math.sqrt((1.0 + p.c1 ** 2) / 3.0) + p.c1 * (eta + f2 / 3.0))
    if term_epsf > 0.0:
        epsf = max(term_epsf, _EM20) ** (-1.0 / p.nexp)
        dam += dpla / max(epsf, _EM20)

    if dam >= p.dc:
        dam = p.dc
        sign[:] = 0.0

    return sign, pla, dam, p.c_solid


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
    """3D solid continuum constitutive update for /MAT/LAW72."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(6, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(float)
    nel = len(sig_arr)

    epsp_arr = np.zeros(nel, dtype=float)
    if epsp is not None:
        ep_in = np.atleast_1d(epsp).astype(float)
        epsp_arr[:min(nel, len(ep_in))] = ep_in[:nel]

    dam_arr = np.zeros(nel, dtype=float)
    if extra is not None and isinstance(extra, dict):
        for k in ("uvar72", "uvar", "damage", "dam"):
            if k in extra and extra[k] is not None:
                u = np.atleast_1d(extra[k]).astype(float)
                dam_arr[:min(nel, len(u))] = u[:nel]
                break

    off_arr = np.ones(nel, dtype=float)
    if extra is not None and isinstance(extra, dict) and "off" in extra:
        o = np.atleast_1d(extra["off"]).astype(float)
        off_arr[:min(nel, len(o))] = o[:nel]

    sig_out = np.zeros_like(sig_arr)
    epsp_out = np.zeros(nel, dtype=float)
    dam_out = np.zeros(nel, dtype=float)
    c_out = np.zeros(nel, dtype=float)

    for i in range(nel):
        s_i, ep_i, d_i, c_i = _solid_update_single(
            p, sig_arr[i], deps_arr[i], epsp_arr[i], dam_arr[i], off=off_arr[i]
        )
        sig_out[i] = s_i
        epsp_out[i] = ep_i
        dam_out[i] = d_i
        c_out[i] = c_i

    if extra is not None and isinstance(extra, dict):
        extra["uvar72"] = dam_out
        extra["uvar"] = dam_out
        extra["damage"] = dam_out

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
    p: Law72Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    epsp0: float,
    dam0: float,
    off: float = 1.0,
) -> Tuple[np.ndarray, float, float, float]:
    """Single 2D plane-stress shell update matching sigeps72c.F."""
    if off < 0.1:
        return np.zeros(len(sig0), dtype=float), epsp0, dam0, 0.0

    # Trial stresses: in-plane [xx, yy, xy]
    sign = np.empty_like(sig0, dtype=float)
    sign[0] = sig0[0] + p.a11_2d * deps[0] + p.a12_2d * deps[1]
    sign[1] = sig0[1] + p.a12_2d * deps[0] + p.a11_2d * deps[1]
    sign[2] = sig0[2] + p.g * deps[2]
    if len(sig0) >= 5:
        sign[3] = sig0[3] + p.g * deps[3]
        sign[4] = sig0[4] + p.g * deps[4]

    dam = dam0
    beta = 1.0
    if 1.0 <= dam <= p.dc:
        beta = max((p.dc - dam) / max(p.dc - 1.0, _EM20), 0.0) ** p.mexp

    cc = epsp0 + p.eps0
    yld = max(beta * p.sigy * (cc ** p.nexp), _EM10)

    # 2D Hill equivalent stress
    f_hill = (
        (p.f_hill + p.h_hill) * (sign[1] ** 2)
        + (p.g_hill + p.h_hill) * (sign[0] ** 2)
        - 2.0 * p.h_hill * sign[0] * sign[1]
        + 2.0 * p.n_hill * (sign[2] ** 2)
    )
    sighl = math.sqrt(max(0.0, f_hill))
    phi = sighl - yld

    pla = epsp0
    dpla = 0.0

    if phi > 0.0 and sighl > _EM20:
        for _ in range(5):
            inv_sighl = 1.0 / max(sighl, _EM20)
            normxx = (p.g_hill * sign[0] + p.h_hill * (sign[0] - sign[1])) * inv_sighl
            normyy = (p.f_hill * sign[1] + p.h_hill * (sign[1] - sign[0])) * inv_sighl
            normxy = 2.0 * p.n_hill * sign[2] * inv_sighl

            dfdsig2 = (
                normxx * (p.a11_2d * normxx + p.a12_2d * normyy)
                + normyy * (p.a11_2d * normyy + p.a12_2d * normxx)
                + normxy * normxy * p.g
            )
            h_slope = beta * p.nexp * p.sigy * (cc ** (p.nexp - 1.0))
            sig_dfdsig = sign[0] * normxx + sign[1] * normyy + sign[2] * normxy
            dpla_dlam = sig_dfdsig / max(yld, _EM20)

            dphi_dlam = -dfdsig2 - h_slope * dpla_dlam
            if abs(dphi_dlam) < _EM20:
                break
            dlam = -phi / dphi_dlam
            if dlam <= 0.0:
                break

            sign[0] -= dlam * (p.a11_2d * normxx + p.a12_2d * normyy)
            sign[1] -= dlam * (p.a11_2d * normyy + p.a12_2d * normxx)
            sign[2] -= dlam * normxy * p.g

            ddep = dlam * dpla_dlam
            dpla += max(0.0, ddep)
            pla += max(0.0, ddep)

            cc = pla + p.eps0
            yld = max(beta * p.sigy * (cc ** p.nexp), _EM10)

            f_hill = (
                (p.f_hill + p.h_hill) * (sign[1] ** 2)
                + (p.g_hill + p.h_hill) * (sign[0] ** 2)
                - 2.0 * p.h_hill * sign[0] * sign[1]
                + 2.0 * p.n_hill * (sign[2] ** 2)
            )
            sighl = math.sqrt(max(0.0, f_hill))
            phi = sighl - yld
            if abs(phi) < 1.0e-5 * yld:
                break

    # MMC Plane Stress Evaluation
    pres = (sign[0] + sign[1]) / 3.0
    svm = math.sqrt(max(0.0, sign[0] ** 2 + sign[1] ** 2 - sign[0] * sign[1] + 3.0 * sign[2] ** 2))
    eta = np.clip(pres / max(svm, _EM20), -2.0 / 3.0, 2.0 / 3.0)
    cos3theta = np.clip(-13.5 * eta * (eta ** 2 - 1.0 / 3.0), -1.0, 1.0)
    theta = 1.0 - (2.0 / math.pi) * math.acos(cos3theta)

    f1 = math.cos(theta * math.pi / 6.0)
    f2 = math.sin(theta * math.pi / 6.0)
    f3 = p.c3 + (math.sqrt(3.0) / (2.0 - math.sqrt(3.0))) * (1.0 - p.c3) * (1.0 / max(f1, _EM20) - 1.0)

    term_epsf = (p.sigy / max(p.c2, _EM20)) * f3 * (f1 * math.sqrt((1.0 + p.c1 ** 2) / 3.0) + p.c1 * (eta + f2 / 3.0))
    if term_epsf > 0.0:
        epsf = max(term_epsf, _EM20) ** (-1.0 / p.nexp)
        dam += dpla / max(epsf, _EM20)

    if dam >= p.dc:
        dam = p.dc
        sign[:] = 0.0

    return sign, pla, dam, p.c_shell


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
    """2D plane-stress shell constitutive update for /MAT/LAW72."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(3, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(float)
    nel = len(sig_arr)

    epsp_arr = np.zeros(nel, dtype=float)
    if epsp is not None:
        ep_in = np.atleast_1d(epsp).astype(float)
        epsp_arr[:min(nel, len(ep_in))] = ep_in[:nel]

    dam_arr = np.zeros(nel, dtype=float)
    if extra is not None and isinstance(extra, dict):
        for k in ("uvar72", "uvar", "damage", "dam"):
            if k in extra and extra[k] is not None:
                u = np.atleast_1d(extra[k]).astype(float)
                dam_arr[:min(nel, len(u))] = u[:nel]
                break

    off_arr = np.ones(nel, dtype=float)
    if extra is not None and isinstance(extra, dict) and "off" in extra:
        o = np.atleast_1d(extra["off"]).astype(float)
        off_arr[:min(nel, len(o))] = o[:nel]

    sig_out = np.zeros_like(sig_arr)
    epsp_out = np.zeros(nel, dtype=float)
    dam_out = np.zeros(nel, dtype=float)
    c_out = np.zeros(nel, dtype=float)

    for i in range(nel):
        s_i, ep_i, d_i, c_i = _shell_update_single(
            p, sig_arr[i], deps_arr[i], epsp_arr[i], dam_arr[i], off=off_arr[i]
        )
        sig_out[i] = s_i
        epsp_out[i] = ep_i
        dam_out[i] = d_i
        c_out[i] = c_i

    if extra is not None and isinstance(extra, dict):
        extra["uvar72"] = dam_out
        extra["uvar"] = dam_out
        extra["damage"] = dam_out

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
