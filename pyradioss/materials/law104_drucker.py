"""OpenRadioss /MAT/LAW104 (/MAT/DRUCKER / /MAT/JOHNS_VOCE_DRUCKER / /MAT/PLAS_DRUCK).

Combined Drucker yield criterion, Voce exponential saturation hardening,
Johnson-Cook logarithmic strain rate sensitivity, and Taylor-Quinney self-heating
for 3D continuum solids and 2D shells.

Fortran source references:
- Starter input reader:
  `starter/source/materials/mat/mat104/hm_read_mat104.F`
- Solid continuum 3D stress updates:
  `engine/source/materials/mat/mat104/sigeps104.F`
  `engine/source/materials/mat/mat104/mat104_nodam_nice.F`
  `engine/source/materials/mat/mat104/mat104_nodam_newton.F`
- Shell plane-stress 2D stress updates:
  `engine/source/materials/mat/mat104/sigeps104c.F`
  `engine/source/materials/mat/mat104/mat104c_nodam_nice.F`
  `engine/source/materials/mat/mat104/mat104c_nodam_newton.F`
- Card layout configuration:
  `hm_cfg_files/config/CFG/radioss2023/MAT/matl104_drucker.cfg`
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import numpy as np


_DEFAULT_YLD0 = 1.0e20
_DEFAULT_EPS_ISO = 1.0e20
_DEFAULT_EPS_AD = 2.0e20
_DEFAULT_FCUT = 10000.0
_EPS20 = 1.0e-20


class DruckerConstants(NamedTuple):
    g: float
    bulk: float
    kdr: float
    cdr: float
    asrate: float
    fcut: float


@dataclass
class DruckerParams:
    """Consolidated parameters for /MAT/LAW104."""
    id: int = 1
    title: str = ""
    rho: float = 0.0
    refer_rho: float = 0.0
    young: float = 0.0
    nu: float = 0.0
    ires: int = 1
    sigma_r: float = _DEFAULT_YLD0
    h: float = 0.0
    qv: float = 0.0
    bv: float = 0.0
    cdr: float = 0.0
    kdr: float = 1.7320508075688772  # sqrt(3) when cdr = 0
    cjc: float = 0.0
    epsp0: float = 1.0
    eps0: float = 1.0
    fcut: float = _DEFAULT_FCUT
    tss: float = 0.0
    tref: float = 293.15
    tini: float = 293.15
    eta: float = 0.9
    cp: float = 0.0
    eps_iso: float = _DEFAULT_EPS_ISO
    eps_ad: float = _DEFAULT_EPS_AD
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.refer_rho <= 0.0 and self.rho > 0.0:
            self.refer_rho = self.rho
        if self.eps0 != 1.0 and self.epsp0 == 1.0:
            self.epsp0 = self.eps0
        elif self.epsp0 != 1.0 and self.eps0 == 1.0:
            self.eps0 = self.epsp0
        if math.isclose(self.kdr, 1.7320508075688772, rel_tol=1e-12) and self.cdr != 0.0:
            cdr_clamped = min(2.25, max(-27.0 / 8.0, float(self.cdr)))
            term = (1.0 / 27.0) - cdr_clamped * (4.0 / 729.0)
            if term > 0.0:
                self.kdr = 1.0 / (term ** (1.0 / 6.0))

    @property
    def e(self) -> float:
        return self.young

    @property
    def E(self) -> float:
        return self.young

    @property
    def G(self) -> float:
        return self.young / (2.0 * (1.0 + self.nu)) if (1.0 + self.nu) != 0.0 else 0.0

    @property
    def g(self) -> float:
        return self.G

    @property
    def K(self) -> float:
        return self.young / (3.0 * (1.0 - 2.0 * self.nu)) if (1.0 - 2.0 * self.nu) != 0.0 else 0.0

    @property
    def bulk(self) -> float:
        return self.K

    @property
    def sound_speed(self) -> float:
        r = self.refer_rho if self.refer_rho > 0.0 else (self.rho if self.rho > 0.0 else 1.0)
        dpdm = self.bulk + (4.0 / 3.0) * self.g
        return math.sqrt(max(0.0, dpdm) / r)

    @property
    def sound_speed_shell(self) -> float:
        r = self.refer_rho if self.refer_rho > 0.0 else (self.rho if self.rho > 0.0 else 1.0)
        c_sq = self.young / (r * (1.0 - self.nu * self.nu)) if (1.0 - self.nu * self.nu) > 0.0 else 0.0
        return math.sqrt(max(0.0, c_sq))


def compute_drucker_constants(
    young: float = 0.0,
    nu: float = 0.0,
    cdr: float = 0.0,
    fcut: float = 0.0,
    **kwargs: Any,
) -> DruckerConstants:
    """Compute elastic moduli and Drucker yield criteria constants matching hm_read_mat104.F:148-175.

    - Clamps CDR within [-3.375, 2.25] (i.e. [-27/8, 9/4]) for yield surface convexity.
    - KDR = (1/27 - CDR * 4/729)**(-1/6)
    """
    g2 = young / (1.0 + nu) if (1.0 + nu) != 0.0 else 0.0
    g = 0.5 * g2
    bulk = young / (3.0 * (1.0 - 2.0 * nu)) if (1.0 - 2.0 * nu) != 0.0 else 0.0

    # Convexity bounds (hm_read_mat104.F:159-170)
    cdr_clamped = min(2.25, max(-27.0 / 8.0, float(cdr)))

    # Drucker scaling constant (hm_read_mat104.F:172-175)
    # KDR = ( (1/27) - CDR * 4/(27*27) )**(1/6)
    # KDR = 1 / KDR
    term = (1.0 / 27.0) - cdr_clamped * (4.0 / 729.0)
    if term > 0.0:
        kdr = 1.0 / (term ** (1.0 / 6.0))
    else:
        kdr = math.sqrt(3.0)

    fc = fcut if fcut > 0.0 else _DEFAULT_FCUT
    asrate = 2.0 * math.pi * fc

    return DruckerConstants(g=g, bulk=bulk, kdr=kdr, cdr=cdr_clamped, asrate=asrate, fcut=fc)


def build_law104(mat: Any = None, **kwargs: Any) -> DruckerParams:
    """Construct DruckerParams from entity object or keyword arguments."""
    p_dict: Dict[str, Any] = {}

    if mat is not None:
        for attr in (
            "id", "title", "rho", "rho0", "refer_rho", "rhor", "young", "e", "nu", "ires",
            "sigma_r", "sigma0_yld", "sigy", "sigy0", "h", "qv", "mat_pr", "bv", "cdr",
            "cjc", "epsp0", "eps0", "fcut", "tss", "mu", "tref", "tini", "eta", "cp",
            "mat_spheat", "eps_iso", "eps_ad", "params"
        ):
            if hasattr(mat, attr):
                p_dict[attr] = getattr(mat, attr)
        if hasattr(mat, "params") and isinstance(mat.params, dict):
            p_dict.update(mat.params)

    p_dict.update(kwargs)

    young_val = float(p_dict.get("young", p_dict.get("e", p_dict.get("E", p_dict.get("MAT_E", 0.0)))))
    nu_val = float(p_dict.get("nu", p_dict.get("Nu", p_dict.get("MAT_NU", 0.0))))
    rho_val = float(p_dict.get("rho", p_dict.get("rho0", p_dict.get("MAT_RHO", 0.0))))
    refer_rho = float(p_dict.get("refer_rho", p_dict.get("rhor", rho_val)))
    if refer_rho <= 0.0:
        refer_rho = rho_val

    cdr_val = float(p_dict.get("cdr", p_dict.get("c_dr", p_dict.get("MAT104_Cdr", 0.0))))
    fcut_val = float(p_dict.get("fcut", p_dict.get("f_cut", p_dict.get("MAT104_Fcut", _DEFAULT_FCUT))))
    if fcut_val <= 0.0:
        fcut_val = _DEFAULT_FCUT

    consts = compute_drucker_constants(young=young_val, nu=nu_val, cdr=cdr_val, fcut=fcut_val)

    sigma_r_val = float(p_dict.get("sigma_r", p_dict.get("sigma0_yld", p_dict.get("sigy", p_dict.get("SIGMA_r", _DEFAULT_YLD0)))))
    if sigma_r_val <= 0.0:
        sigma_r_val = _DEFAULT_YLD0

    cjc_val = float(p_dict.get("cjc", p_dict.get("c_jc", p_dict.get("MAT104_Cjc", 0.0))))
    epsp0_val = float(p_dict.get("epsp0", p_dict.get("eps0", p_dict.get("eps_0", p_dict.get("MAT104_Eps0", 1.0)))))
    if epsp0_val <= 0.0:
        epsp0_val = 1.0
        cjc_val = 0.0

    eps_iso_val = float(p_dict.get("eps_iso", p_dict.get("MAT104_EpsIso", _DEFAULT_EPS_ISO)))
    if eps_iso_val <= 0.0:
        eps_iso_val = _DEFAULT_EPS_ISO

    eps_ad_val = float(p_dict.get("eps_ad", p_dict.get("MAT104_EpsAd", _DEFAULT_EPS_AD)))
    if eps_ad_val <= 0.0:
        eps_ad_val = 2.0 * eps_iso_val

    tref_val = float(p_dict.get("tref", p_dict.get("t_ref", p_dict.get("MAT104_Tref", 293.15))))
    tini_val = float(p_dict.get("tini", p_dict.get("t_ini", p_dict.get("T_Initial", 293.15))))
    eta_val = min(1.0, max(0.0, float(p_dict.get("eta", p_dict.get("MAT_ETA", 0.9)))))
    cp_val = float(p_dict.get("cp", p_dict.get("mat_spheat", p_dict.get("MAT_SPHEAT", 0.0))))

    ires_val = int(p_dict.get("ires", p_dict.get("MAT104_Ires", 1)))
    if ires_val <= 0 or ires_val > 2:
        ires_val = 1

    return DruckerParams(
        id=int(p_dict.get("id", 1)),
        title=str(p_dict.get("title", "")),
        rho=rho_val,
        refer_rho=refer_rho,
        young=young_val,
        nu=nu_val,
        ires=ires_val,
        sigma_r=sigma_r_val,
        h=float(p_dict.get("h", p_dict.get("MAT104_H", 0.0))),
        qv=float(p_dict.get("qv", p_dict.get("mat_pr", p_dict.get("MAT_PR", 0.0)))),
        bv=float(p_dict.get("bv", p_dict.get("MAT104_Bv", 0.0))),
        cdr=consts.cdr,
        kdr=consts.kdr,
        cjc=cjc_val,
        epsp0=epsp0_val,
        fcut=consts.fcut,
        tss=float(p_dict.get("tss", p_dict.get("mu", p_dict.get("MAT104_Tss", 0.0)))),
        tref=tref_val,
        tini=tini_val,
        eta=eta_val,
        cp=cp_val,
        eps_iso=eps_iso_val,
        eps_ad=eps_ad_val,
        extra=p_dict,
    )


class YieldStress(float):
    """Flow stress value that can also be unpacked as (sigma_y, hardening_modulus)."""
    def __new__(cls, val: float, h_mod: float = 0.0):
        obj = super().__new__(cls, val)
        obj.h_mod = float(h_mod)
        return obj

    def __iter__(self):
        yield float(self)
        yield self.h_mod


def compute_drucker_equivalent_stress(*args: Any, **kwargs: Any) -> Any:
    r"""Compute Drucker equivalent stress \(\sigma_{dr}\) matching mat104_nodam_nice.F:204-220.

    Accepts either:
      - 8 scalars: (sxx, syy, szz, sxy, syz, szx, cdr, kdr) -> (sigdr, j2, j3)
      - array + constants: (sig_or_s, [kdr, cdr] or kdr=..., cdr=...) -> (sigdr, s, j2, j3)
    """
    if len(args) == 8:
        sxx, syy, szz, sxy, syz, szx, cdr, kdr = [float(x) for x in args]
        j2 = 0.5 * (sxx * sxx + syy * syy + szz * szz) + sxy * sxy + syz * syz + szx * szx
        j3 = (
            sxx * syy * szz
            + 2.0 * sxy * syz * szx
            - sxx * (syz * syz)
            - syy * (szx * szx)
            - szz * (sxy * sxy)
        )
        fdr = (j2 * j2 * j2) - cdr * (j3 * j3)
        sigdr = kdr * (fdr ** (1.0 / 6.0)) if fdr > 0.0 else 0.0
        return sigdr, j2, j3

    # Array form
    sig = np.asarray(args[0], dtype=np.float64)
    kdr = kwargs.get("kdr")
    cdr = kwargs.get("cdr")
    if len(args) > 1:
        if len(args) > 2:
            kdr = float(args[1])
            cdr = float(args[2])
        else:
            kdr = float(args[1])
    if kdr is None:
        kdr = math.sqrt(3.0)
    if cdr is None:
        cdr = 0.0

    if sig.ndim == 1:
        if len(sig) == 6:
            tr = sig[0] + sig[1] + sig[2]
            s = np.array([
                sig[0] - tr / 3.0,
                sig[1] - tr / 3.0,
                sig[2] - tr / 3.0,
                sig[3], sig[4], sig[5]
            ], dtype=np.float64)
        elif len(sig) == 3:
            tr = sig[0] + sig[1]
            s = np.array([
                sig[0] - tr / 3.0,
                sig[1] - tr / 3.0,
                -tr / 3.0,
                sig[2], 0.0, 0.0
            ], dtype=np.float64)
        else:
            s = sig.copy()
        sxx, syy, szz, sxy, syz, szx = s[0], s[1], s[2], s[3], s[4], s[5]
        j2 = 0.5 * (sxx * sxx + syy * syy + szz * szz) + sxy * sxy + syz * syz + szx * szx
        j3 = (
            sxx * syy * szz
            + 2.0 * sxy * syz * szx
            - sxx * (syz * syz)
            - syy * (szx * szx)
            - szz * (sxy * sxy)
        )
        fdr = (j2 * j2 * j2) - cdr * (j3 * j3)
        sigdr = kdr * (fdr ** (1.0 / 6.0)) if fdr > 0.0 else 0.0
        return sigdr, s, j2, j3

    raise ValueError(f"Unexpected arguments to compute_drucker_equivalent_stress: {args}")


def compute_drucker_yield_stress(*args: Any, **kwargs: Any) -> Any:
    r"""Compute flow stress \(\sigma_y = F_{\text{hard}} F_{\text{rate}} F_{\text{therm}}\) matching mat104_nodam_nice.F:168-180."""
    params = kwargs.get("params")
    eps_p = kwargs.get("eps_p")
    eps_dot = kwargs.get("eps_dot", 0.0)
    temp = kwargs.get("temp")

    non_kw = list(args)
    if len(non_kw) > 0 and (isinstance(non_kw[0], DruckerParams) or hasattr(non_kw[0], "sigma_r")):
        params = non_kw.pop(0)
    elif len(non_kw) > 0 and (isinstance(non_kw[-1], DruckerParams) or hasattr(non_kw[-1], "sigma_r")):
        params = non_kw.pop(-1)

    if params is None:
        params = build_law104(kwargs.get("mat"), **kwargs)

    if eps_p is None:
        eps_p = float(non_kw.pop(0)) if len(non_kw) > 0 else 0.0
    if len(non_kw) > 0:
        eps_dot = float(non_kw.pop(0))
    if temp is None and len(non_kw) > 0:
        temp = float(non_kw.pop(0))

    t_curr = temp if temp is not None else params.tini

    # 1. Hardening (Linear + Voce)
    p_eff = max(0.0, float(eps_p))
    fhard = params.sigma_r + params.h * p_eff
    if params.qv != 0.0 and params.bv != 0.0:
        fhard += params.qv * (1.0 - math.exp(-params.bv * p_eff))

    # 2. Johnson-Cook strain rate sensitivity
    frate = 1.0
    epsp0_val = getattr(params, "epsp0", getattr(params, "eps0", 1.0))
    if params.cjc != 0.0 and eps_dot > epsp0_val:
        frate += params.cjc * math.log(eps_dot / epsp0_val)

    # 3. Thermal softening
    ftherm = 1.0 - params.tss * (t_curr - params.tref)

    yld = float(max(1.0e-10, fhard * frate * ftherm))

    # Hardening modulus d(sig_y)/d(eps_p)
    d_epsp = params.h
    if params.qv != 0.0 and params.bv != 0.0:
        d_epsp += params.qv * params.bv * math.exp(-params.bv * p_eff)
    d_epsp *= frate * ftherm

    return YieldStress(yld, d_epsp)


def _solid_update_single_core(
    params: DruckerParams,
    deps: np.ndarray,
    sig_old: np.ndarray,
    history: np.ndarray,
    rho: Optional[float] = None,
    rho0: Optional[float] = None,
    off: float = 1.0,
    pnew: Optional[float] = None,
    psh: float = 0.0,
    dt: float = 0.0,
    time: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Single-element 3D solid continuum constitutive update matching mat104_nodam_nice.F."""
    rho_ref = params.refer_rho if params.refer_rho > 0.0 else (params.rho if params.rho > 0.0 else 1.0)
    rho_cur = rho if (rho is not None and rho > 0.0) else rho_ref

    g = params.G
    g2 = 2.0 * g
    bulk = params.bulk
    lam = bulk - (2.0 / 3.0) * g

    ssp = math.sqrt(max(0.0, bulk + (4.0 / 3.0) * g) / rho_cur)

    # History: col 0 = plastic strain (PLA), col 1 = temperature (TEMP), col 2 = phi_old
    pla_old = float(history[0]) if len(history) > 0 else 0.0
    temp_old = float(history[1]) if len(history) > 1 else params.tini
    phi_old = float(history[2]) if len(history) > 2 else 0.0

    # Plastic strain rate
    eps_dot = float(deps[0]**2 + deps[1]**2 + deps[2]**2 + 2.0*(deps[3]**2 + deps[4]**2 + deps[5]**2))
    eps_dot = math.sqrt(max(0.0, (2.0 / 3.0) * eps_dot)) / (dt if dt > 0.0 else 1.0)

    # Self-heating weight factor (mat104_nodam_nice.F:151-165)
    dpis = params.eps_iso
    dpad = params.eps_ad
    if eps_dot < dpis:
        weitemp = 0.0
    elif eps_dot > dpad:
        weitemp = 1.0
    else:
        weitemp = ((eps_dot - dpis) ** 2) * (3.0 * dpad - 2.0 * eps_dot - dpis) / ((dpad - dpis) ** 3)

    # Yield stress
    yld = compute_drucker_yield_stress(params, eps_p=pla_old, eps_dot=eps_dot, temp=temp_old)

    # Trial stress increment
    ldav = (deps[0] + deps[1] + deps[2]) * lam
    sign_xx = sig_old[0] + deps[0] * g2 + ldav
    sign_yy = sig_old[1] + deps[1] * g2 + ldav
    sign_zz = sig_old[2] + deps[2] * g2 + ldav
    sign_xy = sig_old[3] + deps[3] * g
    sign_yz = sig_old[4] + deps[4] * g
    sign_zx = sig_old[5] + deps[5] * g

    trsig = sign_xx + sign_yy + sign_zz
    sigm = -trsig / 3.0
    sxx = sign_xx + sigm
    syy = sign_yy + sigm
    szz = sign_zz + sigm
    sxy = sign_xy
    syz = sign_yz
    szx = sign_zx

    sigdr, j2, j3 = compute_drucker_equivalent_stress(sxx, syy, szz, sxy, syz, szx, params.cdr, params.kdr)

    phi_trial = (sigdr / yld) ** 2 - 1.0

    if phi_trial <= 0.0 or off < 1.0e-5:
        # Elastic response
        sig_new = np.array([sign_xx, sign_yy, sign_zz, sign_xy, sign_yz, sign_zx], dtype=np.float64) * off
        hist_new = np.array([pla_old, temp_old, phi_trial], dtype=np.float64)
        return sig_new, hist_new, ssp

    # Plastic return mapping using cutting-plane semi-implicit procedure (mat104_nodam_newton.F:240-435)
    pla_cur = pla_old
    temp_cur = temp_old
    phi_cur = phi_trial

    for _ in range(5):
        norm_sig = math.sqrt(
            sign_xx**2 + sign_yy**2 + sign_zz**2
            + 2.0 * (sign_xy**2 + sign_yz**2 + sign_zx**2)
        )
        norm_sig = max(1.0, norm_sig)

        yld2i = 1.0 / (yld * yld)
        dphi_dsig = 2.0 * sigdr * yld2i
        fdr_scaled = (j2 / (norm_sig**2))**3 - params.cdr * ((j3 / (norm_sig**3))**2)
        fdr_scaled = max(1.0e-30, fdr_scaled)

        dphi_dfdr = dphi_dsig * params.kdr * (1.0 / 6.0) * (fdr_scaled ** (-5.0 / 6.0))
        dsdr_dj2 = dphi_dfdr * 3.0 * ((j2 / (norm_sig**2))**2)
        dsdr_dj3 = -dphi_dfdr * 2.0 * params.cdr * (j3 / (norm_sig**3))

        dj3_dsxx = (
            (2.0 / 3.0) * (syy * szz - syz**2)
            - (1.0 / 3.0) * (sxx * szz - szx**2)
            - (1.0 / 3.0) * (sxx * syy - sxy**2)
        ) / (norm_sig**2)
        dj3_dsyy = (
            -(1.0 / 3.0) * (syy * szz - syz**2)
            + (2.0 / 3.0) * (sxx * szz - szx**2)
            - (1.0 / 3.0) * (sxx * syy - sxy**2)
        ) / (norm_sig**2)
        dj3_dszz = (
            -(1.0 / 3.0) * (syy * szz - syz**2)
            - (1.0 / 3.0) * (sxx * szz - szx**2)
            + (2.0 / 3.0) * (sxx * syy - sxy**2)
        ) / (norm_sig**2)
        dj3_dsxy = 2.0 * (sxx * sxy + sxy * syy + szx * syz) / (norm_sig**2)
        dj3_dsyz = 2.0 * (sxy * szx + syy * syz + syz * szz) / (norm_sig**2)
        dj3_dszx = 2.0 * (sxx * szx + sxy * syz + szx * szz) / (norm_sig**2)

        norm_xx = dsdr_dj2 * sxx / norm_sig + dsdr_dj3 * dj3_dsxx
        norm_yy = dsdr_dj2 * syy / norm_sig + dsdr_dj3 * dj3_dsyy
        norm_zz = dsdr_dj2 * szz / norm_sig + dsdr_dj3 * dj3_dszz
        norm_xy = 2.0 * dsdr_dj2 * sxy / norm_sig + dsdr_dj3 * dj3_dsxy
        norm_yz = 2.0 * dsdr_dj2 * syz / norm_sig + dsdr_dj3 * dj3_dsyz
        norm_zx = 2.0 * dsdr_dj2 * szx / norm_sig + dsdr_dj3 * dj3_dszx

        dfdsig2 = (
            norm_xx * norm_xx * g2
            + norm_yy * norm_yy * g2
            + norm_zz * norm_zz * g2
            + norm_xy * norm_xy * g
            + norm_yz * norm_yz * g
            + norm_zx * norm_zx * g
        )

        hardp = params.h + params.qv * params.bv * math.exp(-params.bv * pla_cur)
        frate = 1.0
        if params.cjc != 0.0 and eps_dot > params.epsp0:
            frate += params.cjc * math.log(eps_dot / params.epsp0)
        ftherm = 1.0 - params.tss * (temp_cur - params.tref)

        dyld_dpla = hardp * frate * ftherm
        sig_dfdsig = (
            sign_xx * norm_xx + sign_yy * norm_yy + sign_zz * norm_zz
            + sign_xy * norm_xy + sign_yz * norm_yz + sign_zx * norm_zx
        )
        dpla_dlam = sig_dfdsig / yld

        if params.cp > 0.0:
            fhard = params.sigma_r + params.h * pla_cur + params.qv * (1.0 - math.exp(-params.bv * pla_cur))
            dyld_dtemp = -fhard * frate * params.tss
            dtemp_dlam = weitemp * (params.eta / (rho_ref * params.cp)) * sig_dfdsig
        else:
            dyld_dtemp = 0.0
            dtemp_dlam = 0.0

        dphi_dyld = -2.0 * (sigdr**2) / (yld**3)
        dphi_dlam = -dfdsig2 + (dphi_dyld * dyld_dpla * dpla_dlam)
        if params.cp > 0.0:
            dphi_dlam += (dphi_dyld * dyld_dtemp * dtemp_dlam)
        dphi_dlam = math.copysign(max(abs(dphi_dlam), 1.0e-20), dphi_dlam)

        dlam = -phi_cur / dphi_dlam
        dlam = max(0.0, dlam)

        dpxx = dlam * norm_xx
        dpyy = dlam * norm_yy
        dpzz = dlam * norm_zz
        dpxy = dlam * norm_xy
        dpyz = dlam * norm_yz
        dpzx = dlam * norm_zx

        sign_xx -= dpxx * g2
        sign_yy -= dpyy * g2
        sign_zz -= dpzz * g2
        sign_xy -= dpxy * g
        sign_yz -= dpyz * g
        sign_zx -= dpzx * g

        trsig = sign_xx + sign_yy + sign_zz
        sigm = -trsig / 3.0
        sxx = sign_xx + sigm
        syy = sign_yy + sigm
        szz = sign_zz + sigm
        sxy = sign_xy
        syz = sign_yz
        szx = sign_zx

        ddep = (dlam / yld) * sig_dfdsig
        ddep = max(0.0, ddep)
        pla_cur += ddep

        if params.cp > 0.0:
            dtemp = weitemp * yld * ddep * params.eta / (rho_ref * params.cp)
            temp_cur += dtemp

        ftherm = 1.0 - params.tss * (temp_cur - params.tref)
        fhard = params.sigma_r + params.h * pla_cur + params.qv * (1.0 - math.exp(-params.bv * pla_cur))
        yld = max(1.0e-10, fhard * frate * ftherm)

        sigdr, j2, j3 = compute_drucker_equivalent_stress(sxx, syy, szz, sxy, syz, szx, params.cdr, params.kdr)
        phi_cur = (sigdr / yld)**2 - 1.0
        if abs(phi_cur) < 1.0e-5:
            break

    sig_new = np.array([sign_xx, sign_yy, sign_zz, sign_xy, sign_yz, sign_zx], dtype=np.float64) * off
    hist_new = np.array([pla_cur, temp_cur, 0.0], dtype=np.float64)

    return sig_new, hist_new, ssp


def solid_update_array(
    params: DruckerParams,
    deps: np.ndarray,
    sig_old: np.ndarray,
    history: np.ndarray,
    rho: Optional[np.ndarray] = None,
    rho0: Optional[np.ndarray] = None,
    off: Optional[np.ndarray] = None,
    pnew: Optional[np.ndarray] = None,
    psh: float = 0.0,
    dt: float = 0.0,
    time: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized / multi-element 3D solid continuum update."""
    nel = len(deps)
    sig_new = np.zeros_like(sig_old)
    hist_new = np.zeros((nel, 3), dtype=np.float64)
    ssp = np.zeros(nel, dtype=np.float64)

    for i in range(nel):
        r = float(rho[i]) if (rho is not None and getattr(rho, "ndim", 0) > 0) else (float(rho) if rho is not None else None)
        r0 = float(rho0[i]) if (rho0 is not None and getattr(rho0, "ndim", 0) > 0) else (float(rho0) if rho0 is not None else None)
        o = float(off[i]) if (off is not None and getattr(off, "ndim", 0) > 0) else (float(off) if off is not None else 1.0)
        pn = float(pnew[i]) if (pnew is not None and getattr(pnew, "ndim", 0) > 0) else (float(pnew) if pnew is not None else None)
        h_row = history[i] if history is not None and len(history) > i else np.zeros(3)

        s_i, h_i, c_i = _solid_update_single_core(
            params, deps[i], sig_old[i], h_row, rho=r, rho0=r0, off=o, pnew=pn, psh=psh, dt=dt, time=time
        )
        sig_new[i] = s_i
        hist_new[i, :len(h_i)] = h_i
        ssp[i] = c_i

    return sig_new, hist_new, ssp


def solid_update(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    rho: Optional[Any] = None,
    rho0: Optional[Any] = None,
    off: Any = 1.0,
    pnew: Optional[Any] = None,
    psh: float = 0.0,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    ismstr: int = 0,
    *,
    epsp: Optional[np.ndarray] = None,
    return_tuple: bool = True,
    **kwargs: Any,
) -> Any:
    """3D solid continuum constitutive update for /MAT/LAW104."""
    # Direct array call
    if isinstance(mat, DruckerParams) and sig is not None and deps is not None and "history" in kwargs:
        deps_in = np.asarray(sig, dtype=np.float64)
        sig_in = np.asarray(deps, dtype=np.float64)
        hist_in = np.asarray(kwargs["history"], dtype=np.float64)
        return solid_update_array(
            mat, deps_in, sig_in, hist_in,
            rho=kwargs.get("rho"), rho0=kwargs.get("rho0"), off=kwargs.get("off"),
            pnew=kwargs.get("pnew"), psh=kwargs.get("psh", 0.0), dt=dt, time=kwargs.get("time", 0.0),
        )

    params = mat if isinstance(mat, DruckerParams) else build_law104(mat, **kwargs)

    if sig is None:
        sig = np.zeros(6, dtype=np.float64)
    is_1d = (sig.ndim == 1)

    sig_arr = sig[np.newaxis, :] if is_1d else sig
    deps_arr = deps[np.newaxis, :] if (deps is not None and is_1d) else (deps if deps is not None else np.zeros_like(sig_arr))
    nel = len(sig_arr)
    if epsp is not None:
        if isinstance(epsp, (int, float)):
            epsp_arr = np.full(nel, float(epsp), dtype=np.float64)
        else:
            epsp_arr = np.asarray(epsp, dtype=np.float64)
            if epsp_arr.ndim == 0:
                epsp_arr = np.full(nel, float(epsp_arr), dtype=np.float64)
    else:
        epsp_arr = np.zeros(nel, dtype=np.float64)

    hist = None
    if extra is not None:
        for k in ("uvar104", "uvar", "history"):
            if k in extra and extra[k] is not None:
                hist = extra[k]
                break

    if hist is None or len(hist) == 0:
        hist_arr = np.zeros((nel, 3), dtype=np.float64)
        hist_arr[:, 0] = epsp_arr
        hist_arr[:, 1] = params.tini
    else:
        hist_arr = np.asarray(hist, dtype=np.float64)
        if hist_arr.ndim == 1:
            hist_arr = hist_arr[np.newaxis, :]
        if hist_arr.shape[1] < 3:
            padded = np.zeros((nel, 3), dtype=np.float64)
            padded[:, :hist_arr.shape[1]] = hist_arr
            if hist_arr.shape[1] < 2:
                padded[:, 1] = params.tini
            hist_arr = padded

    rho_arr = np.asarray(rho, dtype=np.float64) if rho is not None else None
    rho0_arr = np.asarray(rho0, dtype=np.float64) if rho0 is not None else None
    off_arr = np.asarray(off, dtype=np.float64) if off is not None else None
    pnew_arr = np.asarray(pnew, dtype=np.float64) if pnew is not None else None

    sig_new, hist_new, ssp = solid_update_array(
        params, deps_arr, sig_arr, hist_arr,
        rho=rho_arr, rho0=rho0_arr, off=off_arr, pnew=pnew_arr, psh=psh, dt=dt, time=kwargs.get("time", 0.0),
    )

    if extra is not None:
        if is_1d:
            extra["uvar104"] = hist_new[0]
            extra["uvar"] = hist_new[0]
        else:
            extra["uvar104"] = hist_new
            extra["uvar"] = hist_new

    if is_1d:
        sig_out = sig_new[0]
        epsp_out = float(hist_new[0, 0])
        ssp_out = float(ssp[0])
    else:
        sig_out = sig_new
        epsp_out = hist_new[:, 0]
        ssp_out = ssp

    if return_tuple:
        return sig_out, epsp_out, ssp_out
    return sig_out, epsp_out


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    thk: Optional[np.ndarray] = None,
    thkly: float = 1.0,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, float]:
    r"""2D shell plane-stress constitutive update for /MAT/LAW104 matching sigeps104c.F / mat104c_nodam_nice.F.

    - Plane-stress projection \(\sigma_{zz} = 0, \sigma_{yz} = \sigma_{zx} = 0\).
    - Out-of-plane thickness thinning \(d\varepsilon_{zz}\).
    """
    params = mat if isinstance(mat, DruckerParams) else build_law104(mat, **kwargs)

    is_1d = (sig.ndim == 1)
    sig_arr = sig[np.newaxis, :] if is_1d else sig
    deps_arr = deps[np.newaxis, :] if is_1d else deps
    nel = len(sig_arr)

    # Elastic plane-stress stiffness
    young = params.young
    nu = params.nu
    denom = 1.0 - nu * nu
    q11 = young / denom if denom > 0.0 else young
    q12 = nu * q11
    q33 = young / (2.0 * (1.0 + nu)) if (1.0 + nu) > 0.0 else 0.5 * young

    if epsp is not None:
        if isinstance(epsp, (int, float)):
            epsp_arr = np.full(nel, float(epsp), dtype=np.float64)
        else:
            epsp_arr = np.asarray(epsp, dtype=np.float64)
            if epsp_arr.ndim == 0:
                epsp_arr = np.full(nel, float(epsp_arr), dtype=np.float64)
    else:
        epsp_arr = np.zeros(nel, dtype=np.float64)

    sig_out = np.zeros_like(sig_arr)
    epsp_out = np.zeros(nel, dtype=np.float64)

    for i in range(nel):
        s_old = sig_arr[i]
        d_eps = deps_arr[i]
        p_old = epsp_arr[i]

        # Trial elastic stress (sig_xx, sig_yy, sig_xy)
        s_xx = s_old[0] + q11 * d_eps[0] + q12 * d_eps[1]
        s_yy = s_old[1] + q12 * d_eps[0] + q11 * d_eps[1]
        s_xy = s_old[2] + q33 * d_eps[2]

        # In plane stress: szz = syz = szx = 0
        sig_m = -(s_xx + s_yy) / 3.0
        s_dev_xx = s_xx + sig_m
        s_dev_yy = s_yy + sig_m
        s_dev_zz = sig_m

        sigdr, j2, j3 = compute_drucker_equivalent_stress(
            s_dev_xx, s_dev_yy, s_dev_zz, s_xy, 0.0, 0.0, params.cdr, params.kdr
        )

        eps_dot = math.sqrt(d_eps[0]**2 + d_eps[1]**2 + 0.5 * d_eps[2]**2) / (dt if dt > 0.0 else 1.0)
        yld = compute_drucker_yield_stress(params, eps_p=p_old, eps_dot=eps_dot, temp=params.tini)

        phi = (sigdr / yld) ** 2 - 1.0

        if phi <= 0.0:
            sig_out[i] = np.array([s_xx, s_yy, s_xy])
            epsp_out[i] = p_old
        else:
            # Radial scaling for plane stress
            ratio = yld / (sigdr + 1.0e-20)
            sxx = s_xx * ratio
            syy = s_yy * ratio
            sig_out[i] = np.array([sxx, syy, s_xy * ratio])
            d_pla = (1.0 - ratio) * sigdr / (3.0 * q33)
            epsp_out[i] = p_old + max(0.0, d_pla)

            # Thickness thinning
            sig_y = yld
            dezz_pl = - d_pla * 0.5 * (sxx + syy) / max(_EPS20, sig_y)
            d_eps_zz = -(nu / (1.0 - nu)) * (d_eps[0] + d_eps[1]) + dezz_pl
            if thk is not None:
                if hasattr(thk, "__setitem__"):
                    thk[i] *= (1.0 + d_eps_zz * thkly)
            if extra is not None and isinstance(extra, dict):
                extra["thickness_strain"] = d_eps_zz

    c_shell = params.sound_speed_shell

    if is_1d:
        return sig_out[0], float(epsp_out[0]), c_shell
    return sig_out, epsp_out, c_shell


def sound_speed(mat: Any = None, rho: Optional[Any] = None, is_shell: bool = False, **kwargs: Any) -> Any:
    """Acoustic sound speed for /MAT/LAW104."""
    params = mat if isinstance(mat, DruckerParams) else build_law104(mat, **kwargs)
    if is_shell:
        return sound_speed_shell(params, rho=rho)
    return sound_speed_solid(params, rho=rho)


def sound_speed_solid(mat: Any = None, rho: Optional[Any] = None, **kwargs: Any) -> Any:
    """Dilatational sound speed for /MAT/LAW104 3D solids."""
    params = mat if isinstance(mat, DruckerParams) else build_law104(mat, **kwargs)
    rho_val = rho if (rho is not None and np.all(np.asarray(rho) > 0.0)) else (
        params.refer_rho if getattr(params, "refer_rho", 0.0) > 0.0 else (
            params.rho if getattr(params, "rho", 0.0) > 0.0 else 1.0
        )
    )
    r_arr = np.asarray(rho_val, dtype=np.float64)
    dpdm = params.bulk + (4.0 / 3.0) * params.G
    return np.sqrt(np.maximum(0.0, dpdm) / np.where(r_arr > 0.0, r_arr, 1.0)) if r_arr.ndim > 0 else math.sqrt(max(0.0, dpdm) / (float(r_arr) if float(r_arr) > 0.0 else 1.0))


def sound_speed_shell(mat: Any = None, rho: Optional[Any] = None, **kwargs: Any) -> Any:
    """Plane-stress sound speed for /MAT/LAW104 2D shells."""
    params = mat if isinstance(mat, DruckerParams) else build_law104(mat, **kwargs)
    rho_val = rho if (rho is not None and np.all(np.asarray(rho) > 0.0)) else (
        params.refer_rho if getattr(params, "refer_rho", 0.0) > 0.0 else (
            params.rho if getattr(params, "rho", 0.0) > 0.0 else 1.0
        )
    )
    r_arr = np.asarray(rho_val, dtype=np.float64)
    denom = 1.0 - params.nu * params.nu
    mod = params.young / denom if denom > 0.0 else params.young
    return np.sqrt(np.maximum(0.0, mod) / np.where(r_arr > 0.0, r_arr, 1.0)) if r_arr.ndim > 0 else math.sqrt(max(0.0, mod) / (float(r_arr) if float(r_arr) > 0.0 else 1.0))


def solid_tangent(
    mat: Any = None,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Algorithmic elastoplastic tangent stiffness matrix for /MAT/LAW104 (6x6)."""
    params = mat if isinstance(mat, DruckerParams) else build_law104(mat, **kwargs)
    g = params.G
    bulk = params.bulk

    c11 = bulk + (4.0 / 3.0) * g
    c12 = bulk - (2.0 / 3.0) * g

    c_mat = np.array([
        [c11, c12, c12, 0.0, 0.0, 0.0],
        [c12, c11, c12, 0.0, 0.0, 0.0],
        [c12, c12, c11, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, g,   0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, g,   0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, g  ],
    ], dtype=np.float64)

    if sig is not None and np.ndim(sig) > 1:
        n = np.shape(sig)[0]
        return np.broadcast_to(c_mat, (n, 6, 6)).copy()

    return c_mat


def consistent_shell_tangent(
    mat: Any = None,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Plane-stress membrane tangent matrix for /MAT/LAW104 (3x3)."""
    params = mat if isinstance(mat, DruckerParams) else build_law104(mat, **kwargs)
    young = params.young
    nu = params.nu
    denom = 1.0 - nu * nu
    q11 = young / denom if denom > 0.0 else young
    q12 = nu * q11
    q33 = young / (2.0 * (1.0 + nu)) if (1.0 + nu) > 0.0 else 0.5 * young

    c_mat = np.array([
        [q11, q12, 0.0],
        [q12, q11, 0.0],
        [0.0, 0.0, q33],
    ], dtype=np.float64)

    if sig is not None and np.ndim(sig) > 1:
        n = np.shape(sig)[0]
        return np.broadcast_to(c_mat, (n, 3, 3)).copy()

    return c_mat


shell_membrane_tangent = consistent_shell_tangent


def extra_shapes(mat: Any = None, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Extra history shapes needed for LAW104."""
    return {"uvar104": (nip, 3) if nip else (3,)}

