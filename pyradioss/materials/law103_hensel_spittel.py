r"""LAW103 — Hensel-Spittel hot metal forming material model (/MAT/LAW103, /MAT/HENSEL_SPITTEL, /MAT/PLAS_HENS).

Fortran origins:
- ``starter/source/materials/mat/mat103/hm_read_mat103.F`` (starter card reader, defaults, unit conversions, checks)
- ``engine/source/materials/mat/mat103/sigeps103.F`` (3D continuum solid constitutive update)
- ``hm_cfg_files/config/CFG/radioss2020/MAT/matl103_hensel_spittel.cfg`` (CFG attributes & 6-card format)

Theory & Constitutive Formulation
----------------------------------
LAW103 implements the empirical Hensel-Spittel flow stress equation widely used in metal
forming simulations (hot rolling, extrusion, forging) to describe thermomechanical
viscoplastic behavior with strain hardening, dynamic recovery, strain-rate sensitivity,
and temperature dependence:

.. math::
    \sigma_y = A_0 \cdot \text{YLD\_H} \cdot \text{YLD\_SR} \cdot \text{YLD\_T}

where:
- Total equivalent strain:
  .. math:: \varepsilon = \varepsilon_0 + \varepsilon_p
- Hardening scale factor:
  .. math:: \text{YLD\_H} = \varepsilon^{m_2} \cdot \exp(m_4 / \varepsilon) \cdot \exp(m_7 \varepsilon)
- Strain rate scale factor (converted to \(\text{s}^{-1}\)):
  .. math:: \text{YLD\_SR} = (\dot{\varepsilon} \cdot \text{TIME\_FAC})^{m_3}
- Temperature scale factor (\(\theta = T - 273.15\) in \(^\circ\text{C}\)):
  .. math:: \text{YLD\_T} = \exp(m_1 \theta) \cdot (1 + \varepsilon)^{m_5 \theta}

Analytical hardening modulus:
.. math::
    H_m = \frac{\partial \sigma_y}{\partial \varepsilon} = m_7 \sigma_y + \sigma_y \frac{m_2 - m_4 / \varepsilon}{\varepsilon}

Radial return projection:
.. math::
    \text{ratio} = \min\left(1, \frac{\sigma_y}{\max(\sigma_{\text{vm}}, 10^{-20})}\right)
.. math::
    \Delta\varepsilon_p = \frac{(1 - \text{ratio})\sigma_{\text{vm}}}{\max(3G + H_m, 10^{-20})}
.. math::
    \sigma_y^{\text{actual}} = \max(0, \sigma_y + \Delta\varepsilon_p H_m)
.. math::
    \mathbf{s}^{\text{new}} = \mathbf{s}^{\text{trial}} \cdot \min\left(1, \frac{\sigma_y^{\text{actual}}}{\max(\sigma_{\text{vm}}, 10^{-20})}\right)

Adiabatic plastic work dissipation:
.. math::
    \Delta T = \frac{\eta \cdot \sigma_y^{\text{actual}} \cdot \Delta\varepsilon_p}{\rho C_p}

Acoustic dilatational sound speed:
.. math::
    c = \sqrt{\frac{K + \frac{4}{3}G}{\rho_0}}

Element compatibility:
3D continuum solid elements (Hexa8, Tetra4, Penta6, Pyra5) and SPH particles.
Rejects 2D shells (ANCMSG 305) and 1D elements (ANCMSG 306).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

_EM14 = 1.0e-14
_EM20 = 1.0e-20
_T0K = 273.15
_DEFAULT_PMIN = -1.0e30
_DEFAULT_RCP = 1.0e30


@dataclass
class HenselSpittelParams:
    """Consolidated material parameters for /MAT/LAW103 (/MAT/HENSEL_SPITTEL)."""
    id: int = 1
    title: str = ""
    law: int = 103
    law_name: str = "LAW103"
    rho: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    a0: float = 0.0
    m1: float = 0.0
    m2: float = 0.0
    m3: float = 0.0
    m4: float = 0.0
    m5: float = 0.0
    m7: float = 0.0
    fsmooth: int = 0
    fcut: float = 0.0
    eps0: float = 0.0
    pmin: float = _DEFAULT_PMIN
    rcp: float = _DEFAULT_RCP
    t0: float = 293.15
    eta: float = 0.9
    time_fac: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def rho0(self) -> float:
        return self.rho

    @property
    def rhor(self) -> float:
        return self.refer_rho if self.refer_rho > 0.0 else self.rho

    @property
    def E(self) -> float:
        return self.e

    @property
    def nu_val(self) -> float:
        return self.nu

    @property
    def G(self) -> float:
        if (1.0 + self.nu) != 0.0:
            return self.e / (2.0 * (1.0 + self.nu))
        return 0.0

    @property
    def g(self) -> float:
        return self.G

    @property
    def K(self) -> float:
        denom = 1.0 - 2.0 * self.nu
        if denom != 0.0:
            return self.e / (3.0 * denom)
        return 0.0

    @property
    def bulk(self) -> float:
        return self.K

    @property
    def k(self) -> float:
        return self.K

    @property
    def eps_0(self) -> float:
        return self.eps0

    @property
    def rhocp(self) -> float:
        return self.rcp

    @property
    def sound_speed(self) -> float:
        rho_ref = self.rho if self.rho > 0.0 else 1.0
        return math.sqrt(max(0.0, self.bulk + (4.0 / 3.0) * self.g) / rho_ref)


def build_law103(mat: Any = None, **kwargs: Any) -> HenselSpittelParams:
    """Build HenselSpittelParams from a Model entity or keyword dictionary."""
    if isinstance(mat, HenselSpittelParams):
        return mat

    params_dict: Dict[str, Any] = {}
    if mat is not None:
        if hasattr(mat, "id"):
            params_dict["id"] = getattr(mat, "id", 1)
        if hasattr(mat, "title"):
            params_dict["title"] = getattr(mat, "title", "")
        if hasattr(mat, "rho"):
            params_dict["rho"] = getattr(mat, "rho", 0.0)
        elif hasattr(mat, "rho0"):
            params_dict["rho"] = getattr(mat, "rho0", 0.0)
        if hasattr(mat, "refer_rho"):
            params_dict["refer_rho"] = getattr(mat, "refer_rho", 0.0)
        elif hasattr(mat, "rhor"):
            params_dict["refer_rho"] = getattr(mat, "rhor", 0.0)
        if hasattr(mat, "e"):
            params_dict["e"] = getattr(mat, "e", 0.0)
        elif hasattr(mat, "E"):
            params_dict["e"] = getattr(mat, "E", 0.0)
        if hasattr(mat, "nu"):
            params_dict["nu"] = getattr(mat, "nu", 0.0)
        if hasattr(mat, "a0"):
            params_dict["a0"] = getattr(mat, "a0", 0.0)
        if hasattr(mat, "m1"):
            params_dict["m1"] = getattr(mat, "m1", 0.0)
        if hasattr(mat, "m2"):
            params_dict["m2"] = getattr(mat, "m2", 0.0)
        if hasattr(mat, "m3"):
            params_dict["m3"] = getattr(mat, "m3", 0.0)
        if hasattr(mat, "m4"):
            params_dict["m4"] = getattr(mat, "m4", 0.0)
        if hasattr(mat, "m5"):
            params_dict["m5"] = getattr(mat, "m5", 0.0)
        if hasattr(mat, "m7"):
            params_dict["m7"] = getattr(mat, "m7", 0.0)
        if hasattr(mat, "fsmooth"):
            params_dict["fsmooth"] = getattr(mat, "fsmooth", 0)
        if hasattr(mat, "fcut"):
            params_dict["fcut"] = getattr(mat, "fcut", 0.0)
        if hasattr(mat, "eps0"):
            params_dict["eps0"] = getattr(mat, "eps0", 0.0)
        elif hasattr(mat, "eps_0"):
            params_dict["eps0"] = getattr(mat, "eps_0", 0.0)
        if hasattr(mat, "pmin"):
            params_dict["pmin"] = getattr(mat, "pmin", _DEFAULT_PMIN)
        if hasattr(mat, "rcp"):
            params_dict["rcp"] = getattr(mat, "rcp", _DEFAULT_RCP)
        elif hasattr(mat, "rhocp"):
            params_dict["rcp"] = getattr(mat, "rhocp", _DEFAULT_RCP)
        if hasattr(mat, "t0"):
            params_dict["t0"] = getattr(mat, "t0", 293.15)
        if hasattr(mat, "eta"):
            params_dict["eta"] = getattr(mat, "eta", 0.9)
        if hasattr(mat, "time_fac"):
            params_dict["time_fac"] = getattr(mat, "time_fac", 1.0)
        if hasattr(mat, "params") and isinstance(mat.params, dict):
            for k, v in mat.params.items():
                params_dict.setdefault(k, v)

    for k, v in kwargs.items():
        kl = k.lower()
        if kl in ("material_id", "mat_id", "id", "mid"):
            params_dict["id"] = int(v)
        elif kl == "title":
            params_dict["title"] = str(v)
        elif kl in ("rho", "rho0", "rho_i", "initial_density"):
            params_dict["rho"] = float(v)
        elif kl in ("refer_rho", "rhor", "ref_rho", "reference_density"):
            params_dict["refer_rho"] = float(v)
        elif kl in ("e", "young", "young_modulus"):
            params_dict["e"] = float(v)
        elif kl in ("nu", "poisson", "poisson_ratio"):
            params_dict["nu"] = float(v)
        elif kl in ("a0", "mat103_a0"):
            params_dict["a0"] = float(v)
        elif kl in ("m1", "mat103_m1"):
            params_dict["m1"] = float(v)
        elif kl in ("m2", "mat103_m2"):
            params_dict["m2"] = float(v)
        elif kl in ("m3", "mat103_m3"):
            params_dict["m3"] = float(v)
        elif kl in ("m4", "mat103_m4"):
            params_dict["m4"] = float(v)
        elif kl in ("m5", "mat103_m5"):
            params_dict["m5"] = float(v)
        elif kl in ("m7", "mat103_m7"):
            params_dict["m7"] = float(v)
        elif kl in ("fsmooth", "mat_fsmooth"):
            params_dict["fsmooth"] = int(v)
        elif kl in ("fcut", "f_cut"):
            params_dict["fcut"] = float(v)
        elif kl in ("eps0", "eps_0", "mat_srp"):
            params_dict["eps0"] = float(v)
        elif kl in ("pmin", "mat_pc", "p_min"):
            params_dict["pmin"] = float(v)
        elif kl in ("rcp", "rhocp", "mat_spheat", "rho_cp"):
            params_dict["rcp"] = float(v)
        elif kl in ("t0", "mat_t0", "tini", "t_ini"):
            params_dict["t0"] = float(v)
        elif kl in ("eta", "mat103_eta"):
            params_dict["eta"] = float(v)
        elif kl in ("time_fac", "fac_t_work"):
            params_dict["time_fac"] = float(v)

    rho_val = float(params_dict.get("rho", 0.0))
    refer_rho = float(params_dict.get("refer_rho", 0.0))
    if refer_rho == 0.0:
        refer_rho = rho_val

    pmin_val = float(params_dict.get("pmin", _DEFAULT_PMIN))
    if pmin_val == 0.0:
        pmin_val = _DEFAULT_PMIN

    rcp_val = float(params_dict.get("rcp", _DEFAULT_RCP))
    if rcp_val == 0.0:
        rcp_val = _DEFAULT_RCP

    eta_val = min(1.0, max(0.0, float(params_dict.get("eta", 0.9))))

    return HenselSpittelParams(
        id=int(params_dict.get("id", 1)),
        title=str(params_dict.get("title", "")),
        rho=rho_val,
        refer_rho=refer_rho,
        e=float(params_dict.get("e", 0.0)),
        nu=float(params_dict.get("nu", 0.0)),
        a0=float(params_dict.get("a0", 0.0)),
        m1=float(params_dict.get("m1", 0.0)),
        m2=float(params_dict.get("m2", 0.0)),
        m3=float(params_dict.get("m3", 0.0)),
        m4=float(params_dict.get("m4", 0.0)),
        m5=float(params_dict.get("m5", 0.0)),
        m7=float(params_dict.get("m7", 0.0)),
        fsmooth=int(params_dict.get("fsmooth", 0)),
        fcut=float(params_dict.get("fcut", 0.0)),
        eps0=float(params_dict.get("eps0", 0.0)),
        pmin=pmin_val,
        rcp=rcp_val,
        t0=float(params_dict.get("t0", 293.15)),
        eta=eta_val,
        time_fac=float(params_dict.get("time_fac", 1.0)),
        extra=params_dict,
    )


def compute_hensel_spittel_flow_stress(
    a0_or_params: Any,
    m1: float = 0.0,
    m2: float = 0.0,
    m3: float = 0.0,
    m4: float = 0.0,
    m5: float = 0.0,
    m7: float = 0.0,
    eps: float = 0.0,
    eps_dot: float = 0.0,
    temp: Optional[float] = None,
    theta: Optional[float] = None,
    time_fac: float = 1.0,
    *,
    eps_p: Optional[float] = None,
    **kwargs: Any,
) -> float:
    r"""Compute Hensel-Spittel yield flow stress matching sigeps103.F lines 148-197.

    .. math::
        \sigma_y = A_0 \cdot \text{YLD\_H} \cdot \text{YLD\_SR} \cdot \text{YLD\_T}

    Accepts either an explicit parameter list:
      compute_hensel_spittel_flow_stress(a0, m1, m2, m3, m4, m5, m7, eps, eps_dot=..., temp=...)
    or a params object:
      compute_hensel_spittel_flow_stress(params, eps_p=..., eps_dot=..., temp=...)
    """
    if isinstance(a0_or_params, HenselSpittelParams) or hasattr(a0_or_params, "a0"):
        p = a0_or_params
        a0 = p.a0
        m1 = p.m1
        m2 = p.m2
        m3 = p.m3
        m4 = p.m4
        m5 = p.m5
        m7 = p.m7
        time_fac = getattr(p, "time_fac", 1.0)
        eff_p = eps_p if eps_p is not None else kwargs.get("epsp", eps)
        eps0 = getattr(p, "eps0", 0.0)
        eps = max(eps0, eps0 + eff_p) if eff_p > 0.0 else max(eps0, eps0)
    else:
        a0 = float(a0_or_params)
        if eps_p is not None and eps == 0.0:
            eps = eps_p

    if a0 <= 0.0:
        return 0.0

    # 1. Hardening scale factor (sigeps103.F:148-165)
    yld_h = 1.0
    if m2 != 0.0 and eps > 0.0:
        yld_h *= (eps ** m2)
    if m4 != 0.0 and eps > 0.0:
        val4 = m4 / eps
        yld_h *= math.exp(val4) if val4 < 700.0 else 1.0e300
    if m7 != 0.0 and eps > 0.0:
        val7 = m7 * eps
        yld_h *= math.exp(val7) if val7 < 700.0 else 1.0e300

    # 2. Strain rate scale factor (sigeps103.F:167-175)
    if m3 != 0.0:
        rate = eps_dot * time_fac
        if rate > 0.0:
            yld_sr = rate ** m3
        else:
            yld_sr = 0.0
    else:
        yld_sr = 1.0

    # 3. Temperature scale factor (sigeps103.F:177-193)
    if theta is None:
        if temp is not None:
            th = temp - _T0K
        else:
            th = 20.0
    else:
        th = float(theta)

    if m1 != 0.0 and m5 != 0.0:
        exp1 = math.exp(th * m1) if (th * m1) < 700.0 else 1.0e300
        pow5 = ((1.0 + eps) ** (th * m5)) if (1.0 + eps) > 0.0 else 1.0
        yld_t = exp1 * pow5
    elif m1 != 0.0:
        yld_t = math.exp(th * m1) if (th * m1) < 700.0 else 1.0e300
    elif m5 != 0.0:
        yld_t = ((1.0 + eps) ** (th * m5)) if (1.0 + eps) > 0.0 else 1.0
    else:
        yld_t = 1.0

    return float(a0 * yld_h * yld_sr * yld_t)


def compute_hensel_spittel_hardening_modulus(
    a0_or_params: Any,
    m1: float = 0.0,
    m2: float = 0.0,
    m3: float = 0.0,
    m4: float = 0.0,
    m5: float = 0.0,
    m7: float = 0.0,
    eps: float = 0.0,
    eps_dot: float = 0.0,
    temp: Optional[float] = None,
    theta: Optional[float] = None,
    yld: Optional[float] = None,
    time_fac: float = 1.0,
    *,
    eps_p: Optional[float] = None,
    **kwargs: Any,
) -> float:
    r"""Compute Hensel-Spittel plastic hardening modulus matching sigeps103.F lines 201-206.

    .. math::
        H_m = m_7 \sigma_y + \sigma_y \frac{m_2 - m_4 / \varepsilon}{\varepsilon}
    """
    if isinstance(a0_or_params, HenselSpittelParams) or hasattr(a0_or_params, "a0"):
        p = a0_or_params
        a0 = p.a0
        m1 = p.m1
        m2 = p.m2
        m3 = p.m3
        m4 = p.m4
        m5 = p.m5
        m7 = p.m7
        time_fac = getattr(p, "time_fac", 1.0)
        eff_p = eps_p if eps_p is not None else kwargs.get("epsp", eps)
        eps0 = getattr(p, "eps0", 0.0)
        eps = max(eps0, eps0 + eff_p) if eff_p > 0.0 else max(eps0, eps0)
    else:
        a0 = float(a0_or_params)
        if eps_p is not None and eps == 0.0:
            eps = eps_p

    if yld is None:
        yld = compute_hensel_spittel_flow_stress(
            a0, m1, m2, m3, m4, m5, m7, eps, eps_dot=eps_dot, temp=temp, theta=theta, time_fac=time_fac
        )
    hm = m7 * yld
    if eps > 0.0:
        hm += yld * (m2 - m4 / eps) / eps
    return float(hm)


def init_history(n: int = 1, t0: float = 293.15) -> np.ndarray:
    r"""Initialize internal history array for n integration points.

    State variables per integration point:
    - Col 0: Cumulative equivalent plastic strain (\(\varepsilon_p\), PLA)
    - Col 1: Current temperature in Kelvin (T, UVAR(:, 1))
    """
    hist = np.zeros((n, 2), dtype=np.float64)
    hist[:, 1] = t0
    return hist


def _solid_update_single_core(
    params: HenselSpittelParams,
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
    """Single-element 3D solid continuum constitutive update matching sigeps103.F."""
    g = params.g
    g2 = 2.0 * g
    g3 = 3.0 * g
    bulk = params.bulk

    rho_ref = rho0 if (rho0 is not None and rho0 > 0.0) else params.rho
    if rho_ref <= 0.0:
        rho_ref = 1.0

    # History variables: [pla, temp_K]
    pla = float(history[0])
    temp_k = float(history[1])
    if temp_k <= 0.0:
        temp_k = params.t0

    # Hydrostatic pressure and mean volumetric strain (sigeps103.F:107-109)
    pold = -(sig_old[0] + sig_old[1] + sig_old[2]) / 3.0
    davg = (deps[0] + deps[1] + deps[2]) / 3.0
    eps = params.eps0 + pla

    # Elastic trial deviatoric stress tensor (sigeps103.F:114-121)
    s1 = sig_old[0] + pold + g2 * (deps[0] - davg)
    s2 = sig_old[1] + pold + g2 * (deps[1] - davg)
    s3 = sig_old[2] + pold + g2 * (deps[2] - davg)
    s4 = sig_old[3] + g * deps[3]
    s5 = sig_old[4] + g * deps[4]
    s6 = sig_old[5] + g * deps[5]

    # Sound speed (sigeps103.F:126-127)
    dpdm = bulk + (4.0 / 3.0) * g
    ssp = math.sqrt(max(0.0, dpdm) / rho_ref)

    # Von Mises equivalent stress (sigeps103.F:132-135)
    j2 = 0.5 * (s1 * s1 + s2 * s2 + s3 * s3) + s4 * s4 + s5 * s5 + s6 * s6
    svm = math.sqrt(max(0.0, 3.0 * j2))

    # Temperature in Celsius (sigeps103.F:139-143)
    theta = temp_k - _T0K

    # Equivalent strain rate epsd
    if dt > 0.0:
        de_dev_xx = deps[0] - davg
        de_dev_yy = deps[1] - davg
        de_dev_zz = deps[2] - davg
        de_dev_xy = 0.5 * deps[3]
        de_dev_yz = 0.5 * deps[4]
        de_dev_zx = 0.5 * deps[5]
        de_j2 = 0.5 * (de_dev_xx ** 2 + de_dev_yy ** 2 + de_dev_zz ** 2) + de_dev_xy ** 2 + de_dev_yz ** 2 + de_dev_zx ** 2
        epsd = math.sqrt(max(0.0, 2.0 / 3.0 * (2.0 * de_j2))) / dt
    else:
        epsd = 0.0

    # Hensel-Spittel Yield Stress (sigeps103.F:148-197)
    yld = compute_hensel_spittel_flow_stress(
        params.a0, params.m1, params.m2, params.m3, params.m4, params.m5, params.m7,
        eps, eps_dot=epsd, theta=theta, time_fac=params.time_fac
    )

    # Hardening modulus (sigeps103.F:201-206)
    hm = compute_hensel_spittel_hardening_modulus(
        params.a0, params.m1, params.m2, params.m3, params.m4, params.m5, params.m7,
        eps, eps_dot=epsd, theta=theta, yld=yld, time_fac=params.time_fac
    )

    # Radial return projection (sigeps103.F:210-225)
    svm_safe = max(svm, _EM20)
    ratio = min(1.0, yld / svm_safe)

    # Plastic strain increment
    denom_g3_hm = max(g3 + hm, _EM20)
    dpla = (1.0 - ratio) * svm / denom_g3_hm

    # Actual yield stress and updated ratio
    yld_actual = max(0.0, yld + dpla * hm)
    ratio_actual = min(1.0, yld_actual / svm_safe)

    # Updated deviatoric stress
    s1 *= ratio_actual
    s2 *= ratio_actual
    s3 *= ratio_actual
    s4 *= ratio_actual
    s5 *= ratio_actual
    s6 *= ratio_actual

    # Cumulative plastic strain and temperature update (sigeps103.F:223-225)
    pla_new = pla + dpla
    if params.rcp > 0.0 and params.rcp < 1.0e29:
        delta_temp = params.eta * yld_actual * dpla / params.rcp
    else:
        delta_temp = 0.0
    temp_new = temp_k + delta_temp

    # Hydrostatic pressure: bulk elasticity if pnew not passed
    if pnew is None:
        pnew_val = pold - 3.0 * bulk * davg
    else:
        pnew_val = float(pnew)

    # Pressure cutoff Pmin
    if pnew_val < params.pmin:
        pnew_val = params.pmin

    ptot = pnew_val + psh

    # Total Cauchy stress tensor
    sig_new = np.empty(6, dtype=np.float64)
    sig_new[0] = s1 * off - ptot
    sig_new[1] = s2 * off - ptot
    sig_new[2] = s3 * off - ptot
    sig_new[3] = s4 * off
    sig_new[4] = s5 * off
    sig_new[5] = s6 * off

    history_new = np.array([pla_new, temp_new], dtype=np.float64)
    return sig_new, history_new, ssp


def solid_update_single(
    params: HenselSpittelParams,
    sig_or_deps: Any,
    deps_or_sig: Any,
    history: Optional[np.ndarray] = None,
    rho: Optional[float] = None,
    rho0: Optional[float] = None,
    off: float = 1.0,
    pnew: Optional[float] = None,
    psh: float = 0.0,
    *,
    epsp: Optional[float] = None,
    dt: float = 0.0,
    time: float = 0.0,
    **kwargs: Any,
) -> Any:
    """Single-element 3D solid continuum constitutive update matching sigeps103.F.

    Supports both solver convention:
      solid_update_single(params, sig_old, deps, epsp=epsp, dt=dt) -> (sig_new, epsp_new)
    and low-level Fortran array convention:
      solid_update_single(params, deps, sig_old, history) -> (sig_new, history_new, ssp)
    """
    if epsp is not None:
        sig_old = np.asarray(sig_or_deps, dtype=np.float64)
        deps = np.asarray(deps_or_sig, dtype=np.float64)
        temp_init = kwargs.get("temp", params.t0)
        hist = np.array([float(epsp), float(temp_init)], dtype=np.float64)
        sig_new, hist_new, ssp = _solid_update_single_core(
            params, deps, sig_old, hist, rho=rho, rho0=rho0, off=off, pnew=pnew, psh=psh, dt=dt, time=time
        )
        return sig_new, float(hist_new[0])

    if history is not None:
        deps = np.asarray(sig_or_deps, dtype=np.float64)
        sig_old = np.asarray(deps_or_sig, dtype=np.float64)
        hist = np.asarray(history, dtype=np.float64)
        return _solid_update_single_core(
            params, deps, sig_old, hist, rho=rho, rho0=rho0, off=off, pnew=pnew, psh=psh, dt=dt, time=time
        )

    sig_old = np.asarray(sig_or_deps, dtype=np.float64)
    deps = np.asarray(deps_or_sig, dtype=np.float64)
    hist = np.array([0.0, params.t0], dtype=np.float64)
    return _solid_update_single_core(
        params, deps, sig_old, hist, rho=rho, rho0=rho0, off=off, pnew=pnew, psh=psh, dt=dt, time=time
    )


def solid_update_array(
    params: HenselSpittelParams,
    deps: np.ndarray,
    sig_old: np.ndarray,
    history: np.ndarray,
    rho: Optional[np.ndarray] = None,
    rho0: Optional[np.ndarray] = None,
    off: Optional[np.ndarray] = None,
    pnew: Optional[np.ndarray] = None,
    psh: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    time: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized 3D solid continuum constitutive update matching sigeps103.F."""
    deps = np.asarray(deps, dtype=np.float64)
    sig_old = np.asarray(sig_old, dtype=np.float64)
    history = np.asarray(history, dtype=np.float64)
    nel = deps.shape[0]

    if off is None:
        off_arr = np.ones(nel, dtype=np.float64)
    else:
        off_arr = np.asarray(off, dtype=np.float64)

    if psh is None:
        psh_arr = np.zeros(nel, dtype=np.float64)
    elif isinstance(psh, (int, float)):
        psh_arr = np.full(nel, float(psh), dtype=np.float64)
    else:
        psh_arr = np.asarray(psh, dtype=np.float64)

    rho_ref_arr = np.full(nel, params.rho, dtype=np.float64)
    if rho0 is not None:
        rho0_in = np.asarray(rho0, dtype=np.float64)
        valid = (rho0_in > 0.0)
        rho_ref_arr[valid] = rho0_in[valid]
    zero_rho = (rho_ref_arr <= 0.0)
    if np.any(zero_rho):
        rho_ref_arr[zero_rho] = 1.0

    g = params.g
    g2 = 2.0 * g
    g3 = 3.0 * g
    bulk = params.bulk

    # Hydrostatic pressure and mean strain increment (sigeps103.F:107-109)
    pold = -(sig_old[:, 0] + sig_old[:, 1] + sig_old[:, 2]) / 3.0
    davg = (deps[:, 0] + deps[:, 1] + deps[:, 2]) / 3.0
    pla = history[:, 0]
    temp_k = history[:, 1].copy()
    uninit_temp = (temp_k <= 0.0)
    if np.any(uninit_temp):
        temp_k[uninit_temp] = params.t0

    eps = params.eps0 + pla

    # Trial deviatoric stress (sigeps103.F:114-121)
    s1 = sig_old[:, 0] + pold + g2 * (deps[:, 0] - davg)
    s2 = sig_old[:, 1] + pold + g2 * (deps[:, 1] - davg)
    s3 = sig_old[:, 2] + pold + g2 * (deps[:, 2] - davg)
    s4 = sig_old[:, 3] + g * deps[:, 3]
    s5 = sig_old[:, 4] + g * deps[:, 4]
    s6 = sig_old[:, 5] + g * deps[:, 5]

    # Sound speed (sigeps103.F:126-127)
    dpdm = bulk + (4.0 / 3.0) * g
    ssp = np.sqrt(np.maximum(0.0, dpdm) / rho_ref_arr)

    # Von Mises equivalent stress
    j2 = 0.5 * (s1 * s1 + s2 * s2 + s3 * s3) + s4 * s4 + s5 * s5 + s6 * s6
    svm = np.sqrt(np.maximum(0.0, 3.0 * j2))

    # Temperature in Celsius (sigeps103.F:139-143)
    theta = temp_k - _T0K

    # Equivalent strain rate
    if dt > 0.0:
        de_dev_xx = deps[:, 0] - davg
        de_dev_yy = deps[:, 1] - davg
        de_dev_zz = deps[:, 2] - davg
        de_dev_xy = 0.5 * deps[:, 3]
        de_dev_yz = 0.5 * deps[:, 4]
        de_dev_zx = 0.5 * deps[:, 5]
        de_j2 = 0.5 * (de_dev_xx ** 2 + de_dev_yy ** 2 + de_dev_zz ** 2) + de_dev_xy ** 2 + de_dev_yz ** 2 + de_dev_zx ** 2
        epsd = np.sqrt(np.maximum(0.0, 2.0 / 3.0 * (2.0 * de_j2))) / dt
    else:
        epsd = np.zeros(nel, dtype=np.float64)

    # Yield stress vectorization
    # Hardening factor
    yld_h = np.ones(nel, dtype=np.float64)
    pos_eps = (eps > 0.0)
    if params.m2 != 0.0 and np.any(pos_eps):
        yld_h[pos_eps] *= (eps[pos_eps] ** params.m2)
    if params.m4 != 0.0 and np.any(pos_eps):
        yld_h[pos_eps] *= np.exp(np.clip(params.m4 / eps[pos_eps], -700.0, 700.0))
    if params.m7 != 0.0 and np.any(pos_eps):
        yld_h[pos_eps] *= np.exp(np.clip(params.m7 * eps[pos_eps], -700.0, 700.0))

    # Strain rate factor
    if params.m3 != 0.0:
        rate = epsd * params.time_fac
        yld_sr = np.where(rate > 0.0, rate ** params.m3, 0.0)
    else:
        yld_sr = np.ones(nel, dtype=np.float64)

    # Temperature factor
    if params.m1 != 0.0 and params.m5 != 0.0:
        exp1 = np.exp(np.clip(theta * params.m1, -700.0, 700.0))
        pow5 = np.where((1.0 + eps) > 0.0, (1.0 + eps) ** (theta * params.m5), 1.0)
        yld_t = exp1 * pow5
    elif params.m1 != 0.0:
        yld_t = np.exp(np.clip(theta * params.m1, -700.0, 700.0))
    elif params.m5 != 0.0:
        yld_t = np.where((1.0 + eps) > 0.0, (1.0 + eps) ** (theta * params.m5), 1.0)
    else:
        yld_t = np.ones(nel, dtype=np.float64)

    yld = params.a0 * yld_h * yld_sr * yld_t

    # Hardening modulus (sigeps103.F:201-206)
    hm = params.m7 * yld
    if np.any(pos_eps):
        hm[pos_eps] += yld[pos_eps] * (params.m2 - params.m4 / eps[pos_eps]) / eps[pos_eps]

    # Radial return projection (sigeps103.F:210-225)
    svm_safe = np.maximum(svm, _EM20)
    ratio = np.minimum(1.0, yld / svm_safe)

    denom_g3_hm = np.maximum(g3 + hm, _EM20)
    dpla = (1.0 - ratio) * svm / denom_g3_hm

    yld_actual = np.maximum(0.0, yld + dpla * hm)
    ratio_actual = np.minimum(1.0, yld_actual / svm_safe)

    s1 *= ratio_actual
    s2 *= ratio_actual
    s3 *= ratio_actual
    s4 *= ratio_actual
    s5 *= ratio_actual
    s6 *= ratio_actual

    pla_new = pla + dpla
    if params.rcp > 0.0 and params.rcp < 1.0e29:
        delta_temp = params.eta * yld_actual * dpla / params.rcp
    else:
        delta_temp = 0.0
    temp_new = temp_k + delta_temp

    # Hydrostatic pressure
    if pnew is None:
        pnew_arr = pold - 3.0 * bulk * davg
    else:
        pnew_arr = np.asarray(pnew, dtype=np.float64)

    pnew_arr = np.maximum(params.pmin, pnew_arr)
    ptot_arr = pnew_arr + psh_arr

    sig_new = np.empty((nel, 6), dtype=np.float64)
    sig_new[:, 0] = s1 * off_arr - ptot_arr
    sig_new[:, 1] = s2 * off_arr - ptot_arr
    sig_new[:, 2] = s3 * off_arr - ptot_arr
    sig_new[:, 3] = s4 * off_arr
    sig_new[:, 4] = s5 * off_arr
    sig_new[:, 5] = s6 * off_arr

    history_new = np.empty((nel, 2), dtype=np.float64)
    history_new[:, 0] = pla_new
    history_new[:, 1] = temp_new

    return sig_new, history_new, ssp


def solid_update(
    mat: Any = None,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    eps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    ismstr: int = 0,
    *,
    epsp: Optional[np.ndarray] = None,
    return_tuple: bool = True,
    **kwargs: Any,
) -> Any:
    """3D solid continuum Hensel-Spittel stress update.

    Supports both standard solver group call:
      solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
    and direct array call:
      solid_update(params, deps, sig_old, history=history)
    """
    # Direct array call
    if isinstance(mat, HenselSpittelParams) and sig is not None and deps is not None and "history" in kwargs:
        deps_in = np.asarray(sig, dtype=np.float64)
        sig_in = np.asarray(deps, dtype=np.float64)
        hist_in = np.asarray(kwargs["history"], dtype=np.float64)
        return solid_update_array(
            mat,
            deps_in,
            sig_in,
            hist_in,
            rho=kwargs.get("rho"),
            rho0=kwargs.get("rho0"),
            off=kwargs.get("off"),
            pnew=kwargs.get("pnew"),
            psh=kwargs.get("psh", 0.0),
            dt=dt,
            time=kwargs.get("time", 0.0),
        )

    # Standard element / materials.solid_update call:
    params = mat if isinstance(mat, HenselSpittelParams) else build_law103(mat, **kwargs)

    if sig is None:
        sig = np.zeros(6, dtype=np.float64)
    is_1d = (sig.ndim == 1)

    sig_arr = sig[np.newaxis, :] if is_1d else sig
    deps_arr = deps[np.newaxis, :] if (deps is not None and is_1d) else (deps if deps is not None else np.zeros_like(sig_arr))
    nel = len(sig_arr)
    epsp_arr = epsp[np.newaxis] if (epsp is not None and is_1d) else (epsp if epsp is not None else np.zeros(nel))

    # Retrieve history variables from extra
    hist = None
    if extra is not None:
        for k in ("uvar103", "uvar", "history"):
            if k in extra and extra[k] is not None:
                hist = extra[k]
                break

    if hist is None or len(hist) == 0:
        hist = init_history(nel, t0=params.t0)
        hist[:, 0] = epsp_arr
    elif hist.ndim == 1:
        hist = hist[np.newaxis, :]

    rho_arr = extra.get("rho") if extra else None
    rho0_arr = extra.get("rho0") if extra else None
    off_arr = extra.get("off") if extra else None
    pnew_arr = extra.get("pnew") if extra else None
    psh_arr = extra.get("psh", 0.0) if extra else 0.0
    time_val = extra.get("time", 0.0) if extra else 0.0

    sig_new, hist_new, ssp = solid_update_array(
        params,
        deps_arr,
        sig_arr,
        hist,
        rho=rho_arr,
        rho0=rho0_arr,
        off=off_arr,
        pnew=pnew_arr,
        psh=psh_arr,
        dt=dt,
        time=time_val,
    )

    if extra is not None:
        extra["uvar103"] = hist_new
        extra["uvar"] = hist_new
        extra["history"] = hist_new
        extra["temp"] = hist_new[:, 1]

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


def sound_speed(mat: Any = None, rho: Optional[Any] = None, **kwargs: Any) -> Any:
    """Acoustic sound speed for LAW103 (Hensel-Spittel)."""
    params = mat if isinstance(mat, HenselSpittelParams) else build_law103(mat, **kwargs)
    c_base = params.sound_speed
    if rho is None:
        return c_base

    rho_arr = np.asarray(rho, dtype=np.float64)
    if rho_arr.ndim == 0:
        r = float(rho_arr)
        return math.sqrt(max(0.0, params.bulk + (4.0 / 3.0) * params.g) / (r if r > 0.0 else 1.0))
    else:
        safe_r = np.where(rho_arr > 0.0, rho_arr, 1.0)
        return np.sqrt(np.maximum(0.0, params.bulk + (4.0 / 3.0) * params.g) / safe_r)


def solid_tangent(
    mat: Any = None,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Algorithmic elastoplastic tangent stiffness matrix for LAW103 (6x6)."""
    params = mat if isinstance(mat, HenselSpittelParams) else build_law103(mat, **kwargs)
    g = params.g
    k_bulk = params.bulk

    # Elastic stiffness tensor (Voigt: xx, yy, zz, xy, yz, zx)
    c11 = k_bulk + (4.0 / 3.0) * g
    c12 = k_bulk - (2.0 / 3.0) * g

    d_e = np.array([
        [c11, c12, c12, 0.0, 0.0, 0.0],
        [c12, c11, c12, 0.0, 0.0, 0.0],
        [c12, c12, c11, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, g,   0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, g,   0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, g  ],
    ], dtype=np.float64)

    if sig is None:
        return d_e

    sig_arr = np.asarray(sig, dtype=np.float64)
    if sig_arr.ndim == 2:
        sig_arr = sig_arr[0]

    p = -(sig_arr[0] + sig_arr[1] + sig_arr[2]) / 3.0
    s = np.array([
        sig_arr[0] + p,
        sig_arr[1] + p,
        sig_arr[2] + p,
        sig_arr[3],
        sig_arr[4],
        sig_arr[5],
    ], dtype=np.float64)

    j2 = 0.5 * (s[0] ** 2 + s[1] ** 2 + s[2] ** 2) + s[3] ** 2 + s[4] ** 2 + s[5] ** 2
    svm = math.sqrt(max(0.0, 3.0 * j2))
    if svm < 1.0e-12:
        return d_e

    epsp = 0.0
    temp_k = params.t0
    if extra is not None:
        hist = extra.get("uvar103") or extra.get("uvar") or extra.get("history")
        if hist is not None:
            hist_arr = np.asarray(hist, dtype=np.float64)
            if hist_arr.ndim == 2:
                epsp = float(hist_arr[0, 0])
                temp_k = float(hist_arr[0, 1])
            elif hist_arr.ndim == 1:
                epsp = float(hist_arr[0])
                if len(hist_arr) > 1:
                    temp_k = float(hist_arr[1])

    eps = params.eps0 + epsp
    theta = temp_k - _T0K

    yld = compute_hensel_spittel_flow_stress(
        params.a0, params.m1, params.m2, params.m3, params.m4, params.m5, params.m7,
        eps, eps_dot=0.0, theta=theta, time_fac=params.time_fac
    )

    if svm <= yld:
        return d_e

    hm = compute_hensel_spittel_hardening_modulus(
        params.a0, params.m1, params.m2, params.m3, params.m4, params.m5, params.m7,
        eps, eps_dot=0.0, theta=theta, yld=yld, time_fac=params.time_fac
    )

    sqrt_2j2 = math.sqrt(max(1.0e-20, 2.0 * j2))
    n_vec = np.array([
        s[0] / sqrt_2j2,
        s[1] / sqrt_2j2,
        s[2] / sqrt_2j2,
        s[3] / sqrt_2j2 * math.sqrt(2.0),
        s[4] / sqrt_2j2 * math.sqrt(2.0),
        s[5] / sqrt_2j2 * math.sqrt(2.0),
    ], dtype=np.float64)

    denom = 3.0 * g + hm
    if denom <= 0.0:
        denom = 3.0 * g

    factor = (4.0 * g * g) / denom
    d_ep = d_e - factor * np.outer(n_vec, n_vec)
    return d_ep
