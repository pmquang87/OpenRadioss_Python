"""
/MAT/LAW124 (/MAT/CDPM2) — Concrete Damage Plasticity Model 2 (CDPM2).

Fortran origin:
  - Starter: ``starter/source/materials/mat/mat124/hm_read_mat124.F``
  - Engine:  ``engine/source/materials/mat/mat124/sigeps124.F``
  - CFG:     ``hm_cfg_files/config/CFG/radioss*/MAT/matl124_cdpm2.cfg``

Mathematical Formulation:
  - Elasticity: Young's modulus E, Poisson's ratio nu, shear G, bulk K, Lame lambda.
  - Invariants: Haigh-Westergaard coordinates (sigma_m, rho, theta) from stress tensor.
  - Yield Surface: Menetrey-Willam criterion F_p with Willam-Warnke elliptic function r(theta, e).
  - Hardening: Dual hardening variables q_h1(kappa) and q_h2(kappa) driven by plastic work.
  - Return Mapping: Semi-implicit cutting-plane Newton method with 6 iterations,
    including separate apex return when hydrostatic stress exceeds apex limit sigma_apex.
  - Damage: Independent tensile (omega_t) and compressive (omega_c) scalar damage variables
    with linear, bilinear, or exponential softening.
  - Unilateral Effect: Spectral decomposition sigma = (1 - omega_t)*sigma^+ + (1 - omega_c)*sigma^-,
    recovering 100% initial compressive stiffness upon crack closure under load reversal.
  - Regularisation: Crack-band regularization with characteristic element length h.
  - Rate Effects: Optional CEB-FIP model code strain-rate enhancement (IRATE=2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..model.entities import Material

# Numerical thresholds matching OpenRadioss engine/source/materials/mat/mat124/sigeps124.F
_EM20 = 1.0e-20
_EM08 = 1.0e-8
_ZEP999 = 0.999
_SQR3 = math.sqrt(3.0)
_SQR6 = math.sqrt(6.0)
_SQRT_3_OVER_2 = math.sqrt(1.5)
_THIRD = 1.0 / 3.0
_TWO_THIRD = 2.0 / 3.0
_FOUR_OVER_3 = 4.0 / 3.0


@dataclass
class Law124Params:
    """Parameters for /MAT/LAW124 (/MAT/CDPM2) concrete damage plasticity model.

    Faithful to ``hm_read_mat124.F``.
    """
    E: float = 30000.0          # Young's modulus
    nu: float = 0.2             # Poisson's ratio (0 <= nu < 0.5)
    rho0: float = 2.4e-6        # Initial density (e.g. g/mm3 or t/mm3)
    fc: float = 30.0            # Uniaxial compressive strength (> 0)
    ft: float = 3.0             # Uniaxial tensile strength (> 0)
    ecc: float = 0.0            # Eccentricity (if 0, computed from fc, ft)
    qh0: float = 0.3            # Initial hardening parameter (default 0.3)
    hp: float = 0.0             # Hardening modulus (default 0.0)
    ah: float = 0.08            # Hardening ductility parameter 1 (default 0.08)
    bh: float = 0.003           # Hardening ductility parameter 2 (default 0.003)
    ch: float = 2.0             # Hardening ductility parameter 3 (default 2.0)
    dh: float = 1.0e-6          # Hardening ductility parameter 4 (default 1e-6)
    as_: float = 15.0           # Damage ductility measure (default 15.0)
    bs: float = 1.0             # Damage ductility parameter (default 1.0)
    df: float = 0.85            # Constant dilation parameter (default 0.85)
    dflag: int = 1              # Damage flag (1=standard spectral, 2=isotropic, 3=mult, 4=none)
    dtype: int = 2              # Tensile damage type (1=linear, 2=bilinear, 3=exponential)
    ireg: int = 2               # Regularization flag (1=no reg h=1, 2=crack band h=Le)
    wf: float = 0.1             # First displacement threshold wf
    wf1: float = 0.0            # Second displacement threshold wf1 (if 0, 0.15*wf)
    ft1: float = 0.0            # Second tensile strength ft1 (if 0, 0.3*ft)
    efc: float = 1.0e-4         # Compressive inelastic strain threshold (default 1e-4)
    irate: int = 1              # Strain rate effect flag (1=none, 2=CEB-FIP rate model)
    fcut: float = 0.0           # Strain rate filter frequency
    idel: int = 1               # Element deletion flag (1=no delete, 2=delete at D=0.999)
    # Rate effect parameters
    fc0: float = 10.0           # Reference compressive strength (10 MPa)
    epst0: float = 30.0e-6      # Reference tensile strain rate
    epstmax: float = 1.0        # Max tensile strain rate
    deltas: float = 0.0         # Rate parameter delta_s
    betas: float = 0.0          # Rate parameter beta_s
    epsc0: float = 30.0e-6      # Reference compressive strain rate
    epscmax: float = 30.0       # Max compressive strain rate
    alphas: float = 0.0         # Rate parameter alpha_s
    gammas: float = 0.0         # Rate parameter gamma_s
    # Convenience fracture energy inputs (mapped to wf and efc if provided)
    Gf: Optional[float] = None  # Tensile fracture energy
    Gc: Optional[float] = None  # Compressive fracture energy

    # Derived quantities computed in __post_init__
    G: float = field(init=False, default=0.0)
    G2: float = field(init=False, default=0.0)
    lam: float = field(init=False, default=0.0)
    bulk: float = field(init=False, default=0.0)
    m0: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.nu < 0.0 or self.nu >= 0.5:
            raise ValueError(f"Poisson's ratio must satisfy 0 <= nu < 0.5, got {self.nu} (ANCMSG 49)")
        if self.E <= 0.0:
            raise ValueError(f"Young's modulus E must be positive, got {self.E}")
        if self.fc <= 0.0:
            raise ValueError(f"Compressive strength fc must be positive, got {self.fc}")
        if self.ft <= 0.0:
            raise ValueError(f"Tensile strength ft must be positive, got {self.ft}")

        # Elastic moduli (hm_read_mat124.F lines 140-143)
        self.G2 = self.E / (1.0 + self.nu)
        self.G = 0.5 * self.G2
        self.lam = self.G2 * self.nu / (1.0 - 2.0 * self.nu)
        self.bulk = self.E / (3.0 * (1.0 - 2.0 * self.nu))

        # Defaults for bilinear softening (lines 145-151)
        # Check if Gf was supplied
        if self.Gf is not None and self.Gf > 0.0:
            if self.dtype == 1:
                self.wf = 2.0 * self.Gf / self.ft
            elif self.dtype == 2:
                self.wf = self.Gf / (0.225 * self.ft)
            elif self.dtype == 3:
                self.wf = self.Gf / self.ft

        if self.wf1 == 0.0:
            self.wf1 = 0.15 * self.wf
        if self.ft1 == 0.0:
            self.ft1 = 0.3 * self.ft

        # Hardening defaults (lines 153-168)
        if self.qh0 == 0.0:
            self.qh0 = 0.3
        if self.ah == 0.0:
            self.ah = 0.08
        if self.bh == 0.0:
            self.bh = 0.003
        if self.ch == 0.0:
            self.ch = 2.0
        if self.dh == 0.0:
            self.dh = 1.0e-6

        # Dilation and inelastic strain defaults (lines 170-183)
        if self.df == 0.0:
            self.df = 0.85
        if self.efc == 0.0:
            self.efc = 1.0e-4
        if self.as_ == 0.0:
            self.as_ = 15.0
        if self.bs == 0.0:
            self.bs = 1.0

        # Eccentricity default (lines 185-188)
        if self.ecc <= 0.0:
            fc116 = 1.16 * self.fc
            epsi = self.ft * (fc116**2 - self.fc**2) / (fc116 * (self.fc**2 - self.ft**2))
            self.ecc = (1.0 + epsi) / (2.0 - epsi)

        # Friction parameter m0 (line 190)
        self.m0 = 3.0 * ((self.fc**2 - self.ft**2) / (self.fc * self.ft)) * (self.ecc / (self.ecc + 1.0))

        # Flag bounds (lines 192-204)
        if self.idel <= 0:
            self.idel = 1
        self.idel = min(max(1, int(self.idel)), 2)

        if self.dflag <= 0:
            self.dflag = 1
        self.dflag = min(max(1, int(self.dflag)), 4)

        if self.dtype <= 0:
            self.dtype = 2
        self.dtype = min(max(1, int(self.dtype)), 3)

        if self.ireg <= 0:
            self.ireg = 2
        self.ireg = min(max(1, int(self.ireg)), 2)

        # Rate parameters (lines 206-220)
        if self.irate <= 0:
            self.irate = 1
        self.irate = min(max(1, int(self.irate)), 2)
        if self.irate > 1:
            if self.fc0 <= 0.0:
                self.fc0 = 10.0
            self.deltas = 1.0 / (1.0 + 8.0 * (self.fc / self.fc0))
            self.betas = math.exp(6.0 * self.deltas - 2.0)
            self.alphas = 1.0 / (5.0 + 9.0 * (self.fc / self.fc0))
            self.gammas = math.exp(6.156 * self.alphas - 2.0)
            if self.fcut == 0.0:
                self.fcut = 10000.0


def compute_haigh_westergaard(sig: np.ndarray) -> Tuple[float, float, float, float, float, np.ndarray, float]:
    """Compute Haigh-Westergaard coordinates and invariants from stress tensor.

    Args:
        sig: [s_xx, s_yy, s_zz, s_xy, s_yz, s_zx] (Voigt engineering 6-vector)

    Returns:
        sigma_m: Mean hydrostatic stress (tr(sig)/3)
        rho: Deviatoric radius sqrt(2*J2)
        theta: Lode angle in radians [0, pi/3]
        cos_theta: cos(theta)
        sin_theta: sin(theta)
        s: Deviatoric stress vector (6,)
        J2: Second deviatoric invariant
    """
    tr_sig = sig[0] + sig[1] + sig[2]
    sigma_m = tr_sig * _THIRD
    s = np.array([
        sig[0] - sigma_m,
        sig[1] - sigma_m,
        sig[2] - sigma_m,
        sig[3],
        sig[4],
        sig[5],
    ], dtype=np.float64)

    j2 = 0.5 * (s[0]**2 + s[1]**2 + s[2]**2) + s[3]**2 + s[4]**2 + s[5]**2
    j2 = max(j2, _EM20)
    rho = math.sqrt(2.0 * j2)

    j3 = (s[0] * s[1] * s[2] + 2.0 * s[3] * s[5] * s[4]
          - s[0] * s[4]**2 - s[2] * s[3]**2 - s[1] * s[5]**2)

    cos3th = (1.5 * _SQR3) * j3 / (j2**1.5)
    cos3th = max(-1.0, min(1.0, cos3th))
    theta = _THIRD * math.acos(cos3th)
    cos_th = math.cos(theta)
    sin_th = math.sin(theta)

    return sigma_m, rho, theta, cos_th, sin_th, s, j2


def r_willam_warnke(cos_th: float, ecc: float) -> Tuple[float, float]:
    """Evaluate Willam-Warnke elliptic function r(cos(theta), e) and its derivative dr/d(cos(theta)).

    Faithful to ``sigeps124.F`` lines 327-330 and 493-501.
    """
    e2 = ecc**2
    one_minus_e2 = 1.0 - e2
    two_e_minus_one = 2.0 * ecc - 1.0

    u = 4.0 * one_minus_e2 * cos_th**2 + two_e_minus_one**2
    u_prim = 8.0 * one_minus_e2 * cos_th

    rad = max(4.0 * one_minus_e2 * cos_th**2 + 5.0 * e2 - 4.0 * ecc, 0.0)
    sqrt_rad = math.sqrt(rad)
    v = 2.0 * one_minus_e2 * cos_th + two_e_minus_one * sqrt_rad
    v = max(v, _EM20)

    # Derivative dV/d(cos(theta))
    if sqrt_rad > _EM20:
        v_prim = 2.0 * one_minus_e2 + two_e_minus_one * (8.0 * one_minus_e2 * cos_th) / (2.0 * sqrt_rad)
    else:
        v_prim = 2.0 * one_minus_e2

    r = u / v
    dr_dcosth = (u_prim * v - u * v_prim) / (v**2)
    return r, dr_dcosth


def eval_hardening(kappa: float, qh0: float, hp: float) -> Tuple[float, float, float, float]:
    """Evaluate hardening functions q_h1, q_h2 and their kappa derivatives.

    Faithful to ``sigeps124.F`` lines 333-344, 391-395, and 576-586.
    """
    if kappa < 0.0:
        qh1 = qh0
        qh2 = 1.0
        dqh1 = (1.0 - qh0) * 3.0 - hp * 2.0
        dqh2 = 0.0
    elif kappa < 1.0:
        kap2 = kappa**2
        kap3 = kappa * kap2
        qh1 = qh0 + (1.0 - qh0) * (kap3 - 3.0 * kap2 + 3.0 * kappa) - hp * (kap3 - 3.0 * kap2 + 2.0 * kappa)
        qh2 = 1.0
        dqh1 = (1.0 - qh0) * (3.0 * kap2 - 6.0 * kappa + 3.0) - hp * (3.0 * kap2 - 6.0 * kappa + 2.0)
        dqh2 = 0.0
    else:
        qh1 = 1.0
        qh2 = 1.0 + (kappa - 1.0) * hp
        dqh1 = 0.0
        dqh2 = hp
    return qh1, qh2, dqh1, dqh2


def eval_ductility_xsih(sigma_m: float, fc: float, ah: float, bh: float, ch: float, dh: float) -> float:
    """Compute hardening ductility measure xsih from hydrostatic stress.

    Faithful to ``sigeps124.F`` lines 398-405.
    """
    rh = -(sigma_m / fc) - _THIRD
    if rh >= 0.0:
        xsih = ah - (ah - bh) * math.exp(-rh / ch)
    else:
        eh = bh - dh
        fh = (bh - dh) * ch / (ah - bh)
        xsih = eh * math.exp(rh / fh) + dh
    return max(xsih, _EM20)


def spectral_decomposition(sig_undamaged: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform spectral decomposition of 3D symmetric stress tensor into positive and negative parts.

    Returns:
        sig_pos: Tensile stress tensor (6,)
        sig_neg: Compressive stress tensor (6,)
        evals: Principal stresses [sigma_1, sigma_2, sigma_3]
        evecs: Matrix of eigenvectors (3, 3) where column j is eigenvector j
    """
    mat = np.array([
        [sig_undamaged[0], sig_undamaged[3], sig_undamaged[5]],
        [sig_undamaged[3], sig_undamaged[1], sig_undamaged[4]],
        [sig_undamaged[5], sig_undamaged[4], sig_undamaged[2]],
    ], dtype=np.float64)

    evals, evecs = np.linalg.eigh(mat)

    sig_pos_mat = np.zeros((3, 3), dtype=np.float64)
    sig_neg_mat = np.zeros((3, 3), dtype=np.float64)

    for j in range(3):
        val = evals[j]
        v = evecs[:, j]
        vv = np.outer(v, v)
        if val > 0.0:
            sig_pos_mat += val * vv
        else:
            sig_neg_mat += val * vv

    sig_pos = np.array([
        sig_pos_mat[0, 0],
        sig_pos_mat[1, 1],
        sig_pos_mat[2, 2],
        sig_pos_mat[0, 1],
        sig_pos_mat[1, 2],
        sig_pos_mat[0, 2],
    ], dtype=np.float64)

    sig_neg = np.array([
        sig_neg_mat[0, 0],
        sig_neg_mat[1, 1],
        sig_neg_mat[2, 2],
        sig_neg_mat[0, 1],
        sig_neg_mat[1, 2],
        sig_neg_mat[0, 2],
    ], dtype=np.float64)

    return sig_pos, sig_neg, evals, evecs


def solid_step(
    params: Law124Params,
    sig_old: np.ndarray,
    deps: np.ndarray,
    uvar: np.ndarray,
    dmg: np.ndarray,
    le: float = 1.0,
    eps_dot: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Execute single-element stress update for /MAT/LAW124 (CDPM2).

    Args:
        params: Constitutive parameters
        sig_old: Stress tensor at start of cycle [xx, yy, zz, xy, yz, zx] (6,)
        deps: Total strain increment [deps_xx, deps_yy, deps_zz, deps_xy, deps_yz, deps_zx]
              with engineering shear strains deps_xy = 2*deps_12
        uvar: Internal variables (16,)
        dmg: Damage state (3,) [D_scalar, omega_t, omega_c]
        le: Characteristic element length
        eps_dot: Effective strain rate for rate effect

    Returns:
        sig_new: Updated degraded stress tensor (6,)
        uvar_new: Updated internal variables (16,)
        dmg_new: Updated damage array (3,)
        dpla: Plastic strain increment scalar
        c: Acoustic sound speed
    """
    uvar_out = np.copy(uvar)
    dmg_out = np.copy(dmg)

    # Characteristic length initialization
    if params.ireg > 1:
        if uvar_out[15] <= 0.0:
            uvar_out[15] = le if le > 0.0 else 1.0
        h = uvar_out[15]
    else:
        h = 1.0

    # Elastic trial stress computation
    ldav_deps = (deps[0] + deps[1] + deps[2]) * params.lam
    sig_trial = np.array([
        sig_old[0] + deps[0] * params.G2 + ldav_deps,
        sig_old[1] + deps[1] * params.G2 + ldav_deps,
        sig_old[2] + deps[2] * params.G2 + ldav_deps,
        sig_old[3] + deps[3] * params.G,
        sig_old[4] + deps[4] * params.G,
        sig_old[5] + deps[5] * params.G,
    ], dtype=np.float64)

    # Dynamic strength scaling for strain rate (IRATE=2)
    fc = params.fc
    ft = params.ft
    ft1 = params.ft1

    if params.irate > 1 and eps_dot > 0.0:
        if eps_dot <= params.epst0:
            alpha_rt = 1.0
        elif eps_dot <= params.epstmax:
            alpha_rt = (eps_dot / params.epst0)**params.deltas
        else:
            alpha_rt = params.betas * (eps_dot / params.epst0)**_THIRD

        if eps_dot <= params.epsc0:
            alpha_rc = 1.0
        elif eps_dot <= params.epscmax:
            alpha_rc = (eps_dot / params.epsc0)**(1.026 * params.alphas)
        else:
            alpha_rc = params.gammas * (eps_dot / params.epsc0)**_THIRD

        mat_trial = np.array([
            [sig_trial[0], sig_trial[3], sig_trial[5]],
            [sig_trial[3], sig_trial[1], sig_trial[4]],
            [sig_trial[5], sig_trial[4], sig_trial[2]],
        ], dtype=np.float64)
        e_vals = np.linalg.eigvalsh(mat_trial)
        norm_p = max(float(np.sum(e_vals**2)), _EM20)
        alpha_c_rate = 0.0
        for val in e_vals:
            sig_pt = max(val, 0.0)
            sig_pc = min(val, 0.0)
            alpha_c_rate += sig_pc * (sig_pt + sig_pc) / norm_p

        arate = (1.0 - alpha_c_rate) * alpha_rt + alpha_c_rate * alpha_rc
        uvar_out[14] = arate
        fc = fc * arate
        ft = ft * arate
        ft1 = ft1 * arate

    # Invariants of trial stress
    sigma_m, rho, theta, cos_th, sin_th, s, j2 = compute_haigh_westergaard(sig_trial)
    rcos, dr_dcosth = r_willam_warnke(cos_th, params.ecc)

    kappa = float(uvar_out[0])
    qh1, qh2, dqh1_dkap, dqh2_dkap = eval_hardening(kappa, params.qh0, params.hp)

    # Apex stress and functions
    apex = qh2 * fc / params.m0
    fapex = sigma_m - apex

    cl = (rho * rcos / (_SQR6 * fc)) + sigma_m / fc
    bl = (sigma_m / fc) + rho / (_SQR6 * fc)
    al = (1.0 - qh1) * (bl**2) + _SQRT_3_OVER_2 * (rho / fc)
    fp = al**2 + params.m0 * (qh1**2) * qh2 * cl - (qh1**2) * (qh2**2)

    dp = np.zeros(6, dtype=np.float64)
    sign = np.copy(sig_trial)
    niter = 6

    if fapex > 0.0:
        # Apex return mapping with cutting plane (Newton method)
        for _ in range(niter):
            dqh2 = params.hp if kappa >= 1.0 else 0.0
            xsih = eval_ductility_xsih(sigma_m, fc, params.ah, params.bh, params.ch, params.dh)
            dfapex_dkap = dqh2 * (fc / params.m0) + 3.0 * params.bulk * xsih
            if dfapex_dkap == 0.0:
                break
            delta_kap = fapex / dfapex_dkap
            kappa += delta_kap
            sigma_m -= 3.0 * params.bulk * xsih * delta_kap
            qh2_new = 1.0 + (kappa - 1.0) * params.hp if kappa >= 1.0 else 1.0
            apex = qh2_new * fc / params.m0
            fapex = sigma_m - apex

        ds_xx = sign[0] - sigma_m
        ds_yy = sign[1] - sigma_m
        ds_zz = sign[2] - sigma_m
        dp[0] = (1.0 / params.E) * (ds_xx - params.nu * ds_yy - params.nu * ds_zz)
        dp[1] = (1.0 / params.E) * (-params.nu * ds_xx + ds_yy - params.nu * ds_zz)
        dp[2] = (1.0 / params.E) * (-params.nu * ds_xx - params.nu * ds_yy + ds_zz)
        dp[3] = (1.0 / params.G) * sign[3]
        dp[4] = (1.0 / params.G) * sign[4]
        dp[5] = (1.0 / params.G) * sign[5]

        uvar_out[1] += dp[0]
        uvar_out[2] += dp[1]
        uvar_out[3] += dp[2]
        uvar_out[4] += dp[3]
        uvar_out[5] += dp[4]
        uvar_out[6] += dp[5]

        sign[0] = sigma_m
        sign[1] = sigma_m
        sign[2] = sigma_m
        sign[3] = 0.0
        sign[4] = 0.0
        sign[5] = 0.0
        j2 = _EM20
        rho = math.sqrt(2.0 * j2)
        cl = sigma_m / fc

    elif fp > 0.0:
        # Regular return mapping with cutting plane (Newton method)
        for _ in range(niter):
            # 1a) Normal to yield surface
            dfp_drhob = (2.0 * al / fc) * (2.0 * ((1.0 - qh1) / _SQR6) * bl + _SQRT_3_OVER_2) + \
                        params.m0 * (qh1**2) * qh2 * rcos / (_SQR6 * fc)
            dfp_dsigm = (4.0 * al * bl * (1.0 - qh1) / fc) + params.m0 * (qh1**2) * qh2 / fc

            cos3th_val = (1.5 * _SQR3) * (s[0] * s[1] * s[2] + 2.0 * s[3] * s[5] * s[4]
                                          - s[0] * s[4]**2 - s[2] * s[3]**2 - s[1] * s[5]**2) / (j2**1.5)
            cos3th_val = max(-1.0, min(1.0, cos3th_val))
            denom_th = max(math.sqrt(max(1.0 - cos3th_val**2, 0.0)), _EM08)

            j3_val = (s[0] * s[1] * s[2] + 2.0 * s[3] * s[5] * s[4]
                      - s[0] * s[4]**2 - s[2] * s[3]**2 - s[1] * s[5]**2)
            dth_dj2 = 3.0 * _SQR3 * j3_val / (4.0 * (j2**2) * math.sqrt(j2) * denom_th)
            dth_dj3 = -_SQR3 / (2.0 * j2 * math.sqrt(j2) * denom_th)

            dfp_dth = params.m0 * (qh1**2) * qh2 * (rho / (_SQR6 * fc)) * dr_dcosth * (-sin_th)

            dj3_dsxx = _TWO_THIRD * (s[1] * s[2] - s[4]**2) - _THIRD * (s[0] * s[2] - s[5]**2) - _THIRD * (s[0] * s[1] - s[3]**2)
            dj3_dsyy = -_THIRD * (s[1] * s[2] - s[4]**2) + _TWO_THIRD * (s[0] * s[2] - s[5]**2) - _THIRD * (s[0] * s[1] - s[3]**2)
            dj3_dszz = -_THIRD * (s[1] * s[2] - s[4]**2) - _THIRD * (s[0] * s[2] - s[5]**2) + _TWO_THIRD * (s[0] * s[1] - s[3]**2)
            dj3_dsxy = 2.0 * (s[0] * s[3] + s[3] * s[1] + s[5] * s[4])
            dj3_dsyz = 2.0 * (s[3] * s[5] + s[1] * s[4] + s[4] * s[2])
            dj3_dszx = 2.0 * (s[0] * s[5] + s[3] * s[4] + s[5] * s[2])

            normxx = dfp_drhob * (s[0] / rho) + dfp_dsigm * _THIRD + dfp_dth * (dth_dj2 * s[0] + dth_dj3 * dj3_dsxx)
            normyy = dfp_drhob * (s[1] / rho) + dfp_dsigm * _THIRD + dfp_dth * (dth_dj2 * s[1] + dth_dj3 * dj3_dsyy)
            normzz = dfp_drhob * (s[2] / rho) + dfp_dsigm * _THIRD + dfp_dth * (dth_dj2 * s[2] + dth_dj3 * dj3_dszz)
            normxy = 2.0 * dfp_drhob * (s[3] / rho) + dfp_dth * (dth_dj2 * 2.0 * s[3] + dth_dj3 * dj3_dsxy)
            normyz = 2.0 * dfp_drhob * (s[4] / rho) + dfp_dth * (dth_dj2 * 2.0 * s[4] + dth_dj3 * dj3_dsyz)
            normzx = 2.0 * dfp_drhob * (s[5] / rho) + dfp_dth * (dth_dj2 * 2.0 * s[5] + dth_dj3 * dj3_dszx)

            # 1b) Normal to plastic potential G_p
            dgp_drhob = (2.0 * al / fc) * (2.0 * ((1.0 - qh1) / _SQR6) * bl + _SQRT_3_OVER_2) + \
                        (qh1**2) * params.m0 / (_SQR6 * fc)
            ag = (3.0 * ft * qh2 / fc) + params.m0 * 0.5
            denom_bg = max(
                math.log(max(ag, _EM20)) - math.log(max(2.0 * params.df - 1.0, _EM20))
                - math.log(max(3.0 * qh2 + params.m0 * 0.5, _EM20)) + math.log(max(params.df + 1.0, _EM20)),
                _EM20
            )
            bg = ((qh2 * _THIRD) * (1.0 + ft / fc)) / denom_bg
            dmg_dsigm = ag * math.exp(min(max((sigma_m - qh2 * ft * _THIRD) / (bg * fc), -50.0), 50.0))
            dgp_dsigm = (4.0 * al * bl * (1.0 - qh1) / fc) + ((qh1**2) / fc) * dmg_dsigm

            normpxx = dgp_drhob * (s[0] / rho) + dgp_dsigm * _THIRD
            normpyy = dgp_drhob * (s[1] / rho) + dgp_dsigm * _THIRD
            normpzz = dgp_drhob * (s[2] / rho) + dgp_dsigm * _THIRD
            normpxy = 2.0 * dgp_drhob * (s[3] / rho)
            normpyz = 2.0 * dgp_drhob * (s[4] / rho)
            normpzx = 2.0 * dgp_drhob * (s[5] / rho)

            # 2) Plastic multiplier denominator DFP_DLAMBDA
            tr_dgpds = normpxx + normpyy + normpzz
            dfpdsig2 = normxx * (normpxx * params.G2 + params.lam * tr_dgpds) + \
                       normyy * (normpyy * params.G2 + params.lam * tr_dgpds) + \
                       normzz * (normpzz * params.G2 + params.lam * tr_dgpds) + \
                       normxy * normpxy * params.G + \
                       normyz * normpyz * params.G + \
                       normzx * normpzx * params.G

            dfp_dqh1 = -2.0 * al * (bl**2) + 2.0 * qh1 * qh2 * (params.m0 * cl - qh2)
            dfp_dqh2 = (qh1**2) * (params.m0 * cl - 2.0 * qh2)
            dfp_dkap = min(dfp_dqh1 * dqh1_dkap + dfp_dqh2 * dqh2_dkap, 0.0)

            xsih = eval_ductility_xsih(sigma_m, fc, params.ah, params.bh, params.ch, params.dh)
            normgp = math.sqrt(_THIRD * (dgp_dsigm**2) + dgp_drhob**2)
            dkap_dlam = (normgp / xsih) * (2.0 * cos_th)**2

            dfp_dlam = -dfpdsig2 + dfp_dkap * dkap_dlam
            if abs(dfp_dlam) < _EM20:
                dfp_dlam = -_EM20 if dfp_dlam < 0 else _EM20

            dlam = -fp / dfp_dlam

            dpxx = dlam * normpxx
            dpyy = dlam * normpyy
            dpzz = dlam * normpzz
            dpxy = dlam * normpxy
            dpyz = dlam * normpyz
            dpzx = dlam * normpzx

            dp[0] += dpxx
            dp[1] += dpyy
            dp[2] += dpzz
            dp[3] += dpxy
            dp[4] += dpyz
            dp[5] += dpzx

            uvar_out[1] += dpxx
            uvar_out[2] += dpyy
            uvar_out[3] += dpzz
            uvar_out[4] += dpxy
            uvar_out[5] += dpyz
            uvar_out[6] += dpzx

            kappa = max(kappa + dkap_dlam * dlam, 0.0)

            tr_dep = dpxx + dpyy + dpzz
            ldav_dep = tr_dep * params.lam
            sign[0] -= (dpxx * params.G2 + ldav_dep)
            sign[1] -= (dpyy * params.G2 + ldav_dep)
            sign[2] -= (dpzz * params.G2 + ldav_dep)
            sign[3] -= dpxy * params.G
            sign[4] -= dpyz * params.G
            sign[5] -= dpzx * params.G

            sigma_m, rho, theta, cos_th, sin_th, s, j2 = compute_haigh_westergaard(sign)
            rcos, dr_dcosth = r_willam_warnke(cos_th, params.ecc)
            qh1, qh2, dqh1_dkap, dqh2_dkap = eval_hardening(kappa, params.qh0, params.hp)
            cl = (rho * rcos / (_SQR6 * fc)) + sigma_m / fc
            bl = (sigma_m / fc) + rho / (_SQR6 * fc)
            al = (1.0 - qh1) * (bl**2) + _SQRT_3_OVER_2 * (rho / fc)
            fp = al**2 + params.m0 * (qh1**2) * qh2 * cl - (qh1**2) * (qh2**2)

    uvar_out[0] = kappa

    dpla = math.sqrt(dp[0]**2 + dp[1]**2 + dp[2]**2 + 0.5 * (dp[3]**2 + dp[4]**2 + dp[5]**2))

    sig_pos, sig_neg, evals, evecs = spectral_decomposition(sign)

    omega_t = float(dmg_out[1])
    omega_c = float(dmg_out[2])

    if params.dflag < 4:
        eps0 = ft / params.E
        term_eps = eps0 * params.m0 * 0.5 * cl
        eps_eq = term_eps + math.sqrt(max(term_eps**2 + 1.5 * ((eps0 * rho / fc)**2), 0.0))
        uvar_out[7] = eps_eq

        if sigma_m <= 0.0:
            rs = -_SQR6 * sigma_m / max(rho, _EM20)
        else:
            rs = 0.0
        xsis = 1.0 + (params.as_ - 1.0) * (rs**params.bs)
        xsis = max(xsis, _EM20)

        norm_sigp = max(float(np.sum(evals**2)), _EM20)
        alpha_c = 0.0
        for val in evals:
            sig_pt = max(val, 0.0)
            sig_pc = min(val, 0.0)
            alpha_c += sig_pc * (sig_pt + sig_pc) / norm_sigp

        kapdt = float(uvar_out[8])
        kapdt1 = float(uvar_out[9])
        kapdt2 = float(uvar_out[10])
        kapdc = float(uvar_out[11])
        kapdc1 = float(uvar_out[12])
        kapdc2 = float(uvar_out[13])

        if eps_eq >= (eps0 - _EM08):
            kapdt2 += max(eps_eq - kapdt, 0.0) / xsis
            kapdt1 += dpla / xsis
            kapdt = max(eps_eq, kapdt)
            eps_inel_t = kapdt1 + omega_t * kapdt2

            if params.dtype == 1:
                num = params.E * kapdt * params.wf - ft * params.wf + ft * kapdt1 * h
                den = max(params.E * kapdt * params.wf - ft * h * kapdt2, _EM20)
                omega_t = num / den
            elif params.dtype == 2:
                h_eps = h * eps_inel_t
                if 0.0 <= h_eps < params.wf1:
                    num = params.E * kapdt * params.wf1 - ft * params.wf1 - (ft1 - ft) * kapdt1 * h
                    den = max(params.E * kapdt * params.wf1 + (ft1 - ft) * h * kapdt2, _EM20)
                    omega_t = num / den
                elif params.wf1 <= h_eps < params.wf:
                    dwf = params.wf - params.wf1
                    num = params.E * kapdt * dwf - ft1 * dwf + ft1 * h * kapdt1 - ft1 * params.wf1
                    den = max(params.E * kapdt * dwf - ft1 * h * kapdt2, _EM20)
                    omega_t = num / den
                else:
                    omega_t = _ZEP999
            elif params.dtype == 3:
                omega_t = 1.0 - math.exp(-max(h * eps_inel_t / params.wf, 0.0))

            omega_t = max(omega_t, float(dmg_out[1]))
            omega_t = min(max(omega_t, 0.0), _ZEP999)

            kapdc2 += alpha_c * max(eps_eq - kapdc, 0.0) / xsis
            betac = (ft * qh2 * math.sqrt(_TWO_THIRD)) / (max(rho * math.sqrt(1.0 + 2.0 * (params.df**2)), _EM20))
            kapdc1 += alpha_c * betac * dpla / xsis
            kapdc = max(alpha_c * eps_eq, kapdc)
            eps_inel_c = kapdc1 + omega_c * kapdc2
            omega_c = 1.0 - math.exp(-max(eps_inel_c / params.efc, 0.0))
            omega_c = max(omega_c, float(dmg_out[2]))
            omega_c = min(max(omega_c, 0.0), _ZEP999)

        uvar_out[8] = kapdt
        uvar_out[9] = kapdt1
        uvar_out[10] = kapdt2
        uvar_out[11] = kapdc
        uvar_out[12] = kapdc1
        uvar_out[13] = kapdc2

    dmg_out[1] = omega_t
    dmg_out[2] = omega_c

    sig_final = np.copy(sign)
    if params.dflag == 1:
        dmg_out[0] = min(min(omega_t, omega_c), _ZEP999)
        sig_final = (1.0 - omega_t) * sig_pos + (1.0 - omega_c) * sig_neg
    elif params.dflag == 2:
        dmg_out[0] = min(omega_t, _ZEP999)
        sig_final = (1.0 - dmg_out[0]) * sign
    elif params.dflag == 3:
        dmg_out[0] = min(1.0 - (1.0 - omega_t) * (1.0 - omega_c), _ZEP999)
        sig_final = (1.0 - dmg_out[0]) * sign
    else:
        dmg_out[0] = 0.0
        sig_final = sign

    if dmg_out[0] >= _ZEP999 and params.idel > 1:
        sig_final[:] = 0.0
        dmg_out[0] = 1.0

    c = math.sqrt((params.bulk + _FOUR_OVER_3 * params.G) / params.rho0)

    return sig_final, uvar_out, dmg_out, dpla, c


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Solid constitutive update vectorized over element group."""
    params = resolve(mat)
    n = sig.shape[0]

    if extra is None:
        extra = {}

    uvar = extra.get("uvar")
    if uvar is None or uvar.shape != (n, 16):
        uvar = np.zeros((n, 16), dtype=np.float64)
        extra["uvar"] = uvar

    dmg = extra.get("dmg")
    if dmg is None or dmg.shape != (n, 3):
        dmg = np.zeros((n, 3), dtype=np.float64)
        extra["dmg"] = dmg

    le_arr = extra.get("le")
    if le_arr is None or len(le_arr) != n:
        le_arr = np.ones(n, dtype=np.float64)

    epsp_out = np.zeros(n, dtype=np.float64) if epsp is None else np.copy(epsp)
    sig_out = np.zeros_like(sig)
    soundsp = np.zeros(n, dtype=np.float64)

    eps_dot_arr = np.zeros(n, dtype=np.float64)
    if dt > 0.0 and params.irate > 1:
        for i in range(n):
            de = deps[i]
            eps_dot_arr[i] = math.sqrt(
                de[0]**2 + de[1]**2 + de[2]**2 + 0.5 * (de[3]**2 + de[4]**2 + de[5]**2)
            ) / dt

    for i in range(n):
        s_i, uv_i, d_i, dpla, c_i = solid_step(
            params=params,
            sig_old=sig[i],
            deps=deps[i],
            uvar=uvar[i],
            dmg=dmg[i],
            le=float(le_arr[i]),
            eps_dot=float(eps_dot_arr[i]),
        )
        sig_out[i] = s_i
        uvar[i] = uv_i
        dmg[i] = d_i
        epsp_out[i] += dpla
        soundsp[i] = c_i

    return sig_out, epsp_out, soundsp


def sound_speed(mat: Any, rho: Optional[float] = None) -> float:
    """Compute acoustic dilatational sound speed c = sqrt((K + 4/3*G)/rho)."""
    params = resolve(mat)
    density = rho if (rho is not None and rho > 0.0) else params.rho0
    if density <= 0.0:
        density = 2.4e-6
    return math.sqrt((params.bulk + _FOUR_OVER_3 * params.G) / density)


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Return 6x6 initial isotropic elastic tangent stiffness matrix."""
    params = resolve(mat)
    c_el = np.zeros((6, 6), dtype=np.float64)
    lam = params.lam
    g2 = params.G2
    g = params.G

    c_el[0, 0] = lam + g2
    c_el[0, 1] = lam
    c_el[0, 2] = lam

    c_el[1, 0] = lam
    c_el[1, 1] = lam + g2
    c_el[1, 2] = lam

    c_el[2, 0] = lam
    c_el[2, 1] = lam
    c_el[2, 2] = lam + g2

    c_el[3, 3] = g
    c_el[4, 4] = g
    c_el[5, 5] = g

    return c_el


def consistent_solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    pert: float = 1.0e-7,
) -> np.ndarray:
    """Compute 6x6 algorithmic consistent tangent d(Delta sigma)/d(Delta eps) via central finite differences."""
    params = resolve(mat)
    if sig is None:
        sig = np.zeros(6, dtype=np.float64)
    if deps is None:
        deps = np.zeros(6, dtype=np.float64)

    uvar = extra.get("uvar") if extra else None
    if uvar is None:
        uvar = np.zeros(16, dtype=np.float64)
    elif uvar.ndim == 2:
        uvar = uvar[0]

    dmg = extra.get("dmg") if extra else None
    if dmg is None:
        dmg = np.zeros(3, dtype=np.float64)
    elif dmg.ndim == 2:
        dmg = dmg[0]

    le = 1.0
    if extra and "le" in extra:
        le_val = extra["le"]
        le = float(le_val[0]) if hasattr(le_val, "__len__") else float(le_val)

    tang = np.zeros((6, 6), dtype=np.float64)

    for j in range(6):
        deps_plus = np.copy(deps)
        deps_minus = np.copy(deps)
        deps_plus[j] += pert
        deps_minus[j] -= pert

        s_plus, _, _, _, _ = solid_step(params, sig, deps_plus, uvar, dmg, le=le)
        s_minus, _, _, _, _ = solid_step(params, sig, deps_minus, uvar, dmg, le=le)

        tang[:, j] = (s_plus - s_minus) / (2.0 * pert)

    return tang


def extra_shapes(mat: Any, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """Return dictionary of extra state arrays needed by /MAT/LAW124 elements."""
    return {
        "uvar": (nip, 16) if nip > 1 else (16,),
        "dmg": (nip, 3) if nip > 1 else (3,),
    }


def resolve(mat: Any) -> Law124Params:
    """Resolve Law124Params from Material entity, dict, or existing Law124Params."""
    if isinstance(mat, Law124Params):
        return mat
    if isinstance(mat, Material):
        p = mat.params or {}
        return Law124Params(
            E=float(p.get("E", p.get("MAT_E", 30000.0))),
            nu=float(p.get("nu", p.get("MAT_NU", 0.2))),
            rho0=float(getattr(mat, "rho0", p.get("rho", p.get("RHO", 2.4e-6)))),
            fc=float(p.get("fc", p.get("MAT_FC", 30.0))),
            ft=float(p.get("ft", p.get("MAT_FT", 3.0))),
            ecc=float(p.get("ecc", p.get("MAT_ECC", 0.0))),
            qh0=float(p.get("qh0", p.get("MAT_QH0", 0.3))),
            hp=float(p.get("hp", p.get("MAT_HP", 0.0))),
            ah=float(p.get("ah", p.get("MAT_AH", 0.08))),
            bh=float(p.get("bh", p.get("MAT_BH", 0.003))),
            ch=float(p.get("ch", p.get("MAT_CH", 2.0))),
            dh=float(p.get("dh", p.get("MAT_DH", 1.0e-6))),
            as_=float(p.get("as", p.get("as_", p.get("MAT_AS", 15.0)))),
            bs=float(p.get("bs", p.get("MAT_BS", 1.0))),
            df=float(p.get("df", p.get("MAT_DF", 0.85))),
            dflag=int(p.get("dflag", p.get("DFLAG", 1))),
            dtype=int(p.get("dtype", p.get("DTYPE", 2))),
            ireg=int(p.get("ireg", p.get("IREG", 2))),
            wf=float(p.get("wf", p.get("MAT_WF", 0.1))),
            wf1=float(p.get("wf1", p.get("MAT_WF1", 0.0))),
            ft1=float(p.get("ft1", p.get("MAT_FT1", 0.0))),
            efc=float(p.get("efc", p.get("MAT_EFC", 1.0e-4))),
            irate=int(p.get("irate", p.get("IRATE", 1))),
            fcut=float(p.get("fcut", p.get("FCUT", 0.0))),
            idel=int(p.get("idel", p.get("IDEL", 1))),
            Gf=float(p.get("Gf", p.get("GF", 0.0))) if (p.get("Gf") or p.get("GF")) else None,
            Gc=float(p.get("Gc", p.get("GC", 0.0))) if (p.get("Gc") or p.get("GC")) else None,
        )
    if isinstance(mat, dict):
        return Law124Params(
            E=float(mat.get("E", mat.get("MAT_E", 30000.0))),
            nu=float(mat.get("nu", mat.get("MAT_NU", 0.2))),
            rho0=float(mat.get("rho0", mat.get("rho", mat.get("RHO", 2.4e-6)))),
            fc=float(mat.get("fc", mat.get("MAT_FC", 30.0))),
            ft=float(mat.get("ft", mat.get("MAT_FT", 3.0))),
            ecc=float(mat.get("ecc", mat.get("MAT_ECC", 0.0))),
            qh0=float(mat.get("qh0", mat.get("MAT_QH0", 0.3))),
            hp=float(mat.get("hp", mat.get("MAT_HP", 0.0))),
            ah=float(mat.get("ah", mat.get("MAT_AH", 0.08))),
            bh=float(mat.get("bh", mat.get("MAT_BH", 0.003))),
            ch=float(mat.get("ch", mat.get("MAT_CH", 2.0))),
            dh=float(mat.get("dh", mat.get("MAT_DH", 1.0e-6))),
            as_=float(mat.get("as", mat.get("as_", mat.get("MAT_AS", 15.0)))),
            bs=float(mat.get("bs", mat.get("MAT_BS", 1.0))),
            df=float(mat.get("df", mat.get("MAT_DF", 0.85))),
            dflag=int(mat.get("dflag", mat.get("DFLAG", 1))),
            dtype=int(mat.get("dtype", mat.get("DTYPE", 2))),
            ireg=int(mat.get("ireg", mat.get("IREG", 2))),
            wf=float(mat.get("wf", mat.get("MAT_WF", 0.1))),
            wf1=float(mat.get("wf1", mat.get("MAT_WF1", 0.0))),
            ft1=float(mat.get("ft1", mat.get("MAT_FT1", 0.0))),
            efc=float(mat.get("efc", mat.get("MAT_EFC", 1.0e-4))),
            irate=int(mat.get("irate", mat.get("IRATE", 1))),
            fcut=float(mat.get("fcut", mat.get("FCUT", 0.0))),
            idel=int(mat.get("idel", mat.get("IDEL", 1))),
            Gf=float(mat.get("Gf", mat.get("GF", 0.0))) if (mat.get("Gf") or mat.get("GF")) else None,
            Gc=float(mat.get("Gc", mat.get("GC", 0.0))) if (mat.get("Gc") or mat.get("GC")) else None,
        )
    return Law124Params()


def build_law124(rec: Any) -> Material:
    """Constructor for /MAT/LAW124 (/MAT/CDPM2) from starter GenericMaterialRecord."""
    p = rec.params or {}
    e = float(p.get("MAT_E", p.get("E", 30000.0)))
    nu = float(p.get("MAT_NU", p.get("nu", 0.2)))
    params_dict = {
        "E": e if e > 0.0 else 30000.0,
        "nu": nu if 0.0 <= nu < 0.5 else 0.2,
        "MAT_E": e,
        "MAT_NU": nu,
        "IRATE": int(p.get("IRATE", p.get("irate", 1))),
        "FCUT": float(p.get("FCUT", p.get("fcut", 0.0))),
        "IDEL": int(p.get("IDEL", p.get("idel", 1))),
        "MAT_ECC": float(p.get("MAT_ECC", p.get("ecc", 0.0))),
        "MAT_QH0": float(p.get("MAT_QH0", p.get("qh0", 0.3))),
        "MAT_FT": float(p.get("MAT_FT", p.get("ft", 3.0))),
        "MAT_FC": float(p.get("MAT_FC", p.get("fc", 30.0))),
        "MAT_HP": float(p.get("MAT_HP", p.get("hp", 0.0))),
        "MAT_AH": float(p.get("MAT_AH", p.get("ah", 0.08))),
        "MAT_BH": float(p.get("MAT_BH", p.get("bh", 0.003))),
        "MAT_CH": float(p.get("MAT_CH", p.get("ch", 2.0))),
        "MAT_DH": float(p.get("MAT_DH", p.get("dh", 1.0e-6))),
        "MAT_AS": float(p.get("MAT_AS", p.get("as", p.get("as_", 15.0)))),
        "MAT_BS": float(p.get("MAT_BS", p.get("bs", 1.0))),
        "MAT_DF": float(p.get("MAT_DF", p.get("df", 0.85))),
        "DFLAG": int(p.get("DFLAG", p.get("dflag", 1))),
        "DTYPE": int(p.get("DTYPE", p.get("dtype", 2))),
        "IREG": int(p.get("IREG", p.get("ireg", 2))),
        "MAT_WF": float(p.get("MAT_WF", p.get("wf", 0.1))),
        "MAT_WF1": float(p.get("MAT_WF1", p.get("wf1", 0.0))),
        "MAT_FT1": float(p.get("MAT_FT1", p.get("ft1", 0.0))),
        "MAT_EFC": float(p.get("MAT_EFC", p.get("efc", 1.0e-4))),
        "irate": int(p.get("IRATE", p.get("irate", 1))),
        "fcut": float(p.get("FCUT", p.get("fcut", 0.0))),
        "idel": int(p.get("IDEL", p.get("idel", 1))),
        "ecc": float(p.get("MAT_ECC", p.get("ecc", 0.0))),
        "qh0": float(p.get("MAT_QH0", p.get("qh0", 0.3))),
        "ft": float(p.get("MAT_FT", p.get("ft", 3.0))),
        "fc": float(p.get("MAT_FC", p.get("fc", 30.0))),
        "hp": float(p.get("MAT_HP", p.get("hp", 0.0))),
        "ah": float(p.get("MAT_AH", p.get("ah", 0.08))),
        "bh": float(p.get("MAT_BH", p.get("bh", 0.003))),
        "ch": float(p.get("MAT_CH", p.get("ch", 2.0))),
        "dh": float(p.get("MAT_DH", p.get("dh", 1.0e-6))),
        "as": float(p.get("MAT_AS", p.get("as", p.get("as_", 15.0)))),
        "bs": float(p.get("MAT_BS", p.get("bs", 1.0))),
        "df": float(p.get("MAT_DF", p.get("df", 0.85))),
        "dflag": int(p.get("DFLAG", p.get("dflag", 1))),
        "dtype": int(p.get("DTYPE", p.get("dtype", 2))),
        "ireg": int(p.get("IREG", p.get("ireg", 2))),
        "wf": float(p.get("MAT_WF", p.get("wf", 0.1))),
        "wf1": float(p.get("MAT_WF1", p.get("wf1", 0.0))),
        "ft1": float(p.get("MAT_FT1", p.get("ft1", 0.0))),
        "efc": float(p.get("MAT_EFC", p.get("efc", 1.0e-4))),
    }
    rho = getattr(rec, "density", p.get("rho", 2.4e-6))
    return Material(id=rec.id, law=124, rho0=rho, title=getattr(rec, "title", ""), params=params_dict)


def _register() -> None:
    """Register LAW124/CDPM2 constructors in MAT_PHYSICS_REGISTRY."""
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY["LAW124"] = build_law124
    MAT_PHYSICS_REGISTRY["CDPM2"] = build_law124
    MAT_PHYSICS_REGISTRY["LAW124_CDPM2"] = build_law124


_register()
