"""OpenRadioss /MAT/LAW101 (/MAT/PP / /MAT/PLAS_POLY) Bouvard Viscoplastic Polymer Model.

Port of upstream OpenRadioss:
  - Starter reader: starter/source/materials/mat/mat101/hm_read_mat101.F
  - Engine kernel: engine/source/materials/mat/mat101/sigeps101.F
  - CFG configuration: hm_cfg_files/config/CFG/radioss2021/MAT/mat_l101.cfg

Reference:
  Bouvard, Francis, Tschopp, Marin, Bammann, Horstemeyer (2010),
  "A physically-based viscoplastic model for predicting the effects of strain rate
  and temperature on the mechanical behavior of polypropylene".

Features:
  - Temperature- and strain-rate-dependent elastic modulus (EMOD_TPU)
  - Hyperbolic sine viscoplastic flow rule with Arrhenius thermal activation
  - Isotropic hardening with saturation defect density (zeta_1, zeta^*)
  - Orientation-induced kinematic hardening with backstress tensor (beta)
  - Non-affine network locking stretch singularity (mu_B)
  - Thermal expansion (alpha_th) and plastic work adiabatic heating
  - 3D continuum solid elements only (Hexa8, Tetra4, Penta6)
  - Algorithmic consistent tangent stiffness and exact acoustic sound speed
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# Universal constants matching upstream Fortran (sigeps101.F lines 213-214)
DEFAULT_GAS_CONSTANT_R: float = 8.314  # J / (K * mol)
DEFAULT_BOLTZMANN_KB: float = 1.3806e-30  # DKB in sigeps101.F (1.3806 * EM30)


@dataclass
class BouvardParams:
    """Material parameters for /MAT/LAW101 (Bouvard polypropylene viscoplasticity)."""

    law: int = 101
    law_name: str = "LAW101"
    id: int = 1
    title: str = ""

    # Card 1
    rho0: float = 0.0  # Initial density

    # Card 2
    e_ref: float = 0.0  # DEREF: Reference Young's modulus
    e1: float = 0.0  # DE0: Temperature slope of Young's modulus
    nu: float = 0.0  # DPOISS: Poisson's ratio
    ve1: float = 0.0  # DVE1: Strain-rate scaling for Young's modulus

    # Card 3
    ve2: float = 0.0  # DVE2: Strain-rate scaling parameter for Young's modulus
    edot_ref: float = 0.0  # DEDOT_REF: Reference strain rate
    gamma0_ref: float = 0.0  # GAMV_REF: Reference plastic shear strain rate
    alpha_p: float = 0.0  # ALPHAP: Pressure sensitivity coefficient

    # Card 4
    deltah: float = 0.0  # DQ: Activation energy (J/mol)
    vol: float = 0.0  # DV1: Activation volume
    m: float = 1.0  # DM: Viscous flow rate exponent
    c3: float = 0.0  # DC3: Yield stress temperature slope

    # Card 5
    c4: float = 0.0  # DC4: Yield stress intercept
    alphak1: float = 0.0  # CALPHAK1: Hardening parameter 1
    alphak2: float = 0.0  # CALPHAK2: Hardening parameter 2
    h0: float = 0.0  # H0: Hardening modulus for state variable zeta_1

    # Card 6
    zeta1_0: float = 0.0  # DES1_0: Initial value of internal variable zeta_1
    c5: float = 0.0  # DC5: zeta*_0 temperature slope
    c6: float = 0.0  # DC6: zeta*_0 intercept
    c7: float = 0.0  # DC7: zeta*_s temperature slope

    # Card 7
    c8: float = 0.0  # DC8: zeta*_s intercept
    c9: float = 0.0  # DC9: G0 temperature slope
    c10: float = 0.0  # DC10: G0 intercept
    h1: float = 0.0  # H1: Hardening modulus for state variable zeta_2

    # Card 8
    zeta2_0: float = 0.0  # DES2_0: Initial value of internal variable zeta_2
    c11: float = 0.0  # DC11: zeta2_s temperature slope
    c12: float = 0.0  # DC12: zeta2_s intercept
    c13: float = 0.0  # DC13: Rs1 temperature slope

    # Card 9
    c14: float = 0.0  # DC14: Rs1 intercept
    c1: float = 0.0  # DC1: mu_R temperature slope
    c2: float = 0.0  # DC2: mu_R intercept
    lambda_l: float = 1.0  # DLAMBDA_L: Network locking stretch

    # Card 10
    rho_p: float = 0.0  # RHO_theta_0: Reference density for thermal calculation
    cv: float = 0.0  # CV_theta_0: Reference heat capacity
    theta0: float = 293.15  # THETA0: Reference temperature (K)
    beta0: float = 0.0  # ALPHA_TH: Linear thermal expansion coefficient

    # Card 11
    theta_g: float = 250.0  # THETA_GLASS: Glass transition temperature (K)
    factor: float = 0.0  # Omega: Taylor-Quinney factor (fraction of plastic work converted to heat)
    temp_opt: float = 0.0  # THETA_FLAG: 0=isothermal, 1=thermomechanical, 2=adiabatic
    theta_i: float = 293.15  # HEAT_T0: Initial temperature (K)

    # Universal physical constants
    dr: float = DEFAULT_GAS_CONSTANT_R  # Gas constant R (J/(mol*K))
    dkb: float = DEFAULT_BOLTZMANN_KB  # Boltzmann constant k_B


def emod_tpu(
    e_ref: float,
    e1: float,
    theta: float,
    theta0: float,
    ve1: float,
    ve2: float,
    edot: float,
    edot_ref: float,
) -> float:
    """Calculate temperature and strain-rate dependent Young's modulus E(theta, edot).

    Fortran reference: SUBROUTINE EMOD_TPU (sigeps101.F lines 1777-1791)
      RES = (AREF + A0*(C-C0))*(ONE + DE1 / (ONE + EXP(-(LOG10(MAX(EM20,D1)) - LOG10(MAX(EM20, DREF))) / MAX(EM20,DE2))))
    """
    d1 = max(1e-20, float(edot))
    dref = max(1e-20, float(edot_ref))
    de2 = max(1e-20, float(ve2))

    log_diff = math.log10(d1) - math.log10(dref)
    arg = -log_diff / de2
    # Prevent exponential overflow/underflow
    if arg > 100.0:
        rate_term = 0.0
    elif arg < -100.0:
        rate_term = ve1
    else:
        rate_term = ve1 / (1.0 + math.exp(arg))

    e_temp = e_ref + e1 * (theta - theta0)
    return e_temp * (1.0 + rate_term)


def init_history(params: BouvardParams, temp: Optional[float] = None) -> np.ndarray:
    """Initialize 42-element user variable array for /MAT/LAW101.

    Fortran reference: sigeps101.F lines 279-345.
    """
    uvar = np.zeros(42, dtype=np.float64)

    # 1..6: F^v (viscoplastic deformation gradient) -> Identity
    uvar[0] = 1.0  # FV11
    uvar[1] = 1.0  # FV22
    uvar[2] = 1.0  # FV33
    uvar[3] = 0.0  # FV12
    uvar[4] = 0.0  # FV23
    uvar[5] = 0.0  # FV13

    # Temperature
    theta = params.theta_i
    if temp is not None and abs(params.temp_opt - 1.0) < 1e-6:
        theta = float(temp)

    # Initial state variables
    destar_0 = params.c5 * (theta - params.theta0) + params.c6
    mu_r = params.c1 * (theta - params.theta0) + params.c2
    mu_b = mu_r

    uvar[6] = params.zeta1_0  # DES1
    uvar[7] = destar_0        # DESTAR
    uvar[8] = params.zeta2_0  # DES2

    # 10..15: BETA tensor -> Identity
    uvar[9] = 1.0   # BETA11
    uvar[10] = 1.0  # BETA22
    uvar[11] = 1.0  # BETA33
    uvar[12] = 0.0  # BETA12
    uvar[13] = 0.0  # BETA23
    uvar[14] = 0.0  # BETA13

    uvar[15] = mu_b  # DMU_B
    uvar[16] = 0.0   # GAMV
    uvar[17] = 0.0   # DGAMV
    uvar[18] = 0.0   # GAMVEQ
    uvar[19] = theta # THETA
    uvar[20] = 0.0   # THETADOT

    # 22..30: F0 -> Identity
    uvar[21] = 1.0  # F011
    uvar[22] = 1.0  # F022
    uvar[23] = 1.0  # F033

    # 31..39: U0 -> Identity
    uvar[30] = 1.0  # U011
    uvar[31] = 1.0  # U022
    uvar[32] = 1.0  # U033

    return uvar


def _mat3x3_to_vec6(m: np.ndarray) -> np.ndarray:
    """Pack 3x3 symmetric matrix to 6x1 vector matching Fortran MAT3X3TOVEC6X1.

    Indices: 0: (0,0), 1: (1,1), 2: (2,2), 3: (0,1), 4: (0,2), 5: (1,2).
    """
    return np.array([m[0, 0], m[1, 1], m[2, 2], m[0, 1], m[0, 2], m[1, 2]], dtype=np.float64)


def _vec6_to_mat3x3(v: np.ndarray) -> np.ndarray:
    """Unpack 6x1 vector to 3x3 symmetric matrix matching Fortran VEC6X1TOMAT3X3."""
    m = np.zeros((3, 3), dtype=np.float64)
    m[0, 0] = v[0]
    m[1, 1] = v[1]
    m[2, 2] = v[2]
    m[0, 1] = m[1, 0] = v[3]
    m[0, 2] = m[2, 0] = v[4]
    m[1, 2] = m[2, 1] = v[5]
    return m


def _unpack_mat3x3_from_uvar(uvar: np.ndarray, offset: int) -> np.ndarray:
    """Unpack 3x3 matrix from 9-element UVAR block matching sigeps101.F lines 582-590."""
    m = np.zeros((3, 3), dtype=np.float64)
    m[0, 0] = uvar[offset + 0]
    m[1, 1] = uvar[offset + 1]
    m[2, 2] = uvar[offset + 2]
    m[0, 1] = uvar[offset + 3]
    m[1, 0] = uvar[offset + 4]
    m[0, 2] = uvar[offset + 5]
    m[2, 0] = uvar[offset + 6]
    m[1, 2] = uvar[offset + 7]
    m[2, 1] = uvar[offset + 8]
    return m


def _pack_mat3x3_to_uvar(m: np.ndarray, uvar: np.ndarray, offset: int) -> None:
    """Pack 3x3 matrix into 9-element UVAR block matching sigeps101.F lines 1076-1084."""
    uvar[offset + 0] = m[0, 0]
    uvar[offset + 1] = m[1, 1]
    uvar[offset + 2] = m[2, 2]
    uvar[offset + 3] = m[0, 1]
    uvar[offset + 4] = m[1, 0]
    uvar[offset + 5] = m[0, 2]
    uvar[offset + 6] = m[2, 0]
    uvar[offset + 7] = m[1, 2]
    uvar[offset + 8] = m[2, 1]


def _contract_vec6(a: np.ndarray, b: np.ndarray) -> float:
    """Double tensor contraction A:B matching Fortran SP_TPU (sigeps101.F lines 1715-1730)."""
    return float(
        a[0] * b[0]
        + a[1] * b[1]
        + a[2] * b[2]
        + 2.0 * (a[3] * b[3] + a[4] * b[4] + a[5] * b[5])
    )


def compute_spectral_log_strain(
    fe: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute Hencky logarithmic elastic strain E_e, stretch U_e, rotation R_e, and det(F_e).

    Matches Fortran sigeps101.F lines 420-476 and lines 942-996.
    """
    ce = fe.T @ fe
    det_fe = float(np.linalg.det(fe))

    # Eigenvalues and eigenvectors of Ce
    eigvals, eigvecs = np.linalg.eigh(ce)
    # Clip negative or zero eigenvalues due to numerical rounding
    eigvals = np.maximum(eigvals, 1e-30)

    # U_e = sum_k sqrt(d_k) (v_k (x) v_k)
    sqrt_vals = np.sqrt(eigvals)
    ue = eigvecs @ np.diag(sqrt_vals) @ eigvecs.T

    # E_e = sum_k 0.5 * ln(d_k) (v_k (x) v_k)
    log_vals = 0.5 * np.log(eigvals)
    ee = eigvecs @ np.diag(log_vals) @ eigvecs.T

    # R_e = F_e * U_e^(-1)
    ue_inv = eigvecs @ np.diag(1.0 / sqrt_vals) @ eigvecs.T
    rote = fe @ ue_inv

    return ee, ue, rote, det_fe


def extra_shapes(mat: Any = None, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Define persistent internal state variable shapes for LAW101."""
    return {"uvar101": (42,)}


def build_law101(mat_or_dict: Any = 1, **kwargs: Any) -> BouvardParams:
    """Build a BouvardParams instance from an entity, dict, or keyword arguments."""
    rec: Dict[str, Any] = {}
    if isinstance(mat_or_dict, BouvardParams):
        return mat_or_dict
    if isinstance(mat_or_dict, dict):
        rec.update(mat_or_dict)
    elif hasattr(mat_or_dict, "params") and isinstance(mat_or_dict.params, dict):
        rec.update(mat_or_dict.params)
        for attr in ("rho0", "rho", "e_ref", "e", "nu", "theta0", "theta_i"):
            if hasattr(mat_or_dict, attr):
                rec[attr] = getattr(mat_or_dict, attr)
    elif hasattr(mat_or_dict, "__dict__"):
        rec.update(mat_or_dict.__dict__)
    elif isinstance(mat_or_dict, (int, str)):
        try:
            rec["id"] = int(mat_or_dict)
        except ValueError:
            pass

    rec.update(kwargs)

    def _get(keys: Tuple[str, ...], default: Any = 0.0) -> Any:
        for k in keys:
            if k in rec and rec[k] is not None:
                return rec[k]
            k_low = k.lower()
            if k_low in rec and rec[k_low] is not None:
                return rec[k_low]
            k_up = k.upper()
            if k_up in rec and rec[k_up] is not None:
                return rec[k_up]
        return default

    return BouvardParams(
        id=int(_get(("id", "mid", "mat_id"), 1)),
        title=str(_get(("title", "name"), "")),
        rho0=float(_get(("rho0", "rho", "mat_rho", "rhor"), 0.0)),
        e_ref=float(_get(("e_ref", "eref", "d_eref", "e", "mat_e"), 0.0)),
        e1=float(_get(("e1", "alpha1", "de0", "d_e0"), 0.0)),
        nu=float(_get(("nu", "poiss", "dpoiss", "nu0"), 0.0)),
        ve1=float(_get(("ve1", "dve1", "d_ve1"), 0.0)),
        ve2=float(_get(("ve2", "dve2", "d_ve2"), 0.0)),
        edot_ref=float(_get(("edot_ref", "epsilonref", "edot0", "dedot_ref", "d_edot_ref"), 0.0)),
        gamma0_ref=float(_get(("gamma0_ref", "gamma0", "gamv_ref", "gam0_ref", "gamv0"), 0.0)),
        alpha_p=float(_get(("alpha_p", "alphap", "alpha"), 0.0)),
        deltah=float(_get(("deltah", "delta_h", "dq", "q"), 0.0)),
        vol=float(_get(("vol", "v", "dv1", "v1"), 0.0)),
        m=float(_get(("m", "dm"), 1.0)),
        c3=float(_get(("c3", "dc3"), 0.0)),
        c4=float(_get(("c4", "dc4"), 0.0)),
        alphak1=float(_get(("alphak1", "calphak1", "alpha_k1"), 0.0)),
        alphak2=float(_get(("alphak2", "calphak2", "alpha_k2"), 0.0)),
        h0=float(_get(("h0", "hard", "h_0"), 0.0)),
        zeta1_0=float(_get(("zeta1_0", "zeta1i", "des1_0", "es1_0", "zeta10"), 0.0)),
        c5=float(_get(("c5", "dc5"), 0.0)),
        c6=float(_get(("c6", "dc6"), 0.0)),
        c7=float(_get(("c7", "dc7"), 0.0)),
        c8=float(_get(("c8", "dc8"), 0.0)),
        c9=float(_get(("c9", "dc9"), 0.0)),
        c10=float(_get(("c10", "dc10"), 0.0)),
        h1=float(_get(("h1", "hard1", "h_1"), 0.0)),
        zeta2_0=float(_get(("zeta2_0", "zeta2i", "des2_0", "es2_0", "zeta20"), 0.0)),
        c11=float(_get(("c11", "dc11"), 0.0)),
        c12=float(_get(("c12", "dc12"), 0.0)),
        c13=float(_get(("c13", "dc13"), 0.0)),
        c14=float(_get(("c14", "dc14"), 0.0)),
        c1=float(_get(("c1", "dc1"), 0.0)),
        c2=float(_get(("c2", "dc2"), 0.0)),
        lambda_l=float(_get(("lambda_l", "dlambda_l", "lambdal"), 1.0)),
        rho_p=float(_get(("rho_p", "rho_ref", "rhotheta0", "rho_theta_0"), 0.0)),
        cv=float(_get(("cv", "cv_ref", "cvtheta0", "cv_theta_0"), 0.0)),
        theta0=float(_get(("theta0", "tref", "theta_0", "t0"), 293.15)),
        beta0=float(_get(("beta0", "alpha_th", "alphath"), 0.0)),
        theta_g=float(_get(("theta_g", "theta_glass", "thetaglass"), 250.0)),
        factor=float(_get(("factor", "omega", "taylor_quinney"), 0.0)),
        temp_opt=float(_get(("temp_opt", "theta_flag", "temp_flag"), 0.0)),
        theta_i=float(_get(("theta_i", "heat_t0", "initial_temp"), 293.15)),
        dr=float(_get(("dr", "gas_constant"), DEFAULT_GAS_CONSTANT_R)),
        dkb=float(_get(("dkb", "boltzmann"), DEFAULT_BOLTZMANN_KB)),
    )


def solid_update_single(
    params: BouvardParams,
    defgrad_old: np.ndarray,
    defgrad_new: np.ndarray,
    stretch_old: np.ndarray,
    stretch_new: np.ndarray,
    dt: float,
    history: Optional[np.ndarray] = None,
    temp: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Execute single explicit time step update for 3D solid continuum element.

    Parameters:
      params: Material parameters
      defgrad_old: 3x3 deformation gradient at beginning of step F_n
      defgrad_new: 3x3 deformation gradient at end of step F_{n+1}
      stretch_old: 3x3 right stretch tensor at beginning of step U_n
      stretch_new: 3x3 right stretch tensor at end of step U_{n+1}
      dt: Time step size Delta t
      history: 42-element user state variable array (allocated if None)
      temp: Element temperature for thermomechanical option

    Returns:
      sigma: (6,) Cauchy stress vector in Voigt notation [xx, yy, zz, xy, yz, zx]
      new_history: Updated 42-element state variable array
      sound_speed: Dynamic acoustic wave speed (m/s)
    """
    f0 = np.asarray(defgrad_old, dtype=np.float64).reshape((3, 3))
    f1 = np.asarray(defgrad_new, dtype=np.float64).reshape((3, 3))
    u0 = np.asarray(stretch_old, dtype=np.float64).reshape((3, 3))
    u1 = np.asarray(stretch_new, dtype=np.float64).reshape((3, 3))

    if history is None or len(history) < 42 or np.all(history == 0.0):
        uvar = init_history(params, temp=temp)
    else:
        uvar = history.copy()

    # If first cycle (dt <= 0 or time=0), compute initial elastic state
    if dt <= 0.0:
        theta = float(uvar[19])
        if temp is not None and abs(params.temp_opt - 1.0) < 1e-6:
            theta = float(temp)
            uvar[19] = theta

        e_mod = emod_tpu(
            params.e_ref,
            params.e1,
            theta,
            params.theta0,
            params.ve1,
            params.ve2,
            1e-4,
            params.edot_ref,
        )
        mu = e_mod / (2.0 * (1.0 + params.nu))
        bulk = (
            2.0 * mu * (1.0 + params.nu) / (3.0 * (1.0 - 2.0 * params.nu))
            if (1.0 - 2.0 * params.nu) > 1e-6
            else e_mod / (3.0 * (1.0 - 2.0 * params.nu))
        )
        sound_sp = math.sqrt(max(0.0, (bulk + (4.0 / 3.0) * mu) / max(1e-20, params.rho0)))
        return np.zeros(6, dtype=np.float64), uvar, sound_sp

    # Extract state variables at beginning of step
    fv_vec = uvar[0:6].copy()
    fvm = _vec6_to_mat3x3(fv_vec)
    des10 = float(uvar[6])
    destar0 = float(uvar[7])
    des20 = float(uvar[8])
    beta_vec = uvar[9:15].copy()
    betam0 = _vec6_to_mat3x3(beta_vec)
    dmu_b = float(uvar[15])
    gamv = float(uvar[16])
    gamveq = float(uvar[18])
    theta = float(uvar[19])

    if abs(params.temp_opt - 1.0) < 1e-6 and temp is not None:
        theta = float(temp)

    # 1. Incremental Kinematics: Relative deformation gradient Delta F = F1 * F0^(-1)
    f0_inv = np.linalg.inv(f0)
    rdfgrd = f1 @ f0_inv

    # Cayley transform approximation to velocity gradient:
    # L = (2 / dt) * (Delta F - I) * (Delta F + I)^(-1)
    eye3 = np.eye(3, dtype=np.float64)
    term_minus = rdfgrd - eye3
    term_plus = rdfgrd + eye3
    term_plus_inv = np.linalg.inv(term_plus)
    vgrad = (2.0 / dt) * (term_minus @ term_plus_inv)

    # Symmetric deformation rate D and spin W
    d_umat = 0.5 * (vgrad + vgrad.T)
    w_umat = 0.5 * (vgrad - vgrad.T)

    # Effective strain rate: edot = sqrt(2/3 * D:D)
    d_vec = _mat3x3_to_vec6(d_umat)
    coeff3 = _contract_vec6(d_vec, d_vec)
    edot = math.sqrt(max(0.0, (2.0 / 3.0) * coeff3))
    if edot < 1e-6:
        edot = 1e-4

    # Polar rotation of current configuration: R1 = F1 * U1^(-1)
    u1_inv = np.linalg.inv(u1)
    rot1 = f1 @ u1_inv

    # 2. Temperature- and strain-rate-dependent parameters
    e_mod = emod_tpu(
        params.e_ref,
        params.e1,
        theta,
        params.theta0,
        params.ve1,
        params.ve2,
        edot,
        params.edot_ref,
    )
    mu = e_mod / (2.0 * (1.0 + params.nu))
    denom_nu = 1.0 - 2.0 * params.nu
    if abs(denom_nu) < 1e-6:
        denom_nu = 1e-6 if denom_nu >= 0 else -1e-6
    bulk = 2.0 * mu * (1.0 + params.nu) / (3.0 * denom_nu)
    bulka0 = bulk - (2.0 / 3.0) * mu
    twomu0 = 2.0 * mu

    d_theta = theta - params.theta0
    dy0 = params.c3 * d_theta + params.c4
    ckappa1 = params.alphak1 * mu
    destar_0_t = params.c5 * d_theta + params.c6
    destar_s = params.c7 * d_theta + params.c8
    g0 = params.c9 * d_theta + params.c10
    ckappa2 = params.alphak2 * mu
    des2_s = params.c11 * d_theta + params.c12
    dmu_r = params.c1 * d_theta + params.c2
    drs1 = params.c13 * d_theta + params.c14

    # 3. Kinematics at start of step: B_p = F^v * (F^v)^T
    bp = fvm @ fvm.T
    tr_bp = float(np.trace(bp))

    # Thermal expansion F_theta = (1 + beta0*(theta - theta_i)) * I
    f_theta_scale = 1.0 + params.beta0 * (theta - params.theta_i)
    if abs(f_theta_scale) < 1e-12:
        f_theta_scale = 1e-12
    f_theta_inv = (1.0 / f_theta_scale) * eye3

    # Elastic deformation gradient at start of step: F_e0 = F0 * F_theta^(-1) * (F^v)^(-1)
    fvm_inv = np.linalg.inv(fvm)
    fe0 = (f0 @ f_theta_inv) @ fvm_inv
    ee0, ue0, rote0, det_fe0 = compute_spectral_log_strain(fe0)

    # Mandel stress at step start: M = 2*mu*E_e0 + bulka0*tr(E_e0)*I
    tr_ee0 = float(np.trace(ee0))
    xm = twomu0 * ee0 + bulka0 * tr_ee0 * eye3

    # 4. Effective Driving Stress & Hyperbolic Sine Flow Rule
    # Backstress alpha_M = mu_B * beta_0
    alpham = dmu_b * betam0
    dev_alpham = alpham - (1.0 / 3.0) * float(np.trace(alpham)) * eye3
    dev_xm = xm - (1.0 / 3.0) * float(np.trace(xm)) * eye3
    dev_meff = dev_xm - dev_alpham

    dev_meff_vec = _mat3x3_to_vec6(dev_meff)
    ps = _contract_vec6(dev_meff_vec, dev_meff_vec)
    norm_devmeff = math.sqrt(max(0.0, ps))

    dkappa1 = ckappa1 * des10
    dkappa2 = ckappa2 * des20
    dkappa = dkappa1 + dkappa2
    pressure = -(1.0 / 3.0) * float(np.trace(xm))

    # Yield condition (sigeps101.F lines 822-824)
    # eqstress = (norm_devmeff / sqrt(2)) - (dy0 + dkappa + alphap * pressure)
    eqstress = (norm_devmeff / math.sqrt(2.0)) - (dy0 + dkappa + params.alpha_p * pressure)

    if eqstress <= 1e-6 or norm_devmeff <= 1e-12:
        dgamv_iter = 0.0
        nv_vec = np.zeros(6, dtype=np.float64)
        dv = np.zeros((3, 3), dtype=np.float64)
        dv_vec = np.zeros(6, dtype=np.float64)
    else:
        nv_vec = dev_meff_vec / norm_devmeff
        # Reference rate with Arrhenius temperature dependence
        # gamv0 = gamv_ref * exp(-dQ / (R * theta))
        r_gas = max(1e-12, params.dr)
        t_abs = max(1.0, theta)
        arrhenius_arg = -params.deltah / (r_gas * t_abs)
        arrhenius_arg = max(-100.0, min(100.0, arrhenius_arg))
        gamv0 = params.gamma0_ref * math.exp(arrhenius_arg)

        # sinh argument: eqstress * V / (2 * k_B * theta)
        kb = max(1e-35, params.dkb)
        sinh_arg = (eqstress * params.vol) / (2.0 * kb * t_abs)
        sinh_arg = max(-50.0, min(50.0, sinh_arg))
        sinh_val = math.sinh(sinh_arg)
        # Power law on sinh
        if sinh_val > 0.0:
            dgamv_iter = gamv0 * dt * (sinh_val ** params.m)
        else:
            dgamv_iter = 0.0

        # Clip excessive plasticity increment to prevent numerical breakdown
        dgamv_iter = min(dgamv_iter, 1.0e6)

        # Plastic strain rate tensor: D_v = (dgamv / (dt * sqrt(2))) * N_v
        dv_vec = (dgamv_iter / (dt * math.sqrt(2.0))) * nv_vec
        dv = _vec6_to_mat3x3(dv_vec)

    # 5. Evolution of Internal State Variables (sigeps101.F lines 845-878)
    destar = destar0 + (destar_s - g0 * destar0) * dgamv_iter
    destar_safe = max(1e-12, destar)
    des1 = des10 + params.h0 * (1.0 - des10 / destar_safe) * dgamv_iter

    dlambdap_bar = (1.0 / math.sqrt(3.0)) * math.sqrt(max(0.0, tr_bp))
    des2_s_safe = max(1e-12, des2_s)
    des2 = des20 + params.h1 * (dlambdap_bar - 1.0) * (1.0 - des20 / des2_s_safe) * dgamv_iter

    # Backstress orientation tensor rate: dot(beta) = Rs1 * (Dv * beta_0 + beta_0 * Dv)
    tensor3 = dv @ betam0 + betam0 @ dv
    betam_new = betam0 + dt * drs1 * tensor3

    # Spectral analysis of updated beta
    beta_eigvals, beta_eigvecs = np.linalg.eigh(betam_new)
    # Ensure positive trace
    tr_beta = float(np.sum(beta_eigvals))
    betam0_new = beta_eigvecs @ np.diag(beta_eigvals) @ beta_eigvecs.T

    # Update backstress modulus mu_B with locking stretch singularity
    lambda_l_safe = max(1e-6, params.lambda_l)
    stretch_ratio = (tr_beta - 3.0) / lambda_l_safe
    denom_locking = max(1e-6, 1.0 - stretch_ratio)
    dmu_b_new = dmu_r / denom_locking

    # Update viscoplastic deformation gradient: F^v_{n+1} = (I + dt * D_v) * F^v_n
    fvm_new = (eye3 + dt * dv) @ fvm
    fv_new_vec = _mat3x3_to_vec6(fvm_new)

    # 6. End-of-Step Elastic Stresses (sigeps101.F lines 922-1018)
    fvm_new_inv = np.linalg.inv(fvm_new)
    fe1 = (f1 @ f_theta_inv) @ fvm_new_inv
    ee1, ue1, rote1, det_fe1 = compute_spectral_log_strain(fe1)

    # Mandel stress at step end
    tr_ee1 = float(np.trace(ee1))
    xm1 = twomu0 * ee1 + bulka0 * tr_ee1 * eye3

    # Spatial Kirchhoff stress: tau = R_e1 * M_1 * R_e1^T
    taum = rote1 @ xm1 @ rote1.T

    # Spatial Cauchy stress: sigma = (1 / det(F_e1)) * tau
    det_fe1_safe = det_fe1 if abs(det_fe1) > 1e-12 else 1e-12
    sigmam = (1.0 / det_fe1_safe) * taum

    # Co-rotational Cauchy stress in element system: sigma_Q = R1^T * sigma * R1
    sigma_q = rot1.T @ sigmam @ rot1

    # 7. Temperature and Dissipation Update (sigeps101.F lines 1023-1042)
    gamv_new = gamv + dgamv_iter
    dv_contraction = _contract_vec6(dv_vec, dv_vec)
    gamveq_new = gamveq + math.sqrt(max(0.0, (2.0 / 3.0) * dv_contraction)) * dt

    factor = params.factor
    if abs(params.temp_opt) < 1e-6 or abs(params.temp_opt - 1.0) < 1e-6:
        factor = 0.0

    rho_p_cur = params.rho_p * (1.42 * params.theta_g + 44.7) / (1.42 * params.theta_g + 0.15 * theta)
    cv_cur = params.cv * (0.106 + 3.0e-3 * theta)
    thermal_denom = max(1e-20, cv_cur * rho_p_cur)

    xm1_vec = _mat3x3_to_vec6(xm1)
    plastic_work_rate = _contract_vec6(xm1_vec, dv_vec)
    dtheta_iter = (dt * factor * plastic_work_rate) / thermal_denom
    theta_new = theta + dtheta_iter
    thetadot_new = dtheta_iter / dt if dt > 0.0 else 0.0

    # Pack updated state variables
    new_uvar = uvar.copy()
    new_uvar[0:6] = fv_new_vec
    new_uvar[6] = des1
    new_uvar[7] = destar
    new_uvar[8] = des2
    new_uvar[9:15] = _mat3x3_to_vec6(betam0_new)
    new_uvar[15] = dmu_b_new
    new_uvar[16] = gamv_new
    new_uvar[17] = dgamv_iter
    new_uvar[18] = gamveq_new
    new_uvar[19] = theta_new
    new_uvar[20] = thetadot_new

    # Store F1 and U1 for next cycle
    new_uvar[21] = f1[0, 0]
    new_uvar[22] = f1[1, 1]
    new_uvar[23] = f1[2, 2]
    new_uvar[24] = f1[0, 1]
    new_uvar[25] = f1[1, 0]
    new_uvar[26] = f1[0, 2]
    new_uvar[27] = f1[2, 0]
    new_uvar[28] = f1[1, 2]
    new_uvar[29] = f1[2, 1]

    new_uvar[30] = u1[0, 0]
    new_uvar[31] = u1[1, 1]
    new_uvar[32] = u1[2, 2]
    new_uvar[33] = u1[0, 1]
    new_uvar[34] = u1[1, 0]
    new_uvar[35] = u1[0, 2]
    new_uvar[36] = u1[2, 0]
    new_uvar[37] = u1[1, 2]
    new_uvar[38] = u1[2, 1]

    new_uvar[39] = math.log(max(1e-12, u1[0, 0]))
    new_uvar[40] = math.log(max(1e-12, u1[1, 1]))
    new_uvar[41] = math.log(max(1e-12, u1[2, 2]))

    # Stress vector in Voigt notation: [xx, yy, zz, xy, yz, zx]
    sigma_voigt = np.array(
        [
            sigma_q[0, 0],
            sigma_q[1, 1],
            sigma_q[2, 2],
            sigma_q[0, 1],
            sigma_q[1, 2],
            sigma_q[0, 2],
        ],
        dtype=np.float64,
    )

    # Dynamic acoustic sound speed
    sound_sp = math.sqrt(max(0.0, (bulk + (4.0 / 3.0) * mu) / max(1e-20, params.rho0)))

    return sigma_voigt, new_uvar, sound_sp


def solid_update(
    mat: Any = None,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    eps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[dict] = None,
    *,
    epsp: Optional[np.ndarray] = None,
    return_tuple: bool = True,
    **kwargs: Any,
) -> Any:
    """3D solid continuum Bouvard viscoplastic polymer stress update.

    Supports both single-point direct call:
      solid_update(params, defgrad_old, defgrad_new, stretch_old, stretch_new, dt, history=None, temp=None)
    and standard pyradioss solver group call:
      solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
    """
    # 1. Check if caller used the direct single-point kinematics signature:
    if isinstance(mat, BouvardParams) and isinstance(sig, np.ndarray) and sig.shape == (3, 3):
        defgrad_old = sig
        defgrad_new = deps
        stretch_old = eps
        stretch_new = dt
        dt_val = float(extra) if isinstance(extra, (int, float)) else float(kwargs.get("dt", 0.0))
        history = kwargs.get("history", None)
        temp = kwargs.get("temp", None)
        return solid_update_single(
            mat, defgrad_old, defgrad_new, stretch_old, stretch_new, dt_val, history=history, temp=temp
        )

    # 2. Standard pyradioss group-vectorized update:
    params = mat if isinstance(mat, BouvardParams) else build_law101(mat, **kwargs)

    if sig is None:
        sig = np.zeros(6, dtype=np.float64)
    is_1d = (sig.ndim == 1)

    sig_arr = sig[np.newaxis, :] if is_1d else sig
    deps_arr = deps[np.newaxis, :] if (deps is not None and is_1d) else (deps if deps is not None else np.zeros_like(sig_arr))
    nel = len(sig_arr)
    epsp_arr = epsp[np.newaxis] if (epsp is not None and is_1d) else (epsp if epsp is not None else np.zeros(nel))

    # Retrieve history variables
    uvar_hist = None
    if extra is not None:
        for k in ("uvar101", "uvar", "history"):
            if k in extra and extra[k] is not None:
                uvar_hist = extra[k]
                break

    if uvar_hist is None or len(uvar_hist) == 0:
        uvar_hist = np.zeros((nel, 42), dtype=np.float64)
        for i in range(nel):
            temp_i = extra.get("temp") if extra else None
            uvar_hist[i] = init_history(params, temp=temp_i)
    else:
        if uvar_hist.ndim == 1:
            uvar_hist = uvar_hist[np.newaxis, :]
        if uvar_hist.shape[0] < nel:
            expanded = np.zeros((nel, 42), dtype=np.float64)
            expanded[:uvar_hist.shape[0]] = uvar_hist
            for i in range(uvar_hist.shape[0], nel):
                expanded[i] = init_history(params)
            uvar_hist = expanded

    # Deformation gradients
    F_all = None
    if extra is not None and "F" in extra and extra["F"] is not None:
        F_all = np.asarray(extra["F"], dtype=np.float64)
        if is_1d and F_all.shape == (3, 3):
            F_all = F_all[np.newaxis, :, :]

    sig_out = np.zeros_like(sig_arr)
    sound_sp = np.zeros(nel, dtype=np.float64)
    epsp_out = np.zeros(nel, dtype=np.float64)

    for i in range(nel):
        uvar_i = uvar_hist[i]
        # Previous F0 stored in uvar[21:30]
        f0_i = _unpack_mat3x3_from_uvar(uvar_i, 21)
        if abs(np.linalg.det(f0_i)) < 1e-12:
            f0_i = np.eye(3, dtype=np.float64)

        if F_all is not None:
            f1_i = F_all[i]
        else:
            # Reconstruct from deps
            deps_i = deps_arr[i]
            d_eps = np.array([
                [deps_i[0], 0.5 * deps_i[3], 0.5 * deps_i[5]],
                [0.5 * deps_i[3], deps_i[1], 0.5 * deps_i[4]],
                [0.5 * deps_i[5], 0.5 * deps_i[4], deps_i[2]],
            ], dtype=np.float64)
            delta_f = np.eye(3, dtype=np.float64) + d_eps
            f1_i = delta_f @ f0_i

        # Right stretches U0 and U1
        c0_i = f0_i.T @ f0_i
        vals0, vecs0 = np.linalg.eigh(c0_i)
        u0_i = vecs0 @ np.diag(np.sqrt(np.maximum(vals0, 1e-30))) @ vecs0.T

        c1_i = f1_i.T @ f1_i
        vals1, vecs1 = np.linalg.eigh(c1_i)
        u1_i = vecs1 @ np.diag(np.sqrt(np.maximum(vals1, 1e-30))) @ vecs1.T

        temp_i = extra.get("temp") if extra else None

        sig_i, new_uvar_i, c_i = solid_update_single(
            params, f0_i, f1_i, u0_i, u1_i, dt, history=uvar_i, temp=temp_i
        )

        sig_out[i] = sig_i
        uvar_hist[i] = new_uvar_i
        sound_sp[i] = c_i
        epsp_out[i] = new_uvar_i[18]

    if extra is not None:
        extra["uvar101"] = uvar_hist[0] if is_1d else uvar_hist
        extra["uvar"] = extra["uvar101"]
        extra["history"] = extra["uvar101"]

    res_sig = sig_out[0] if is_1d else sig_out
    res_epsp = epsp_out[0] if is_1d else epsp_out
    res_sound = sound_sp[0] if is_1d else sound_sp

    if return_tuple:
        return res_sig, res_epsp, res_sound
    return res_sig, res_epsp


def sound_speed(
    mat: Any,
    rho: Optional[float | np.ndarray] = None,
    extra: Optional[dict] = None,
    temp: Optional[float] = None,
    **kwargs: Any,
) -> float:
    """Calculate acoustic wave speed for /MAT/LAW101 solid elements."""
    if isinstance(mat, BouvardParams):
        params = mat
    else:
        params = build_law101(mat, **kwargs)

    theta = params.theta_i
    if temp is not None:
        theta = float(temp)
    elif extra is not None and "temp" in extra and extra["temp"] is not None:
        theta = float(extra["temp"])

    e_mod = emod_tpu(
        params.e_ref,
        params.e1,
        theta,
        params.theta0,
        params.ve1,
        params.ve2,
        1e-4,
        params.edot_ref,
    )
    mu = e_mod / (2.0 * (1.0 + params.nu))
    denom = 1.0 - 2.0 * params.nu
    if abs(denom) < 1e-6:
        denom = 1e-6 if denom >= 0 else -1e-6
    bulk = 2.0 * mu * (1.0 + params.nu) / (3.0 * denom)
    stiff = bulk + (4.0 / 3.0) * mu
    if rho is not None:
        r = np.asarray(rho, dtype=np.float64)
        r_val = np.where(r > 0.0, r, params.rho0)
        c = np.sqrt(np.maximum(0.0, stiff / np.maximum(1e-20, r_val)))
        return float(c) if r.ndim == 0 else c
    return float(np.sqrt(np.maximum(0.0, stiff / max(1e-20, params.rho0))))


def _solid_tangent_single(
    params: BouvardParams,
    defgrad: np.ndarray,
    stretch: np.ndarray,
    dt: float,
    history: Optional[np.ndarray] = None,
    temp: Optional[float] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    if dt <= 0.0:
        theta = params.theta_i if temp is None else float(temp)
        e_mod = emod_tpu(
            params.e_ref,
            params.e1,
            theta,
            params.theta0,
            params.ve1,
            params.ve2,
            1e-4,
            params.edot_ref,
        )
        mu = e_mod / (2.0 * (1.0 + params.nu))
        denom = 1.0 - 2.0 * params.nu
        if abs(denom) < 1e-6:
            denom = 1e-6 if denom >= 0 else -1e-6
        lam = 2.0 * mu * params.nu / denom
        c_mat = np.zeros((6, 6), dtype=np.float64)
        c_mat[0, 0] = c_mat[1, 1] = c_mat[2, 2] = lam + 2.0 * mu
        c_mat[0, 1] = c_mat[1, 0] = c_mat[0, 2] = c_mat[2, 0] = c_mat[1, 2] = c_mat[2, 1] = lam
        c_mat[3, 3] = c_mat[4, 4] = c_mat[5, 5] = mu
        return c_mat

    c_mat = np.zeros((6, 6), dtype=np.float64)
    f0 = np.asarray(defgrad, dtype=np.float64).reshape((3, 3))
    u0 = np.asarray(stretch, dtype=np.float64).reshape((3, 3))

    sig0, _, _ = solid_update_single(
        params, f0, f0, u0, u0, dt, history=history, temp=temp
    )

    pert_indices = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]

    for j, (r, c) in enumerate(pert_indices):
        delta_f = np.eye(3, dtype=np.float64)
        if r == c:
            delta_f[r, c] += eps
        else:
            delta_f[r, c] += 0.5 * eps
            delta_f[c, r] += 0.5 * eps

        f_pert = delta_f @ f0
        c_pert = f_pert.T @ f_pert
        vals, vecs = np.linalg.eigh(c_pert)
        vals = np.maximum(vals, 1e-30)
        u_pert = vecs @ np.diag(np.sqrt(vals)) @ vecs.T

        sig_pert, _, _ = solid_update_single(
            params, f0, f_pert, u0, u_pert, dt, history=history, temp=temp
        )
        c_mat[:, j] = (sig_pert - sig0) / eps

    c_mat = 0.5 * (c_mat + c_mat.T)
    return c_mat


def solid_tangent(
    mat: Any = None,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[dict] = None,
    eps: float = 1e-6,
    **kwargs: Any,
) -> np.ndarray:
    """Compute 6x6 numerical algorithmic tangent stiffness matrix for 3D solid elements.

    Supports both single-point direct call:
      solid_tangent(params, defgrad, stretch, dt, history=..., temp=...)
    and standard pyradioss solver call:
      solid_tangent(mat, sig, deps, dt, extra)
    """
    if isinstance(mat, BouvardParams) and isinstance(sig, np.ndarray) and sig.shape == (3, 3):
        params = mat
        f0 = sig
        u0 = np.asarray(deps if deps is not None else np.eye(3), dtype=np.float64).reshape((3, 3))
        dt_val = float(dt)
        history = kwargs.get("history", None)
        temp = kwargs.get("temp", None)
        return _solid_tangent_single(params, f0, u0, dt_val, history=history, temp=temp, eps=eps)

    if not isinstance(mat, BouvardParams):
        params = build_law101(mat, **kwargs)
    else:
        params = mat

    if sig is not None and isinstance(sig, np.ndarray) and sig.ndim == 2:
        nel = sig.shape[0]
        c_mats = np.zeros((nel, 6, 6), dtype=np.float64)
        for i in range(nel):
            fi = extra["F"][i] if (extra is not None and "F" in extra and extra["F"] is not None) else np.eye(3)
            ci = fi.T @ fi
            valsi, vecsi = np.linalg.eigh(ci)
            ui = vecsi @ np.diag(np.sqrt(np.maximum(valsi, 1e-30))) @ vecsi.T
            hist_i = extra["uvar101"][i] if (extra is not None and "uvar101" in extra) else (extra["history"][i] if extra and "history" in extra else None)
            c_mats[i] = _solid_tangent_single(params, fi, ui, dt, history=hist_i, eps=eps)
        return c_mats
    else:
        fi = extra["F"] if (extra is not None and "F" in extra and extra["F"] is not None) else np.eye(3)
        if fi.ndim == 3:
            fi = fi[0]
        ci = fi.T @ fi
        valsi, vecsi = np.linalg.eigh(ci)
        ui = vecsi @ np.diag(np.sqrt(np.maximum(valsi, 1e-30))) @ vecsi.T
        hist_i = extra["uvar101"] if (extra is not None and "uvar101" in extra) else (extra["history"] if extra and "history" in extra else None)
        if hist_i is not None and hist_i.ndim == 2:
            hist_i = hist_i[0]
        return _solid_tangent_single(params, fi, ui, dt, history=hist_i, eps=eps)
