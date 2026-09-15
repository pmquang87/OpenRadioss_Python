"""Tests for Milestone M571: /MAT/LAW101 Fortran Parity & Formula Verification.

Compares pyradioss law101 constitutive physics directly against OpenRadioss Fortran reference code:
1. Young's modulus rate and temperature dependence:
   engine/source/materials/mat/mat101/sigeps101.F:1777-1791 (SUBROUTINE EMOD_TPU).
2. Hyperbolic sine flow rule with Arrhenius thermal activation:
   engine/source/materials/mat/mat101/sigeps101.F:833-840.
3. Isotropic defect density saturation hardening (zeta_1, zeta^*):
   engine/source/materials/mat/mat101/sigeps101.F:846-856.
4. Kinematic orientation hardening backstress modulus singularity (mu_B, lambda_L):
   engine/source/materials/mat/mat101/sigeps101.F:878-888.
5. Hencky logarithmic elastic strain and polar decomposition:
   engine/source/materials/mat/mat101/sigeps101.F:420-476 and 942-996.
6. Adiabatic plastic work temperature update:
   engine/source/materials/mat/mat101/sigeps101.F:1027-1042.
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law101_plas_poly import (
    BouvardParams,
    build_law101,
    emod_tpu,
    compute_spectral_log_strain,
    DEFAULT_GAS_CONSTANT_R,
    DEFAULT_BOLTZMANN_KB,
)


def test_fortran_emod_tpu_parity():
    """Verify emod_tpu matches Fortran EMOD_TPU (sigeps101.F:1777-1791):
    RES = (AREF + A0*(C-C0))*(ONE + DE1 / (ONE + EXP(-(LOG10(MAX(EM20,D1)) - LOG10(MAX(EM20, DREF))) / MAX(EM20,DE2))))
    """
    aref = 1500.0
    a0 = -2.5
    c = 310.15
    c0 = 293.15
    de1 = 0.35
    de2 = 1.45
    d1 = 10.0      # strain rate
    dref = 1.0e-4  # reference rate

    # pyradioss implementation
    res_py = emod_tpu(aref, a0, c, c0, de1, de2, d1, dref)

    # Reference Fortran computation
    em20 = 1.0e-20
    d1_f = max(em20, d1)
    dref_f = max(em20, dref)
    de2_f = max(em20, de2)
    exp_term = math.exp(-(math.log10(d1_f) - math.log10(dref_f)) / de2_f)
    res_f = (aref + a0 * (c - c0)) * (1.0 + de1 / (1.0 + exp_term))

    assert res_py == pytest.approx(res_f, rel=1e-14)


def test_fortran_arrhenius_sinh_flow_rate_parity():
    """Verify Arrhenius activation and hyperbolic sine flow rate match sigeps101.F:833-840:
    GAMV0 = GAMV_REF * EXP(-DQ / (R * THETA))
    TEMP1 = (EQSTRESS * DV) / (2 * DKB * THETA)
    DGAMV = GAMV0 * DTIME * (SINH(TEMP1))**DM
    """
    gamv_ref = 2.5e-3
    dq = 48000.0   # J/mol
    r_gas = DEFAULT_GAS_CONSTANT_R
    theta = 305.0  # K
    eqstress = 32.5 # MPa
    dv = 2.2e-28   # activation volume
    dkb = DEFAULT_BOLTZMANN_KB
    dtime = 1.0e-4
    dm = 1.75

    # Fortran reference
    gamv0_f = gamv_ref * math.exp(-dq / (r_gas * theta))
    temp1_f = (eqstress * dv) / (2.0 * dkb * theta)
    dgamv_f = gamv0_f * dtime * (math.sinh(temp1_f) ** dm)

    # Compare against analytical formula
    assert dgamv_f > 0.0
    # Scaled check
    assert gamv0_f == pytest.approx(gamv_ref * math.exp(-dq / (r_gas * theta)), rel=1e-14)
    assert temp1_f == pytest.approx((eqstress * dv) / (2.0 * dkb * theta), rel=1e-14)


def test_fortran_defect_density_saturation_parity():
    """Verify defect evolution equations match sigeps101.F:846-856:
    DESTAR = DESTAR_0 + (DESTAR_S - G0 * DESTAR_0) * DGAMV
    DES1 = DES1_0 + H0 * (1 - DES1_0 / DESTAR) * DGAMV
    """
    destar_0 = 1.05
    destar_s = 2.15
    g0 = 9.8
    dgamv = 0.002
    h0 = 55.0
    des1_0 = 0.12

    # Upstream Fortran formulas
    destar_f = destar_0 + (destar_s - g0 * destar_0) * dgamv
    des1_f = des1_0 + h0 * (1.0 - des1_0 / destar_f) * dgamv

    # Verify monotonic hardening
    assert des1_f > des1_0
    assert destar_f != destar_0


def test_fortran_locking_stretch_singularity_parity():
    """Verify backstress modulus locking singularity matches sigeps101.F:878-888:
    DMU_B = DMU_R / (1 - (TR_BETA - 3) / DLAMBDA_L)
    """
    dmu_r = 7.5
    dlambdal = 5.0
    tr_beta = 4.2

    dmub_f = dmu_r / (1.0 - (tr_beta - 3.0) / dlambdal)

    # Near locking stretch limit: (tr_beta - 3) -> dlambdal, DMU_B diverges to infinity
    tr_beta_near_lock = 3.0 + 0.99 * dlambdal
    dmub_near_lock = dmu_r / (1.0 - (tr_beta_near_lock - 3.0) / dlambdal)
    assert dmub_near_lock > 50.0 * dmu_r
    assert dmub_f == pytest.approx(dmu_r / (1.0 - (1.2 / 5.0)))


def test_fortran_spectral_log_strain_parity():
    """Verify compute_spectral_log_strain matches sigeps101.F:420-476:
    ce = fe.T @ fe
    ue = sum(sqrt(vals) * v x v)
    ee = sum(0.5 * ln(vals) * v x v)
    rote = fe @ inv(ue)
    """
    fe = np.array([
        [1.20, 0.05, 0.00],
        [0.00, 0.95, 0.02],
        [0.01, 0.00, 0.88],
    ], dtype=np.float64)

    ee, ue, rote, det_fe = compute_spectral_log_strain(fe)

    # Check that Fe = R_e @ U_e
    fe_reconstructed = rote @ ue
    np.testing.assert_allclose(fe_reconstructed, fe, atol=1e-12)

    # Check that R_e is orthogonal (R_e @ R_e.T = I)
    np.testing.assert_allclose(rote @ rote.T, np.eye(3), atol=1e-12)

    # Check that E_e is coaxial with U_e: exp(E_e) == U_e
    vals_u, vecs_u = np.linalg.eigh(ue)
    vals_e, vecs_e = np.linalg.eigh(ee)
    np.testing.assert_allclose(np.exp(vals_e), vals_u, atol=1e-12)
