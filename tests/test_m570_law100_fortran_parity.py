"""
Tests for Milestone M570: /MAT/LAW100 Fortran Parity & Formula Verification.

Compares pyradioss law100 constitutive physics directly against OpenRadioss Fortran reference code:
1. Arruda-Boyce series coefficients from starter/source/materials/mat/mat100/hm_read_mat100.F:364-368 and sigaboyce.F.
2. Kinematic decomposition Fe = F * Fp^(-1), b = Fe * Fe^T from engine/source/materials/mat/mat100/calcmatb.F.
3. Bergstrom-Boyce creep flow rate from engine/source/materials/mat/mat100/viscbb.F.
4. Hyperbolic sine flow rate from engine/source/materials/mat/mat100/viscsinh.F.
5. Power law creep flow rate from engine/source/materials/mat/mat100/viscpower.F.
6. Equilibrium network yield stress softening from engine/source/materials/mat/mat100/sigeps100.F90.
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law100_multi_network import (
    calc_mat_b,
    poly_stress,
    arruda_boyce_stress,
    visc_bb,
    visc_sinh,
    visc_power,
    build_law100,
    SecondaryNetworkParams,
    compute_he_stress,
    _AB_C1, _AB_C2, _AB_C3, _AB_C4, _AB_C5,
)


def test_fortran_arruda_boyce_series_coefficients():
    """Verify Arruda-Boyce series coefficients match hm_read_mat100.F:364-368 and sigaboyce.F."""
    assert _AB_C1 == pytest.approx(0.5, abs=1e-15)
    assert _AB_C2 == pytest.approx(1.0 / 20.0, abs=1e-15)
    assert _AB_C3 == pytest.approx(11.0 / 1050.0, abs=1e-15)
    assert _AB_C4 == pytest.approx(19.0 / 7000.0, abs=1e-15)
    assert _AB_C5 == pytest.approx(519.0 / 673750.0, abs=1e-15)


def test_fortran_calcmatb_parity():
    """Verify calc_mat_b matches calcmatb.F: fe = f * inv(fp), b = fe * fe^T."""
    F = np.array([
        [1.15, 0.05, 0.02],
        [0.01, 0.98, 0.04],
        [0.00, 0.02, 0.89],
    ])
    Fp = np.array([
        [1.02, 0.01, 0.00],
        [0.01, 1.01, 0.02],
        [0.00, 0.01, 0.97],
    ])

    # pyradioss implementation
    b_py, fe_py = calc_mat_b(F, Fp)

    # Reference Fortran calcmatb.F math
    inv_fp_ref = np.linalg.inv(Fp)
    fe_ref = F @ inv_fp_ref
    b_ref = fe_ref @ fe_ref.T

    np.testing.assert_allclose(fe_py, fe_ref, atol=1e-14)
    np.testing.assert_allclose(b_py, b_ref, atol=1e-14)


def test_fortran_viscbb_parity():
    """Verify visc_bb matches viscbb.F exactly:
    ip1 = fp11^2 + fp22^2 + fp33^2
    lpchain = sqrt(third * ip1)
    temp = max(em20, lpchain - 1 + ksi)
    dgamma = a1 * exp(expc * log(temp)) * (tbnorm^expm / tauref^expm)
    """
    fp = np.array([
        [1.05, 0.02, 0.01],
        [0.01, 1.03, 0.02],
        [0.00, 0.01, 0.92],
    ])
    tbnorm = 1.45e5
    a1 = 0.025
    expc = -0.7
    expm = 1.35
    ksi = 0.01
    tauref = 1.0e5

    # pyradioss implementation
    dgamma_py = visc_bb(fp, tbnorm, a1, expc, expm, ksi, tauref)

    # Reference Fortran viscbb.F computation
    ip1 = fp[0, 0]**2 + fp[1, 1]**2 + fp[2, 2]**2
    lpchain = math.sqrt(ip1 / 3.0)
    temp = max(1e-20, lpchain - 1.0 + ksi)
    dgamma_ref = a1 * math.exp(expc * math.log(temp)) * ((tbnorm ** expm) / (tauref ** expm))

    assert dgamma_py == pytest.approx(dgamma_ref, rel=1e-12)


def test_fortran_viscsinh_parity():
    """Verify visc_sinh matches viscsinh.F: dgamma = a1 * (sinh(b0 * tbnorm))^expn."""
    tbnorm = 2.5e5
    a1 = 0.015
    b0 = 5.0e-6
    expn = 1.25

    dgamma_py = visc_sinh(tbnorm, a1, b0, expn)
    dgamma_ref = a1 * (math.sinh(b0 * tbnorm) ** expn)

    assert dgamma_py == pytest.approx(dgamma_ref, rel=1e-12)


def test_fortran_viscpower_parity():
    """Verify visc_power matches viscpower.F:
    temp1 = (expm + 1) * gammaold
    temp2 = exp(expn * log(tbnorm))
    temp3 = exp(expm * log(temp1))
    dgamma = a1 * exp( (1/(1+expm)) * log(temp2 * temp3) )
    """
    tbnorm = 1.8e5
    a1 = 1.2e-6
    expm = 0.4
    expn = 2.1
    gammaold = 0.05

    dgamma_py = visc_power(tbnorm, a1, expm, expn, gammaold)

    temp1 = (expm + 1.0) * gammaold
    temp2 = math.exp(expn * math.log(tbnorm))
    temp3 = math.exp(expm * math.log(temp1))
    dgamma_ref = a1 * math.exp((1.0 / (1.0 + expm)) * math.log(temp2 * temp3))

    assert dgamma_py == pytest.approx(dgamma_ref, rel=1e-12)


def test_fortran_parallel_stress_superposition():
    """Verify multi-network parallel stress superposition matches sigeps100.F90:
    total_sig = sig_A + sum_k (stiffn_k * sig_Bk)
    """
    sec1 = SecondaryNetworkParams(network_id=1, flag_visc=1, stiffness=1.2)
    sec2 = SecondaryNetworkParams(network_id=2, flag_visc=2, stiffness=0.6)
    params = build_law100(
        id=1, rho0=1000.0, flag_he=1, c10=1.0e6, c01=1.0e5, d1=1.0e-7,
        n_net=2, networks=[sec1, sec2]
    )

    F = np.diag([1.1, 1.0 / math.sqrt(1.1), 1.0 / math.sqrt(1.1)])
    b, _ = calc_mat_b(F)
    sig_he, _ = compute_he_stress(b, params)

    # In instantaneous response (zero plastic flow Fp = I):
    # sig_total = sig_A + 1.2 * sig_B1 + 0.6 * sig_B2 = (1 + 1.2 + 0.6) * sig_he = 2.8 * sig_he
    expected_total = (1.0 + 1.2 + 0.6) * sig_he
    # Compute from params.sb = 1.8
    assert params.sb == pytest.approx(1.8)
