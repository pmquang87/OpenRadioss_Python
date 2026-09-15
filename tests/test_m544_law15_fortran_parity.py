"""
Fortran Parity Verification Tests for Milestone M544:
/MAT/LAW15, /MAT/CHANG, /MAT/PLAS_ANISO, /MAT/COMP_CHANG.

Fortran source code references:
  - starter/source/materials/mat/mat015/hm_read_mat15.F
  - engine/source/materials/mat/mat015/sigeps15c.F
  - engine/source/materials/mat/mat015/m15cplrc.F
  - engine/source/materials/mat/mat015/m15crak.F
  - hm_cfg_files/config/CFG/radioss110/MAT/matl15_chang.cfg

Auditor 2A Verification Tasks:
  1. Orthotropic plane stress elasticity:
     C11 = E1 / (1 - nu12*nu21), C22 = E2 / (1 - nu12*nu21),
     C12 = nu12*E2 / (1 - nu12*nu21), C33 = G12.
  2. Sound speed:
     c = sqrt(max(C1, Gmax) / rho0) where C1 = max(E1, E2) / (1 - nu12*nu21),
     Gmax = max(G12, G23, G31).
  3. Tsai-Wu plasticity (m15cplrc.F):
     F1 = 1/sigyt1 - 1/sigyc1, F2 = 1/sigyt2 - 1/sigyc2,
     F11 = 1/(sigyt1*sigyc1), F22 = 1/(sigyt2*sigyc2), F33 = 1/(s12^2),
     F12 = -0.5 * alpha * sqrt(F11*F22).
     W_vec = F1*s1 + F2*s2 + F11*s1^2 + F22*s2^2 + 2*F12*s1*s2 + F33*s12^2.
     f_yld = min(fmax, (1 + b * wpla^n) * epspfac).
     epspfac with log law and Cowper-Symonds.
     Plastic normal and return.
  4. Chang-Chang failure criteria (m15crak.F):
     - Fiber tension: ef2 = (s11/s1)^2 + beta * (s12/s12_str)^2 >= 1.
     - Fiber compression: efc2 = (s11/c1)^2 >= 1.
     - Matrix tension: em2 = (s22/c2)^2 + (s12/s12_str)^2 >= 1.
     - Matrix compression: emc2 = (s22 / (2*s12_str))^2 + (s12/s12_str)^2
                                   + (s22/c2) * ((c2/(2*s12_str))^2 - 1) >= 1.
  5. Post-failure exponential stress relaxation:
     dam = exp(-(time - tfail)/tmax); if dam < 0.01: dam = 0.0.
     Matrix failure: s22, s12, s23, s31 scaled by dam.
     Fiber failure: all stresses scaled by dam.
  6. Element deletion and itype handling (itype = 0..6).
  7. 5-component transverse shear stress support.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials import law15_chang


# ============================================================================
# 1. Orthotropic Plane Stress Elasticity Parity
# ============================================================================

def test_orthotropic_plane_stress_elastic_matrix():
    """Verify C11 = E1 / (1 - nu12*nu21), C22 = E2 / (1 - nu12*nu21),
    C12 = nu12*E2 / (1 - nu12*nu21), C33 = G12, and symmetry C12 == C21."""
    E1 = 150000.0
    E2 = 10000.0
    nu12 = 0.3
    G12 = 5000.0

    nu21 = nu12 * E2 / E1
    detc = 1.0 - nu12 * nu21

    expected_c11 = E1 / detc
    expected_c22 = E2 / detc
    expected_c12 = nu12 * E2 / detc
    expected_c33 = G12

    mat = law15_chang.build_law15(
        E1=E1, E2=E2, nu12=nu12, G12=G12,
        rho0=1.5e-9,
        sigyt1=1e10, sigyt2=1e10, sigt12=1e10,
    )

    C_mat = law15_chang.shell_membrane_tangent(mat)

    assert C_mat[0, 0] == pytest.approx(expected_c11, rel=1e-6)
    assert C_mat[1, 1] == pytest.approx(expected_c22, rel=1e-6)
    assert C_mat[0, 1] == pytest.approx(expected_c12, rel=1e-6)
    assert C_mat[1, 0] == pytest.approx(expected_c12, rel=1e-6)
    assert C_mat[2, 2] == pytest.approx(expected_c33, rel=1e-6)
    assert C_mat[0, 2] == 0.0
    assert C_mat[1, 2] == 0.0


def test_elastic_predictor_stress_response():
    """Verify incremental Hooke update matches C_mat @ deps exactly in intact state."""
    E1 = 120000.0
    E2 = 8000.0
    nu12 = 0.25
    G12 = 4000.0

    mat = law15_chang.build_law15(
        E1=E1, E2=E2, nu12=nu12, G12=G12,
        rho0=1.5e-9,
        sigyt1=1e10, sigyt2=1e10, sigt12=1e10,
    )

    C_mat = law15_chang.shell_membrane_tangent(mat)
    deps = np.array([1.2e-4, -0.8e-4, 2.5e-4])

    expected_sig = C_mat @ deps
    sig, epsp, _ = law15_chang.shell_update(mat, np.zeros(3), deps)

    assert sig[0] == pytest.approx(expected_sig[0], rel=1e-6)
    assert sig[1] == pytest.approx(expected_sig[1], rel=1e-6)
    assert sig[2] == pytest.approx(expected_sig[2], rel=1e-6)
    assert epsp == 0.0


# ============================================================================
# 2. Sound Speed Parity (hm_read_mat15.F:266-269)
# ============================================================================

def test_sound_speed_c1_dominates():
    """Verify c = sqrt(max(C1, Gmax) / rho0) when C1 > Gmax."""
    E1 = 160000.0
    E2 = 12000.0
    nu12 = 0.3
    G12 = 4000.0
    G23 = 3000.0
    G31 = 4500.0
    rho0 = 1.6e-9

    nu21 = nu12 * E2 / E1
    detc = 1.0 - nu12 * nu21
    C1 = max(E1, E2) / detc
    Gmax = max(G12, G23, G31)
    assert C1 > Gmax

    expected_c = math.sqrt(C1 / rho0)
    computed_c = law15_chang.sound_speed_law15(E1, E2, nu12, G12, G23, G31, rho0)
    assert computed_c == pytest.approx(expected_c, rel=1e-7)


def test_sound_speed_gmax_dominates():
    """Verify c = sqrt(Gmax / rho0) when Gmax > C1 (unusual composite regime)."""
    E1 = 1000.0
    E2 = 1000.0
    nu12 = 0.1
    G12 = 2000.0  # G12 > C1
    G23 = 1500.0
    G31 = 1000.0
    rho0 = 1.0e-9

    Gmax = max(G12, G23, G31)
    expected_c = math.sqrt(Gmax / rho0)
    computed_c = law15_chang.sound_speed_law15(E1, E2, nu12, G12, G23, G31, rho0)
    assert computed_c == pytest.approx(expected_c, rel=1e-7)


# ============================================================================
# 3. Tsai-Wu Plasticity Formulations (m15cplrc.F)
# ============================================================================

def test_tsai_wu_coefficients_exact():
    """Verify Tsai-Wu coefficients against analytical formulas:
      F1 = 1/sigyt1 - 1/sigyc1, F2 = 1/sigyt2 - 1/sigyc2,
      F11 = 1/(sigyt1*sigyc1), F22 = 1/(sigyt2*sigyc2), F33 = 1/(s12^2),
      F12 = -0.5 * alpha * sqrt(F11*F22).
    """
    sigyt1 = 1500.0
    sigyc1 = 1000.0
    sigyt2 = 50.0
    sigyc2 = 150.0
    s12 = 70.0
    alpha = 0.85

    expected_F1 = 1.0 / sigyt1 - 1.0 / sigyc1
    expected_F2 = 1.0 / sigyt2 - 1.0 / sigyc2
    expected_F11 = 1.0 / (sigyt1 * sigyc1)
    expected_F22 = 1.0 / (sigyt2 * sigyc2)
    expected_F33 = 1.0 / (s12 ** 2)
    expected_F12 = -0.5 * alpha * math.sqrt(expected_F11 * expected_F22)

    coeffs = law15_chang.tsai_wu_coefficients(
        sigyt1, sigyc1, sigyt2, sigyc2, s12, alpha=alpha
    )

    assert coeffs["F1"] == pytest.approx(expected_F1, rel=1e-8)
    assert coeffs["F2"] == pytest.approx(expected_F2, rel=1e-8)
    assert coeffs["F11"] == pytest.approx(expected_F11, rel=1e-8)
    assert coeffs["F22"] == pytest.approx(expected_F22, rel=1e-8)
    assert coeffs["F33"] == pytest.approx(expected_F33, rel=1e-8)
    assert coeffs["F12"] == pytest.approx(expected_F12, rel=1e-8)


def test_tsai_wu_yield_criterion_eval():
    """Verify W_vec = F1*s1 + F2*s2 + F11*s1^2 + F22*s2^2 + 2*F12*s1*s2 + F33*s12^2."""
    F1 = -0.0003
    F2 = 0.015
    F11 = 8.0e-7
    F22 = 1.5e-4
    F33 = 2.0e-4
    F12 = -3.0e-6

    s1 = 800.0
    s2 = 30.0
    s12 = 45.0

    expected_W = (
        F1 * s1
        + F2 * s2
        + F11 * (s1 ** 2)
        + F22 * (s2 ** 2)
        + F33 * (s12 ** 2)
        + 2.0 * F12 * s1 * s2
    )

    computed_W = law15_chang.tsai_wu_yield_criterion(
        s1, s2, s12, F1, F2, F11, F22, F33, F12
    )
    assert computed_W == pytest.approx(expected_W, rel=1e-8)


def test_hardening_yield_eval():
    """Verify flow stress f_yld = min(fmax, (1 + b * wpla^n) * epspfac)."""
    b = 50.0
    n = 0.6
    wpla = 0.05
    epspfac = 1.15
    fmax = 1.8

    expected_fyld = min(fmax, (1.0 + b * (wpla ** n)) * epspfac)
    computed_fyld = law15_chang.hardening_yield(wpla, b, n, epspfac=epspfac, fmax=fmax)
    assert computed_fyld == pytest.approx(expected_fyld, rel=1e-8)


def test_strain_rate_factor_log_and_power():
    """Verify strain rate factor:
      Log law (Fortran m15cplrc.F:177-186): epspfac = 1 + c * log(eps_dot / epdr)
      Cowper-Symonds: epspfac = 1 + (eps_dot / c)^(1/epdr) if strflag != 0 else 1.0.
    """
    eps_dot = 100.0
    c = 0.08
    epdr = 1.0

    # 1. Fortran log law
    expected_log = 1.0 + c * math.log(eps_dot / epdr)
    computed_log = law15_chang.strain_rate_factor(eps_dot, c=c, epdr=epdr, formulation="log")
    assert computed_log == pytest.approx(expected_log, rel=1e-7)

    # 2. Cowper-Symonds power law
    expected_cs = 1.0 + (eps_dot / c) ** (1.0 / epdr)
    computed_cs = law15_chang.strain_rate_factor(
        eps_dot, c=c, epdr=epdr, formulation="cowper_symonds"
    )
    assert computed_cs == pytest.approx(expected_cs, rel=1e-7)

    # 3. strflag == 0 gives 1.0
    assert law15_chang.strain_rate_factor(eps_dot, c=c, epdr=epdr, strflag=0) == 1.0

    # 4. eps_dot <= epdr gives 1.0 in log law
    assert law15_chang.strain_rate_factor(0.5, c=c, epdr=1.0, formulation="log") == 1.0


# ============================================================================
# 4. Chang-Chang Failure Criteria Parity (m15crak.F)
# ============================================================================

def test_chang_chang_fiber_tension():
    """Verify fiber tension mode: ef2 = (s11/s1)^2 + beta * (s12/s12_str)^2 >= 1."""
    s1 = 1800.0
    s12_str = 70.0
    beta_s = 0.9

    mat = law15_chang.build_law15(
        E1=150000.0, E2=10000.0, nu12=0.3, G12=4500.0,
        s1=s1, s12=s12_str, beta_s=beta_s,
        sigyt1=1e10, sigyt2=1e10, sigt12=1e10,
        tmax=0.01, itype=0,
    )

    # Point 1: Below failure threshold (ef2 < 1.0)
    s11_sub = 1200.0
    s12_sub = 30.0
    ef2_sub = (s11_sub / s1) ** 2 + beta_s * (s12_sub / s12_str) ** 2
    assert ef2_sub < 1.0

    detc = 1.0 - 0.3 * (0.3 * 10000.0 / 150000.0)
    A11 = 150000.0 / detc
    deps_sub = np.array([s11_sub / A11, 0.0, s12_sub / 4500.0])
    extra_sub = {}
    sig_sub, _, _ = law15_chang.shell_update(mat, np.zeros(3), deps_sub, extra=extra_sub)
    assert extra_sub["damt15"][0, 0] == 1.0  # Intact

    # Point 2: Above failure threshold (ef2 >= 1.0)
    s11_fail = 1600.0
    s12_fail = 55.0
    ef2_fail = (s11_fail / s1) ** 2 + beta_s * (s12_fail / s12_str) ** 2
    assert ef2_fail >= 1.0

    deps_fail = np.array([s11_fail / A11, 0.0, s12_fail / 4500.0])
    extra_fail = {}
    sig_fail, _, _ = law15_chang.shell_update(mat, np.zeros(3), deps_fail, extra=extra_fail)
    assert extra_fail["damt15"][0, 0] < 1.0  # Fiber damaged!


def test_chang_chang_fiber_compression():
    """Verify fiber compression mode: efc2 = (s11/c1)^2 >= 1 when s11 <= 0."""
    c1 = 1200.0
    mat = law15_chang.build_law15(
        E1=150000.0, E2=10000.0, nu12=0.3, G12=4500.0,
        c1=c1,
        sigyt1=1e10, sigyt2=1e10, sigt12=1e10,
        tmax=0.01, itype=0,
    )

    detc = 1.0 - 0.3 * (0.3 * 10000.0 / 150000.0)
    A11 = 150000.0 / detc

    # Under threshold
    extra_sub = {}
    law15_chang.shell_update(mat, np.zeros(3), np.array([-1000.0 / A11, 0.0, 0.0]), extra=extra_sub)
    assert extra_sub["damt15"][0, 0] == 1.0

    # Over threshold
    extra_fail = {}
    law15_chang.shell_update(mat, np.zeros(3), np.array([-1300.0 / A11, 0.0, 0.0]), extra=extra_fail)
    assert extra_fail["damt15"][0, 0] < 1.0


def test_chang_chang_matrix_tension():
    """Verify matrix tension mode: em2 = (s22/c2)^2 + (s12/s12_str)^2 >= 1 when s22 >= 0."""
    c2 = 50.0
    s12_str = 60.0
    mat = law15_chang.build_law15(
        E1=150000.0, E2=10000.0, nu12=0.3, G12=4500.0,
        c2=c2, s12=s12_str, s1=1e10,
        sigyt1=1e10, sigyt2=1e10, sigt12=1e10,
        tmax=0.01, itype=0,
    )

    detc = 1.0 - 0.3 * (0.3 * 10000.0 / 150000.0)
    A22 = 10000.0 / detc

    # Matrix tension failure with combined tension and shear
    s22 = 40.0
    s12 = 45.0
    em2 = (s22 / c2) ** 2 + (s12 / s12_str) ** 2
    assert em2 > 1.0  # (40/50)^2 + (45/60)^2 = 0.64 + 0.5625 = 1.2025 >= 1

    extra = {}
    law15_chang.shell_update(mat, np.zeros(3), np.array([0.0, s22 / A22, s12 / 4500.0]), extra=extra)
    assert extra["damt15"][0, 0] == 1.0  # Fiber intact
    assert extra["damt15"][0, 1] < 1.0   # Matrix cracked!


def test_chang_chang_matrix_compression():
    """Verify matrix compression mode:
      emc2 = (s22 / (2*s12_str))^2 + (s12/s12_str)^2 + (s22/c2) * ((c2/(2*s12_str))^2 - 1) >= 1
      when s22 < 0.
    """
    c2 = 160.0
    s12_str = 60.0
    mat = law15_chang.build_law15(
        E1=150000.0, E2=10000.0, nu12=0.3, G12=4500.0,
        c2=c2, s12=s12_str, s1=1e10,
        sigyt1=1e10, sigyt2=1e10, sigt12=1e10,
        tmax=0.01, itype=0,
    )

    detc = 1.0 - 0.3 * (0.3 * 10000.0 / 150000.0)
    A22 = 10000.0 / detc

    s22 = -180.0
    s12 = 20.0
    term1 = (s22 / (2.0 * s12_str)) ** 2
    term2 = (s12 / s12_str) ** 2
    ratio = (c2 / (2.0 * s12_str)) ** 2 - 1.0
    term3 = (s22 / c2) * ratio
    emc2 = term1 + term2 + term3
    assert emc2 >= 1.0

    extra = {}
    law15_chang.shell_update(mat, np.zeros(3), np.array([0.0, s22 / A22, s12 / 4500.0]), extra=extra)
    assert extra["damt15"][0, 0] == 1.0
    assert extra["damt15"][0, 1] < 1.0


# ============================================================================
# 5. Post-Failure Exponential Relaxation Parity
# ============================================================================

def test_matrix_failure_relaxation_stresses():
    """Verify that when matrix fails:
      dam = exp(-(time - tfail)/tmax).
      s22, s12 are scaled by dam.
      s11 remains unaffected (carried by intact fibers).
    """
    tmax = 0.004
    mat = law15_chang.build_law15(
        E1=100000.0, E2=10000.0, nu12=0.3, G12=4000.0,
        c2=50.0, s1=1e10,
        sigyt1=1e10, sigyt2=1e10, sigt12=1e10,
        tmax=tmax, itype=0,
    )

    detc = 1.0 - 0.3 * (0.3 * 10000.0 / 100000.0)
    A11 = 100000.0 / detc
    A22 = 10000.0 / detc

    # Step 1: initiate matrix failure at t = 0.001
    extra = {"time": 0.001}
    deps = np.array([500.0 / A11, 70.0 / A22, 20.0 / 4000.0])
    sig1, _, _ = law15_chang.shell_update(mat, np.zeros(3), deps, extra=extra)

    assert extra["damt15"][0, 0] == 1.0  # Fiber intact
    assert extra["damt15"][0, 1] < 1.0   # Matrix failed
    initial_s11 = sig1[0]
    initial_s22 = sig1[1]

    # Step 2: advance to t = 0.001 + tmax = 0.005 s
    extra["time"] = 0.005
    sig2, _, _ = law15_chang.shell_update(mat, sig1, np.zeros(3), extra=extra)

    expected_dam = math.exp(-1.0)
    assert extra["damt15"][0, 1] == pytest.approx(expected_dam, rel=1e-3)
    # Transverse stress decays with dam
    assert sig2[1] == pytest.approx(initial_s22 * expected_dam, rel=1e-3)
    # Fiber direction s11 is NOT decayed by matrix failure!
    assert sig2[0] == pytest.approx(initial_s11, rel=1e-3)

    # Step 3: advance far beyond (> 5*tmax), threshold < 0.01 zeroes out dam
    extra["time"] = 0.05
    sig3, _, _ = law15_chang.shell_update(mat, sig2, np.zeros(3), extra=extra)
    assert extra["damt15"][0, 1] == 0.0
    assert sig3[1] == 0.0
    assert sig3[2] == 0.0
    # s11 remains intact
    assert sig3[0] == pytest.approx(initial_s11, rel=1e-3)


def test_fiber_failure_relaxation_all_stresses():
    """Verify that when fiber fails, ALL stresses (s11, s22, s12) are scaled by dam."""
    tmax = 0.002
    mat = law15_chang.build_law15(
        E1=100000.0, E2=10000.0, nu12=0.3, G12=4000.0,
        s1=800.0,
        sigyt1=1e10, sigyt2=1e10, sigt12=1e10,
        tmax=tmax, itype=0,
    )

    detc = 1.0 - 0.3 * (0.3 * 10000.0 / 100000.0)
    A11 = 100000.0 / detc

    extra = {"time": 0.001}
    deps = np.array([1000.0 / A11, 20.0 / 10000.0, 10.0 / 4000.0])
    sig1, _, _ = law15_chang.shell_update(mat, np.zeros(3), deps, extra=extra)

    assert extra["damt15"][0, 0] < 1.0
    initial_s11 = sig1[0]
    initial_s22 = sig1[1]

    # Advance by tmax
    extra["time"] = 0.003
    sig2, _, _ = law15_chang.shell_update(mat, sig1, np.zeros(3), extra=extra)

    expected_dam = math.exp(-1.0)
    assert extra["damt15"][0, 0] == pytest.approx(expected_dam, rel=1e-3)
    assert sig2[0] == pytest.approx(initial_s11 * expected_dam, rel=1e-3)
    assert sig2[1] == pytest.approx(initial_s22 * expected_dam, rel=1e-3)

    # Advance to full relaxation
    extra["time"] = 0.03
    sig3, _, _ = law15_chang.shell_update(mat, sig2, np.zeros(3), extra=extra)
    assert extra["damt15"][0, 0] == 0.0
    assert np.all(sig3 == 0.0)


# ============================================================================
# 6. Element Deletion & Itype Handling (IOFF cases 0 to 6)
# ============================================================================

@pytest.mark.parametrize("itype", [0, 1, 2, 3, 4, 6])
def test_itype_wpmax_deletion(itype: int):
    """Verify that exceeding wpmax deletes the element for all itype options."""
    mat = law15_chang.build_law15(
        E1=50000.0, E2=5000.0, nu12=0.3, G12=2000.0,
        sigyt1=100.0, sigyt2=30.0, sigt12=20.0,
        wpmax=2.0, wpref=1.0,
        s1=1e10, s2=1e10, c1=1e10, c2=1e10, s12=1e10,
        itype=itype,
    )
    extra = {}
    # Large strain inducing plastic work > 2.0
    deps = np.array([0.0, 0.15, 0.0])
    sig, epsp, _ = law15_chang.shell_update(mat, np.zeros(3), deps, extra=extra)

    assert extra["off15"][0] == 0.0
    assert np.all(sig == 0.0)


def test_itype_2_fiber_tension_deletion():
    """itype=2: shell deleted if Wp >= Wpmax OR tensile failure in dir 1."""
    mat = law15_chang.build_law15(
        E1=100000.0, E2=10000.0, nu12=0.3, G12=4000.0,
        s1=1000.0,  # Tensile fiber strength
        sigyt1=1e10, sigyt2=1e10, sigt12=1e10,
        itype=2,
    )
    detc = 1.0 - 0.3 * (0.3 * 10000.0 / 100000.0)
    A11 = 100000.0 / detc

    extra = {}
    sig, _, _ = law15_chang.shell_update(mat, np.zeros(3), np.array([1200.0 / A11, 0.0, 0.0]), extra=extra)

    assert extra["off15"][0] == 0.0
    assert np.all(sig == 0.0)


def test_itype_3_matrix_tension_deletion():
    """itype=3: shell deleted if Wp >= Wpmax OR tensile failure in dir 2."""
    mat = law15_chang.build_law15(
        E1=100000.0, E2=10000.0, nu12=0.3, G12=4000.0,
        c2=60.0, s1=1e10,  # Tensile matrix strength
        sigyt1=1e10, sigyt2=1e10, sigt12=1e10,
        itype=3,
    )
    detc = 1.0 - 0.3 * (0.3 * 10000.0 / 100000.0)
    A22 = 10000.0 / detc

    extra = {}
    sig, _, _ = law15_chang.shell_update(mat, np.zeros(3), np.array([0.0, 80.0 / A22, 0.0]), extra=extra)

    assert extra["off15"][0] == 0.0
    assert np.all(sig == 0.0)


def test_itype_0_no_deletion_on_cracking():
    """itype=0: element is NOT deleted on cracking alone, stresses relax exponentially."""
    mat = law15_chang.build_law15(
        E1=100000.0, E2=10000.0, nu12=0.3, G12=4000.0,
        s1=1000.0,
        sigyt1=1e10, sigyt2=1e10, sigt12=1e10,
        tmax=0.01,
        itype=0,
    )
    detc = 1.0 - 0.3 * (0.3 * 10000.0 / 100000.0)
    A11 = 100000.0 / detc

    extra = {"time": 0.001}
    sig, _, _ = law15_chang.shell_update(mat, np.zeros(3), np.array([1200.0 / A11, 0.0, 0.0]), extra=extra)

    # itype=0 does not delete on fiber failure; off15 remains 1.0
    assert extra["off15"][0] == 1.0
    assert extra["damt15"][0, 0] < 1.0
    assert sig[0] > 0.0


# ============================================================================
# 7. 5-Component Transverse Shear Support Parity (sigeps15c.F)
# ============================================================================

def test_five_component_stress_update():
    """Verify that 5-component [s11, s22, s12, s23, s31] updates correctly
    with transverse shear moduli G23 and G31, and decays under matrix failure."""
    E1 = 120000.0
    E2 = 8000.0
    nu12 = 0.3
    G12 = 4000.0
    G23 = 2500.0
    G31 = 3500.0
    c2 = 50.0

    mat = law15_chang.build_law15(
        E1=E1, E2=E2, nu12=nu12, G12=G12, G23=G23, G31=G31,
        c2=c2, s1=1e10,
        sigyt1=1e10, sigyt2=1e10, sigt12=1e10,
        tmax=0.005, itype=0,
    )

    detc = 1.0 - 0.3 * (0.3 * 8000.0 / 120000.0)
    A11 = 120000.0 / detc
    A22 = 8000.0 / detc

    deps_5 = np.array([100.0 / A11, 60.0 / A22, 10.0 / G12, 1.0e-3, 2.0e-3])

    # Step 1: matrix cracking occurs at t = 0.001
    extra = {"time": 0.001}
    sig1, _, _ = law15_chang.shell_update(mat, np.zeros(5), deps_5, extra=extra)

    assert sig1.shape == (5,)
    assert sig1[3] == pytest.approx(G23 * 1.0e-3)
    assert sig1[4] == pytest.approx(G31 * 2.0e-3)
    assert extra["damt15"][0, 1] < 1.0

    # Step 2: advance time by tmax -> transverse shears s23 and s31 decay by exp(-1)
    extra["time"] = 0.006
    sig2, _, _ = law15_chang.shell_update(mat, sig1, np.zeros(5), extra=extra)

    expected_dam = math.exp(-1.0)
    assert sig2[3] == pytest.approx(sig1[3] * expected_dam, rel=1e-3)
    assert sig2[4] == pytest.approx(sig1[4] * expected_dam, rel=1e-3)


# ============================================================================
# 8. Consistent Algorithmic Tangent Major Symmetry
# ============================================================================

def test_consistent_tangent_major_symmetry():
    """Verify that consistent plane-stress tangent satisfies major symmetry."""
    mat = law15_chang.build_law15(
        E1=140000.0, E2=9000.0, nu12=0.28, G12=4200.0,
        sigyt1=600.0, sigyc1=500.0, sigyt2=60.0, sigyc2=80.0,
        sigt12=45.0, sigc12=45.0,
        b=20.0, n=0.8,
        s1=1e10, s2=1e10, c1=1e10, c2=1e10, s12=1e10,
    )

    # Tangent at plastic yield state with symmetric=True
    sig_yield = np.array([400.0, 40.0, 20.0])
    c_tan = law15_chang.consistent_shell_tangent(mat, sig_yield, symmetric=True)

    assert np.allclose(c_tan, c_tan.T, atol=1e-10)
