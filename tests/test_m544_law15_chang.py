"""
Tests for LAW15 — Chang-Chang Composite Material Law
(/MAT/LAW15, /MAT/CHANG, /MAT/PLAS_ANISO, /MAT/COMP_CHANG).
Milestone M544: LAW15 Kernel Builder.

Fortran references:
  - starter/source/materials/mat/mat015/hm_read_mat15.F
  - engine/source/materials/mat/mat015/sigeps15c.F
  - engine/source/materials/mat/mat015/m15cplrc.F
  - engine/source/materials/mat/mat015/m15crak.F
  - hm_cfg_files/config/CFG/radioss110/MAT/matl15_chang.cfg

Verification coverage:
  1. Parameter parsing, defaults, fallbacks, and Tsai-Wu coefficients.
  2. Sound speed calculation and materials package dispatch.
  3. Persistent extra state shapes (damt15, sigr15, wpla15, off15).
  4. solid_update rejection (NotImplementedError: shell only).
  5. Elastic plane stress response (Hooke's law in 1, 2, and 12 shear).
  6. Tsai-Wu plasticity yield criterion and normal flow return (m15cplrc.F).
  7. Strain rate enhancement (logarithmic rate law with src and srp).
  8. Chang-Chang failure criteria (m15crak.F):
     - Tensile fiber mode
     - Compressive fiber mode
     - Tensile matrix mode
     - Compressive matrix mode
  9. Post-failure relaxation dynamics (exponential decay with Tmax).
  10. Element deletion & layer failure logic (itype / ioff and wpmax).
  11. Consistent plane-stress algorithmic tangent (consistent_shell_tangent).
  12. Materials registry and package level dispatch.
"""

from __future__ import annotations

import math
from typing import Any, Dict
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.materials import law15_chang
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY


# ============================================================================
# 1. Parameter Constructor build_law15 & Tsai-Wu Coefficients
# ============================================================================

def test_law15_constructor_and_defaults():
    """Verify default constructor parameters and derived compliance / sound speed."""
    mat = law15_chang.build_law15(
        id=15,
        rho0=1.6e-9,
        E1=150000.0,
        E2=9000.0,
        nu12=0.32,
        G12=4500.0,
        G23=3000.0,
        G31=4500.0,
        sigyt1=1800.0,
        sigyc1=1200.0,
        sigyt2=40.0,
        sigyc2=160.0,
        sigt12=60.0,
        sigc12=60.0,
        alpha=1.0,
        b=100.0,
        n=0.5,
        fmax=2000.0,
        wpmax=50.0,
        wpref=1.0,
        itype=2,
        src=0.05,
        srp=1.0,
        strflag=1,
        beta_s=0.8,
        tmax=0.01,
        s1=2000.0,
        s2=50.0,
        s12=70.0,
        c1=1400.0,
        c2=180.0,
    )
    assert mat.id == 15
    assert mat.law == 15
    assert mat.rho0 == 1.6e-9
    assert mat.title == "LAW15_CHANG"

    p = mat.params
    assert p["E1"] == 150000.0
    assert p["E2"] == 9000.0
    assert p["nu12"] == 0.32
    assert p["G12"] == 4500.0
    assert p["G23"] == 3000.0
    assert p["G31"] == 4500.0

    # nu21 = nu12 * E2 / E1
    expected_nu21 = 0.32 * 9000.0 / 150000.0
    assert p["nu21"] == pytest.approx(expected_nu21)

    expected_detc = 1.0 - 0.32 * expected_nu21
    assert p["detc"] == pytest.approx(expected_detc)

    expected_c1 = 150000.0 / expected_detc
    assert p["C1"] == pytest.approx(expected_c1)

    # Sound speed: sqrt(max(C1, Gmax) / rho0)
    expected_ssp = math.sqrt(expected_c1 / 1.6e-9)
    assert p["ssp"] == pytest.approx(expected_ssp)
    assert law15_chang.sound_speed(mat) == pytest.approx(expected_ssp)


def test_law15_fallbacks():
    """Verify fallback values when optional strengths are omitted."""
    mat = law15_chang.build_law15(
        id=1,
        rho0=1.5e-9,
        E1=100000.0,
        E2=10000.0,
        nu12=0.3,
        G12=4000.0,
        G23=3000.0,
        G31=4000.0,
        sigyt1=1000.0,
        sigyt2=50.0,
        sigt12=40.0,
        # Omit sigyc1, sigyc2, sigc12, and Chang-Chang strengths s1, s2, c1, c2, s12
    )
    p = mat.params
    # Compressive yield defaults to tensile yield
    assert p["sigyc1"] == 1000.0
    assert p["sigyc2"] == 50.0
    # Shear compressive yield defaults to shear tensile yield
    assert p["sigc12"] == 40.0
    # Chang-Chang failure strengths default to corresponding yield strengths
    assert p["s1"] == 1000.0
    assert p["s2"] == 50.0
    assert p["c1"] == 1000.0
    assert p["c2"] == 50.0
    assert p["s12"] == 40.0


def test_law15_tsai_wu_coefficients():
    """Verify Tsai-Wu coefficients formulas against manual arithmetic."""
    coeffs = law15_chang.tsai_wu_coefficients(
        sigyt1=1000.0,
        sigyc1=800.0,
        sigyt2=50.0,
        sigyc2=100.0,
        sig12=(40.0, 40.0),
        alpha=1.0,
    )
    # F1 = 1/1000 - 1/800 = 0.001 - 0.00125 = -0.00025
    assert coeffs["F1"] == pytest.approx(1.0 / 1000.0 - 1.0 / 800.0)
    # F2 = 1/50 - 1/100 = 0.02 - 0.01 = 0.01
    assert coeffs["F2"] == pytest.approx(1.0 / 50.0 - 1.0 / 100.0)
    # F11 = 1/(1000*800) = 1.25e-6
    assert coeffs["F11"] == pytest.approx(1.0 / 800000.0)
    # F22 = 1/(50*100) = 2.0e-4
    assert coeffs["F22"] == pytest.approx(1.0 / 5000.0)
    # F33 = 1/(40*40) = 6.25e-4
    assert coeffs["F33"] == pytest.approx(1.0 / 1600.0)
    # F12 = -0.5 * alpha * sqrt(F11 * F22)
    expected_f12 = -0.5 * 1.0 * math.sqrt((1.0 / 800000.0) * (1.0 / 5000.0))
    assert coeffs["F12"] == pytest.approx(expected_f12)


# ============================================================================
# 2. Sound Speed and Extra Shapes
# ============================================================================

def test_law15_sound_speed():
    """Verify sound speed computation with custom and default density."""
    mat = law15_chang.build_law15(
        E1=140000.0,
        E2=10000.0,
        nu12=0.28,
        G12=5000.0,
        G23=3000.0,
        G31=5000.0,
        rho0=2.0e-9,
    )
    c_def = law15_chang.sound_speed(mat)
    assert c_def > 0.0

    # Custom rho
    c_custom = law15_chang.sound_speed(mat, rho=8.0e-9)
    assert c_custom == pytest.approx(c_def * 0.5)

    # Materials dispatch
    assert materials.sound_speed(mat) == pytest.approx(c_def)


def test_law15_extra_shapes():
    """Verify persistent extra state array shapes for nip layers."""
    mat = law15_chang.build_law15()

    shapes_nip = law15_chang.extra_shapes(mat, nip=5)
    assert shapes_nip["damt15"] == (5, 2)
    assert shapes_nip["sigr15"] == (5, 6)
    assert shapes_nip["wpla15"] == (5,)
    assert shapes_nip["off15"] == (5,)

    shapes_single = law15_chang.extra_shapes(mat, nip=None)
    assert shapes_single["damt15"] == (2,)
    assert shapes_single["sigr15"] == (6,)
    assert shapes_single["wpla15"] == ()
    assert shapes_single["off15"] == ()

    # Dispatch via materials package
    m_shapes = materials.extra_shapes(mat, nip=4)
    assert m_shapes["damt15"] == (4, 2)
    assert m_shapes["sigr15"] == (4, 6)
    assert m_shapes["wpla15"] == (4,)
    assert m_shapes["off15"] == (4,)


# ============================================================================
# 3. Solid Update Rejection
# ============================================================================

def test_law15_solid_update_raises():
    """Verify solid update raises NotImplementedError for shells only."""
    mat = law15_chang.build_law15()
    with pytest.raises(NotImplementedError, match="shell elements only"):
        law15_chang.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)))

    with pytest.raises(NotImplementedError, match="shell elements only"):
        materials.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), epsp=None, dt=0.0)


# ============================================================================
# 4. Elastic Plane-Stress Response
# ============================================================================

def test_law15_elastic_plane_stress():
    """Verify linear orthotropic elasticity for uncracked, unyielded material."""
    E1 = 120000.0
    E2 = 8000.0
    nu12 = 0.3
    G12 = 4000.0
    rho0 = 1.5e-9

    mat = law15_chang.build_law15(
        E1=E1,
        E2=E2,
        nu12=nu12,
        G12=G12,
        G23=3000.0,
        G31=4000.0,
        rho0=rho0,
        sigyt1=1e10,  # High yield to remain purely elastic
        sigyt2=1e10,
        sigt12=1e10,
    )

    nu21 = nu12 * E2 / E1
    detc = 1.0 - nu12 * nu21
    A11 = E1 / detc
    A22 = E2 / detc
    A12 = nu21 * A11

    # 1. Uniaxial longitudinal strain
    deps1 = np.array([1.0e-4, 0.0, 0.0])
    sig1, epsp1, c1 = law15_chang.shell_update(mat, np.zeros(3), deps1)
    assert sig1[0] == pytest.approx(A11 * 1.0e-4, rel=1e-5)
    assert sig1[1] == pytest.approx(A12 * 1.0e-4, rel=1e-5)
    assert sig1[2] == pytest.approx(0.0, abs=1e-10)
    assert epsp1 == 0.0

    # 2. Uniaxial transverse strain
    deps2 = np.array([0.0, 1.0e-4, 0.0])
    sig2, epsp2, c2 = law15_chang.shell_update(mat, np.zeros(3), deps2)
    assert sig2[0] == pytest.approx(A12 * 1.0e-4, rel=1e-5)
    assert sig2[1] == pytest.approx(A22 * 1.0e-4, rel=1e-5)
    assert sig2[2] == pytest.approx(0.0, abs=1e-10)

    # 3. Pure shear strain
    deps3 = np.array([0.0, 0.0, 2.0e-4])
    sig3, epsp3, c3 = law15_chang.shell_update(mat, np.zeros(3), deps3)
    assert sig3[0] == pytest.approx(0.0, abs=1e-10)
    assert sig3[1] == pytest.approx(0.0, abs=1e-10)
    assert sig3[2] == pytest.approx(G12 * 2.0e-4, rel=1e-5)

    # 4. 2D array vectorization
    deps_2d = np.array([deps1, deps2, deps3])
    sigs, epsps, cs = law15_chang.shell_update(mat, np.zeros((3, 3)), deps_2d)
    assert sigs.shape == (3, 3)
    assert sigs[0, 0] == pytest.approx(sig1[0])
    assert sigs[1, 1] == pytest.approx(sig2[1])
    assert sigs[2, 2] == pytest.approx(sig3[2])


# ============================================================================
# 5. Tsai-Wu Plasticity Yield and Return
# ============================================================================

def test_law15_tsai_wu_plasticity():
    """Verify that exceeding Tsai-Wu yield limit triggers normal plastic flow return."""
    mat = law15_chang.build_law15(
        E1=100000.0,
        E2=10000.0,
        nu12=0.3,
        G12=4000.0,
        G23=3000.0,
        G31=4000.0,
        rho0=1.5e-9,
        sigyt1=500.0,
        sigyc1=500.0,
        sigyt2=50.0,
        sigyc2=50.0,
        sigt12=40.0,
        sigc12=40.0,
        b=10.0,
        n=1.0,
        # High failure limits to isolate plasticity from failure
        s1=1e10, s2=1e10, c1=1e10, c2=1e10, s12=1e10,
    )

    # Elastic limit in transverse direction is sigyt2 = 50 MPa
    # Apply strain that would give elastic stress = 100 MPa
    nu21 = 0.3 * 10000.0 / 100000.0
    detc = 1.0 - 0.3 * nu21
    A22 = 10000.0 / detc
    target_strain = 100.0 / A22

    deps = np.array([0.0, target_strain, 0.0])
    sig, epsp, c = law15_chang.shell_update(mat, np.zeros(3), deps)

    # Stress must be reduced by plastic return
    assert sig[1] < 100.0
    # Plastic work / strain must be positive
    assert epsp > 0.0


def test_law15_strain_rate_sensitivity():
    """Verify that rate coefficient src increases yield strength at higher strain rates."""
    mat = law15_chang.build_law15(
        E1=100000.0,
        E2=10000.0,
        nu12=0.3,
        G12=4000.0,
        G23=3000.0,
        G31=4000.0,
        rho0=1.5e-9,
        sigyt1=500.0,
        sigyc1=500.0,
        sigyt2=50.0,
        sigyc2=50.0,
        sigt12=40.0,
        sigc12=40.0,
        src=0.1,   # 10% rate sensitivity
        srp=1.0,   # Reference strain rate 1.0 /s
        s1=1e10, s2=1e10, c1=1e10, c2=1e10, s12=1e10,
    )

    # Small dt -> very high strain rate 1e5 /s
    deps = np.array([0.0, 0.01, 0.0])
    sig_fast, epsp_fast, _ = law15_chang.shell_update(mat, np.zeros(3), deps, dt=1.0e-7)

    # Quasi-static dt -> strain rate 1.0 /s
    sig_slow, epsp_slow, _ = law15_chang.shell_update(mat, np.zeros(3), deps, dt=0.01)

    # Fast loading experiences dynamic hardening, retaining higher flow stress
    assert sig_fast[1] > sig_slow[1]


# ============================================================================
# 6. Chang-Chang Failure Criteria (m15crak.F)
# ============================================================================

def test_law15_tensile_fiber_breakage():
    """Verify tensile fiber failure mode ef2 = (s11/s1)^2 + beta_s*(s12/s12)^2 >= 1."""
    s1_limit = 1500.0
    s12_limit = 80.0
    beta_s = 1.0
    mat = law15_chang.build_law15(
        E1=150000.0,
        E2=10000.0,
        nu12=0.3,
        G12=4000.0,
        G23=3000.0,
        G31=4000.0,
        rho0=1.5e-9,
        sigyt1=1e10, sigyt2=1e10, sigt12=1e10,
        s1=s1_limit,
        s12=s12_limit,
        beta_s=beta_s,
        tmax=0.001,
        itype=0,
    )

    # Apply strain exceeding tensile fiber strength
    detc = 1.0 - 0.3 * (0.3 * 10000.0 / 150000.0)
    A11 = 150000.0 / detc
    deps = np.array([1600.0 / A11, 0.0, 0.0])

    extra = {}
    sig, epsp, _ = law15_chang.shell_update(mat, np.zeros(3), deps, extra=extra)

    # Fiber damage flag damt15[0] must be set (< 1.0)
    damt = extra["damt15"]
    assert damt[0, 0] < 1.0
    # Matrix intact
    assert damt[0, 1] == 1.0


def test_law15_compressive_fiber_breakage():
    """Verify compressive fiber failure mode efc2 = (s11/c1)^2 >= 1."""
    c1_limit = 1000.0
    mat = law15_chang.build_law15(
        E1=150000.0,
        E2=10000.0,
        nu12=0.3,
        G12=4000.0,
        G23=3000.0,
        G31=4000.0,
        rho0=1.5e-9,
        sigyt1=1e10, sigyt2=1e10, sigt12=1e10,
        c1=c1_limit,
        tmax=0.001,
        itype=0,
    )

    detc = 1.0 - 0.3 * (0.3 * 10000.0 / 150000.0)
    A11 = 150000.0 / detc
    deps = np.array([-1100.0 / A11, 0.0, 0.0])

    extra = {}
    sig, epsp, _ = law15_chang.shell_update(mat, np.zeros(3), deps, extra=extra)

    damt = extra["damt15"]
    assert damt[0, 0] < 1.0


def test_law15_tensile_matrix_cracking():
    """Verify tensile matrix cracking mode em2 = (s22/c2)^2 + (s12/s12)^2 >= 1."""
    c2_limit = 60.0
    mat = law15_chang.build_law15(
        E1=150000.0,
        E2=10000.0,
        nu12=0.3,
        G12=4000.0,
        G23=3000.0,
        G31=4000.0,
        rho0=1.5e-9,
        sigyt1=1e10, sigyt2=1e10, sigt12=1e10,
        s1=1e10,
        c2=c2_limit,
        tmax=0.001,
        itype=0,
    )

    detc = 1.0 - 0.3 * (0.3 * 10000.0 / 150000.0)
    A22 = 10000.0 / detc
    deps = np.array([0.0, 70.0 / A22, 0.0])

    extra = {}
    sig, epsp, _ = law15_chang.shell_update(mat, np.zeros(3), deps, extra=extra)

    damt = extra["damt15"]
    # Fiber intact
    assert damt[0, 0] == 1.0
    # Matrix cracked
    assert damt[0, 1] < 1.0


def test_law15_compressive_matrix_cracking():
    """Verify compressive matrix cracking criterion emc2 >= 1."""
    c2_limit = 150.0
    s12_limit = 60.0
    mat = law15_chang.build_law15(
        E1=150000.0,
        E2=10000.0,
        nu12=0.3,
        G12=4000.0,
        G23=3000.0,
        G31=4000.0,
        rho0=1.5e-9,
        sigyt1=1e10, sigyt2=1e10, sigt12=1e10,
        s1=1e10,
        c2=c2_limit,
        s12=s12_limit,
        tmax=0.001,
        itype=0,
    )

    detc = 1.0 - 0.3 * (0.3 * 10000.0 / 150000.0)
    A22 = 10000.0 / detc
    deps = np.array([0.0, -180.0 / A22, 0.0])

    extra = {}
    sig, epsp, _ = law15_chang.shell_update(mat, np.zeros(3), deps, extra=extra)

    damt = extra["damt15"]
    assert damt[0, 0] == 1.0
    assert damt[0, 1] < 1.0


# ============================================================================
# 7. Post-Failure Relaxation & Stresses Decay
# ============================================================================

def test_law15_post_failure_relaxation():
    """Verify exponential relaxation damt = exp(-(t - tfail)/tmax) and stress decay."""
    tmax = 0.005
    mat = law15_chang.build_law15(
        E1=100000.0,
        E2=10000.0,
        nu12=0.3,
        G12=4000.0,
        G23=3000.0,
        G31=4000.0,
        rho0=1.5e-9,
        sigyt1=1e10, sigyt2=1e10, sigt12=1e10,
        s1=1000.0,
        tmax=tmax,
        itype=0,
    )

    extra = {"time": 0.001}
    deps = np.array([0.02, 0.0, 0.0])  # Exceeds s1 = 1000 MPa

    # Step 1: failure initiation at t = 0.001
    sig1, _, _ = law15_chang.shell_update(mat, np.zeros(3), deps, extra=extra)
    assert extra["damt15"][0, 0] == 0.999
    assert extra["sigr15"][0, 5] == 0.001

    # Step 2: advance time by tmax (0.005 s) to t = 0.006
    extra["time"] = 0.006
    sig2, _, _ = law15_chang.shell_update(mat, sig1, np.zeros(3), extra=extra)

    # Damage factor should decay to exp(-1) ~ 0.367879
    expected_decay = math.exp(-1.0)
    assert extra["damt15"][0, 0] == pytest.approx(expected_decay, rel=1e-3)
    assert sig2[0] == pytest.approx(sig1[0] * expected_decay, rel=1e-3)

    # Step 3: advance time far beyond (t = 0.1 s, > 5*tmax)
    extra["time"] = 0.1
    sig3, _, _ = law15_chang.shell_update(mat, sig2, np.zeros(3), extra=extra)
    # Fully decayed below 0.01 threshold -> zero
    assert extra["damt15"][0, 0] == 0.0
    assert sig3[0] == 0.0


# ============================================================================
# 8. Element Deletion & Layer Failure (Itype / IOFF)
# ============================================================================

def test_law15_element_deletion_on_fiber_failure():
    """Verify that if itype != 0, fiber failure zeros stresses and clears off15 flag."""
    mat = law15_chang.build_law15(
        E1=100000.0,
        E2=10000.0,
        nu12=0.3,
        G12=4000.0,
        G23=3000.0,
        G31=4000.0,
        rho0=1.5e-9,
        sigyt1=1e10, sigyt2=1e10, sigt12=1e10,
        s1=1000.0,
        itype=2,  # itype != 0 enables element deletion on fiber failure
    )

    deps = np.array([0.02, 0.0, 0.0])
    extra = {}
    sig, _, _ = law15_chang.shell_update(mat, np.zeros(3), deps, extra=extra)

    assert extra["off15"][0] == 0.0
    assert np.all(sig == 0.0)


def test_law15_element_deletion_on_wpmax():
    """Verify that exceeding wpmax zeros stresses and clears off15."""
    mat = law15_chang.build_law15(
        E1=100000.0,
        E2=10000.0,
        nu12=0.3,
        G12=4000.0,
        G23=3000.0,
        G31=4000.0,
        rho0=1.5e-9,
        sigyt1=200.0,
        sigyc1=200.0,
        sigyt2=50.0,
        sigyc2=50.0,
        sigt12=40.0,
        sigc12=40.0,
        wpmax=3.0,
        wpref=1.0,
        s1=1e10, s2=1e10, c1=1e10, c2=1e10, s12=1e10,
        itype=1,
    )

    # Huge plastic deformation driving wpla > wpmax
    deps = np.array([0.0, 0.2, 0.0])
    extra = {}
    sig, epsp, _ = law15_chang.shell_update(mat, np.zeros(3), deps, extra=extra)

    assert extra["off15"][0] == 0.0
    assert np.all(sig == 0.0)


# ============================================================================
# 9. Consistent Algorithmic Tangent (consistent_shell_tangent)
# ============================================================================

def test_law15_consistent_shell_tangent_elastic():
    """Verify consistent tangent matches elastic membrane tangent in elastic range."""
    mat = law15_chang.build_law15(
        E1=100000.0,
        E2=10000.0,
        nu12=0.3,
        G12=4000.0,
        G23=3000.0,
        G31=4000.0,
        rho0=1.5e-9,
        sigyt1=1e10, sigyt2=1e10, sigt12=1e10,
    )

    c_el = law15_chang.shell_membrane_tangent(mat)
    sig = np.array([100.0, 20.0, 10.0])
    c_alg = law15_chang.consistent_shell_tangent(mat, sig)

    assert np.allclose(c_alg, c_el)


def test_law15_consistent_shell_tangent_plastic():
    """Verify elastoplastic tangent is softer than elastic tangent under yield."""
    mat = law15_chang.build_law15(
        E1=100000.0,
        E2=10000.0,
        nu12=0.3,
        G12=4000.0,
        G23=3000.0,
        G31=4000.0,
        rho0=1.5e-9,
        sigyt1=500.0,
        sigyc1=500.0,
        sigyt2=50.0,
        sigyc2=50.0,
        sigt12=40.0,
        sigc12=40.0,
        b=10.0,
        n=1.0,
        s1=1e10, s2=1e10, c1=1e10, c2=1e10, s12=1e10,
    )

    c_el = law15_chang.shell_membrane_tangent(mat)
    # Stress on the yield surface
    sig_yield = np.array([0.0, 50.0, 0.0])
    c_ep = law15_chang.consistent_shell_tangent(mat, sig_yield)

    # Plasticity softens the tangent: C_ep[1, 1] < C_el[1, 1]
    assert c_ep[1, 1] < c_el[1, 1]


def test_law15_tangent_when_deleted():
    """Verify tangent is zero when element is deleted (off15 <= 0)."""
    mat = law15_chang.build_law15()
    extra = {"off15": np.array([0.0])}
    c_tan = law15_chang.consistent_shell_tangent(mat, np.zeros(3), extra=extra)
    assert np.all(c_tan == 0.0)


# ============================================================================
# 10. Materials Registry and Package-Level Dispatch
# ============================================================================

def test_law15_registration_and_aliases():
    """Verify registry keys /MAT/LAW15, /MAT/CHANG, /MAT/PLAS_ANISO, /MAT/COMP_CHANG."""
    for key in (15, "15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG"):
        assert key in MAT_PHYSICS_REGISTRY
        builder = MAT_PHYSICS_REGISTRY[key]
        mat = builder(E1=100000.0, E2=10000.0, nu12=0.3, G12=4000.0)
        assert mat.law == 15
        assert mat.law_name == "LAW15"


def test_law15_package_level_dispatch():
    """Verify materials.shell_update, materials.sound_speed, materials.consistent_shell_tangent."""
    mat = law15_chang.build_law15(
        E1=100000.0,
        E2=10000.0,
        nu12=0.3,
        G12=4000.0,
        G23=3000.0,
        G31=4000.0,
        rho0=1.5e-9,
    )

    # 1. sound_speed
    ssp = materials.sound_speed(mat)
    assert ssp > 0.0

    # 2. shell_membrane_tangent
    c_mem = materials.shell_membrane_tangent(mat)
    assert c_mem.shape == (3, 3)

    # 3. shell_layer_tangent / consistent_shell_tangent
    c_lay = materials.shell_layer_tangent(mat, np.zeros((1, 3)))
    assert c_lay.shape == (1, 3, 3)

    # 4. shell_update
    sig, epsp = materials.shell_update(mat, np.zeros(3), np.array([1e-4, 0.0, 0.0]))
    assert sig[0] > 0.0
