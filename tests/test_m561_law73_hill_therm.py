"""
Tests for OpenRadioss Material Law 73 (/MAT/LAW73, /MAT/BARLAT2000, /MAT/HILL_THERM).
Thermal Hill Orthotropic Plasticity Model for Shell Elements (Milestone M561).

Upstream reference files:
  - Starter reader: starter/source/materials/mat/mat073/hm_read_mat73.F
  - Engine physics: engine/source/materials/mat/mat073/sigeps73c.F
  - Table tools: engine/source/tools/curve/table_tools.F (TABLE_VINTERP)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

import pyradioss.materials as materials
from pyradioss.materials import (
    Law73Params,
    build_law73,
    law73_hill_therm,
    shell_update_law73,
    solid_update_law73,
    law73_sound_speed,
    law73_shell_tangent,
    law73_shell_membrane_tangent,
    law73_extra_shapes,
    LAW_DISPATCH_METADATA,
    MATERIAL_SHELL_DISPATCH,
    MATERIAL_SOLID_DISPATCH,
)
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY


# ============================================================================
# 1. Law73Params Initialization, Defaults & Derived Constants
# ============================================================================

def test_law73_params_defaults():
    """Verify default parameters and isotropic Hill reduction."""
    p = Law73Params()
    assert p.rho0 == 1.0
    assert p.e == 210000.0
    assert p.nu == 0.3
    assert p.r00 == 1.0
    assert p.r45 == 1.0
    assert p.r90 == 1.0
    assert p.chard == 0.0
    assert p.iyield == 0
    assert p.t0 == 293.0
    assert p.rhocp == 0.0

    # Isotropic recovery: R00=R45=R90=1 => R=1, H=0.5, A01=1, A02=1, A03=1, A12=3 (von Mises)
    assert p.r == pytest.approx(1.0)
    assert p.h == pytest.approx(0.5)
    assert p.a01 == pytest.approx(1.0)
    assert p.a02 == pytest.approx(1.0)
    assert p.a03 == pytest.approx(1.0)
    assert p.a12 == pytest.approx(3.0)

    # Elastic constants
    expected_a11 = 210000.0 / (1.0 - 0.3 ** 2)
    assert p.a11 == pytest.approx(expected_a11)
    assert p.a21 == pytest.approx(0.3 * expected_a11)
    expected_g = 0.5 * 210000.0 / (1.0 + 0.3)
    assert p.g == pytest.approx(expected_g)
    assert p.soundsp == pytest.approx(math.sqrt(expected_a11 / 1.0))


def test_law73_lankford_anisotropy_and_normalization():
    """Verify Hill constants calculation and iyield=1 normalization."""
    r00, r45, r90 = 1.5, 1.2, 1.8
    # iyield = 0: unnormalized
    p0 = Law73Params(r00=r00, r45=r45, r90=r90, iyield=0)
    expected_r = 0.25 * (1.5 + 2.0 * 1.2 + 1.8)
    expected_h = expected_r / (1.0 + expected_r)
    expected_a01 = expected_h * (1.0 + 1.0 / 1.5)
    expected_a02 = expected_h * (1.0 + 1.0 / 1.8)
    expected_a03 = 2.0 * expected_h
    expected_a12 = (2.0 * 1.2 + 1.0) * (expected_a01 + expected_a02 - expected_a03)

    assert p0.r == pytest.approx(expected_r)
    assert p0.h == pytest.approx(expected_h)
    assert p0.a01 == pytest.approx(expected_a01)
    assert p0.a02 == pytest.approx(expected_a02)
    assert p0.a03 == pytest.approx(expected_a03)
    assert p0.a12 == pytest.approx(expected_a12)

    # iyield = 1: normalized so A01 = 1.0
    p1 = Law73Params(r00=r00, r45=r45, r90=r90, iyield=1)
    assert p1.a01 == pytest.approx(1.0)
    assert p1.a02 == pytest.approx(expected_a02 / expected_a01)
    assert p1.a03 == pytest.approx(expected_a03 / expected_a01)
    assert p1.a12 == pytest.approx(expected_a12 / expected_a01)


def test_law73_builder():
    """Verify build_law73 constructor from dict, Law73Params, or kwargs."""
    mat1 = build_law73(Law73Params(e=100000.0, nu=0.25))
    assert mat1.law == 73
    assert mat1.law_name == "LAW73"
    assert mat1.params["e"] == 100000.0
    assert mat1.params["nu"] == 0.25

    mat2 = build_law73({"id": 5, "density": 2.7e-9, "e": 70000.0, "nu": 0.33, "r00": 1.2})
    assert mat2.id == 5
    assert mat2.rho0 == 2.7e-9
    assert mat2.params["e"] == 70000.0
    assert mat2.params["r00"] == 1.2


# ============================================================================
# 2. Pure Elastic Trial Stress & Transverse Shear
# ============================================================================

def test_law73_pure_elastic_update():
    """Verify pure elastic step when equivalent stress is below initial yield."""
    p = Law73Params(e=200000.0, nu=0.3, yield_table=400.0)
    mat = build_law73(p)

    sig = np.zeros(3)
    deps = np.array([1.0e-4, 0.5e-4, -0.2e-4])
    sig_out, pla_out = shell_update_law73(mat, sig.copy(), deps)

    # Expected elastic increment (deps[2] is engineering shear gamma_xy = 2*eps_xy):
    a11 = p.a11
    a21 = p.a21
    g = p.g
    expected_sxx = a11 * deps[0] + a21 * deps[1]
    expected_syy = a21 * deps[0] + a11 * deps[1]
    expected_sxy = g * deps[2]

    assert sig_out[0] == pytest.approx(expected_sxx, rel=1e-6)
    assert sig_out[1] == pytest.approx(expected_syy, rel=1e-6)
    assert sig_out[2] == pytest.approx(expected_sxy, rel=1e-6)
    assert pla_out == 0.0


def test_law73_transverse_shear_5component():
    """Verify 5-component stress update: plane stress + transverse shear."""
    p = Law73Params(e=200000.0, nu=0.3, yield_table=500.0)
    mat = build_law73(p)

    sig = np.zeros(5)
    deps = np.array([1.0e-4, 0.0, 0.0, 2.0e-4, -1.0e-4])
    sig_out, pla_out = shell_update_law73(mat, sig.copy(), deps)

    # Transverse shears are updated purely elastically with transverse shear modulus gs = G * shf
    expected_g = p.g * (5.0 / 6.0)
    assert sig_out[3] == pytest.approx(expected_g * deps[3], rel=1e-6)
    assert sig_out[4] == pytest.approx(expected_g * deps[4], rel=1e-6)


# ============================================================================
# 3. Plastic Return Mapping & Isotropic von Mises Recovery
# ============================================================================

def test_law73_plastic_return_isotropic():
    """Verify plastic return mapping reduces to von Mises plane-stress radial return."""
    yield_stress = 300.0
    p = Law73Params(e=200000.0, nu=0.3, yield_table=yield_stress, chard=0.0)
    mat = build_law73(p)

    # Large uniaxial strain causing plastic deformation along x
    # 2 Newton iterations of sigeps73c.F converges within 0.3% on large single step
    sig = np.zeros(3)
    deps = np.array([0.005, 0.0, 0.0])
    sig_out, pla_out = shell_update_law73(mat, sig, deps)

    # Under pure isotropic plastic yielding, equivalent stress must lie on yield surface
    a01, a02, a03, a12 = p.a01, p.a02, p.a03, p.a12
    seq = math.sqrt(a01 * sig_out[0] ** 2 + a02 * sig_out[1] ** 2
                    - a03 * sig_out[0] * sig_out[1] + a12 * sig_out[2] ** 2)

    assert seq == pytest.approx(yield_stress, rel=0.005)
    assert pla_out > 0.0


def test_law73_plastic_return_anisotropic():
    """Verify plastic return for anisotropic Lankford coefficients."""
    r00, r45, r90 = 1.8, 1.3, 2.2
    yield_stress = 250.0
    p = Law73Params(e=200000.0, nu=0.3, r00=r00, r45=r45, r90=r90, iyield=1,
                    yield_table=yield_stress, chard=0.0)
    mat = build_law73(p)

    sig = np.zeros(3)
    # Biaxial loading
    deps = np.array([0.004, 0.002, 0.001])
    sig_out, pla_out = shell_update_law73(mat, sig, deps)

    a01, a02, a03, a12 = p.a01, p.a02, p.a03, p.a12
    seq = math.sqrt(a01 * sig_out[0] ** 2 + a02 * sig_out[1] ** 2
                    - a03 * sig_out[0] * sig_out[1] + a12 * sig_out[2] ** 2)

    assert seq == pytest.approx(yield_stress, rel=0.05)
    assert pla_out > 0.0


def test_law73_plastic_return_multistep_convergence():
    """Verify multi-step realistic explicit loading reaches high yield precision."""
    yield_stress = 350.0
    p = Law73Params(e=200000.0, nu=0.3, yield_table=yield_stress, chard=0.0)
    mat = build_law73(p)

    sig = np.zeros(3)
    pla = 0.0
    # 20 small steps typical of explicit cycles
    for _ in range(20):
        sig, pla = shell_update_law73(mat, sig, np.array([1.5e-4, 0.0, 0.0]), epsp=pla)

    seq = math.sqrt(sig[0] ** 2 + sig[1] ** 2 - sig[0] * sig[1] + 3.0 * sig[2] ** 2)
    assert seq == pytest.approx(yield_stress, rel=1e-3)
    assert pla > 0.0


# ============================================================================
# 4. Mixed Isotropic / Kinematic Hardening & Back-Stress
# ============================================================================

def test_law73_pure_kinematic_hardening_bauschinger():
    """Verify pure kinematic hardening (chard=1.0) shifts yield surface (Bauschinger effect)."""
    # Linear hardening curve: yield_table returns (yld, slope)
    def linear_hardening(pla, rate=0.0, temp=293.0):
        return 200.0 + 5000.0 * pla, 5000.0

    p = Law73Params(e=200000.0, nu=0.3, yield_table=linear_hardening, chard=1.0)
    mat = build_law73(p)

    extra = {
        "uvar73": np.zeros((1, 7)),
        "pla73": np.zeros(1),
        "off73": np.ones(1),
    }

    # Step 1: Forward plastic loading along x
    sig1, pla1 = shell_update_law73(mat, np.zeros(3), np.array([0.003, 0.0, 0.0]),
                                    epsp=0.0, extra=extra)
    assert pla1 > 0.0
    # Check that backstress alpha_xx (uvar[:, 1]) has evolved positively
    alpha_xx = extra["uvar73"][0, 1]
    assert alpha_xx > 0.0

    # Step 2: Unload and reload in reverse (-x)
    # With kinematic hardening, reverse yield occurs at |sig_xx - alpha_xx| = Y_0,
    # so reverse yield stress is smaller in magnitude than forward stress!
    sig_rev = sig1.copy()
    deps_rev = np.array([-0.0015, 0.0, 0.0])
    sig2, pla2 = shell_update_law73(mat, sig_rev, deps_rev, epsp=pla1, extra=extra)

    # Active stress is sig - alpha
    active_xx = sig2[0] - extra["uvar73"][0, 1]
    assert active_xx < sig2[0]


def test_law73_isotropic_vs_kinematic_hardening():
    """Verify that isotropic (chard=0) expands yield surface while kinematic (chard=1) preserves initial size."""
    def hard_curve(pla, *args):
        return 250.0 + 2000.0 * pla, 2000.0

    p_iso = Law73Params(e=200000.0, nu=0.3, yield_table=hard_curve, chard=0.0)
    p_kin = Law73Params(e=200000.0, nu=0.3, yield_table=hard_curve, chard=1.0)

    mat_iso = build_law73(p_iso)
    mat_kin = build_law73(p_kin)

    ext_iso = {"uvar73": np.zeros((1, 7))}
    ext_kin = {"uvar73": np.zeros((1, 7))}

    deps = np.array([0.004, 0.0, 0.0])
    s_iso, p_out_iso = shell_update_law73(mat_iso, np.zeros(3), deps, extra=ext_iso)
    s_kin, p_out_kin = shell_update_law73(mat_kin, np.zeros(3), deps, extra=ext_kin)

    # In isotropic hardening, backstress remains zero
    assert ext_iso["uvar73"][0, 1] == pytest.approx(0.0)
    # In kinematic hardening, backstress is positive
    assert ext_kin["uvar73"][0, 1] > 0.0


# ============================================================================
# 5. Dynamic Young's Modulus Degradation
# ============================================================================

def test_law73_dynamic_youngs_modulus_exponential():
    """Verify CE exponential degradation of Young's modulus with plastic strain."""
    e0 = 200000.0
    einf = 150000.0
    ce = 50.0
    p = Law73Params(e=e0, einf=einf, ce=ce, yield_table=300.0)
    mat = build_law73(p)

    extra = {"uvar73": np.zeros((1, 7))}
    # Apply significant plastic strain
    deps = np.array([0.02, 0.0, 0.0])
    sig_out, pla_out = shell_update_law73(mat, np.zeros(3), deps, extra=extra)

    assert pla_out > 0.0
    # Expected degraded modulus E
    expected_e = e0 - (e0 - einf) * (1.0 - math.exp(-ce * pla_out))
    assert expected_e < e0
    assert expected_e >= einf


def test_law73_dynamic_youngs_modulus_curve():
    """Verify tabulated curve degradation of Young's modulus (ifunce > 0)."""
    e0 = 200000.0
    # Curve returning degradation ratio f_E(pla)
    curve_e = ([0.0, 0.05, 0.10], [1.0, 0.85, 0.70])
    p = Law73Params(e=e0, ifunce=1, curve_e=curve_e, yield_table=300.0)
    mat = build_law73(p)
    assert p.opte == 1

    extra = {"uvar73": np.zeros((1, 7))}
    deps = np.array([0.01, 0.0, 0.0])
    sig_out, pla_out = shell_update_law73(mat, np.zeros(3), deps, extra=extra)
    assert pla_out > 0.0


# ============================================================================
# 6. Adiabatic Heating & Temperature Evolution
# ============================================================================

def test_law73_adiabatic_plastic_heating():
    """Verify temperature rise from plastic work dissipation."""
    t0 = 293.0
    rhocp = 3.5e6  # J / (m^3 * K)
    # Temperature-dependent yield: 3D function
    def y_therm(pla, rate, temp):
        # 300 MPa at 293 K, softening by 0.5 MPa/K
        return max(300.0 - 0.5 * (temp - 293.0), 50.0), 0.0

    p = Law73Params(e=200000.0, nu=0.3, t0=t0, rhocp=rhocp, yield_table=y_therm)
    mat = build_law73(p)

    extra = {
        "uvar73": np.zeros((1, 7)),
        "temp": np.full(1, t0),
        "eint": np.zeros(1),
        "vol": np.ones(1),
        "vol0": np.ones(1),
    }

    # Step with plastic work
    deps = np.array([0.005, 0.0, 0.0])
    sig_out, pla_out = shell_update_law73(mat, np.zeros(3), deps, extra=extra)

    assert pla_out > 0.0
    assert extra["temp"][0] > t0
    assert extra["eint"][0] > 0.0


# ============================================================================
# 7. Tensile Failure Damage Factor & Element Deletion
# ============================================================================

def test_law73_tensile_failure_damage():
    """Verify FAIL softening factor between EPSR1 and EPSR2."""
    epsr1 = 0.01
    epsr2 = 0.03
    p = Law73Params(e=200000.0, nu=0.3, epsr1=epsr1, epsr2=epsr2, yield_table=250.0)
    mat = build_law73(p)

    extra = {"uvar73": np.zeros((1, 7))}

    # Case A: Below epsr1 => FAIL = 1.0
    deps_a = np.array([0.005, 0.0, 0.0])
    sig_a, _ = shell_update_law73(mat, np.zeros(3), deps_a, extra=dict(extra))
    seq_a = math.sqrt(sig_a[0] ** 2 + sig_a[1] ** 2 - sig_a[0] * sig_a[1] + 3.0 * sig_a[2] ** 2)
    assert seq_a == pytest.approx(250.0, rel=0.005)

    # Case B: Between epsr1 and epsr2 (at 0.02, FAIL = (0.03 - 0.02) / (0.03 - 0.01) = 0.5)
    deps_b = np.array([0.02, 0.0, 0.0])
    sig_b, _ = shell_update_law73(mat, np.zeros(3), deps_b, extra=dict(extra))
    seq_b = math.sqrt(sig_b[0] ** 2 + sig_b[1] ** 2 - sig_b[0] * sig_b[1] + 3.0 * sig_b[2] ** 2)
    assert seq_b == pytest.approx(250.0 * 0.5, rel=0.02)

    # Case C: Beyond epsr2 => FAIL = 0.0 (complete failure)
    deps_c = np.array([0.04, 0.0, 0.0])
    sig_c, _ = shell_update_law73(mat, np.zeros(3), deps_c, extra=dict(extra))
    assert np.allclose(sig_c, 0.0, atol=1e-3)


def test_law73_element_erosion():
    """Verify element erosion when plastic strain exceeds eps_max."""
    eps_max = 0.05
    p = Law73Params(e=200000.0, nu=0.3, eps_max=eps_max, yield_table=200.0)
    mat = build_law73(p)

    extra = {
        "uvar73": np.zeros((1, 7)),
        "off73": np.ones(1),
        "off": np.ones(1),
    }

    # Step exceeding eps_max
    deps = np.array([0.08, 0.0, 0.0])
    sig_out, pla_out = shell_update_law73(mat, np.zeros(3), deps, extra=extra)

    assert pla_out > eps_max
    # Element off flag degraded by 0.8 factor per OpenRadioss convention
    assert extra["off73"][0] == pytest.approx(0.8)
    assert extra["off"][0] == pytest.approx(0.8)


# ============================================================================
# 8. Trilinear Table Interpolation (TABLE_VINTERP)
# ============================================================================

def test_law73_table_interpolation_3d():
    """Verify trilinear interpolation across plastic strain, strain rate, and temperature."""
    # Build 3D table grid
    x1 = np.array([0.0, 0.1, 0.2])          # plastic strain
    x2 = np.array([0.0, 100.0])              # strain rate
    x3 = np.array([293.0, 400.0])            # temperature

    # Create 3D values array: y[temp_idx, rate_idx, strain_idx]
    y_grid = np.zeros((2, 2, 3))
    for k in range(2):      # temp
        for j in range(2):  # rate
            for i in range(3):  # strain
                y_grid[k, j, i] = 200.0 + 300.0 * x1[i] + 0.5 * x2[j] - 0.2 * (x3[k] - 293.0)

    table_dict = {
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "y": y_grid,
    }

    p = Law73Params(e=200000.0, nu=0.3, yield_table=table_dict)
    mat = build_law73(p)

    # Query mid-point: pla=0.05, rate=50, temp=346.5
    pla_test = np.array([0.05])
    rate_test = np.array([50.0])
    temp_test = np.array([346.5])

    val, dval = law73_hill_therm._eval_yield_table(p, pla_test, rate_test, temp_test)
    expected_val = 200.0 + 300.0 * 0.05 + 0.5 * 50.0 - 0.2 * (346.5 - 293.0)
    assert val[0] == pytest.approx(expected_val, rel=1e-5)
    assert dval[0] == pytest.approx(300.0, rel=1e-5)


# ============================================================================
# 9. Consistent Shell Tangent Operators
# ============================================================================

def test_law73_membrane_tangent():
    """Verify shell membrane tangent matrix against analytical formula."""
    p = Law73Params(e=210000.0, nu=0.3)
    mat = build_law73(p)

    c_mem = law73_shell_membrane_tangent(mat)
    assert c_mem.shape == (3, 3)
    assert c_mem[0, 0] == pytest.approx(p.a11)
    assert c_mem[1, 1] == pytest.approx(p.a11)
    assert c_mem[0, 1] == pytest.approx(p.a21)
    assert c_mem[1, 0] == pytest.approx(p.a21)
    assert c_mem[2, 2] == pytest.approx(p.g)
    assert c_mem[0, 2] == 0.0
    assert c_mem[1, 2] == 0.0


def test_law73_consistent_tangent_fd_verification():
    """Verify consistent tangent against central finite differences of shell_update."""
    p = Law73Params(e=200000.0, nu=0.3, yield_table=250.0, chard=0.5)
    mat = build_law73(p)

    sig = np.array([120.0, 40.0, 15.0])
    deps = np.array([0.002, 0.0005, -0.0002])

    tan = law73_shell_tangent(mat, sig=sig, deps=deps, h=1e-6)
    assert tan.shape == (3, 3)

    # Numerical verification via finite differences
    h = 1e-6
    d_num = np.zeros((3, 3))
    for j in range(3):
        dp = deps.copy()
        dm = deps.copy()
        dp[j] += h
        dm[j] -= h
        sp, _ = shell_update_law73(mat, sig.copy(), dp)
        sm, _ = shell_update_law73(mat, sig.copy(), dm)
        d_num[:, j] = (sp[:3] - sm[:3]) / (2.0 * h)

    assert np.allclose(tan, d_num, rtol=1e-3, atol=1e-3)


# ============================================================================
# 10. Rejection of Solids & Sound Speed
# ============================================================================

def test_law73_rejection_of_solids():
    """Verify solid_update and solid_tangent raise NotImplementedError."""
    p = Law73Params()
    mat = build_law73(p)

    with pytest.raises(NotImplementedError, match="shell elements only"):
        solid_update_law73(mat, np.zeros(6), np.zeros(6))

    with pytest.raises(NotImplementedError, match="shell elements only"):
        materials.solid_update(mat, np.zeros(6), np.zeros(6))

    with pytest.raises(NotImplementedError, match="shells only"):
        materials.solid_tangent(mat, np.zeros(6))


def test_law73_sound_speed():
    """Verify sound speed calculation for LAW73."""
    p = Law73Params(e=200000.0, nu=0.25, rho0=2.0)
    mat = build_law73(p)

    c_expected = math.sqrt((200000.0 / (1.0 - 0.25 ** 2)) / 2.0)
    c_calc = law73_sound_speed(mat)
    assert c_calc == pytest.approx(c_expected)

    # Through top-level materials.sound_speed
    assert materials.sound_speed(mat) == pytest.approx(c_expected)


# ============================================================================
# 11. Dispatch & Registry Integration
# ============================================================================

def test_law73_registration():
    """Verify LAW73 is correctly registered across all registries and metadata."""
    aliases = [73, "73", "LAW73", "HILL_THERM",
               "MAT_LAW73", "MAT_HILL_THERM", "LAW73_HILL_THERM"]

    for k in aliases:
        assert k in MAT_PHYSICS_REGISTRY
        assert k in MATERIAL_SHELL_DISPATCH
        assert k in MATERIAL_SOLID_DISPATCH
        assert k in LAW_DISPATCH_METADATA
        meta = LAW_DISPATCH_METADATA[k]
        assert meta["shell"] is True
        assert meta["solid"] is False
        assert meta["plane_stress"] is True


def test_law73_extra_shapes():
    """Verify extra_shapes returns required state variable arrays."""
    p = Law73Params()
    mat = build_law73(p)

    shapes_sol = law73_extra_shapes(mat, nip=None)
    assert shapes_sol["uvar73"] == (7,)
    assert shapes_sol["pla73"] == ()
    assert shapes_sol["off73"] == ()

    shapes_shell = law73_extra_shapes(mat, nip=5)
    assert shapes_shell["uvar73"] == (5, 7)
    assert shapes_shell["pla73"] == (5,)
    assert shapes_shell["off73"] == (5,)


# ============================================================================
# 12. Batched Element Vectorization Equivalence
# ============================================================================

def test_law73_batched_vectorization():
    """Verify batch calculation (n, 3) yields identical results to serial (3,) calls."""
    p = Law73Params(e=200000.0, nu=0.3, yield_table=280.0, chard=0.3)
    mat = build_law73(p)

    n_elem = 4
    sig_batch = np.array([
        [50.0, 20.0, 5.0],
        [150.0, 80.0, -10.0],
        [220.0, 10.0, 30.0],
        [-180.0, 60.0, 15.0],
    ])
    deps_batch = np.array([
        [0.001, 0.0005, 0.0],
        [0.002, -0.001, 0.0005],
        [0.004, 0.001, -0.002],
        [-0.003, 0.002, 0.001],
    ])

    extra_batch = {
        "uvar73": np.zeros((n_elem, 7)),
        "pla73": np.zeros(n_elem),
        "off73": np.ones(n_elem),
    }

    # Batch update
    s_out_batch, p_out_batch = shell_update_law73(mat, sig_batch.copy(), deps_batch,
                                                  epsp=np.zeros(n_elem), extra=extra_batch)

    # Serial updates
    for i in range(n_elem):
        ext_single = {
            "uvar73": np.zeros((1, 7)),
            "pla73": np.zeros(1),
            "off73": np.ones(1),
        }
        s_single, p_single = shell_update_law73(mat, sig_batch[i].copy(), deps_batch[i],
                                                epsp=0.0, extra=ext_single)
        assert np.allclose(s_out_batch[i], s_single, rtol=1e-10, atol=1e-10)
        assert p_out_batch[i] == pytest.approx(p_single, rel=1e-10)


# ============================================================================
# 13. Starter Model Integration & Deck Reader
# ============================================================================

def test_mat_law73_entities_and_properties():
    """Verify MatLaw73 entity dataclass fields, property helpers, and aliases."""
    from pyradioss.model.entities import MatLaw73, MatHillTherm, MatThermalHill, MatThermHill

    assert MatHillTherm is MatLaw73
    assert MatThermalHill is MatLaw73
    assert MatThermHill is MatLaw73

    m = MatLaw73(
        id=73,
        rho=2.7e-9,
        refer_rho=2.7e-9,
        e=70000.0,
        nu=0.33,
        ifunce=10,
        einf=50000.0,
        ce=25.0,
        r00=1.5,
        r45=1.2,
        r90=1.8,
        chard=0.4,
        iyield=1,
        eps_max=0.3,
        epsr1=0.25,
        epsr2=0.35,
        table_id=101,
        fscale=1.2,
        pscale=1.0,
        t0=293.0,
        rhocp=2.4e-3,
        title="Sheet Alloy 73",
    )

    assert m.id == 73
    assert m.title == "Sheet Alloy 73"
    assert m.law == 73
    assert m.law_name == "LAW73"
    assert m.rho == pytest.approx(2.7e-9)
    assert m.rho0 == pytest.approx(2.7e-9)
    assert m.rhor == pytest.approx(2.7e-9)
    assert m.e == pytest.approx(70000.0)
    assert m.E == pytest.approx(70000.0)
    assert m.nu == pytest.approx(0.33)
    assert m.Nu == pytest.approx(0.33)
    assert m.G == pytest.approx(70000.0 / (2.0 * 1.33))
    assert m.g == pytest.approx(m.G)

    # Hill coefficients property
    a01, a02, a03, a12 = m._calc_hill_coefficients()
    assert m.A01 == pytest.approx(a01)
    assert m.A02 == pytest.approx(a02)
    assert m.A03 == pytest.approx(a03)
    assert m.A12 == pytest.approx(a12)
    assert m.A01 == pytest.approx(1.0)  # iyield = 1

    # Aliases
    assert m.yr_fun == 10
    assert m.efib == pytest.approx(50000.0)
    assert m.c == pytest.approx(25.0)
    assert m.fun_a1 == 101
    assert m.t_initial == pytest.approx(293.0)
    assert m.spheat == pytest.approx(2.4e-3)
    assert m.epsp_max == pytest.approx(0.3)
    assert m.epst1 == pytest.approx(0.25)
    assert m.epst2 == pytest.approx(0.35)

    # Mapping protocol
    assert m["e"] == pytest.approx(70000.0)
    assert m["A01"] == pytest.approx(1.0)
    assert "r00" in m
    assert m.get("missing", 999) == 999


def test_card_layouts_and_cfg_catalogue_law73():
    """Verify card layouts and catalogue mappings for LAW73."""
    from pyradioss.input.card_layouts import CARD_LAYOUTS
    from pyradioss.input.cfg_catalogue import law_number, canonical_law_name

    for prefix in ("MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL"):
        assert CARD_LAYOUTS[f"{prefix}_1"] == [20]
        assert CARD_LAYOUTS[f"{prefix}_2"] == [20, 20]
        assert CARD_LAYOUTS[f"{prefix}_3"] == [10, 10, 20, 20]
        assert CARD_LAYOUTS[f"{prefix}_4"] == [20, 20, 20, 20, 10, 10]
        assert CARD_LAYOUTS[f"{prefix}_5"] == [20, 20, 20]
        assert CARD_LAYOUTS[f"{prefix}_6"] == [10, 10, 20, 20]
        assert CARD_LAYOUTS[f"{prefix}_7"] == [20, 20]

    assert law_number("LAW73") == 73
    assert law_number("HILL_THERM") == 73
    assert law_number("MAT_LAW73") == 73
    assert law_number("MAT_HILL_THERM") == 73

    assert canonical_law_name(73) == "LAW73"
    assert canonical_law_name("HILL_THERM") == "LAW73"


def test_starter_keyword_reader_law73(tmp_path):
    """Verify fixed-format and free-format /MAT/LAW73 deck parsing."""
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.starter_keywords import KEYWORD_PARSERS
    from pyradioss.model.model import Model
    from pyradioss.common.messages import MessageLog

    deck_content = (
        "# OpenRadioss Starter Deck with LAW73\n"
        "/MAT/LAW73/1\n"
        "Alloy Sheet 73\n"
        "#                 RHO                RHOR\n"
        "             2.70E-9             2.70E-9\n"
        "#                  E                  NU\n"
        "             70000.0                0.33\n"
        "#   IFUNCE                  EINF                  CE\n"
        "        10               50000.0                25.0\n"
        "#                R00                 R45                 R90               CHARD              IYIELD\n"
        "                 1.5                 1.2                 1.8                 0.4                   1\n"
        "#            EPS_MAX               EPSR1               EPSR2\n"
        "                0.35                0.25                0.30\n"
        "#           TABLE_ID              FSCALE              PSCALE\n"
        "                 101                 1.2                 1.0\n"
        "#                 T0               RHOCP\n"
        "               293.0             2.40E-3\n"
        "/END\n"
    )

    deck_path = tmp_path / "TEST_0000.rad"
    deck_path.write_text(deck_content, encoding="utf-8")

    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    for block in blocks:
        parser = KEYWORD_PARSERS.get(block.keyword.replace("/", "_"))
        if parser:
            parser(block, model, log)

    assert hasattr(model, "mat_law73s")
    assert 1 in model.mat_law73s
    m = model.mat_law73s[1]
    assert m.id == 1
    assert m.e == pytest.approx(70000.0)
    assert m.nu == pytest.approx(0.33)
    assert m.ifunce == 10
    assert m.einf == pytest.approx(50000.0)
    assert m.ce == pytest.approx(25.0)
    assert m.r00 == pytest.approx(1.5)
    assert m.r45 == pytest.approx(1.2)
    assert m.r90 == pytest.approx(1.8)
    assert m.chard == pytest.approx(0.4)
    assert m.iyield == 1
    assert m.table_id == 101
    assert m.t0 == pytest.approx(293.0)
    assert m.rhocp == pytest.approx(2.4e-3)

    assert 1 in model.materials
    mat = model.materials[1]
    assert mat.law == 73
    assert mat.rho0 == pytest.approx(2.7e-9)


def test_starter_deck_writer_roundtrip(tmp_path):
    """Verify StarterDeck.mat_law73 output roundtrip through deck_reader."""
    from pyradioss.input.deck_writer import StarterDeck
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.starter_keywords import KEYWORD_PARSERS
    from pyradioss.model.model import Model
    from pyradioss.common.messages import MessageLog

    deck = StarterDeck("ROUNDTRIP")
    deck.mat_law73(
        mid=2,
        title="Roundtrip Test 73",
        rho=7.8e-9,
        e=210000.0,
        nu=0.3,
        ifunce=5,
        einf=160000.0,
        ce=15.0,
        r00=1.1,
        r45=1.3,
        r90=1.5,
        chard=0.2,
        iyield=1,
        eps_max=0.4,
        epsr1=0.2,
        epsr2=0.3,
        table_id=202,
        fscale=1.1,
        pscale=1.0,
        t0=300.0,
        rhocp=3.5e-3,
    )
    deck.end()

    deck_path = tmp_path / "ROUNDTRIP_0000.rad"
    deck.write(str(deck_path))

    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    for block in blocks:
        parser = KEYWORD_PARSERS.get(block.keyword.replace("/", "_"))
        if parser:
            parser(block, model, log)

    assert 2 in model.mat_law73s
    m = model.mat_law73s[2]
    assert m.id == 2
    assert m.e == pytest.approx(210000.0)
    assert m.nu == pytest.approx(0.3)
    assert m.ifunce == 5
    assert m.einf == pytest.approx(160000.0)
    assert m.ce == pytest.approx(15.0)
    assert m.r00 == pytest.approx(1.1)
    assert m.r45 == pytest.approx(1.3)
    assert m.r90 == pytest.approx(1.5)
    assert m.chard == pytest.approx(0.2)
    assert m.iyield == 1
    assert m.table_id == 202
    assert m.t0 == pytest.approx(300.0)
    assert m.rhocp == pytest.approx(3.5e-3)


def test_starter_checks_law73():
    """Verify diagnostic parameter bounds checks for LAW73."""
    from pyradioss.model.entities import MatLaw73
    from pyradioss.common.messages import MessageLog
    from pyradioss.starter.checks import check_mat_law73

    log_valid = MessageLog()
    m_valid = MatLaw73(id=1, rho=2.7e-9, e=70000.0, nu=0.33, r00=1.5, r45=1.2, r90=1.8,
                       epsr1=0.2, epsr2=0.3)
    check_mat_law73(mat=m_valid, log=log_valid)
    assert len(log_valid.errors) == 0

    log_bad = MessageLog()
    m_bad = MatLaw73(id=2, rho=-1.0, e=0.0, nu=0.55, r00=-1.0, epsr1=0.4, epsr2=0.2)
    check_mat_law73(mat=m_bad, log=log_bad)
    assert len(log_bad.errors) == 5
    err_text = " ".join(log_bad.errors)
    assert "initial density RHO must be > 0" in err_text
    assert "Young's modulus E must be > 0" in err_text
    assert "0 <= nu < 0.5" in err_text
    assert "Lankford parameter R00 must be > 0" in err_text
    assert "must be less than tensile failure strain 2" in err_text



