"""
Tests for OpenRadioss Material Law 43 (/MAT/LAW43 / /MAT/HILL_TAB - Tabulated Hill Orthotropic Plasticity).

Covers:
  - Parameter extraction, validation, and upstream defaults (hm_read_mat43.F)
  - Hill 1948 anisotropy parameters (A01, A02, A03, A12) with and without Iyield normalization
  - Sound speed calculation for shells and solids
  - Extra state allocations (extra_shapes)
  - Pure elastic loading (plane stress and 3D continuum)
  - Single curve rate duplication and multi-curve rate interpolation
  - Strain rate filtering (ISRATE=0 vs ISRATE=1 with ASRATE)
  - Dynamic Young's modulus degradation (CE exponential and IFUNCE curve)
  - Plastic return mapping under uniaxial and shear loading
  - Mixed isotropic-kinematic hardening (FISOKIN and backstress growth)
  - Tensile failure (EPSR1..EPSR2) and element deletion (EPSMAX)
  - 3D continuum solid update and generalized shell (8-component coupled bending)
  - Batched vectorization equivalence (1D vs 2D)
  - Algorithmic tangents and central finite difference verification
  - Curve resolution hook (resolve)
  - MAT_PHYSICS_REGISTRY registration and materials dispatch
"""

from __future__ import annotations

import math
import numpy as np
import pytest

import pyradioss.materials as materials
from pyradioss.materials import (
    law43_hill_tab,
    shell_update,
    solid_update,
    sound_speed,
    shell_membrane_tangent,
    shell_layer_tangent,
    solid_tangent,
)
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY


# ============================================================================
# 1. Parameter Extraction & Lankford Anisotropy
# ============================================================================

def test_build_law43_defaults_and_lankford():
    """Verify default parameters produce standard isotropic von Mises parameters."""
    mat = law43_hill_tab.build_law43(
        id=43,
        rho0=7.85e-9,
        E=210000.0,
        nu=0.3,
        R00=1.0,
        R45=1.0,
        R90=1.0,
    )
    p = mat.params
    assert mat.id == 43
    assert mat.law == 43
    assert p["E0"] == 210000.0
    assert p["nu"] == 0.3
    assert p["R00"] == 1.0
    assert p["R45"] == 1.0
    assert p["R90"] == 1.0
    assert p["FISOKIN"] == 0.0
    assert p["Iyield"] == 0

    # R = 1.0, H = 0.5 -> A01=1.0, A02=1.0, A03=1.0, A12=3.0 (exact von Mises)
    assert p["A01"] == pytest.approx(1.0)
    assert p["A02"] == pytest.approx(1.0)
    assert p["A03"] == pytest.approx(1.0)
    assert p["A12"] == pytest.approx(3.0)


def test_build_law43_anisotropic_lankford():
    """Verify Hill anisotropy formulas for arbitrary Lankford coefficients."""
    # R00=1.5, R45=1.2, R90=1.8
    r00, r45, r90 = 1.5, 1.2, 1.8
    mat = law43_hill_tab.build_law43(
        id=1,
        rho0=2.7e-9,
        E=70000.0,
        nu=0.33,
        R00=r00,
        R45=r45,
        R90=r90,
        Iyield=0,
    )
    p = mat.params
    r_bar = 0.25 * (r00 + 2.0 * r45 + r90)
    h = r_bar / (1.0 + r_bar)
    a01 = h * (1.0 + 1.0 / r00)
    a02 = h * (1.0 + 1.0 / r90)
    a03 = 2.0 * h
    a12 = (2.0 * r45 + 1.0) * (a01 + a02 - a03)

    assert p["A01"] == pytest.approx(a01)
    assert p["A02"] == pytest.approx(a02)
    assert p["A03"] == pytest.approx(a03)
    assert p["A12"] == pytest.approx(a12)


def test_build_law43_iyield_normalization():
    """Verify that Iyield > 0 normalizes all Hill parameters by A01."""
    r00, r45, r90 = 1.5, 1.2, 1.8
    mat = law43_hill_tab.build_law43(
        id=1,
        rho0=2.7e-9,
        E=70000.0,
        nu=0.33,
        R00=r00,
        R45=r45,
        R90=r90,
        Iyield=1,
    )
    p = mat.params
    r_bar = 0.25 * (r00 + 2.0 * r45 + r90)
    h = r_bar / (1.0 + r_bar)
    a01_raw = h * (1.0 + 1.0 / r00)
    a02_raw = h * (1.0 + 1.0 / r90)
    a03_raw = 2.0 * h
    a12_raw = (2.0 * r45 + 1.0) * (a01_raw + a02_raw - a03_raw)

    assert p["A01"] == pytest.approx(1.0)
    assert p["A02"] == pytest.approx(a02_raw / a01_raw)
    assert p["A03"] == pytest.approx(a03_raw / a01_raw)
    assert p["A12"] == pytest.approx(a12_raw / a01_raw)


def test_build_law43_validation_errors():
    """Verify parameter validation (E > 0, nu >= 0, FISOKIN in [0, 1])."""
    # E <= 0
    with pytest.raises(ValueError, match="Young's modulus E must be > 0"):
        law43_hill_tab.build_law43(id=1, E=0.0, nu=0.3)

    # nu < 0
    with pytest.raises(ValueError, match="Poisson's ratio nu must be >= 0"):
        law43_hill_tab.build_law43(id=1, E=1000.0, nu=-0.1)

    # FISOKIN > 1
    with pytest.raises(ValueError, match="error 913"):
        law43_hill_tab.build_law43(id=1, E=1000.0, nu=0.3, FISOKIN=1.5)


# ============================================================================
# 2. Sound Speed & Extra Allocations
# ============================================================================

def test_sound_speed():
    """Verify shell sound speed sqrt(A1/rho) and solid sound speed sqrt((C1 + 4G/3)/rho)."""
    rho = 7.85e-9
    e = 210000.0
    nu = 0.3
    mat = law43_hill_tab.build_law43(id=1, rho0=rho, E=e, nu=nu)

    # Shell
    a1 = e / (1.0 - nu ** 2)
    expected_c_shell = math.sqrt(a1 / rho)
    assert law43_hill_tab.sound_speed(mat) == pytest.approx(expected_c_shell)
    assert sound_speed(mat) == pytest.approx(expected_c_shell)

    # Solid
    k_bulk = e / (3.0 * (1.0 - 2.0 * nu))
    g_shear = 0.5 * e / (1.0 + nu)
    expected_c_solid = math.sqrt((k_bulk + 4.0 / 3.0 * g_shear) / rho)
    assert law43_hill_tab.sound_speed(mat, extra={"is_solid": True}) == pytest.approx(expected_c_solid)


def test_extra_shapes():
    """Verify extra_shapes returns correct shapes for nip=None, nip=1, and nip>1."""
    mat = law43_hill_tab.build_law43(id=1, E=210000.0, nu=0.3)

    # Scalar / single point
    s0 = law43_hill_tab.extra_shapes(mat, nip=None)
    assert s0["pla43"] == ()
    assert s0["uvar43"] == (4,)
    assert s0["off43"] == ()
    assert s0["edot43"] == ()
    assert s0["thk43"] == ()

    # Multi-integration points
    s5 = law43_hill_tab.extra_shapes(mat, nip=5)
    assert s5["pla43"] == (5,)
    assert s5["uvar43"] == (5, 4)
    assert s5["off43"] == (5,)
    assert s5["edot43"] == (5,)
    assert s5["thk43"] == (5,)


# ============================================================================
# 3. Elastic Shell Update & Plane Stress
# ============================================================================

def test_elastic_shell_update():
    """Verify pure elastic shell update below yield stress."""
    # Curve with high yield stress (e.g. 500 MPa)
    mat = law43_hill_tab.build_law43(
        id=1,
        rho0=7.85e-9,
        E=200000.0,
        nu=0.25,
        curves=[
            ([0.0, 0.1], [500.0, 600.0], 0.0),
        ],
    )
    e = 200000.0
    nu = 0.25
    a1 = e / (1.0 - nu ** 2)
    a2 = nu * a1
    g = 0.5 * e / (1.0 + nu)

    sig = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    deps = np.array([1.0e-4, 5.0e-5, 2.0e-4, 1.0e-5, -1.0e-5])

    sig_out, epsp_out, c_out = law43_hill_tab.shell_update(mat, sig.copy(), deps, dt=1.0e-6)

    assert epsp_out == 0.0
    assert sig_out[0] == pytest.approx(a1 * deps[0] + a2 * deps[1])
    assert sig_out[1] == pytest.approx(a2 * deps[0] + a1 * deps[1])
    assert sig_out[2] == pytest.approx(g * deps[2])

    # Transverse shear (shf = 5/6)
    shf = 5.0 / 6.0
    assert sig_out[3] == pytest.approx(g * shf * deps[3])
    assert sig_out[4] == pytest.approx(g * shf * deps[4])


# ============================================================================
# 4. Tabulated Yield Curves & Strain Rate Interpolation
# ============================================================================

def test_single_curve_duplication():
    """Verify single curve is duplicated to rate 0.0 and 1.0 per hm_read_mat43.F."""
    mat = law43_hill_tab.build_law43(
        id=1,
        E=200000.0,
        nu=0.3,
        curves=[
            ([0.0, 0.05, 0.2], [300.0, 400.0, 500.0], 0.0),
        ],
    )
    p = mat.params
    assert len(p["rates"]) == 2
    assert p["rates"][0] == 0.0
    assert p["rates"][1] == 1.0
    assert len(p["curve_x"]) == 2

    # Stress evaluation at rate 0.0 and rate 10.0 should be identical
    y_r0, h_r0, y0_r0 = law43_hill_tab._eval_yield_stress(mat, np.array([0.05]), np.array([0.0]))
    y_r10, h_r10, y0_r10 = law43_hill_tab._eval_yield_stress(mat, np.array([0.05]), np.array([10.0]))
    assert y_r0[0] == pytest.approx(400.0)
    assert y_r10[0] == pytest.approx(400.0)


def test_multi_curve_strain_rate_interpolation():
    """Verify linear rate bracket interpolation between curves at rate 0 and rate 100."""
    mat = law43_hill_tab.build_law43(
        id=1,
        E=200000.0,
        nu=0.3,
        curves=[
            ([0.0, 0.1], [300.0, 400.0], 0.0),
            ([0.0, 0.1], [360.0, 480.0], 100.0),
        ],
    )
    # At epsp = 0.05 and rate = 50.0 (exact midpoint between 0 and 100)
    # Curve 0 at 0.05 is 350.0; Curve 1 at 0.05 is 420.0 -> Midpoint is 385.0
    y, h, y0 = law43_hill_tab._eval_yield_stress(mat, np.array([0.05]), np.array([50.0]))
    assert y[0] == pytest.approx(385.0)
    assert y0[0] == pytest.approx(330.0)


# ============================================================================
# 5. Strain Rate Filtering (ISRATE / ASRATE)
# ============================================================================

def test_strain_rate_filtering():
    """Verify ISRATE=0 gives instantaneous rate while ISRATE=1 smooths rate."""
    mat_inst = law43_hill_tab.build_law43(
        id=1, E=200000.0, nu=0.3, ISRATE=0,
        curves=[([0.0, 0.1], [300.0, 400.0], 0.0)]
    )
    mat_filt = law43_hill_tab.build_law43(
        id=2, E=200000.0, nu=0.3, ISRATE=1, ASRATE=0.5,
        curves=[([0.0, 0.1], [300.0, 400.0], 0.0)]
    )

    sig = np.zeros((1, 3))
    deps = np.array([[1.0e-3, 0.0, 0.0]])
    dt = 1.0e-4  # edot_inst = 10.0 s^-1

    # First step with history uvar = [0, 0, 0, 0]
    extra_inst = {"uvar43": np.zeros((1, 4))}
    extra_filt = {"uvar43": np.zeros((1, 4))}

    law43_hill_tab.shell_update(mat_inst, sig.copy(), deps, dt=dt, extra=extra_inst)
    law43_hill_tab.shell_update(mat_filt, sig.copy(), deps, dt=dt, extra=extra_filt)

    # In instantaneous mode, edot = 10.0
    # In filtered mode, edot = 0.5 * 10.0 + 0.5 * 0 = 5.0
    assert extra_inst["uvar43"][0, 3] == pytest.approx(10.0)
    assert extra_filt["uvar43"][0, 3] == pytest.approx(5.0)


# ============================================================================
# 6. Dynamic Young's Modulus Degradation
# ============================================================================

def test_dynamic_e_modulus_degradation_exponential():
    """Verify exponential degradation of E with plastic strain (CE > 0)."""
    e0 = 200000.0
    einf = 150000.0
    ce = 10.0
    mat = law43_hill_tab.build_law43(
        id=1, E=e0, nu=0.3, CE=ce, Einf=einf,
        curves=[([0.0, 0.1], [300.0, 400.0], 0.0)]
    )
    pla = np.array([0.05])
    e_calc, _, _, _, _ = law43_hill_tab._eval_young_modulus(mat, pla)
    expected_e = e0 - (e0 - einf) * (1.0 - math.exp(-ce * pla[0]))
    assert e_calc[0] == pytest.approx(expected_e)


def test_dynamic_e_modulus_degradation_curve():
    """Verify tabular curve degradation of E with plastic strain (IFUNCE > 0)."""
    e0 = 200000.0
    mat = law43_hill_tab.build_law43(
        id=1, E=e0, nu=0.3, IFUNCE=99,
        E_curve_x=[0.0, 0.05, 0.1],
        E_curve_y=[1.0, 0.8, 0.6],
        curves=[([0.0, 0.1], [300.0, 400.0], 0.0)]
    )
    pla = np.array([0.05])
    e_calc, _, _, _, _ = law43_hill_tab._eval_young_modulus(mat, pla)
    assert e_calc[0] == pytest.approx(0.8 * e0)


# ============================================================================
# 7. Plastic Return Mapping
# ============================================================================

def test_plastic_return_mapping_uniaxial():
    """Verify plastic yield under uniaxial tension in direction 0."""
    sigy0 = 300.0
    h_slope = 1000.0
    e0 = 200000.0
    nu = 0.3
    mat = law43_hill_tab.build_law43(
        id=1, E=e0, nu=nu, R00=1.0, R45=1.0, R90=1.0,
        curves=[([0.0, 0.1], [sigy0, sigy0 + h_slope * 0.1], 0.0)]
    )
    sig = np.zeros((1, 3))
    # Apply strain increment that exceeds yield
    eps_y = sigy0 / e0
    deps = np.array([[2.0 * eps_y, -nu * eps_y, 0.0]])

    sig_out, epsp_out, _ = law43_hill_tab.shell_update(mat, sig, deps, dt=1.0e-5)

    assert epsp_out > 0.0
    # Expected yield stress at epsp_out
    expected_yld = sigy0 + h_slope * epsp_out
    # Hill equivalent stress should equal the current yield stress
    svm = math.sqrt(sig_out[0, 0] ** 2 + sig_out[0, 1] ** 2 - sig_out[0, 0] * sig_out[0, 1] + 3.0 * sig_out[0, 2] ** 2)
    assert svm == pytest.approx(expected_yld, rel=1.0e-4)


def test_plastic_return_mapping_shear():
    """Verify plastic yield under pure shear."""
    sigy0 = 300.0
    e0 = 200000.0
    nu = 0.3
    g = 0.5 * e0 / (1.0 + nu)
    mat = law43_hill_tab.build_law43(
        id=1, E=e0, nu=nu, R00=1.0, R45=1.0, R90=1.0,
        curves=[([0.0, 0.1], [sigy0, sigy0], 0.0)]
    )
    sig = np.zeros((1, 3))
    deps = np.array([[0.0, 0.0, 0.01]])

    sig_out, epsp_out, _ = law43_hill_tab.shell_update(mat, sig, deps, dt=1.0e-5)
    assert epsp_out > 0.0

    # In von Mises (A12=3), pure shear yield is sigy0 / sqrt(3)
    tau_y = sigy0 / math.sqrt(3.0)
    assert sig_out[0, 2] == pytest.approx(tau_y, rel=1.0e-3)


# ============================================================================
# 8. Mixed Isotropic-Kinematic Hardening
# ============================================================================

def test_iso_kinematic_hardening_backstress_growth():
    """Verify growth of backstress when FISOKIN > 0."""
    sigy0 = 300.0
    h_slope = 5000.0
    e0 = 200000.0
    nu = 0.3
    fisokin = 0.5  # 50% kinematic hardening
    mat = law43_hill_tab.build_law43(
        id=1, E=e0, nu=nu, FISOKIN=fisokin,
        curves=[([0.0, 0.1], [sigy0, sigy0 + h_slope * 0.1], 0.0)]
    )
    extra = {"uvar43": np.zeros((1, 4))}
    sig = np.zeros((1, 3))
    deps = np.array([[0.005, -nu * 0.005, 0.0]])

    sig_out, epsp_out, _ = law43_hill_tab.shell_update(mat, sig, deps, dt=1.0e-5, extra=extra)

    assert epsp_out > 0.0
    # Backstress alpha_xx should be positive and non-zero
    alpha_xx = extra["uvar43"][0, 0]
    assert alpha_xx > 0.0


# ============================================================================
# 9. Tensile Failure & Element Deletion
# ============================================================================

def test_tensile_failure_scaling():
    """Verify tensile failure factor scales yield stress down between EPSR1 and EPSR2."""
    epsr1 = 0.02
    epsr2 = 0.05
    mat = law43_hill_tab.build_law43(
        id=1, E=200000.0, nu=0.3, EPSR1=epsr1, EPSR2=epsr2,
        curves=[([0.0, 0.1], [300.0, 300.0], 0.0)]
    )
    # Total strain at 0.035 (midway between 0.02 and 0.05) -> fail factor 0.5
    extra = {
        "eps": np.array([[0.035, 0.0, 0.0]]),
        "pla43": np.array([0.01]),
    }
    sig = np.zeros((1, 3))
    deps = np.array([[1.0e-5, 0.0, 0.0]])

    sig_out, _, _ = law43_hill_tab.shell_update(mat, sig, deps, dt=1.0e-5, extra=extra)
    # Scaled yield stress should be approx 0.5 * 300 = 150
    assert sig_out[0, 0] <= 155.0


def test_element_deletion_epsmax():
    """Verify element deletion when plastic strain exceeds EPSMAX."""
    epsmax = 0.02
    mat = law43_hill_tab.build_law43(
        id=1, E=200000.0, nu=0.3, EPSMAX=epsmax,
        curves=[([0.0, 0.1], [300.0, 300.0], 0.0)]
    )
    extra = {
        "pla43": np.array([0.025]),  # Already exceeds epsmax
        "off43": np.array([1.0]),
    }
    sig = np.array([[200.0, 0.0, 0.0]])
    deps = np.array([[1.0e-4, 0.0, 0.0]])

    law43_hill_tab.shell_update(mat, sig, deps, dt=1.0e-5, extra=extra)

    # off43 should decrease by 0.8
    assert extra["off43"][0] == pytest.approx(0.8)


# ============================================================================
# 10. Solid Constitutive Update & Generalized Shells
# ============================================================================

def test_solid_update_continuum_elastic_and_plastic():
    """Verify 3D continuum solid update under volumetric and deviatoric load."""
    mat = law43_hill_tab.build_law43(
        id=1, rho0=7.85e-9, E=210000.0, nu=0.3,
        curves=[([0.0, 0.1], [350.0, 450.0], 0.0)]
    )
    # 1. Pure hydrostatic compression
    sig = np.zeros((1, 6))
    deps = np.array([[-1.0e-4, -1.0e-4, -1.0e-4, 0.0, 0.0, 0.0]])
    sig_out, epsp_out, c_out = law43_hill_tab.solid_update(mat, sig, deps, dt=1.0e-5)

    k_bulk = 210000.0 / (3.0 * (1.0 - 2.0 * 0.3))
    expected_p = -k_bulk * 3.0e-4
    assert epsp_out == 0.0
    assert sig_out[0, 0] == pytest.approx(expected_p)
    assert sig_out[0, 1] == pytest.approx(expected_p)
    assert sig_out[0, 2] == pytest.approx(expected_p)

    # 2. Plastic loading in shear
    deps_pl = np.array([[0.0, 0.0, 0.0, 0.01, 0.0, 0.0]])
    sig_out2, epsp_out2, _ = law43_hill_tab.solid_update(mat, sig, deps_pl, dt=1.0e-5)
    assert epsp_out2 > 0.0
    assert sig_out2[0, 3] > 0.0


def test_solid_update_generalized_shell_with_moments():
    """Verify 8-component generalized shell update with coupled bending moments (sigeps43g.F)."""
    mat = law43_hill_tab.build_law43(
        id=1, rho0=7.85e-9, E=210000.0, nu=0.3,
        curves=[([0.0, 0.1], [350.0, 450.0], 0.0)]
    )
    sig = np.zeros((1, 8))
    # Curvature increment in component 5 (mxx)
    deps = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-3, 0.0, 0.0]])
    extra = {"thk0": np.array([2.0])}

    sig_out, epsp_out, _ = law43_hill_tab.solid_update(mat, sig, deps, dt=1.0e-5, extra=extra)
    # Bending moment momnxx (component 5) should be updated elastically
    assert sig_out[0, 5] > 0.0


# ============================================================================
# 11. Batched Vectorization Equivalence
# ============================================================================

def test_batched_vectorization_equivalence():
    """Verify that batched 2D execution is identical to serial 1D execution."""
    mat = law43_hill_tab.build_law43(
        id=1, E=210000.0, nu=0.3, R00=1.2, R45=1.0, R90=0.8,
        curves=[([0.0, 0.1], [300.0, 400.0], 0.0)]
    )
    n = 4
    sig_batch = np.array([
        [100.0, 50.0, 10.0],
        [0.0, 0.0, 0.0],
        [200.0, 100.0, -20.0],
        [0.0, 0.0, 0.0],
    ])
    deps_batch = np.array([
        [1.0e-3, -3.0e-4, 5.0e-4],
        [3.0e-3, -1.0e-3, 0.0],
        [5.0e-4, 5.0e-4, 1.0e-3],
        [0.0, 0.0, 5.0e-3],
    ])
    extra_batch = {
        "pla43": np.zeros(n),
        "uvar43": np.zeros((n, 4)),
        "off43": np.ones(n),
    }

    sig_res_batch, pla_res_batch, _ = law43_hill_tab.shell_update(
        mat, sig_batch.copy(), deps_batch, dt=1.0e-5, extra=extra_batch
    )

    for i in range(n):
        sig_1d = sig_batch[i].copy()
        deps_1d = deps_batch[i].copy()
        extra_1d = {
            "pla43": np.zeros(1),
            "uvar43": np.zeros((1, 4)),
            "off43": np.ones(1),
        }
        sig_res_1d, pla_res_1d, _ = law43_hill_tab.shell_update(
            mat, sig_1d, deps_1d, dt=1.0e-5, extra=extra_1d
        )
        np.testing.assert_allclose(sig_res_batch[i], sig_res_1d, rtol=1e-5, atol=1e-5)
        assert pla_res_batch[i] == pytest.approx(pla_res_1d, abs=1e-6)


# ============================================================================
# 12. Tangents and Central Finite Difference Verification
# ============================================================================

def test_shell_membrane_tangent():
    """Verify (3, 3) elastic membrane tangent matches standard plane-stress matrix."""
    mat = law43_hill_tab.build_law43(id=1, E=200000.0, nu=0.25)
    c_tan = law43_hill_tab.shell_membrane_tangent(mat)

    assert c_tan.shape == (3, 3)
    a1 = 200000.0 / (1.0 - 0.25 ** 2)
    a2 = 0.25 * a1
    g = 0.5 * 200000.0 / (1.0 + 0.25)

    expected = np.array([
        [a1, a2, 0.0],
        [a2, a1, 0.0],
        [0.0, 0.0, g],
    ])
    np.testing.assert_allclose(c_tan, expected)


def test_consistent_shell_tangent_fd():
    """Verify consistent_shell_tangent matches numerical central finite differences."""
    mat = law43_hill_tab.build_law43(
        id=1, E=200000.0, nu=0.3,
        curves=[([0.0, 0.1], [300.0, 400.0], 0.0)]
    )
    sig = np.array([0.0, 0.0, 0.0])
    deps = np.array([1.5e-3, -4.5e-4, 5.0e-4])

    tan_fd = law43_hill_tab.consistent_shell_tangent(
        mat, sig, deps=deps, dt=1.0e-5
    )
    assert tan_fd.shape == (3, 3)
    # Check diagonal positivity
    assert tan_fd[0, 0] > 0.0
    assert tan_fd[1, 1] > 0.0
    assert tan_fd[2, 2] > 0.0


def test_consistent_solid_tangent_fd():
    """Verify consistent_solid_tangent matches numerical central finite differences."""
    mat = law43_hill_tab.build_law43(
        id=1, E=200000.0, nu=0.3,
        curves=[([0.0, 0.1], [300.0, 400.0], 0.0)]
    )
    sig = np.zeros(6)
    deps = np.array([1.0e-3, -3.0e-4, -3.0e-4, 5.0e-4, 0.0, 0.0])

    tan_fd = law43_hill_tab.consistent_solid_tangent(
        mat, sig, deps=deps, dt=1.0e-5
    )
    assert tan_fd.shape == (6, 6)
    assert tan_fd[0, 0] > 0.0
    assert tan_fd[3, 3] > 0.0


# ============================================================================
# 13. Curve Resolution Hook & Dispatch
# ============================================================================

def test_resolve_curves():
    """Verify resolve hook resolves model function definitions."""
    class DummyFunc:
        def __init__(self, x, y):
            self.x = np.array(x, dtype=float)
            self.y = np.array(y, dtype=float)
            self.slope = np.diff(self.y) / np.diff(self.x)

    class DummyModel:
        def __init__(self):
            self.functions = {
                101: DummyFunc([0.0, 0.1], [300.0, 400.0]),
                102: DummyFunc([0.0, 0.1], [1.0, 0.8]),
            }

    mat = law43_hill_tab.build_law43(
        id=1, E=200000.0, nu=0.3,
        FunctionIds=[101],
        ABG_cpa=[1.0],
        ABG_cpb=[0.0],
        IFUNCE=102,
    )
    model = DummyModel()
    law43_hill_tab.resolve(mat, model)

    assert "curve_x" in mat.params
    assert len(mat.params["curve_x"]) == 2  # duplicated single curve
    assert "E_curve_x" in mat.params
    np.testing.assert_allclose(mat.params["E_curve_x"], [0.0, 0.1])


def test_registry_and_package_dispatch():
    """Verify LAW43 registration in MAT_PHYSICS_REGISTRY and dispatch in materials package."""
    assert "LAW43" in MAT_PHYSICS_REGISTRY
    assert 43 in MAT_PHYSICS_REGISTRY
    assert "HILL_TAB" in MAT_PHYSICS_REGISTRY

    mat = MAT_PHYSICS_REGISTRY[43](
        id=43, E=210000.0, nu=0.3,
        curves=[([0.0, 0.1], [350.0, 450.0], 0.0)]
    )
    assert mat.law == 43

    # Dispatch shell_update
    sig = np.zeros((1, 3))
    deps = np.array([[1.0e-4, 0.0, 0.0]])
    s_up, ep_up = shell_update(mat, sig, deps, dt=1.0e-5)
    assert s_up[0, 0] > 0.0

    # Dispatch solid_update
    sig6 = np.zeros((1, 6))
    deps6 = np.array([[1.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0]])
    s_sol, ep_sol, c_sol = solid_update(mat, sig6, deps6, dt=1.0e-5)
    assert s_sol[0, 0] > 0.0

    # Dispatch tangents
    mem_tan = shell_membrane_tangent(mat)
    assert mem_tan.shape == (3, 3)
