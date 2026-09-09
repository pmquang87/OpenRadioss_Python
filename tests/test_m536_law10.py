"""Comprehensive unit test suite for Milestone M536: /MAT/LAW10 (/MAT/SOIL, /MAT/DPRAG).

Fortran origins & reference files:
- ``engine/source/materials/mat/mat010/m10law.F`` (solid constitutive update)
- ``common_source/eos/compaction.F90`` (compaction equation of state)
- ``starter/source/materials/mat/mat010/hm_read_mat10.F`` (starter reader, defaults, parameter estimation)
- ``C:\\OpenRadioss\\hm_cfg_files\\config\\CFG\\radioss2020\\MAT\\matl10_law10.cfg`` (CFG card specification)

Covers all 32 required milestone tests:
1.  test_law10_parameter_validation_valid
2.  test_law10_parameter_validation_negative_density
3.  test_law10_parameter_validation_negative_young
4.  test_law10_parameter_validation_invalid_nu
5.  test_law10_parameter_defaults
6.  test_law10_pstar_calculation_linear
7.  test_law10_pstar_calculation_quadratic_real_roots
8.  test_law10_pstar_calculation_quadratic_no_real_roots
9.  test_law10_pure_hydrostatic_compression_cubic
10. test_law10_pure_hydrostatic_tension_linear
11. test_law10_compaction_unloading_iform1
12. test_law10_compaction_unloading_iform2
13. test_law10_fracture_pressure_cutoff
14. test_law10_elastic_deviatoric_shear
15. test_law10_plastic_radial_return
16. test_law10_plastic_strain_accumulation
17. test_law10_pressure_dependent_yielding
18. test_law10_von_mises_cap_cutoff
19. test_law10_tension_cutoff_shear_collapse
20. test_law10_closure_pressure_shear_collapse
21. test_law10_sound_speed_elastic
22. test_law10_sound_speed_compaction
23. test_law10_consistent_solid_tangent_elastic
24. test_law10_consistent_solid_tangent_plastic
25. test_law10_tangent_numerical_directional_derivative
26. test_law10_spatial_rotation_objectivity
27. test_law10_shell_update_raises_not_implemented
28. test_law10_batched_vectorization_equivalence
29. test_law10_empty_array_safety
30. test_law10_zero_dt_no_op
31. test_law10_deck_writer_roundtrip
32. test_law10_starter_checks_integration
plus supplementary tests for registration, element deactivation ('off'), and calling conventions.
"""

from __future__ import annotations

import math
import tempfile
import numpy as np
import pytest

from pyradioss.materials.law10_soil import (
    build_law10,
    _ensure_params,
    solid_update,
    shell_update,
    sound_speed,
    consistent_solid_tangent,
)
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.starter.checks import check_model


# =============================================================================
# 1. Parameter Validation and Defaults (Tests 1-8)
# =============================================================================

def test_law10_parameter_validation_valid():
    """Test 1: Valid parameters produce expected derived values (G, K, c1, Amax, pmin, pstar)."""
    rec = {
        "id": 1,
        "MAT_RHO": 2000.0,
        "MAT_E": 3.0e7,
        "MAT_NU": 0.25,
        "MAT_A0": 1.0e6,
        "MAT_A1": 0.5,
        "MAT_A2": 0.0,
    }
    mat = build_law10(rec)
    p = mat.params
    assert mat.law == 10
    assert mat.rho0 == 2000.0

    expected_g = 3.0e7 / (2.0 * (1.0 + 0.25))  # 1.2e7
    expected_k = 3.0e7 / (3.0 * (1.0 - 2.0 * 0.25))  # 2.0e7
    assert math.isclose(p["G"], expected_g)
    assert math.isclose(p["K"], expected_k)

    # Derived defaults
    assert p["c1"] == expected_k
    assert p["Amax"] == 1e20
    assert p["pmin"] == -1e30
    assert p["pext"] == 0.0
    # Linear root: pstar = -A0 / A1
    assert math.isclose(p["pstar"], -1.0e6 / 0.5)


def test_law10_parameter_validation_negative_density():
    """Test 2: Raises ValueError when rho0 <= 0."""
    with pytest.raises(ValueError, match="Density rho0 must be > 0"):
        build_law10({"MAT_RHO": -100.0, "MAT_E": 1.0e7, "MAT_NU": 0.2})

    with pytest.raises(ValueError, match="Density rho0 must be > 0"):
        build_law10({"MAT_RHO": 0.0, "MAT_E": 1.0e7, "MAT_NU": 0.2})


def test_law10_parameter_validation_negative_young():
    """Test 3: Raises ValueError when E <= 0."""
    with pytest.raises(ValueError, match="Young's modulus E must be > 0"):
        build_law10({"MAT_RHO": 2000.0, "MAT_E": -1.0e7, "MAT_NU": 0.2})

    with pytest.raises(ValueError, match="Young's modulus E must be > 0"):
        build_law10({"MAT_RHO": 2000.0, "MAT_E": 0.0, "MAT_NU": 0.2})


def test_law10_parameter_validation_invalid_nu():
    """Test 4: Raises ValueError when nu < 0 or nu >= 0.5."""
    with pytest.raises(ValueError, match="Poisson's ratio nu must be in"):
        build_law10({"MAT_RHO": 2000.0, "MAT_E": 1.0e7, "MAT_NU": 0.5})

    with pytest.raises(ValueError, match="Poisson's ratio nu must be in"):
        build_law10({"MAT_RHO": 2000.0, "MAT_E": 1.0e7, "MAT_NU": 0.6})

    with pytest.raises(ValueError, match="Poisson's ratio nu must be in"):
        build_law10({"MAT_RHO": 2000.0, "MAT_E": 1.0e7, "MAT_NU": -0.1})


def test_law10_parameter_defaults():
    """Test 5: Verifies Amax defaults to 1e20, pmin defaults to -1e30, pext defaults to 0.0."""
    p = _ensure_params({"rho0": 1800.0, "E": 2.5e7, "nu": 0.3})
    assert p["Amax"] == 1e20
    assert p["pmin"] == -1e30
    assert p["pext"] == 0.0
    assert p["psh"] == 0.0
    assert p["c0"] == 0.0
    assert p["c2"] == 0.0
    assert p["c3"] == 0.0
    assert p["iform"] == 1
    assert p["mue_max"] == 1e20
    assert math.isclose(p["bunl"], p["K"])


def test_law10_pstar_calculation_linear():
    """Test 6: A2 == 0, A1 != 0 gives pstar = -A0 / A1."""
    p = _ensure_params({"rho0": 1000.0, "E": 1e6, "nu": 0.2, "A0": 500.0, "A1": 2.0, "A2": 0.0})
    assert math.isclose(p["pstar"], -250.0)


def test_law10_pstar_calculation_quadratic_real_roots():
    """Test 7: A2 != 0 with positive delta gives root pstar = (-A1 + sqrt(delta)) / (2*A2)."""
    p = _ensure_params({"rho0": 1000.0, "E": 1e6, "nu": 0.2, "A0": 100.0, "A1": 20.0, "A2": 0.5})
    delta = 20.0**2 - 4.0 * 100.0 * 0.5  # 400 - 200 = 200 > 0
    expected = (-20.0 + math.sqrt(delta)) / (2.0 * 0.5)
    assert math.isclose(p["pstar"], expected)


def test_law10_pstar_calculation_quadratic_no_real_roots():
    """Test 8: A2 != 0 with negative delta gives extremum vertex pstar = -A1 / (2*A2); A1=A2=0 gives -1e30."""
    p1 = _ensure_params({"rho0": 1000.0, "E": 1e6, "nu": 0.2, "A0": 1000.0, "A1": 10.0, "A2": 1.0})
    # delta = 100 - 4000 < 0
    expected = -10.0 / (2.0 * 1.0)
    assert math.isclose(p1["pstar"], expected)

    # Constant envelope A1 == 0, A2 == 0
    p2 = _ensure_params({"rho0": 1000.0, "E": 1e6, "nu": 0.2, "A0": 100.0, "A1": 0.0, "A2": 0.0})
    assert p2["pstar"] == -1e30


# =============================================================================
# 2. Compaction EOS & Pressure Response (Tests 9-13)
# =============================================================================

def test_law10_pure_hydrostatic_compression_cubic():
    """Test 9: Under volumetric compression (mu > 0), pressure follows cubic EOS P(mu) = c0 + c1*mu + (c2 + c3*mu)*mu^2."""
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 3.0e7,
        "MAT_NU": 0.25,
        "c0": 1.0e4,
        "c1": 2.0e6,
        "c2": 5.0e6,
        "c3": 1.0e6,
        "A0": 1.0e12,  # Strictly elastic deviator
    })
    sig = np.zeros((1, 6), dtype=float)
    deps = np.array([[-0.01, -0.01, -0.01, 0.0, 0.0, 0.0]], dtype=float)
    extra = {}

    sig_new = solid_update(mat, sig, d_eps=deps, dt=1e-4, extra=extra)

    mu = 0.03
    mu2 = mu * mu
    expected_p = 1.0e4 + 2.0e6 * mu + (5.0e6 + 1.0e6 * mu) * mu2

    np.testing.assert_allclose(sig_new[0, 0], -expected_p, rtol=1e-5)
    np.testing.assert_allclose(sig_new[0, 1], -expected_p, rtol=1e-5)
    np.testing.assert_allclose(sig_new[0, 2], -expected_p, rtol=1e-5)
    np.testing.assert_allclose(sig_new[0, 3:], 0.0, atol=1e-12)
    assert math.isclose(extra["mu_bak"][0], 0.03)


def test_law10_pure_hydrostatic_tension_linear():
    """Test 10: Under volumetric tension (mu < 0), pressure follows linear EOS P(mu) = c0 + c1*mu."""
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 3.0e7,
        "MAT_NU": 0.25,
        "c0": 2000.0,
        "c1": 5.0e6,
        "c2": 8.0e6,
        "c3": 3.0e6,
        "pmin": -1.0e8,
        "A0": 1.0e12,
    })
    sig = np.zeros((1, 6), dtype=float)
    # Tensile volumetric strain: deps_ii > 0 -> mu = -tr(deps) = -0.012
    deps = np.array([[+0.004, +0.004, +0.004, 0.0, 0.0, 0.0]], dtype=float)
    extra = {}

    sig_new = solid_update(mat, sig, d_eps=deps, dt=1e-4, extra=extra)

    mu = -0.012
    expected_p = 2000.0 + 5.0e6 * mu  # Linear in tension, higher-order terms omitted

    np.testing.assert_allclose(sig_new[0, 0], -expected_p, rtol=1e-5)
    np.testing.assert_allclose(sig_new[0, 1], -expected_p, rtol=1e-5)
    np.testing.assert_allclose(sig_new[0, 2], -expected_p, rtol=1e-5)
    np.testing.assert_allclose(sig_new[0, 3:], 0.0, atol=1e-12)


def test_law10_compaction_unloading_iform1():
    """Test 11: Compaction unloading with iform=1 uses constant slope bunl."""
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 3.0e7,
        "MAT_NU": 0.25,
        "c0": 0.0,
        "c1": 1.0e6,
        "c2": 2.0e6,
        "c3": 0.0,
        "bunl": 6.0e6,
        "iform": 1,
        "A0": 1.0e12,
    })
    sig = np.zeros((1, 6), dtype=float)
    # 1. Load to mu = 0.06
    extra = {}
    deps_load = np.array([[-0.02, -0.02, -0.02, 0.0, 0.0, 0.0]])
    sig_load = solid_update(mat, sig, d_eps=deps_load, dt=1e-4, extra=extra)
    p_bak = 1.0e6 * 0.06 + 2.0e6 * (0.06**2)
    np.testing.assert_allclose(sig_load[0, 0], -p_bak, rtol=1e-5)
    assert math.isclose(extra["mu_bak"][0], 0.06)

    # 2. Unload to mu = 0.045 (delta_mu = -0.015, tension increment)
    deps_unl = np.array([[+0.005, +0.005, +0.005, 0.0, 0.0, 0.0]])
    sig_unl = solid_update(mat, sig_load, d_eps=deps_unl, dt=1e-4, extra=extra)
    # p_unl = p_bak - (mu_bak - mu) * bunl = p_bak - 0.015 * 6e6
    expected_p_unl = p_bak - 0.015 * 6.0e6
    np.testing.assert_allclose(sig_unl[0, 0], -expected_p_unl, rtol=1e-5)
    # mu_bak remains 0.06
    assert math.isclose(extra["mu_bak"][0], 0.06)


def test_law10_compaction_unloading_iform2():
    """Test 12: Compaction unloading with iform=2 uses continuous slope alpha*bunl + (1-alpha)*c1."""
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 3.0e7,
        "MAT_NU": 0.25,
        "c0": 0.0,
        "c1": 1.0e6,
        "c2": 2.0e6,
        "c3": 0.0,
        "bunl": 5.0e6,
        "mue_max": 0.1,
        "iform": 2,
        "A0": 1.0e12,
    })
    sig = np.zeros((1, 6), dtype=float)
    extra = {}
    # Load to mu = 0.05
    deps_load = np.array([[-0.05/3.0, -0.05/3.0, -0.05/3.0, 0.0, 0.0, 0.0]])
    sig_load = solid_update(mat, sig, d_eps=deps_load, dt=1e-4, extra=extra)
    p_bak = 1.0e6 * 0.05 + 2.0e6 * (0.05**2)

    # Unload to mu = 0.04 (delta_mu = -0.01)
    deps_unl = np.array([[+0.01/3.0, +0.01/3.0, +0.01/3.0, 0.0, 0.0, 0.0]])
    sig_unl = solid_update(mat, sig_load, d_eps=deps_unl, dt=1e-4, extra=extra)

    # alpha = mu_bak / mue_max = 0.05 / 0.1 = 0.5
    # b_eff = 0.5 * 5e6 + 0.5 * 1e6 = 3e6
    expected_p_unl = p_bak - 0.01 * 3.0e6
    np.testing.assert_allclose(sig_unl[0, 0], -expected_p_unl, rtol=1e-5)


def test_law10_fracture_pressure_cutoff():
    """Test 13: Tension exceeding fracture pressure pmin clamps pressure to pmin and zero-collapses deviatoric stress."""
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 1.0e7,
        "MAT_NU": 0.25,
        "c0": 0.0,
        "c1": 1.0e7,
        "pmin": -2.0e4,
        "A0": 1.0e6,
    })
    sig = np.zeros((1, 6), dtype=float)
    # Tensile expansion producing P = -5e4 < pmin (-2e4), with simultaneous shear strain
    # mu = -0.005 -> P = 1e7 * (-0.005) = -5e4
    deps = np.array([[+0.005/3.0, +0.005/3.0, +0.005/3.0, 2.0e-3, 0.0, 0.0]])
    extra = {}
    sig_out = solid_update(mat, sig, d_eps=deps, dt=1e-4, extra=extra)

    # Clamped pressure: P = pmin = -2.0e4 -> sigma_ii = -(-2e4) = +2e4
    np.testing.assert_allclose(sig_out[0, :3], 2.0e4, rtol=1e-5)
    # Deviatoric stresses completely zeroed
    np.testing.assert_allclose(sig_out[0, 3:], 0.0, atol=1e-12)
    assert extra["g0"][0] == 0.0
    assert extra["ratio"][0] == 0.0


# =============================================================================
# 3. Deviatoric Plasticity & Yield Surface (Tests 14-20)
# =============================================================================

def test_law10_elastic_deviatoric_shear():
    """Test 14: Pure shear strain within elastic limit preserves trial stress with ratio=1.0 and zero plastic strain."""
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 1.0e7,
        "MAT_NU": 0.25,
        "A0": 1.0e6,
        "A1": 0.0,
        "A2": 0.0,
        "c0": 0.0,
        "c1": 1.0e7,
    })
    g = 1.0e7 / (2.0 * 1.25)  # 4.0e6
    sig = np.zeros((1, 6), dtype=float)
    deps = np.array([[0.0, 0.0, 0.0, 1.0e-4, 0.0, 0.0]])
    extra = {}

    sig_out = solid_update(mat, sig, d_eps=deps, dt=1e-4, extra=extra)
    # s_xy = G * dgamma = 4.0e6 * 1.0e-4 = 400.0
    # J2 = 400^2 = 1.6e5 <= A0 (1e6)
    np.testing.assert_allclose(sig_out[0, 3], 400.0)
    assert extra["ratio"][0] == 1.0
    assert extra["epxe"][0] == 0.0


def test_law10_plastic_radial_return():
    """Test 15: Radial return scales deviatoric stress onto yield surface J2 = G0."""
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 1.0e7,
        "MAT_NU": 0.25,
        "A0": 1.0e6,
        "A1": 0.0,
        "A2": 0.0,
        "c0": 0.0,
        "c1": 1.0e7,
    })
    sig = np.zeros((1, 6), dtype=float)
    deps = np.array([[0.0, 0.0, 0.0, 1.0e-3, 0.0, 0.0]])
    extra = {}

    sig_out = solid_update(mat, sig, d_eps=deps, dt=1e-4, extra=extra)
    # s_tr_xy = 4000.0 -> J2_tr = 1.6e7 > 1e6
    # ratio = sqrt(1e6 / 1.6e7) = 0.25 -> s_xy = 1000.0
    np.testing.assert_allclose(sig_out[0, 3], 1000.0, rtol=1e-5)
    np.testing.assert_allclose(extra["ratio"][0], 0.25, rtol=1e-5)
    j2_new = sig_out[0, 3] ** 2
    np.testing.assert_allclose(j2_new, 1.0e6, rtol=1e-5)


def test_law10_plastic_strain_accumulation():
    """Test 16: Plastic strain increments accumulate correctly across sequential load steps."""
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 1.0e7,
        "MAT_NU": 0.25,
        "A0": 1.0e6,
        "A1": 0.0,
        "c0": 0.0,
        "c1": 1.0e7,
    })
    g = 1.0e7 / (2.0 * 1.25)  # 4e6
    sig = np.zeros((1, 6), dtype=float)
    extra = {}

    # Step 1: deps_xy = 1e-3
    deps1 = np.array([[0.0, 0.0, 0.0, 1.0e-3, 0.0, 0.0]])
    sig1 = solid_update(mat, sig, d_eps=deps1, dt=1e-4, extra=extra)
    ratio1 = 0.25
    expected_dpla1 = (1.0 - ratio1) * math.sqrt(3.0 * 1.6e7) / (3.0 * g)
    np.testing.assert_allclose(extra["epxe"][0], expected_dpla1, rtol=1e-5)

    # Step 2: additional plastic shear deps_xy = 5e-4
    deps2 = np.array([[0.0, 0.0, 0.0, 5.0e-4, 0.0, 0.0]])
    sig2 = solid_update(mat, sig1, d_eps=deps2, dt=1e-4, extra=extra)
    # s_tr_xy = 1000.0 + 4e6 * 5e-4 = 3000.0 -> J2_tr = 9e6
    ratio2 = math.sqrt(1e6 / 9e6)  # 1/3
    expected_dpla2 = (1.0 - ratio2) * math.sqrt(3.0 * 9.0e6) / (3.0 * g)
    expected_total_epxe = expected_dpla1 + expected_dpla2
    np.testing.assert_allclose(extra["epxe"][0], expected_total_epxe, rtol=1e-5)


def test_law10_pressure_dependent_yielding():
    """Test 17: Higher confining pressure elevates Drucker-Prager yield limit G0 and increases shear strength."""
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 1.0e7,
        "MAT_NU": 0.25,
        "A0": 1.0e6,
        "A1": 3.0,
        "A2": 0.0,
        "c0": 0.0,
        "c1": 1.0e7,
    })
    # Case A: Low pressure P = 1e5 -> G0 = 1e6 + 3*1e5 = 1.3e6
    deps_a = np.array([[-1e5/1e7/3.0, -1e5/1e7/3.0, -1e5/1e7/3.0, 2.0e-3, 0.0, 0.0]])
    extra_a = {}
    sig_a = solid_update(mat, np.zeros((1, 6)), d_eps=deps_a, dt=1e-4, extra=extra_a)

    # Case B: High pressure P = 8e5 -> G0 = 1e6 + 3*8e5 = 3.4e6
    deps_b = np.array([[-8e5/1e7/3.0, -8e5/1e7/3.0, -8e5/1e7/3.0, 2.0e-3, 0.0, 0.0]])
    extra_b = {}
    sig_b = solid_update(mat, np.zeros((1, 6)), d_eps=deps_b, dt=1e-4, extra=extra_b)

    assert extra_b["g0"][0] > extra_a["g0"][0]
    assert sig_b[0, 3] > sig_a[0, 3]
    np.testing.assert_allclose(sig_a[0, 3] ** 2, 1.3e6, rtol=1e-5)
    np.testing.assert_allclose(sig_b[0, 3] ** 2, 3.4e6, rtol=1e-5)


def test_law10_von_mises_cap_cutoff():
    """Test 18: Yield envelope G0 is capped at Amax under high confining pressure."""
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 1.0e7,
        "MAT_NU": 0.25,
        "A0": 1.0e6,
        "A1": 5.0,
        "Amax": 2.2e6,  # Cap at 2.2e6
        "c0": 0.0,
        "c1": 1.0e7,
    })
    # P = 1e6 -> uncapped G0 = 1e6 + 5*1e6 = 6e6 > Amax
    deps = np.array([[-1e6/1e7/3.0, -1e6/1e7/3.0, -1e6/1e7/3.0, 2.0e-3, 0.0, 0.0]])
    extra = {}
    sig_out = solid_update(mat, np.zeros((1, 6)), d_eps=deps, dt=1e-4, extra=extra)

    np.testing.assert_allclose(extra["g0"][0], 2.2e6, rtol=1e-5)
    np.testing.assert_allclose(sig_out[0, 3] ** 2, 2.2e6, rtol=1e-5)


def test_law10_tension_cutoff_shear_collapse():
    """Test 19: When pressure reaches tension cutoff pmin, shear strength collapses to zero."""
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 1.0e7,
        "MAT_NU": 0.25,
        "A0": 1.0e6,
        "pmin": -5.0e3,
        "c0": 0.0,
        "c1": 1.0e7,
    })
    # Tensile expansion producing P = -1e4 <= pmin (-5e3)
    deps = np.array([[+1e4/1e7/3.0, +1e4/1e7/3.0, +1e4/1e7/3.0, 1.0e-3, 0.0, 0.0]])
    extra = {}
    sig_out = solid_update(mat, np.zeros((1, 6)), d_eps=deps, dt=1e-4, extra=extra)

    assert extra["g0"][0] == 0.0
    assert extra["ratio"][0] == 0.0
    np.testing.assert_allclose(sig_out[0, 3:], 0.0, atol=1e-12)


def test_law10_closure_pressure_shear_collapse():
    """Test 20: When P_tot <= pstar (pressure axis root closure), shear strength collapses to zero."""
    # A0 = 1e6, A1 = 2.0, A2 = 0 -> pstar = -1e6 / 2.0 = -5e5
    # Let pmin = -1e9 so that pmin does not trigger first
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 1.0e7,
        "MAT_NU": 0.25,
        "A0": 1.0e6,
        "A1": 2.0,
        "A2": 0.0,
        "pmin": -1.0e9,
        "c0": 0.0,
        "c1": 1.0e7,
    })
    # Expand to P = -6e5 < pstar (-5e5)
    deps = np.array([[+6e5/1e7/3.0, +6e5/1e7/3.0, +6e5/1e7/3.0, 1.0e-3, 0.0, 0.0]])
    extra = {}
    sig_out = solid_update(mat, np.zeros((1, 6)), d_eps=deps, dt=1e-4, extra=extra)

    assert extra["g0"][0] == 0.0
    assert extra["ratio"][0] == 0.0
    np.testing.assert_allclose(sig_out[0, 3:], 0.0, atol=1e-12)


# =============================================================================
# 4. Wave Speed & Tangent Operators (Tests 21-25)
# =============================================================================

def test_law10_sound_speed_elastic():
    """Test 21: Elastic sound speed c = sqrt((K + 4/3*G) / rho0) when c1=K, bunl=K."""
    mat = build_law10({
        "MAT_RHO": 2200.0,
        "MAT_E": 3.3e7,
        "MAT_NU": 0.2,
    })
    g = 3.3e7 / (2.0 * 1.2)
    k = 3.3e7 / (3.0 * (1.0 - 2.0 * 0.2))
    expected_c = math.sqrt((k + (4.0 / 3.0) * g) / 2200.0)

    c = sound_speed(mat)
    assert math.isclose(c, expected_c)


def test_law10_sound_speed_compaction():
    """Test 22: Compaction sound speed uses K_eff = max(c1, bunl) and handles density array."""
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 3.0e7,
        "MAT_NU": 0.25,
        "c1": 2.0e7,
        "bunl": 5.0e7,
    })
    g = 3.0e7 / (2.0 * 1.25)
    k_eff = 5.0e7
    expected_c = math.sqrt((k_eff + (4.0 / 3.0) * g) / 2000.0)

    assert math.isclose(sound_speed(mat), expected_c)

    # Test density array
    rho_arr = np.array([2000.0, 1800.0])
    c_arr = sound_speed(mat, rho=rho_arr)
    assert c_arr.shape == (2,)
    assert math.isclose(c_arr[0], expected_c)
    assert math.isclose(c_arr[1], math.sqrt((k_eff + (4.0 / 3.0) * g) / 1800.0))


def test_law10_consistent_solid_tangent_elastic():
    """Test 23: In elastic regime, consistent tangent matches 3D isotropic elasticity matrix."""
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 1.2e7,
        "MAT_NU": 0.2,
        "A0": 1.0e12,
    })
    p = mat.params
    g = p["G"]
    k = p["K"]
    sig = np.zeros((1, 6), dtype=float)
    extra = {"ratio": np.array([1.0]), "j2": np.array([0.0])}

    d_tangent = consistent_solid_tangent(mat, sig, extra=extra)
    assert d_tangent.shape == (1, 6, 6)

    c11 = k + (4.0 / 3.0) * g
    c12 = k - (2.0 / 3.0) * g
    np.testing.assert_allclose(d_tangent[0, 0, 0], c11)
    np.testing.assert_allclose(d_tangent[0, 1, 1], c11)
    np.testing.assert_allclose(d_tangent[0, 2, 2], c11)
    np.testing.assert_allclose(d_tangent[0, 0, 1], c12)
    np.testing.assert_allclose(d_tangent[0, 3, 3], g)
    np.testing.assert_allclose(d_tangent[0, 4, 4], g)
    np.testing.assert_allclose(d_tangent[0, 5, 5], g)


def test_law10_consistent_solid_tangent_plastic():
    """Test 24: In plastic regime, tangent exhibits stiffness reduction and finite plastic flow contribution."""
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 1.0e7,
        "MAT_NU": 0.25,
        "A0": 1.0e6,
        "A1": 0.0,
        "c0": 0.0,
        "c1": 1.0e7,
    })
    g = mat.params["G"]
    sig = np.zeros((1, 6), dtype=float)
    deps = np.array([[0.0, 0.0, 0.0, 2.0e-3, 0.0, 0.0]])
    extra = {}
    sig_cur = solid_update(mat, sig, d_eps=deps, dt=1e-4, extra=extra)

    d_tang = consistent_solid_tangent(mat, sig_cur, extra=extra)
    assert d_tang.shape == (1, 6, 6)
    assert not np.isnan(d_tang).any()
    assert not np.isinf(d_tang).any()
    # Shear tangent D44 must be strictly lower than elastic shear modulus G
    assert d_tang[0, 3, 3] < g


def test_law10_tangent_numerical_directional_derivative():
    """Test 25: Tangent directional derivative D : delta_eps matches finite-difference perturbation of solid_update."""
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 1.0e7,
        "MAT_NU": 0.25,
        "A0": 1.0e6,
        "A1": 0.0,
        "c0": 0.0,
        "c1": 1.0e7,
    })
    sig_base = np.zeros((1, 6), dtype=float)
    deps_base = np.array([[0.0, 0.0, 0.0, 2.0e-3, 0.0, 0.0]])
    extra = {}
    sig_cur = solid_update(mat, sig_base, d_eps=deps_base, dt=1e-4, extra=extra)

    d_tang = consistent_solid_tangent(mat, sig_cur, extra=extra)[0]

    # Perturb strain increment by delta_eps
    delta_eps = np.array([1e-6, -5e-7, -5e-7, 2e-6, 0.0, 0.0])
    deps_perturbed = deps_base + delta_eps[None, :]

    extra_p = {}
    sig_perturbed = solid_update(mat, sig_base, d_eps=deps_perturbed, dt=1e-4, extra=extra_p)

    d_sig_fd = sig_perturbed[0] - sig_cur[0]
    d_sig_tangent = d_tang @ delta_eps

    np.testing.assert_allclose(d_sig_tangent, d_sig_fd, rtol=1e-3, atol=1e-2)


# =============================================================================
# 5. Objectivity, Safety & Invariance (Tests 26-30)
# =============================================================================

def _tensor_to_voigt_stress(T: np.ndarray) -> np.ndarray:
    return np.array([T[0, 0], T[1, 1], T[2, 2], T[0, 1], T[1, 2], T[2, 0]])


def _voigt_to_tensor_stress(v: np.ndarray) -> np.ndarray:
    return np.array([
        [v[0], v[3], v[5]],
        [v[3], v[1], v[4]],
        [v[5], v[4], v[2]],
    ])


def _tensor_to_voigt_strain(T: np.ndarray) -> np.ndarray:
    return np.array([T[0, 0], T[1, 1], T[2, 2], 2.0 * T[0, 1], 2.0 * T[1, 2], 2.0 * T[2, 0]])


def test_law10_spatial_rotation_objectivity():
    """Test 26: Spatial rotation objectivity sigma'(Q eps Q^T) = Q sigma(eps) Q^T."""
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 1.0e7,
        "MAT_NU": 0.25,
        "A0": 1.0e6,
        "A1": 2.0,
        "c0": 0.0,
        "c1": 1.0e7,
    })
    # Rotation of 90 degrees around Z axis
    theta = math.pi / 2.0
    Q = np.array([
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta),  math.cos(theta), 0.0],
        [0.0,             0.0,             1.0],
    ])

    sig_tensor = np.array([
        [1000.0,  500.0,    0.0],
        [ 500.0, -2000.0,   0.0],
        [   0.0,    0.0,  -500.0],
    ])
    deps_tensor = np.array([
        [-0.002,  0.001,  0.0],
        [ 0.001, -0.003,  0.0],
        [   0.0,    0.0, -0.001],
    ])

    sig_voigt = _tensor_to_voigt_stress(sig_tensor)[None, :]
    deps_voigt = _tensor_to_voigt_strain(deps_tensor)[None, :]

    sig_rot_tensor = Q @ sig_tensor @ Q.T
    deps_rot_tensor = Q @ deps_tensor @ Q.T

    sig_rot_voigt = _tensor_to_voigt_stress(sig_rot_tensor)[None, :]
    deps_rot_voigt = _tensor_to_voigt_strain(deps_rot_tensor)[None, :]

    sig_new = solid_update(mat, sig_voigt, d_eps=deps_voigt, dt=1e-4)
    sig_new_rot = solid_update(mat, sig_rot_voigt, d_eps=deps_rot_voigt, dt=1e-4)

    expected_rotated = Q @ _voigt_to_tensor_stress(sig_new[0]) @ Q.T
    actual_rotated = _voigt_to_tensor_stress(sig_new_rot[0])

    np.testing.assert_allclose(actual_rotated, expected_rotated, rtol=1e-5, atol=1e-4)


def test_law10_shell_update_raises_not_implemented():
    """Test 27: Calling shell_update raises NotImplementedError (solids only)."""
    mat = build_law10({"MAT_RHO": 2000.0, "MAT_E": 1.0e7, "MAT_NU": 0.25})
    sig = np.zeros((1, 3), dtype=float)
    with pytest.raises(NotImplementedError, match="3D solid elements only"):
        shell_update(mat, sig)


def test_law10_batched_vectorization_equivalence():
    """Test 28: Batched vectorized solid_update produces identical output to element-by-element iteration."""
    mat = build_law10({
        "MAT_RHO": 2000.0,
        "MAT_E": 2.0e7,
        "MAT_NU": 0.25,
        "A0": 1.0e6,
        "A1": 1.5,
        "c0": 1.0e4,
        "c1": 2.0e6,
        "c2": 5.0e6,
        "bunl": 4.0e6,
    })
    n = 6
    np.random.seed(42)
    sig_batch = np.random.randn(n, 6) * 1.0e4
    deps_batch = np.random.randn(n, 6) * 1.0e-3
    mu_bak_batch = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    extra_batch = {"mu_bak": mu_bak_batch.copy()}

    sig_out_batch = solid_update(mat, sig_batch, d_eps=deps_batch, dt=1e-4, extra=extra_batch)

    # Element-by-element evaluation
    sig_out_single = np.empty_like(sig_batch)
    for i in range(n):
        extra_i = {"mu_bak": np.array([mu_bak_batch[i]])}
        out_i = solid_update(mat, sig_batch[i:i+1], d_eps=deps_batch[i:i+1], dt=1e-4, extra=extra_i)
        sig_out_single[i] = out_i[0]

    np.testing.assert_allclose(sig_out_batch, sig_out_single, rtol=1e-12, atol=1e-12)


def test_law10_empty_array_safety():
    """Test 29: Calling solid_update and consistent_solid_tangent with empty (0, 6) arrays succeeds cleanly."""
    mat = build_law10({"MAT_RHO": 1800.0, "MAT_E": 1.0e7, "MAT_NU": 0.3})
    sig_empty = np.empty((0, 6), dtype=float)
    deps_empty = np.empty((0, 6), dtype=float)

    out = solid_update(mat, sig_empty, d_eps=deps_empty, dt=1e-3)
    assert out.shape == (0, 6)

    d_tang = consistent_solid_tangent(mat, sig_empty)
    assert d_tang.shape == (0, 6, 6)


def test_law10_zero_dt_no_op():
    """Test 30: Calling solid_update with dt=0.0 returns unmodified stress tensor."""
    mat = build_law10({"MAT_RHO": 1800.0, "MAT_E": 1.0e7, "MAT_NU": 0.3})
    sig = np.array([[100.0, 50.0, 25.0, 10.0, 5.0, 2.0]], dtype=float)
    out = solid_update(mat, sig, dt=0.0)
    np.testing.assert_allclose(out, sig)


# =============================================================================
# 6. Deck Writer & Starter Integration (Tests 31-32)
# =============================================================================

def test_law10_deck_writer_roundtrip():
    """Test 31: Deck writer StarterDeck.mat_law10 renders 7 cards and starter parses into active Material."""
    deck = StarterDeck("M536_TEST")
    deck.mat_law10(
        mat_id=1,
        title="sand_layer",
        rho0=1900.0,
        rhor=1900.0,
        e=4.0e7,
        nu=0.3,
        a0=2.0e6,
        a1=0.8,
        a2=0.01,
        amax=5.0e6,
        c0=100.0,
        c1=3.0e7,
        c2=200.0,
        c3=50.0,
        pmin=-5.0e5,
        pext=101325.0,
        b=3.5e7,
        mue_max=0.25,
    )
    rendered = deck.render()
    with tempfile.NamedTemporaryFile(delete=False, suffix="_0000.rad", mode="w") as f:
        f.write(rendered)
        path = f.name

    log = MessageLog()
    blocks = read_deck(path)
    model = Model()
    parse_starter_deck(blocks, model, log)

    assert len(log.errors) == 0
    assert 1 in model.materials
    mat = model.materials[1]
    assert mat.law == 10
    assert mat.rho0 == 1900.0
    assert mat.params["E"] == 4.0e7
    assert mat.params["nu"] == 0.3
    assert mat.params["A0"] == 2.0e6
    assert mat.params["A1"] == 0.8
    assert mat.params["A2"] == 0.01
    assert mat.params["Amax"] == 5.0e6
    assert mat.params["c0"] == 100.0
    assert mat.params["c1"] == 3.0e7
    assert mat.params["c2"] == 200.0
    assert mat.params["c3"] == 50.0
    assert mat.params["pmin"] == -5.0e5
    assert mat.params["pext"] == 101325.0
    assert mat.params["bunl"] == 3.5e7
    assert mat.params["mue_max"] == 0.25


def test_law10_starter_checks_integration():
    """Test 32: check_model accepts LAW10 on solid bricks and tetras, rejects LAW10 on shells."""
    mat = build_law10({"MAT_RHO": 2000.0, "MAT_E": 1.0e7, "MAT_NU": 0.25})
    mat.id = 1

    class DummyGroup:
        def __init__(self, slices):
            self.n = 1
            self.state = {"slices": slices}

    # 1. Solid element group passes check_model
    solid_model = Model()
    solid_model.node_ids = np.arange(8)
    solid_model.bricks = DummyGroup([(slice(0, 1), mat, None)])
    log_solid = MessageLog()
    check_model(solid_model, log_solid)
    mat_errors = [e for e in log_solid.errors if "not ported" in e or "does not support" in e]
    assert len(mat_errors) == 0

    # 2. Shell element group fails check_model (LAW10 not in _ALLOWED_LAWS["shells"])
    shell_model = Model()
    shell_model.node_ids = np.arange(4)
    shell_model.shells = DummyGroup([(slice(0, 1), mat, None)])
    log_shell = MessageLog()
    check_model(shell_model, log_shell)
    shell_errors = [e for e in log_shell.errors if "not ported" in e or "does not support" in e]
    assert len(shell_errors) > 0


# =============================================================================
# 7. Supplementary Tests
# =============================================================================

def test_law10_auto_estimation():
    """Supplementary: Automatic estimation of missing mue_max or bunl per hm_read_mat10.F."""
    # 1. Missing mue_max estimated from bunl (c3 == 0, c2 != 0)
    p1 = _ensure_params({
        "rho0": 1000.0, "E": 1e6, "nu": 0.0,
        "c1": 1.0e6, "c2": 5.0e5, "c3": 0.0,
        "bunl": 2.0e6, "mue_max": 0.0,
    })
    expected_mumx = (2.0e6 - 1.0e6) / (2.0 * 5.0e5)
    assert math.isclose(p1["mue_max"], expected_mumx)

    # 2. Missing bunl estimated from mue_max (c3 == 0, c2 != 0)
    p2 = _ensure_params({
        "rho0": 1000.0, "E": 1e6, "nu": 0.0,
        "c1": 1.0e6, "c2": 5.0e5, "c3": 0.0,
        "bunl": 0.0, "mue_max": 0.5,
    })
    expected_bunl = 1.0e6 + 2.0 * 5.0e5 * 0.5
    assert math.isclose(p2["bunl"], expected_bunl)


def test_law10_registration():
    """Supplementary: Check that all LAW10 aliases are registered in MAT_PHYSICS_REGISTRY."""
    for key in ("LAW10", "SOIL", "DPRAG", "DPRAG1", "10"):
        assert key in MAT_PHYSICS_REGISTRY
        fn = MAT_PHYSICS_REGISTRY[key]
        assert callable(fn)


def test_law10_element_inactive_off():
    """Supplementary: Element deactivation flag off=0 relaxes deviatoric stress to zero."""
    mat = build_law10({"MAT_RHO": 2000.0, "MAT_E": 1.0e7, "MAT_NU": 0.25, "A0": 1.0e6})
    sig = np.zeros((1, 6), dtype=float)
    deps = np.array([[0.0, 0.0, 0.0, 1.0e-3, 0.0, 0.0]])
    extra = {"off": np.array([0.0])}
    sig_out = solid_update(mat, sig, d_eps=deps, dt=1e-4, extra=extra)
    np.testing.assert_allclose(sig_out[0, 3], 0.0)


def test_law10_calling_conventions():
    """Supplementary: Calling conventions with eps_dot, d_eps, or kernel tuple return."""
    mat = build_law10({"MAT_RHO": 2000.0, "MAT_E": 1.0e7, "MAT_NU": 0.25, "A0": 1.0e6})
    sig = np.zeros((1, 6), dtype=float)
    dt = 1e-4

    eps_dot = np.array([[0.0, 0.0, 0.0, 10.0, 0.0, 0.0]])
    res1 = solid_update(mat, sig, eps_dot, dt)

    deps = eps_dot * dt
    res2 = solid_update(mat, sig, d_eps=deps, dt=dt)
    np.testing.assert_allclose(res1, res2)

    epsp = np.zeros(1)
    extra = {}
    sig_k, epsp_k, c_k = solid_update(mat, sig, deps, epsp, dt, extra, return_tuple=True)
    np.testing.assert_allclose(sig_k, res1)
    assert c_k is not None
