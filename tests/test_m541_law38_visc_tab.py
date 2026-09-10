"""
Tests for OpenRadioss Material Law 38 (/MAT/LAW38, /MAT/VISC_TAB - Tabulated Viscoelastic Foam).

Covers:
  - build_law38 parameter extraction and defaults (hm_read_mat38.F)
  - Elastic compression and tension (linear and stiffening)
  - Multi-curve rate dependency and power-law interpolation (sigeps38.F)
  - Dedicated unloading curve (IUNLOAD != 0)
  - Confined air pressure with porosity, relaxation, and pressure cap (KCOMPAIR = 1, 2)
  - Hysteresis and damage decay (KRECOVER = 0, 1, 2; KDECAY = 0, 1, 2)
  - Instability control (IMSTA >= 1)
  - Tension cutoff element deletion (TENSIONCUT)
  - Vectorization / batched execution consistency (1-element vs N-elements)
  - Longitudinal sound speed computation
  - Consistent algorithmic tangent tensor (n, 6, 6)
  - shell_update raising NotImplementedError
  - Material registry and dispatcher wiring
"""

import math
import numpy as np
import pytest

from pyradioss.model.entities import Material
from pyradioss.materials.law38_visc_tab import (
    build_law38,
    solid_update,
    sound_speed,
    consistent_solid_tangent,
    shell_update,
    extra_shapes,
    _eval_curve,
)
import pyradioss.materials as mats


# ----------------------------------------------------------------------------
# 1. Parameter Extraction & Construction Tests
# ----------------------------------------------------------------------------

def test_build_law38_defaults():
    """Verify default values and clamping matching hm_read_mat38.F."""
    rec = {
        "id": 101,
        "rho0": 1.2e-3,
        "MAT_E": 200.0,
    }
    mat = build_law38(rec)
    assert mat.id == 101
    assert mat.law == 38
    assert mat.rho0 == pytest.approx(1.2e-3)
    p = mat.params
    assert p["e0"] == pytest.approx(200.0)
    assert p["efinal"] == pytest.approx(200.0)  # Efinal clamped to >= E0
    assert p["ratedamp"] == pytest.approx(0.5)
    assert p["hyster"] == pytest.approx(1.0)
    assert p["theta"] == pytest.approx(0.67)
    assert p["pscale"] == pytest.approx(1.0)
    assert p["exponas"] == pytest.approx(1.0)
    assert p["exponbs"] == pytest.approx(1.0)
    assert p["funload"] == pytest.approx(1.0)
    assert p["tolerance"] == pytest.approx(1.0)
    assert p["maxpres"] >= 1.0e20
    assert p["tensioncut"] >= 1.0e20


def test_build_law38_validation_errors():
    """Verify required positive density and modulus."""
    with pytest.raises(ValueError, match="Density rho0 must be > 0"):
        build_law38({"id": 1, "rho0": 0.0, "MAT_E": 100.0})

    with pytest.raises(ValueError, match="Initial Young modulus E0 must be > 0"):
        build_law38({"id": 1, "rho0": 1.0, "MAT_E": -50.0})


def test_build_law38_poisson_clamping():
    """Verify Poisson ratios are clamped below 0.499."""
    rec = {
        "id": 2,
        "rho0": 1.0,
        "MAT_E": 100.0,
        "MAT_NU": 0.6,
        "MAT_NUt": 0.55,
    }
    mat = build_law38(rec)
    assert mat.params["nu_t"] == pytest.approx(0.499)
    assert mat.params["nu_c"] == pytest.approx(0.499)


# ----------------------------------------------------------------------------
# 2. Curve Evaluation Helper Tests
# ----------------------------------------------------------------------------

def test_eval_curve_interpolation_and_extrapolation():
    """Test _eval_curve with tuple (xs, ys) and piecewise slopes."""
    xs = np.array([0.0, 0.1, 0.2, 0.5])
    ys = np.array([0.0, 10.0, 15.0, 30.0])
    curve = (xs, ys)

    # Point at node
    val, slope = _eval_curve(curve, 0.1)
    assert val == pytest.approx(10.0)

    # Point inside first interval
    val, slope = _eval_curve(curve, 0.05)
    assert val == pytest.approx(5.0)
    assert slope == pytest.approx(100.0)  # (10 - 0) / 0.1

    # Point inside second interval
    val, slope = _eval_curve(curve, 0.15)
    assert val == pytest.approx(12.5)
    assert slope == pytest.approx(50.0)  # (15 - 10) / 0.1

    # Below lower bound (extrapolated with first slope)
    val, slope = _eval_curve(curve, -0.05)
    assert val == pytest.approx(-5.0)
    assert slope == pytest.approx(100.0)

    # Above upper bound (extrapolated with last slope)
    val, slope = _eval_curve(curve, 0.6)
    assert val == pytest.approx(30.0 + 0.1 * 50.0)
    assert slope == pytest.approx(50.0)


# ----------------------------------------------------------------------------
# 3. Elastic Compression and Tension Tests
# ----------------------------------------------------------------------------

def test_elastic_compression_uniaxial():
    """Verify simple uniaxial compression along X axis (TOTAL formulation)."""
    # Linear elastic curve: y = 200 * x
    xs = np.array([0.0, 0.1, 0.5])
    ys = np.array([0.0, 20.0, 100.0])  # slope 200
    mat = build_law38({
        "id": 10,
        "rho0": 1.0,
        "MAT_E": 200.0,
        "MAT_NU": 0.0,
        "MAT_NUt": 0.0,
        "load_curves": [(xs, ys)],
        "fscale_tab": [1.0],
        "eps_tab": [0.0],
    })

    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 0] = -0.05  # -5% compression in X
    dt = 1.0e-3
    extra = {}

    sig_new, epsp_out, c = solid_update(mat, sig, deps, dt=dt, extra=extra)

    # In compression, engineering strain is negative, so Cauchy stress is negative
    # lambda_x = exp(-0.05), Cauchy stress conversion: psc_x / (lambda_y * lambda_z) = psc_x / 1.0 = psc_x
    # nominal strain ean_x = exp(-0.05) - 1 ~= -0.04877
    # strain = -ean_x ~= +0.04877
    # PSN = -200 * strain = -9.754
    assert sig_new[0, 0] < 0.0  # Compressive
    assert abs(sig_new[0, 0] - (-200.0 * (1.0 - math.exp(-0.05)))) < 1e-3
    assert sig_new[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert sig_new[0, 2] == pytest.approx(0.0, abs=1e-6)
    assert c[0] > 0.0


def test_tension_stiffening():
    """Verify tension regime: linear in tension with stiffening towards Efinal."""
    e0 = 100.0
    efinal = 500.0
    lamda = 2.0
    # Upstream hm_read_mat38.F:253 forces EPSFIN=1.0 for any real value
    epsfin = 1.0
    mat = build_law38({
        "id": 11,
        "rho0": 1.0,
        "MAT_E": e0,
        "MAT_Efinal": efinal,
        "MAT_Lamda": lamda,
        "MAT_Epsfinal": epsfin,
        "MAT_NU": 0.0,
        "MAT_NUt": 0.0,
        "load_curves": [([0.0, 1.0], [0.0, 100.0])],
    })

    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 0] = 0.05  # Tension along X
    dt = 1.0e-3
    extra = {}

    sig_new, _, _ = solid_update(mat, sig, deps, dt=dt, extra=extra)

    # Volumer J = exp(0.05) * exp(0) * exp(0) = exp(0.05)
    J = math.exp(0.05)
    tmp1 = math.exp(-lamda * (J - 1.0 + epsfin))
    ei_expected = efinal + (e0 - efinal) * (1.0 - tmp1)
    ean_x = J - 1.0
    psn_x = ei_expected * ean_x
    # Cauchy stress: psn_x / (lambda_y * lambda_z) = psn_x / 1.0
    assert sig_new[0, 0] > 0.0  # Tensile stress
    assert sig_new[0, 0] == pytest.approx(psn_x, rel=1e-4)


# ----------------------------------------------------------------------------
# 4. Multi-Curve Rate Dependency Tests
# ----------------------------------------------------------------------------

def test_multi_curve_strain_rate_interpolation():
    """Verify rate interpolation between static and dynamic curves with power law."""
    # Curve 1 (rate 0): y = 100 * eps
    # Curve 2 (rate 100): y = 200 * eps
    c1 = (np.array([0.0, 0.2]), np.array([0.0, 20.0]))
    c2 = (np.array([0.0, 0.2]), np.array([0.0, 40.0]))

    mat = build_law38({
        "id": 12,
        "rho0": 1.0,
        "MAT_E": 100.0,
        "MAT_NU": 0.0,
        "MAT_NUt": 0.0,
        "load_curves": [c1, c2],
        "fscale_tab": [1.0, 1.0],
        "eps_tab": [0.0, 100.0],
        "MAT_EXP1": 1.0,  # exponas
        "MAT_EXP2": 1.0,  # exponbs
        "DAMP1": 1.0,     # no rate damping
    })

    # Apply strain increment corresponding to rate 50.0
    # rate = deps / dt = 0.05 / 0.001 = 50.0
    # Ratio = (50 - 0) / (100 - 0) = 0.5
    dt = 0.001
    deps = np.zeros((1, 6))
    deps[0, 0] = -0.05

    sig_new, _, _ = solid_update(mat, np.zeros((1, 6)), deps, dt=dt)

    # In sigeps38.F:
    # EL = exp(-0.05)
    # EAN = EL - 1
    # STRAIN = -EAN = 1 - exp(-0.05)
    # Nominal strain rate: EBN = EBR * EL = (deps/dt) * EL = 50 * exp(-0.05)
    # Ratio = EBN / 100.0
    strain = 1.0 - math.exp(-0.05)
    rate = 50.0 * math.exp(-0.05)
    ratio = rate / 100.0
    psn1 = -100.0 * strain
    psn2 = -200.0 * strain
    expected_psn = psn2 + (psn1 - psn2) * (1.0 - ratio)

    assert sig_new[0, 0] == pytest.approx(expected_psn, rel=1e-3)


def test_viscosity_limiting():
    """Verify that dynamic viscosity limits the rate-dependent stress jump."""
    c1 = (np.array([0.0, 0.2]), np.array([0.0, 20.0]))
    c2 = (np.array([0.0, 0.2]), np.array([0.0, 200.0]))

    # Set VISCOSITY to a small cap
    max_visc = 0.05
    mat = build_law38({
        "id": 13,
        "rho0": 1.0,
        "MAT_E": 100.0,
        "MAT_NU": 0.0,
        "MAT_NUt": 0.0,
        "load_curves": [c1, c2],
        "fscale_tab": [1.0, 1.0],
        "eps_tab": [0.0, 100.0],
        "MAT_MaxVisc": max_visc,
        "DAMP1": 1.0,
    })

    dt = 0.001
    deps = np.zeros((1, 6))
    deps[0, 0] = -0.05

    sig_new, _, _ = solid_update(mat, np.zeros((1, 6)), deps, dt=dt)

    strain = 1.0 - math.exp(-0.05)
    rate = 50.0 * math.exp(-0.05)
    psn1 = -100.0 * strain
    # PSN = PSN1 - VISC * (rate - rate_s)
    expected_psn = psn1 - max_visc * rate
    assert sig_new[0, 0] == pytest.approx(expected_psn, rel=1e-3)


# ----------------------------------------------------------------------------
# 5. Closed-Cell Air Pressure Tests
# ----------------------------------------------------------------------------

def test_closed_cell_air_pressure_analytical():
    """Verify analytical air pressure computation P_air = P0 * (J - 1)/(J - phi) * exp(-relaxp * t)."""
    p0 = 0.1
    phi = 0.3
    relaxp = 2.0
    mat = build_law38({
        "id": 14,
        "rho0": 1.0,
        "MAT_E": 100.0,
        "MAT_NU": 0.0,
        "MAT_NUt": 0.0,
        "MAT_Kair": 1,
        "MAT_P0": p0,
        "MAT_POROS": phi,
        "MAT_PR": relaxp,
        "MAT_PMAX": 10.0,
        "load_curves": [([0.0, 1.0], [0.0, 100.0])],
    })

    # Hydrostatic compression: all 3 directions compress
    deps = np.full((1, 6), 0.0)
    deps[0, 0] = -0.05
    deps[0, 1] = -0.05
    deps[0, 2] = -0.05

    t = 0.1
    extra = {"time": t}

    sig_new, _, _ = solid_update(mat, np.zeros((1, 6)), deps, dt=0.01, extra=extra)

    J = math.exp(-0.05) ** 3  # ~= 0.8607
    p_air_expected = (p0 * (J - 1.0) / (J - phi)) * math.exp(-relaxp * t)

    # In hydrostatic compression, air pressure is added to each normal stress
    # Without air pressure, normal stresses are psc_x
    # Check that air pressure contribution is present and negative (compressive)
    assert p_air_expected < 0.0
    # Compare with a run without air pressure
    mat_no_air = build_law38({
        "id": 14,
        "rho0": 1.0,
        "MAT_E": 100.0,
        "MAT_NU": 0.0,
        "MAT_NUt": 0.0,
        "MAT_Kair": 0,
        "load_curves": [([0.0, 1.0], [0.0, 100.0])],
    })
    sig_no_air, _, _ = solid_update(mat_no_air, np.zeros((1, 6)), deps, dt=0.01)

    diff = sig_new[0, 0] - sig_no_air[0, 0]
    assert diff == pytest.approx(p_air_expected, rel=1e-3)


def test_air_pressure_cap():
    """Verify that air pressure is capped at -MAXPRES under extreme compression."""
    maxpres = 0.5
    mat = build_law38({
        "id": 15,
        "rho0": 1.0,
        "MAT_E": 100.0,
        "MAT_NU": 0.0,
        "MAT_NUt": 0.0,
        "MAT_Kair": 1,
        "MAT_P0": 10.0,   # Large P0 to trigger cap
        "MAT_POROS": 0.8,
        "MAT_PMAX": maxpres,
        "load_curves": [([0.0, 1.0], [0.0, 100.0])],
    })

    deps = np.full((1, 6), 0.0)
    deps[0, 0] = -0.1
    deps[0, 1] = -0.1
    deps[0, 2] = -0.1

    extra = {}
    solid_update(mat, np.zeros((1, 6)), deps, dt=0.01, extra=extra)

    uvar = extra["uv38"]
    # UVAR(16) stores air pressure
    p_air = uvar[0, 15]
    assert p_air >= -maxpres  # capped at -maxpres


# ----------------------------------------------------------------------------
# 6. Hysteresis Loops and Unloading Recovery Tests
# ----------------------------------------------------------------------------

def test_hysteresis_krecover_0_and_1():
    """Verify hysteresis decay factor under loading and unloading for KRECOVER=0 and 1."""
    # Step 1: Loading in compression
    mat = build_law38({
        "id": 16,
        "rho0": 1.0,
        "MAT_E": 100.0,
        "MAT_NU": 0.0,
        "MAT_NUt": 0.0,
        "MAT_RELX": 5.0,  # beta
        "MAT_HYST": 0.8,  # hyster
        "Gflag": 0,       # krecover = 0 (no recovery, accumulate strain)
        "Vflag": 0,       # kdecay = 0 (decay on loading and unloading)
        "load_curves": [([0.0, 1.0], [0.0, 100.0])],
    })

    extra = {}
    sig1, _, _ = solid_update(mat, np.zeros((1, 6)), np.array([[-0.1, 0, 0, 0, 0, 0]]), dt=0.01, extra=extra)

    # Verify strain was accumulated in UVAR(26)
    uvar = extra["uv38"]
    acc_strain = uvar[0, 25]
    assert acc_strain > 0.0

    decay = uvar[0, 28]
    expected_decay = min(1.0, 0.8 * (1.0 - math.exp(-5.0 * acc_strain)))
    assert decay == pytest.approx(expected_decay, rel=1e-3)

    # Step 2: Unload (deps > 0 while in compression)
    sig2, _, _ = solid_update(mat, sig1, np.array([[+0.04, 0, 0, 0, 0, 0]]), dt=0.01, extra=extra)
    # Under KRECOVER=0, acc_strain does not decrement on unloading
    assert extra["uv38"][0, 25] == pytest.approx(acc_strain)

    # Now compare with KRECOVER=1 (recovery on unloading)
    extra_rec1 = {}
    solid_update(mat, np.zeros((1, 6)), np.array([[-0.1, 0, 0, 0, 0, 0]]), dt=0.01, extra=extra_rec1)
    mat.params["krecover"] = 1
    solid_update(mat, sig1, np.array([[+0.04, 0, 0, 0, 0, 0]]), dt=0.01, extra=extra_rec1)
    # UVAR(26) should decrease by ECN
    assert extra_rec1["uv38"][0, 25] < acc_strain


def test_hysteresis_krecover_2_energy_based():
    """Verify energy-based recovery KRECOVER=2 using internal energy Eint."""
    mat = build_law38({
        "id": 17,
        "rho0": 1.0,
        "MAT_E": 100.0,
        "MAT_NU": 0.0,
        "MAT_NUt": 0.0,
        "MAT_RELX": 2.0,  # beta
        "MAT_HYST": 0.3,  # hyster
        "Gflag": 2,       # krecover = 2
        "load_curves": [([0.0, 1.0], [0.0, 100.0])],
    })

    # Cycle 1: peak energy 10.0
    extra = {"time": 0.01, "eint": np.array([10.0])}
    sig1, _, _ = solid_update(mat, np.zeros((1, 6)), np.array([[-0.1, 0, 0, 0, 0, 0]]), dt=0.01, extra=extra)
    # At peak energy, eint == max_eint -> PUI = 1 -> EFAC = 0
    assert extra["uv38"][0, 28] == pytest.approx(0.0)

    # Cycle 2: lower energy 5.0 (unloading)
    extra["eint"] = np.array([5.0])
    extra["time"] = 0.02
    sig2, _, _ = solid_update(mat, sig1, np.array([[+0.04, 0, 0, 0, 0, 0]]), dt=0.01, extra=extra)
    # PUI = (5 / 10)^2 = 0.25 -> EFAC = (1 - 0.3) * (1 - 0.25) = 0.7 * 0.75 = 0.525
    assert extra["uv38"][0, 28] == pytest.approx(0.525, rel=1e-3)


# ----------------------------------------------------------------------------
# 7. Tension Cutoff Element Deletion Tests
# ----------------------------------------------------------------------------

def test_tension_cutoff_element_deletion():
    """Verify that exceeding TENSIONCUT sets stress to zero and off=0."""
    tensioncut = 50.0
    mat = build_law38({
        "id": 18,
        "rho0": 1.0,
        "MAT_E": 100.0,
        "MAT_NU": 0.0,
        "MAT_NUt": 0.0,
        "MAT_CUTOFF": tensioncut,
        "load_curves": [([0.0, 1.0], [0.0, 100.0])],
    })

    # Small tension below cutoff: 100 * 0.2 = 20 < 50
    extra = {"off": np.array([1.0])}
    sig1, _, _ = solid_update(mat, np.zeros((1, 6)), np.array([[0.2, 0, 0, 0, 0, 0]]), dt=0.01, extra=extra)
    assert extra["off"][0] == 1.0
    assert sig1[0, 0] > 0.0

    # Large tension above cutoff: 100 * 0.8 = 80 > 50
    sig2, _, _ = solid_update(mat, sig1, np.array([[0.8, 0, 0, 0, 0, 0]]), dt=0.01, extra=extra)
    assert extra["off"][0] == 0.0
    assert np.all(sig2[0] == 0.0)

    # Once deleted, element remains inactive
    sig3, _, _ = solid_update(mat, sig2, np.array([[-0.1, 0, 0, 0, 0, 0]]), dt=0.01, extra=extra)
    assert np.all(sig3[0] == 0.0)


# ----------------------------------------------------------------------------
# 8. Vectorization / Batched Execution Tests
# ----------------------------------------------------------------------------

def test_vectorization_consistency():
    """Verify that running N elements together matches running them individually."""
    n_elems = 10
    mat = build_law38({
        "id": 19,
        "rho0": 1.0,
        "MAT_E": 150.0,
        "MAT_NU": 0.2,
        "MAT_NUt": 0.1,
        "MAT_RV": 1.5,
        "load_curves": [([0.0, 0.5], [0.0, 75.0])],
    })

    rng = np.random.default_rng(42)
    deps_batch = rng.uniform(-0.05, 0.05, (n_elems, 6))
    sig_batch = rng.uniform(-5.0, 5.0, (n_elems, 6))
    dt = 0.001

    extra_batch = {}
    sig_out_batch, _, c_batch = solid_update(mat, sig_batch.copy(), deps_batch.copy(), dt=dt, extra=extra_batch)

    # Run individually
    for i in range(n_elems):
        extra_single = {}
        sig_out_single, _, c_single = solid_update(
            mat, sig_batch[i:i+1].copy(), deps_batch[i:i+1].copy(), dt=dt, extra=extra_single
        )
        np.testing.assert_allclose(sig_out_batch[i], sig_out_single[0], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(c_batch[i], c_single[0], rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------------
# 9. Sound Speed & Tangent Tests
# ----------------------------------------------------------------------------

def test_sound_speed():
    """Verify P-wave longitudinal sound speed formula."""
    e0 = 300.0
    nu = 0.25
    rho0 = 1.5
    mat = build_law38({
        "id": 20,
        "rho0": rho0,
        "MAT_E": e0,
        "MAT_NU": nu,
        "MAT_NUt": nu,
    })

    c = sound_speed(mat)
    k = e0 / (3.0 * (1.0 - 2.0 * nu))
    g = e0 / (2.0 * (1.0 + nu))
    expected_c = math.sqrt((k + 4.0/3.0 * g) / rho0)
    assert c == pytest.approx(expected_c, rel=1e-5)


def test_consistent_solid_tangent_shape_and_symmetry():
    """Verify consistent solid tangent shape (n, 6, 6) and symmetry."""
    mat = build_law38({
        "id": 21,
        "rho0": 1.0,
        "MAT_E": 200.0,
        "MAT_NU": 0.3,
    })

    n = 5
    eps = np.zeros((n, 6))
    deps = np.zeros((n, 6))
    dt = 0.001
    extra = {}

    D = consistent_solid_tangent(mat, eps, deps, dt, extra)
    assert D.shape == (n, 6, 6)

    # Verify symmetry D == D.T for each element
    for i in range(n):
        np.testing.assert_allclose(D[i], D[i].T, atol=1e-10)

    # Verify positive definiteness
    for i in range(n):
        eigvals = np.linalg.eigvalsh(D[i])
        assert np.all(eigvals > 0.0)


# ----------------------------------------------------------------------------
# 10. Shell Update & Integration Tests
# ----------------------------------------------------------------------------

def test_shell_update_not_implemented():
    """Verify shell_update raises NotImplementedError."""
    mat = build_law38({"id": 22, "rho0": 1.0, "MAT_E": 100.0})
    with pytest.raises(NotImplementedError, match="3D solid elements only"):
        shell_update(mat, np.zeros((1, 3)), np.zeros((1, 3)), None, 0.0)


def test_materials_dispatch_integration():
    """Verify dispatch through pyradioss.materials top-level functions."""
    mat = build_law38({"id": 23, "rho0": 1.0, "MAT_E": 100.0})
    assert mat.law == 38

    # Extra shapes
    shapes = mats.extra_shapes(mat)
    assert "uv38" in shapes
    assert shapes["uv38"] == (33,)
    assert "eps38" in shapes

    # Solid update dispatch
    sig = np.zeros((1, 6))
    deps = np.array([[-0.01, 0, 0, 0, 0, 0]])
    extra = {"eps38": np.zeros((1, 6)), "uv38": np.zeros((1, 33))}
    sig_new, epsp, c = mats.solid_update(mat, sig, deps, None, dt=0.001, extra=extra)
    assert sig_new.shape == (1, 6)
    assert c.shape == (1,)

    # Tangent dispatch
    D = mats.solid_tangent(mat, sig, epsp, deps, extra=extra)
    assert D.shape == (1, 6, 6)


# ----------------------------------------------------------------------------
# 11. Dedicated Unloading, Poisson Coupling, Instability & Rotation Tests
# ----------------------------------------------------------------------------

def test_dedicated_unloading_curve():
    """Verify dedicated unloading curve (IUNLOAD != 0)."""
    # Loading curve: y = 200 * strain
    # Unloading curve: y = 80 * strain
    load_c = (np.array([0.0, 0.2]), np.array([0.0, 40.0]))
    unload_c = (np.array([0.0, 0.2]), np.array([0.0, 16.0]))

    mat = build_law38({
        "id": 24,
        "rho0": 1.0,
        "MAT_E": 200.0,
        "MAT_NU": 0.0,
        "MAT_NUt": 0.0,
        "FUN_B4": 1,
        "MAT_ALPHA6": 1.0,
        "MAT_EPSF2": 0.0,  # runload = edots[0]
        "DAMP1": 1.0,      # instantaneous rate response
        "load_curves": [load_c],
        "unload_curve": unload_c,
    })

    # Step 1: Load to -0.1
    extra = {}
    sig1, _, _ = solid_update(mat, np.zeros((1, 6)), np.array([[-0.1, 0, 0, 0, 0, 0]]), dt=0.01, extra=extra)

    # Step 2: Unload to -0.06 (deps = +0.04 while in compression)
    sig2, _, _ = solid_update(mat, sig1, np.array([[+0.04, 0, 0, 0, 0, 0]]), dt=0.01, extra=extra)

    # During unloading with runload == edots[0], stress follows dedicated unloading curve (slope 80)
    # Expected stress is approx -80 * strain
    current_strain = 1.0 - math.exp(-0.06)
    expected_stress = -80.0 * current_strain
    assert sig2[0, 0] == pytest.approx(expected_stress, rel=1e-2)


def test_poisson_ratio_coupling():
    """Verify Poisson ratio update V12, V23, V31 and coupling matrix."""
    mat = build_law38({
        "id": 25,
        "rho0": 1.0,
        "MAT_E": 100.0,
        "MAT_NU": 0.3,
        "MAT_NUt": 0.1,
        "MAT_RV": 2.0,
        "load_curves": [([0.0, 1.0], [0.0, 100.0])],
    })

    # Apply biaxial strain
    deps = np.zeros((1, 6))
    deps[0, 0] = -0.05
    deps[0, 1] = -0.02
    extra = {}

    sig, _, _ = solid_update(mat, np.zeros((1, 6)), deps, dt=0.01, extra=extra)

    # Both X and Y stresses should be non-zero and compressive
    assert sig[0, 0] < 0.0
    assert sig[0, 1] < 0.0
    # Cross-coupling through Poisson matrix induces Z stress even though deps_zz = 0
    assert sig[0, 2] < 0.0


def test_instability_control():
    """Verify material instability control (IMSTA >= 1)."""
    # Softening / plateau curve where secant modulus is low compared to stress differences
    curve = (np.array([0.0, 0.1, 0.5]), np.array([0.0, 20.0, 22.0]))
    mat_no_sta = build_law38({
        "id": 26,
        "rho0": 1.0,
        "MAT_E": 200.0,
        "MAT_Iinsta": 0,
        "load_curves": [curve],
    })
    mat_sta = build_law38({
        "id": 27,
        "rho0": 1.0,
        "MAT_E": 200.0,
        "MAT_Iinsta": 1,
        "load_curves": [curve],
    })

    # Unbalanced stress state: large compressive strain in X (in plateau region), tension in Y
    deps = np.zeros((1, 6))
    deps[0, 0] = -0.4
    deps[0, 1] = 0.2
    deps[0, 2] = 0.0

    sig_no, _, _ = solid_update(mat_no_sta, np.zeros((1, 6)), deps, dt=0.01)
    sig_sta, _, _ = solid_update(mat_sta, np.zeros((1, 6)), deps, dt=0.01)

    # Instability control modifies stresses and moduli
    assert not np.allclose(sig_no, sig_sta)


def test_axis_rotation_history_transformation():
    """Verify CHECKAXES and DREH history variable rotation when axes rotate."""
    mat = build_law38({
        "id": 28,
        "rho0": 1.0,
        "MAT_E": 100.0,
        "MAT_Tol": 0.01,  # Small tolerance to trigger axis rotation
        "load_curves": [([0.0, 1.0], [0.0, 100.0])],
    })

    # Step 1: Strain purely in X (principal axes along coordinate axes)
    extra = {}
    deps1 = np.array([[-0.05, 0, 0, 0, 0, 0]])
    sig1, _, _ = solid_update(mat, np.zeros((1, 6)), deps1, dt=0.01, extra=extra)

    # Initial principal axes: identity
    uvar = extra["uv38"]
    dprao = uvar[0, 16:25].copy()

    # Step 2: Apply strong shear strain to rotate principal axes by 45 degrees
    deps2 = np.array([[0, 0, 0, 0.2, 0, 0]])
    sig2, _, _ = solid_update(mat, sig1, deps2, dt=0.01, extra=extra)

    # Stored principal directions should have updated due to 45-degree rotation
    dpra_new = extra["uv38"][0, 16:25]
    assert not np.allclose(dprao, dpra_new)


def test_incremental_formulation():
    """Verify incremental formulation (ITOTAL = 2)."""
    mat = build_law38({
        "id": 29,
        "rho0": 1.0,
        "MAT_E": 100.0,
        "ITOTAL": 2,  # Incremental formulation
        "load_curves": [([0.0, 1.0], [0.0, 100.0])],
    })

    extra = {}
    sig1, _, _ = solid_update(mat, np.zeros((1, 6)), np.array([[-0.02, 0, 0, 0, 0, 0]]), dt=0.01, extra=extra)
    sig2, _, _ = solid_update(mat, sig1, np.array([[-0.02, 0, 0, 0, 0, 0]]), dt=0.01, extra=extra)

    # Cumulative stress should increase with each step
    assert abs(sig2[0, 0]) > abs(sig1[0, 0])

