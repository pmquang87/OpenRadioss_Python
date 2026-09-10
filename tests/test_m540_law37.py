"""Tests for Milestone M540: LAW37 / BIPHAS / BIPHASIC constitutive physics.

Upstream Fortran origins:
- Kernel: engine/source/materials/mat/mat037/sigeps37.F
- Reader: starter/source/materials/mat/mat037/hm_read_mat37.F
- CFG Schema: hm_cfg_files/config/CFG/radioss2018/MAT/matl37_biphas.cfg
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import Material
import pyradioss.materials as pm
from pyradioss.materials import law37_biphas as l37
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY


# =============================================================================
# Helper Fixtures
# =============================================================================

def _make_law37(
    rho_l0: float = 1000.0,
    c_l: float = 2.2e9,
    alpha1: float = 0.8,
    nu_l: float = 1e-6,
    nu_vol_l: float = 0.0,
    rho_g0: float = 1.2,
    gamma: float = 1.4,
    p0: float = 1.01325e5,
    nu_g: float = 1.5e-5,
    nu_vol_g: float = 0.0,
    rho0: float = 0.0,
    psh: float = 0.0,
    isolver: Optional[int] = None,
    mat_id: int = 1,
    title: str = "BIPHAS_TEST",
    **kwargs,
) -> Material:
    params = {
        "rho_l0": rho_l0,
        "c_l": c_l,
        "alpha1": alpha1,
        "nu_l": nu_l,
        "nu_vol_l": nu_vol_l,
        "rho_g0": rho_g0,
        "gamma": gamma,
        "p0": p0,
        "nu_g": nu_g,
        "nu_vol_g": nu_vol_g,
        "rho0": rho0,
        "psh": psh,
        **kwargs,
    }
    if isolver is not None:
        params["isolver"] = isolver
    rec = {
        "id": mat_id,
        "title": title,
        "params": params,
    }
    return l37.build_law37(rec)


# =============================================================================
# 1. Parameter Extraction & Constructor Tests
# =============================================================================

def test_build_law37_defaults():
    mat = _make_law37()
    assert mat.id == 1
    assert mat.law == 37
    assert mat.law_name == "LAW37"
    assert mat.title == "BIPHAS_TEST"

    # Check default density calculation: rho_l0 * alpha1 + (1 - alpha1) * rho_g0
    expected_rho0 = 1000.0 * 0.8 + 0.2 * 1.2
    assert math.isclose(mat.rho0, expected_rho0, rel_tol=1e-12)
    assert math.isclose(mat.params["rho0"], expected_rho0, rel_tol=1e-12)

    # When psh == 0.0: pshift = -p0
    assert mat.params["psh"] == 0.0
    assert math.isclose(mat.params["pshift"], -1.01325e5, rel_tol=1e-12)

    # Solver default
    assert mat.params["isolver"] == 1


def test_build_law37_cfg_keys():
    rec = {
        "id": 2,
        "title": "CFG_BIPHAS",
        "params": {
            "Lqud_Rho_l": 998.0,
            "C_l": 2.15e9,
            "ALPHA1": 0.5,
            "Nu_l": 1.1e-6,
            "Bulk_Ratio_l": 2.0e-6,
            "Lqud_Rho_g": 1.18,
            "Lqud_Gamma_bulk": 1.4,
            "Lqud_P0": 1.0e5,
            "Nu_g": 1.8e-5,
            "Bulk_Ratio_g": 3.0e-5,
            "MAT_RHO": 500.0,
            "MAT_PSH": 1e-20,
        },
    }
    mat = l37.build_law37(rec)
    assert mat.id == 2
    assert mat.rho0 == 500.0
    assert mat.params["rho_l0"] == 998.0
    assert mat.params["c_l"] == 2.15e9
    assert mat.params["alpha1"] == 0.5
    assert mat.params["nu_l"] == 1.1e-6
    assert mat.params["nu_vol_l"] == 2.0e-6
    assert mat.params["rho_g0"] == 1.18
    assert mat.params["gamma"] == 1.4
    assert mat.params["p0"] == 1.0e5
    assert mat.params["nu_g"] == 1.8e-5
    assert mat.params["nu_vol_g"] == 3.0e-5
    assert math.isclose(mat.params["pshift"], -1e-20, rel_tol=1e-12)


def test_build_law37_clamping_and_validation():
    # Test alpha1 clamping to [0, 1]
    mat_high = _make_law37(alpha1=1.5)
    assert mat_high.params["alpha1"] == 1.0

    mat_low = _make_law37(alpha1=-0.2)
    assert mat_low.params["alpha1"] == 0.0

    # Negative densities clamped to 1e-20
    mat_neg = _make_law37(rho_l0=-10.0, rho_g0=-5.0)
    assert mat_neg.params["rho_l0"] == 1e-20
    assert mat_neg.params["rho_g0"] == 1e-20


def test_build_law37_solver_selection():
    # Explicit isolver 2
    mat_nr = _make_law37(isolver=2)
    assert mat_nr.params["isolver"] == 2

    # INT22 > 0 selects Newton solver
    mat_int22 = _make_law37(INT22=1.0)
    assert mat_int22.params["isolver"] == 2


def test_registry():
    for key in (37, "37", "LAW37", "BIPHAS", "BIPHASIC"):
        assert key in MAT_PHYSICS_REGISTRY
        assert MAT_PHYSICS_REGISTRY[key] is l37.build_law37


def test_extra_shapes_and_needs_env():
    mat = _make_law37()
    shapes = pm.extra_shapes(mat)
    assert "uv37" in shapes
    assert shapes["uv37"] == (5,)

    assert pm.needs_env(mat) is True


def test_shell_update_not_implemented():
    mat = _make_law37()
    sig = np.zeros((1, 3))
    deps = np.zeros((1, 3))
    with pytest.raises(NotImplementedError, match="LAW37 \\(biphasic fluid/gas\\) is implemented for 3D solid and SPH elements only\\."):
        l37.shell_update(mat, sig, deps)


# =============================================================================
# 2. Constitutive Stress Update (ISOLVER = 1 Legacy Solver)
# =============================================================================

def test_solid_update_empty():
    mat = _make_law37()
    sig = np.zeros((0, 6))
    deps = np.zeros((0, 6))
    s_out, epsp_out, c_out = l37.solid_update(mat, sig, deps, dt=0.01)
    assert s_out.shape == (0, 6)
    assert epsp_out is None
    assert c_out.shape == (0,)


def test_solid_update_legacy_initialization_and_equilibrium():
    mat = _make_law37(isolver=1, alpha1=0.8, nu_l=0.0, nu_g=0.0)
    nel = 2
    sig = np.zeros((nel, 6))
    deps = np.zeros((nel, 6))
    extra = {"rho": np.full(nel, mat.rho0)}

    # First cycle: uv37 is uninitialized (all zero)
    s_out, epsp_out, c_out = l37.solid_update(mat, sig, deps, dt=1e-5, extra=extra)

    # Initial state should be at equilibrium (zero deviatoric stress, pressure = 0 when psh = 0)
    # Since pshift = -P0, and initial P = P0, P_rel = P + pshift = 0
    np.testing.assert_allclose(s_out, 0.0, atol=1e-3)

    # Check uv37 initialized values
    uv37 = extra["uv37"]
    assert uv37.shape == (nel, 5)
    # rho1 = rho_l0, rho2 = rho_g0
    np.testing.assert_allclose(uv37[:, 1], mat.params["rho_g0"], rtol=1e-6)
    np.testing.assert_allclose(uv37[:, 2], mat.params["rho_l0"], rtol=1e-6)
    # Sound speed: c = sqrt(C_l / rho1)
    expected_c = math.sqrt(mat.params["c_l"] / mat.params["rho_l0"])
    np.testing.assert_allclose(c_out, expected_c, rtol=1e-6)


def test_solid_update_legacy_compression():
    mat = _make_law37(isolver=1, nu_l=0.0, nu_g=0.0)
    nel = 1
    sig = np.zeros((nel, 6))
    deps = np.zeros((nel, 6))
    extra = {"rho": np.array([mat.rho0])}

    # Cycle 1: initialization at reference density
    l37.solid_update(mat, sig, deps, dt=1e-4, extra=extra)

    # Cycle 2: negative strain increment corresponds to compression (deps_vol < 0)
    deps_comp = np.full((nel, 6), -1e-4)
    # Under compression, current density increases
    compressed_rho = mat.rho0 * 1.01
    extra["rho"] = np.array([compressed_rho])

    s_out, _, c_out = l37.solid_update(mat, sig, deps_comp, dt=1e-4, extra=extra)

    # Under compression, pressure increases (sigma_ii becomes negative)
    assert s_out[0, 0] < 0.0
    assert s_out[0, 1] < 0.0
    assert s_out[0, 2] < 0.0
    # No shear stress when shear strain rate and viscosity are zero
    assert s_out[0, 3] == 0.0
    assert s_out[0, 4] == 0.0
    assert s_out[0, 5] == 0.0


def test_solid_update_viscous_stresses():
    nu_l = 1e-4
    nu_vol_l = 2e-4
    mat = _make_law37(isolver=1, nu_l=nu_l, nu_vol_l=nu_vol_l, nu_g=1e-5, nu_vol_g=2e-5)
    nel = 1
    sig = np.zeros((nel, 6))
    dt = 1e-3
    # Apply pure shear strain increment: deps_xy = 1e-4
    deps = np.array([[0.0, 0.0, 0.0, 1e-4, 0.0, 0.0]])
    extra = {"rho": np.array([mat.rho0])}

    s_out, _, _ = l37.solid_update(mat, sig, deps, dt=dt, extra=extra)

    # Shear stress sigma_xy = mu * (deps_xy / dt)
    uv37 = extra["uv37"]
    b1 = uv37[0, 0]
    b2 = mat.rho0 - b1
    rho1 = uv37[0, 2]
    rho2 = uv37[0, 1]
    expected_mu = (b1 * rho1 * nu_l + b2 * rho2 * 1e-5) / mat.rho0
    expected_sig_xy = expected_mu * (1e-4 / dt)
    assert math.isclose(s_out[0, 3], expected_sig_xy, rel_tol=1e-6)


# =============================================================================
# 3. Constitutive Stress Update (ISOLVER = 2 Newton-Raphson Solver)
# =============================================================================

def test_solid_update_newton_equilibrium():
    mat = _make_law37(isolver=2, alpha1=0.7, nu_l=0.0, nu_g=0.0)
    nel = 2
    sig = np.zeros((nel, 6))
    deps = np.zeros((nel, 6))
    extra = {"rho": np.full(nel, mat.rho0)}

    s_out, _, c_out = l37.solid_update(mat, sig, deps, dt=1e-4, extra=extra)

    # Equilibrium stress is 0
    np.testing.assert_allclose(s_out, 0.0, atol=1e-3)

    # Wood's mixture sound speed
    uv37 = extra["uv37"]
    r1 = mat.params["c_l"] / mat.params["rho_l0"]
    ssp1 = r1 * uv37[:, 2]
    ssp2 = mat.params["gamma"] * mat.params["p0"] * (uv37[:, 1] / mat.params["rho_g0"]) ** mat.params["gamma"]
    ssp = uv37[:, 3] / ssp1 + uv37[:, 4] / ssp2
    expected_c = np.sqrt(1.0 / (ssp * mat.rho0))
    np.testing.assert_allclose(c_out, expected_c, rtol=1e-6)


def test_solid_update_newton_pure_gas():
    # alpha1 = 0 -> pure gas
    mat = _make_law37(isolver=2, alpha1=0.0, rho_g0=1.2, p0=1e5, gamma=1.4)
    nel = 1
    sig = np.zeros((nel, 6))
    deps = np.zeros((nel, 6))
    extra = {"rho": np.array([1.2])}

    s_out, _, c_out = l37.solid_update(mat, sig, deps, dt=1e-4, extra=extra)
    uv37 = extra["uv37"]
    assert uv37[0, 0] == 0.0
    assert uv37[0, 3] == 0.0
    assert uv37[0, 4] == 1.0
    # Pure gas sound speed: sqrt(gamma * P0 / rho_g0)
    expected_c = math.sqrt(1.4 * 1e5 / 1.2)
    assert math.isclose(c_out[0], expected_c, rel_tol=1e-6)


def test_solid_update_newton_pure_liquid():
    # alpha1 = 1.0 -> pure liquid
    mat = _make_law37(isolver=2, alpha1=1.0, rho_l0=1000.0, c_l=2.2e9)
    nel = 1
    sig = np.zeros((nel, 6))
    deps = np.zeros((nel, 6))
    extra = {"rho": np.array([1000.0])}

    s_out, _, c_out = l37.solid_update(mat, sig, deps, dt=1e-4, extra=extra)
    uv37 = extra["uv37"]
    assert uv37[0, 3] == 1.0
    assert uv37[0, 4] == 0.0
    # Pure liquid sound speed: sqrt(C_l / rho_l0)
    expected_c = math.sqrt(2.2e9 / 1000.0)
    assert math.isclose(c_out[0], expected_c, rel_tol=1e-6)


def test_solid_update_boundary_case():
    # When gamma * c_l < 1e-30 (boundary element)
    mat = _make_law37(c_l=0.0, gamma=0.0)
    nel = 1
    sig = np.array([[10.0, 20.0, 30.0, 1.0, 2.0, 3.0]])
    deps = np.zeros((nel, 6))
    extra = {"rho": np.array([1000.0])}

    s_out, _, c_out = l37.solid_update(mat, sig, deps, dt=1e-4, extra=extra)
    assert math.isclose(c_out[0], 1e-30, rel_tol=1e-12)
    # Stresses remain unchanged for boundary element
    np.testing.assert_allclose(s_out, sig)


# =============================================================================
# 4. Sound Speed API Tests
# =============================================================================

def test_sound_speed_scalar_and_array():
    mat_legacy = _make_law37(isolver=1)
    c_scalar = l37.sound_speed(mat_legacy, rho=1000.0)
    assert isinstance(c_scalar, float)
    assert math.isclose(c_scalar, math.sqrt(2.2e9 / 1000.0), rel_tol=1e-6)

    c_array = l37.sound_speed(mat_legacy, rho=np.array([1000.0, 1000.0]))
    assert isinstance(c_array, np.ndarray)
    assert c_array.shape == (2,)

    mat_nr = _make_law37(isolver=2)
    c_nr = l37.sound_speed(mat_nr, rho=mat_nr.rho0)
    assert isinstance(c_nr, float)
    assert c_nr > 0.0


# =============================================================================
# 5. Consistent Solid Tangent Matrix Tests
# =============================================================================

def test_consistent_solid_tangent_dimensions_and_symmetry():
    mat = _make_law37(nu_l=1e-4, nu_vol_l=2e-4)
    nel = 3
    sig = np.zeros((nel, 6))
    extra = {"rho": np.full(nel, mat.rho0), "dt": 1e-4}

    D = l37.consistent_solid_tangent(mat, sig=sig, extra=extra, dt=1e-4)
    assert D.shape == (nel, 6, 6)

    # Check symmetry: D[i, j, k] == D[i, k, j]
    for i in range(nel):
        np.testing.assert_allclose(D[i], D[i].T, atol=1e-10)


def test_consistent_solid_tangent_values():
    nu_l = 1e-3
    nu_vol_l = 2e-3
    mat = _make_law37(nu_l=nu_l, nu_vol_l=nu_vol_l)
    nel = 1
    sig = np.zeros((nel, 6))
    dt = 1e-4
    rho = mat.rho0
    extra = {"rho": np.array([rho]), "dt": dt}

    D = l37.consistent_solid_tangent(mat, sig=sig, extra=extra, dt=dt)
    c = l37.sound_speed(mat, rho=rho)
    kt = rho * (c ** 2)

    alpha1 = mat.params["alpha1"]
    b1 = alpha1 * rho
    b2 = (1.0 - alpha1) * rho
    mu = (b1 * mat.params["rho_l0"] * nu_l + b2 * mat.params["rho_g0"] * mat.params["nu_g"]) / rho
    mu_vol = (b1 * mat.params["rho_l0"] * nu_vol_l + b2 * mat.params["rho_g0"] * mat.params["nu_vol_g"]) / rho
    gt = mu / dt
    k_vol = mu_vol / dt
    bulk = kt + k_vol

    # Normal components:
    expected_D00 = bulk + 2.0 * gt
    expected_D01 = bulk
    assert math.isclose(D[0, 0, 0], expected_D00, rel_tol=1e-6)
    assert math.isclose(D[0, 0, 1], expected_D01, rel_tol=1e-6)

    # Shear components:
    assert math.isclose(D[0, 3, 3], gt, rel_tol=1e-6)
    assert math.isclose(D[0, 4, 4], gt, rel_tol=1e-6)
    assert math.isclose(D[0, 5, 5], gt, rel_tol=1e-6)


def test_consistent_solid_tangent_empty():
    mat = _make_law37()
    sig = np.zeros((0, 6))
    D = l37.consistent_solid_tangent(mat, sig=sig)
    assert D.shape == (0, 6, 6)


# =============================================================================
# 6. Global Materials Package Dispatch Tests
# =============================================================================

def test_pm_dispatch():
    mat = _make_law37()
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))

    # Dispatch solid_update
    s_out, epsp_out, c_out = pm.solid_update(mat, sig, deps, epsp=None, dt=1e-4)
    assert s_out.shape == (1, 6)

    # Dispatch sound_speed
    c = pm.sound_speed(mat)
    assert isinstance(c, float)
    assert c > 0.0

    # Dispatch shell_update raises
    with pytest.raises(NotImplementedError):
        pm.shell_update(mat, sig[:, :3], deps[:, :3], epsp=None, dt=1e-4)

    # Dispatch solid_tangent
    D = pm.solid_tangent(mat, sig, epsp=None, epsp_incr=None, extra={"dt": 1e-4})
    assert D.shape == (1, 6, 6)


# =============================================================================
# 7. Starter Checks & Deck Reader Tests
# =============================================================================

def test_check_mat_law37_valid():
    from pyradioss.common.messages import MessageLog
    from pyradioss.starter.checks import check_mat_law37

    mat = _make_law37()
    log = MessageLog()
    check_mat_law37(mat, log)
    assert len(log.errors) == 0


def test_check_mat_law37_errors():
    from pyradioss.common.messages import MessageLog
    from pyradioss.starter.checks import check_mat_law37

    # Bad liquid density
    mat = _make_law37()
    mat.params["rho_l0"] = 0.0
    log = MessageLog()
    check_mat_law37(mat, log)
    assert len(log.errors) > 0
    assert any("liquid reference density" in m for m in log.errors)

    # Bad liquid bulk modulus
    mat = _make_law37()
    mat.params["c_l"] = 0.0
    log = MessageLog()
    check_mat_law37(mat, log)
    assert len(log.errors) > 0
    assert any("liquid bulk modulus" in m for m in log.errors)


def test_starter_deck_biphas_roundtrip(tmp_path):
    from pyradioss.input.deck_writer import StarterDeck
    from pyradioss.input.starter_keywords import parse_starter_deck

    d = StarterDeck("LAW37_TEST")
    d.mat_biphas(
        id=1,
        rho_l0=1000.0,
        c_l=2.2e9,
        alpha1=0.75,
        nu_l=1.0e-6,
        nu_vol_l=2.0e-6,
        rho_g0=1.2,
        gamma_g=1.4,
        p0_g=1.0e5,
        nu_g=1.5e-5,
        nu_vol_g=2.5e-5,
        rho=750.3,
        pshift=1e-20,
        title="WaterAir",
    )
    deck_path = tmp_path / "test_0000.rad"
    d.write(str(deck_path))
    assert deck_path.exists()

    model = parse_starter_deck(str(deck_path))
    assert 1 in model.materials
    mat = model.materials[1]
    assert getattr(mat, "inactive", False) is False
    assert mat.law == 37
    assert math.isclose(mat.params["rho_l0"], 1000.0, rel_tol=1e-6)
    assert math.isclose(mat.params["c_l"], 2.2e9, rel_tol=1e-6)
    assert math.isclose(mat.params["alpha1"], 0.75, rel_tol=1e-6)


# =============================================================================
# 8. Advanced Physics & Numerical Verification Tests
# =============================================================================

def test_consistent_solid_tangent_viscous_perturbation():
    """Verify tangent matrix matches numerical directional derivative of viscous stress."""
    nu_l = 2e-3
    nu_vol_l = 4e-3
    mat = _make_law37(nu_l=nu_l, nu_vol_l=nu_vol_l, nu_g=1e-4, nu_vol_g=2e-4)
    nel = 1
    dt = 1e-4
    rho = mat.rho0

    extra_base = {"rho": np.array([rho]), "dt": dt}
    D = l37.consistent_solid_tangent(mat, sig=np.zeros((1, 6)), extra=extra_base, dt=dt)

    eps = 1e-7
    # For each strain component, test perturbation
    for comp in range(6):
        deps_0 = np.zeros((1, 6))
        deps_p = np.zeros((1, 6))
        deps_p[0, comp] = eps

        # Initialize uv37 for both runs with identical base state
        extra_0 = {"rho": np.array([rho]), "dt": dt}
        extra_p = {"rho": np.array([rho]), "dt": dt}
        l37.solid_update(mat, np.zeros((1, 6)), deps_0, dt=dt, extra=extra_0)
        l37.solid_update(mat, np.zeros((1, 6)), deps_0, dt=dt, extra=extra_p)

        sig_0, _, _ = l37.solid_update(mat, np.zeros((1, 6)), deps_0, dt=dt, extra=extra_0)
        sig_p, _, _ = l37.solid_update(mat, np.zeros((1, 6)), deps_p, dt=dt, extra=extra_p)

        # Derivative of viscous stress d(sig_v)/d(deps)
        # Viscous part of tangent matrix:
        D_num = (sig_p[0] - sig_0[0]) / eps

        # Viscous part of analytical D:
        uv37 = extra_0["uv37"]
        b1 = uv37[0, 0]
        b2 = rho - b1
        rho1 = uv37[0, 2]
        rho2 = uv37[0, 1]
        mu = (b1 * rho1 * nu_l + b2 * rho2 * mat.params["nu_g"]) / rho
        mu_vol = (b1 * rho1 * nu_vol_l + b2 * rho2 * mat.params["nu_vol_g"]) / rho
        gt = mu / dt
        k_vol = mu_vol / dt

        D_visc = np.zeros((6, 6))
        for i in range(3):
            for j in range(3):
                D_visc[i, j] += k_vol
                if i == j:
                    D_visc[i, j] += 2.0 * gt
        for s in (3, 4, 5):
            D_visc[s, s] += gt

        np.testing.assert_allclose(D_num, D_visc[:, comp], atol=1e-5, rtol=1e-5)


def test_multi_element_vectorization():
    """Verify batch calculation on 50 elements gives identical results to single element calls."""
    mat = _make_law37(isolver=1, nu_l=1e-5, nu_vol_l=2e-5)
    n = 50
    np.random.seed(42)
    sig_batch = np.zeros((n, 6))
    deps_batch = np.random.randn(n, 6) * 1e-4
    rho_batch = mat.rho0 * (1.0 + np.random.randn(n) * 0.02)
    dt = 1e-5

    extra_batch = {"rho": rho_batch.copy()}
    # Cycle 1: initialization
    l37.solid_update(mat, sig_batch, np.zeros((n, 6)), dt=dt, extra=extra_batch)
    # Cycle 2: update
    s_batch, _, c_batch = l37.solid_update(mat, sig_batch, deps_batch, dt=dt, extra=extra_batch)

    # Per-element verification
    for i in range(n):
        sig_single = np.zeros((1, 6))
        extra_single = {"rho": np.array([rho_batch[i]])}
        l37.solid_update(mat, sig_single, np.zeros((1, 6)), dt=dt, extra=extra_single)
        s_single, _, c_single = l37.solid_update(
            mat, sig_single, deps_batch[i:i+1], dt=dt, extra=extra_single
        )
        np.testing.assert_allclose(s_batch[i], s_single[0], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(c_batch[i], c_single[0], rtol=1e-10, atol=1e-10)


def test_newton_convergence_across_mass_fractions():
    """Test Newton solver (isolver=2) across all phases: pure gas, two-phase, pure liquid."""
    for alpha1 in (0.0, 1e-4, 0.1, 0.5, 0.9, 0.9999, 1.0):
        mat = _make_law37(isolver=2, alpha1=alpha1)
        nel = 1
        sig = np.zeros((nel, 6))
        deps = np.zeros((nel, 6))
        extra = {"rho": np.array([mat.rho0])}

        s_out, _, c_out = l37.solid_update(mat, sig, deps, dt=1e-4, extra=extra)
        assert not np.any(np.isnan(s_out))
        assert not np.any(np.isnan(c_out))
        assert c_out[0] > 0.0
