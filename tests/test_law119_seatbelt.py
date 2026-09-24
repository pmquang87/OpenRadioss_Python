"""Unit tests for LAW119 seatbelt material model (/MAT/LAW119, /MAT/SH_SEATBELT, /PROP/SEATBELT).

Verifies:
1. Material creation via build_law119 factory and Law119Seatbelt entity.
2. 2D Shell membrane tension vs compression tagging (RCOMP wrinkling model).
3. 2D Shell coating layer stress contribution (ECOAT, NUCOAT, TCOAT).
4. 1D Cable/Belt tension-only mechanics: zero compressive stress/force (slack/folding).
5. 1D Cable/Belt nonlinear loading, unloading, and hysteresis.
6. Sliding friction model (Euler-Eytelwein capstan wrap formula and velocity decay).
7. Folding resistance / ribbon bending moment model.
8. Retractor and pretensioner behavior (sensor lock, pull-in curve, load limiter).
9. Solid update stub and tangent per material template API.
10. Sound speed and material registry integration.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials.law119_seatbelt import (
    Law119Seatbelt,
    RetractorState,
    build_law119,
    cable_update,
    compute_force,
    consistent_shell_tangent,
    extra_shapes,
    folding_moment,
    retractor_update,
    shell_membrane_tangent,
    shell_update,
    sliding_friction,
    slipring_friction,
    solid_update,
    sound_speed,
    spring_update,
    tangent,
)


# ------------------------------------------------------------------------------
# 1. Creation and Parameters
# ------------------------------------------------------------------------------

def test_law119_creation_and_properties():
    """Verify initialization and derived orthotropic constants."""
    params = {
        "STIFF1": 2000.0,
        "E22": 200.0,
        "NU12": 0.2,
        "G12": 150.0,
        "RE": 0.05,
        "LMIN": 5.0,
        "DAMP1": 0.1,
    }
    mat = Law119Seatbelt(id=1, rho0=1.5e-3, title="Belt1", params=params)

    assert mat.id == 1
    assert mat.law == 119
    assert mat.rho0 == 1.5e-3
    assert mat.e11 == 2000.0
    assert mat.e22 == 200.0
    assert mat.nu12 == 0.2
    assert mat.g12 == 150.0
    assert mat.rcomp == 0.05
    assert mat.lmin == 5.0

    # Orthotropic derived constants:
    # fscalet = E22 / E11 = 200 / 2000 = 0.1
    # nu21 = nu12 * fscalet = 0.2 * 0.1 = 0.02
    # det = 1 / (1 - 0.2 * 0.02) = 1 / 0.996
    assert math.isclose(mat.fscalet, 0.1, rel_tol=1e-6)
    assert math.isclose(mat.nu21, 0.02, rel_tol=1e-6)
    assert math.isclose(mat.det, 1.0 / 0.996, rel_tol=1e-6)
    assert math.isclose(mat.a11, 2000.0 / 0.996, rel_tol=1e-6)
    assert math.isclose(mat.a22, mat.a11 * 0.1, rel_tol=1e-6)


def test_build_law119_factory():
    """Verify build_law119 factory handles dictionaries and objects."""
    d = {
        "id": 42,
        "title": "FactoryBelt",
        "rho0": 1.2e-3,
        "params": {
            "STIFF1": 1500.0,
            "NU12": 0.15,
            "RE": 0.02,
        },
    }
    mat = build_law119(d)
    assert isinstance(mat, Law119Seatbelt)
    assert mat.id == 42
    assert mat.title == "FactoryBelt"
    assert mat.e11 == 1500.0
    assert mat.nu12 == 0.15
    assert mat.rcomp == 0.02


# ------------------------------------------------------------------------------
# 2. 2D Shell Mechanics (sigeps119c.F & law119_membrane.F)
# ------------------------------------------------------------------------------

def test_law119_shell_tension_vs_compression():
    """Verify in-plane shell tension branch (beta=1.0) vs compression branch (beta=RCOMP)."""
    rcomp = 0.05
    mat = Law119Seatbelt(id=1, rho0=1.0, params={"STIFF1": 1000.0, "NU12": 0.0, "RE": rcomp})

    # State containers
    extra_t = extra_shapes(mat, nip=1)
    extra_c = extra_shapes(mat, nip=1)

    # 1. Biaxial tension: deps = [+0.01, +0.01, 0.0]
    sig_init = np.zeros((1, 3), dtype=float)
    deps_t = np.array([[0.01, 0.01, 0.0]], dtype=float)
    sig_t, _ = shell_update(mat, sig_init.copy(), deps_t, extra=extra_t)

    # 2. Biaxial compression: deps = [-0.01, -0.01, 0.0]
    deps_c = np.array([[-0.01, -0.01, 0.0]], dtype=float)
    sig_c, _ = shell_update(mat, sig_init.copy(), deps_c, extra=extra_c)

    # In tension: sigma_xx = A11 * eps_xx * 1.0
    # In compression: sigma_xx = A11 * eps_xx * RCOMP
    assert sig_t[0, 0] > 0.0
    assert sig_c[0, 0] < 0.0
    ratio = abs(sig_c[0, 0] / sig_t[0, 0])
    assert math.isclose(ratio, rcomp, rel_tol=1e-5)


def test_law119_shell_coating():
    """Verify coating layer contributes additional stiffness when ECOAT & TCOAT > 0."""
    mat_no_coat = Law119Seatbelt(id=1, params={"STIFF1": 1000.0, "NU12": 0.0, "LMIN": 1.0})
    mat_coat = Law119Seatbelt(id=2, params={
        "STIFF1": 1000.0,
        "NU12": 0.0,
        "LMIN": 1.0,
        "ECOAT": 200.0,
        "TCOAT": 0.1,
        "NUCOAT": 0.0,
    })

    deps = np.array([[0.01, 0.0, 0.0]], dtype=float)
    sig0 = np.zeros((1, 3), dtype=float)

    sig1, _ = shell_update(mat_no_coat, sig0.copy(), deps, extra={})
    sig2, _ = shell_update(mat_coat, sig0.copy(), deps, extra={})

    # Coated element must carry higher stress due to coating layer
    assert sig2[0, 0] > sig1[0, 0]


# ------------------------------------------------------------------------------
# 3. 1D Cable/Belt Mechanics: Tension-Only Slack & Folding
# ------------------------------------------------------------------------------

def test_law119_1d_tension_only_slack():
    """Verify 1D belt is strictly tension-only: zero compression stress/force when compressed."""
    mat = Law119Seatbelt(id=1, params={"STIFF1": 5000.0, "LMIN": 10.0})

    L0 = 10.0
    state = None

    # Step 1: Tension elongation delta = +1.0 (L = 11.0)
    F_tens, k_tan, state = cable_update(mat, L=11.0, L0=L0, v_rel=0.0, state=state)
    assert F_tens > 0.0
    assert k_tan > 0.0
    assert math.isclose(F_tens, 5000.0 * 1.0, rel_tol=1e-5)

    # Step 2: Compression / slack delta = -2.0 (L = 8.0)
    # Seatbelt accommodates compression by folding/slack without compressive resistance
    F_comp, k_tan_comp, state = cable_update(mat, L=8.0, L0=L0, v_rel=0.0, state=state)
    assert F_comp == 0.0
    assert k_tan_comp == 0.0


def test_law119_1d_hysteresis_unloading():
    """Verify loading and hysteresis unloading with permanent plastic elongation."""
    # Custom loading curve: F = 10000 * eps
    # Unloading curve: steeper slope F = 20000 * eps
    loading_curve = lambda eps: 10000.0 * eps
    unloading_curve = lambda eps: 25000.0 * eps

    mat = Law119Seatbelt(id=1, params={
        "STIFF1": 10000.0,
        "LMIN": 10.0,
        "FUN_L": loading_curve,
        "FUN_UL": unloading_curve,
    })

    L0 = 10.0
    state = None

    # 1. Primary tensile loading to eps = 0.10 (L = 11.0)
    F_peak, _, state = cable_update(mat, L=11.0, L0=L0, v_rel=0.0, state=state)
    assert math.isclose(F_peak, 1000.0, rel_tol=1e-3)
    assert state["eps_max"] == 0.10

    # 2. Unload to eps = 0.08 (L = 10.8)
    F_unl, _, state = cable_update(mat, L=10.8, L0=L0, v_rel=0.0, state=state)
    # In unloading, force drops rapidly along unloading curve
    assert F_unl < F_peak
    assert F_unl > 0.0

    # 3. Unload fully until slack
    F_slack, _, state = cable_update(mat, L=10.0, L0=L0, v_rel=0.0, state=state)
    assert F_slack == 0.0


def test_law119_compute_force_wrapper():
    """Verify compute_force convenience function."""
    mat = Law119Seatbelt(id=1, params={"STIFF1": 2000.0})
    F, state = compute_force(delta_L=0.5, L0=1.0, mat=mat)
    assert math.isclose(F, 1000.0, rel_tol=1e-5)
    assert state["force_old"] == F


# ------------------------------------------------------------------------------
# 4. Folding and Sliding Friction Models
# ------------------------------------------------------------------------------

def test_law119_sliding_friction_capstan():
    """Verify Euler-Eytelwein capstan wrap friction relation T2 = T1 * exp(mu * theta)."""
    t1 = 100.0
    theta = math.pi / 2.0  # 90-degree wrap around D-ring
    mu = 0.25

    t2, f_fric = sliding_friction(t1=t1, theta=theta, mu=mu, v_rel=1.0)

    # Theoretical capstan formula:
    # T2 = 100 * exp(0.25 * pi/2) = 100 * exp(0.392699) = 148.0968
    expected_t2 = 100.0 * math.exp(mu * theta)
    assert math.isclose(t2, expected_t2, rel_tol=1e-5)
    assert math.isclose(f_fric, expected_t2 - t1, rel_tol=1e-5)


def test_law119_sliding_friction_static_dynamic_decay():
    """Verify exponential velocity decay between static and dynamic friction."""
    t1 = 100.0
    theta = 1.0
    mu_dyn = 0.2
    mu_stat = 0.4
    decay = 2.0

    # Quasi-static (v_rel near 0): uses static friction
    t2_stat, _ = sliding_friction(t1, theta, mu=mu_dyn, mu_stat=mu_stat, v_rel=0.0, decay=decay)
    expected_stat = t1 * math.exp(mu_stat * theta)
    assert math.isclose(t2_stat, expected_stat, rel_tol=1e-5)

    # High speed: decays to dynamic friction
    t2_dyn, _ = sliding_friction(t1, theta, mu=mu_dyn, mu_stat=mu_stat, v_rel=10.0, decay=decay)
    expected_dyn = t1 * math.exp(mu_dyn * theta)
    assert math.isclose(t2_dyn, expected_dyn, rel_tol=1e-3)


def test_law119_folding_resistance():
    """Verify belt folding moment resistance increases with angle up to M_max."""
    k_fold = 50.0
    m_max = 20.0

    # Small angle: elastic moment M = k_fold * theta
    m1 = folding_moment(theta=0.1, k_fold=k_fold, m_max=m_max)
    assert math.isclose(m1, 5.0, rel_tol=1e-6)

    # Large angle: clamped to plastic folding limit m_max
    m2 = folding_moment(theta=1.0, k_fold=k_fold, m_max=m_max)
    assert math.isclose(m2, 20.0, rel_tol=1e-6)

    # Reverse direction
    m3 = folding_moment(theta=-0.2, k_fold=k_fold, m_max=m_max)
    assert math.isclose(m3, -10.0, rel_tol=1e-6)


# ------------------------------------------------------------------------------
# 5. Retractor and Pretensioner Behavior
# ------------------------------------------------------------------------------

def test_law119_retractor_payout_and_locking():
    """Verify retractor free payout when unlocked, and locking upon sensor trigger."""
    state = RetractorState()

    # Step 1: Unlocked payout: belt extends freely with minimal rewind resistance (<= 5 N)
    f1, state = retractor_update(state, pullout_disp=0.05, pullout_vel=0.5, dt=0.01)
    assert not state.locked
    assert f1 <= 5.0
    assert state.payout == 0.05

    # Step 2: Lock triggered by crash sensor
    f2, state = retractor_update(state, pullout_disp=0.06, pullout_vel=0.5, dt=0.01, sensor_lock=True)
    assert state.locked
    # Once locked, retractor develops high holding force resisting pullout
    assert f2 >= 500.0


def test_law119_retractor_pretensioner_and_load_limiter():
    """Verify pretensioner pull-in and load limiter energy absorption."""
    state = RetractorState()

    # Pretensioner fires with 2000 N pull-in force
    f_pretens, state = retractor_update(
        state,
        pullout_disp=0.0,
        pullout_vel=-1.0,
        dt=0.01,
        sensor_pretens=True,
        pretens_force=2000.0,
    )
    assert state.pretens_active
    assert state.locked
    assert math.isclose(f_pretens, 2000.0, rel_tol=1e-3)

    # Occupant forward motion creates massive belt tension, but load limiter caps force at 4000 N
    f_limit, state = retractor_update(
        state,
        pullout_disp=0.5,
        pullout_vel=1.0,
        dt=0.01,
        load_limit=4000.0,
    )
    assert math.isclose(f_limit, 4000.0, rel_tol=1e-3)


# ------------------------------------------------------------------------------
# 6. Template API and Material Registry Integration
# ------------------------------------------------------------------------------

def test_law119_solid_update_and_tangent_template_stubs():
    """Verify solid_update stub and tangent API conform to material template."""
    # Element-group stub call
    class DummyGroup:
        mat = Law119Seatbelt(id=1, params={})

    group = DummyGroup()
    # solid_update(group, x, u, ur, dt, fint, mint) -> returns None without error
    res_solid = solid_update(group, None, None, None, 0.001, np.zeros(3), np.zeros(3))
    assert res_solid is None

    # tangent(group) -> returns None per template
    res_tan_group = tangent(group)
    assert res_tan_group is None

    # Constitutive update: raises NotImplementedError for solid elements
    mat = Law119Seatbelt(id=1, params={})
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    with pytest.raises(NotImplementedError):
        solid_update(mat, sig, deps)

    # Tangent for material entity returns (3, 3) membrane tangent
    tan_mat = tangent(mat)
    assert isinstance(tan_mat, np.ndarray)
    assert tan_mat.shape == (3, 3)


def test_law119_sound_speed():
    """Verify characteristic acoustic sound speed calculation."""
    mat = Law119Seatbelt(id=1, rho0=1.0e-3, params={"STIFF1": 2500.0, "NU12": 0.0})
    c = sound_speed(mat)
    # c = sqrt(E / rho) = sqrt(2500 / 1e-3) = sqrt(2.5e6) = 1581.1388
    assert math.isclose(c, math.sqrt(2500.0 / 1.0e-3), rel_tol=1e-5)
