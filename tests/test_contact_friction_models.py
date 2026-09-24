"""Unit tests for /FRICTION advanced friction models ported from OpenRadioss.

Upstream references:
- hm_read_friction_models.F
- hm_read_friction.F
- hm_read_friction_orientations.F
- i7for3.F (lines 1860-2295, 2475-2600)
"""

import math
import numpy as np
import pytest

from pyradioss.contact.friction_models import (
    FrictionModel,
    compute_friction_coefficient,
    compute_friction_force,
)


def test_static_to_dynamic_velocity_transition():
    """Test static-to-dynamic exponential velocity transition:
    mu(v_rel) = mu_d + (mu_s - mu_d) * exp(-c * |v_rel|)
    """
    mu_s = 0.55
    mu_d = 0.25
    decay_c = 2.0  # 1/(m/s)

    model = FrictionModel(
        mu_0=mu_s,
        mu_s=mu_s,
        mu_d=mu_d,
        decay_coef=decay_c,
    )

    # 1. At v = 0, mu should equal static friction mu_s
    mu_0 = compute_friction_coefficient(model, v_rel=0.0)
    assert np.isclose(mu_0, mu_s, atol=1.0e-12)

    # 2. At very high velocity (v >> 1/c), mu should approach dynamic friction mu_d
    mu_inf = compute_friction_coefficient(model, v_rel=100.0)
    assert np.isclose(mu_inf, mu_d, atol=1.0e-6)

    # 3. Intermediate velocities
    for v in [0.1, 0.5, 1.0, 2.0, 5.0]:
        expected = mu_d + (mu_s - mu_d) * math.exp(-decay_c * v)
        val = compute_friction_coefficient(model, v_rel=v)
        assert np.isclose(val, expected, atol=1.0e-12)

    # 4. Monotonicity check over velocity sweep
    v_sweep = np.linspace(0.0, 5.0, 50)
    mu_sweep = compute_friction_coefficient(model, v_rel=v_sweep)
    diffs = np.diff(mu_sweep)
    assert np.all(diffs <= 0.0), "Friction must monotonically decrease from static to dynamic"

    # 5. Velocity sign invariance (|v_rel|)
    mu_pos = compute_friction_coefficient(model, v_rel=1.5)
    mu_neg = compute_friction_coefficient(model, v_rel=-1.5)
    assert np.isclose(mu_pos, mu_neg, atol=1.0e-14)


def test_pressure_power_law_scaling():
    """Test pressure dependence power-law model:
    mu(P) = mu_0 * (P / P_0) ** n_p
    """
    mu_0 = 0.35
    p_0 = 50.0  # MPa
    n_p = -0.2  # Softening with pressure

    model = FrictionModel(
        mu_0=mu_0,
        p_0=p_0,
        n_p=n_p,
    )

    # 1. At reference pressure P = P_0, mu should equal mu_0
    mu_ref = compute_friction_coefficient(model, P=p_0)
    assert np.isclose(mu_ref, mu_0, atol=1.0e-12)

    # 2. At P = 2 * P_0 with n_p = -0.2
    p_high = 2.0 * p_0
    expected_high = mu_0 * (2.0 ** n_p)
    mu_high = compute_friction_coefficient(model, P=p_high)
    assert np.isclose(mu_high, expected_high, atol=1.0e-12)

    # 3. Positive pressure exponent (e.g. pressure-hardening)
    model_pos = FrictionModel(mu_0=0.2, p_0=10.0, n_p=0.5)
    p_quad = 40.0
    expected_quad = 0.2 * ((40.0 / 10.0) ** 0.5)  # 0.2 * 2.0 = 0.4
    assert np.isclose(compute_friction_coefficient(model_pos, P=p_quad), expected_quad, atol=1.0e-12)

    # 4. Array of pressures
    p_arr = np.array([25.0, 50.0, 100.0, 200.0])
    mu_arr = compute_friction_coefficient(model, P=p_arr)
    expected_arr = mu_0 * (p_arr / p_0) ** n_p
    assert np.allclose(mu_arr, expected_arr, atol=1.0e-12)


def test_temperature_dependence():
    """Test temperature softening model:
    mu(T) = mu_0 * max(0.0, 1.0 - a_T * (T - T_0))
    """
    mu_0 = 0.4
    t_0 = 293.15  # 20 C in Kelvin
    a_t = 0.002   # 1/K

    model = FrictionModel(
        mu_0=mu_0,
        t_0=t_0,
        a_t=a_t,
    )

    # 1. At T = T_0, mu equals mu_0
    assert np.isclose(compute_friction_coefficient(model, T=t_0), mu_0, atol=1.0e-12)

    # 2. At T = T_0 + 100 K
    t_hot = t_0 + 100.0
    expected_hot = mu_0 * (1.0 - a_t * 100.0)  # 0.4 * 0.8 = 0.32
    assert np.isclose(compute_friction_coefficient(model, T=t_hot), expected_hot, atol=1.0e-12)

    # 3. High temperature clamp (no negative friction)
    t_extreme = t_0 + 1000.0  # (1 - 0.002 * 1000) = -1.0 -> clamped to 0 / mu_min
    assert compute_friction_coefficient(model, T=t_extreme) >= model.mu_min


def test_anisotropic_friction_elliptical_variation():
    """Test anisotropic / orthotropic friction elliptical variation with sliding angle:
    mu(theta) = sqrt((mu_1 * cos(theta))**2 + (mu_2 * sin(theta))**2)
    """
    mu_1 = 0.60  # Longitudinal
    mu_2 = 0.30  # Transverse
    ortho_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    model = FrictionModel(
        is_orthotropic=True,
        mu_1=mu_1,
        mu_2=mu_2,
        ortho_dir=ortho_axis,
    )

    # 1. Longitudinal sliding (theta = 0) -> mu = mu_1
    mu_long = compute_friction_coefficient(model, theta=0.0)
    assert np.isclose(mu_long, mu_1, atol=1.0e-12)

    # Test via direction vector [1, 0, 0]
    mu_long_vec = compute_friction_coefficient(model, dir_tangent=np.array([1.0, 0.0, 0.0]))
    assert np.isclose(mu_long_vec, mu_1, atol=1.0e-12)

    # Opposite longitudinal direction [-1, 0, 0] -> still mu_1
    mu_long_neg = compute_friction_coefficient(model, dir_tangent=np.array([-1.0, 0.0, 0.0]))
    assert np.isclose(mu_long_neg, mu_1, atol=1.0e-12)

    # 2. Transverse sliding (theta = pi/2) -> mu = mu_2
    mu_trans = compute_friction_coefficient(model, theta=math.pi / 2.0)
    assert np.isclose(mu_trans, mu_2, atol=1.0e-12)

    # Test via direction vector [0, 1, 0]
    mu_trans_vec = compute_friction_coefficient(model, dir_tangent=np.array([0.0, 1.0, 0.0]))
    assert np.isclose(mu_trans_vec, mu_2, atol=1.0e-12)

    # 3. 45-degree sliding (theta = pi/4)
    # mu(pi/4) = sqrt(0.5 * mu_1^2 + 0.5 * mu_2^2)
    expected_45 = math.sqrt(0.5 * (mu_1 ** 2 + mu_2 ** 2))
    mu_45 = compute_friction_coefficient(model, theta=math.pi / 4.0)
    assert np.isclose(mu_45, expected_45, atol=1.0e-12)

    # Via 45-degree tangent vector [1, 1, 0]
    mu_45_vec = compute_friction_coefficient(model, dir_tangent=np.array([1.0, 1.0, 0.0]))
    assert np.isclose(mu_45_vec, expected_45, atol=1.0e-12)

    # 4. Array of angles across all 4 quadrants
    thetas = np.linspace(0.0, 2.0 * math.pi, 25)
    mu_thetas = compute_friction_coefficient(model, theta=thetas)
    expected_thetas = np.sqrt((mu_1 * np.cos(thetas)) ** 2 + (mu_2 * np.sin(thetas)) ** 2)
    assert np.allclose(mu_thetas, expected_thetas, atol=1.0e-12)


def test_combined_velocity_pressure_temperature_friction():
    """Test combined velocity-decay, pressure power-law, and temperature softening friction:
    mu = (mu_d + (mu_s - mu_d) * exp(-c * v)) * (P / P_0)**n_p * (1 - a_T * (T - T_0))
    """
    mu_s = 0.60
    mu_d = 0.25
    c_decay = 1.5
    p_0 = 20.0
    n_p = -0.15
    t_0 = 293.15
    a_t = 0.0015

    model = FrictionModel(
        mu_0=mu_s,
        mu_s=mu_s,
        mu_d=mu_d,
        decay_coef=c_decay,
        p_0=p_0,
        n_p=n_p,
        t_0=t_0,
        a_t=a_t,
    )

    # Reference state: v = 0, P = P_0, T = T_0 -> mu = mu_s
    mu_ref = compute_friction_coefficient(model, v_rel=0.0, P=p_0, T=t_0)
    assert np.isclose(mu_ref, mu_s, atol=1.0e-12)

    # High velocity at reference P, T -> mu = mu_d
    mu_high_v = compute_friction_coefficient(model, v_rel=50.0, P=p_0, T=t_0)
    assert np.isclose(mu_high_v, mu_d, atol=1.0e-6)

    # General state: v = 1.2, P = 40.0, T = 350.0
    v_test = 1.2
    p_test = 40.0
    t_test = 350.0

    term_v = mu_d + (mu_s - mu_d) * math.exp(-c_decay * v_test)
    term_p = (p_test / p_0) ** n_p
    term_t = 1.0 - a_t * (t_test - t_0)
    expected_combined = term_v * term_p * term_t

    val_combined = compute_friction_coefficient(model, v_rel=v_test, P=p_test, T=t_test)
    assert np.isclose(val_combined, expected_combined, atol=1.0e-12)


def test_compute_friction_force_vector_and_batch():
    """Test compute_friction_force for 1D vectors, zero velocity, and batch arrays."""
    mu_0 = 0.3
    model = FrictionModel(mu_0=mu_0)

    # 1. 1D vector sliding along X: v = [2.0, 0.0, 0.0], Fn = 1500.0 N
    v_vec = np.array([2.0, 0.0, 0.0])
    fn = 1500.0
    f_fric = compute_friction_force(model, F_normal=fn, v_rel_vec=v_vec)

    # Direction opposes velocity -> [-1, 0, 0]
    expected_force = np.array([-mu_0 * fn, 0.0, 0.0])
    assert np.allclose(f_fric, expected_force, atol=1.0e-12)
    assert np.isclose(np.linalg.norm(f_fric), mu_0 * fn, atol=1.0e-12)

    # 2. Negative normal force (compression) should use absolute value |Fn|
    f_fric_neg = compute_friction_force(model, F_normal=-fn, v_rel_vec=v_vec)
    assert np.allclose(f_fric_neg, expected_force, atol=1.0e-12)

    # 3. Zero velocity produces zero friction force
    v_zero = np.zeros(3)
    assert np.allclose(compute_friction_force(model, F_normal=fn, v_rel_vec=v_zero), np.zeros(3))

    # 4. Arbitrary 3D sliding vector [3, 4, 0] (magnitude 5)
    v_3d = np.array([3.0, 4.0, 0.0])
    f_3d = compute_friction_force(model, F_normal=100.0, v_rel_vec=v_3d)
    expected_3d = -mu_0 * 100.0 * (v_3d / 5.0)
    assert np.allclose(f_3d, expected_3d, atol=1.0e-12)

    # 5. Batch / Multi-point evaluation (N x 3)
    v_batch = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    fn_batch = np.array([100.0, 200.0, 300.0])
    f_batch = compute_friction_force(model, F_normal=fn_batch, v_rel_vec=v_batch)

    assert np.allclose(f_batch[0], np.array([-30.0, 0.0, 0.0]))
    assert np.allclose(f_batch[1], np.array([0.0, -60.0, 0.0]))
    assert np.allclose(f_batch[2], np.array([0.0, 0.0, 0.0]))


def test_openradioss_mfrot_models():
    """Test OpenRadioss standard MFROT laws (0=Coulomb, 1=Viscous, 2=Darmstadt, 3=Renard, 4=ExpDecay)."""
    # MFROT 0: Coulomb
    m0 = FrictionModel(mu_0=0.25, mfrot=0)
    assert np.isclose(compute_friction_coefficient(m0, v_rel=10.0, P=5.0), 0.25)

    # MFROT 1: Viscous polynomial: mu = mu_0 + (c1 + c4*p)*p + (c2 + c3*p)*v + c5*v^2
    m1 = FrictionModel(
        mu_0=0.2,
        mfrot=1,
        c1=0.01,
        c2=0.02,
        c3=0.001,
        c4=0.0005,
        c5=0.002,
    )
    p = 4.0
    v = 3.0
    expected_m1 = 0.2 + (0.01 + 0.0005 * p) * p + (0.02 + 0.001 * p) * v + 0.002 * (v ** 2)
    assert np.isclose(compute_friction_coefficient(m1, v_rel=v, P=p), expected_m1, atol=1.0e-12)

    # MFROT 2: Darmstadt: mu = mu_0 + c1*exp(c2*v)*p^2 + c3*exp(c4*v)*p + c5*exp(c6*v)
    m2 = FrictionModel(
        mu_0=0.15,
        mfrot=2,
        c1=0.001,
        c2=0.05,
        c3=0.005,
        c4=0.02,
        c5=0.01,
        c6=0.03,
    )
    expected_m2 = (
        0.15
        + 0.001 * math.exp(0.05 * v) * (p ** 2)
        + 0.005 * math.exp(0.02 * v) * p
        + 0.01 * math.exp(0.03 * v)
    )
    assert np.isclose(compute_friction_coefficient(m2, v_rel=v, P=p), expected_m2, atol=1.0e-12)

    # MFROT 3: Renard law
    m3 = FrictionModel(
        mfrot=3,
        c1=0.1,   # mu_min
        c2=0.35,  # mu_max
        c3=0.2,   # mu at v_cr1
        c4=0.3,   # mu at v_cr2
        c5=1.0,   # v_cr1
        c6=5.0,   # v_cr2
    )
    # v = 0 -> c1 = 0.1
    assert np.isclose(compute_friction_coefficient(m3, v_rel=0.0), 0.1, atol=1.0e-12)
    # v = c5 = 1.0 -> c3 = 0.2
    assert np.isclose(compute_friction_coefficient(m3, v_rel=1.0), 0.2, atol=1.0e-12)
    # v = c6 = 5.0 -> c4 = 0.3
    assert np.isclose(compute_friction_coefficient(m3, v_rel=5.0), 0.3, atol=1.0e-12)

    # MFROT 4: Exponential decay mapped from C1, C2
    m4 = FrictionModel(
        mu_0=0.5,
        mfrot=4,
        c1=0.2,   # dynamic coefficient
        c2=1.0,   # decay rate
    )
    assert np.isclose(compute_friction_coefficient(m4, v_rel=0.0), 0.5, atol=1.0e-12)
    assert np.isclose(compute_friction_coefficient(m4, v_rel=1.0), 0.2 + 0.3 * math.exp(-1.0), atol=1.0e-12)


def test_friction_model_from_dict():
    """Test building FrictionModel from deck / starter card dictionary."""
    data = {
        "id": 12,
        "title": "TIRE_ROAD_FRICTION",
        "mu_0": 0.7,
        "mu_s": 0.85,
        "mu_d": 0.45,
        "decay_coef": 2.5,
        "p_0": 0.25,
        "n_p": -0.1,
        "t_0": 298.15,
        "a_t": 0.002,
        "is_orthotropic": True,
        "mu_1": 0.85,
        "mu_2": 0.60,
        "ortho_dir": [1.0, 0.0, 0.0],
    }
    model = FrictionModel.from_dict(data)
    assert model.id == 12
    assert model.title == "TIRE_ROAD_FRICTION"
    assert model.mu_0 == 0.7
    assert model.mu_s == 0.85
    assert model.mu_d == 0.45
    assert model.decay_coef == 2.5
    assert model.p_0 == 0.25
    assert model.n_p == -0.1
    assert model.is_orthotropic is True
    assert model.mu_1 == 0.85
    assert model.mu_2 == 0.60
    assert np.allclose(model.ortho_dir, [1.0, 0.0, 0.0])
