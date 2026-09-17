"""
Milestone M587 Test Suite: Fortran-faithful Advanced Springs in pyradioss:
- /PROP/TYPE19 (SPR_TORS): Torsion Spring (Fortran r1tors.F / rforc3.F)
- /PROP/TYPE44 (SPR_CRUS): Crushing Spring with energy absorption (Fortran ruser44.F)
- /PROP/TYPE46 (SPR_MUSCLE): Active Muscle Spring with Hill-type dynamics (Fortran ruser46.F)

Tests:
1. /PROP/TYPE19: Relative twist angle, opposite moments, zero transverse force.
2. /PROP/TYPE19: Dynamic torsional vibration, frequency, and energy conservation.
3. /PROP/TYPE19: Rotational critical time step.
4. /PROP/TYPE44: Elastic compression, crushing plateau, plastic offset delta_p.
5. /PROP/TYPE44: Elastic unloading with slope K_unload back to zero force at delta_p.
6. /PROP/TYPE44: Plastic dissipation energy accounting and dynamic energy balance.
7. /PROP/TYPE46: Isometric active force scaling with activation alpha(t).
8. /PROP/TYPE46: Force-length and force-velocity Hill relationships.
9. /PROP/TYPE46: Passive elasticity and dynamic energy conservation.
10. Mixed spring group: TYPE4, TYPE19, TYPE44, TYPE46 coexistence in one group.
11. Starter deck reader parsing for /PROP/TYPE19, /PROP/TYPE44, /PROP/TYPE46 (fixed & free).
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.entities import Property, PropType19, PropType44, PropType46
from pyradioss.model.model import ElementGroup, Model
from pyradioss.elements import spring, spring_advanced


# ============================================================================
# 1. /PROP/TYPE19 Torsion Spring Tests
# ============================================================================

def test_type19_torsion_kinematics_and_moment_balance():
    """Verify TYPE19 torsion spring kinematics: relative angle theta about
    the line of nodes, applied opposite moments M1 = +M*a, M2 = -M*a,
    and zero transverse forces."""
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    # Spring aligned with X axis: (0,0,0) to (2,0,0)
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=np.float64)

    prop = Property(
        id=1, type=19, title="Torsion_Prop",
        params={"k_theta": 1000.0, "c_theta": 50.0, "inertia": 2.0, "mass": 1.0}
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )

    log = MessageLog()
    node_idx, massn, inertn = spring.init_group(group, model, log)
    assert not log.errors
    assert inertn is not None
    assert np.allclose(inertn, [1.0, 1.0])  # lumped half/half rotational inertia

    # Coordinate positions
    x = model.x0.copy()
    # Rotational velocities: Node 1 omega = [0,0,0], Node 2 omega = [10, 0, 0] rad/s (pure twist about X)
    vr = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
    ], dtype=np.float64)
    v = np.zeros_like(x)
    dt = 0.01

    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))

    dt_crit = spring.forces(group, x, v, vr, dt, fint, mint)

    # Relative angle theta after dt=0.01: theta = 10 * 0.01 = 0.1 rad
    theta = group.state["t19_theta"][0]
    assert theta == pytest.approx(0.1)

    # Moment M = K_theta * theta + C_theta * theta_dot = 1000 * 0.1 + 50 * 10 = 100 + 500 = 600
    M = group.state["t19_moment"][0]
    assert M == pytest.approx(600.0)

    # Restoring moments: node 1 gets +M*a = [+600, 0, 0], node 2 gets -M*a = [-600, 0, 0]
    assert mint[0, 0] == pytest.approx(600.0)
    assert mint[1, 0] == pytest.approx(-600.0)
    assert np.allclose(mint[:, 1:], 0.0)

    # Zero translational forces
    assert np.allclose(fint, 0.0)

    # Work booked into eint = 0.5 * (0 + 600) * 10 * 0.01 = 30.0
    eint = group.state["eint"][0]
    assert eint == pytest.approx(30.0)


def test_type19_dynamic_torsional_vibration_and_energy():
    """Simulate free torsional vibration of a two-node system connected by
    a TYPE19 torsion spring and verify conservation of total energy
    (kinetic + internal)."""
    k_theta = 5000.0
    inertia_node = 0.5  # per node

    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float64)

    prop = Property(
        id=1, type=19, title="Torsion",
        params={"k_theta": k_theta, "c_theta": 0.0, "inertia": 2.0 * inertia_node, "mass": 1.0}
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )

    log = MessageLog()
    spring.init_group(group, model, log)

    x = model.x0.copy()
    v = np.zeros_like(x)
    # Initial angular velocity: node 1 at -20 rad/s, node 2 at +20 rad/s
    vr = np.array([
        [-20.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
    ], dtype=np.float64)

    dt = 1.0e-4
    n_steps = 1000

    # Initial kinetic energy: 0.5 * I * (vr1^2 + vr2^2) = 0.5 * 0.5 * (400 + 400) = 200.0 J
    ke_initial = 0.5 * inertia_node * float((vr ** 2).sum())
    assert ke_initial == pytest.approx(200.0)

    for _ in range(n_steps):
        fint = np.zeros((2, 3))
        mint = np.zeros((2, 3))
        spring.forces(group, x, v, vr, dt, fint, mint)

        # Angular acceleration: alpha = mint / I
        alpha = mint / inertia_node
        # Leap-frog update
        vr += alpha * dt

    ke_final = 0.5 * inertia_node * float((vr ** 2).sum())
    eint_final = group.state["eint"][0]
    total_energy = ke_final + eint_final

    # Energy conservation within 0.1%
    assert total_energy == pytest.approx(ke_initial, rel=1e-3)


def test_type19_critical_time_step():
    """Verify analytical rotational critical time step dt = I / (sqrt(C^2 + I*K) + C)."""
    prop = Property(
        id=1, type=19, title="Torsion",
        params={"k_theta": 1000.0, "c_theta": 20.0, "inertia": 0.5, "mass": 1.0}
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    model = Model()
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    spring.init_group(group, model, MessageLog())

    x = model.x0.copy()
    v = np.zeros_like(x)
    vr = np.zeros_like(x)
    dt_crit = spring.forces(group, x, v, vr, 1e-4, np.zeros((2, 3)), np.zeros((2, 3)))

    I = 0.5
    K = 1000.0
    C = 20.0
    expected_dt = I / (math.sqrt(C * C + I * K) + C)
    assert dt_crit[0] == pytest.approx(expected_dt, rel=1e-5)


# ============================================================================
# 2. /PROP/TYPE44 Crushing Spring Tests
# ============================================================================

def test_type44_crushing_elastic_and_plateau():
    """Verify TYPE44 crushing spring:
    - Elastic response when |F| < F_yield
    - Crushing plateau at F = -F_yield when compressed past yield
    - Non-recoverable plastic displacement delta_p
    """
    k_unload = 10000.0
    f_yield = 2000.0

    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    # Initial length L0 = 10.0
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
    ], dtype=np.float64)

    prop = Property(
        id=1, type=44, title="Crush_Spring",
        params={"k_unload": k_unload, "f_yield": f_yield, "mass": 2.0}
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )

    spring.init_group(group, model, MessageLog())

    # 1. Elastic compression: delta = -0.1 -> F = 10000 * (-0.1) = -1000 N (below yield 2000 N)
    x = np.array([[0.0, 0.0, 0.0], [9.9, 0.0, 0.0]])
    fint = np.zeros((2, 3))
    spring.forces(group, x, None, None, 0.001, fint, None)

    assert group.state["force"][0] == pytest.approx(-1000.0)
    assert group.state["t44_delta_p"][0] == pytest.approx(0.0)  # no plastic deformation yet

    # 2. Crushing compression: delta = -0.5 -> F_trial = -5000 N (< -2000 N)
    # Force must clamp to -F_yield = -2000 N
    # Permanent plastic displacement: delta_p = delta + F_yield / K = -0.5 + 2000 / 10000 = -0.3
    x = np.array([[0.0, 0.0, 0.0], [9.5, 0.0, 0.0]])
    fint[:] = 0.0
    spring.forces(group, x, None, None, 0.001, fint, None)

    assert group.state["force"][0] == pytest.approx(-2000.0)
    assert group.state["t44_delta_p"][0] == pytest.approx(-0.3)
    assert group.state["t44_e_plastic"][0] == pytest.approx(f_yield * 0.3)


def test_type44_elastic_unloading_with_permanent_offset():
    """Verify TYPE44 elastic unloading with slope K_unload after crushing:
    Force returns to zero at permanent deformation delta = delta_p."""
    k_unload = 5000.0
    f_yield = 1000.0

    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model.x0 = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])

    prop = Property(
        id=1, type=44, title="Crush",
        params={"k_unload": k_unload, "f_yield": f_yield, "mass": 1.0}
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )

    spring.init_group(group, model, MessageLog())

    # Crush element by 1.0 m (L = 4.0 m)
    # F_yield = 1000 N, elastic yield delta_y = 1000 / 5000 = 0.2 m
    # delta_p = -1.0 + 0.2 = -0.8 m
    x_crush = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    spring.forces(group, x_crush, None, None, 0.001, np.zeros((2, 3)), None)
    assert group.state["t44_delta_p"][0] == pytest.approx(-0.8)

    # Now unload to L = 4.1 m (delta = -0.9 m, trial force = 5000 * (-0.9 - (-0.8)) = -500 N)
    x_unload1 = np.array([[0.0, 0.0, 0.0], [4.1, 0.0, 0.0]])
    spring.forces(group, x_unload1, None, None, 0.001, np.zeros((2, 3)), None)
    assert group.state["force"][0] == pytest.approx(-500.0)
    assert group.state["t44_delta_p"][0] == pytest.approx(-0.8)  # delta_p unchanged during unloading

    # Unload to L = 4.2 m (delta = -0.8 m == delta_p -> F must be exactly 0.0)
    x_zero = np.array([[0.0, 0.0, 0.0], [4.2, 0.0, 0.0]])
    spring.forces(group, x_zero, None, None, 0.001, np.zeros((2, 3)), None)
    assert group.state["force"][0] == pytest.approx(0.0, abs=1e-9)


def test_type44_energy_dissipation_and_conservation():
    """Verify that TYPE44 crushing work is fully accounted for in eint."""
    k_unload = 10000.0
    f_yield = 1000.0

    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model.x0 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    prop = Property(
        id=1, type=44, title="Crush",
        params={"k_unload": k_unload, "f_yield": f_yield, "mass": 1.0}
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )

    spring.init_group(group, model, MessageLog())

    # Move node 2 inward with constant velocity v = -1.0 m/s for 0.5 s (total delta = -0.5 m)
    # Elastic limit: delta_e = 1000 / 10000 = 0.1 m -> elastic energy = 0.5 * 1000 * 0.1 = 50 J
    # Plastic crushing: delta_p = 0.5 - 0.1 = 0.4 m -> plastic energy = 1000 * 0.4 = 400 J
    # Expected total internal energy = 450 J
    dt = 1e-3
    n_steps = 500
    x = model.x0.copy()
    v = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

    for _ in range(n_steps):
        x += v * dt
        fint = np.zeros((2, 3))
        spring.forces(group, x, v, None, dt, fint, None)

    eint = group.state["eint"][0]
    e_plastic = group.state["t44_e_plastic"][0]

    assert e_plastic == pytest.approx(400.0, rel=1e-2)
    assert eint == pytest.approx(450.0, rel=1e-2)


# ============================================================================
# 3. /PROP/TYPE46 Muscle Spring Tests
# ============================================================================

def test_type46_isometric_active_force_and_activation():
    """Verify TYPE46 muscle spring active force generation:
    F_active = F_max * alpha(t) * f_L(L/L0) * f_v(v/v_max)."""
    f_max = 3000.0
    v_max = 10.0
    l_opt = 1.0

    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    prop = Property(
        id=1, type=46, title="Muscle",
        params={
            "f_max": f_max, "v_max": v_max, "l_opt": l_opt,
            "activation": 0.8,  # 80% constant activation
            "k_pe": 500.0, "mass": 1.0
        }
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )

    spring.init_group(group, model, MessageLog())

    # Case A: Isometric at optimal length L = L0 = 1.0, v = 0
    # f_L(1.0) = 1.0, f_v(0) = 1.0, F_passive = 0
    # F = 3000 * 0.8 * 1.0 * 1.0 = 2400 N
    x = model.x0.copy()
    v = np.zeros_like(x)
    fint = np.zeros((2, 3))
    spring.forces(group, x, v, None, 0.001, fint, None)

    assert group.state["force"][0] == pytest.approx(2400.0)
    # Muscle is in tension -> pulls node 1 towards node 2 (+X) and node 2 towards node 1 (-X)
    assert fint[0, 0] == pytest.approx(2400.0)
    assert fint[1, 0] == pytest.approx(-2400.0)


def test_type46_force_length_and_velocity_relationships():
    """Verify TYPE46 Hill force-length parabolic relationship and
    hyperbolic force-velocity shortening relationship."""
    f_max = 1000.0
    v_max = 8.0
    l_opt = 1.0

    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    prop = Property(
        id=1, type=46, title="Muscle",
        params={
            "f_max": f_max, "v_max": v_max, "l_opt": l_opt,
            "activation": 1.0, "k_pe": 0.0, "mass": 1.0
        }
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )

    spring.init_group(group, model, MessageLog())

    # 1. Force-length scaling: L = 1.25 m (stretch ratio lambda = 1.25)
    # Default parabolic: f_L = 1 - ((lambda - 1) / 0.5)^2 = 1 - (0.25 / 0.5)^2 = 1 - 0.25 = 0.75
    x_stretch = np.array([[0.0, 0.0, 0.0], [1.25, 0.0, 0.0]])
    v_zero = np.zeros_like(x_stretch)
    spring.forces(group, x_stretch, v_zero, None, 0.001, np.zeros((2, 3)), None)
    assert group.state["force"][0] == pytest.approx(750.0)

    # 2. Force-velocity scaling: shortening at vc = 0.5 * v_max (4.0 m/s shortening)
    # Ldot = -4.0 m/s -> vc = -(-4/8) = 0.5
    # Hill curve: (1 - vc) / (1 + vc / 0.25) = (1 - 0.5) / (1 + 2.0) = 0.5 / 3.0 = 1/6
    x_opt = model.x0.copy()
    v_shorten = np.array([[0.0, 0.0, 0.0], [-4.0, 0.0, 0.0]])
    spring.forces(group, x_opt, v_shorten, None, 0.001, np.zeros((2, 3)), None)
    expected_fv = 1.0 / 6.0
    assert group.state["force"][0] == pytest.approx(1000.0 * expected_fv, rel=1e-3)


def test_type46_passive_elasticity_and_energy_conservation():
    """Verify TYPE46 passive elasticity for L > L0 and work booking into eint."""
    f_max = 0.0   # passive only
    k_pe = 2000.0
    l_opt = 2.0

    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model.x0 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    prop = Property(
        id=1, type=46, title="Muscle_Passive",
        params={"f_max": f_max, "k_pe": k_pe, "l_opt": l_opt, "mass": 1.0}
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )

    spring.init_group(group, model, MessageLog())

    # Stretch from L=2.0 to L=2.2 (delta = 0.2 m)
    # F_passive = K_pe * (L - L0) = 2000 * 0.2 = 400 N
    # Expected elastic work = 0.5 * 2000 * 0.2^2 = 40 J
    dt = 1e-3
    n_steps = 200
    x = model.x0.copy()
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # stretching at 1.0 m/s

    for _ in range(n_steps):
        x += v * dt
        spring.forces(group, x, v, None, dt, np.zeros((2, 3)), None)

    assert group.state["force"][0] == pytest.approx(400.0, rel=1e-2)
    assert group.state["eint"][0] == pytest.approx(40.0, rel=1e-2)


# ============================================================================
# 4. Mixed Spring Group Test
# ============================================================================

def test_mixed_spring_group_coexistence():
    """Verify that a single ElementGroup containing TYPE4 (axial), TYPE19 (torsion),
    TYPE44 (crushing), and TYPE46 (muscle) correctly dispatches forces and moments."""
    model = Model()
    model.node_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    model.x0 = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],  # elem 0: TYPE4
        [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],  # elem 1: TYPE19
        [4.0, 0.0, 0.0], [5.0, 0.0, 0.0],  # elem 2: TYPE44
        [6.0, 0.0, 0.0], [7.0, 0.0, 0.0],  # elem 3: TYPE46
    ], dtype=np.float64)

    p4 = Property(id=1, type=4, title="P_Axial", params={"k": 500.0, "mass": 1.0})
    p19 = Property(id=2, type=19, title="P_Torsion", params={"k_theta": 200.0, "inertia": 0.2, "mass": 1.0})
    p44 = Property(id=3, type=44, title="P_Crush", params={"k_unload": 1000.0, "f_yield": 100.0, "mass": 1.0})
    p46 = Property(id=4, type=46, title="P_Muscle", params={"f_max": 800.0, "activation": 1.0, "mass": 1.0})

    group = ElementGroup(
        ids=np.array([1, 2, 3, 4]),
        conn=np.array([
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ], dtype=np.int64),
        part=np.zeros(4, dtype=np.int64),
        state={
            "slices": [
                (slice(0, 1), None, p4),
                (slice(1, 2), None, p19),
                (slice(2, 3), None, p44),
                (slice(3, 4), None, p46),
            ]
        }
    )

    log = MessageLog()
    spring.init_group(group, model, log)
    assert not log.errors

    x = model.x0.copy()
    # Elem 0: stretch by 0.1 m
    x[1, 0] += 0.1
    # Elem 1: no stretch, but rotational velocity twist
    vr = np.zeros_like(x)
    vr[3, 0] = 5.0  # 5 rad/s
    # Elem 2: compress by 0.2 m (crush)
    x[5, 0] -= 0.2
    # Elem 3: isometric at optimal length

    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))
    dt = 0.01

    dt_crit = spring.forces(group, x, np.zeros_like(x), vr, dt, fint, mint)

    # 1. Elem 0 (TYPE4): F = 500 * 0.1 = 50 N
    assert fint[0, 0] == pytest.approx(50.0)
    assert fint[1, 0] == pytest.approx(-50.0)

    # 2. Elem 1 (TYPE19): M = 200 * (5.0 * 0.01) = 10.0 N*m
    assert mint[2, 0] == pytest.approx(10.0)
    assert mint[3, 0] == pytest.approx(-10.0)

    # 3. Elem 2 (TYPE44): F = -100 N (clamped at yield)
    assert fint[4, 0] == pytest.approx(-100.0)
    assert fint[5, 0] == pytest.approx(100.0)

    # 4. Elem 3 (TYPE46): F = 800 N (active muscle force)
    assert fint[6, 0] == pytest.approx(800.0)
    assert fint[7, 0] == pytest.approx(-800.0)

    # All 4 elements returned positive critical time steps
    assert np.all(dt_crit > 0.0)


# ============================================================================
# 5. Starter Deck Parsing Tests
# ============================================================================

def test_starter_parsing_advanced_spring_props(tmp_path: Path):
    """Verify Starter deck parsing for /PROP/TYPE19, /PROP/TYPE44, /PROP/TYPE46."""
    deck = """# RADIOSS STARTER DECK
/BEGIN
Advanced Springs Test
1 1
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 2.0 0.0 0.0
4 3.0 0.0 0.0
5 4.0 0.0 0.0
6 5.0 0.0 0.0
/SPRING/1
1 1 2
/SPRING/2
2 3 4
/SPRING/3
3 5 6
/PROP/TYPE19/1
Torsion Spring
1.5 0.5 2500.0 15.0
/PROP/TYPE44/2
Crushing Spring
2.0 0.0 10000.0
8000.0 0.0 0.0 0.0 0
/PROP/TYPE46/3
Muscle Spring
1.0 1500.0 1.0 8.0 200.0
/END
"""
    p = tmp_path / "TEST_0000.rad"
    p.write_text(deck.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)

    assert not log.errors

    # Check PropType19
    assert 1 in model.props_type19 or 1 in model.prop_type19s
    p19 = model.props_type19[1]
    assert p19.mass == pytest.approx(1.5)
    assert p19.inertia == pytest.approx(0.5)
    assert p19.k == pytest.approx(2500.0)
    assert p19.c == pytest.approx(15.0)

    # Check PropType44
    assert 2 in model.prop_type44s
    p44 = model.prop_type44s[2]
    assert p44.mass == pytest.approx(2.0)
    assert p44.stiff1 == pytest.approx(10000.0)

    # Check PropType46
    assert 3 in model.prop_type46s
    p46 = model.prop_type46s[3]
    assert p46.mass == pytest.approx(1.0)
    assert p46.stiff0 == pytest.approx(1500.0)  # first float on card
