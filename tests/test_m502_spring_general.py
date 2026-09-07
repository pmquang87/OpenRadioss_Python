"""
Unit tests for Milestone M502:
/PROP/TYPE8 (SPR_GENE) & /PROP/TYPE13 (SPR_BEAM) 6-DOF General Spring Elements
Hardening, Moving Skew Dynamics, Rotational Equilibrium & Kinematics.

Fortran origin:
  engine/source/elements/spring/rforc3.F
  engine/source/elements/spring/r2def3.F
  engine/source/elements/spring/r2coor3.F
  engine/source/elements/spring/r2cum3.F
  engine/source/elements/spring/r2len3.F
  starter/source/properties/spring/hm_read_prop08.F
  starter/source/properties/spring/hm_read_prop13.F
"""

import math
from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.constants import EM20, EP30
from pyradioss.common.messages import MessageLog
from pyradioss.elements import spring, spring_general
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck, read_prop
from pyradioss.model.entities import Part, Property
from pyradioss.model.model import ElementGroup, Model
from pyradioss.model.skew import SkewFrame, SkewSet
from pyradioss.starter.initialization import build_element_groups


# ---------------------------------------------------------------------------
# Helper to build a 2-node spring model with given property
# ---------------------------------------------------------------------------
def _make_spring_model(prop, x0_nodes=None):
    if x0_nodes is None:
        x0_nodes = np.array([[0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0]])
    m = Model()
    m.add_nodes(np.array([1, 2]), x0_nodes)
    g = ElementGroup(ids=np.array([1]), conn=np.array([[0, 1]]),
                     part=np.array([0]))
    g.state["slices"] = [(slice(0, 1), None, prop)]
    m.springs = g
    return m, g


# ============================================================================
# 1. 6-DOF Uncoupled Stiffness (Pure Translations & Pure Rotations)
# ============================================================================
def test_6dof_uncoupled_translations():
    """Verify independent stiffness response on each translation axis (X, Y, Z)."""
    k = [100.0, 200.0, 300.0, 0.0, 0.0, 0.0]
    p = Property(id=1, type=8, params={
        "mass": 2.0, "inertia": 1.0, "skew_id": 0,
        "k1": k[0], "k2": k[1], "k3": k[2],
        "k4": k[3], "k5": k[4], "k6": k[5],
    })
    m, g = _make_spring_model(p, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    log = MessageLog()
    spring.init_group(g, m, log)

    # Displace node 2 by dx=0.1, dy=0.2, dz=-0.3
    x = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, -0.3]])
    v = np.zeros((2, 3))
    vr = np.zeros((2, 3))
    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    dt = 1e-4

    dtc = spring.forces(g, x, v, vr, dt, fint, mint)

    # Expected forces: F = K * delta
    # Node 1 gets +F, Node 2 gets -F
    expected_F = np.array([100.0 * 0.1, 200.0 * 0.2, 300.0 * -0.3])
    np.testing.assert_allclose(fint[0], expected_F, rtol=1e-6)
    np.testing.assert_allclose(fint[1], -expected_F, rtol=1e-6)
    np.testing.assert_allclose(mint, 0.0, atol=1e-12)


def test_6dof_uncoupled_rotations():
    """Verify independent stiffness response on each rotation axis (RX, RY, RZ)."""
    k = [0.0, 0.0, 0.0, 400.0, 500.0, 600.0]
    p = Property(id=1, type=8, params={
        "mass": 1.0, "inertia": 2.0, "skew_id": 0,
        "k1": k[0], "k2": k[1], "k3": k[2],
        "k4": k[3], "k5": k[4], "k6": k[5],
    })
    m, g = _make_spring_model(p, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    log = MessageLog()
    spring.init_group(g, m, log)

    x = m.x0.copy()
    v = np.zeros((2, 3))
    # Angular velocity on node 2: wx=10, wy=-20, wz=30 rad/s
    vr = np.array([[0.0, 0.0, 0.0], [10.0, -20.0, 30.0]])
    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    dt = 0.005

    spring.forces(g, x, v, vr, dt, fint, mint)

    # Expected accumulated rotation: theta = wr * dt
    # Expected moments: M = K * theta
    expected_theta = np.array([10.0, -20.0, 30.0]) * dt
    expected_M = np.array([400.0, 500.0, 600.0]) * expected_theta

    np.testing.assert_allclose(mint[0], expected_M, rtol=1e-6)
    np.testing.assert_allclose(mint[1], -expected_M, rtol=1e-6)
    np.testing.assert_allclose(fint, 0.0, atol=1e-12)


# ============================================================================
# 2. Multi-Cycle Reversible Rotation Integration
# ============================================================================
def test_rotation_rate_integration_and_reversibility():
    """Verify rate-integrated rotation accumulation across steps and reversible return to zero."""
    p = Property(id=1, type=8, params={
        "mass": 1.0, "inertia": 2.0, "skew_id": 0,
        "k6": 1000.0,
    })
    m, g = _make_spring_model(p, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    log = MessageLog()
    spring.init_group(g, m, log)

    x = m.x0.copy()
    v = np.zeros((2, 3))
    dt = 0.01

    # Spin forward for 5 steps at wz = 2.0 rad/s -> theta_z = 0.1 rad
    for step in range(5):
        vr = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        fint = np.zeros((2, 3))
        mint = np.zeros((2, 3))
        spring.forces(g, x, v, vr, dt, fint, mint)

    gen = g.state["gen6"]
    np.testing.assert_allclose(gen["theta"][0], [0.0, 0.0, 0.1], rtol=1e-6)
    np.testing.assert_allclose(gen["moment"][0], [0.0, 0.0, 100.0], rtol=1e-6)

    # Spin backward for 5 steps at wz = -2.0 rad/s -> theta_z returns to 0.0
    for step in range(5):
        vr = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -2.0]])
        fint = np.zeros((2, 3))
        mint = np.zeros((2, 3))
        spring.forces(g, x, v, vr, dt, fint, mint)

    np.testing.assert_allclose(gen["theta"][0], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(gen["moment"][0], [0.0, 0.0, 0.0], atol=1e-12)


# ============================================================================
# 3. Viscous Damping Forces on All 6 DOFs
# ============================================================================
def test_viscous_damping_translations_and_rotations():
    """Verify viscous damping force C * v_rel and moment C * w_rel."""
    p = Property(id=1, type=8, params={
        "mass": 1.0, "inertia": 1.0, "skew_id": 0,
        "k1": 0.0, "c1": 15.0,
        "k2": 0.0, "c2": 25.0,
        "k3": 0.0, "c3": 35.0,
        "k4": 0.0, "c4": 45.0,
        "k5": 0.0, "c5": 55.0,
        "k6": 0.0, "c6": 65.0,
    })
    m, g = _make_spring_model(p, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    log = MessageLog()
    spring.init_group(g, m, log)

    x = m.x0.copy()
    v = np.array([[0.0, 0.0, 0.0], [2.0, -3.0, 4.0]])
    vr = np.array([[0.0, 0.0, 0.0], [5.0, -6.0, 7.0]])
    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    dt = 1e-3

    spring.forces(g, x, v, vr, dt, fint, mint)

    expected_F = np.array([15.0 * 2.0, 25.0 * -3.0, 35.0 * 4.0])
    expected_M = np.array([45.0 * 5.0, 55.0 * -6.0, 65.0 * 7.0])

    np.testing.assert_allclose(fint[0], expected_F, rtol=1e-6)
    np.testing.assert_allclose(fint[1], -expected_F, rtol=1e-6)
    np.testing.assert_allclose(mint[0], expected_M, rtol=1e-6)
    np.testing.assert_allclose(mint[1], -expected_M, rtol=1e-6)


# ============================================================================
# 4. Static Skew Coordinate System (TYPE8 with SKEW)
# ============================================================================
def test_static_skew_frame_transformation():
    """Verify TYPE8 spring resolved in a static rotated skew frame."""
    # Skew rotated by 90 degrees around global Z:
    # Local X\' = (0, 1, 0)
    # Local Y\' = (-1, 0, 0)
    # Local Z\' = (0, 0, 1)
    skews = SkewSet()
    skews.axes = np.zeros((2, 3, 3))
    skews.axes[0] = np.eye(3)
    skews.axes[1] = np.array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    p = Property(id=1, type=8, params={
        "mass": 2.0, "inertia": 1.0, "skew_id": 1, "skew_row": 1,
        "k1": 500.0,   # local X\' stiffness (acts along global Y)
        "k2": 100.0,   # local Y\' stiffness (acts along global -X)
        "k3": 50.0,
    })
    m, g = _make_spring_model(p, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    m.skews = skews
    log = MessageLog()
    spring.init_group(g, m, log)

    # Displace node 2 by +0.2 along global Y:
    # In local skew: delta_X\' = d . e1 = (0, 0.2, 0) . (0, 1, 0) = 0.2
    # delta_Y\' = 0, delta_Z\' = 0
    x = np.array([[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]])
    v = np.zeros((2, 3))
    vr = np.zeros((2, 3))
    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    dt = 1e-4

    spring.forces(g, x, v, vr, dt, fint, mint)

    # Local force: F_X\' = 500.0 * 0.2 = 100.0, F_Y\' = 0, F_Z\' = 0
    # Global force: fvec = F_X\' * e1 = 100.0 * (0, 1, 0) = (0, 100, 0)
    expected_fvec = np.array([0.0, 100.0, 0.0])
    np.testing.assert_allclose(fint[0], expected_fvec, rtol=1e-6)
    np.testing.assert_allclose(fint[1], -expected_fvec, rtol=1e-6)


# ============================================================================
# 5. Moving Skew Coordinate System (TYPE8 with /SKEW/MOV)
# ============================================================================
def test_moving_skew_frame_dynamic_reloading():
    """Verify TYPE8 spring on a /SKEW/MOV reloads its axes each cycle as the skew turns."""
    skews = SkewSet()
    skews.axes = np.zeros((2, 3, 3))
    skews.axes[0] = np.eye(3)
    skews.axes[1] = np.eye(3)   # initially identity
    skews._mov_rows = np.array([1])  # mark row 1 as moving

    p = Property(id=1, type=8, params={
        "mass": 2.0, "inertia": 1.0, "skew_id": 1, "skew_row": 1,
        "k1": 1000.0,  # along local X\'
        "k2": 0.0,
        "k3": 0.0,
    })
    m, g = _make_spring_model(p, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    m.skews = skews
    log = MessageLog()
    spring.init_group(g, m, log)

    # Step 1: Skew is along global X. Stretch node 2 along global X by 0.1
    x = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    v = np.zeros((2, 3))
    vr = np.zeros((2, 3))
    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    dt = 1e-4
    spring.forces(g, x, v, vr, dt, fint, mint)
    np.testing.assert_allclose(fint[0], [100.0, 0.0, 0.0], rtol=1e-6)

    # Step 2: Skew rotates 90 degrees around Z: local X\' is now global Y!
    skews.axes[1] = np.array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    # Stretch node 2 along global Y by 0.1
    x = np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]])
    fint.fill(0.0)
    mint.fill(0.0)
    spring.forces(g, x, v, vr, dt, fint, mint)

    # Restoring force must now act along global Y!
    np.testing.assert_allclose(fint[0], [0.0, 100.0, 0.0], rtol=1e-6)
    np.testing.assert_allclose(fint[1], [0.0, -100.0, 0.0], rtol=1e-6)


# ============================================================================
# 6. TYPE13 Spring-Beam Element Frame Construction
# ============================================================================
def test_type13_element_frame_construction():
    """Verify TYPE13 builds an orthonormal element frame along N1->N2."""
    p = Property(id=1, type=13, params={
        "mass": 2.0, "inertia": 1.0, "skew_id": 0,
        "k1": 1000.0, "k2": 200.0, "k3": 300.0,
    })
    # Element along diagonal (1, 1, 0)
    L = math.sqrt(2.0)
    x0 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    m, g = _make_spring_model(p, x0)
    log = MessageLog()
    spring.init_group(g, m, log)

    gen = g.state["gen6"]
    e1 = gen["e1"][0]
    e2 = gen["e2"][0]
    e3 = gen["e3"][0]

    # e1 must be unit vector along (1, 1, 0)
    expected_e1 = np.array([1.0 / L, 1.0 / L, 0.0])
    np.testing.assert_allclose(e1, expected_e1, rtol=1e-6)
    # Check orthonormality
    np.testing.assert_allclose(np.linalg.norm(e1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(e2), 1.0, rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(e3), 1.0, rtol=1e-6)
    np.testing.assert_allclose(np.dot(e1, e2), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.dot(e1, e3), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.dot(e2, e3), 0.0, atol=1e-12)
    # Right-handed frame: det([e1, e2, e3]) == 1.0
    R = np.column_stack([e1, e2, e3])
    np.testing.assert_allclose(np.linalg.det(R), 1.0, rtol=1e-6)


def test_type13_coincident_nodes_fallback():
    """Verify TYPE13 with coincident nodes (L <= EM20) falls back to global axes."""
    p = Property(id=1, type=13, params={
        "mass": 1.0, "inertia": 1.0, "skew_id": 0,
        "k1": 100.0,
    })
    # Coincident nodes at origin
    x0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    m, g = _make_spring_model(p, x0)
    log = MessageLog()
    spring.init_group(g, m, log)

    gen = g.state["gen6"]
    np.testing.assert_allclose(gen["e1"][0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(gen["e2"][0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(gen["e3"][0], [0.0, 0.0, 1.0], atol=1e-12)


# ============================================================================
# 7. Linear Momentum Conservation
# ============================================================================
def test_exact_linear_momentum_conservation():
    """Verify sum(F) = 0 for arbitrary 3D orientations and deformations."""
    p = Property(id=1, type=8, params={
        "mass": 2.0, "inertia": 1.0, "skew_id": 0,
        "k1": 123.0, "k2": 456.0, "k3": 789.0,
        "c1": 12.0, "c2": 34.0, "c3": 56.0,
    })
    m, g = _make_spring_model(p, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    log = MessageLog()
    spring.init_group(g, m, log)

    x = np.array([[1.1, 1.9, 3.2], [4.5, 4.8, 6.7]])
    v = np.array([[0.5, -0.2, 1.0], [-0.3, 0.8, -0.4]])
    vr = np.zeros((2, 3))
    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    dt = 1e-3

    spring.forces(g, x, v, vr, dt, fint, mint)

    # Newton\'s third law: f1 + f2 == 0
    total_force = fint[0] + fint[1]
    np.testing.assert_allclose(total_force, 0.0, atol=1e-14)


# ============================================================================
# 8. Rotational Equilibrium (iequil=1) vs Discrete Mode (iequil=0)
# ============================================================================
def test_rotational_equilibrium_iequil():
    """Verify Fortran r2cum3.F moment arm correction when iequil=1.
    Exact total angular momentum conservation: sum(M) + sum(r x F) = 0.
    """
    # Spring length d = 2.0 along X, transverse force along Y (k2 > 0)
    p_no_eq = Property(id=1, type=8, params={
        "mass": 2.0, "inertia": 1.0, "skew_id": 0, "iequil": 0,
        "k2": 1000.0,
    })
    p_eq = Property(id=2, type=8, params={
        "mass": 2.0, "inertia": 1.0, "skew_id": 0, "iequil": 1,
        "k2": 1000.0,
    })

    # Case 1: iequil = 0 (discrete mode)
    m0, g0 = _make_spring_model(p_no_eq, np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    spring.init_group(g0, m0, MessageLog())
    x = np.array([[0.0, 0.0, 0.0], [2.0, 0.1, 0.0]])  # dy = 0.1
    fint0 = np.zeros((2, 3))
    mint0 = np.zeros((2, 3))
    spring.forces(g0, x, None, None, 1e-4, fint0, mint0)

    # In discrete mode, pure transverse translation generates zero nodal moments
    np.testing.assert_allclose(mint0, 0.0, atol=1e-12)
    # Total external torque about origin is r2 x f2 != 0
    torque0 = np.cross(x[0], fint0[0]) + np.cross(x[1], fint0[1]) + mint0[0] + mint0[1]
    assert np.linalg.norm(torque0) > 10.0

    # Case 2: iequil = 1 (rotational equilibrium mode)
    m1, g1 = _make_spring_model(p_eq, np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    spring.init_group(g1, m1, MessageLog())
    fint1 = np.zeros((2, 3))
    mint1 = np.zeros((2, 3))
    spring.forces(g1, x, None, None, 1e-4, fint1, mint1)

    # Forces are unchanged:
    np.testing.assert_allclose(fint1, fint0, rtol=1e-6)

    # Total angular momentum (torque about origin + internal moments) MUST BE ZERO!
    # torque = x1 x f1 + x2 x f2 + m1 + m2 == 0
    torque1 = np.cross(x[0], fint1[0]) + np.cross(x[1], fint1[1]) + mint1[0] + mint1[1]
    np.testing.assert_allclose(torque1, 0.0, atol=1e-13)


# ============================================================================
# 9. Critical Time Step with Exact Damping Reduction
# ============================================================================
def test_critical_time_step_damping_reduction():
    """Verify dt = M / (sqrt(C^2 + M*K) + C) matching Fortran r2len3.F:182."""
    mass = 2.0
    k = 800.0
    c = 10.0

    # Analytical formula:
    # dt_tr = mass / (sqrt(c^2 + mass * k) + c)
    dt_analytical = mass / (math.sqrt(c * c + mass * k) + c)

    p = Property(id=1, type=8, params={
        "mass": mass, "inertia": 0.5, "skew_id": 0,
        "k1": k, "c1": c,
    })
    m, g = _make_spring_model(p)
    spring.init_group(g, m, MessageLog())

    dtc = spring.forces(g, m.x0, None, None, 1e-4, np.zeros((2, 3)), np.zeros((2, 3)))
    np.testing.assert_allclose(dtc[0], dt_analytical, rtol=1e-7)


def test_critical_time_step_undamped_limit():
    """Verify undamped critical time step matches 1 / sqrt(K / M)."""
    mass = 4.0
    k = 100.0
    p = Property(id=1, type=8, params={
        "mass": mass, "inertia": 0.0, "skew_id": 0,
        "k1": k, "c1": 0.0,
    })
    m, g = _make_spring_model(p)
    spring.init_group(g, m, MessageLog())

    dtc = spring.forces(g, m.x0, None, None, 1e-4, np.zeros((2, 3)), np.zeros((2, 3)))
    expected_dt = 1.0 / math.sqrt(k / mass)  # 1 / sqrt(25) = 0.2
    np.testing.assert_allclose(dtc[0], expected_dt, rtol=1e-7)


# ============================================================================
# 10. Energy Accounting (Work into eint)
# ============================================================================
def test_energy_accounting_elastic_work():
    """Verify work done by spring force accumulates into internal energy eint."""
    k = 500.0
    p = Property(id=1, type=8, params={
        "mass": 1.0, "inertia": 1.0, "skew_id": 0,
        "k1": k,
    })
    m, g = _make_spring_model(p, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    spring.init_group(g, m, MessageLog())

    # Slowly stretch along X from 0 to delta_max in N steps with constant velocity
    delta_max = 0.2
    n_steps = 100
    dt = 0.001
    v_x = delta_max / (n_steps * dt)
    v = np.array([[0.0, 0.0, 0.0], [v_x, 0.0, 0.0]])
    x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    for step in range(n_steps):
        x[1, 0] += v_x * dt
        spring.forces(g, x, v, None, dt, np.zeros((2, 3)), np.zeros((2, 3)))

    gen = g.state["gen6"]
    # Theoretical elastic strain energy: 0.5 * K * delta_max^2
    e_theoretical = 0.5 * k * (delta_max ** 2)
    # Energy integration is Riemann sum of F * v * dt
    np.testing.assert_allclose(gen["eint"][0], e_theoretical, rtol=0.02)


# ============================================================================
# 11. Mixed-Element Group (TYPE4 + TYPE8 + TYPE13)
# ============================================================================
def test_mixed_spring_group_initialization_and_forces():
    """Verify an ElementGroup mixing TYPE4 axial, TYPE8 general, and TYPE13 spring-beam."""
    p4 = Property(id=1, type=4, params={"mass": 2.0, "k": 100.0, "c": 0.0})
    p8 = Property(id=2, type=8, params={"mass": 4.0, "inertia": 1.0, "skew_id": 0, "k1": 200.0})
    p13 = Property(id=3, type=13, params={"mass": 6.0, "inertia": 2.0, "skew_id": 0, "k1": 300.0})

    m = Model()
    # 6 nodes: [0, 1] for elem 0, [2, 3] for elem 1, [4, 5] for elem 2
    nodes_xyz = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
    ])
    m.add_nodes(np.arange(1, 7), nodes_xyz)

    g = ElementGroup(
        ids=np.array([1, 2, 3]),
        conn=np.array([[0, 1], [2, 3], [4, 5]]),
        part=np.array([0, 1, 2])
    )
    g.state["slices"] = [
        (slice(0, 1), None, p4),
        (slice(1, 2), None, p8),
        (slice(2, 3), None, p13),
    ]

    log = MessageLog()
    node_idx, massn, inertn = spring.init_group(g, m, log)
    assert not log.errors

    # Check mass distribution:
    # Elem 0 (TYPE4): mass 2.0 -> 1.0 each on slots 0, 1
    # Elem 1 (TYPE8): mass 4.0 -> 2.0 each on slots 2, 3
    # Elem 2 (TYPE13): mass 6.0 -> 3.0 each on slots 4, 5
    np.testing.assert_allclose(massn, [1.0, 1.0, 2.0, 2.0, 3.0, 3.0])

    # Check inertia distribution:
    # Elem 0 (TYPE4): 0.0
    # Elem 1 (TYPE8): inertia 1.0 -> 0.5 each on slots 2, 3
    # Elem 2 (TYPE13): inertia 2.0 -> 1.0 each on slots 4, 5
    np.testing.assert_allclose(inertn, [0.0, 0.0, 0.5, 0.5, 1.0, 1.0])

    # Compute forces
    x = m.x0.copy()
    # Stretch all 3 elements along X by 0.1
    x[1, 0] += 0.1
    x[3, 0] += 0.1
    x[5, 0] += 0.1

    fint = np.zeros((6, 3))
    mint = np.zeros((6, 3))
    dtc = spring.forces(g, x, None, None, 1e-4, fint, mint)

    assert len(dtc) == 3
    # Elem 0 force: 100 * 0.1 = 10
    np.testing.assert_allclose(fint[0], [10.0, 0.0, 0.0], rtol=1e-5)
    # Elem 1 force: 200 * 0.1 = 20
    np.testing.assert_allclose(fint[2], [20.0, 0.0, 0.0], rtol=1e-5)
    # Elem 2 force: 300 * 0.1 = 30
    np.testing.assert_allclose(fint[4], [30.0, 0.0, 0.0], rtol=1e-5)


# ============================================================================
# 12. Keyword Parsing & Starter Round-Trip
# ============================================================================
def test_starter_deck_type8_and_type13_parsing(tmp_path: Path):
    """Verify parsing /PROP/TYPE8 and /PROP/TYPE13 cards through parse_starter_deck."""
    deck_text = """\
# OpenRadioss Starter Deck
/BEGIN
Test Deck M502
      2021         0
                  kg                  mm                  ms
                  kg                  mm                  ms
/NODE
         1                 0.0                 0.0                 0.0
         2                 1.0                 0.0                 0.0
/PART/1
Spring Part 1
         1         0         0
/PROP/TYPE8/1
General Spring Prop 8
                 2.5                 1.5         0         0         0         0         0         1
               120.0                12.0                 0.0                 0.0                 0.0
                   0                   0                   0                   0                   0
                 0.0                 0.0                 1.0                 1.0
               240.0                24.0                 0.0                 0.0                 0.0
                   0                   0                   0                   0                   0
                 0.0                 0.0                 1.0                 1.0
               360.0                36.0                 0.0                 0.0                 0.0
                   0                   0                   0                   0                   0
                 0.0                 0.0                 1.0                 1.0
               480.0                48.0                 0.0                 0.0                 0.0
                   0                   0                   0                   0                   0
                 0.0                 0.0                 1.0                 1.0
               600.0                60.0                 0.0                 0.0                 0.0
                   0                   0                   0                   0                   0
                 0.0                 0.0                 1.0                 1.0
               720.0                72.0                 0.0                 0.0                 0.0
                   0                   0                   0                   0                   0
                 0.0                 0.0                 1.0                 1.0
/SPRING/1
         1         1         2
/END
"""
    p = tmp_path / "TEST_0000.rad"
    p.write_text(deck_text, encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert not log.errors

    # Verify property parsed
    prop = model.properties[1]
    assert prop.type == 8
    assert prop.params["mass"] == pytest.approx(2.5)
    assert prop.params["inertia"] == pytest.approx(1.5)
    assert prop.params["iequil"] == 1
    assert prop.params["k1"] == pytest.approx(120.0)
    assert prop.params["c1"] == pytest.approx(12.0)
    assert prop.params["k6"] == pytest.approx(720.0)
    assert prop.params["c6"] == pytest.approx(72.0)

    # Verify build_element_groups wires it correctly
    build_element_groups(model, log)
    assert not log.errors
    spring_group = model.springs
    assert spring_group.n == 1


# ============================================================================
# 13. Defensive Edge Cases
# ============================================================================
def test_defensive_edge_cases():
    """Verify resilience against empty indices, zero dt, zero stiffness, and missing skews."""
    p = Property(id=1, type=8, params={
        "mass": 0.0, "inertia": 0.0, "skew_id": 999, "skew_row": 999,
        "k1": 0.0, "c1": 0.0,
    })
    m, g = _make_spring_model(p)
    log = MessageLog()
    spring.init_group(g, m, log)

    # 1. Empty idx6 call
    dtc_empty = spring_general.forces6(g, m.x0, None, None, 1e-4, None, None, np.array([], dtype=int))
    assert len(dtc_empty) == 0

    # 2. Zero dt call
    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    dtc_zero = spring_general.forces6(g, m.x0, None, None, 0.0, fint, mint, np.array([0]))
    assert dtc_zero[0] == EP30

    # 3. Negative dt clamped safely
    dtc_neg = spring_general.forces6(g, m.x0, None, None, -0.05, fint, mint, np.array([0]))
    assert dtc_neg[0] == EP30

    # 4. None arrays for fint, mint, v, vr
    dtc_none = spring_general.forces6(g, m.x0, None, None, 1e-4, None, None, np.array([0]))
    assert len(dtc_none) == 1


# ============================================================================
# 14. TYPE13 (SPR_BEAM) Deck Parsing & Initialization
# ============================================================================
def test_starter_deck_type13_spr_beam_parsing(tmp_path: Path):
    """Verify parsing /PROP/SPR_BEAM (TYPE13) deck in free format."""
    deck_text = """\
# OpenRadioss Starter Deck
/BEGIN
Test Deck SPR_BEAM M502
      2021         0
                  kg                  mm                  ms
                  kg                  mm                  ms
/NODE
         1                 0.0                 0.0                 0.0
         2                 0.0                 3.0                 4.0
/PART/1
Spring Beam Part 1
         1         0         0
/PROP/SPR_BEAM/1
Spring Beam Prop 13
                 3.0                 0.8         0
               150.0                15.0
                   0
               250.0                25.0
                   0
               350.0                35.0
                   0
               450.0                45.0
                   0
               550.0                55.0
                   0
               650.0                65.0
                   0
/SPRING/1
         1         1         2
/END
"""
    p = tmp_path / "TEST_0000.rad"
    p.write_text(deck_text, encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert not log.errors

    prop = model.properties[1]
    assert prop.type == 13
    assert prop.params["mass"] == pytest.approx(3.0)
    assert prop.params["inertia"] == pytest.approx(0.8)
    assert prop.params["k1"] == pytest.approx(150.0)
    assert prop.params["c1"] == pytest.approx(15.0)
    assert prop.params["k6"] == pytest.approx(650.0)
    assert prop.params["c6"] == pytest.approx(65.0)

    build_element_groups(model, log)
    assert not log.errors
    spring_group = model.springs
    assert spring_group.n == 1


# ============================================================================
# 15. Rotational Time Step Governing (dt_rot < dt_tr)
# ============================================================================
def test_critical_time_step_rotational_governing():
    """Verify that when rotational stiffness/inertia ratio is higher, dt_rot governs."""
    # Translation: mass=10.0, k=100.0 => dt_tr = 1 / sqrt(10) = 0.3162
    # Rotation: inertia=0.01, kr=1000.0 => dt_rot = 1 / sqrt(100000) = 0.003162
    p = Property(id=1, type=8, params={
        "mass": 10.0, "inertia": 0.01, "skew_id": 0,
        "k1": 100.0,
        "k4": 1000.0,
    })
    m, g = _make_spring_model(p)
    spring.init_group(g, m, MessageLog())

    dtc = spring.forces(g, m.x0, None, None, 1e-4, np.zeros((2, 3)), np.zeros((2, 3)))
    expected_dt_rot = 1.0 / math.sqrt(1000.0 / 0.01)
    np.testing.assert_allclose(dtc[0], expected_dt_rot, rtol=1e-6)
    assert dtc[0] < 0.01


# ============================================================================
# 16. Simultaneous 3D Translations, Rotations & Energy Accumulation
# ============================================================================
def test_simultaneous_3d_translations_and_rotations_with_damping():
    """Verify simultaneous translation + rotation with both stiffness and damping."""
    p = Property(id=1, type=8, params={
        "mass": 2.0, "inertia": 1.0, "skew_id": 0,
        "k1": 100.0, "k2": 150.0, "k3": 200.0,
        "c1": 5.0, "c2": 10.0, "c3": 15.0,
        "k4": 300.0, "k5": 400.0, "k6": 500.0,
        "c4": 20.0, "c5": 25.0, "c6": 30.0,
    })
    m, g = _make_spring_model(p, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    spring.init_group(g, m, MessageLog())

    dt = 0.001
    x = np.array([[0.0, 0.0, 0.0], [0.1, -0.2, 0.3]])
    v = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, -1.0]])
    vr = np.array([[0.0, 0.0, 0.0], [-2.0, 1.5, 3.0]])

    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    spring.forces(g, x, v, vr, dt, fint, mint)

    gen = g.state["gen6"]
    # Total translations: delta = (0.1, -0.2, 0.3)
    # v_rel = (1.0, 2.0, -1.0)
    expected_F = np.array([
        100.0 * 0.1 + 5.0 * 1.0,
        150.0 * -0.2 + 10.0 * 2.0,
        200.0 * 0.3 + 15.0 * -1.0,
    ])
    np.testing.assert_allclose(gen["force"][0], expected_F, rtol=1e-6)
    np.testing.assert_allclose(fint[0], expected_F, rtol=1e-6)
    np.testing.assert_allclose(fint[1], -expected_F, rtol=1e-6)

    # Rotations: theta = (-2.0, 1.5, 3.0) * dt
    expected_theta = np.array([-2.0, 1.5, 3.0]) * dt
    expected_M = np.array([
        300.0 * expected_theta[0] + 20.0 * -2.0,
        400.0 * expected_theta[1] + 25.0 * 1.5,
        500.0 * expected_theta[2] + 30.0 * 3.0,
    ])
    np.testing.assert_allclose(gen["moment"][0], expected_M, rtol=1e-6)
    np.testing.assert_allclose(mint[0], expected_M, rtol=1e-6)
    np.testing.assert_allclose(mint[1], -expected_M, rtol=1e-6)

    # Internal energy must be positive (work done by forces + moments)
    assert gen["eint"][0] > 0.0


# ============================================================================
# 17. Rotational Equilibrium under Oblique 3D Displacements
# ============================================================================
def test_rotational_equilibrium_oblique_3d():
    """Verify exact total torque conservation sum(M) + sum(r x F) = 0 for arbitrary 3D geometry."""
    p = Property(id=1, type=8, params={
        "mass": 3.0, "inertia": 2.0, "skew_id": 0, "iequil": 1,
        "k1": 150.0, "k2": 250.0, "k3": 350.0,
        "c1": 10.0, "c2": 20.0, "c3": 30.0,
        "k4": 450.0, "k5": 550.0, "k6": 650.0,
        "c4": 40.0, "c5": 50.0, "c6": 60.0,
    })
    # Oblique nodes
    x0 = np.array([[1.5, -2.5, 3.5], [4.5, 1.5, -0.5]])
    m, g = _make_spring_model(p, x0)
    spring.init_group(g, m, MessageLog())

    # Deformed coordinates with complex displacement & velocities
    x = np.array([[1.2, -2.1, 3.8], [4.8, 1.9, -0.2]])
    v = np.array([[0.3, -0.4, 0.5], [-0.2, 0.6, -0.7]])
    vr = np.array([[1.2, -1.5, 0.8], [-0.9, 1.1, -1.4]])

    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    dt = 1e-4
    spring.forces(g, x, v, vr, dt, fint, mint)

    # 1. Total force must be zero
    np.testing.assert_allclose(fint[0] + fint[1], 0.0, atol=1e-13)

    # 2. Total torque about origin must be zero:
    # T = x0 x f0 + x1 x f1 + m0 + m1 == 0
    total_torque = np.cross(x[0], fint[0]) + np.cross(x[1], fint[1]) + mint[0] + mint[1]
    np.testing.assert_allclose(total_torque, 0.0, atol=1e-12)
