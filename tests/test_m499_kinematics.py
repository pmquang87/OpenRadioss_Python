# pyradioss - Milestone M499: Kinematics, Boundary Conditions & Loads Unit Tests
import pytest
import numpy as np

from pyradioss.model.model import Model
from pyradioss.model.entities import (
    NodeGroup, Surface, BoundaryCondition, Gravity, ConcentratedLoad,
    ImposedVelocity, ImposedDisplacement, PressureLoad
)
from pyradioss.common.tables import FunctTable
from pyradioss.common.messages import MessageLog
from pyradioss.engine.kinematics import LoadsAndConstraints


def test_bcs_global_translation():
    """Verify global translational BCS constraints (X, Y, Z) and zero external work."""
    model = Model()
    model.node_ids = np.arange(3, dtype=np.int64)
    model.mass = np.ones(3, dtype=np.float64)
    model.x0 = np.zeros((3, 3), dtype=np.float64)
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([0], dtype=np.int64))
    model.node_groups[2] = NodeGroup(id=2, node_idx=np.array([1], dtype=np.int64))

    # Node 0 fixed in X (tra 1), Node 1 fixed in Y and Z (tra 2, 3)
    bc1 = BoundaryCondition(id=1, grnod_id=1, fix_tra=[True, False, False], fix_rot=[False, False, False])
    bc2 = BoundaryCondition(id=2, grnod_id=2, fix_tra=[False, True, True], fix_rot=[False, False, False])
    model.bcs = [bc1, bc2]

    log = MessageLog()
    lc = LoadsAndConstraints(model, log)

    v = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)
    vr = np.zeros((3, 3), dtype=np.float64)

    w = lc.apply_kinematic(t=0.01, v=v, vr=vr, mass=model.mass, x=model.x0, dt=0.001)

    # Work for fixed boundary conditions must be exactly zero
    assert w == 0.0
    # Node 0: X zeroed
    assert np.allclose(v[0], [0.0, 2.0, 3.0])
    # Node 1: Y and Z zeroed
    assert np.allclose(v[1], [4.0, 0.0, 0.0])
    # Node 2: free
    assert np.allclose(v[2], [7.0, 8.0, 9.0])


def test_bcs_global_rotation():
    """Verify global rotational BCS constraints on angular velocity vr."""
    model = Model()
    model.node_ids = np.arange(2, dtype=np.int64)
    model.mass = np.ones(2, dtype=np.float64)
    model.x0 = np.zeros((2, 3), dtype=np.float64)
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([0], dtype=np.int64))

    # Node 0 fixed in RX and RZ
    bc = BoundaryCondition(id=1, grnod_id=1, fix_tra=[False, False, False], fix_rot=[True, False, True])
    model.bcs = [bc]

    log = MessageLog()
    lc = LoadsAndConstraints(model, log)

    v = np.ones((2, 3), dtype=np.float64)
    vr = np.array([
        [10.0, 20.0, 30.0],
        [40.0, 50.0, 60.0],
    ], dtype=np.float64)

    w = lc.apply_kinematic(t=0.01, v=v, vr=vr, mass=model.mass, x=model.x0, dt=0.001)
    assert w == 0.0
    assert np.allclose(vr[0], [0.0, 20.0, 0.0])
    assert np.allclose(vr[1], [40.0, 50.0, 60.0])
    # Translational velocity unaffected
    assert np.allclose(v, 1.0)


def test_bcs_skewed_coordinate_system():
    """Verify skewed BCS removing velocity component projected along skew axis."""
    model = Model()
    model.node_ids = np.arange(1, dtype=np.int64)
    model.mass = np.array([1.0], dtype=np.float64)
    model.x0 = np.zeros((1, 3), dtype=np.float64)
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([0], dtype=np.int64))

    # Skew rotated 45 deg around Z:
    # e1 = [1/sqrt(2), 1/sqrt(2), 0], e2 = [-1/sqrt(2), 1/sqrt(2), 0], e3 = [0, 0, 1]
    s2 = np.sqrt(2.0)
    e1 = np.array([1.0 / s2, 1.0 / s2, 0.0])
    e2 = np.array([-1.0 / s2, 1.0 / s2, 0.0])
    e3 = np.array([0.0, 0.0, 1.0])

    class MockSkews:
        axes = {1: np.array([e1, e2, e3])}

    model.skews = MockSkews()

    # Fixed along axis 0 (e1)
    bc = BoundaryCondition(id=1, grnod_id=1, fix_tra=[True, False, False], fix_rot=[False, False, False])
    bc.skew_row = 1
    model.bcs = [bc]

    log = MessageLog()
    lc = LoadsAndConstraints(model, log)

    # Initial velocity has components along e1 and e2:
    # v = 10 * e1 + 5 * e2
    v_init = 10.0 * e1 + 5.0 * e2
    v = v_init.reshape((1, 3)).copy()
    vr = np.zeros((1, 3), dtype=np.float64)

    w = lc.apply_kinematic(t=0.01, v=v, vr=vr, mass=model.mass, x=model.x0, dt=0.001)

    assert w == 0.0
    # Component along e1 must be removed completely:
    assert np.isclose(np.dot(v[0], e1), 0.0, atol=1e-12)
    # Component along e2 must be preserved:
    assert np.isclose(np.dot(v[0], e2), 5.0, atol=1e-12)
    assert np.allclose(v[0], 5.0 * e2, atol=1e-12)


def test_frozen_massless_node_treatment():
    """Verify frozen nodes (mass >= 1e29) auto-fixing and release on driven DOFs."""
    model = Model()
    model.node_ids = np.arange(2, dtype=np.int64)
    model.mass = np.array([1e30, 2.5], dtype=np.float64)
    model.x0 = np.zeros((2, 3), dtype=np.float64)
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([0], dtype=np.int64))

    # Node 0 driven by /IMPVEL in DOF 0 (X)
    fct = FunctTable(fct_id=1, x=[0.0, 1.0], y=[15.0, 15.0])
    model.functions[1] = fct
    imp = ImposedVelocity(id=1, grnod_id=1, dof=0, funct_id=1, scale=1.0, xscale=1.0, tstart=0.0, tstop=1.0)
    model.impvel = [imp]

    log = MessageLog()
    lc = LoadsAndConstraints(model, log)

    # Node 0: DOF 0 was released from auto-fix, DOFs 1, 2 remain fixed
    assert lc.fix_tra[0, 0] is False or lc.fix_tra[0, 0] == 0
    assert lc.fix_tra[0, 1] is True or lc.fix_tra[0, 1] == 1
    assert lc.fix_tra[0, 2] is True or lc.fix_tra[0, 2] == 1

    v = np.zeros((2, 3), dtype=np.float64)
    vr = np.zeros((2, 3), dtype=np.float64)
    w = lc.apply_kinematic(t=0.01, v=v, vr=vr, mass=model.mass, x=model.x0, dt=0.001)

    # Velocity in X imposed to 15.0
    assert np.isclose(v[0, 0], 15.0)
    # Frozen mass must not contribute to work
    assert w == 0.0


def test_gravity_force_accumulation():
    """Verify /GRAV gravity load evaluation ignoring frozen nodes."""
    model = Model()
    model.node_ids = np.arange(2, dtype=np.int64)
    model.mass = np.array([10.0, 1e30], dtype=np.float64)
    model.mass0 = model.mass.copy()
    model.x0 = np.zeros((2, 3), dtype=np.float64)

    fct = FunctTable(fct_id=1, x=[0.0, 1.0], y=[9.81, 9.81])
    model.functions[1] = fct
    # Gravity along -Z
    grav = Gravity(id=1, grnod_id=None, direction=np.array([0.0, 0.0, -1.0]), funct_id=1, scale=1.0)
    model.gravity = [grav]

    log = MessageLog()
    lc = LoadsAndConstraints(model, log)

    fext = np.zeros((2, 3), dtype=np.float64)
    lc.external_forces(t=0.1, fext=fext, x=model.x0)

    # Node 0: Fz = 10 * 9.81 * (-1) = -98.1
    assert np.allclose(fext[0], [0.0, 0.0, -98.1])
    # Node 1 (frozen): 0 gravity force
    assert np.allclose(fext[1], [0.0, 0.0, 0.0])


def test_cload_forces_and_sensors():
    """Verify /CLOAD concentrated forces with sensor gating and time scaling."""
    model = Model()
    model.node_ids = np.arange(1, dtype=np.int64)
    model.mass = np.array([5.0], dtype=np.float64)
    model.x0 = np.zeros((1, 3), dtype=np.float64)
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([0], dtype=np.int64))

    # Function: F(t) = 100 * t
    fct = FunctTable(fct_id=1, x=[0.0, 1.0], y=[0.0, 100.0])
    model.functions[1] = fct

    # /CLOAD with sens_id=10, time_scale=2.0, scale=1.5 along +X
    cload = ConcentratedLoad(
        id=1, grnod_id=1, direction=np.array([1.0, 0.0, 0.0]),
        funct_id=1, scale=1.5, sens_id=10, time_scale=2.0
    )
    model.cloads = [cload]

    log = MessageLog()
    lc = LoadsAndConstraints(model, log)

    class MockSensors:
        def shifted_time(self, sens_id, t):
            # Sensor fires at t=0.04
            if t < 0.04:
                return None
            return t - 0.04

    sensors = MockSensors()

    # Before sensor fires (t = 0.02)
    fext = np.zeros((1, 3), dtype=np.float64)
    lc.external_forces(t=0.02, fext=fext, x=model.x0, sensors=sensors)
    assert np.allclose(fext[0], [0.0, 0.0, 0.0])

    # After sensor fires (t = 0.06): shifted time = 0.06 - 0.04 = 0.02
    # with time_scale=2.0: te = 0.02 / 2.0 = 0.01
    # fct.eval(0.01) = 1.0, F = 1.5 * 1.0 = 1.5 in +X
    fext = np.zeros((1, 3), dtype=np.float64)
    lc.external_forces(t=0.06, fext=fext, x=model.x0, sensors=sensors)
    assert np.isclose(fext[0, 0], 1.5)


def test_pload_quad_follower_pressure():
    """Verify /PLOAD follower pressure on 4-node quad segment with structure rotation."""
    model = Model()
    model.node_ids = np.arange(4, dtype=np.int64)
    model.mass = np.ones(4, dtype=np.float64)

    # Flat quad in XY plane: width=2, height=3, area=6.0, normal=+Z
    x0 = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 3.0, 0.0],
        [0.0, 3.0, 0.0],
    ], dtype=np.float64)
    model.x0 = x0

    surf = Surface(id=1, segments=np.array([[0, 1, 2, 3]], dtype=np.int64))
    model.surfaces[1] = surf

    fct = FunctTable(fct_id=1, x=[0.0, 1.0], y=[10.0, 10.0])
    model.functions[1] = fct

    pload = PressureLoad(id=1, surf_id=1, funct_id=1, scale=1.0)
    model.ploads = [pload]

    log = MessageLog()
    lc = LoadsAndConstraints(model, log)

    fext = np.zeros((4, 3), dtype=np.float64)
    lc.external_forces(t=0.01, fext=fext, x=x0)

    # Total area vector = [0, 0, 6], pressure = 10 -> Total force = [0, 0, 60]
    # Each corner receives 1/4 -> 15.0 in +Z
    assert np.allclose(fext[:, 0], 0.0)
    assert np.allclose(fext[:, 1], 0.0)
    assert np.allclose(fext[:, 2], 15.0)
    assert np.isclose(np.sum(fext[:, 2]), 60.0)

    # Now rotate the quad 90 degrees around X axis: normal points in -Y
    x_rot = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 0.0, 3.0],
        [0.0, 0.0, 3.0],
    ], dtype=np.float64)

    fext_rot = np.zeros((4, 3), dtype=np.float64)
    lc.external_forces(t=0.01, fext=fext_rot, x=x_rot)

    # Pressure must follow the surface orientation! Force is in -Y direction
    assert np.allclose(fext_rot[:, 0], 0.0)
    assert np.allclose(fext_rot[:, 1], -15.0)
    assert np.allclose(fext_rot[:, 2], 0.0)
    assert np.isclose(np.sum(fext_rot[:, 1]), -60.0)


def test_pload_triangle_segment():
    """Verify /PLOAD on 3-node triangular segment with 1/3 corner weight lumping."""
    model = Model()
    model.node_ids = np.arange(3, dtype=np.int64)
    model.mass = np.ones(3, dtype=np.float64)

    # Right triangle in XY plane: (0,0,0), (2,0,0), (0,2,0). Area = 0.5 * 2 * 2 = 2.0
    x0 = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ], dtype=np.float64)
    model.x0 = x0

    # Triangle segment with n4 = n3 (convention)
    surf = Surface(id=2, segments=np.array([[0, 1, 2, 2]], dtype=np.int64))
    model.surfaces[2] = surf

    fct = FunctTable(fct_id=2, x=[0.0, 1.0], y=[6.0, 6.0])
    model.functions[2] = fct

    pload = PressureLoad(id=2, surf_id=2, funct_id=2, scale=1.0)
    model.ploads = [pload]

    log = MessageLog()
    lc = LoadsAndConstraints(model, log)

    fext = np.zeros((3, 3), dtype=np.float64)
    lc.external_forces(t=0.01, fext=fext, x=x0)

    # Area vector = [0, 0, 2.0], pressure = 6.0 -> Total force = [0, 0, 12.0]
    # Each corner receives 1/3 -> 4.0 in +Z
    assert np.allclose(fext[:, 0], 0.0)
    assert np.allclose(fext[:, 1], 0.0)
    assert np.allclose(fext[:, 2], 4.0)
    assert np.isclose(np.sum(fext[:, 2]), 12.0)


def test_impvel_translation_and_time_window():
    """Verify translational /IMPVEL enforcement and [tstart, tstop] gating."""
    model = Model()
    model.node_ids = np.arange(1, dtype=np.int64)
    model.mass = np.array([5.0], dtype=np.float64)
    model.x0 = np.zeros((1, 3), dtype=np.float64)
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([0], dtype=np.int64))

    fct = FunctTable(fct_id=1, x=[0.0, 1.0], y=[20.0, 20.0])
    model.functions[1] = fct

    # Active inside [0.01, 0.05]
    imp = ImposedVelocity(
        id=1, grnod_id=1, dof=0, funct_id=1, scale=1.0, xscale=1.0,
        tstart=0.01, tstop=0.05
    )
    model.impvel = [imp]

    log = MessageLog()
    lc = LoadsAndConstraints(model, log)

    vr = np.zeros((1, 3), dtype=np.float64)

    # 1. Before window (t = 0.005): free
    v = np.array([[2.0, 0.0, 0.0]], dtype=np.float64)
    w1 = lc.apply_kinematic(t=0.005, v=v, vr=vr, mass=model.mass, x=model.x0, dt=0.001)
    assert np.isclose(v[0, 0], 2.0)
    assert w1 == 0.0

    # 2. Inside window (t = 0.02): enforced to 20.0
    v = np.array([[2.0, 0.0, 0.0]], dtype=np.float64)
    w2 = lc.apply_kinematic(t=0.02, v=v, vr=vr, mass=model.mass, x=model.x0, dt=0.001)
    assert np.isclose(v[0, 0], 20.0)
    assert w2 > 0.0

    # 3. After window (t = 0.06): free
    v = np.array([[12.0, 0.0, 0.0]], dtype=np.float64)
    w3 = lc.apply_kinematic(t=0.06, v=v, vr=vr, mass=model.mass, x=model.x0, dt=0.001)
    assert np.isclose(v[0, 0], 12.0)
    assert w3 == 0.0


def test_impvel_rotational():
    """Verify rotational /IMPVEL updating angular velocity vr against rotational inertia."""
    model = Model()
    model.node_ids = np.arange(1, dtype=np.int64)
    model.mass = np.array([10.0], dtype=np.float64)
    model.x0 = np.zeros((1, 3), dtype=np.float64)
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([0], dtype=np.int64))

    fct = FunctTable(fct_id=1, x=[0.0, 1.0], y=[50.0, 50.0])
    model.functions[1] = fct

    # dof=4 corresponds to rotational YY (dof - 3 = 1)
    imp = ImposedVelocity(
        id=1, grnod_id=1, dof=4, funct_id=1, scale=1.0, xscale=1.0,
        tstart=0.0, tstop=1.0
    )
    model.impvel = [imp]

    log = MessageLog()
    lc = LoadsAndConstraints(model, log)

    v = np.zeros((1, 3), dtype=np.float64)
    vr = np.array([[0.0, 5.0, 0.0]], dtype=np.float64)
    inertia = np.array([2.0], dtype=np.float64)

    w = lc.apply_kinematic(t=0.01, v=v, vr=vr, mass=model.mass, x=model.x0, dt=0.001, inertia=inertia)

    # vr[0, 1] updated to 50.0
    assert np.isclose(vr[0, 1], 50.0)
    # v is unaffected
    assert np.allclose(v, 0.0)


def test_impdisp_exact_landing():
    """Verify /IMPDISP exact target position landing: v = (x0 + d - x) / dt."""
    model = Model()
    model.node_ids = np.arange(1, dtype=np.int64)
    model.mass = np.array([1.0], dtype=np.float64)
    # Initial position x0 = 10.0 in X
    model.x0 = np.array([[10.0, 0.0, 0.0]], dtype=np.float64)
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([0], dtype=np.int64))

    # Target displacement function: d(t) = 2.0
    fct = FunctTable(fct_id=1, x=[0.0, 1.0], y=[2.0, 2.0])
    model.functions[1] = fct

    imp = ImposedDisplacement(
        id=1, grnod_id=1, dof=0, funct_id=1, scale=1.0, xscale=1.0,
        tstart=0.0, tstop=1.0
    )
    model.impdisp = [imp]

    log = MessageLog()
    lc = LoadsAndConstraints(model, log)

    # Current position drifted slightly: x = 11.8 (target is x0 + d = 10.0 + 2.0 = 12.0)
    x_curr = np.array([[11.8, 0.0, 0.0]], dtype=np.float64)
    dt = 0.01
    v = np.zeros((1, 3), dtype=np.float64)
    vr = np.zeros((1, 3), dtype=np.float64)

    lc.apply_kinematic(t=0.01, v=v, vr=vr, mass=model.mass, x=x_curr, dt=dt)

    # Required velocity to land on 12.0 is (12.0 - 11.8) / 0.01 = 20.0
    assert np.isclose(v[0, 0], 20.0)
    # Target position after dt is exactly 12.0
    x_next = x_curr[0, 0] + v[0, 0] * dt
    assert np.isclose(x_next, 12.0)


def test_skewed_impvel():
    """Verify skewed /IMPVEL imposing velocity strictly along skew unit axis."""
    model = Model()
    model.node_ids = np.arange(1, dtype=np.int64)
    model.mass = np.array([1.0], dtype=np.float64)
    model.x0 = np.zeros((1, 3), dtype=np.float64)
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([0], dtype=np.int64))

    s2 = np.sqrt(2.0)
    e1 = np.array([1.0 / s2, 1.0 / s2, 0.0])
    e2 = np.array([-1.0 / s2, 1.0 / s2, 0.0])
    e3 = np.array([0.0, 0.0, 1.0])

    class MockSkews:
        axes = {1: np.array([e1, e2, e3])}

    model.skews = MockSkews()

    fct = FunctTable(fct_id=1, x=[0.0, 1.0], y=[10.0, 10.0])
    model.functions[1] = fct

    # Impose along axis 0 (e1)
    imp = ImposedVelocity(
        id=1, grnod_id=1, dof=0, funct_id=1, scale=1.0, xscale=1.0,
        tstart=0.0, tstop=1.0
    )
    imp.skew_row = 1
    model.impvel = [imp]

    log = MessageLog()
    lc = LoadsAndConstraints(model, log)

    # Initial velocity has transverse motion 7.0 along e2
    v = (7.0 * e2).reshape((1, 3)).copy()
    vr = np.zeros((1, 3), dtype=np.float64)

    lc.apply_kinematic(t=0.01, v=v, vr=vr, mass=model.mass, x=model.x0, dt=0.001)

    # Velocity along e1 is now 10.0
    assert np.isclose(np.dot(v[0], e1), 10.0)
    # Velocity along e2 remains 7.0
    assert np.isclose(np.dot(v[0], e2), 7.0)


def test_work_accounting_impulsive_start():
    """Verify that leapfrog midstep work booking matches Delta KE at impulsive start."""
    model = Model()
    model.node_ids = np.arange(1, dtype=np.int64)
    m = 8.0
    model.mass = np.array([m], dtype=np.float64)
    model.x0 = np.zeros((1, 3), dtype=np.float64)
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([0], dtype=np.int64))

    v_imp = 25.0
    fct = FunctTable(fct_id=1, x=[0.0, 1.0], y=[v_imp, v_imp])
    model.functions[1] = fct

    imp = ImposedVelocity(
        id=1, grnod_id=1, dof=0, funct_id=1, scale=1.0, xscale=1.0,
        tstart=0.0, tstop=1.0
    )
    model.impvel = [imp]

    log = MessageLog()
    lc = LoadsAndConstraints(model, log)

    # Node starts at rest
    v = np.zeros((1, 3), dtype=np.float64)
    v_old = np.zeros((1, 3), dtype=np.float64)
    vr = np.zeros((1, 3), dtype=np.float64)

    w = lc.apply_kinematic(t=0.001, v=v, vr=vr, mass=model.mass, x=model.x0, dt=0.001, v_old=v_old)

    # Kinetic energy acquired: KE = 0.5 * m * v_imp^2
    ke_expected = 0.5 * m * (v_imp ** 2)
    # Work booked by constraint must match KE identically (no 50% impulsive error!)
    assert np.isclose(w, ke_expected)


def test_work_accounting_opposing_force():
    """Verify that work booked against opposing force equals F_opp * v * dt."""
    model = Model()
    model.node_ids = np.arange(1, dtype=np.int64)
    m = 10.0
    model.mass = np.array([m], dtype=np.float64)
    model.x0 = np.zeros((1, 3), dtype=np.float64)
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([0], dtype=np.int64))

    v_const = 20.0
    fct = FunctTable(fct_id=1, x=[0.0, 1.0], y=[v_const, v_const])
    model.functions[1] = fct

    imp = ImposedVelocity(
        id=1, grnod_id=1, dof=0, funct_id=1, scale=1.0, xscale=1.0,
        tstart=0.0, tstop=1.0
    )
    model.impvel = [imp]

    log = MessageLog()
    lc = LoadsAndConstraints(model, log)

    # Node was moving at v_const before the step
    v_old = np.array([[v_const, 0.0, 0.0]], dtype=np.float64)
    # Opposing external force F_ext = -100.0 slowed down free velocity:
    # a = F / m = -10.0, Delta v = a * dt = -0.01 for dt=0.001
    dt = 0.001
    F_opp = 100.0
    v_free = np.array([[v_const - (F_opp / m) * dt, 0.0, 0.0]], dtype=np.float64)
    vr = np.zeros((1, 3), dtype=np.float64)

    w = lc.apply_kinematic(t=0.01, v=v_free, vr=vr, mass=model.mass, x=model.x0, dt=dt, v_old=v_old)

    # Work done by the kinematic constraint must equal F_opp * v_const * dt
    w_expected = F_opp * v_const * dt
    assert np.isclose(w, w_expected)


def test_defensive_missing_entities():
    """Verify that missing node groups, surfaces, and functions log warnings and don't crash."""
    model = Model()
    model.node_ids = np.arange(2, dtype=np.int64)
    model.mass = np.ones(2, dtype=np.float64)
    model.x0 = np.zeros((2, 3), dtype=np.float64)

    # Referencing non-existent groups and functions
    bc = BoundaryCondition(id=1, grnod_id=999, fix_tra=[True, True, True], fix_rot=[True, True, True])
    model.bcs = [bc]

    grav = Gravity(id=1, grnod_id=888, direction=np.array([0, 0, -1]), funct_id=777)
    model.gravity = [grav]

    cload = ConcentratedLoad(id=1, grnod_id=666, direction=np.array([1, 0, 0]), funct_id=555)
    model.cloads = [cload]

    imp = ImposedVelocity(id=1, grnod_id=444, dof=0, funct_id=333)
    model.impvel = [imp]

    pload = PressureLoad(id=1, surf_id=222, funct_id=111)
    model.ploads = [pload]

    log = MessageLog()
    # Must initialize cleanly without KeyError
    lc = LoadsAndConstraints(model, log)

    fext = np.zeros((2, 3), dtype=np.float64)
    lc.external_forces(t=0.01, fext=fext, x=model.x0)

    v = np.zeros((2, 3), dtype=np.float64)
    vr = np.zeros((2, 3), dtype=np.float64)
    w = lc.apply_kinematic(t=0.01, v=v, vr=vr, mass=model.mass, x=model.x0, dt=0.001)

    assert w == 0.0
    assert len(log.warnings) > 0
