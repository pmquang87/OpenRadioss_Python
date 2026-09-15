"""tests/test_m500_rigid_body.py — Comprehensive unit tests for /RBODY and /RBE2
rigid body mechanics, kinematics, boundary conditions, and hardening (M500).

Fortran origin:
  engine/source/constraints/general/rbody/rbyfor.F  (force/moment gather)
  engine/source/constraints/general/rbody/rbycor.F  (6-DOF Newton-Euler & velocity scatter)
  engine/source/constraints/general/rbody/rbyact.F  (activation & sensor bookkeeping)
  starter/source/constraints/general/rbody/inirby.F (mass, COG, inertia tensor)
"""

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.engine.kinematics import LoadsAndConstraints
from pyradioss.engine.rigid_body import (
    RigidBodyEngine,
    _exp_rotation,
    _orthonormalize,
    _skew,
    build_rigid_bodies,
)
from pyradioss.model.entities import (
    BoundaryCondition,
    ImposedDisplacement,
    ImposedVelocity,
    NodeGroup,
    RigidBody,
)
from pyradioss.model.model import Model
from pyradioss.model.skew import SkewSet


class _DummyFunc:
    def __init__(self, val: float = 1.0, slope: float = 0.0):
        self.val = float(val)
        self.slope = float(slope)

    def eval(self, t: float) -> float:
        return self.val + self.slope * float(t)


def _setup_basic_rbody(n_slaves: int = 4, master_pos=None, slave_radius: float = 1.0,
                       mass_per_node: float = 2.0):
    """Create a minimal clean model and loads for testing RigidBodyEngine."""
    model = Model()
    num_nodes = 1 + n_slaves
    model.node_ids = np.arange(1, num_nodes + 1, dtype=np.int64)

    x0 = np.zeros((num_nodes, 3), dtype=np.float64)
    if master_pos is not None:
        x0[0] = master_pos

    # Distribute slaves evenly on a ring in XY plane around master
    slaves = np.arange(1, num_nodes, dtype=np.int64)
    for i, s in enumerate(slaves):
        theta = 2.0 * np.pi * i / n_slaves
        x0[s] = x0[0] + np.array([slave_radius * np.cos(theta), slave_radius * np.sin(theta), 0.0])

    model.x0 = x0
    model.x = x0.copy()
    model.v = np.zeros((num_nodes, 3), dtype=np.float64)
    model.vr = np.zeros((num_nodes, 3), dtype=np.float64)
    model.mass = np.full(num_nodes, mass_per_node, dtype=np.float64)
    model.mass0 = model.mass.copy()
    model.inertia = np.full(num_nodes, 0.1, dtype=np.float64)

    # Compute exact inertia tensor about master/COG
    r = x0[slaves] - x0[0]
    r2 = np.einsum("nb,nb->n", r, r)
    J0 = np.eye(3) * float((mass_per_node * r2).sum()) - np.einsum("n,nb,nc->bc", np.full(n_slaves, mass_per_node), r, r)
    total_mass = float(model.mass.sum())

    rb = RigidBody(
        id=1,
        master_id=1,
        grnod_id=10,
        master=0,
        slaves=slaves,
        mass_total=total_mass,
        xg=x0[0].copy(),
        J=J0,
        kind="RBODY",
    )
    model.rbodies = [rb]

    grp = NodeGroup(id=10, node_ids=model.node_ids[slaves], node_idx=slaves)
    model.node_groups[10] = grp

    log = MessageLog()
    loads = LoadsAndConstraints(model, log)
    return model, rb, loads, log


# --------------------------------------------------------------------------
# 1. Rodrigues exponential map properties
# --------------------------------------------------------------------------
def test_rodrigues_exponential_map_properties():
    """Verify exact SO(3) mathematical properties of _exp_rotation and _orthonormalize."""
    w = np.array([2.0, -3.0, 6.0], dtype=np.float64)  # norm = 7.0
    dt = 0.05
    theta = 7.0 * 0.05  # 0.35 rad

    R = _exp_rotation(w, dt)

    # Orthogonality: R @ R.T == I
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-14)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-14)

    # Unit determinant: det(R) == 1
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-14)

    # Trace formula: tr(R) = 1 + 2*cos(theta)
    assert np.isclose(np.trace(R), 1.0 + 2.0 * np.cos(theta), atol=1e-14)

    # Axis invariance: R @ w == w
    assert np.allclose(R @ w, w, atol=1e-14)

    # Small-angle limit: theta < 1e-12 uses Taylor expansion
    w_tiny = np.array([1e-13, 0.0, 0.0])
    R_tiny = _exp_rotation(w_tiny, 1.0)
    assert np.allclose(R_tiny, np.eye(3) + _skew(w_tiny), atol=1e-15)

    # Polar projection via SVD
    R_perturbed = R + 1e-5 * np.ones((3, 3))
    R_ortho = _orthonormalize(R_perturbed)
    assert np.allclose(R_ortho @ R_ortho.T, np.eye(3), atol=1e-14)


# --------------------------------------------------------------------------
# 2. Rigid velocity scatter and zero strain rate
# --------------------------------------------------------------------------
def test_rigid_velocity_scatter_and_zero_strain():
    """Verify linear rigid velocity field v_i = v_ref + w x r_i and zero strain rate."""
    model, rb, loads, log = _setup_basic_rbody(n_slaves=4, master_pos=np.array([1.0, 2.0, 3.0]))
    rbe = RigidBodyEngine(rb, model, loads, log)

    v_ref = np.array([2.5, -1.0, 4.0])
    w = np.array([0.5, 1.2, -3.0])
    rbe.v_ref = v_ref.copy()
    rbe.w = w.copy()

    # Scatter rigid velocity field
    v_nodes = rbe._rigid_field(model.x[rbe.nodes])

    for i, n in enumerate(rbe.nodes):
        r_n = model.x[n] - rbe.x_ref
        v_expect = v_ref + np.cross(w, r_n)
        assert np.allclose(v_nodes[i], v_expect, atol=1e-14)

    # Zero strain-rate identity: for all pairs (i, j), (v_i - v_j) . (x_i - x_j) == 0
    for i in range(len(rbe.nodes)):
        for j in range(i + 1, len(rbe.nodes)):
            dv = v_nodes[i] - v_nodes[j]
            dx = model.x[rbe.nodes[i]] - model.x[rbe.nodes[j]]
            assert np.isclose(np.dot(dv, dx), 0.0, atol=1e-13)


# --------------------------------------------------------------------------
# 3. Discrete power and virtual work identity
# --------------------------------------------------------------------------
def test_discrete_power_and_virtual_work_identity():
    """Verify that rigid body constraints satisfy sum f_i . v_i == F . v_ref + T . w."""
    model, rb, loads, log = _setup_basic_rbody(n_slaves=4)
    rbe = RigidBodyEngine(rb, model, loads, log)

    # Prescribe arbitrary 3D external, internal, and contact forces and moments
    N = model.numnod
    np.random.seed(42)
    fint = np.random.randn(N, 3) * 100.0
    fext = np.random.randn(N, 3) * 50.0
    fcont = np.random.randn(N, 3) * 30.0
    mint = np.random.randn(N, 3) * 10.0

    v = np.zeros((N, 3))
    vr = np.zeros((N, 3))
    dt = 0.001

    rbe.advance(fint, fext, fcont, mint, v, vr, model.x, dt, t_next=dt)

    # Calculate discrete power from nodal perspective:
    f_total = fint[rbe.nodes] + fext[rbe.nodes] + fcont[rbe.nodes]
    p_nodal = np.sum(f_total * v[rbe.nodes]) + np.sum(mint[rbe.nodes] * vr[rbe.nodes])

    # Calculate discrete power from resultant body perspective:
    p_body = np.dot(rbe.f_res, rbe.v_ref) + np.dot(rbe.m_res, rbe.w)

    assert np.isclose(p_nodal, p_body, atol=1e-10)


# --------------------------------------------------------------------------
# 4. Torque-free angular momentum and energy conservation
# --------------------------------------------------------------------------
def test_free_flight_angular_momentum_conservation():
    """Verify exact conservation of L and kinetic energy in torque-free flight."""
    model, rb, loads, log = _setup_basic_rbody(n_slaves=4)
    # Set non-isotropic inertia tensor
    rb.J = np.diag([20.0, 30.0, 50.0])
    rbe = RigidBodyEngine(rb, model, loads, log)

    # Give initial angular momentum
    L0 = np.array([12.0, -8.0, 15.0])
    rbe.L = L0.copy()
    rbe.w = np.linalg.solve(rbe.J0, rbe.L)
    rbe.v_ref = np.array([1.0, -2.0, 3.0])

    dt = 0.002
    N = model.numnod
    f_zero = np.zeros((N, 3))
    m_zero = np.zeros((N, 3))

    initial_ke = 0.5 * rbe.M * np.dot(rbe.v_ref, rbe.v_ref) + 0.5 * np.dot(rbe.w, rbe.L)

    for cycle in range(20):
        t_next = (cycle + 1) * dt
        rbe.advance(f_zero, f_zero, f_zero, m_zero, model.v, model.vr, model.x, dt, t_next)
        rbe.enforce(model.x, model.v, dt)

        # Angular momentum L must remain constant to machine precision
        assert np.allclose(rbe.L, L0, atol=1e-13)

        # Kinetic energy must remain constant
        current_ke = 0.5 * rbe.M * np.dot(rbe.v_ref, rbe.v_ref) + 0.5 * np.dot(rbe.w, rbe.L)
        assert np.isclose(current_ke, initial_ke, atol=1e-12)


# --------------------------------------------------------------------------
# 5. Pivoted physical pendulum & Huygens-Steiner transport
# --------------------------------------------------------------------------
def test_pivoted_physical_pendulum_huygens_steiner():
    """Verify parallel-axis theorem transport and pivot mode kinematics."""
    model, rb, loads, log = _setup_basic_rbody(n_slaves=4, master_pos=np.array([0.0, 0.0, 0.0]))
    # Offset COG relative to master
    rb.xg = np.array([0.0, 0.0, -2.0])

    # Fix all translations on master node (turn into pivot)
    bc = BoundaryCondition(id=1, grnod_id=1, fix_tra=[True, True, True], fix_rot=[False, False, False])
    model.bcs = [bc]
    model.node_groups[1] = NodeGroup(id=1, node_ids=np.array([1]), node_idx=np.array([0]))

    loads_pivot = LoadsAndConstraints(model, log)
    rbe = RigidBodyEngine(rb, model, loads_pivot, log)

    assert rbe.pivot is True
    assert np.allclose(rbe.v_ref, np.zeros(3))
    assert np.allclose(rbe.x_ref, model.x0[0])

    # Check Huygens-Steiner transport: J_pivot = J_cog + M*(|c|^2 I - c outer c)
    c = rb.xg - model.x0[0]
    J_expected = rb.J + rbe.M * (np.eye(3) * float(c @ c) - np.outer(c, c))
    assert np.allclose(rbe.J0, J_expected, atol=1e-14)

    # Advance under a net force: v_ref must remain strictly zero
    N = model.numnod
    fext = np.zeros((N, 3))
    fext[1] = np.array([0.0, 0.0, 100.0])  # Force along Z creates torque about Y
    rbe.advance(np.zeros((N, 3)), fext, np.zeros((N, 3)), np.zeros((N, 3)),
                model.v, model.vr, model.x, dt=0.001, t_next=0.001)

    assert np.allclose(rbe.v_ref, np.zeros(3))
    assert rbe.w[1] != 0.0  # Torque about Y induces rotation


# --------------------------------------------------------------------------
# 6. Global translational and rotational BCS
# --------------------------------------------------------------------------
def test_global_translational_and_rotational_bcs():
    """Verify that /BCS on master clamps specific translation and rotation DOFs."""
    model, rb, loads, log = _setup_basic_rbody(n_slaves=4)
    # Fix translation along X and rotation about Z
    bc = BoundaryCondition(id=1, grnod_id=1, fix_tra=[True, False, False], fix_rot=[False, False, True])
    model.bcs = [bc]
    model.node_groups[1] = NodeGroup(id=1, node_ids=np.array([1]), node_idx=np.array([0]))

    loads_bc = LoadsAndConstraints(model, log)
    rbe = RigidBodyEngine(rb, model, loads_bc, log)

    assert bool(rbe.fix_tra[0]) is True
    assert bool(rbe.fix_tra[1]) is False
    assert bool(rbe.fix_rot[2]) is True
    assert bool(rbe.fix_rot[0]) is False

    # Apply forces and torques along both free and constrained directions
    N = model.numnod
    fext = np.zeros((N, 3))
    fext[0] = np.array([500.0, 200.0, 0.0])
    mint = np.zeros((N, 3))
    mint[0] = np.array([10.0, 0.0, 50.0])

    rbe.advance(np.zeros((N, 3)), fext, np.zeros((N, 3)), mint,
                model.v, model.vr, model.x, dt=0.001, t_next=0.001)

    # Constrained DOFs must remain zero
    assert rbe.v_ref[0] == 0.0
    assert rbe.v_ref[1] > 0.0
    assert rbe.w[2] == 0.0
    assert rbe.w[0] != 0.0


# --------------------------------------------------------------------------
# 7. Skewed boundary conditions projection
# --------------------------------------------------------------------------
def test_skewed_bcs_projection():
    """Verify that skewed /BCS projects out motion along arbitrary 3D local coordinate axes."""
    model, rb, loads, log = _setup_basic_rbody(n_slaves=4)

    # Create a skew coordinate system rotated 45 degrees around Z
    skews = SkewSet()
    theta = np.pi / 4.0
    c, s = np.cos(theta), np.sin(theta)
    axes_45 = np.array([
        [c, s, 0.0],
        [-s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    skews.axes = [np.eye(3), axes_45]
    skews.origins = [np.zeros(3), np.zeros(3)]
    skews.is_mov = [False, False]
    model.skews = skews

    # Constrain local X axis in translation
    bc = BoundaryCondition(id=1, grnod_id=1, fix_tra=[True, False, False], fix_rot=[False, False, False], skew_row=1)
    model.bcs = [bc]
    model.node_groups[1] = NodeGroup(id=1, node_ids=np.array([1]), node_idx=np.array([0]))

    loads_skew = LoadsAndConstraints(model, log)
    rbe = RigidBodyEngine(rb, model, loads_skew, log)

    # Set velocity with components along both local axes
    e_local_x = axes_45[0]
    e_local_y = axes_45[1]
    rbe.v_ref = 10.0 * e_local_x + 5.0 * e_local_y

    rbe._apply_skew_bcs(rbe.v_ref, rbe.w)

    # Local X component should be exactly 0, local Y component preserved
    assert np.isclose(np.dot(rbe.v_ref, e_local_x), 0.0, atol=1e-14)
    assert np.isclose(np.dot(rbe.v_ref, e_local_y), 5.0, atol=1e-14)


# --------------------------------------------------------------------------
# 8. Translational /IMPVEL drive and work booking
# --------------------------------------------------------------------------
def test_translational_impvel_and_work_booking():
    """Verify master translational /IMPVEL enforces velocity and books external work."""
    model, rb, loads, log = _setup_basic_rbody(n_slaves=4)
    model.functions[100] = _DummyFunc(val=25.0)

    # Impose velocity on master along Y (dof 1)
    imp = ImposedVelocity(id=1, grnod_id=1, dof=1, funct_id=100, scale=1.0)
    model.impvel = [imp]
    model.node_groups[1] = NodeGroup(id=1, node_ids=np.array([1]), node_idx=np.array([0]))

    loads_imp = LoadsAndConstraints(model, log)
    rbe = RigidBodyEngine(rb, model, loads_imp, log)

    assert len(rbe.drives) == 1
    assert rbe.drives[0][0] == 1

    N = model.numnod
    dt = 0.001
    wext = rbe.advance(np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3)),
                       model.v, model.vr, model.x, dt=dt, t_next=dt)

    assert np.isclose(rbe.v_ref[1], 25.0, atol=1e-14)
    # External work = M * (v_imp - v_free) * v_imp = M * 25.0 * 25.0
    expected_work = rbe.M * 25.0 * 25.0
    assert np.isclose(wext, expected_work, atol=1e-10)


# --------------------------------------------------------------------------
# 9. Rotational /IMPVEL drive and spin work booking
# --------------------------------------------------------------------------
def test_rotational_impvel_and_spin_work_booking():
    """Verify rotational /IMPVEL enforces spin and books rotational leapfrog work."""
    model, rb, loads, log = _setup_basic_rbody(n_slaves=4)
    model.functions[200] = _DummyFunc(val=10.0)

    # Impose angular velocity about Z (dof 5 in Radioss, rdof = 2)
    imp = ImposedVelocity(id=1, grnod_id=1, dof=5, funct_id=200, scale=1.0)
    model.impvel = [imp]
    model.node_groups[1] = NodeGroup(id=1, node_ids=np.array([1]), node_idx=np.array([0]))

    loads_imp = LoadsAndConstraints(model, log)
    rbe = RigidBodyEngine(rb, model, loads_imp, log)

    assert len(rbe.rot_drives) == 1
    assert rbe.rot_drives[0][0] == 2  # rdof 2 corresponds to dof 5 (Z)

    N = model.numnod
    dt = 0.001
    wext = rbe.advance(np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3)),
                       model.v, model.vr, model.x, dt=dt, t_next=dt)

    assert np.isclose(rbe.w[2], 10.0, atol=1e-14)
    # Work booking: dL . (w_old + w_end)/2 = (J_zz * 10) * (0 + 10)/2 = 0.5 * J_zz * 100
    expected_work = 0.5 * rbe.J0[2, 2] * 100.0
    assert np.isclose(wext, expected_work, atol=1e-10)


# --------------------------------------------------------------------------
# 10. Translational and rotational /IMPDISP exact target landing
# --------------------------------------------------------------------------
def test_translational_and_rotational_impdisp():
    """Verify that /IMPDISP drives master position to x0 + d(t) without integration drift."""
    model, rb, loads, log = _setup_basic_rbody(n_slaves=4)
    model.functions[300] = _DummyFunc(val=0.0, slope=10.0)  # d(t) = 10 * t

    # Impose displacement on master along X (dof 0)
    imp_disp = ImposedDisplacement(id=1, grnod_id=1, dof=0, funct_id=300, scale=1.0)
    model.impdisp = [imp_disp]
    model.node_groups[1] = NodeGroup(id=1, node_ids=np.array([1]), node_idx=np.array([0]))

    loads_imp = LoadsAndConstraints(model, log)
    rbe = RigidBodyEngine(rb, model, loads_imp, log)

    N = model.numnod
    dt = 0.01
    t = dt
    rbe.advance(np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3)),
                model.v, model.vr, model.x, dt=dt, t_next=t)
    rbe.enforce(model.x, model.v, dt)

    # Master node must have moved by exactly 10.0 * 0.01 = 0.1
    expected_x = model.x0[0, 0] + 0.1
    assert np.isclose(model.x[0, 0], expected_x, atol=1e-12)


# --------------------------------------------------------------------------
# 11. Skewed imposed motion drives
# --------------------------------------------------------------------------
def test_skewed_imposed_motion():
    """Verify master /IMPVEL and /IMPDISP along skew coordinate axes."""
    model, rb, loads, log = _setup_basic_rbody(n_slaves=4)

    skews = SkewSet()
    theta = np.pi / 6.0
    axes_30 = np.array([
        [np.cos(theta), np.sin(theta), 0.0],
        [-np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    skews.axes = [np.eye(3), axes_30]
    skews.origins = [np.zeros(3), np.zeros(3)]
    skews.is_mov = [False, False]
    model.skews = skews

    model.functions[400] = _DummyFunc(val=12.0)
    imp_skew = ImposedVelocity(id=1, grnod_id=1, dof=0, funct_id=400, scale=1.0, skew_row=1)
    model.impvel = [imp_skew]
    model.node_groups[1] = NodeGroup(id=1, node_ids=np.array([1]), node_idx=np.array([0]))

    loads_skew = LoadsAndConstraints(model, log)
    rbe = RigidBodyEngine(rb, model, loads_skew, log)

    assert len(rbe.skew_drives) == 1

    dt = 0.001
    N = model.numnod
    rbe.advance(np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3)),
                model.v, model.vr, model.x, dt=dt, t_next=dt)

    # Component of v_ref along skew local axis 0 must equal 12.0
    v_local = np.dot(rbe.v_ref, axes_30[0])
    assert np.isclose(v_local, 12.0, atol=1e-12)


# --------------------------------------------------------------------------
# 12. Kinematic clash resolution & warning emission
# --------------------------------------------------------------------------
def test_kinematic_clash_resolution_warning():
    """Verify that slave nodes under /BCS or /IMPVEL warn and yield to rigid body."""
    model, rb, loads, log = _setup_basic_rbody(n_slaves=4)

    # Put a BCS and an IMPVEL on slave node 2 (index 2)
    slave_node_id = int(model.node_ids[2])
    bc_slave = BoundaryCondition(id=1, grnod_id=50, fix_tra=[True, True, True], fix_rot=[False, False, False])
    imp_slave = ImposedVelocity(id=2, grnod_id=50, dof=0, funct_id=0, scale=1.0)
    model.bcs = [bc_slave]
    model.impvel = [imp_slave]
    model.node_groups[50] = NodeGroup(id=50, node_ids=np.array([slave_node_id]), node_idx=np.array([2]))

    log_check = MessageLog()
    loads_clash = LoadsAndConstraints(model, log_check)

    # LoadsAndConstraints has the slave node in impvel initially
    assert np.isin(2, loads_clash.impvel[0][0])

    rbe = RigidBodyEngine(rb, model, loads_clash, log_check)

    # Slave node 2 must be stripped out of loads.impvel to prevent clash
    assert not np.isin(2, loads_clash.impvel[0][0])

    # Warning messages must have been logged
    warnings_text = "\n".join(log_check.warnings)
    assert "touches slave node(s) — the rigid body wins" in warnings_text
    assert "/IMPVEL drives slave node(s)" in warnings_text


# --------------------------------------------------------------------------
# 13. Effective mass augmentation (finalize_mass)
# --------------------------------------------------------------------------
def test_effective_mass_augmentation():
    """Verify finalize_mass folds /INTER/TYPE2 secondary inertia into body M and J."""
    model, rb, loads, log = _setup_basic_rbody(n_slaves=4)
    rbe = RigidBodyEngine(rb, model, loads, log)

    initial_M = rbe.M
    initial_J = rbe.J0.copy()

    # Add tied secondary mass delta dm = 3.0 to slave node 1
    mass_eff = model.mass.copy()
    mass_eff[1] += 3.0

    rbe.finalize_mass(mass_eff)

    assert np.isclose(rbe.M, initial_M + 3.0, atol=1e-14)
    r1 = rbe.r0[1]  # slave node 1 offset
    expected_dJ = 3.0 * (np.eye(3) * float(r1 @ r1) - np.outer(r1, r1))
    assert np.allclose(rbe.J0, initial_J + expected_dJ, atol=1e-14)


# --------------------------------------------------------------------------
# 14. Finite rotation and placement (enforce)
# --------------------------------------------------------------------------
def test_enforce_finite_rotation_and_placement():
    """Verify node placement x = x_ref + R r0 and velocity re-scattering in enforce()."""
    model, rb, loads, log = _setup_basic_rbody(n_slaves=4)
    rbe = RigidBodyEngine(rb, model, loads, log)

    rbe.v_ref = np.array([10.0, 0.0, 0.0])
    rbe.w = np.array([0.0, 0.0, np.pi / 2.0])  # 90 deg/sec
    dt = 1.0  # rotates by 90 degrees

    rbe.enforce(model.x, model.v, dt)

    # After 90 deg rotation about Z, node at [1, 0, 0] relative to master lands on [0, 1, 0]
    expected_R = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert np.allclose(rbe.R, expected_R, atol=1e-12)

    for i, n in enumerate(rbe.nodes):
        expected_pos = rbe.x_ref + expected_R @ rbe.r0[i]
        assert np.allclose(model.x[n], expected_pos, atol=1e-12)


# --------------------------------------------------------------------------
# 15. Restart resumption from saved state
# --------------------------------------------------------------------------
def test_restart_resumption():
    """Verify that dynamic state dictionary accurately restores R, L, v_ref, w, x_ref, xg."""
    model, rb, loads, log = _setup_basic_rbody(n_slaves=4)

    saved_state = {
        "R": np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64),
        "L": np.array([1.0, 2.0, 3.0]),
        "v_ref": np.array([4.0, 5.0, 6.0]),
        "w": np.array([0.1, 0.2, 0.3]),
        "x_ref": np.array([10.0, 20.0, 30.0]),
        "xg": np.array([11.0, 21.0, 31.0]),
    }

    log_resume = MessageLog()
    rbe = RigidBodyEngine(rb, model, loads, log_resume, saved=saved_state)

    assert np.allclose(rbe.R, saved_state["R"])
    assert np.allclose(rbe.L, saved_state["L"])
    assert np.allclose(rbe.v_ref, saved_state["v_ref"])
    assert np.allclose(rbe.w, saved_state["w"])
    assert np.allclose(rbe.x_ref, saved_state["x_ref"])
    assert np.allclose(rbe.xg, saved_state["xg"])


# --------------------------------------------------------------------------
# 16. Chain detection and rejection in build_rigid_bodies
# --------------------------------------------------------------------------
def test_chain_detection_and_rejection():
    """Verify build_rigid_bodies detects rigid body chains and raises NotImplementedError."""
    model = Model()
    model.node_ids = np.arange(1, 5, dtype=np.int64)

    # Body 1: master=0, slaves=[1, 2]
    rb1 = RigidBody(id=1, kind="RBODY", master_id=1, grnod_id=10, master=0, slaves=np.array([1, 2], dtype=np.int64), mass_total=10.0, xg=np.zeros(3), J=np.eye(3))
    # Body 2: master=2 (slave of Body 1!), slaves=[3]
    rb2 = RigidBody(id=2, kind="RBODY", master_id=3, grnod_id=20, master=2, slaves=np.array([3], dtype=np.int64), mass_total=5.0, xg=np.zeros(3), J=np.eye(3))
    model.rbodies = [rb1, rb2]

    log = MessageLog()
    loads = LoadsAndConstraints(model, log)

    with pytest.raises(NotImplementedError, match="rigid-body CHAIN"):
        build_rigid_bodies(model, loads, log)


# --------------------------------------------------------------------------
# 17. Defensive handling of singular inertia and empty arrays
# --------------------------------------------------------------------------
def test_defensive_singular_inertia_and_empty_handling():
    """Verify robust Moore-Penrose pseudo-inverse fallback on singular inertia tensors."""
    model, rb, loads, log = _setup_basic_rbody(n_slaves=2)

    # Make J0 singular (e.g., all slave masses collinear on X axis, J_xx = 0)
    rb.J = np.diag([0.0, 10.0, 10.0])

    rbe = RigidBodyEngine(rb, model, loads, log)

    # Must solve for spin without crashing
    assert not np.isnan(rbe.w).any()

    # Advance must also not raise LinAlgError
    N = model.numnod
    rbe.advance(np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3)),
                model.v, model.vr, model.x, dt=0.001, t_next=0.001)

    assert not np.isnan(rbe.w).any()

    # Check properties
    assert np.allclose(rbe.x_cg, rbe.xg)
    assert np.allclose(rbe.v_cg, rbe.v_ref)
