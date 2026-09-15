"""
Milestone M487: /MPC Multi-Point Constraints Unit Tests & Documentation Hardening.

Direct unit test coverage for `pyradioss/engine/mpc.py` (MpcConstraints, build_mpc)
and `pyradioss/input/starter_keywords.py` (read_mpc).

Upstream OpenRadioss Fortran reference files:
  - starter/source/constraints/general/mpc/hm_read_mpc.F
  - engine/source/tools/lagmul/lag_mpc.F
  - starter/source/tools/lagmul/lgmini_mpc.F
  - config/CFG/radioss110/RBODY/mpc.cfg
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.engine.mpc import MpcConstraints, build_mpc
from pyradioss.input.deck_reader import Card, KeywordBlock
from pyradioss.input.starter_keywords import read_mpc
from pyradioss.model.entities import Mpc, RigidBody
from pyradioss.model.model import Model


class DummyLoads:
    """Minimal mock for boundary conditions / loads."""
    def __init__(self, numnod: int):
        self.fix_tra = np.zeros((numnod, 3), dtype=bool)
        self.fix_rot = np.zeros((numnod, 3), dtype=bool)


def _make_model(numnod=4, masses=None, inertias=None, mpcs=None, rbodies=None):
    """Create a minimal mock Model for testing MpcConstraints."""
    model = Model()
    model.node_ids = np.arange(1, numnod + 1, dtype=np.int64)
    model._id2idx = {int(nid): i for i, nid in enumerate(model.node_ids)}
    model.x = np.zeros((numnod, 3), dtype=np.float64)
    model.x0 = np.zeros((numnod, 3), dtype=np.float64)
    model.v = np.zeros((numnod, 3), dtype=np.float64)
    if masses is None:
        model.mass = np.ones(numnod, dtype=np.float64) * 2.0
    else:
        model.mass = np.asarray(masses, dtype=np.float64)
    if inertias is None:
        model.inertia = np.ones(numnod, dtype=np.float64) * 0.5
    else:
        model.inertia = np.asarray(inertias, dtype=np.float64)
    model.mpcs = mpcs or []
    model.rbodies = rbodies or []
    return model


# ============================================================================
# 1. Construction & G-Matrix Assembly Tests
# ============================================================================

def test_mpc_build_none_when_no_mpc():
    """build_mpc returns None when model has no /MPC entities."""
    model = _make_model(numnod=3, mpcs=[])
    log = MessageLog()
    c = build_mpc(model, None, log)
    assert c is None


def test_mpc_build_none_when_all_mpcs_invalid():
    """build_mpc returns None when all /MPC entities are invalid (e.g. unknown nodes)."""
    model = _make_model(numnod=3)
    # Reference unknown node 99
    model.mpcs = [Mpc(id=1, node_ids=[1, 99], dofs=[1, 1], coefs=[1.0, -1.0])]
    log = MessageLog()
    c = build_mpc(model, None, log)
    assert c is None
    assert any("unknown node id" in err for err in log.errors)


def test_mpc_g_matrix_assembly_simple():
    """Single row: u1_x - u2_x = 0."""
    model = _make_model(numnod=3)
    model.mpcs = [Mpc(id=1, node_ids=[1, 2], dofs=[1, 1], coefs=[1.0, -1.0])]
    log = MessageLog()
    c = MpcConstraints(model, loads=None, log=log)
    assert len(c) == 1
    assert c.nc == 1
    # 2 distinct columns (node 0 DOF 0 and node 1 DOF 0)
    assert c.G.shape == (1, 2)
    assert np.allclose(c.G, [[1.0, -1.0]])
    assert np.array_equal(c.col_node, [0, 1])
    assert np.array_equal(c.col_dof, [0, 0])
    assert np.all(c.tra)


def test_mpc_g_matrix_duplicate_terms_sum():
    """Multiple terms in the same row on the same node and DOF sum coefficients."""
    model = _make_model(numnod=3)
    # Node 1 DOF 1 has 1.0 + 0.5 = 1.5, node 2 DOF 1 has -1.5
    model.mpcs = [Mpc(id=1, node_ids=[1, 1, 2], dofs=[1, 1, 1], coefs=[1.0, 0.5, -1.5])]
    log = MessageLog()
    c = MpcConstraints(model, loads=None, log=log)
    assert len(c) == 1
    assert c.G.shape == (1, 2)
    assert np.allclose(c.G, [[1.5, -1.5]])


def test_mpc_g_matrix_multiple_independent_rows():
    """Two independent rows: u1_x - u2_x = 0 and u3_y - u4_y = 0."""
    model = _make_model(numnod=4)
    model.mpcs = [
        Mpc(id=1, node_ids=[1, 2], dofs=[1, 1], coefs=[1.0, -1.0]),
        Mpc(id=2, node_ids=[3, 4], dofs=[2, 2], coefs=[1.0, -1.0]),
    ]
    log = MessageLog()
    c = MpcConstraints(model, loads=None, log=log)
    assert len(c) == 2
    assert c.G.shape == (2, 4)
    # Row 0 acts on columns for (node 0 dof 0) and (node 1 dof 0)
    # Row 1 acts on columns for (node 2 dof 1) and (node 3 dof 1)
    assert np.allclose(c.G[0, :2], [1.0, -1.0])
    assert np.allclose(c.G[0, 2:], [0.0, 0.0])
    assert np.allclose(c.G[1, :2], [0.0, 0.0])
    assert np.allclose(c.G[1, 2:], [1.0, -1.0])


def test_mpc_shared_dof_across_rows():
    """Two rows coupling a shared node: u1_x - u2_x = 0 and u2_x - u3_x = 0."""
    model = _make_model(numnod=3)
    model.mpcs = [
        Mpc(id=1, node_ids=[1, 2], dofs=[1, 1], coefs=[1.0, -1.0]),
        Mpc(id=2, node_ids=[2, 3], dofs=[1, 1], coefs=[1.0, -1.0]),
    ]
    log = MessageLog()
    c = MpcConstraints(model, loads=None, log=log)
    assert len(c) == 2
    # Columns are nodes 0, 1, 2 for dof 0 -> 3 columns
    assert c.G.shape == (2, 3)
    expected_G = np.array([
        [1.0, -1.0, 0.0],
        [0.0, 1.0, -1.0]
    ])
    assert np.allclose(c.G, expected_G)


def test_mpc_rotational_dofs_valid():
    """Rotational DOFs (4..6) with positive inertia are supported."""
    model = _make_model(numnod=2, inertias=[0.1, 0.2])
    # DOF 4 (rx) and DOF 6 (rz)
    model.mpcs = [Mpc(id=1, node_ids=[1, 2], dofs=[4, 6], coefs=[2.0, -3.0])]
    log = MessageLog()
    c = MpcConstraints(model, loads=None, log=log)
    assert len(c) == 1
    assert not np.any(c.tra)  # both are rotational
    assert np.array_equal(c.col_dof, [3, 5])  # internally 0..5


def test_mpc_rotational_dof_zero_inertia_rejected():
    """Rotational DOF on node with inertia <= 0 is rejected with an error log."""
    model = _make_model(numnod=2, inertias=[0.0, 0.2])
    model.mpcs = [Mpc(id=1, node_ids=[1, 2], dofs=[4, 4], coefs=[1.0, -1.0])]
    log = MessageLog()
    c = MpcConstraints(model, loads=None, log=log)
    assert len(c) == 0
    assert any("carries no rotational inertia" in err for err in log.errors)


def test_mpc_rigid_body_clash_warning():
    """MPC term on a rigid body node triggers a kinematic clash warning."""
    model = _make_model(numnod=3)
    rb = RigidBody(id=1, kind="RBODY", master_id=1, grnod_id=1)
    rb.master = 0
    rb.slaves = np.array([1])
    model.rbodies = [rb]
    model.mpcs = [Mpc(id=1, node_ids=[1, 3], dofs=[1, 1], coefs=[1.0, -1.0])]
    log = MessageLog()
    c = MpcConstraints(model, loads=None, log=log)
    assert len(c) == 1
    assert any("kinematic clash" in w for w in log.warnings)


# ============================================================================
# 2. Boundary Condition & Ground DOF Tests
# ============================================================================

def test_mpc_fixed_bcs_dof_treated_as_ground():
    """/BCS-fixed DOF is marked fixed; its minv entry is zeroed."""
    model = _make_model(numnod=2, masses=[2.0, 4.0], inertias=[1.0, 1.0])
    loads = DummyLoads(numnod=2)
    # Node 0 DOF 0 (X) is fixed by BCS
    loads.fix_tra[0, 0] = True
    model.mpcs = [Mpc(id=1, node_ids=[1, 2], dofs=[1, 1], coefs=[1.0, -1.0])]
    log = MessageLog()
    c = MpcConstraints(model, loads=loads, log=log)
    assert np.array_equal(c.fixed, [True, False])
    inv_m = 1.0 / model.mass
    inv_I = 1.0 / model.inertia
    m_cols = c._minv_cols(inv_m, inv_I)
    # Node 0 fixed -> m_cols[0] == 0.0; Node 1 free -> 1.0 / 4.0 = 0.25
    assert m_cols[0] == 0.0
    assert m_cols[1] == 0.25


# ============================================================================
# 3. Force Transfer & Acceleration Constraint Tests
# ============================================================================

def test_transfer_forces_acceleration_satisfaction():
    """After transfer_forces, the acceleration satisfies G a = 0."""
    model = _make_model(numnod=3, masses=[1.0, 2.0, 3.0])
    # u1 + 2*u2 - 3*u3 = 0
    model.mpcs = [Mpc(id=1, node_ids=[1, 2, 3], dofs=[1, 1, 1], coefs=[1.0, 2.0, -3.0])]
    c = MpcConstraints(model, loads=None, log=MessageLog())

    inv_mass = 1.0 / model.mass
    inv_inertia = 1.0 / model.inertia
    fint = np.array([[10.0, 0.0, 0.0],
                     [-5.0, 0.0, 0.0],
                     [ 2.0, 0.0, 0.0]])
    fcont = np.zeros_like(fint)
    fext = np.zeros_like(fint)
    mint = np.zeros_like(fint)

    c.transfer_forces(fint, fcont, fext, mint, inv_mass, inv_inertia)

    # Compute updated accelerations on X
    a = fint[:, 0] * inv_mass
    # Constraint equation: a[0] + 2*a[1] - 3*a[2] must equal 0
    res = a[0] + 2.0 * a[1] - 3.0 * a[2]
    assert np.isclose(res, 0.0, atol=1e-12)


def test_transfer_forces_equal_mass_symmetry():
    """Equal masses connected by u1_x - u2_x = 0 under force F on node 1 only."""
    model = _make_model(numnod=2, masses=[2.0, 2.0])
    model.mpcs = [Mpc(id=1, node_ids=[1, 2], dofs=[1, 1], coefs=[1.0, -1.0])]
    c = MpcConstraints(model, loads=None, log=MessageLog())

    inv_mass = 1.0 / model.mass
    inv_inertia = 1.0 / model.inertia
    F = 100.0
    fint = np.array([[F, 0.0, 0.0],
                     [0.0, 0.0, 0.0]])
    fcont = np.zeros_like(fint)
    fext = np.zeros_like(fint)
    mint = np.zeros_like(fint)

    c.transfer_forces(fint, fcont, fext, mint, inv_mass, inv_inertia)

    a = fint[:, 0] * inv_mass
    # Symmetrical acceleration a1 == a2 == F / (m1 + m2) = 100 / 4 = 25.0
    assert np.isclose(a[0], 25.0)
    assert np.isclose(a[1], 25.0)


def test_transfer_forces_unequal_mass():
    """Unequal masses m1=1.0, m2=3.0 connected by u1 - u2 = 0 under F=40.0."""
    model = _make_model(numnod=2, masses=[1.0, 3.0])
    model.mpcs = [Mpc(id=1, node_ids=[1, 2], dofs=[1, 1], coefs=[1.0, -1.0])]
    c = MpcConstraints(model, loads=None, log=MessageLog())

    inv_mass = 1.0 / model.mass
    inv_inertia = 1.0 / model.inertia
    F = 40.0
    fint = np.array([[F, 0.0, 0.0],
                     [0.0, 0.0, 0.0]])
    fcont = np.zeros_like(fint)
    fext = np.zeros_like(fint)
    mint = np.zeros_like(fint)

    c.transfer_forces(fint, fcont, fext, mint, inv_mass, inv_inertia)

    a = fint[:, 0] * inv_mass
    # Common acceleration a = F / (m1 + m2) = 40 / 4 = 10.0
    assert np.isclose(a[0], 10.0)
    assert np.isclose(a[1], 10.0)


def test_transfer_forces_rotational_moment_coupling():
    """Rotational DOFs update mint to satisfy angular acceleration constraint."""
    model = _make_model(numnod=2, inertias=[2.0, 3.0])
    # theta1_x - theta2_x = 0 (DOF 4)
    model.mpcs = [Mpc(id=1, node_ids=[1, 2], dofs=[4, 4], coefs=[1.0, -1.0])]
    c = MpcConstraints(model, loads=None, log=MessageLog())

    inv_mass = 1.0 / model.mass
    inv_inertia = 1.0 / model.inertia
    fint = np.zeros((2, 3))
    fcont = np.zeros_like(fint)
    fext = np.zeros_like(fint)
    M = 50.0
    mint = np.array([[M, 0.0, 0.0],
                     [0.0, 0.0, 0.0]])

    c.transfer_forces(fint, fcont, fext, mint, inv_mass, inv_inertia)

    alpha = mint[:, 0] * inv_inertia
    # Common angular acceleration alpha = M / (I1 + I2) = 50 / 5 = 10.0
    assert np.isclose(alpha[0], 10.0)
    assert np.isclose(alpha[1], 10.0)


def test_transfer_forces_zero_virtual_work():
    """Lagrange multiplier force has zero virtual power on any conforming velocity."""
    model = _make_model(numnod=3, masses=[1.0, 2.0, 1.5])
    model.mpcs = [Mpc(id=1, node_ids=[1, 2, 3], dofs=[1, 1, 1], coefs=[2.0, -1.0, -1.0])]
    c = MpcConstraints(model, loads=None, log=MessageLog())

    inv_mass = 1.0 / model.mass
    inv_inertia = 1.0 / model.inertia
    fint_orig = np.array([[12.0, 0.0, 0.0],
                          [-4.0, 0.0, 0.0],
                          [ 0.0, 0.0, 0.0]])
    fint = fint_orig.copy()
    c.transfer_forces(fint, np.zeros_like(fint), np.zeros_like(fint),
                      np.zeros_like(fint), inv_mass, inv_inertia)

    f_lagrange = fint[:, 0] - fint_orig[:, 0]

    # Any conforming velocity field satisfies 2*v1 - v2 - v3 = 0
    # Let v1 = 3.0, v2 = 4.0, v3 = 2.0 (2*3 - 4 - 2 = 0)
    v_conforming = np.array([3.0, 4.0, 2.0])
    power = np.dot(f_lagrange, v_conforming)
    assert np.isclose(power, 0.0, atol=1e-12)


# ============================================================================
# 4. Velocity Cleanup / Enforce Tests
# ============================================================================

def test_enforce_velocity_projection():
    """Non-conforming velocities are projected to satisfy G v = 0."""
    model = _make_model(numnod=2, masses=[1.0, 1.0])
    model.mpcs = [Mpc(id=1, node_ids=[1, 2], dofs=[1, 1], coefs=[1.0, -1.0])]
    c = MpcConstraints(model, loads=None, log=MessageLog())

    inv_mass = 1.0 / model.mass
    inv_inertia = 1.0 / model.inertia

    v = np.array([[10.0, 0.0, 0.0],
                  [ 2.0, 0.0, 0.0]])
    vr = np.zeros_like(v)

    c.enforce(v, vr, inv_mass, inv_inertia)

    # u1_x - u2_x = 0; with equal masses, mean velocity is (10 + 2)/2 = 6.0
    assert np.isclose(v[0, 0], 6.0)
    assert np.isclose(v[1, 0], 6.0)
    assert np.isclose(v[0, 0] - v[1, 0], 0.0, atol=1e-12)


def test_enforce_already_satisfied_no_op():
    """If G v == 0 initially, enforce exits early and leaves velocities unchanged."""
    model = _make_model(numnod=2)
    model.mpcs = [Mpc(id=1, node_ids=[1, 2], dofs=[1, 1], coefs=[1.0, -1.0])]
    c = MpcConstraints(model, loads=None, log=MessageLog())

    inv_mass = 1.0 / model.mass
    inv_inertia = 1.0 / model.inertia

    v = np.array([[5.0, 1.0, 2.0],
                  [5.0, 3.0, 4.0]])
    v_orig = v.copy()
    vr = np.zeros_like(v)

    c.enforce(v, vr, inv_mass, inv_inertia)
    assert np.array_equal(v, v_orig)


def test_enforce_rotational_velocity():
    """Rotational velocities are projected onto G vr = 0."""
    model = _make_model(numnod=2, inertias=[2.0, 2.0])
    model.mpcs = [Mpc(id=1, node_ids=[1, 2], dofs=[5, 5], coefs=[1.0, -1.0])]  # ry
    c = MpcConstraints(model, loads=None, log=MessageLog())

    v = np.zeros((2, 3))
    vr = np.array([[0.0, 8.0, 0.0],
                   [0.0, 2.0, 0.0]])

    c.enforce(v, vr, 1.0 / model.mass, 1.0 / model.inertia)
    # Average is 5.0
    assert np.isclose(vr[0, 1], 5.0)
    assert np.isclose(vr[1, 1], 5.0)


def test_enforce_fixed_dof_drives_free_dof():
    """When one DOF is /BCS-fixed, cleanup moves the free DOF to match."""
    model = _make_model(numnod=2, masses=[1.0, 1.0])
    loads = DummyLoads(numnod=2)
    loads.fix_tra[0, 0] = True  # Node 0 fixed
    model.mpcs = [Mpc(id=1, node_ids=[1, 2], dofs=[1, 1], coefs=[1.0, -1.0])]
    c = MpcConstraints(model, loads=loads, log=MessageLog())

    v = np.array([[0.0, 0.0, 0.0],   # fixed node has v=0
                  [7.0, 0.0, 0.0]])  # free node has v=7
    vr = np.zeros_like(v)

    c.enforce(v, vr, 1.0 / model.mass, 1.0 / model.inertia)
    # Fixed node remains 0.0; free node is driven to 0.0
    assert np.isclose(v[0, 0], 0.0)
    assert np.isclose(v[1, 0], 0.0)


# ============================================================================
# 5. Solver Singularity & Least-Squares Fallback Tests
# ============================================================================

def test_solve_redundant_rows_least_squares():
    """Redundant/singular row sets fall back to least-squares and log warning once."""
    model = _make_model(numnod=2)
    # Two identical rows: rank 1, but nc=2
    model.mpcs = [
        Mpc(id=1, node_ids=[1, 2], dofs=[1, 1], coefs=[1.0, -1.0]),
        Mpc(id=2, node_ids=[1, 2], dofs=[1, 1], coefs=[1.0, -1.0]),
    ]
    log = MessageLog()
    c = MpcConstraints(model, loads=None, log=log)

    inv_mass = 1.0 / model.mass
    inv_inertia = 1.0 / model.inertia

    fint = np.array([[10.0, 0.0, 0.0],
                     [ 0.0, 0.0, 0.0]])
    c.transfer_forces(fint, np.zeros_like(fint), np.zeros_like(fint),
                      np.zeros_like(fint), inv_mass, inv_inertia)

    assert c._warned_singular
    assert any("least-squares multipliers used" in w for w in log.warnings)
    # Constraint is still satisfied
    a = fint[:, 0] * inv_mass
    assert np.isclose(a[0], a[1])


# ============================================================================
# 6. Starter Keyword Parsing Tests (read_mpc)
# ============================================================================

def _make_block(user_id: int, title: str, lines: list[str], fixed: bool = False) -> KeywordBlock:
    cards = [Card(title)] + [Card(line) for line in lines]
    return KeywordBlock(
        keyword="MPC",
        parts=["MPC", str(user_id)],
        user_id=user_id,
        cards=cards,
        fixed=fixed,
    )


def test_read_mpc_3_tokens_free_format():
    """Standard 3-token lines: node_ID dof coef."""
    block = _make_block(10, "equality", ["1 1 1.0", "2 1 -1.0"])
    model = Model()
    log = MessageLog()
    read_mpc(block, model, log)
    assert not log.errors
    assert len(model.mpcs) == 1
    m = model.mpcs[0]
    assert m.id == 10
    assert m.title == "equality"
    assert m.node_ids == [1, 2]
    assert m.dofs == [1, 1]
    assert m.coefs == [1.0, -1.0]


def test_read_mpc_4_tokens_standard_radioss():
    """Standard Radioss 4-token lines: node_ID Idof skew_ID alpha."""
    block = _make_block(20, "radioss card", ["1 2 0 2.5", "2 2 0 -2.5"])
    model = Model()
    log = MessageLog()
    read_mpc(block, model, log)
    assert not log.errors
    assert len(model.mpcs) == 1
    m = model.mpcs[0]
    assert m.id == 20
    assert m.node_ids == [1, 2]
    assert m.dofs == [2, 2]
    assert m.coefs == [2.5, -2.5]


def test_read_mpc_fixed_format():
    """Fixed format %10d%10d%10d%20lg."""
    block = _make_block(
        30,
        "fixed card",
        [
            "         1         1         0                 1.0",
            "         2         1         0                -1.0",
        ],
        fixed=True,
    )
    model = Model()
    log = MessageLog()
    read_mpc(block, model, log)
    assert not log.errors
    assert len(model.mpcs) == 1
    m = model.mpcs[0]
    assert m.id == 30
    assert m.node_ids == [1, 2]
    assert m.dofs == [1, 1]
    assert m.coefs == [1.0, -1.0]


def test_read_mpc_default_coef_when_zero():
    """Upstream hm_read_mpc.F: IF (COEF==ZERO) COEF = ONE."""
    block = _make_block(40, "zero coef", ["1 1 0 0.0", "2 1 0 0.0"])
    model = Model()
    log = MessageLog()
    read_mpc(block, model, log)
    assert not log.errors
    m = model.mpcs[0]
    assert m.coefs == [1.0, 1.0]


def test_read_mpc_skew_warning():
    """Non-zero skew_ID logs warning that local skew frame is unported."""
    block = _make_block(50, "skew card", ["1 1 5 1.0", "2 1 0 -1.0"])
    model = Model()
    log = MessageLog()
    read_mpc(block, model, log)
    assert not log.errors
    assert any("local skew_ID=5 not ported" in w for w in log.warnings)


def test_read_mpc_validation_errors():
    """Reject invalid DOF and constraint with < 2 terms."""
    # Invalid DOF 7
    block1 = _make_block(60, "bad dof", ["1 7 1.0", "2 1 -1.0"])
    model = Model()
    log = MessageLog()
    read_mpc(block1, model, log)
    assert any("dof must be 1..6" in err for err in log.errors)
    assert len(model.mpcs) == 0

    # Only 1 term
    block2 = _make_block(61, "one term", ["1 1 1.0"])
    log = MessageLog()
    read_mpc(block2, model, log)
    assert any("needs at least two terms" in err for err in log.errors)
    assert len(model.mpcs) == 0


# ============================================================================
# 7. End-to-End Dynamic Simulation Test
# ============================================================================

def test_mpc_end_to_end_dynamic_motion():
    """Explicit central-difference time integration of 2 tied masses.

    Mass 1 (m=2) and Mass 2 (m=3) tied by u1_x = u2_x.
    Sinusoidal force applied to Mass 1.
    Both masses must stay perfectly synchronized (x1 == x2, v1 == v2) for all 50 steps.
    """
    model = _make_model(numnod=2, masses=[2.0, 3.0])
    model.mpcs = [Mpc(id=1, node_ids=[1, 2], dofs=[1, 1], coefs=[1.0, -1.0])]
    c = MpcConstraints(model, loads=None, log=MessageLog())

    inv_mass = 1.0 / model.mass
    inv_inertia = 1.0 / model.inertia

    dt = 0.01
    n_steps = 50

    for step in range(n_steps):
        t = step * dt
        # Applied force on node 0
        F = 10.0 * np.sin(5.0 * t)
        fint = np.array([[F, 0.0, 0.0],
                         [0.0, 0.0, 0.0]])
        fcont = np.zeros_like(fint)
        fext = np.zeros_like(fint)
        mint = np.zeros_like(fint)

        # 1. Transfer constraint forces
        c.transfer_forces(fint, fcont, fext, mint, inv_mass, inv_inertia)

        # 2. Acceleration update
        a = fint * inv_mass[:, None]
        assert np.isclose(a[0, 0], a[1, 0], atol=1e-11)

        # 3. Velocity update (leapfrog)
        model.v += a * dt

        # 4. Kinematic velocity cleanup
        c.enforce(model.v, np.zeros_like(model.v), inv_mass, inv_inertia)
        assert np.isclose(model.v[0, 0], model.v[1, 0], atol=1e-11)

        # 5. Position update
        model.x += model.v * dt
        assert np.isclose(model.x[0, 0], model.x[1, 0], atol=1e-11)
