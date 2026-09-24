"""
Milestone M586 Test Suite: Fortran-faithful Cylindrical Joints (/CYL_JOINT)
and Rigid Links (/RLINK) in pyradioss engine kinematics.

Tests:
1. Starter parsing for both /CYL_JOINT and /RLINK (fixed & free format, node lists).
2. Cylindrical joint sliding along axis without resistance while completely resisting
   perpendicular motion.
3. Exact linear and angular momentum conservation between main and secondary nodes.
4. Rigid link translational locking (Tx, Ty, Tz) and mass-weighted center of mass motion.
5. Rigid link partial DOFs decoupling.
6. End-to-end explicit engine run with /CYL_JOINT verifying motion and energy conservation.
7. End-to-end explicit engine run with /RLINK verifying tied motion and energy balance.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.engine_keywords import parse_engine_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.entities import CylJoint, NodeGroup, RigidLink
from pyradioss.model.model import EngineControls, Model
from pyradioss.starter.starter import run_starter
from pyradioss.engine.cyl_joint import CylJointEngine, build_cyl_joints
from pyradioss.engine.rlink import RigidLinkEngine, build_rlinks
from pyradioss.engine.engine import _integrate


def _parse_starter(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


# ============================================================================
# 1. Starter Parsing Tests
# ============================================================================

def test_m586_starter_parsing_cyl_joint_and_rlink(tmp_path: Path):
    """Verify parsing of /CYL_JOINT and /RLINK in both fixed and free formats,
    including node groups and additional secondary node cards.
    """
    deck = """# RADIOSS STARTER DECK
/BEGIN
M586 Starter Test
1 1
/NODE
1 0.0 0.0 0.0
2 10.0 0.0 0.0
3 5.0 0.0 0.0
4 5.0 1.0 0.0
10 0.0 0.0 0.0
11 1.0 0.0 0.0
12 2.0 0.0 0.0
/GRNOD/NODE/100
Group_100
3
/CYL_JOINT/1
Cylindrical Joint via Grnod
1 2 100
/CYL_JOINT/2
Cylindrical Joint via Node List
1 2 0
4
/RLINK/1
Rigid Link Fixed DOFs via Grnod
111000 0 100 0
/RLINK/2
Rigid Link via Node List
111111 0 0 0
10 11 12
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert not log.errors

    # Check /CYL_JOINT/1
    assert 1 in model.cyl_joints
    cj1 = model.cyl_joints[1]
    assert cj1.node_id1 == 1
    assert cj1.node_id2 == 2
    assert cj1.grnod_id == 100

    # Check /CYL_JOINT/2 (with secondary nodes card)
    assert 2 in model.cyl_joints
    cj2 = model.cyl_joints[2]
    assert cj2.node_id1 == 1
    assert cj2.node_id2 == 2
    assert 4 in cj2.secondary_nodes

    # Check /RLINK/1
    assert 1 in model.rlinks
    rl1 = model.rlinks[1]
    assert rl1.dofs == (1, 1, 1, 0, 0, 0)
    assert rl1.tx == 1 and rl1.ty == 1 and rl1.tz == 1
    assert rl1.rx == 0 and rl1.ry == 0 and rl1.rz == 0
    assert rl1.grnod_id == 100

    # Check /RLINK/2 (with node list card)
    assert 2 in model.rlinks
    rl2 = model.rlinks[2]
    assert rl2.dofs == (1, 1, 1, 1, 1, 1)
    assert rl2.node_ids == [10, 11, 12]


# ============================================================================
# 2. Cylindrical Joint Sliding and Perpendicular Resistance
# ============================================================================

def test_m586_cyl_joint_sliding_and_perpendicular_resistance():
    """Verify that secondary nodes slide along the axis with zero resistance,
    while motion perpendicular to the axis is completely constrained.
    """
    model = Model()
    model.node_ids = np.array([1, 2, 3], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1, 3: 2}
    # Axis along X from (0,0,0) to (10,0,0); secondary node 3 at (4,0,0)
    model.x = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ], dtype=np.float64)
    model.mass = np.array([10.0, 10.0, 2.0], dtype=np.float64)

    joint = CylJoint(id=1, node_id1=1, node_id2=2, secondary_nodes=[3])
    log = MessageLog()
    engine = CylJointEngine(joint, model, log)
    assert engine.is_valid

    # Case A: Pure axial velocity -> should be 100% preserved
    v = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [15.5, 0.0, 0.0],  # pure axial
    ], dtype=np.float64)
    engine.apply_velocity(v, model.x, mass=None)
    assert v[2, 0] == pytest.approx(15.5)
    assert v[2, 1] == pytest.approx(0.0)
    assert v[2, 2] == pytest.approx(0.0)

    # Case B: Pure axial acceleration -> should be 100% preserved
    a = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-42.0, 0.0, 0.0],
    ], dtype=np.float64)
    engine.apply_acceleration(a, model.x, mass=None)
    assert a[2, 0] == pytest.approx(-42.0)
    assert a[2, 1] == pytest.approx(0.0)
    assert a[2, 2] == pytest.approx(0.0)

    # Case C: Perpendicular velocity while axis is stationary -> perpendicular component eliminated
    v = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [7.0, 8.0, -9.0],  # v_par = 7.0, v_perp = (0, 8, -9)
    ], dtype=np.float64)
    engine.apply_velocity(v, model.x, mass=None)
    assert v[2, 0] == pytest.approx(7.0)
    assert v[2, 1] == pytest.approx(0.0, abs=1e-12)
    assert v[2, 2] == pytest.approx(0.0, abs=1e-12)

    # Case D: Axis itself is moving in Y and Z: v_A = (0, 2, 4), v_B = (0, 12, 14)
    # At xi = 4.0, wA = 0.6, wB = 0.4:
    # v_axis = 0.6 * (0, 2, 4) + 0.4 * (0, 12, 14) = (0, 6.0, 8.0)
    v = np.array([
        [0.0, 2.0, 4.0],
        [0.0, 12.0, 14.0],
        [5.0, 0.0, 0.0],  # starts with v_par = 5.0
    ], dtype=np.float64)
    engine.apply_velocity(v, model.x, mass=None)
    assert v[2, 0] == pytest.approx(5.0)    # axial preserved
    assert v[2, 1] == pytest.approx(6.0)    # matched axis Y
    assert v[2, 2] == pytest.approx(8.0)    # matched axis Z


# ============================================================================
# 3. Exact Linear and Angular Momentum Conservation in Cylindrical Joint
# ============================================================================

def test_m586_cyl_joint_momentum_conservation():
    """Verify that reaction impulses and forces transferred between secondary
    and main nodes conserve both linear and angular momentum to machine precision.
    """
    model = Model()
    model.node_ids = np.array([1, 2, 3], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1, 3: 2}

    # Arbitrary axis orientation in 3D:
    xA = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    xB = np.array([5.0, 8.0, 7.0], dtype=np.float64)
    vec = xB - xA
    S = float(np.linalg.norm(vec))
    n = vec / S

    # Place secondary node along axis at xi = 0.35 * S
    xi = 0.35 * S
    xS = xA + xi * n

    model.x = np.array([xA, xB, xS], dtype=np.float64)
    mA, mB, mS = 4.5, 3.0, 2.5
    mass = np.array([mA, mB, mS], dtype=np.float64)
    model.mass = mass

    joint = CylJoint(id=1, node_id1=1, node_id2=2, secondary_nodes=[3])
    log = MessageLog()
    engine = CylJointEngine(joint, model, log)

    # Arbitrary initial velocities
    v = np.array([
        [2.3, -1.1, 4.0],
        [-0.8, 3.5, 1.2],
        [5.0, -4.0, 2.0],
    ], dtype=np.float64)

    # Initial linear and angular momentum
    P_initial = (mass[:, None] * v).sum(axis=0)
    L_initial = sum(np.cross(model.x[i], mass[i] * v[i]) for i in range(3))

    # Apply velocity constraint with mass coupling
    engine.apply_velocity(v, model.x, mass=mass)

    # Final linear and angular momentum
    P_final = (mass[:, None] * v).sum(axis=0)
    L_final = sum(np.cross(model.x[i], mass[i] * v[i]) for i in range(3))

    # Assert exact conservation (< 1e-14)
    np.testing.assert_allclose(P_final, P_initial, atol=1e-14)
    np.testing.assert_allclose(L_final, L_initial, atol=1e-14)

    # Test force transfer conservation
    fint = np.array([
        [10.0, -5.0, 2.0],
        [3.0, 8.0, -4.0],
        [-7.0, 12.0, 15.0],
    ], dtype=np.float64)
    fext = np.zeros_like(fint)
    fcont = np.zeros_like(fint)
    mint = np.zeros_like(fint)

    F_initial = fint.sum(axis=0)
    T_initial = sum(np.cross(model.x[i], fint[i]) for i in range(3))

    engine.transfer_forces(fint, fext, fcont, mint, model.x)

    F_final = fint.sum(axis=0)
    T_final = sum(np.cross(model.x[i], fint[i]) for i in range(3))

    np.testing.assert_allclose(F_final, F_initial, atol=1e-14)
    np.testing.assert_allclose(T_final, T_initial, atol=1e-14)


# ============================================================================
# 4. Rigid Link Translational Locking and Center-of-Mass Kinematics
# ============================================================================

def test_m586_rigid_link_translational_locking_and_center_of_mass():
    """Verify that /RLINK ties nodes rigidly along active translational DOFs
    and sets accelerations and velocities to mass-weighted center of mass (rlink1.F).
    """
    model = Model()
    model.node_ids = np.array([1, 2, 3], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1, 3: 2}

    model.x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
        [3.0, 0.0, 4.0],
    ], dtype=np.float64)
    mass = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    model.mass = mass

    rlink = RigidLink(id=1, dofs=(1, 1, 1, 0, 0, 0), node_ids=[1, 2, 3])
    log = MessageLog()
    engine = RigidLinkEngine(rlink, model, log)
    assert engine.is_valid
    assert len(engine.node_idx) == 3

    # Nodal velocities:
    v = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 6.0],
    ], dtype=np.float64)

    # Mass-weighted v_cm = (1*[1,0,0] + 2*[0,3,0] + 3*[0,0,6]) / 6 = [1/6, 1.0, 3.0]
    expected_v_cm = np.array([1.0 / 6.0, 1.0, 3.0])

    P_initial = (mass[:, None] * v).sum(axis=0)
    engine.apply_velocity(v, mass)
    P_final = (mass[:, None] * v).sum(axis=0)

    # All nodes must have exactly v_cm
    for i in range(3):
        np.testing.assert_allclose(v[i], expected_v_cm, atol=1e-14)
    # Linear momentum strictly preserved
    np.testing.assert_allclose(P_final, P_initial, atol=1e-14)

    # Accelerations:
    a = np.array([
        [6.0, 0.0, 0.0],
        [0.0, 12.0, 0.0],
        [0.0, 0.0, 18.0],
    ], dtype=np.float64)
    expected_a_cm = np.array([1.0, 4.0, 9.0])

    engine.apply_acceleration(a, mass)
    for i in range(3):
        np.testing.assert_allclose(a[i], expected_a_cm, atol=1e-14)

    # Check geometric enforcement (enforce relative offsets)
    x = model.x.copy()
    # Apply enforce to capture reference offsets
    engine.enforce(x)
    initial_relative_dist = np.linalg.norm(x[1] - x[0])

    # Move nodes by v_cm * dt
    dt = 0.01
    x += v * dt
    engine.enforce(x)
    new_relative_dist = np.linalg.norm(x[1] - x[0])
    assert new_relative_dist == pytest.approx(initial_relative_dist, abs=1e-14)


# ============================================================================
# 5. Rigid Link Partial DOFs Decoupling
# ============================================================================

def test_m586_rigid_link_partial_dofs():
    """Verify that inactive DOFs remain completely independent."""
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.mass = np.array([2.0, 2.0], dtype=np.float64)

    # Tied only in X and Z: Tx=1, Ty=0, Tz=1
    rlink = RigidLink(id=1, dofs=(1, 0, 1, 0, 0, 0), node_ids=[1, 2])
    log = MessageLog()
    engine = RigidLinkEngine(rlink, model, log)

    v = np.array([
        [2.0, 10.0, 4.0],
        [6.0, 20.0, 8.0],
    ], dtype=np.float64)

    engine.apply_velocity(v, model.mass)

    # X and Z should be averaged: (2+6)/2 = 4, (4+8)/2 = 6
    assert v[0, 0] == pytest.approx(4.0)
    assert v[1, 0] == pytest.approx(4.0)
    assert v[0, 2] == pytest.approx(6.0)
    assert v[1, 2] == pytest.approx(6.0)

    # Y is untied -> must remain unchanged!
    assert v[0, 1] == pytest.approx(10.0)
    assert v[1, 1] == pytest.approx(20.0)


# ============================================================================
# 6. End-to-End Explicit Engine Run with /CYL_JOINT
# ============================================================================

def test_m586_cyl_joint_end_to_end_engine(tmp_path: Path):
    """Run an explicit dynamic simulation of a mass sliding on a cylindrical
    joint guide rail, verifying free sliding, constraint satisfaction, and
    energy conservation.
    """
    deck_starter = """# RADIOSS STARTER DECK
/BEGIN
Cylindrical Joint Explicit Run
/MAT/LAW1/1
Elastic
7.8e-9
210000.0 0.3
/PROP/TYPE1/1
Shell_Prop
1 1 1 0 0 0 0
1.0 1.0 1.0
1.0 0.833333
/PART/1
Part_Guide
1 1
/NODE
1 0.0 0.0 0.0
2 50.0 0.0 0.0
3 0.0 5.0 0.0
4 50.0 5.0 0.0
5 10.0 0.0 0.0
6 15.0 0.0 0.0
7 10.0 5.0 0.0
8 15.0 5.0 0.0
/SHELL/1
1 1 2 4 3
2 5 6 8 7
/GRNOD/NODE/10
Secondary_Group
5 6
/CYL_JOINT/1
Guide_Rail_Joint
1 2 10
/GRNOD/NODE/30
Rail_Ends
1 2 3 4
/BCS/1
Fix_Rail_Ends
111 111 0 30
/GRNOD/NODE/20
Slider_Nodes
5 6 7 8
/INIVEL/TRA/1
Initial_Velocity
20.0 0.0 0.0 20
/END
"""
    deck_engine = """# RADIOSS ENGINE DECK
/TITLE
Cylindrical Joint Engine
/RUN/CYL_RUN/1
0.001
/DT/NODA/CST
0.8 5.0e-7
/PRINT/-100
/ANIM/DT
0.0005
/END
"""
    p_sta = tmp_path / "CYL_RUN_0000.rad"
    p_sta.write_text(deck_starter.strip() + "\n", encoding="ascii")
    model = run_starter(str(p_sta))

    p_eng = tmp_path / "CYL_RUN_0001.rad"
    p_eng.write_text(deck_engine.strip() + "\n", encoding="ascii")
    blocks_eng = read_deck(str(p_eng))
    controls = parse_engine_deck(blocks_eng, MessageLog())

    # Run explicit time integration
    res_model = _integrate(model, controls, MessageLog(), str(tmp_path), "CYL_RUN", run_num=1)

    # Node 5 should slide freely along X: x_final > 10.0
    x_final = res_model.x[res_model.node_index(5)]
    v_final = res_model.v[res_model.node_index(5)]

    assert x_final[0] > 10.01
    assert v_final[0] == pytest.approx(20.0, rel=1e-3)
    # Perpendicular displacement and velocity must stay near zero
    assert abs(x_final[1]) < 1e-5
    assert abs(x_final[2]) < 1e-5
    assert abs(v_final[1]) < 1e-5
    assert abs(v_final[2]) < 1e-5


# ============================================================================
# 7. End-to-End Explicit Engine Run with /RLINK
# ============================================================================

def test_m586_rigid_link_end_to_end_engine(tmp_path: Path):
    """Run an explicit dynamic simulation with two nodes tied by /RLINK,
    verifying tied motion and energy conservation.
    """
    deck_starter = """# RADIOSS STARTER DECK
/BEGIN
Rigid Link Explicit Run
/MAT/LAW1/1
Elastic
7.8e-9
210000.0 0.3
/PROP/TYPE1/1
Shell_Prop
1 1 1 0 0 0 0
1.0 1.0 1.0
1.0 0.833333
/PART/1
Part_Body
1 1
/NODE
1 0.0 0.0 0.0
2 5.0 0.0 0.0
3 0.0 5.0 0.0
4 5.0 5.0 0.0
/SHELL/1
1 1 2 4 3
/GRNOD/NODE/20
Tied_Pair
1 2
/RLINK/1
Link_Pair
1 1 1 0 0 0 0 20 0
/INIVEL/TRA/1
Initial_Vel
10.0 5.0 0.0 20
/END
"""
    deck_engine = """# RADIOSS ENGINE DECK
/TITLE
Rigid Link Engine
/RUN/RLINK_RUN/1
0.01
/DT/NODA/CST
0.9 1.0e-5
/PRINT/-100
/ANIM/DT
0.005
/END
"""
    p_sta = tmp_path / "RLINK_RUN_0000.rad"
    p_sta.write_text(deck_starter.strip() + "\n", encoding="ascii")
    model = run_starter(str(p_sta))

    p_eng = tmp_path / "RLINK_RUN_0001.rad"
    p_eng.write_text(deck_engine.strip() + "\n", encoding="ascii")
    blocks_eng = read_deck(str(p_eng))
    controls = parse_engine_deck(blocks_eng, MessageLog())

    # Run explicit time integration
    res_model = _integrate(model, controls, MessageLog(), str(tmp_path), "RLINK_RUN", run_num=1)

    # Nodes 1 and 2 tied by /RLINK must have identical translational velocities
    idx1 = res_model.node_index(1)
    idx2 = res_model.node_index(2)

    v1 = res_model.v[idx1]
    v2 = res_model.v[idx2]

    # Tied in X, Y, Z
    np.testing.assert_allclose(v1, v2, atol=1e-4)

    # Relative distance between node 1 and 2 in X must stay locked at 5.0
    x1 = res_model.x[idx1]
    x2 = res_model.x[idx2]
    assert (x2[0] - x1[0]) == pytest.approx(5.0, rel=1e-3)
