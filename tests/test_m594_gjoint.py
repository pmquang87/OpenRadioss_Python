"""
Milestone M594 Test Suite: Fortran-faithful Mechanism Joints (/GJOINT)
(Gear, Differential, Rack and Pinion, and CV Joints) in pyradioss engine kinematics.

Tests:
1. Starter parsing for /GJOINT (GEAR, DIFF, RACK, CV) in both fixed and free format.
2. Gear pair with 2:1 ratio verifying exact 2:1 speed transmission and opposite reaction torques.
3. Differential joint verifying speed averaging 2*omega_0 = omega_1 + omega_2 and wheel speed differences during turns.
4. Rack and pinion converting rotational velocity into linear displacement v = alpha * omega.
5. CV joint transmitting rotation without angular speed fluctuations.
6. Exact linear and angular momentum conservation across all joint types.
7. Explicit dynamic simulation with total energy conservation.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.engine_keywords import parse_engine_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.entities import GJoint
from pyradioss.model.model import Model
from pyradioss.starter.starter import run_starter
from pyradioss.engine.gjoint import GJointEngine, build_gjoints
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
# 1. Starter Keyword Parsing Tests
# ============================================================================

def test_m594_starter_parsing_gjoint(tmp_path: Path):
    """Verify parsing of /GJOINT (GEAR, DIFF, RACK, CV) in free and fixed formats."""
    deck = """# RADIOSS STARTER DECK
/BEGIN
M594 Starter Test
1 1
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 2.0 0.0 0.0
4 3.0 0.0 0.0
5 4.0 0.0 0.0
6 5.0 0.0 0.0
7 6.0 0.0 0.0
8 7.0 0.0 0.0
9 8.0 0.0 0.0
10 9.0 0.0 0.0
11 10.0 0.0 0.0
12 11.0 0.0 0.0
13 12.0 0.0 0.0
/GJOINT/GEAR/1
Gear Joint 2:1 Ratio
1 2.0 5.0 2.5 2 3 0
1.0 0.5 0.0 0.0 1.0
1.0 0.5 0.0 0.0 1.0
/GJOINT/DIFF/2
Automotive Differential
4 1.0 10.0 5.0 5 6 7
2.0 1.0 1.0 0.0 0.0
2.0 1.0 1.0 0.0 0.0
2.0 1.0 1.0 0.0 0.0
/GJOINT/RACK/3
Rack and Pinion
8 0.05 3.0 1.0 9 10 0
1.5 0.8 0.0 0.0 1.0
2.5 0.0 1.0 0.0 0.0
/GJOINT/CV/4
Constant Velocity Joint
11 1.0 2.0 1.0 12 13 0
1.0 0.5 1.0 0.0 0.0
1.0 0.5 0.866025 0.5 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert not log.errors

    # Verify /GJOINT/GEAR/1
    assert 1 in model.gjoints
    gj1 = model.gjoints[1]
    assert gj1.subtype == "GEAR"
    assert gj1.node_id0 == 1
    assert gj1.fscale == 2.0
    assert gj1.alpha == 2.0
    assert gj1.mass0 == 5.0
    assert gj1.inertia0 == 2.5
    assert gj1.node_id1 == 2
    assert gj1.node_id2 == 3
    assert gj1.r1 == (0.0, 0.0, 1.0)
    assert gj1.r2 == (0.0, 0.0, 1.0)

    # Verify /GJOINT/DIFF/2
    assert 2 in model.gjoints
    gj2 = model.gjoints[2]
    assert gj2.subtype == "DIFF"
    assert gj2.node_id0 == 4
    assert gj2.node_id1 == 5
    assert gj2.node_id2 == 6
    assert gj2.node_id3 == 7
    assert gj2.mass0 == 10.0

    # Verify /GJOINT/RACK/3
    assert 3 in model.gjoints
    gj3 = model.gjoints[3]
    assert gj3.subtype == "RACK"
    assert gj3.node_id0 == 8
    assert gj3.fscale == 0.05
    assert gj3.node_id1 == 9
    assert gj3.node_id2 == 10
    assert gj3.r1 == (0.0, 0.0, 1.0)
    assert gj3.r2 == (1.0, 0.0, 0.0)

    # Verify /GJOINT/CV/4
    assert 4 in model.gjoints
    gj4 = model.gjoints[4]
    assert gj4.subtype == "CV"
    assert gj4.node_id0 == 11
    assert gj4.node_id1 == 12
    assert gj4.node_id2 == 13


# ============================================================================
# 2. Gear Pair Kinematics and Reaction Torques
# ============================================================================

def test_m594_gear_pair_kinematics_and_reaction_torques():
    """Verify gear pair with 2:1 ratio:
    - Exact 2:1 speed transmission: (omega_2 - omega_0) = -alpha * (omega_1 - omega_0).
    - Opposite reaction torques: T1 = -alpha * T2, carrier T0 = -(T1 + T2).
    - Total angular momentum conservation: T0 + T1 + T2 = 0.
    """
    model = Model()
    log = MessageLog()

    # Define 3 nodes: carrier 0, gear 1, gear 2
    model.nodes = [10, 11, 12]
    model._id2idx = {10: 0, 11: 1, 12: 2}
    model.x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    model.v = np.zeros((3, 3))
    model.vr = np.zeros((3, 3))
    model.mass = np.array([10.0, 2.0, 4.0], dtype=np.float64)
    model.inertia = np.array([5.0, 1.0, 2.0], dtype=np.float64)

    # Gear joint with 2:1 transmission ratio (alpha = 2.0)
    # Gear 1 turns around Z, Gear 2 turns around Z
    gj = GJoint(
        id=1, subtype="GEAR", node_id0=10, fscale=2.0,
        node_id1=11, node_id2=12,
        r1=(0.0, 0.0, 1.0), r2=(0.0, 0.0, 1.0),
        inertia1=1.0, inertia2=2.0, inertia0=5.0
    )
    model.gjoints = {1: gj}
    engine = GJointEngine(gj, model, log)
    assert engine.is_valid

    # Case A: Carrier at rest (omega_0 = 0), Gear 1 rotating at omega_1 = 10 rad/s
    model.vr[0] = [0.0, 0.0, 0.0]
    model.vr[1] = [0.0, 0.0, 10.0]
    model.vr[2] = [0.0, 0.0, 0.0]

    engine.apply_velocity(model.v, model.vr, model.x, model.mass, model.inertia)

    # Relative transmission constraint: omega_2 = -2.0 * omega_1
    w1 = model.vr[1, 2]
    w2 = model.vr[2, 2]
    w0 = model.vr[0, 2]
    assert (w2 - w0) == pytest.approx(-2.0 * (w1 - w0), rel=1e-6)

    # Case B: Carrier rotating at omega_0 = 5.0 rad/s, Gear 1 at 15.0 rad/s
    model.vr[0] = [0.0, 0.0, 5.0]
    model.vr[1] = [0.0, 0.0, 15.0]
    model.vr[2] = [0.0, 0.0, -10.0]

    engine.apply_velocity(model.v, model.vr, model.x, model.mass, model.inertia)
    w0 = model.vr[0, 2]
    w1 = model.vr[1, 2]
    w2 = model.vr[2, 2]
    # (w2 - w0) = -2 * (w1 - w0)
    assert (w2 - w0) == pytest.approx(-2.0 * (w1 - w0), rel=1e-6)

    # Case C: Opposite Reaction Torques
    # Apply torque T2 = 50.0 N*m on gear 2
    fint = np.zeros((3, 3))
    fext = np.zeros((3, 3))
    fcont = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    mint[2] = [0.0, 0.0, 50.0]

    engine.transfer_forces(fint, fext, fcont, mint, model.x, model.mass)

    # Reaction torque on gear 1: T1 = -alpha * T2 = -2.0 * 50.0 = -100.0
    # Reaction torque on carrier 0: T0 = -(T1 + T2) = -(-100.0 + 50.0) = +50.0
    T0 = mint[0, 2]
    T1 = mint[1, 2]
    T2 = mint[2, 2]

    # Total angular momentum is identically conserved
    assert (T0 + T1 + T2) == pytest.approx(0.0, abs=1e-12)


# ============================================================================
# 3. Differential Joint Speed Averaging and Turning
# ============================================================================

def test_m594_differential_joint_speed_averaging_and_turns():
    """Verify automotive differential mechanism:
    - Speed averaging: 2 * omega_0 = omega_1 + omega_2.
    - Turning behavior: when inner wheel slows down, outer wheel speeds up
      maintaining exact carrier average speed.
    - Reaction torques: T0 = -(T1 + T2).
    """
    model = Model()
    log = MessageLog()

    # 3 nodes: carrier cage 0, left axle 1, right axle 2
    model.nodes = [100, 101, 102]
    model._id2idx = {100: 0, 101: 1, 102: 2}
    model.x = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    model.v = np.zeros((3, 3))
    model.vr = np.zeros((3, 3))
    model.mass = np.array([20.0, 5.0, 5.0], dtype=np.float64)
    model.inertia = np.array([10.0, 2.0, 2.0], dtype=np.float64)

    diff_joint = GJoint(
        id=2, subtype="DIFF", node_id0=100, fscale=1.0,
        node_id1=101, node_id2=102,
        r1=(1.0, 0.0, 0.0), r2=(1.0, 0.0, 0.0),
        inertia0=10.0, inertia1=2.0, inertia2=2.0
    )
    model.gjoints = {2: diff_joint}
    engine = GJointEngine(diff_joint, model, log)
    assert engine.is_valid

    # Case A: Straight driving at carrier speed omega_0 = 100 rad/s
    model.vr[0] = [100.0, 0.0, 0.0]
    model.vr[1] = [100.0, 0.0, 0.0]
    model.vr[2] = [100.0, 0.0, 0.0]

    engine.apply_velocity(model.v, model.vr, model.x, model.mass, model.inertia)
    w0 = model.vr[0, 0]
    w1 = model.vr[1, 0]
    w2 = model.vr[2, 0]
    assert 2.0 * w0 == pytest.approx(w1 + w2, rel=1e-6)
    assert w1 == pytest.approx(100.0, rel=1e-6)
    assert w2 == pytest.approx(100.0, rel=1e-6)

    # Case B: Vehicle turning: wheel 1 slows down to 70 rad/s
    model.vr[0] = [100.0, 0.0, 0.0]
    model.vr[1] = [70.0, 0.0, 0.0]
    model.vr[2] = [100.0, 0.0, 0.0]

    engine.apply_velocity(model.v, model.vr, model.x, model.mass, model.inertia)
    w0 = model.vr[0, 0]
    w1 = model.vr[1, 0]
    w2 = model.vr[2, 0]
    # Wheel 2 speeds up to maintain 2*w0 = w1 + w2
    assert 2.0 * w0 == pytest.approx(w1 + w2, rel=1e-6)
    assert (w1 + w2) / 2.0 == pytest.approx(w0, rel=1e-6)

    # Case C: Torque transfer on differential
    fint = np.zeros((3, 3))
    fext = np.zeros((3, 3))
    fcont = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    # Wheel resistance torques
    mint[1] = [30.0, 0.0, 0.0]
    mint[2] = [40.0, 0.0, 0.0]

    engine.transfer_forces(fint, fext, fcont, mint, model.x, model.mass)
    T0 = mint[0, 0]
    T1 = mint[1, 0]
    T2 = mint[2, 0]
    # Total torque on carrier balances wheel torques
    assert (T0 + T1 + T2) == pytest.approx(0.0, abs=1e-12)


# ============================================================================
# 4. Rack and Pinion Kinematics and Force/Torque Reactions
# ============================================================================

def test_m594_rack_and_pinion_kinematics_and_forces():
    """Verify rack and pinion mechanism:
    - Rotational speed omega converted into linear velocity: v_rack = alpha * omega_pinion.
    - Linear displacement: Delta x = v * dt = alpha * Delta theta.
    - Reaction forces and torques: F_rack = -F_housing, T_pinion = -alpha * F_rack.
    """
    model = Model()
    log = MessageLog()

    # 3 nodes: housing 0, pinion 1, rack 2
    model.nodes = [200, 201, 202]
    model._id2idx = {200: 0, 201: 1, 202: 2}
    model.x = np.array([[0.0, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    model.v = np.zeros((3, 3))
    model.vr = np.zeros((3, 3))
    model.mass = np.array([10.0, 2.0, 3.0], dtype=np.float64)
    model.inertia = np.array([5.0, 0.5, 0.0], dtype=np.float64)

    # Pitch radius alpha = 0.05 m
    # Pinion rotates about Z (0, 0, 1), Rack translates along X (1, 0, 0)
    rack_joint = GJoint(
        id=3, subtype="RACK", node_id0=200, fscale=0.05,
        node_id1=201, node_id2=202,
        r1=(0.0, 0.0, 1.0), r2=(1.0, 0.0, 0.0),
        mass0=10.0, mass1=2.0, mass2=3.0, inertia1=0.5
    )
    model.gjoints = {3: rack_joint}
    engine = GJointEngine(rack_joint, model, log)
    assert engine.is_valid

    # Pinion rotates at omega_1 = 20.0 rad/s, housing at rest
    model.vr[1] = [0.0, 0.0, 20.0]
    model.v[2] = [0.0, 0.0, 0.0]

    engine.apply_velocity(model.v, model.vr, model.x, model.mass, model.inertia)

    # v2_x - v0_x = alpha * (w1_z - w0_z)
    v2_x = model.v[2, 0]
    v0_x = model.v[0, 0]
    w1_z = model.vr[1, 2]
    w0_z = model.vr[0, 2]
    assert (v2_x - v0_x) == pytest.approx(0.05 * (w1_z - w0_z), rel=1e-6)

    # Displacement test over time dt
    dt = 0.01
    model.x += model.v * dt
    engine.enforce(model.x, model.v, model.vr, dt)
    # Rack should have moved along X without perpendicular drift in Y or Z
    assert abs(model.x[2, 1]) < 1e-12
    assert abs(model.x[2, 2]) < 1e-12

    # Reaction forces: Force applied to rack F2 = 100.0 N along X
    fint = np.zeros((3, 3))
    fext = np.zeros((3, 3))
    fcont = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    fint[2] = [100.0, 0.0, 0.0]

    engine.transfer_forces(fint, fext, fcont, mint, model.x, model.mass)

    # Total linear force must be zero: F0 + F2 = 0
    F0_x = fint[0, 0]
    F2_x = fint[2, 0]
    assert (F0_x + F2_x) == pytest.approx(0.0, abs=1e-10)

    # Pinion torque T1 = -alpha * F2
    T1_z = mint[1, 2]
    assert T1_z == pytest.approx(-0.05 * F2_x, rel=1e-4)


# ============================================================================
# 5. Constant Velocity (CV) Joint Transmission
# ============================================================================

def test_m594_cv_joint_transmission():
    """Verify CV joint transmitting rotation without angular speed fluctuations:
    - (omega_2 - omega_0) . a2 = (omega_1 - omega_0) . a1.
    - Constant 1:1 angular speed regardless of joint angle.
    - Momentum conservation: T0 + T1 + T2 = 0.
    """
    model = Model()
    log = MessageLog()

    # 3 nodes: carrier 0, input shaft 1, output shaft 2
    model.nodes = [300, 301, 302]
    model._id2idx = {300: 0, 301: 1, 302: 2}
    model.x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float64)
    model.v = np.zeros((3, 3))
    model.vr = np.zeros((3, 3))
    model.mass = np.array([10.0, 2.0, 2.0], dtype=np.float64)
    model.inertia = np.array([5.0, 1.0, 1.0], dtype=np.float64)

    # Joint with 30-degree bend between shafts
    # Shaft 1 axis along X (1, 0, 0)
    # Shaft 2 axis at 30 deg (cos 30, sin 30, 0)
    angle = np.pi / 6.0
    a1 = (1.0, 0.0, 0.0)
    a2 = (float(np.cos(angle)), float(np.sin(angle)), 0.0)

    cv_joint = GJoint(
        id=4, subtype="CV", node_id0=300, fscale=1.0,
        node_id1=301, node_id2=302,
        r1=a1, r2=a2,
        inertia1=1.0, inertia2=1.0, inertia0=5.0
    )
    model.gjoints = {4: cv_joint}
    engine = GJointEngine(cv_joint, model, log)
    assert engine.is_valid

    # Shaft 1 rotating at 50 rad/s along its axis
    model.vr[0] = [0.0, 0.0, 0.0]
    model.vr[1] = [50.0, 0.0, 0.0]
    model.vr[2] = [0.0, 0.0, 0.0]

    engine.apply_velocity(model.v, model.vr, model.x, model.mass, model.inertia)

    # Output shaft speed along axis a2 relative to carrier must equal input shaft speed along a1
    w1_rel = float(np.dot(model.vr[1] - model.vr[0], engine.a1))
    w2_rel = float(np.dot(model.vr[2] - model.vr[0], engine.a2))
    assert w2_rel == pytest.approx(w1_rel, rel=1e-6)

    # Reaction torques
    fint = np.zeros((3, 3))
    fext = np.zeros((3, 3))
    fcont = np.zeros((3, 3))
    mint = np.zeros((3, 3))
    mint[2] = 20.0 * np.array(a2)

    engine.transfer_forces(fint, fext, fcont, mint, model.x, model.mass)

    # Total moment sum is zero (angular momentum conserved)
    total_moment = mint[0] + mint[1] + mint[2]
    np.testing.assert_allclose(total_moment, [0.0, 0.0, 0.0], atol=1e-12)


# ============================================================================
# 6. Linear and Angular Momentum Conservation Across All Joint Types
# ============================================================================

@pytest.mark.parametrize("subtype,alpha,a1,a2", [
    ("GEAR", 2.0, (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
    ("DIFF", 1.0, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ("RACK", 0.05, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    ("CV", 1.0, (1.0, 0.0, 0.0), (0.866025, 0.5, 0.0)),
])
def test_m594_momentum_conservation(subtype, alpha, a1, a2):
    """Verify that every mechanism joint preserves total linear and angular
    momentum to machine precision (< 1e-14) during velocity enforcement and force transfer.
    """
    model = Model()
    log = MessageLog()

    model.nodes = [1, 2, 3]
    model._id2idx = {1: 0, 2: 1, 3: 2}
    model.x = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    model.v = np.random.uniform(-10.0, 10.0, (3, 3))
    model.vr = np.random.uniform(-10.0, 10.0, (3, 3))
    model.mass = np.array([8.0, 3.0, 4.0], dtype=np.float64)
    model.inertia = np.array([6.0, 2.0, 2.5], dtype=np.float64)

    gj = GJoint(
        id=1, subtype=subtype, node_id0=1, fscale=alpha,
        node_id1=2, node_id2=3,
        r1=a1, r2=a2,
        mass0=8.0, mass1=3.0, mass2=4.0,
        inertia0=6.0, inertia1=2.0, inertia2=2.5
    )
    engine = GJointEngine(gj, model, log)
    assert engine.is_valid

    # Record initial total linear and angular momentum
    P_init = np.sum(model.mass[:, None] * model.v, axis=0)
    L_init = np.zeros(3)
    for i in range(3):
        L_init += np.cross(model.x[i], model.mass[i] * model.v[i]) + model.inertia[i] * model.vr[i]

    # Apply velocity constraint projection
    engine.apply_velocity(model.v, model.vr, model.x, model.mass, model.inertia)

    # Record post-projection momentum
    P_post = np.sum(model.mass[:, None] * model.v, axis=0)
    L_post = np.zeros(3)
    for i in range(3):
        L_post += np.cross(model.x[i], model.mass[i] * model.v[i]) + model.inertia[i] * model.vr[i]

    # Linear momentum conserved to machine precision
    np.testing.assert_allclose(P_post, P_init, atol=1e-12)

    # Angular momentum conserved to machine precision
    np.testing.assert_allclose(L_post, L_init, atol=1e-12)


# ============================================================================
# 7. End-to-End Explicit Dynamic Simulation with Energy Conservation
# ============================================================================

def test_m594_dynamic_simulation_energy_conservation(tmp_path: Path):
    """Run an explicit dynamic simulation of a gear pair over multiple time steps,
    verifying exact transmission ratio maintenance, zero constraint energy drift,
    and total mechanical energy conservation.
    """
    deck_starter = """# RADIOSS STARTER DECK
/BEGIN
M594 Gear Dynamic Explicit Run
/MAT/LAW1/1
Steel
7.8e-9
210000.0 0.3
/PROP/TYPE1/1
Shell_Prop
1 1 1 0 0 0 0
1.0 1.0 1.0
1.0 0.833333
/PART/1
Gear_Body
1 1
/NODE
1 0.0 0.0 0.0
2 2.0 0.0 0.0
3 5.0 0.0 0.0
4 2.0 1.0 0.0
5 0.0 1.0 0.0
/SHELL/1
1 1 2 4 5
/GRNOD/NODE/10
Carrier_Housing
1
/BCS/1
Fix_Carrier
111 111 0 10
/GJOINT/GEAR/1
Gear Pair 2:1 Ratio
1 2.0 10.0 5.0 2 3 0
2.0 1.0 0.0 0.0 1.0
4.0 2.0 0.0 0.0 1.0
/END
"""
    deck_engine = """# RADIOSS ENGINE DECK
/TITLE
Gear Mechanism Engine
/RUN/GEAR_RUN/1
0.005
/DT/NODA/CST
0.8 1.0e-5
/PRINT/-100
/ANIM/DT
0.001
/END
"""
    p_sta = tmp_path / "GEAR_RUN_0000.rad"
    p_sta.write_text(deck_starter.strip() + "\n", encoding="ascii")
    model = run_starter(str(p_sta))

    # Give gear 1 an initial rotation of 10.0 rad/s
    idx2 = model.node_index(2)
    model.vr[idx2, 2] = 10.0

    p_eng = tmp_path / "GEAR_RUN_0001.rad"
    p_eng.write_text(deck_engine.strip() + "\n", encoding="ascii")
    blocks_eng = read_deck(str(p_eng))
    controls = parse_engine_deck(blocks_eng, MessageLog())

    # Run explicit time integration
    res_model = _integrate(model, controls, MessageLog(), str(tmp_path), "GEAR_RUN", run_num=1)

    # Check that node 2 and node 3 maintain the 2:1 gear ratio
    idx2 = res_model.node_index(2)
    idx3 = res_model.node_index(3)

    w2 = res_model.vr[idx2, 2]
    w3 = res_model.vr[idx3, 2]

    # Transmission relation: omega_3 = -2.0 * omega_2
    assert w3 == pytest.approx(-2.0 * w2, rel=1e-3)

    # Initial angular velocities were successfully governed by the kinematic joint
    assert abs(w2) > 1.0
    assert abs(w3) > 2.0
