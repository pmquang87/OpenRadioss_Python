"""
Unit tests for /PROP/TYPE12 (/PROP/SPR_PUL: 3-Node Pulley Spring).

Fortran origin:
  engine/source/elements/spring/r3def3.F    (kinematics & sliding)
  engine/source/elements/spring/redef3.F90  (constitutive law)
  engine/source/elements/spring/r3cum3.F    (nodal force scatter)
  engine/source/elements/spring/r3bilan.F   (energy accounting)
  engine/source/elements/spring/r3len3.F    (critical time step)
  starter/source/properties/spring/hm_read_prop12.F (property reader)
  starter/source/elements/reader/hm_read_spring.F   (3-node spring reader)
  starter/source/elements/spring/r3buf3.F   (initial lengths)
  starter/source/elements/spring/rmas12.F   (lumped massing)
"""

import math
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import spring, spring_advanced
from pyradioss.model.entities import Property, PropType12
from pyradioss.model.model import ElementGroup, Model
from pyradioss.input.deck_reader import Card, KeywordBlock
from pyradioss.input.starter_keywords import read_prop, read_spring
from pyradioss.input.prop_reader import parse_spr_pul


class MockFunction:
    """Mock Radioss function curve."""
    def __init__(self, fid: int, x_pts, y_pts):
        self.id = fid
        self.x = np.array(x_pts, dtype=np.float64)
        self.y = np.array(y_pts, dtype=np.float64)

    def eval(self, val: float) -> float:
        return float(np.interp(val, self.x, self.y))


def test_type12_keyword_parsing_fixed_and_free():
    """Verify parsing of /PROP/TYPE12 and /PROP/SPR_PUL 5-card definitions."""
    # Fixed format 5 cards test (radioss2017 layout)
    card1 = f"{1.2:>20.1f}" + " " * 30 + f"{0:>10d}" + f"{0:>10d}" + f"{0:>10d}" + f"{0.25:>20.2f}"
    card2 = f"{500.0:>20.1f}" + f"{5.0:>20.1f}" + f"{1.0:>20.1f}" + f"{0.0:>20.1f}" + f"{1.0:>20.1f}"
    card3 = f"{0:>10d}" * 5 + " " * 10 + f"{-10.0:>20.1f}" + f"{10.0:>20.1f}"
    card4 = f"{1.0:>20.1f}" + f"{0.0:>20.1f}" + f"{1.0:>20.1f}" + f"{1.0:>20.1f}"
    card5 = f"{0:>10d}" + f"{0:>10d}" + f"{1.0:>20.1f}" + f"{1.0:>20.1f}" + f"{-100.0:>20.1f}" + f"{100.0:>20.1f}"
    block = KeywordBlock(
        keyword="PROP/TYPE12",
        parts=["PROP", "TYPE12", "100"],
        user_id=100,
        fixed=True,
        cards=[
            Card("PulleySpringTest"),
            Card(card1),
            Card(card2),
            Card(card3),
            Card(card4),
            Card(card5),
        ]
    )
    model1 = Model()
    log1 = MessageLog()
    read_prop(block, model1, log1)
    prop = model1.properties.get(100)
    assert prop is not None
    assert prop.type == 12
    assert prop.id == 100
    assert pytest.approx(prop.params["mass"]) == 1.2
    assert pytest.approx(prop.params["stiff1"]) == 500.0
    assert pytest.approx(prop.params["damp1"]) == 5.0
    assert pytest.approx(prop.params["fric"]) == 0.25
    assert pytest.approx(prop.params["min_rup1"]) == -10.0
    assert pytest.approx(prop.params["max_rup1"]) == 10.0
    assert pytest.approx(prop.params["f_min"]) == -100.0
    assert pytest.approx(prop.params["f_max"]) == 100.0

    # Free format (comma-separated) test
    block_free = KeywordBlock(
        keyword="PROP/SPR_PUL",
        parts=["PROP", "SPR_PUL", "200"],
        user_id=200,
        fixed=False,
        cards=[
            Card("FreeFormatPulley"),
            Card("2.5, 0, 0, 1, 0.35"),
            Card("1000.0, 10.0, 1.0, 0.0, 1.0"),
            Card("0, 0, 0, 0, 0, -5.0, 15.0"),
            Card("1.0, 0.0, 1.0, 1.0"),
            Card("0, 0, 1.5, 0.8, -50.0, 75.0"),
        ]
    )
    model2 = Model()
    log2 = MessageLog()
    read_prop(block_free, model2, log2)
    prop2 = model2.properties.get(200)
    assert prop2 is not None
    assert prop2.type == 12
    assert prop2.id == 200
    assert pytest.approx(prop2.params["mass"]) == 2.5
    assert pytest.approx(prop2.params["stiff1"]) == 1000.0
    assert pytest.approx(prop2.params["damp1"]) == 10.0
    assert prop2.params["ileng"] == 1
    assert pytest.approx(prop2.params["fric"]) == 0.35
    assert pytest.approx(prop2.params["scale2"]) == 1.5
    assert pytest.approx(prop2.params["scale3"]) == 0.8
    assert pytest.approx(prop2.params["f_min"]) == -50.0
    assert pytest.approx(prop2.params["f_max"]) == 75.0


def test_spring_3node_reading():
    """Verify reading 3-node spring elements from /SPRING."""
    # 3 nodes: elem_id, n1, n2, n3
    block = KeywordBlock(
        keyword="SPRING",
        parts=["SPRING", "1"],
        user_id=1,
        fixed=True,
        cards=[
            Card("         1         1         2         3"),
            Card("         2         4         5          "), # 2-node spring
        ]
    )
    model = Model()
    read_spring(block, model, MessageLog())
    assert len(model.raw_elems["SPRING"]) == 2
    e1 = model.raw_elems["SPRING"][0]
    assert e1[0] == 1 # elem_id
    assert e1[1] == 1 # part_id
    assert e1[2] == [1, 2, 3] # nodes
    e2 = model.raw_elems["SPRING"][1]
    assert e2[0] == 2
    assert e2[1] == 1
    assert e2[2] == [4, 5]


def test_type12_kinematics_and_l0():
    """Verify TYPE12 initial length L0 = L01 + L02 per r3buf3.F."""
    model = Model()
    # 3 nodes forming a right angle:
    # N1 at (0, 0, 0)
    # N2 (pulley) at (3, 0, 0) => L01 = 3.0
    # N3 at (3, 4, 0)          => L02 = 4.0
    # Total cable length L0 = 3.0 + 4.0 = 7.0
    model.node_ids = np.array([1, 2, 3], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1, 3: 2}
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [3.0, 4.0, 0.0],
    ], dtype=np.float64)

    prop = Property(
        id=1, type=12, title="PulleyGeom",
        params={"mass": 4.0, "stiff1": 1000.0, "fric": 0.0}
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1, 2]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    assert pytest.approx(group.state["L0"][0]) == 7.0


def test_type12_lumped_mass_distribution():
    """Verify lumped mass distribution (0.25*M, 0.50*M, 0.25*M) per rmas12.F."""
    model = Model()
    model.node_ids = np.array([10, 20, 30], dtype=np.int64)
    model._id2idx = {10: 0, 20: 1, 30: 2}
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=np.float64)

    M = 16.0
    prop = Property(
        id=1, type=12, title="PulleyMass",
        params={"mass": M, "stiff1": 1000.0, "ileng": 0}
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1, 2]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    node_idx, massn, inertn = spring.init_group(group, model, MessageLog())

    # Check node masses assembled
    nodal_masses = np.zeros(3)
    np.add.at(nodal_masses, node_idx, massn)
    assert pytest.approx(nodal_masses[0]) == 0.25 * M # Node 1: 4.0
    assert pytest.approx(nodal_masses[1]) == 0.50 * M # Node 2: 8.0 (pulley)
    assert pytest.approx(nodal_masses[2]) == 0.25 * M # Node 3: 4.0
    assert pytest.approx(np.sum(nodal_masses)) == M

    # Test unit length mass (ileng = 1)
    # L0 = 1.0 + 1.0 = 2.0. Total mass = 16.0 * 2.0 = 32.0
    prop_ileng = Property(
        id=2, type=12, title="PulleyMassIleng",
        params={"mass": M, "stiff1": 1000.0, "ileng": 1}
    )
    group_ileng = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1, 2]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop_ileng)]}
    )
    node_idx2, massn2, _ = spring.init_group(group_ileng, model, MessageLog())
    nodal_masses2 = np.zeros(3)
    np.add.at(nodal_masses2, node_idx2, massn2)
    assert pytest.approx(nodal_masses2[0]) == 0.25 * 32.0 # 8.0
    assert pytest.approx(nodal_masses2[1]) == 0.50 * 32.0 # 16.0
    assert pytest.approx(nodal_masses2[2]) == 0.25 * 32.0 # 8.0


def test_type12_zero_net_nodal_force():
    """Verify that net nodal force is strictly zero (f1 + f2 + f3 == 0) for arbitrary 3D geometry."""
    model = Model()
    model.node_ids = np.array([1, 2, 3], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1, 3: 2}
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 0.5],
        [-0.5, 3.0, 1.5],
    ], dtype=np.float64)

    prop = Property(
        id=1, type=12, title="PulleyNetZero",
        params={"mass": 1.0, "stiff1": 5000.0, "damp1": 20.0, "fric": 0.3}
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1, 2]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    # Current displaced configuration with elongation and relative sliding
    x_curr = np.array([
        [-0.1, -0.05, 0.02],
        [1.0, 2.0, 0.5],
        [-0.6, 3.1, 1.6],
    ], dtype=np.float64)
    v_curr = np.array([
        [1.5, 0.2, -0.1],
        [0.0, 0.0, 0.0],
        [-0.8, 1.2, 0.5],
    ], dtype=np.float64)

    fint = np.zeros((3, 3), dtype=np.float64)
    spring.forces(group, x_curr, v_curr, None, 0.001, fint, None)

    # Net force across all 3 nodes must balance to 0
    f_total = np.sum(fint, axis=0)
    np.testing.assert_allclose(f_total, [0.0, 0.0, 0.0], atol=1e-12)


def test_type12_capstan_friction():
    """Verify Capstan friction formula Fmax = F * tanh(0.5 * mu * beta) per r3def3.F."""
    model = Model()
    model.node_ids = np.array([1, 2, 3], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1, 3: 2}
    # 90-degree wrap angle:
    # N1 at (0, 1, 0), N2 at (0, 0, 0), N3 at (1, 0, 0)
    # a1 = (0, -1, 0), a2 = (-1, 0, 0)
    # cos_theta = a1 . a2 = 0 => beta = pi - pi/2 = pi/2 = 90 deg
    model.x0 = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float64)

    mu = 0.4
    K = 10000.0
    prop = Property(
        id=1, type=12, title="CapstanPulley",
        params={"mass": 1.0, "stiff1": K, "damp1": 0.0, "fric": mu}
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1, 2]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    # Apply uniform cable stretch Delta L = 0.05 => F_qs = 10000 * 0.05 = 500 N
    # And high relative velocity sliding over pulley to saturate friction
    x_curr = np.array([
        [0.0, 1.025, 0.0],
        [0.0, 0.0, 0.0],
        [1.025, 0.0, 0.0],
    ], dtype=np.float64) # Delta L = 0.025 + 0.025 = 0.05

    # With v1 = 20.0 m/s and dt = 0.001 s:
    # ddx = 20.0 m/s, ddx * dt * K = 200.0 N > fmax = 152.1 N => saturated!
    v_curr = np.array([
        [0.0, 20.0, 0.0],   # pulling node 1 away fast (vl1 = 20.0)
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],    # node 3 stationary (vl2 = 0)
    ], dtype=np.float64)    # ddx = vl1 - vl2 = 20.0

    fint = np.zeros((3, 3), dtype=np.float64)
    dt = 0.001
    spring.forces(group, x_curr, v_curr, None, dt, fint, None)

    F = 500.0
    beta = math.pi / 2.0
    expected_fmax = F * math.tanh(0.5 * mu * beta)

    # In r3def3.F:
    # Branch 1 tension T1 = F + DF
    # Branch 2 tension T2 = F - DF
    # Since sliding pulls toward 1, DF > 0, saturated at fmax
    actual_df = group.state["t12_df"][0]
    assert pytest.approx(actual_df, rel=1e-3) == expected_fmax
    assert pytest.approx(group.state["force"][0], rel=1e-3) == F


def test_type12_friction_direction():
    """Verify that friction force DF correctly opposes sliding velocity direction."""
    model = Model()
    model.node_ids = np.array([1, 2, 3], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1, 3: 2}
    # 90 deg wrap angle so beta = pi/2 > 0
    model.x0 = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float64)

    prop = Property(
        id=1, type=12, title="FrictionDir",
        params={"mass": 1.0, "stiff1": 2000.0, "fric": 0.2}
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1, 2]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    # Stretch cable: Delta L = 0.1, F = 200
    x_curr = np.array([
        [0.0, 1.05, 0.0],
        [0.0, 0.0, 0.0],
        [1.05, 0.0, 0.0],
    ], dtype=np.float64)

    # Case A: Node 1 moves away along +y (lengthening branch 1 faster => DDX > 0 => DF > 0)
    v_a = np.array([
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float64)
    fint_a = np.zeros((3, 3), dtype=np.float64)
    spring.forces(group, x_curr, v_a, None, 0.001, fint_a, None)
    df_a = group.state["t12_df"][0]
    assert df_a > 0.0

    # Reset state
    group.state["t12_dfs"][0] = 0.0
    group.state["t12_df"][0] = 0.0

    # Case B: Node 3 moves away along +x (lengthening branch 2 faster => DDX < 0 => DF < 0)
    v_b = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=np.float64)
    fint_b = np.zeros((3, 3), dtype=np.float64)
    spring.forces(group, x_curr, v_b, None, 0.001, fint_b, None)
    df_b = group.state["t12_df"][0]
    assert df_b < 0.0


def test_type12_energy_accounting():
    """Verify that internal energy tracked in eint matches mechanical work done."""
    model = Model()
    model.node_ids = np.array([1, 2, 3], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1, 3: 2}
    model.x0 = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float64)

    K = 1000.0
    prop = Property(
        id=1, type=12, title="PulleyEnergy",
        params={"mass": 1.0, "stiff1": K, "damp1": 0.0, "fric": 0.25}
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1, 2]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    # Step-by-step stretch and sliding
    dt = 0.0005
    n_steps = 40
    x = model.x0.copy()
    v = np.zeros((3, 3))
    v[0, 1] = 1.0   # Node 1 moves in +y
    v[2, 0] = 0.5   # Node 3 moves in +x (slower)

    w_ext = 0.0
    fint_old = np.zeros((3, 3))
    for step in range(n_steps):
        x += v * dt
        fint = np.zeros((3, 3))
        spring.forces(group, x, v, None, dt, fint, None)
        # Trapezoidal external work done on the system (-fint is external load balancing fint)
        w_ext += np.sum(-0.5 * (fint_old + fint) * v) * dt
        fint_old = fint.copy()

    eint = group.state["eint"][0]
    # eint should match external work within 1% trapezoidal integration tolerance
    assert pytest.approx(eint, rel=1e-2) == w_ext
    assert eint > 0.0


def test_type12_rupture_limits():
    """Verify failure / rupture when elongation exceeds max_rup1."""
    model = Model()
    model.node_ids = np.array([1, 2, 3], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1, 3: 2}
    model.x0 = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float64) # L0 = 2.0

    prop = Property(
        id=1, type=12, title="PulleyRupture",
        params={"mass": 1.0, "stiff1": 1000.0, "max_rup1": 0.2}
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1, 2]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    # Displace below rupture: delta L = 0.15 < 0.2
    x1 = np.array([[0.0, 1.15, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    fint1 = np.zeros((3, 3))
    spring.forces(group, x1, None, None, 0.001, fint1, None)
    assert group.state["off"][0] == 1.0
    assert group.state["force"][0] > 0.0

    # Displace past rupture: delta L = 0.25 > 0.2
    x2 = np.array([[0.0, 1.25, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    fint2 = np.zeros((3, 3))
    spring.forces(group, x2, None, None, 0.001, fint2, None)
    assert group.state["off"][0] == 0.0
    assert group.state["force"][0] == 0.0
    np.testing.assert_allclose(fint2, 0.0)
