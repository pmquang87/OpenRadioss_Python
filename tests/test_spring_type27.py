"""
Unit tests for /PROP/TYPE27 (/PROP/SPR_BDAMP: Spring with Bilinear Damping and Nonlinear Exponent).

Fortran origin:
  engine/source/elements/spring/r27def3.F
  engine/source/elements/spring/rforc3.F
  starter/source/properties/spring/hm_read_prop27.F
"""

import math
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import spring, spring_advanced
from pyradioss.model.entities import Property, PropType27
from pyradioss.model.model import ElementGroup, Model
from pyradioss.input.deck_reader import Card, KeywordBlock
from pyradioss.input.starter_keywords import read_prop_spr_bdamp


class MockFunction:
    """Mock Radioss function curve."""
    def __init__(self, fid: int, x_pts, y_pts):
        self.id = fid
        self.x = np.array(x_pts, dtype=np.float64)
        self.y = np.array(y_pts, dtype=np.float64)

    def eval(self, val: float) -> float:
        return float(np.interp(val, self.x, self.y))


def test_type27_nonlinear_exponent():
    """Verify nonlinear power-law stiffness: F_K = K * sign(DL) * |DL|^n."""
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

    # K = 1000.0, n = 1.5, C = 0.0, itens = 1 (active in tension)
    prop = Property(
        id=1, type=27, title="NonlinearExpSpring",
        params={
            "mass": 4.0,
            "stiff": 1000.0,
            "damp": 0.0,
            "nexp": 1.5,
            "itens": 1,
            "ileng": 0,
        }
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    # Step 1: Displace node 2 by +0.04 (DL = 0.04)
    x = model.x0.copy()
    x[1, 0] = 1.04
    v = np.zeros_like(x)
    vr = np.zeros_like(x)
    dt = 0.01

    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    dt_crit = spring.forces(group, x, v, vr, dt, fint, mint)

    # |DL|^1.5 = (0.04)^1.5 = 0.008
    # F = 1000.0 * 0.008 = 8.0 N
    assert group.state["force"][0] == pytest.approx(8.0)
    assert fint[0, 0] == pytest.approx(8.0)
    assert fint[1, 0] == pytest.approx(-8.0)
    # Energy: 0.5 * 0.04 * 8.0 = 0.16 J
    assert group.state["eint"][0] == pytest.approx(0.16)


def test_type27_assembly_rule():
    """Verify force assembly rule (r27def3.F lines 259-265):
    If |FK| > |FD|: F = FK + FD
    Else: F = 2*FK, K_eff = 2*K, C_eff = 0
    """
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

    # Case A: |FK| > |FD|
    # K = 100.0, C = 10.0, n = 1.0, itens = 1
    # DL = 0.1, dt = 0.01 -> dv = 0.1 / 0.01 = 10.0
    # FK = 100.0 * 0.1 = 10.0
    # FD = 10.0 * 10.0 = 100.0 -> Wait, here |FK| < |FD|!
    # Let's adjust so FK = 200.0 > FD = 50.0:
    prop_a = Property(
        id=1, type=27, title="AssemblyRuleA",
        params={
            "mass": 2.0,
            "stiff": 2000.0,  # FK = 2000 * 0.1 = 200.0
            "damp": 5.0,      # FD = 5 * 10.0 = 50.0
            "nexp": 1.0,
            "itens": 1,
            "ileng": 0,
        }
    )
    group_a = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop_a)]}
    )
    spring.init_group(group_a, model, MessageLog())

    x = model.x0.copy()
    x[1, 0] = 1.1
    v = np.zeros_like(x)
    vr = np.zeros_like(x)
    dt = 0.01
    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    spring.forces(group_a, x, v, vr, dt, fint, mint)

    # |FK| = 200 > |FD| = 50 -> F = 200 + 50 = 250.0 N
    assert group_a.state["force"][0] == pytest.approx(250.0)

    # Case B: |FK| <= |FD|
    # FK = 20.0, FD = 50.0 -> F = 2 * FK = 40.0 N
    prop_b = Property(
        id=2, type=27, title="AssemblyRuleB",
        params={
            "mass": 2.0,
            "stiff": 200.0,   # FK = 200 * 0.1 = 20.0
            "damp": 5.0,      # FD = 5 * 10.0 = 50.0
            "nexp": 1.0,
            "itens": 1,
            "ileng": 0,
        }
    )
    group_b = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop_b)]}
    )
    spring.init_group(group_b, model, MessageLog())
    fint.fill(0.0)
    spring.forces(group_b, x, v, vr, dt, fint, mint)
    assert group_b.state["force"][0] == pytest.approx(40.0)


def test_type27_compression_gap():
    """Verify gap behavior in compression: inactive until DL < GAP."""
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

    # GAP = -0.05, ITENS = 0 (compression only)
    prop = Property(
        id=1, type=27, title="GapSpring",
        params={
            "mass": 1.0,
            "stiff": 1000.0,
            "damp": 0.0,
            "nexp": 1.0,
            "gap": -0.05,
            "itens": 0,
            "ileng": 0,
        }
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    x = model.x0.copy()
    v = np.zeros_like(x)
    vr = np.zeros_like(x)
    dt = 0.01
    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))

    # Test 1: DL = -0.02 > GAP (-0.05): Should be in gap (0 force)
    x[1, 0] = 0.98
    spring.forces(group, x, v, vr, dt, fint, mint)
    assert group.state["force"][0] == pytest.approx(0.0)

    # Test 2: DL = -0.08 < GAP (-0.05): Active!
    # delta = DL - GAP = -0.08 - (-0.05) = -0.03
    # FK = 1000.0 * (-0.03) = -30.0 N
    x[1, 0] = 0.92
    spring.forces(group, x, v, vr, dt, fint, mint)
    assert group.state["force"][0] == pytest.approx(-30.0)


def test_type27_force_smoothing_filter():
    """Verify first-order filter for spring force when FSMOOTH > 0 and FCUT > 0."""
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

    # FCUT = 50.0 Hz, FSMOOTH = 1
    fcut = 50.0
    dt = 0.001
    # omega_cut = 2 * pi * dt * fc = 2 * pi * 0.001 * 50 = 0.1 * pi = 0.314159265
    # alpha = omega_cut / (omega_cut + 1)
    omega_cut = 2.0 * math.pi * dt * fcut
    alpha = omega_cut / (omega_cut + 1.0)

    prop = Property(
        id=1, type=27, title="SmoothSpring",
        params={
            "mass": 1.0,
            "stiff": 1000.0,
            "damp": 0.0,
            "nexp": 1.0,
            "itens": 1,
            "fsmooth": 1,
            "fcut": fcut,
        }
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    x = model.x0.copy()
    x[1, 0] = 1.05  # DL = 0.05 -> raw F = 50.0
    v = np.zeros_like(x)
    vr = np.zeros_like(x)
    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))

    spring.forces(group, x, v, vr, dt, fint, mint)

    # F_old = 0.0, raw F = 50.0 -> F_filtered = alpha * 50.0 + (1 - alpha) * 0.0
    expected_f = alpha * 50.0
    assert group.state["force"][0] == pytest.approx(expected_f, rel=1e-5)


def test_type27_rupture():
    """Verify displacement rupture (IFAIL=1) and force rupture (IFAIL=2)."""
    model = Model()
    model.node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1, 3: 2, 4: 3}
    model.x0 = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0], [1.0, 2.0, 0.0],
    ], dtype=np.float64)

    # Element 0: IFAIL = 1 (disp rupture at delta_max = 0.05)
    prop1 = Property(
        id=1, type=27, title="DispRupture",
        params={
            "mass": 1.0, "stiff": 1000.0, "itens": 1,
            "ifail": 1, "delta_max": 0.05,
        }
    )
    # Element 1: IFAIL = 2 (force rupture at delta_max = 100.0 N)
    prop2 = Property(
        id=2, type=27, title="ForceRupture",
        params={
            "mass": 1.0, "stiff": 1000.0, "itens": 1,
            "ifail": 2, "delta_max": 100.0,
        }
    )
    group = ElementGroup(
        ids=np.array([1, 2]),
        conn=np.array([[0, 1], [2, 3]], dtype=np.int64),
        part=np.array([0, 1]),
        state={"slices": [
            (slice(0, 1), None, prop1),
            (slice(1, 2), None, prop2),
        ]}
    )
    spring.init_group(group, model, MessageLog())

    x = model.x0.copy()
    # Elem 0: DL = 0.06 > 0.05 -> Ruptures!
    x[1, 0] = 1.06
    # Elem 1: DL = 0.12 -> F_raw = 120.0 > 100.0 -> Ruptures!
    x[3, 0] = 1.12

    v = np.zeros_like(x)
    vr = np.zeros_like(x)
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    spring.forces(group, x, v, vr, 0.01, fint, mint)

    assert group.state["off"][0] == 0.0
    assert group.state["force"][0] == 0.0
    assert group.state["off"][1] == 0.0
    assert group.state["force"][1] == 0.0


def test_type27_functions():
    """Verify function-based stiffness and damping curves (fun1 and fun2)."""
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

    # Function 10 for FK: FK(delta) = 500 * delta
    model.functions[10] = MockFunction(10, [0.0, 0.1, 0.2], [0.0, 50.0, 100.0])
    # Function 20 for FD: FD(dv) = 20 * dv
    model.functions[20] = MockFunction(20, [0.0, 5.0, 10.0], [0.0, 100.0, 200.0])

    prop = Property(
        id=1, type=27, title="FuncSpring",
        params={
            "mass": 1.0,
            "stiff": 100.0,
            "damp": 10.0,
            "itens": 1,
            "fun1": 10,
            "fscale1": 2.0,  # 2x force scaling on FK
            "ascale1": 1.0,
            "fun2": 20,
            "fscale2": 1.5,  # 1.5x force scaling on FD
            "ascale2": 1.0,
        }
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    x = model.x0.copy()
    x[1, 0] = 1.1  # DL = 0.1 -> fun1 gives 50.0 * 2.0 = 100.0
    # dt = 0.01 -> dv = 0.1 / 0.01 = 10.0 -> fun2 gives 200.0 * 1.5 = 300.0
    # FK = 100.0, FD = 300.0. Since |FK| <= |FD|, F = 2 * FK = 200.0
    v = np.zeros_like(x)
    vr = np.zeros_like(x)
    dt = 0.01
    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    spring.forces(group, x, v, vr, dt, fint, mint)

    assert group.state["force"][0] == pytest.approx(200.0)


def test_type27_starter_keyword_parsing():
    """Verify /PROP/TYPE27 keyword parsing in starter_keywords.py."""
    model = Model()
    log = MessageLog()

    # Fixed card format (starts with title card)
    cards_text = [
        "Spring BDAMP Fixed Test",
        f"{1.5:20.6f}{'':30}{0:10d}{0:10d}{0:10d}{1:10d}{0:10d}",
        f"{1500.0:20.6f}{25.0:20.6f}{1.2:20.6f}{-0.08:20.6f}{0.15:20.6f}",
        f"{-0.02:20.6f}{'':50}{1:10d}{100.0:20.6f}",
        f"{10:10d}{20:10d}{1.0:20.6f}{2.0:20.6f}{1.0:20.6f}{1.5:20.6f}",
    ]
    cards = [Card(raw=t, source=f"test:{i+1}") for i, t in enumerate(cards_text)]
    block = KeywordBlock(
        keyword="/PROP/TYPE27/101",
        parts=["PROP", "TYPE27", "101"],
        user_id=101,
        fixed=True,
        cards=cards,
    )
    read_prop_spr_bdamp(block, model, log)

    assert 101 in model.prop_spr_bdamps
    p27 = model.prop_spr_bdamps[101]
    assert p27.mass == pytest.approx(1.5)
    assert p27.itens == 1
    assert p27.stiff == pytest.approx(1500.0)
    assert p27.damp == pytest.approx(25.0)
    assert p27.nexp == pytest.approx(1.2)
    assert p27.delta_min == pytest.approx(-0.08)
    assert p27.delta_max == pytest.approx(0.15)
    assert p27.gap == pytest.approx(-0.02)
    assert p27.fsmooth == 1
    assert p27.fcut == pytest.approx(100.0)
    assert p27.fct_id1 == 10
    assert p27.fct_id2 == 20
    assert p27.fscale1 == pytest.approx(2.0)
    assert p27.fscale2 == pytest.approx(1.5)

    assert 101 in model.properties
    prop = model.properties[101]
    assert isinstance(prop, Property)
    assert prop.type == 27
    assert prop.params["k"] == pytest.approx(1500.0)
    assert prop.params["c"] == pytest.approx(25.0)

    # Free format test
    block_free = KeywordBlock(
        keyword="/PROP/SPR_BDAMP/102",
        parts=["PROP", "SPR_BDAMP", "102"],
        user_id=102,
        fixed=False,
        cards=[
            Card(raw="Free format spring", source="test:1"),
            Card(raw="2.0 0 0 0 1 0", source="test:2"),
            Card(raw="3000.0 50.0 1.0 -0.1 0.2", source="test:3"),
            Card(raw="-0.05 1 50.0", source="test:4"),
            Card(raw="11 21 1.0 1.0 1.0 1.0", source="test:5"),
        ],
    )
    read_prop_spr_bdamp(block_free, model, log)
    assert 102 in model.prop_spr_bdamps
    p27_free = model.prop_spr_bdamps[102]
    assert p27_free.mass == pytest.approx(2.0)
    assert p27_free.stiff == pytest.approx(3000.0)
    assert p27_free.damp == pytest.approx(50.0)
    assert p27_free.gap == pytest.approx(-0.05)
    assert 102 in model.properties
    assert model.properties[102].type == 27

