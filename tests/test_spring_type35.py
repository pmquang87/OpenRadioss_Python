"""
Unit tests for /PROP/TYPE35 (/PROP/STITCH / /PROP/SEW: Progressive Damage Stitch Spring).

Fortran origin:
  engine/source/elements/spring/ruser35.F
  engine/source/elements/spring/rforc3.F
  starter/source/properties/spring/hm_read_prop35.F
  starter/source/elements/spring/rini35.F
"""

import math
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import spring, spring_advanced
from pyradioss.model.entities import Property, PropType35
from pyradioss.model.model import ElementGroup, Model
from pyradioss.input.deck_reader import Card, KeywordBlock
from pyradioss.input.starter_keywords import read_prop_type35
from pyradioss.input.prop_reader import parse_stitch


class MockFunction:
    """Mock Radioss function curve."""
    def __init__(self, fid: int, x_pts, y_pts):
        self.id = fid
        self.x = np.array(x_pts, dtype=np.float64)
        self.y = np.array(y_pts, dtype=np.float64)

    def eval(self, val: float) -> float:
        return float(np.interp(val, self.x, self.y))


def test_type35_elastic_loading():
    """Verify TYPE35 elastic loading, displacement integration, and energy accounting."""
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

    # L0 = 1.0, AMAS = 2.0, ELASTIF = 1000.0, XLIM1 = 10.0 (no rupture)
    prop = Property(
        id=1, type=35, title="StitchElastic",
        params={
            "amas": 2.0,
            "elastif": 1000.0,
            "xlim1": 10.0,
            "fscal": 1.0,
        }
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    # Step 1: Move node 2 with vx = 1.0 for dt = 0.01 (DX = 0.01)
    x = np.array([[0.0, 0.0, 0.0], [1.01, 0.0, 0.0]], dtype=np.float64)
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    vr = np.zeros_like(x)
    dt = 0.01

    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    dt_crit = spring.forces(group, x, v, vr, dt, fint, mint)

    # DX = 0.01 / 1.01 = 0.00990099...
    # FX = 1000.0 * DX * 1.0
    expected_dx = 0.01 * 1.0 / 1.01
    expected_fx = 1000.0 * expected_dx
    assert group.state["force"][0] == pytest.approx(expected_fx, rel=1e-5)
    assert group.state["t35_uvar1"][0] == pytest.approx(expected_dx, rel=1e-5)
    assert group.state["t35_uvar2"][0] == pytest.approx(1.0)
    assert group.state["t35_uvar3"][0] == pytest.approx(0.0)
    assert fint[0, 0] == pytest.approx(expected_fx, rel=1e-5)
    assert fint[1, 0] == pytest.approx(-expected_fx, rel=1e-5)

    # Energy: 0.5 * (0 + expected_fx) * 1.0 * 0.01
    assert group.state["eint"][0] == pytest.approx(0.5 * expected_fx * 0.01, rel=1e-5)

    # dt_crit check: omega = 2 * sqrt( (1000/1.01) / 2 )
    expected_k = 1000.0 / 1.01
    expected_omega = 2.0 * math.sqrt(expected_k / 2.0)
    expected_dt = 2.0 / expected_omega
    assert dt_crit[0] == pytest.approx(expected_dt, rel=1e-5)


def test_type35_yield_capping_with_functions():
    """Verify yield capping with initial user functions f1 (tension) and f2 (compression)."""
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

    # Functions:
    # 10: initial tension capped at 15.0
    # 20: initial compression capped at -12.0
    model.functions = {
        10: MockFunction(10, [0.0, 1.0], [15.0, 15.0]),
        20: MockFunction(20, [-1.0, 0.0], [-12.0, -12.0]),
    }

    prop = Property(
        id=1, type=35, title="StitchYield",
        params={
            "amas": 1.0,
            "elastif": 10000.0,
            "xlim1": 10.0,
            "fun_a1": 10,
            "fun_b1": 20,
            "fscal": 1.0,
        }
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    # Step 1: Tension loading. Elastic force would be 10000 * 0.01 = 100, but capped at 15.0
    x = np.array([[0.0, 0.0, 0.0], [1.01, 0.0, 0.0]], dtype=np.float64)
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    vr = np.zeros_like(x)
    dt = 0.01

    fint = np.zeros((2, 3))
    spring.forces(group, x, v, vr, dt, fint, np.zeros((2, 3)))
    assert group.state["force"][0] == pytest.approx(15.0)

    # Step 2: Compression loading. Large negative velocity. Force clamped at -12.0
    v = np.array([[0.0, 0.0, 0.0], [-5.0, 0.0, 0.0]], dtype=np.float64)
    x = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]], dtype=np.float64)
    spring.forces(group, x, v, vr, dt, fint, np.zeros((2, 3)))
    assert group.state["force"][0] == pytest.approx(-12.0)


def test_type35_progressive_damage_delay_and_rupture():
    """Verify damage delay accumulation (D2), degradation (D1), and rupture (XLIM1)."""
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

    # Functions:
    # 1: initial tension (50.0)
    # 2: initial comp (-50.0)
    # 3: final tension (25.0)
    # 4: final comp (-25.0)
    model.functions = {
        1: MockFunction(1, [0.0, 2.0], [50.0, 50.0]),
        2: MockFunction(2, [-2.0, 0.0], [-50.0, -50.0]),
        3: MockFunction(3, [0.0, 2.0], [25.0, 25.0]),
        4: MockFunction(4, [-2.0, 0.0], [-25.0, -25.0]),
    }

    # XLIM1 = 0.05, D1 = 0.2, D2 = 0.5, ILOAD = 0
    prop = Property(
        id=1, type=35, title="StitchDamage",
        params={
            "amas": 1.0,
            "elastif": 1000.0,
            "xlim1": 0.05,
            "damg": 0.2,   # D1
            "fdelay": 0.5, # D2
            "iload": 0,
            "fun_a1": 1,
            "fun_b1": 2,
            "fun_c1": 3,
            "fun_d1": 4,
            "fscal": 1.0,
        }
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    # Cycle 1: Displace by 0.06 (X >= XLIM1 = 0.05)
    # At start of cycle 1: UVAR3 = 0, FR_WAVE = 0
    # Yield bound is initial (FMX = 50.0).
    # At end of cycle 1: X >= XLIM1 -> FR_WAVE becomes 1.0. UVAR3 is still 0.0.
    x = np.array([[0.0, 0.0, 0.0], [1.06, 0.0, 0.0]], dtype=np.float64)
    v = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=np.float64)
    dt = 0.01  # DX = 0.01 * 6.0 / 1.06 ~ 0.0566

    fint = np.zeros((2, 3))
    spring.forces(group, x, v, np.zeros_like(x), dt, fint, np.zeros((2, 3)))
    assert group.state["t35_fr_wave"][0] == 1.0
    assert group.state["t35_uvar3"][0] == 0.0
    assert group.state["t35_uvar2"][0] == 1.0
    assert group.state["off"][0] == 1.0

    # Cycle 2: Keep X >= XLIM1.
    # At start of cycle 2: FR_WAVE == 1.0 and UVAR3 < 1.0 -> UVAR3 += D2 (0.0 + 0.5 = 0.5).
    # Since UVAR3 > 0, yield bound switches to function 3 (final tension FMX = UVAR2 * 25.0 = 25.0)!
    x = np.array([[0.0, 0.0, 0.0], [1.07, 0.0, 0.0]], dtype=np.float64)
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    spring.forces(group, x, v, np.zeros_like(x), dt, fint, np.zeros((2, 3)))
    assert group.state["t35_uvar3"][0] == pytest.approx(0.5)
    assert group.state["t35_fr_wave"][0] == 1.0
    assert group.state["t35_uvar2"][0] == 1.0
    assert group.state["off"][0] == 1.0

    # Cycle 3: Still X >= XLIM1.
    # At start of cycle 3: FR_WAVE == 1.0 and UVAR3 < 1.0 -> UVAR3 += D2 (0.5 + 0.5 = 1.0).
    # At end of cycle 3: X >= XLIM1 and UVAR3 >= 1.0 -> OFF becomes 0 (rupture)!
    x = np.array([[0.0, 0.0, 0.0], [1.08, 0.0, 0.0]], dtype=np.float64)
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    dt_crit = spring.forces(group, x, v, np.zeros_like(x), dt, fint, np.zeros((2, 3)))

    assert group.state["t35_uvar3"][0] == pytest.approx(1.0)
    assert group.state["off"][0] == 0.0
    assert group.state["force"][0] == 0.0
    assert dt_crit[0] == pytest.approx(1e30)


def test_type35_iload_flag():
    """Verify that when ILOAD = 1, damage delay only accumulates during loading (DX > 0)."""
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

    prop = Property(
        id=1, type=35, title="StitchILoad",
        params={
            "amas": 1.0,
            "elastif": 1000.0,
            "xlim1": 0.05,
            "damg": 0.2,
            "fdelay": 0.5,
            "iload": 1,  # Only accumulate damage when DX > 0
        }
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]}
    )
    spring.init_group(group, model, MessageLog())

    # Cycle 1: Extend past XLIM1
    x = np.array([[0.0, 0.0, 0.0], [1.06, 0.0, 0.0]], dtype=np.float64)
    v = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=np.float64)
    dt = 0.01
    fint = np.zeros((2, 3))
    spring.forces(group, x, v, np.zeros_like(x), dt, fint, np.zeros((2, 3)))
    assert group.state["t35_fr_wave"][0] == 1.0
    assert group.state["t35_uvar3"][0] == 0.0

    # Cycle 2: Unloading (vx < 0, DX < 0)
    # Because ILOAD = 1 and DX <= 0, UVAR3 does NOT increase!
    v = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
    x = np.array([[0.0, 0.0, 0.0], [1.055, 0.0, 0.0]], dtype=np.float64)
    spring.forces(group, x, v, np.zeros_like(x), dt, fint, np.zeros((2, 3)))
    assert group.state["t35_uvar3"][0] == 0.0  # Kept at 0.0!


def test_type35_starter_keyword_3card_parsing():
    """Verify parsing /PROP/TYPE35 3-card format (hm_read_prop35.F)."""
    model = Model()
    log = MessageLog()

    lines = [
        "1         Stitch3Card",
        f"{0.05:20.6f}{1500.0:20.6f}{0.08:20.6f}{0.0:20.6f}{100.0:20.6f}",
        f"{0.1:20.6f}{0.2:20.6f}{0.0:20.6f}{1.0:20.6f}",
        f"{10:10d}{20:10d}{30:10d}{40:10d}",
    ]
    block = KeywordBlock(
        keyword="/PROP/TYPE35/1",
        parts=["PROP", "TYPE35", "1"],
        user_id=1,
        source="deck.rad:10",
        cards=[Card(raw=l) for l in lines],
        fixed=True,
    )
    read_prop_type35(block, model, log)

    assert 1 in model.prop_type35s
    p35 = model.prop_type35s[1]
    assert p35.amas == pytest.approx(0.05)
    assert p35.elastif == pytest.approx(1500.0)
    assert p35.xlim1 == pytest.approx(0.08)
    assert p35.xk == pytest.approx(100.0)
    assert p35.damg == pytest.approx(0.1)
    assert p35.fdelay == pytest.approx(0.2)
    assert p35.fun_a1 == 10
    assert p35.fun_b1 == 20
    assert p35.fun_c1 == 30
    assert p35.fun_d1 == 40

    prop = model.properties[1]
    assert prop.type == 35
    assert prop.params["amas"] == pytest.approx(0.05)
    assert prop.params["elastif"] == pytest.approx(1500.0)


def test_type35_starter_keyword_2card_stitch_parsing():
    """Verify parsing /PROP/STITCH 2-card format (prop_p35_stitch.cfg)."""
    model = Model()
    log = MessageLog()

    lines = [
        "2         Stitch2Card",
        f"{0.02:20.6f}{2000.0:20.6f}{0.04:20.6f}{50.0:20.6f}",
        f"{11:10d}{21:10d}{31:10d}{41:10d}{0.15:20.6f}{0.25:20.6f}",
    ]
    block = KeywordBlock(
        keyword="/PROP/STITCH/2",
        parts=["PROP", "STITCH", "2"],
        user_id=2,
        source="deck.rad:20",
        cards=[Card(raw=l) for l in lines],
        fixed=True,
    )
    read_prop_type35(block, model, log)

    assert 2 in model.prop_type35s
    p35 = model.prop_type35s[2]
    assert p35.amas == pytest.approx(0.02)
    assert p35.elastif == pytest.approx(2000.0)
    assert p35.xlim1 == pytest.approx(0.04)
    assert p35.xk == pytest.approx(50.0)
    assert p35.fun_a1 == 11
    assert p35.fun_b1 == 21
    assert p35.fun_c1 == 31
    assert p35.fun_d1 == 41
    assert p35.damg == pytest.approx(0.15)
    assert p35.fdelay == pytest.approx(0.25)


def test_type35_prop_reader_stitch_parsing():
    """Verify prop_reader.parse_stitch handles both card formats."""
    log = MessageLog()
    lines = [
        "3         StitchPropReader",
        f"{0.03:20.6f}{1800.0:20.6f}{0.06:20.6f}{80.0:20.6f}",
        f"{1:10d}{2:10d}{3:10d}{4:10d}{0.12:20.6f}{0.34:20.6f}",
    ]
    block = KeywordBlock(
        keyword="/PROP/STITCH/3",
        parts=["PROP", "STITCH", "3"],
        user_id=3,
        source="deck.rad:30",
        cards=[Card(raw=l) for l in lines],
        fixed=True,
    )
    prop = parse_stitch(block, log)
    assert prop.type == 35
    assert prop.params["amas"] == pytest.approx(0.03)
    assert prop.params["elastif"] == pytest.approx(1800.0)
    assert prop.params["xlim1"] == pytest.approx(0.06)
    assert prop.params["damg"] == pytest.approx(0.12)
    assert prop.params["fdelay"] == pytest.approx(0.34)
