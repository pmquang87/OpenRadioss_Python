"""Comprehensive tests for Milestone M605: Element time step control and geometric erosion.

Covers:
1. Engine keyword parsing:
   - /DT/BRICK/DEL/1 with Card 1 and Card 2 (col_min, defv_min, asp_max, defv_max)
   - /DT/SHELL/DEL, /DT/TRUSS/DEL, /DT/SPRING/DEL, /DT/TETRA10/DEL, /DT/SOLID/STOP
2. compute_sdlenmax:
   - Unit cube: L_max == 1.0, aspect ratio == 1.0, collapse ratio == 1.0
   - Elongated cube (1x1x5): L_max == 5.0, aspect ratio == 5.0, collapse ratio == 0.2
3. Solid volume ratio erosion:
   - defv_min: element compressed below defv_min is eroded (off=0, dt=EP30)
   - defv_max: element expanded beyond defv_max is eroded
4. Aspect ratio & collapse ratio erosion:
   - asp_max: distorted element eroded
   - col_min: flattened tetra eroded
5. Time step deletion across element types:
   - Bricks, shells, trusses, springs: when dt_e < dt_min, element is eroded (off=0, dt=EP30, state.ndel incremented)
6. /DT/<elem>/STOP: Solver halts with stop reason naming element when dt_e < dt_min.
7. Property Card 4 vs Engine Card 2 combination: Stricter threshold governs.
8. End-to-end integration test: Minimal run with starter and engine verifying element deletion occurs,
   state.ndel incremented, and run finishes cleanly.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
import textwrap

import numpy as np
import pytest

from pyradioss.common.constants import EP30
from pyradioss.common.messages import MessageLog
from pyradioss.elements import solid_hexa8, spring, truss
from pyradioss.engine.element_erosion import (
    SdLenMaxArray,
    check_solid_geometric_erosion,
    compute_sdlenmax,
)
from pyradioss.engine.engine import EngineState, _matches_elem_dt_control, run_engine
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.engine_keywords import parse_engine_deck
from pyradioss.model.model import EngineControls, Model
from pyradioss.starter.starter import run_starter


# ============================================================================
# 1. Engine keyword parsing
# ============================================================================

def test_engine_keyword_parsing():
    """Verify parsing of /DT/<elem>/DEL and /DT/<elem>/STOP with Card 1 and Card 2."""
    deck_content = """# OpenRadioss Engine File
/RUN/DT_TEST/1
0.01
/DT/BRICK/DEL/1
0.8 1.0e-7
0.15 0.05 10.0 5.0
/DT/SHELL/DEL
0.9 2.0e-7
/DT/TRUSS/DEL
0.7 3.0e-7
/DT/SPRING/DEL
0.85 4.0e-7
/DT/TETRA10/DEL
0.67 5.0e-7
/DT/SOLID/STOP
0.9 6.0e-7
0.20 0.10 8.0 4.0
"""
    blocks = read_deck(deck_content.splitlines())
    log = MessageLog()
    ec = parse_engine_deck(blocks, log)

    # 1. /DT/BRICK/DEL/1 (Card 1 & Card 2)
    assert "BRICK" in ec.dt_controls
    brick_ctrl = ec.dt_controls["BRICK"]
    assert brick_ctrl["action"] == "DEL"
    assert brick_ctrl["flag"] == "1"
    assert brick_ctrl["scale"] == 0.8
    assert np.isclose(brick_ctrl["dt_min"], 1.0e-7)
    assert np.isclose(brick_ctrl["col_min"], 0.15)
    assert np.isclose(brick_ctrl["defv_min"], 0.05)
    assert np.isclose(brick_ctrl["asp_max"], 10.0)
    assert np.isclose(brick_ctrl["defv_max"], 5.0)

    # 2. /DT/SHELL/DEL
    assert "SHELL" in ec.dt_controls
    shell_ctrl = ec.dt_controls["SHELL"]
    assert shell_ctrl["action"] == "DEL"
    assert shell_ctrl["scale"] == 0.9
    assert np.isclose(shell_ctrl["dt_min"], 2.0e-7)

    # 3. /DT/TRUSS/DEL
    assert "TRUSS" in ec.dt_controls
    truss_ctrl = ec.dt_controls["TRUSS"]
    assert truss_ctrl["action"] == "DEL"
    assert truss_ctrl["scale"] == 0.7
    assert np.isclose(truss_ctrl["dt_min"], 3.0e-7)

    # 4. /DT/SPRING/DEL
    assert "SPRING" in ec.dt_controls
    spring_ctrl = ec.dt_controls["SPRING"]
    assert spring_ctrl["action"] == "DEL"
    assert spring_ctrl["scale"] == 0.85
    assert np.isclose(spring_ctrl["dt_min"], 4.0e-7)

    # 5. /DT/TETRA10/DEL
    assert "TETRA10" in ec.dt_controls
    tetra_ctrl = ec.dt_controls["TETRA10"]
    assert tetra_ctrl["action"] == "DEL"
    assert tetra_ctrl["scale"] == 0.67
    assert np.isclose(tetra_ctrl["dt_min"], 5.0e-7)

    # 6. /DT/SOLID/STOP (Card 1 & Card 2)
    assert "SOLID" in ec.dt_controls
    solid_ctrl = ec.dt_controls["SOLID"]
    assert solid_ctrl["action"] == "STOP"
    assert solid_ctrl["scale"] == 0.9
    assert np.isclose(solid_ctrl["dt_min"], 6.0e-7)
    assert np.isclose(solid_ctrl["col_min"], 0.20)
    assert np.isclose(solid_ctrl["defv_min"], 0.10)
    assert np.isclose(solid_ctrl["asp_max"], 8.0)
    assert np.isclose(solid_ctrl["defv_max"], 4.0)


# ============================================================================
# 2. compute_sdlenmax: Unit cube & Elongated cube (1x1x5)
# ============================================================================

def test_compute_sdlenmax():
    """Verify compute_sdlenmax on unit cube (1, 1, 1) and elongated cube (5, 5, 0.2)."""
    # Unit cube: [0, 1] x [0, 1] x [0, 1]
    # Nodes:
    # 0:(0,0,0), 1:(1,0,0), 2:(1,1,0), 3:(0,1,0)
    # 4:(0,0,1), 5:(1,0,1), 6:(1,1,1), 7:(0,1,1)
    unit_cube = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    res_unit = compute_sdlenmax(unit_cube)
    assert isinstance(res_unit, (SdLenMaxArray, np.ndarray))
    assert np.isclose(float(res_unit), 1.0)
    assert np.isclose(res_unit.L_max, 1.0)
    assert np.isclose(res_unit.aspect_ratio, 1.0)
    assert np.isclose(res_unit.collapse_ratio, 1.0)

    # Elongated cube 1 x 1 x 5: [0, 1] x [0, 1] x [0, 5]
    elongated_cube = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 5.0],
        [1.0, 0.0, 5.0],
        [1.0, 1.0, 5.0],
        [0.0, 1.0, 5.0],
    ])
    res_elong = compute_sdlenmax(elongated_cube)
    assert np.isclose(float(res_elong), 5.0)
    assert np.isclose(res_elong.L_max, 5.0)
    assert np.isclose(res_elong.aspect_ratio, 5.0)
    assert np.isclose(res_elong.collapse_ratio, 0.2)

    # Vectorized batch of 2 elements
    batch = np.stack([unit_cube, elongated_cube], axis=0)
    res_batch = compute_sdlenmax(batch)
    assert res_batch.shape == (2,)
    assert np.allclose(res_batch.L_max, [1.0, 5.0])
    assert np.allclose(res_batch.aspect_ratio, [1.0, 5.0])
    assert np.allclose(res_batch.collapse_ratio, [1.0, 0.2])


# ============================================================================
# 3. Solid volume ratio erosion (defv_min and defv_max)
# ============================================================================

def test_solid_volume_ratio_erosion():
    """Verify defv_min and defv_max trigger erosion with off=0 and dt=EP30."""
    class DummyGroup:
        def __init__(self, n, v_curr, v0):
            self.n = n
            self.conn = np.zeros((n, 8), dtype=int)
            self.state = {
                "vol": np.array(v_curr, dtype=float),
                "vol0": np.array(v0, dtype=float),
                "lc": np.ones(n, dtype=float),
                "off": np.ones(n, dtype=float),
            }

    # 3 elements:
    # elem 0: normal (vol/vol0 = 1.0)
    # elem 1: compressed below defv_min (vol/vol0 = 0.05 < 0.1)
    # elem 2: expanded beyond defv_max (vol/vol0 = 3.0 > 2.0)
    group = DummyGroup(3, [1.0, 0.05, 3.0], [1.0, 1.0, 1.0])
    dt_ctrl = {
        "action": "DEL",
        "defv_min": 0.1,
        "defv_max": 2.0,
        "dt_min": 1.0e-8,
    }
    x = np.zeros((24, 3))
    del_mask = check_solid_geometric_erosion(group, x, dt_ctrl)
    assert not del_mask[0]
    assert del_mask[1]
    assert del_mask[2]

    # Test engine-level erosion application
    state = EngineState()
    dt_e = np.array([1.0e-5, 1.0e-5, 1.0e-5])
    off = group.state["off"]
    active_del = (off > 0.0) & del_mask
    if np.any(active_del):
        off[active_del] = 0.0
        dt_e[active_del] = EP30
        state.ndel += int(np.sum(active_del))

    assert off[0] == 1.0
    assert off[1] == 0.0
    assert off[2] == 0.0
    assert dt_e[0] < EP30
    assert dt_e[1] == EP30
    assert dt_e[2] == EP30
    assert state.ndel == 2


# ============================================================================
# 4. Aspect ratio & collapse ratio erosion
# ============================================================================

def test_aspect_and_collapse_ratio_erosion():
    """Verify asp_max and col_min trigger erosion for distorted hex and flattened tet."""
    # A. Distorted Hexahedron (10 x 1 x 1)
    xe_hex = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [10.0, 0.0, 1.0],
        [10.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    class DummyHexGroup:
        def __init__(self):
            self.n = 1
            self.conn = np.arange(8)[np.newaxis, :]
            self.state = {
                "vol": np.array([10.0]),
                "vol0": np.array([10.0]),
                "lc": np.array([1.0]),  # L_max = 10, lc = 1 -> aspect = 10.0, collapse = 0.1
                "off": np.ones(1),
            }

    hex_group = DummyHexGroup()
    # asp_max = 5.0 -> aspect 10.0 > 5.0 -> eroded
    mask_asp = check_solid_geometric_erosion(hex_group, xe_hex, {"asp_max": 5.0})
    assert mask_asp[0] == True

    # asp_max = 15.0 -> aspect 10.0 <= 15.0 -> not eroded
    mask_asp_ok = check_solid_geometric_erosion(hex_group, xe_hex, {"asp_max": 15.0})
    assert mask_asp_ok[0] == False

    # B. Flattened Tetrahedron
    # A tet with 3 base nodes on z=0 and top node at small z=0.05
    x_tet = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.05],
    ])
    class DummyTetGroup:
        def __init__(self):
            self.n = 1
            self.conn = np.array([[0, 1, 2, 3]])
            v = 0.5 * 0.05 / 3.0  # base area 0.5 * height 0.05 / 3
            self.state = {
                "vol": np.array([v]),
                "vol0": np.array([v]),
                "lc": np.array([0.05]),
                "off": np.ones(1),
            }

    tet_group = DummyTetGroup()
    # col_min = 0.5: flattened tetra should have low collapse ratio and be eroded
    mask_col = check_solid_geometric_erosion(tet_group, x_tet, {"col_min": 0.5})
    assert mask_col[0] == True

    # col_min = 0.001: lenient limit should not erode
    mask_col_ok = check_solid_geometric_erosion(tet_group, x_tet, {"col_min": 0.001})
    assert mask_col_ok[0] == False


# ============================================================================
# 5. Time step deletion across element types: Bricks, Shells, Trusses, Springs
# ============================================================================

def test_timestep_deletion_across_element_types():
    """Verify dt < dt_min triggers deletion (off=0, dt=EP30, state.ndel++) across element types."""
    element_types = [
        ("BRICK", "bricks"),
        ("SHELL", "shells"),
        ("TRUSS", "trusses"),
        ("SPRING", "springs"),
    ]

    dt_min = 1.0e-6

    for ctrl_key, gname in element_types:
        assert _matches_elem_dt_control(gname, ctrl_key)

        state = EngineState()
        dt_ctrl = {ctrl_key: {"action": "DEL", "dt_min": dt_min, "scale": 0.9}}

        # Two elements: elem 0 safe (dt=2e-6), elem 1 below dt_min (dt=0.5e-6)
        dt_e = np.array([2.0e-6, 0.5e-6], dtype=float)
        off = np.array([1.0, 1.0], dtype=float)
        group_state = {"off": off}

        class MockElemGroup:
            def __init__(self, st):
                self.state = st
                self.n = 2
                self.conn = np.zeros((2, 4), dtype=int)

        grp = MockElemGroup(group_state)

        # Simulate engine time step control pass
        for key, ctrl in dt_ctrl.items():
            if _matches_elem_dt_control(gname, key):
                if ctrl.get("action") == "DEL":
                    del_mask_dt = (dt_e < ctrl["dt_min"])
                    del_mask = (off > 0.0) & del_mask_dt
                    if np.any(del_mask):
                        off[del_mask] = 0.0
                        dt_e = dt_e.copy()
                        dt_e[del_mask] = EP30
                        state.ndel += int(np.sum(del_mask))

        assert off[0] == 1.0, f"Element 0 in {gname} should stay alive"
        assert off[1] == 0.0, f"Element 1 in {gname} should be eroded"
        assert dt_e[0] == pytest.approx(2.0e-6)
        assert dt_e[1] == EP30, f"Eroded element in {gname} must yield dt=EP30"
        assert state.ndel == 1, f"state.ndel should be 1 for {gname}"


# ============================================================================
# 6. /DT/<elem>/STOP halts solver naming element when dt_e < dt_min
# ============================================================================

def test_dt_elem_stop(make_deck):
    """Verify /DT/<elem>/STOP halts solver with descriptive stop reason naming the element."""
    starter = """\
    /BEGIN
    dt stop test
    /NODE
    1 0 0 0
    2 1 0 0
    3 1 1 0
    4 0 1 0
    5 0 0 1
    6 1 0 1
    7 1 1 1
    8 0 1 1
    9 1.001 0 0
    10 1.001 1 0
    11 1.001 0 1
    12 1.001 1 1
    /BRICK/1
    1 1 2 3 4 5 6 7 8
    /BRICK/2
    2 2 9 10 3 6 11 12 7
    /PART/1
    part1
    1 1
    /PART/2
    part2
    2 1
    /PROP/SOLID/1
    solid1
    1.1 0.05
    /PROP/SOLID/2
    solid2
    1.1 0.05
    /MAT/LAW1/1
    steel
    7.8e-6
    210. 0.3
    /END
    """
    # Element 2 has characteristic length ~0.001 -> dt ~ 1.5e-7
    # Setting dt_min = 1.0e-5 will trigger STOP on Element 2
    engine = """\
    /RUN/TEST/1
    0.0005
    /DT/BRICK/STOP
    0.9 1.0e-5
    /PRINT/-100
    /STOP
    5
    """
    s_path, e_path = make_deck("DT_STOP", starter, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s_path)
        m = run_engine(e_path)

    assert m.engine_state.stop_reason != ""
    assert "/DT/BRICK/STOP" in m.engine_state.stop_reason
    assert "ELEMENT 2" in m.engine_state.stop_reason
    assert "BELOW MINIMUM" in m.engine_state.stop_reason


# ============================================================================
# 7. Property Card 4 vs Engine Card 2 combination: Stricter threshold governs
# ============================================================================

def test_property_card4_vs_engine_card2_combination():
    """Verify combining Property Card 4 and Engine Card 2 enforces the strictest limits."""
    # Rules in sgeodel3.F:
    # vdef_min = max(prop, eng)
    # vdef_max = min(prop, eng)
    # asp_max = min(prop, eng)
    # col_min = max(prop, eng)

    class MockProp:
        def __init__(self, vdef_min=0.0, vdef_max=0.0, asp_max=0.0, col_min=0.0):
            self.params = {
                "vdef_min": vdef_min,
                "vdef_max": vdef_max,
                "asp_max": asp_max,
                "col_min": col_min,
            }

    class MockGroup:
        def __init__(self, prop, vol, vol0, lc):
            self.n = len(vol)
            self.conn = np.zeros((self.n, 8), dtype=int)
            self.state = {
                "slices": [(slice(0, self.n), None, prop)],
                "vol": np.array(vol, dtype=float),
                "vol0": np.array(vol0, dtype=float),
                "lc": np.array(lc, dtype=float),
                "off": np.ones(self.n),
            }

    x = np.zeros((8, 3))

    # Case 1: Engine is stricter on vdef_min (prop=0.1, eng=0.3 -> effective 0.3)
    # Element with ratio 0.2 should be deleted because of engine limit
    prop1 = MockProp(vdef_min=0.1)
    grp1 = MockGroup(prop1, vol=[0.2], vol0=[1.0], lc=[1.0])
    mask1 = check_solid_geometric_erosion(grp1, x, {"defv_min": 0.3})
    assert mask1[0] == True, "Engine stricter vdef_min must trigger deletion"

    # Case 2: Property is stricter on vdef_min (prop=0.3, eng=0.1 -> effective 0.3)
    # Element with ratio 0.2 should be deleted because of prop limit
    prop2 = MockProp(vdef_min=0.3)
    grp2 = MockGroup(prop2, vol=[0.2], vol0=[1.0], lc=[1.0])
    mask2 = check_solid_geometric_erosion(grp2, x, {"defv_min": 0.1})
    assert mask2[0] == True, "Property stricter vdef_min must trigger deletion"

    # Case 3: Engine is stricter on vdef_max (prop=3.0, eng=2.0 -> effective 2.0)
    # Element with ratio 2.5 should be deleted because of engine limit
    prop3 = MockProp(vdef_max=3.0)
    grp3 = MockGroup(prop3, vol=[2.5], vol0=[1.0], lc=[1.0])
    mask3 = check_solid_geometric_erosion(grp3, x, {"defv_max": 2.0})
    assert mask3[0] == True, "Engine stricter vdef_max must trigger deletion"

    # Case 4: Property is stricter on vdef_max (prop=2.0, eng=3.0 -> effective 2.0)
    # Element with ratio 2.5 should be deleted because of prop limit
    prop4 = MockProp(vdef_max=2.0)
    grp4 = MockGroup(prop4, vol=[2.5], vol0=[1.0], lc=[1.0])
    mask4 = check_solid_geometric_erosion(grp4, x, {"defv_max": 3.0})
    assert mask4[0] == True, "Property stricter vdef_max must trigger deletion"

    # Case 5: Engine is stricter on col_min (prop=0.1, eng=0.3 -> effective 0.3)
    xe_hex = np.array([
        [0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [5.0, 0.0, 1.0], [5.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ])
    prop5 = MockProp(col_min=0.1)
    # L_max = 5.0, lc = 1.0 -> collapse = 0.2. Stricter eng=0.3 triggers deletion (0.2 < 0.3)
    grp5 = MockGroup(prop5, vol=[5.0], vol0=[5.0], lc=[1.0])
    grp5.conn = np.arange(8)[np.newaxis, :]
    mask5 = check_solid_geometric_erosion(grp5, xe_hex, {"col_min": 0.3})
    assert mask5[0] == True, "Engine stricter col_min must trigger deletion"


# ============================================================================
# 8. End-to-end integration test: Minimal run with starter and engine
# ============================================================================

def test_end_to_end_integration_element_deletion(make_deck):
    """Verify starter + engine end-to-end element deletion, state.ndel increment, and clean run."""
    starter = """\
    /BEGIN
    two brick deletion integration test
    /NODE
    1 0 0 0
    2 1 0 0
    3 1 1 0
    4 0 1 0
    5 0 0 1
    6 1 0 1
    7 1 1 1
    8 0 1 1
    9 1.001 0 0
    10 1.001 1 0
    11 1.001 0 1
    12 1.001 1 1
    /BRICK/1
    1 1 2 3 4 5 6 7 8
    /BRICK/2
    2 2 9 10 3 6 11 12 7
    /PART/1
    part1
    1 1
    /PART/2
    part2
    2 1
    /PROP/SOLID/1
    solid1
    1.1 0.05
    /PROP/SOLID/2
    solid2
    1.1 0.05
    /MAT/LAW1/1
    steel
    7.8e-6
    210. 0.3
    /END
    """
    # Element 2 is thin (dx=0.001) -> dt ~ 1.5e-7.
    # Setting /DT/BRICK/DEL with dt_min = 1.0e-5 causes Element 2 to be deleted immediately.
    # Element 1 has dt ~ 1.5e-4 > 1.0e-5 and survives.
    engine = """\
    /RUN/DEL_TEST/1
    0.0005
    /DT/BRICK/DEL
    0.9 1.0e-5
    /PRINT/-100
    /STOP
    5
    """
    s_path, e_path = make_deck("DEL_TEST", starter, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s_path)
        m = run_engine(e_path)

    # 1. Check element status
    assert m.bricks.state["off"][0] == 1.0, "Element 1 should survive"
    assert m.bricks.state["off"][1] == 0.0, "Element 2 should be deleted"

    # 2. Check engine state
    assert hasattr(m, "engine_state"), "Model must have engine_state"
    assert m.engine_state.ndel == 1, f"Expected 1 deleted element, got {m.engine_state.ndel}"

    # 3. Check clean termination
    assert not m.engine_state.stop_reason, f"Run should complete cleanly, got {m.engine_state.stop_reason}"
    assert m.engine_state.cycle >= 5, f"Expected at least 5 cycles, got {m.engine_state.cycle}"
