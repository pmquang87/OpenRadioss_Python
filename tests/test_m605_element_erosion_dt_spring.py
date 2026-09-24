"""Tests for Milestone M605: element erosion helper, engine keywords, and spring erosion."""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.common.constants import EP30
from pyradioss.elements import spring
from pyradioss.engine.element_erosion import (
    check_solid_geometric_erosion,
    compute_sdlenmax,
)
from pyradioss.input.deck_reader import Card, KeywordBlock
from pyradioss.model.model import EngineControls, Model


def test_compute_sdlenmax_unit_cube():
    """Verify compute_sdlenmax on a unit cube returns 1.0."""
    # 8 nodes of a unit cube: [0, 1] x [0, 1] x [0, 1]
    # bottom face: 0:(0,0,0), 1:(1,0,0), 2:(1,1,0), 3:(0,1,0)
    # top face:    4:(0,0,1), 5:(1,0,1), 6:(1,1,1), 7:(0,1,1)
    xe = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    # 2D input (single element)
    l_max = compute_sdlenmax(xe)
    assert np.isclose(float(l_max), 1.0)

    # 3D input (batch of 2 elements)
    xe_batch = np.stack([xe, xe * 2.0], axis=0)
    l_max_batch = compute_sdlenmax(xe_batch)
    assert l_max_batch.shape == (2,)
    assert np.isclose(l_max_batch[0], 1.0)
    assert np.isclose(l_max_batch[1], 2.0)


def test_check_solid_geometric_erosion_volume_ratio():
    """Verify volume ratio limits defv_min and defv_max trigger deletion."""
    class DummyGroup:
        def __init__(self, n, v_curr, v0):
            self.n = n
            self.conn = np.zeros((n, 8), dtype=int)
            self.state = {
                "vol": np.array(v_curr, dtype=float),
                "vol0": np.array(v0, dtype=float),
                "lc": np.ones(n, dtype=float),
            }

    # 3 elements: normal, under compressed (del by defv_min), over expanded (del by defv_max)
    group = DummyGroup(3, [1.0, 0.05, 3.0], [1.0, 1.0, 1.0])
    dt_ctrl = {
        "defv_min": 0.1,
        "defv_max": 2.0,
    }
    x = np.zeros((24, 3))
    del_mask = check_solid_geometric_erosion(group, x, dt_ctrl)
    assert del_mask[0] == False
    assert del_mask[1] == True
    assert del_mask[2] == True


def test_check_solid_geometric_erosion_aspect_and_collapse_hex():
    """Verify aspect ratio and collapse ratio deletion on hexahedra."""
    # Stretched box 10 x 1 x 1
    xe = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [10.0, 0.0, 1.0],
        [10.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    x = xe
    conn = np.arange(8)[np.newaxis, :]

    class DummyGroup:
        def __init__(self):
            self.n = 1
            self.conn = conn
            self.state = {
                "vol": np.array([10.0]),
                "vol0": np.array([10.0]),
                "lc": np.array([1.0]),  # lc = 1.0, l_max = 10.0 -> asp = 10.0, col = 0.1
            }

    group = DummyGroup()

    # With asp_max = 5.0 -> deleted
    mask_asp = check_solid_geometric_erosion(group, x, {"asp_max": 5.0})
    assert mask_asp[0] == True

    # With asp_max = 15.0 -> not deleted
    mask_asp_ok = check_solid_geometric_erosion(group, x, {"asp_max": 15.0})
    assert mask_asp_ok[0] == False

    # With col_min = 0.2 -> deleted (since col = 0.1 < 0.2)
    mask_col = check_solid_geometric_erosion(group, x, {"col_min": 0.2})
    assert mask_col[0] == True

    # With col_min = 0.05 -> not deleted (col = 0.1 >= 0.05)
    mask_col_ok = check_solid_geometric_erosion(group, x, {"col_min": 0.05})
    assert mask_col_ok[0] == False


def test_check_solid_geometric_erosion_aspect_and_collapse_tet():
    """Verify aspect and collapse ratio deletion on tetrahedra."""
    # 4-node tet
    conn = np.array([[0, 1, 2, 3]])
    x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    class DummyGroup:
        def __init__(self):
            self.n = 1
            self.conn = conn
            self.state = {
                "vol": np.array([1.0 / 6.0]),
                "vol0": np.array([1.0 / 6.0]),
                "lc": np.array([0.5]),
            }

    group = DummyGroup()
    # al = sqrt((1/6) / 0.5^3) = sqrt(4/3) = 1.1547
    # asp = 1.24 * sqrt(3) * al = 1.24 * 2.0 = 2.48
    # col = 1.0 / asp = 1.0 / 2.48 = 0.4032

    del_asp = check_solid_geometric_erosion(group, x, {"asp_max": 2.0})
    assert del_asp[0] == True

    del_col = check_solid_geometric_erosion(group, x, {"col_min": 0.5})
    assert del_col[0] == True

    del_none = check_solid_geometric_erosion(group, x, {"asp_max": 5.0, "col_min": 0.2})
    assert del_none[0] == False


def test_check_solid_geometric_erosion_property_combination():
    """Verify property-level limits are combined with dt_ctrl."""
    class DummyProp:
        def __init__(self):
            self.params = {
                "vdef_min": 0.2,
                "vdef_max": 3.0,
                "asp_max": 8.0,
                "col_min": 0.1,
            }

    class DummyGroup:
        def __init__(self):
            self.n = 1
            self.conn = np.zeros((1, 8), dtype=int)
            self.state = {
                "slices": [(slice(0, 1), None, DummyProp())],
                "vol": np.array([0.15]),
                "vol0": np.array([1.0]),
            }

    group = DummyGroup()
    x = np.zeros((8, 3))
    # dt_ctrl has defv_min = 0.1, but prop has 0.2 -> combined vdef_min = max(0.2, 0.1) = 0.2
    # vol/vol0 = 0.15 < 0.2 -> deleted
    del_mask = check_solid_geometric_erosion(group, x, {"defv_min": 0.1})
    assert del_mask[0] == True


def test_spring_erosion():
    """Verify spring element off initialization and erosion force/dt zeroing."""
    class DummyGroup:
        def __init__(self):
            self.n = 2
            self.conn = np.array([[0, 1], [1, 2]])
            self.state = {
                "slices": [],
            }

    class DummyModel:
        def __init__(self):
            self.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    grp = DummyGroup()
    mdl = DummyModel()
    spring.init_group(grp, mdl, None)

    assert "off" in grp.state
    assert np.all(grp.state["off"] == 1.0)

    # Set k and mass for both springs
    grp.state["k"] = np.array([100.0, 100.0])
    grp.state["mass"] = np.array([1.0, 1.0])
    grp.state["cdamp"] = np.array([0.0, 0.0])

    # Deactivate spring 1: off[1] = 0.0
    grp.state["off"][1] = 0.0

    x = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [2.1, 0.0, 0.0]])
    v = np.zeros_like(x)
    fint = np.zeros_like(x)

    dtc = spring.forces(grp, x, v, None, 0.001, fint, None)

    # Spring 0 is active: non-zero force and finite dt
    assert grp.state["force"][0] > 0.0
    assert dtc[0] < EP30

    # Spring 1 is eroded: zero force and EP30 dt
    assert grp.state["force"][1] == 0.0
    assert dtc[1] == EP30
    assert np.allclose(fint[2], 0.0)


def test_engine_keywords_dt_solid_and_card2():
    """Verify /DT/BRICK and /DT/SOLID recognition and Card 2 geometric limits parsing."""
    deck_content = """# OpenRadioss Engine File
/RUN/TEST/1
0.01
/DT/BRICK/DEL/1
0.8 1.0e-7
0.15 0.05 10.0 5.0
/DT/SOLID/STOP
0.9 2.0e-7
0.20 0.10 8.0 4.0
/DT/TETRA10/DEL
0.7 3.0e-7
"""
    from pyradioss.input.deck_reader import read_deck
    blocks = read_deck(deck_content.splitlines())
    from pyradioss.input.engine_keywords import parse_engine_deck
    from pyradioss.common.messages import MessageLog
    log = MessageLog()
    ec = parse_engine_deck(blocks, log)

    assert "BRICK" in ec.dt_controls
    brick_ctrl = ec.dt_controls["BRICK"]
    assert brick_ctrl["action"] == "DEL"
    assert brick_ctrl["scale"] == 0.8
    assert np.isclose(brick_ctrl["dt_min"], 1.0e-7)
    assert np.isclose(brick_ctrl["col_min"], 0.15)
    assert np.isclose(brick_ctrl["defv_min"], 0.05)
    assert np.isclose(brick_ctrl["asp_max"], 10.0)
    assert np.isclose(brick_ctrl["defv_max"], 5.0)

    assert "SOLID" in ec.dt_controls
    solid_ctrl = ec.dt_controls["SOLID"]
    assert solid_ctrl["action"] == "STOP"
    assert solid_ctrl["scale"] == 0.9
    assert np.isclose(solid_ctrl["dt_min"], 2.0e-7)
    assert np.isclose(solid_ctrl["col_min"], 0.20)
    assert np.isclose(solid_ctrl["defv_min"], 0.10)
    assert np.isclose(solid_ctrl["asp_max"], 8.0)
    assert np.isclose(solid_ctrl["defv_max"], 4.0)

    assert "TETRA10" in ec.dt_controls
    tetra_ctrl = ec.dt_controls["TETRA10"]
    assert tetra_ctrl["action"] == "DEL"
    assert tetra_ctrl["scale"] == 0.7
    assert np.isclose(tetra_ctrl["dt_min"], 3.0e-7)
