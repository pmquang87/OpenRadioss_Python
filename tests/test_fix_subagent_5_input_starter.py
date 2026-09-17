"""
Unit tests for Wave 2 Subagent 5: Input & Starter fixes (BUG-INP-01 to BUG-INP-08).
"""

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.card_layouts import split_fixed
from pyradioss.input.deck_reader import Card, KeywordBlock, parse_fortran_float
from pyradioss.input.deck_writer import _title_cards
from pyradioss.input.mat_reader import _is_numeric_card as mat_is_numeric_card
from pyradioss.input.prop_reader import _is_num as prop_is_num
from pyradioss.input.starter_keywords import (
    _is_numeric_card as kw_is_numeric_card,
    _ival,
    read_rwall,
    read_sect,
    read_transform,
)
from pyradioss.model import Model
from pyradioss.starter.initialization import _nodes_in_box, build_element_groups
from pyradioss.starter.starter import apply_transforms


# ---------------------------------------------------------------------------
# BUG-INP-01: Fortran compact float detection in card numeric checkers
# ---------------------------------------------------------------------------
def test_bug_inp_01_fortran_float_in_numeric_card_detectors():
    # Compact Fortran exponents without 'e'/'E', e.g. 1.234-05, 2.5+02
    card_compact = Card(raw=" 1.234-05  2.5+02  -3.4D-01")
    card_text = Card(raw=" MATERIAL_NAME  ELASTIC ")

    # prop_reader._is_num
    assert prop_is_num(card_compact) is True
    assert prop_is_num(card_text) is False

    # mat_reader._is_numeric_card
    assert mat_is_numeric_card(card_compact) is True
    assert mat_is_numeric_card(card_text) is False

    # starter_keywords._is_numeric_card
    assert kw_is_numeric_card(card_compact) is True
    assert kw_is_numeric_card(card_text) is False

    # deck_writer._title_cards
    block_numeric = KeywordBlock(
        keyword="MAT/1",
        parts=["MAT", "1"],
        user_id=1,
        cards=[card_compact],
    )
    title, body = _title_cards(block_numeric)
    # If the first card is numeric, title is blank and card is kept in body
    assert title == ""
    assert len(body) == 1


# ---------------------------------------------------------------------------
# BUG-INP-02: _ival parsing compact Fortran floats
# ---------------------------------------------------------------------------
def test_bug_inp_02_ival_fortran_float():
    assert _ival(" 10 ") == 10
    assert _ival(" 1.0E+02 ") == 100
    assert _ival(" 2.5D+01 ") == 25
    assert _ival(" 1.5+02 ") == 150
    assert _ival(" 1.0-02 ") == 0
    assert _ival("", default=42) == 42


# ---------------------------------------------------------------------------
# BUG-INP-03: read_transform_matrix with short token list
# ---------------------------------------------------------------------------
def test_bug_inp_03_read_transform_matrix_short_cards():
    model = Model()
    log = MessageLog()

    c1 = Card(raw="1 1.0 0.0 0.0 0.0 0")
    c2 = Card(raw="0.0 1.0 0.0 0.0")
    c3 = Card(raw="0.0")  # Only 1 token on card 3 (m31), m32..tz omitted

    block = KeywordBlock(
        keyword="TRANSFORM/MATRIX/1",
        parts=["TRANSFORM", "MATRIX", "1"],
        user_id=1,
        cards=[c1, c2, c3],
        fixed=False,
    )

    read_transform(block, model, log)
    assert len(model.transforms) == 1
    tr = model.transforms[0]
    assert tr[1] == "MATRIX"
    mat_3x3 = tr[3]
    # m31 should be 0.0, m32 should default to 0.0, m33 to 1.0
    assert mat_3x3[2] == (0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# BUG-INP-04: TRANSFORM/TRA node-pair vector replaces (not adds to) translation
# ---------------------------------------------------------------------------
def test_bug_inp_04_transform_tra_nodepair_replaces():
    model = Model()
    model.node_ids = np.array([1, 2, 3], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1, 3: 2}
    model.x0 = np.array([
        [0.0, 0.0, 0.0],   # Node 1
        [5.0, 10.0, 15.0], # Node 2
        [1.0, 1.0, 1.0],   # Node 3 (to transform)
    ], dtype=np.float64)

    class NodeGroup:
        id = 1
        node_idx = np.array([2], dtype=np.int64)
        node_ids = [3]

    model.node_groups[1] = NodeGroup()

    # Initial vector specified on card was (100.0, 200.0, 300.0),
    # but node pair (1, 2) is specified. Per Fortran lectrans.F:224-226,
    # the vector from node 1 to node 2 (5, 10, 15) replaces (tx, ty, tz).
    model.transforms = [
        (1, "TRA", 1, 100.0, 200.0, 300.0, 1, 2, 0, 0)
    ]

    log = MessageLog()
    apply_transforms(model, log)

    assert len(log.errors) == 0
    # Node 3 was [1.0, 1.0, 1.0] -> should be [6.0, 11.0, 16.0], NOT [106.0, 211.0, 316.0]
    assert np.allclose(model.x0[2], [6.0, 11.0, 16.0])


# ---------------------------------------------------------------------------
# BUG-INP-05: TRANSFORM/ROT & SYM requires both n1 > 0 and n2 > 0
# ---------------------------------------------------------------------------
def test_bug_inp_05_transform_rot_requires_both_nodes():
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([
        [999.0, 999.0, 999.0],  # Node 1
        [1.0, 0.0, 0.0],        # Node 2 (to transform)
    ], dtype=np.float64)

    class NodeGroup:
        id = 1
        node_idx = np.array([1], dtype=np.int64)
        node_ids = [2]

    model.node_groups[1] = NodeGroup()

    # Axis defined by points p1=(0,0,0) and p2=(0,0,1) (Z-axis).
    # Node 1 is specified (n1=1), but n2=0. Coordinates p1, p2 from card
    # must NOT be overridden by Node 1!
    model.transforms = [
        (1, "ROT", 1, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0, 1, 0, 0)
    ]

    log = MessageLog()
    apply_transforms(model, log)

    # Rotating [1, 0, 0] by 90 deg around Z-axis -> [0, 1, 0]
    assert np.allclose(model.x0[1], [0.0, 1.0, 0.0], atol=1e-6)


# ---------------------------------------------------------------------------
# BUG-INP-06: /BOX/RECTA with skew and corner nodes
# ---------------------------------------------------------------------------
def test_bug_inp_06_box_recta_skew_corner_nodes():
    model = Model()
    # 3 nodes:
    # Node 1 at (0, 0, 0), Node 2 at (10, 0, 0) - corner nodes for box in global
    # Node 3 at (0, 5, 0) in global
    model.node_ids = np.array([1, 2, 3], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1, 3: 2}
    model.x0 = np.array([
        [0.0, 0.0, 0.0],   # Node 1
        [10.0, 0.0, 0.0],  # Node 2
        [0.0, 5.0, 0.0],   # Node 3
    ], dtype=np.float64)

    # Define a skew rotated 90 deg about Z:
    # X_skew = Y_global, Y_skew = -X_global, Z_skew = Z_global
    model.skews._by_key[("SKEW", 1)] = 1
    model.skews.origins = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    model.skews.axes = np.array([
        np.eye(3),
        [[0.0, 1.0, 0.0],
         [-1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0]]
    ], dtype=np.float64)

    class DummyBox:
        id = 1
        kind = "RECTA"
        iskew = 1
        node1 = 1  # (0, 0, 0) in global -> (0, 0, 0) in skew
        node2 = 3  # (0, 5, 0) in global -> (5, 0, 0) in skew
        corner_min = np.array([0.0, 0.0, 0.0])
        corner_max = np.array([0.0, 0.0, 0.0])

    log = MessageLog()
    inside_idx = _nodes_in_box(model, DummyBox(), log, "TEST")

    # In skew coords:
    # Node 1: (0, 0, 0) -> inside [0..5, 0..0, 0..0]
    # Node 2: (0, -10, 0) -> outside
    # Node 3: (5, 0, 0) -> inside [0..5, 0..0, 0..0]
    assert 0 in inside_idx  # Node 1
    assert 2 in inside_idx  # Node 3
    assert 1 not in inside_idx  # Node 2


# ---------------------------------------------------------------------------
# BUG-INP-07: Split card layouts for RWALL_BOX, RWALL_CONE, SECT_CUT
# ---------------------------------------------------------------------------
def test_bug_inp_07_split_card_layouts():
    model = Model()
    log = MessageLog()

    # /SECT/CUT: Card 0 = Title, Card 1 = Origin (20, 20, 20), Card 2 = Normal (20, 20, 20)
    c_title = Card(raw="Cut Section Title")
    c1 = Card(raw=f"{1.0:>20.4f}{2.0:>20.4f}{3.0:>20.4f}")
    c2 = Card(raw=f"{0.0:>20.4f}{1.0:>20.4f}{0.0:>20.4f}")
    block_sect = KeywordBlock(
        keyword="SECT/CUT/1",
        parts=["SECT", "CUT", "1"],
        user_id=1,
        cards=[c_title, c1, c2],
        fixed=True,
    )
    read_sect(block_sect, model, log)
    assert 1 in model.sect_cuts
    sect = model.sect_cuts[1]
    assert np.allclose(sect.orig, [1.0, 2.0, 3.0])
    assert np.allclose(sect.normal, [0.0, 1.0, 0.0])

    # /RWALL/BOX: Card 0 = Title, Card 1 = params, Card 2 = P1, Card 3 = P2
    c_rtitle = Card(raw="Rigid Wall Box Title")
    c_r1 = Card(raw=f"{1:>10d}{0:>10d}{100.0:>10.2f}{0.0:>10.2f}")
    c_r2 = Card(raw=f"{1.0:>20.4f}{2.0:>20.4f}{3.0:>20.4f}")
    c_r3 = Card(raw=f"{4.0:>20.4f}{5.0:>20.4f}{6.0:>20.4f}")
    block_rwall = KeywordBlock(
        keyword="RWALL/BOX/1",
        parts=["RWALL", "BOX", "1"],
        user_id=1,
        cards=[c_rtitle, c_r1, c_r2, c_r3],
        fixed=True,
    )
    read_rwall(block_rwall, model, log)
    assert 1 in model.rwall_boxes
    rw = model.rwall_boxes[1]
    assert np.allclose(rw.p1, [1.0, 2.0, 3.0])
    assert np.allclose(rw.p2, [4.0, 5.0, 6.0])

    # /RWALL/CONE: Card 0 = Title, Card 1 = params, Card 2 = Apex, Card 3 = Axis + Angle
    c_ctitle = Card(raw="Rigid Wall Cone Title")
    c_c1 = Card(raw=f"{2:>10d}{0:>10d}{100.0:>10.2f}{0.0:>10.2f}")
    c_c2 = Card(raw=f"{0.0:>20.4f}{0.0:>20.4f}{0.0:>20.4f}")
    c_c3 = Card(raw=f"{0.0:>20.4f}{0.0:>20.4f}{1.0:>20.4f}{45.0:>20.4f}")
    block_cone = KeywordBlock(
        keyword="RWALL/CONE/2",
        parts=["RWALL", "CONE", "2"],
        user_id=2,
        cards=[c_ctitle, c_c1, c_c2, c_c3],
        fixed=True,
    )
    read_rwall(block_cone, model, log)
    assert 2 in model.rwall_cones
    rc = model.rwall_cones[2]
    assert np.allclose(rc.apex, [0.0, 0.0, 0.0])
    assert np.allclose(rc.axis, [0.0, 0.0, 1.0])
    assert rc.angle == pytest.approx(45.0)


# ---------------------------------------------------------------------------
# BUG-INP-08: build_element_groups drops defective elements individually
# ---------------------------------------------------------------------------
def test_bug_inp_08_build_element_groups_defective_element():
    model = Model()
    # Define nodes 1, 2, 3, 4
    model.node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1, 3: 2, 4: 3}
    model.x0 = np.zeros((4, 3))

    class DummyPart:
        id = 1
        mat_id = 1
        prop_id = 1

    class DummyMat:
        id = 1

    class DummyProp:
        id = 1
        type = 1
        params = {}

    model.parts[1] = DummyPart()
    model.materials[1] = DummyMat()
    model.properties[1] = DummyProp()

    # Raw shells:
    # Elem 101: valid nodes [1, 2, 3, 4]
    # Elem 102: invalid node [1, 2, 999, 4] (Node 999 not defined)
    model.raw_elems["SHELL"] = [
        (101, 1, [1, 2, 3, 4]),
        (102, 1, [1, 2, 999, 4]),
    ]

    log = MessageLog()
    build_element_groups(model, log)

    # Should log an error for element 102
    assert any("102" in err and "unknown node id" in err for err in log.errors)

    # But element group SHELL should NOT be dropped!
    assert hasattr(model, "shells")
    assert model.shells is not None
    assert len(model.shells.ids) == 1
    assert model.shells.ids[0] == 101
