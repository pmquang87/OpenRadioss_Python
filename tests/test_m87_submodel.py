"""
Milestone M87 — /SUBMODEL and /ENDSUB Container Architecture & Scoping.

Tests:
1. Parser for /SUBMODEL (fixed and free format) with card offsets and title
2. Node tagging and hierarchy tracking in Model across nested submodels
3. /TRANSFORM scoping with sub_id targeting nodes in specific submodels (TRA, ROT, SYM, SCA)
4. Unmatched /ENDSUB warning handling
"""

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.model.entities import Submodel
from pyradioss.starter.starter import run_starter


_MINIMAL_ELEM_DECK = """/PROP/TYPE1/1
Shell_Prop
1 1 1 0 0 0 0
1.0 1.0 1.0 0 0 0 0 0
1.0 0.833333
/MAT/LAW1/1
Linear_Elastic
7.8e-9
210000.0 0.3
/PART/1
Plate
1 1
/NODE
101 0.0 0.0 0.0
102 10.0 0.0 0.0
103 10.0 10.0 0.0
104 0.0 10.0 0.0
/SHELL/1
1 101 102 103 104
"""


def test_submodel_parsing_fixed_and_free(tmp_path):
    # Free format: /SUBMODEL/sub_id
    deck_free = f"""/BEGIN
TEST_SUBMODEL_FREE
/SUBMODEL/100
Submodel Free Format
1000 100 200 300 400 500 600
{_MINIMAL_ELEM_DECK}
/ENDSUB
/END
"""
    p1 = tmp_path / "TEST_FREE_0000.rad"
    p1.write_text(deck_free)
    log1 = MessageLog()
    m1 = run_starter(str(p1), log1)
    assert len(log1.errors) == 0
    assert 100 in m1.submodels
    sm1 = m1.submodels[100]
    assert sm1.id == 100
    assert sm1.title == "Submodel Free Format"
    assert sm1.off_def == 1000
    assert sm1.off_nod == 100
    assert sm1.off_ele == 200
    assert sm1.off_part == 300
    assert sm1.off_mat == 400
    assert sm1.off_type == 500
    assert sm1.off_sub == 600

    # Fixed format: /SUBMODEL/sub_id
    deck_fixed = f"""# RADIOSS STARTER
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_SUBMODEL_FIXED
      2021         0
/SUBMODEL/200
Submodel Fixed Format
      2000       200       300       400       500       600       700
/PROP/TYPE1/1
Shell_Prop
         1         1         1         0         0         0         0
                 1.0                 1.0                 1.0
                 1.0            0.833333
/MAT/LAW1/1
Linear_Elastic
              7.8e-9
            210000.0                 0.3
/PART/1
Plate
         1         1
/NODE
       101                 0.0                 0.0                 0.0
       102                10.0                 0.0                 0.0
       103                10.0                10.0                 0.0
       104                 0.0                10.0                 0.0
/SHELL/1
         1       101       102       103       104
/ENDSUB
/END
"""
    p2 = tmp_path / "TEST_FIXED_0000.rad"
    p2.write_text(deck_fixed)
    log2 = MessageLog()
    m2 = run_starter(str(p2), log2)
    assert len(log2.errors) == 0
    assert 200 in m2.submodels
    sm2 = m2.submodels[200]
    assert sm2.id == 200
    assert sm2.title == "Submodel Fixed Format"
    assert sm2.off_def == 2000
    assert sm2.off_nod == 200
    assert sm2.off_ele == 300
    assert sm2.off_part == 400
    assert sm2.off_mat == 500
    assert sm2.off_type == 600
    assert sm2.off_sub == 700


def test_submodel_node_tagging_and_nesting(tmp_path):
    deck = f"""/BEGIN
TEST_NESTED_SUBMODELS
/NODE
1 0.0 0.0 0.0
/SUBMODEL/1
Submodel 1
0 0 0 0 0 0 0
/NODE
2 1.0 0.0 0.0
/SUBMODEL/2
Submodel 2 (Nested)
0 0 0 0 0 0 0
/NODE
3 2.0 0.0 0.0
/ENDSUB
/NODE
4 3.0 0.0 0.0
/ENDSUB
/NODE
5 4.0 0.0 0.0
{_MINIMAL_ELEM_DECK}
/END
"""
    p = tmp_path / "TEST_NESTED_0000.rad"
    p.write_text(deck)
    log = MessageLog()
    m = run_starter(str(p), log)
    assert len(log.errors) == 0

    idx1 = m.node_index(1)
    idx2 = m.node_index(2)
    idx3 = m.node_index(3)
    idx4 = m.node_index(4)
    idx5 = m.node_index(5)

    assert m.node_submodel[idx1] == 0   # Root
    assert m.node_submodel[idx2] == 1   # Submodel 1
    assert m.node_submodel[idx3] == 2   # Submodel 2
    assert m.node_submodel[idx4] == 1   # Back to Submodel 1 after inner /ENDSUB
    assert m.node_submodel[idx5] == 0   # Root after outer /ENDSUB


def test_transform_with_submodel_scoping(tmp_path):
    # Test /TRANSFORM/TRA, ROT, SYM, SCA scoping by sub_id
    deck = f"""/BEGIN
TEST_TRANSFORM_SUBMODEL
/NODE
1 0.0 0.0 0.0
/SUBMODEL/10
Submodel 10
0 0 0 0 0 0 0
/NODE
2 0.0 0.0 0.0
3 1.0 0.0 0.0
/ENDSUB
/SUBMODEL/20
Submodel 20
0 0 0 0 0 0 0
/NODE
4 1.0 1.0 0.0
/ENDSUB
/TRANSFORM/TRA/1
Translate Submodel 10 by (5, 5, 0)
0 5.0 5.0 0.0 0 0 10
/TRANSFORM/SCA/2
Scale Submodel 20 by factor (2, 2, 1) around origin
0 2.0 2.0 1.0 0 20
{_MINIMAL_ELEM_DECK}
/END
"""
    p = tmp_path / "TEST_TR_SUB_0000.rad"
    p.write_text(deck)
    log = MessageLog()
    m = run_starter(str(p), log)
    assert len(log.errors) == 0

    idx1 = m.node_index(1)
    idx2 = m.node_index(2)
    idx3 = m.node_index(3)
    idx4 = m.node_index(4)

    # Node 1 is outside submodels -> unchanged at (0, 0, 0)
    np.testing.assert_allclose(m.x0[idx1], [0.0, 0.0, 0.0], atol=1e-6)
    # Nodes 2 & 3 are in submodel 10 -> translated by (5, 5, 0)
    np.testing.assert_allclose(m.x0[idx2], [5.0, 5.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(m.x0[idx3], [6.0, 5.0, 0.0], atol=1e-6)
    # Node 4 is in submodel 20 -> scaled to (2.0, 2.0, 0.0)
    np.testing.assert_allclose(m.x0[idx4], [2.0, 2.0, 0.0], atol=1e-6)


def test_endsub_unmatched_warning(tmp_path):
    deck = f"""/BEGIN
TEST_UNMATCHED_ENDSUB
/ENDSUB
{_MINIMAL_ELEM_DECK}
/END
"""
    p = tmp_path / "TEST_UNMATCHED_0000.rad"
    p.write_text(deck)
    log = MessageLog()
    m = run_starter(str(p), log)
    assert len(log.errors) == 0
    assert any("/ENDSUB encountered without an active /SUBMODEL" in str(w) for w in log.warnings)
