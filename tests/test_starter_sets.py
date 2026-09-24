"""Tests for /SET/GENERAL resolution into model.node_groups and model.surfaces.

Upstream Fortran reference:
- starter/source/model/sets/hm_set.F
- starter/source/model/sets/fill_igr.F
"""

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.starter.starter import run_starter


_BASE_DECK = """\
/MAT/LAW1/1
Elastic
              7.8e-9
            210000.0                 0.3
/PROP/TYPE1/1
Shell_Prop
         1         1         1         0         0         0         0
                 1.0                 1.0                 1.0
                 1.0            0.833333
/PART/1
Part_one
         1         1
/NODE
       101                 0.0                 0.0                 0.0
       102                10.0                 0.0                 0.0
       103                10.0                10.0                 0.0
       104                 0.0                10.0                 0.0
/SHELL/1
         1       101       102       103       104
"""


def _run(tmp_path, deck, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="utf-8")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


def test_set_general_node_group_resolved(tmp_path):
    """Assert /SET/GENERAL/1 with KEY NODE resolves to model.node_groups[1] and is accessible by /BCS/1."""
    deck = f"""\
/BEGIN
TEST_SET_GENERAL_NODE
{_BASE_DECK}
/SET/GENERAL/1
Node Group From General Set
NODE
101 102
/BCS/1
Boundary condition on node group 1
111 111 0 1
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Starter errors: {log.errors}"
    assert 1 in model.node_groups
    ng = model.node_groups[1]
    assert ng.id == 1
    assert ng.title == "Node Group From General Set"
    assert ng.node_idx is not None
    # User node IDs 101, 102 correspond to 0-based indices 0, 1
    assert set(ng.node_idx) == {0, 1}
    assert any(bc.id == 1 and bc.grnod_id == 1 for bc in model.bcs)


def test_set_general_node_group_inline_key(tmp_path):
    """Test /SET/GENERAL/1 with KEY and IDs on the same card or header."""
    deck = f"""\
/BEGIN
TEST_SET_GENERAL_INLINE
{_BASE_DECK}
/SET/GENERAL/10
Inline Node Set
NODE 103 104
/BCS/2
BCS on group 10
111 111 0 10
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Starter errors: {log.errors}"
    assert 10 in model.node_groups
    ng = model.node_groups[10]
    assert ng.node_idx is not None
    assert set(ng.node_idx) == {2, 3}


def test_set_general_node_group_rbody(tmp_path):
    """Test /SET/GENERAL node group referenced by /RBODY."""
    deck = f"""\
/BEGIN
TEST_SET_GENERAL_RBODY
{_BASE_DECK}
/NODE
       200                 5.0                 5.0                 0.0
/SET/GENERAL/3
Rigid Body Secondary Nodes
NODE
101 102 103 104
/RBODY/2
Rigid Body 2
200 3 0 0 0
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Starter errors: {log.errors}"
    assert 3 in model.node_groups
    assert any(rb.id == 2 and rb.grnod_id == 3 for rb in model.rbodies)
    (rb,) = [rb for rb in model.rbodies if rb.id == 2]
    assert rb.grnod_id == 3


def test_set_general_surface_seg_resolved(tmp_path):
    """Test /SET/GENERAL/2 with KEY SEG resolves to model.surfaces[2]."""
    deck = f"""\
/BEGIN
TEST_SET_GENERAL_SEG
{_BASE_DECK}
/SET/GENERAL/2
Surface Segment Set
SEG
101 102 103 104
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Starter errors: {log.errors}"
    assert 2 in model.surfaces
    s = model.surfaces[2]
    assert s.id == 2
    assert s.title == "Surface Segment Set"
    assert s.segments is not None
    assert len(s.segments) == 1
    assert np.array_equal(s.segments[0], [0, 1, 2, 3])


def test_set_general_surface_part_e_resolved(tmp_path):
    """Test /SET/GENERAL/5 with KEY PART_E resolves to model.surfaces[5]."""
    deck = f"""\
/BEGIN
TEST_SET_GENERAL_PART_E
{_BASE_DECK}
/SET/GENERAL/5
Surface From Part_E
PART_E
1
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Starter errors: {log.errors}"
    assert 5 in model.surfaces
    s = model.surfaces[5]
    assert s.id == 5
    assert s.title == "Surface From Part_E"
    assert s.segments is not None
    assert len(s.segments) == 1
