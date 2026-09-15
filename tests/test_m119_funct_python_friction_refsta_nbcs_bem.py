"""Tests for Milestone M119: Python Functions (/FUNCT_PYTHON),
Generalized Multi-Pair Friction (/FRICTION), Reference State Geometry (/REFSTA, /EREF),
Non-Linear Boundary Conditions (/NBCS), ALE MUSCL High-Order Advection (/ALE/MUSCL),
and Boundary Element Method (/BEM).
"""
from __future__ import annotations

from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.starter.starter import run_starter
from pyradioss.starter.starter import StarterError


_BOILERPLATE = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
Test_Deck_M119
      2022         0
/MAT/LAW1/1
LinearElastic
              7.8e-9
            210000.0                 0.3
/PROP/TYPE1/1
Dummy_Shell
         1         1         1         0         0         0         0
                 1.0                 1.0                 1.0
                 1.0            0.833333
/PART/1
Part_Shell_1
         1         1
/PART/2
Part_Shell_2
         1         1
/NODE
         1                 0.0                 0.0                 0.0
         2                 1.0                 0.0                 0.0
         3                 1.0                 1.0                 0.0
         4                 0.0                 1.0                 0.0
/SHELL/1
         1         1         2         3         4
/SKEW/MOV/1
Skew1
         1         2         3         X
/GRPART/1
PartGroup1
         1
/GRPART/2
PartGroup2
         2
"""


def _run(tmp_path: Path, deck: str, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="ascii")
    log = MessageLog()
    try:
        model = run_starter(str(p), log)
    except StarterError:
        model = None
    return model, log


def test_m119_funct_python(tmp_path):
    deck = _BOILERPLATE + """\
/FUNCT_PYTHON/10
def my_custom_curve(x, y):
    import math
    return (x + y) * math.sin(x)
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 10 in model.funct_pythons
    fp = model.funct_pythons[10]
    assert fp.id == 10
    assert len(fp.lines) == 3
    assert "def my_custom_curve" in fp.lines[0]
    assert "math.sin(x)" in fp.lines[2]


def test_m119_friction_fixed_and_free(tmp_path):
    deck = _BOILERPLATE + """\
/FRICTION/1
Friction_Model_One
         1         1                10.0         1
                 1.0                 2.0                 3.0                 4.0                 5.0
                 6.0                 0.2                 0.8
         0         0         1         2                   1
                 0.1                 0.2                 0.3                 0.4                 0.5
                 0.6                0.15                 0.9
                 1.1                 1.2                 1.3                 1.4                 1.5
                 1.6                0.25                0.85
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.friction_models
    fm = model.friction_models[1]
    assert fm.id == 1
    assert fm.title == "Friction_Model_One"
    assert fm.ifric == 1
    assert fm.ifiltr == 1
    assert fm.xfreq == 10.0
    assert fm.iform == 1
    assert fm.c1 == 1.0
    assert fm.fric == 0.2
    assert fm.vis_f == 0.8
    assert len(fm.pairs) == 1
    p = fm.pairs[0]
    assert p.part_id1 == 1
    assert p.part_id2 == 2
    assert p.idir == 1
    assert p.c1 == 0.1
    assert p.fric == 0.15
    assert p.c1_dir2 == 1.1
    assert p.fric_dir2 == 0.25


def test_m119_friction_free(tmp_path):
    deck = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
Free_Friction_Test
/MAT/LAW1/1
LinearElastic
7.8e-9
210000.0 0.3
/PROP/TYPE1/1
Dummy_Shell
1 1 1 0 0 0 0
1.0 1.0 1.0
1.0 0.833333
/PART/1
Part_Shell_1
1 1
/PART/2
Part_Shell_2
1 1
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
/SHELL/1
1 1 2 3 4
/FRICTION/2
Friction_Model_Two
2 0 0.0 2
0.5 0.5 0.5 0.5 0.5
0.5 0.3 1.0
0 0 1 2 0
0.2 0.2 0.2 0.2 0.2
0.2 0.1 1.0
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 2 in model.friction_models
    fm = model.friction_models[2]
    assert fm.ifric == 2
    assert fm.iform == 2
    assert len(fm.pairs) == 1
    assert fm.pairs[0].part_id1 == 1
    assert fm.pairs[0].fric == 0.1


def test_m119_refsta_and_eref(tmp_path):
    deck = _BOILERPLATE + """\
/REFSTA
         1                 0.0                 0.0                 0.0
         2                 1.1                 0.0                 0.0
         3                 1.1                 1.1                 0.0
         4                 0.0                 1.1                 0.0
/EREF/1
ElementRefPart1
/EREF/SHELL/2
ElementRefShellPart2
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 2 in model.refsta_nodes
    assert model.refsta_nodes[2].x == 1.1
    assert 1 in model.eref_specs
    assert 2 in model.eref_specs
    assert model.eref_specs[2].subtype == "SHELL"


def test_m119_nbcs(tmp_path):
    deck = _BOILERPLATE + """\
/NBCS/5
NonlinearBCS5
   111 000         1         1
   000 111         0         2
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 5 in model.nbcs_blocks
    nb = model.nbcs_blocks[5]
    assert nb.title == "NonlinearBCS5"
    assert len(nb.nodes) == 2
    assert nb.nodes[0].tx == 1
    assert nb.nodes[0].ty == 1
    assert nb.nodes[0].tz == 1
    assert nb.nodes[0].wx == 0
    assert nb.nodes[0].skew_id == 1
    assert nb.nodes[0].node_id == 1
    assert nb.nodes[1].wx == 1
    assert nb.nodes[1].wy == 1
    assert nb.nodes[1].wz == 1
    assert nb.nodes[1].node_id == 2


def test_m119_ale_muscl_and_bem(tmp_path):
    deck = _BOILERPLATE + """\
/ALE/MUSCL
                 1.5
/BEM/10
BoundaryElementMethodBlock
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert model.ale_muscl is not None
    assert model.ale_muscl.beta == 1.5
    assert 10 in model.bem_models
    assert model.bem_models[10].title == "BoundaryElementMethodBlock"


def test_m119_cross_reference_errors(tmp_path):
    deck = _BOILERPLATE + """\
/FRICTION/99
FaultyFriction
         1         1                10.0         1
                 1.0                 2.0                 3.0                 4.0                 5.0
                 6.0                 0.2                 0.8
         0         0       999         2                   0
                 0.1                 0.2                 0.3                 0.4                 0.5
                 0.6                0.15                 0.9
/NBCS/99
FaultyNBCS
   111 111       888       777
/REFSTA
       666                 0.0                 0.0                 0.0
/END
"""
    model, log = _run(tmp_path, deck)
    err_text = " ".join(log.errors)
    assert "/FRICTION/99: part 999 not defined" in err_text
    assert "/NBCS/99: node 777 not defined" in err_text
    assert "/NBCS/99: skew 888 not defined" in err_text
    assert "/REFSTA: node 666 not defined" in err_text
