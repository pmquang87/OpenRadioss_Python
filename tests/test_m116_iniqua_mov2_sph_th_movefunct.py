"""Tests for Milestone M116: Initial Quad / State Cards (/INIQUA, /INISTA),
Moving Reference Systems (/FRAME/MOV2, /SKEW/MOV2), SPH Reserve Buffers (/SPH/RESERVE),
Extended Time History Channels (/TH/NSTRAND, /TH/SPHCEL, /TH/MODE, /TH/CYL_JO, /TH/GR*),
and Function Curve Transformations (/MOVE_FUNCT, /FUNCT/MOVE).
"""
from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np

from pyradioss.common.messages import MessageLog
from pyradioss.starter.starter import run_starter
from pyradioss.starter.starter import StarterError


_BOILERPLATE = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
Test_Deck_M116
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
/PROP/TYPE2/2
Dummy_Truss
         1         1
                 1.0
/PROP/TYPE3/3
Dummy_Beam
         1         1
                 1.0                 1.0                 1.0
                 1.0                 1.0                 1.0                 1.0
/PROP/TYPE14/4
Dummy_Solid
         1         1
/PART/1
Part_Shell_1
         1         1
/PART/2
Part_Shell_2
         1         1
/PART/3
Part_Beam_3
         3         1
/PART/4
Part_Solid_4
         4         1
/NODE
         1                 0.0                 0.0                 0.0
         2                 1.0                 0.0                 0.0
         3                 1.0                 1.0                 0.0
         4                 0.0                 1.0                 0.0
         5                 0.0                 0.0                 1.0
         6                 1.0                 0.0                 1.0
         7                 1.0                 1.0                 1.0
         8                 0.0                 1.0                 1.0
/SHELL/1
         1         1         2         3         4
         2         1         2         3         4
         3         1         2         3         4
         4         1         2         3         4
/BRICK/4
         5         1         2         3         4         5         6         7         8
/BEAM/3
         6         1         2         3
/FUNCT/1
TestCurve1
                 0.0                 0.0
                 1.0                 2.0
/FUNCT/2
TestCurve2
                 0.0                 0.0
                 2.0                 4.0
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


def test_m116_iniqua_and_inista(tmp_path):
    deck = _BOILERPLATE + """\
/INIQUA/EPSP
         1                0.05
/INIQUA/THICK
         2                 1.5
/INISTA/SHELL/EPSP
         4                0.12
/INISTA/SOLID/EPSP
         5                0.08
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.ini_shells
    assert model.ini_shells[1].epsp == 0.05
    assert 2 in model.ini_shells
    assert model.ini_shells[2].thick == 1.5
    assert 4 in model.ini_shells
    assert model.ini_shells[4].epsp == 0.12
    assert 5 in model.ini_bricks
    assert model.ini_bricks[5].epsp == 0.08


def test_m116_frame_skew_mov2(tmp_path):
    deck = _BOILERPLATE + """\
/FRAME/MOV2/10
MovingFrame10
         1         2         3
/SKEW/MOV2/20
MovingSkew20
         1         2         4
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 10 in model.skews
    assert 20 in model.skews
    assert model.skews.has_moving() is True

    sf10 = next(e for e in model.skews.entries if e.id == 10)
    assert sf10.kind == "FRAME"
    assert sf10.subtype == "MOV2"
    assert sf10.imov == 2
    assert sf10.n1 == 1 and sf10.n2 == 2 and sf10.n3 == 3

    sf20 = next(e for e in model.skews.entries if e.id == 20)
    assert sf20.kind == "SKEW"
    assert sf20.subtype == "MOV2"
    assert sf20.imov == 2
    assert sf20.n1 == 1 and sf20.n2 == 2 and sf20.n3 == 4


def test_m116_sph_reserve(tmp_path):
    deck = _BOILERPLATE + """\
/SPH/RESERVE/1
       500
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.sph_reserves
    assert model.sph_reserves[1].part_id == 1
    assert model.sph_reserves[1].np_particles == 500


def test_m116_move_funct(tmp_path):
    deck = _BOILERPLATE + """\
/MOVE_FUNCT/1
ScaleCurve1
                 2.0                 3.0                 0.5                 1.0
/FUNCT/MOVE/2
ScaleCurve2
                 1.5                 2.5                 0.0                 0.0
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    # Verify function transformations were applied
    f1 = model.functions[1]
    # Original: (0, 0), (1, 2)
    # Transformed: x' = x*2.0 + 0.5 -> (0.5, 2.5); y' = y*3.0 + 1.0 -> (1.0, 7.0)
    assert np.allclose(f1.x, [0.5, 2.5])
    assert np.allclose(f1.y, [1.0, 7.0])

    f2 = model.functions[2]
    # Original: (0, 0), (2, 4)
    # Transformed: x' = x*1.5 -> (0.0, 3.0); y' = y*2.5 -> (0.0, 10.0)
    assert np.allclose(f2.x, [0.0, 3.0])
    assert np.allclose(f2.y, [0.0, 10.0])


def test_m116_th_extended_kinds(tmp_path):
    deck = _BOILERPLATE + """\
/GRSHEL/SHEL/1
GroupShell1
         1         2
/TH/SPHCEL/1
TH_SPH
DEF
         1
/TH/NSTRAND/2
TH_Strand
DEF
         1
/TH/MODE/3
TH_Mode
DEF
         1
/TH/CYL_JO/4
TH_CylJoint
DEF
         1
/TH/GRSHEL/5
TH_GroupShell
DEF
         1
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert len(model.th_requests) == 5
    req_kinds = [r.kind for r in model.th_requests]
    assert "SPHCEL" in req_kinds
    assert "NSTRAND" in req_kinds
    assert "MODE" in req_kinds
    assert "CYL_JO" in req_kinds
    assert "GRSHEL" in req_kinds


def test_m116_cross_reference_errors(tmp_path):
    deck = _BOILERPLATE + """\
/SPH/RESERVE/999
       100
/MOVE_FUNCT/888
ScaleNonExistent
                 1.0                 1.0                 0.0                 0.0
/END
"""
    model, log = _run(tmp_path, deck)
    err_text = " ".join(log.errors)
    assert "/SPH/RESERVE/999: part 999 not defined" in err_text
    assert "/MOVE_FUNCT/888: function 888 not defined" in err_text
