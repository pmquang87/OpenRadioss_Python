"""Tests for Milestone M138: Extended Element Initial Thermodynamic States, Boundary Condition Constraints & Coordinate Symmetry Suite
(/INIBRI/TEMP, /INIBRI/PRES, /INIBRI/VOID, /INISHE/TEMP, /INISHE/ENER, /INITRU/TEMP, /INIBEA/TEMP, /INISPR/TEMP,
 /BCS/TRA, /BCS/ROT, /TRANSFORM/SYMET, /TRANSFORM/SCALE, /TRANSFORM/TRANSL, /TRANSFORM/ROTATE, /EBCS/PERIODIC, /EBCS/CYCLIC).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_m138_element_initial_states(tmp_path):
    deck = """/BEGIN
Test Element Initial States
/INIBRI/TEMP
1 298.15
/INIBRI/PRES
1 101325.0
/INIBRI/VOID
1 0.05
/INISHE/TEMP
10 350.0
/INISHE/ENER
10 1250.0
/INISH3/TEMP
20 400.0
/INITRU/TEMP
30 300.0
/INIBEA/TEMP
40 310.0
/INISPR/TEMP
50 295.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    # Brick initial state
    assert 1 in model.ini_bricks
    b = model.ini_bricks[1]
    assert b.temp == pytest.approx(298.15)
    assert b.pres == pytest.approx(101325.0)
    assert b.void == pytest.approx(0.05)

    # Shell initial state
    assert 10 in model.ini_shells
    s = model.ini_shells[10]
    assert s.temp == pytest.approx(350.0)
    assert s.em == pytest.approx(1250.0)

    # 3-node shell initial state
    assert 20 in model.ini_shells
    s3 = model.ini_shells[20]
    assert s3.temp == pytest.approx(400.0)

    # Truss initial state
    assert 30 in model.ini_trusses
    t = model.ini_trusses[30]
    assert t.temp == pytest.approx(300.0)

    # Beam initial state
    assert 40 in model.ini_beams
    bm = model.ini_beams[40]
    assert bm.temp == pytest.approx(310.0)

    # Spring initial state
    assert 50 in model.ini_springs
    sp = model.ini_springs[50]
    assert sp.temp == pytest.approx(295.0)


def test_m138_bcs_tra_and_rot(tmp_path):
    deck = """/BEGIN
Test BCS Tra and Rot
/BCS/TRA/1
Translation Clamp
110 0 100
/BCS/ROT/2
Rotation Clamp
001 0 200
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert len(model.bcs) == 2

    bc1 = model.bcs[0]
    assert bc1.id == 1
    assert bc1.grnod_id == 100
    assert np.array_equal(bc1.fix_tra, [True, True, False])
    assert np.array_equal(bc1.fix_rot, [False, False, False])

    bc2 = model.bcs[1]
    assert bc2.id == 2
    assert bc2.grnod_id == 200
    assert np.array_equal(bc2.fix_tra, [False, False, False])
    assert np.array_equal(bc2.fix_rot, [False, False, True])


def test_m138_transform_mirror_and_aliases(tmp_path):
    deck = """/BEGIN
Test Transform Aliases
/TRANSFORM/SYMET/1
Mirror Symmetry
10 0.0 0.0 0.0 0 0 0
0.0 0.0 1.0
/TRANSFORM/SCALE/2
Scale Transform
20 2.0 2.0 2.0 0 0
/TRANSFORM/TRANSL/3
Translate Transform
30 10.0 20.0 30.0 0 0 0
/TRANSFORM/ROTATE/4
Rotate Transform
40 0.0 0.0 0.0 0 0 0
0.0 0.0 1.0 45.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert len(model.transforms) == 4


def test_m138_ebcs_periodic_and_cyclic(tmp_path):
    deck = """/BEGIN
Test EBCS Periodic and Cyclic
/EBCS/PERIODIC/1
Periodic Frontier
10 20 1 100
/EBCS/CYCLIC/2
Cyclic Frontier
30 40 2 200
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 1 in model.ebcs_periodics
    p = model.ebcs_periodics[1]
    assert p.id == 1
    assert p.surf1_id == 10
    assert p.surf2_id == 20
    assert p.skew_id == 1
    assert p.grpart_id == 100

    assert 2 in model.ebcs_cyclics
    c = model.ebcs_cyclics[2]
    assert c.id == 2
    assert c.surf1_id == 30
    assert c.surf2_id == 40
    assert c.skew_id == 2
    assert c.grpart_id == 200
