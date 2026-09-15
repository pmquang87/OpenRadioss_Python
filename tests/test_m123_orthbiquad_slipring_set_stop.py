"""Tests for Milestone M123: /FAIL/ORTHBIQUAD, /SLIPRING/SHELL, /SET/* entity set aliases,
and /STOP Engine termination controls.
"""
from __future__ import annotations

from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.input.engine_keywords import parse_engine_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_m123_fail_orthbiquad(tmp_path):
    deck = """/BEGIN
Test FAIL ORTHBIQUAD
/MAT/LAW1/1
Steel
7.85e-9
210000.0 0.3
/FAIL/ORTHBIQUAD/1
0.5 1 2 0.05
0.1 0.2 0.3 0.4 0.5
1.0e-3 0.02 1.0 10 20 5.0
0.9 0.8 1.1 1.2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.fail_orthbiquads
    fo = model.fail_orthbiquads[1]
    assert fo.id == 1
    assert fo.mat_id == 1
    assert fo.p_thickfail == 0.5
    assert fo.m_flag == 1
    assert fo.s_flag == 2
    assert fo.inst_start == 0.05
    assert fo.c1 == 0.1
    assert fo.c2 == 0.2
    assert fo.c3 == 0.3
    assert fo.c4 == 0.4
    assert fo.c5 == 0.5
    assert fo.eps_dot0 == 1.0e-3
    assert fo.c_jc == 0.02
    assert fo.fct_id_rate == 10
    assert fo.fct_id_el == 20
    assert fo.ei_ref == 5.0
    assert fo.r1 == 0.9
    assert fo.r2 == 0.8
    assert fo.r4 == 1.1
    assert fo.r5 == 1.2


def test_m123_slipring_shell(tmp_path):
    deck = """/BEGIN
Test Slipring Shell
/SLIPRING/SHELL/10
Seatbelt Slipring
1 2 100 5 1 0.25 0.01
10 20 0.15 1.0 1.0 1.0
30 40 0.25 1.0 1.0 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 10 in model.slipring_shells
    sr = model.slipring_shells[10]
    assert sr.id == 10
    assert sr.title == "Seatbelt Slipring"
    assert sr.el_set1 == 1
    assert sr.el_set2 == 2
    assert sr.node_set == 100
    assert sr.sens_id == 5
    assert sr.flow_flag == 1
    assert sr.a == 0.25
    assert sr.ed_factor == 0.01
    assert sr.fric_d == 0.15
    assert sr.fric_s == 0.25


def test_m123_set_aliases(tmp_path):
    deck = """/BEGIN
Test Set Aliases
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.0 1.0 0.0
4 1.0 1.0 0.0
/SET/SHELL/1
Shell Set
101 102 103
/SETS/BRIC/2
Brick Set
201 202
/SET/NODE/3
Node Set
1 2 3 4
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert "SHEL" in model.egroups
    assert 1 in model.egroups["SHEL"]
    assert model.egroups["SHEL"][1].elem_ids == [101, 102, 103]

    assert "BRIC" in model.egroups
    assert 2 in model.egroups["BRIC"]
    assert model.egroups["BRIC"][2].elem_ids == [201, 202]

    assert 3 in model.node_groups
    assert set(model.node_groups[3].node_ids) == {1, 2, 3, 4}


def test_m123_engine_stop_and_anim(tmp_path):
    engine_deck = """\
# Engine Deck M123
/RUN/TestRun/1
10.0
/STOP/NSTEP
50000
/STOP/TSTOP
0.085
/STOP/TIMET
3600.0
/ANIM/MASS
/ANIM/SHELL/THICK
/ANIM/NODA/TEMP
/ANIM/BRICK/DAMP
"""
    p = tmp_path / "TEST_0001.rad"
    p.write_text(engine_deck, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    ec = parse_engine_deck(blocks, log)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert ec.stop_nstep == 50000
    assert ec.stop_tstop == 0.085
    assert ec.stop_timet == 3600.0
    assert "MASS" in ec.anim_elem or "MASS" in ec.anim_vect or any("MASS" in x for x in ec.anim_elem)
    assert any("THICK" in x for x in ec.anim_elem)
