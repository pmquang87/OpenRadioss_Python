"""Tests for Milestone M126: Extended Failure Models Suite (/FAIL/PUCK, /FAIL/SAHRAEI,
/FAIL/SYAZWAN, /FAIL/TAB2, /FAIL/GENE1) and /TH/SENSOR Time-History Requests.
"""
from __future__ import annotations

from pathlib import Path
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


def test_m126_fail_puck(tmp_path):
    deck = """/BEGIN
Test FAIL PUCK
/MAT/LAW1/10
Composite Material
1.5e-9
140000.0 0.3
/FAIL/PUCK/10
1200.0 50.0 70.0 800.0 200.0
0.35 0.25 0.25 1.0e-5 1 1
1000.0
501
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 10 in model.fail_pucks

    p = model.fail_pucks[10]
    assert p.mat_id == 10
    assert p.sigma_1t == 1200.0
    assert p.sigma_2t == 50.0
    assert p.sigma_12 == 70.0
    assert p.sigma_1c == 800.0
    assert p.sigma_2c == 200.0
    assert p.p12_pos == 0.35
    assert p.p12_neg == 0.25
    assert p.p22_neg == 0.25
    assert p.tau_max == 1.0e-5
    assert p.ifail_sh == 1
    assert p.ifail_so == 1
    assert p.fcut == 1000.0
    assert p.fail_id == 501


def test_m126_fail_sahraei(tmp_path):
    deck = """/BEGIN
Test FAIL SAHRAEI
/MAT/LAW2/15
Metal
7.8e-9
210000.0 0.3
500.0 300.0 0.5
/FAIL/SAHRAEI/15
1 1 1 1 0.15 2 5.0
1 1 0.25 0.5
601
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 15 in model.fail_sahraeis

    s = model.fail_sahraeis[15]
    assert s.mat_id == 15
    assert s.fct_ratio == 1
    assert s.num == 1
    assert s.den == 1
    assert s.ordi == 1
    assert s.vol_strain == 0.15
    assert s.fct_elsize == 2
    assert s.el_ref == 5.0
    assert s.comp_dir == 1
    assert s.idel == 1
    assert s.max_comp_strain == 0.25
    assert s.ratio == 0.5
    assert s.fail_id == 601


def test_m126_fail_syazwan(tmp_path):
    deck = """/BEGIN
Test FAIL SYAZWAN
/MAT/LAW2/20
Metal
7.8e-9
210000.0 0.3
500.0 300.0 0.5
/FAIL/SYAZWAN/20
1 0.02
0.1 0.2 0.3 0.4 0.5 0.6
701
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 20 in model.fail_syazwans

    sy = model.fail_syazwans[20]
    assert sy.mat_id == 20
    assert sy.icard == 1
    assert sy.epfmin == 0.02
    assert sy.coeffs == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert sy.fail_id == 701


def test_m126_fail_tab2(tmp_path):
    deck = """/BEGIN
Test FAIL TAB2
/MAT/LAW2/25
Metal
7.8e-9
210000.0 0.3
500.0 300.0 0.5
/FAIL/TAB2/25
101 1.0 1 0.5
1.5 0.9 102 0.8
103 2.5 1.2
801
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 25 in model.fail_tab2s

    t = model.fail_tab2s[25]
    assert t.mat_id == 25
    assert t.epsf_id == 101
    assert t.fcrit == 1.0
    assert t.failip == 1
    assert t.pthk == 0.5
    assert t.n == 1.5
    assert t.dcrit == 0.9
    assert t.inst_id == 102
    assert t.ecrit == 0.8
    assert t.fct_exp == 103
    assert t.exp_ref == 2.5
    assert t.exp == 1.2
    assert t.fail_id == 801


def test_m126_fail_gene1(tmp_path):
    deck = """/BEGIN
Test FAIL GENE1
/MAT/LAW2/30
Metal
7.8e-9
210000.0 0.3
500.0 300.0 0.5
/FAIL/GENE1/30
-100.0 1000.0 800.0 1.0 1.0e-6
1 100.0 600.0 500.0 2.0
2 50.0 0.2 0.3 0.1
901
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 30 in model.fail_gene1s

    g = model.fail_gene1s[30]
    assert g.mat_id == 30
    assert g.pmin == -100.0
    assert g.pmax == 1000.0
    assert g.sigp1_max == 800.0
    assert g.time_max == 1.0
    assert g.dtmin == 1.0e-6
    assert g.fct_idsm == 1
    assert g.eps_dot_sm == 100.0
    assert g.sig_max == 600.0
    assert g.sigr == 500.0
    assert g.k == 2.0
    assert g.fct_idps == 2
    assert g.eps_dot_ps == 50.0
    assert g.eps_max == 0.2
    assert g.eps_eff == 0.3
    assert g.eps_vol == 0.1
    assert g.fail_id == 901


def test_m126_th_sensor(tmp_path):
    deck = """/BEGIN
Test TH SENSOR
/TH/SENSOR/1
Sensor Output
DEF
1 2 3
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert len(model.th_requests) == 1
    th = model.th_requests[0]
    assert th.kind == "SENSOR"
    assert th.ids == [1, 2, 3]


def test_m126_fixed_format(tmp_path):
    deck = """#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
Test Fixed Format
      2022         0
/MAT/LAW1/10
Composite Material
              1.5e-9
            140000.0                 0.3
/FAIL/PUCK/10
              1200.0                50.0                70.0               800.0               200.0
                0.35                0.25                0.25              1.0e-5         1         1
              1000.0
       501
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 10 in model.fail_pucks
    p = model.fail_pucks[10]
    assert p.sigma_1t == pytest.approx(1200.0)
    assert p.sigma_2t == pytest.approx(50.0)
    assert p.p12_pos == pytest.approx(0.35)
    assert p.fail_id == 501

