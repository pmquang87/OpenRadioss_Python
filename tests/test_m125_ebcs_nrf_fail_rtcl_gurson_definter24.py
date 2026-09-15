"""Tests for Milestone M125: Non-Reflecting Boundary Conditions (/EBCS/NRF),
/FAIL/RTCL and /FAIL/GURSON Failure Models, and /DEF_INTER/TYPE24 Contact Defaults.
"""
from __future__ import annotations

from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.starter.starter import run_starter


def _parse_starter(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_m125_ebcs_nrf(tmp_path):
    deck = """/BEGIN
Test EBCS NRF
/EBCS/NRF/1
Non Reflecting Frontier
10
0.05 0.02
/EBCS/NON_REFLECT/2
Non Reflecting Frontier 2
20
0.1 0.04
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.ebcs_nrfs
    assert 2 in model.ebcs_nrfs

    ebcs1 = model.ebcs_nrfs[1]
    assert ebcs1.id == 1
    assert ebcs1.title == "Non Reflecting Frontier"
    assert ebcs1.surf_id == 10
    assert ebcs1.tcar_p == 0.05
    assert ebcs1.tcar_vf == 0.02

    ebcs2 = model.ebcs_nrfs[2]
    assert ebcs2.id == 2
    assert ebcs2.title == "Non Reflecting Frontier 2"
    assert ebcs2.surf_id == 20
    assert ebcs2.tcar_p == 0.1
    assert ebcs2.tcar_vf == 0.04


def test_m125_fail_rtcl(tmp_path):
    deck = """/BEGIN
Test FAIL RTCL
/MAT/LAW1/10
Elastic Material
7.8e-9
210000.0 0.3
/FAIL/RTCL/10
0.25 1 0.15
101
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 10 in model.fail_rtcls

    f_rtcl = model.fail_rtcls[10]
    assert f_rtcl.mat_id == 10
    assert f_rtcl.epscal == 0.25
    assert f_rtcl.inst == 1
    assert f_rtcl.n == 0.15
    assert f_rtcl.fail_id == 101


def test_m125_fail_gurson(tmp_path):
    deck = """/BEGIN
Test FAIL GURSON
/MAT/LAW2/5
Johnson Cook
7.8e-9
210000.0 0.3
500.0 300.0 0.5
/FAIL/GURSON/5
1.5 1.0 1
0.3 0.05 1.2
0.15 0.25 0.01
2.0 0.05 1.0
201
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 5 in model.fail_gursons

    f_gurson = model.fail_gursons[5]
    assert f_gurson.mat_id == 5
    assert f_gurson.q1 == 1.5
    assert f_gurson.q2 == 1.0
    assert f_gurson.iloc == 1
    assert f_gurson.eps_n == 0.3
    assert f_gurson.a_s == 0.05
    assert f_gurson.k_w == 1.2
    assert f_gurson.f_c == 0.15
    assert f_gurson.f_r == 0.25
    assert f_gurson.f_0 == 0.01
    assert f_gurson.r_len == 2.0
    assert f_gurson.h_chi == 0.05
    assert f_gurson.le_max == 1.0
    assert f_gurson.fail_id == 201


def test_m125_def_inter_type24(tmp_path):
    deck = """/BEGIN
Test DEF INTER TYPE24
/DEF_INTER/TYPE24
1000 1 1 1000 1000 1 1000 1000 1000
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert "TYPE24" in model.def_inter
    d24 = model.def_inter["TYPE24"]
    assert d24["istf"] == 1000
    assert d24["igap"] == 1
    assert d24["irem_i2"] == 1
    assert d24["idel"] == 1000
    assert d24["itied"] == 1000
    assert d24["ishape"] == 1
    assert d24["irs"] == 1000
    assert d24["iedge"] == 1000
    assert d24["ipen"] == 1000


def test_m125_fixed_format(tmp_path):
    deck = """#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
Test Fixed Format
      2022         0
/EBCS/NRF/1
Fixed NRF
        10
                0.05                0.02
/MAT/LAW1/10
Elastic Material
              7.8e-9
            210000.0                 0.3
/FAIL/RTCL/10
                0.25         1                0.15
       101
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.ebcs_nrfs
    assert model.ebcs_nrfs[1].surf_id == 10
    assert model.ebcs_nrfs[1].tcar_p == pytest.approx(0.05)
    assert 10 in model.fail_rtcls
    assert model.fail_rtcls[10].epscal == pytest.approx(0.25)
    assert model.fail_rtcls[10].fail_id == 101

