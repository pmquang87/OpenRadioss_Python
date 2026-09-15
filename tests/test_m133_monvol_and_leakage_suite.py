"""Tests for Milestone M133: Extended Airbag & Monitored Volume Models Suite & Fabric Leakage
(/MONVOL/TYPE1-11, /MONVOL/AIRBAG, /MONVOL/COMMU, /MONVOL/FVMBAG, /MONVOL/PART, and /LEAK).
"""
from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np

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


def test_m133_monvol_numeric_type_aliases(tmp_path):
    deck = """/BEGIN
Test MONVOL numeric type aliases
/MONVOL/TYPE1/1
Monvol Area Type 1
100
1.0 1.0 1.0 1.0 1.0
/MONVOL/TYPE2/2
Monvol Pres Type 2
100 1.013e-4 1.0 1
/MONVOL/TYPE3/3
Monvol Gas Type 3
100 1 1.0
1.0 1.0 1.0 1.0 1.0
1.4 0.0 0.0 0.0 0.0
0.0 1.013e-4 293.15 0 0
/MONVOL/TYPE9/9
Monvol Commu1 Type 9
100 1 1.0
1.0 1.0 1.0 1.0 1.0
1.4 0.0 0.0 0.0 0.0 0 0
/MONVOL/TYPE10/10
Monvol LFluid Type 10
100
1.0 1.0
1000.0
1 1 1.0 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 1 in model.monvol_areas
    assert model.monvol_areas[1].surf_id == 100

    assert 2 in model.monvol_pres
    assert model.monvol_pres[2].surf_id == 100

    assert 3 in model.monvol_gases
    assert model.monvol_gases[3].surf_id == 100

    assert 9 in model.monvol_commus
    assert model.monvol_commus[9].surf_id == 100

    assert 10 in model.monvol_lfluids
    assert model.monvol_lfluids[10].surf_id == 100


def test_m133_monvol_airbag_type4(tmp_path):
    deck = """/BEGIN
Test MONVOL AIRBAG Type 4
/MONVOL/AIRBAG/10
Multi Gas Airbag 10
100
1.0 1.0 1.0 1.0 1.0
0.0 1.013e-4 293.15 0 0
1.4 1000.0 0.0 0.0
1
1.4 1000.0 0.0 0.0
1 1 1.0 2 1.0 0
1 101 102 103
1
200 0.05 0.6 0.1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 10 in model.monvol_airbags
    ab = model.monvol_airbags[10]
    assert ab.surf_id == 100
    assert ab.gammai == pytest.approx(1.4)
    assert ab.cpai == pytest.approx(1000.0)
    assert ab.njet == 1
    assert len(ab.jets) == 1
    assert ab.jets[0].fct_id_mass == 1
    assert ab.jets[0].n1 == 101
    assert ab.nvent == 1
    assert len(ab.vents) == 1
    assert ab.vents[0].surf_id_v == 200
    assert ab.vents[0].avent == pytest.approx(0.05)


def test_m133_monvol_commu_type5(tmp_path):
    deck = """/BEGIN
Test MONVOL COMMU Type 5
/MONVOL/COMMU/20
Communicating Volume 20
100
1.0 1.0 1.0 1.0 1.0
0.0 1.013e-4 293.15 0 0
1.4 1000.0 0.0 0.0
0
0
21 22 23
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 20 in model.monvol_commu_type5
    cm = model.monvol_commu_type5[20]
    assert cm.surf_id == 100
    assert cm.comm_ids == [21, 22, 23]


def test_m133_monvol_part(tmp_path):
    deck = """/BEGIN
Test MONVOL PART
/MONVOL/PART/30
Part based monvol
5 PRES 1.013e-4
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 30 in model.monvol_parts
    mp = model.monvol_parts[30]
    assert mp.part_id == 5
    assert mp.monvol_type == "PRES"
    assert mp.pini == pytest.approx(1.013e-4)


def test_m133_leak_mat_and_part_and_area(tmp_path):
    deck = """/BEGIN
Test LEAK models
/LEAK/MAT/1
Standard Fabric Leakage
1 1.0 1.0
0.5 10 1.0
0.1 0.2 11 12 1.0 1.0
/LEAK/PART/2
Micromechanical Leakage
5 1.0 1.0
0.0 0 1.0
0.005 0.001
0.1 1.0 0.2
/LEAK/AREA/3
Area Fabric Leakage
2 1.0 1.0
0.8 20 1.0
0.0 0.0 21 22 1.0 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 1 in model.leak_mats
    l1 = model.leak_mats[1]
    assert l1.ileakage == 1
    assert l1.acoeft1 == pytest.approx(0.5)
    assert l1.bcoeft1 == pytest.approx(0.1)

    assert 2 in model.leak_parts
    l2 = model.leak_parts[2]
    assert l2.ileakage == 5
    assert l2.length == pytest.approx(0.005)
    assert l2.thick == pytest.approx(0.001)
    assert l2.c1 == pytest.approx(0.1)
    assert l2.c2 == pytest.approx(1.0)
    assert l2.c3 == pytest.approx(0.2)

    assert 3 in model.leak_areas
    l3 = model.leak_areas[3]
    assert l3.ileakage == 2
    assert l3.acoeft1 == pytest.approx(0.8)
