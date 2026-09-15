"""Tests for Milestone M144: Extended Interfaces (/INTER/TYPE9, /INTER/TYPE10, /INTER/TYPE16, /INTER/TYPE17, /DEF_INTER),
Bolt Preloads (/PRELOAD/BOLT, /SECT/BOLT), and Hydrostatic Surface Loads (/LOAD/HYDRO).
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


def test_m144_interfaces_type9_16_17(tmp_path):
    deck = """/BEGIN
Test Extended Interfaces
/INTER/TYPE9/9
Tied FSI Interface
10 20
0.15 0.02 0.0 10.0 0.05
/INTER/TYPE16/16
Lagrange Multiplier Tied Interface
30 40 1.25
/INTER/TYPE17/17
Hertz Cylinder Contact Interface
50 60
1.0 0.2 0.01 5.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert len(model.interfaces) == 3

    int9 = [i for i in model.interfaces if i.id == 9][0]
    assert int9.type == 9
    assert int9.surf_id == 10
    assert int9.surf_id1 == 20
    assert int9.fric == pytest.approx(0.15)
    assert int9.gap == pytest.approx(0.02)
    assert int9.tstop == pytest.approx(10.0)
    assert int9.visc == pytest.approx(0.05)

    int16 = [i for i in model.interfaces if i.id == 16][0]
    assert int16.type == 16
    assert int16.surf_id == 30
    assert int16.surf_id1 == 40
    assert int16.stfac == pytest.approx(1.25)

    int17 = [i for i in model.interfaces if i.id == 17][0]
    assert int17.type == 17
    assert int17.surf_id == 50
    assert int17.surf_id1 == 60
    assert int17.fric == pytest.approx(0.2)
    assert int17.gap == pytest.approx(0.01)
    assert int17.radius == pytest.approx(5.0)


def test_m144_def_inter_types(tmp_path):
    deck = """/BEGIN
Test Default Interface Parameters
/DEF_INTER/TYPE9
1 2 0 1 0
/DEF_INTER/TYPE10
1000 1 0 1
/DEF_INTER/TYPE16
2000 1
/DEF_INTER/TYPE17
3000 2 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert "TYPE9" in model.def_inter
    assert model.def_inter["TYPE9"]["istf"] == 1
    assert model.def_inter["TYPE9"]["igap"] == 2

    assert "TYPE10" in model.def_inter
    assert model.def_inter["TYPE10"]["istf"] == 1000
    assert model.def_inter["TYPE10"]["itied"] == 1

    assert "TYPE16" in model.def_inter
    assert model.def_inter["TYPE16"]["istf"] == 2000

    assert "TYPE17" in model.def_inter
    assert model.def_inter["TYPE17"]["istf"] == 3000
    assert model.def_inter["TYPE17"]["iform"] == 1


def test_m144_preload_bolt_and_load_hydro(tmp_path):
    deck = """/BEGIN
Test Bolt Preload and Hydrostatic Load
/PRELOAD/BOLT/1
Bolt Pretension Load
10 0 5 15000.0 0.0 0.05 250.0
/LOAD/HYDRO/2
Water Tank Pressure
20 0 1000.0 1.5 9.81
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 1 in model.preload_bolts
    pb = model.preload_bolts[1]
    assert pb.sect_id == 10
    assert pb.fct_id == 5
    assert pb.preload == pytest.approx(15000.0)
    assert pb.tstop == pytest.approx(0.05)
    assert pb.torque == pytest.approx(250.0)

    assert 2 in model.load_hydros
    lh = model.load_hydros[2]
    assert lh.surf_id == 20
    assert lh.density == pytest.approx(1000.0)
    assert lh.z_free == pytest.approx(1.5)
    assert lh.gravity == pytest.approx(9.81)
