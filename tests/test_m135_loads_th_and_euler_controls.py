"""Tests for Milestone M135: Extended Loadings, Time History Requests & Eulerian Controls Suite
(/LOAD/GRAV, /LOAD/BODY, /LOAD/HEAT, /TH extended kinds, /EULER/BCS, /HEAT/BCS).
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


def test_m135_extended_loadings(tmp_path):
    deck = """/BEGIN
Test Extended Loadings
/LOAD/GRAV/1
Gravity loading on node group
10 0.0 0.0 -9.81 100 1.0 0
/LOAD/BODY/2
Volumetric body force on part group
20 0.0 -1.0 0.0 101 2.5 1
/LOAD/HEAT/3
Thermal heat flux on part
30 500.0 102 1.0 0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 1 in model.load_gravities
    g = model.load_gravities[1]
    assert g.id == 1
    assert g.grnod_id == 10
    assert g.dir_vector == (0.0, 0.0, -9.81)
    assert g.funct_id == 100
    assert g.scale == pytest.approx(1.0)

    assert 2 in model.load_bodies
    b = model.load_bodies[2]
    assert b.id == 2
    assert b.grpart_id == 20
    assert b.dir_vector == (0.0, -1.0, 0.0)
    assert b.funct_id == 101
    assert b.scale == pytest.approx(2.5)
    assert b.sens_id == 1

    assert 3 in model.load_therms
    th = model.load_therms[3]
    assert th.id == 3
    assert th.group_id == 30
    assert th.flux == pytest.approx(500.0)
    assert th.funct_id == 102
    assert th.scale == pytest.approx(1.0)


def test_m135_expanded_th_requests(tmp_path):
    deck = """/BEGIN
Test Expanded Time History Requests
/TH/MONVOL/1
Monitored Volume TH
PRES VOL TEMP
1 2
/TH/AIRBAG/2
Airbag TH
MASS VOL ENER
10
/TH/ALE/3
ALE TH
DEF
100 101
/TH/LAGMUL/4
Lagrange Multiplier TH
FORCE MOMENT
50
/TH/STACK/5
Stack Composite TH
SIGX SIGY
200
/TH/WAVE_SHAPER/6
Wave Shaper TH
RADIUS TIME
300
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert len(model.th_requests) == 6
    kinds = [r.kind for r in model.th_requests]
    assert kinds == ["MONVOL", "AIRBAG", "ALE", "LAGMUL", "STACK", "WAVE_SHAPER"]


def test_m135_euler_and_heat_bcs(tmp_path):
    deck = """/BEGIN
Test Eulerian and Thermal Boundary Conditions
/EULER/BCS/1
Eulerian Inflow Boundary
10 INFLOW 100.0 0.0 0.0
/HEAT/BCS/2
Thermal Temperature Boundary
20 TEMP 373.15 200 1.0 0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 1 in model.euler_bcs
    eb = model.euler_bcs[1]
    assert eb.id == 1
    assert eb.grnod_id == 10
    assert eb.bcs_type == "INFLOW"
    assert eb.val1 == pytest.approx(100.0)

    assert 2 in model.heat_bcs
    hb = model.heat_bcs[2]
    assert hb.id == 2
    assert hb.group_id == 20
    assert hb.bcs_type == "TEMP"
    assert hb.tval == pytest.approx(373.15)
    assert hb.funct_id == 200
    assert hb.scale == pytest.approx(1.0)
