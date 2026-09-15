"""Tests for Milestone M205:
- Flow Inflow/Outflow Boundaries: /FLOW, /FLOW/INFLOW, /FLOW/OUTFLOW, /ALE/FLOW
- Cavity Enclosure Radiation: /HEAT/RAD_CAV, /HEAT/CAV, /RAD_CAV
- Torsional & Bending Springs: /PROP/TYPE19, /PROP/SPR_TORS, /PROP/SPR_BEND
- Blast, Fluid & Laser Load Aliases: /LOAD/PBLAST, /LOAD/PFLUID, /LOAD/LASER
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import FlowBoundary, HeatRadCav, PropSpringTors, PropSpringBend


def _parse_starter(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_m205_flow_boundaries(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Flow Boundaries Test
1 1
/FLOW/INFLOW/101
Inflow Surface Boundary
       201         1         5
                 1.2               1e5               300.0                50.0                 0.0
/ALE/FLOW/102
Outflow Surface Boundary
       202         2         0
                 1.0               1e5               293.15                0.0                 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 101 in model.flow_boundaries
    f1 = model.flow_boundaries[101]
    assert f1.id == 101
    assert f1.subtype == "INFLOW"
    assert f1.surf_id == 201
    assert f1.flow_type == 1
    assert f1.sens_id == 5
    assert pytest.approx(f1.rho) == 1.2
    assert pytest.approx(f1.pres) == 1e5
    assert pytest.approx(f1.temp) == 300.0
    assert pytest.approx(f1.vx) == 50.0
    assert model.flows is model.flow_boundaries

    assert 102 in model.flow_boundaries
    f2 = model.flow_boundaries[102]
    assert f2.id == 102
    assert f2.surf_id == 202
    assert f2.flow_type == 2


def test_m205_heat_rad_cav(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Cavity Radiation Test
1 1
/HEAT/RAD_CAV/301
Enclosure Cavity Radiation
       101       102                 0.8                 0.9                 0.75
/RAD_CAV/302
Cavity 2
       103       104                 1.0                 1.0                 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 301 in model.heat_rad_cavs
    rc1 = model.heat_rad_cavs[301]
    assert rc1.surf_id1 == 101
    assert rc1.surf_id2 == 102
    assert pytest.approx(rc1.emissivity1) == 0.8
    assert pytest.approx(rc1.emissivity2) == 0.9
    assert pytest.approx(rc1.view_factor) == 0.75

    assert 302 in model.heat_rad_cavs
    rc2 = model.heat_rad_cavs[302]
    assert rc2.surf_id1 == 103
    assert rc2.surf_id2 == 104


def test_m205_prop_spring_tors_and_bend(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Torsion and Bending Properties Test
1 1
/PROP/TYPE19/401
Torsional Spring
                 0.5              5000.0                10.0
/PROP/SPR_BEND/402
Bending Spring
                 0.2              8000.0                25.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 401 in model.props_type19
    p19 = model.props_type19[401]
    assert p19.id == 401
    assert pytest.approx(p19.mass) == 0.5
    assert pytest.approx(p19.k) == 5000.0
    assert pytest.approx(p19.c) == 10.0
    assert model.props_spr_tors is model.props_type19

    assert 402 in model.props_type20
    p20 = model.props_type20[402]
    assert p20.id == 402
    assert pytest.approx(p20.mass) == 0.2
    assert pytest.approx(p20.k) == 8000.0
    assert pytest.approx(p20.c) == 25.0
    assert model.props_spr_bend is model.props_type20


def test_m205_loads_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Loads Aliases Test
1 1
/LOAD/PBLAST/501
Blast Pressure Load
       101         1         1         0         2         0                             0
                 0.0                 0.0                 5.0                 0.0               100.0
/LOAD/PFLUID/502
Fluid Hydrostatic Load
       102
                 0.0                 0.0                 0.0
                 0.0                 0.0                -1.0
              1000.0                 9.81              1e5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 501 in model.load_pblasts
    assert model.pblasts is model.load_pblasts
    assert 501 in model.pblasts

    assert 502 in model.load_pfluids
    assert model.pfluids is model.load_pfluids
    assert 502 in model.pfluids
    assert model.lasers is model.load_lasers
