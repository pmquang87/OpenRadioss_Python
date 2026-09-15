"""Tests for Milestone M216:
- Ladevèze EVR Failure Criterion: /FAIL/LAD_EVR, /FAIL/LADEVEZE_EVR
- Engine Console Monitor Directive: /MONITOR, /ENG/MONITOR
- Engine Noise Filter Directive: /NOIS, /ENG/NOIS
- Distance Gap Sensor Trigger: /SENSOR/TIME_GAP, /SENSOR/GAP
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_m216_fail_lad_evr(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Ladeveze EVR Failure Criterion Test
1 1
/FAIL/LAD_EVR/77
Ladeveze EVR Material Failure
                0.25                0.75                 1.5                 0.9         2                0.35
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 77 in model.fail_lad_evrs
    fe = model.fail_lad_evrs[77]
    assert pytest.approx(fe.yo) == 0.25
    assert pytest.approx(fe.yc) == 0.75
    assert pytest.approx(fe.ymax) == 1.5
    assert pytest.approx(fe.d_max) == 0.9
    assert fe.ifail_sh == 2
    assert pytest.approx(fe.gam) == 0.35


def test_m216_eng_monitor(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Monitor Test
1 1
/MONITOR/1
Nodal Monitor 1
       101         4                0.01
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.monitors
    mon = model.monitors[1]
    assert mon.node_id == 101
    assert mon.ivar_type == 4
    assert pytest.approx(mon.dt_print) == 0.01


def test_m216_eng_nois(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Noise Filter Test
1 1
/ENG/NOIS
              1000.0         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert pytest.approx(model.nois_freq) == 1000.0
    assert model.nois_type == 1


def test_m216_sensor_gap(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Distance Gap Sensor Test
1 1
/SENSOR/TIME_GAP/55
Distance Gap Sensor 55
        10        20                 5.0                0.02         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 55 in model.sensor_gaps
    assert any(s.id == 55 and s.kind == "GAP" for s in model.sensors)
    sg = model.sensor_gaps[55]
    assert sg.node1 == 10
    assert sg.node2 == 20
    assert pytest.approx(sg.d_gap) == 5.0
    assert pytest.approx(sg.t_delay) == 0.02
    assert sg.isens_mode == 1
