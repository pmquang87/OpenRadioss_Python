"""Tests for Milestone M121: Extended Sensors (/SENSOR/GAUGE, /SENSOR/HIC,
/SENSOR/WORK, /SENSOR/RWALL, /SENSOR/XSECTION, /SENSOR/DIST_SURF), /GAUGE/POINT,
/SPHGLO, /ANALY, and cross-reference validation.
"""
from __future__ import annotations

from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.starter.checks import check_model


def _parse_starter(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_m121_sensor_gauge_hic_work_free(tmp_path):
    deck = """/BEGIN
Test Deck M121
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
/SENSOR/GAUGE/1
Gauge Sensor
0.05
2
10 1.5e6 0.001
20 2.0e6 0.002
/SENSOR/HIC/2
HIC Sensor
0.01
100 X 0.015 1000.0 9.81 0.001
/SENSOR/WORK/3
Work Sensor
0.02
1 2 5000.0 0.001
10 20 30 40
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    s1 = next(s for s in model.sensors if s.id == 1)
    assert s1.kind == "GAUGE"
    assert s1.tdelay == 0.05
    assert len(s1.gauge_entries) == 2
    assert s1.gauge_entries[0] == (10, 1.5e6, 0.001)
    assert s1.gauge_entries[1] == (20, 2.0e6, 0.002)

    s2 = next(s for s in model.sensors if s.id == 2)
    assert s2.kind == "HIC"
    assert s2.tdelay == 0.01
    assert s2.accel_id == 100
    assert s2.dir == "X"
    assert s2.hic_period == 0.015
    assert s2.hic_val == 1000.0
    assert s2.gravity == 9.81
    assert s2.tmin == 0.001

    s3 = next(s for s in model.sensors if s.id == 3)
    assert s3.kind == "WORK"
    assert s3.tdelay == 0.02
    assert s3.node_id1 == 1
    assert s3.node_id2 == 2
    assert s3.work_max == 5000.0
    assert s3.tmin == 0.001
    assert s3.sect_id == 10
    assert s3.int_id == 20
    assert s3.rbody_id == 30
    assert s3.rwall_id == 40


def test_m121_sensor_rwall_xsection_dist_surf_free(tmp_path):
    deck = """/BEGIN
Test Deck M121 Sensors
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.0 1.0 0.0
4 1.0 1.0 0.0
/SENSOR/RWALL/4
RWall Sensor
0.0
50 FN 100.0 5000.0 0.005
/SENSOR/XSECTION/5
Section Sensor
0.0
60 TM 10.0 500.0 0.002
/SENSOR/DIST_SURF/6
Dist Surf Sensor
0.01
1 0 2 3 4
0.05 0.5 0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    s4 = next(s for s in model.sensors if s.id == 4)
    assert s4.kind == "RWALL"
    assert s4.rwall_id == 50
    assert s4.dir == "FN"
    assert s4.fmin == 100.0
    assert s4.fmax == 5000.0
    assert s4.tmin == 0.005

    s5 = next(s for s in model.sensors if s.id == 5)
    assert s5.kind == "XSECTION"
    assert s5.sect_id == 60
    assert s5.dir == "TM"
    assert s5.fmin == 10.0
    assert s5.fmax == 500.0
    assert s5.tmin == 0.002

    s6 = next(s for s in model.sensors if s.id == 6)
    assert s6.kind == "DIST_SURF"
    assert s6.node_id1 == 1
    assert s6.node_id2 == 2
    assert s6.node_id3 == 3
    assert s6.node_id4 == 4
    assert s6.dmin == 0.05
    assert s6.dmax == 0.5
    assert s6.tmin == 0.001


def test_m121_gauge_point(tmp_path):
    deck = """/BEGIN
Test Point Gauge
/GAUGE/POINT/10
Pressure Shock Gauge
0.0 0.0 0.0 0.1 PointA
1.0 2.0 3.0 0.2 PointB
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 10 in model.gauge_points
    gp = model.gauge_points[10]
    assert gp.title == "Pressure Shock Gauge"
    assert len(gp.points) == 2
    assert gp.points[0] == (0.0, 0.0, 0.0, 0.1, "PointA")
    assert gp.points[1] == (1.0, 2.0, 3.0, 0.2, "PointB")


def test_m121_sphglo_and_analy(tmp_path):
    deck = """/BEGIN
Test Global Controls
/SPHGLO
0.3 500 150 150 1
/ANALY
2 1 2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert model.sphglo is not None
    assert model.sphglo.alpha_sort == 0.3
    assert model.sphglo.maxsph == 500
    assert model.sphglo.lneigh == 150
    assert model.sphglo.nneigh == 150
    assert model.sphglo.isol2sph == 1

    assert model.analy is not None
    assert model.analy.n2d3d == 2
    assert model.analy.iparith == 1
    assert model.analy.isubcyc == 2


def test_m121_cross_reference_checks(tmp_path):
    deck = """/BEGIN
Cross Ref Test
/NODE
1 0.0 0.0 0.0
/SENSOR/HIC/1
HIC
0.0
999 X 0.01 1000.0 9.81 0.001
/SENSOR/RWALL/2
RWALL
0.0
888 FN 10.0 100.0 0.001
/SENSOR/XSECTION/3
SECT
0.0
777 TF 10.0 100.0 0.001
/SENSOR/GAUGE/4
GAUGE
0.0
1
666 1.0 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    check_log = MessageLog()
    check_model(model, check_log)
    err_msgs = " ".join(check_log.errors)
    assert "unknown accelerometer 999" in err_msgs
    assert "unknown rigid wall 888" in err_msgs
    assert "unknown section 777" in err_msgs
    assert "unknown gauge 666" in err_msgs


def test_m121_engine_element_dt_controls(tmp_path):
    from pyradioss.input.engine_keywords import parse_engine_deck

    engine_deck = """\
# Engine Deck M121 DT Element Controls
/RUN/TestRun/1
10.0
/DT/BRICK/CST
0.8 1.0e-7
/DT/SHELL/DEL
0.67 5.0e-8
/DT/INTER/STOP
0.9 2.0e-7
/DT/QUAD/CST/1
0.5 1.5e-7
"""
    p = tmp_path / "TEST_0001.rad"
    p.write_text(engine_deck, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    ec = parse_engine_deck(blocks, log)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert "BRICK" in ec.dt_controls
    assert ec.dt_controls["BRICK"]["action"] == "CST"
    assert ec.dt_controls["BRICK"]["scale"] == 0.8
    assert ec.dt_controls["BRICK"]["dt_min"] == 1.0e-7

    assert "SHELL" in ec.dt_controls
    assert ec.dt_controls["SHELL"]["action"] == "DEL"
    assert ec.dt_controls["SHELL"]["scale"] == 0.67
    assert ec.dt_controls["SHELL"]["dt_min"] == 5.0e-8

    assert "INTER" in ec.dt_controls
    assert ec.dt_controls["INTER"]["action"] == "STOP"
    assert ec.dt_controls["INTER"]["scale"] == 0.9
    assert ec.dt_controls["INTER"]["dt_min"] == 2.0e-7

    assert "QUAD" in ec.dt_controls
    assert ec.dt_controls["QUAD"]["action"] == "CST"
    assert ec.dt_controls["QUAD"]["flag"] == "1"
    assert ec.dt_controls["QUAD"]["scale"] == 0.5
    assert ec.dt_controls["QUAD"]["dt_min"] == 1.5e-7

