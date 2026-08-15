"""Tests for Milestone M136: Extended Entity Groups, Rigid Wall Geometries, Cross Sections & Advanced Sensors Suite
(/GRNOD/LINE, /GR*/BOX, /GR*/SURF, /RWALL/BOX, /RWALL/CONE, /SECT/BOX, /SECT/CUT, /SENSOR/SPH, /SENSOR/AIRBAG, /SENSOR/SHELL, /SENSOR/SOLID).
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


def test_m136_extended_groups(tmp_path):
    deck = """/BEGIN
Test Extended Groups
/GRNOD/LINE/1
Line boundary nodes
10 20
/GRNOD/SUB/2
Subset node group
1 2
/GRSHEL/BOX/3
Shells in box
100
/GRBRIC/SURF/4
Bricks on surface
200
/GRQUAD/SUB/5
Quad subset
10 11
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 1 in model.node_groups
    assert model.node_groups[1].line_ids == [10, 20]

    assert 2 in model.node_groups
    assert model.node_groups[2].grnod_ids == [1, 2]

    assert 3 in model.egroups["SHEL"]
    assert model.egroups["SHEL"][3].box_ids == [100]

    assert 4 in model.egroups["BRIC"]
    assert model.egroups["BRIC"][4].surf_ids == [200]

    assert 5 in model.egroups["QUAD"]
    assert model.egroups["QUAD"][5].group_ids == [10, 11]


def test_m136_rwall_box_and_cone(tmp_path):
    deck = """/BEGIN
Test Rigid Wall Box and Cone
/RWALL/BOX/1
Rigid Box Wall
10 0 100 0
0.0 0.0 0.0 10.0 10.0 10.0
/RWALL/CONE/2
Rigid Conical Wall
20 1 200 0
0.0 0.0 5.0 0.0 0.0 1.0 45.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 1 in model.rwall_boxes
    rb = model.rwall_boxes[1]
    assert rb.id == 1
    assert rb.node_id == 10
    assert rb.grnod_id == 100
    assert rb.p1 == (0.0, 0.0, 0.0)
    assert rb.p2 == (10.0, 10.0, 10.0)

    assert 2 in model.rwall_cones
    rc = model.rwall_cones[2]
    assert rc.id == 2
    assert rc.node_id == 20
    assert rc.slide == 1
    assert rc.grnod_id == 200
    assert rc.apex == (0.0, 0.0, 5.0)
    assert rc.axis == (0.0, 0.0, 1.0)
    assert rc.angle == pytest.approx(45.0)


def test_m136_sect_box_and_cut(tmp_path):
    deck = """/BEGIN
Test Section Box and Cut
/SECT/BOX/1
Section cutting by box
100 200 1
/SECT/CUT/2
Section plane cut
0.0 0.0 10.0 0.0 0.0 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 1 in model.sect_boxes
    sb = model.sect_boxes[1]
    assert sb.id == 1
    assert sb.box_id == 100
    assert sb.grnod_id == 200
    assert sb.frame_id == 1

    assert 2 in model.sect_cuts
    sc = model.sect_cuts[2]
    assert sc.id == 2
    assert sc.orig == (0.0, 0.0, 10.0)
    assert sc.normal == (0.0, 0.0, 1.0)


def test_m136_subsystem_sensors(tmp_path):
    deck = """/BEGIN
Test Subsystem Sensors
/SENSOR/SPH/1
SPH Sensor
0.001
10 1000.0 2000.0 0.005
/SENSOR/AIRBAG/2
Airbag Sensor
0.0
5 1e5 2e5 0.01
/SENSOR/SHELL/3
Shell Sensor
0.0
20 0.2 0.5 0.0
/SENSOR/SOLID/4
Solid Sensor
0.0
30 1e8 5e8 0.002
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert len(model.sensors) == 4

    s1 = model.sensors[0]
    assert s1.id == 1
    assert s1.kind == "SPH"
    assert s1.tdelay == pytest.approx(0.001)
    assert s1.target_id == 10
    assert s1.dmin == pytest.approx(1000.0)
    assert s1.dmax == pytest.approx(2000.0)
    assert s1.tmin == pytest.approx(0.005)

    s2 = model.sensors[1]
    assert s2.id == 2
    assert s2.kind == "AIRBAG"
    assert s2.target_id == 5
    assert s2.dmin == pytest.approx(1e5)
    assert s2.dmax == pytest.approx(2e5)

    s3 = model.sensors[2]
    assert s3.id == 3
    assert s3.kind == "SHELL"
    assert s3.target_id == 20
    assert s3.dmin == pytest.approx(0.2)
    assert s3.dmax == pytest.approx(0.5)

    s4 = model.sensors[3]
    assert s4.id == 4
    assert s4.kind == "SOLID"
    assert s4.target_id == 30
    assert s4.dmin == pytest.approx(1e8)
    assert s4.dmax == pytest.approx(5e8)
