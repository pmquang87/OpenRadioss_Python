"""Tests for Milestone M201: Substructure TH, Group TH Part, Wave Shaper DFS, Sensors, and Composite Properties Suite."""
from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    ThSubs, ThPartGroup, DfsWavSha, SensorDistSurf, SensorSensAndOr, PropPcompp, PropType51
)


def _parse_starter(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_m201_th_subs(tmp_path: Path):
    deck_text = """/BEGIN
Test M201 TH_SUBS
/TH/SUBS/101
Substructure Time History 1
DEF
         1         2         3
/ATH/SUBS/102
Substructure Time History 2
ENERGY DISP
10 20
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)
    assert not log.errors
    assert 101 in model.th_subs
    assert 102 in model.th_subs

    s1 = model.th_subs[101]
    assert isinstance(s1, ThSubs)
    assert s1.id == 101
    assert "DEF" in s1.vars or len(s1.vars) > 0
    assert 1 in s1.subs_ids
    assert 2 in s1.subs_ids
    assert 3 in s1.subs_ids

    s2 = model.th_subs[102]
    assert s2.id == 102
    assert s2.prefix == "ATH"
    assert "ENERGY" in s2.vars
    assert "DISP" in s2.vars
    assert s2.subs_ids == [10, 20]


def test_m201_thpart_group(tmp_path: Path):
    deck_text = """/BEGIN
Test M201 THPART Group
/THPART/GRSHEL/1
Shell Group Part TH
        12
/THPART/GRBRIC/2
Solid Group Part TH
        34
/THPART/GRBEAM/3
Beam Group Part TH
56
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)
    assert not log.errors
    assert 1 in model.th_part_groups
    assert 2 in model.th_part_groups
    assert 3 in model.th_part_groups

    p1 = model.th_part_groups[1]
    assert isinstance(p1, ThPartGroup)
    assert p1.elem_type == "SHEL"
    assert p1.grelem_id == 12

    p2 = model.th_part_groups[2]
    assert p2.elem_type == "BRIC"
    assert p2.grelem_id == 34

    p3 = model.th_part_groups[3]
    assert p3.elem_type == "BEAM"
    assert p3.grelem_id == 56


def test_m201_dfs_wav_sha(tmp_path: Path):
    deck_text = """/BEGIN
Test M201 DFS WAV SHA
/DFS/WAV_SHA/1
Wave Shaper Detonator Point
                1.23                4.56                7.89                0.01         5         9
/WAVE/2
Wave Alias Detonator Point
10.0 20.0 30.0 0.05 6 11
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)
    assert not log.errors
    assert 1 in model.dfs_wav_shas
    assert 2 in model.dfs_wav_shas

    w1 = model.dfs_wav_shas[1]
    assert isinstance(w1, DfsWavSha)
    assert pytest.approx(w1.xdet, 1e-4) == 1.23
    assert pytest.approx(w1.ydet, 1e-4) == 4.56
    assert pytest.approx(w1.zdet, 1e-4) == 7.89
    assert pytest.approx(w1.tdet, 1e-4) == 0.01
    assert w1.mat_id == 5
    assert w1.grnod_id == 9

    w2 = model.dfs_wav_shas[2]
    assert pytest.approx(w2.xdet, 1e-4) == 10.0
    assert pytest.approx(w2.ydet, 1e-4) == 20.0
    assert pytest.approx(w2.zdet, 1e-4) == 30.0
    assert pytest.approx(w2.tdet, 1e-4) == 0.05
    assert w2.mat_id == 6
    assert w2.grnod_id == 11


def test_m201_sensor_dist_surf(tmp_path: Path):
    deck_text = """/BEGIN
Test M201 SENSOR DIST SURF
/SENSOR/DIST_SURF/1
Surface Distance Sensor
                 0.5
        10         1        20        30        40
                0.01                 5.0                 0.0                 0.1
/SENSOR/TYPE17/2
Type17 Surface Distance Sensor
0.2
15 2 25 35 45
0.02 6.0 0.0 0.15
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)
    assert not log.errors
    assert 1 in model.sensors_dist_surf
    assert 2 in model.sensors_dist_surf

    s1 = model.sensors_dist_surf[1]
    assert isinstance(s1, SensorDistSurf)
    assert s1.id == 1
    assert pytest.approx(s1.tdelay, 1e-4) == 0.5
    assert s1.node_id == 10
    assert s1.surf_id == 1
    assert s1.node_id1 == 20
    assert s1.node_id2 == 30
    assert s1.node_id3 == 40
    assert pytest.approx(s1.dist_min, 1e-4) == 0.01
    assert pytest.approx(s1.dist_max, 1e-4) == 5.0
    assert pytest.approx(s1.tmin, 1e-4) == 0.1

    s2 = model.sensors_dist_surf[2]
    assert s2.id == 2
    assert pytest.approx(s2.tdelay, 1e-4) == 0.2
    assert s2.node_id == 15
    assert s2.surf_id == 2
    assert s2.node_id1 == 25
    assert s2.node_id2 == 35
    assert s2.node_id3 == 45
    assert pytest.approx(s2.dist_min, 1e-4) == 0.02
    assert pytest.approx(s2.dist_max, 1e-4) == 6.0
    assert pytest.approx(s2.tmin, 1e-4) == 0.15


def test_m201_sensor_sens_and_or(tmp_path: Path):
    deck_text = """/BEGIN
Test M201 SENSOR SENS AND OR
/SENSOR/SENS_AND_OR/1
Compound Logic Sensor
0.1
10 20
/SENSOR/OR/2
Logical OR Sensor
0.05
30 40
/SENSOR/SENS/3
Type 3 Sensor
0.2
50 60
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)
    assert not log.errors
    assert 1 in model.sensors_sens_and_or
    assert 2 in model.sensors_sens_and_or
    assert 3 in model.sensors_sens_and_or

    s1 = model.sensors_sens_and_or[1]
    assert isinstance(s1, SensorSensAndOr)
    assert s1.id == 1
    assert pytest.approx(s1.tdelay, 1e-4) == 0.1
    assert s1.sensor_id1 == 10
    assert s1.sensor_id2 == 20

    s2 = model.sensors_sens_and_or[2]
    assert s2.id == 2
    assert s2.logic_type == "OR"
    assert pytest.approx(s2.tdelay, 1e-4) == 0.05
    assert s2.sensor_id1 == 30
    assert s2.sensor_id2 == 40

    s3 = model.sensors_sens_and_or[3]
    assert s3.id == 3
    assert pytest.approx(s3.tdelay, 1e-4) == 0.2
    assert s3.sensor_id1 == 50
    assert s3.sensor_id2 == 60


def test_m201_prop_pcompp(tmp_path: Path):
    deck_text = """/BEGIN
Test M201 PROP PCOMPP
/PROP/PCOMPP/100
Composite Laminate Property
       555
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)
    assert not log.errors
    assert 100 in model.props_pcompp
    assert 100 in model.properties

    p = model.props_pcompp[100]
    assert isinstance(p, PropPcompp)
    assert p.id == 100
    assert p.laminate_id == 555
    assert model.properties[100].params["laminate_id"] == 555


def test_m201_prop_p51_and_type51(tmp_path: Path):
    deck_text = """/BEGIN
Test M201 PROP P51 and TYPE51
/PROP/TYPE51/201
Cohesive Layered Shell Type51
         1         2         3         4                1.25                0.15
                0.01                0.02                0.03                0.04                0.05
                0.83         5         6                0.07
                1.00                0.00                0.00         1         2         3         4
/PROP/P51/202
P51 Alias Property
2 1 0 0 2.5 0.2
0.02 0.03 0.04 0.05 0.06
0.85 6 7 0.08
0.0 1.0 0.0 2 3 4 5
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)
    assert not log.errors
    assert 201 in model.props_type51
    assert 202 in model.props_type51

    p1 = model.props_type51[201]
    assert isinstance(p1, PropType51)
    assert p1.id == 201
    assert p1.ishell == 1
    assert p1.ismstr == 2
    assert p1.ish3n == 3
    assert p1.idrill == 4
    assert pytest.approx(p1.pthk, 1e-4) == 1.25
    assert pytest.approx(p1.zshift, 1e-4) == 0.15
    assert pytest.approx(p1.hm, 1e-4) == 0.01
    assert pytest.approx(p1.hf, 1e-4) == 0.02
    assert pytest.approx(p1.hr, 1e-4) == 0.03
    assert pytest.approx(p1.dm, 1e-4) == 0.04
    assert pytest.approx(p1.dn, 1e-4) == 0.05
    assert pytest.approx(p1.ashear, 1e-4) == 0.83
    assert p1.iint == 5
    assert p1.ithick == 6
    assert pytest.approx(p1.failexp, 1e-4) == 0.07
    assert pytest.approx(p1.vx, 1e-4) == 1.0
    assert pytest.approx(p1.vy, 1e-4) == 0.0
    assert pytest.approx(p1.vz, 1e-4) == 0.0
    assert p1.idsk == 1
    assert p1.iorth == 2
    assert p1.ipos == 3
    assert p1.irp == 4

    p2 = model.props_type51[202]
    assert p2.id == 202
    assert p2.ishell == 2
    assert p2.ismstr == 1
    assert pytest.approx(p2.pthk, 1e-4) == 2.5
    assert pytest.approx(p2.zshift, 1e-4) == 0.2
    assert pytest.approx(p2.ashear, 1e-4) == 0.85
    assert p2.iint == 6
    assert p2.ithick == 7
