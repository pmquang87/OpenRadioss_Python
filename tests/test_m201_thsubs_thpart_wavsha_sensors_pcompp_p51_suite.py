"""Test suite for Milestone M201: Substructure TH, Group TH Part, Wave Shaper DFS,
Surface Distance & Compound Boolean Sensors, and PCOMPP/P51 Composite Properties Suite.
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


def test_th_subs_parsing(tmp_path: Path):
    """Verify /TH/SUBS and /ATH/SUBS in free format."""
    deck_text = """# RADIOSS STARTER DECK
/BEGIN
Substructure TH Test
                  1                 1
/TH/SUBS/1
Substructure Time History 1
DEF
101 102 103
/ATH/SUBS/2
Substructure Time History 2
ENERGY DISP
201 202
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)

    assert 1 in model.th_subs
    th1 = model.th_subs[1]
    assert th1.id == 1
    assert th1.title == "Substructure Time History 1"
    assert th1.prefix in ("TH", "TH_SUBS")
    assert th1.vars == ["DEF"]
    assert th1.subs_ids == [101, 102, 103]

    assert 2 in model.th_subs
    th2 = model.th_subs[2]
    assert th2.id == 2
    assert th2.title == "Substructure Time History 2"
    assert th2.prefix in ("ATH", "ATH_SUBS")
    assert th2.vars == ["ENERGY", "DISP"]
    assert th2.subs_ids == [201, 202]

    # Verify th_requests
    assert len(model.th_requests) >= 2


def test_thpart_group_parsing(tmp_path: Path):
    """Verify /THPART/GRSHEL, /THPART/GRBEAM, /THPART/GRBRIC, etc."""
    deck_text = """# RADIOSS STARTER DECK
/BEGIN
THPart Group Test
                  1                 1
/THPART/GRSHEL/10
Shell Group TH
501
/THPART/GRBEAM/20
Beam Group TH
502
/THPART/GRBRIC/30
Brick Group TH
503
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)

    assert 10 in model.th_part_groups
    th_shel = model.th_part_groups[10]
    assert th_shel.id == 10
    assert th_shel.title == "Shell Group TH"
    assert th_shel.elem_type == "SHEL"
    assert th_shel.grelem_id == 501

    assert 20 in model.th_part_groups
    th_beam = model.th_part_groups[20]
    assert th_beam.id == 20
    assert th_beam.elem_type == "BEAM"
    assert th_beam.grelem_id == 502

    assert 30 in model.th_part_groups
    th_bric = model.th_part_groups[30]
    assert th_bric.id == 30
    assert th_bric.elem_type == "BRIC"
    assert th_bric.grelem_id == 503


def test_dfs_wav_sha_and_wave_parsing(tmp_path: Path):
    """Verify /DFS/WAV_SHA and /WAVE."""
    deck_text = """# RADIOSS STARTER DECK
/BEGIN
DFS WAV_SHA Test
                  1                 1
/DFS/WAV_SHA/1
Wave Shaper Detonation Point 1
1.5 2.5 3.5 0.005 3 42
/WAVE/2
Wave Shaper Detonation Point 2
10.0 20.0 30.0 0.010 4 99
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)

    assert 1 in model.dfs_wav_shas
    ws1 = model.dfs_wav_shas[1]
    assert ws1.id == 1
    assert ws1.title == "Wave Shaper Detonation Point 1"
    assert ws1.xdet == 1.5
    assert ws1.ydet == 2.5
    assert ws1.zdet == 3.5
    assert ws1.tdet == 0.005
    assert ws1.mat_id == 3
    assert ws1.grnod_id == 42

    assert 2 in model.dfs_wav_shas
    ws2 = model.dfs_wav_shas[2]
    assert ws2.id == 2
    assert ws2.xdet == 10.0
    assert ws2.ydet == 20.0
    assert ws2.zdet == 30.0
    assert ws2.tdet == 0.010
    assert ws2.mat_id == 4
    assert ws2.grnod_id == 99

    assert len(model.det_points) >= 2


def test_sensor_dist_surf_and_sens_and_or_parsing(tmp_path: Path):
    """Verify /SENSOR/DIST_SURF, /SENSOR/SENS_AND_OR, /SENSOR/AND, /SENSOR/OR, /SENSOR/WORK."""
    deck_text = """# RADIOSS STARTER DECK
/BEGIN
Sensor Suite Test
                  1                 1
/SENSOR/DIST_SURF/1
Surface Distance Sensor
0.001
10 100 20 30 40
0.5 5.0 0.0 0.002
/SENSOR/SENS_AND_OR/2
Logical And/Or Sensor
0.0005
3 4
/SENSOR/AND/3
Logical And Sensor
0.0
5 6
/SENSOR/OR/4
Logical Or Sensor
0.0
7 8
/SENSOR/WORK/5
Work Sensor
0.001
1001 1002 500.0 0.01
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)

    # Sensor DIST_SURF
    assert 1 in model.sensors_dist_surf
    s_dist = model.sensors_dist_surf[1]
    assert s_dist.id == 1
    assert s_dist.title == "Surface Distance Sensor"
    assert s_dist.tdelay == 0.001
    assert s_dist.node_id == 10
    assert s_dist.surf_id == 100
    assert s_dist.dist_min == 0.5
    assert s_dist.dist_max == 5.0
    assert s_dist.tmin == 0.002

    # Sensor SENS_AND_OR
    assert 2 in model.sensors_sens_and_or
    s_logic = model.sensors_sens_and_or[2]
    assert s_logic.id == 2
    assert s_logic.title == "Logical And/Or Sensor"
    assert s_logic.sensor_id1 == 3
    assert s_logic.sensor_id2 == 4
    assert s_logic.t_delay == 0.0005

    assert 3 in model.sensors_sens_and_or
    assert model.sensors_sens_and_or[3].logic_type == "AND"
    assert model.sensors_sens_and_or[3].sensor_id1 == 5
    assert model.sensors_sens_and_or[3].sensor_id2 == 6

    assert 4 in model.sensors_sens_and_or
    assert model.sensors_sens_and_or[4].logic_type == "OR"
    assert model.sensors_sens_and_or[4].sensor_id1 == 7
    assert model.sensors_sens_and_or[4].sensor_id2 == 8

    # Sensor WORK
    assert 5 in model.sensors_work
    s_work = model.sensors_work[5]
    assert s_work.id == 5
    assert s_work.title == "Work Sensor"
    assert s_work.object_id == 1001
    assert s_work.sens_type == 1002
    assert s_work.w_max == 500.0


def test_prop_pcompp_and_type51_parsing(tmp_path: Path):
    """Verify /PROP/PCOMPP, /PROP/TYPE51, /PROP/P51, /PROP/TSH_P51."""
    deck_text = """# RADIOSS STARTER DECK
/BEGIN
Composite Properties Test
                  1                 1
/PROP/PCOMPP/101
Ply-based Composite Property
77
/PROP/TYPE51/201
Cohesive Layered Shell Property
1 2 3 4 2.5 0.5
0.1 0.2 0.3 0.01 0.02
0.833333 5 1 0.05
1.0 0.0 0.0 1 2 3 4
/PROP/P51/202
P51 Property Alias
1 1 1 1 1.0 0.0
0.0 0.0 0.0 0.0 0.0
0.833333 3 0 0.0
1.0 0.0 0.0 0 0 0 0
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)

    # PCOMPP
    assert 101 in model.props_pcompp
    pcomp = model.props_pcompp[101]
    assert pcomp.id == 101
    assert pcomp.title == "Ply-based Composite Property"
    assert pcomp.laminate_id == 77
    assert 101 in model.properties
    assert model.properties[101].params["laminate_id"] == 77

    # TYPE51
    assert 201 in model.props_type51
    p51 = model.props_type51[201]
    assert p51.id == 201
    assert p51.title == "Cohesive Layered Shell Property"
    assert p51.ishell == 1
    assert p51.ismstr == 2
    assert p51.ish3n == 3
    assert p51.idrill == 4
    assert p51.pthk == 2.5
    assert p51.zshift == 0.5
    assert p51.hm == 0.1
    assert p51.hf == 0.2
    assert p51.hr == 0.3
    assert p51.dm == 0.01
    assert p51.dn == 0.02
    assert p51.iint == 5
    assert p51.ithick == 1
    assert p51.failexp == 0.05
    assert p51.idsk == 1
    assert p51.iorth == 2
    assert p51.ipos == 3
    assert p51.irp == 4

    # P51 alias
    assert 202 in model.props_type51
    p51_2 = model.props_type51[202]
    assert p51_2.id == 202
    assert p51_2.pthk == 1.0


def test_m201_fixed_format_deck_parsing(tmp_path: Path):
    """Verify fixed-format card parsing for M201 keywords."""
    deck_text = """# RADIOSS STARTER DECK
/BEGIN
Fixed Format M201 Test
      2022         0
/DFS/WAV_SHA/1
Wave Shaper Fixed
                 1.0                 2.0                 3.0              0.0001        10        20
/SENSOR/DIST_SURF/2
Fixed Dist Surf
               0.001
        10       100        20        30        40
                 0.1                 1.0                                                  0.0005
/PROP/PCOMPP/3
Fixed PCOMPP
        88
/PROP/TYPE51/4
Fixed TYPE51
        10        20        30        40                 1.5                 0.2
                 0.1                 0.2                 0.3                0.01                0.02
            0.833333        10        20                 0.0
                 1.0                 0.0                 0.0         1         2         3         4
/END
"""
    model, log = _parse_starter(tmp_path, deck_text)

    # DFS WAV_SHA fixed
    assert 1 in model.dfs_wav_shas
    ws = model.dfs_wav_shas[1]
    assert ws.xdet == 1.0
    assert ws.ydet == 2.0
    assert ws.zdet == 3.0
    assert ws.tdet == 0.0001
    assert ws.mat_id == 10
    assert ws.grnod_id == 20

    # SENSOR DIST_SURF fixed
    assert 2 in model.sensors_dist_surf
    s_dist = model.sensors_dist_surf[2]
    assert s_dist.node_id == 10
    assert s_dist.surf_id == 100
    assert s_dist.dist_min == 0.1
    assert s_dist.dist_max == 1.0
    assert s_dist.tmin == 0.0005

    # PROP PCOMPP fixed
    assert 3 in model.props_pcompp
    assert model.props_pcompp[3].laminate_id == 88

    # PROP TYPE51 fixed
    assert 4 in model.props_type51
    p51 = model.props_type51[4]
    assert p51.ishell == 10
    assert p51.ismstr == 20
    assert p51.ish3n == 30
    assert p51.idrill == 40
    assert p51.pthk == 1.5
    assert p51.zshift == 0.2
    assert p51.hm == 0.1
    assert p51.iint == 10
    assert p51.ithick == 20
    assert p51.idsk == 1
    assert p51.iorth == 2
    assert p51.ipos == 3
    assert p51.irp == 4
