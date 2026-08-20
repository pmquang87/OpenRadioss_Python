"""Tests for Milestone M283: Inievo Failure Model, EngEntropyProduction, ScrewNutJoint, and SensorSpringTotalForce."""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, deck_text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(deck_text, encoding="utf-8")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(p))
    parse_starter_deck(blocks, model, log)
    return model, log


def test_m283_fail_inievo_fixed(tmp_path: Path):
    c1 = f"{1:>10d}{1:>10d}{1:>10d}{'':>40s}{1:>10d}{0.05:>20.4f}"
    c2 = f"{1:>10d}{1:>10d}{1:>10d}{1:>10d}"
    c3 = f"{1:>10d}{0.01:>20.4f}{1.0:>20.4f}{2.5:>20.4f}"
    c4 = f"{2:>10d}{0.05:>20.4f}{1.2:>20.4f}"
    c5 = f"{0.1:>20.4f}{0.5:>20.4f}{50.0:>20.4f}"
    c6 = f"{930:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Inievo Failure Model Fixed Format Test
2022 0
/FAIL/INIEVO/930
Inievo Multi-Criterion Damage Initiation Evolution
{c1}
{c2}
{c3}
{c4}
{c5}
{c6}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 930 in model.fail_inievos
    fi = model.fail_inievos[930]
    assert fi.ninievo == 1
    assert fi.ishear == 1
    assert fi.ilen == 1
    assert fi.failip == 1
    assert pytest.approx(fi.pthk) == 0.05
    assert len(fi.models) == 1
    assert fi.models[0]["tab_id"] == 1
    assert pytest.approx(fi.models[0]["disp"]) == 0.1
    assert fi.fail_id == 930
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "INIEVO"


def test_m283_fail_inievo_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Inievo Free Format Test
/FAIL/INIEVO/931
1, 1, 1, 1, 0.04
1, 1, 1, 1
2, 0.02, 1.0, 3.0
3, 0.08, 1.1
0.2, 0.4, 60.0
931
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 931 in model.fail_inievos
    fi = model.fail_inievos[931]
    assert fi.ninievo == 1
    assert pytest.approx(fi.pthk) == 0.04
    assert len(fi.models) == 1
    assert fi.models[0]["tab_id"] == 2
    assert pytest.approx(fi.models[0]["disp"]) == 0.2
    assert fi.fail_id == 931


def test_m283_fail_inievo_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Inievo Aliases Test
/FAIL/INI_EVO/932
1, 0, 0, 0, 0.0
1, 1, 1, 1
10, 0.01, 1.0, 1.0
0, 0.0, 1.0
0.1, 0.1, 10.0
/FAIL/INIEVO_MODEL/933
1, 0, 0, 0, 0.0
1, 1, 1, 1
10, 0.01, 1.0, 1.0
0, 0.0, 1.0
0.1, 0.1, 10.0
/FAIL/INIEVO_LAW/934
1, 0, 0, 0, 0.0
1, 1, 1, 1
10, 0.01, 1.0, 1.0
0, 0.0, 1.0
0.1, 0.1, 10.0
/FAIL/DAMAGE_INIEVO/935
1, 0, 0, 0, 0.0
1, 1, 1, 1
10, 0.01, 1.0, 1.0
0, 0.0, 1.0
0.1, 0.1, 10.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 932 in model.fail_inievos
    assert 933 in model.fail_inievos
    assert 934 in model.fail_inievos
    assert 935 in model.fail_inievos
    assert len(model.raw_fails) == 4


def test_m283_eng_entropy_production(tmp_path: Path):
    c1 = f"{0.00035:>20.6f}{8:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Entropy Production Fixed and Free Format Test
2022 0
/ENG/ENTROPY_PRODUCTION/1
Fixed Entropy Production
{c1}
/ENG/ENTROPY_PRODUCTION/2
Free Entropy Production
0.0007, 16
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_entropy_productions
    assert 2 in model.eng_entropy_productions
    ep1 = model.eng_entropy_productions[1]
    assert pytest.approx(ep1.dt_entropy) == 0.00035
    assert ep1.sens_id == 8
    ep2 = model.eng_entropy_productions[2]
    assert pytest.approx(ep2.dt_entropy) == 0.0007
    assert ep2.sens_id == 16


def test_m283_eng_entropy_production_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Entropy Production Aliases Test
/ENG/ENTROPY_PROD/11
0.001, 10
/ENG/EENTROPY/12
0.002, 11
/ENG/ENTROPY_RATE/13
0.003, 12
/ENG/THERMAL_ENTROPY/14
0.004, 13
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.eng_entropy_productions
    assert 12 in model.eng_entropy_productions
    assert 13 in model.eng_entropy_productions
    assert 14 in model.eng_entropy_productions
    assert pytest.approx(model.eng_entropy_productions[11].dt_entropy) == 0.001
    assert pytest.approx(model.eng_entropy_productions[12].dt_entropy) == 0.002
    assert pytest.approx(model.eng_entropy_productions[13].dt_entropy) == 0.003
    assert pytest.approx(model.eng_entropy_productions[14].dt_entropy) == 0.004


def test_m283_screw_nut_joint(tmp_path: Path):
    c1 = f"{141:>10d}{142:>10d}{143:>10d}{5.5e6:>20.4f}{4:>10d}{2.0e-6:>20.6e}"
    c2 = f"{5.0:>20.4f}{10.0:>20.4f}{1.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Screw Nut Joint Fixed and Free Format Test
2022 0
/LAGMUL/SCREW_NUT_JOINT/1
Fixed Screw Nut Joint
{c1}
{c2}
/LAGMUL/SCREW_NUT_JOINT/2
Free Screw Nut Joint
241, 242, 243, 6.2e6, 5, 3.5e-6
4.0, 8.0, 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_screw_nut_joints
    assert 2 in model.lagmul_screw_nut_joints
    sn1 = model.lagmul_screw_nut_joints[1]
    assert sn1.node1 == 141
    assert sn1.node2 == 142
    assert sn1.node3 == 143
    assert pytest.approx(sn1.stiff) == 5.5e6
    assert sn1.skew_id == 4
    assert pytest.approx(sn1.tol) == 2.0e-6
    assert pytest.approx(sn1.pitch) == 5.0
    assert pytest.approx(sn1.lead) == 10.0
    assert pytest.approx(sn1.axis_z) == 1.0

    sn2 = model.lagmul_screw_nut_joints[2]
    assert sn2.node1 == 241
    assert sn2.node2 == 242
    assert sn2.node3 == 243
    assert pytest.approx(sn2.stiff) == 6.2e6
    assert sn2.skew_id == 5
    assert pytest.approx(sn2.tol) == 3.5e-6
    assert pytest.approx(sn2.pitch) == 4.0
    assert pytest.approx(sn2.lead) == 8.0
    assert pytest.approx(sn2.axis_z) == 1.0


def test_m283_screw_nut_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Screw Nut Joint Aliases Test
/SCREW_NUT_JOINT/61
601, 602, 603, 1.0e6, 0, 1.0e-6
5.0, 5.0, 1.0
/LAGMUL/SCREW_NUT/62
604, 605, 606, 1.0e6, 0, 1.0e-6
5.0, 5.0, 1.0
/SCREW_NUT/63
607, 608, 609, 1.0e6, 0, 1.0e-6
5.0, 5.0, 1.0
/SCREW_NUT_MECHANISM/64
610, 611, 612, 1.0e6, 0, 1.0e-6
5.0, 5.0, 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 61 in model.lagmul_screw_nut_joints
    assert 62 in model.lagmul_screw_nut_joints
    assert 63 in model.lagmul_screw_nut_joints
    assert 64 in model.lagmul_screw_nut_joints
    assert model.lagmul_screw_nut_joints[61].node1 == 601
    assert model.lagmul_screw_nut_joints[62].node1 == 604
    assert model.lagmul_screw_nut_joints[63].node1 == 607
    assert model.lagmul_screw_nut_joints[64].node1 == 610


def test_m283_sensor_spring_total_force(tmp_path: Path):
    c1 = f"{771:>10d}{15500.0:>20.4f}{0.0045:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Force Fixed and Free Format Test
2022 0
/SENSOR/SPRING_TOTAL_FORCE/1
Fixed Spring Total Force Sensor
{c1}
/SENSOR/SPRING_TOTAL_FORCE/2
Free Spring Total Force Sensor
772, 18500.0, 0.0065
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_forces
    assert 2 in model.sensor_spring_total_forces
    s1 = model.sensor_spring_total_forces[1]
    assert s1.spring_id == 771
    assert pytest.approx(s1.f_tot_max) == 15500.0
    assert pytest.approx(s1.t_delay) == 0.0045

    s2 = model.sensor_spring_total_forces[2]
    assert s2.spring_id == 772
    assert pytest.approx(s2.f_tot_max) == 18500.0
    assert pytest.approx(s2.t_delay) == 0.0065

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_TOTAL_FORCE"
    assert model.sensors[1].kind == "SPRING_TOTAL_FORCE"


def test_m283_sensor_spring_total_force_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Force Aliases Test
/SENSOR/SPRING_TOT_FORCE/11
871, 14000.0, 0.001
/SENSOR/SPRING_FORCE_TOTAL/12
872, 16000.0, 0.002
/SENSOR/TOTAL_FORCE_SPRING/13
873, 19000.0, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.sensor_spring_total_forces
    assert 12 in model.sensor_spring_total_forces
    assert 13 in model.sensor_spring_total_forces
    assert model.sensor_spring_total_forces[11].spring_id == 871
    assert pytest.approx(model.sensor_spring_total_forces[11].f_tot_max) == 14000.0
    assert model.sensor_spring_total_forces[12].spring_id == 872
    assert model.sensor_spring_total_forces[13].spring_id == 873
