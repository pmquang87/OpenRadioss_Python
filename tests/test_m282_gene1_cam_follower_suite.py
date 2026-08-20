"""Tests for Milestone M282: Gene1 Failure Model, EngHelmholtzEnergy, CardanJoint, and SensorSpringNormalWork."""

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


def test_m282_fail_gene1_fixed(tmp_path: Path):
    c1 = f"{-100.0:>20.4f}{150.0:>20.4f}{400.0:>20.4f}{0.05:>20.4f}{1.0e-7:>20.6e}"
    c2 = f"{101:>10d}{0:>10d}{0.001:>20.4f}{500.0:>20.4f}{200.0:>20.4f}{0.8:>20.4f}"
    c3 = f"{102:>10d}{0:>10d}{0.002:>20.4f}{0.35:>20.4f}{0.40:>20.4f}{0.15:>20.4f}"
    c4 = f"{-0.2:>20.4f}{0.45:>20.4f}{103:>10d}{104:>10d}{105:>10d}"
    c5 = f"{106:>10d}{1:>10d}{0.005:>20.4f}{5:>10d}{1:>10d}{2:>10d}{0:>10d}{0.12:>20.4f}"
    c6 = f"{0.05:>20.4f}{1.5:>20.4f}{3:>10d}{0:>10d}{800.0:>20.4f}{1:>10d}"
    c7 = f"{107:>10d}{0:>10d}{1.2:>20.4f}{2.5:>20.4f}"
    c8 = f"{920:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Gene1 Failure Model Fixed Format Test
2022 0
/FAIL/GENE1/920
Gene1 Multi-Criterion Failure Model
{c1}
{c2}
{c3}
{c4}
{c5}
{c6}
{c7}
{c8}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 920 in model.fail_gene1s
    fg = model.fail_gene1s[920]
    assert pytest.approx(fg.pmin) == -100.0
    assert pytest.approx(fg.pmax) == 150.0
    assert pytest.approx(fg.sigp1_max) == 400.0
    assert pytest.approx(fg.tmax) == 0.05
    assert pytest.approx(fg.dtmin) == 1.0e-7
    assert fg.fct_idsm == 101
    assert pytest.approx(fg.sig_max) == 500.0
    assert pytest.approx(fg.sigr) == 200.0
    assert pytest.approx(fg.kf) == 0.8
    assert fg.fail_id == 920
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "GENE1"


def test_m282_fail_gene1_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Gene1 Free Format Test
/FAIL/GENE1/921
-50.0, 120.0, 350.0, 0.02, 2.0e-7
201, 0.001, 450.0, 180.0, 0.75
202, 0.002, 0.30, 0.35, 0.10
-0.15, 0.40, 203, 204, 205
206, 1, 0.005, 4, 1, 2, 0.10
0.04, 1.2, 2, 750.0, 1
207, 1.1, 2.0
921
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 921 in model.fail_gene1s
    fg = model.fail_gene1s[921]
    assert pytest.approx(fg.pmin) == -50.0
    assert pytest.approx(fg.pmax) == 120.0
    assert pytest.approx(fg.sigp1_max) == 350.0
    assert pytest.approx(fg.tmax) == 0.02
    assert fg.fct_idsm == 201
    assert pytest.approx(fg.sig_max) == 450.0
    assert fg.fail_id == 921


def test_m282_fail_gene1_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Gene1 Aliases Test
/FAIL/GENERIC1/922
-10.0, 100.0, 300.0, 0.01, 1.0e-7
/FAIL/GENE1_MODEL/923
-10.0, 100.0, 300.0, 0.01, 1.0e-7
/FAIL/GENE1_LAW/924
-10.0, 100.0, 300.0, 0.01, 1.0e-7
/FAIL/GENERIC_FAILURE_1/925
-10.0, 100.0, 300.0, 0.01, 1.0e-7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 922 in model.fail_gene1s
    assert 923 in model.fail_gene1s
    assert 924 in model.fail_gene1s
    assert 925 in model.fail_gene1s
    assert len(model.raw_fails) == 4


def test_m282_eng_helmholtz_energy(tmp_path: Path):
    c1 = f"{0.00025:>20.6f}{7:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Helmholtz Energy Fixed and Free Format Test
2022 0
/ENG/HELMHOLTZ_ENERGY/1
Fixed Helmholtz Energy
{c1}
/ENG/HELMHOLTZ_ENERGY/2
Free Helmholtz Energy
0.0005, 14
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_helmholtz_energies
    assert 2 in model.eng_helmholtz_energies
    he1 = model.eng_helmholtz_energies[1]
    assert pytest.approx(he1.dt_helm) == 0.00025
    assert he1.sens_id == 7
    he2 = model.eng_helmholtz_energies[2]
    assert pytest.approx(he2.dt_helm) == 0.0005
    assert he2.sens_id == 14


def test_m282_eng_helmholtz_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Helmholtz Energy Aliases Test
/ENG/HELMHOLTZ_WORK/11
0.001, 10
/ENG/EHELM/12
0.002, 11
/ENG/HELM_ENERGY/13
0.003, 12
/ENG/FREE_ENERGY/14
0.004, 13
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.eng_helmholtz_energies
    assert 12 in model.eng_helmholtz_energies
    assert 13 in model.eng_helmholtz_energies
    assert 14 in model.eng_helmholtz_energies
    assert pytest.approx(model.eng_helmholtz_energies[11].dt_helm) == 0.001
    assert pytest.approx(model.eng_helmholtz_energies[12].dt_helm) == 0.002
    assert pytest.approx(model.eng_helmholtz_energies[13].dt_helm) == 0.003
    assert pytest.approx(model.eng_helmholtz_energies[14].dt_helm) == 0.004


def test_m282_cam_follower_joint(tmp_path: Path):
    c1 = f"{131:>10d}{132:>10d}{133:>10d}{4.5e6:>20.4f}{6:>10d}{3.0e-6:>20.6e}"
    c2 = f"{0.0:>20.4f}{0.0:>20.4f}{1.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Cam Follower Joint Fixed and Free Format Test
2022 0
/LAGMUL/CAM_FOLLOWER_JOINT/1
Fixed Cam Follower Joint
{c1}
{c2}
/LAGMUL/CAM_FOLLOWER_JOINT/2
Free Cam Follower Joint
231, 232, 233, 3.8e6, 7, 4.5e-6
1.0, 0.0, 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_cam_follower_joints
    assert 2 in model.lagmul_cam_follower_joints
    cj1 = model.lagmul_cam_follower_joints[1]
    assert cj1.node1 == 131
    assert cj1.node2 == 132
    assert cj1.node3 == 133
    assert pytest.approx(cj1.stiff) == 4.5e6
    assert cj1.skew_id == 6
    assert pytest.approx(cj1.tol) == 3.0e-6
    assert pytest.approx(cj1.axis_x) == 0.0
    assert pytest.approx(cj1.axis_y) == 0.0
    assert pytest.approx(cj1.axis_z) == 1.0

    cj2 = model.lagmul_cam_follower_joints[2]
    assert cj2.node1 == 231
    assert cj2.node2 == 232
    assert cj2.node3 == 233
    assert pytest.approx(cj2.stiff) == 3.8e6
    assert cj2.skew_id == 7
    assert pytest.approx(cj2.tol) == 4.5e-6
    assert pytest.approx(cj2.axis_x) == 1.0
    assert pytest.approx(cj2.axis_y) == 0.0
    assert pytest.approx(cj2.axis_z) == 0.0


def test_m282_cam_follower_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Cam Follower Joint Aliases Test
/CAM_FOLLOWER_JOINT/51
501, 502, 503, 1.0e6, 0, 1.0e-6
0.0, 1.0, 0.0
/LAGMUL/CAM_FOLLOWER/52
504, 505, 506, 1.0e6, 0, 1.0e-6
0.0, 0.0, 1.0
/CAM_FOLLOWER/53
507, 508, 509, 1.0e6, 0, 1.0e-6
1.0, 0.0, 0.0
/CAM_FOLLOWER_MECHANISM/54
510, 511, 512, 1.0e6, 0, 1.0e-6
0.0, 1.0, 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 51 in model.lagmul_cam_follower_joints
    assert 52 in model.lagmul_cam_follower_joints
    assert 53 in model.lagmul_cam_follower_joints
    assert 54 in model.lagmul_cam_follower_joints
    assert model.lagmul_cam_follower_joints[51].node1 == 501
    assert model.lagmul_cam_follower_joints[52].node1 == 504
    assert model.lagmul_cam_follower_joints[53].node1 == 507
    assert model.lagmul_cam_follower_joints[54].node1 == 510


def test_m282_sensor_spring_normal_work(tmp_path: Path):
    c1 = f"{761:>10d}{6200.0:>20.4f}{0.0035:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Normal Work Fixed and Free Format Test
2022 0
/SENSOR/SPRING_NORMAL_WORK/1
Fixed Spring Normal Work Sensor
{c1}
/SENSOR/SPRING_NORMAL_WORK/2
Free Spring Normal Work Sensor
762, 8800.0, 0.0055
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_normal_works
    assert 2 in model.sensor_spring_normal_works
    s1 = model.sensor_spring_normal_works[1]
    assert s1.spring_id == 761
    assert pytest.approx(s1.w_norm_max) == 6200.0
    assert pytest.approx(s1.t_delay) == 0.0035

    s2 = model.sensor_spring_normal_works[2]
    assert s2.spring_id == 762
    assert pytest.approx(s2.w_norm_max) == 8800.0
    assert pytest.approx(s2.t_delay) == 0.0055

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_NORMAL_WORK"
    assert model.sensors[1].kind == "SPRING_NORMAL_WORK"


def test_m282_sensor_spring_normal_work_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Normal Work Aliases Test
/SENSOR/SPRING_NORM_WORK/11
861, 8500.0, 0.001
/SENSOR/SPRING_WORK_NORMAL/12
862, 9600.0, 0.002
/SENSOR/NORMAL_WORK_SPRING/13
863, 10700.0, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.sensor_spring_normal_works
    assert 12 in model.sensor_spring_normal_works
    assert 13 in model.sensor_spring_normal_works
    assert model.sensor_spring_normal_works[11].spring_id == 861
    assert pytest.approx(model.sensor_spring_normal_works[11].w_norm_max) == 8500.0
    assert model.sensor_spring_normal_works[12].spring_id == 862
    assert model.sensor_spring_normal_works[13].spring_id == 863
