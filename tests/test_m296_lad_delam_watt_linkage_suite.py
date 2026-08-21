"""Tests for Milestone M296: LadDelam Failure Model, EngMagnetocaloricEnergy, WattLinkageJoint, and SensorSpringResultantJerk."""

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


def test_m296_fail_lad_delam_fixed(tmp_path: Path):
    c1 = f"{280.0:>20.4f}{650.0:>20.4f}{820.0:>20.4f}{1.75:>20.4f}{0.950:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1055:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Delamination Model Fixed Format Test
2022 0
/FAIL/LAD_DELAM/1055
Ladeveze Interlaminar Fracture Criterion
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1055 in model.fail_laddelams
    fld = model.fail_laddelams[1055]
    assert pytest.approx(fld.g_1c) == 280.0
    assert pytest.approx(fld.g_2c) == 650.0
    assert pytest.approx(fld.g_3c) == 820.0
    assert pytest.approx(fld.gamma_delam) == 1.75
    assert pytest.approx(fld.d_delam_max) == 0.950
    assert fld.ifail_sh == 1
    assert fld.ifail_so == 2
    assert fld.fail_id == 1055
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DELAM"


def test_m296_fail_lad_delam_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Delamination Free Format Test
/FAIL/LAD_DELAM/1056
320.0, 700.0, 900.0, 2.0, 0.920
1, 1
1056
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1056 in model.fail_laddelams
    fld = model.fail_laddelams[1056]
    assert pytest.approx(fld.g_1c) == 320.0
    assert pytest.approx(fld.g_2c) == 700.0
    assert pytest.approx(fld.g_3c) == 900.0
    assert pytest.approx(fld.gamma_delam) == 2.0
    assert pytest.approx(fld.d_delam_max) == 0.920
    assert fld.fail_id == 1056


def test_m296_fail_lad_delam_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Delamination Aliases Test
/FAIL/LADEVEZE_INTERLAMINAR_FRACTURE/1057
250.0, 600.0, 750.0, 1.5, 0.98
1, 1
/FAIL/LAD_DELAMINATION/1058
250.0, 600.0, 750.0, 1.5, 0.98
1, 1
/FAIL/LAD_DELAM_MODEL/1059
250.0, 600.0, 750.0, 1.5, 0.98
1, 1
/FAIL/LAD_DELAM_LAW/1060
250.0, 600.0, 750.0, 1.5, 0.98
1, 1
/FAIL/LADEVEZE_INTERLAMINAR_DELAMINATION/1061
250.0, 600.0, 750.0, 1.5, 0.98
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1057 in model.fail_laddelams
    assert 1058 in model.fail_laddelams
    assert 1059 in model.fail_laddelams
    assert 1060 in model.fail_laddelams
    assert 1061 in model.fail_laddelams
    assert len(model.raw_fails) == 5


def test_m296_eng_magnetocaloric_energy(tmp_path: Path):
    c1 = f"{0.00045:>20.6f}{48:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Magnetocaloric Energy Fixed and Free Format Test
2022 0
/ENG/MAGNETOCALORIC_ENERGY/1
Fixed Magnetocaloric Energy Output
{c1}
/ENG/MAGNETOCALORIC_ENERGY/2
Free Magnetocaloric Energy Output
0.0009, 96
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_magnetocaloric_energies
    assert 2 in model.eng_magnetocaloric_energies
    me1 = model.eng_magnetocaloric_energies[1]
    assert pytest.approx(me1.dt_magnetocaloric) == 0.00045
    assert me1.sens_id == 48
    me2 = model.eng_magnetocaloric_energies[2]
    assert pytest.approx(me2.dt_magnetocaloric) == 0.0009
    assert me2.sens_id == 96


def test_m296_eng_magnetocaloric_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Magnetocaloric Energy Aliases Test
/ENG/MC_WORK/21
0.001, 20
/ENG/EMAGNETOCALORIC/22
0.002, 21
/ENG/MAGNETOCALORIC_DISSIPATION/23
0.003, 22
/ENG/EM_MAGNETOCALORIC/24
0.004, 23
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 21 in model.eng_magnetocaloric_energies
    assert 22 in model.eng_magnetocaloric_energies
    assert 23 in model.eng_magnetocaloric_energies
    assert 24 in model.eng_magnetocaloric_energies
    assert pytest.approx(model.eng_magnetocaloric_energies[21].dt_magnetocaloric) == 0.001
    assert pytest.approx(model.eng_magnetocaloric_energies[22].dt_magnetocaloric) == 0.002
    assert pytest.approx(model.eng_magnetocaloric_energies[23].dt_magnetocaloric) == 0.003
    assert pytest.approx(model.eng_magnetocaloric_energies[24].dt_magnetocaloric) == 0.004


def test_m296_watt_linkage_joint(tmp_path: Path):
    c1 = f"{791:>10d}{792:>10d}{793:>10d}{7.5e6:>20.4f}{1:>10d}{1.4e-6:>20.6e}"
    c2 = f"{100.0:>20.4f}{40.0:>20.4f}{40.0:>20.4f}{50.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Watt Linkage Joint Fixed and Free Format Test
2022 0
/LAGMUL/WATT_LINKAGE_JOINT/1
Fixed Watt Linkage Joint
{c1}
{c2}
/LAGMUL/WATT_LINKAGE_JOINT/2
Free Watt Linkage Joint
891, 892, 893, 8.2e6, 2, 1.8e-6
120.0, 50.0, 50.0, 60.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_watt_linkage_joints
    assert 2 in model.lagmul_watt_linkage_joints
    wt1 = model.lagmul_watt_linkage_joints[1]
    assert wt1.node1 == 791
    assert wt1.node2 == 792
    assert wt1.node3 == 793
    assert pytest.approx(wt1.stiff) == 7.5e6
    assert wt1.skew_id == 1
    assert pytest.approx(wt1.tol) == 1.4e-6
    assert pytest.approx(wt1.ground_len) == 100.0
    assert pytest.approx(wt1.link1_len) == 40.0
    assert pytest.approx(wt1.link2_len) == 40.0
    assert pytest.approx(wt1.coupler_len) == 50.0

    wt2 = model.lagmul_watt_linkage_joints[2]
    assert wt2.node1 == 891
    assert wt2.node2 == 892
    assert wt2.node3 == 893
    assert pytest.approx(wt2.stiff) == 8.2e6
    assert wt2.skew_id == 2
    assert pytest.approx(wt2.tol) == 1.8e-6
    assert pytest.approx(wt2.ground_len) == 120.0
    assert pytest.approx(wt2.link1_len) == 50.0
    assert pytest.approx(wt2.link2_len) == 50.0
    assert pytest.approx(wt2.coupler_len) == 60.0


def test_m296_watt_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Watt Linkage Joint Aliases Test
/WATT_LINKAGE_JOINT/31
971, 972, 973, 2.0e6, 0, 1.0e-6
80.0, 30.0, 30.0, 40.0
/LAGMUL/WATT_LINKAGE/32
974, 975, 976, 2.0e6, 0, 1.0e-6
80.0, 30.0, 30.0, 40.0
/WATT_LINKAGE/33
977, 978, 979, 2.0e6, 0, 1.0e-6
80.0, 30.0, 30.0, 40.0
/WATT_STRAIGHT_LINE_MECHANISM/34
980, 981, 982, 2.0e6, 0, 1.0e-6
80.0, 30.0, 30.0, 40.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 31 in model.lagmul_watt_linkage_joints
    assert 32 in model.lagmul_watt_linkage_joints
    assert 33 in model.lagmul_watt_linkage_joints
    assert 34 in model.lagmul_watt_linkage_joints
    assert model.lagmul_watt_linkage_joints[31].node1 == 971
    assert model.lagmul_watt_linkage_joints[32].node1 == 974
    assert model.lagmul_watt_linkage_joints[33].node1 == 977
    assert model.lagmul_watt_linkage_joints[34].node1 == 980


def test_m296_sensor_spring_resultant_jerk(tmp_path: Path):
    c1 = f"{918:>10d}{3.4e8:>20.4f}{0.0045:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Resultant Jerk Fixed and Free Format Test
2022 0
/SENSOR/SPRING_RESULTANT_JERK/1
Fixed Spring Resultant Jerk Sensor
{c1}
/SENSOR/SPRING_RESULTANT_JERK/2
Free Spring Resultant Jerk Sensor
919, 4.2e8, 0.0065
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_resultant_jerks
    assert 2 in model.sensor_spring_resultant_jerks
    s1 = model.sensor_spring_resultant_jerks[1]
    assert s1.spring_id == 918
    assert pytest.approx(s1.jres_max) == 3.4e8
    assert pytest.approx(s1.t_delay) == 0.0045

    s2 = model.sensor_spring_resultant_jerks[2]
    assert s2.spring_id == 919
    assert pytest.approx(s2.jres_max) == 4.2e8
    assert pytest.approx(s2.t_delay) == 0.0065

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_RESULTANT_JERK"
    assert model.sensors[1].kind == "SPRING_RESULTANT_JERK"


def test_m296_sensor_spring_resultant_jerk_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Resultant Jerk Aliases Test
/SENSOR/SPRING_RES_JERK/61
991, 2.1e8, 0.0015
/SENSOR/SPRING_JERK_RES/62
992, 2.2e8, 0.0025
/SENSOR/RESULTANT_JERK_SPRING/63
993, 2.3e8, 0.0035
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 61 in model.sensor_spring_resultant_jerks
    assert 62 in model.sensor_spring_resultant_jerks
    assert 63 in model.sensor_spring_resultant_jerks
    assert model.sensor_spring_resultant_jerks[61].spring_id == 991
    assert pytest.approx(model.sensor_spring_resultant_jerks[61].jres_max) == 2.1e8
    assert model.sensor_spring_resultant_jerks[62].spring_id == 992
    assert model.sensor_spring_resultant_jerks[63].spring_id == 993
