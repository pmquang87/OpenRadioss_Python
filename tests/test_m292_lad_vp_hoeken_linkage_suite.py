"""Tests for Milestone M292: LadViscoPlast Failure Model, EngDielectricLossEnergy, HoekenLinkageJoint, and SensorSpringBendingAcceleration."""

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


def test_m292_fail_lad_visco_plast_fixed(tmp_path: Path):
    c1 = f"{150.0:>20.4f}{2.500:>20.4f}{450.0:>20.4f}{0.850:>20.4f}{0.985:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1018:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Viscoplastic Failure Model Fixed Format Test
2022 0
/FAIL/LAD_VISCO_PLAST/1018
Ladeveze Dynamic Viscoplastic Damage Evolution
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1018 in model.fail_ladviscoplasts
    flvp = model.fail_ladviscoplasts[1018]
    assert pytest.approx(flvp.gamma_vp) == 150.0
    assert pytest.approx(flvp.m_vp) == 2.500
    assert pytest.approx(flvp.a_vp) == 450.0
    assert pytest.approx(flvp.p_vp) == 0.850
    assert pytest.approx(flvp.d_max_vp) == 0.985
    assert flvp.ifail_sh == 1
    assert flvp.ifail_so == 2
    assert flvp.fail_id == 1018
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_VISCO_PLAST"


def test_m292_fail_lad_visco_plast_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Viscoplastic Free Format Test
/FAIL/LAD_VISCO_PLAST/1019
120.0, 2.200, 400.0, 0.800, 0.950
1, 1
1019
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1019 in model.fail_ladviscoplasts
    flvp = model.fail_ladviscoplasts[1019]
    assert pytest.approx(flvp.gamma_vp) == 120.0
    assert pytest.approx(flvp.d_max_vp) == 0.950
    assert flvp.fail_id == 1019


def test_m292_fail_lad_visco_plast_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Viscoplastic Aliases Test
/FAIL/LADEVEZE_VISCO_PLASTIC/1020
130.0, 2.30, 420.0, 0.82, 0.96
1, 1
/FAIL/LAD_VP/1021
130.0, 2.30, 420.0, 0.82, 0.96
1, 1
/FAIL/LAD_VP_MODEL/1022
130.0, 2.30, 420.0, 0.82, 0.96
1, 1
/FAIL/LAD_VP_LAW/1023
130.0, 2.30, 420.0, 0.82, 0.96
1, 1
/FAIL/LADEVEZE_RATE_SENSITIVE_DAMAGE/1024
130.0, 2.30, 420.0, 0.82, 0.96
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1020 in model.fail_ladviscoplasts
    assert 1021 in model.fail_ladviscoplasts
    assert 1022 in model.fail_ladviscoplasts
    assert 1023 in model.fail_ladviscoplasts
    assert 1024 in model.fail_ladviscoplasts
    assert len(model.raw_fails) == 5


def test_m292_eng_dielectric_loss_energy(tmp_path: Path):
    c1 = f"{0.00065:>20.6f}{18:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Dielectric Loss Energy Fixed and Free Format Test
2022 0
/ENG/DIELECTRIC_LOSS_ENERGY/1
Fixed Dielectric Loss Energy Output
{c1}
/ENG/DIELECTRIC_LOSS_ENERGY/2
Free Dielectric Loss Energy Output
0.0013, 36
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_dielectric_loss_energies
    assert 2 in model.eng_dielectric_loss_energies
    de1 = model.eng_dielectric_loss_energies[1]
    assert pytest.approx(de1.dt_dielectric) == 0.00065
    assert de1.sens_id == 18
    de2 = model.eng_dielectric_loss_energies[2]
    assert pytest.approx(de2.dt_dielectric) == 0.0013
    assert de2.sens_id == 36


def test_m292_eng_dielectric_loss_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Dielectric Loss Energy Aliases Test
/ENG/DIELECTRIC_WORK/11
0.001, 10
/ENG/EDIELECTRIC/12
0.002, 11
/ENG/DIELECTRIC_DISSIPATION/13
0.003, 12
/ENG/EM_DIELECTRIC_LOSS/14
0.004, 13
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.eng_dielectric_loss_energies
    assert 12 in model.eng_dielectric_loss_energies
    assert 13 in model.eng_dielectric_loss_energies
    assert 14 in model.eng_dielectric_loss_energies
    assert pytest.approx(model.eng_dielectric_loss_energies[11].dt_dielectric) == 0.001
    assert pytest.approx(model.eng_dielectric_loss_energies[12].dt_dielectric) == 0.002
    assert pytest.approx(model.eng_dielectric_loss_energies[13].dt_dielectric) == 0.003
    assert pytest.approx(model.eng_dielectric_loss_energies[14].dt_dielectric) == 0.004


def test_m292_hoeken_linkage_joint(tmp_path: Path):
    c1 = f"{751:>10d}{752:>10d}{753:>10d}{8.8e6:>20.4f}{1:>10d}{1.5e-6:>20.6e}"
    c2 = f"{25.0:>20.4f}{50.0:>20.4f}{62.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Hoeken Linkage Joint Fixed and Free Format Test
2022 0
/LAGMUL/HOEKEN_LINKAGE_JOINT/1
Fixed Hoeken Linkage Joint
{c1}
{c2}
/LAGMUL/HOEKEN_LINKAGE_JOINT/2
Free Hoeken Linkage Joint
851, 852, 853, 9.6e6, 2, 1.9e-6
30.0, 60.0, 75.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_hoeken_linkage_joints
    assert 2 in model.lagmul_hoeken_linkage_joints
    hk1 = model.lagmul_hoeken_linkage_joints[1]
    assert hk1.node1 == 751
    assert hk1.node2 == 752
    assert hk1.node3 == 753
    assert pytest.approx(hk1.stiff) == 8.8e6
    assert hk1.skew_id == 1
    assert pytest.approx(hk1.tol) == 1.5e-6
    assert pytest.approx(hk1.crank_len) == 25.0
    assert pytest.approx(hk1.rocker_len) == 50.0
    assert pytest.approx(hk1.coupler_len) == 62.5

    hk2 = model.lagmul_hoeken_linkage_joints[2]
    assert hk2.node1 == 851
    assert hk2.node2 == 852
    assert hk2.node3 == 853
    assert pytest.approx(hk2.stiff) == 9.6e6
    assert hk2.skew_id == 2
    assert pytest.approx(hk2.tol) == 1.9e-6
    assert pytest.approx(hk2.crank_len) == 30.0
    assert pytest.approx(hk2.rocker_len) == 60.0
    assert pytest.approx(hk2.coupler_len) == 75.0


def test_m292_hoeken_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Hoeken Linkage Joint Aliases Test
/HOEKEN_LINKAGE_JOINT/31
951, 952, 953, 1.0e6, 0, 1.0e-6
20.0, 40.0, 50.0
/LAGMUL/HOEKEN_LINKAGE/32
954, 955, 956, 1.0e6, 0, 1.0e-6
20.0, 40.0, 50.0
/HOEKEN_LINKAGE/33
957, 958, 959, 1.0e6, 0, 1.0e-6
20.0, 40.0, 50.0
/HOEKEN_STRAIGHT_LINE_MECHANISM/34
960, 961, 962, 1.0e6, 0, 1.0e-6
20.0, 40.0, 50.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 31 in model.lagmul_hoeken_linkage_joints
    assert 32 in model.lagmul_hoeken_linkage_joints
    assert 33 in model.lagmul_hoeken_linkage_joints
    assert 34 in model.lagmul_hoeken_linkage_joints
    assert model.lagmul_hoeken_linkage_joints[31].node1 == 951
    assert model.lagmul_hoeken_linkage_joints[32].node1 == 954
    assert model.lagmul_hoeken_linkage_joints[33].node1 == 957
    assert model.lagmul_hoeken_linkage_joints[34].node1 == 960


def test_m292_sensor_spring_bending_acceleration(tmp_path: Path):
    c1 = f"{881:>10d}{26000.0:>20.4f}{0.0032:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Bending Acceleration Fixed and Free Format Test
2022 0
/SENSOR/SPRING_BENDING_ACCELERATION/1
Fixed Spring Bending Acceleration Sensor
{c1}
/SENSOR/SPRING_BENDING_ACCELERATION/2
Free Spring Bending Acceleration Sensor
882, 31000.0, 0.0052
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_bending_accelerations
    assert 2 in model.sensor_spring_bending_accelerations
    s1 = model.sensor_spring_bending_accelerations[1]
    assert s1.spring_id == 881
    assert pytest.approx(s1.alphab_max) == 26000.0
    assert pytest.approx(s1.t_delay) == 0.0032

    s2 = model.sensor_spring_bending_accelerations[2]
    assert s2.spring_id == 882
    assert pytest.approx(s2.alphab_max) == 31000.0
    assert pytest.approx(s2.t_delay) == 0.0052

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_BENDING_ACCELERATION"
    assert model.sensors[1].kind == "SPRING_BENDING_ACCELERATION"


def test_m292_sensor_spring_bending_acceleration_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Bending Acceleration Aliases Test
/SENSOR/SPRING_BEND_ACC/91
981, 20000.0, 0.001
/SENSOR/SPRING_ACC_BEND/92
982, 21000.0, 0.002
/SENSOR/BENDING_ACCELERATION_SPRING/93
983, 22000.0, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 91 in model.sensor_spring_bending_accelerations
    assert 92 in model.sensor_spring_bending_accelerations
    assert 93 in model.sensor_spring_bending_accelerations
    assert model.sensor_spring_bending_accelerations[91].spring_id == 981
    assert pytest.approx(model.sensor_spring_bending_accelerations[91].alphab_max) == 20000.0
    assert model.sensor_spring_bending_accelerations[92].spring_id == 982
    assert model.sensor_spring_bending_accelerations[93].spring_id == 983
