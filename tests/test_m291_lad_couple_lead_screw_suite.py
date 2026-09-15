"""Tests for Milestone M291: LadCouple Failure Model, EngPlasmonicEnergy, LeadScrewJoint, and SensorSpringTorsionalAcceleration."""

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


def test_m291_fail_lad_couple_fixed(tmp_path: Path):
    c1 = f"{300.0:>20.4f}{0.0025:>20.4f}{0.0450:>20.4f}{0.9500:>20.4f}{1.1500:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1008:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Failure Model Fixed Format Test
2022 0
/FAIL/LAD_COUPLE/1008
Ladeveze Thermo-Mechanical Damage Evolution
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1008 in model.fail_ladcouples
    flc = model.fail_ladcouples[1008]
    assert pytest.approx(flc.t_ref) == 300.0
    assert pytest.approx(flc.beta_th) == 0.0025
    assert pytest.approx(flc.c_th) == 0.0450
    assert pytest.approx(flc.d_th_max) == 0.9500
    assert pytest.approx(flc.gamma_th) == 1.1500
    assert flc.ifail_sh == 1
    assert flc.ifail_so == 2
    assert flc.fail_id == 1008
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE"


def test_m291_fail_lad_couple_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Free Format Test
/FAIL/LAD_COUPLE/1009
295.0, 0.0020, 0.040, 0.920, 1.100
1, 1
1009
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1009 in model.fail_ladcouples
    flc = model.fail_ladcouples[1009]
    assert pytest.approx(flc.t_ref) == 295.0
    assert pytest.approx(flc.gamma_th) == 1.100
    assert flc.fail_id == 1009


def test_m291_fail_lad_couple_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Aliases Test
/FAIL/LADEVEZE_COUPLED/1010
298.0, 0.0022, 0.042, 0.93, 1.12
1, 1
/FAIL/LAD_COUPLED/1011
298.0, 0.0022, 0.042, 0.93, 1.12
1, 1
/FAIL/LAD_COUPLE_MODEL/1012
298.0, 0.0022, 0.042, 0.93, 1.12
1, 1
/FAIL/LAD_COUPLE_LAW/1013
298.0, 0.0022, 0.042, 0.93, 1.12
1, 1
/FAIL/LADEVEZE_THERMO_COUPLED/1014
298.0, 0.0022, 0.042, 0.93, 1.12
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1010 in model.fail_ladcouples
    assert 1011 in model.fail_ladcouples
    assert 1012 in model.fail_ladcouples
    assert 1013 in model.fail_ladcouples
    assert 1014 in model.fail_ladcouples
    assert len(model.raw_fails) == 5


def test_m291_eng_plasmonic_energy(tmp_path: Path):
    c1 = f"{0.00075:>20.6f}{16:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Plasmonic Energy Fixed and Free Format Test
2022 0
/ENG/PLASMONIC_ENERGY/1
Fixed Plasmonic Energy Output
{c1}
/ENG/PLASMONIC_ENERGY/2
Free Plasmonic Energy Output
0.0015, 32
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_plasmonic_energies
    assert 2 in model.eng_plasmonic_energies
    pe1 = model.eng_plasmonic_energies[1]
    assert pytest.approx(pe1.dt_plasmon) == 0.00075
    assert pe1.sens_id == 16
    pe2 = model.eng_plasmonic_energies[2]
    assert pytest.approx(pe2.dt_plasmon) == 0.0015
    assert pe2.sens_id == 32


def test_m291_eng_plasmonic_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Plasmonic Energy Aliases Test
/ENG/PLASMONIC_WORK/11
0.001, 10
/ENG/EPLASMON/12
0.002, 11
/ENG/SURFACE_PLASMON_ENERGY/13
0.003, 12
/ENG/EM_PLASMON_ENERGY/14
0.004, 13
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.eng_plasmonic_energies
    assert 12 in model.eng_plasmonic_energies
    assert 13 in model.eng_plasmonic_energies
    assert 14 in model.eng_plasmonic_energies
    assert pytest.approx(model.eng_plasmonic_energies[11].dt_plasmon) == 0.001
    assert pytest.approx(model.eng_plasmonic_energies[12].dt_plasmon) == 0.002
    assert pytest.approx(model.eng_plasmonic_energies[13].dt_plasmon) == 0.003
    assert pytest.approx(model.eng_plasmonic_energies[14].dt_plasmon) == 0.004


def test_m291_lead_screw_joint(tmp_path: Path):
    c1 = f"{701:>10d}{702:>10d}{703:>10d}{9.5e6:>20.4f}{1:>10d}{1.4e-6:>20.6e}"
    c2 = f"{5.0:>20.4f}{30.0:>20.4f}{0.92:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Lead Screw Joint Fixed and Free Format Test
2022 0
/LAGMUL/LEAD_SCREW_JOINT/1
Fixed Lead Screw Joint
{c1}
{c2}
/LAGMUL/LEAD_SCREW_JOINT/2
Free Lead Screw Joint
801, 802, 803, 1.05e7, 2, 1.8e-6
10.0, 45.0, 0.95
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_lead_screw_joints
    assert 2 in model.lagmul_lead_screw_joints
    ls1 = model.lagmul_lead_screw_joints[1]
    assert ls1.node1 == 701
    assert ls1.node2 == 702
    assert ls1.node3 == 703
    assert pytest.approx(ls1.stiff) == 9.5e6
    assert ls1.skew_id == 1
    assert pytest.approx(ls1.tol) == 1.4e-6
    assert pytest.approx(ls1.pitch_lead) == 5.0
    assert pytest.approx(ls1.thread_angle) == 30.0
    assert pytest.approx(ls1.helix_efficiency) == 0.92

    ls2 = model.lagmul_lead_screw_joints[2]
    assert ls2.node1 == 801
    assert ls2.node2 == 802
    assert ls2.node3 == 803
    assert pytest.approx(ls2.stiff) == 1.05e7
    assert ls2.skew_id == 2
    assert pytest.approx(ls2.tol) == 1.8e-6
    assert pytest.approx(ls2.pitch_lead) == 10.0
    assert pytest.approx(ls2.thread_angle) == 45.0
    assert pytest.approx(ls2.helix_efficiency) == 0.95


def test_m291_lead_screw_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Lead Screw Joint Aliases Test
/LEAD_SCREW_JOINT/41
981, 982, 983, 1.0e6, 0, 1.0e-6
6.0, 30.0, 0.90
/LAGMUL/LEAD_SCREW/42
984, 985, 986, 1.0e6, 0, 1.0e-6
6.0, 30.0, 0.90
/LEAD_SCREW/43
987, 988, 989, 1.0e6, 0, 1.0e-6
6.0, 30.0, 0.90
/BALL_SCREW_MECHANISM/44
990, 991, 992, 1.0e6, 0, 1.0e-6
6.0, 30.0, 0.90
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 41 in model.lagmul_lead_screw_joints
    assert 42 in model.lagmul_lead_screw_joints
    assert 43 in model.lagmul_lead_screw_joints
    assert 44 in model.lagmul_lead_screw_joints
    assert model.lagmul_lead_screw_joints[41].node1 == 981
    assert model.lagmul_lead_screw_joints[42].node1 == 984
    assert model.lagmul_lead_screw_joints[43].node1 == 987
    assert model.lagmul_lead_screw_joints[44].node1 == 990


def test_m291_sensor_spring_torsional_acceleration(tmp_path: Path):
    c1 = f"{871:>10d}{18000.0:>20.4f}{0.0025:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Acceleration Fixed and Free Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_ACCELERATION/1
Fixed Spring Torsional Acceleration Sensor
{c1}
/SENSOR/SPRING_TORSIONAL_ACCELERATION/2
Free Spring Torsional Acceleration Sensor
872, 22000.0, 0.0045
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_accelerations
    assert 2 in model.sensor_spring_torsional_accelerations
    s1 = model.sensor_spring_torsional_accelerations[1]
    assert s1.spring_id == 871
    assert pytest.approx(s1.alphat_max) == 18000.0
    assert pytest.approx(s1.t_delay) == 0.0025

    s2 = model.sensor_spring_torsional_accelerations[2]
    assert s2.spring_id == 872
    assert pytest.approx(s2.alphat_max) == 22000.0
    assert pytest.approx(s2.t_delay) == 0.0045

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_TORSIONAL_ACCELERATION"
    assert model.sensors[1].kind == "SPRING_TORSIONAL_ACCELERATION"


def test_m291_sensor_spring_torsional_acceleration_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Torsional Acceleration Aliases Test
/SENSOR/SPRING_TORS_ACC/81
971, 15000.0, 0.001
/SENSOR/SPRING_ACC_TORS/82
972, 16000.0, 0.002
/SENSOR/TORSIONAL_ACCELERATION_SPRING/83
973, 17000.0, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 81 in model.sensor_spring_torsional_accelerations
    assert 82 in model.sensor_spring_torsional_accelerations
    assert 83 in model.sensor_spring_torsional_accelerations
    assert model.sensor_spring_torsional_accelerations[81].spring_id == 971
    assert pytest.approx(model.sensor_spring_torsional_accelerations[81].alphat_max) == 15000.0
    assert model.sensor_spring_torsional_accelerations[82].spring_id == 972
    assert model.sensor_spring_torsional_accelerations[83].spring_id == 973
