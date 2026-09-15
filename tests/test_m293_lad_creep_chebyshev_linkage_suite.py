"""Tests for Milestone M293: LadCreep Failure Model, EngMagneticHysteresisEnergy, ChebyshevLinkageJoint, and SensorSpringTotalAngularAcceleration."""

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


def test_m293_fail_lad_creep_fixed(tmp_path: Path):
    c1 = f"{2.5e-5:>20.6e}{4.200:>20.4f}{145000.0:>20.2f}{373.15:>20.2f}{0.975:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1028:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Creep Failure Model Fixed Format Test
2022 0
/FAIL/LAD_CREEP/1028
Ladeveze Tertiary Creep Damage Evolution
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1028 in model.fail_ladcreeps
    flc = model.fail_ladcreeps[1028]
    assert pytest.approx(flc.a_creep) == 2.5e-5
    assert pytest.approx(flc.n_creep) == 4.200
    assert pytest.approx(flc.q_creep) == 145000.0
    assert pytest.approx(flc.t_creep_ref) == 373.15
    assert pytest.approx(flc.d_creep_max) == 0.975
    assert flc.ifail_sh == 1
    assert flc.ifail_so == 2
    assert flc.fail_id == 1028
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_CREEP"


def test_m293_fail_lad_creep_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Creep Free Format Test
/FAIL/LAD_CREEP/1029
1.8e-5, 3.800, 135000.0, 350.0, 0.940
1, 1
1029
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1029 in model.fail_ladcreeps
    flc = model.fail_ladcreeps[1029]
    assert pytest.approx(flc.a_creep) == 1.8e-5
    assert pytest.approx(flc.d_creep_max) == 0.940
    assert flc.fail_id == 1029


def test_m293_fail_lad_creep_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Creep Aliases Test
/FAIL/LADEVEZE_CREEP/1030
2.0e-5, 4.0, 140000.0, 360.0, 0.95
1, 1
/FAIL/LAD_CREEP_DAMAGE/1031
2.0e-5, 4.0, 140000.0, 360.0, 0.95
1, 1
/FAIL/LAD_CREEP_MODEL/1032
2.0e-5, 4.0, 140000.0, 360.0, 0.95
1, 1
/FAIL/LAD_CREEP_LAW/1033
2.0e-5, 4.0, 140000.0, 360.0, 0.95
1, 1
/FAIL/LADEVEZE_TERTIARY_CREEP/1034
2.0e-5, 4.0, 140000.0, 360.0, 0.95
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1030 in model.fail_ladcreeps
    assert 1031 in model.fail_ladcreeps
    assert 1032 in model.fail_ladcreeps
    assert 1033 in model.fail_ladcreeps
    assert 1034 in model.fail_ladcreeps
    assert len(model.raw_fails) == 5


def test_m293_eng_magnetic_hysteresis_energy(tmp_path: Path):
    c1 = f"{0.00055:>20.6f}{22:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Magnetic Hysteresis Energy Fixed and Free Format Test
2022 0
/ENG/MAGNETIC_HYSTERESIS_ENERGY/1
Fixed Magnetic Hysteresis Energy Output
{c1}
/ENG/MAGNETIC_HYSTERESIS_ENERGY/2
Free Magnetic Hysteresis Energy Output
0.0011, 44
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_magnetic_hysteresis_energies
    assert 2 in model.eng_magnetic_hysteresis_energies
    he1 = model.eng_magnetic_hysteresis_energies[1]
    assert pytest.approx(he1.dt_hysteresis) == 0.00055
    assert he1.sens_id == 22
    he2 = model.eng_magnetic_hysteresis_energies[2]
    assert pytest.approx(he2.dt_hysteresis) == 0.0011
    assert he2.sens_id == 44


def test_m293_eng_magnetic_hysteresis_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Magnetic Hysteresis Energy Aliases Test
/ENG/MAG_HYST_WORK/11
0.001, 10
/ENG/EMAGHYST/12
0.002, 11
/ENG/HYSTERESIS_LOSS_ENERGY/13
0.003, 12
/ENG/EM_HYSTERESIS_LOSS/14
0.004, 13
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.eng_magnetic_hysteresis_energies
    assert 12 in model.eng_magnetic_hysteresis_energies
    assert 13 in model.eng_magnetic_hysteresis_energies
    assert 14 in model.eng_magnetic_hysteresis_energies
    assert pytest.approx(model.eng_magnetic_hysteresis_energies[11].dt_hysteresis) == 0.001
    assert pytest.approx(model.eng_magnetic_hysteresis_energies[12].dt_hysteresis) == 0.002
    assert pytest.approx(model.eng_magnetic_hysteresis_energies[13].dt_hysteresis) == 0.003
    assert pytest.approx(model.eng_magnetic_hysteresis_energies[14].dt_hysteresis) == 0.004


def test_m293_chebyshev_linkage_joint(tmp_path: Path):
    c1 = f"{761:>10d}{762:>10d}{763:>10d}{7.5e6:>20.4f}{1:>10d}{1.6e-6:>20.6e}"
    c2 = f"{80.0:>20.4f}{100.0:>20.4f}{50.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Chebyshev Linkage Joint Fixed and Free Format Test
2022 0
/LAGMUL/CHEBYSHEV_LINKAGE_JOINT/1
Fixed Chebyshev Linkage Joint
{c1}
{c2}
/LAGMUL/CHEBYSHEV_LINKAGE_JOINT/2
Free Chebyshev Linkage Joint
861, 862, 863, 8.4e6, 2, 2.1e-6
90.0, 112.5, 56.25
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_chebyshev_linkage_joints
    assert 2 in model.lagmul_chebyshev_linkage_joints
    cb1 = model.lagmul_chebyshev_linkage_joints[1]
    assert cb1.node1 == 761
    assert cb1.node2 == 762
    assert cb1.node3 == 763
    assert pytest.approx(cb1.stiff) == 7.5e6
    assert cb1.skew_id == 1
    assert pytest.approx(cb1.tol) == 1.6e-6
    assert pytest.approx(cb1.base_len) == 80.0
    assert pytest.approx(cb1.crank_len) == 100.0
    assert pytest.approx(cb1.coupler_len) == 50.0

    cb2 = model.lagmul_chebyshev_linkage_joints[2]
    assert cb2.node1 == 861
    assert cb2.node2 == 862
    assert cb2.node3 == 863
    assert pytest.approx(cb2.stiff) == 8.4e6
    assert cb2.skew_id == 2
    assert pytest.approx(cb2.tol) == 2.1e-6
    assert pytest.approx(cb2.base_len) == 90.0
    assert pytest.approx(cb2.crank_len) == 112.5
    assert pytest.approx(cb2.coupler_len) == 56.25


def test_m293_chebyshev_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Chebyshev Linkage Joint Aliases Test
/CHEBYSHEV_LINKAGE_JOINT/21
961, 962, 963, 1.0e6, 0, 1.0e-6
60.0, 75.0, 37.5
/LAGMUL/CHEBYSHEV_LINKAGE/22
964, 965, 966, 1.0e6, 0, 1.0e-6
60.0, 75.0, 37.5
/CHEBYSHEV_LINKAGE/23
967, 968, 969, 1.0e6, 0, 1.0e-6
60.0, 75.0, 37.5
/CHEBYSHEV_STRAIGHT_LINE_MECHANISM/24
970, 971, 972, 1.0e6, 0, 1.0e-6
60.0, 75.0, 37.5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 21 in model.lagmul_chebyshev_linkage_joints
    assert 22 in model.lagmul_chebyshev_linkage_joints
    assert 23 in model.lagmul_chebyshev_linkage_joints
    assert 24 in model.lagmul_chebyshev_linkage_joints
    assert model.lagmul_chebyshev_linkage_joints[21].node1 == 961
    assert model.lagmul_chebyshev_linkage_joints[22].node1 == 964
    assert model.lagmul_chebyshev_linkage_joints[23].node1 == 967
    assert model.lagmul_chebyshev_linkage_joints[24].node1 == 970


def test_m293_sensor_spring_total_angular_acceleration(tmp_path: Path):
    c1 = f"{891:>10d}{36000.0:>20.4f}{0.0038:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Angular Acceleration Fixed and Free Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_ACCELERATION/1
Fixed Spring Total Angular Acceleration Sensor
{c1}
/SENSOR/SPRING_TOTAL_ANGULAR_ACCELERATION/2
Free Spring Total Angular Acceleration Sensor
892, 42000.0, 0.0058
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_angular_accelerations
    assert 2 in model.sensor_spring_total_angular_accelerations
    s1 = model.sensor_spring_total_angular_accelerations[1]
    assert s1.spring_id == 891
    assert pytest.approx(s1.alpha_tot_max) == 36000.0
    assert pytest.approx(s1.t_delay) == 0.0038

    s2 = model.sensor_spring_total_angular_accelerations[2]
    assert s2.spring_id == 892
    assert pytest.approx(s2.alpha_tot_max) == 42000.0
    assert pytest.approx(s2.t_delay) == 0.0058

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_ACCELERATION"
    assert model.sensors[1].kind == "SPRING_TOTAL_ANGULAR_ACCELERATION"


def test_m293_sensor_spring_total_angular_acceleration_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Angular Acceleration Aliases Test
/SENSOR/SPRING_TOT_ANG_ACC/71
991, 28000.0, 0.001
/SENSOR/SPRING_ACC_ANG_TOT/72
992, 29000.0, 0.002
/SENSOR/TOTAL_ANGULAR_ACCELERATION_SPRING/73
993, 30000.0, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 71 in model.sensor_spring_total_angular_accelerations
    assert 72 in model.sensor_spring_total_angular_accelerations
    assert 73 in model.sensor_spring_total_angular_accelerations
    assert model.sensor_spring_total_angular_accelerations[71].spring_id == 991
    assert pytest.approx(model.sensor_spring_total_angular_accelerations[71].alpha_tot_max) == 28000.0
    assert model.sensor_spring_total_angular_accelerations[72].spring_id == 992
    assert model.sensor_spring_total_angular_accelerations[73].spring_id == 993
