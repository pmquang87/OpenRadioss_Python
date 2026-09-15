"""Tests for Milestone M303: LadFiberMatrixDebonding Failure Model, EngElectrostrictiveEnergy, SylvesterKempeLinkageJoint, and SensorSpringTorsionalAccelerationRate."""

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


def test_m303_fail_lad_fiber_matrix_debonding_fixed(tmp_path: Path):
    c1 = f"{1.9:>20.4f}{22.5:>20.4f}{145.0:>20.4f}{0.28:>20.4f}{0.985:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1125:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Fiber Matrix Debonding Model Fixed Format Test
2022 0
/FAIL/LAD_FIBER_MATRIX_DEBONDING/1125
Ladeveze Interface Debonding Criterion
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1125 in model.fail_ladfibermatrixdebondings
    flfmd = model.fail_ladfibermatrixdebondings[1125]
    assert pytest.approx(flfmd.y0_fmd) == 1.9
    assert pytest.approx(flfmd.yc_fmd) == 22.5
    assert pytest.approx(flfmd.tau_fmd_crit) == 145.0
    assert pytest.approx(flfmd.mu_fmd_fric) == 0.28
    assert pytest.approx(flfmd.d_fmd_max) == 0.985
    assert flfmd.ifail_sh == 1
    assert flfmd.ifail_so == 2
    assert flfmd.fail_id == 1125
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_FIBER_MATRIX_DEBONDING"


def test_m303_fail_lad_fiber_matrix_debonding_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Fiber Matrix Debonding Free Format Test
/FAIL/LAD_FIBER_MATRIX_DEBONDING/1126
2.6, 28.0, 160.0, 0.32, 0.960
1, 1
1126
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1126 in model.fail_ladfibermatrixdebondings
    flfmd = model.fail_ladfibermatrixdebondings[1126]
    assert pytest.approx(flfmd.y0_fmd) == 2.6
    assert pytest.approx(flfmd.yc_fmd) == 28.0
    assert pytest.approx(flfmd.tau_fmd_crit) == 160.0
    assert pytest.approx(flfmd.mu_fmd_fric) == 0.32
    assert pytest.approx(flfmd.d_fmd_max) == 0.960
    assert flfmd.fail_id == 1126


def test_m303_fail_lad_fiber_matrix_debonding_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Fiber Matrix Debonding Aliases Test
/FAIL/LADEVEZE_FIBER_MATRIX_DEBOND/1127
1.6, 19.5, 130.0, 0.25, 0.97
1, 1
/FAIL/LAD_FMD/1128
1.6, 19.5, 130.0, 0.25, 0.97
1, 1
/FAIL/LAD_FMD_MODEL/1129
1.6, 19.5, 130.0, 0.25, 0.97
1, 1
/FAIL/LAD_FMD_LAW/1130
1.6, 19.5, 130.0, 0.25, 0.97
1, 1
/FAIL/LADEVEZE_INTERFACE_DEBONDING/1131
1.6, 19.5, 130.0, 0.25, 0.97
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1127 in model.fail_ladfibermatrixdebondings
    assert 1128 in model.fail_ladfibermatrixdebondings
    assert 1129 in model.fail_ladfibermatrixdebondings
    assert 1130 in model.fail_ladfibermatrixdebondings
    assert 1131 in model.fail_ladfibermatrixdebondings
    assert len(model.raw_fails) == 5


def test_m303_eng_electrostrictive_energy(tmp_path: Path):
    c1 = f"{0.00015:>20.6f}{35:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Electrostrictive Energy Fixed and Free Format Test
2022 0
/ENG/ELECTROSTRICTIVE_ENERGY/1
Fixed Electrostrictive Energy Output
{c1}
/ENG/ELECTROSTRICTIVE_ENERGY/2
Free Electrostrictive Energy Output
0.00030, 70
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrostrictive_energies
    assert 2 in model.eng_electrostrictive_energies
    ese1 = model.eng_electrostrictive_energies[1]
    assert pytest.approx(ese1.dt_electrostrictive) == 0.00015
    assert ese1.sens_id == 35
    ese2 = model.eng_electrostrictive_energies[2]
    assert pytest.approx(ese2.dt_electrostrictive) == 0.00030
    assert ese2.sens_id == 70


def test_m303_eng_electrostrictive_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Electrostrictive Energy Aliases Test
/ENG/ES_WORK/81
0.001, 80
/ENG/EELECTROSTRICTIVE/82
0.002, 81
/ENG/ELECTROSTRICTION_DISSIPATION/83
0.003, 82
/ENG/EM_ELECTROSTRICTIVE/84
0.004, 83
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 81 in model.eng_electrostrictive_energies
    assert 82 in model.eng_electrostrictive_energies
    assert 83 in model.eng_electrostrictive_energies
    assert 84 in model.eng_electrostrictive_energies
    assert pytest.approx(model.eng_electrostrictive_energies[81].dt_electrostrictive) == 0.001
    assert pytest.approx(model.eng_electrostrictive_energies[82].dt_electrostrictive) == 0.002
    assert pytest.approx(model.eng_electrostrictive_energies[83].dt_electrostrictive) == 0.003
    assert pytest.approx(model.eng_electrostrictive_energies[84].dt_electrostrictive) == 0.004


def test_m303_sylvester_kempe_linkage_joint(tmp_path: Path):
    c1 = f"{901:>10d}{902:>10d}{903:>10d}{8.6e6:>20.4f}{1:>10d}{1.0e-6:>20.6e}"
    c2 = f"{32.0:>20.4f}{48.0:>20.4f}{24.0:>20.4f}{1.50:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sylvester-Kempe Linkage Joint Fixed and Free Format Test
2022 0
/LAGMUL/SYLVESTER_KEMPE_LINKAGE_JOINT/1
Fixed Sylvester-Kempe Linkage Joint
{c1}
{c2}
/LAGMUL/SYLVESTER_KEMPE_LINKAGE_JOINT/2
Free Sylvester-Kempe Linkage Joint
1001, 1002, 1003, 9.3e6, 2, 1.4e-6
36.0, 54.0, 27.0, 1.75
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_sylvester_kempe_linkage_joints
    assert 2 in model.lagmul_sylvester_kempe_linkage_joints
    skj1 = model.lagmul_sylvester_kempe_linkage_joints[1]
    assert skj1.node1 == 901
    assert skj1.node2 == 902
    assert skj1.node3 == 903
    assert pytest.approx(skj1.stiff) == 8.6e6
    assert skj1.skew_id == 1
    assert pytest.approx(skj1.tol) == 1.0e-6
    assert pytest.approx(skj1.arm_len_a) == 32.0
    assert pytest.approx(skj1.arm_len_b) == 48.0
    assert pytest.approx(skj1.base_dist) == 24.0
    assert pytest.approx(skj1.angular_multiplier) == 1.50

    skj2 = model.lagmul_sylvester_kempe_linkage_joints[2]
    assert skj2.node1 == 1001
    assert skj2.node2 == 1002
    assert skj2.node3 == 1003
    assert pytest.approx(skj2.stiff) == 9.3e6
    assert skj2.skew_id == 2
    assert pytest.approx(skj2.tol) == 1.4e-6
    assert pytest.approx(skj2.arm_len_a) == 36.0
    assert pytest.approx(skj2.arm_len_b) == 54.0
    assert pytest.approx(skj2.base_dist) == 27.0
    assert pytest.approx(skj2.angular_multiplier) == 1.75


def test_m303_sylvester_kempe_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sylvester-Kempe Linkage Joint Aliases Test
/SYLVESTER_KEMPE_LINKAGE_JOINT/91
1031, 1032, 1033, 5.8e6, 0, 1.0e-6
28.0, 42.0, 21.0, 1.25
/LAGMUL/SYLVESTER_KEMPE_LINKAGE/92
1034, 1035, 1036, 5.8e6, 0, 1.0e-6
28.0, 42.0, 21.0, 1.25
/SYLVESTER_KEMPE_LINKAGE/93
1037, 1038, 1039, 5.8e6, 0, 1.0e-6
28.0, 42.0, 21.0, 1.25
/SYLVESTER_KEMPE_QUAD_INVERSOR/94
1040, 1041, 1042, 5.8e6, 0, 1.0e-6
28.0, 42.0, 21.0, 1.25
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 91 in model.lagmul_sylvester_kempe_linkage_joints
    assert 92 in model.lagmul_sylvester_kempe_linkage_joints
    assert 93 in model.lagmul_sylvester_kempe_linkage_joints
    assert 94 in model.lagmul_sylvester_kempe_linkage_joints
    assert model.lagmul_sylvester_kempe_linkage_joints[91].node1 == 1031
    assert model.lagmul_sylvester_kempe_linkage_joints[92].node1 == 1034
    assert model.lagmul_sylvester_kempe_linkage_joints[93].node1 == 1037
    assert model.lagmul_sylvester_kempe_linkage_joints[94].node1 == 1040


def test_m303_sensor_spring_torsional_acceleration_rate(tmp_path: Path):
    c1 = f"{988:>10d}{2.9e7:>20.4f}{0.0082:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Acceleration Rate Fixed and Free Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_ACCELERATION_RATE/1
Fixed Spring Torsional Acceleration Rate Sensor
{c1}
/SENSOR/SPRING_TORSIONAL_ACCELERATION_RATE/2
Free Spring Torsional Acceleration Rate Sensor
989, 4.8e7, 0.0108
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_acceleration_rates
    assert 2 in model.sensor_spring_torsional_acceleration_rates
    s1 = model.sensor_spring_torsional_acceleration_rates[1]
    assert s1.spring_id == 988
    assert pytest.approx(s1.jtors_rate_max) == 2.9e7
    assert pytest.approx(s1.t_delay) == 0.0082

    s2 = model.sensor_spring_torsional_acceleration_rates[2]
    assert s2.spring_id == 989
    assert pytest.approx(s2.jtors_rate_max) == 4.8e7
    assert pytest.approx(s2.t_delay) == 0.0108

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_TORSIONAL_ACCELERATION_RATE"
    assert model.sensors[1].kind == "SPRING_TORSIONAL_ACCELERATION_RATE"


def test_m303_sensor_spring_torsional_acceleration_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Torsional Acceleration Rate Aliases Test
/SENSOR/SPRING_TORS_ACC_RATE/101
1018, 2.15e7, 0.0027
/SENSOR/SPRING_RATE_ACC_TORS/102
1019, 2.35e7, 0.0037
/SENSOR/TORSIONAL_ACCELERATION_RATE_SPRING/103
1020, 2.55e7, 0.0047
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 101 in model.sensor_spring_torsional_acceleration_rates
    assert 102 in model.sensor_spring_torsional_acceleration_rates
    assert 103 in model.sensor_spring_torsional_acceleration_rates
    assert model.sensor_spring_torsional_acceleration_rates[101].spring_id == 1018
    assert pytest.approx(model.sensor_spring_torsional_acceleration_rates[101].jtors_rate_max) == 2.15e7
    assert model.sensor_spring_torsional_acceleration_rates[102].spring_id == 1019
    assert model.sensor_spring_torsional_acceleration_rates[103].spring_id == 1020
