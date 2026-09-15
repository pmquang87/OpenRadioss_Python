"""Tests for Milestone M302: LadFracture Failure Model, EngThermophotonicEnergy, KempeLinkageJoint, and SensorSpringTransverseAccelerationRate."""

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


def test_m302_fail_lad_fracture_fixed(tmp_path: Path):
    c1 = f"{1.7:>20.4f}{20.5:>20.4f}{1.6:>20.4f}{0.0045:>20.6f}{0.982:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1115:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Fracture Model Fixed Format Test
2022 0
/FAIL/LAD_FRACTURE/1115
Ladeveze Dynamic Cohesive Fracture Criterion
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1115 in model.fail_ladfractures
    flf = model.fail_ladfractures[1115]
    assert pytest.approx(flf.y0_frac) == 1.7
    assert pytest.approx(flf.yc_frac) == 20.5
    assert pytest.approx(flf.gamma_cohes) == 1.6
    assert pytest.approx(flf.l_char) == 0.0045
    assert pytest.approx(flf.d_frac_max) == 0.982
    assert flf.ifail_sh == 1
    assert flf.ifail_so == 2
    assert flf.fail_id == 1115
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_FRACTURE"


def test_m302_fail_lad_fracture_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Fracture Free Format Test
/FAIL/LAD_FRACTURE/1116
2.4, 25.0, 1.8, 0.0060, 0.965
1, 1
1116
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1116 in model.fail_ladfractures
    flf = model.fail_ladfractures[1116]
    assert pytest.approx(flf.y0_frac) == 2.4
    assert pytest.approx(flf.yc_frac) == 25.0
    assert pytest.approx(flf.gamma_cohes) == 1.8
    assert pytest.approx(flf.l_char) == 0.0060
    assert pytest.approx(flf.d_frac_max) == 0.965
    assert flf.fail_id == 1116


def test_m302_fail_lad_fracture_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Fracture Aliases Test
/FAIL/LADEVEZE_DYNAMIC_FRACTURE/1117
1.4, 17.5, 1.5, 0.0035, 0.97
1, 1
/FAIL/LAD_FRAC/1118
1.4, 17.5, 1.5, 0.0035, 0.97
1, 1
/FAIL/LAD_FRACTURE_MODEL/1119
1.4, 17.5, 1.5, 0.0035, 0.97
1, 1
/FAIL/LAD_FRACTURE_LAW/1120
1.4, 17.5, 1.5, 0.0035, 0.97
1, 1
/FAIL/LADEVEZE_COHESIVE_FRACTURE/1121
1.4, 17.5, 1.5, 0.0035, 0.97
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1117 in model.fail_ladfractures
    assert 1118 in model.fail_ladfractures
    assert 1119 in model.fail_ladfractures
    assert 1120 in model.fail_ladfractures
    assert 1121 in model.fail_ladfractures
    assert len(model.raw_fails) == 5


def test_m302_eng_thermophotonic_energy(tmp_path: Path):
    c1 = f"{0.00018:>20.6f}{38:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Thermophotonic Energy Fixed and Free Format Test
2022 0
/ENG/THERMOPHOTONIC_ENERGY/1
Fixed Thermophotonic Energy Output
{c1}
/ENG/THERMOPHOTONIC_ENERGY/2
Free Thermophotonic Energy Output
0.00036, 76
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_thermophotonic_energies
    assert 2 in model.eng_thermophotonic_energies
    tpe1 = model.eng_thermophotonic_energies[1]
    assert pytest.approx(tpe1.dt_thermophotonic) == 0.00018
    assert tpe1.sens_id == 38
    tpe2 = model.eng_thermophotonic_energies[2]
    assert pytest.approx(tpe2.dt_thermophotonic) == 0.00036
    assert tpe2.sens_id == 76


def test_m302_eng_thermophotonic_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Thermophotonic Energy Aliases Test
/ENG/TP_WORK/71
0.001, 70
/ENG/ETHERMOPHOTONIC/72
0.002, 71
/ENG/THERMOPHOTONIC_DISSIPATION/73
0.003, 72
/ENG/EM_THERMOPHOTONIC/74
0.004, 73
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 71 in model.eng_thermophotonic_energies
    assert 72 in model.eng_thermophotonic_energies
    assert 73 in model.eng_thermophotonic_energies
    assert 74 in model.eng_thermophotonic_energies
    assert pytest.approx(model.eng_thermophotonic_energies[71].dt_thermophotonic) == 0.001
    assert pytest.approx(model.eng_thermophotonic_energies[72].dt_thermophotonic) == 0.002
    assert pytest.approx(model.eng_thermophotonic_energies[73].dt_thermophotonic) == 0.003
    assert pytest.approx(model.eng_thermophotonic_energies[74].dt_thermophotonic) == 0.004


def test_m302_kempe_linkage_joint(tmp_path: Path):
    c1 = f"{891:>10d}{892:>10d}{893:>10d}{8.4e6:>20.4f}{1:>10d}{1.0e-6:>20.6e}"
    c2 = f"{25.0:>20.4f}{62.5:>20.4f}{62.5:>20.4f}{105.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Kempe Linkage Joint Fixed and Free Format Test
2022 0
/LAGMUL/KEMPE_LINKAGE_JOINT/1
Fixed Kempe Linkage Joint
{c1}
{c2}
/LAGMUL/KEMPE_LINKAGE_JOINT/2
Free Kempe Linkage Joint
991, 992, 993, 9.1e6, 2, 1.3e-6
30.0, 75.0, 75.0, 126.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_kempe_linkage_joints
    assert 2 in model.lagmul_kempe_linkage_joints
    kj1 = model.lagmul_kempe_linkage_joints[1]
    assert kj1.node1 == 891
    assert kj1.node2 == 892
    assert kj1.node3 == 893
    assert pytest.approx(kj1.stiff) == 8.4e6
    assert kj1.skew_id == 1
    assert pytest.approx(kj1.tol) == 1.0e-6
    assert pytest.approx(kj1.arm_len1) == 25.0
    assert pytest.approx(kj1.arm_len2) == 62.5
    assert pytest.approx(kj1.cross_len) == 62.5
    assert pytest.approx(kj1.travel_span) == 105.0

    kj2 = model.lagmul_kempe_linkage_joints[2]
    assert kj2.node1 == 991
    assert kj2.node2 == 992
    assert kj2.node3 == 993
    assert pytest.approx(kj2.stiff) == 9.1e6
    assert kj2.skew_id == 2
    assert pytest.approx(kj2.tol) == 1.3e-6
    assert pytest.approx(kj2.arm_len1) == 30.0
    assert pytest.approx(kj2.arm_len2) == 75.0
    assert pytest.approx(kj2.cross_len) == 75.0
    assert pytest.approx(kj2.travel_span) == 126.0


def test_m302_kempe_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Kempe Linkage Joint Aliases Test
/KEMPE_LINKAGE_JOINT/81
1011, 1012, 1013, 5.5e6, 0, 1.0e-6
20.0, 50.0, 50.0, 84.0
/LAGMUL/KEMPE_LINKAGE/82
1014, 1015, 1016, 5.5e6, 0, 1.0e-6
20.0, 50.0, 50.0, 84.0
/KEMPE_LINKAGE/83
1017, 1018, 1019, 5.5e6, 0, 1.0e-6
20.0, 50.0, 50.0, 84.0
/KEMPE_UNIVERSAL_MECHANISM/84
1020, 1021, 1022, 5.5e6, 0, 1.0e-6
20.0, 50.0, 50.0, 84.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 81 in model.lagmul_kempe_linkage_joints
    assert 82 in model.lagmul_kempe_linkage_joints
    assert 83 in model.lagmul_kempe_linkage_joints
    assert 84 in model.lagmul_kempe_linkage_joints
    assert model.lagmul_kempe_linkage_joints[81].node1 == 1011
    assert model.lagmul_kempe_linkage_joints[82].node1 == 1014
    assert model.lagmul_kempe_linkage_joints[83].node1 == 1017
    assert model.lagmul_kempe_linkage_joints[84].node1 == 1020


def test_m302_sensor_spring_transverse_acceleration_rate(tmp_path: Path):
    c1 = f"{978:>10d}{2.7e7:>20.4f}{0.0078:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Transverse Acceleration Rate Fixed and Free Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_ACCELERATION_RATE/1
Fixed Spring Transverse Acceleration Rate Sensor
{c1}
/SENSOR/SPRING_TRANSVERSE_ACCELERATION_RATE/2
Free Spring Transverse Acceleration Rate Sensor
979, 4.5e7, 0.0098
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_transverse_acceleration_rates
    assert 2 in model.sensor_spring_transverse_acceleration_rates
    s1 = model.sensor_spring_transverse_acceleration_rates[1]
    assert s1.spring_id == 978
    assert pytest.approx(s1.jtrans_rate_max) == 2.7e7
    assert pytest.approx(s1.t_delay) == 0.0078

    s2 = model.sensor_spring_transverse_acceleration_rates[2]
    assert s2.spring_id == 979
    assert pytest.approx(s2.jtrans_rate_max) == 4.5e7
    assert pytest.approx(s2.t_delay) == 0.0098

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_ACCELERATION_RATE"
    assert model.sensors[1].kind == "SPRING_TRANSVERSE_ACCELERATION_RATE"


def test_m302_sensor_spring_transverse_acceleration_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Transverse Acceleration Rate Aliases Test
/SENSOR/SPRING_TRANS_ACC_RATE/98
1008, 2.05e7, 0.0026
/SENSOR/SPRING_RATE_ACC_TRANS/99
1009, 2.25e7, 0.0036
/SENSOR/TRANSVERSE_ACCELERATION_RATE_SPRING/100
1010, 2.45e7, 0.0046
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 98 in model.sensor_spring_transverse_acceleration_rates
    assert 99 in model.sensor_spring_transverse_acceleration_rates
    assert 100 in model.sensor_spring_transverse_acceleration_rates
    assert model.sensor_spring_transverse_acceleration_rates[98].spring_id == 1008
    assert pytest.approx(model.sensor_spring_transverse_acceleration_rates[98].jtrans_rate_max) == 2.05e7
    assert model.sensor_spring_transverse_acceleration_rates[99].spring_id == 1009
    assert model.sensor_spring_transverse_acceleration_rates[100].spring_id == 1010
