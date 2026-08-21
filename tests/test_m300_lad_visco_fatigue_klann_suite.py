"""Tests for Milestone M300: LadViscoFatigue Failure Model, EngThermogalvanicEnergy, KlannLinkageJoint, and SensorSpringTotalAccelerationRate."""

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


def test_m300_fail_lad_visco_fatigue_fixed(tmp_path: Path):
    c1 = f"{1.5:>20.4f}{19.0:>20.4f}{2.4:>20.4f}{0.0035:>20.6f}{0.980:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1095:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Visco-Fatigue Damage Model Fixed Format Test
2022 0
/FAIL/LAD_VISCO_FATIGUE/1095
Ladeveze Visco-Fatigue Damage Criterion
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1095 in model.fail_ladviscofatigues
    flvf = model.fail_ladviscofatigues[1095]
    assert pytest.approx(flvf.y0_vf) == 1.5
    assert pytest.approx(flvf.yc_vf) == 19.0
    assert pytest.approx(flvf.beta_vf) == 2.4
    assert pytest.approx(flvf.tau_relax) == 0.0035
    assert pytest.approx(flvf.d_vf_max) == 0.980
    assert flvf.ifail_sh == 1
    assert flvf.ifail_so == 2
    assert flvf.fail_id == 1095
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_VISCO_FATIGUE"


def test_m300_fail_lad_visco_fatigue_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Visco-Fatigue Damage Free Format Test
/FAIL/LAD_VISCO_FATIGUE/1096
2.2, 24.0, 2.8, 0.0050, 0.955
1, 1
1096
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1096 in model.fail_ladviscofatigues
    flvf = model.fail_ladviscofatigues[1096]
    assert pytest.approx(flvf.y0_vf) == 2.2
    assert pytest.approx(flvf.yc_vf) == 24.0
    assert pytest.approx(flvf.beta_vf) == 2.8
    assert pytest.approx(flvf.tau_relax) == 0.0050
    assert pytest.approx(flvf.d_vf_max) == 0.955
    assert flvf.fail_id == 1096


def test_m300_fail_lad_visco_fatigue_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Visco-Fatigue Damage Aliases Test
/FAIL/LADEVEZE_VISCO_FATIGUE/1097
1.0, 15.0, 2.0, 0.002, 0.96
1, 1
/FAIL/LAD_VF/1098
1.0, 15.0, 2.0, 0.002, 0.96
1, 1
/FAIL/LAD_VISCO_FATIGUE_MODEL/1099
1.0, 15.0, 2.0, 0.002, 0.96
1, 1
/FAIL/LAD_VISCO_FATIGUE_LAW/1100
1.0, 15.0, 2.0, 0.002, 0.96
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_FATIGUE/1101
1.0, 15.0, 2.0, 0.002, 0.96
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1097 in model.fail_ladviscofatigues
    assert 1098 in model.fail_ladviscofatigues
    assert 1099 in model.fail_ladviscofatigues
    assert 1100 in model.fail_ladviscofatigues
    assert 1101 in model.fail_ladviscofatigues
    assert len(model.raw_fails) == 5


def test_m300_eng_thermogalvanic_energy(tmp_path: Path):
    c1 = f"{0.00015:>20.6f}{36:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Thermogalvanic Energy Fixed and Free Format Test
2022 0
/ENG/THERMOGALVANIC_ENERGY/1
Fixed Thermogalvanic Energy Output
{c1}
/ENG/THERMOGALVANIC_ENERGY/2
Free Thermogalvanic Energy Output
0.0003, 72
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_thermogalvanic_energies
    assert 2 in model.eng_thermogalvanic_energies
    tge1 = model.eng_thermogalvanic_energies[1]
    assert pytest.approx(tge1.dt_thermogalvanic) == 0.00015
    assert tge1.sens_id == 36
    tge2 = model.eng_thermogalvanic_energies[2]
    assert pytest.approx(tge2.dt_thermogalvanic) == 0.0003
    assert tge2.sens_id == 72


def test_m300_eng_thermogalvanic_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Thermogalvanic Energy Aliases Test
/ENG/TG_WORK/51
0.001, 50
/ENG/ETHERMOGALVANIC/52
0.002, 51
/ENG/THERMOGALVANIC_DISSIPATION/53
0.003, 52
/ENG/EM_THERMOGALVANIC/54
0.004, 53
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 51 in model.eng_thermogalvanic_energies
    assert 52 in model.eng_thermogalvanic_energies
    assert 53 in model.eng_thermogalvanic_energies
    assert 54 in model.eng_thermogalvanic_energies
    assert pytest.approx(model.eng_thermogalvanic_energies[51].dt_thermogalvanic) == 0.001
    assert pytest.approx(model.eng_thermogalvanic_energies[52].dt_thermogalvanic) == 0.002
    assert pytest.approx(model.eng_thermogalvanic_energies[53].dt_thermogalvanic) == 0.003
    assert pytest.approx(model.eng_thermogalvanic_energies[54].dt_thermogalvanic) == 0.004


def test_m300_klann_linkage_joint(tmp_path: Path):
    c1 = f"{871:>10d}{872:>10d}{873:>10d}{8.5e6:>20.4f}{1:>10d}{1.0e-6:>20.6e}"
    c2 = f"{35.0:>20.4f}{80.0:>20.4f}{95.0:>20.4f}{120.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Klann Linkage Joint Fixed and Free Format Test
2022 0
/LAGMUL/KLANN_LINKAGE_JOINT/1
Fixed Klann Linkage Joint
{c1}
{c2}
/LAGMUL/KLANN_LINKAGE_JOINT/2
Free Klann Linkage Joint
971, 972, 973, 9.2e6, 2, 1.4e-6
40.0, 90.0, 110.0, 140.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_klann_linkage_joints
    assert 2 in model.lagmul_klann_linkage_joints
    kj1 = model.lagmul_klann_linkage_joints[1]
    assert kj1.node1 == 871
    assert kj1.node2 == 872
    assert kj1.node3 == 873
    assert pytest.approx(kj1.stiff) == 8.5e6
    assert kj1.skew_id == 1
    assert pytest.approx(kj1.tol) == 1.0e-6
    assert pytest.approx(kj1.crank_len) == 35.0
    assert pytest.approx(kj1.rocker_len) == 80.0
    assert pytest.approx(kj1.leg_upper_len) == 95.0
    assert pytest.approx(kj1.leg_lower_len) == 120.0

    kj2 = model.lagmul_klann_linkage_joints[2]
    assert kj2.node1 == 971
    assert kj2.node2 == 972
    assert kj2.node3 == 973
    assert pytest.approx(kj2.stiff) == 9.2e6
    assert kj2.skew_id == 2
    assert pytest.approx(kj2.tol) == 1.4e-6
    assert pytest.approx(kj2.crank_len) == 40.0
    assert pytest.approx(kj2.rocker_len) == 90.0
    assert pytest.approx(kj2.leg_upper_len) == 110.0
    assert pytest.approx(kj2.leg_lower_len) == 140.0


def test_m300_klann_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Klann Linkage Joint Aliases Test
/KLANN_LINKAGE_JOINT/61
981, 982, 983, 5.0e6, 0, 1.0e-6
30.0, 70.0, 85.0, 100.0
/LAGMUL/KLANN_LINKAGE/62
984, 985, 986, 5.0e6, 0, 1.0e-6
30.0, 70.0, 85.0, 100.0
/KLANN_LINKAGE/63
987, 988, 989, 5.0e6, 0, 1.0e-6
30.0, 70.0, 85.0, 100.0
/KLANN_WALKING_MECHANISM/64
990, 991, 992, 5.0e6, 0, 1.0e-6
30.0, 70.0, 85.0, 100.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 61 in model.lagmul_klann_linkage_joints
    assert 62 in model.lagmul_klann_linkage_joints
    assert 63 in model.lagmul_klann_linkage_joints
    assert 64 in model.lagmul_klann_linkage_joints
    assert model.lagmul_klann_linkage_joints[61].node1 == 981
    assert model.lagmul_klann_linkage_joints[62].node1 == 984
    assert model.lagmul_klann_linkage_joints[63].node1 == 987
    assert model.lagmul_klann_linkage_joints[64].node1 == 990


def test_m300_sensor_spring_total_acceleration_rate(tmp_path: Path):
    c1 = f"{958:>10d}{2.6e7:>20.4f}{0.0075:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Acceleration Rate Fixed and Free Format Test
2022 0
/SENSOR/SPRING_TOTAL_ACCELERATION_RATE/1
Fixed Spring Total Acceleration Rate Sensor
{c1}
/SENSOR/SPRING_TOTAL_ACCELERATION_RATE/2
Free Spring Total Acceleration Rate Sensor
959, 4.4e7, 0.0095
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_acceleration_rates
    assert 2 in model.sensor_spring_total_acceleration_rates
    s1 = model.sensor_spring_total_acceleration_rates[1]
    assert s1.spring_id == 958
    assert pytest.approx(s1.jrate_max) == 2.6e7
    assert pytest.approx(s1.t_delay) == 0.0075

    s2 = model.sensor_spring_total_acceleration_rates[2]
    assert s2.spring_id == 959
    assert pytest.approx(s2.jrate_max) == 4.4e7
    assert pytest.approx(s2.t_delay) == 0.0095

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_TOTAL_ACCELERATION_RATE"
    assert model.sensors[1].kind == "SPRING_TOTAL_ACCELERATION_RATE"


def test_m300_sensor_spring_total_acceleration_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Acceleration Rate Aliases Test
/SENSOR/SPRING_TOT_ACC_RATE/91
995, 2.0e7, 0.0025
/SENSOR/SPRING_RATE_ACC_TOT/92
996, 2.2e7, 0.0035
/SENSOR/TOTAL_ACCELERATION_RATE_SPRING/93
997, 2.4e7, 0.0045
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 91 in model.sensor_spring_total_acceleration_rates
    assert 92 in model.sensor_spring_total_acceleration_rates
    assert 93 in model.sensor_spring_total_acceleration_rates
    assert model.sensor_spring_total_acceleration_rates[91].spring_id == 995
    assert pytest.approx(model.sensor_spring_total_acceleration_rates[91].jrate_max) == 2.0e7
    assert model.sensor_spring_total_acceleration_rates[92].spring_id == 996
    assert model.sensor_spring_total_acceleration_rates[93].spring_id == 997
