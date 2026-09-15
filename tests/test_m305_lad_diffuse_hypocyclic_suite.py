"""Tests for Milestone M305: LadDiffuseDamage Failure Model, EngThermophotonicEmissionEnergy, HypocyclicLinkageJoint, and SensorSpringTotalAngularAccelerationRate."""

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


def test_m305_fail_lad_diffuse_damage_fixed(tmp_path: Path):
    c1 = f"{1.8:>20.4f}{24.0:>20.4f}{0.0055:>20.6f}{0.0085:>20.6f}{0.988:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1145:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Diffuse Damage Model Fixed Format Test
2022 0
/FAIL/LAD_DIFFUSE_DAMAGE/1145
Ladeveze Nonlocal Regularized Diffuse Damage
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1145 in model.fail_laddiffusedamages
    fldd = model.fail_laddiffusedamages[1145]
    assert pytest.approx(fldd.y0_diff) == 1.8
    assert pytest.approx(fldd.yc_diff) == 24.0
    assert pytest.approx(fldd.c_reg_diff) == 0.0055
    assert pytest.approx(fldd.l_nonlocal) == 0.0085
    assert pytest.approx(fldd.d_diff_max) == 0.988
    assert fldd.ifail_sh == 1
    assert fldd.ifail_so == 2
    assert fldd.fail_id == 1145
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DIFFUSE_DAMAGE"


def test_m305_fail_lad_diffuse_damage_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Diffuse Damage Free Format Test
/FAIL/LAD_DIFFUSE_DAMAGE/1146
2.5, 30.0, 0.0075, 0.0105, 0.965
1, 1
1146
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1146 in model.fail_laddiffusedamages
    fldd = model.fail_laddiffusedamages[1146]
    assert pytest.approx(fldd.y0_diff) == 2.5
    assert pytest.approx(fldd.yc_diff) == 30.0
    assert pytest.approx(fldd.c_reg_diff) == 0.0075
    assert pytest.approx(fldd.l_nonlocal) == 0.0105
    assert pytest.approx(fldd.d_diff_max) == 0.965
    assert fldd.fail_id == 1146


def test_m305_fail_lad_diffuse_damage_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Diffuse Damage Aliases Test
/FAIL/LADEVEZE_DIFFUSE_DAMAGE/1147
1.5, 20.0, 0.0045, 0.0065, 0.97
1, 1
/FAIL/LAD_DIFFUSE/1148
1.5, 20.0, 0.0045, 0.0065, 0.97
1, 1
/FAIL/LAD_DIFFUSE_MODEL/1149
1.5, 20.0, 0.0045, 0.0065, 0.97
1, 1
/FAIL/LAD_DIFFUSE_LAW/1150
1.5, 20.0, 0.0045, 0.0065, 0.97
1, 1
/FAIL/LADEVEZE_NONLOCAL_DAMAGE/1151
1.5, 20.0, 0.0045, 0.0065, 0.97
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1147 in model.fail_laddiffusedamages
    assert 1148 in model.fail_laddiffusedamages
    assert 1149 in model.fail_laddiffusedamages
    assert 1150 in model.fail_laddiffusedamages
    assert 1151 in model.fail_laddiffusedamages
    assert len(model.raw_fails) == 5


def test_m305_eng_thermophotonic_emission_energy(tmp_path: Path):
    c1 = f"{0.00014:>20.6f}{34:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Thermophotonic Emission Energy Fixed and Free Format Test
2022 0
/ENG/THERMOPHOTONIC_EMISSION_ENERGY/1
Fixed Thermophotonic Emission Energy Output
{c1}
/ENG/THERMOPHOTONIC_EMISSION_ENERGY/2
Free Thermophotonic Emission Energy Output
0.00028, 68
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_thermophotonic_emission_energies
    assert 2 in model.eng_thermophotonic_emission_energies
    tpe1 = model.eng_thermophotonic_emission_energies[1]
    assert pytest.approx(tpe1.dt_tpe) == 0.00014
    assert tpe1.sens_id == 34
    tpe2 = model.eng_thermophotonic_emission_energies[2]
    assert pytest.approx(tpe2.dt_tpe) == 0.00028
    assert tpe2.sens_id == 68


def test_m305_eng_thermophotonic_emission_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Thermophotonic Emission Energy Aliases Test
/ENG/TPE_WORK/101
0.001, 100
/ENG/ETHERMOPHOTONIC_EMISSION/102
0.002, 101
/ENG/THERMOPHOTONIC_EMISSION_DISSIPATION/103
0.003, 102
/ENG/EM_THERMOPHOTONIC_EMISSION/104
0.004, 103
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 101 in model.eng_thermophotonic_emission_energies
    assert 102 in model.eng_thermophotonic_emission_energies
    assert 103 in model.eng_thermophotonic_emission_energies
    assert 104 in model.eng_thermophotonic_emission_energies
    assert pytest.approx(model.eng_thermophotonic_emission_energies[101].dt_tpe) == 0.001
    assert pytest.approx(model.eng_thermophotonic_emission_energies[102].dt_tpe) == 0.002
    assert pytest.approx(model.eng_thermophotonic_emission_energies[103].dt_tpe) == 0.003
    assert pytest.approx(model.eng_thermophotonic_emission_energies[104].dt_tpe) == 0.004


def test_m305_hypocyclic_linkage_joint(tmp_path: Path):
    c1 = f"{921:>10d}{922:>10d}{923:>10d}{8.8e6:>20.4f}{1:>10d}{1.0e-6:>20.6e}"
    c2 = f"{80.0:>20.4f}{40.0:>20.4f}{160.0:>20.4f}{0.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Hypocyclic Linkage Joint Fixed and Free Format Test
2022 0
/LAGMUL/HYPOCYCLIC_LINKAGE_JOINT/1
Fixed Hypocyclic Linkage Joint
{c1}
{c2}
/LAGMUL/HYPOCYCLIC_LINKAGE_JOINT/2
Free Hypocyclic Linkage Joint
1021, 1022, 1023, 9.5e6, 2, 1.6e-6
100.0, 50.0, 200.0, 0.523
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_hypocyclic_linkage_joints
    assert 2 in model.lagmul_hypocyclic_linkage_joints
    hlj1 = model.lagmul_hypocyclic_linkage_joints[1]
    assert hlj1.node1 == 921
    assert hlj1.node2 == 922
    assert hlj1.node3 == 923
    assert pytest.approx(hlj1.stiff) == 8.8e6
    assert hlj1.skew_id == 1
    assert pytest.approx(hlj1.tol) == 1.0e-6
    assert pytest.approx(hlj1.radius_outer) == 80.0
    assert pytest.approx(hlj1.radius_inner) == 40.0
    assert pytest.approx(hlj1.stroke_travel) == 160.0
    assert pytest.approx(hlj1.phase_angle) == 0.0

    hlj2 = model.lagmul_hypocyclic_linkage_joints[2]
    assert hlj2.node1 == 1021
    assert hlj2.node2 == 1022
    assert hlj2.node3 == 1023
    assert pytest.approx(hlj2.stiff) == 9.5e6
    assert hlj2.skew_id == 2
    assert pytest.approx(hlj2.tol) == 1.6e-6
    assert pytest.approx(hlj2.radius_outer) == 100.0
    assert pytest.approx(hlj2.radius_inner) == 50.0
    assert pytest.approx(hlj2.stroke_travel) == 200.0
    assert pytest.approx(hlj2.phase_angle) == 0.523


def test_m305_hypocyclic_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Hypocyclic Linkage Joint Aliases Test
/HYPOCYCLIC_LINKAGE_JOINT/105
1071, 1072, 1073, 6.2e6, 0, 1.0e-6
60.0, 30.0, 120.0, 0.0
/LAGMUL/HYPOCYCLIC_LINKAGE/106
1074, 1075, 1076, 6.2e6, 0, 1.0e-6
60.0, 30.0, 120.0, 0.0
/HYPOCYCLIC_LINKAGE/107
1077, 1078, 1079, 6.2e6, 0, 1.0e-6
60.0, 30.0, 120.0, 0.0
/TUSI_COUPLE_MECHANISM/108
1080, 1081, 1082, 6.2e6, 0, 1.0e-6
60.0, 30.0, 120.0, 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 105 in model.lagmul_hypocyclic_linkage_joints
    assert 106 in model.lagmul_hypocyclic_linkage_joints
    assert 107 in model.lagmul_hypocyclic_linkage_joints
    assert 108 in model.lagmul_hypocyclic_linkage_joints
    assert model.lagmul_hypocyclic_linkage_joints[105].node1 == 1071
    assert model.lagmul_hypocyclic_linkage_joints[106].node1 == 1074
    assert model.lagmul_hypocyclic_linkage_joints[107].node1 == 1077
    assert model.lagmul_hypocyclic_linkage_joints[108].node1 == 1080


def test_m305_sensor_spring_total_angular_acceleration_rate(tmp_path: Path):
    c1 = f"{1008:>10d}{3.3e7:>20.4f}{0.0092:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Angular Acceleration Rate Fixed and Free Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_ACCELERATION_RATE/1
Fixed Spring Total Angular Acceleration Rate Sensor
{c1}
/SENSOR/SPRING_TOTAL_ANGULAR_ACCELERATION_RATE/2
Free Spring Total Angular Acceleration Rate Sensor
1009, 5.4e7, 0.0124
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_angular_acceleration_rates
    assert 2 in model.sensor_spring_total_angular_acceleration_rates
    s1 = model.sensor_spring_total_angular_acceleration_rates[1]
    assert s1.spring_id == 1008
    assert pytest.approx(s1.jtot_ang_rate_max) == 3.3e7
    assert pytest.approx(s1.t_delay) == 0.0092

    s2 = model.sensor_spring_total_angular_acceleration_rates[2]
    assert s2.spring_id == 1009
    assert pytest.approx(s2.jtot_ang_rate_max) == 5.4e7
    assert pytest.approx(s2.t_delay) == 0.0124

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_ACCELERATION_RATE"
    assert model.sensors[1].kind == "SPRING_TOTAL_ANGULAR_ACCELERATION_RATE"


def test_m305_sensor_spring_total_angular_acceleration_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Angular Acceleration Rate Aliases Test
/SENSOR/SPRING_TOT_ANG_ACC_RATE/107
1038, 2.35e7, 0.0029
/SENSOR/SPRING_RATE_ACC_ANG_TOT/108
1039, 2.55e7, 0.0039
/SENSOR/TOTAL_ANGULAR_ACCELERATION_RATE_SPRING/109
1040, 2.75e7, 0.0049
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 107 in model.sensor_spring_total_angular_acceleration_rates
    assert 108 in model.sensor_spring_total_angular_acceleration_rates
    assert 109 in model.sensor_spring_total_angular_acceleration_rates
    assert model.sensor_spring_total_angular_acceleration_rates[107].spring_id == 1038
    assert pytest.approx(model.sensor_spring_total_angular_acceleration_rates[107].jtot_ang_rate_max) == 2.35e7
    assert model.sensor_spring_total_angular_acceleration_rates[108].spring_id == 1039
    assert model.sensor_spring_total_angular_acceleration_rates[109].spring_id == 1040
