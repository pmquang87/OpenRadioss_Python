"""Tests for Milestone M297: LadTcAsymmetry Failure Model, EngThermoelectricEnergy, HartLinkageJoint, and SensorSpringTorsionalJerk."""

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


def test_m297_fail_lad_tc_asymmetry_fixed(tmp_path: Path):
    c1 = f"{1.2:>20.4f}{15.0:>20.4f}{2.4:>20.4f}{28.0:>20.4f}{0.965:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1065:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Tension-Compression Asymmetry Model Fixed Format Test
2022 0
/FAIL/LAD_TC_ASYMMETRY/1065
Ladeveze Asymmetric Damage Criterion
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1065 in model.fail_ladtcasymmetries
    fltc = model.fail_ladtcasymmetries[1065]
    assert pytest.approx(fltc.y0_t) == 1.2
    assert pytest.approx(fltc.yc_t) == 15.0
    assert pytest.approx(fltc.y0_c) == 2.4
    assert pytest.approx(fltc.yc_c) == 28.0
    assert pytest.approx(fltc.d_tc_max) == 0.965
    assert fltc.ifail_sh == 1
    assert fltc.ifail_so == 2
    assert fltc.fail_id == 1065
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TC_ASYMMETRY"


def test_m297_fail_lad_tc_asymmetry_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Tension-Compression Asymmetry Free Format Test
/FAIL/LAD_TC_ASYMMETRY/1066
1.5, 18.0, 3.0, 32.0, 0.940
1, 1
1066
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1066 in model.fail_ladtcasymmetries
    fltc = model.fail_ladtcasymmetries[1066]
    assert pytest.approx(fltc.y0_t) == 1.5
    assert pytest.approx(fltc.yc_t) == 18.0
    assert pytest.approx(fltc.y0_c) == 3.0
    assert pytest.approx(fltc.yc_c) == 32.0
    assert pytest.approx(fltc.d_tc_max) == 0.940
    assert fltc.fail_id == 1066


def test_m297_fail_lad_tc_asymmetry_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Tension-Compression Asymmetry Aliases Test
/FAIL/LADEVEZE_TENSION_COMPRESSION_ASYMMETRY/1067
1.0, 12.0, 2.0, 24.0, 0.95
1, 1
/FAIL/LAD_TC_ASYM/1068
1.0, 12.0, 2.0, 24.0, 0.95
1, 1
/FAIL/LAD_TC_MODEL/1069
1.0, 12.0, 2.0, 24.0, 0.95
1, 1
/FAIL/LAD_TC_LAW/1070
1.0, 12.0, 2.0, 24.0, 0.95
1, 1
/FAIL/LADEVEZE_ASYMMETRIC_DAMAGE/1071
1.0, 12.0, 2.0, 24.0, 0.95
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1067 in model.fail_ladtcasymmetries
    assert 1068 in model.fail_ladtcasymmetries
    assert 1069 in model.fail_ladtcasymmetries
    assert 1070 in model.fail_ladtcasymmetries
    assert 1071 in model.fail_ladtcasymmetries
    assert len(model.raw_fails) == 5


def test_m297_eng_thermoelectric_energy(tmp_path: Path):
    c1 = f"{0.00045:>20.6f}{52:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Thermoelectric Energy Fixed and Free Format Test
2022 0
/ENG/THERMOELECTRIC_ENERGY/1
Fixed Thermoelectric Energy Output
{c1}
/ENG/THERMOELECTRIC_ENERGY/2
Free Thermoelectric Energy Output
0.0009, 104
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_thermoelectric_energies
    assert 2 in model.eng_thermoelectric_energies
    te1 = model.eng_thermoelectric_energies[1]
    assert pytest.approx(te1.dt_thermoelectric) == 0.00045
    assert te1.sens_id == 52
    te2 = model.eng_thermoelectric_energies[2]
    assert pytest.approx(te2.dt_thermoelectric) == 0.0009
    assert te2.sens_id == 104


def test_m297_eng_thermoelectric_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Thermoelectric Energy Aliases Test
/ENG/TE_WORK/21
0.001, 20
/ENG/ETHERMOELECTRIC/22
0.002, 21
/ENG/THERMOELECTRIC_DISSIPATION/23
0.003, 22
/ENG/EM_THERMOELECTRIC/24
0.004, 23
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 21 in model.eng_thermoelectric_energies
    assert 22 in model.eng_thermoelectric_energies
    assert 23 in model.eng_thermoelectric_energies
    assert 24 in model.eng_thermoelectric_energies
    assert pytest.approx(model.eng_thermoelectric_energies[21].dt_thermoelectric) == 0.001
    assert pytest.approx(model.eng_thermoelectric_energies[22].dt_thermoelectric) == 0.002
    assert pytest.approx(model.eng_thermoelectric_energies[23].dt_thermoelectric) == 0.003
    assert pytest.approx(model.eng_thermoelectric_energies[24].dt_thermoelectric) == 0.004


def test_m297_hart_linkage_joint(tmp_path: Path):
    c1 = f"{801:>10d}{802:>10d}{803:>10d}{7.5e6:>20.4f}{1:>10d}{1.4e-6:>20.6e}"
    c2 = f"{100.0:>20.4f}{40.0:>20.4f}{80.0:>20.4f}{0.50:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Hart Linkage Joint Fixed and Free Format Test
2022 0
/LAGMUL/HART_LINKAGE_JOINT/1
Fixed Hart Linkage Joint
{c1}
{c2}
/LAGMUL/HART_LINKAGE_JOINT/2
Free Hart Linkage Joint
901, 902, 903, 8.2e6, 2, 1.8e-6
120.0, 50.0, 100.0, 0.60
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_hart_linkage_joints
    assert 2 in model.lagmul_hart_linkage_joints
    hj1 = model.lagmul_hart_linkage_joints[1]
    assert hj1.node1 == 801
    assert hj1.node2 == 802
    assert hj1.node3 == 803
    assert pytest.approx(hj1.stiff) == 7.5e6
    assert hj1.skew_id == 1
    assert pytest.approx(hj1.tol) == 1.4e-6
    assert pytest.approx(hj1.base_len) == 100.0
    assert pytest.approx(hj1.short_link_len) == 40.0
    assert pytest.approx(hj1.long_link_len) == 80.0
    assert pytest.approx(hj1.coupler_ratio) == 0.50

    hj2 = model.lagmul_hart_linkage_joints[2]
    assert hj2.node1 == 901
    assert hj2.node2 == 902
    assert hj2.node3 == 903
    assert pytest.approx(hj2.stiff) == 8.2e6
    assert hj2.skew_id == 2
    assert pytest.approx(hj2.tol) == 1.8e-6
    assert pytest.approx(hj2.base_len) == 120.0
    assert pytest.approx(hj2.short_link_len) == 50.0
    assert pytest.approx(hj2.long_link_len) == 100.0
    assert pytest.approx(hj2.coupler_ratio) == 0.60


def test_m297_hart_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Hart Linkage Joint Aliases Test
/HART_LINKAGE_JOINT/31
981, 982, 983, 2.0e6, 0, 1.0e-6
80.0, 30.0, 60.0, 0.5
/LAGMUL/HART_LINKAGE/32
984, 985, 986, 2.0e6, 0, 1.0e-6
80.0, 30.0, 60.0, 0.5
/HART_LINKAGE/33
987, 988, 989, 2.0e6, 0, 1.0e-6
80.0, 30.0, 60.0, 0.5
/HART_STRAIGHT_LINE_INVERSOR/34
990, 991, 992, 2.0e6, 0, 1.0e-6
80.0, 30.0, 60.0, 0.5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 31 in model.lagmul_hart_linkage_joints
    assert 32 in model.lagmul_hart_linkage_joints
    assert 33 in model.lagmul_hart_linkage_joints
    assert 34 in model.lagmul_hart_linkage_joints
    assert model.lagmul_hart_linkage_joints[31].node1 == 981
    assert model.lagmul_hart_linkage_joints[32].node1 == 984
    assert model.lagmul_hart_linkage_joints[33].node1 == 987
    assert model.lagmul_hart_linkage_joints[34].node1 == 990


def test_m297_sensor_spring_torsional_jerk(tmp_path: Path):
    c1 = f"{928:>10d}{1.5e7:>20.4f}{0.0045:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Jerk Fixed and Free Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_JERK/1
Fixed Spring Torsional Jerk Sensor
{c1}
/SENSOR/SPRING_TORSIONAL_JERK/2
Free Spring Torsional Jerk Sensor
929, 2.8e7, 0.0065
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_jerks
    assert 2 in model.sensor_spring_torsional_jerks
    s1 = model.sensor_spring_torsional_jerks[1]
    assert s1.spring_id == 928
    assert pytest.approx(s1.jtors_max) == 1.5e7
    assert pytest.approx(s1.t_delay) == 0.0045

    s2 = model.sensor_spring_torsional_jerks[2]
    assert s2.spring_id == 929
    assert pytest.approx(s2.jtors_max) == 2.8e7
    assert pytest.approx(s2.t_delay) == 0.0065

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_TORSIONAL_JERK"
    assert model.sensors[1].kind == "SPRING_TORSIONAL_JERK"


def test_m297_sensor_spring_torsional_jerk_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Torsional Jerk Aliases Test
/SENSOR/SPRING_TORS_JERK/61
995, 1.1e7, 0.0015
/SENSOR/SPRING_JERK_TORS/62
996, 1.2e7, 0.0025
/SENSOR/TORSIONAL_JERK_SPRING/63
997, 1.3e7, 0.0035
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 61 in model.sensor_spring_torsional_jerks
    assert 62 in model.sensor_spring_torsional_jerks
    assert 63 in model.sensor_spring_torsional_jerks
    assert model.sensor_spring_torsional_jerks[61].spring_id == 995
    assert pytest.approx(model.sensor_spring_torsional_jerks[61].jtors_max) == 1.1e7
    assert model.sensor_spring_torsional_jerks[62].spring_id == 996
    assert model.sensor_spring_torsional_jerks[63].spring_id == 997
