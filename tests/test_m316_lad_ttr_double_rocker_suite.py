"""Tests for Milestone M316: LadTransverseTensionRate Failure Model, EngThermoplasmonicEnergy, FourBarDoubleRockerJoint, and SensorSpringNormalSnapRate."""

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


def test_m316_fail_lad_transverse_tension_rate_fixed(tmp_path: Path):
    c1 = f"{2.75:>20.4f}{38.5:>20.4f}{0.068:>20.4f}{1.48:>20.4f}{0.986:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1255:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Tension Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_TENSION_RATE/1255
Ladeveze Transverse Tension Rate Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1255 in model.fail_ladtransversetensionrates
    flttr = model.fail_ladtransversetensionrates[1255]
    assert pytest.approx(flttr.y0_ttr) == 2.75
    assert pytest.approx(flttr.yc_ttr) == 38.5
    assert pytest.approx(flttr.c_rate_ttr) == 0.068
    assert pytest.approx(flttr.p_rate_ttr) == 1.48
    assert pytest.approx(flttr.d_ttr_max) == 0.986
    assert flttr.ifail_sh == 1
    assert flttr.ifail_so == 2
    assert flttr.fail_id == 1255
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_TENSION_RATE"


def test_m316_fail_lad_transverse_tension_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Tension Rate Free Format Test
/FAIL/LAD_TRANSVERSE_TENSION_RATE/1256
3.55, 49.0, 0.088, 1.7, 0.965
1, 1
1256
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1256 in model.fail_ladtransversetensionrates
    flttr = model.fail_ladtransversetensionrates[1256]
    assert pytest.approx(flttr.y0_ttr) == 3.55
    assert pytest.approx(flttr.yc_ttr) == 49.0
    assert pytest.approx(flttr.c_rate_ttr) == 0.088
    assert pytest.approx(flttr.p_rate_ttr) == 1.7
    assert pytest.approx(flttr.d_ttr_max) == 0.965
    assert flttr.fail_id == 1256


def test_m316_fail_lad_transverse_tension_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Tension Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_TENSION_RATE/1257
2.3, 31.0, 0.050, 1.4, 0.95
1, 1
/FAIL/LAD_TTR/1258
2.3, 31.0, 0.050, 1.4, 0.95
1, 1
/FAIL/LAD_TTR_MODEL/1259
2.3, 31.0, 0.050, 1.4, 0.95
1, 1
/FAIL/LAD_TTR_LAW/1260
2.3, 31.0, 0.050, 1.4, 0.95
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_MICROCRACKING/1261
2.3, 31.0, 0.050, 1.4, 0.95
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1257 in model.fail_ladtransversetensionrates
    assert 1258 in model.fail_ladtransversetensionrates
    assert 1259 in model.fail_ladtransversetensionrates
    assert 1260 in model.fail_ladtransversetensionrates
    assert 1261 in model.fail_ladtransversetensionrates
    assert len(model.raw_fails) == 5


def test_m316_eng_thermoplasmonic_energy(tmp_path: Path):
    c1 = f"{0.00042:>20.6f}{62:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Thermoplasmonic Energy Fixed and Free Format Test
2022 0
/ENG/THERMOPLASMONIC_ENERGY/1
Fixed Thermoplasmonic Energy Output
{c1}
/ENG/THERMOPLASMONIC_ENERGY/2
Free Thermoplasmonic Energy Output
0.00084, 124
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_thermoplasmonic_energies
    assert 2 in model.eng_thermoplasmonic_energies
    tpl1 = model.eng_thermoplasmonic_energies[1]
    assert pytest.approx(tpl1.dt_tpl) == 0.00042
    assert tpl1.sens_id == 62
    tpl2 = model.eng_thermoplasmonic_energies[2]
    assert pytest.approx(tpl2.dt_tpl) == 0.00084
    assert tpl2.sens_id == 124


def test_m316_eng_thermoplasmonic_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Thermoplasmonic Energy Aliases Test
/ENG/TPL_WORK/211
0.001, 210
/ENG/ETHERMOPLASMONIC/212
0.002, 211
/ENG/THERMOPLASMONIC_DISSIPATION/213
0.003, 212
/ENG/EM_THERMOPLASMONIC/214
0.004, 213
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 211 in model.eng_thermoplasmonic_energies
    assert 212 in model.eng_thermoplasmonic_energies
    assert 213 in model.eng_thermoplasmonic_energies
    assert 214 in model.eng_thermoplasmonic_energies
    assert pytest.approx(model.eng_thermoplasmonic_energies[211].dt_tpl) == 0.001
    assert pytest.approx(model.eng_thermoplasmonic_energies[212].dt_tpl) == 0.002
    assert pytest.approx(model.eng_thermoplasmonic_energies[213].dt_tpl) == 0.003
    assert pytest.approx(model.eng_thermoplasmonic_energies[214].dt_tpl) == 0.004


def test_m316_four_bar_double_rocker_joint(tmp_path: Path):
    c1 = f"{1011:>10d}{1012:>10d}{1013:>10d}{1.08e7:>20.4f}{1:>10d}{1.0e-6:>20.6e}"
    c2 = f"{90.0:>20.4f}{150.0:>20.4f}{110.0:>20.4f}{80.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Four-Bar Double Rocker Joint Fixed and Free Format Test
2022 0
/LAGMUL/FOUR_BAR_DOUBLE_ROCKER_JOINT/1
Fixed Four-Bar Double Rocker Joint
{c1}
{c2}
/LAGMUL/FOUR_BAR_DOUBLE_ROCKER_JOINT/2
Free Four-Bar Double Rocker Joint
1111, 1112, 1113, 1.25e7, 2, 2.8e-6
105.0, 175.0, 130.0, 95.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_four_bar_double_rocker_joints
    assert 2 in model.lagmul_four_bar_double_rocker_joints
    fdrj1 = model.lagmul_four_bar_double_rocker_joints[1]
    assert fdrj1.node1 == 1011
    assert fdrj1.node2 == 1012
    assert fdrj1.node3 == 1013
    assert pytest.approx(fdrj1.stiff) == 1.08e7
    assert fdrj1.skew_id == 1
    assert pytest.approx(fdrj1.tol) == 1.0e-6
    assert pytest.approx(fdrj1.input_rocker_len) == 90.0
    assert pytest.approx(fdrj1.coupler_len) == 150.0
    assert pytest.approx(fdrj1.output_rocker_len) == 110.0
    assert pytest.approx(fdrj1.ground_len) == 80.0

    fdrj2 = model.lagmul_four_bar_double_rocker_joints[2]
    assert fdrj2.node1 == 1111
    assert fdrj2.node2 == 1112
    assert fdrj2.node3 == 1113
    assert pytest.approx(fdrj2.stiff) == 1.25e7
    assert fdrj2.skew_id == 2
    assert pytest.approx(fdrj2.tol) == 2.8e-6
    assert pytest.approx(fdrj2.input_rocker_len) == 105.0
    assert pytest.approx(fdrj2.coupler_len) == 175.0
    assert pytest.approx(fdrj2.output_rocker_len) == 130.0
    assert pytest.approx(fdrj2.ground_len) == 95.0


def test_m316_four_bar_double_rocker_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Four-Bar Double Rocker Joint Aliases Test
/FOUR_BAR_DOUBLE_ROCKER_JOINT/215
1181, 1182, 1183, 8.1e6, 0, 1.0e-6
80.0, 130.0, 100.0, 70.0
/LAGMUL/FOUR_BAR_DOUBLE_ROCKER/216
1184, 1185, 1186, 8.1e6, 0, 1.0e-6
80.0, 130.0, 100.0, 70.0
/FOUR_BAR_DOUBLE_ROCKER/217
1187, 1188, 1189, 8.1e6, 0, 1.0e-6
80.0, 130.0, 100.0, 70.0
/DOUBLE_ROCKER_MECHANISM/218
1190, 1191, 1192, 8.1e6, 0, 1.0e-6
80.0, 130.0, 100.0, 70.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 215 in model.lagmul_four_bar_double_rocker_joints
    assert 216 in model.lagmul_four_bar_double_rocker_joints
    assert 217 in model.lagmul_four_bar_double_rocker_joints
    assert 218 in model.lagmul_four_bar_double_rocker_joints
    assert model.lagmul_four_bar_double_rocker_joints[215].node1 == 1181
    assert model.lagmul_four_bar_double_rocker_joints[216].node1 == 1184
    assert model.lagmul_four_bar_double_rocker_joints[217].node1 == 1187
    assert model.lagmul_four_bar_double_rocker_joints[218].node1 == 1190


def test_m316_sensor_spring_normal_snap_rate(tmp_path: Path):
    c1 = f"{1118:>10d}{5.9e7:>20.4f}{0.0185:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Normal Snap Rate Fixed and Free Format Test
2022 0
/SENSOR/SPRING_NORMAL_SNAP_RATE/1
Fixed Spring Normal Snap Rate Sensor
{c1}
/SENSOR/SPRING_NORMAL_SNAP_RATE/2
Free Spring Normal Snap Rate Sensor
1119, 8.1e7, 0.0255
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_normal_snap_rates
    assert 2 in model.sensor_spring_normal_snap_rates
    s1 = model.sensor_spring_normal_snap_rates[1]
    assert s1.spring_id == 1118
    assert pytest.approx(s1.jnorm_crackle_max) == 5.9e7
    assert pytest.approx(s1.t_delay) == 0.0185

    s2 = model.sensor_spring_normal_snap_rates[2]
    assert s2.spring_id == 1119
    assert pytest.approx(s2.jnorm_crackle_max) == 8.1e7
    assert pytest.approx(s2.t_delay) == 0.0255

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_NORMAL_SNAP_RATE"
    assert model.sensors[1].kind == "SPRING_NORMAL_SNAP_RATE"


def test_m316_sensor_spring_normal_snap_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Normal Snap Rate Aliases Test
/SENSOR/SPRING_NORM_SNAP_RATE/140
1148, 3.45e7, 0.0045
/SENSOR/SPRING_RATE_SNAP_NORM/141
1149, 3.65e7, 0.0055
/SENSOR/NORMAL_SNAP_RATE_SPRING/142
1150, 3.85e7, 0.0065
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 140 in model.sensor_spring_normal_snap_rates
    assert 141 in model.sensor_spring_normal_snap_rates
    assert 142 in model.sensor_spring_normal_snap_rates
    assert model.sensor_spring_normal_snap_rates[140].spring_id == 1148
    assert pytest.approx(model.sensor_spring_normal_snap_rates[140].jnorm_crackle_max) == 3.45e7
    assert model.sensor_spring_normal_snap_rates[141].spring_id == 1149
    assert model.sensor_spring_normal_snap_rates[142].spring_id == 1150
