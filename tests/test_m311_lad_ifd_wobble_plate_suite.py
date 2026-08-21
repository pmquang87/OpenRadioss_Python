"""Tests for Milestone M311: LadInterfacialDelamination Failure Model, EngThermomagnetoelectricEnergy, WobblePlateMechanismJoint, and SensorSpringTotalJerkRate."""

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


def test_m311_fail_lad_interfacial_delamination_fixed(tmp_path: Path):
    c1 = f"{1.75:>20.4f}{28.5:>20.4f}{1.45:>20.4f}{2.5e6:>20.4e}{0.982:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1205:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Interfacial Delamination Fixed Format Test
2022 0
/FAIL/LAD_INTERFACIAL_DELAMINATION/1205
Ladeveze Interfacial Delamination Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1205 in model.fail_ladinterfacialdelaminations
    flifd = model.fail_ladinterfacialdelaminations[1205]
    assert pytest.approx(flifd.y0_ifd) == 1.75
    assert pytest.approx(flifd.yc_ifd) == 28.5
    assert pytest.approx(flifd.b_ifd_mix) == 1.45
    assert pytest.approx(flifd.k_ifd_penalty) == 2.5e6
    assert pytest.approx(flifd.d_ifd_max) == 0.982
    assert flifd.ifail_sh == 1
    assert flifd.ifail_so == 2
    assert flifd.fail_id == 1205
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_INTERFACIAL_DELAMINATION"


def test_m311_fail_lad_interfacial_delamination_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Interfacial Delamination Free Format Test
/FAIL/LAD_INTERFACIAL_DELAMINATION/1206
2.4, 36.0, 1.6, 3.2e6, 0.972
1, 1
1206
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1206 in model.fail_ladinterfacialdelaminations
    flifd = model.fail_ladinterfacialdelaminations[1206]
    assert pytest.approx(flifd.y0_ifd) == 2.4
    assert pytest.approx(flifd.yc_ifd) == 36.0
    assert pytest.approx(flifd.b_ifd_mix) == 1.6
    assert pytest.approx(flifd.k_ifd_penalty) == 3.2e6
    assert pytest.approx(flifd.d_ifd_max) == 0.972
    assert flifd.fail_id == 1206


def test_m311_fail_lad_interfacial_delamination_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Interfacial Delamination Aliases Test
/FAIL/LADEVEZE_INTERFACIAL_DELAMINATION/1207
1.5, 25.0, 1.2, 2.0e6, 0.96
1, 1
/FAIL/LAD_IFD/1208
1.5, 25.0, 1.2, 2.0e6, 0.96
1, 1
/FAIL/LAD_IFD_MODEL/1209
1.5, 25.0, 1.2, 2.0e6, 0.96
1, 1
/FAIL/LAD_IFD_LAW/1210
1.5, 25.0, 1.2, 2.0e6, 0.96
1, 1
/FAIL/LADEVEZE_COHESIVE_DELAMINATION/1211
1.5, 25.0, 1.2, 2.0e6, 0.96
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1207 in model.fail_ladinterfacialdelaminations
    assert 1208 in model.fail_ladinterfacialdelaminations
    assert 1209 in model.fail_ladinterfacialdelaminations
    assert 1210 in model.fail_ladinterfacialdelaminations
    assert 1211 in model.fail_ladinterfacialdelaminations
    assert len(model.raw_fails) == 5


def test_m311_eng_thermomagnetoelectric_energy(tmp_path: Path):
    c1 = f"{0.00025:>20.6f}{45:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Thermomagnetoelectric Energy Fixed and Free Format Test
2022 0
/ENG/THERMOMAGNETOELECTRIC_ENERGY/1
Fixed Thermomagnetoelectric Energy Output
{c1}
/ENG/THERMOMAGNETOELECTRIC_ENERGY/2
Free Thermomagnetoelectric Energy Output
0.00050, 90
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_thermomagnetoelectric_energies
    assert 2 in model.eng_thermomagnetoelectric_energies
    tme1 = model.eng_thermomagnetoelectric_energies[1]
    assert pytest.approx(tme1.dt_tme) == 0.00025
    assert tme1.sens_id == 45
    tme2 = model.eng_thermomagnetoelectric_energies[2]
    assert pytest.approx(tme2.dt_tme) == 0.00050
    assert tme2.sens_id == 90


def test_m311_eng_thermomagnetoelectric_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Thermomagnetoelectric Energy Aliases Test
/ENG/TME_WORK/161
0.001, 160
/ENG/ETHERMOMAGNETOELECTRIC/162
0.002, 161
/ENG/THERMOMAGNETOELECTRIC_DISSIPATION/163
0.003, 162
/ENG/EM_THERMOMAGNETOELECTRIC/164
0.004, 163
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 161 in model.eng_thermomagnetoelectric_energies
    assert 162 in model.eng_thermomagnetoelectric_energies
    assert 163 in model.eng_thermomagnetoelectric_energies
    assert 164 in model.eng_thermomagnetoelectric_energies
    assert pytest.approx(model.eng_thermomagnetoelectric_energies[161].dt_tme) == 0.001
    assert pytest.approx(model.eng_thermomagnetoelectric_energies[162].dt_tme) == 0.002
    assert pytest.approx(model.eng_thermomagnetoelectric_energies[163].dt_tme) == 0.003
    assert pytest.approx(model.eng_thermomagnetoelectric_energies[164].dt_tme) == 0.004


def test_m311_wobble_plate_mechanism_joint(tmp_path: Path):
    c1 = f"{981:>10d}{982:>10d}{983:>10d}{9.4e6:>20.4f}{1:>10d}{1.0e-6:>20.6e}"
    c2 = f"{80.0:>20.4f}{0.25:>20.4f}{65.0:>20.4f}{5:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Wobble Plate Mechanism Joint Fixed and Free Format Test
2022 0
/LAGMUL/WOBBLE_PLATE_MECHANISM_JOINT/1
Fixed Wobble Plate Mechanism Joint
{c1}
{c2}
/LAGMUL/WOBBLE_PLATE_MECHANISM_JOINT/2
Free Wobble Plate Mechanism Joint
1081, 1082, 1083, 1.08e7, 2, 2.2e-6
95.0, 0.32, 75.0, 7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_wobble_plate_mechanism_joints
    assert 2 in model.lagmul_wobble_plate_mechanism_joints
    wpmj1 = model.lagmul_wobble_plate_mechanism_joints[1]
    assert wpmj1.node1 == 981
    assert wpmj1.node2 == 982
    assert wpmj1.node3 == 983
    assert pytest.approx(wpmj1.stiff) == 9.4e6
    assert wpmj1.skew_id == 1
    assert pytest.approx(wpmj1.tol) == 1.0e-6
    assert pytest.approx(wpmj1.plate_radius) == 80.0
    assert pytest.approx(wpmj1.nutation_angle) == 0.25
    assert pytest.approx(wpmj1.stroke_travel) == 65.0
    assert wpmj1.piston_count == 5

    wpmj2 = model.lagmul_wobble_plate_mechanism_joints[2]
    assert wpmj2.node1 == 1081
    assert wpmj2.node2 == 1082
    assert wpmj2.node3 == 1083
    assert pytest.approx(wpmj2.stiff) == 1.08e7
    assert wpmj2.skew_id == 2
    assert pytest.approx(wpmj2.tol) == 2.2e-6
    assert pytest.approx(wpmj2.plate_radius) == 95.0
    assert pytest.approx(wpmj2.nutation_angle) == 0.32
    assert pytest.approx(wpmj2.stroke_travel) == 75.0
    assert wpmj2.piston_count == 7


def test_m311_wobble_plate_mechanism_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Wobble Plate Mechanism Joint Aliases Test
/WOBBLE_PLATE_MECHANISM_JOINT/165
1131, 1132, 1133, 6.9e6, 0, 1.0e-6
70.0, 0.2, 50.0, 4
/LAGMUL/WOBBLE_PLATE_MECHANISM/166
1134, 1135, 1136, 6.9e6, 0, 1.0e-6
70.0, 0.2, 50.0, 4
/WOBBLE_PLATE_MECHANISM/167
1137, 1138, 1139, 6.9e6, 0, 1.0e-6
70.0, 0.2, 50.0, 4
/SWASH_WOBBLE_AXIAL_JOINT/168
1140, 1141, 1142, 6.9e6, 0, 1.0e-6
70.0, 0.2, 50.0, 4
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 165 in model.lagmul_wobble_plate_mechanism_joints
    assert 166 in model.lagmul_wobble_plate_mechanism_joints
    assert 167 in model.lagmul_wobble_plate_mechanism_joints
    assert 168 in model.lagmul_wobble_plate_mechanism_joints
    assert model.lagmul_wobble_plate_mechanism_joints[165].node1 == 1131
    assert model.lagmul_wobble_plate_mechanism_joints[166].node1 == 1134
    assert model.lagmul_wobble_plate_mechanism_joints[167].node1 == 1137
    assert model.lagmul_wobble_plate_mechanism_joints[168].node1 == 1140


def test_m311_sensor_spring_total_jerk_rate(tmp_path: Path):
    c1 = f"{1068:>10d}{4.5e7:>20.4f}{0.0135:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Jerk Rate Fixed and Free Format Test
2022 0
/SENSOR/SPRING_TOTAL_JERK_RATE/1
Fixed Spring Total Jerk Rate Sensor
{c1}
/SENSOR/SPRING_TOTAL_JERK_RATE/2
Free Spring Total Jerk Rate Sensor
1069, 6.7e7, 0.0182
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_jerk_rates
    assert 2 in model.sensor_spring_total_jerk_rates
    s1 = model.sensor_spring_total_jerk_rates[1]
    assert s1.spring_id == 1068
    assert pytest.approx(s1.jtot_snap_max) == 4.5e7
    assert pytest.approx(s1.t_delay) == 0.0135

    s2 = model.sensor_spring_total_jerk_rates[2]
    assert s2.spring_id == 1069
    assert pytest.approx(s2.jtot_snap_max) == 6.7e7
    assert pytest.approx(s2.t_delay) == 0.0182

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_TOTAL_JERK_RATE"
    assert model.sensors[1].kind == "SPRING_TOTAL_JERK_RATE"


def test_m311_sensor_spring_total_jerk_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Jerk Rate Aliases Test
/SENSOR/SPRING_TOT_JERK_RATE/125
1098, 2.95e7, 0.0035
/SENSOR/SPRING_RATE_JERK_TOT/126
1099, 3.15e7, 0.0045
/SENSOR/TOTAL_JERK_RATE_SPRING/127
1100, 3.35e7, 0.0055
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 125 in model.sensor_spring_total_jerk_rates
    assert 126 in model.sensor_spring_total_jerk_rates
    assert 127 in model.sensor_spring_total_jerk_rates
    assert model.sensor_spring_total_jerk_rates[125].spring_id == 1098
    assert pytest.approx(model.sensor_spring_total_jerk_rates[125].jtot_snap_max) == 2.95e7
    assert model.sensor_spring_total_jerk_rates[126].spring_id == 1099
    assert model.sensor_spring_total_jerk_rates[127].spring_id == 1100
