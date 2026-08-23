"""Tests for Milestone M333: LadCoupleDynamicCrush Failure Model, EngFlexothermoelectricResonanceEnergy, FrankeLinkageJoint, and SensorSpringTotalLockRate."""

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


def test_m333_fail_lad_couple_dynamic_crush_fixed(tmp_path: Path):
    c1 = f"{162.0:>20.4f}{450.0:>20.4f}{32.5:>20.4f}{1.58:>20.4f}{0.980:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1490:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Dynamic Crush Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_DYNAMIC_CRUSH/1490
Ladeveze Dynamic Coupled Crush Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1490 in model.fail_ladcoupledynamiccrushs
    fcdc = model.fail_ladcoupledynamiccrushs[1490]
    assert pytest.approx(fcdc.sigma_cdc0) == 162.0
    assert pytest.approx(fcdc.sigma_cdcc) == 450.0
    assert pytest.approx(fcdc.gamma_cdc) == 32.5
    assert pytest.approx(fcdc.p_cdc) == 1.58
    assert pytest.approx(fcdc.d_cdc_max) == 0.980
    assert fcdc.ifail_sh == 1
    assert fcdc.ifail_so == 2
    assert fcdc.fail_id == 1490
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_DYNAMIC_CRUSH"


def test_m333_fail_lad_couple_dynamic_crush_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Dynamic Crush Free Format Test
/FAIL/LAD_COUPLE_DYNAMIC_CRUSH/1491
175.0, 510.0, 36.0, 1.70, 0.960
1, 1
1491
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1491 in model.fail_ladcoupledynamiccrushs
    fcdc = model.fail_ladcoupledynamiccrushs[1491]
    assert pytest.approx(fcdc.sigma_cdc0) == 175.0
    assert pytest.approx(fcdc.sigma_cdcc) == 510.0
    assert pytest.approx(fcdc.gamma_cdc) == 36.0
    assert pytest.approx(fcdc.p_cdc) == 1.70
    assert pytest.approx(fcdc.d_cdc_max) == 0.960
    assert fcdc.fail_id == 1491


def test_m333_fail_lad_couple_dynamic_crush_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Dynamic Crush Aliases Test
/FAIL/LADEVEZE_COUPLED_DYNAMIC_CRUSH/1492
130.0, 370.0, 24.0, 1.30, 0.93
1, 1
/FAIL/LAD_CDC/1493
130.0, 370.0, 24.0, 1.30, 0.93
1, 1
/FAIL/LAD_CDC_MODEL/1494
130.0, 370.0, 24.0, 1.30, 0.93
1, 1
/FAIL/LAD_CDC_LAW/1495
130.0, 370.0, 24.0, 1.30, 0.93
1, 1
/FAIL/LADEVEZE_COUPLED_DYNAMIC_CRUSHING/1496
130.0, 370.0, 24.0, 1.30, 0.93
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1492 in model.fail_ladcoupledynamiccrushs
    assert 1493 in model.fail_ladcoupledynamiccrushs
    assert 1494 in model.fail_ladcoupledynamiccrushs
    assert 1495 in model.fail_ladcoupledynamiccrushs
    assert 1496 in model.fail_ladcoupledynamiccrushs
    assert len(model.raw_fails) == 5


def test_m333_fail_lad_couple_dynamic_crush_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_COUPLE_DYNAMIC_CRUSH/1497
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m333_eng_flexothermoelectric_resonance_energy(tmp_path: Path):
    c1 = f"{0.00128:>20.6f}{175:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermoelectric Resonance Energy Fixed and Free Format Test
2022 0
/ENG/FLEXOTHERMOELECTRIC_RESONANCE_ENERGY/1
Fixed Flexothermoelectric Resonance Energy Output
{c1}
/ENG/FLEXOTHERMOELECTRIC_RESONANCE_ENERGY/2
Free Flexothermoelectric Resonance Energy Output
0.00168, 330
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermoelectric_resonance_energies
    assert 2 in model.eng_flexothermoelectric_resonance_energies
    fter1 = model.eng_flexothermoelectric_resonance_energies[1]
    assert pytest.approx(fter1.dt_fter) == 0.00128
    assert fter1.sens_id == 175
    fter2 = model.eng_flexothermoelectric_resonance_energies[2]
    assert pytest.approx(fter2.dt_fter) == 0.00168
    assert fter2.sens_id == 330


def test_m333_eng_flexothermoelectric_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermoelectric Resonance Energy Aliases Test
/ENG/FLEXOTHERM_ELEC_RES_WORK/3
0.00030, 85
/ENG/EFLEXOTHERMOELECTRICRESONANCE/4
0.00035, 95
/ENG/FLEXOTHERMOELECTRIC_RESONANCE_DISSIPATION/5
0.00040, 105
/ENG/EM_FLEXOTHERMOELECTRIC_RESONANCE/6
0.00045, 115
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermoelectric_resonance_energies
    assert 4 in model.eng_flexothermoelectric_resonance_energies
    assert 5 in model.eng_flexothermoelectric_resonance_energies
    assert 6 in model.eng_flexothermoelectric_resonance_energies


def test_m333_eng_flexothermoelectric_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOELECTRIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m333_lagmul_franke_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{191:>10d}{192:>10d}{193:>10d}{1.45e7:>20.1f}{45:>10d}{2.5e-5:>20.6e}"
    c2 = f"{96.0:>20.4f}{89.5:>20.4f}{78.0:>20.4f}{34.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Franke Linkage Joint Fixed Format Test
2022 0
/FRANKE_LINKAGE_JOINT/155
Franke Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 155 in model.lagmul_franke_linkage_joints
    joint = model.lagmul_franke_linkage_joints[155]
    assert joint.node1 == 191
    assert joint.node2 == 192
    assert joint.node3 == 193
    assert pytest.approx(joint.stiff) == 1.45e7
    assert joint.skew_id == 45
    assert pytest.approx(joint.tol) == 2.5e-5
    assert pytest.approx(joint.link_len_a) == 96.0
    assert pytest.approx(joint.link_len_b) == 89.5
    assert pytest.approx(joint.twist_angle_alpha) == 78.0
    assert pytest.approx(joint.offset_distance_f) == 34.0


def test_m333_lagmul_franke_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Franke Linkage Joint Free Format Test
/LAGMUL/FRANKE_LINKAGE_JOINT/156
291, 292, 293, 8.5e6, 65, 3.8e-5
98.0, 91.0, 82.0, 36.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 156 in model.lagmul_franke_linkage_joints
    joint = model.lagmul_franke_linkage_joints[156]
    assert joint.node1 == 291
    assert joint.node2 == 292
    assert joint.node3 == 293
    assert pytest.approx(joint.stiff) == 8.5e6
    assert joint.skew_id == 65
    assert pytest.approx(joint.tol) == 3.8e-5
    assert pytest.approx(joint.link_len_a) == 98.0
    assert pytest.approx(joint.link_len_b) == 91.0
    assert pytest.approx(joint.twist_angle_alpha) == 82.0
    assert pytest.approx(joint.offset_distance_f) == 36.0


def test_m333_lagmul_franke_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Franke Linkage Joint Aliases Test
/LAGMUL/FRANKE_LINKAGE/157
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/FRANKE_LINKAGE/158
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/FRANKE_SPATIAL_MECHANISM/159
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/FRANKE_6R_MECHANISM/160
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/FRANKE_OVERCONSTRAINED_MECHANISM/161
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 157 in model.lagmul_franke_linkage_joints
    assert 158 in model.lagmul_franke_linkage_joints
    assert 159 in model.lagmul_franke_linkage_joints
    assert 160 in model.lagmul_franke_linkage_joints
    assert 161 in model.lagmul_franke_linkage_joints


def test_m333_lagmul_franke_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FRANKE_LINKAGE_JOINT/162
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m333_sensor_spring_total_lock_rate_fixed(tmp_path: Path):
    c1 = f"{951:>10d}{2.45e9:>20.1f}{0.0080:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Total Lock Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_LOCK_RATE/188
Spring Total Lock Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 188 in model.sensor_spring_total_lock_rates
    sensor = model.sensor_spring_total_lock_rates[188]
    assert sensor.spring_id == 951
    assert pytest.approx(sensor.jtot_lock_max) == 2.45e9
    assert pytest.approx(sensor.t_delay) == 0.0080
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_LOCK_RATE"


def test_m333_sensor_spring_total_lock_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Total Lock Rate Sensor Free Format Test
/SENSOR/SPRING_TOTAL_LOCK_RATE/189
952, 2.55e9, 0.0100
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 189 in model.sensor_spring_total_lock_rates
    sensor = model.sensor_spring_total_lock_rates[189]
    assert sensor.spring_id == 952
    assert pytest.approx(sensor.jtot_lock_max) == 2.55e9
    assert pytest.approx(sensor.t_delay) == 0.0100


def test_m333_sensor_spring_total_lock_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Total Lock Rate Sensor Aliases Test
/SENSOR/SPRING_TOT_LOCK_RATE/190
953, 3.0e9, 0.0020
/SENSOR/SPRING_RATE_LOCK_TOT/191
954, 3.0e9, 0.0020
/SENSOR/TOTAL_LOCK_RATE_SPRING/192
955, 3.0e9, 0.0020
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 190 in model.sensor_spring_total_lock_rates
    assert 191 in model.sensor_spring_total_lock_rates
    assert 192 in model.sensor_spring_total_lock_rates
    assert len(model.sensors) == 3


def test_m333_sensor_spring_total_lock_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TOTAL_LOCK_RATE/193
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
