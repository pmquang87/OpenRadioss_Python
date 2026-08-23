"""Tests for Milestone M338: LadDynamicMicrobucklingRate Failure Model, EngFlexothermoelectromagneticResonanceEnergy, HuntLinkageJoint, and SensorSpringTransverseDropRate."""

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


def test_m338_fail_lad_dynamic_microbuckling_rate_fixed(tmp_path: Path):
    c1 = f"{188.0:>20.4f}{520.0:>20.4f}{41.0:>20.4f}{1.80:>20.4f}{0.955:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1540:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Microbuckling Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_MICROBUCKLING_RATE/1540
Ladeveze Dynamic Microbuckling Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1540 in model.fail_laddynamicmicrobucklingrates
    fdmbr = model.fail_laddynamicmicrobucklingrates[1540]
    assert pytest.approx(fdmbr.sigma_dmbr0) == 188.0
    assert pytest.approx(fdmbr.sigma_dmbrc) == 520.0
    assert pytest.approx(fdmbr.gamma_dmbr) == 41.0
    assert pytest.approx(fdmbr.p_dmbr) == 1.80
    assert pytest.approx(fdmbr.d_dmbr_max) == 0.955
    assert fdmbr.ifail_sh == 1
    assert fdmbr.ifail_so == 2
    assert fdmbr.fail_id == 1540
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_MICROBUCKLING_RATE"


def test_m338_fail_lad_dynamic_microbuckling_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Microbuckling Rate Free Format Test
/FAIL/LAD_DYNAMIC_MICROBUCKLING_RATE/1541
205.0, 590.0, 46.0, 1.95, 0.930
1, 1
1541
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1541 in model.fail_laddynamicmicrobucklingrates
    fdmbr = model.fail_laddynamicmicrobucklingrates[1541]
    assert pytest.approx(fdmbr.sigma_dmbr0) == 205.0
    assert pytest.approx(fdmbr.sigma_dmbrc) == 590.0
    assert pytest.approx(fdmbr.gamma_dmbr) == 46.0
    assert pytest.approx(fdmbr.p_dmbr) == 1.95
    assert pytest.approx(fdmbr.d_dmbr_max) == 0.930
    assert fdmbr.fail_id == 1541


def test_m338_fail_lad_dynamic_microbuckling_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Microbuckling Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_MICROBUCKLING_RATE/1542
155.0, 435.0, 32.0, 1.55, 0.935
1, 1
/FAIL/LAD_DMBR/1543
155.0, 435.0, 32.0, 1.55, 0.935
1, 1
/FAIL/LAD_DMBR_MODEL/1544
155.0, 435.0, 32.0, 1.55, 0.935
1, 1
/FAIL/LAD_DMBR_LAW/1545
155.0, 435.0, 32.0, 1.55, 0.935
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_MICROBUCKLING/1546
155.0, 435.0, 32.0, 1.55, 0.935
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1542 in model.fail_laddynamicmicrobucklingrates
    assert 1543 in model.fail_laddynamicmicrobucklingrates
    assert 1544 in model.fail_laddynamicmicrobucklingrates
    assert 1545 in model.fail_laddynamicmicrobucklingrates
    assert 1546 in model.fail_laddynamicmicrobucklingrates
    assert len(model.raw_fails) == 5


def test_m338_fail_lad_dynamic_microbuckling_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_DYNAMIC_MICROBUCKLING_RATE/1547
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m338_eng_flexothermoelectromagnetic_resonance_energy(tmp_path: Path):
    c1 = f"{0.00155:>20.6f}{225:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermoelectromagnetic Resonance Energy Fixed and Free Format Test
2022 0
/ENG/FLEXOTHERMOELECTROMAGNETIC_RESONANCE_ENERGY/1
Fixed Flexothermoelectromagnetic Resonance Energy Output
{c1}
/ENG/FLEXOTHERMOELECTROMAGNETIC_RESONANCE_ENERGY/2
Free Flexothermoelectromagnetic Resonance Energy Output
0.00205, 380
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermoelectromagnetic_resonance_energies
    assert 2 in model.eng_flexothermoelectromagnetic_resonance_energies
    ftemr1 = model.eng_flexothermoelectromagnetic_resonance_energies[1]
    assert pytest.approx(ftemr1.dt_ftemr) == 0.00155
    assert ftemr1.sens_id == 225
    ftemr2 = model.eng_flexothermoelectromagnetic_resonance_energies[2]
    assert pytest.approx(ftemr2.dt_ftemr) == 0.00205
    assert ftemr2.sens_id == 380


def test_m338_eng_flexothermoelectromagnetic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermoelectromagnetic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_EM_RES_WORK/3
0.00048, 105
/ENG/EFLEXOTHERMOELECTROMAGNETICRESONANCE/4
0.00052, 115
/ENG/FLEXOTHERMOELECTROMAGNETIC_RESONANCE_DISSIPATION/5
0.00058, 125
/ENG/EM_FLEXOTHERMOELECTROMAGNETIC_RESONANCE/6
0.00062, 135
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermoelectromagnetic_resonance_energies
    assert 4 in model.eng_flexothermoelectromagnetic_resonance_energies
    assert 5 in model.eng_flexothermoelectromagnetic_resonance_energies
    assert 6 in model.eng_flexothermoelectromagnetic_resonance_energies


def test_m338_eng_flexothermoelectromagnetic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOELECTROMAGNETIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m338_lagmul_hunt_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{241:>10d}{242:>10d}{243:>10d}{1.95e7:>20.1f}{62:>10d}{3.0e-5:>20.6e}"
    c2 = f"{115.0:>20.4f}{105.0:>20.4f}{92.0:>20.4f}{48.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Hunt Linkage Joint Fixed Format Test
2022 0
/HUNT_LINKAGE_JOINT/205
Hunt Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 205 in model.lagmul_hunt_linkage_joints
    joint = model.lagmul_hunt_linkage_joints[205]
    assert joint.node1 == 241
    assert joint.node2 == 242
    assert joint.node3 == 243
    assert pytest.approx(joint.stiff) == 1.95e7
    assert joint.skew_id == 62
    assert pytest.approx(joint.tol) == 3.0e-5
    assert pytest.approx(joint.link_len_a) == 115.0
    assert pytest.approx(joint.link_len_b) == 105.0
    assert pytest.approx(joint.twist_angle_alpha) == 92.0
    assert pytest.approx(joint.offset_distance_f) == 48.0


def test_m338_lagmul_hunt_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Hunt Linkage Joint Free Format Test
/LAGMUL/HUNT_LINKAGE_JOINT/206
341, 342, 343, 9.5e6, 82, 4.0e-5
118.0, 108.0, 94.0, 50.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 206 in model.lagmul_hunt_linkage_joints
    joint = model.lagmul_hunt_linkage_joints[206]
    assert joint.node1 == 341
    assert joint.node2 == 342
    assert joint.node3 == 343
    assert pytest.approx(joint.stiff) == 9.5e6
    assert joint.skew_id == 82
    assert pytest.approx(joint.tol) == 4.0e-5
    assert pytest.approx(joint.link_len_a) == 118.0
    assert pytest.approx(joint.link_len_b) == 108.0
    assert pytest.approx(joint.twist_angle_alpha) == 94.0
    assert pytest.approx(joint.offset_distance_f) == 50.0


def test_m338_lagmul_hunt_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Hunt Linkage Joint Aliases Test
/LAGMUL/HUNT_LINKAGE/207
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/HUNT_LINKAGE/208
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/HUNT_SPATIAL_MECHANISM/209
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/HUNT_6R_MECHANISM/210
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/HUNT_SCREW_MECHANISM/211
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 207 in model.lagmul_hunt_linkage_joints
    assert 208 in model.lagmul_hunt_linkage_joints
    assert 209 in model.lagmul_hunt_linkage_joints
    assert 210 in model.lagmul_hunt_linkage_joints
    assert 211 in model.lagmul_hunt_linkage_joints


def test_m338_lagmul_hunt_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/HUNT_LINKAGE_JOINT/212
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m338_sensor_spring_transverse_drop_rate_fixed(tmp_path: Path):
    c1 = f"{995:>10d}{3.65e9:>20.1f}{0.0105:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Transverse Drop Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_DROP_RATE/238
Spring Transverse Drop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 238 in model.sensor_spring_transverse_drop_rates
    sensor = model.sensor_spring_transverse_drop_rates[238]
    assert sensor.spring_id == 995
    assert pytest.approx(sensor.jtrans_drop_max) == 3.65e9
    assert pytest.approx(sensor.t_delay) == 0.0105
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_DROP_RATE"


def test_m338_sensor_spring_transverse_drop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Transverse Drop Rate Sensor Free Format Test
/SENSOR/SPRING_TRANSVERSE_DROP_RATE/239
996, 3.75e9, 0.0130
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 239 in model.sensor_spring_transverse_drop_rates
    sensor = model.sensor_spring_transverse_drop_rates[239]
    assert sensor.spring_id == 996
    assert pytest.approx(sensor.jtrans_drop_max) == 3.75e9
    assert pytest.approx(sensor.t_delay) == 0.0130


def test_m338_sensor_spring_transverse_drop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Transverse Drop Rate Sensor Aliases Test
/SENSOR/SPRING_TRANS_DROP_RATE/240
997, 3.9e9, 0.0045
/SENSOR/SPRING_RATE_DROP_TRANS/241
998, 3.9e9, 0.0045
/SENSOR/TRANSVERSE_DROP_RATE_SPRING/242
999, 3.9e9, 0.0045
/SENSOR/SPRING_DROP_TRANS/243
1000, 3.9e9, 0.0045
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 240 in model.sensor_spring_transverse_drop_rates
    assert 241 in model.sensor_spring_transverse_drop_rates
    assert 242 in model.sensor_spring_transverse_drop_rates
    assert 243 in model.sensor_spring_transverse_drop_rates
    assert len(model.sensors) == 4


def test_m338_sensor_spring_transverse_drop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TRANSVERSE_DROP_RATE/244
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
