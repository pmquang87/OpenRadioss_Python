"""Tests for Milestone M339: LadTransverseMicrobucklingRate Failure Model, EngFlexothermoelectroacousticResonanceEnergy, BakerLineLinkageJoint, and SensorSpringTotalDropRate."""

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


def test_m339_fail_lad_transverse_microbuckling_rate_fixed(tmp_path: Path):
    c1 = f"{192.0:>20.4f}{530.0:>20.4f}{42.5:>20.4f}{1.82:>20.4f}{0.950:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1550:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Microbuckling Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_MICROBUCKLING_RATE/1550
Ladeveze Transverse Microbuckling Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1550 in model.fail_ladtransversemicrobucklingrates
    ftmbr = model.fail_ladtransversemicrobucklingrates[1550]
    assert pytest.approx(ftmbr.sigma_tmbr0) == 192.0
    assert pytest.approx(ftmbr.sigma_tmbrc) == 530.0
    assert pytest.approx(ftmbr.gamma_tmbr) == 42.5
    assert pytest.approx(ftmbr.p_tmbr) == 1.82
    assert pytest.approx(ftmbr.d_tmbr_max) == 0.950
    assert ftmbr.ifail_sh == 1
    assert ftmbr.ifail_so == 2
    assert ftmbr.fail_id == 1550
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_MICROBUCKLING_RATE"


def test_m339_fail_lad_transverse_microbuckling_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Microbuckling Rate Free Format Test
/FAIL/LAD_TRANSVERSE_MICROBUCKLING_RATE/1551
210.0, 605.0, 48.0, 1.98, 0.925
1, 1
1551
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1551 in model.fail_ladtransversemicrobucklingrates
    ftmbr = model.fail_ladtransversemicrobucklingrates[1551]
    assert pytest.approx(ftmbr.sigma_tmbr0) == 210.0
    assert pytest.approx(ftmbr.sigma_tmbrc) == 605.0
    assert pytest.approx(ftmbr.gamma_tmbr) == 48.0
    assert pytest.approx(ftmbr.p_tmbr) == 1.98
    assert pytest.approx(ftmbr.d_tmbr_max) == 0.925
    assert ftmbr.fail_id == 1551


def test_m339_fail_lad_transverse_microbuckling_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Microbuckling Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_MICROBUCKLING_RATE/1552
160.0, 445.0, 34.0, 1.60, 0.94
1, 1
/FAIL/LAD_TMBR/1553
160.0, 445.0, 34.0, 1.60, 0.94
1, 1
/FAIL/LAD_TMBR_MODEL/1554
160.0, 445.0, 34.0, 1.60, 0.94
1, 1
/FAIL/LAD_TMBR_LAW/1555
160.0, 445.0, 34.0, 1.60, 0.94
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_MICROBUCKLING/1556
160.0, 445.0, 34.0, 1.60, 0.94
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1552 in model.fail_ladtransversemicrobucklingrates
    assert 1553 in model.fail_ladtransversemicrobucklingrates
    assert 1554 in model.fail_ladtransversemicrobucklingrates
    assert 1555 in model.fail_ladtransversemicrobucklingrates
    assert 1556 in model.fail_ladtransversemicrobucklingrates
    assert len(model.raw_fails) == 5


def test_m339_fail_lad_transverse_microbuckling_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_TRANSVERSE_MICROBUCKLING_RATE/1557
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m339_eng_flexothermoelectroacoustic_resonance_energy(tmp_path: Path):
    c1 = f"{0.00158:>20.6f}{235:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermoelectroacoustic Resonance Energy Fixed and Free Format Test
2022 0
/ENG/FLEXOTHERMOELECTROACOUSTIC_RESONANCE_ENERGY/1
Fixed Flexothermoelectroacoustic Resonance Energy Output
{c1}
/ENG/FLEXOTHERMOELECTROACOUSTIC_RESONANCE_ENERGY/2
Free Flexothermoelectroacoustic Resonance Energy Output
0.00212, 390
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermoelectroacoustic_resonance_energies
    assert 2 in model.eng_flexothermoelectroacoustic_resonance_energies
    ftear1 = model.eng_flexothermoelectroacoustic_resonance_energies[1]
    assert pytest.approx(ftear1.dt_ftear) == 0.00158
    assert ftear1.sens_id == 235
    ftear2 = model.eng_flexothermoelectroacoustic_resonance_energies[2]
    assert pytest.approx(ftear2.dt_ftear) == 0.00212
    assert ftear2.sens_id == 390


def test_m339_eng_flexothermoelectroacoustic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermoelectroacoustic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_EA_RES_WORK/3
0.00050, 108
/ENG/EFLEXOTHERMOELECTROACOUSTICRESONANCE/4
0.00055, 118
/ENG/FLEXOTHERMOELECTROACOUSTIC_RESONANCE_DISSIPATION/5
0.00060, 128
/ENG/EM_FLEXOTHERMOELECTROACOUSTIC_RESONANCE/6
0.00065, 138
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermoelectroacoustic_resonance_energies
    assert 4 in model.eng_flexothermoelectroacoustic_resonance_energies
    assert 5 in model.eng_flexothermoelectroacoustic_resonance_energies
    assert 6 in model.eng_flexothermoelectroacoustic_resonance_energies


def test_m339_eng_flexothermoelectroacoustic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOELECTROACOUSTIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m339_lagmul_baker_line_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{251:>10d}{252:>10d}{253:>10d}{2.05e7:>20.1f}{65:>10d}{3.2e-5:>20.6e}"
    c2 = f"{118.0:>20.4f}{108.0:>20.4f}{95.0:>20.4f}{52.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Baker Line Linkage Joint Fixed Format Test
2022 0
/BAKER_LINE_LINKAGE_JOINT/215
Baker Line Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 215 in model.lagmul_baker_line_linkage_joints
    joint = model.lagmul_baker_line_linkage_joints[215]
    assert joint.node1 == 251
    assert joint.node2 == 252
    assert joint.node3 == 253
    assert pytest.approx(joint.stiff) == 2.05e7
    assert joint.skew_id == 65
    assert pytest.approx(joint.tol) == 3.2e-5
    assert pytest.approx(joint.link_len_a) == 118.0
    assert pytest.approx(joint.link_len_b) == 108.0
    assert pytest.approx(joint.twist_angle_alpha) == 95.0
    assert pytest.approx(joint.offset_distance_f) == 52.0


def test_m339_lagmul_baker_line_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Baker Line Linkage Joint Free Format Test
/LAGMUL/BAKER_LINE_LINKAGE_JOINT/216
351, 352, 353, 9.8e6, 85, 4.2e-5
120.0, 110.0, 96.0, 54.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 216 in model.lagmul_baker_line_linkage_joints
    joint = model.lagmul_baker_line_linkage_joints[216]
    assert joint.node1 == 351
    assert joint.node2 == 352
    assert joint.node3 == 353
    assert pytest.approx(joint.stiff) == 9.8e6
    assert joint.skew_id == 85
    assert pytest.approx(joint.tol) == 4.2e-5
    assert pytest.approx(joint.link_len_a) == 120.0
    assert pytest.approx(joint.link_len_b) == 110.0
    assert pytest.approx(joint.twist_angle_alpha) == 96.0
    assert pytest.approx(joint.offset_distance_f) == 54.0


def test_m339_lagmul_baker_line_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Baker Line Linkage Joint Aliases Test
/LAGMUL/BAKER_LINE_LINKAGE/217
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/BAKER_LINE_LINKAGE/218
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/BAKER_LINE_SYMMETRIC_MECHANISM/219
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/BAKER_LINE_6R_MECHANISM/220
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/BAKER_LINE_OVERCONSTRAINED_MECHANISM/221
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 217 in model.lagmul_baker_line_linkage_joints
    assert 218 in model.lagmul_baker_line_linkage_joints
    assert 219 in model.lagmul_baker_line_linkage_joints
    assert 220 in model.lagmul_baker_line_linkage_joints
    assert 221 in model.lagmul_baker_line_linkage_joints


def test_m339_lagmul_baker_line_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/BAKER_LINE_LINKAGE_JOINT/222
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m339_sensor_spring_total_drop_rate_fixed(tmp_path: Path):
    c1 = f"{998:>10d}{3.85e9:>20.1f}{0.0115:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Total Drop Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_DROP_RATE/248
Spring Total Drop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 248 in model.sensor_spring_total_drop_rates
    sensor = model.sensor_spring_total_drop_rates[248]
    assert sensor.spring_id == 998
    assert pytest.approx(sensor.jtot_drop_max) == 3.85e9
    assert pytest.approx(sensor.t_delay) == 0.0115
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_DROP_RATE"


def test_m339_sensor_spring_total_drop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Total Drop Rate Sensor Free Format Test
/SENSOR/SPRING_TOTAL_DROP_RATE/249
999, 3.95e9, 0.0140
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 249 in model.sensor_spring_total_drop_rates
    sensor = model.sensor_spring_total_drop_rates[249]
    assert sensor.spring_id == 999
    assert pytest.approx(sensor.jtot_drop_max) == 3.95e9
    assert pytest.approx(sensor.t_delay) == 0.0140


def test_m339_sensor_spring_total_drop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Total Drop Rate Sensor Aliases Test
/SENSOR/SPRING_TOT_DROP_RATE/250
1001, 4.1e9, 0.0050
/SENSOR/SPRING_RATE_DROP_TOT/251
1002, 4.1e9, 0.0050
/SENSOR/TOTAL_DROP_RATE_SPRING/252
1003, 4.1e9, 0.0050
/SENSOR/SPRING_DROP_TOT/253
1004, 4.1e9, 0.0050
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 250 in model.sensor_spring_total_drop_rates
    assert 251 in model.sensor_spring_total_drop_rates
    assert 252 in model.sensor_spring_total_drop_rates
    assert 253 in model.sensor_spring_total_drop_rates
    assert len(model.sensors) == 4


def test_m339_sensor_spring_total_drop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TOTAL_DROP_RATE/254
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
