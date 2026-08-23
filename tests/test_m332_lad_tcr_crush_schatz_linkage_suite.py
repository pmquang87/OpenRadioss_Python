"""Tests for Milestone M332: LadTransverseCrushRate Failure Model, EngFlexothermomagneticResonanceEnergy, SchatzLinkageJoint, and SensorSpringTransverseLockRate."""

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


def test_m332_fail_lad_transverse_crush_rate_fixed(tmp_path: Path):
    c1 = f"{155.0:>20.4f}{435.0:>20.4f}{30.5:>20.4f}{1.52:>20.4f}{0.982:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1480:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Crush Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_CRUSH_RATE/1480
Ladeveze Dynamic Transverse Compressive Crush Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1480 in model.fail_ladtransversecrushrates
    ftcr = model.fail_ladtransversecrushrates[1480]
    assert pytest.approx(ftcr.sigma_tcr0) == 155.0
    assert pytest.approx(ftcr.sigma_tcrc) == 435.0
    assert pytest.approx(ftcr.gamma_tcr) == 30.5
    assert pytest.approx(ftcr.p_tcr) == 1.52
    assert pytest.approx(ftcr.d_tcr_max) == 0.982
    assert ftcr.ifail_sh == 1
    assert ftcr.ifail_so == 2
    assert ftcr.fail_id == 1480
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_CRUSH_RATE"


def test_m332_fail_lad_transverse_crush_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Crush Rate Free Format Test
/FAIL/LAD_TRANSVERSE_CRUSH_RATE/1481
168.0, 495.0, 34.0, 1.65, 0.965
1, 1
1481
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1481 in model.fail_ladtransversecrushrates
    ftcr = model.fail_ladtransversecrushrates[1481]
    assert pytest.approx(ftcr.sigma_tcr0) == 168.0
    assert pytest.approx(ftcr.sigma_tcrc) == 495.0
    assert pytest.approx(ftcr.gamma_tcr) == 34.0
    assert pytest.approx(ftcr.p_tcr) == 1.65
    assert pytest.approx(ftcr.d_tcr_max) == 0.965
    assert ftcr.fail_id == 1481


def test_m332_fail_lad_transverse_crush_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Crush Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_CRUSH_RATE/1482
125.0, 360.0, 22.0, 1.25, 0.94
1, 1
/FAIL/LAD_TRANS_CRUSH_RATE/1483
125.0, 360.0, 22.0, 1.25, 0.94
1, 1
/FAIL/LAD_TCR_CRUSH/1484
125.0, 360.0, 22.0, 1.25, 0.94
1, 1
/FAIL/LAD_TCR_CRUSH_MODEL/1485
125.0, 360.0, 22.0, 1.25, 0.94
1, 1
/FAIL/LAD_TCR_CRUSH_LAW/1486
125.0, 360.0, 22.0, 1.25, 0.94
1, 1
/FAIL/LADEVEZE_TRANSVERSE_CRUSHING_RATE/1487
125.0, 360.0, 22.0, 1.25, 0.94
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1482 in model.fail_ladtransversecrushrates
    assert 1483 in model.fail_ladtransversecrushrates
    assert 1484 in model.fail_ladtransversecrushrates
    assert 1485 in model.fail_ladtransversecrushrates
    assert 1486 in model.fail_ladtransversecrushrates
    assert 1487 in model.fail_ladtransversecrushrates
    assert len(model.raw_fails) == 6


def test_m332_fail_lad_transverse_crush_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_TRANSVERSE_CRUSH_RATE/1488
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m332_eng_flexothermomagnetic_resonance_energy(tmp_path: Path):
    c1 = f"{0.00122:>20.6f}{165:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermomagnetic Resonance Energy Fixed and Free Format Test
2022 0
/ENG/FLEXOTHERMOMAGNETIC_RESONANCE_ENERGY/1
Fixed Flexothermomagnetic Resonance Energy Output
{c1}
/ENG/FLEXOTHERMOMAGNETIC_RESONANCE_ENERGY/2
Free Flexothermomagnetic Resonance Energy Output
0.00162, 320
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermomagnetic_resonance_energies
    assert 2 in model.eng_flexothermomagnetic_resonance_energies
    ftmr1 = model.eng_flexothermomagnetic_resonance_energies[1]
    assert pytest.approx(ftmr1.dt_ftmr) == 0.00122
    assert ftmr1.sens_id == 165
    ftmr2 = model.eng_flexothermomagnetic_resonance_energies[2]
    assert pytest.approx(ftmr2.dt_ftmr) == 0.00162
    assert ftmr2.sens_id == 320


def test_m332_eng_flexothermomagnetic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermomagnetic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_MAG_RES_WORK/3
0.00028, 82
/ENG/EFLEXOTHERMOMAGNETICRESONANCE/4
0.00032, 92
/ENG/FLEXOTHERMOMAGNETIC_RESONANCE_DISSIPATION/5
0.00038, 102
/ENG/EM_FLEXOTHERMOMAGNETIC_RESONANCE/6
0.00042, 112
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermomagnetic_resonance_energies
    assert 4 in model.eng_flexothermomagnetic_resonance_energies
    assert 5 in model.eng_flexothermomagnetic_resonance_energies
    assert 6 in model.eng_flexothermomagnetic_resonance_energies


def test_m332_eng_flexothermomagnetic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOMAGNETIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m332_lagmul_schatz_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{181:>10d}{182:>10d}{183:>10d}{1.35e7:>20.1f}{42:>10d}{2.8e-5:>20.6e}"
    c2 = f"{92.0:>20.4f}{86.5:>20.4f}{76.0:>20.4f}{32.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Schatz Linkage Joint Fixed Format Test
2022 0
/SCHATZ_LINKAGE_JOINT/145
Schatz Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 145 in model.lagmul_schatz_linkage_joints
    joint = model.lagmul_schatz_linkage_joints[145]
    assert joint.node1 == 181
    assert joint.node2 == 182
    assert joint.node3 == 183
    assert pytest.approx(joint.stiff) == 1.35e7
    assert joint.skew_id == 42
    assert pytest.approx(joint.tol) == 2.8e-5
    assert pytest.approx(joint.link_len_a) == 92.0
    assert pytest.approx(joint.link_len_b) == 86.5
    assert pytest.approx(joint.twist_angle_alpha) == 76.0
    assert pytest.approx(joint.offset_distance_f) == 32.0


def test_m332_lagmul_schatz_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Schatz Linkage Joint Free Format Test
/LAGMUL/SCHATZ_LINKAGE_JOINT/146
281, 282, 283, 8.8e6, 62, 4.2e-5
95.0, 89.0, 80.0, 34.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 146 in model.lagmul_schatz_linkage_joints
    joint = model.lagmul_schatz_linkage_joints[146]
    assert joint.node1 == 281
    assert joint.node2 == 282
    assert joint.node3 == 283
    assert pytest.approx(joint.stiff) == 8.8e6
    assert joint.skew_id == 62
    assert pytest.approx(joint.tol) == 4.2e-5
    assert pytest.approx(joint.link_len_a) == 95.0
    assert pytest.approx(joint.link_len_b) == 89.0
    assert pytest.approx(joint.twist_angle_alpha) == 80.0
    assert pytest.approx(joint.offset_distance_f) == 34.0


def test_m332_lagmul_schatz_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Schatz Linkage Joint Aliases Test
/LAGMUL/SCHATZ_LINKAGE/147
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/SCHATZ_LINKAGE/148
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/SCHATZ_SPATIAL_MECHANISM/149
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/SCHATZ_6R_MECHANISM/150
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/TURBULA_MECHANISM/151
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 147 in model.lagmul_schatz_linkage_joints
    assert 148 in model.lagmul_schatz_linkage_joints
    assert 149 in model.lagmul_schatz_linkage_joints
    assert 150 in model.lagmul_schatz_linkage_joints
    assert 151 in model.lagmul_schatz_linkage_joints


def test_m332_lagmul_schatz_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SCHATZ_LINKAGE_JOINT/152
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m332_sensor_spring_transverse_lock_rate_fixed(tmp_path: Path):
    c1 = f"{941:>10d}{2.25e9:>20.1f}{0.0075:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Transverse Lock Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_LOCK_RATE/178
Spring Transverse Lock Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 178 in model.sensor_spring_transverse_lock_rates
    sensor = model.sensor_spring_transverse_lock_rates[178]
    assert sensor.spring_id == 941
    assert pytest.approx(sensor.jtrans_lock_max) == 2.25e9
    assert pytest.approx(sensor.t_delay) == 0.0075
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_LOCK_RATE"


def test_m332_sensor_spring_transverse_lock_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Transverse Lock Rate Sensor Free Format Test
/SENSOR/SPRING_TRANSVERSE_LOCK_RATE/179
942, 2.35e9, 0.0095
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 179 in model.sensor_spring_transverse_lock_rates
    sensor = model.sensor_spring_transverse_lock_rates[179]
    assert sensor.spring_id == 942
    assert pytest.approx(sensor.jtrans_lock_max) == 2.35e9
    assert pytest.approx(sensor.t_delay) == 0.0095


def test_m332_sensor_spring_transverse_lock_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Transverse Lock Rate Sensor Aliases Test
/SENSOR/SPRING_TRANS_LOCK_RATE/180
943, 2.9e9, 0.0015
/SENSOR/SPRING_RATE_LOCK_TRANS/181
944, 2.9e9, 0.0015
/SENSOR/TRANSVERSE_LOCK_RATE_SPRING/182
945, 2.9e9, 0.0015
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 180 in model.sensor_spring_transverse_lock_rates
    assert 181 in model.sensor_spring_transverse_lock_rates
    assert 182 in model.sensor_spring_transverse_lock_rates
    assert len(model.sensors) == 3


def test_m332_sensor_spring_transverse_lock_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TRANSVERSE_LOCK_RATE/183
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
