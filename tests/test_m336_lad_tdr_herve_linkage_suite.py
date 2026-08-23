"""Tests for Milestone M336: LadTransverseDelaminationRate Failure Model, EngFlexoelectroacousticResonanceEnergy, HerveLinkageJoint, and SensorSpringTotalAngularLockRate."""

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


def test_m336_fail_lad_transverse_delamination_rate_fixed(tmp_path: Path):
    c1 = f"{182.0:>20.4f}{495.0:>20.4f}{38.0:>20.4f}{1.72:>20.4f}{0.965:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1520:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Delamination Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_DELAMINATION_RATE/1520
Ladeveze Transverse Delamination Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1520 in model.fail_ladtransversedelaminationrates
    ftdr = model.fail_ladtransversedelaminationrates[1520]
    assert pytest.approx(ftdr.sigma_tdr0) == 182.0
    assert pytest.approx(ftdr.sigma_tdrc) == 495.0
    assert pytest.approx(ftdr.gamma_tdr) == 38.0
    assert pytest.approx(ftdr.p_tdr) == 1.72
    assert pytest.approx(ftdr.d_tdr_max) == 0.965
    assert ftdr.ifail_sh == 1
    assert ftdr.ifail_so == 2
    assert ftdr.fail_id == 1520
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_DELAMINATION_RATE"


def test_m336_fail_lad_transverse_delamination_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Delamination Rate Free Format Test
/FAIL/LAD_TRANSVERSE_DELAMINATION_RATE/1521
195.0, 560.0, 42.0, 1.85, 0.940
1, 1
1521
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1521 in model.fail_ladtransversedelaminationrates
    ftdr = model.fail_ladtransversedelaminationrates[1521]
    assert pytest.approx(ftdr.sigma_tdr0) == 195.0
    assert pytest.approx(ftdr.sigma_tdrc) == 560.0
    assert pytest.approx(ftdr.gamma_tdr) == 42.0
    assert pytest.approx(ftdr.p_tdr) == 1.85
    assert pytest.approx(ftdr.d_tdr_max) == 0.940
    assert ftdr.fail_id == 1521


def test_m336_fail_lad_transverse_delamination_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Delamination Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_DELAMINATION_RATE/1522
145.0, 410.0, 28.0, 1.45, 0.92
1, 1
/FAIL/LAD_TDR/1523
145.0, 410.0, 28.0, 1.45, 0.92
1, 1
/FAIL/LAD_TDR_MODEL/1524
145.0, 410.0, 28.0, 1.45, 0.92
1, 1
/FAIL/LAD_TDR_LAW/1525
145.0, 410.0, 28.0, 1.45, 0.92
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_DELAMINATION/1526
145.0, 410.0, 28.0, 1.45, 0.92
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1522 in model.fail_ladtransversedelaminationrates
    assert 1523 in model.fail_ladtransversedelaminationrates
    assert 1524 in model.fail_ladtransversedelaminationrates
    assert 1525 in model.fail_ladtransversedelaminationrates
    assert 1526 in model.fail_ladtransversedelaminationrates
    assert len(model.raw_fails) == 5


def test_m336_fail_lad_transverse_delamination_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_TRANSVERSE_DELAMINATION_RATE/1527
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m336_eng_flexoelectroacoustic_resonance_energy(tmp_path: Path):
    c1 = f"{0.00148:>20.6f}{205:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexoelectroacoustic Resonance Energy Fixed and Free Format Test
2022 0
/ENG/FLEXOELECTROACOUSTIC_RESONANCE_ENERGY/1
Fixed Flexoelectroacoustic Resonance Energy Output
{c1}
/ENG/FLEXOELECTROACOUSTIC_RESONANCE_ENERGY/2
Free Flexoelectroacoustic Resonance Energy Output
0.00192, 360
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexoelectroacoustic_resonance_energies
    assert 2 in model.eng_flexoelectroacoustic_resonance_energies
    fear1 = model.eng_flexoelectroacoustic_resonance_energies[1]
    assert pytest.approx(fear1.dt_fear) == 0.00148
    assert fear1.sens_id == 205
    fear2 = model.eng_flexoelectroacoustic_resonance_energies[2]
    assert pytest.approx(fear2.dt_fear) == 0.00192
    assert fear2.sens_id == 360


def test_m336_eng_flexoelectroacoustic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexoelectroacoustic Resonance Energy Aliases Test
/ENG/FLEXOELEC_AC_RES_WORK/3
0.00042, 98
/ENG/EFLEXOELECTROACOUSTICRESONANCE/4
0.00048, 108
/ENG/FLEXOELECTROACOUSTIC_RESONANCE_DISSIPATION/5
0.00052, 118
/ENG/EM_FLEXOELECTROACOUSTIC_RESONANCE/6
0.00058, 128
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexoelectroacoustic_resonance_energies
    assert 4 in model.eng_flexoelectroacoustic_resonance_energies
    assert 5 in model.eng_flexoelectroacoustic_resonance_energies
    assert 6 in model.eng_flexoelectroacoustic_resonance_energies


def test_m336_eng_flexoelectroacoustic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOELECTROACOUSTIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m336_lagmul_herve_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{221:>10d}{222:>10d}{223:>10d}{1.75e7:>20.1f}{55:>10d}{2.5e-5:>20.6e}"
    c2 = f"{108.0:>20.4f}{98.5:>20.4f}{85.0:>20.4f}{42.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Herve Linkage Joint Fixed Format Test
2022 0
/HERVE_LINKAGE_JOINT/185
Herve Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 185 in model.lagmul_herve_linkage_joints
    joint = model.lagmul_herve_linkage_joints[185]
    assert joint.node1 == 221
    assert joint.node2 == 222
    assert joint.node3 == 223
    assert pytest.approx(joint.stiff) == 1.75e7
    assert joint.skew_id == 55
    assert pytest.approx(joint.tol) == 2.5e-5
    assert pytest.approx(joint.link_len_a) == 108.0
    assert pytest.approx(joint.link_len_b) == 98.5
    assert pytest.approx(joint.twist_angle_alpha) == 85.0
    assert pytest.approx(joint.offset_distance_f) == 42.0


def test_m336_lagmul_herve_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Herve Linkage Joint Free Format Test
/LAGMUL/HERVE_LINKAGE_JOINT/186
321, 322, 323, 8.5e6, 75, 3.5e-5
110.0, 99.0, 88.0, 44.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 186 in model.lagmul_herve_linkage_joints
    joint = model.lagmul_herve_linkage_joints[186]
    assert joint.node1 == 321
    assert joint.node2 == 322
    assert joint.node3 == 323
    assert pytest.approx(joint.stiff) == 8.5e6
    assert joint.skew_id == 75
    assert pytest.approx(joint.tol) == 3.5e-5
    assert pytest.approx(joint.link_len_a) == 110.0
    assert pytest.approx(joint.link_len_b) == 99.0
    assert pytest.approx(joint.twist_angle_alpha) == 88.0
    assert pytest.approx(joint.offset_distance_f) == 44.0


def test_m336_lagmul_herve_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Herve Linkage Joint Aliases Test
/LAGMUL/HERVE_LINKAGE/187
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/HERVE_LINKAGE/188
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/HERVE_SPATIAL_MECHANISM/189
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/HERVE_6R_MECHANISM/190
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/HERVE_ISOCLINE_MECHANISM/191
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 187 in model.lagmul_herve_linkage_joints
    assert 188 in model.lagmul_herve_linkage_joints
    assert 189 in model.lagmul_herve_linkage_joints
    assert 190 in model.lagmul_herve_linkage_joints
    assert 191 in model.lagmul_herve_linkage_joints


def test_m336_lagmul_herve_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/HERVE_LINKAGE_JOINT/192
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m336_sensor_spring_total_angular_lock_rate_fixed(tmp_path: Path):
    c1 = f"{981:>10d}{3.15e9:>20.1f}{0.0095:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Total Angular Lock Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_LOCK_RATE/218
Spring Total Angular Lock Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 218 in model.sensor_spring_total_angular_lock_rates
    sensor = model.sensor_spring_total_angular_lock_rates[218]
    assert sensor.spring_id == 981
    assert pytest.approx(sensor.jtot_ang_lock_max) == 3.15e9
    assert pytest.approx(sensor.t_delay) == 0.0095
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_LOCK_RATE"


def test_m336_sensor_spring_total_angular_lock_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Total Angular Lock Rate Sensor Free Format Test
/SENSOR/SPRING_TOTAL_ANGULAR_LOCK_RATE/219
982, 3.25e9, 0.0120
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 219 in model.sensor_spring_total_angular_lock_rates
    sensor = model.sensor_spring_total_angular_lock_rates[219]
    assert sensor.spring_id == 982
    assert pytest.approx(sensor.jtot_ang_lock_max) == 3.25e9
    assert pytest.approx(sensor.t_delay) == 0.0120


def test_m336_sensor_spring_total_angular_lock_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Total Angular Lock Rate Sensor Aliases Test
/SENSOR/SPRING_TOT_ANG_LOCK_RATE/220
983, 3.6e9, 0.0035
/SENSOR/SPRING_RATE_LOCK_ANG_TOT/221
984, 3.6e9, 0.0035
/SENSOR/TOTAL_ANGULAR_LOCK_RATE_SPRING/222
985, 3.6e9, 0.0035
/SENSOR/SPRING_LOCK_ANG_TOT/223
986, 3.6e9, 0.0035
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 220 in model.sensor_spring_total_angular_lock_rates
    assert 221 in model.sensor_spring_total_angular_lock_rates
    assert 222 in model.sensor_spring_total_angular_lock_rates
    assert 223 in model.sensor_spring_total_angular_lock_rates
    assert len(model.sensors) == 4


def test_m336_sensor_spring_total_angular_lock_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TOTAL_ANGULAR_LOCK_RATE/224
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
