"""Tests for Milestone M335: LadDynamicDelaminationRate Failure Model, EngFlexoelectromagneticResonanceEnergy, BorelLinkageJoint, and SensorSpringBendingLockRate."""

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


def test_m335_fail_lad_dynamic_delamination_rate_fixed(tmp_path: Path):
    c1 = f"{178.0:>20.4f}{480.0:>20.4f}{36.5:>20.4f}{1.66:>20.4f}{0.970:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1510:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Delamination Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_DELAMINATION_RATE/1510
Ladeveze Dynamic Delamination Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1510 in model.fail_laddynamicdelaminationrates
    fddr = model.fail_laddynamicdelaminationrates[1510]
    assert pytest.approx(fddr.sigma_ddr0) == 178.0
    assert pytest.approx(fddr.sigma_ddrc) == 480.0
    assert pytest.approx(fddr.gamma_ddr) == 36.5
    assert pytest.approx(fddr.p_ddr) == 1.66
    assert pytest.approx(fddr.d_ddr_max) == 0.970
    assert fddr.ifail_sh == 1
    assert fddr.ifail_so == 2
    assert fddr.fail_id == 1510
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_DELAMINATION_RATE"


def test_m335_fail_lad_dynamic_delamination_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Delamination Rate Free Format Test
/FAIL/LAD_DYNAMIC_DELAMINATION_RATE/1511
190.0, 540.0, 40.0, 1.80, 0.950
1, 1
1511
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1511 in model.fail_laddynamicdelaminationrates
    fddr = model.fail_laddynamicdelaminationrates[1511]
    assert pytest.approx(fddr.sigma_ddr0) == 190.0
    assert pytest.approx(fddr.sigma_ddrc) == 540.0
    assert pytest.approx(fddr.gamma_ddr) == 40.0
    assert pytest.approx(fddr.p_ddr) == 1.80
    assert pytest.approx(fddr.d_ddr_max) == 0.950
    assert fddr.fail_id == 1511


def test_m335_fail_lad_dynamic_delamination_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Delamination Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_DELAMINATION_RATE/1512
140.0, 390.0, 26.0, 1.40, 0.91
1, 1
/FAIL/LAD_DDR/1513
140.0, 390.0, 26.0, 1.40, 0.91
1, 1
/FAIL/LAD_DDR_MODEL/1514
140.0, 390.0, 26.0, 1.40, 0.91
1, 1
/FAIL/LAD_DDR_LAW/1515
140.0, 390.0, 26.0, 1.40, 0.91
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_DELAMINATION/1516
140.0, 390.0, 26.0, 1.40, 0.91
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1512 in model.fail_laddynamicdelaminationrates
    assert 1513 in model.fail_laddynamicdelaminationrates
    assert 1514 in model.fail_laddynamicdelaminationrates
    assert 1515 in model.fail_laddynamicdelaminationrates
    assert 1516 in model.fail_laddynamicdelaminationrates
    assert len(model.raw_fails) == 5


def test_m335_fail_lad_dynamic_delamination_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_DYNAMIC_DELAMINATION_RATE/1517
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m335_eng_flexoelectromagnetic_resonance_energy(tmp_path: Path):
    c1 = f"{0.00142:>20.6f}{195:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexoelectromagnetic Resonance Energy Fixed and Free Format Test
2022 0
/ENG/FLEXOELECTROMAGNETIC_RESONANCE_ENERGY/1
Fixed Flexoelectromagnetic Resonance Energy Output
{c1}
/ENG/FLEXOELECTROMAGNETIC_RESONANCE_ENERGY/2
Free Flexoelectromagnetic Resonance Energy Output
0.00182, 350
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexoelectromagnetic_resonance_energies
    assert 2 in model.eng_flexoelectromagnetic_resonance_energies
    femr1 = model.eng_flexoelectromagnetic_resonance_energies[1]
    assert pytest.approx(femr1.dt_femr) == 0.00142
    assert femr1.sens_id == 195
    femr2 = model.eng_flexoelectromagnetic_resonance_energies[2]
    assert pytest.approx(femr2.dt_femr) == 0.00182
    assert femr2.sens_id == 350


def test_m335_eng_flexoelectromagnetic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexoelectromagnetic Resonance Energy Aliases Test
/ENG/FLEXOELEC_MAG_RES_WORK/3
0.00040, 95
/ENG/EFLEXOELECTROMAGNETICRESONANCE/4
0.00045, 105
/ENG/FLEXOELECTROMAGNETIC_RESONANCE_DISSIPATION/5
0.00050, 115
/ENG/EM_FLEXOELECTROMAGNETIC_RESONANCE/6
0.00055, 125
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexoelectromagnetic_resonance_energies
    assert 4 in model.eng_flexoelectromagnetic_resonance_energies
    assert 5 in model.eng_flexoelectromagnetic_resonance_energies
    assert 6 in model.eng_flexoelectromagnetic_resonance_energies


def test_m335_eng_flexoelectromagnetic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOELECTROMAGNETIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m335_lagmul_borel_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{211:>10d}{212:>10d}{213:>10d}{1.65e7:>20.1f}{52:>10d}{2.0e-5:>20.6e}"
    c2 = f"{102.0:>20.4f}{95.5:>20.4f}{82.0:>20.4f}{38.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Borel Linkage Joint Fixed Format Test
2022 0
/BOREL_LINKAGE_JOINT/175
Borel Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 175 in model.lagmul_borel_linkage_joints
    joint = model.lagmul_borel_linkage_joints[175]
    assert joint.node1 == 211
    assert joint.node2 == 212
    assert joint.node3 == 213
    assert pytest.approx(joint.stiff) == 1.65e7
    assert joint.skew_id == 52
    assert pytest.approx(joint.tol) == 2.0e-5
    assert pytest.approx(joint.link_len_a) == 102.0
    assert pytest.approx(joint.link_len_b) == 95.5
    assert pytest.approx(joint.twist_angle_alpha) == 82.0
    assert pytest.approx(joint.offset_distance_f) == 38.0


def test_m335_lagmul_borel_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Borel Linkage Joint Free Format Test
/LAGMUL/BOREL_LINKAGE_JOINT/176
311, 312, 313, 8.0e6, 72, 3.2e-5
105.0, 96.0, 86.0, 40.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 176 in model.lagmul_borel_linkage_joints
    joint = model.lagmul_borel_linkage_joints[176]
    assert joint.node1 == 311
    assert joint.node2 == 312
    assert joint.node3 == 313
    assert pytest.approx(joint.stiff) == 8.0e6
    assert joint.skew_id == 72
    assert pytest.approx(joint.tol) == 3.2e-5
    assert pytest.approx(joint.link_len_a) == 105.0
    assert pytest.approx(joint.link_len_b) == 96.0
    assert pytest.approx(joint.twist_angle_alpha) == 86.0
    assert pytest.approx(joint.offset_distance_f) == 40.0


def test_m335_lagmul_borel_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Borel Linkage Joint Aliases Test
/LAGMUL/BOREL_LINKAGE/177
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/BOREL_LINKAGE/178
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/BOREL_SPATIAL_MECHANISM/179
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/BOREL_6R_MECHANISM/180
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/BOREL_BIVECTOR_MECHANISM/181
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 177 in model.lagmul_borel_linkage_joints
    assert 178 in model.lagmul_borel_linkage_joints
    assert 179 in model.lagmul_borel_linkage_joints
    assert 180 in model.lagmul_borel_linkage_joints
    assert 181 in model.lagmul_borel_linkage_joints


def test_m335_lagmul_borel_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/BOREL_LINKAGE_JOINT/182
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m335_sensor_spring_bending_lock_rate_fixed(tmp_path: Path):
    c1 = f"{971:>10d}{2.85e9:>20.1f}{0.0090:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Bending Lock Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_BENDING_LOCK_RATE/208
Spring Bending Lock Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 208 in model.sensor_spring_bending_lock_rates
    sensor = model.sensor_spring_bending_lock_rates[208]
    assert sensor.spring_id == 971
    assert pytest.approx(sensor.jbend_lock_max) == 2.85e9
    assert pytest.approx(sensor.t_delay) == 0.0090
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_BENDING_LOCK_RATE"


def test_m335_sensor_spring_bending_lock_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Bending Lock Rate Sensor Free Format Test
/SENSOR/SPRING_BENDING_LOCK_RATE/209
972, 2.95e9, 0.0110
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 209 in model.sensor_spring_bending_lock_rates
    sensor = model.sensor_spring_bending_lock_rates[209]
    assert sensor.spring_id == 972
    assert pytest.approx(sensor.jbend_lock_max) == 2.95e9
    assert pytest.approx(sensor.t_delay) == 0.0110


def test_m335_sensor_spring_bending_lock_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Bending Lock Rate Sensor Aliases Test
/SENSOR/SPRING_BEND_LOCK_RATE/210
973, 3.4e9, 0.0030
/SENSOR/SPRING_RATE_LOCK_BEND/211
974, 3.4e9, 0.0030
/SENSOR/BENDING_LOCK_RATE_SPRING/212
975, 3.4e9, 0.0030
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 210 in model.sensor_spring_bending_lock_rates
    assert 211 in model.sensor_spring_bending_lock_rates
    assert 212 in model.sensor_spring_bending_lock_rates
    assert len(model.sensors) == 3


def test_m335_sensor_spring_bending_lock_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_BENDING_LOCK_RATE/213
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
