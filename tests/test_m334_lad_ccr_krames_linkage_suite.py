"""Tests for Milestone M334: LadCoupleCrushRate Failure Model, EngFlexothermoacousticResonanceEnergy, KramesLinkageJoint, and SensorSpringTorsionalLockRate."""

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


def test_m334_fail_lad_couple_crush_rate_fixed(tmp_path: Path):
    c1 = f"{170.0:>20.4f}{465.0:>20.4f}{35.0:>20.4f}{1.62:>20.4f}{0.975:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1500:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Crush Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_CRUSH_RATE/1500
Ladeveze Rate Dependent Coupled Crush Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1500 in model.fail_ladcouplecrushrates
    fccr = model.fail_ladcouplecrushrates[1500]
    assert pytest.approx(fccr.sigma_ccr0) == 170.0
    assert pytest.approx(fccr.sigma_ccrc) == 465.0
    assert pytest.approx(fccr.gamma_ccr) == 35.0
    assert pytest.approx(fccr.p_ccr) == 1.62
    assert pytest.approx(fccr.d_ccr_max) == 0.975
    assert fccr.ifail_sh == 1
    assert fccr.ifail_so == 2
    assert fccr.fail_id == 1500
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_CRUSH_RATE"


def test_m334_fail_lad_couple_crush_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Crush Rate Free Format Test
/FAIL/LAD_COUPLE_CRUSH_RATE/1501
182.0, 525.0, 38.0, 1.75, 0.955
1, 1
1501
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1501 in model.fail_ladcouplecrushrates
    fccr = model.fail_ladcouplecrushrates[1501]
    assert pytest.approx(fccr.sigma_ccr0) == 182.0
    assert pytest.approx(fccr.sigma_ccrc) == 525.0
    assert pytest.approx(fccr.gamma_ccr) == 38.0
    assert pytest.approx(fccr.p_ccr) == 1.75
    assert pytest.approx(fccr.d_ccr_max) == 0.955
    assert fccr.fail_id == 1501


def test_m334_fail_lad_couple_crush_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Crush Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_CRUSH_RATE/1502
135.0, 380.0, 25.0, 1.35, 0.92
1, 1
/FAIL/LAD_CCR/1503
135.0, 380.0, 25.0, 1.35, 0.92
1, 1
/FAIL/LAD_CCR_MODEL/1504
135.0, 380.0, 25.0, 1.35, 0.92
1, 1
/FAIL/LAD_CCR_LAW/1505
135.0, 380.0, 25.0, 1.35, 0.92
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_CRUSH/1506
135.0, 380.0, 25.0, 1.35, 0.92
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1502 in model.fail_ladcouplecrushrates
    assert 1503 in model.fail_ladcouplecrushrates
    assert 1504 in model.fail_ladcouplecrushrates
    assert 1505 in model.fail_ladcouplecrushrates
    assert 1506 in model.fail_ladcouplecrushrates
    assert len(model.raw_fails) == 5


def test_m334_fail_lad_couple_crush_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_COUPLE_CRUSH_RATE/1507
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m334_eng_flexothermoacoustic_resonance_energy(tmp_path: Path):
    c1 = f"{0.00135:>20.6f}{185:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermoacoustic Resonance Energy Fixed and Free Format Test
2022 0
/ENG/FLEXOTHERMOACOUSTIC_RESONANCE_ENERGY/1
Fixed Flexothermoacoustic Resonance Energy Output
{c1}
/ENG/FLEXOTHERMOACOUSTIC_RESONANCE_ENERGY/2
Free Flexothermoacoustic Resonance Energy Output
0.00175, 340
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermoacoustic_resonance_energies
    assert 2 in model.eng_flexothermoacoustic_resonance_energies
    ftar1 = model.eng_flexothermoacoustic_resonance_energies[1]
    assert pytest.approx(ftar1.dt_ftar) == 0.00135
    assert ftar1.sens_id == 185
    ftar2 = model.eng_flexothermoacoustic_resonance_energies[2]
    assert pytest.approx(ftar2.dt_ftar) == 0.00175
    assert ftar2.sens_id == 340


def test_m334_eng_flexothermoacoustic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermoacoustic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_AC_RES_WORK/3
0.00035, 90
/ENG/EFLEXOTHERMOACOUSTICRESONANCE/4
0.00040, 100
/ENG/FLEXOTHERMOACOUSTIC_RESONANCE_DISSIPATION/5
0.00045, 110
/ENG/EM_FLEXOTHERMOACOUSTIC_RESONANCE/6
0.00050, 120
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermoacoustic_resonance_energies
    assert 4 in model.eng_flexothermoacoustic_resonance_energies
    assert 5 in model.eng_flexothermoacoustic_resonance_energies
    assert 6 in model.eng_flexothermoacoustic_resonance_energies


def test_m334_eng_flexothermoacoustic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOACOUSTIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m334_lagmul_krames_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{201:>10d}{202:>10d}{203:>10d}{1.55e7:>20.1f}{48:>10d}{2.2e-5:>20.6e}"
    c2 = f"{98.0:>20.4f}{92.5:>20.4f}{80.0:>20.4f}{36.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Krames Linkage Joint Fixed Format Test
2022 0
/KRAMES_LINKAGE_JOINT/165
Krames Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 165 in model.lagmul_krames_linkage_joints
    joint = model.lagmul_krames_linkage_joints[165]
    assert joint.node1 == 201
    assert joint.node2 == 202
    assert joint.node3 == 203
    assert pytest.approx(joint.stiff) == 1.55e7
    assert joint.skew_id == 48
    assert pytest.approx(joint.tol) == 2.2e-5
    assert pytest.approx(joint.link_len_a) == 98.0
    assert pytest.approx(joint.link_len_b) == 92.5
    assert pytest.approx(joint.twist_angle_alpha) == 80.0
    assert pytest.approx(joint.offset_distance_f) == 36.0


def test_m334_lagmul_krames_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Krames Linkage Joint Free Format Test
/LAGMUL/KRAMES_LINKAGE_JOINT/166
301, 302, 303, 8.2e6, 68, 3.5e-5
100.0, 93.0, 84.0, 38.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 166 in model.lagmul_krames_linkage_joints
    joint = model.lagmul_krames_linkage_joints[166]
    assert joint.node1 == 301
    assert joint.node2 == 302
    assert joint.node3 == 303
    assert pytest.approx(joint.stiff) == 8.2e6
    assert joint.skew_id == 68
    assert pytest.approx(joint.tol) == 3.5e-5
    assert pytest.approx(joint.link_len_a) == 100.0
    assert pytest.approx(joint.link_len_b) == 93.0
    assert pytest.approx(joint.twist_angle_alpha) == 84.0
    assert pytest.approx(joint.offset_distance_f) == 38.0


def test_m334_lagmul_krames_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Krames Linkage Joint Aliases Test
/LAGMUL/KRAMES_LINKAGE/167
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/KRAMES_LINKAGE/168
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/KRAMES_SPATIAL_MECHANISM/169
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/KRAMES_6R_MECHANISM/170
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/KRAMES_SYMMETRICAL_MECHANISM/171
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 167 in model.lagmul_krames_linkage_joints
    assert 168 in model.lagmul_krames_linkage_joints
    assert 169 in model.lagmul_krames_linkage_joints
    assert 170 in model.lagmul_krames_linkage_joints
    assert 171 in model.lagmul_krames_linkage_joints


def test_m334_lagmul_krames_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/KRAMES_LINKAGE_JOINT/172
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m334_sensor_spring_torsional_lock_rate_fixed(tmp_path: Path):
    c1 = f"{961:>10d}{2.65e9:>20.1f}{0.0085:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Torsional Lock Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_LOCK_RATE/198
Spring Torsional Lock Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 198 in model.sensor_spring_torsional_lock_rates
    sensor = model.sensor_spring_torsional_lock_rates[198]
    assert sensor.spring_id == 961
    assert pytest.approx(sensor.jtors_lock_max) == 2.65e9
    assert pytest.approx(sensor.t_delay) == 0.0085
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_LOCK_RATE"


def test_m334_sensor_spring_torsional_lock_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Torsional Lock Rate Sensor Free Format Test
/SENSOR/SPRING_TORSIONAL_LOCK_RATE/199
962, 2.75e9, 0.0105
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 199 in model.sensor_spring_torsional_lock_rates
    sensor = model.sensor_spring_torsional_lock_rates[199]
    assert sensor.spring_id == 962
    assert pytest.approx(sensor.jtors_lock_max) == 2.75e9
    assert pytest.approx(sensor.t_delay) == 0.0105


def test_m334_sensor_spring_torsional_lock_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Torsional Lock Rate Sensor Aliases Test
/SENSOR/SPRING_TORS_LOCK_RATE/200
963, 3.2e9, 0.0025
/SENSOR/SPRING_RATE_LOCK_TORS/201
964, 3.2e9, 0.0025
/SENSOR/TORSIONAL_LOCK_RATE_SPRING/202
965, 3.2e9, 0.0025
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 200 in model.sensor_spring_torsional_lock_rates
    assert 201 in model.sensor_spring_torsional_lock_rates
    assert 202 in model.sensor_spring_torsional_lock_rates
    assert len(model.sensors) == 3


def test_m334_sensor_spring_torsional_lock_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TORSIONAL_LOCK_RATE/203
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
