"""Tests for Milestone M327: LadCoupleCreep Failure Model, EngPyroelectricResonanceEnergy, BakerLinkageJoint, and SensorSpringTotalPopRate."""

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


def test_m327_fail_lad_couple_creep_fixed(tmp_path: Path):
    c1 = f"{1.5e-6:>20.8f}{3.50:>20.4f}{45.0:>20.2f}{18.5:>20.4f}{0.975:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1390:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Couple Creep Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_CREEP/1390
Ladeveze Coupled Damage Creep Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1390 in model.fail_ladcouplecreeps
    fcc = model.fail_ladcouplecreeps[1390]
    assert pytest.approx(fcc.a_c) == 1.5e-6
    assert pytest.approx(fcc.n_c) == 3.50
    assert pytest.approx(fcc.q_c) == 45.0
    assert pytest.approx(fcc.gamma_c) == 18.5
    assert pytest.approx(fcc.d_cc_max) == 0.975
    assert fcc.ifail_sh == 1
    assert fcc.ifail_so == 2
    assert fcc.fail_id == 1390
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_CREEP"


def test_m327_fail_lad_couple_creep_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Couple Creep Free Format Test
/FAIL/LAD_COUPLE_CREEP/1391
2.2e-6, 4.0, 50.0, 22.0, 0.960
1, 1
1391
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1391 in model.fail_ladcouplecreeps
    fcc = model.fail_ladcouplecreeps[1391]
    assert pytest.approx(fcc.a_c) == 2.2e-6
    assert pytest.approx(fcc.n_c) == 4.0
    assert pytest.approx(fcc.q_c) == 50.0
    assert pytest.approx(fcc.gamma_c) == 22.0
    assert pytest.approx(fcc.d_cc_max) == 0.960
    assert fcc.fail_id == 1391


def test_m327_fail_lad_couple_creep_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Couple Creep Aliases Test
/FAIL/LADEVEZE_COUPLED_CREEP/1392
1.0e-6, 3.0, 40.0, 15.0, 0.95
1, 1
/FAIL/LAD_CC/1393
1.0e-6, 3.0, 40.0, 15.0, 0.95
1, 1
/FAIL/LAD_CC_MODEL/1394
1.0e-6, 3.0, 40.0, 15.0, 0.95
1, 1
/FAIL/LAD_CC_LAW/1395
1.0e-6, 3.0, 40.0, 15.0, 0.95
1, 1
/FAIL/LADEVEZE_DAMAGE_CREEP_COUPLING/1396
1.0e-6, 3.0, 40.0, 15.0, 0.95
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1392 in model.fail_ladcouplecreeps
    assert 1393 in model.fail_ladcouplecreeps
    assert 1394 in model.fail_ladcouplecreeps
    assert 1395 in model.fail_ladcouplecreeps
    assert 1396 in model.fail_ladcouplecreeps
    assert len(model.raw_fails) == 5


def test_m327_fail_lad_couple_creep_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_COUPLE_CREEP/1397
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m327_eng_pyroelectric_resonance_energy(tmp_path: Path):
    c1 = f"{0.00065:>20.6f}{115:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Pyroelectric Resonance Energy Fixed and Free Format Test
2022 0
/ENG/PYROELECTRIC_RESONANCE_ENERGY/1
Fixed Pyroelectric Resonance Energy Output
{c1}
/ENG/PYROELECTRIC_RESONANCE_ENERGY/2
Free Pyroelectric Resonance Energy Output
0.00105, 230
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_pyroelectric_resonance_energies
    assert 2 in model.eng_pyroelectric_resonance_energies
    pre1 = model.eng_pyroelectric_resonance_energies[1]
    assert pytest.approx(pre1.dt_pyr) == 0.00065
    assert pre1.sens_id == 115
    pre2 = model.eng_pyroelectric_resonance_energies[2]
    assert pytest.approx(pre2.dt_pyr) == 0.00105
    assert pre2.sens_id == 230


def test_m327_eng_pyroelectric_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Pyroelectric Resonance Energy Aliases Test
/ENG/PYRO_RES_WORK/3
0.00025, 65
/ENG/EPYRORESONANCE/4
0.00030, 75
/ENG/PYROELECTRIC_RESONANCE_DISSIPATION/5
0.00035, 85
/ENG/EM_PYROELECTRIC_RESONANCE/6
0.00040, 95
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_pyroelectric_resonance_energies
    assert 4 in model.eng_pyroelectric_resonance_energies
    assert 5 in model.eng_pyroelectric_resonance_energies
    assert 6 in model.eng_pyroelectric_resonance_energies


def test_m327_eng_pyroelectric_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/PYROELECTRIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m327_lagmul_baker_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{131:>10d}{132:>10d}{133:>10d}{8.5e6:>20.1f}{25:>10d}{2.0e-5:>20.6e}"
    c2 = f"{64.0:>20.4f}{58.5:>20.4f}{52.0:>20.4f}{14.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Baker Linkage Joint Fixed Format Test
2022 0
/BAKER_LINKAGE_JOINT/95
Baker Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 95 in model.lagmul_baker_linkage_joints
    joint = model.lagmul_baker_linkage_joints[95]
    assert joint.node1 == 131
    assert joint.node2 == 132
    assert joint.node3 == 133
    assert pytest.approx(joint.stiff) == 8.5e6
    assert joint.skew_id == 25
    assert pytest.approx(joint.tol) == 2.0e-5
    assert pytest.approx(joint.link_len_a) == 64.0
    assert pytest.approx(joint.link_len_b) == 58.5
    assert pytest.approx(joint.twist_angle_alpha) == 52.0
    assert pytest.approx(joint.offset_distance_d) == 14.5


def test_m327_lagmul_baker_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Baker Linkage Joint Free Format Test
/LAGMUL/BAKER_LINKAGE_JOINT/96
231, 232, 233, 6.2e6, 40, 3.5e-5
68.0, 62.0, 55.0, 16.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 96 in model.lagmul_baker_linkage_joints
    joint = model.lagmul_baker_linkage_joints[96]
    assert joint.node1 == 231
    assert joint.node2 == 232
    assert joint.node3 == 233
    assert pytest.approx(joint.stiff) == 6.2e6
    assert joint.skew_id == 40
    assert pytest.approx(joint.tol) == 3.5e-5
    assert pytest.approx(joint.link_len_a) == 68.0
    assert pytest.approx(joint.link_len_b) == 62.0
    assert pytest.approx(joint.twist_angle_alpha) == 55.0
    assert pytest.approx(joint.offset_distance_d) == 16.0


def test_m327_lagmul_baker_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Baker Linkage Joint Aliases Test
/LAGMUL/BAKER_LINKAGE/97
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/BAKER_LINKAGE/98
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/BAKER_4R_5R_6R_MECHANISM/99
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/BAKER_SPATIAL_MECHANISM/100
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 97 in model.lagmul_baker_linkage_joints
    assert 98 in model.lagmul_baker_linkage_joints
    assert 99 in model.lagmul_baker_linkage_joints
    assert 100 in model.lagmul_baker_linkage_joints


def test_m327_lagmul_baker_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/BAKER_LINKAGE_JOINT/101
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m327_sensor_spring_total_pop_rate_fixed(tmp_path: Path):
    c1 = f"{811:>10d}{1.25e9:>20.1f}{0.0050:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Total Pop Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_POP_RATE/128
Spring Total Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 128 in model.sensor_spring_total_pop_rates
    sensor = model.sensor_spring_total_pop_rates[128]
    assert sensor.spring_id == 811
    assert pytest.approx(sensor.jtot_pop_max) == 1.25e9
    assert pytest.approx(sensor.t_delay) == 0.0050
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_POP_RATE"


def test_m327_sensor_spring_total_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Total Pop Rate Sensor Free Format Test
/SENSOR/SPRING_TOTAL_POP_RATE/129
812, 1.35e9, 0.0070
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 129 in model.sensor_spring_total_pop_rates
    sensor = model.sensor_spring_total_pop_rates[129]
    assert sensor.spring_id == 812
    assert pytest.approx(sensor.jtot_pop_max) == 1.35e9
    assert pytest.approx(sensor.t_delay) == 0.0070


def test_m327_sensor_spring_total_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Total Pop Rate Sensor Aliases Test
/SENSOR/SPRING_TOT_POP_RATE/130
813, 1.8e9, 0.001
/SENSOR/SPRING_RATE_POP_TOT/131
814, 1.8e9, 0.001
/SENSOR/TOTAL_POP_RATE_SPRING/132
815, 1.8e9, 0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 130 in model.sensor_spring_total_pop_rates
    assert 131 in model.sensor_spring_total_pop_rates
    assert 132 in model.sensor_spring_total_pop_rates
    assert len(model.sensors) == 3


def test_m327_sensor_spring_total_pop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TOTAL_POP_RATE/133
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
