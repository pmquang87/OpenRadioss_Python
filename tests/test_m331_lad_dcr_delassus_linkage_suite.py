"""Tests for Milestone M331: LadDynamicCrushRate Failure Model, EngFlexomagnetoelectricResonanceEnergy, DelassusLinkageJoint, and SensorSpringNormalLockRate."""

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


def test_m331_fail_lad_dynamic_crush_rate_fixed(tmp_path: Path):
    c1 = f"{145.0:>20.4f}{420.0:>20.4f}{28.5:>20.4f}{1.45:>20.4f}{0.985:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1470:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Crush Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_CRUSH_RATE/1470
Ladeveze Dynamic Compressive Crush Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1470 in model.fail_laddynamiccrushrates
    fdcr = model.fail_laddynamiccrushrates[1470]
    assert pytest.approx(fdcr.sigma_cr0) == 145.0
    assert pytest.approx(fdcr.sigma_crc) == 420.0
    assert pytest.approx(fdcr.gamma_cr) == 28.5
    assert pytest.approx(fdcr.p_cr) == 1.45
    assert pytest.approx(fdcr.d_dcr_max) == 0.985
    assert fdcr.ifail_sh == 1
    assert fdcr.ifail_so == 2
    assert fdcr.fail_id == 1470
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_CRUSH_RATE"


def test_m331_fail_lad_dynamic_crush_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Crush Rate Free Format Test
/FAIL/LAD_DYNAMIC_CRUSH_RATE/1471
160.0, 480.0, 32.0, 1.60, 0.970
1, 1
1471
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1471 in model.fail_laddynamiccrushrates
    fdcr = model.fail_laddynamiccrushrates[1471]
    assert pytest.approx(fdcr.sigma_cr0) == 160.0
    assert pytest.approx(fdcr.sigma_crc) == 480.0
    assert pytest.approx(fdcr.gamma_cr) == 32.0
    assert pytest.approx(fdcr.p_cr) == 1.60
    assert pytest.approx(fdcr.d_dcr_max) == 0.970
    assert fdcr.fail_id == 1471


def test_m331_fail_lad_dynamic_crush_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Crush Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_CRUSH_RATE/1472
120.0, 350.0, 20.0, 1.2, 0.95
1, 1
/FAIL/LAD_DCR/1473
120.0, 350.0, 20.0, 1.2, 0.95
1, 1
/FAIL/LAD_DCR_MODEL/1474
120.0, 350.0, 20.0, 1.2, 0.95
1, 1
/FAIL/LAD_DCR_LAW/1475
120.0, 350.0, 20.0, 1.2, 0.95
1, 1
/FAIL/LADEVEZE_DYNAMIC_COMPRESSIVE_CRUSH_RATE/1476
120.0, 350.0, 20.0, 1.2, 0.95
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1472 in model.fail_laddynamiccrushrates
    assert 1473 in model.fail_laddynamiccrushrates
    assert 1474 in model.fail_laddynamiccrushrates
    assert 1475 in model.fail_laddynamiccrushrates
    assert 1476 in model.fail_laddynamiccrushrates
    assert len(model.raw_fails) == 5


def test_m331_fail_lad_dynamic_crush_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_DYNAMIC_CRUSH_RATE/1477
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m331_eng_flexomagnetoelectric_resonance_energy(tmp_path: Path):
    c1 = f"{0.00115:>20.6f}{155:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexomagnetoelectric Resonance Energy Fixed and Free Format Test
2022 0
/ENG/FLEXOMAGNETOELECTRIC_RESONANCE_ENERGY/1
Fixed Flexomagnetoelectric Resonance Energy Output
{c1}
/ENG/FLEXOMAGNETOELECTRIC_RESONANCE_ENERGY/2
Free Flexomagnetoelectric Resonance Energy Output
0.00155, 310
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexomagnetoelectric_resonance_energies
    assert 2 in model.eng_flexomagnetoelectric_resonance_energies
    fmer1 = model.eng_flexomagnetoelectric_resonance_energies[1]
    assert pytest.approx(fmer1.dt_fmer) == 0.00115
    assert fmer1.sens_id == 155
    fmer2 = model.eng_flexomagnetoelectric_resonance_energies[2]
    assert pytest.approx(fmer2.dt_fmer) == 0.00155
    assert fmer2.sens_id == 310


def test_m331_eng_flexomagnetoelectric_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexomagnetoelectric Resonance Energy Aliases Test
/ENG/FLEXOMAG_ELEC_RES_WORK/3
0.00025, 78
/ENG/EFLEXOMAGNETOELECTRICRESONANCE/4
0.00030, 88
/ENG/FLEXOMAGNETOELECTRIC_RESONANCE_DISSIPATION/5
0.00035, 98
/ENG/EM_FLEXOMAGNETOELECTRIC_RESONANCE/6
0.00040, 108
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexomagnetoelectric_resonance_energies
    assert 4 in model.eng_flexomagnetoelectric_resonance_energies
    assert 5 in model.eng_flexomagnetoelectric_resonance_energies
    assert 6 in model.eng_flexomagnetoelectric_resonance_energies


def test_m331_eng_flexomagnetoelectric_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOMAGNETOELECTRIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m331_lagmul_delassus_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{171:>10d}{172:>10d}{173:>10d}{1.25e7:>20.1f}{38:>10d}{3.2e-5:>20.6e}"
    c2 = f"{88.0:>20.4f}{82.5:>20.4f}{74.0:>20.4f}{30.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Delassus Linkage Joint Fixed Format Test
2022 0
/DELASSUS_LINKAGE_JOINT/135
Delassus Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 135 in model.lagmul_delassus_linkage_joints
    joint = model.lagmul_delassus_linkage_joints[135]
    assert joint.node1 == 171
    assert joint.node2 == 172
    assert joint.node3 == 173
    assert pytest.approx(joint.stiff) == 1.25e7
    assert joint.skew_id == 38
    assert pytest.approx(joint.tol) == 3.2e-5
    assert pytest.approx(joint.link_len_a) == 88.0
    assert pytest.approx(joint.link_len_b) == 82.5
    assert pytest.approx(joint.twist_angle_alpha) == 74.0
    assert pytest.approx(joint.offset_distance_f) == 30.0


def test_m331_lagmul_delassus_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Delassus Linkage Joint Free Format Test
/LAGMUL/DELASSUS_LINKAGE_JOINT/136
271, 272, 273, 9.1e6, 58, 4.8e-5
92.0, 86.0, 78.0, 32.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 136 in model.lagmul_delassus_linkage_joints
    joint = model.lagmul_delassus_linkage_joints[136]
    assert joint.node1 == 271
    assert joint.node2 == 272
    assert joint.node3 == 273
    assert pytest.approx(joint.stiff) == 9.1e6
    assert joint.skew_id == 58
    assert pytest.approx(joint.tol) == 4.8e-5
    assert pytest.approx(joint.link_len_a) == 92.0
    assert pytest.approx(joint.link_len_b) == 86.0
    assert pytest.approx(joint.twist_angle_alpha) == 78.0
    assert pytest.approx(joint.offset_distance_f) == 32.0


def test_m331_lagmul_delassus_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Delassus Linkage Joint Aliases Test
/LAGMUL/DELASSUS_LINKAGE/137
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/DELASSUS_LINKAGE/138
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/DELASSUS_SPATIAL_MECHANISM/139
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/DELASSUS_6R_MECHANISM/140
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 137 in model.lagmul_delassus_linkage_joints
    assert 138 in model.lagmul_delassus_linkage_joints
    assert 139 in model.lagmul_delassus_linkage_joints
    assert 140 in model.lagmul_delassus_linkage_joints


def test_m331_lagmul_delassus_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/DELASSUS_LINKAGE_JOINT/141
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m331_sensor_spring_normal_lock_rate_fixed(tmp_path: Path):
    c1 = f"{931:>10d}{2.05e9:>20.1f}{0.0070:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Normal Lock Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_NORMAL_LOCK_RATE/168
Spring Normal Lock Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 168 in model.sensor_spring_normal_lock_rates
    sensor = model.sensor_spring_normal_lock_rates[168]
    assert sensor.spring_id == 931
    assert pytest.approx(sensor.jnorm_lock_max) == 2.05e9
    assert pytest.approx(sensor.t_delay) == 0.0070
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_NORMAL_LOCK_RATE"


def test_m331_sensor_spring_normal_lock_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Normal Lock Rate Sensor Free Format Test
/SENSOR/SPRING_NORMAL_LOCK_RATE/169
932, 2.15e9, 0.0090
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 169 in model.sensor_spring_normal_lock_rates
    sensor = model.sensor_spring_normal_lock_rates[169]
    assert sensor.spring_id == 932
    assert pytest.approx(sensor.jnorm_lock_max) == 2.15e9
    assert pytest.approx(sensor.t_delay) == 0.0090


def test_m331_sensor_spring_normal_lock_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Normal Lock Rate Sensor Aliases Test
/SENSOR/SPRING_NORM_LOCK_RATE/170
933, 2.8e9, 0.001
/SENSOR/SPRING_RATE_LOCK_NORM/171
934, 2.8e9, 0.001
/SENSOR/NORMAL_LOCK_RATE_SPRING/172
935, 2.8e9, 0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 170 in model.sensor_spring_normal_lock_rates
    assert 171 in model.sensor_spring_normal_lock_rates
    assert 172 in model.sensor_spring_normal_lock_rates
    assert len(model.sensors) == 3


def test_m331_sensor_spring_normal_lock_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_NORMAL_LOCK_RATE/173
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
