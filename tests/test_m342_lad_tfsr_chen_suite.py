"""Tests for Milestone M342: LadTransverseFiberSplittingRate Failure Model, EngFlexothermophotonicResonanceEnergy, ChenLinkageJoint, and SensorSpringTotalAngularDropRate."""

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


def test_m342_fail_lad_transverse_fiber_splitting_rate_fixed(tmp_path: Path):
    c1 = f"{218.0:>20.4f}{595.0:>20.4f}{49.5:>20.4f}{1.96:>20.4f}{0.955:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1580:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Fiber Splitting Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_FIBER_SPLITTING_RATE/1580
Ladeveze Transverse Fiber Splitting Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1580 in model.fail_ladtransversefibersplittingrates
    ftfsr = model.fail_ladtransversefibersplittingrates[1580]
    assert pytest.approx(ftfsr.sigma_tfsr0) == 218.0
    assert pytest.approx(ftfsr.sigma_tfsrc) == 595.0
    assert pytest.approx(ftfsr.gamma_tfsr) == 49.5
    assert pytest.approx(ftfsr.p_tfsr) == 1.96
    assert pytest.approx(ftfsr.d_tfsr_max) == 0.955
    assert ftfsr.ifail_sh == 1
    assert ftfsr.ifail_so == 2
    assert ftfsr.fail_id == 1580
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_FIBER_SPLITTING_RATE"


def test_m342_fail_lad_transverse_fiber_splitting_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Fiber Splitting Rate Free Format Test
/FAIL/LAD_TRANSVERSE_FIBER_SPLITTING_RATE/1581
230.0, 645.0, 53.0, 2.15, 0.935
1, 1
1581
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1581 in model.fail_ladtransversefibersplittingrates
    ftfsr = model.fail_ladtransversefibersplittingrates[1581]
    assert pytest.approx(ftfsr.sigma_tfsr0) == 230.0
    assert pytest.approx(ftfsr.sigma_tfsrc) == 645.0
    assert pytest.approx(ftfsr.gamma_tfsr) == 53.0
    assert pytest.approx(ftfsr.p_tfsr) == 2.15
    assert pytest.approx(ftfsr.d_tfsr_max) == 0.935
    assert ftfsr.fail_id == 1581


def test_m342_fail_lad_transverse_fiber_splitting_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Fiber Splitting Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_FIBER_SPLITTING_RATE/1582
180.0, 490.0, 39.0, 1.72, 0.942
1, 1
/FAIL/LAD_TFSR/1583
180.0, 490.0, 39.0, 1.72, 0.942
1, 1
/FAIL/LAD_TFSR_MODEL/1584
180.0, 490.0, 39.0, 1.72, 0.942
1, 1
/FAIL/LAD_TFSR_LAW/1585
180.0, 490.0, 39.0, 1.72, 0.942
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_FIBER_SPLITTING/1586
180.0, 490.0, 39.0, 1.72, 0.942
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1582 in model.fail_ladtransversefibersplittingrates
    assert 1583 in model.fail_ladtransversefibersplittingrates
    assert 1584 in model.fail_ladtransversefibersplittingrates
    assert 1585 in model.fail_ladtransversefibersplittingrates
    assert 1586 in model.fail_ladtransversefibersplittingrates
    assert len(model.raw_fails) == 5


def test_m342_fail_lad_transverse_fiber_splitting_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_TRANSVERSE_FIBER_SPLITTING_RATE/1587
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m342_eng_flexothermophotonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00075:>20.6f}{155:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermophotonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOPHOTONIC_RESONANCE_ENERGY/1
Flexothermophotonic Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermophotonic_resonance_energies
    eng = model.eng_flexothermophotonic_resonance_energies[1]
    assert pytest.approx(eng.dt_ftpr) == 0.00075
    assert eng.sens_id == 155


def test_m342_eng_flexothermophotonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexothermophotonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOPHOTONIC_RESONANCE_ENERGY/2
0.00088, 165
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexothermophotonic_resonance_energies
    eng = model.eng_flexothermophotonic_resonance_energies[2]
    assert pytest.approx(eng.dt_ftpr) == 0.00088
    assert eng.sens_id == 165


def test_m342_eng_flexothermophotonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermophotonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_PHOTON_RES_WORK/3
0.00058, 122
/ENG/EFLEXOTHERMOPHOTONICRESONANCE/4
0.00060, 128
/ENG/FLEXOTHERMOPHOTONIC_RESONANCE_DISSIPATION/5
0.00068, 138
/ENG/EM_FLEXOTHERMOPHOTONIC_RESONANCE/6
0.00072, 148
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermophotonic_resonance_energies
    assert 4 in model.eng_flexothermophotonic_resonance_energies
    assert 5 in model.eng_flexothermophotonic_resonance_energies
    assert 6 in model.eng_flexothermophotonic_resonance_energies


def test_m342_eng_flexothermophotonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOPHOTONIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m342_lagmul_chen_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{281:>10d}{282:>10d}{283:>10d}{2.35e7:>20.1f}{75:>10d}{4.0e-5:>20.6e}"
    c2 = f"{132.0:>20.4f}{122.0:>20.4f}{105.0:>20.4f}{62.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Chen Linkage Joint Fixed Format Test
2022 0
/CHEN_LINKAGE_JOINT/245
Chen Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 245 in model.lagmul_chen_linkage_joints
    joint = model.lagmul_chen_linkage_joints[245]
    assert joint.node1 == 281
    assert joint.node2 == 282
    assert joint.node3 == 283
    assert pytest.approx(joint.stiff) == 2.35e7
    assert joint.skew_id == 75
    assert pytest.approx(joint.tol) == 4.0e-5
    assert pytest.approx(joint.link_len_a) == 132.0
    assert pytest.approx(joint.link_len_b) == 122.0
    assert pytest.approx(joint.twist_angle_alpha) == 105.0
    assert pytest.approx(joint.offset_distance_e) == 62.0


def test_m342_lagmul_chen_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Chen Linkage Joint Free Format Test
/LAGMUL/CHEN_LINKAGE_JOINT/246
381, 382, 383, 12.0e6, 95, 5.0e-5
135.0, 125.0, 108.0, 65.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 246 in model.lagmul_chen_linkage_joints
    joint = model.lagmul_chen_linkage_joints[246]
    assert joint.node1 == 381
    assert joint.node2 == 382
    assert joint.node3 == 383
    assert pytest.approx(joint.stiff) == 12.0e6
    assert joint.skew_id == 95
    assert pytest.approx(joint.tol) == 5.0e-5
    assert pytest.approx(joint.link_len_a) == 135.0
    assert pytest.approx(joint.link_len_b) == 125.0
    assert pytest.approx(joint.twist_angle_alpha) == 108.0
    assert pytest.approx(joint.offset_distance_e) == 65.0


def test_m342_lagmul_chen_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Chen Linkage Joint Aliases Test
/LAGMUL/CHEN_LINKAGE/247
1, 2, 3, 1e6, 0, 1e-6
15.0, 15.0, 40.0, 7.0
/CHEN_LINKAGE/248
1, 2, 3, 1e6, 0, 1e-6
15.0, 15.0, 40.0, 7.0
/CHEN_SPATIAL_MECHANISM/249
1, 2, 3, 1e6, 0, 1e-6
15.0, 15.0, 40.0, 7.0
/CHEN_6R_MECHANISM/250
1, 2, 3, 1e6, 0, 1e-6
15.0, 15.0, 40.0, 7.0
/CHEN_OVERCONSTRAINED_MECHANISM/251
1, 2, 3, 1e6, 0, 1e-6
15.0, 15.0, 40.0, 7.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 247 in model.lagmul_chen_linkage_joints
    assert 248 in model.lagmul_chen_linkage_joints
    assert 249 in model.lagmul_chen_linkage_joints
    assert 250 in model.lagmul_chen_linkage_joints
    assert 251 in model.lagmul_chen_linkage_joints


def test_m342_lagmul_chen_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/CHEN_LINKAGE_JOINT/252
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m342_sensor_spring_total_angular_drop_rate_fixed(tmp_path: Path):
    c1 = f"{1025:>10d}{4.55e9:>20.1f}{0.0145:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Total Angular Drop Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_DROP_RATE/278
Spring Total Angular Drop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 278 in model.sensor_spring_total_angular_drop_rates
    sensor = model.sensor_spring_total_angular_drop_rates[278]
    assert sensor.spring_id == 1025
    assert pytest.approx(sensor.jtot_ang_drop_max) == 4.55e9
    assert pytest.approx(sensor.t_delay) == 0.0145
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_DROP_RATE"


def test_m342_sensor_spring_total_angular_drop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Total Angular Drop Rate Sensor Free Format Test
/SENSOR/SPRING_TOTAL_ANGULAR_DROP_RATE/279
1026, 4.65e9, 0.0170
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 279 in model.sensor_spring_total_angular_drop_rates
    sensor = model.sensor_spring_total_angular_drop_rates[279]
    assert sensor.spring_id == 1026
    assert pytest.approx(sensor.jtot_ang_drop_max) == 4.65e9
    assert pytest.approx(sensor.t_delay) == 0.0170


def test_m342_sensor_spring_total_angular_drop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Total Angular Drop Rate Sensor Aliases Test
/SENSOR/SPRING_TOT_ANG_DROP_RATE/280
1027, 4.7e9, 0.0075
/SENSOR/SPRING_RATE_DROP_ANG_TOT/281
1028, 4.7e9, 0.0075
/SENSOR/TOTAL_ANGULAR_DROP_RATE_SPRING/282
1029, 4.7e9, 0.0075
/SENSOR/SPRING_DROP_ANG_TOT/283
1030, 4.7e9, 0.0075
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 280 in model.sensor_spring_total_angular_drop_rates
    assert 281 in model.sensor_spring_total_angular_drop_rates
    assert 282 in model.sensor_spring_total_angular_drop_rates
    assert 283 in model.sensor_spring_total_angular_drop_rates
    assert len(model.sensors) == 4


def test_m342_sensor_spring_total_angular_drop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TOTAL_ANGULAR_DROP_RATE/284
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
