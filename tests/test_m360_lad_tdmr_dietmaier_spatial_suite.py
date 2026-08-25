"""Tests for Milestone M360: LadTransverseDelaminationMicrodebondingRate Failure Model, EngFlexothermoplasmonexcitonphononmagnonpolaritonicResonanceEnergy, DietmaierSpatialLinkageJoint, and SensorSpringTotalAngularPopRate."""

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


def test_m360_fail_lad_transverse_delamination_microdebonding_rate_fixed(tmp_path: Path):
    c1 = f"{144.0:>20.4f}{432.0:>20.4f}{56.0:>20.4f}{2.50:>20.4f}{0.980:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1760:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Delamination Micro-Debonding Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_DELAMINATION_MICRODEBONDING_RATE/1760
Ladeveze Transverse Delamination Microdebonding Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1760 in model.fail_ladtransversedelaminationmicrodebondingrates
    ftdmr = model.fail_ladtransversedelaminationmicrodebondingrates[1760]
    assert pytest.approx(ftdmr.sigma_tdmr0) == 144.0
    assert pytest.approx(ftdmr.sigma_tdmrc) == 432.0
    assert pytest.approx(ftdmr.gamma_tdmr) == 56.0
    assert pytest.approx(ftdmr.p_tdmr) == 2.50
    assert pytest.approx(ftdmr.d_tdmr_max) == 0.980
    assert ftdmr.ifail_sh == 1
    assert ftdmr.ifail_so == 2
    assert ftdmr.fail_id == 1760
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_DELAMINATION_MICRODEBONDING_RATE"


def test_m360_fail_lad_transverse_delamination_microdebonding_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Delamination Micro-Debonding Rate Free Format Test
/FAIL/LAD_TRANSVERSE_DELAMINATION_MICRODEBONDING_RATE/1761
154.0, 462.0, 62.0, 2.70, 0.966
1, 1
1761
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1761 in model.fail_ladtransversedelaminationmicrodebondingrates
    ftdmr = model.fail_ladtransversedelaminationmicrodebondingrates[1761]
    assert pytest.approx(ftdmr.sigma_tdmr0) == 154.0
    assert pytest.approx(ftdmr.sigma_tdmrc) == 462.0
    assert pytest.approx(ftdmr.gamma_tdmr) == 62.0
    assert pytest.approx(ftdmr.p_tdmr) == 2.70
    assert pytest.approx(ftdmr.d_tdmr_max) == 0.966
    assert ftdmr.fail_id == 1761


def test_m360_fail_lad_transverse_delamination_microdebonding_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Delamination Micro-Debonding Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_DELAMINATION_MICRODEBONDING_RATE/1762
134.0, 402.0, 50.0, 2.14, 0.994
1, 1
/FAIL/LAD_TDMR/1763
134.0, 402.0, 50.0, 2.14, 0.994
1, 1
/FAIL/LAD_TDMR_MODEL/1764
134.0, 402.0, 50.0, 2.14, 0.994
1, 1
/FAIL/LAD_TDMR_LAW/1765
134.0, 402.0, 50.0, 2.14, 0.994
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_DELAMINATION_MICRODEBONDING/1766
134.0, 402.0, 50.0, 2.14, 0.994
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1762 in model.fail_ladtransversedelaminationmicrodebondingrates
    assert 1763 in model.fail_ladtransversedelaminationmicrodebondingrates
    assert 1764 in model.fail_ladtransversedelaminationmicrodebondingrates
    assert 1765 in model.fail_ladtransversedelaminationmicrodebondingrates
    assert 1766 in model.fail_ladtransversedelaminationmicrodebondingrates
    assert len(model.raw_fails) == 5


def test_m360_fail_lad_transverse_delamination_microdebonding_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_TRANSVERSE_DELAMINATION_MICRODEBONDING_RATE/1767
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m360_eng_flexothermoplasmonexcitonphononmagnonpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00160:>20.6f}{250:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermoplasmonexcitonphononmagnonpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOPLASMONEXCITONPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY/1
Flexothermoplasmonexcitonphononmagnonpolaritonic Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermoplasmonexcitonphononmagnonpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonexcitonphononmagnonpolaritonic_resonance_energies[1]
    assert pytest.approx(eng.dt_ftpepmpr) == 0.00160
    assert eng.sens_id == 250


def test_m360_eng_flexothermoplasmonexcitonphononmagnonpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexothermoplasmonexcitonphononmagnonpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOPLASMONEXCITONPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY/2
0.00180, 260
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexothermoplasmonexcitonphononmagnonpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonexcitonphononmagnonpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_ftpepmpr) == 0.00180
    assert eng.sens_id == 260


def test_m360_eng_flexothermoplasmonexcitonphononmagnonpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermoplasmonexcitonphononmagnonpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_PLASMON_EXCITON_PHONON_MAGNON_POLARITON_RES_WORK/3
0.00120, 185
/ENG/EFLEXOTHERMOPLASMONEXCITONPHONONMAGNONPOLARITONICRESONANCE/4
0.00124, 190
/ENG/FLEXOTHERMOPLASMONEXCITONPHONONMAGNONPOLARITONIC_RESONANCE_DISSIPATION/5
0.00130, 200
/ENG/EM_FLEXOTHERMOPLASMONEXCITONPHONONMAGNONPOLARITONIC_RESONANCE/6
0.00136, 210
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermoplasmonexcitonphononmagnonpolaritonic_resonance_energies
    assert 4 in model.eng_flexothermoplasmonexcitonphononmagnonpolaritonic_resonance_energies
    assert 5 in model.eng_flexothermoplasmonexcitonphononmagnonpolaritonic_resonance_energies
    assert 6 in model.eng_flexothermoplasmonexcitonphononmagnonpolaritonic_resonance_energies


def test_m360_eng_flexothermoplasmonexcitonphononmagnonpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOPLASMONEXCITONPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m360_lagmul_dietmaier_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{461:>10d}{462:>10d}{463:>10d}{4.15e7:>20.1f}{128:>10d}{3.2e-5:>20.6e}"
    c2 = f"{215.0:>20.4f}{205.0:>20.4f}{185.0:>20.4f}{135.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Dietmaier Spatial Linkage Joint Fixed Format Test
2022 0
/DIETMAIER_SPATIAL_LINKAGE_JOINT/425
Dietmaier Spatial Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 425 in model.lagmul_dietmaier_spatial_linkage_joints
    joint = model.lagmul_dietmaier_spatial_linkage_joints[425]
    assert joint.node1 == 461
    assert joint.node2 == 462
    assert joint.node3 == 463
    assert pytest.approx(joint.stiff) == 4.15e7
    assert joint.skew_id == 128
    assert pytest.approx(joint.tol) == 3.2e-5
    assert pytest.approx(joint.link_len_a) == 215.0
    assert pytest.approx(joint.link_len_b) == 205.0
    assert pytest.approx(joint.twist_angle_alpha) == 185.0
    assert pytest.approx(joint.offset_distance_s) == 135.0


def test_m360_lagmul_dietmaier_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Dietmaier Spatial Linkage Joint Free Format Test
/LAGMUL/DIETMAIER_SPATIAL_LINKAGE_JOINT/426
561, 562, 563, 27.0e6, 148, 4.2e-5
216.0, 206.0, 186.0, 136.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 426 in model.lagmul_dietmaier_spatial_linkage_joints
    joint = model.lagmul_dietmaier_spatial_linkage_joints[426]
    assert joint.node1 == 561
    assert joint.node2 == 562
    assert joint.node3 == 563
    assert pytest.approx(joint.stiff) == 27.0e6
    assert joint.skew_id == 148
    assert pytest.approx(joint.tol) == 4.2e-5
    assert pytest.approx(joint.link_len_a) == 216.0
    assert pytest.approx(joint.link_len_b) == 206.0
    assert pytest.approx(joint.twist_angle_alpha) == 186.0
    assert pytest.approx(joint.offset_distance_s) == 136.0


def test_m360_lagmul_dietmaier_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Dietmaier Spatial Linkage Joint Aliases Test
/LAGMUL/DIETMAIER_SPATIAL_LINKAGE/427
1, 2, 3, 1e6, 0, 1e-6
56.0, 56.0, 102.0, 42.0
/DIETMAIER_SPATIAL_LINKAGE/428
1, 2, 3, 1e6, 0, 1e-6
56.0, 56.0, 102.0, 42.0
/DIETMAIER_SPATIAL_MULTI_LOOP_MECHANISM/429
1, 2, 3, 1e6, 0, 1e-6
56.0, 56.0, 102.0, 42.0
/DIETMAIER_SPATIAL_ISOMORPHIC_MECHANISM/430
1, 2, 3, 1e6, 0, 1e-6
56.0, 56.0, 102.0, 42.0
/DIETMAIER_SPATIAL_6R_MECHANISM/431
1, 2, 3, 1e6, 0, 1e-6
56.0, 56.0, 102.0, 42.0
/DIETMAIER_SPATIAL_OVERCONSTRAINED_MECHANISM/432
1, 2, 3, 1e6, 0, 1e-6
56.0, 56.0, 102.0, 42.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 427 in model.lagmul_dietmaier_spatial_linkage_joints
    assert 428 in model.lagmul_dietmaier_spatial_linkage_joints
    assert 429 in model.lagmul_dietmaier_spatial_linkage_joints
    assert 430 in model.lagmul_dietmaier_spatial_linkage_joints
    assert 431 in model.lagmul_dietmaier_spatial_linkage_joints
    assert 432 in model.lagmul_dietmaier_spatial_linkage_joints


def test_m360_lagmul_dietmaier_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/DIETMAIER_SPATIAL_LINKAGE_JOINT/433
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m360_sensor_spring_total_angular_pop_rate_fixed(tmp_path: Path):
    c1 = f"{1205:>10d}{7.25e9:>20.1f}{0.0325:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Total Angular Pop Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE/458
Spring Total Angular Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 458 in model.sensor_spring_total_angular_pop_rates
    sensor = model.sensor_spring_total_angular_pop_rates[458]
    assert sensor.spring_id == 1205
    assert pytest.approx(sensor.jtot_ang_pop_max) == 7.25e9
    assert pytest.approx(sensor.t_delay) == 0.0325
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_POP_RATE"


def test_m360_sensor_spring_total_angular_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Total Angular Pop Rate Sensor Free Format Test
/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE/459
1206, 7.35e9, 0.0350
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 459 in model.sensor_spring_total_angular_pop_rates
    sensor = model.sensor_spring_total_angular_pop_rates[459]
    assert sensor.spring_id == 1206
    assert pytest.approx(sensor.jtot_ang_pop_max) == 7.35e9
    assert pytest.approx(sensor.t_delay) == 0.0350


def test_m360_sensor_spring_total_angular_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Total Angular Pop Rate Sensor Aliases Test
/SENSOR/SPRING_TOT_ANG_POP_RATE/460
1207, 7.4e9, 0.0255
/SENSOR/SPRING_RATE_POP_ANG_TOT/461
1208, 7.4e9, 0.0255
/SENSOR/TOTAL_ANGULAR_POP_RATE_SPRING/462
1209, 7.4e9, 0.0255
/SENSOR/SPRING_POP_ANG_TOT/463
1210, 7.4e9, 0.0255
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 460 in model.sensor_spring_total_angular_pop_rates
    assert 461 in model.sensor_spring_total_angular_pop_rates
    assert 462 in model.sensor_spring_total_angular_pop_rates
    assert 463 in model.sensor_spring_total_angular_pop_rates
    assert len(model.sensors) == 4


def test_m360_sensor_spring_total_angular_pop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE/464
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
