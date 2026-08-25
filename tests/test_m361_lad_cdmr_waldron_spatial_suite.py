"""Tests for Milestone M361: LadCoupleDelaminationMicrodebondingRate Failure Model, EngFlexomagnetoplasmonicResonanceEnergy, WaldronSpatialLinkageJoint, and SensorSpringNormalCrackleRate."""

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


def test_m361_fail_lad_couple_delamination_microdebonding_rate_fixed(tmp_path: Path):
    c1 = f"{146.0:>20.4f}{438.0:>20.4f}{58.0:>20.4f}{2.55:>20.4f}{0.978:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1770:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Delamination Micro-Debonding Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_DELAMINATION_MICRODEBONDING_RATE/1770
Ladeveze Coupled Delamination Microdebonding Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1770 in model.fail_ladcoupledelaminationmicrodebondingrates
    fcdmr = model.fail_ladcoupledelaminationmicrodebondingrates[1770]
    assert pytest.approx(fcdmr.sigma_cdmr0) == 146.0
    assert pytest.approx(fcdmr.sigma_cdmrc) == 438.0
    assert pytest.approx(fcdmr.gamma_cdmr) == 58.0
    assert pytest.approx(fcdmr.p_cdmr) == 2.55
    assert pytest.approx(fcdmr.d_cdmr_max) == 0.978
    assert fcdmr.ifail_sh == 1
    assert fcdmr.ifail_so == 2
    assert fcdmr.fail_id == 1770
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_DELAMINATION_MICRODEBONDING_RATE"


def test_m361_fail_lad_couple_delamination_microdebonding_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Delamination Micro-Debonding Rate Free Format Test
/FAIL/LAD_COUPLE_DELAMINATION_MICRODEBONDING_RATE/1771
156.0, 468.0, 64.0, 2.75, 0.964
1, 1
1771
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1771 in model.fail_ladcoupledelaminationmicrodebondingrates
    fcdmr = model.fail_ladcoupledelaminationmicrodebondingrates[1771]
    assert pytest.approx(fcdmr.sigma_cdmr0) == 156.0
    assert pytest.approx(fcdmr.sigma_cdmrc) == 468.0
    assert pytest.approx(fcdmr.gamma_cdmr) == 64.0
    assert pytest.approx(fcdmr.p_cdmr) == 2.75
    assert pytest.approx(fcdmr.d_cdmr_max) == 0.964
    assert fcdmr.fail_id == 1771


def test_m361_fail_lad_couple_delamination_microdebonding_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Delamination Micro-Debonding Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_DELAMINATION_MICRODEBONDING_RATE/1772
136.0, 408.0, 52.0, 2.16, 0.996
1, 1
/FAIL/LAD_CDMR/1773
136.0, 408.0, 52.0, 2.16, 0.996
1, 1
/FAIL/LAD_CDMR_MODEL/1774
136.0, 408.0, 52.0, 2.16, 0.996
1, 1
/FAIL/LAD_CDMR_LAW/1775
136.0, 408.0, 52.0, 2.16, 0.996
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_DELAMINATION_MICRODEBONDING/1776
136.0, 408.0, 52.0, 2.16, 0.996
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1772 in model.fail_ladcoupledelaminationmicrodebondingrates
    assert 1773 in model.fail_ladcoupledelaminationmicrodebondingrates
    assert 1774 in model.fail_ladcoupledelaminationmicrodebondingrates
    assert 1775 in model.fail_ladcoupledelaminationmicrodebondingrates
    assert 1776 in model.fail_ladcoupledelaminationmicrodebondingrates
    assert len(model.raw_fails) == 5


def test_m361_fail_lad_couple_delamination_microdebonding_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_COUPLE_DELAMINATION_MICRODEBONDING_RATE/1777
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m361_eng_flexomagnetoplasmonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00165:>20.6f}{255:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexomagnetoplasmonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPLASMONIC_RESONANCE_ENERGY/1
Flexomagnetoplasmonic Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexomagnetoplasmonic_resonance_energies
    eng = model.eng_flexomagnetoplasmonic_resonance_energies[1]
    assert pytest.approx(eng.dt_fmpr) == 0.00165
    assert eng.sens_id == 255


def test_m361_eng_flexomagnetoplasmonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexomagnetoplasmonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPLASMONIC_RESONANCE_ENERGY/2
0.00185, 265
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexomagnetoplasmonic_resonance_energies
    eng = model.eng_flexomagnetoplasmonic_resonance_energies[2]
    assert pytest.approx(eng.dt_fmpr) == 0.00185
    assert eng.sens_id == 265


def test_m361_eng_flexomagnetoplasmonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexomagnetoplasmonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PLASMON_RES_WORK/3
0.00125, 190
/ENG/EFLEXOMAGNETOPLASMONICRESONANCE/4
0.00129, 195
/ENG/FLEXOMAGNETOPLASMONIC_RESONANCE_DISSIPATION/5
0.00135, 205
/ENG/EM_FLEXOMAGNETOPLASMONIC_RESONANCE/6
0.00142, 215
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexomagnetoplasmonic_resonance_energies
    assert 4 in model.eng_flexomagnetoplasmonic_resonance_energies
    assert 5 in model.eng_flexomagnetoplasmonic_resonance_energies
    assert 6 in model.eng_flexomagnetoplasmonic_resonance_energies


def test_m361_eng_flexomagnetoplasmonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOMAGNETOPLASMONIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m361_lagmul_waldron_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{471:>10d}{472:>10d}{473:>10d}{4.25e7:>20.1f}{130:>10d}{3.0e-5:>20.6e}"
    c2 = f"{220.0:>20.4f}{210.0:>20.4f}{190.0:>20.4f}{140.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Waldron Spatial Linkage Joint Fixed Format Test
2022 0
/WALDRON_SPATIAL_LINKAGE_JOINT/435
Waldron Spatial Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 435 in model.lagmul_waldron_spatial_linkage_joints
    joint = model.lagmul_waldron_spatial_linkage_joints[435]
    assert joint.node1 == 471
    assert joint.node2 == 472
    assert joint.node3 == 473
    assert pytest.approx(joint.stiff) == 4.25e7
    assert joint.skew_id == 130
    assert pytest.approx(joint.tol) == 3.0e-5
    assert pytest.approx(joint.link_len_a) == 220.0
    assert pytest.approx(joint.link_len_b) == 210.0
    assert pytest.approx(joint.twist_angle_alpha) == 190.0
    assert pytest.approx(joint.offset_distance_s) == 140.0


def test_m361_lagmul_waldron_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Waldron Spatial Linkage Joint Free Format Test
/LAGMUL/WALDRON_SPATIAL_LINKAGE_JOINT/436
571, 572, 573, 28.0e6, 150, 4.0e-5
222.0, 212.0, 192.0, 142.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 436 in model.lagmul_waldron_spatial_linkage_joints
    joint = model.lagmul_waldron_spatial_linkage_joints[436]
    assert joint.node1 == 571
    assert joint.node2 == 572
    assert joint.node3 == 573
    assert pytest.approx(joint.stiff) == 28.0e6
    assert joint.skew_id == 150
    assert pytest.approx(joint.tol) == 4.0e-5
    assert pytest.approx(joint.link_len_a) == 222.0
    assert pytest.approx(joint.link_len_b) == 212.0
    assert pytest.approx(joint.twist_angle_alpha) == 192.0
    assert pytest.approx(joint.offset_distance_s) == 142.0


def test_m361_lagmul_waldron_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Waldron Spatial Linkage Joint Aliases Test
/LAGMUL/WALDRON_SPATIAL_LINKAGE/437
1, 2, 3, 1e6, 0, 1e-6
58.0, 58.0, 104.0, 44.0
/WALDRON_SPATIAL_LINKAGE/438
1, 2, 3, 1e6, 0, 1e-6
58.0, 58.0, 104.0, 44.0
/WALDRON_SPATIAL_MULTI_LOOP_MECHANISM/439
1, 2, 3, 1e6, 0, 1e-6
58.0, 58.0, 104.0, 44.0
/WALDRON_SPATIAL_HYBRID_MECHANISM/440
1, 2, 3, 1e6, 0, 1e-6
58.0, 58.0, 104.0, 44.0
/WALDRON_SPATIAL_6R_MECHANISM/441
1, 2, 3, 1e6, 0, 1e-6
58.0, 58.0, 104.0, 44.0
/WALDRON_SPATIAL_OVERCONSTRAINED_MECHANISM/442
1, 2, 3, 1e6, 0, 1e-6
58.0, 58.0, 104.0, 44.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 437 in model.lagmul_waldron_spatial_linkage_joints
    assert 438 in model.lagmul_waldron_spatial_linkage_joints
    assert 439 in model.lagmul_waldron_spatial_linkage_joints
    assert 440 in model.lagmul_waldron_spatial_linkage_joints
    assert 441 in model.lagmul_waldron_spatial_linkage_joints
    assert 442 in model.lagmul_waldron_spatial_linkage_joints


def test_m361_lagmul_waldron_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/WALDRON_SPATIAL_LINKAGE_JOINT/443
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m361_sensor_spring_normal_crackle_rate_fixed(tmp_path: Path):
    c1 = f"{1215:>10d}{7.45e9:>20.1f}{0.0335:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Normal Crackle Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_NORMAL_CRACKLE_RATE/468
Spring Normal Crackle Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 468 in model.sensor_spring_normal_crackle_rates
    sensor = model.sensor_spring_normal_crackle_rates[468]
    assert sensor.spring_id == 1215
    assert pytest.approx(sensor.jnorm_crk_max) == 7.45e9
    assert pytest.approx(sensor.t_delay) == 0.0335
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_NORMAL_CRACKLE_RATE"


def test_m361_sensor_spring_normal_crackle_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Normal Crackle Rate Sensor Free Format Test
/SENSOR/SPRING_NORMAL_CRACKLE_RATE/469
1216, 7.55e9, 0.0360
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 469 in model.sensor_spring_normal_crackle_rates
    sensor = model.sensor_spring_normal_crackle_rates[469]
    assert sensor.spring_id == 1216
    assert pytest.approx(sensor.jnorm_crk_max) == 7.55e9
    assert pytest.approx(sensor.t_delay) == 0.0360


def test_m361_sensor_spring_normal_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Normal Crackle Rate Sensor Aliases Test
/SENSOR/SPRING_NORM_CRACKLE_RATE/470
1217, 7.6e9, 0.0265
/SENSOR/SPRING_RATE_CRACKLE_NORM/471
1218, 7.6e9, 0.0265
/SENSOR/NORMAL_CRACKLE_RATE_SPRING/472
1219, 7.6e9, 0.0265
/SENSOR/SPRING_CRACKLE_NORM/473
1220, 7.6e9, 0.0265
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 470 in model.sensor_spring_normal_crackle_rates
    assert 471 in model.sensor_spring_normal_crackle_rates
    assert 472 in model.sensor_spring_normal_crackle_rates
    assert 473 in model.sensor_spring_normal_crackle_rates
    assert len(model.sensors) == 4


def test_m361_sensor_spring_normal_crackle_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_NORMAL_CRACKLE_RATE/474
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
