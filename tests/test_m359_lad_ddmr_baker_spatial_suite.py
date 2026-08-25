"""Tests for Milestone M359: LadDynamicDelaminationMicrodebondingRate Failure Model, EngFlexothermoexcitonphononmagnonpolaritonicResonanceEnergy, BakerSpatialLinkageJoint, and SensorSpringBendingPopRate."""

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


def test_m359_fail_lad_dynamic_delamination_microdebonding_rate_fixed(tmp_path: Path):
    c1 = f"{142.0:>20.4f}{426.0:>20.4f}{54.0:>20.4f}{2.45:>20.4f}{0.982:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1750:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Delamination Micro-Debonding Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_DELAMINATION_MICRODEBONDING_RATE/1750
Ladeveze Dynamic Delamination Microdebonding Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1750 in model.fail_laddynamicdelaminationmicrodebondingrates
    fddmr = model.fail_laddynamicdelaminationmicrodebondingrates[1750]
    assert pytest.approx(fddmr.sigma_ddmr0) == 142.0
    assert pytest.approx(fddmr.sigma_ddmrc) == 426.0
    assert pytest.approx(fddmr.gamma_ddmr) == 54.0
    assert pytest.approx(fddmr.p_ddmr) == 2.45
    assert pytest.approx(fddmr.d_ddmr_max) == 0.982
    assert fddmr.ifail_sh == 1
    assert fddmr.ifail_so == 2
    assert fddmr.fail_id == 1750
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_DELAMINATION_MICRODEBONDING_RATE"


def test_m359_fail_lad_dynamic_delamination_microdebonding_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Delamination Micro-Debonding Rate Free Format Test
/FAIL/LAD_DYNAMIC_DELAMINATION_MICRODEBONDING_RATE/1751
152.0, 456.0, 60.0, 2.65, 0.968
1, 1
1751
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1751 in model.fail_laddynamicdelaminationmicrodebondingrates
    fddmr = model.fail_laddynamicdelaminationmicrodebondingrates[1751]
    assert pytest.approx(fddmr.sigma_ddmr0) == 152.0
    assert pytest.approx(fddmr.sigma_ddmrc) == 456.0
    assert pytest.approx(fddmr.gamma_ddmr) == 60.0
    assert pytest.approx(fddmr.p_ddmr) == 2.65
    assert pytest.approx(fddmr.d_ddmr_max) == 0.968
    assert fddmr.fail_id == 1751


def test_m359_fail_lad_dynamic_delamination_microdebonding_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Delamination Micro-Debonding Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_DELAMINATION_MICRODEBONDING_RATE/1752
132.0, 396.0, 48.0, 2.12, 0.992
1, 1
/FAIL/LAD_DDMR/1753
132.0, 396.0, 48.0, 2.12, 0.992
1, 1
/FAIL/LAD_DDMR_MODEL/1754
132.0, 396.0, 48.0, 2.12, 0.992
1, 1
/FAIL/LAD_DDMR_LAW/1755
132.0, 396.0, 48.0, 2.12, 0.992
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_DELAMINATION_MICRODEBONDING/1756
132.0, 396.0, 48.0, 2.12, 0.992
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1752 in model.fail_laddynamicdelaminationmicrodebondingrates
    assert 1753 in model.fail_laddynamicdelaminationmicrodebondingrates
    assert 1754 in model.fail_laddynamicdelaminationmicrodebondingrates
    assert 1755 in model.fail_laddynamicdelaminationmicrodebondingrates
    assert 1756 in model.fail_laddynamicdelaminationmicrodebondingrates
    assert len(model.raw_fails) == 5


def test_m359_fail_lad_dynamic_delamination_microdebonding_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_DYNAMIC_DELAMINATION_MICRODEBONDING_RATE/1757
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m359_eng_flexothermoexcitonphononmagnonpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00155:>20.6f}{245:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermoexcitonphononmagnonpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOEXCITONPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY/1
Flexothermoexcitonphononmagnonpolaritonic Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermoexcitonphononmagnonpolaritonic_resonance_energies
    eng = model.eng_flexothermoexcitonphononmagnonpolaritonic_resonance_energies[1]
    assert pytest.approx(eng.dt_ftepmpr) == 0.00155
    assert eng.sens_id == 245


def test_m359_eng_flexothermoexcitonphononmagnonpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Flexothermoexcitonphononmagnonpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOEXCITONPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY/2
0.00175, 255
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_flexothermoexcitonphononmagnonpolaritonic_resonance_energies
    eng = model.eng_flexothermoexcitonphononmagnonpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_ftepmpr) == 0.00175
    assert eng.sens_id == 255


def test_m359_eng_flexothermoexcitonphononmagnonpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermoexcitonphononmagnonpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_EXCITON_PHONON_MAGNON_POLARITON_RES_WORK/3
0.00115, 182
/ENG/EFLEXOTHERMOEXCITONPHONONMAGNONPOLARITONICRESONANCE/4
0.00119, 188
/ENG/FLEXOTHERMOEXCITONPHONONMAGNONPOLARITONIC_RESONANCE_DISSIPATION/5
0.00125, 198
/ENG/EM_FLEXOTHERMOEXCITONPHONONMAGNONPOLARITONIC_RESONANCE/6
0.00132, 208
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermoexcitonphononmagnonpolaritonic_resonance_energies
    assert 4 in model.eng_flexothermoexcitonphononmagnonpolaritonic_resonance_energies
    assert 5 in model.eng_flexothermoexcitonphononmagnonpolaritonic_resonance_energies
    assert 6 in model.eng_flexothermoexcitonphononmagnonpolaritonic_resonance_energies


def test_m359_eng_flexothermoexcitonphononmagnonpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMOEXCITONPHONONMAGNONPOLARITONIC_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m359_lagmul_baker_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{451:>10d}{452:>10d}{453:>10d}{4.05e7:>20.1f}{126:>10d}{3.4e-5:>20.6e}"
    c2 = f"{210.0:>20.4f}{200.0:>20.4f}{180.0:>20.4f}{130.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Baker Spatial Linkage Joint Fixed Format Test
2022 0
/BAKER_SPATIAL_LINKAGE_JOINT/415
Baker Spatial Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 415 in model.lagmul_baker_spatial_linkage_joints
    joint = model.lagmul_baker_spatial_linkage_joints[415]
    assert joint.node1 == 451
    assert joint.node2 == 452
    assert joint.node3 == 453
    assert pytest.approx(joint.stiff) == 4.05e7
    assert joint.skew_id == 126
    assert pytest.approx(joint.tol) == 3.4e-5
    assert pytest.approx(joint.link_len_a) == 210.0
    assert pytest.approx(joint.link_len_b) == 200.0
    assert pytest.approx(joint.twist_angle_alpha) == 180.0
    assert pytest.approx(joint.offset_distance_s) == 130.0


def test_m359_lagmul_baker_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Baker Spatial Linkage Joint Free Format Test
/LAGMUL/BAKER_SPATIAL_LINKAGE_JOINT/416
551, 552, 553, 26.0e6, 146, 4.4e-5
212.0, 202.0, 182.0, 132.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 416 in model.lagmul_baker_spatial_linkage_joints
    joint = model.lagmul_baker_spatial_linkage_joints[416]
    assert joint.node1 == 551
    assert joint.node2 == 552
    assert joint.node3 == 553
    assert pytest.approx(joint.stiff) == 26.0e6
    assert joint.skew_id == 146
    assert pytest.approx(joint.tol) == 4.4e-5
    assert pytest.approx(joint.link_len_a) == 212.0
    assert pytest.approx(joint.link_len_b) == 202.0
    assert pytest.approx(joint.twist_angle_alpha) == 182.0
    assert pytest.approx(joint.offset_distance_s) == 132.0


def test_m359_lagmul_baker_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Baker Spatial Linkage Joint Aliases Test
/LAGMUL/BAKER_SPATIAL_LINKAGE/417
1, 2, 3, 1e6, 0, 1e-6
54.0, 54.0, 100.0, 40.0
/BAKER_SPATIAL_LINKAGE/418
1, 2, 3, 1e6, 0, 1e-6
54.0, 54.0, 100.0, 40.0
/BAKER_SPATIAL_MULTI_LOOP_MECHANISM/419
1, 2, 3, 1e6, 0, 1e-6
54.0, 54.0, 100.0, 40.0
/BAKER_SPATIAL_SYMMETRIC_MECHANISM/420
1, 2, 3, 1e6, 0, 1e-6
54.0, 54.0, 100.0, 40.0
/BAKER_SPATIAL_6R_MECHANISM/421
1, 2, 3, 1e6, 0, 1e-6
54.0, 54.0, 100.0, 40.0
/BAKER_SPATIAL_OVERCONSTRAINED_MECHANISM/422
1, 2, 3, 1e6, 0, 1e-6
54.0, 54.0, 100.0, 40.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 417 in model.lagmul_baker_spatial_linkage_joints
    assert 418 in model.lagmul_baker_spatial_linkage_joints
    assert 419 in model.lagmul_baker_spatial_linkage_joints
    assert 420 in model.lagmul_baker_spatial_linkage_joints
    assert 421 in model.lagmul_baker_spatial_linkage_joints
    assert 422 in model.lagmul_baker_spatial_linkage_joints


def test_m359_lagmul_baker_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/BAKER_SPATIAL_LINKAGE_JOINT/423
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m359_sensor_spring_bending_pop_rate_fixed(tmp_path: Path):
    c1 = f"{1195:>10d}{7.05e9:>20.1f}{0.0315:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Bending Pop Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_BENDING_POP_RATE/448
Spring Bending Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 448 in model.sensor_spring_bending_pop_rates
    sensor = model.sensor_spring_bending_pop_rates[448]
    assert sensor.spring_id == 1195
    assert pytest.approx(sensor.jbend_pop_max) == 7.05e9
    assert pytest.approx(sensor.t_delay) == 0.0315
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_BENDING_POP_RATE"


def test_m359_sensor_spring_bending_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Bending Pop Rate Sensor Free Format Test
/SENSOR/SPRING_BENDING_POP_RATE/449
1196, 7.15e9, 0.0340
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 449 in model.sensor_spring_bending_pop_rates
    sensor = model.sensor_spring_bending_pop_rates[449]
    assert sensor.spring_id == 1196
    assert pytest.approx(sensor.jbend_pop_max) == 7.15e9
    assert pytest.approx(sensor.t_delay) == 0.0340


def test_m359_sensor_spring_bending_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Bending Pop Rate Sensor Aliases Test
/SENSOR/SPRING_BEND_POP_RATE/450
1197, 7.2e9, 0.0245
/SENSOR/SPRING_RATE_POP_BEND/451
1198, 7.2e9, 0.0245
/SENSOR/BENDING_POP_RATE_SPRING/452
1199, 7.2e9, 0.0245
/SENSOR/SPRING_POP_BEND/453
1200, 7.2e9, 0.0245
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 450 in model.sensor_spring_bending_pop_rates
    assert 451 in model.sensor_spring_bending_pop_rates
    assert 452 in model.sensor_spring_bending_pop_rates
    assert 453 in model.sensor_spring_bending_pop_rates
    assert len(model.sensors) == 4


def test_m359_sensor_spring_bending_pop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_BENDING_POP_RATE/454
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
