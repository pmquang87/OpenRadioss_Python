"""Tests for Milestone M407: LadDynamicCoreShearingRate Failure Model, EngFlexothermoexcitonicmagnonicphononicpolaritonicResonanceEnergy, GaussSpatialLinkageJoint, and SensorSpringBendingCrackleRate."""

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


def test_m407_fail_lad_dynamic_core_shearing_rate_fixed(tmp_path: Path):
    c1 = f"{595.0:>20.4f}{1785.0:>20.4f}{375.0:>20.4f}{7.85:>20.4f}{0.825:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2210:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Core Shearing Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_CORE_SHEARING_RATE/2210
Ladeveze Dynamic Core Shearing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2210 in model.fail_laddynamiccoreshearingrates
    fdcsr = model.fail_laddynamiccoreshearingrates[2210]
    assert pytest.approx(fdcsr.sigma_dcsr0) == 595.0
    assert pytest.approx(fdcsr.sigma_dcsrc) == 1785.0
    assert pytest.approx(fdcsr.gamma_dcsr) == 375.0
    assert pytest.approx(fdcsr.p_dcsr) == 7.85
    assert pytest.approx(fdcsr.d_dcsr_max) == 0.825
    assert fdcsr.ifail_sh == 1
    assert fdcsr.ifail_so == 2
    assert fdcsr.fail_id == 2210
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_CORE_SHEARING_RATE"


def test_m407_fail_lad_dynamic_core_shearing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Core Shearing Rate Free Format Test
/FAIL/LAD_DYNAMIC_CORE_SHEARING_RATE/2211
605.0, 1815.0, 385.0, 8.05, 0.815
1, 1
2211
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2211 in model.fail_laddynamiccoreshearingrates
    fdcsr = model.fail_laddynamiccoreshearingrates[2211]
    assert pytest.approx(fdcsr.sigma_dcsr0) == 605.0
    assert pytest.approx(fdcsr.sigma_dcsrc) == 1815.0
    assert pytest.approx(fdcsr.gamma_dcsr) == 385.0
    assert pytest.approx(fdcsr.p_dcsr) == 8.05
    assert pytest.approx(fdcsr.d_dcsr_max) == 0.815
    assert fdcsr.fail_id == 2211


def test_m407_fail_lad_dynamic_core_shearing_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Core Shearing Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_CORE_SHEARING_RATE/2212
585.0, 1755.0, 370.0, 7.65, 0.845
1, 1
/FAIL/LAD_DCSR/2213
585.0, 1755.0, 370.0, 7.65, 0.845
1, 1
/FAIL/LAD_DCSR_MODEL/2214
585.0, 1755.0, 370.0, 7.65, 0.845
1, 1
/FAIL/LAD_DCSR_LAW/2215
585.0, 1755.0, 370.0, 7.65, 0.845
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_CORE_SHEARING/2216
585.0, 1755.0, 370.0, 7.65, 0.845
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2212 in model.fail_laddynamiccoreshearingrates
    assert 2213 in model.fail_laddynamiccoreshearingrates
    assert 2214 in model.fail_laddynamiccoreshearingrates
    assert 2215 in model.fail_laddynamiccoreshearingrates
    assert 2216 in model.fail_laddynamiccoreshearingrates
    assert len(model.raw_fails) == 5


def test_m407_fail_lad_dynamic_core_shearing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_CORE_SHEARING_RATE/2217
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m407_eng_flexothermoexcitonicmagnonicphononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0295:>20.4f}{515:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexothermoexcitonicmagnonicphononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/415
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 415 in model.eng_flexothermoexcitonicmagnonicphononicpolaritonic_resonance_energies
    eng = model.eng_flexothermoexcitonicmagnonicphononicpolaritonic_resonance_energies[415]
    assert pytest.approx(eng.dt_ftempp) == 0.0295
    assert eng.sens_id == 515


def test_m407_eng_flexothermoexcitonicmagnonicphononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexothermoexcitonicmagnonicphononicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/416
0.0305, 516
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 416 in model.eng_flexothermoexcitonicmagnonicphononicpolaritonic_resonance_energies
    eng = model.eng_flexothermoexcitonicmagnonicphononicpolaritonic_resonance_energies[416]
    assert pytest.approx(eng.dt_ftempp) == 0.0305
    assert eng.sens_id == 516


def test_m407_eng_flexothermoexcitonicmagnonicphononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexothermoexcitonicmagnonicphononicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_EXCITONIC_MAGNONIC_PHONONIC_POLARITON_RES_WORK/417
0.0315, 517
/ENG/EFLEXOTHERMOEXCITONICMAGNONICPHONONICPOLARITONICRESONANCE/418
0.0325, 518
/ENG/FLEXOTHERMOEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_DISSIPATION/419
0.0335, 519
/ENG/ET_FLEXOTHERMOEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE/420
0.0345, 520
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 417 in model.eng_flexothermoexcitonicmagnonicphononicpolaritonic_resonance_energies
    assert 418 in model.eng_flexothermoexcitonicmagnonicphononicpolaritonic_resonance_energies
    assert 419 in model.eng_flexothermoexcitonicmagnonicphononicpolaritonic_resonance_energies
    assert 420 in model.eng_flexothermoexcitonicmagnonicphononicpolaritonic_resonance_energies


def test_m407_eng_flexothermoexcitonicmagnonicphononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOTHERMOEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/421
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m407_lagmul_gauss_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{501:>10d}{502:>10d}{503:>10d}{3.56e7:>20.4f}{219:>10d}{2.55e-4:>20.4e}"
    c2 = f"{285.0:>20.4f}{268.0:>20.4f}{295.0:>20.4f}{218.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Gauss Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/GAUSS_SPATIAL_LINKAGE_JOINT/465
Gauss Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 465 in model.lagmul_gauss_spatial_linkage_joints
    joint = model.lagmul_gauss_spatial_linkage_joints[465]
    assert joint.node1 == 501
    assert joint.node2 == 502
    assert joint.node3 == 503
    assert pytest.approx(joint.stiff) == 3.56e7
    assert joint.skew_id == 219
    assert pytest.approx(joint.tol) == 2.55e-4
    assert pytest.approx(joint.link_len_a) == 285.0
    assert pytest.approx(joint.link_len_b) == 268.0
    assert pytest.approx(joint.twist_angle_alpha) == 295.0
    assert pytest.approx(joint.offset_distance_s) == 218.0
    assert pytest.approx(joint.offset_distance_r) == 218.0
    assert pytest.approx(joint.offset_distance_v) == 218.0
    assert pytest.approx(joint.offset_distance_h) == 218.0
    assert pytest.approx(joint.offset_distance_u) == 218.0


def test_m407_lagmul_gauss_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Gauss Spatial Linkage Joint Free Format Test
/LAGMUL/GAUSS_SPATIAL_LINKAGE_JOINT/466
601, 602, 603, 3.80e7, 220, 2.65e-4
305.0, 275.0, 315.0, 230.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 466 in model.lagmul_gauss_spatial_linkage_joints
    joint = model.lagmul_gauss_spatial_linkage_joints[466]
    assert joint.node1 == 601
    assert joint.node2 == 602
    assert joint.node3 == 603
    assert pytest.approx(joint.stiff) == 3.80e7
    assert joint.skew_id == 220
    assert pytest.approx(joint.tol) == 2.65e-4
    assert pytest.approx(joint.link_len_a) == 305.0
    assert pytest.approx(joint.link_len_b) == 275.0
    assert pytest.approx(joint.twist_angle_alpha) == 315.0
    assert pytest.approx(joint.offset_distance_s) == 230.0


def test_m407_lagmul_gauss_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Gauss Spatial Linkage Joint Aliases Test
/GAUSS_SPATIAL_LINKAGE_JOINT/467
701, 702, 703, 1.0e6, 0, 1.0e-6
245.0, 235.0, 245.0, 195.0
/LAGMUL/GAUSS_SPATIAL_LINKAGE/468
701, 702, 703, 1.0e6, 0, 1.0e-6
245.0, 235.0, 245.0, 195.0
/GAUSS_SPATIAL_LINKAGE/469
701, 702, 703, 1.0e6, 0, 1.0e-6
245.0, 235.0, 245.0, 195.0
/GAUSS_SPATIAL_MULTI_LOOP_MECHANISM/470
701, 702, 703, 1.0e6, 0, 1.0e-6
245.0, 235.0, 245.0, 195.0
/GAUSS_SPATIAL_SYMMETRIC_MECHANISM/471
701, 702, 703, 1.0e6, 0, 1.0e-6
245.0, 235.0, 245.0, 195.0
/GAUSS_SPATIAL_6R_MECHANISM/472
701, 702, 703, 1.0e6, 0, 1.0e-6
245.0, 235.0, 245.0, 195.0
/GAUSS_SPATIAL_OVERCONSTRAINED_MECHANISM/473
701, 702, 703, 1.0e6, 0, 1.0e-6
245.0, 235.0, 245.0, 195.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(467, 474):
        assert jid in model.lagmul_gauss_spatial_linkage_joints


def test_m407_lagmul_gauss_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/GAUSS_SPATIAL_LINKAGE_JOINT/474
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m407_sensor_spring_bending_crackle_rate_fixed(tmp_path: Path):
    c1 = f"{1548:>10d}{4.78e8:>20.4f}{0.1525:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Bending Crackle Rate Fixed Format Test
2022 0
/SENSOR/SPRING_BENDING_CRACKLE_RATE/1
Fixed Spring Bending Crackle Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_bending_crackle_rates
    s1 = model.sensor_spring_bending_crackle_rates[1]
    assert s1.spring_id == 1548
    assert pytest.approx(s1.jbend_crk_max) == 4.78e8
    assert pytest.approx(s1.jbend_shot_max) == 4.78e8
    assert pytest.approx(s1.jbend_drop_max) == 4.78e8
    assert pytest.approx(s1.jbend_lock_max) == 4.78e8
    assert pytest.approx(s1.jbend_pop_max) == 4.78e8
    assert pytest.approx(s1.jbend_snp_max) == 4.78e8
    assert pytest.approx(s1.t_delay) == 0.1525
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_BENDING_CRACKLE_RATE"


def test_m407_sensor_spring_bending_crackle_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Bending Crackle Rate Free Format Test
/SENSOR/SPRING_BENDING_CRACKLE_RATE/2
Free Spring Bending Crackle Rate Sensor
1549, 5.38e8, 0.2515
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_bending_crackle_rates
    s2 = model.sensor_spring_bending_crackle_rates[2]
    assert s2.spring_id == 1549
    assert pytest.approx(s2.jbend_crk_max) == 5.38e8
    assert pytest.approx(s2.t_delay) == 0.2515


def test_m407_sensor_spring_bending_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Bending Crackle Rate Aliases Test
/SENSOR/SPRING_BEND_CRACKLE_RATE/411
1569, 2.995e8, 0.1865
/SENSOR/SPRING_RATE_CRACKLE_BEND/412
1570, 3.015e8, 0.1875
/SENSOR/BENDING_CRACKLE_RATE_SPRING/413
1571, 3.035e8, 0.1885
/SENSOR/SPRING_CRACKLE_BEND/414
1572, 3.055e8, 0.1895
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(411, 415):
        assert sid in model.sensor_spring_bending_crackle_rates


def test_m407_sensor_spring_bending_crackle_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_BENDING_CRACKLE_RATE/415
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
