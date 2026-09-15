"""Tests for Milestone M396: LadTransverseDelaminationRate Failure Model, EngFlexothermoplasmonicexcitonicpolaritonicResonanceEnergy, DieselSpatialLinkageJoint, and SensorSpringTotalAngularShotRate."""

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


def test_m396_fail_lad_transverse_delamination_rate_fixed(tmp_path: Path):
    c1 = f"{530.0:>20.4f}{1590.0:>20.4f}{295.0:>20.4f}{6.85:>20.4f}{0.915:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2100:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Delamination Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_DELAMINATION_RATE/2100
Ladeveze Transverse Delamination Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2100 in model.fail_ladtransversedelaminationrates
    ftdlr = model.fail_ladtransversedelaminationrates[2100]
    assert pytest.approx(ftdlr.sigma_tdlr0) == 530.0
    assert pytest.approx(ftdlr.sigma_tdlrc) == 1590.0
    assert pytest.approx(ftdlr.gamma_tdlr) == 295.0
    assert pytest.approx(ftdlr.p_tdlr) == 6.85
    assert pytest.approx(ftdlr.d_tdlr_max) == 0.915
    assert ftdlr.ifail_sh == 1
    assert ftdlr.ifail_so == 2
    assert ftdlr.fail_id == 2100
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_DELAMINATION_RATE"


def test_m396_fail_lad_transverse_delamination_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Delamination Rate Free Format Test
/FAIL/LAD_TRANSVERSE_DELAMINATION_RATE/2101
540.0, 1620.0, 305.0, 7.05, 0.895
1, 1
2101
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2101 in model.fail_ladtransversedelaminationrates
    ftdlr = model.fail_ladtransversedelaminationrates[2101]
    assert pytest.approx(ftdlr.sigma_tdlr0) == 540.0
    assert pytest.approx(ftdlr.sigma_tdlrc) == 1620.0
    assert pytest.approx(ftdlr.gamma_tdlr) == 305.0
    assert pytest.approx(ftdlr.p_tdlr) == 7.05
    assert pytest.approx(ftdlr.d_tdlr_max) == 0.895
    assert ftdlr.fail_id == 2101


def test_m396_fail_lad_transverse_delamination_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Delamination Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_DELAMINATION_RATE/2102
520.0, 1560.0, 290.0, 6.65, 0.925
1, 1
/FAIL/LAD_TDLR/2103
520.0, 1560.0, 290.0, 6.65, 0.925
1, 1
/FAIL/LAD_TDLR_MODEL/2104
520.0, 1560.0, 290.0, 6.65, 0.925
1, 1
/FAIL/LAD_TDLR_LAW/2105
520.0, 1560.0, 290.0, 6.65, 0.925
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_DELAMINATION/2106
520.0, 1560.0, 290.0, 6.65, 0.925
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2102 in model.fail_ladtransversedelaminationrates
    assert 2103 in model.fail_ladtransversedelaminationrates
    assert 2104 in model.fail_ladtransversedelaminationrates
    assert 2105 in model.fail_ladtransversedelaminationrates
    assert 2106 in model.fail_ladtransversedelaminationrates
    assert len(model.raw_fails) == 5


def test_m396_fail_lad_transverse_delamination_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_TRANSVERSE_DELAMINATION_RATE/2107
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m396_eng_flexothermoplasmonicexcitonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0185:>20.4f}{405:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexothermoplasmonicexcitonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY/305
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 305 in model.eng_flexothermoplasmonicexcitonicpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonicexcitonicpolaritonic_resonance_energies[305]
    assert pytest.approx(eng.dt_ftpep) == 0.0185
    assert eng.sens_id == 405


def test_m396_eng_flexothermoplasmonicexcitonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexothermoplasmonicexcitonicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY/306
0.0195, 406
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 306 in model.eng_flexothermoplasmonicexcitonicpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonicexcitonicpolaritonic_resonance_energies[306]
    assert pytest.approx(eng.dt_ftpep) == 0.0195
    assert eng.sens_id == 406


def test_m396_eng_flexothermoplasmonicexcitonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexothermoplasmonicexcitonicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_PLASMONIC_EXCITONIC_POLARITON_RES_WORK/307
0.0205, 407
/ENG/EFLEXOTHERMOPLASMONICEXCITONICPOLARITONICRESONANCE/308
0.0215, 408
/ENG/FLEXOTHERMOPLASMONICEXCITONICPOLARITONIC_RESONANCE_DISSIPATION/309
0.0225, 409
/ENG/ET_FLEXOTHERMOPLASMONICEXCITONICPOLARITONIC_RESONANCE/310
0.0235, 410
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 307 in model.eng_flexothermoplasmonicexcitonicpolaritonic_resonance_energies
    assert 308 in model.eng_flexothermoplasmonicexcitonicpolaritonic_resonance_energies
    assert 309 in model.eng_flexothermoplasmonicexcitonicpolaritonic_resonance_energies
    assert 310 in model.eng_flexothermoplasmonicexcitonicpolaritonic_resonance_energies


def test_m396_eng_flexothermoplasmonicexcitonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOTHERMOPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY/311
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m396_lagmul_diesel_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{391:>10d}{392:>10d}{393:>10d}{2.46e7:>20.4f}{109:>10d}{1.45e-4:>20.4e}"
    c2 = f"{175.0:>20.4f}{158.0:>20.4f}{185.0:>20.4f}{108.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Diesel Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/DIESEL_SPATIAL_LINKAGE_JOINT/355
Diesel Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 355 in model.lagmul_diesel_spatial_linkage_joints
    joint = model.lagmul_diesel_spatial_linkage_joints[355]
    assert joint.node1 == 391
    assert joint.node2 == 392
    assert joint.node3 == 393
    assert pytest.approx(joint.stiff) == 2.46e7
    assert joint.skew_id == 109
    assert pytest.approx(joint.tol) == 1.45e-4
    assert pytest.approx(joint.link_len_a) == 175.0
    assert pytest.approx(joint.link_len_b) == 158.0
    assert pytest.approx(joint.twist_angle_alpha) == 185.0
    assert pytest.approx(joint.offset_distance_s) == 108.0
    assert pytest.approx(joint.offset_distance_r) == 108.0
    assert pytest.approx(joint.offset_distance_v) == 108.0
    assert pytest.approx(joint.offset_distance_h) == 108.0
    assert pytest.approx(joint.offset_distance_u) == 108.0


def test_m396_lagmul_diesel_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Diesel Spatial Linkage Joint Free Format Test
/LAGMUL/DIESEL_SPATIAL_LINKAGE_JOINT/356
491, 492, 493, 2.70e7, 110, 1.55e-4
195.0, 165.0, 205.0, 120.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 356 in model.lagmul_diesel_spatial_linkage_joints
    joint = model.lagmul_diesel_spatial_linkage_joints[356]
    assert joint.node1 == 491
    assert joint.node2 == 492
    assert joint.node3 == 493
    assert pytest.approx(joint.stiff) == 2.70e7
    assert joint.skew_id == 110
    assert pytest.approx(joint.tol) == 1.55e-4
    assert pytest.approx(joint.link_len_a) == 195.0
    assert pytest.approx(joint.link_len_b) == 165.0
    assert pytest.approx(joint.twist_angle_alpha) == 205.0
    assert pytest.approx(joint.offset_distance_s) == 120.0


def test_m396_lagmul_diesel_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Diesel Spatial Linkage Joint Aliases Test
/DIESEL_SPATIAL_LINKAGE_JOINT/357
591, 592, 593, 1.0e6, 0, 1.0e-6
135.0, 125.0, 135.0, 85.0
/LAGMUL/DIESEL_SPATIAL_LINKAGE/358
591, 592, 593, 1.0e6, 0, 1.0e-6
135.0, 125.0, 135.0, 85.0
/DIESEL_SPATIAL_LINKAGE/359
591, 592, 593, 1.0e6, 0, 1.0e-6
135.0, 125.0, 135.0, 85.0
/DIESEL_SPATIAL_MULTI_LOOP_MECHANISM/360
591, 592, 593, 1.0e6, 0, 1.0e-6
135.0, 125.0, 135.0, 85.0
/DIESEL_SPATIAL_SYMMETRIC_MECHANISM/361
591, 592, 593, 1.0e6, 0, 1.0e-6
135.0, 125.0, 135.0, 85.0
/DIESEL_SPATIAL_6R_MECHANISM/362
591, 592, 593, 1.0e6, 0, 1.0e-6
135.0, 125.0, 135.0, 85.0
/DIESEL_SPATIAL_OVERCONSTRAINED_MECHANISM/363
591, 592, 593, 1.0e6, 0, 1.0e-6
135.0, 125.0, 135.0, 85.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(357, 364):
        assert jid in model.lagmul_diesel_spatial_linkage_joints


def test_m396_lagmul_diesel_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/DIESEL_SPATIAL_LINKAGE_JOINT/364
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m396_sensor_spring_total_angular_shot_rate_fixed(tmp_path: Path):
    c1 = f"{1438:>10d}{3.68e8:>20.4f}{0.1145:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Angular Shot Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_SHOT_RATE/1
Fixed Spring Total Angular Shot Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_angular_shot_rates
    s1 = model.sensor_spring_total_angular_shot_rates[1]
    assert s1.spring_id == 1438
    assert pytest.approx(s1.jtot_ang_shot_max) == 3.68e8
    assert pytest.approx(s1.jtot_ang_drop_max) == 3.68e8
    assert pytest.approx(s1.jtot_ang_lock_max) == 3.68e8
    assert pytest.approx(s1.jtot_ang_pop_max) == 3.68e8
    assert pytest.approx(s1.jtot_ang_snp_max) == 3.68e8
    assert pytest.approx(s1.jtot_ang_crackle_max) == 3.68e8
    assert pytest.approx(s1.t_delay) == 0.1145
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_SHOT_RATE"


def test_m396_sensor_spring_total_angular_shot_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Angular Shot Rate Free Format Test
/SENSOR/SPRING_TOTAL_ANGULAR_SHOT_RATE/2
Free Spring Total Angular Shot Rate Sensor
1439, 4.28e8, 0.1595
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_angular_shot_rates
    s2 = model.sensor_spring_total_angular_shot_rates[2]
    assert s2.spring_id == 1439
    assert pytest.approx(s2.jtot_ang_shot_max) == 4.28e8
    assert pytest.approx(s2.t_delay) == 0.1595


def test_m396_sensor_spring_total_angular_shot_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Angular Shot Rate Aliases Test
/SENSOR/SPRING_TOT_ANG_SHOT_RATE/391
1459, 1.895e8, 0.0765
/SENSOR/SPRING_RATE_SHOT_ANG_TOT/392
1460, 1.915e8, 0.0775
/SENSOR/TOTAL_ANGULAR_SHOT_RATE_SPRING/393
1461, 1.935e8, 0.0785
/SENSOR/SPRING_SHOT_ANG_TOT/394
1462, 1.955e8, 0.0795
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(391, 395):
        assert sid in model.sensor_spring_total_angular_shot_rates


def test_m396_sensor_spring_total_angular_shot_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TOTAL_ANGULAR_SHOT_RATE/395
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
