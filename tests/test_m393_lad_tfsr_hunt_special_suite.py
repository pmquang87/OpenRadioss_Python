"""Tests for Milestone M393: LadTransverseFiberSplittingRate Failure Model, EngFlexothermoplasmonicexcitonicmagnonicpolaritonicResonanceEnergy, HuntSpecialSpatialLinkageJoint, and SensorSpringTotalShotRate."""

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


def test_m393_fail_lad_transverse_fiber_splitting_rate_fixed(tmp_path: Path):
    c1 = f"{450.0:>20.4f}{1350.0:>20.4f}{235.0:>20.4f}{5.95:>20.4f}{0.955:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2070:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Fiber Splitting Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_FIBER_SPLITTING_RATE/2070
Ladeveze Transverse Fiber Splitting Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2070 in model.fail_ladtransversefibersplittingrates
    ftfsr = model.fail_ladtransversefibersplittingrates[2070]
    assert pytest.approx(ftfsr.sigma_tfsr0) == 450.0
    assert pytest.approx(ftfsr.sigma_tfsrc) == 1350.0
    assert pytest.approx(ftfsr.gamma_tfsr) == 235.0
    assert pytest.approx(ftfsr.p_tfsr) == 5.95
    assert pytest.approx(ftfsr.d_tfsr_max) == 0.955
    assert ftfsr.ifail_sh == 1
    assert ftfsr.ifail_so == 2
    assert ftfsr.fail_id == 2070
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_FIBER_SPLITTING_RATE"


def test_m393_fail_lad_transverse_fiber_splitting_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Fiber Splitting Rate Free Format Test
/FAIL/LAD_TRANSVERSE_FIBER_SPLITTING_RATE/2071
460.0, 1380.0, 245.0, 6.15, 0.935
1, 1
2071
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2071 in model.fail_ladtransversefibersplittingrates
    ftfsr = model.fail_ladtransversefibersplittingrates[2071]
    assert pytest.approx(ftfsr.sigma_tfsr0) == 460.0
    assert pytest.approx(ftfsr.sigma_tfsrc) == 1380.0
    assert pytest.approx(ftfsr.gamma_tfsr) == 245.0
    assert pytest.approx(ftfsr.p_tfsr) == 6.15
    assert pytest.approx(ftfsr.d_tfsr_max) == 0.935
    assert ftfsr.fail_id == 2071


def test_m393_fail_lad_transverse_fiber_splitting_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Fiber Splitting Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_FIBER_SPLITTING_RATE/2072
440.0, 1320.0, 230.0, 5.75, 0.965
1, 1
/FAIL/LAD_TFSR/2073
440.0, 1320.0, 230.0, 5.75, 0.965
1, 1
/FAIL/LAD_TFSR_MODEL/2074
440.0, 1320.0, 230.0, 5.75, 0.965
1, 1
/FAIL/LAD_TFSR_LAW/2075
440.0, 1320.0, 230.0, 5.75, 0.965
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_FIBER_SPLITTING/2076
440.0, 1320.0, 230.0, 5.75, 0.965
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2072 in model.fail_ladtransversefibersplittingrates
    assert 2073 in model.fail_ladtransversefibersplittingrates
    assert 2074 in model.fail_ladtransversefibersplittingrates
    assert 2075 in model.fail_ladtransversefibersplittingrates
    assert 2076 in model.fail_ladtransversefibersplittingrates
    assert len(model.raw_fails) == 5


def test_m393_fail_lad_transverse_fiber_splitting_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_TRANSVERSE_FIBER_SPLITTING_RATE/2077
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m393_eng_flexothermoplasmonicexcitonicmagnonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0145:>20.4f}{375:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexothermoplasmonicexcitonicmagnonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/275
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 275 in model.eng_flexothermoplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonicexcitonicmagnonicpolaritonic_resonance_energies[275]
    assert pytest.approx(eng.dt_ftpempr) == 0.0145
    assert eng.sens_id == 375


def test_m393_eng_flexothermoplasmonicexcitonicmagnonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexothermoplasmonicexcitonicmagnonicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/276
0.0155, 376
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 276 in model.eng_flexothermoplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonicexcitonicmagnonicpolaritonic_resonance_energies[276]
    assert pytest.approx(eng.dt_ftpempr) == 0.0155
    assert eng.sens_id == 376


def test_m393_eng_flexothermoplasmonicexcitonicmagnonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexothermoplasmonicexcitonicmagnonicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_PLASMON_EXCITON_MAGNON_POLARITON_RES_WORK/277
0.0165, 377
/ENG/EFLEXOTHERMOPLASMONICEXCITONICMAGNONICPOLARITONICRESONANCE/278
0.0175, 378
/ENG/FLEXOTHERMOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_DISSIPATION/279
0.0185, 379
/ENG/ET_FLEXOTHERMOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE/280
0.0195, 380
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 277 in model.eng_flexothermoplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    assert 278 in model.eng_flexothermoplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    assert 279 in model.eng_flexothermoplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    assert 280 in model.eng_flexothermoplasmonicexcitonicmagnonicpolaritonic_resonance_energies


def test_m393_eng_flexothermoplasmonicexcitonicmagnonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOTHERMOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/281
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m393_lagmul_hunt_special_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{361:>10d}{362:>10d}{363:>10d}{2.16e7:>20.4f}{79:>10d}{1.15e-4:>20.4e}"
    c2 = f"{145.0:>20.4f}{128.0:>20.4f}{155.0:>20.4f}{78.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Hunt Special Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/HUNT_SPECIAL_SPATIAL_LINKAGE_JOINT/325
Hunt Special Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 325 in model.lagmul_hunt_special_spatial_linkage_joints
    joint = model.lagmul_hunt_special_spatial_linkage_joints[325]
    assert joint.node1 == 361
    assert joint.node2 == 362
    assert joint.node3 == 363
    assert pytest.approx(joint.stiff) == 2.16e7
    assert joint.skew_id == 79
    assert pytest.approx(joint.tol) == 1.15e-4
    assert pytest.approx(joint.link_len_a) == 145.0
    assert pytest.approx(joint.link_len_b) == 128.0
    assert pytest.approx(joint.twist_angle_alpha) == 155.0
    assert pytest.approx(joint.offset_distance_s) == 78.0
    assert pytest.approx(joint.offset_distance_r) == 78.0
    assert pytest.approx(joint.offset_distance_v) == 78.0
    assert pytest.approx(joint.offset_distance_h) == 78.0
    assert pytest.approx(joint.offset_distance_u) == 78.0


def test_m393_lagmul_hunt_special_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Hunt Special Spatial Linkage Joint Free Format Test
/LAGMUL/HUNT_SPECIAL_SPATIAL_LINKAGE_JOINT/326
461, 462, 463, 2.40e7, 80, 1.25e-4
165.0, 135.0, 175.0, 90.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 326 in model.lagmul_hunt_special_spatial_linkage_joints
    joint = model.lagmul_hunt_special_spatial_linkage_joints[326]
    assert joint.node1 == 461
    assert joint.node2 == 462
    assert joint.node3 == 463
    assert pytest.approx(joint.stiff) == 2.40e7
    assert joint.skew_id == 80
    assert pytest.approx(joint.tol) == 1.25e-4
    assert pytest.approx(joint.link_len_a) == 165.0
    assert pytest.approx(joint.link_len_b) == 135.0
    assert pytest.approx(joint.twist_angle_alpha) == 175.0
    assert pytest.approx(joint.offset_distance_s) == 90.0


def test_m393_lagmul_hunt_special_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Hunt Special Spatial Linkage Joint Aliases Test
/HUNT_SPECIAL_SPATIAL_LINKAGE_JOINT/327
561, 562, 563, 1.0e6, 0, 1.0e-6
105.0, 95.0, 105.0, 55.0
/LAGMUL/HUNT_SPECIAL_SPATIAL_LINKAGE/328
561, 562, 563, 1.0e6, 0, 1.0e-6
105.0, 95.0, 105.0, 55.0
/HUNT_SPECIAL_SPATIAL_LINKAGE/329
561, 562, 563, 1.0e6, 0, 1.0e-6
105.0, 95.0, 105.0, 55.0
/HUNT_SPECIAL_SPATIAL_MULTI_LOOP_MECHANISM/330
561, 562, 563, 1.0e6, 0, 1.0e-6
105.0, 95.0, 105.0, 55.0
/HUNT_SPECIAL_SPATIAL_SYMMETRIC_MECHANISM/331
561, 562, 563, 1.0e6, 0, 1.0e-6
105.0, 95.0, 105.0, 55.0
/HUNT_SPECIAL_SPATIAL_6R_MECHANISM/332
561, 562, 563, 1.0e6, 0, 1.0e-6
105.0, 95.0, 105.0, 55.0
/HUNT_SPECIAL_SPATIAL_OVERCONSTRAINED_MECHANISM/333
561, 562, 563, 1.0e6, 0, 1.0e-6
105.0, 95.0, 105.0, 55.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(327, 334):
        assert jid in model.lagmul_hunt_special_spatial_linkage_joints


def test_m393_lagmul_hunt_special_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/HUNT_SPECIAL_SPATIAL_LINKAGE_JOINT/334
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m393_sensor_spring_total_shot_rate_fixed(tmp_path: Path):
    c1 = f"{1408:>10d}{3.38e8:>20.4f}{0.1025:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Shot Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_SHOT_RATE/1
Fixed Spring Total Shot Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_shot_rates
    s1 = model.sensor_spring_total_shot_rates[1]
    assert s1.spring_id == 1408
    assert pytest.approx(s1.jtot_shot_max) == 3.38e8
    assert pytest.approx(s1.jtot_drop_max) == 3.38e8
    assert pytest.approx(s1.jtot_lock_max) == 3.38e8
    assert pytest.approx(s1.jtot_pop_max) == 3.38e8
    assert pytest.approx(s1.jtot_snp_max) == 3.38e8
    assert pytest.approx(s1.jtot_crackle_max) == 3.38e8
    assert pytest.approx(s1.t_delay) == 0.1025
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_SHOT_RATE"


def test_m393_sensor_spring_total_shot_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Shot Rate Free Format Test
/SENSOR/SPRING_TOTAL_SHOT_RATE/2
Free Spring Total Shot Rate Sensor
1409, 3.98e8, 0.1295
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_shot_rates
    s2 = model.sensor_spring_total_shot_rates[2]
    assert s2.spring_id == 1409
    assert pytest.approx(s2.jtot_shot_max) == 3.98e8
    assert pytest.approx(s2.t_delay) == 0.1295


def test_m393_sensor_spring_total_shot_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Shot Rate Aliases Test
/SENSOR/SPRING_TOT_SHOT_RATE/391
1429, 1.595e8, 0.0465
/SENSOR/SPRING_RATE_SHOT_TOT/392
1430, 1.615e8, 0.0475
/SENSOR/TOTAL_SHOT_RATE_SPRING/393
1431, 1.635e8, 0.0485
/SENSOR/SPRING_SHOT_TOT/394
1432, 1.655e8, 0.0495
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(391, 395):
        assert sid in model.sensor_spring_total_shot_rates


def test_m393_sensor_spring_total_shot_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TOTAL_SHOT_RATE/395
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
