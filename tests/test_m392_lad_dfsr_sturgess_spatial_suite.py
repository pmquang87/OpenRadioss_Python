"""Tests for Milestone M392: LadDynamicFiberSplittingRate Failure Model, EngFlexothermophononicplasmonicexcitonicmagnonicpolaritonicResonanceEnergy, SturgessSpatialLinkageJoint, and SensorSpringTransverseShotRate."""

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


def test_m392_fail_lad_dynamic_fiber_splitting_rate_fixed(tmp_path: Path):
    c1 = f"{420.0:>20.4f}{1260.0:>20.4f}{215.0:>20.4f}{5.65:>20.4f}{0.965:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2060:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Fiber Splitting Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_FIBER_SPLITTING_RATE/2060
Ladeveze Dynamic Fiber Splitting Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2060 in model.fail_laddynamicfibersplittingrates
    fdfsr = model.fail_laddynamicfibersplittingrates[2060]
    assert pytest.approx(fdfsr.sigma_dfsr0) == 420.0
    assert pytest.approx(fdfsr.sigma_dfsrc) == 1260.0
    assert pytest.approx(fdfsr.gamma_dfsr) == 215.0
    assert pytest.approx(fdfsr.p_dfsr) == 5.65
    assert pytest.approx(fdfsr.d_dfsr_max) == 0.965
    assert fdfsr.ifail_sh == 1
    assert fdfsr.ifail_so == 2
    assert fdfsr.fail_id == 2060
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_FIBER_SPLITTING_RATE"


def test_m392_fail_lad_dynamic_fiber_splitting_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Fiber Splitting Rate Free Format Test
/FAIL/LAD_DYNAMIC_FIBER_SPLITTING_RATE/2061
430.0, 1290.0, 225.0, 5.85, 0.945
1, 1
2061
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2061 in model.fail_laddynamicfibersplittingrates
    fdfsr = model.fail_laddynamicfibersplittingrates[2061]
    assert pytest.approx(fdfsr.sigma_dfsr0) == 430.0
    assert pytest.approx(fdfsr.sigma_dfsrc) == 1290.0
    assert pytest.approx(fdfsr.gamma_dfsr) == 225.0
    assert pytest.approx(fdfsr.p_dfsr) == 5.85
    assert pytest.approx(fdfsr.d_dfsr_max) == 0.945
    assert fdfsr.fail_id == 2061


def test_m392_fail_lad_dynamic_fiber_splitting_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Fiber Splitting Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_FIBER_SPLITTING_RATE/2062
410.0, 1230.0, 210.0, 5.45, 0.975
1, 1
/FAIL/LAD_DFSR/2063
410.0, 1230.0, 210.0, 5.45, 0.975
1, 1
/FAIL/LAD_DFSR_MODEL/2064
410.0, 1230.0, 210.0, 5.45, 0.975
1, 1
/FAIL/LAD_DFSR_LAW/2065
410.0, 1230.0, 210.0, 5.45, 0.975
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_FIBER_SPLITTING/2066
410.0, 1230.0, 210.0, 5.45, 0.975
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2062 in model.fail_laddynamicfibersplittingrates
    assert 2063 in model.fail_laddynamicfibersplittingrates
    assert 2064 in model.fail_laddynamicfibersplittingrates
    assert 2065 in model.fail_laddynamicfibersplittingrates
    assert 2066 in model.fail_laddynamicfibersplittingrates
    assert len(model.raw_fails) == 5


def test_m392_fail_lad_dynamic_fiber_splitting_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_FIBER_SPLITTING_RATE/2067
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m392_eng_flexothermophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0125:>20.4f}{365:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexothermophononicplasmonicexcitonicmagnonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOPHONONICPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/265
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 265 in model.eng_flexothermophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_flexothermophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energies[265]
    assert pytest.approx(eng.dt_ftppempr) == 0.0125
    assert eng.sens_id == 365


def test_m392_eng_flexothermophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexothermophononicplasmonicexcitonicmagnonicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOPHONONICPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/266
0.0135, 366
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 266 in model.eng_flexothermophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_flexothermophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energies[266]
    assert pytest.approx(eng.dt_ftppempr) == 0.0135
    assert eng.sens_id == 366


def test_m392_eng_flexothermophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexothermophononicplasmonicexcitonicmagnonicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_PHONON_PLASMON_EXCITON_MAGNON_POLARITON_RES_WORK/267
0.0145, 367
/ENG/EFLEXOTHERMOPHONONICPLASMONICEXCITONICMAGNONICPOLARITONICRESONANCE/268
0.0155, 368
/ENG/FLEXOTHERMOPHONONICPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_DISSIPATION/269
0.0165, 369
/ENG/ET_FLEXOTHERMOPHONONICPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE/270
0.0175, 370
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 267 in model.eng_flexothermophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    assert 268 in model.eng_flexothermophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    assert 269 in model.eng_flexothermophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    assert 270 in model.eng_flexothermophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energies


def test_m392_eng_flexothermophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOTHERMOPHONONICPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/271
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m392_lagmul_sturgess_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{351:>10d}{352:>10d}{353:>10d}{2.06e7:>20.4f}{69:>10d}{1.05e-4:>20.4e}"
    c2 = f"{135.0:>20.4f}{118.0:>20.4f}{145.0:>20.4f}{68.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sturgess Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/STURGESS_SPATIAL_LINKAGE_JOINT/315
Sturgess Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 315 in model.lagmul_sturgess_spatial_linkage_joints
    joint = model.lagmul_sturgess_spatial_linkage_joints[315]
    assert joint.node1 == 351
    assert joint.node2 == 352
    assert joint.node3 == 353
    assert pytest.approx(joint.stiff) == 2.06e7
    assert joint.skew_id == 69
    assert pytest.approx(joint.tol) == 1.05e-4
    assert pytest.approx(joint.link_len_a) == 135.0
    assert pytest.approx(joint.link_len_b) == 118.0
    assert pytest.approx(joint.twist_angle_alpha) == 145.0
    assert pytest.approx(joint.offset_distance_s) == 68.0
    assert pytest.approx(joint.offset_distance_r) == 68.0
    assert pytest.approx(joint.offset_distance_v) == 68.0
    assert pytest.approx(joint.offset_distance_h) == 68.0
    assert pytest.approx(joint.offset_distance_u) == 68.0


def test_m392_lagmul_sturgess_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sturgess Spatial Linkage Joint Free Format Test
/LAGMUL/STURGESS_SPATIAL_LINKAGE_JOINT/316
451, 452, 453, 2.30e7, 70, 1.15e-4
155.0, 125.0, 165.0, 80.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 316 in model.lagmul_sturgess_spatial_linkage_joints
    joint = model.lagmul_sturgess_spatial_linkage_joints[316]
    assert joint.node1 == 451
    assert joint.node2 == 452
    assert joint.node3 == 453
    assert pytest.approx(joint.stiff) == 2.30e7
    assert joint.skew_id == 70
    assert pytest.approx(joint.tol) == 1.15e-4
    assert pytest.approx(joint.link_len_a) == 155.0
    assert pytest.approx(joint.link_len_b) == 125.0
    assert pytest.approx(joint.twist_angle_alpha) == 165.0
    assert pytest.approx(joint.offset_distance_s) == 80.0


def test_m392_lagmul_sturgess_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sturgess Spatial Linkage Joint Aliases Test
/STURGESS_SPATIAL_LINKAGE_JOINT/317
551, 552, 553, 1.0e6, 0, 1.0e-6
95.0, 85.0, 95.0, 45.0
/LAGMUL/STURGESS_SPATIAL_LINKAGE/318
551, 552, 553, 1.0e6, 0, 1.0e-6
95.0, 85.0, 95.0, 45.0
/STURGESS_SPATIAL_LINKAGE/319
551, 552, 553, 1.0e6, 0, 1.0e-6
95.0, 85.0, 95.0, 45.0
/STURGESS_SPATIAL_MULTI_LOOP_MECHANISM/320
551, 552, 553, 1.0e6, 0, 1.0e-6
95.0, 85.0, 95.0, 45.0
/STURGESS_SPATIAL_SYMMETRIC_MECHANISM/321
551, 552, 553, 1.0e6, 0, 1.0e-6
95.0, 85.0, 95.0, 45.0
/STURGESS_SPATIAL_6R_MECHANISM/322
551, 552, 553, 1.0e6, 0, 1.0e-6
95.0, 85.0, 95.0, 45.0
/STURGESS_SPATIAL_OVERCONSTRAINED_MECHANISM/323
551, 552, 553, 1.0e6, 0, 1.0e-6
95.0, 85.0, 95.0, 45.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(317, 324):
        assert jid in model.lagmul_sturgess_spatial_linkage_joints


def test_m392_lagmul_sturgess_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/STURGESS_SPATIAL_LINKAGE_JOINT/324
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m392_sensor_spring_transverse_shot_rate_fixed(tmp_path: Path):
    c1 = f"{1398:>10d}{3.28e8:>20.4f}{0.0985:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Transverse Shot Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_SHOT_RATE/1
Fixed Spring Transverse Shot Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_transverse_shot_rates
    s1 = model.sensor_spring_transverse_shot_rates[1]
    assert s1.spring_id == 1398
    assert pytest.approx(s1.jtrans_shot_max) == 3.28e8
    assert pytest.approx(s1.jtrans_drop_max) == 3.28e8
    assert pytest.approx(s1.jtrans_lock_max) == 3.28e8
    assert pytest.approx(s1.jtrans_pop_max) == 3.28e8
    assert pytest.approx(s1.jtrans_snp_max) == 3.28e8
    assert pytest.approx(s1.jtrans_crackle_max) == 3.28e8
    assert pytest.approx(s1.t_delay) == 0.0985
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_SHOT_RATE"


def test_m392_sensor_spring_transverse_shot_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Transverse Shot Rate Free Format Test
/SENSOR/SPRING_TRANSVERSE_SHOT_RATE/2
Free Spring Transverse Shot Rate Sensor
1399, 3.88e8, 0.1195
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_transverse_shot_rates
    s2 = model.sensor_spring_transverse_shot_rates[2]
    assert s2.spring_id == 1399
    assert pytest.approx(s2.jtrans_shot_max) == 3.88e8
    assert pytest.approx(s2.t_delay) == 0.1195


def test_m392_sensor_spring_transverse_shot_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Transverse Shot Rate Aliases Test
/SENSOR/SPRING_TRANS_SHOT_RATE/391
1419, 1.495e8, 0.0365
/SENSOR/SPRING_RATE_SHOT_TRANS/392
1420, 1.515e8, 0.0375
/SENSOR/TRANSVERSE_SHOT_RATE_SPRING/393
1421, 1.535e8, 0.0385
/SENSOR/SPRING_SHOT_TRANS/394
1422, 1.555e8, 0.0395
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(391, 395):
        assert sid in model.sensor_spring_transverse_shot_rates


def test_m392_sensor_spring_transverse_shot_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TRANSVERSE_SHOT_RATE/395
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
