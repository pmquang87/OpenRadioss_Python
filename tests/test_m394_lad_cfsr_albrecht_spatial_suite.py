"""Tests for Milestone M394: LadCoupleFiberSplittingRate Failure Model, EngFlexothermophononicexcitonicmagnonicpolaritonicResonanceEnergy, AlbrechtSpatialLinkageJoint, and SensorSpringTorsionalShotRate."""

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


def test_m394_fail_lad_couple_fiber_splitting_rate_fixed(tmp_path: Path):
    c1 = f"{480.0:>20.4f}{1440.0:>20.4f}{255.0:>20.4f}{6.25:>20.4f}{0.945:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2080:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Fiber Splitting Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_FIBER_SPLITTING_RATE/2080
Ladeveze Coupled Fiber Splitting Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2080 in model.fail_ladcouplefibersplittingrates
    fcfsr = model.fail_ladcouplefibersplittingrates[2080]
    assert pytest.approx(fcfsr.sigma_cfsr0) == 480.0
    assert pytest.approx(fcfsr.sigma_cfsrc) == 1440.0
    assert pytest.approx(fcfsr.gamma_cfsr) == 255.0
    assert pytest.approx(fcfsr.p_cfsr) == 6.25
    assert pytest.approx(fcfsr.d_cfsr_max) == 0.945
    assert fcfsr.ifail_sh == 1
    assert fcfsr.ifail_so == 2
    assert fcfsr.fail_id == 2080
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_FIBER_SPLITTING_RATE"


def test_m394_fail_lad_couple_fiber_splitting_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Fiber Splitting Rate Free Format Test
/FAIL/LAD_COUPLE_FIBER_SPLITTING_RATE/2081
490.0, 1470.0, 265.0, 6.45, 0.925
1, 1
2081
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2081 in model.fail_ladcouplefibersplittingrates
    fcfsr = model.fail_ladcouplefibersplittingrates[2081]
    assert pytest.approx(fcfsr.sigma_cfsr0) == 490.0
    assert pytest.approx(fcfsr.sigma_cfsrc) == 1470.0
    assert pytest.approx(fcfsr.gamma_cfsr) == 265.0
    assert pytest.approx(fcfsr.p_cfsr) == 6.45
    assert pytest.approx(fcfsr.d_cfsr_max) == 0.925
    assert fcfsr.fail_id == 2081


def test_m394_fail_lad_couple_fiber_splitting_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Fiber Splitting Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_FIBER_SPLITTING_RATE/2082
470.0, 1410.0, 250.0, 6.05, 0.955
1, 1
/FAIL/LAD_CFSR/2083
470.0, 1410.0, 250.0, 6.05, 0.955
1, 1
/FAIL/LAD_CFSR_MODEL/2084
470.0, 1410.0, 250.0, 6.05, 0.955
1, 1
/FAIL/LAD_CFSR_LAW/2085
470.0, 1410.0, 250.0, 6.05, 0.955
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_FIBER_SPLITTING/2086
470.0, 1410.0, 250.0, 6.05, 0.955
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2082 in model.fail_ladcouplefibersplittingrates
    assert 2083 in model.fail_ladcouplefibersplittingrates
    assert 2084 in model.fail_ladcouplefibersplittingrates
    assert 2085 in model.fail_ladcouplefibersplittingrates
    assert 2086 in model.fail_ladcouplefibersplittingrates
    assert len(model.raw_fails) == 5


def test_m394_fail_lad_couple_fiber_splitting_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_COUPLE_FIBER_SPLITTING_RATE/2087
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m394_eng_flexothermophononicexcitonicmagnonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0165:>20.4f}{385:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexothermophononicexcitonicmagnonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOPHONONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/285
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 285 in model.eng_flexothermophononicexcitonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_flexothermophononicexcitonicmagnonicpolaritonic_resonance_energies[285]
    assert pytest.approx(eng.dt_ftpempr) == 0.0165
    assert eng.sens_id == 385


def test_m394_eng_flexothermophononicexcitonicmagnonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexothermophononicexcitonicmagnonicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOPHONONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/286
0.0175, 386
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 286 in model.eng_flexothermophononicexcitonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_flexothermophononicexcitonicmagnonicpolaritonic_resonance_energies[286]
    assert pytest.approx(eng.dt_ftpempr) == 0.0175
    assert eng.sens_id == 386


def test_m394_eng_flexothermophononicexcitonicmagnonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexothermophononicexcitonicmagnonicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_PHONON_EXCITON_MAGNON_POLARITON_RES_WORK/287
0.0185, 387
/ENG/EFLEXOTHERMOPHONONICEXCITONICMAGNONICPOLARITONICRESONANCE/288
0.0195, 388
/ENG/FLEXOTHERMOPHONONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_DISSIPATION/289
0.0205, 389
/ENG/ET_FLEXOTHERMOPHONONICEXCITONICMAGNONICPOLARITONIC_RESONANCE/290
0.0215, 390
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 287 in model.eng_flexothermophononicexcitonicmagnonicpolaritonic_resonance_energies
    assert 288 in model.eng_flexothermophononicexcitonicmagnonicpolaritonic_resonance_energies
    assert 289 in model.eng_flexothermophononicexcitonicmagnonicpolaritonic_resonance_energies
    assert 290 in model.eng_flexothermophononicexcitonicmagnonicpolaritonic_resonance_energies


def test_m394_eng_flexothermophononicexcitonicmagnonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOTHERMOPHONONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/291
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m394_lagmul_albrecht_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{371:>10d}{372:>10d}{373:>10d}{2.26e7:>20.4f}{89:>10d}{1.25e-4:>20.4e}"
    c2 = f"{155.0:>20.4f}{138.0:>20.4f}{165.0:>20.4f}{88.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Albrecht Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/ALBRECHT_SPATIAL_LINKAGE_JOINT/335
Albrecht Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 335 in model.lagmul_albrecht_spatial_linkage_joints
    joint = model.lagmul_albrecht_spatial_linkage_joints[335]
    assert joint.node1 == 371
    assert joint.node2 == 372
    assert joint.node3 == 373
    assert pytest.approx(joint.stiff) == 2.26e7
    assert joint.skew_id == 89
    assert pytest.approx(joint.tol) == 1.25e-4
    assert pytest.approx(joint.link_len_a) == 155.0
    assert pytest.approx(joint.link_len_b) == 138.0
    assert pytest.approx(joint.twist_angle_alpha) == 165.0
    assert pytest.approx(joint.offset_distance_s) == 88.0
    assert pytest.approx(joint.offset_distance_r) == 88.0
    assert pytest.approx(joint.offset_distance_v) == 88.0
    assert pytest.approx(joint.offset_distance_h) == 88.0
    assert pytest.approx(joint.offset_distance_u) == 88.0


def test_m394_lagmul_albrecht_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Albrecht Spatial Linkage Joint Free Format Test
/LAGMUL/ALBRECHT_SPATIAL_LINKAGE_JOINT/336
471, 472, 473, 2.50e7, 90, 1.35e-4
175.0, 145.0, 185.0, 100.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 336 in model.lagmul_albrecht_spatial_linkage_joints
    joint = model.lagmul_albrecht_spatial_linkage_joints[336]
    assert joint.node1 == 471
    assert joint.node2 == 472
    assert joint.node3 == 473
    assert pytest.approx(joint.stiff) == 2.50e7
    assert joint.skew_id == 90
    assert pytest.approx(joint.tol) == 1.35e-4
    assert pytest.approx(joint.link_len_a) == 175.0
    assert pytest.approx(joint.link_len_b) == 145.0
    assert pytest.approx(joint.twist_angle_alpha) == 185.0
    assert pytest.approx(joint.offset_distance_s) == 100.0


def test_m394_lagmul_albrecht_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Albrecht Spatial Linkage Joint Aliases Test
/ALBRECHT_SPATIAL_LINKAGE_JOINT/337
571, 572, 573, 1.0e6, 0, 1.0e-6
115.0, 105.0, 115.0, 65.0
/LAGMUL/ALBRECHT_SPATIAL_LINKAGE/338
571, 572, 573, 1.0e6, 0, 1.0e-6
115.0, 105.0, 115.0, 65.0
/ALBRECHT_SPATIAL_LINKAGE/339
571, 572, 573, 1.0e6, 0, 1.0e-6
115.0, 105.0, 115.0, 65.0
/ALBRECHT_SPATIAL_MULTI_LOOP_MECHANISM/340
571, 572, 573, 1.0e6, 0, 1.0e-6
115.0, 105.0, 115.0, 65.0
/ALBRECHT_SPATIAL_SYMMETRIC_MECHANISM/341
571, 572, 573, 1.0e6, 0, 1.0e-6
115.0, 105.0, 115.0, 65.0
/ALBRECHT_SPATIAL_6R_MECHANISM/342
571, 572, 573, 1.0e6, 0, 1.0e-6
115.0, 105.0, 115.0, 65.0
/ALBRECHT_SPATIAL_OVERCONSTRAINED_MECHANISM/343
571, 572, 573, 1.0e6, 0, 1.0e-6
115.0, 105.0, 115.0, 65.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(337, 344):
        assert jid in model.lagmul_albrecht_spatial_linkage_joints


def test_m394_lagmul_albrecht_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/ALBRECHT_SPATIAL_LINKAGE_JOINT/344
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m394_sensor_spring_torsional_shot_rate_fixed(tmp_path: Path):
    c1 = f"{1418:>10d}{3.48e8:>20.4f}{0.1065:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Shot Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_SHOT_RATE/1
Fixed Spring Torsional Shot Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_shot_rates
    s1 = model.sensor_spring_torsional_shot_rates[1]
    assert s1.spring_id == 1418
    assert pytest.approx(s1.jtors_shot_max) == 3.48e8
    assert pytest.approx(s1.jtors_drop_max) == 3.48e8
    assert pytest.approx(s1.jtors_lock_max) == 3.48e8
    assert pytest.approx(s1.jtors_pop_max) == 3.48e8
    assert pytest.approx(s1.jtors_snp_max) == 3.48e8
    assert pytest.approx(s1.jtors_crackle_max) == 3.48e8
    assert pytest.approx(s1.t_delay) == 0.1065
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_SHOT_RATE"


def test_m394_sensor_spring_torsional_shot_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Torsional Shot Rate Free Format Test
/SENSOR/SPRING_TORSIONAL_SHOT_RATE/2
Free Spring Torsional Shot Rate Sensor
1419, 4.08e8, 0.1395
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_torsional_shot_rates
    s2 = model.sensor_spring_torsional_shot_rates[2]
    assert s2.spring_id == 1419
    assert pytest.approx(s2.jtors_shot_max) == 4.08e8
    assert pytest.approx(s2.t_delay) == 0.1395


def test_m394_sensor_spring_torsional_shot_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Torsional Shot Rate Aliases Test
/SENSOR/SPRING_TORS_SHOT_RATE/391
1439, 1.695e8, 0.0565
/SENSOR/SPRING_RATE_SHOT_TORS/392
1440, 1.715e8, 0.0575
/SENSOR/TORSIONAL_SHOT_RATE_SPRING/393
1441, 1.735e8, 0.0585
/SENSOR/SPRING_SHOT_TORS/394
1442, 1.755e8, 0.0595
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(391, 395):
        assert sid in model.sensor_spring_torsional_shot_rates


def test_m394_sensor_spring_torsional_shot_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TORSIONAL_SHOT_RATE/395
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
