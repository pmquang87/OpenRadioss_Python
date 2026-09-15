"""Tests for Milestone M412: LadCoupleCoreDebondingRate Failure Model, EngElectrothermoflexoplasmonicexcitonicphononicpolaritonicResonanceEnergy, PoissonSpatialLinkageJoint, and SensorSpringTorsionalPopRate."""

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


def test_m412_fail_lad_couple_core_debonding_rate_fixed(tmp_path: Path):
    c1 = f"{645.0:>20.4f}{1935.0:>20.4f}{425.0:>20.4f}{8.85:>20.4f}{0.785:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2260:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Core Debonding Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_CORE_DEBONDING_RATE/2260
Ladeveze Coupled Core Debonding Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2260 in model.fail_ladcouplecoredebondingrates
    fccd = model.fail_ladcouplecoredebondingrates[2260]
    assert pytest.approx(fccd.sigma_ccd0) == 645.0
    assert pytest.approx(fccd.sigma_ccdc) == 1935.0
    assert pytest.approx(fccd.gamma_ccd) == 425.0
    assert pytest.approx(fccd.p_ccd) == 8.85
    assert pytest.approx(fccd.d_ccd_max) == 0.785
    assert fccd.ifail_sh == 1
    assert fccd.ifail_so == 2
    assert fccd.fail_id == 2260
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_CORE_DEBONDING_RATE"


def test_m412_fail_lad_couple_core_debonding_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Core Debonding Rate Free Format Test
/FAIL/LAD_COUPLE_CORE_DEBONDING_RATE/2261
655.0, 1965.0, 435.0, 9.05, 0.775
1, 1
2261
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2261 in model.fail_ladcouplecoredebondingrates
    fccd = model.fail_ladcouplecoredebondingrates[2261]
    assert pytest.approx(fccd.sigma_ccd0) == 655.0
    assert pytest.approx(fccd.sigma_ccdc) == 1965.0
    assert pytest.approx(fccd.gamma_ccd) == 435.0
    assert pytest.approx(fccd.p_ccd) == 9.05
    assert pytest.approx(fccd.d_ccd_max) == 0.775
    assert fccd.fail_id == 2261


def test_m412_fail_lad_couple_core_debonding_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Core Debonding Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_CORE_DEBONDING_RATE/2262
635.0, 1905.0, 420.0, 8.65, 0.795
1, 1
/FAIL/LAD_CCDR/2263
635.0, 1905.0, 420.0, 8.65, 0.795
1, 1
/FAIL/LAD_CCDR_MODEL/2264
635.0, 1905.0, 420.0, 8.65, 0.795
1, 1
/FAIL/LAD_CCDR_LAW/2265
635.0, 1905.0, 420.0, 8.65, 0.795
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_CORE_DEBONDING/2266
635.0, 1905.0, 420.0, 8.65, 0.795
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2262 in model.fail_ladcouplecoredebondingrates
    assert 2263 in model.fail_ladcouplecoredebondingrates
    assert 2264 in model.fail_ladcouplecoredebondingrates
    assert 2265 in model.fail_ladcouplecoredebondingrates
    assert 2266 in model.fail_ladcouplecoredebondingrates
    assert len(model.raw_fails) == 5


def test_m412_fail_lad_couple_core_debonding_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_COUPLE_CORE_DEBONDING_RATE/2267
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m412_eng_electrothermoflexoplasmonicexcitonicphononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0345:>20.4f}{575:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexoplasmonicexcitonicphononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY/475
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 475 in model.eng_electrothermoflexoplasmonicexcitonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexoplasmonicexcitonicphononicpolaritonic_resonance_energies[475]
    assert pytest.approx(eng.dt_etfpepp) == 0.0345
    assert eng.sens_id == 575


def test_m412_eng_electrothermoflexoplasmonicexcitonicphononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexoplasmonicexcitonicphononicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY/476
0.0355, 576
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 476 in model.eng_electrothermoflexoplasmonicexcitonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexoplasmonicexcitonicphononicpolaritonic_resonance_energies[476]
    assert pytest.approx(eng.dt_etfpepp) == 0.0355
    assert eng.sens_id == 576


def test_m412_eng_electrothermoflexoplasmonicexcitonicphononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexoplasmonicexcitonicphononicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_PLASMON_EXCITON_PHONON_POLARITON_RES_WORK/477
0.0365, 577
/ENG/EELECTROTHERMOFLEXOPLASMONICEXCITONICPHONONICPOLARITONICRESONANCE/478
0.0375, 578
/ENG/ELECTROTHERMOFLEXOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE_DISSIPATION/479
0.0385, 579
/ENG/ET_ELECTROTHERMOFLEXOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE/480
0.0395, 580
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 477 in model.eng_electrothermoflexoplasmonicexcitonicphononicpolaritonic_resonance_energies
    assert 478 in model.eng_electrothermoflexoplasmonicexcitonicphononicpolaritonic_resonance_energies
    assert 479 in model.eng_electrothermoflexoplasmonicexcitonicphononicpolaritonic_resonance_energies
    assert 480 in model.eng_electrothermoflexoplasmonicexcitonicphononicpolaritonic_resonance_energies


def test_m412_eng_electrothermoflexoplasmonicexcitonicphononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY/481
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m412_lagmul_poisson_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{561:>10d}{562:>10d}{563:>10d}{4.15e7:>20.4f}{265:>10d}{3.15e-4:>20.4e}"
    c2 = f"{325.0:>20.4f}{305.0:>20.4f}{335.0:>20.4f}{255.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Poisson Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/POISSON_SPATIAL_LINKAGE_JOINT/525
Poisson Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 525 in model.lagmul_poisson_spatial_linkage_joints
    joint = model.lagmul_poisson_spatial_linkage_joints[525]
    assert joint.node1 == 561
    assert joint.node2 == 562
    assert joint.node3 == 563
    assert pytest.approx(joint.stiff) == 4.15e7
    assert joint.skew_id == 265
    assert pytest.approx(joint.tol) == 3.15e-4
    assert pytest.approx(joint.link_len_a) == 325.0
    assert pytest.approx(joint.link_len_b) == 305.0
    assert pytest.approx(joint.twist_angle_alpha) == 335.0
    assert pytest.approx(joint.offset_distance_s) == 255.0
    assert pytest.approx(joint.offset_distance_r) == 255.0
    assert pytest.approx(joint.offset_distance_v) == 255.0
    assert pytest.approx(joint.offset_distance_h) == 255.0
    assert pytest.approx(joint.offset_distance_u) == 255.0


def test_m412_lagmul_poisson_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Poisson Spatial Linkage Joint Free Format Test
/LAGMUL/POISSON_SPATIAL_LINKAGE_JOINT/526
661, 662, 663, 4.45e7, 266, 3.25e-4
345.0, 315.0, 355.0, 270.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 526 in model.lagmul_poisson_spatial_linkage_joints
    joint = model.lagmul_poisson_spatial_linkage_joints[526]
    assert joint.node1 == 661
    assert joint.node2 == 662
    assert joint.node3 == 663
    assert pytest.approx(joint.stiff) == 4.45e7
    assert joint.skew_id == 266
    assert pytest.approx(joint.tol) == 3.25e-4
    assert pytest.approx(joint.link_len_a) == 345.0
    assert pytest.approx(joint.link_len_b) == 315.0
    assert pytest.approx(joint.twist_angle_alpha) == 355.0
    assert pytest.approx(joint.offset_distance_s) == 270.0


def test_m412_lagmul_poisson_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Poisson Spatial Linkage Joint Aliases Test
/POISSON_SPATIAL_LINKAGE_JOINT/527
761, 762, 763, 1.0e6, 0, 1.0e-6
285.0, 275.0, 285.0, 235.0
/LAGMUL/POISSON_SPATIAL_LINKAGE/528
761, 762, 763, 1.0e6, 0, 1.0e-6
285.0, 275.0, 285.0, 235.0
/POISSON_SPATIAL_LINKAGE/529
761, 762, 763, 1.0e6, 0, 1.0e-6
285.0, 275.0, 285.0, 235.0
/POISSON_SPATIAL_MULTI_LOOP_MECHANISM/530
761, 762, 763, 1.0e6, 0, 1.0e-6
285.0, 275.0, 285.0, 235.0
/POISSON_SPATIAL_SYMMETRIC_MECHANISM/531
761, 762, 763, 1.0e6, 0, 1.0e-6
285.0, 275.0, 285.0, 235.0
/POISSON_SPATIAL_6R_MECHANISM/532
761, 762, 763, 1.0e6, 0, 1.0e-6
285.0, 275.0, 285.0, 235.0
/POISSON_SPATIAL_OVERCONSTRAINED_MECHANISM/533
761, 762, 763, 1.0e6, 0, 1.0e-6
285.0, 275.0, 285.0, 235.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(527, 534):
        assert jid in model.lagmul_poisson_spatial_linkage_joints


def test_m412_lagmul_poisson_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/POISSON_SPATIAL_LINKAGE_JOINT/534
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m412_sensor_spring_torsional_pop_rate_fixed(tmp_path: Path):
    c1 = f"{1608:>10d}{5.38e8:>20.4f}{0.1845:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Pop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_POP_RATE/1
Fixed Spring Torsional Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_pop_rates
    s1 = model.sensor_spring_torsional_pop_rates[1]
    assert s1.spring_id == 1608
    assert pytest.approx(s1.jtor_pop_max) == 5.38e8
    assert pytest.approx(s1.jtor_snp_max) == 5.38e8
    assert pytest.approx(s1.jtor_crackle_max) == 5.38e8
    assert pytest.approx(s1.jtor_shot_max) == 5.38e8
    assert pytest.approx(s1.jtor_drop_max) == 5.38e8
    assert pytest.approx(s1.jtor_lock_max) == 5.38e8
    assert pytest.approx(s1.jtor_crk_max) == 5.38e8
    assert pytest.approx(s1.t_delay) == 0.1845
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_POP_RATE"


def test_m412_sensor_spring_torsional_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Torsional Pop Rate Free Format Test
/SENSOR/SPRING_TORSIONAL_POP_RATE/2
Free Spring Torsional Pop Rate Sensor
1609, 5.98e8, 0.2835
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_torsional_pop_rates
    s2 = model.sensor_spring_torsional_pop_rates[2]
    assert s2.spring_id == 1609
    assert pytest.approx(s2.jtor_pop_max) == 5.98e8
    assert pytest.approx(s2.t_delay) == 0.2835


def test_m412_sensor_spring_torsional_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Torsional Pop Rate Aliases Test
/SENSOR/SPRING_TORS_POP_RATE/471
1629, 3.495e8, 0.2365
/SENSOR/SPRING_RATE_POP_TORS/472
1630, 3.515e8, 0.2375
/SENSOR/TORSIONAL_POP_RATE_SPRING/473
1631, 3.535e8, 0.2385
/SENSOR/SPRING_POP_TORS/474
1632, 3.555e8, 0.2395
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(471, 475):
        assert sid in model.sensor_spring_torsional_pop_rates


def test_m412_sensor_spring_torsional_pop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TORSIONAL_POP_RATE/475
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
