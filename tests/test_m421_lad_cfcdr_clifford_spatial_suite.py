"""Tests for Milestone M421: LadCoupleFacesheetCoreDebondingRate Failure Model, EngElectrothermoflexomagnetoplasmonicexcitonicmagnonicphononicpolaritonicResonanceEnergy, CliffordSpatialLinkageJoint, and SensorSpringNormalDropRate."""

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


def test_m421_fail_lad_couple_facesheet_core_debonding_rate_fixed(tmp_path: Path):
    c1 = f"{765.0:>20.4f}{2295.0:>20.4f}{545.0:>20.4f}{11.25:>20.4f}{0.665:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2350:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Facesheet-Core Debonding Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_FACESHEET_CORE_DEBONDING_RATE/2350
Ladeveze Coupled Facesheet-Core Debonding Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2350 in model.fail_ladcouplefacesheetcoredebondingrates
    fcfcd = model.fail_ladcouplefacesheetcoredebondingrates[2350]
    assert pytest.approx(fcfcd.sigma_cfcd0) == 765.0
    assert pytest.approx(fcfcd.sigma_cfcdc) == 2295.0
    assert pytest.approx(fcfcd.gamma_cfcd) == 545.0
    assert pytest.approx(fcfcd.p_cfcd) == 11.25
    assert pytest.approx(fcfcd.d_cfcd_max) == 0.665
    assert fcfcd.ifail_sh == 1
    assert fcfcd.ifail_so == 2
    assert fcfcd.fail_id == 2350
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_FACESHEET_CORE_DEBONDING_RATE"


def test_m421_fail_lad_couple_facesheet_core_debonding_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Facesheet-Core Debonding Rate Free Format Test
/FAIL/LAD_COUPLE_FACESHEET_CORE_DEBONDING_RATE/2351
775.0, 2325.0, 555.0, 11.45, 0.655
1, 1
2351
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2351 in model.fail_ladcouplefacesheetcoredebondingrates
    fcfcd = model.fail_ladcouplefacesheetcoredebondingrates[2351]
    assert pytest.approx(fcfcd.sigma_cfcd0) == 775.0
    assert pytest.approx(fcfcd.sigma_cfcdc) == 2325.0
    assert pytest.approx(fcfcd.gamma_cfcd) == 555.0
    assert pytest.approx(fcfcd.p_cfcd) == 11.45
    assert pytest.approx(fcfcd.d_cfcd_max) == 0.655
    assert fcfcd.fail_id == 2351


def test_m421_fail_lad_couple_facesheet_core_debonding_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Facesheet-Core Debonding Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_FACESHEET_CORE_DEBONDING_RATE/2352
755.0, 2265.0, 540.0, 11.05, 0.675
1, 1
/FAIL/LAD_CFCDR/2353
755.0, 2265.0, 540.0, 11.05, 0.675
1, 1
/FAIL/LAD_CFCDR_MODEL/2354
755.0, 2265.0, 540.0, 11.05, 0.675
1, 1
/FAIL/LAD_CFCDR_LAW/2355
755.0, 2265.0, 540.0, 11.05, 0.675
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_FACESHEET_CORE_DEBONDING/2356
755.0, 2265.0, 540.0, 11.05, 0.675
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2352 in model.fail_ladcouplefacesheetcoredebondingrates
    assert 2353 in model.fail_ladcouplefacesheetcoredebondingrates
    assert 2354 in model.fail_ladcouplefacesheetcoredebondingrates
    assert 2355 in model.fail_ladcouplefacesheetcoredebondingrates
    assert 2356 in model.fail_ladcouplefacesheetcoredebondingrates
    assert len(model.raw_fails) == 5


def test_m421_fail_lad_couple_facesheet_core_debonding_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_COUPLE_FACESHEET_CORE_DEBONDING_RATE/2357
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m421_eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0525:>20.4f}{755:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicexcitonicmagnonicphononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/655
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 655 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies[655]
    assert pytest.approx(eng.dt_etfmpxmppp) == 0.0525
    assert eng.sens_id == 755


def test_m421_eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicexcitonicmagnonicphononicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/656
0.0535, 756
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 656 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies[656]
    assert pytest.approx(eng.dt_etfmpxmppp) == 0.0535
    assert eng.sens_id == 756


def test_m421_eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicexcitonicmagnonicphononicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_EXCITON_MAGNON_PHONON_POLARITON_RES_WORK/657
0.0545, 757
/ENG/EELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONICRESONANCE/658
0.0555, 758
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_DISSIPATION/659
0.0565, 759
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE/660
0.0575, 760
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 657 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies
    assert 658 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies
    assert 659 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies
    assert 660 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies


def test_m421_eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/661
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m421_lagmul_clifford_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{741:>10d}{742:>10d}{743:>10d}{5.95e7:>20.4f}{355:>10d}{5.05e-4:>20.4e}"
    c2 = f"{415.0:>20.4f}{395.0:>20.4f}{425.0:>20.4f}{345.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Clifford Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/CLIFFORD_SPATIAL_LINKAGE_JOINT/705
Clifford Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 705 in model.lagmul_clifford_spatial_linkage_joints
    joint = model.lagmul_clifford_spatial_linkage_joints[705]
    assert joint.node1 == 741
    assert joint.node2 == 742
    assert joint.node3 == 743
    assert pytest.approx(joint.stiff) == 5.95e7
    assert joint.skew_id == 355
    assert pytest.approx(joint.tol) == 5.05e-4
    assert pytest.approx(joint.link_len_a) == 415.0
    assert pytest.approx(joint.link_len_b) == 395.0
    assert pytest.approx(joint.twist_angle_alpha) == 425.0
    assert pytest.approx(joint.offset_distance_s) == 345.0
    assert pytest.approx(joint.offset_distance_r) == 345.0
    assert pytest.approx(joint.offset_distance_v) == 345.0
    assert pytest.approx(joint.offset_distance_h) == 345.0
    assert pytest.approx(joint.offset_distance_u) == 345.0


def test_m421_lagmul_clifford_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Clifford Spatial Linkage Joint Free Format Test
/LAGMUL/CLIFFORD_SPATIAL_LINKAGE_JOINT/706
841, 842, 843, 6.25e7, 356, 5.15e-4
435.0, 405.0, 445.0, 360.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 706 in model.lagmul_clifford_spatial_linkage_joints
    joint = model.lagmul_clifford_spatial_linkage_joints[706]
    assert joint.node1 == 841
    assert joint.node2 == 842
    assert joint.node3 == 843
    assert pytest.approx(joint.stiff) == 6.25e7
    assert joint.skew_id == 356
    assert pytest.approx(joint.tol) == 5.15e-4
    assert pytest.approx(joint.link_len_a) == 435.0
    assert pytest.approx(joint.link_len_b) == 405.0
    assert pytest.approx(joint.twist_angle_alpha) == 445.0
    assert pytest.approx(joint.offset_distance_s) == 360.0


def test_m421_lagmul_clifford_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Clifford Spatial Linkage Joint Aliases Test
/CLIFFORD_SPATIAL_LINKAGE_JOINT/707
941, 942, 943, 1.0e6, 0, 1.0e-6
375.0, 365.0, 375.0, 325.0
/LAGMUL/CLIFFORD_SPATIAL_LINKAGE/708
941, 942, 943, 1.0e6, 0, 1.0e-6
375.0, 365.0, 375.0, 325.0
/CLIFFORD_SPATIAL_LINKAGE/709
941, 942, 943, 1.0e6, 0, 1.0e-6
375.0, 365.0, 375.0, 325.0
/CLIFFORD_SPATIAL_MULTI_LOOP_MECHANISM/710
941, 942, 943, 1.0e6, 0, 1.0e-6
375.0, 365.0, 375.0, 325.0
/CLIFFORD_SPATIAL_SYMMETRIC_MECHANISM/711
941, 942, 943, 1.0e6, 0, 1.0e-6
375.0, 365.0, 375.0, 325.0
/CLIFFORD_SPATIAL_6R_MECHANISM/712
941, 942, 943, 1.0e6, 0, 1.0e-6
375.0, 365.0, 375.0, 325.0
/CLIFFORD_SPATIAL_OVERCONSTRAINED_MECHANISM/713
941, 942, 943, 1.0e6, 0, 1.0e-6
375.0, 365.0, 375.0, 325.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(707, 714):
        assert jid in model.lagmul_clifford_spatial_linkage_joints


def test_m421_lagmul_clifford_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/CLIFFORD_SPATIAL_LINKAGE_JOINT/714
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m421_sensor_spring_normal_drop_rate_fixed(tmp_path: Path):
    c1 = f"{1788:>10d}{7.18e8:>20.4f}{0.2745:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Normal Drop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_NORMAL_DROP_RATE/1
Fixed Spring Normal Drop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_normal_drop_rates
    s1 = model.sensor_spring_normal_drop_rates[1]
    assert s1.spring_id == 1788
    assert pytest.approx(s1.jnorm_drop_max) == 7.18e8
    assert pytest.approx(s1.jnorm_lock_max) == 7.18e8
    assert pytest.approx(s1.jnorm_snp_max) == 7.18e8
    assert pytest.approx(s1.jnorm_crackle_max) == 7.18e8
    assert pytest.approx(s1.jnorm_shot_max) == 7.18e8
    assert pytest.approx(s1.jnorm_pop_max) == 7.18e8
    assert pytest.approx(s1.jnorm_crk_max) == 7.18e8
    assert pytest.approx(s1.t_delay) == 0.2745
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_NORMAL_DROP_RATE"


def test_m421_sensor_spring_normal_drop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Normal Drop Rate Free Format Test
/SENSOR/SPRING_NORMAL_DROP_RATE/2
Free Spring Normal Drop Rate Sensor
1789, 7.78e8, 0.3735
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_normal_drop_rates
    s2 = model.sensor_spring_normal_drop_rates[2]
    assert s2.spring_id == 1789
    assert pytest.approx(s2.jnorm_drop_max) == 7.78e8
    assert pytest.approx(s2.t_delay) == 0.3735


def test_m421_sensor_spring_normal_drop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Normal Drop Rate Aliases Test
/SENSOR/SPRING_NORM_DROP_RATE/651
1809, 4.395e8, 0.3265
/SENSOR/SPRING_RATE_DROP_NORM/652
1810, 4.415e8, 0.3275
/SENSOR/NORMAL_DROP_RATE_SPRING/653
1811, 4.435e8, 0.3285
/SENSOR/SPRING_DROP_NORM/654
1812, 4.455e8, 0.3295
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(651, 655):
        assert sid in model.sensor_spring_normal_drop_rates


def test_m421_sensor_spring_normal_drop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_NORMAL_DROP_RATE/655
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
