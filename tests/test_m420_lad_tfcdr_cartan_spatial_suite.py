"""Tests for Milestone M420: LadTransverseFacesheetCoreDebondingRate Failure Model, EngElectrothermoflexomagnetoplasmonicmagnonicphononicpolaritonicResonanceEnergy, CartanSpatialLinkageJoint, and SensorSpringTotalAngularLockRate."""

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


def test_m420_fail_lad_transverse_facesheet_core_debonding_rate_fixed(tmp_path: Path):
    c1 = f"{755.0:>20.4f}{2265.0:>20.4f}{535.0:>20.4f}{11.05:>20.4f}{0.675:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2340:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Facesheet-Core Debonding Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_FACESHEET_CORE_DEBONDING_RATE/2340
Ladeveze Transverse Facesheet-Core Debonding Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2340 in model.fail_ladtransversefacesheetcoredebondingrates
    ftfcd = model.fail_ladtransversefacesheetcoredebondingrates[2340]
    assert pytest.approx(ftfcd.sigma_tfcd0) == 755.0
    assert pytest.approx(ftfcd.sigma_tfcdc) == 2265.0
    assert pytest.approx(ftfcd.gamma_tfcd) == 535.0
    assert pytest.approx(ftfcd.p_tfcd) == 11.05
    assert pytest.approx(ftfcd.d_tfcd_max) == 0.675
    assert ftfcd.ifail_sh == 1
    assert ftfcd.ifail_so == 2
    assert ftfcd.fail_id == 2340
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_FACESHEET_CORE_DEBONDING_RATE"


def test_m420_fail_lad_transverse_facesheet_core_debonding_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Facesheet-Core Debonding Rate Free Format Test
/FAIL/LAD_TRANSVERSE_FACESHEET_CORE_DEBONDING_RATE/2341
765.0, 2295.0, 545.0, 11.25, 0.665
1, 1
2341
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2341 in model.fail_ladtransversefacesheetcoredebondingrates
    ftfcd = model.fail_ladtransversefacesheetcoredebondingrates[2341]
    assert pytest.approx(ftfcd.sigma_tfcd0) == 765.0
    assert pytest.approx(ftfcd.sigma_tfcdc) == 2295.0
    assert pytest.approx(ftfcd.gamma_tfcd) == 545.0
    assert pytest.approx(ftfcd.p_tfcd) == 11.25
    assert pytest.approx(ftfcd.d_tfcd_max) == 0.665
    assert ftfcd.fail_id == 2341


def test_m420_fail_lad_transverse_facesheet_core_debonding_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Facesheet-Core Debonding Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_FACESHEET_CORE_DEBONDING_RATE/2342
745.0, 2235.0, 530.0, 10.85, 0.685
1, 1
/FAIL/LAD_TFCDR/2343
745.0, 2235.0, 530.0, 10.85, 0.685
1, 1
/FAIL/LAD_TFCDR_MODEL/2344
745.0, 2235.0, 530.0, 10.85, 0.685
1, 1
/FAIL/LAD_TFCDR_LAW/2345
745.0, 2235.0, 530.0, 10.85, 0.685
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_FACESHEET_CORE_DEBONDING/2346
745.0, 2235.0, 530.0, 10.85, 0.685
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2342 in model.fail_ladtransversefacesheetcoredebondingrates
    assert 2343 in model.fail_ladtransversefacesheetcoredebondingrates
    assert 2344 in model.fail_ladtransversefacesheetcoredebondingrates
    assert 2345 in model.fail_ladtransversefacesheetcoredebondingrates
    assert 2346 in model.fail_ladtransversefacesheetcoredebondingrates
    assert len(model.raw_fails) == 5


def test_m420_fail_lad_transverse_facesheet_core_debonding_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_TRANSVERSE_FACESHEET_CORE_DEBONDING_RATE/2347
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m420_eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0505:>20.4f}{735:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/635
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 635 in model.eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energies[635]
    assert pytest.approx(eng.dt_etfmpmppp) == 0.0505
    assert eng.sens_id == 735


def test_m420_eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/636
0.0515, 736
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 636 in model.eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energies[636]
    assert pytest.approx(eng.dt_etfmpmppp) == 0.0515
    assert eng.sens_id == 736


def test_m420_eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_MAGNON_PHONON_POLARITON_RES_WORK/637
0.0525, 737
/ENG/EELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPHONONICPOLARITONICRESONANCE/638
0.0535, 738
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_DISSIPATION/639
0.0545, 739
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE/640
0.0555, 740
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 637 in model.eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energies
    assert 638 in model.eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energies
    assert 639 in model.eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energies
    assert 640 in model.eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energies


def test_m420_eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/641
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m420_lagmul_cartan_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{721:>10d}{722:>10d}{723:>10d}{5.75e7:>20.4f}{345:>10d}{4.85e-4:>20.4e}"
    c2 = f"{405.0:>20.4f}{385.0:>20.4f}{415.0:>20.4f}{335.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Cartan Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/CARTAN_SPATIAL_LINKAGE_JOINT/685
Cartan Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 685 in model.lagmul_cartan_spatial_linkage_joints
    joint = model.lagmul_cartan_spatial_linkage_joints[685]
    assert joint.node1 == 721
    assert joint.node2 == 722
    assert joint.node3 == 723
    assert pytest.approx(joint.stiff) == 5.75e7
    assert joint.skew_id == 345
    assert pytest.approx(joint.tol) == 4.85e-4
    assert pytest.approx(joint.link_len_a) == 405.0
    assert pytest.approx(joint.link_len_b) == 385.0
    assert pytest.approx(joint.twist_angle_alpha) == 415.0
    assert pytest.approx(joint.offset_distance_s) == 335.0
    assert pytest.approx(joint.offset_distance_r) == 335.0
    assert pytest.approx(joint.offset_distance_v) == 335.0
    assert pytest.approx(joint.offset_distance_h) == 335.0
    assert pytest.approx(joint.offset_distance_u) == 335.0


def test_m420_lagmul_cartan_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Cartan Spatial Linkage Joint Free Format Test
/LAGMUL/CARTAN_SPATIAL_LINKAGE_JOINT/686
821, 822, 823, 6.05e7, 346, 4.95e-4
425.0, 395.0, 435.0, 350.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 686 in model.lagmul_cartan_spatial_linkage_joints
    joint = model.lagmul_cartan_spatial_linkage_joints[686]
    assert joint.node1 == 821
    assert joint.node2 == 822
    assert joint.node3 == 823
    assert pytest.approx(joint.stiff) == 6.05e7
    assert joint.skew_id == 346
    assert pytest.approx(joint.tol) == 4.95e-4
    assert pytest.approx(joint.link_len_a) == 425.0
    assert pytest.approx(joint.link_len_b) == 395.0
    assert pytest.approx(joint.twist_angle_alpha) == 435.0
    assert pytest.approx(joint.offset_distance_s) == 350.0


def test_m420_lagmul_cartan_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Cartan Spatial Linkage Joint Aliases Test
/CARTAN_SPATIAL_LINKAGE_JOINT/687
921, 922, 923, 1.0e6, 0, 1.0e-6
365.0, 355.0, 365.0, 315.0
/LAGMUL/CARTAN_SPATIAL_LINKAGE/688
921, 922, 923, 1.0e6, 0, 1.0e-6
365.0, 355.0, 365.0, 315.0
/CARTAN_SPATIAL_LINKAGE/689
921, 922, 923, 1.0e6, 0, 1.0e-6
365.0, 355.0, 365.0, 315.0
/CARTAN_SPATIAL_MULTI_LOOP_MECHANISM/690
921, 922, 923, 1.0e6, 0, 1.0e-6
365.0, 355.0, 365.0, 315.0
/CARTAN_SPATIAL_SYMMETRIC_MECHANISM/691
921, 922, 923, 1.0e6, 0, 1.0e-6
365.0, 355.0, 365.0, 315.0
/CARTAN_SPATIAL_6R_MECHANISM/692
921, 922, 923, 1.0e6, 0, 1.0e-6
365.0, 355.0, 365.0, 315.0
/CARTAN_SPATIAL_OVERCONSTRAINED_MECHANISM/693
921, 922, 923, 1.0e6, 0, 1.0e-6
365.0, 355.0, 365.0, 315.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(687, 694):
        assert jid in model.lagmul_cartan_spatial_linkage_joints


def test_m420_lagmul_cartan_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/CARTAN_SPATIAL_LINKAGE_JOINT/694
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m420_sensor_spring_total_angular_lock_rate_fixed(tmp_path: Path):
    c1 = f"{1768:>10d}{6.98e8:>20.4f}{0.2645:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Angular Lock Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_LOCK_RATE/1
Fixed Spring Total Angular Lock Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_angular_lock_rates
    s1 = model.sensor_spring_total_angular_lock_rates[1]
    assert s1.spring_id == 1768
    assert pytest.approx(s1.jtot_ang_lock_max) == 6.98e8
    assert pytest.approx(s1.jtot_ang_snp_max) == 6.98e8
    assert pytest.approx(s1.jtot_ang_crackle_max) == 6.98e8
    assert pytest.approx(s1.jtot_ang_shot_max) == 6.98e8
    assert pytest.approx(s1.jtot_ang_drop_max) == 6.98e8
    assert pytest.approx(s1.jtot_ang_pop_max) == 6.98e8
    assert pytest.approx(s1.jtot_ang_crk_max) == 6.98e8
    assert pytest.approx(s1.t_delay) == 0.2645
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_LOCK_RATE"


def test_m420_sensor_spring_total_angular_lock_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Angular Lock Rate Free Format Test
/SENSOR/SPRING_TOTAL_ANGULAR_LOCK_RATE/2
Free Spring Total Angular Lock Rate Sensor
1769, 7.58e8, 0.3635
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_angular_lock_rates
    s2 = model.sensor_spring_total_angular_lock_rates[2]
    assert s2.spring_id == 1769
    assert pytest.approx(s2.jtot_ang_lock_max) == 7.58e8
    assert pytest.approx(s2.t_delay) == 0.3635


def test_m420_sensor_spring_total_angular_lock_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Angular Lock Rate Aliases Test
/SENSOR/SPRING_TOT_ANG_LOCK_RATE/631
1789, 4.295e8, 0.3165
/SENSOR/SPRING_RATE_LOCK_ANG_TOT/632
1790, 4.315e8, 0.3175
/SENSOR/TOTAL_ANGULAR_LOCK_RATE_SPRING/633
1791, 4.335e8, 0.3185
/SENSOR/SPRING_LOCK_ANG_TOT/634
1792, 4.355e8, 0.3195
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(631, 635):
        assert sid in model.sensor_spring_total_angular_lock_rates


def test_m420_sensor_spring_total_angular_lock_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TOTAL_ANGULAR_LOCK_RATE/635
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
