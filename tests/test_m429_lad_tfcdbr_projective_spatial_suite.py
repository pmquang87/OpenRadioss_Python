"""Tests for Milestone M429: LadTransverseFacesheetCoreDebondingRate Failure Model, EngElectrothermoflexomagnetoplasmonicexcitonicphononicmagnonpolaritonicResonanceEnergy, ProjectiveSpatialLinkageJoint, and SensorSpringTotalAngularRateOfChange."""

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


def test_m429_fail_lad_transverse_facesheet_core_debonding_rate_fixed(tmp_path: Path):
    c1 = f"{915.0:>20.4f}{2745.0:>20.4f}{695.0:>20.4f}{14.25:>20.4f}{0.535:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2430:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Facesheet Core Debonding Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_FACESHEET_CORE_DEBONDING_RATE/2430
Ladeveze Transverse Facesheet Core Debonding Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2430 in model.fail_ladtransversefacesheetcoredebondingrates
    ftfcd = model.fail_ladtransversefacesheetcoredebondingrates[2430]
    assert pytest.approx(ftfcd.sigma_tfcd0) == 915.0
    assert pytest.approx(ftfcd.sigma_tfcdc) == 2745.0
    assert pytest.approx(ftfcd.gamma_tfcd) == 695.0
    assert pytest.approx(ftfcd.p_tfcd) == 14.25
    assert pytest.approx(ftfcd.d_tfcd_max) == 0.535
    assert ftfcd.ifail_sh == 1
    assert ftfcd.ifail_so == 2
    assert ftfcd.fail_id == 2430
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_FACESHEET_CORE_DEBONDING_RATE"


def test_m429_fail_lad_transverse_facesheet_core_debonding_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Facesheet Core Debonding Rate Free Format Test
/FAIL/LAD_TRANSVERSE_FACESHEET_CORE_DEBONDING_RATE/2431
925.0, 2775.0, 705.0, 14.45, 0.525
1, 1
2431
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2431 in model.fail_ladtransversefacesheetcoredebondingrates
    ftfcd = model.fail_ladtransversefacesheetcoredebondingrates[2431]
    assert pytest.approx(ftfcd.sigma_tfcd0) == 925.0
    assert pytest.approx(ftfcd.sigma_tfcdc) == 2775.0
    assert pytest.approx(ftfcd.gamma_tfcd) == 705.0
    assert pytest.approx(ftfcd.p_tfcd) == 14.45
    assert pytest.approx(ftfcd.d_tfcd_max) == 0.525
    assert ftfcd.fail_id == 2431


def test_m429_fail_lad_transverse_facesheet_core_debonding_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Facesheet Core Debonding Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_FACESHEET_CORE_DEBONDING_RATE/2432
905.0, 2715.0, 685.0, 14.05, 0.545
1, 1
/FAIL/LAD_TFCDBR/2433
905.0, 2715.0, 685.0, 14.05, 0.545
1, 1
/FAIL/LAD_TFCDBR_MODEL/2434
905.0, 2715.0, 685.0, 14.05, 0.545
1, 1
/FAIL/LAD_TFCDBR_LAW/2435
905.0, 2715.0, 685.0, 14.05, 0.545
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_FACESHEET_CORE_DEBONDING/2436
905.0, 2715.0, 685.0, 14.05, 0.545
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2432 in model.fail_ladtransversefacesheetcoredebondingrates
    assert 2433 in model.fail_ladtransversefacesheetcoredebondingrates
    assert 2434 in model.fail_ladtransversefacesheetcoredebondingrates
    assert 2435 in model.fail_ladtransversefacesheetcoredebondingrates
    assert 2436 in model.fail_ladtransversefacesheetcoredebondingrates
    assert len(model.raw_fails) == 5


def test_m429_fail_lad_transverse_facesheet_core_debonding_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_TRANSVERSE_FACESHEET_CORE_DEBONDING_RATE/2437
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m429_eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0835:>20.4f}{845:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicexcitonicphononicmagnonpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY/745
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 745 in model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonpolaritonic_resonance_energies[745]
    assert pytest.approx(eng.dt_etfmpexpmpp) == 0.0835
    assert eng.sens_id == 845


def test_m429_eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicexcitonicphononicmagnonpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY/746
0.0845, 846
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 746 in model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonpolaritonic_resonance_energies[746]
    assert pytest.approx(eng.dt_etfmpexpmpp) == 0.0845
    assert eng.sens_id == 846


def test_m429_eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicexcitonicphononicmagnonpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_EXCITON_PHONON_MAGNON_POLARITON_RES_WORK/747
0.0855, 847
/ENG/EELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICMAGNONPOLARITONICRESONANCE/748
0.0865, 848
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICMAGNONPOLARITONIC_RESONANCE_DISSIPATION/749
0.0875, 849
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICMAGNONPOLARITONIC_RESONANCE/750
0.0885, 850
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 747 in model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonpolaritonic_resonance_energies
    assert 748 in model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonpolaritonic_resonance_energies
    assert 749 in model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonpolaritonic_resonance_energies
    assert 750 in model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonpolaritonic_resonance_energies


def test_m429_eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY/751
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m429_lagmul_projective_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{831:>10d}{832:>10d}{833:>10d}{7.15e7:>20.4f}{435:>10d}{6.25e-4:>20.4e}"
    c2 = f"{495.0:>20.4f}{475.0:>20.4f}{505.0:>20.4f}{425.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Projective Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/PROJECTIVE_SPATIAL_LINKAGE_JOINT/795
Projective Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 795 in model.lagmul_projective_spatial_linkage_joints
    joint = model.lagmul_projective_spatial_linkage_joints[795]
    assert joint.node1 == 831
    assert joint.node2 == 832
    assert joint.node3 == 833
    assert pytest.approx(joint.stiff) == 7.15e7
    assert joint.skew_id == 435
    assert pytest.approx(joint.tol) == 6.25e-4
    assert pytest.approx(joint.link_len_a) == 495.0
    assert pytest.approx(joint.link_len_b) == 475.0
    assert pytest.approx(joint.twist_angle_alpha) == 505.0
    assert pytest.approx(joint.offset_distance_s) == 425.0
    assert pytest.approx(joint.offset_distance_r) == 425.0
    assert pytest.approx(joint.offset_distance_v) == 425.0
    assert pytest.approx(joint.offset_distance_h) == 425.0
    assert pytest.approx(joint.offset_distance_u) == 425.0
    assert pytest.approx(joint.offset_distance_f) == 425.0


def test_m429_lagmul_projective_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Projective Spatial Linkage Joint Free Format Test
/LAGMUL/PROJECTIVE_SPATIAL_LINKAGE_JOINT/796
931, 932, 933, 7.35e7, 436, 6.35e-4
515.0, 485.0, 525.0, 440.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 796 in model.lagmul_projective_spatial_linkage_joints
    joint = model.lagmul_projective_spatial_linkage_joints[796]
    assert joint.node1 == 931
    assert joint.node2 == 932
    assert joint.node3 == 933
    assert pytest.approx(joint.stiff) == 7.35e7
    assert joint.skew_id == 436
    assert pytest.approx(joint.tol) == 6.35e-4
    assert pytest.approx(joint.link_len_a) == 515.0
    assert pytest.approx(joint.link_len_b) == 485.0
    assert pytest.approx(joint.twist_angle_alpha) == 525.0
    assert pytest.approx(joint.offset_distance_s) == 440.0


def test_m429_lagmul_projective_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Projective Spatial Linkage Joint Aliases Test
/PROJECTIVE_SPATIAL_LINKAGE_JOINT/797
1031, 1032, 1033, 1.0e6, 0, 1.0e-6
455.0, 445.0, 455.0, 405.0
/LAGMUL/PROJECTIVE_SPATIAL_LINKAGE/798
1031, 1032, 1033, 1.0e6, 0, 1.0e-6
455.0, 445.0, 455.0, 405.0
/PROJECTIVE_SPATIAL_LINKAGE/799
1031, 1032, 1033, 1.0e6, 0, 1.0e-6
455.0, 445.0, 455.0, 405.0
/PROJECTIVE_SPATIAL_MULTI_LOOP_MECHANISM/800
1031, 1032, 1033, 1.0e6, 0, 1.0e-6
455.0, 445.0, 455.0, 405.0
/PROJECTIVE_SPATIAL_SYMMETRIC_MECHANISM/801
1031, 1032, 1033, 1.0e6, 0, 1.0e-6
455.0, 445.0, 455.0, 405.0
/PROJECTIVE_SPATIAL_6R_MECHANISM/802
1031, 1032, 1033, 1.0e6, 0, 1.0e-6
455.0, 445.0, 455.0, 405.0
/PROJECTIVE_SPATIAL_OVERCONSTRAINED_MECHANISM/803
1031, 1032, 1033, 1.0e6, 0, 1.0e-6
455.0, 445.0, 455.0, 405.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(797, 804):
        assert jid in model.lagmul_projective_spatial_linkage_joints


def test_m429_lagmul_projective_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/PROJECTIVE_SPATIAL_LINKAGE_JOINT/804
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m429_sensor_spring_total_angular_rate_of_change_fixed(tmp_path: Path):
    c1 = f"{1878:>10d}{8.38e8:>20.4f}{0.3845:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Angular Rate Of Change Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_RATE_OF_CHANGE/1
Fixed Spring Total Angular Rate Of Change Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_angular_rate_of_changes
    s1 = model.sensor_spring_total_angular_rate_of_changes[1]
    assert s1.spring_id == 1878
    assert pytest.approx(s1.jtot_ang_roc_max) == 8.38e8
    assert pytest.approx(s1.jtot_ang_lock_max) == 8.38e8
    assert pytest.approx(s1.jtot_ang_snp_max) == 8.38e8
    assert pytest.approx(s1.jtot_ang_crackle_max) == 8.38e8
    assert pytest.approx(s1.jtot_ang_shot_max) == 8.38e8
    assert pytest.approx(s1.jtot_ang_pop_max) == 8.38e8
    assert pytest.approx(s1.jtot_ang_crk_max) == 8.38e8
    assert pytest.approx(s1.jtot_ang_drop_rate_max) == 8.38e8
    assert pytest.approx(s1.jtot_ang_rate_max) == 8.38e8
    assert pytest.approx(s1.jtot_ang_roc_rate_max) == 8.38e8
    assert pytest.approx(s1.t_delay) == 0.3845
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_RATE_OF_CHANGE"


def test_m429_sensor_spring_total_angular_rate_of_change_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Angular Rate Of Change Free Format Test
/SENSOR/SPRING_TOTAL_ANGULAR_RATE_OF_CHANGE/2
Free Spring Total Angular Rate Of Change Sensor
1879, 8.98e8, 0.4835
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_angular_rate_of_changes
    s2 = model.sensor_spring_total_angular_rate_of_changes[2]
    assert s2.spring_id == 1879
    assert pytest.approx(s2.jtot_ang_roc_max) == 8.98e8
    assert pytest.approx(s2.t_delay) == 0.4835


def test_m429_sensor_spring_total_angular_rate_of_change_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Angular Rate Of Change Aliases Test
/SENSOR/SPRING_TOT_ANG_RATE_OF_CHANGE/741
1899, 5.495e8, 0.4065
/SENSOR/SPRING_RATE_OF_CHANGE_ANG_TOT/742
1900, 5.515e8, 0.4075
/SENSOR/TOTAL_ANGULAR_RATE_OF_CHANGE_SPRING/743
1901, 5.535e8, 0.4085
/SENSOR/SPRING_ROC_ANG_TOT/744
1902, 5.555e8, 0.4095
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(741, 745):
        assert sid in model.sensor_spring_total_angular_rate_of_changes


def test_m429_sensor_spring_total_angular_rate_of_change_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TOTAL_ANGULAR_RATE_OF_CHANGE/745
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
