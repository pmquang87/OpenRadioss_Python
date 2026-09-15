"""Tests for Milestone M428: LadDynamicFacesheetCoreDebondingRate Failure Model, EngElectrothermoflexomagnetoexcitonicphononicmagnonpolaritonicResonanceEnergy, ConformalSpatialLinkageJoint, and SensorSpringBendingRateOfChange."""

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


def test_m428_fail_lad_dynamic_facesheet_core_debonding_rate_fixed(tmp_path: Path):
    c1 = f"{895.0:>20.4f}{2685.0:>20.4f}{675.0:>20.4f}{13.85:>20.4f}{0.555:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2420:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Facesheet Core Debonding Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_FACESHEET_CORE_DEBONDING_RATE/2420
Ladeveze Dynamic Facesheet Core Debonding Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2420 in model.fail_laddynamicfacesheetcoredebondingrates
    fdfcd = model.fail_laddynamicfacesheetcoredebondingrates[2420]
    assert pytest.approx(fdfcd.sigma_dfcd0) == 895.0
    assert pytest.approx(fdfcd.sigma_dfcdc) == 2685.0
    assert pytest.approx(fdfcd.gamma_dfcd) == 675.0
    assert pytest.approx(fdfcd.p_dfcd) == 13.85
    assert pytest.approx(fdfcd.d_dfcd_max) == 0.555
    assert fdfcd.ifail_sh == 1
    assert fdfcd.ifail_so == 2
    assert fdfcd.fail_id == 2420
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_FACESHEET_CORE_DEBONDING_RATE"


def test_m428_fail_lad_dynamic_facesheet_core_debonding_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Facesheet Core Debonding Rate Free Format Test
/FAIL/LAD_DYNAMIC_FACESHEET_CORE_DEBONDING_RATE/2421
905.0, 2715.0, 685.0, 14.05, 0.545
1, 1
2421
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2421 in model.fail_laddynamicfacesheetcoredebondingrates
    fdfcd = model.fail_laddynamicfacesheetcoredebondingrates[2421]
    assert pytest.approx(fdfcd.sigma_dfcd0) == 905.0
    assert pytest.approx(fdfcd.sigma_dfcdc) == 2715.0
    assert pytest.approx(fdfcd.gamma_dfcd) == 685.0
    assert pytest.approx(fdfcd.p_dfcd) == 14.05
    assert pytest.approx(fdfcd.d_dfcd_max) == 0.545
    assert fdfcd.fail_id == 2421


def test_m428_fail_lad_dynamic_facesheet_core_debonding_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Facesheet Core Debonding Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_FACESHEET_CORE_DEBONDING_RATE/2422
885.0, 2655.0, 665.0, 13.65, 0.565
1, 1
/FAIL/LAD_DFCDBR/2423
885.0, 2655.0, 665.0, 13.65, 0.565
1, 1
/FAIL/LAD_DFCDBR_MODEL/2424
885.0, 2655.0, 665.0, 13.65, 0.565
1, 1
/FAIL/LAD_DFCDBR_LAW/2425
885.0, 2655.0, 665.0, 13.65, 0.565
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_FACESHEET_CORE_DEBONDING/2426
885.0, 2655.0, 665.0, 13.65, 0.565
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2422 in model.fail_laddynamicfacesheetcoredebondingrates
    assert 2423 in model.fail_laddynamicfacesheetcoredebondingrates
    assert 2424 in model.fail_laddynamicfacesheetcoredebondingrates
    assert 2425 in model.fail_laddynamicfacesheetcoredebondingrates
    assert 2426 in model.fail_laddynamicfacesheetcoredebondingrates
    assert len(model.raw_fails) == 5


def test_m428_fail_lad_dynamic_facesheet_core_debonding_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_FACESHEET_CORE_DEBONDING_RATE/2427
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m428_eng_electrothermoflexomagnetoexcitonicphononicmagnonpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0775:>20.4f}{835:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetoexcitonicphononicmagnonpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY/735
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 735 in model.eng_electrothermoflexomagnetoexcitonicphononicmagnonpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoexcitonicphononicmagnonpolaritonic_resonance_energies[735]
    assert pytest.approx(eng.dt_etfmexpmpp) == 0.0775
    assert eng.sens_id == 835


def test_m428_eng_electrothermoflexomagnetoexcitonicphononicmagnonpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetoexcitonicphononicmagnonpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY/736
0.0785, 836
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 736 in model.eng_electrothermoflexomagnetoexcitonicphononicmagnonpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoexcitonicphononicmagnonpolaritonic_resonance_energies[736]
    assert pytest.approx(eng.dt_etfmexpmpp) == 0.0785
    assert eng.sens_id == 836


def test_m428_eng_electrothermoflexomagnetoexcitonicphononicmagnonpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetoexcitonicphononicmagnonpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_PHONON_MAGNON_POLARITON_RES_WORK/737
0.0795, 837
/ENG/EELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICMAGNONPOLARITONICRESONANCE/738
0.0805, 838
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICMAGNONPOLARITONIC_RESONANCE_DISSIPATION/739
0.0815, 839
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICMAGNONPOLARITONIC_RESONANCE/740
0.0825, 840
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 737 in model.eng_electrothermoflexomagnetoexcitonicphononicmagnonpolaritonic_resonance_energies
    assert 738 in model.eng_electrothermoflexomagnetoexcitonicphononicmagnonpolaritonic_resonance_energies
    assert 739 in model.eng_electrothermoflexomagnetoexcitonicphononicmagnonpolaritonic_resonance_energies
    assert 740 in model.eng_electrothermoflexomagnetoexcitonicphononicmagnonpolaritonic_resonance_energies


def test_m428_eng_electrothermoflexomagnetoexcitonicphononicmagnonpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY/741
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m428_lagmul_conformal_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{821:>10d}{822:>10d}{823:>10d}{6.95e7:>20.4f}{425:>10d}{6.05e-4:>20.4e}"
    c2 = f"{485.0:>20.4f}{465.0:>20.4f}{495.0:>20.4f}{415.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Conformal Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/CONFORMAL_SPATIAL_LINKAGE_JOINT/785
Conformal Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 785 in model.lagmul_conformal_spatial_linkage_joints
    joint = model.lagmul_conformal_spatial_linkage_joints[785]
    assert joint.node1 == 821
    assert joint.node2 == 822
    assert joint.node3 == 823
    assert pytest.approx(joint.stiff) == 6.95e7
    assert joint.skew_id == 425
    assert pytest.approx(joint.tol) == 6.05e-4
    assert pytest.approx(joint.link_len_a) == 485.0
    assert pytest.approx(joint.link_len_b) == 465.0
    assert pytest.approx(joint.twist_angle_alpha) == 495.0
    assert pytest.approx(joint.offset_distance_s) == 415.0
    assert pytest.approx(joint.offset_distance_r) == 415.0
    assert pytest.approx(joint.offset_distance_v) == 415.0
    assert pytest.approx(joint.offset_distance_h) == 415.0
    assert pytest.approx(joint.offset_distance_u) == 415.0
    assert pytest.approx(joint.offset_distance_f) == 415.0


def test_m428_lagmul_conformal_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Conformal Spatial Linkage Joint Free Format Test
/LAGMUL/CONFORMAL_SPATIAL_LINKAGE_JOINT/786
921, 922, 923, 7.15e7, 426, 6.15e-4
505.0, 475.0, 515.0, 430.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 786 in model.lagmul_conformal_spatial_linkage_joints
    joint = model.lagmul_conformal_spatial_linkage_joints[786]
    assert joint.node1 == 921
    assert joint.node2 == 922
    assert joint.node3 == 923
    assert pytest.approx(joint.stiff) == 7.15e7
    assert joint.skew_id == 426
    assert pytest.approx(joint.tol) == 6.15e-4
    assert pytest.approx(joint.link_len_a) == 505.0
    assert pytest.approx(joint.link_len_b) == 475.0
    assert pytest.approx(joint.twist_angle_alpha) == 515.0
    assert pytest.approx(joint.offset_distance_s) == 430.0


def test_m428_lagmul_conformal_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Conformal Spatial Linkage Joint Aliases Test
/CONFORMAL_SPATIAL_LINKAGE_JOINT/787
1021, 1022, 1023, 1.0e6, 0, 1.0e-6
445.0, 435.0, 445.0, 395.0
/LAGMUL/CONFORMAL_SPATIAL_LINKAGE/788
1021, 1022, 1023, 1.0e6, 0, 1.0e-6
445.0, 435.0, 445.0, 395.0
/CONFORMAL_SPATIAL_LINKAGE/789
1021, 1022, 1023, 1.0e6, 0, 1.0e-6
445.0, 435.0, 445.0, 395.0
/CONFORMAL_SPATIAL_MULTI_LOOP_MECHANISM/790
1021, 1022, 1023, 1.0e6, 0, 1.0e-6
445.0, 435.0, 445.0, 395.0
/CONFORMAL_SPATIAL_SYMMETRIC_MECHANISM/791
1021, 1022, 1023, 1.0e6, 0, 1.0e-6
445.0, 435.0, 445.0, 395.0
/CONFORMAL_SPATIAL_6R_MECHANISM/792
1021, 1022, 1023, 1.0e6, 0, 1.0e-6
445.0, 435.0, 445.0, 395.0
/CONFORMAL_SPATIAL_OVERCONSTRAINED_MECHANISM/793
1021, 1022, 1023, 1.0e6, 0, 1.0e-6
445.0, 435.0, 445.0, 395.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(787, 794):
        assert jid in model.lagmul_conformal_spatial_linkage_joints


def test_m428_lagmul_conformal_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/CONFORMAL_SPATIAL_LINKAGE_JOINT/794
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m428_sensor_spring_bending_rate_of_change_fixed(tmp_path: Path):
    c1 = f"{1868:>10d}{8.18e8:>20.4f}{0.3645:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Bending Rate Of Change Fixed Format Test
2022 0
/SENSOR/SPRING_BENDING_RATE_OF_CHANGE/1
Fixed Spring Bending Rate Of Change Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_bending_rate_of_changes
    s1 = model.sensor_spring_bending_rate_of_changes[1]
    assert s1.spring_id == 1868
    assert pytest.approx(s1.jbend_roc_max) == 8.18e8
    assert pytest.approx(s1.jbend_lock_max) == 8.18e8
    assert pytest.approx(s1.jbend_snp_max) == 8.18e8
    assert pytest.approx(s1.jbend_crackle_max) == 8.18e8
    assert pytest.approx(s1.jbend_shot_max) == 8.18e8
    assert pytest.approx(s1.jbend_pop_max) == 8.18e8
    assert pytest.approx(s1.jbend_crk_max) == 8.18e8
    assert pytest.approx(s1.jbend_drop_rate_max) == 8.18e8
    assert pytest.approx(s1.jbend_rate_max) == 8.18e8
    assert pytest.approx(s1.jbend_roc_rate_max) == 8.18e8
    assert pytest.approx(s1.t_delay) == 0.3645
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_BENDING_RATE_OF_CHANGE"


def test_m428_sensor_spring_bending_rate_of_change_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Bending Rate Of Change Free Format Test
/SENSOR/SPRING_BENDING_RATE_OF_CHANGE/2
Free Spring Bending Rate Of Change Sensor
1869, 8.78e8, 0.4635
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_bending_rate_of_changes
    s2 = model.sensor_spring_bending_rate_of_changes[2]
    assert s2.spring_id == 1869
    assert pytest.approx(s2.jbend_roc_max) == 8.78e8
    assert pytest.approx(s2.t_delay) == 0.4635


def test_m428_sensor_spring_bending_rate_of_change_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Bending Rate Of Change Aliases Test
/SENSOR/SPRING_BEND_RATE_OF_CHANGE/731
1889, 5.295e8, 0.3965
/SENSOR/SPRING_RATE_OF_CHANGE_BEND/732
1890, 5.315e8, 0.3975
/SENSOR/BENDING_RATE_OF_CHANGE_SPRING/733
1891, 5.335e8, 0.3985
/SENSOR/SPRING_ROC_BEND/734
1892, 5.355e8, 0.3995
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(731, 735):
        assert sid in model.sensor_spring_bending_rate_of_changes


def test_m428_sensor_spring_bending_rate_of_change_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_BENDING_RATE_OF_CHANGE/735
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
