"""Tests for Milestone M419: LadDynamicFacesheetCoreDebondingRate Failure Model, EngElectrothermoflexomagnetoplasmonicexcitonicphononicpolaritonicResonanceEnergy, HausdorffSpatialLinkageJoint, and SensorSpringBendingLockRate."""

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


def test_m419_fail_lad_dynamic_facesheet_core_debonding_rate_fixed(tmp_path: Path):
    c1 = f"{745.0:>20.4f}{2235.0:>20.4f}{525.0:>20.4f}{10.85:>20.4f}{0.685:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2330:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Facesheet-Core Debonding Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_FACESHEET_CORE_DEBONDING_RATE/2330
Ladeveze Dynamic Facesheet-Core Debonding Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2330 in model.fail_laddynamicfacesheetcoredebondingrates
    fdfcd = model.fail_laddynamicfacesheetcoredebondingrates[2330]
    assert pytest.approx(fdfcd.sigma_dfcd0) == 745.0
    assert pytest.approx(fdfcd.sigma_dfcdc) == 2235.0
    assert pytest.approx(fdfcd.gamma_dfcd) == 525.0
    assert pytest.approx(fdfcd.p_dfcd) == 10.85
    assert pytest.approx(fdfcd.d_dfcd_max) == 0.685
    assert fdfcd.ifail_sh == 1
    assert fdfcd.ifail_so == 2
    assert fdfcd.fail_id == 2330
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_FACESHEET_CORE_DEBONDING_RATE"


def test_m419_fail_lad_dynamic_facesheet_core_debonding_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Facesheet-Core Debonding Rate Free Format Test
/FAIL/LAD_DYNAMIC_FACESHEET_CORE_DEBONDING_RATE/2331
755.0, 2265.0, 535.0, 11.05, 0.675
1, 1
2331
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2331 in model.fail_laddynamicfacesheetcoredebondingrates
    fdfcd = model.fail_laddynamicfacesheetcoredebondingrates[2331]
    assert pytest.approx(fdfcd.sigma_dfcd0) == 755.0
    assert pytest.approx(fdfcd.sigma_dfcdc) == 2265.0
    assert pytest.approx(fdfcd.gamma_dfcd) == 535.0
    assert pytest.approx(fdfcd.p_dfcd) == 11.05
    assert pytest.approx(fdfcd.d_dfcd_max) == 0.675
    assert fdfcd.fail_id == 2331


def test_m419_fail_lad_dynamic_facesheet_core_debonding_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Facesheet-Core Debonding Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_FACESHEET_CORE_DEBONDING_RATE/2332
735.0, 2205.0, 520.0, 10.65, 0.695
1, 1
/FAIL/LAD_DFCDR/2333
735.0, 2205.0, 520.0, 10.65, 0.695
1, 1
/FAIL/LAD_DFCDR_MODEL/2334
735.0, 2205.0, 520.0, 10.65, 0.695
1, 1
/FAIL/LAD_DFCDR_LAW/2335
735.0, 2205.0, 520.0, 10.65, 0.695
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_FACESHEET_CORE_DEBONDING/2336
735.0, 2205.0, 520.0, 10.65, 0.695
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2332 in model.fail_laddynamicfacesheetcoredebondingrates
    assert 2333 in model.fail_laddynamicfacesheetcoredebondingrates
    assert 2334 in model.fail_laddynamicfacesheetcoredebondingrates
    assert 2335 in model.fail_laddynamicfacesheetcoredebondingrates
    assert 2336 in model.fail_laddynamicfacesheetcoredebondingrates
    assert len(model.raw_fails) == 5


def test_m419_fail_lad_dynamic_facesheet_core_debonding_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_FACESHEET_CORE_DEBONDING_RATE/2337
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m419_eng_electrothermoflexomagnetoplasmonicexcitonicphononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0485:>20.4f}{715:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicexcitonicphononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY/615
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 615 in model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicpolaritonic_resonance_energies[615]
    assert pytest.approx(eng.dt_etfmpeppp) == 0.0485
    assert eng.sens_id == 715


def test_m419_eng_electrothermoflexomagnetoplasmonicexcitonicphononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicexcitonicphononicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY/616
0.0495, 716
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 616 in model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicpolaritonic_resonance_energies[616]
    assert pytest.approx(eng.dt_etfmpeppp) == 0.0495
    assert eng.sens_id == 716


def test_m419_eng_electrothermoflexomagnetoplasmonicexcitonicphononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicexcitonicphononicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_EXCITON_PHONON_POLARITON_RES_WORK/617
0.0505, 717
/ENG/EELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICPOLARITONICRESONANCE/618
0.0515, 718
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE_DISSIPATION/619
0.0525, 719
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE/620
0.0535, 720
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 617 in model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicpolaritonic_resonance_energies
    assert 618 in model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicpolaritonic_resonance_energies
    assert 619 in model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicpolaritonic_resonance_energies
    assert 620 in model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicpolaritonic_resonance_energies


def test_m419_eng_electrothermoflexomagnetoplasmonicexcitonicphononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY/621
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m419_lagmul_hausdorff_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{701:>10d}{702:>10d}{703:>10d}{5.55e7:>20.4f}{335:>10d}{4.65e-4:>20.4e}"
    c2 = f"{395.0:>20.4f}{375.0:>20.4f}{405.0:>20.4f}{325.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Hausdorff Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/HAUSDORFF_SPATIAL_LINKAGE_JOINT/665
Hausdorff Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 665 in model.lagmul_hausdorff_spatial_linkage_joints
    joint = model.lagmul_hausdorff_spatial_linkage_joints[665]
    assert joint.node1 == 701
    assert joint.node2 == 702
    assert joint.node3 == 703
    assert pytest.approx(joint.stiff) == 5.55e7
    assert joint.skew_id == 335
    assert pytest.approx(joint.tol) == 4.65e-4
    assert pytest.approx(joint.link_len_a) == 395.0
    assert pytest.approx(joint.link_len_b) == 375.0
    assert pytest.approx(joint.twist_angle_alpha) == 405.0
    assert pytest.approx(joint.offset_distance_s) == 325.0
    assert pytest.approx(joint.offset_distance_r) == 325.0
    assert pytest.approx(joint.offset_distance_v) == 325.0
    assert pytest.approx(joint.offset_distance_h) == 325.0
    assert pytest.approx(joint.offset_distance_u) == 325.0


def test_m419_lagmul_hausdorff_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Hausdorff Spatial Linkage Joint Free Format Test
/LAGMUL/HAUSDORFF_SPATIAL_LINKAGE_JOINT/666
801, 802, 803, 5.85e7, 336, 4.75e-4
415.0, 385.0, 425.0, 340.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 666 in model.lagmul_hausdorff_spatial_linkage_joints
    joint = model.lagmul_hausdorff_spatial_linkage_joints[666]
    assert joint.node1 == 801
    assert joint.node2 == 802
    assert joint.node3 == 803
    assert pytest.approx(joint.stiff) == 5.85e7
    assert joint.skew_id == 336
    assert pytest.approx(joint.tol) == 4.75e-4
    assert pytest.approx(joint.link_len_a) == 415.0
    assert pytest.approx(joint.link_len_b) == 385.0
    assert pytest.approx(joint.twist_angle_alpha) == 425.0
    assert pytest.approx(joint.offset_distance_s) == 340.0


def test_m419_lagmul_hausdorff_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Hausdorff Spatial Linkage Joint Aliases Test
/HAUSDORFF_SPATIAL_LINKAGE_JOINT/667
901, 902, 903, 1.0e6, 0, 1.0e-6
355.0, 345.0, 355.0, 305.0
/LAGMUL/HAUSDORFF_SPATIAL_LINKAGE/668
901, 902, 903, 1.0e6, 0, 1.0e-6
355.0, 345.0, 355.0, 305.0
/HAUSDORFF_SPATIAL_LINKAGE/669
901, 902, 903, 1.0e6, 0, 1.0e-6
355.0, 345.0, 355.0, 305.0
/HAUSDORFF_SPATIAL_MULTI_LOOP_MECHANISM/670
901, 902, 903, 1.0e6, 0, 1.0e-6
355.0, 345.0, 355.0, 305.0
/HAUSDORFF_SPATIAL_SYMMETRIC_MECHANISM/671
901, 902, 903, 1.0e6, 0, 1.0e-6
355.0, 345.0, 355.0, 305.0
/HAUSDORFF_SPATIAL_6R_MECHANISM/672
901, 902, 903, 1.0e6, 0, 1.0e-6
355.0, 345.0, 355.0, 305.0
/HAUSDORFF_SPATIAL_OVERCONSTRAINED_MECHANISM/673
901, 902, 903, 1.0e6, 0, 1.0e-6
355.0, 345.0, 355.0, 305.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(667, 674):
        assert jid in model.lagmul_hausdorff_spatial_linkage_joints


def test_m419_lagmul_hausdorff_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/HAUSDORFF_SPATIAL_LINKAGE_JOINT/674
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m419_sensor_spring_bending_lock_rate_fixed(tmp_path: Path):
    c1 = f"{1748:>10d}{6.78e8:>20.4f}{0.2545:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Bending Lock Rate Fixed Format Test
2022 0
/SENSOR/SPRING_BENDING_LOCK_RATE/1
Fixed Spring Bending Lock Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_bending_lock_rates
    s1 = model.sensor_spring_bending_lock_rates[1]
    assert s1.spring_id == 1748
    assert pytest.approx(s1.jbend_lock_max) == 6.78e8
    assert pytest.approx(s1.jbend_snp_max) == 6.78e8
    assert pytest.approx(s1.jbend_crackle_max) == 6.78e8
    assert pytest.approx(s1.jbend_shot_max) == 6.78e8
    assert pytest.approx(s1.jbend_drop_max) == 6.78e8
    assert pytest.approx(s1.jbend_pop_max) == 6.78e8
    assert pytest.approx(s1.jbend_crk_max) == 6.78e8
    assert pytest.approx(s1.t_delay) == 0.2545
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_BENDING_LOCK_RATE"


def test_m419_sensor_spring_bending_lock_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Bending Lock Rate Free Format Test
/SENSOR/SPRING_BENDING_LOCK_RATE/2
Free Spring Bending Lock Rate Sensor
1749, 7.38e8, 0.3535
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_bending_lock_rates
    s2 = model.sensor_spring_bending_lock_rates[2]
    assert s2.spring_id == 1749
    assert pytest.approx(s2.jbend_lock_max) == 7.38e8
    assert pytest.approx(s2.t_delay) == 0.3535


def test_m419_sensor_spring_bending_lock_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Bending Lock Rate Aliases Test
/SENSOR/SPRING_BEND_LOCK_RATE/611
1769, 4.195e8, 0.3065
/SENSOR/SPRING_RATE_LOCK_BEND/612
1770, 4.215e8, 0.3075
/SENSOR/BENDING_LOCK_RATE_SPRING/613
1771, 4.235e8, 0.3085
/SENSOR/SPRING_LOCK_BEND/614
1772, 4.255e8, 0.3095
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(611, 615):
        assert sid in model.sensor_spring_bending_lock_rates


def test_m419_sensor_spring_bending_lock_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_BENDING_LOCK_RATE/615
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
