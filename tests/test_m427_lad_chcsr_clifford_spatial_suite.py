"""Tests for Milestone M427: LadCoupleHoneycombCoreShearingRate Failure Model, EngElectrothermoflexomagnetoplasmonicphononicmagnonpolaritonicResonanceEnergy, CliffordSpatialLinkageJoint, and SensorSpringTorsionalRateOfChange."""

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


def test_m427_fail_lad_couple_honeycomb_core_shearing_rate_fixed(tmp_path: Path):
    c1 = f"{875.0:>20.4f}{2625.0:>20.4f}{655.0:>20.4f}{13.45:>20.4f}{0.575:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2410:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Honeycomb Core Shearing Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_HONEYCOMB_CORE_SHEARING_RATE/2410
Ladeveze Coupled Honeycomb Core Shearing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2410 in model.fail_ladcouplehoneycombcoreshearingrates
    fchcs = model.fail_ladcouplehoneycombcoreshearingrates[2410]
    assert pytest.approx(fchcs.sigma_chcs0) == 875.0
    assert pytest.approx(fchcs.sigma_chcsc) == 2625.0
    assert pytest.approx(fchcs.gamma_chcs) == 655.0
    assert pytest.approx(fchcs.p_chcs) == 13.45
    assert pytest.approx(fchcs.d_chcs_max) == 0.575
    assert fchcs.ifail_sh == 1
    assert fchcs.ifail_so == 2
    assert fchcs.fail_id == 2410
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_HONEYCOMB_CORE_SHEARING_RATE"


def test_m427_fail_lad_couple_honeycomb_core_shearing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Honeycomb Core Shearing Rate Free Format Test
/FAIL/LAD_COUPLE_HONEYCOMB_CORE_SHEARING_RATE/2411
885.0, 2655.0, 665.0, 13.65, 0.565
1, 1
2411
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2411 in model.fail_ladcouplehoneycombcoreshearingrates
    fchcs = model.fail_ladcouplehoneycombcoreshearingrates[2411]
    assert pytest.approx(fchcs.sigma_chcs0) == 885.0
    assert pytest.approx(fchcs.sigma_chcsc) == 2655.0
    assert pytest.approx(fchcs.gamma_chcs) == 665.0
    assert pytest.approx(fchcs.p_chcs) == 13.65
    assert pytest.approx(fchcs.d_chcs_max) == 0.565
    assert fchcs.fail_id == 2411


def test_m427_fail_lad_couple_honeycomb_core_shearing_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Honeycomb Core Shearing Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_HONEYCOMB_CORE_SHEARING_RATE/2412
865.0, 2595.0, 645.0, 13.25, 0.585
1, 1
/FAIL/LAD_CHCSR/2413
865.0, 2595.0, 645.0, 13.25, 0.585
1, 1
/FAIL/LAD_CHCSR_MODEL/2414
865.0, 2595.0, 645.0, 13.25, 0.585
1, 1
/FAIL/LAD_CHCSR_LAW/2415
865.0, 2595.0, 645.0, 13.25, 0.585
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_HONEYCOMB_CORE_SHEARING/2416
865.0, 2595.0, 645.0, 13.25, 0.585
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2412 in model.fail_ladcouplehoneycombcoreshearingrates
    assert 2413 in model.fail_ladcouplehoneycombcoreshearingrates
    assert 2414 in model.fail_ladcouplehoneycombcoreshearingrates
    assert 2415 in model.fail_ladcouplehoneycombcoreshearingrates
    assert 2416 in model.fail_ladcouplehoneycombcoreshearingrates
    assert len(model.raw_fails) == 5


def test_m427_fail_lad_couple_honeycomb_core_shearing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_COUPLE_HONEYCOMB_CORE_SHEARING_RATE/2417
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m427_eng_electrothermoflexomagnetoplasmonicphononicmagnonpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0715:>20.4f}{825:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicphononicmagnonpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY/725
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 725 in model.eng_electrothermoflexomagnetoplasmonicphononicmagnonpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicphononicmagnonpolaritonic_resonance_energies[725]
    assert pytest.approx(eng.dt_etfmppmpp) == 0.0715
    assert eng.sens_id == 825


def test_m427_eng_electrothermoflexomagnetoplasmonicphononicmagnonpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicphononicmagnonpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY/726
0.0725, 826
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 726 in model.eng_electrothermoflexomagnetoplasmonicphononicmagnonpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicphononicmagnonpolaritonic_resonance_energies[726]
    assert pytest.approx(eng.dt_etfmppmpp) == 0.0725
    assert eng.sens_id == 826


def test_m427_eng_electrothermoflexomagnetoplasmonicphononicmagnonpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicphononicmagnonpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_PHONON_MAGNON_POLARITON_RES_WORK/727
0.0735, 827
/ENG/EELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICMAGNONPOLARITONICRESONANCE/728
0.0745, 828
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICMAGNONPOLARITONIC_RESONANCE_DISSIPATION/729
0.0755, 829
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICMAGNONPOLARITONIC_RESONANCE/730
0.0765, 830
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 727 in model.eng_electrothermoflexomagnetoplasmonicphononicmagnonpolaritonic_resonance_energies
    assert 728 in model.eng_electrothermoflexomagnetoplasmonicphononicmagnonpolaritonic_resonance_energies
    assert 729 in model.eng_electrothermoflexomagnetoplasmonicphononicmagnonpolaritonic_resonance_energies
    assert 730 in model.eng_electrothermoflexomagnetoplasmonicphononicmagnonpolaritonic_resonance_energies


def test_m427_eng_electrothermoflexomagnetoplasmonicphononicmagnonpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY/731
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m427_lagmul_clifford_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{811:>10d}{812:>10d}{813:>10d}{6.75e7:>20.4f}{415:>10d}{5.85e-4:>20.4e}"
    c2 = f"{475.0:>20.4f}{455.0:>20.4f}{485.0:>20.4f}{405.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Clifford Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/CLIFFORD_SPATIAL_LINKAGE_JOINT/775
Clifford Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 775 in model.lagmul_clifford_spatial_linkage_joints
    joint = model.lagmul_clifford_spatial_linkage_joints[775]
    assert joint.node1 == 811
    assert joint.node2 == 812
    assert joint.node3 == 813
    assert pytest.approx(joint.stiff) == 6.75e7
    assert joint.skew_id == 415
    assert pytest.approx(joint.tol) == 5.85e-4
    assert pytest.approx(joint.link_len_a) == 475.0
    assert pytest.approx(joint.link_len_b) == 455.0
    assert pytest.approx(joint.twist_angle_alpha) == 485.0
    assert pytest.approx(joint.offset_distance_s) == 405.0
    assert pytest.approx(joint.offset_distance_r) == 405.0
    assert pytest.approx(joint.offset_distance_v) == 405.0
    assert pytest.approx(joint.offset_distance_h) == 405.0
    assert pytest.approx(joint.offset_distance_u) == 405.0
    assert pytest.approx(joint.offset_distance_f) == 405.0


def test_m427_lagmul_clifford_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Clifford Spatial Linkage Joint Free Format Test
/LAGMUL/CLIFFORD_SPATIAL_LINKAGE_JOINT/776
911, 912, 913, 6.95e7, 416, 5.95e-4
495.0, 465.0, 505.0, 420.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 776 in model.lagmul_clifford_spatial_linkage_joints
    joint = model.lagmul_clifford_spatial_linkage_joints[776]
    assert joint.node1 == 911
    assert joint.node2 == 912
    assert joint.node3 == 913
    assert pytest.approx(joint.stiff) == 6.95e7
    assert joint.skew_id == 416
    assert pytest.approx(joint.tol) == 5.95e-4
    assert pytest.approx(joint.link_len_a) == 495.0
    assert pytest.approx(joint.link_len_b) == 465.0
    assert pytest.approx(joint.twist_angle_alpha) == 505.0
    assert pytest.approx(joint.offset_distance_s) == 420.0


def test_m427_lagmul_clifford_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Clifford Spatial Linkage Joint Aliases Test
/CLIFFORD_SPATIAL_LINKAGE_JOINT/777
1011, 1012, 1013, 1.0e6, 0, 1.0e-6
435.0, 425.0, 435.0, 385.0
/LAGMUL/CLIFFORD_SPATIAL_LINKAGE/778
1011, 1012, 1013, 1.0e6, 0, 1.0e-6
435.0, 425.0, 435.0, 385.0
/CLIFFORD_SPATIAL_LINKAGE/779
1011, 1012, 1013, 1.0e6, 0, 1.0e-6
435.0, 425.0, 435.0, 385.0
/CLIFFORD_SPATIAL_MULTI_LOOP_MECHANISM/780
1011, 1012, 1013, 1.0e6, 0, 1.0e-6
435.0, 425.0, 435.0, 385.0
/CLIFFORD_SPATIAL_SYMMETRIC_MECHANISM/781
1011, 1012, 1013, 1.0e6, 0, 1.0e-6
435.0, 425.0, 435.0, 385.0
/CLIFFORD_SPATIAL_6R_MECHANISM/782
1011, 1012, 1013, 1.0e6, 0, 1.0e-6
435.0, 425.0, 435.0, 385.0
/CLIFFORD_SPATIAL_OVERCONSTRAINED_MECHANISM/783
1011, 1012, 1013, 1.0e6, 0, 1.0e-6
435.0, 425.0, 435.0, 385.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(777, 784):
        assert jid in model.lagmul_clifford_spatial_linkage_joints


def test_m427_lagmul_clifford_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/CLIFFORD_SPATIAL_LINKAGE_JOINT/784
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m427_sensor_spring_torsional_rate_of_change_fixed(tmp_path: Path):
    c1 = f"{1858:>10d}{7.98e8:>20.4f}{0.3445:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Rate Of Change Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_RATE_OF_CHANGE/1
Fixed Spring Torsional Rate Of Change Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_rate_of_changes
    s1 = model.sensor_spring_torsional_rate_of_changes[1]
    assert s1.spring_id == 1858
    assert pytest.approx(s1.jtors_roc_max) == 7.98e8
    assert pytest.approx(s1.jtors_lock_max) == 7.98e8
    assert pytest.approx(s1.jtors_snp_max) == 7.98e8
    assert pytest.approx(s1.jtors_crackle_max) == 7.98e8
    assert pytest.approx(s1.jtors_shot_max) == 7.98e8
    assert pytest.approx(s1.jtors_pop_max) == 7.98e8
    assert pytest.approx(s1.jtors_crk_max) == 7.98e8
    assert pytest.approx(s1.jtors_drop_rate_max) == 7.98e8
    assert pytest.approx(s1.jtors_rate_max) == 7.98e8
    assert pytest.approx(s1.jtors_roc_rate_max) == 7.98e8
    assert pytest.approx(s1.t_delay) == 0.3445
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_RATE_OF_CHANGE"


def test_m427_sensor_spring_torsional_rate_of_change_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Torsional Rate Of Change Free Format Test
/SENSOR/SPRING_TORSIONAL_RATE_OF_CHANGE/2
Free Spring Torsional Rate Of Change Sensor
1859, 8.58e8, 0.4435
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_torsional_rate_of_changes
    s2 = model.sensor_spring_torsional_rate_of_changes[2]
    assert s2.spring_id == 1859
    assert pytest.approx(s2.jtors_roc_max) == 8.58e8
    assert pytest.approx(s2.t_delay) == 0.4435


def test_m427_sensor_spring_torsional_rate_of_change_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Torsional Rate Of Change Aliases Test
/SENSOR/SPRING_TORS_RATE_OF_CHANGE/721
1879, 5.095e8, 0.3865
/SENSOR/SPRING_RATE_OF_CHANGE_TORS/722
1880, 5.115e8, 0.3875
/SENSOR/TORSIONAL_RATE_OF_CHANGE_SPRING/723
1881, 5.135e8, 0.3885
/SENSOR/SPRING_ROC_TORS/724
1882, 5.155e8, 0.3895
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(721, 725):
        assert sid in model.sensor_spring_torsional_rate_of_changes


def test_m427_sensor_spring_torsional_rate_of_change_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TORSIONAL_RATE_OF_CHANGE/725
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
