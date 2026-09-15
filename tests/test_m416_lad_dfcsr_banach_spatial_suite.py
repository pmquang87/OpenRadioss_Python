"""Tests for Milestone M416: LadDynamicFacesheetCoreShearingRate Failure Model, EngElectrothermoflexomagnetoexcitonicphononicpolaritonicResonanceEnergy, BanachSpatialLinkageJoint, and SensorSpringTransverseLockRate."""

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


def test_m416_fail_lad_dynamic_facesheet_core_shearing_rate_fixed(tmp_path: Path):
    c1 = f"{715.0:>20.4f}{2145.0:>20.4f}{495.0:>20.4f}{10.25:>20.4f}{0.715:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2300:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Facesheet-Core Shearing Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_FACESHEET_CORE_SHEARING_RATE/2300
Ladeveze Dynamic Facesheet-Core Shearing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2300 in model.fail_laddynamicfacesheetcoreshearingrates
    fdfcs = model.fail_laddynamicfacesheetcoreshearingrates[2300]
    assert pytest.approx(fdfcs.sigma_dfcs0) == 715.0
    assert pytest.approx(fdfcs.sigma_dfcsc) == 2145.0
    assert pytest.approx(fdfcs.gamma_dfcs) == 495.0
    assert pytest.approx(fdfcs.p_dfcs) == 10.25
    assert pytest.approx(fdfcs.d_dfcs_max) == 0.715
    assert fdfcs.ifail_sh == 1
    assert fdfcs.ifail_so == 2
    assert fdfcs.fail_id == 2300
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_FACESHEET_CORE_SHEARING_RATE"


def test_m416_fail_lad_dynamic_facesheet_core_shearing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Facesheet-Core Shearing Rate Free Format Test
/FAIL/LAD_DYNAMIC_FACESHEET_CORE_SHEARING_RATE/2301
725.0, 2175.0, 505.0, 10.45, 0.705
1, 1
2301
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2301 in model.fail_laddynamicfacesheetcoreshearingrates
    fdfcs = model.fail_laddynamicfacesheetcoreshearingrates[2301]
    assert pytest.approx(fdfcs.sigma_dfcs0) == 725.0
    assert pytest.approx(fdfcs.sigma_dfcsc) == 2175.0
    assert pytest.approx(fdfcs.gamma_dfcs) == 505.0
    assert pytest.approx(fdfcs.p_dfcs) == 10.45
    assert pytest.approx(fdfcs.d_dfcs_max) == 0.705
    assert fdfcs.fail_id == 2301


def test_m416_fail_lad_dynamic_facesheet_core_shearing_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Facesheet-Core Shearing Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_FACESHEET_CORE_SHEARING_RATE/2302
705.0, 2115.0, 490.0, 10.05, 0.725
1, 1
/FAIL/LAD_DFCSR/2303
705.0, 2115.0, 490.0, 10.05, 0.725
1, 1
/FAIL/LAD_DFCSR_MODEL/2304
705.0, 2115.0, 490.0, 10.05, 0.725
1, 1
/FAIL/LAD_DFCSR_LAW/2305
705.0, 2115.0, 490.0, 10.05, 0.725
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_FACESHEET_CORE_SHEARING/2306
705.0, 2115.0, 490.0, 10.05, 0.725
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2302 in model.fail_laddynamicfacesheetcoreshearingrates
    assert 2303 in model.fail_laddynamicfacesheetcoreshearingrates
    assert 2304 in model.fail_laddynamicfacesheetcoreshearingrates
    assert 2305 in model.fail_laddynamicfacesheetcoreshearingrates
    assert 2306 in model.fail_laddynamicfacesheetcoreshearingrates
    assert len(model.raw_fails) == 5


def test_m416_fail_lad_dynamic_facesheet_core_shearing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_FACESHEET_CORE_SHEARING_RATE/2307
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m416_eng_electrothermoflexomagnetoexcitonicphononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0425:>20.4f}{655:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetoexcitonicphononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY/555
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 555 in model.eng_electrothermoflexomagnetoexcitonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoexcitonicphononicpolaritonic_resonance_energies[555]
    assert pytest.approx(eng.dt_etfmeppp) == 0.0425
    assert eng.sens_id == 655


def test_m416_eng_electrothermoflexomagnetoexcitonicphononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetoexcitonicphononicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY/556
0.0435, 656
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 556 in model.eng_electrothermoflexomagnetoexcitonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoexcitonicphononicpolaritonic_resonance_energies[556]
    assert pytest.approx(eng.dt_etfmeppp) == 0.0435
    assert eng.sens_id == 656


def test_m416_eng_electrothermoflexomagnetoexcitonicphononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetoexcitonicphononicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_PHONON_POLARITON_RES_WORK/557
0.0445, 657
/ENG/EELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICPOLARITONICRESONANCE/558
0.0455, 658
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICPOLARITONIC_RESONANCE_DISSIPATION/559
0.0465, 659
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICPOLARITONIC_RESONANCE/560
0.0475, 660
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 557 in model.eng_electrothermoflexomagnetoexcitonicphononicpolaritonic_resonance_energies
    assert 558 in model.eng_electrothermoflexomagnetoexcitonicphononicpolaritonic_resonance_energies
    assert 559 in model.eng_electrothermoflexomagnetoexcitonicphononicpolaritonic_resonance_energies
    assert 560 in model.eng_electrothermoflexomagnetoexcitonicphononicpolaritonic_resonance_energies


def test_m416_eng_electrothermoflexomagnetoexcitonicphononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY/561
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m416_lagmul_banach_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{641:>10d}{642:>10d}{643:>10d}{4.95e7:>20.4f}{305:>10d}{4.05e-4:>20.4e}"
    c2 = f"{365.0:>20.4f}{345.0:>20.4f}{375.0:>20.4f}{295.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Banach Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/BANACH_SPATIAL_LINKAGE_JOINT/605
Banach Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 605 in model.lagmul_banach_spatial_linkage_joints
    joint = model.lagmul_banach_spatial_linkage_joints[605]
    assert joint.node1 == 641
    assert joint.node2 == 642
    assert joint.node3 == 643
    assert pytest.approx(joint.stiff) == 4.95e7
    assert joint.skew_id == 305
    assert pytest.approx(joint.tol) == 4.05e-4
    assert pytest.approx(joint.link_len_a) == 365.0
    assert pytest.approx(joint.link_len_b) == 345.0
    assert pytest.approx(joint.twist_angle_alpha) == 375.0
    assert pytest.approx(joint.offset_distance_s) == 295.0
    assert pytest.approx(joint.offset_distance_r) == 295.0
    assert pytest.approx(joint.offset_distance_v) == 295.0
    assert pytest.approx(joint.offset_distance_h) == 295.0
    assert pytest.approx(joint.offset_distance_u) == 295.0


def test_m416_lagmul_banach_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Banach Spatial Linkage Joint Free Format Test
/LAGMUL/BANACH_SPATIAL_LINKAGE_JOINT/606
741, 742, 743, 5.25e7, 306, 4.15e-4
385.0, 355.0, 395.0, 310.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 606 in model.lagmul_banach_spatial_linkage_joints
    joint = model.lagmul_banach_spatial_linkage_joints[606]
    assert joint.node1 == 741
    assert joint.node2 == 742
    assert joint.node3 == 743
    assert pytest.approx(joint.stiff) == 5.25e7
    assert joint.skew_id == 306
    assert pytest.approx(joint.tol) == 4.15e-4
    assert pytest.approx(joint.link_len_a) == 385.0
    assert pytest.approx(joint.link_len_b) == 355.0
    assert pytest.approx(joint.twist_angle_alpha) == 395.0
    assert pytest.approx(joint.offset_distance_s) == 310.0


def test_m416_lagmul_banach_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Banach Spatial Linkage Joint Aliases Test
/BANACH_SPATIAL_LINKAGE_JOINT/607
841, 842, 843, 1.0e6, 0, 1.0e-6
325.0, 315.0, 325.0, 275.0
/LAGMUL/BANACH_SPATIAL_LINKAGE/608
841, 842, 843, 1.0e6, 0, 1.0e-6
325.0, 315.0, 325.0, 275.0
/BANACH_SPATIAL_LINKAGE/609
841, 842, 843, 1.0e6, 0, 1.0e-6
325.0, 315.0, 325.0, 275.0
/BANACH_SPATIAL_MULTI_LOOP_MECHANISM/610
841, 842, 843, 1.0e6, 0, 1.0e-6
325.0, 315.0, 325.0, 275.0
/BANACH_SPATIAL_SYMMETRIC_MECHANISM/611
841, 842, 843, 1.0e6, 0, 1.0e-6
325.0, 315.0, 325.0, 275.0
/BANACH_SPATIAL_6R_MECHANISM/612
841, 842, 843, 1.0e6, 0, 1.0e-6
325.0, 315.0, 325.0, 275.0
/BANACH_SPATIAL_OVERCONSTRAINED_MECHANISM/613
841, 842, 843, 1.0e6, 0, 1.0e-6
325.0, 315.0, 325.0, 275.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(607, 614):
        assert jid in model.lagmul_banach_spatial_linkage_joints


def test_m416_lagmul_banach_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/BANACH_SPATIAL_LINKAGE_JOINT/614
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m416_sensor_spring_transverse_lock_rate_fixed(tmp_path: Path):
    c1 = f"{1688:>10d}{6.18e8:>20.4f}{0.2245:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Transverse Lock Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_LOCK_RATE/1
Fixed Spring Transverse Lock Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_transverse_lock_rates
    s1 = model.sensor_spring_transverse_lock_rates[1]
    assert s1.spring_id == 1688
    assert pytest.approx(s1.jtrans_lock_max) == 6.18e8
    assert pytest.approx(s1.jtrans_snp_max) == 6.18e8
    assert pytest.approx(s1.jtrans_crackle_max) == 6.18e8
    assert pytest.approx(s1.jtrans_shot_max) == 6.18e8
    assert pytest.approx(s1.jtrans_drop_max) == 6.18e8
    assert pytest.approx(s1.jtrans_pop_max) == 6.18e8
    assert pytest.approx(s1.jtrans_crk_max) == 6.18e8
    assert pytest.approx(s1.t_delay) == 0.2245
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_LOCK_RATE"


def test_m416_sensor_spring_transverse_lock_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Transverse Lock Rate Free Format Test
/SENSOR/SPRING_TRANSVERSE_LOCK_RATE/2
Free Spring Transverse Lock Rate Sensor
1689, 6.78e8, 0.3235
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_transverse_lock_rates
    s2 = model.sensor_spring_transverse_lock_rates[2]
    assert s2.spring_id == 1689
    assert pytest.approx(s2.jtrans_lock_max) == 6.78e8
    assert pytest.approx(s2.t_delay) == 0.3235


def test_m416_sensor_spring_transverse_lock_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Transverse Lock Rate Aliases Test
/SENSOR/SPRING_TRANS_LOCK_RATE/551
1709, 3.895e8, 0.2765
/SENSOR/SPRING_RATE_LOCK_TRANS/552
1710, 3.915e8, 0.2775
/SENSOR/TRANSVERSE_LOCK_RATE_SPRING/553
1711, 3.935e8, 0.2785
/SENSOR/SPRING_LOCK_TRANS/554
1712, 3.955e8, 0.2795
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(551, 555):
        assert sid in model.sensor_spring_transverse_lock_rates


def test_m416_sensor_spring_transverse_lock_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TRANSVERSE_LOCK_RATE/555
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
