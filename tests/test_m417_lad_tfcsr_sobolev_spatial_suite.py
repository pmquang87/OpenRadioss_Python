"""Tests for Milestone M417: LadTransverseFacesheetCoreShearingRate Failure Model, EngElectrothermoflexomagnetomagnonicphononicpolaritonicResonanceEnergy, SobolevSpatialLinkageJoint, and SensorSpringTotalLockRate."""

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


def test_m417_fail_lad_transverse_facesheet_core_shearing_rate_fixed(tmp_path: Path):
    c1 = f"{725.0:>20.4f}{2175.0:>20.4f}{505.0:>20.4f}{10.45:>20.4f}{0.705:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2310:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Facesheet-Core Shearing Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_FACESHEET_CORE_SHEARING_RATE/2310
Ladeveze Transverse Facesheet-Core Shearing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2310 in model.fail_ladtransversefacesheetcoreshearingrates
    ftfcs = model.fail_ladtransversefacesheetcoreshearingrates[2310]
    assert pytest.approx(ftfcs.sigma_tfcs0) == 725.0
    assert pytest.approx(ftfcs.sigma_tfcsc) == 2175.0
    assert pytest.approx(ftfcs.gamma_tfcs) == 505.0
    assert pytest.approx(ftfcs.p_tfcs) == 10.45
    assert pytest.approx(ftfcs.d_tfcs_max) == 0.705
    assert ftfcs.ifail_sh == 1
    assert ftfcs.ifail_so == 2
    assert ftfcs.fail_id == 2310
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_FACESHEET_CORE_SHEARING_RATE"


def test_m417_fail_lad_transverse_facesheet_core_shearing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Facesheet-Core Shearing Rate Free Format Test
/FAIL/LAD_TRANSVERSE_FACESHEET_CORE_SHEARING_RATE/2311
735.0, 2205.0, 515.0, 10.65, 0.695
1, 1
2311
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2311 in model.fail_ladtransversefacesheetcoreshearingrates
    ftfcs = model.fail_ladtransversefacesheetcoreshearingrates[2311]
    assert pytest.approx(ftfcs.sigma_tfcs0) == 735.0
    assert pytest.approx(ftfcs.sigma_tfcsc) == 2205.0
    assert pytest.approx(ftfcs.gamma_tfcs) == 515.0
    assert pytest.approx(ftfcs.p_tfcs) == 10.65
    assert pytest.approx(ftfcs.d_tfcs_max) == 0.695
    assert ftfcs.fail_id == 2311


def test_m417_fail_lad_transverse_facesheet_core_shearing_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Facesheet-Core Shearing Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_FACESHEET_CORE_SHEARING_RATE/2312
715.0, 2145.0, 500.0, 10.25, 0.715
1, 1
/FAIL/LAD_TFCSR/2313
715.0, 2145.0, 500.0, 10.25, 0.715
1, 1
/FAIL/LAD_TFCSR_MODEL/2314
715.0, 2145.0, 500.0, 10.25, 0.715
1, 1
/FAIL/LAD_TFCSR_LAW/2315
715.0, 2145.0, 500.0, 10.25, 0.715
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_FACESHEET_CORE_SHEARING/2316
715.0, 2145.0, 500.0, 10.25, 0.715
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2312 in model.fail_ladtransversefacesheetcoreshearingrates
    assert 2313 in model.fail_ladtransversefacesheetcoreshearingrates
    assert 2314 in model.fail_ladtransversefacesheetcoreshearingrates
    assert 2315 in model.fail_ladtransversefacesheetcoreshearingrates
    assert 2316 in model.fail_ladtransversefacesheetcoreshearingrates
    assert len(model.raw_fails) == 5


def test_m417_fail_lad_transverse_facesheet_core_shearing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_TRANSVERSE_FACESHEET_CORE_SHEARING_RATE/2317
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m417_eng_electrothermoflexomagnetomagnonicphononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0445:>20.4f}{675:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetomagnonicphononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/575
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 575 in model.eng_electrothermoflexomagnetomagnonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetomagnonicphononicpolaritonic_resonance_energies[575]
    assert pytest.approx(eng.dt_etfmmppp) == 0.0445
    assert eng.sens_id == 675


def test_m417_eng_electrothermoflexomagnetomagnonicphononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetomagnonicphononicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/576
0.0455, 676
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 576 in model.eng_electrothermoflexomagnetomagnonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetomagnonicphononicpolaritonic_resonance_energies[576]
    assert pytest.approx(eng.dt_etfmmppp) == 0.0455
    assert eng.sens_id == 676


def test_m417_eng_electrothermoflexomagnetomagnonicphononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetomagnonicphononicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_MAGNON_PHONON_POLARITON_RES_WORK/577
0.0465, 677
/ENG/EELECTROTHERMOFLEXOMAGNETOMAGNONICPHONONICPOLARITONICRESONANCE/578
0.0475, 678
/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICPHONONICPOLARITONIC_RESONANCE_DISSIPATION/579
0.0485, 679
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOMAGNONICPHONONICPOLARITONIC_RESONANCE/580
0.0495, 680
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 577 in model.eng_electrothermoflexomagnetomagnonicphononicpolaritonic_resonance_energies
    assert 578 in model.eng_electrothermoflexomagnetomagnonicphononicpolaritonic_resonance_energies
    assert 579 in model.eng_electrothermoflexomagnetomagnonicphononicpolaritonic_resonance_energies
    assert 580 in model.eng_electrothermoflexomagnetomagnonicphononicpolaritonic_resonance_energies


def test_m417_eng_electrothermoflexomagnetomagnonicphononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/581
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m417_lagmul_sobolev_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{661:>10d}{662:>10d}{663:>10d}{5.15e7:>20.4f}{315:>10d}{4.25e-4:>20.4e}"
    c2 = f"{375.0:>20.4f}{355.0:>20.4f}{385.0:>20.4f}{305.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sobolev Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/SOBOLEV_SPATIAL_LINKAGE_JOINT/625
Sobolev Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 625 in model.lagmul_sobolev_spatial_linkage_joints
    joint = model.lagmul_sobolev_spatial_linkage_joints[625]
    assert joint.node1 == 661
    assert joint.node2 == 662
    assert joint.node3 == 663
    assert pytest.approx(joint.stiff) == 5.15e7
    assert joint.skew_id == 315
    assert pytest.approx(joint.tol) == 4.25e-4
    assert pytest.approx(joint.link_len_a) == 375.0
    assert pytest.approx(joint.link_len_b) == 355.0
    assert pytest.approx(joint.twist_angle_alpha) == 385.0
    assert pytest.approx(joint.offset_distance_s) == 305.0
    assert pytest.approx(joint.offset_distance_r) == 305.0
    assert pytest.approx(joint.offset_distance_v) == 305.0
    assert pytest.approx(joint.offset_distance_h) == 305.0
    assert pytest.approx(joint.offset_distance_u) == 305.0


def test_m417_lagmul_sobolev_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sobolev Spatial Linkage Joint Free Format Test
/LAGMUL/SOBOLEV_SPATIAL_LINKAGE_JOINT/626
761, 762, 763, 5.45e7, 316, 4.35e-4
395.0, 365.0, 405.0, 320.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 626 in model.lagmul_sobolev_spatial_linkage_joints
    joint = model.lagmul_sobolev_spatial_linkage_joints[626]
    assert joint.node1 == 761
    assert joint.node2 == 762
    assert joint.node3 == 763
    assert pytest.approx(joint.stiff) == 5.45e7
    assert joint.skew_id == 316
    assert pytest.approx(joint.tol) == 4.35e-4
    assert pytest.approx(joint.link_len_a) == 395.0
    assert pytest.approx(joint.link_len_b) == 365.0
    assert pytest.approx(joint.twist_angle_alpha) == 405.0
    assert pytest.approx(joint.offset_distance_s) == 320.0


def test_m417_lagmul_sobolev_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sobolev Spatial Linkage Joint Aliases Test
/SOBOLEV_SPATIAL_LINKAGE_JOINT/627
861, 862, 863, 1.0e6, 0, 1.0e-6
335.0, 325.0, 335.0, 285.0
/LAGMUL/SOBOLEV_SPATIAL_LINKAGE/628
861, 862, 863, 1.0e6, 0, 1.0e-6
335.0, 325.0, 335.0, 285.0
/SOBOLEV_SPATIAL_LINKAGE/629
861, 862, 863, 1.0e6, 0, 1.0e-6
335.0, 325.0, 335.0, 285.0
/SOBOLEV_SPATIAL_MULTI_LOOP_MECHANISM/630
861, 862, 863, 1.0e6, 0, 1.0e-6
335.0, 325.0, 335.0, 285.0
/SOBOLEV_SPATIAL_SYMMETRIC_MECHANISM/631
861, 862, 863, 1.0e6, 0, 1.0e-6
335.0, 325.0, 335.0, 285.0
/SOBOLEV_SPATIAL_6R_MECHANISM/632
861, 862, 863, 1.0e6, 0, 1.0e-6
335.0, 325.0, 335.0, 285.0
/SOBOLEV_SPATIAL_OVERCONSTRAINED_MECHANISM/633
861, 862, 863, 1.0e6, 0, 1.0e-6
335.0, 325.0, 335.0, 285.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(627, 634):
        assert jid in model.lagmul_sobolev_spatial_linkage_joints


def test_m417_lagmul_sobolev_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/SOBOLEV_SPATIAL_LINKAGE_JOINT/634
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m417_sensor_spring_total_lock_rate_fixed(tmp_path: Path):
    c1 = f"{1708:>10d}{6.38e8:>20.4f}{0.2345:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Lock Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_LOCK_RATE/1
Fixed Spring Total Lock Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_lock_rates
    s1 = model.sensor_spring_total_lock_rates[1]
    assert s1.spring_id == 1708
    assert pytest.approx(s1.jtot_lock_max) == 6.38e8
    assert pytest.approx(s1.jtot_snp_max) == 6.38e8
    assert pytest.approx(s1.jtot_crackle_max) == 6.38e8
    assert pytest.approx(s1.jtot_shot_max) == 6.38e8
    assert pytest.approx(s1.jtot_drop_max) == 6.38e8
    assert pytest.approx(s1.jtot_pop_max) == 6.38e8
    assert pytest.approx(s1.jtot_crk_max) == 6.38e8
    assert pytest.approx(s1.t_delay) == 0.2345
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_LOCK_RATE"


def test_m417_sensor_spring_total_lock_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Lock Rate Free Format Test
/SENSOR/SPRING_TOTAL_LOCK_RATE/2
Free Spring Total Lock Rate Sensor
1709, 6.98e8, 0.3335
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_lock_rates
    s2 = model.sensor_spring_total_lock_rates[2]
    assert s2.spring_id == 1709
    assert pytest.approx(s2.jtot_lock_max) == 6.98e8
    assert pytest.approx(s2.t_delay) == 0.3335


def test_m417_sensor_spring_total_lock_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Lock Rate Aliases Test
/SENSOR/SPRING_TOT_LOCK_RATE/571
1729, 3.995e8, 0.2865
/SENSOR/SPRING_RATE_LOCK_TOT/572
1730, 4.015e8, 0.2875
/SENSOR/TOTAL_LOCK_RATE_SPRING/573
1731, 4.035e8, 0.2885
/SENSOR/SPRING_LOCK_TOT/574
1732, 4.055e8, 0.2895
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(571, 575):
        assert sid in model.sensor_spring_total_lock_rates


def test_m417_sensor_spring_total_lock_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TOTAL_LOCK_RATE/575
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
