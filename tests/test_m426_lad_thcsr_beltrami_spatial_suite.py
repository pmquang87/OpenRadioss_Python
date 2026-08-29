"""Tests for Milestone M426: LadTransverseHoneycombCoreShearingRate Failure Model, EngElectrothermoflexomagnetoplasmonicmagnonicphononicpolaritonicResonanceEnergy, BeltramiSpatialLinkageJoint, and SensorSpringTotalAngularDropRate."""

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


def test_m426_fail_lad_transverse_honeycomb_core_shearing_rate_fixed(tmp_path: Path):
    c1 = f"{855.0:>20.4f}{2565.0:>20.4f}{635.0:>20.4f}{13.05:>20.4f}{0.595:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2400:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Honeycomb Core Shearing Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_HONEYCOMB_CORE_SHEARING_RATE/2400
Ladeveze Transverse Honeycomb Core Shearing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2400 in model.fail_ladtransversehoneycombcoreshearingrates
    fthcs = model.fail_ladtransversehoneycombcoreshearingrates[2400]
    assert pytest.approx(fthcs.sigma_thcs0) == 855.0
    assert pytest.approx(fthcs.sigma_thcsc) == 2565.0
    assert pytest.approx(fthcs.gamma_thcs) == 635.0
    assert pytest.approx(fthcs.p_thcs) == 13.05
    assert pytest.approx(fthcs.d_thcs_max) == 0.595
    assert fthcs.ifail_sh == 1
    assert fthcs.ifail_so == 2
    assert fthcs.fail_id == 2400
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_HONEYCOMB_CORE_SHEARING_RATE"


def test_m426_fail_lad_transverse_honeycomb_core_shearing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Honeycomb Core Shearing Rate Free Format Test
/FAIL/LAD_TRANSVERSE_HONEYCOMB_CORE_SHEARING_RATE/2401
865.0, 2595.0, 645.0, 13.25, 0.585
1, 1
2401
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2401 in model.fail_ladtransversehoneycombcoreshearingrates
    fthcs = model.fail_ladtransversehoneycombcoreshearingrates[2401]
    assert pytest.approx(fthcs.sigma_thcs0) == 865.0
    assert pytest.approx(fthcs.sigma_thcsc) == 2595.0
    assert pytest.approx(fthcs.gamma_thcs) == 645.0
    assert pytest.approx(fthcs.p_thcs) == 13.25
    assert pytest.approx(fthcs.d_thcs_max) == 0.585
    assert fthcs.fail_id == 2401


def test_m426_fail_lad_transverse_honeycomb_core_shearing_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Honeycomb Core Shearing Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_HONEYCOMB_CORE_SHEARING_RATE/2402
845.0, 2535.0, 625.0, 12.85, 0.605
1, 1
/FAIL/LAD_THCSR/2403
845.0, 2535.0, 625.0, 12.85, 0.605
1, 1
/FAIL/LAD_THCSR_MODEL/2404
845.0, 2535.0, 625.0, 12.85, 0.605
1, 1
/FAIL/LAD_THCSR_LAW/2405
845.0, 2535.0, 625.0, 12.85, 0.605
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_HONEYCOMB_CORE_SHEARING/2406
845.0, 2535.0, 625.0, 12.85, 0.605
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2402 in model.fail_ladtransversehoneycombcoreshearingrates
    assert 2403 in model.fail_ladtransversehoneycombcoreshearingrates
    assert 2404 in model.fail_ladtransversehoneycombcoreshearingrates
    assert 2405 in model.fail_ladtransversehoneycombcoreshearingrates
    assert 2406 in model.fail_ladtransversehoneycombcoreshearingrates
    assert len(model.raw_fails) == 5


def test_m426_fail_lad_transverse_honeycomb_core_shearing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_TRANSVERSE_HONEYCOMB_CORE_SHEARING_RATE/2407
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m426_eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0655:>20.4f}{815:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/715
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 715 in model.eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energies[715]
    assert pytest.approx(eng.dt_etfmpxmpp) == 0.0655
    assert eng.sens_id == 815


def test_m426_eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/716
0.0665, 816
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 716 in model.eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energies[716]
    assert pytest.approx(eng.dt_etfmpxmpp) == 0.0665
    assert eng.sens_id == 816


def test_m426_eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_MAGNON_PHONON_POLARITON_RES_WORK/717
0.0675, 817
/ENG/EELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPHONONICPOLARITONICRESONANCE/718
0.0685, 818
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_DISSIPATION/719
0.0695, 819
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE/720
0.0705, 820
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 717 in model.eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energies
    assert 718 in model.eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energies
    assert 719 in model.eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energies
    assert 720 in model.eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energies


def test_m426_eng_electrothermoflexomagnetoplasmonicmagnonicphononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/721
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m426_lagmul_beltrami_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{801:>10d}{802:>10d}{803:>10d}{6.55e7:>20.4f}{405:>10d}{5.65e-4:>20.4e}"
    c2 = f"{465.0:>20.4f}{445.0:>20.4f}{475.0:>20.4f}{395.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Beltrami Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/BELTRAMI_SPATIAL_LINKAGE_JOINT/765
Beltrami Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 765 in model.lagmul_beltrami_spatial_linkage_joints
    joint = model.lagmul_beltrami_spatial_linkage_joints[765]
    assert joint.node1 == 801
    assert joint.node2 == 802
    assert joint.node3 == 803
    assert pytest.approx(joint.stiff) == 6.55e7
    assert joint.skew_id == 405
    assert pytest.approx(joint.tol) == 5.65e-4
    assert pytest.approx(joint.link_len_a) == 465.0
    assert pytest.approx(joint.link_len_b) == 445.0
    assert pytest.approx(joint.twist_angle_alpha) == 475.0
    assert pytest.approx(joint.offset_distance_s) == 395.0
    assert pytest.approx(joint.offset_distance_r) == 395.0
    assert pytest.approx(joint.offset_distance_v) == 395.0
    assert pytest.approx(joint.offset_distance_h) == 395.0
    assert pytest.approx(joint.offset_distance_u) == 395.0
    assert pytest.approx(joint.offset_distance_f) == 395.0


def test_m426_lagmul_beltrami_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Beltrami Spatial Linkage Joint Free Format Test
/LAGMUL/BELTRAMI_SPATIAL_LINKAGE_JOINT/766
901, 902, 903, 6.85e7, 406, 5.75e-4
485.0, 455.0, 495.0, 410.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 766 in model.lagmul_beltrami_spatial_linkage_joints
    joint = model.lagmul_beltrami_spatial_linkage_joints[766]
    assert joint.node1 == 901
    assert joint.node2 == 902
    assert joint.node3 == 903
    assert pytest.approx(joint.stiff) == 6.85e7
    assert joint.skew_id == 406
    assert pytest.approx(joint.tol) == 5.75e-4
    assert pytest.approx(joint.link_len_a) == 485.0
    assert pytest.approx(joint.link_len_b) == 455.0
    assert pytest.approx(joint.twist_angle_alpha) == 495.0
    assert pytest.approx(joint.offset_distance_s) == 410.0


def test_m426_lagmul_beltrami_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Beltrami Spatial Linkage Joint Aliases Test
/BELTRAMI_SPATIAL_LINKAGE_JOINT/767
1001, 1002, 1003, 1.0e6, 0, 1.0e-6
425.0, 415.0, 425.0, 375.0
/LAGMUL/BELTRAMI_SPATIAL_LINKAGE/768
1001, 1002, 1003, 1.0e6, 0, 1.0e-6
425.0, 415.0, 425.0, 375.0
/BELTRAMI_SPATIAL_LINKAGE/769
1001, 1002, 1003, 1.0e6, 0, 1.0e-6
425.0, 415.0, 425.0, 375.0
/BELTRAMI_SPATIAL_MULTI_LOOP_MECHANISM/770
1001, 1002, 1003, 1.0e6, 0, 1.0e-6
425.0, 415.0, 425.0, 375.0
/BELTRAMI_SPATIAL_SYMMETRIC_MECHANISM/771
1001, 1002, 1003, 1.0e6, 0, 1.0e-6
425.0, 415.0, 425.0, 375.0
/BELTRAMI_SPATIAL_6R_MECHANISM/772
1001, 1002, 1003, 1.0e6, 0, 1.0e-6
425.0, 415.0, 425.0, 375.0
/BELTRAMI_SPATIAL_OVERCONSTRAINED_MECHANISM/773
1001, 1002, 1003, 1.0e6, 0, 1.0e-6
425.0, 415.0, 425.0, 375.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(767, 774):
        assert jid in model.lagmul_beltrami_spatial_linkage_joints


def test_m426_lagmul_beltrami_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/BELTRAMI_SPATIAL_LINKAGE_JOINT/774
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m426_sensor_spring_total_angular_drop_rate_fixed(tmp_path: Path):
    c1 = f"{1848:>10d}{7.78e8:>20.4f}{0.3245:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Angular Drop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_DROP_RATE/1
Fixed Spring Total Angular Drop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_angular_drop_rates
    s1 = model.sensor_spring_total_angular_drop_rates[1]
    assert s1.spring_id == 1848
    assert pytest.approx(s1.jtot_ang_drop_max) == 7.78e8
    assert pytest.approx(s1.jtot_ang_lock_max) == 7.78e8
    assert pytest.approx(s1.jtot_ang_snp_max) == 7.78e8
    assert pytest.approx(s1.jtot_ang_crackle_max) == 7.78e8
    assert pytest.approx(s1.jtot_ang_shot_max) == 7.78e8
    assert pytest.approx(s1.jtot_ang_pop_max) == 7.78e8
    assert pytest.approx(s1.jtot_ang_crk_max) == 7.78e8
    assert pytest.approx(s1.jtot_ang_drop_rate_max) == 7.78e8
    assert pytest.approx(s1.jtot_ang_rate_max) == 7.78e8
    assert pytest.approx(s1.t_delay) == 0.3245
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_DROP_RATE"


def test_m426_sensor_spring_total_angular_drop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Angular Drop Rate Free Format Test
/SENSOR/SPRING_TOTAL_ANGULAR_DROP_RATE/2
Free Spring Total Angular Drop Rate Sensor
1849, 8.38e8, 0.4235
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_angular_drop_rates
    s2 = model.sensor_spring_total_angular_drop_rates[2]
    assert s2.spring_id == 1849
    assert pytest.approx(s2.jtot_ang_drop_max) == 8.38e8
    assert pytest.approx(s2.t_delay) == 0.4235


def test_m426_sensor_spring_total_angular_drop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Angular Drop Rate Aliases Test
/SENSOR/SPRING_TOT_ANG_DROP_RATE/711
1869, 4.895e8, 0.3765
/SENSOR/SPRING_RATE_DROP_ANG_TOT/712
1870, 4.915e8, 0.3775
/SENSOR/TOTAL_ANGULAR_DROP_RATE_SPRING/713
1871, 4.935e8, 0.3785
/SENSOR/SPRING_DROP_ANG_TOT/714
1872, 4.955e8, 0.3795
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(711, 715):
        assert sid in model.sensor_spring_total_angular_drop_rates


def test_m426_sensor_spring_total_angular_drop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TOTAL_ANGULAR_DROP_RATE/715
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
