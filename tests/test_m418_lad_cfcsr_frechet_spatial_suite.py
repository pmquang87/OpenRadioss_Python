"""Tests for Milestone M418: LadCoupleFacesheetCoreShearingRate Failure Model, EngElectrothermoflexomagnetoexcitonicmagnonicphononicpolaritonicResonanceEnergy, FrechetSpatialLinkageJoint, and SensorSpringTorsionalLockRate."""

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


def test_m418_fail_lad_couple_facesheet_core_shearing_rate_fixed(tmp_path: Path):
    c1 = f"{735.0:>20.4f}{2205.0:>20.4f}{515.0:>20.4f}{10.65:>20.4f}{0.695:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2320:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Facesheet-Core Shearing Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_FACESHEET_CORE_SHEARING_RATE/2320
Ladeveze Coupled Facesheet-Core Shearing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2320 in model.fail_ladcouplefacesheetcoreshearingrates
    fcfcs = model.fail_ladcouplefacesheetcoreshearingrates[2320]
    assert pytest.approx(fcfcs.sigma_cfcs0) == 735.0
    assert pytest.approx(fcfcs.sigma_cfcsc) == 2205.0
    assert pytest.approx(fcfcs.gamma_cfcs) == 515.0
    assert pytest.approx(fcfcs.p_cfcs) == 10.65
    assert pytest.approx(fcfcs.d_cfcs_max) == 0.695
    assert fcfcs.ifail_sh == 1
    assert fcfcs.ifail_so == 2
    assert fcfcs.fail_id == 2320
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_FACESHEET_CORE_SHEARING_RATE"


def test_m418_fail_lad_couple_facesheet_core_shearing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Facesheet-Core Shearing Rate Free Format Test
/FAIL/LAD_COUPLE_FACESHEET_CORE_SHEARING_RATE/2321
745.0, 2235.0, 525.0, 10.85, 0.685
1, 1
2321
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2321 in model.fail_ladcouplefacesheetcoreshearingrates
    fcfcs = model.fail_ladcouplefacesheetcoreshearingrates[2321]
    assert pytest.approx(fcfcs.sigma_cfcs0) == 745.0
    assert pytest.approx(fcfcs.sigma_cfcsc) == 2235.0
    assert pytest.approx(fcfcs.gamma_cfcs) == 525.0
    assert pytest.approx(fcfcs.p_cfcs) == 10.85
    assert pytest.approx(fcfcs.d_cfcs_max) == 0.685
    assert fcfcs.fail_id == 2321


def test_m418_fail_lad_couple_facesheet_core_shearing_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Facesheet-Core Shearing Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_FACESHEET_CORE_SHEARING_RATE/2322
725.0, 2175.0, 510.0, 10.45, 0.705
1, 1
/FAIL/LAD_CFCSR/2323
725.0, 2175.0, 510.0, 10.45, 0.705
1, 1
/FAIL/LAD_CFCSR_MODEL/2324
725.0, 2175.0, 510.0, 10.45, 0.705
1, 1
/FAIL/LAD_CFCSR_LAW/2325
725.0, 2175.0, 510.0, 10.45, 0.705
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_FACESHEET_CORE_SHEARING/2326
725.0, 2175.0, 510.0, 10.45, 0.705
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2322 in model.fail_ladcouplefacesheetcoreshearingrates
    assert 2323 in model.fail_ladcouplefacesheetcoreshearingrates
    assert 2324 in model.fail_ladcouplefacesheetcoreshearingrates
    assert 2325 in model.fail_ladcouplefacesheetcoreshearingrates
    assert 2326 in model.fail_ladcouplefacesheetcoreshearingrates
    assert len(model.raw_fails) == 5


def test_m418_fail_lad_couple_facesheet_core_shearing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_COUPLE_FACESHEET_CORE_SHEARING_RATE/2327
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m418_eng_electrothermoflexomagnetoexcitonicmagnonicphononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0465:>20.4f}{695:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetoexcitonicmagnonicphononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/595
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 595 in model.eng_electrothermoflexomagnetoexcitonicmagnonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoexcitonicmagnonicphononicpolaritonic_resonance_energies[595]
    assert pytest.approx(eng.dt_etfmemppp) == 0.0465
    assert eng.sens_id == 695


def test_m418_eng_electrothermoflexomagnetoexcitonicmagnonicphononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetoexcitonicmagnonicphononicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/596
0.0475, 696
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 596 in model.eng_electrothermoflexomagnetoexcitonicmagnonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoexcitonicmagnonicphononicpolaritonic_resonance_energies[596]
    assert pytest.approx(eng.dt_etfmemppp) == 0.0475
    assert eng.sens_id == 696


def test_m418_eng_electrothermoflexomagnetoexcitonicmagnonicphononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetoexcitonicmagnonicphononicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_MAGNON_PHONON_POLARITON_RES_WORK/597
0.0485, 697
/ENG/EELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONICPHONONICPOLARITONICRESONANCE/598
0.0495, 698
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_DISSIPATION/599
0.0505, 699
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE/600
0.0515, 700
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 597 in model.eng_electrothermoflexomagnetoexcitonicmagnonicphononicpolaritonic_resonance_energies
    assert 598 in model.eng_electrothermoflexomagnetoexcitonicmagnonicphononicpolaritonic_resonance_energies
    assert 599 in model.eng_electrothermoflexomagnetoexcitonicmagnonicphononicpolaritonic_resonance_energies
    assert 600 in model.eng_electrothermoflexomagnetoexcitonicmagnonicphononicpolaritonic_resonance_energies


def test_m418_eng_electrothermoflexomagnetoexcitonicmagnonicphononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/601
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m418_lagmul_frechet_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{681:>10d}{682:>10d}{683:>10d}{5.35e7:>20.4f}{325:>10d}{4.45e-4:>20.4e}"
    c2 = f"{385.0:>20.4f}{365.0:>20.4f}{395.0:>20.4f}{315.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Frechet Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/FRECHET_SPATIAL_LINKAGE_JOINT/645
Frechet Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 645 in model.lagmul_frechet_spatial_linkage_joints
    joint = model.lagmul_frechet_spatial_linkage_joints[645]
    assert joint.node1 == 681
    assert joint.node2 == 682
    assert joint.node3 == 683
    assert pytest.approx(joint.stiff) == 5.35e7
    assert joint.skew_id == 325
    assert pytest.approx(joint.tol) == 4.45e-4
    assert pytest.approx(joint.link_len_a) == 385.0
    assert pytest.approx(joint.link_len_b) == 365.0
    assert pytest.approx(joint.twist_angle_alpha) == 395.0
    assert pytest.approx(joint.offset_distance_s) == 315.0
    assert pytest.approx(joint.offset_distance_r) == 315.0
    assert pytest.approx(joint.offset_distance_v) == 315.0
    assert pytest.approx(joint.offset_distance_h) == 315.0
    assert pytest.approx(joint.offset_distance_u) == 315.0


def test_m418_lagmul_frechet_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Frechet Spatial Linkage Joint Free Format Test
/LAGMUL/FRECHET_SPATIAL_LINKAGE_JOINT/646
781, 782, 783, 5.65e7, 326, 4.55e-4
405.0, 375.0, 415.0, 330.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 646 in model.lagmul_frechet_spatial_linkage_joints
    joint = model.lagmul_frechet_spatial_linkage_joints[646]
    assert joint.node1 == 781
    assert joint.node2 == 782
    assert joint.node3 == 783
    assert pytest.approx(joint.stiff) == 5.65e7
    assert joint.skew_id == 326
    assert pytest.approx(joint.tol) == 4.55e-4
    assert pytest.approx(joint.link_len_a) == 405.0
    assert pytest.approx(joint.link_len_b) == 375.0
    assert pytest.approx(joint.twist_angle_alpha) == 415.0
    assert pytest.approx(joint.offset_distance_s) == 330.0


def test_m418_lagmul_frechet_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Frechet Spatial Linkage Joint Aliases Test
/FRECHET_SPATIAL_LINKAGE_JOINT/647
881, 882, 883, 1.0e6, 0, 1.0e-6
345.0, 335.0, 345.0, 295.0
/LAGMUL/FRECHET_SPATIAL_LINKAGE/648
881, 882, 883, 1.0e6, 0, 1.0e-6
345.0, 335.0, 345.0, 295.0
/FRECHET_SPATIAL_LINKAGE/649
881, 882, 883, 1.0e6, 0, 1.0e-6
345.0, 335.0, 345.0, 295.0
/FRECHET_SPATIAL_MULTI_LOOP_MECHANISM/650
881, 882, 883, 1.0e6, 0, 1.0e-6
345.0, 335.0, 345.0, 295.0
/FRECHET_SPATIAL_SYMMETRIC_MECHANISM/651
881, 882, 883, 1.0e6, 0, 1.0e-6
345.0, 335.0, 345.0, 295.0
/FRECHET_SPATIAL_6R_MECHANISM/652
881, 882, 883, 1.0e6, 0, 1.0e-6
345.0, 335.0, 345.0, 295.0
/FRECHET_SPATIAL_OVERCONSTRAINED_MECHANISM/653
881, 882, 883, 1.0e6, 0, 1.0e-6
345.0, 335.0, 345.0, 295.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(647, 654):
        assert jid in model.lagmul_frechet_spatial_linkage_joints


def test_m418_lagmul_frechet_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/FRECHET_SPATIAL_LINKAGE_JOINT/654
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m418_sensor_spring_torsional_lock_rate_fixed(tmp_path: Path):
    c1 = f"{1728:>10d}{6.58e8:>20.4f}{0.2445:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Lock Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_LOCK_RATE/1
Fixed Spring Torsional Lock Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_lock_rates
    s1 = model.sensor_spring_torsional_lock_rates[1]
    assert s1.spring_id == 1728
    assert pytest.approx(s1.jtors_lock_max) == 6.58e8
    assert pytest.approx(s1.jtors_snp_max) == 6.58e8
    assert pytest.approx(s1.jtors_crackle_max) == 6.58e8
    assert pytest.approx(s1.jtors_shot_max) == 6.58e8
    assert pytest.approx(s1.jtors_drop_max) == 6.58e8
    assert pytest.approx(s1.jtors_pop_max) == 6.58e8
    assert pytest.approx(s1.jtors_crk_max) == 6.58e8
    assert pytest.approx(s1.t_delay) == 0.2445
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_LOCK_RATE"


def test_m418_sensor_spring_torsional_lock_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Torsional Lock Rate Free Format Test
/SENSOR/SPRING_TORSIONAL_LOCK_RATE/2
Free Spring Torsional Lock Rate Sensor
1729, 7.18e8, 0.3435
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_torsional_lock_rates
    s2 = model.sensor_spring_torsional_lock_rates[2]
    assert s2.spring_id == 1729
    assert pytest.approx(s2.jtors_lock_max) == 7.18e8
    assert pytest.approx(s2.t_delay) == 0.3435


def test_m418_sensor_spring_torsional_lock_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Torsional Lock Rate Aliases Test
/SENSOR/SPRING_TORS_LOCK_RATE/591
1749, 4.095e8, 0.2965
/SENSOR/SPRING_RATE_LOCK_TORS/592
1750, 4.115e8, 0.2975
/SENSOR/TORSIONAL_LOCK_RATE_SPRING/593
1751, 4.135e8, 0.2985
/SENSOR/SPRING_LOCK_TORS/594
1752, 4.155e8, 0.2995
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(591, 595):
        assert sid in model.sensor_spring_torsional_lock_rates


def test_m418_sensor_spring_torsional_lock_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TORSIONAL_LOCK_RATE/595
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
