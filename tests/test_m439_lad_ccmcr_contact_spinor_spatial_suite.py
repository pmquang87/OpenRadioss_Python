"""Tests for Milestone M439: LadCoupledCoreMicrocrackingRate Failure Model, EngElectrothermoflexomagnetomagnonicpolaritonicResonanceEnergy, LagmulContactSpinorSpatialLinkageJoint, and SensorSpringTotalCrackleRate."""

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


def test_m439_fail_lad_coupled_core_microcracking_rate_fixed(tmp_path: Path):
    c1 = f"{990.0:>20.4f}{2970.0:>20.4f}{770.0:>20.4f}{15.70:>20.4f}{0.460:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2530:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Core Microcracking Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLED_CORE_MICROCRACKING_RATE/2530
Ladeveze Coupled Core Microcracking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2530 in model.fail_ladcoupledcoremicrocrackingrates
    fccmcr = model.fail_ladcoupledcoremicrocrackingrates[2530]
    assert pytest.approx(fccmcr.sigma_ccmcr0) == 990.0
    assert pytest.approx(fccmcr.sigma_ccmcrc) == 2970.0
    assert pytest.approx(fccmcr.gamma_ccmcr) == 770.0
    assert pytest.approx(fccmcr.p_ccmcr) == 15.70
    assert pytest.approx(fccmcr.d_ccmcr_max) == 0.460
    assert fccmcr.ifail_sh == 1
    assert fccmcr.ifail_so == 2
    assert fccmcr.fail_id == 2530
    assert pytest.approx(fccmcr.sigma_ccmd0) == 990.0
    assert pytest.approx(fccmcr.sigma_ccmdc) == 2970.0
    assert pytest.approx(fccmcr.gamma_ccmd) == 770.0
    assert pytest.approx(fccmcr.p_ccmd) == 15.70
    assert pytest.approx(fccmcr.d_ccmd_max) == 0.460
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLED_CORE_MICROCRACKING_RATE"


def test_m439_fail_lad_coupled_core_microcracking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Core Microcracking Rate Free Format Test
/FAIL/LAD_COUPLED_CORE_MICROCRACKING_RATE/2531
1000.0, 3000.0, 780.0, 15.90, 0.450
1, 1
2531
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2531 in model.fail_ladcoupledcoremicrocrackingrates
    fccmcr = model.fail_ladcoupledcoremicrocrackingrates[2531]
    assert pytest.approx(fccmcr.sigma_ccmcr0) == 1000.0
    assert pytest.approx(fccmcr.sigma_ccmcrc) == 3000.0
    assert pytest.approx(fccmcr.gamma_ccmcr) == 780.0
    assert pytest.approx(fccmcr.p_ccmcr) == 15.90
    assert pytest.approx(fccmcr.d_ccmcr_max) == 0.450
    assert fccmcr.fail_id == 2531


def test_m439_fail_lad_coupled_core_microcracking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Core Microcracking Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_CORE_MICROCRACKING_RATE/2532
980.0, 2940.0, 760.0, 15.50, 0.470
1, 1
/FAIL/LAD_COUPLE_CORE_MICROCRACKING_RATE/2533
980.0, 2940.0, 760.0, 15.50, 0.470
1, 1
/FAIL/LADEVEZE_COUPLE_CORE_MICROCRACKING_RATE/2534
980.0, 2940.0, 760.0, 15.50, 0.470
1, 1
/FAIL/LAD_COUPLED_CORE_MICROCRACK_RATE/2535
980.0, 2940.0, 760.0, 15.50, 0.470
1, 1
/FAIL/LAD_COUPLE_CORE_MICROCRACK_RATE/2536
980.0, 2940.0, 760.0, 15.50, 0.470
1, 1
/FAIL/LAD_CCMCR/2537
980.0, 2940.0, 760.0, 15.50, 0.470
1, 1
/FAIL/LAD_CCMCR_MODEL/2538
980.0, 2940.0, 760.0, 15.50, 0.470
1, 1
/FAIL/LAD_CCMCR_LAW/2539
980.0, 2940.0, 760.0, 15.50, 0.470
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_CORE_MICROCRACKING/2540
980.0, 2940.0, 760.0, 15.50, 0.470
1, 1
/FAIL/LAD_COUPLED_CORE_MICRODAMAGE_RATE/2541
980.0, 2940.0, 760.0, 15.50, 0.470
1, 1
/FAIL/LADEVEZE_COUPLED_CORE_MICRODAMAGE_RATE/2542
980.0, 2940.0, 760.0, 15.50, 0.470
1, 1
/FAIL/LAD_COUPLE_CORE_MICRODAMAGE_RATE/2543
980.0, 2940.0, 760.0, 15.50, 0.470
1, 1
/FAIL/LADEVEZE_COUPLE_CORE_MICRODAMAGE_RATE/2544
980.0, 2940.0, 760.0, 15.50, 0.470
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for mid in range(2532, 2545):
        assert mid in model.fail_ladcoupledcoremicrocrackingrates
        f = model.fail_ladcoupledcoremicrocrackingrates[mid]
        assert pytest.approx(f.sigma_ccmcr0) == 980.0
        assert pytest.approx(f.sigma_ccmcrc) == 2940.0


def test_m439_fail_lad_coupled_core_microcracking_rate_properties():
    from pyradioss.model.entities import FailLadCoupledCoreMicrocrackingRate
    f = FailLadCoupledCoreMicrocrackingRate(mat_id=1)
    f.sigma_ccmd0 = 1050.0
    assert pytest.approx(f.sigma_ccmcr0) == 1050.0
    assert pytest.approx(f.sigma_ccmd0) == 1050.0

    f.sigma_ccmdc = 3150.0
    assert pytest.approx(f.sigma_ccmcrc) == 3150.0
    assert pytest.approx(f.sigma_ccmdc) == 3150.0

    f.gamma_ccmd = 820.0
    assert pytest.approx(f.gamma_ccmcr) == 820.0
    assert pytest.approx(f.gamma_ccmd) == 820.0

    f.p_ccmd = 16.50
    assert pytest.approx(f.p_ccmcr) == 16.50
    assert pytest.approx(f.p_ccmd) == 16.50

    f.d_ccmd_max = 0.420
    assert pytest.approx(f.d_ccmcr_max) == 0.420
    assert pytest.approx(f.d_ccmd_max) == 0.420


def test_m439_eng_electrothermoflexomagnetomagnonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00125:>20.6f}{42:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICPOLARITONIC_RESONANCE_ENERGY/1
Coupled Resonance Energy Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetomagnonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetomagnonicpolaritonic_resonance_energies[1]
    assert pytest.approx(r.dt_etfmgmnp) == 0.00125
    assert pytest.approx(r.dt_etfplp) == 0.00125
    assert r.sens_id == 42


def test_m439_eng_electrothermoflexomagnetomagnonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICPOLARITONIC_RESONANCE_ENERGY/2
0.00225, 43
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetomagnonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetomagnonicpolaritonic_resonance_energies[2]
    assert pytest.approx(r.dt_etfmgmnp) == 0.00225
    assert r.sens_id == 43


def test_m439_eng_electrothermoflexomagnetomagnonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_MAGNON_POLARITON_RES_WORK/3
0.00325, 44
/ENG/EELECTROTHERMOFLEXOMAGNETOMAGNONICPOLARITONICRESONANCE/4
0.00325, 44
/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICPOLARITONIC_RESONANCE_DISSIPATION/5
0.00325, 44
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOMAGNONICPOLARITONIC_RESONANCE/6
0.00325, 44
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPOLARITONIC_RESONANCE_ENERGY/7
0.00325, 44
/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_POLARITON_RES_WORK/8
0.00325, 44
/ENG/EELECTROTHERMOFLEXOMAGNETOPLASMONICPOLARITONICRESONANCE/9
0.00325, 44
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/10
0.00325, 44
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPLASMONICPOLARITONIC_RESONANCE/11
0.00325, 44
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for rid in range(3, 12):
        assert rid in model.eng_electrothermoflexomagnetomagnonicpolaritonic_resonance_energies
        r = model.eng_electrothermoflexomagnetomagnonicpolaritonic_resonance_energies[rid]
        assert pytest.approx(r.dt_etfmgmnp) == 0.00325
        assert r.sens_id == 44


def test_m439_lagmul_contact_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{2.5e6:>20.4f}{5:>10d}{1.0e-5:>20.6e}"
    c2 = f"{12.5:>20.4f}{14.5:>20.4f}{45.0:>20.4f}{3.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Contact Spinor Spatial Linkage Joint Fixed Format Test
2022 0
/CONTACT_SPINOR_SPATIAL_LINKAGE_JOINT/1
Contact Spinor Spatial Mechanism Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_contact_spinor_spatial_linkage_joints
    j = model.lagmul_contact_spinor_spatial_linkage_joints[1]
    assert j.node1 == 101
    assert j.node2 == 102
    assert j.node3 == 103
    assert pytest.approx(j.stiff) == 2.5e6
    assert j.skew_id == 5
    assert pytest.approx(j.tol) == 1.0e-5
    assert pytest.approx(j.link_len_a) == 12.5
    assert pytest.approx(j.link_len_b) == 14.5
    assert pytest.approx(j.twist_angle_alpha) == 45.0
    assert pytest.approx(j.offset_distance_s) == 3.5
    assert pytest.approx(j.offset_distance_r) == 3.5
    assert pytest.approx(j.offset_distance_v) == 3.5
    assert pytest.approx(j.offset_distance_h) == 3.5
    assert pytest.approx(j.offset_distance_u) == 3.5
    assert pytest.approx(j.offset_distance_f) == 3.5


def test_m439_lagmul_contact_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Contact Spinor Spatial Linkage Joint Free Format Test
/LAGMUL/CONTACT_SPINOR_SPATIAL_LINKAGE_JOINT/2
201, 202, 203, 3.5e6, 6, 2.0e-5
15.5, 17.5, 60.0, 4.5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_contact_spinor_spatial_linkage_joints
    j = model.lagmul_contact_spinor_spatial_linkage_joints[2]
    assert j.node1 == 201
    assert j.node2 == 202
    assert j.node3 == 203
    assert pytest.approx(j.stiff) == 3.5e6
    assert j.skew_id == 6
    assert pytest.approx(j.tol) == 2.0e-5
    assert pytest.approx(j.link_len_a) == 15.5
    assert pytest.approx(j.link_len_b) == 17.5
    assert pytest.approx(j.twist_angle_alpha) == 60.0
    assert pytest.approx(j.offset_distance_s) == 4.5


def test_m439_lagmul_contact_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Contact Spinor Spatial Linkage Joint Aliases Test
/LAGMUL/CONTACT_SPINOR_SPATIAL_LINKAGE/3
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/CONTACT_SPINOR_SPATIAL_LINKAGE/4
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/CONTACT_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM/5
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/CONTACT_SPINOR_SPATIAL_SYMMETRIC_MECHANISM/6
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/CONTACT_SPINOR_SPATIAL_6R_MECHANISM/7
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/CONTACT_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM/8
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/CONTACT_TWISTOR_SPATIAL_LINKAGE_JOINT/9
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/CONTACT_TWISTOR_SPATIAL_LINKAGE_JOINT/10
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/CONTACT_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/11
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/CONTACT_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/12
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/13
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/14
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(3, 15):
        assert jid in model.lagmul_contact_spinor_spatial_linkage_joints
        j = model.lagmul_contact_spinor_spatial_linkage_joints[jid]
        assert j.node1 == 301
        assert pytest.approx(j.link_len_a) == 10.0


def test_m439_lagmul_contact_spinor_spatial_linkage_joint_properties():
    from pyradioss.model.entities import LagmulContactSpinorSpatialLinkageJoint
    j = LagmulContactSpinorSpatialLinkageJoint(id=1)
    j.offset_distance_r = 7.5
    assert pytest.approx(j.offset_distance_s) == 7.5
    assert pytest.approx(j.offset_distance_r) == 7.5

    j.offset_distance_v = 8.5
    assert pytest.approx(j.offset_distance_s) == 8.5
    assert pytest.approx(j.offset_distance_v) == 8.5

    j.offset_distance_h = 9.5
    assert pytest.approx(j.offset_distance_s) == 9.5
    assert pytest.approx(j.offset_distance_h) == 9.5

    j.offset_distance_u = 10.5
    assert pytest.approx(j.offset_distance_s) == 10.5
    assert pytest.approx(j.offset_distance_u) == 10.5

    j.offset_distance_f = 11.5
    assert pytest.approx(j.offset_distance_s) == 11.5
    assert pytest.approx(j.offset_distance_f) == 11.5


def test_m439_sensor_spring_total_crackle_rate_fixed(tmp_path: Path):
    c1 = f"{501:>10d}{1.75e12:>20.6e}{0.0055:>20.6f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Crackle Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_CRACKLE_RATE/1
Sensor Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_crackle_rates
    s = model.sensor_spring_total_crackle_rates[1]
    assert s.spring_id == 501
    assert pytest.approx(s.jtot_crk_max) == 1.75e12
    assert pytest.approx(s.jtot_shot_max) == 1.75e12
    assert pytest.approx(s.jtot_drop_max) == 1.75e12
    assert pytest.approx(s.jtot_lock_max) == 1.75e12
    assert pytest.approx(s.jtot_pop_max) == 1.75e12
    assert pytest.approx(s.t_delay) == 0.0055


def test_m439_sensor_spring_total_crackle_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Crackle Rate Free Format Test
/SENSOR/SPRING_TOTAL_CRACKLE_RATE/2
502, 1.85e12, 0.0065
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_crackle_rates
    s = model.sensor_spring_total_crackle_rates[2]
    assert s.spring_id == 502
    assert pytest.approx(s.jtot_crk_max) == 1.85e12
    assert pytest.approx(s.t_delay) == 0.0065


def test_m439_sensor_spring_total_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Crackle Rate Aliases Test
/SENSOR/SPRING_TOT_CRACKLE_RATE/3
503, 1.95e12, 0.0075
/SENSOR/SPRING_CRACKLE_RATE_TOT/4
503, 1.95e12, 0.0075
/SENSOR/TOTAL_CRACKLE_RATE_SPRING/5
503, 1.95e12, 0.0075
/SENSOR/SPRING_CRACKLE_TOT/6
503, 1.95e12, 0.0075
/SENSOR/SPRING_TOTAL_CRK_RATE/7
503, 1.95e12, 0.0075
/SENSOR/SPRING_TOT_CRK_RATE/8
503, 1.95e12, 0.0075
/SENSOR/SPRING_CRK_RATE_TOT/9
503, 1.95e12, 0.0075
/SENSOR/TOTAL_CRK_RATE_SPRING/10
503, 1.95e12, 0.0075
/SENSOR/SPRING_CRK_TOT/11
503, 1.95e12, 0.0075
/SENSOR/SPRING_RESULTANT_CRACKLE_RATE/12
503, 1.95e12, 0.0075
/SENSOR/SPRING_RES_CRACKLE_RATE/13
503, 1.95e12, 0.0075
/SENSOR/SPRING_CRACKLE_RATE_RES/14
503, 1.95e12, 0.0075
/SENSOR/RESULTANT_CRACKLE_RATE_SPRING/15
503, 1.95e12, 0.0075
/SENSOR/SPRING_RESULTANT_CRK_RATE/16
503, 1.95e12, 0.0075
/SENSOR/SPRING_RES_CRK_RATE/17
503, 1.95e12, 0.0075
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(3, 18):
        assert sid in model.sensor_spring_total_crackle_rates
        s = model.sensor_spring_total_crackle_rates[sid]
        assert s.spring_id == 503
        assert pytest.approx(s.jtot_crk_max) == 1.95e12


def test_m439_sensor_spring_total_crackle_rate_properties():
    from pyradioss.model.entities import SensorSpringTotalCrackleRate
    s = SensorSpringTotalCrackleRate(id=1)
    s.jtot_shot_max = 3.1e12
    assert pytest.approx(s.jtot_crk_max) == 3.1e12
    assert pytest.approx(s.jtot_shot_max) == 3.1e12

    s.jtot_drop_max = 3.2e12
    assert pytest.approx(s.jtot_crk_max) == 3.2e12
    assert pytest.approx(s.jtot_drop_max) == 3.2e12

    s.jtot_lock_max = 3.3e12
    assert pytest.approx(s.jtot_crk_max) == 3.3e12
    assert pytest.approx(s.jtot_lock_max) == 3.3e12

    s.jtot_pop_max = 3.4e12
    assert pytest.approx(s.jtot_crk_max) == 3.4e12
    assert pytest.approx(s.jtot_pop_max) == 3.4e12


def test_m439_missing_cards_and_errors(tmp_path: Path):
    deck = """# EMPTY CARDS DECK
/BEGIN
Empty Cards Test
/FAIL/LAD_COUPLED_CORE_MICROCRACKING_RATE/999
/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICPOLARITONIC_RESONANCE_ENERGY/999
/CONTACT_SPINOR_SPATIAL_LINKAGE_JOINT/999
/SENSOR/SPRING_TOTAL_CRACKLE_RATE/999
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) >= 4
