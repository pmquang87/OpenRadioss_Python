"""Tests for Milestone M448: LadCoupledDelaminationMicrocrackingRate Failure Model, EngElectrothermoflexomagnetochiralphononicplasmonicpolaritonicResonanceEnergy, LagmulToposSpinorSpatialLinkageJoint, and SensorSpringTotalAngularPopRate."""

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


def test_m448_fail_lad_coupled_delamination_microcracking_rate_fixed(tmp_path: Path):
    c1 = f"{1070.0:>20.4f}{3210.0:>20.4f}{870.0:>20.4f}{17.20:>20.4f}{0.395:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2700:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Delamination Microcracking Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLED_DELAMINATION_MICROCRACKING_RATE/2700
Ladeveze Coupled Delamination Microcracking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2700 in model.fail_ladcoupleddelaminationmicrocrackingrates
    fcdmcr = model.fail_ladcoupleddelaminationmicrocrackingrates[2700]
    assert pytest.approx(fcdmcr.sigma_cdmcr0) == 1070.0
    assert pytest.approx(fcdmcr.sigma_cdmcrc) == 3210.0
    assert pytest.approx(fcdmcr.gamma_cdmcr) == 870.0
    assert pytest.approx(fcdmcr.p_cdmcr) == 17.20
    assert pytest.approx(fcdmcr.d_cdmcr_max) == 0.395
    assert fcdmcr.ifail_sh == 1
    assert fcdmcr.ifail_so == 2
    assert fcdmcr.fail_id == 2700
    assert pytest.approx(fcdmcr.sigma_cdmc0) == 1070.0
    assert pytest.approx(fcdmcr.sigma_cdmcc) == 3210.0
    assert pytest.approx(fcdmcr.gamma_cdmc) == 870.0
    assert pytest.approx(fcdmcr.p_cdmc) == 17.20
    assert pytest.approx(fcdmcr.d_cdmc_max) == 0.395
    assert pytest.approx(fcdmcr.sigma_cdm0) == 1070.0
    assert pytest.approx(fcdmcr.sigma_cdmc) == 3210.0
    assert pytest.approx(fcdmcr.gamma_cdm) == 870.0
    assert pytest.approx(fcdmcr.p_cdm) == 17.20
    assert pytest.approx(fcdmcr.d_cdm_max) == 0.395
    assert pytest.approx(fcdmcr.sigma_cimcr0) == 1070.0
    assert pytest.approx(fcdmcr.sigma_cimcrc) == 3210.0
    assert pytest.approx(fcdmcr.gamma_cimcr) == 870.0
    assert pytest.approx(fcdmcr.p_cimcr) == 17.20
    assert pytest.approx(fcdmcr.d_cimcr_max) == 0.395
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLED_DELAMINATION_MICROCRACKING_RATE"


def test_m448_fail_lad_coupled_delamination_microcracking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Delamination Microcracking Rate Free Format Test
/FAIL/LAD_COUPLED_DELAMINATION_MICROCRACKING_RATE/2701
1090.0, 3270.0, 890.0, 17.50, 0.385
1, 1
2701
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2701 in model.fail_ladcoupleddelaminationmicrocrackingrates
    fcdmcr = model.fail_ladcoupleddelaminationmicrocrackingrates[2701]
    assert pytest.approx(fcdmcr.sigma_cdmcr0) == 1090.0
    assert pytest.approx(fcdmcr.sigma_cdmcrc) == 3270.0
    assert pytest.approx(fcdmcr.gamma_cdmcr) == 890.0
    assert pytest.approx(fcdmcr.p_cdmcr) == 17.50
    assert pytest.approx(fcdmcr.d_cdmcr_max) == 0.385
    assert fcdmcr.fail_id == 2701


def test_m448_fail_lad_coupled_delamination_microcracking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Delamination Microcracking Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_DELAMINATION_MICROCRACKING_RATE/2702
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LAD_COUPLE_DELAMINATION_MICROCRACKING_RATE/2703
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LADEVEZE_COUPLE_DELAMINATION_MICROCRACKING_RATE/2704
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LAD_COUPLED_DELAMINATION_MICROCRACK_RATE/2705
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LADEVEZE_COUPLED_DELAMINATION_MICROCRACK_RATE/2706
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LAD_COUPLE_DELAMINATION_MICROCRACK_RATE/2707
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LADEVEZE_COUPLE_DELAMINATION_MICROCRACK_RATE/2708
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LAD_COUPLED_DELAM_MICROCRACKING_RATE/2709
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LADEVEZE_COUPLED_DELAM_MICROCRACKING_RATE/2710
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LAD_COUPLE_DELAM_MICROCRACKING_RATE/2711
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LADEVEZE_COUPLE_DELAM_MICROCRACKING_RATE/2712
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LAD_COUPLED_DELAM_MICROCRACK_RATE/2713
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LADEVEZE_COUPLED_DELAM_MICROCRACK_RATE/2714
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LAD_COUPLE_DELAM_MICROCRACK_RATE/2715
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LADEVEZE_COUPLE_DELAM_MICROCRACK_RATE/2716
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LAD_CDDMCR/2717
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LAD_CDDMCR_MODEL/2718
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LAD_CDDMCR_LAW/2719
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_DELAMINATION_MICROCRACKING/2720
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LAD_COUPLED_INTERLAMINAR_MICROCRACKING_RATE/2721
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_MICROCRACKING_RATE/2722
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LAD_COUPLE_INTERLAMINAR_MICROCRACKING_RATE/2723
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LADEVEZE_COUPLE_INTERLAMINAR_MICROCRACKING_RATE/2724
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LAD_COUPLED_INTERLAMINAR_MICROCRACK_RATE/2725
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_MICROCRACK_RATE/2726
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LAD_COUPLE_INTERLAMINAR_MICROCRACK_RATE/2727
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/FAIL/LADEVEZE_COUPLE_INTERLAMINAR_MICROCRACK_RATE/2728
1020.0, 3060.0, 820.0, 16.00, 0.435
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for mid in range(2702, 2729):
        assert mid in model.fail_ladcoupleddelaminationmicrocrackingrates
        f = model.fail_ladcoupleddelaminationmicrocrackingrates[mid]
        assert pytest.approx(f.sigma_cdmcr0) == 1020.0
        assert pytest.approx(f.sigma_cdmcrc) == 3060.0


def test_m448_fail_lad_coupled_delamination_microcracking_rate_properties():
    from pyradioss.model.entities import FailLadCoupledDelaminationMicrocrackingRate
    f = FailLadCoupledDelaminationMicrocrackingRate(mat_id=1)
    f.sigma_cdmc0 = 1130.0
    assert pytest.approx(f.sigma_cdmcr0) == 1130.0
    assert pytest.approx(f.sigma_cdmc0) == 1130.0

    f.sigma_cdmcc = 3390.0
    assert pytest.approx(f.sigma_cdmcrc) == 3390.0
    assert pytest.approx(f.sigma_cdmcc) == 3390.0

    f.gamma_cdmc = 910.0
    assert pytest.approx(f.gamma_cdmcr) == 910.0
    assert pytest.approx(f.gamma_cdmc) == 910.0

    f.p_cdmc = 17.30
    assert pytest.approx(f.p_cdmcr) == 17.30
    assert pytest.approx(f.p_cdmc) == 17.30

    f.d_cdmc_max = 0.375
    assert pytest.approx(f.d_cdmcr_max) == 0.375
    assert pytest.approx(f.d_cdmc_max) == 0.375

    f.sigma_cimcr0 = 1190.0
    assert pytest.approx(f.sigma_cdmcr0) == 1190.0
    assert pytest.approx(f.sigma_cimcr0) == 1190.0

    f.sigma_cimcrc = 3570.0
    assert pytest.approx(f.sigma_cdmcrc) == 3570.0
    assert pytest.approx(f.sigma_cimcrc) == 3570.0

    f.gamma_cimcr = 950.0
    assert pytest.approx(f.gamma_cdmcr) == 950.0
    assert pytest.approx(f.gamma_cimcr) == 950.0

    f.p_cimcr = 18.30
    assert pytest.approx(f.p_cdmcr) == 18.30
    assert pytest.approx(f.p_cimcr) == 18.30

    f.d_cimcr_max = 0.345
    assert pytest.approx(f.d_cdmcr_max) == 0.345
    assert pytest.approx(f.d_cimcr_max) == 0.345


def test_m448_eng_electrothermoflexomagnetochiralphononicplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00215:>20.6f}{71:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Chiral Phonon Plasmon Polaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Chiral Phonon Plasmon Polaritonic Resonance Energy Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralphononicplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetochiralphononicplasmonicpolaritonic_resonance_energies[1]
    assert pytest.approx(r.dt_etfcphplp) == 0.00215
    assert pytest.approx(r.dt_etfplp) == 0.00215
    assert r.sens_id == 71


def test_m448_eng_electrothermoflexomagnetochiralphononicplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Chiral Phonon Plasmon Polaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
0.00315, 72
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralphononicplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetochiralphononicplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(r.dt_etfcphplp) == 0.00315
    assert r.sens_id == 72


def test_m448_eng_electrothermoflexomagnetochiralphononicplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Chiral Phonon Plasmon Polaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_PHONON_PLASMON_POLARITON_RES_WORK/3
0.00415, 73
/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALPHONONICPLASMONICPOLARITONICRESONANCE/4
0.00415, 73
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPHONONICPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/5
0.00415, 73
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALPHONONICPLASMONICPOLARITONIC_RESONANCE/6
0.00415, 73
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY/7
0.00415, 73
/ENG/ELECTRO_THERM_FLEXO_MAG_PHONON_CHIRAL_PLASMON_POLARITON_RES_WORK/8
0.00415, 73
/ENG/EELECTROTHERMOFLEXOMAGNETOPHONONICCHIRALPLASMONICPOLARITONICRESONANCE/9
0.00415, 73
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/10
0.00415, 73
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPHONONICCHIRALPLASMONICPOLARITONIC_RESONANCE/11
0.00415, 73
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for rid in range(3, 12):
        assert rid in model.eng_electrothermoflexomagnetochiralphononicplasmonicpolaritonic_resonance_energies
        r = model.eng_electrothermoflexomagnetochiralphononicplasmonicpolaritonic_resonance_energies[rid]
        assert pytest.approx(r.dt_etfcphplp) == 0.00415
        assert r.sens_id == 73


def test_m448_lagmul_topos_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{3.6e6:>20.4f}{6:>10d}{2.1e-5:>20.6e}"
    c2 = f"{21.0:>20.4f}{23.0:>20.4f}{70.0:>20.4f}{7.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Topos Spinor Spatial Linkage Joint Fixed Format Test
2022 0
/TOPOS_SPINOR_SPATIAL_LINKAGE_JOINT/1
Topos Spinor Spatial Mechanism Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_topos_spinor_spatial_linkage_joints
    j = model.lagmul_topos_spinor_spatial_linkage_joints[1]
    assert j.node1 == 101
    assert j.node2 == 102
    assert j.node3 == 103
    assert pytest.approx(j.stiff) == 3.6e6
    assert j.skew_id == 6
    assert pytest.approx(j.tol) == 2.1e-5
    assert pytest.approx(j.link_len_a) == 21.0
    assert pytest.approx(j.link_len_b) == 23.0
    assert pytest.approx(j.twist_angle_alpha) == 70.0
    assert pytest.approx(j.offset_distance_s) == 7.5
    assert pytest.approx(j.offset_distance_r) == 7.5
    assert pytest.approx(j.offset_distance_v) == 7.5
    assert pytest.approx(j.offset_distance_h) == 7.5
    assert pytest.approx(j.offset_distance_u) == 7.5
    assert pytest.approx(j.offset_distance_f) == 7.5


def test_m448_lagmul_topos_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Topos Spinor Spatial Linkage Joint Free Format Test
/LAGMUL/TOPOS_SPINOR_SPATIAL_LINKAGE_JOINT/2
201, 202, 203, 4.6e6, 7, 3.1e-5
24.0, 26.0, 82.0, 8.5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_topos_spinor_spatial_linkage_joints
    j = model.lagmul_topos_spinor_spatial_linkage_joints[2]
    assert j.node1 == 201
    assert j.node2 == 202
    assert j.node3 == 203
    assert pytest.approx(j.stiff) == 4.6e6
    assert j.skew_id == 7
    assert pytest.approx(j.tol) == 3.1e-5
    assert pytest.approx(j.link_len_a) == 24.0
    assert pytest.approx(j.link_len_b) == 26.0
    assert pytest.approx(j.twist_angle_alpha) == 82.0
    assert pytest.approx(j.offset_distance_s) == 8.5


def test_m448_lagmul_topos_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Topos Spinor Spatial Linkage Joint Aliases Test
/LAGMUL/TOPOS_SPINOR_SPATIAL_LINKAGE/3
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/TOPOS_SPINOR_SPATIAL_LINKAGE/4
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/TOPOS_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM/5
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/TOPOS_SPINOR_SPATIAL_SYMMETRIC_MECHANISM/6
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/TOPOS_SPINOR_SPATIAL_6R_MECHANISM/7
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/TOPOS_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM/8
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/TOPOS_TWISTOR_SPATIAL_LINKAGE_JOINT/9
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/TOPOS_TWISTOR_SPATIAL_LINKAGE_JOINT/10
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/TOPOS_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/11
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/TOPOS_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/12
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/TOPOS_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/13
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/TOPOS_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/14
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(3, 15):
        assert jid in model.lagmul_topos_spinor_spatial_linkage_joints
        j = model.lagmul_topos_spinor_spatial_linkage_joints[jid]
        assert j.node1 == 301
        assert j.node2 == 302
        assert j.node3 == 303


def test_m448_sensor_spring_total_angular_pop_rate_fixed(tmp_path: Path):
    c1 = f"{10:>10d}{1.96e5:>20.4f}{0.0026:>20.6f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Angular Pop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE/1
Total Angular Pop Rate Sensor Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_angular_pop_rates
    s = model.sensor_spring_total_angular_pop_rates[1]
    assert s.spring_id == 10
    assert pytest.approx(s.jtang_pop_max) == 1.96e5
    assert pytest.approx(s.t_delay) == 0.0026
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_POP_RATE"


def test_m448_sensor_spring_total_angular_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Angular Pop Rate Free Format Test
/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE/2
20, 2.96e5, 0.0036
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_angular_pop_rates
    s = model.sensor_spring_total_angular_pop_rates[2]
    assert s.spring_id == 20
    assert pytest.approx(s.jtang_pop_max) == 2.96e5
    assert pytest.approx(s.t_delay) == 0.0036


def test_m448_sensor_spring_total_angular_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Angular Pop Rate Aliases Test
/SENSOR/SPRING_TOT_ANG_POP_RATE/3
30, 3.6e5, 0.001
/SENSOR/SPRING_TOTAL_ANGULAR_POP/4
30, 3.6e5, 0.001
/SENSOR/SPRING_TOT_ANG_POP/5
30, 3.6e5, 0.001
/SENSOR/SPRING_POP_RATE_TOTAL_ANGULAR/6
30, 3.6e5, 0.001
/SENSOR/SPRING_POP_RATE_TOT_ANG/7
30, 3.6e5, 0.001
/SENSOR/TOTAL_ANGULAR_POP_RATE_SPRING/8
30, 3.6e5, 0.001
/SENSOR/SPRING_POP_TOTAL_ANGULAR/9
30, 3.6e5, 0.001
/SENSOR/SPRING_POP_TOT_ANG/10
30, 3.6e5, 0.001
/SENSOR/SPRING_RESULTANT_ANGULAR_POP_RATE/11
30, 3.6e5, 0.001
/SENSOR/SPRING_RES_ANG_POP_RATE/12
30, 3.6e5, 0.001
/SENSOR/SPRING_RESULTANT_ANGULAR_POP/13
30, 3.6e5, 0.001
/SENSOR/SPRING_RES_ANG_POP/14
30, 3.6e5, 0.001
/SENSOR/SPRING_POP_RATE_RESULTANT_ANGULAR/15
30, 3.6e5, 0.001
/SENSOR/SPRING_POP_RATE_RES_ANG/16
30, 3.6e5, 0.001
/SENSOR/RESULTANT_ANGULAR_POP_RATE_SPRING/17
30, 3.6e5, 0.001
/SENSOR/SPRING_POP_RESULTANT_ANGULAR/18
30, 3.6e5, 0.001
/SENSOR/SPRING_POP_RES_ANG/19
30, 3.6e5, 0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(3, 20):
        assert sid in model.sensor_spring_total_angular_pop_rates
        s = model.sensor_spring_total_angular_pop_rates[sid]
        assert s.spring_id == 30
        assert pytest.approx(s.jtang_pop_max) == 3.6e5


def test_m448_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_COUPLED_DELAMINATION_MICROCRACKING_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/TOPOS_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4
