"""Tests for Milestone M451: LadCoupledDelaminationMicrodebondingRate Failure Model, EngElectrothermoflexomagnetochiralspinplasmonicpolaritonicResonanceEnergy, LagmulFoliationSpinorSpatialLinkageJoint, and SensorSpringCoupledCrackleRate."""

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


def test_m451_fail_lad_coupled_delamination_microdebonding_rate_fixed(tmp_path: Path):
    c1 = f"{1070.0:>20.4f}{3210.0:>20.4f}{870.0:>20.4f}{17.0:>20.4f}{0.480:>20.4f}"
    c2 = f"{2:>10d}{2:>10d}"
    c3 = f"{770:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Delamination Microdebonding Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLED_DELAMINATION_MICRODEBONDING_RATE/2770
Ladeveze Coupled Microdebonding Law
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2770 in model.fail_ladcoupleddelaminationmicrodebondingrates
    f = model.fail_ladcoupleddelaminationmicrodebondingrates[2770]
    assert pytest.approx(f.sigma_cdmdbr0) == 1070.0
    assert pytest.approx(f.sigma_cdmdbrc) == 3210.0
    assert pytest.approx(f.gamma_cdmdbr) == 870.0
    assert pytest.approx(f.p_cdmdbr) == 17.0
    assert pytest.approx(f.d_cdmdbr_max) == 0.480
    assert f.ifail_sh == 2
    assert f.ifail_so == 2
    assert f.fail_id == 770
    assert pytest.approx(f.sigma_cdmdb0) == 1070.0
    assert pytest.approx(f.sigma_cdmdbc) == 3210.0
    assert pytest.approx(f.gamma_cdmdb) == 870.0
    assert pytest.approx(f.p_cdmdb) == 17.0
    assert pytest.approx(f.d_cdmdb_max) == 0.480
    assert pytest.approx(f.sigma_cdm0) == 1070.0
    assert pytest.approx(f.sigma_cdmc) == 3210.0
    assert pytest.approx(f.gamma_cdm) == 870.0
    assert pytest.approx(f.p_cdm) == 17.0
    assert pytest.approx(f.d_cdm_max) == 0.480
    assert pytest.approx(f.sigma_cimdbr0) == 1070.0
    assert pytest.approx(f.sigma_cimdbrc) == 3210.0
    assert pytest.approx(f.gamma_cimdbr) == 870.0
    assert pytest.approx(f.p_cimdbr) == 17.0
    assert pytest.approx(f.d_cimdbr_max) == 0.480


def test_m451_fail_lad_coupled_delamination_microdebonding_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Delamination Microdebonding Rate Free Format Test
/FAIL/LAD_COUPLED_DELAMINATION_MICRODEBONDING_RATE/2771
1170.0, 3510.0, 970.0, 19.0, 0.580
1, 1
870
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2771 in model.fail_ladcoupleddelaminationmicrodebondingrates
    f = model.fail_ladcoupleddelaminationmicrodebondingrates[2771]
    assert pytest.approx(f.sigma_cdmdbr0) == 1170.0
    assert pytest.approx(f.sigma_cdmdbrc) == 3510.0
    assert pytest.approx(f.gamma_cdmdbr) == 970.0
    assert pytest.approx(f.p_cdmdbr) == 19.0
    assert pytest.approx(f.d_cdmdbr_max) == 0.580
    assert f.ifail_sh == 1
    assert f.ifail_so == 1
    assert f.fail_id == 870


def test_m451_fail_lad_coupled_delamination_microdebonding_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Delamination Microdebonding Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_DELAMINATION_MICRODEBONDING_RATE/2772
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LAD_COUPLE_DELAMINATION_MICRODEBONDING_RATE/2773
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LADEVEZE_COUPLE_DELAMINATION_MICRODEBONDING_RATE/2774
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LAD_COUPLED_DELAMINATION_MICRODEBOND_RATE/2775
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LADEVEZE_COUPLED_DELAMINATION_MICRODEBOND_RATE/2776
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LAD_COUPLE_DELAMINATION_MICRODEBOND_RATE/2777
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LADEVEZE_COUPLE_DELAMINATION_MICRODEBOND_RATE/2778
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LAD_COUPLED_DELAM_MICRODEBONDING_RATE/2779
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LADEVEZE_COUPLED_DELAM_MICRODEBONDING_RATE/2780
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LAD_COUPLE_DELAM_MICRODEBONDING_RATE/2781
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LADEVEZE_COUPLE_DELAM_MICRODEBONDING_RATE/2782
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LAD_COUPLED_DELAM_MICRODEBOND_RATE/2783
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LADEVEZE_COUPLED_DELAM_MICRODEBOND_RATE/2784
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LAD_COUPLE_DELAM_MICRODEBOND_RATE/2785
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LADEVEZE_COUPLE_DELAM_MICRODEBOND_RATE/2786
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LAD_CDDMDBR/2787
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LAD_CDDMDBR_MODEL/2788
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LAD_CDDMDBR_LAW/2789
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_DELAMINATION_MICRODEBONDING/2790
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LAD_COUPLED_INTERLAMINAR_MICRODEBONDING_RATE/2791
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_MICRODEBONDING_RATE/2792
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LAD_COUPLE_INTERLAMINAR_MICRODEBONDING_RATE/2793
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LADEVEZE_COUPLE_INTERLAMINAR_MICRODEBONDING_RATE/2794
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LAD_COUPLED_INTERLAMINAR_MICRODEBOND_RATE/2795
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_MICRODEBOND_RATE/2796
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LAD_COUPLE_INTERLAMINAR_MICRODEBOND_RATE/2797
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/FAIL/LADEVEZE_COUPLE_INTERLAMINAR_MICRODEBOND_RATE/2798
1070.0, 3210.0, 870.0, 17.0, 0.480
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for mid in range(2772, 2799):
        assert mid in model.fail_ladcoupleddelaminationmicrodebondingrates
        f = model.fail_ladcoupleddelaminationmicrodebondingrates[mid]
        assert pytest.approx(f.sigma_cdmdbr0) == 1070.0
        assert pytest.approx(f.sigma_cdmdbrc) == 3210.0
        assert pytest.approx(f.gamma_cdmdbr) == 870.0
        assert pytest.approx(f.p_cdmdbr) == 17.0
        assert pytest.approx(f.d_cdmdbr_max) == 0.480


def test_m451_eng_electrothermoflexomagnetochiralspinplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00475:>20.6e}{85:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetochiralspinplasmonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSPINPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Engine Electrothermoflexomagnetochiralspinplasmonicpolaritonic Resonance Energy Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralspinplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetochiralspinplasmonicpolaritonic_resonance_energies[1]
    assert pytest.approx(r.dt_etfcspinplp) == 0.00475
    assert r.sens_id == 85


def test_m451_eng_electrothermoflexomagnetochiralspinplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetochiralspinplasmonicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSPINPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
0.00575, 95
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralspinplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetochiralspinplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(r.dt_etfcspinplp) == 0.00575
    assert r.sens_id == 95


def test_m451_eng_electrothermoflexomagnetochiralspinplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetochiralspinplasmonicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_SPIN_PLASMON_POLARITON_RES_WORK/3
0.00475, 85
/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALSPINPLASMONICPOLARITONICRESONANCE/4
0.00475, 85
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSPINPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/5
0.00475, 85
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALSPINPLASMONICPOLARITONIC_RESONANCE/6
0.00475, 85
/ENG/ELECTROTHERMOFLEXOMAGNETOSPINONCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY/7
0.00475, 85
/ENG/ELECTRO_THERM_FLEXO_MAG_SPINON_CHIRAL_PLASMON_POLARITON_RES_WORK/8
0.00475, 85
/ENG/EELECTROTHERMOFLEXOMAGNETOSPINONCHIRALPLASMONICPOLARITONICRESONANCE/9
0.00475, 85
/ENG/ELECTROTHERMOFLEXOMAGNETOSPINONCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/10
0.00475, 85
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOSPINONCHIRALPLASMONICPOLARITONIC_RESONANCE/11
0.00475, 85
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for rid in range(3, 12):
        assert rid in model.eng_electrothermoflexomagnetochiralspinplasmonicpolaritonic_resonance_energies
        r = model.eng_electrothermoflexomagnetochiralspinplasmonicpolaritonic_resonance_energies[rid]
        assert pytest.approx(r.dt_etfcspinplp) == 0.00475
        assert r.sens_id == 85


def test_m451_lagmul_foliation_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{3.9e6:>20.4f}{8:>10d}{2.6e-5:>20.6e}"
    c2 = f"{24.0:>20.4f}{26.0:>20.4f}{78.0:>20.4f}{10.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Foliation Spinor Spatial Linkage Joint Fixed Format Test
2022 0
/FOLIATION_SPINOR_SPATIAL_LINKAGE_JOINT/1
Foliation Spinor Spatial Mechanism Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_foliation_spinor_spatial_linkage_joints
    j = model.lagmul_foliation_spinor_spatial_linkage_joints[1]
    assert j.node1 == 101
    assert j.node2 == 102
    assert j.node3 == 103
    assert pytest.approx(j.stiff) == 3.9e6
    assert j.skew_id == 8
    assert pytest.approx(j.tol) == 2.6e-5
    assert pytest.approx(j.link_len_a) == 24.0
    assert pytest.approx(j.link_len_b) == 26.0
    assert pytest.approx(j.twist_angle_alpha) == 78.0
    assert pytest.approx(j.offset_distance_s) == 10.0
    assert pytest.approx(j.offset_distance_r) == 10.0
    assert pytest.approx(j.offset_distance_v) == 10.0
    assert pytest.approx(j.offset_distance_h) == 10.0
    assert pytest.approx(j.offset_distance_u) == 10.0
    assert pytest.approx(j.offset_distance_f) == 10.0


def test_m451_lagmul_foliation_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Foliation Spinor Spatial Linkage Joint Free Format Test
/LAGMUL/FOLIATION_SPINOR_SPATIAL_LINKAGE_JOINT/2
Foliation Spinor Spatial Mechanism Joint Free
201, 202, 203, 4.9e6, 9, 3.6e-5
34.0, 36.0, 88.0, 15.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_foliation_spinor_spatial_linkage_joints
    j = model.lagmul_foliation_spinor_spatial_linkage_joints[2]
    assert j.node1 == 201
    assert j.node2 == 202
    assert j.node3 == 203
    assert pytest.approx(j.stiff) == 4.9e6
    assert j.skew_id == 9
    assert pytest.approx(j.tol) == 3.6e-5
    assert pytest.approx(j.link_len_a) == 34.0
    assert pytest.approx(j.link_len_b) == 36.0
    assert pytest.approx(j.twist_angle_alpha) == 88.0
    assert pytest.approx(j.offset_distance_s) == 15.0


def test_m451_lagmul_foliation_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Foliation Spinor Spatial Linkage Joint Aliases Test
/FOLIATION_SPINOR_SPATIAL_LINKAGE/3
101, 102, 103, 3.9e6, 8, 2.6e-5
24.0, 26.0, 78.0, 10.0
/LAGMUL/FOLIATION_SPINOR_SPATIAL_LINKAGE/4
101, 102, 103, 3.9e6, 8, 2.6e-5
24.0, 26.0, 78.0, 10.0
/FOLIATION_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM/5
101, 102, 103, 3.9e6, 8, 2.6e-5
24.0, 26.0, 78.0, 10.0
/FOLIATION_SPINOR_SPATIAL_SYMMETRIC_MECHANISM/6
101, 102, 103, 3.9e6, 8, 2.6e-5
24.0, 26.0, 78.0, 10.0
/FOLIATION_SPINOR_SPATIAL_6R_MECHANISM/7
101, 102, 103, 3.9e6, 8, 2.6e-5
24.0, 26.0, 78.0, 10.0
/FOLIATION_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM/8
101, 102, 103, 3.9e6, 8, 2.6e-5
24.0, 26.0, 78.0, 10.0
/LAGMUL/FOLIATION_TWISTOR_SPATIAL_LINKAGE_JOINT/9
101, 102, 103, 3.9e6, 8, 2.6e-5
24.0, 26.0, 78.0, 10.0
/FOLIATION_TWISTOR_SPATIAL_LINKAGE_JOINT/10
101, 102, 103, 3.9e6, 8, 2.6e-5
24.0, 26.0, 78.0, 10.0
/LAGMUL/FOLIATION_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/11
101, 102, 103, 3.9e6, 8, 2.6e-5
24.0, 26.0, 78.0, 10.0
/FOLIATION_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/12
101, 102, 103, 3.9e6, 8, 2.6e-5
24.0, 26.0, 78.0, 10.0
/LAGMUL/FOLIATION_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/13
101, 102, 103, 3.9e6, 8, 2.6e-5
24.0, 26.0, 78.0, 10.0
/FOLIATION_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/14
101, 102, 103, 3.9e6, 8, 2.6e-5
24.0, 26.0, 78.0, 10.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(3, 15):
        assert jid in model.lagmul_foliation_spinor_spatial_linkage_joints
        j = model.lagmul_foliation_spinor_spatial_linkage_joints[jid]
        assert j.node1 == 101
        assert j.node2 == 102
        assert j.node3 == 103
        assert pytest.approx(j.stiff) == 3.9e6


def test_m451_sensor_spring_coupled_crackle_rate_fixed(tmp_path: Path):
    c1 = f"{1205:>10d}{9.55e6:>20.4f}{0.0085:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Coupled Crackle Rate Fixed Format Test
2022 0
/SENSOR/SPRING_COUPLED_CRACKLE_RATE/1
Spring Coupled Crackle Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_coupled_crackle_rates
    sensor = model.sensor_spring_coupled_crackle_rates[1]
    assert sensor.spring_id == 1205
    assert pytest.approx(sensor.jcoup_crk_max) == 9.55e6
    assert pytest.approx(sensor.j_crk_coup_max) == 9.55e6
    assert pytest.approx(sensor.j_crackle_coup_max) == 9.55e6
    assert pytest.approx(sensor.j_coup_crk_max) == 9.55e6
    assert pytest.approx(sensor.j_crk_max) == 9.55e6
    assert pytest.approx(sensor.jcoup_crackle_max) == 9.55e6
    assert pytest.approx(sensor.j_crk_coupled_max) == 9.55e6
    assert pytest.approx(sensor.j_crackle_coupled_max) == 9.55e6
    assert pytest.approx(sensor.jcoup_shot_max) == 9.55e6
    assert pytest.approx(sensor.jcoup_drop_max) == 9.55e6
    assert pytest.approx(sensor.jcoup_lock_max) == 9.55e6
    assert pytest.approx(sensor.jcoup_pop_max) == 9.55e6
    assert pytest.approx(sensor.jcoup_snp_max) == 9.55e6
    assert pytest.approx(sensor.jcoup_snap_max) == 9.55e6
    assert pytest.approx(sensor.jcoup_rate_max) == 9.55e6
    assert pytest.approx(sensor.jcoup_roc_rate_max) == 9.55e6
    assert pytest.approx(sensor.jcoup_drop_rate_max) == 9.55e6
    assert pytest.approx(sensor.jcoup_crk_rate_max) == 9.55e6
    assert pytest.approx(sensor.t_delay) == 0.0085
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_COUPLED_CRACKLE_RATE"


def test_m451_sensor_spring_coupled_crackle_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Coupled Crackle Rate Free Format Test
/SENSOR/SPRING_COUPLED_CRACKLE_RATE/2
45, 5.8e5, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_coupled_crackle_rates
    sensor = model.sensor_spring_coupled_crackle_rates[2]
    assert sensor.spring_id == 45
    assert pytest.approx(sensor.jcoup_crk_max) == 5.8e5
    assert pytest.approx(sensor.t_delay) == 0.003
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_COUPLED_CRACKLE_RATE"


def test_m451_sensor_spring_coupled_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Coupled Crackle Rate Aliases Test
/SENSOR/SPRING_COUPLE_CRACKLE_RATE/3
45, 5.8e5, 0.003
/SENSOR/SPRING_COUPLED_CRK_RATE/4
45, 5.8e5, 0.003
/SENSOR/SPRING_COUPLE_CRK_RATE/5
45, 5.8e5, 0.003
/SENSOR/SPRING_COUPLED_CRACKLE/6
45, 5.8e5, 0.003
/SENSOR/SPRING_COUPLE_CRACKLE/7
45, 5.8e5, 0.003
/SENSOR/SPRING_COUPLED_CRK/8
45, 5.8e5, 0.003
/SENSOR/SPRING_COUPLE_CRK/9
45, 5.8e5, 0.003
/SENSOR/SPRING_POP_RATE_COUPLED_CRACKLE/10
45, 5.8e5, 0.003
/SENSOR/SPRING_POP_RATE_COUPLE_CRACKLE/11
45, 5.8e5, 0.003
/SENSOR/COUPLED_CRACKLE_RATE_SPRING/12
45, 5.8e5, 0.003
/SENSOR/COUPLE_CRACKLE_RATE_SPRING/13
45, 5.8e5, 0.003
/SENSOR/SPRING_CRACKLE_RATE_COUPLED/14
45, 5.8e5, 0.003
/SENSOR/SPRING_CRACKLE_RATE_COUPLE/15
45, 5.8e5, 0.003
/SENSOR/SPRING_CRK_RATE_COUPLED/16
45, 5.8e5, 0.003
/SENSOR/SPRING_CRK_RATE_COUPLE/17
45, 5.8e5, 0.003
/SENSOR/SPRING_RATE_CRACKLE_COUPLED/18
45, 5.8e5, 0.003
/SENSOR/SPRING_RATE_CRACKLE_COUPLE/19
45, 5.8e5, 0.003
/SENSOR/SPRING_CRACKLE_COUPLED/20
45, 5.8e5, 0.003
/SENSOR/SPRING_CRACKLE_COUPLE/21
45, 5.8e5, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(3, 22):
        assert sid in model.sensor_spring_coupled_crackle_rates
        s = model.sensor_spring_coupled_crackle_rates[sid]
        assert s.spring_id == 45
        assert pytest.approx(s.jcoup_crk_max) == 5.8e5


def test_m451_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_COUPLED_DELAMINATION_MICRODEBONDING_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSPINPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/FOLIATION_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_COUPLED_CRACKLE_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4
