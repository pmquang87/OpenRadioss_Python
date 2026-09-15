"""Tests for Milestone M454: LadCoupledMatrixShearDegradationRate Failure Model, EngElectrothermoflexomagnetochiralorbitonplasmonicpolaritonicResonanceEnergy, LagmulBundleSpinorSpatialLinkageJoint, and SensorSpringNormalPopRate."""

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


def test_m454_fail_lad_coupled_matrix_shear_degradation_rate_fixed(tmp_path: Path):
    c1 = f"{1100.0:>20.4f}{3300.0:>20.4f}{900.0:>20.4f}{20.0:>20.4f}{0.510:>20.4f}"
    c2 = f"{2:>10d}{2:>10d}"
    c3 = f"{800:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Matrix Shear Degradation Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLED_MATRIX_SHEAR_DEGRADATION_RATE/2800
Ladeveze Coupled Matrix Shear Degradation Law
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2800 in model.fail_ladcoupledmatrixsheardegradationrates
    f = model.fail_ladcoupledmatrixsheardegradationrates[2800]
    assert pytest.approx(f.sigma_cmsdr0) == 1100.0
    assert pytest.approx(f.sigma_cmsdrc) == 3300.0
    assert pytest.approx(f.gamma_cmsdr) == 900.0
    assert pytest.approx(f.p_cmsdr) == 20.0
    assert pytest.approx(f.d_cmsdr_max) == 0.510
    assert f.ifail_sh == 2
    assert f.ifail_so == 2
    assert f.fail_id == 800
    assert pytest.approx(f.sigma_cmsd0) == 1100.0
    assert pytest.approx(f.sigma_cmsdc) == 3300.0
    assert pytest.approx(f.gamma_cmsd) == 900.0
    assert pytest.approx(f.p_cmsd) == 20.0
    assert pytest.approx(f.d_cmsd_max) == 0.510
    assert pytest.approx(f.sigma_cms0) == 1100.0
    assert pytest.approx(f.sigma_cmsc) == 3300.0
    assert pytest.approx(f.gamma_cms) == 900.0
    assert pytest.approx(f.p_cms) == 20.0
    assert pytest.approx(f.d_cms_max) == 0.510
    assert pytest.approx(f.sigma_csdr0) == 1100.0
    assert pytest.approx(f.sigma_csdrc) == 3300.0
    assert pytest.approx(f.gamma_csdr) == 900.0
    assert pytest.approx(f.p_csdr) == 20.0
    assert pytest.approx(f.d_csdr_max) == 0.510


def test_m454_fail_lad_coupled_matrix_shear_degradation_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Matrix Shear Degradation Rate Free Format Test
/FAIL/LAD_COUPLED_MATRIX_SHEAR_DEGRADATION_RATE/2801
1200.0, 3600.0, 1000.0, 22.0, 0.610
1, 1
900
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2801 in model.fail_ladcoupledmatrixsheardegradationrates
    f = model.fail_ladcoupledmatrixsheardegradationrates[2801]
    assert pytest.approx(f.sigma_cmsdr0) == 1200.0
    assert pytest.approx(f.sigma_cmsdrc) == 3600.0
    assert pytest.approx(f.gamma_cmsdr) == 1000.0
    assert pytest.approx(f.p_cmsdr) == 22.0
    assert pytest.approx(f.d_cmsdr_max) == 0.610
    assert f.ifail_sh == 1
    assert f.ifail_so == 1
    assert f.fail_id == 900


def test_m454_fail_lad_coupled_matrix_shear_degradation_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Matrix Shear Degradation Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_MATRIX_SHEAR_DEGRADATION_RATE/2802
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LAD_COUPLE_MATRIX_SHEAR_DEGRADATION_RATE/2803
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LADEVEZE_COUPLE_MATRIX_SHEAR_DEGRADATION_RATE/2804
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LAD_COUPLED_MATRIX_SHEAR_DEGRADE_RATE/2805
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LADEVEZE_COUPLED_MATRIX_SHEAR_DEGRADE_RATE/2806
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LAD_COUPLE_MATRIX_SHEAR_DEGRADE_RATE/2807
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LADEVEZE_COUPLE_MATRIX_SHEAR_DEGRADE_RATE/2808
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LAD_COUPLED_MATRIX_SHEAR_RATE/2809
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LADEVEZE_COUPLED_MATRIX_SHEAR_RATE/2810
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LAD_COUPLE_MATRIX_SHEAR_RATE/2811
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LADEVEZE_COUPLE_MATRIX_SHEAR_RATE/2812
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LAD_COUPLED_SHEAR_DEGRADATION_RATE/2813
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LADEVEZE_COUPLED_SHEAR_DEGRADATION_RATE/2814
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LAD_COUPLE_SHEAR_DEGRADATION_RATE/2815
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LADEVEZE_COUPLE_SHEAR_DEGRADATION_RATE/2816
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LAD_COUPLED_SHEAR_DEGRADE_RATE/2817
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LADEVEZE_COUPLED_SHEAR_DEGRADE_RATE/2818
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LAD_COUPLE_SHEAR_DEGRADE_RATE/2819
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LADEVEZE_COUPLE_SHEAR_DEGRADE_RATE/2820
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LAD_CMSDR/2821
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LAD_CMSDR_MODEL/2822
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LAD_CMSDR_LAW/2823
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_MATRIX_SHEAR_DEGRADATION/2824
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LAD_COUPLED_MATRIX_SHEAR_DAMAGE_RATE/2825
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LADEVEZE_COUPLED_MATRIX_SHEAR_DAMAGE_RATE/2826
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LAD_COUPLE_MATRIX_SHEAR_DAMAGE_RATE/2827
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/FAIL/LADEVEZE_COUPLE_MATRIX_SHEAR_DAMAGE_RATE/2828
1100.0, 3300.0, 900.0, 20.0, 0.510
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for mid in range(2802, 2829):
        assert mid in model.fail_ladcoupledmatrixsheardegradationrates
        f = model.fail_ladcoupledmatrixsheardegradationrates[mid]
        assert pytest.approx(f.sigma_cmsdr0) == 1100.0
        assert pytest.approx(f.sigma_cmsdrc) == 3300.0
        assert pytest.approx(f.gamma_cmsdr) == 900.0
        assert pytest.approx(f.p_cmsdr) == 20.0
        assert pytest.approx(f.d_cmsdr_max) == 0.510


def test_m454_eng_electrothermoflexomagnetochiralorbitonplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00505:>20.6e}{88:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetochiralorbitonplasmonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALORBITONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Engine Electrothermoflexomagnetochiralorbitonplasmonicpolaritonic Resonance Energy Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralorbitonplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetochiralorbitonplasmonicpolaritonic_resonance_energies[1]
    assert pytest.approx(r.dt_etfcorbitonplp) == 0.00505
    assert r.sens_id == 88


def test_m454_eng_electrothermoflexomagnetochiralorbitonplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetochiralorbitonplasmonicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALORBITONPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
0.00605, 98
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralorbitonplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetochiralorbitonplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(r.dt_etfcorbitonplp) == 0.00605
    assert r.sens_id == 98


def test_m454_eng_electrothermoflexomagnetochiralorbitonplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetochiralorbitonplasmonicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_ORBITON_PLASMON_POLARITON_RES_WORK/3
0.00505, 88
/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALORBITONPLASMONICPOLARITONICRESONANCE/4
0.00505, 88
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALORBITONPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/5
0.00505, 88
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALORBITONPLASMONICPOLARITONIC_RESONANCE/6
0.00505, 88
/ENG/ELECTROTHERMOFLEXOMAGNETOORBITONCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY/7
0.00505, 88
/ENG/ELECTRO_THERM_FLEXO_MAG_ORBITON_CHIRAL_PLASMON_POLARITON_RES_WORK/8
0.00505, 88
/ENG/EELECTROTHERMOFLEXOMAGNETOORBITONCHIRALPLASMONICPOLARITONICRESONANCE/9
0.00505, 88
/ENG/ELECTROTHERMOFLEXOMAGNETOORBITONCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/10
0.00505, 88
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOORBITONCHIRALPLASMONICPOLARITONIC_RESONANCE/11
0.00505, 88
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for rid in range(3, 12):
        assert rid in model.eng_electrothermoflexomagnetochiralorbitonplasmonicpolaritonic_resonance_energies
        r = model.eng_electrothermoflexomagnetochiralorbitonplasmonicpolaritonic_resonance_energies[rid]
        assert pytest.approx(r.dt_etfcorbitonplp) == 0.00505
        assert r.sens_id == 88


def test_m454_lagmul_bundle_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{4.2e6:>20.4f}{11:>10d}{2.9e-5:>20.6e}"
    c2 = f"{27.0:>20.4f}{29.0:>20.4f}{81.0:>20.4f}{13.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Bundle Spinor Spatial Linkage Joint Fixed Format Test
2022 0
/BUNDLE_SPINOR_SPATIAL_LINKAGE_JOINT/1
Bundle Spinor Spatial Mechanism Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_bundle_spinor_spatial_linkage_joints
    j = model.lagmul_bundle_spinor_spatial_linkage_joints[1]
    assert j.node1 == 101
    assert j.node2 == 102
    assert j.node3 == 103
    assert pytest.approx(j.stiff) == 4.2e6
    assert j.skew_id == 11
    assert pytest.approx(j.tol) == 2.9e-5
    assert pytest.approx(j.link_len_a) == 27.0
    assert pytest.approx(j.link_len_b) == 29.0
    assert pytest.approx(j.twist_angle_alpha) == 81.0
    assert pytest.approx(j.offset_distance_s) == 13.0
    assert pytest.approx(j.offset_distance_r) == 13.0
    assert pytest.approx(j.offset_distance_v) == 13.0
    assert pytest.approx(j.offset_distance_h) == 13.0
    assert pytest.approx(j.offset_distance_u) == 13.0
    assert pytest.approx(j.offset_distance_f) == 13.0


def test_m454_lagmul_bundle_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Bundle Spinor Spatial Linkage Joint Free Format Test
/LAGMUL/BUNDLE_SPINOR_SPATIAL_LINKAGE_JOINT/2
Bundle Spinor Spatial Mechanism Joint Free
201, 202, 203, 5.2e6, 12, 3.9e-5
37.0, 39.0, 91.0, 18.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_bundle_spinor_spatial_linkage_joints
    j = model.lagmul_bundle_spinor_spatial_linkage_joints[2]
    assert j.node1 == 201
    assert j.node2 == 202
    assert j.node3 == 203
    assert pytest.approx(j.stiff) == 5.2e6
    assert j.skew_id == 12
    assert pytest.approx(j.tol) == 3.9e-5
    assert pytest.approx(j.link_len_a) == 37.0
    assert pytest.approx(j.link_len_b) == 39.0
    assert pytest.approx(j.twist_angle_alpha) == 91.0
    assert pytest.approx(j.offset_distance_s) == 18.0


def test_m454_lagmul_bundle_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Bundle Spinor Spatial Linkage Joint Aliases Test
/BUNDLE_SPINOR_SPATIAL_LINKAGE/3
101, 102, 103, 4.2e6, 11, 2.9e-5
27.0, 29.0, 81.0, 13.0
/LAGMUL/BUNDLE_SPINOR_SPATIAL_LINKAGE/4
101, 102, 103, 4.2e6, 11, 2.9e-5
27.0, 29.0, 81.0, 13.0
/BUNDLE_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM/5
101, 102, 103, 4.2e6, 11, 2.9e-5
27.0, 29.0, 81.0, 13.0
/BUNDLE_SPINOR_SPATIAL_SYMMETRIC_MECHANISM/6
101, 102, 103, 4.2e6, 11, 2.9e-5
27.0, 29.0, 81.0, 13.0
/BUNDLE_SPINOR_SPATIAL_6R_MECHANISM/7
101, 102, 103, 4.2e6, 11, 2.9e-5
27.0, 29.0, 81.0, 13.0
/BUNDLE_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM/8
101, 102, 103, 4.2e6, 11, 2.9e-5
27.0, 29.0, 81.0, 13.0
/LAGMUL/BUNDLE_TWISTOR_SPATIAL_LINKAGE_JOINT/9
101, 102, 103, 4.2e6, 11, 2.9e-5
27.0, 29.0, 81.0, 13.0
/BUNDLE_TWISTOR_SPATIAL_LINKAGE_JOINT/10
101, 102, 103, 4.2e6, 11, 2.9e-5
27.0, 29.0, 81.0, 13.0
/LAGMUL/BUNDLE_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/11
101, 102, 103, 4.2e6, 11, 2.9e-5
27.0, 29.0, 81.0, 13.0
/BUNDLE_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/12
101, 102, 103, 4.2e6, 11, 2.9e-5
27.0, 29.0, 81.0, 13.0
/LAGMUL/BUNDLE_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/13
101, 102, 103, 4.2e6, 11, 2.9e-5
27.0, 29.0, 81.0, 13.0
/BUNDLE_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/14
101, 102, 103, 4.2e6, 11, 2.9e-5
27.0, 29.0, 81.0, 13.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(3, 15):
        assert jid in model.lagmul_bundle_spinor_spatial_linkage_joints
        j = model.lagmul_bundle_spinor_spatial_linkage_joints[jid]
        assert j.node1 == 101
        assert j.node2 == 102
        assert j.node3 == 103
        assert pytest.approx(j.stiff) == 4.2e6


def test_m454_sensor_spring_normal_pop_rate_fixed(tmp_path: Path):
    c1 = f"{1208:>10d}{9.85e6:>20.4f}{0.0088:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Normal Pop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_NORMAL_POP_RATE/1
Spring Normal Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_normal_pop_rates
    sensor = model.sensor_spring_normal_pop_rates[1]
    assert sensor.spring_id == 1208
    assert pytest.approx(sensor.jnorm_pop_max) == 9.85e6
    assert pytest.approx(sensor.j_pop_norm_max) == 9.85e6
    assert pytest.approx(sensor.j_pop_normal_max) == 9.85e6
    assert pytest.approx(sensor.j_norm_pop_max) == 9.85e6
    assert pytest.approx(sensor.j_pop_max) == 9.85e6
    assert pytest.approx(sensor.j_pop_axial_max) == 9.85e6
    assert pytest.approx(sensor.jnorm_shot_max) == 9.85e6
    assert pytest.approx(sensor.jnorm_drop_max) == 9.85e6
    assert pytest.approx(sensor.jnorm_lock_max) == 9.85e6
    assert pytest.approx(sensor.jnorm_snp_max) == 9.85e6
    assert pytest.approx(sensor.jnorm_snap_max) == 9.85e6
    assert pytest.approx(sensor.jnorm_rate_max) == 9.85e6
    assert pytest.approx(sensor.jnorm_roc_rate_max) == 9.85e6
    assert pytest.approx(sensor.jnorm_drop_rate_max) == 9.85e6
    assert pytest.approx(sensor.jnorm_crk_rate_max) == 9.85e6
    assert pytest.approx(sensor.jnorm_pop_rate_max) == 9.85e6
    assert pytest.approx(sensor.t_delay) == 0.0088
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_NORMAL_POP_RATE"


def test_m454_sensor_spring_normal_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Normal Pop Rate Free Format Test
/SENSOR/SPRING_NORMAL_POP_RATE/2
48, 6.1e5, 0.006
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_normal_pop_rates
    sensor = model.sensor_spring_normal_pop_rates[2]
    assert sensor.spring_id == 48
    assert pytest.approx(sensor.jnorm_pop_max) == 6.1e5
    assert pytest.approx(sensor.t_delay) == 0.006
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_NORMAL_POP_RATE"


def test_m454_sensor_spring_normal_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Normal Pop Rate Aliases Test
/SENSOR/SPRING_NORM_POP_RATE/3
48, 6.1e5, 0.006
/SENSOR/SPRING_NORMAL_POP/4
48, 6.1e5, 0.006
/SENSOR/SPRING_NORM_POP/5
48, 6.1e5, 0.006
/SENSOR/SPRING_AXIAL_POP_RATE/6
48, 6.1e5, 0.006
/SENSOR/SPRING_AXIAL_POP/7
48, 6.1e5, 0.006
/SENSOR/SPRING_POP_RATE_NORMAL/8
48, 6.1e5, 0.006
/SENSOR/SPRING_POP_RATE_NORM/9
48, 6.1e5, 0.006
/SENSOR/SPRING_POP_RATE_AXIAL/10
48, 6.1e5, 0.006
/SENSOR/NORMAL_POP_RATE_SPRING/11
48, 6.1e5, 0.006
/SENSOR/NORM_POP_RATE_SPRING/12
48, 6.1e5, 0.006
/SENSOR/AXIAL_POP_RATE_SPRING/13
48, 6.1e5, 0.006
/SENSOR/SPRING_POP_NORMAL/14
48, 6.1e5, 0.006
/SENSOR/SPRING_POP_NORM/15
48, 6.1e5, 0.006
/SENSOR/SPRING_POP_AXIAL/16
48, 6.1e5, 0.006
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(3, 17):
        assert sid in model.sensor_spring_normal_pop_rates
        s = model.sensor_spring_normal_pop_rates[sid]
        assert s.spring_id == 48
        assert pytest.approx(s.jnorm_pop_max) == 6.1e5


def test_m454_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_COUPLED_MATRIX_SHEAR_DEGRADATION_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALORBITONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/BUNDLE_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_NORMAL_POP_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4
