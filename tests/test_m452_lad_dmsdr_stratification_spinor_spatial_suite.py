"""Tests for Milestone M452: LadDynamicMatrixShearDegradationRate Failure Model, EngElectrothermoflexomagnetochiralspinonplasmonicpolaritonicResonanceEnergy, LagmulStratificationSpinorSpatialLinkageJoint, and SensorSpringTorsionalCrackleRate."""

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


def test_m452_fail_lad_dynamic_matrix_shear_degradation_rate_fixed(tmp_path: Path):
    c1 = f"{1080.0:>20.4f}{3240.0:>20.4f}{880.0:>20.4f}{18.0:>20.4f}{0.490:>20.4f}"
    c2 = f"{2:>10d}{2:>10d}"
    c3 = f"{780:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Matrix Shear Degradation Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_MATRIX_SHEAR_DEGRADATION_RATE/2780
Ladeveze Dynamic Matrix Shear Degradation Law
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2780 in model.fail_laddynamicmatrixsheardegradationrates
    f = model.fail_laddynamicmatrixsheardegradationrates[2780]
    assert pytest.approx(f.sigma_dmsdr0) == 1080.0
    assert pytest.approx(f.sigma_dmsdrc) == 3240.0
    assert pytest.approx(f.gamma_dmsdr) == 880.0
    assert pytest.approx(f.p_dmsdr) == 18.0
    assert pytest.approx(f.d_dmsdr_max) == 0.490
    assert f.ifail_sh == 2
    assert f.ifail_so == 2
    assert f.fail_id == 780
    assert pytest.approx(f.sigma_dmsd0) == 1080.0
    assert pytest.approx(f.sigma_dmsdc) == 3240.0
    assert pytest.approx(f.gamma_dmsd) == 880.0
    assert pytest.approx(f.p_dmsd) == 18.0
    assert pytest.approx(f.d_dmsd_max) == 0.490
    assert pytest.approx(f.sigma_dms0) == 1080.0
    assert pytest.approx(f.sigma_dmsc) == 3240.0
    assert pytest.approx(f.gamma_dms) == 880.0
    assert pytest.approx(f.p_dms) == 18.0
    assert pytest.approx(f.d_dms_max) == 0.490
    assert pytest.approx(f.sigma_dsdr0) == 1080.0
    assert pytest.approx(f.sigma_dsdrc) == 3240.0
    assert pytest.approx(f.gamma_dsdr) == 880.0
    assert pytest.approx(f.p_dsdr) == 18.0
    assert pytest.approx(f.d_dsdr_max) == 0.490


def test_m452_fail_lad_dynamic_matrix_shear_degradation_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Matrix Shear Degradation Rate Free Format Test
/FAIL/LAD_DYNAMIC_MATRIX_SHEAR_DEGRADATION_RATE/2781
1180.0, 3540.0, 980.0, 20.0, 0.590
1, 1
880
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2781 in model.fail_laddynamicmatrixsheardegradationrates
    f = model.fail_laddynamicmatrixsheardegradationrates[2781]
    assert pytest.approx(f.sigma_dmsdr0) == 1180.0
    assert pytest.approx(f.sigma_dmsdrc) == 3540.0
    assert pytest.approx(f.gamma_dmsdr) == 980.0
    assert pytest.approx(f.p_dmsdr) == 20.0
    assert pytest.approx(f.d_dmsdr_max) == 0.590
    assert f.ifail_sh == 1
    assert f.ifail_so == 1
    assert f.fail_id == 880


def test_m452_fail_lad_dynamic_matrix_shear_degradation_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Matrix Shear Degradation Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_MATRIX_SHEAR_DEGRADATION_RATE/2782
1080.0, 3240.0, 880.0, 18.0, 0.490
1, 1
/FAIL/LAD_DYNAMIC_MATRIX_SHEAR_DEGRADE_RATE/2783
1080.0, 3240.0, 880.0, 18.0, 0.490
1, 1
/FAIL/LADEVEZE_DYNAMIC_MATRIX_SHEAR_DEGRADE_RATE/2784
1080.0, 3240.0, 880.0, 18.0, 0.490
1, 1
/FAIL/LAD_DYNAMIC_MATRIX_SHEAR_RATE/2785
1080.0, 3240.0, 880.0, 18.0, 0.490
1, 1
/FAIL/LADEVEZE_DYNAMIC_MATRIX_SHEAR_RATE/2786
1080.0, 3240.0, 880.0, 18.0, 0.490
1, 1
/FAIL/LAD_DYNAMIC_SHEAR_DEGRADATION_RATE/2787
1080.0, 3240.0, 880.0, 18.0, 0.490
1, 1
/FAIL/LADEVEZE_DYNAMIC_SHEAR_DEGRADATION_RATE/2788
1080.0, 3240.0, 880.0, 18.0, 0.490
1, 1
/FAIL/LAD_DYNAMIC_SHEAR_DEGRADE_RATE/2789
1080.0, 3240.0, 880.0, 18.0, 0.490
1, 1
/FAIL/LADEVEZE_DYNAMIC_SHEAR_DEGRADE_RATE/2790
1080.0, 3240.0, 880.0, 18.0, 0.490
1, 1
/FAIL/LAD_DMSDR/2791
1080.0, 3240.0, 880.0, 18.0, 0.490
1, 1
/FAIL/LAD_DMSDR_MODEL/2792
1080.0, 3240.0, 880.0, 18.0, 0.490
1, 1
/FAIL/LAD_DMSDR_LAW/2793
1080.0, 3240.0, 880.0, 18.0, 0.490
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_MATRIX_SHEAR_DEGRADATION/2794
1080.0, 3240.0, 880.0, 18.0, 0.490
1, 1
/FAIL/LAD_DYNAMIC_MATRIX_SHEAR_DAMAGE_RATE/2795
1080.0, 3240.0, 880.0, 18.0, 0.490
1, 1
/FAIL/LADEVEZE_DYNAMIC_MATRIX_SHEAR_DAMAGE_RATE/2796
1080.0, 3240.0, 880.0, 18.0, 0.490
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for mid in range(2782, 2797):
        assert mid in model.fail_laddynamicmatrixsheardegradationrates
        f = model.fail_laddynamicmatrixsheardegradationrates[mid]
        assert pytest.approx(f.sigma_dmsdr0) == 1080.0
        assert pytest.approx(f.sigma_dmsdrc) == 3240.0
        assert pytest.approx(f.gamma_dmsdr) == 880.0
        assert pytest.approx(f.p_dmsdr) == 18.0
        assert pytest.approx(f.d_dmsdr_max) == 0.490


def test_m452_eng_electrothermoflexomagnetochiralspinonplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00485:>20.6e}{86:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetochiralspinonplasmonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSPINONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Engine Electrothermoflexomagnetochiralspinonplasmonicpolaritonic Resonance Energy Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralspinonplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetochiralspinonplasmonicpolaritonic_resonance_energies[1]
    assert pytest.approx(r.dt_etfcspinonplp) == 0.00485
    assert r.sens_id == 86


def test_m452_eng_electrothermoflexomagnetochiralspinonplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetochiralspinonplasmonicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSPINONPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
0.00585, 96
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralspinonplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetochiralspinonplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(r.dt_etfcspinonplp) == 0.00585
    assert r.sens_id == 96


def test_m452_eng_electrothermoflexomagnetochiralspinonplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetochiralspinonplasmonicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_SPINON_PLASMON_POLARITON_RES_WORK/3
0.00485, 86
/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALSPINONPLASMONICPOLARITONICRESONANCE/4
0.00485, 86
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSPINONPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/5
0.00485, 86
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALSPINONPLASMONICPOLARITONIC_RESONANCE/6
0.00485, 86
/ENG/ELECTROTHERMOFLEXOMAGNETOSPINDROPCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY/7
0.00485, 86
/ENG/ELECTRO_THERM_FLEXO_MAG_SPINDROP_CHIRAL_PLASMON_POLARITON_RES_WORK/8
0.00485, 86
/ENG/EELECTROTHERMOFLEXOMAGNETOSPINDROPCHIRALPLASMONICPOLARITONICRESONANCE/9
0.00485, 86
/ENG/ELECTROTHERMOFLEXOMAGNETOSPINDROPCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/10
0.00485, 86
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOSPINDROPCHIRALPLASMONICPOLARITONIC_RESONANCE/11
0.00485, 86
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for rid in range(3, 12):
        assert rid in model.eng_electrothermoflexomagnetochiralspinonplasmonicpolaritonic_resonance_energies
        r = model.eng_electrothermoflexomagnetochiralspinonplasmonicpolaritonic_resonance_energies[rid]
        assert pytest.approx(r.dt_etfcspinonplp) == 0.00485
        assert r.sens_id == 86


def test_m452_lagmul_stratification_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{4.0e6:>20.4f}{9:>10d}{2.7e-5:>20.6e}"
    c2 = f"{25.0:>20.4f}{27.0:>20.4f}{79.0:>20.4f}{11.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Stratification Spinor Spatial Linkage Joint Fixed Format Test
2022 0
/STRATIFICATION_SPINOR_SPATIAL_LINKAGE_JOINT/1
Stratification Spinor Spatial Mechanism Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_stratification_spinor_spatial_linkage_joints
    j = model.lagmul_stratification_spinor_spatial_linkage_joints[1]
    assert j.node1 == 101
    assert j.node2 == 102
    assert j.node3 == 103
    assert pytest.approx(j.stiff) == 4.0e6
    assert j.skew_id == 9
    assert pytest.approx(j.tol) == 2.7e-5
    assert pytest.approx(j.link_len_a) == 25.0
    assert pytest.approx(j.link_len_b) == 27.0
    assert pytest.approx(j.twist_angle_alpha) == 79.0
    assert pytest.approx(j.offset_distance_s) == 11.0
    assert pytest.approx(j.offset_distance_r) == 11.0
    assert pytest.approx(j.offset_distance_v) == 11.0
    assert pytest.approx(j.offset_distance_h) == 11.0
    assert pytest.approx(j.offset_distance_u) == 11.0
    assert pytest.approx(j.offset_distance_f) == 11.0


def test_m452_lagmul_stratification_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Stratification Spinor Spatial Linkage Joint Free Format Test
/LAGMUL/STRATIFICATION_SPINOR_SPATIAL_LINKAGE_JOINT/2
Stratification Spinor Spatial Mechanism Joint Free
201, 202, 203, 5.0e6, 10, 3.7e-5
35.0, 37.0, 89.0, 16.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_stratification_spinor_spatial_linkage_joints
    j = model.lagmul_stratification_spinor_spatial_linkage_joints[2]
    assert j.node1 == 201
    assert j.node2 == 202
    assert j.node3 == 203
    assert pytest.approx(j.stiff) == 5.0e6
    assert j.skew_id == 10
    assert pytest.approx(j.tol) == 3.7e-5
    assert pytest.approx(j.link_len_a) == 35.0
    assert pytest.approx(j.link_len_b) == 37.0
    assert pytest.approx(j.twist_angle_alpha) == 89.0
    assert pytest.approx(j.offset_distance_s) == 16.0


def test_m452_lagmul_stratification_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Stratification Spinor Spatial Linkage Joint Aliases Test
/STRATIFICATION_SPINOR_SPATIAL_LINKAGE/3
101, 102, 103, 4.0e6, 9, 2.7e-5
25.0, 27.0, 79.0, 11.0
/LAGMUL/STRATIFICATION_SPINOR_SPATIAL_LINKAGE/4
101, 102, 103, 4.0e6, 9, 2.7e-5
25.0, 27.0, 79.0, 11.0
/STRATIFICATION_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM/5
101, 102, 103, 4.0e6, 9, 2.7e-5
25.0, 27.0, 79.0, 11.0
/STRATIFICATION_SPINOR_SPATIAL_SYMMETRIC_MECHANISM/6
101, 102, 103, 4.0e6, 9, 2.7e-5
25.0, 27.0, 79.0, 11.0
/STRATIFICATION_SPINOR_SPATIAL_6R_MECHANISM/7
101, 102, 103, 4.0e6, 9, 2.7e-5
25.0, 27.0, 79.0, 11.0
/STRATIFICATION_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM/8
101, 102, 103, 4.0e6, 9, 2.7e-5
25.0, 27.0, 79.0, 11.0
/LAGMUL/STRATIFICATION_TWISTOR_SPATIAL_LINKAGE_JOINT/9
101, 102, 103, 4.0e6, 9, 2.7e-5
25.0, 27.0, 79.0, 11.0
/STRATIFICATION_TWISTOR_SPATIAL_LINKAGE_JOINT/10
101, 102, 103, 4.0e6, 9, 2.7e-5
25.0, 27.0, 79.0, 11.0
/LAGMUL/STRATIFICATION_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/11
101, 102, 103, 4.0e6, 9, 2.7e-5
25.0, 27.0, 79.0, 11.0
/STRATIFICATION_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/12
101, 102, 103, 4.0e6, 9, 2.7e-5
25.0, 27.0, 79.0, 11.0
/LAGMUL/STRATIFICATION_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/13
101, 102, 103, 4.0e6, 9, 2.7e-5
25.0, 27.0, 79.0, 11.0
/STRATIFICATION_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/14
101, 102, 103, 4.0e6, 9, 2.7e-5
25.0, 27.0, 79.0, 11.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(3, 15):
        assert jid in model.lagmul_stratification_spinor_spatial_linkage_joints
        j = model.lagmul_stratification_spinor_spatial_linkage_joints[jid]
        assert j.node1 == 101
        assert j.node2 == 102
        assert j.node3 == 103
        assert pytest.approx(j.stiff) == 4.0e6


def test_m452_sensor_spring_torsional_crackle_rate_fixed(tmp_path: Path):
    c1 = f"{1206:>10d}{9.65e6:>20.4f}{0.0086:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Crackle Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE/1
Spring Torsional Crackle Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_crackle_rates
    sensor = model.sensor_spring_torsional_crackle_rates[1]
    assert sensor.spring_id == 1206
    assert pytest.approx(sensor.jtors_crk_max) == 9.65e6
    assert pytest.approx(sensor.j_crk_tors_max) == 9.65e6
    assert pytest.approx(sensor.j_crackle_tors_max) == 9.65e6
    assert pytest.approx(sensor.j_tors_crk_max) == 9.65e6
    assert pytest.approx(sensor.j_crk_max) == 9.65e6
    assert pytest.approx(sensor.jtors_crackle_max) == 9.65e6
    assert pytest.approx(sensor.j_crk_torsional_max) == 9.65e6
    assert pytest.approx(sensor.j_crackle_torsional_max) == 9.65e6
    assert pytest.approx(sensor.jtors_shot_max) == 9.65e6
    assert pytest.approx(sensor.jtors_drop_max) == 9.65e6
    assert pytest.approx(sensor.jtors_lock_max) == 9.65e6
    assert pytest.approx(sensor.jtors_pop_max) == 9.65e6
    assert pytest.approx(sensor.jtors_snp_max) == 9.65e6
    assert pytest.approx(sensor.jtors_snap_max) == 9.65e6
    assert pytest.approx(sensor.jtors_rate_max) == 9.65e6
    assert pytest.approx(sensor.jtors_roc_rate_max) == 9.65e6
    assert pytest.approx(sensor.jtors_drop_rate_max) == 9.65e6
    assert pytest.approx(sensor.jtors_crk_rate_max) == 9.65e6
    assert pytest.approx(sensor.t_delay) == 0.0086
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_CRACKLE_RATE"


def test_m452_sensor_spring_torsional_crackle_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Torsional Crackle Rate Free Format Test
/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE/2
46, 5.9e5, 0.004
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_torsional_crackle_rates
    sensor = model.sensor_spring_torsional_crackle_rates[2]
    assert sensor.spring_id == 46
    assert pytest.approx(sensor.jtors_crk_max) == 5.9e5
    assert pytest.approx(sensor.t_delay) == 0.004
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_CRACKLE_RATE"


def test_m452_sensor_spring_torsional_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Torsional Crackle Rate Aliases Test
/SENSOR/SPRING_TORSION_CRACKLE_RATE/3
46, 5.9e5, 0.004
/SENSOR/SPRING_TORSIONAL_CRK_RATE/4
46, 5.9e5, 0.004
/SENSOR/SPRING_TORSION_CRK_RATE/5
46, 5.9e5, 0.004
/SENSOR/SPRING_TORS_CRK_RATE/6
46, 5.9e5, 0.004
/SENSOR/SPRING_TORSIONAL_CRACKLE/7
46, 5.9e5, 0.004
/SENSOR/SPRING_TORSION_CRACKLE/8
46, 5.9e5, 0.004
/SENSOR/SPRING_TORS_CRACKLE/9
46, 5.9e5, 0.004
/SENSOR/SPRING_TORSIONAL_CRK/10
46, 5.9e5, 0.004
/SENSOR/SPRING_TORSION_CRK/11
46, 5.9e5, 0.004
/SENSOR/SPRING_TORS_CRK/12
46, 5.9e5, 0.004
/SENSOR/SPRING_TWIST_CRACKLE_RATE/13
46, 5.9e5, 0.004
/SENSOR/SPRING_TWIST_CRK_RATE/14
46, 5.9e5, 0.004
/SENSOR/SPRING_TWIST_CRACKLE/15
46, 5.9e5, 0.004
/SENSOR/SPRING_TWIST_CRK/16
46, 5.9e5, 0.004
/SENSOR/SPRING_POP_RATE_TORSIONAL_CRACKLE/17
46, 5.9e5, 0.004
/SENSOR/SPRING_POP_RATE_TORSION_CRACKLE/18
46, 5.9e5, 0.004
/SENSOR/SPRING_POP_RATE_TWIST_CRACKLE/19
46, 5.9e5, 0.004
/SENSOR/TORSIONAL_CRACKLE_RATE_SPRING/20
46, 5.9e5, 0.004
/SENSOR/TORSION_CRACKLE_RATE_SPRING/21
46, 5.9e5, 0.004
/SENSOR/SPRING_CRACKLE_RATE_TORSIONAL/22
46, 5.9e5, 0.004
/SENSOR/SPRING_CRACKLE_RATE_TORSION/23
46, 5.9e5, 0.004
/SENSOR/SPRING_CRACKLE_RATE_TORS/24
46, 5.9e5, 0.004
/SENSOR/SPRING_CRK_RATE_TORSIONAL/25
46, 5.9e5, 0.004
/SENSOR/SPRING_CRK_RATE_TORSION/26
46, 5.9e5, 0.004
/SENSOR/SPRING_CRK_RATE_TORS/27
46, 5.9e5, 0.004
/SENSOR/SPRING_RATE_CRACKLE_TORSIONAL/28
46, 5.9e5, 0.004
/SENSOR/SPRING_RATE_CRACKLE_TORSION/29
46, 5.9e5, 0.004
/SENSOR/SPRING_RATE_CRACKLE_TORS/30
46, 5.9e5, 0.004
/SENSOR/SPRING_CRACKLE_TORSIONAL/31
46, 5.9e5, 0.004
/SENSOR/SPRING_CRACKLE_TORSION/32
46, 5.9e5, 0.004
/SENSOR/SPRING_CRACKLE_TORS/33
46, 5.9e5, 0.004
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(3, 34):
        assert sid in model.sensor_spring_torsional_crackle_rates
        s = model.sensor_spring_torsional_crackle_rates[sid]
        assert s.spring_id == 46
        assert pytest.approx(s.jtors_crk_max) == 5.9e5


def test_m452_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_DYNAMIC_MATRIX_SHEAR_DEGRADATION_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALSPINONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/STRATIFICATION_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4
