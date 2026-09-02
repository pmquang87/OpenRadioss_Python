"""Tests for Milestone M453: LadTransverseMatrixShearDegradationRate Failure Model, EngElectrothermoflexomagnetochiralholonplasmonicpolaritonicResonanceEnergy, LagmulFibrationSpinorSpatialLinkageJoint, and SensorSpringTotalCrackleRate."""

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


def test_m453_fail_lad_transverse_matrix_shear_degradation_rate_fixed(tmp_path: Path):
    c1 = f"{1090.0:>20.4f}{3270.0:>20.4f}{890.0:>20.4f}{19.0:>20.4f}{0.500:>20.4f}"
    c2 = f"{2:>10d}{2:>10d}"
    c3 = f"{790:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Matrix Shear Degradation Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_MATRIX_SHEAR_DEGRADATION_RATE/2790
Ladeveze Transverse Matrix Shear Degradation Law
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2790 in model.fail_ladtransversematrixsheardegradationrates
    f = model.fail_ladtransversematrixsheardegradationrates[2790]
    assert pytest.approx(f.sigma_tmsdr0) == 1090.0
    assert pytest.approx(f.sigma_tmsdrc) == 3270.0
    assert pytest.approx(f.gamma_tmsdr) == 890.0
    assert pytest.approx(f.p_tmsdr) == 19.0
    assert pytest.approx(f.d_tmsdr_max) == 0.500
    assert f.ifail_sh == 2
    assert f.ifail_so == 2
    assert f.fail_id == 790
    assert pytest.approx(f.sigma_tmsd0) == 1090.0
    assert pytest.approx(f.sigma_tmsdc) == 3270.0
    assert pytest.approx(f.gamma_tmsd) == 890.0
    assert pytest.approx(f.p_tmsd) == 19.0
    assert pytest.approx(f.d_tmsd_max) == 0.500
    assert pytest.approx(f.sigma_tms0) == 1090.0
    assert pytest.approx(f.sigma_tmsc) == 3270.0
    assert pytest.approx(f.gamma_tms) == 890.0
    assert pytest.approx(f.p_tms) == 19.0
    assert pytest.approx(f.d_tms_max) == 0.500
    assert pytest.approx(f.sigma_tsdr0) == 1090.0
    assert pytest.approx(f.sigma_tsdrc) == 3270.0
    assert pytest.approx(f.gamma_tsdr) == 890.0
    assert pytest.approx(f.p_tsdr) == 19.0
    assert pytest.approx(f.d_tsdr_max) == 0.500


def test_m453_fail_lad_transverse_matrix_shear_degradation_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Matrix Shear Degradation Rate Free Format Test
/FAIL/LAD_TRANSVERSE_MATRIX_SHEAR_DEGRADATION_RATE/2791
1190.0, 3570.0, 990.0, 21.0, 0.600
1, 1
890
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2791 in model.fail_ladtransversematrixsheardegradationrates
    f = model.fail_ladtransversematrixsheardegradationrates[2791]
    assert pytest.approx(f.sigma_tmsdr0) == 1190.0
    assert pytest.approx(f.sigma_tmsdrc) == 3570.0
    assert pytest.approx(f.gamma_tmsdr) == 990.0
    assert pytest.approx(f.p_tmsdr) == 21.0
    assert pytest.approx(f.d_tmsdr_max) == 0.600
    assert f.ifail_sh == 1
    assert f.ifail_so == 1
    assert f.fail_id == 890


def test_m453_fail_lad_transverse_matrix_shear_degradation_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Matrix Shear Degradation Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_MATRIX_SHEAR_DEGRADATION_RATE/2792
1090.0, 3270.0, 890.0, 19.0, 0.500
1, 1
/FAIL/LAD_TRANSVERSE_MATRIX_SHEAR_DEGRADE_RATE/2793
1090.0, 3270.0, 890.0, 19.0, 0.500
1, 1
/FAIL/LADEVEZE_TRANSVERSE_MATRIX_SHEAR_DEGRADE_RATE/2794
1090.0, 3270.0, 890.0, 19.0, 0.500
1, 1
/FAIL/LAD_TRANSVERSE_MATRIX_SHEAR_RATE/2795
1090.0, 3270.0, 890.0, 19.0, 0.500
1, 1
/FAIL/LADEVEZE_TRANSVERSE_MATRIX_SHEAR_RATE/2796
1090.0, 3270.0, 890.0, 19.0, 0.500
1, 1
/FAIL/LAD_TRANSVERSE_SHEAR_DEGRADATION_RATE/2797
1090.0, 3270.0, 890.0, 19.0, 0.500
1, 1
/FAIL/LADEVEZE_TRANSVERSE_SHEAR_DEGRADATION_RATE/2798
1090.0, 3270.0, 890.0, 19.0, 0.500
1, 1
/FAIL/LAD_TRANSVERSE_SHEAR_DEGRADE_RATE/2799
1090.0, 3270.0, 890.0, 19.0, 0.500
1, 1
/FAIL/LADEVEZE_TRANSVERSE_SHEAR_DEGRADE_RATE/2800
1090.0, 3270.0, 890.0, 19.0, 0.500
1, 1
/FAIL/LAD_TMSDR/2801
1090.0, 3270.0, 890.0, 19.0, 0.500
1, 1
/FAIL/LAD_TMSDR_MODEL/2802
1090.0, 3270.0, 890.0, 19.0, 0.500
1, 1
/FAIL/LAD_TMSDR_LAW/2803
1090.0, 3270.0, 890.0, 19.0, 0.500
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_MATRIX_SHEAR_DEGRADATION/2804
1090.0, 3270.0, 890.0, 19.0, 0.500
1, 1
/FAIL/LAD_TRANSVERSE_MATRIX_SHEAR_DAMAGE_RATE/2805
1090.0, 3270.0, 890.0, 19.0, 0.500
1, 1
/FAIL/LADEVEZE_TRANSVERSE_MATRIX_SHEAR_DAMAGE_RATE/2806
1090.0, 3270.0, 890.0, 19.0, 0.500
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for mid in range(2792, 2807):
        assert mid in model.fail_ladtransversematrixsheardegradationrates
        f = model.fail_ladtransversematrixsheardegradationrates[mid]
        assert pytest.approx(f.sigma_tmsdr0) == 1090.0
        assert pytest.approx(f.sigma_tmsdrc) == 3270.0
        assert pytest.approx(f.gamma_tmsdr) == 890.0
        assert pytest.approx(f.p_tmsdr) == 19.0
        assert pytest.approx(f.d_tmsdr_max) == 0.500


def test_m453_eng_electrothermoflexomagnetochiralholonplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00495:>20.6e}{87:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetochiralholonplasmonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALHOLONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Engine Electrothermoflexomagnetochiralholonplasmonicpolaritonic Resonance Energy Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralholonplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetochiralholonplasmonicpolaritonic_resonance_energies[1]
    assert pytest.approx(r.dt_etfcholonplp) == 0.00495
    assert r.sens_id == 87


def test_m453_eng_electrothermoflexomagnetochiralholonplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetochiralholonplasmonicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALHOLONPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
0.00595, 97
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralholonplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetochiralholonplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(r.dt_etfcholonplp) == 0.00595
    assert r.sens_id == 97


def test_m453_eng_electrothermoflexomagnetochiralholonplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetochiralholonplasmonicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_HOLON_PLASMON_POLARITON_RES_WORK/3
0.00495, 87
/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALHOLONPLASMONICPOLARITONICRESONANCE/4
0.00495, 87
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALHOLONPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/5
0.00495, 87
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALHOLONPLASMONICPOLARITONIC_RESONANCE/6
0.00495, 87
/ENG/ELECTROTHERMOFLEXOMAGNETOHOLONCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY/7
0.00495, 87
/ENG/ELECTRO_THERM_FLEXO_MAG_HOLON_CHIRAL_PLASMON_POLARITON_RES_WORK/8
0.00495, 87
/ENG/EELECTROTHERMOFLEXOMAGNETOHOLONCHIRALPLASMONICPOLARITONICRESONANCE/9
0.00495, 87
/ENG/ELECTROTHERMOFLEXOMAGNETOHOLONCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/10
0.00495, 87
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOHOLONCHIRALPLASMONICPOLARITONIC_RESONANCE/11
0.00495, 87
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for rid in range(3, 12):
        assert rid in model.eng_electrothermoflexomagnetochiralholonplasmonicpolaritonic_resonance_energies
        r = model.eng_electrothermoflexomagnetochiralholonplasmonicpolaritonic_resonance_energies[rid]
        assert pytest.approx(r.dt_etfcholonplp) == 0.00495
        assert r.sens_id == 87


def test_m453_lagmul_fibration_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{4.1e6:>20.4f}{10:>10d}{2.8e-5:>20.6e}"
    c2 = f"{26.0:>20.4f}{28.0:>20.4f}{80.0:>20.4f}{12.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Fibration Spinor Spatial Linkage Joint Fixed Format Test
2022 0
/FIBRATION_SPINOR_SPATIAL_LINKAGE_JOINT/1
Fibration Spinor Spatial Mechanism Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_fibration_spinor_spatial_linkage_joints
    j = model.lagmul_fibration_spinor_spatial_linkage_joints[1]
    assert j.node1 == 101
    assert j.node2 == 102
    assert j.node3 == 103
    assert pytest.approx(j.stiff) == 4.1e6
    assert j.skew_id == 10
    assert pytest.approx(j.tol) == 2.8e-5
    assert pytest.approx(j.link_len_a) == 26.0
    assert pytest.approx(j.link_len_b) == 28.0
    assert pytest.approx(j.twist_angle_alpha) == 80.0
    assert pytest.approx(j.offset_distance_s) == 12.0
    assert pytest.approx(j.offset_distance_r) == 12.0
    assert pytest.approx(j.offset_distance_v) == 12.0
    assert pytest.approx(j.offset_distance_h) == 12.0
    assert pytest.approx(j.offset_distance_u) == 12.0
    assert pytest.approx(j.offset_distance_f) == 12.0


def test_m453_lagmul_fibration_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Fibration Spinor Spatial Linkage Joint Free Format Test
/LAGMUL/FIBRATION_SPINOR_SPATIAL_LINKAGE_JOINT/2
Fibration Spinor Spatial Mechanism Joint Free
201, 202, 203, 5.1e6, 11, 3.8e-5
36.0, 38.0, 90.0, 17.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_fibration_spinor_spatial_linkage_joints
    j = model.lagmul_fibration_spinor_spatial_linkage_joints[2]
    assert j.node1 == 201
    assert j.node2 == 202
    assert j.node3 == 203
    assert pytest.approx(j.stiff) == 5.1e6
    assert j.skew_id == 11
    assert pytest.approx(j.tol) == 3.8e-5
    assert pytest.approx(j.link_len_a) == 36.0
    assert pytest.approx(j.link_len_b) == 38.0
    assert pytest.approx(j.twist_angle_alpha) == 90.0
    assert pytest.approx(j.offset_distance_s) == 17.0


def test_m453_lagmul_fibration_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Fibration Spinor Spatial Linkage Joint Aliases Test
/FIBRATION_SPINOR_SPATIAL_LINKAGE/3
101, 102, 103, 4.1e6, 10, 2.8e-5
26.0, 28.0, 80.0, 12.0
/LAGMUL/FIBRATION_SPINOR_SPATIAL_LINKAGE/4
101, 102, 103, 4.1e6, 10, 2.8e-5
26.0, 28.0, 80.0, 12.0
/FIBRATION_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM/5
101, 102, 103, 4.1e6, 10, 2.8e-5
26.0, 28.0, 80.0, 12.0
/FIBRATION_SPINOR_SPATIAL_SYMMETRIC_MECHANISM/6
101, 102, 103, 4.1e6, 10, 2.8e-5
26.0, 28.0, 80.0, 12.0
/FIBRATION_SPINOR_SPATIAL_6R_MECHANISM/7
101, 102, 103, 4.1e6, 10, 2.8e-5
26.0, 28.0, 80.0, 12.0
/FIBRATION_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM/8
101, 102, 103, 4.1e6, 10, 2.8e-5
26.0, 28.0, 80.0, 12.0
/LAGMUL/FIBRATION_TWISTOR_SPATIAL_LINKAGE_JOINT/9
101, 102, 103, 4.1e6, 10, 2.8e-5
26.0, 28.0, 80.0, 12.0
/FIBRATION_TWISTOR_SPATIAL_LINKAGE_JOINT/10
101, 102, 103, 4.1e6, 10, 2.8e-5
26.0, 28.0, 80.0, 12.0
/LAGMUL/FIBRATION_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/11
101, 102, 103, 4.1e6, 10, 2.8e-5
26.0, 28.0, 80.0, 12.0
/FIBRATION_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/12
101, 102, 103, 4.1e6, 10, 2.8e-5
26.0, 28.0, 80.0, 12.0
/LAGMUL/FIBRATION_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/13
101, 102, 103, 4.1e6, 10, 2.8e-5
26.0, 28.0, 80.0, 12.0
/FIBRATION_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/14
101, 102, 103, 4.1e6, 10, 2.8e-5
26.0, 28.0, 80.0, 12.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(3, 15):
        assert jid in model.lagmul_fibration_spinor_spatial_linkage_joints
        j = model.lagmul_fibration_spinor_spatial_linkage_joints[jid]
        assert j.node1 == 101
        assert j.node2 == 102
        assert j.node3 == 103
        assert pytest.approx(j.stiff) == 4.1e6


def test_m453_sensor_spring_total_crackle_rate_fixed(tmp_path: Path):
    c1 = f"{1207:>10d}{9.75e6:>20.4f}{0.0087:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Crackle Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_CRACKLE_RATE/1
Spring Total Crackle Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_crackle_rates
    sensor = model.sensor_spring_total_crackle_rates[1]
    assert sensor.spring_id == 1207
    assert pytest.approx(sensor.jtot_crk_max) == 9.75e6
    assert pytest.approx(sensor.j_crk_tot_max) == 9.75e6
    assert pytest.approx(sensor.j_crackle_tot_max) == 9.75e6
    assert pytest.approx(sensor.j_tot_crk_max) == 9.75e6
    assert pytest.approx(sensor.j_crk_max) == 9.75e6
    assert pytest.approx(sensor.jtot_crackle_max) == 9.75e6
    assert pytest.approx(sensor.j_crk_total_max) == 9.75e6
    assert pytest.approx(sensor.j_crackle_total_max) == 9.75e6
    assert pytest.approx(sensor.jtot_shot_max) == 9.75e6
    assert pytest.approx(sensor.jtot_drop_max) == 9.75e6
    assert pytest.approx(sensor.jtot_lock_max) == 9.75e6
    assert pytest.approx(sensor.jtot_pop_max) == 9.75e6
    assert pytest.approx(sensor.jtot_snp_max) == 9.75e6
    assert pytest.approx(sensor.jtot_snap_max) == 9.75e6
    assert pytest.approx(sensor.jtot_rate_max) == 9.75e6
    assert pytest.approx(sensor.jtot_roc_rate_max) == 9.75e6
    assert pytest.approx(sensor.jtot_drop_rate_max) == 9.75e6
    assert pytest.approx(sensor.jtot_crk_rate_max) == 9.75e6
    assert pytest.approx(sensor.t_delay) == 0.0087
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_CRACKLE_RATE"


def test_m453_sensor_spring_total_crackle_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Crackle Rate Free Format Test
/SENSOR/SPRING_TOTAL_CRACKLE_RATE/2
47, 6.0e5, 0.005
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_crackle_rates
    sensor = model.sensor_spring_total_crackle_rates[2]
    assert sensor.spring_id == 47
    assert pytest.approx(sensor.jtot_crk_max) == 6.0e5
    assert pytest.approx(sensor.t_delay) == 0.005
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_CRACKLE_RATE"


def test_m453_sensor_spring_total_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Crackle Rate Aliases Test
/SENSOR/SPRING_TOT_CRACKLE_RATE/3
47, 6.0e5, 0.005
/SENSOR/SPRING_TOTAL_CRK_RATE/4
47, 6.0e5, 0.005
/SENSOR/SPRING_TOT_CRK_RATE/5
47, 6.0e5, 0.005
/SENSOR/SPRING_TOTAL_CRACKLE/6
47, 6.0e5, 0.005
/SENSOR/SPRING_TOT_CRACKLE/7
47, 6.0e5, 0.005
/SENSOR/SPRING_TOTAL_CRK/8
47, 6.0e5, 0.005
/SENSOR/SPRING_TOT_CRK/9
47, 6.0e5, 0.005
/SENSOR/SPRING_RESULTANT_CRACKLE_RATE/10
47, 6.0e5, 0.005
/SENSOR/SPRING_RESULTANT_CRK_RATE/11
47, 6.0e5, 0.005
/SENSOR/SPRING_RES_CRACKLE_RATE/12
47, 6.0e5, 0.005
/SENSOR/SPRING_RES_CRK_RATE/13
47, 6.0e5, 0.005
/SENSOR/SPRING_POP_RATE_TOTAL_CRACKLE/14
47, 6.0e5, 0.005
/SENSOR/SPRING_POP_RATE_TOT_CRACKLE/15
47, 6.0e5, 0.005
/SENSOR/SPRING_POP_RATE_RESULTANT_CRACKLE/16
47, 6.0e5, 0.005
/SENSOR/TOTAL_CRACKLE_RATE_SPRING/17
47, 6.0e5, 0.005
/SENSOR/TOT_CRACKLE_RATE_SPRING/18
47, 6.0e5, 0.005
/SENSOR/RESULTANT_CRACKLE_RATE_SPRING/19
47, 6.0e5, 0.005
/SENSOR/SPRING_CRACKLE_RATE_TOTAL/20
47, 6.0e5, 0.005
/SENSOR/SPRING_CRACKLE_RATE_TOT/21
47, 6.0e5, 0.005
/SENSOR/SPRING_CRACKLE_RATE_RESULTANT/22
47, 6.0e5, 0.005
/SENSOR/SPRING_CRK_RATE_TOTAL/23
47, 6.0e5, 0.005
/SENSOR/SPRING_CRK_RATE_TOT/24
47, 6.0e5, 0.005
/SENSOR/SPRING_CRK_RATE_RESULTANT/25
47, 6.0e5, 0.005
/SENSOR/SPRING_RATE_CRACKLE_TOTAL/26
47, 6.0e5, 0.005
/SENSOR/SPRING_RATE_CRACKLE_TOT/27
47, 6.0e5, 0.005
/SENSOR/SPRING_RATE_CRACKLE_RESULTANT/28
47, 6.0e5, 0.005
/SENSOR/SPRING_CRACKLE_TOTAL/29
47, 6.0e5, 0.005
/SENSOR/SPRING_CRACKLE_TOT/30
47, 6.0e5, 0.005
/SENSOR/SPRING_CRACKLE_RESULTANT/31
47, 6.0e5, 0.005
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(3, 32):
        assert sid in model.sensor_spring_total_crackle_rates
        s = model.sensor_spring_total_crackle_rates[sid]
        assert s.spring_id == 47
        assert pytest.approx(s.jtot_crk_max) == 6.0e5


def test_m453_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_TRANSVERSE_MATRIX_SHEAR_DEGRADATION_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALHOLONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/FIBRATION_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_TOTAL_CRACKLE_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4
