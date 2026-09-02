"""Tests for Milestone M455: LadDynamicFiberTensionRuptureRate Failure Model, EngElectrothermoflexomagnetochiralplasmononplasmonicpolaritonicResonanceEnergy, LagmulConnectionSpinorSpatialLinkageJoint, and SensorSpringTransversePopRate."""

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


def test_m455_fail_lad_dynamic_fiber_tension_rupture_rate_fixed(tmp_path: Path):
    c1 = f"{1110.0:>20.4f}{3330.0:>20.4f}{910.0:>20.4f}{21.0:>20.4f}{0.520:>20.4f}"
    c2 = f"{2:>10d}{2:>10d}"
    c3 = f"{810:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Fiber Tension Rupture Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_FIBER_TENSION_RUPTURE_RATE/2810
Ladeveze Dynamic Fiber Tension Rupture Law
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2810 in model.fail_laddynamicfibertensionrupturerates
    f = model.fail_laddynamicfibertensionrupturerates[2810]
    assert pytest.approx(f.sigma_dftrr0) == 1110.0
    assert pytest.approx(f.sigma_dftrrc) == 3330.0
    assert pytest.approx(f.gamma_dftrr) == 910.0
    assert pytest.approx(f.p_dftrr) == 21.0
    assert pytest.approx(f.d_dftrr_max) == 0.520
    assert f.ifail_sh == 2
    assert f.ifail_so == 2
    assert f.fail_id == 810
    assert pytest.approx(f.sigma_dftr0) == 1110.0
    assert pytest.approx(f.sigma_dftrc) == 3330.0
    assert pytest.approx(f.gamma_dftr) == 910.0
    assert pytest.approx(f.p_dftr) == 21.0
    assert pytest.approx(f.d_dftr_max) == 0.520
    assert pytest.approx(f.sigma_dft0) == 1110.0
    assert pytest.approx(f.sigma_dftc) == 3330.0
    assert pytest.approx(f.gamma_dft) == 910.0
    assert pytest.approx(f.p_dft) == 21.0
    assert pytest.approx(f.d_dft_max) == 0.520
    assert pytest.approx(f.sigma_dfr0) == 1110.0
    assert pytest.approx(f.sigma_dfrc) == 3330.0
    assert pytest.approx(f.gamma_dfr) == 910.0
    assert pytest.approx(f.p_dfr) == 21.0
    assert pytest.approx(f.d_dfr_max) == 0.520


def test_m455_fail_lad_dynamic_fiber_tension_rupture_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Fiber Tension Rupture Rate Free Format Test
/FAIL/LAD_DYNAMIC_FIBER_TENSION_RUPTURE_RATE/2811
1210.0, 3630.0, 1010.0, 23.0, 0.620
1, 1
910
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2811 in model.fail_laddynamicfibertensionrupturerates
    f = model.fail_laddynamicfibertensionrupturerates[2811]
    assert pytest.approx(f.sigma_dftrr0) == 1210.0
    assert pytest.approx(f.sigma_dftrrc) == 3630.0
    assert pytest.approx(f.gamma_dftrr) == 1010.0
    assert pytest.approx(f.p_dftrr) == 23.0
    assert pytest.approx(f.d_dftrr_max) == 0.620
    assert f.ifail_sh == 1
    assert f.ifail_so == 1
    assert f.fail_id == 910


def test_m455_fail_lad_dynamic_fiber_tension_rupture_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Fiber Tension Rupture Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_FIBER_TENSION_RUPTURE_RATE/2812
1110.0, 3330.0, 910.0, 21.0, 0.520
1, 1
/FAIL/LAD_DYNAMIC_FIBER_TENSION_RUPTURE/2813
1110.0, 3330.0, 910.0, 21.0, 0.520
1, 1
/FAIL/LADEVEZE_DYNAMIC_FIBER_TENSION_RUPTURE/2814
1110.0, 3330.0, 910.0, 21.0, 0.520
1, 1
/FAIL/LAD_DYNAMIC_FIBER_TENSION_RATE/2815
1110.0, 3330.0, 910.0, 21.0, 0.520
1, 1
/FAIL/LADEVEZE_DYNAMIC_FIBER_TENSION_RATE/2816
1110.0, 3330.0, 910.0, 21.0, 0.520
1, 1
/FAIL/LAD_DYNAMIC_FIBER_RUPTURE_RATE/2817
1110.0, 3330.0, 910.0, 21.0, 0.520
1, 1
/FAIL/LADEVEZE_DYNAMIC_FIBER_RUPTURE_RATE/2818
1110.0, 3330.0, 910.0, 21.0, 0.520
1, 1
/FAIL/LAD_DYNAMIC_FIBER_TENSILE_RUPTURE_RATE/2819
1110.0, 3330.0, 910.0, 21.0, 0.520
1, 1
/FAIL/LADEVEZE_DYNAMIC_FIBER_TENSILE_RUPTURE_RATE/2820
1110.0, 3330.0, 910.0, 21.0, 0.520
1, 1
/FAIL/LAD_DFTRR/2821
1110.0, 3330.0, 910.0, 21.0, 0.520
1, 1
/FAIL/LAD_DFTRR_MODEL/2822
1110.0, 3330.0, 910.0, 21.0, 0.520
1, 1
/FAIL/LAD_DFTRR_LAW/2823
1110.0, 3330.0, 910.0, 21.0, 0.520
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_FIBER_TENSION_RUPTURE/2824
1110.0, 3330.0, 910.0, 21.0, 0.520
1, 1
/FAIL/LAD_DYNAMIC_FIBER_TENSION_DAMAGE_RATE/2825
1110.0, 3330.0, 910.0, 21.0, 0.520
1, 1
/FAIL/LADEVEZE_DYNAMIC_FIBER_TENSION_DAMAGE_RATE/2826
1110.0, 3330.0, 910.0, 21.0, 0.520
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for mid in range(2812, 2827):
        assert mid in model.fail_laddynamicfibertensionrupturerates
        f = model.fail_laddynamicfibertensionrupturerates[mid]
        assert pytest.approx(f.sigma_dftrr0) == 1110.0
        assert pytest.approx(f.sigma_dftrrc) == 3330.0
        assert pytest.approx(f.gamma_dftrr) == 910.0
        assert pytest.approx(f.p_dftrr) == 21.0
        assert pytest.approx(f.d_dftrr_max) == 0.520


def test_m455_eng_electrothermoflexomagnetochiralplasmononplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00515:>20.6e}{89:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetochiralplasmononplasmonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPLASMONONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Engine Electrothermoflexomagnetochiralplasmononplasmonicpolaritonic Resonance Energy Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralplasmononplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetochiralplasmononplasmonicpolaritonic_resonance_energies[1]
    assert pytest.approx(r.dt_etfcplasmononplp) == 0.00515
    assert r.sens_id == 89


def test_m455_eng_electrothermoflexomagnetochiralplasmononplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetochiralplasmononplasmonicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPLASMONONPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
0.00615, 99
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralplasmononplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetochiralplasmononplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(r.dt_etfcplasmononplp) == 0.00615
    assert r.sens_id == 99


def test_m455_eng_electrothermoflexomagnetochiralplasmononplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetochiralplasmononplasmonicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_PLASMONON_PLASMON_POLARITON_RES_WORK/3
0.00515, 89
/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALPLASMONONPLASMONICPOLARITONICRESONANCE/4
0.00515, 89
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPLASMONONPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/5
0.00515, 89
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALPLASMONONPLASMONICPOLARITONIC_RESONANCE/6
0.00515, 89
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONONCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY/7
0.00515, 89
/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMONON_CHIRAL_PLASMON_POLARITON_RES_WORK/8
0.00515, 89
/ENG/EELECTROTHERMOFLEXOMAGNETOPLASMONONCHIRALPLASMONICPOLARITONICRESONANCE/9
0.00515, 89
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONONCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/10
0.00515, 89
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPLASMONONCHIRALPLASMONICPOLARITONIC_RESONANCE/11
0.00515, 89
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for rid in range(3, 12):
        assert rid in model.eng_electrothermoflexomagnetochiralplasmononplasmonicpolaritonic_resonance_energies
        r = model.eng_electrothermoflexomagnetochiralplasmononplasmonicpolaritonic_resonance_energies[rid]
        assert pytest.approx(r.dt_etfcplasmononplp) == 0.00515
        assert r.sens_id == 89


def test_m455_lagmul_connection_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{4.3e6:>20.4f}{12:>10d}{3.0e-5:>20.6e}"
    c2 = f"{28.0:>20.4f}{30.0:>20.4f}{82.0:>20.4f}{14.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Connection Spinor Spatial Linkage Joint Fixed Format Test
2022 0
/CONNECTION_SPINOR_SPATIAL_LINKAGE_JOINT/1
Connection Spinor Spatial Mechanism Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_connection_spinor_spatial_linkage_joints
    j = model.lagmul_connection_spinor_spatial_linkage_joints[1]
    assert j.node1 == 101
    assert j.node2 == 102
    assert j.node3 == 103
    assert pytest.approx(j.stiff) == 4.3e6
    assert j.skew_id == 12
    assert pytest.approx(j.tol) == 3.0e-5
    assert pytest.approx(j.link_len_a) == 28.0
    assert pytest.approx(j.link_len_b) == 30.0
    assert pytest.approx(j.twist_angle_alpha) == 82.0
    assert pytest.approx(j.offset_distance_s) == 14.0
    assert pytest.approx(j.offset_distance_r) == 14.0
    assert pytest.approx(j.offset_distance_v) == 14.0
    assert pytest.approx(j.offset_distance_h) == 14.0
    assert pytest.approx(j.offset_distance_u) == 14.0
    assert pytest.approx(j.offset_distance_f) == 14.0


def test_m455_lagmul_connection_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Connection Spinor Spatial Linkage Joint Free Format Test
/LAGMUL/CONNECTION_SPINOR_SPATIAL_LINKAGE_JOINT/2
Connection Spinor Spatial Mechanism Joint Free
201, 202, 203, 5.3e6, 13, 4.0e-5
38.0, 40.0, 92.0, 19.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_connection_spinor_spatial_linkage_joints
    j = model.lagmul_connection_spinor_spatial_linkage_joints[2]
    assert j.node1 == 201
    assert j.node2 == 202
    assert j.node3 == 203
    assert pytest.approx(j.stiff) == 5.3e6
    assert j.skew_id == 13
    assert pytest.approx(j.tol) == 4.0e-5
    assert pytest.approx(j.link_len_a) == 38.0
    assert pytest.approx(j.link_len_b) == 40.0
    assert pytest.approx(j.twist_angle_alpha) == 92.0
    assert pytest.approx(j.offset_distance_s) == 19.0


def test_m455_lagmul_connection_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Connection Spinor Spatial Linkage Joint Aliases Test
/CONNECTION_SPINOR_SPATIAL_LINKAGE/3
101, 102, 103, 4.3e6, 12, 3.0e-5
28.0, 30.0, 82.0, 14.0
/LAGMUL/CONNECTION_SPINOR_SPATIAL_LINKAGE/4
101, 102, 103, 4.3e6, 12, 3.0e-5
28.0, 30.0, 82.0, 14.0
/CONNECTION_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM/5
101, 102, 103, 4.3e6, 12, 3.0e-5
28.0, 30.0, 82.0, 14.0
/CONNECTION_SPINOR_SPATIAL_SYMMETRIC_MECHANISM/6
101, 102, 103, 4.3e6, 12, 3.0e-5
28.0, 30.0, 82.0, 14.0
/CONNECTION_SPINOR_SPATIAL_6R_MECHANISM/7
101, 102, 103, 4.3e6, 12, 3.0e-5
28.0, 30.0, 82.0, 14.0
/CONNECTION_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM/8
101, 102, 103, 4.3e6, 12, 3.0e-5
28.0, 30.0, 82.0, 14.0
/LAGMUL/CONNECTION_TWISTOR_SPATIAL_LINKAGE_JOINT/9
101, 102, 103, 4.3e6, 12, 3.0e-5
28.0, 30.0, 82.0, 14.0
/CONNECTION_TWISTOR_SPATIAL_LINKAGE_JOINT/10
101, 102, 103, 4.3e6, 12, 3.0e-5
28.0, 30.0, 82.0, 14.0
/LAGMUL/CONNECTION_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/11
101, 102, 103, 4.3e6, 12, 3.0e-5
28.0, 30.0, 82.0, 14.0
/CONNECTION_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/12
101, 102, 103, 4.3e6, 12, 3.0e-5
28.0, 30.0, 82.0, 14.0
/LAGMUL/CONNECTION_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/13
101, 102, 103, 4.3e6, 12, 3.0e-5
28.0, 30.0, 82.0, 14.0
/CONNECTION_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/14
101, 102, 103, 4.3e6, 12, 3.0e-5
28.0, 30.0, 82.0, 14.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(3, 15):
        assert jid in model.lagmul_connection_spinor_spatial_linkage_joints
        j = model.lagmul_connection_spinor_spatial_linkage_joints[jid]
        assert j.node1 == 101
        assert j.node2 == 102
        assert j.node3 == 103
        assert pytest.approx(j.stiff) == 4.3e6


def test_m455_sensor_spring_transverse_pop_rate_fixed(tmp_path: Path):
    c1 = f"{1209:>10d}{9.95e6:>20.4f}{0.0089:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Transverse Pop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_POP_RATE/1
Spring Transverse Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_transverse_pop_rates
    sensor = model.sensor_spring_transverse_pop_rates[1]
    assert sensor.spring_id == 1209
    assert pytest.approx(sensor.jtrans_pop_max) == 9.95e6
    assert pytest.approx(sensor.j_pop_trans_max) == 9.95e6
    assert pytest.approx(sensor.j_pop_transverse_max) == 9.95e6
    assert pytest.approx(sensor.j_trans_pop_max) == 9.95e6
    assert pytest.approx(sensor.j_pop_max) == 9.95e6
    assert pytest.approx(sensor.j_pop_shear_max) == 9.95e6
    assert pytest.approx(sensor.jtrans_shot_max) == 9.95e6
    assert pytest.approx(sensor.jtrans_drop_max) == 9.95e6
    assert pytest.approx(sensor.jtrans_lock_max) == 9.95e6
    assert pytest.approx(sensor.jtrans_snp_max) == 9.95e6
    assert pytest.approx(sensor.jtrans_snap_max) == 9.95e6
    assert pytest.approx(sensor.jtrans_rate_max) == 9.95e6
    assert pytest.approx(sensor.jtrans_roc_rate_max) == 9.95e6
    assert pytest.approx(sensor.jtrans_drop_rate_max) == 9.95e6
    assert pytest.approx(sensor.jtrans_crk_rate_max) == 9.95e6
    assert pytest.approx(sensor.jtrans_crackle_max) == 9.95e6
    assert pytest.approx(sensor.jtrans_crk_max) == 9.95e6
    assert pytest.approx(sensor.jtrans_pop_rate_max) == 9.95e6
    assert pytest.approx(sensor.t_delay) == 0.0089
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_POP_RATE"


def test_m455_sensor_spring_transverse_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Transverse Pop Rate Free Format Test
/SENSOR/SPRING_TRANSVERSE_POP_RATE/2
49, 6.2e5, 0.007
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_transverse_pop_rates
    sensor = model.sensor_spring_transverse_pop_rates[2]
    assert sensor.spring_id == 49
    assert pytest.approx(sensor.jtrans_pop_max) == 6.2e5
    assert pytest.approx(sensor.t_delay) == 0.007
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_POP_RATE"


def test_m455_sensor_spring_transverse_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Transverse Pop Rate Aliases Test
/SENSOR/SPRING_TRANS_POP_RATE/3
49, 6.2e5, 0.007
/SENSOR/SPRING_TRANSVERSE_POP/4
49, 6.2e5, 0.007
/SENSOR/SPRING_TRANS_POP/5
49, 6.2e5, 0.007
/SENSOR/SPRING_SHEAR_POP_RATE/6
49, 6.2e5, 0.007
/SENSOR/SPRING_SHEAR_POP/7
49, 6.2e5, 0.007
/SENSOR/SPRING_POP_RATE_TRANSVERSE/8
49, 6.2e5, 0.007
/SENSOR/SPRING_POP_RATE_TRANS/9
49, 6.2e5, 0.007
/SENSOR/SPRING_POP_RATE_SHEAR/10
49, 6.2e5, 0.007
/SENSOR/TRANSVERSE_POP_RATE_SPRING/11
49, 6.2e5, 0.007
/SENSOR/TRANS_POP_RATE_SPRING/12
49, 6.2e5, 0.007
/SENSOR/SHEAR_POP_RATE_SPRING/13
49, 6.2e5, 0.007
/SENSOR/SPRING_POP_TRANSVERSE/14
49, 6.2e5, 0.007
/SENSOR/SPRING_POP_TRANS/15
49, 6.2e5, 0.007
/SENSOR/SPRING_POP_SHEAR/16
49, 6.2e5, 0.007
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(3, 17):
        assert sid in model.sensor_spring_transverse_pop_rates
        s = model.sensor_spring_transverse_pop_rates[sid]
        assert s.spring_id == 49
        assert pytest.approx(s.jtrans_pop_max) == 6.2e5


def test_m455_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_DYNAMIC_FIBER_TENSION_RUPTURE_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPLASMONONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/CONNECTION_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_TRANSVERSE_POP_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4
