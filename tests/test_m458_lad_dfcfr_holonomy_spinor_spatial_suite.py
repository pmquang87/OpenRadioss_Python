"""Tests for Milestone M458: LadDynamicFiberCompressionFailureRate Failure Model, EngElectrothermoflexomagnetochiralaxionicplasmonicpolaritonicResonanceEnergy, LagmulHolonomySpinorSpatialLinkageJoint, and SensorSpringTotalAngularPopRate."""

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


def test_m458_fail_lad_dynamic_fiber_compression_failure_rate_fixed(tmp_path: Path):
    c1 = f"{1420.0:>20.4f}{4250.0:>20.4f}{1120.0:>20.4f}{28.0:>20.4f}{0.620:>20.4f}"
    c2 = f"{2:>10d}{2:>10d}"
    c3 = f"{920:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Fiber Compression Failure Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_FIBER_COMPRESSION_FAILURE_RATE/2950
Ladeveze Dynamic Fiber Compression Failure Law
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2950 in model.fail_laddynamicfibercompressionfailurerates
    f = model.fail_laddynamicfibercompressionfailurerates[2950]
    assert pytest.approx(f.sigma_dfcfr0) == 1420.0
    assert pytest.approx(f.sigma_dfcfrc) == 4250.0
    assert pytest.approx(f.gamma_dfcfr) == 1120.0
    assert pytest.approx(f.p_dfcfr) == 28.0
    assert pytest.approx(f.d_dfcfr_max) == 0.620
    assert f.ifail_sh == 2
    assert f.ifail_so == 2
    assert f.fail_id == 920
    assert pytest.approx(f.sigma_dfcf0) == 1420.0
    assert pytest.approx(f.sigma_dfcfc) == 4250.0
    assert pytest.approx(f.gamma_dfcf) == 1120.0
    assert pytest.approx(f.p_dfcf) == 28.0
    assert pytest.approx(f.d_dfcf_max) == 0.620
    assert pytest.approx(f.sigma_dfc0) == 1420.0
    assert pytest.approx(f.sigma_dfcc) == 4250.0
    assert pytest.approx(f.gamma_dfc) == 1120.0
    assert pytest.approx(f.p_dfc) == 28.0
    assert pytest.approx(f.d_dfc_max) == 0.620
    assert pytest.approx(f.sigma_dfr0) == 1420.0
    assert pytest.approx(f.sigma_dfrc) == 4250.0
    assert pytest.approx(f.gamma_dfr) == 1120.0
    assert pytest.approx(f.p_dfr) == 28.0
    assert pytest.approx(f.d_dfr_max) == 0.620


def test_m458_fail_lad_dynamic_fiber_compression_failure_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Fiber Compression Failure Rate Free Format Test
/FAIL/LAD_DYNAMIC_FIBER_COMPRESSION_FAILURE_RATE/2951
1520.0, 4550.0, 1220.0, 30.0, 0.720
1, 1
1020
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2951 in model.fail_laddynamicfibercompressionfailurerates
    f = model.fail_laddynamicfibercompressionfailurerates[2951]
    assert pytest.approx(f.sigma_dfcfr0) == 1520.0
    assert pytest.approx(f.sigma_dfcfrc) == 4550.0
    assert pytest.approx(f.gamma_dfcfr) == 1220.0
    assert pytest.approx(f.p_dfcfr) == 30.0
    assert pytest.approx(f.d_dfcfr_max) == 0.720
    assert f.ifail_sh == 1
    assert f.ifail_so == 1
    assert f.fail_id == 1020


def test_m458_fail_lad_dynamic_fiber_compression_failure_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Fiber Compression Failure Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_FIBER_COMPRESSION_FAILURE_RATE/2952
1420.0, 4250.0, 1120.0, 28.0, 0.620
1, 1
/FAIL/LAD_DYNAMIC_FIBER_COMPRESSION_FAILURE/2953
1420.0, 4250.0, 1120.0, 28.0, 0.620
1, 1
/FAIL/LADEVEZE_DYNAMIC_FIBER_COMPRESSION_FAILURE/2954
1420.0, 4250.0, 1120.0, 28.0, 0.620
1, 1
/FAIL/LAD_DYNAMIC_FIBER_COMPRESSION_RATE/2955
1420.0, 4250.0, 1120.0, 28.0, 0.620
1, 1
/FAIL/LADEVEZE_DYNAMIC_FIBER_COMPRESSION_RATE/2956
1420.0, 4250.0, 1120.0, 28.0, 0.620
1, 1
/FAIL/LAD_DYNAMIC_FIBER_COMPRESSIVE_FAILURE_RATE/2957
1420.0, 4250.0, 1120.0, 28.0, 0.620
1, 1
/FAIL/LADEVEZE_DYNAMIC_FIBER_COMPRESSIVE_FAILURE_RATE/2958
1420.0, 4250.0, 1120.0, 28.0, 0.620
1, 1
/FAIL/LAD_DYNAMIC_FIBER_COMPRESSIVE_FAILURE/2959
1420.0, 4250.0, 1120.0, 28.0, 0.620
1, 1
/FAIL/LADEVEZE_DYNAMIC_FIBER_COMPRESSIVE_FAILURE/2960
1420.0, 4250.0, 1120.0, 28.0, 0.620
1, 1
/FAIL/LAD_DFCFR/2961
1420.0, 4250.0, 1120.0, 28.0, 0.620
1, 1
/FAIL/LAD_DFCFR_MODEL/2962
1420.0, 4250.0, 1120.0, 28.0, 0.620
1, 1
/FAIL/LAD_DFCFR_LAW/2963
1420.0, 4250.0, 1120.0, 28.0, 0.620
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_FIBER_COMPRESSION_FAILURE/2964
1420.0, 4250.0, 1120.0, 28.0, 0.620
1, 1
/FAIL/LAD_DYNAMIC_FIBER_COMPRESSION_DAMAGE_RATE/2965
1420.0, 4250.0, 1120.0, 28.0, 0.620
1, 1
/FAIL/LADEVEZE_DYNAMIC_FIBER_COMPRESSION_DAMAGE_RATE/2966
1420.0, 4250.0, 1120.0, 28.0, 0.620
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for mid in range(2952, 2967):
        assert mid in model.fail_laddynamicfibercompressionfailurerates
        f = model.fail_laddynamicfibercompressionfailurerates[mid]
        assert pytest.approx(f.sigma_dfcfr0) == 1420.0
        assert pytest.approx(f.sigma_dfcfrc) == 4250.0
        assert pytest.approx(f.gamma_dfcfr) == 1120.0
        assert pytest.approx(f.p_dfcfr) == 28.0
        assert pytest.approx(f.d_dfcfr_max) == 0.620


def test_m458_eng_electrothermoflexomagnetochiralaxionicplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00528:>20.6e}{92:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetochiralaxionicplasmonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALAXIONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Engine Electrothermoflexomagnetochiralaxionicplasmonicpolaritonic Resonance Energy Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralaxionicplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetochiralaxionicplasmonicpolaritonic_resonance_energies[1]
    assert pytest.approx(r.dt_etfcaxionicplp) == 0.00528
    assert r.sens_id == 92


def test_m458_eng_electrothermoflexomagnetochiralaxionicplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetochiralaxionicplasmonicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALAXIONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
0.00628, 102
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralaxionicplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetochiralaxionicplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(r.dt_etfcaxionicplp) == 0.00628
    assert r.sens_id == 102


def test_m458_eng_electrothermoflexomagnetochiralaxionicplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetochiralaxionicplasmonicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_AXIONIC_PLASMON_POLARITON_RES_WORK/3
0.00528, 92
/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALAXIONICPLASMONICPOLARITONICRESONANCE/4
0.00528, 92
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALAXIONICPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/5
0.00528, 92
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALAXIONICPLASMONICPOLARITONIC_RESONANCE/6
0.00528, 92
/ENG/ELECTROTHERMOFLEXOMAGNETOAXIONICCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY/7
0.00528, 92
/ENG/ELECTRO_THERM_FLEXO_MAG_AXIONIC_CHIRAL_PLASMON_POLARITON_RES_WORK/8
0.00528, 92
/ENG/EELECTROTHERMOFLEXOMAGNETOAXIONICCHIRALPLASMONICPOLARITONICRESONANCE/9
0.00528, 92
/ENG/ELECTROTHERMOFLEXOMAGNETOAXIONICCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/10
0.00528, 92
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOAXIONICCHIRALPLASMONICPOLARITONIC_RESONANCE/11
0.00528, 92
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for rid in range(3, 12):
        assert rid in model.eng_electrothermoflexomagnetochiralaxionicplasmonicpolaritonic_resonance_energies
        r = model.eng_electrothermoflexomagnetochiralaxionicplasmonicpolaritonic_resonance_energies[rid]
        assert pytest.approx(r.dt_etfcaxionicplp) == 0.00528
        assert r.sens_id == 92


def test_m458_lagmul_holonomy_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{4.6e6:>20.4f}{13:>10d}{3.3e-5:>20.6e}"
    c2 = f"{31.0:>20.4f}{33.0:>20.4f}{85.0:>20.4f}{17.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Holonomy Spinor Spatial Linkage Joint Fixed Format Test
2022 0
/HOLONOMY_SPINOR_SPATIAL_LINKAGE_JOINT/1
Holonomy Spinor Spatial Mechanism Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_holonomy_spinor_spatial_linkage_joints
    j = model.lagmul_holonomy_spinor_spatial_linkage_joints[1]
    assert j.node1 == 101
    assert j.node2 == 102
    assert j.node3 == 103
    assert pytest.approx(j.stiff) == 4.6e6
    assert j.skew_id == 13
    assert pytest.approx(j.tol) == 3.3e-5
    assert pytest.approx(j.link_len_a) == 31.0
    assert pytest.approx(j.link_len_b) == 33.0
    assert pytest.approx(j.twist_angle_alpha) == 85.0
    assert pytest.approx(j.offset_distance_s) == 17.0
    assert pytest.approx(j.offset_distance_r) == 17.0
    assert pytest.approx(j.offset_distance_v) == 17.0
    assert pytest.approx(j.offset_distance_h) == 17.0
    assert pytest.approx(j.offset_distance_u) == 17.0
    assert pytest.approx(j.offset_distance_f) == 17.0


def test_m458_lagmul_holonomy_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Holonomy Spinor Spatial Linkage Joint Free Format Test
/LAGMUL/HOLONOMY_SPINOR_SPATIAL_LINKAGE_JOINT/2
Holonomy Spinor Spatial Mechanism Joint Free
201, 202, 203, 5.6e6, 14, 4.3e-5
41.0, 43.0, 95.0, 22.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_holonomy_spinor_spatial_linkage_joints
    j = model.lagmul_holonomy_spinor_spatial_linkage_joints[2]
    assert j.node1 == 201
    assert j.node2 == 202
    assert j.node3 == 203
    assert pytest.approx(j.stiff) == 5.6e6
    assert j.skew_id == 14
    assert pytest.approx(j.tol) == 4.3e-5
    assert pytest.approx(j.link_len_a) == 41.0
    assert pytest.approx(j.link_len_b) == 43.0
    assert pytest.approx(j.twist_angle_alpha) == 95.0
    assert pytest.approx(j.offset_distance_s) == 22.0


def test_m458_lagmul_holonomy_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Holonomy Spinor Spatial Linkage Joint Aliases Test
/HOLONOMY_SPINOR_SPATIAL_LINKAGE/3
101, 102, 103, 4.6e6, 13, 3.3e-5
31.0, 33.0, 85.0, 17.0
/LAGMUL/HOLONOMY_SPINOR_SPATIAL_LINKAGE/4
101, 102, 103, 4.6e6, 13, 3.3e-5
31.0, 33.0, 85.0, 17.0
/HOLONOMY_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM/5
101, 102, 103, 4.6e6, 13, 3.3e-5
31.0, 33.0, 85.0, 17.0
/HOLONOMY_SPINOR_SPATIAL_SYMMETRIC_MECHANISM/6
101, 102, 103, 4.6e6, 13, 3.3e-5
31.0, 33.0, 85.0, 17.0
/HOLONOMY_SPINOR_SPATIAL_6R_MECHANISM/7
101, 102, 103, 4.6e6, 13, 3.3e-5
31.0, 33.0, 85.0, 17.0
/HOLONOMY_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM/8
101, 102, 103, 4.6e6, 13, 3.3e-5
31.0, 33.0, 85.0, 17.0
/LAGMUL/HOLONOMY_TWISTOR_SPATIAL_LINKAGE_JOINT/9
101, 102, 103, 4.6e6, 13, 3.3e-5
31.0, 33.0, 85.0, 17.0
/HOLONOMY_TWISTOR_SPATIAL_LINKAGE_JOINT/10
101, 102, 103, 4.6e6, 13, 3.3e-5
31.0, 33.0, 85.0, 17.0
/LAGMUL/HOLONOMY_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/11
101, 102, 103, 4.6e6, 13, 3.3e-5
31.0, 33.0, 85.0, 17.0
/HOLONOMY_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/12
101, 102, 103, 4.6e6, 13, 3.3e-5
31.0, 33.0, 85.0, 17.0
/LAGMUL/HOLONOMY_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/13
101, 102, 103, 4.6e6, 13, 3.3e-5
31.0, 33.0, 85.0, 17.0
/HOLONOMY_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/14
101, 102, 103, 4.6e6, 13, 3.3e-5
31.0, 33.0, 85.0, 17.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(3, 15):
        assert jid in model.lagmul_holonomy_spinor_spatial_linkage_joints
        j = model.lagmul_holonomy_spinor_spatial_linkage_joints[jid]
        assert j.node1 == 101
        assert j.node2 == 102
        assert j.node3 == 103
        assert pytest.approx(j.stiff) == 4.6e6


def test_m458_sensor_spring_total_angular_pop_rate_fixed(tmp_path: Path):
    c1 = f"{1212:>10d}{9.98e6:>20.4f}{0.0092:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Angular Pop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE/1
Spring Total Angular Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_angular_pop_rates
    sensor = model.sensor_spring_total_angular_pop_rates[1]
    assert sensor.spring_id == 1212
    assert pytest.approx(sensor.jtot_ang_pop_max) == 9.98e6
    assert pytest.approx(sensor.j_pop_tot_ang_max) == 9.98e6
    assert pytest.approx(sensor.j_pop_total_angular_max) == 9.98e6
    assert pytest.approx(sensor.j_tot_ang_pop_max) == 9.98e6
    assert pytest.approx(sensor.j_pop_max) == 9.98e6
    assert pytest.approx(sensor.j_pop_res_ang_max) == 9.98e6
    assert pytest.approx(sensor.jtot_ang_shot_max) == 9.98e6
    assert pytest.approx(sensor.jtot_ang_drop_max) == 9.98e6
    assert pytest.approx(sensor.jtot_ang_lock_max) == 9.98e6
    assert pytest.approx(sensor.jtot_ang_snp_max) == 9.98e6
    assert pytest.approx(sensor.jtot_ang_snap_max) == 9.98e6
    assert pytest.approx(sensor.jtot_ang_rate_max) == 9.98e6
    assert pytest.approx(sensor.jtot_ang_roc_rate_max) == 9.98e6
    assert pytest.approx(sensor.jtot_ang_drop_rate_max) == 9.98e6
    assert pytest.approx(sensor.jtot_ang_crk_rate_max) == 9.98e6
    assert pytest.approx(sensor.jtot_ang_crackle_max) == 9.98e6
    assert pytest.approx(sensor.jtot_ang_crk_max) == 9.98e6
    assert pytest.approx(sensor.jtot_ang_pop_rate_max) == 9.98e6
    assert pytest.approx(sensor.t_delay) == 0.0092
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_POP_RATE"


def test_m458_sensor_spring_total_angular_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Angular Pop Rate Free Format Test
/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE/2
52, 6.5e5, 0.0095
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_angular_pop_rates
    sensor = model.sensor_spring_total_angular_pop_rates[2]
    assert sensor.spring_id == 52
    assert pytest.approx(sensor.jtot_ang_pop_max) == 6.5e5
    assert pytest.approx(sensor.t_delay) == 0.0095
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_POP_RATE"


def test_m458_sensor_spring_total_angular_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Angular Pop Rate Aliases Test
/SENSOR/SPRING_TOT_ANG_POP_RATE/3
52, 6.5e5, 0.0095
/SENSOR/SPRING_TOTAL_ANGULAR_POP/4
52, 6.5e5, 0.0095
/SENSOR/SPRING_TOT_ANG_POP/5
52, 6.5e5, 0.0095
/SENSOR/SPRING_RESULTANT_ANGULAR_POP_RATE/6
52, 6.5e5, 0.0095
/SENSOR/SPRING_RESULTANT_ANGULAR_POP/7
52, 6.5e5, 0.0095
/SENSOR/SPRING_POP_RATE_TOTAL_ANGULAR/8
52, 6.5e5, 0.0095
/SENSOR/SPRING_POP_RATE_TOT_ANG/9
52, 6.5e5, 0.0095
/SENSOR/SPRING_POP_RATE_RESULTANT_ANGULAR/10
52, 6.5e5, 0.0095
/SENSOR/TOTAL_ANGULAR_POP_RATE_SPRING/11
52, 6.5e5, 0.0095
/SENSOR/TOT_ANG_POP_RATE_SPRING/12
52, 6.5e5, 0.0095
/SENSOR/RESULTANT_ANGULAR_POP_RATE_SPRING/13
52, 6.5e5, 0.0095
/SENSOR/SPRING_POP_TOTAL_ANGULAR/14
52, 6.5e5, 0.0095
/SENSOR/SPRING_POP_TOT_ANG/15
52, 6.5e5, 0.0095
/SENSOR/SPRING_POP_RESULTANT_ANGULAR/16
52, 6.5e5, 0.0095
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(3, 17):
        assert sid in model.sensor_spring_total_angular_pop_rates
        s = model.sensor_spring_total_angular_pop_rates[sid]
        assert s.spring_id == 52
        assert pytest.approx(s.jtot_ang_pop_max) == 6.5e5


def test_m458_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_DYNAMIC_FIBER_COMPRESSION_FAILURE_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALAXIONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/HOLONOMY_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4
