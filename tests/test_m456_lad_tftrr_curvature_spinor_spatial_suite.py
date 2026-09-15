"""Tests for Milestone M456: LadTransverseFiberTensionRuptureRate Failure Model, EngElectrothermoflexomagnetochiralparamagnonplasmonicpolaritonicResonanceEnergy, LagmulCurvatureSpinorSpatialLinkageJoint, and SensorSpringCoupledPopRate."""

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


def test_m456_fail_lad_transverse_fiber_tension_rupture_rate_fixed(tmp_path: Path):
    c1 = f"{1120.0:>20.4f}{3340.0:>20.4f}{920.0:>20.4f}{22.0:>20.4f}{0.530:>20.4f}"
    c2 = f"{2:>10d}{2:>10d}"
    c3 = f"{820:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Fiber Tension Rupture Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_FIBER_TENSION_RUPTURE_RATE/2830
Ladeveze Transverse Fiber Tension Rupture Law
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2830 in model.fail_ladtransversefibertensionrupturerates
    f = model.fail_ladtransversefibertensionrupturerates[2830]
    assert pytest.approx(f.sigma_tftrr0) == 1120.0
    assert pytest.approx(f.sigma_tftrrc) == 3340.0
    assert pytest.approx(f.gamma_tftrr) == 920.0
    assert pytest.approx(f.p_tftrr) == 22.0
    assert pytest.approx(f.d_tftrr_max) == 0.530
    assert f.ifail_sh == 2
    assert f.ifail_so == 2
    assert f.fail_id == 820
    assert pytest.approx(f.sigma_tftr0) == 1120.0
    assert pytest.approx(f.sigma_tftrc) == 3340.0
    assert pytest.approx(f.gamma_tftr) == 920.0
    assert pytest.approx(f.p_tftr) == 22.0
    assert pytest.approx(f.d_tftr_max) == 0.530
    assert pytest.approx(f.sigma_tft0) == 1120.0
    assert pytest.approx(f.sigma_tftc) == 3340.0
    assert pytest.approx(f.gamma_tft) == 920.0
    assert pytest.approx(f.p_tft) == 22.0
    assert pytest.approx(f.d_tft_max) == 0.530
    assert pytest.approx(f.sigma_tfr0) == 1120.0
    assert pytest.approx(f.sigma_tfrc) == 3340.0
    assert pytest.approx(f.gamma_tfr) == 920.0
    assert pytest.approx(f.p_tfr) == 22.0
    assert pytest.approx(f.d_tfr_max) == 0.530


def test_m456_fail_lad_transverse_fiber_tension_rupture_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Fiber Tension Rupture Rate Free Format Test
/FAIL/LAD_TRANSVERSE_FIBER_TENSION_RUPTURE_RATE/2831
1220.0, 3640.0, 1020.0, 24.0, 0.630
1, 1
920
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2831 in model.fail_ladtransversefibertensionrupturerates
    f = model.fail_ladtransversefibertensionrupturerates[2831]
    assert pytest.approx(f.sigma_tftrr0) == 1220.0
    assert pytest.approx(f.sigma_tftrrc) == 3640.0
    assert pytest.approx(f.gamma_tftrr) == 1020.0
    assert pytest.approx(f.p_tftrr) == 24.0
    assert pytest.approx(f.d_tftrr_max) == 0.630
    assert f.ifail_sh == 1
    assert f.ifail_so == 1
    assert f.fail_id == 920


def test_m456_fail_lad_transverse_fiber_tension_rupture_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Fiber Tension Rupture Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_FIBER_TENSION_RUPTURE_RATE/2832
1120.0, 3340.0, 920.0, 22.0, 0.530
1, 1
/FAIL/LAD_TRANSVERSE_FIBER_TENSION_RUPTURE/2833
1120.0, 3340.0, 920.0, 22.0, 0.530
1, 1
/FAIL/LADEVEZE_TRANSVERSE_FIBER_TENSION_RUPTURE/2834
1120.0, 3340.0, 920.0, 22.0, 0.530
1, 1
/FAIL/LAD_TRANSVERSE_FIBER_TENSION_RATE/2835
1120.0, 3340.0, 920.0, 22.0, 0.530
1, 1
/FAIL/LADEVEZE_TRANSVERSE_FIBER_TENSION_RATE/2836
1120.0, 3340.0, 920.0, 22.0, 0.530
1, 1
/FAIL/LAD_TRANSVERSE_FIBER_RUPTURE_RATE/2837
1120.0, 3340.0, 920.0, 22.0, 0.530
1, 1
/FAIL/LADEVEZE_TRANSVERSE_FIBER_RUPTURE_RATE/2838
1120.0, 3340.0, 920.0, 22.0, 0.530
1, 1
/FAIL/LAD_TRANSVERSE_FIBER_TENSILE_RUPTURE_RATE/2839
1120.0, 3340.0, 920.0, 22.0, 0.530
1, 1
/FAIL/LADEVEZE_TRANSVERSE_FIBER_TENSILE_RUPTURE_RATE/2840
1120.0, 3340.0, 920.0, 22.0, 0.530
1, 1
/FAIL/LAD_TFTRR/2841
1120.0, 3340.0, 920.0, 22.0, 0.530
1, 1
/FAIL/LAD_TFTRR_MODEL/2842
1120.0, 3340.0, 920.0, 22.0, 0.530
1, 1
/FAIL/LAD_TFTRR_LAW/2843
1120.0, 3340.0, 920.0, 22.0, 0.530
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_FIBER_TENSION_RUPTURE/2844
1120.0, 3340.0, 920.0, 22.0, 0.530
1, 1
/FAIL/LAD_TRANSVERSE_FIBER_TENSION_DAMAGE_RATE/2845
1120.0, 3340.0, 920.0, 22.0, 0.530
1, 1
/FAIL/LADEVEZE_TRANSVERSE_FIBER_TENSION_DAMAGE_RATE/2846
1120.0, 3340.0, 920.0, 22.0, 0.530
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for mid in range(2832, 2847):
        assert mid in model.fail_ladtransversefibertensionrupturerates
        f = model.fail_ladtransversefibertensionrupturerates[mid]
        assert pytest.approx(f.sigma_tftrr0) == 1120.0
        assert pytest.approx(f.sigma_tftrrc) == 3340.0
        assert pytest.approx(f.gamma_tftrr) == 920.0
        assert pytest.approx(f.p_tftrr) == 22.0
        assert pytest.approx(f.d_tftrr_max) == 0.530


def test_m456_eng_electrothermoflexomagnetochiralparamagnonplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00516:>20.6e}{90:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetochiralparamagnonplasmonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPARAMAGNONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Engine Electrothermoflexomagnetochiralparamagnonplasmonicpolaritonic Resonance Energy Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralparamagnonplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetochiralparamagnonplasmonicpolaritonic_resonance_energies[1]
    assert pytest.approx(r.dt_etfcparamagnonplp) == 0.00516
    assert r.sens_id == 90


def test_m456_eng_electrothermoflexomagnetochiralparamagnonplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetochiralparamagnonplasmonicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPARAMAGNONPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
0.00616, 100
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralparamagnonplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetochiralparamagnonplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(r.dt_etfcparamagnonplp) == 0.00616
    assert r.sens_id == 100


def test_m456_eng_electrothermoflexomagnetochiralparamagnonplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetochiralparamagnonplasmonicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_PARAMAGNON_PLASMON_POLARITON_RES_WORK/3
0.00516, 90
/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALPARAMAGNONPLASMONICPOLARITONICRESONANCE/4
0.00516, 90
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPARAMAGNONPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/5
0.00516, 90
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALPARAMAGNONPLASMONICPOLARITONIC_RESONANCE/6
0.00516, 90
/ENG/ELECTROTHERMOFLEXOMAGNETOPARAMAGNONCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY/7
0.00516, 90
/ENG/ELECTRO_THERM_FLEXO_MAG_PARAMAGNON_CHIRAL_PLASMON_POLARITON_RES_WORK/8
0.00516, 90
/ENG/EELECTROTHERMOFLEXOMAGNETOPARAMAGNONCHIRALPLASMONICPOLARITONICRESONANCE/9
0.00516, 90
/ENG/ELECTROTHERMOFLEXOMAGNETOPARAMAGNONCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/10
0.00516, 90
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPARAMAGNONCHIRALPLASMONICPOLARITONIC_RESONANCE/11
0.00516, 90
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for rid in range(3, 12):
        assert rid in model.eng_electrothermoflexomagnetochiralparamagnonplasmonicpolaritonic_resonance_energies
        r = model.eng_electrothermoflexomagnetochiralparamagnonplasmonicpolaritonic_resonance_energies[rid]
        assert pytest.approx(r.dt_etfcparamagnonplp) == 0.00516
        assert r.sens_id == 90


def test_m456_lagmul_curvature_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{4.4e6:>20.4f}{12:>10d}{3.1e-5:>20.6e}"
    c2 = f"{29.0:>20.4f}{31.0:>20.4f}{83.0:>20.4f}{15.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Curvature Spinor Spatial Linkage Joint Fixed Format Test
2022 0
/CURVATURE_SPINOR_SPATIAL_LINKAGE_JOINT/1
Curvature Spinor Spatial Mechanism Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_curvature_spinor_spatial_linkage_joints
    j = model.lagmul_curvature_spinor_spatial_linkage_joints[1]
    assert j.node1 == 101
    assert j.node2 == 102
    assert j.node3 == 103
    assert pytest.approx(j.stiff) == 4.4e6
    assert j.skew_id == 12
    assert pytest.approx(j.tol) == 3.1e-5
    assert pytest.approx(j.link_len_a) == 29.0
    assert pytest.approx(j.link_len_b) == 31.0
    assert pytest.approx(j.twist_angle_alpha) == 83.0
    assert pytest.approx(j.offset_distance_s) == 15.0
    assert pytest.approx(j.offset_distance_r) == 15.0
    assert pytest.approx(j.offset_distance_v) == 15.0
    assert pytest.approx(j.offset_distance_h) == 15.0
    assert pytest.approx(j.offset_distance_u) == 15.0
    assert pytest.approx(j.offset_distance_f) == 15.0


def test_m456_lagmul_curvature_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Curvature Spinor Spatial Linkage Joint Free Format Test
/LAGMUL/CURVATURE_SPINOR_SPATIAL_LINKAGE_JOINT/2
Curvature Spinor Spatial Mechanism Joint Free
201, 202, 203, 5.4e6, 13, 4.1e-5
39.0, 41.0, 93.0, 20.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_curvature_spinor_spatial_linkage_joints
    j = model.lagmul_curvature_spinor_spatial_linkage_joints[2]
    assert j.node1 == 201
    assert j.node2 == 202
    assert j.node3 == 203
    assert pytest.approx(j.stiff) == 5.4e6
    assert j.skew_id == 13
    assert pytest.approx(j.tol) == 4.1e-5
    assert pytest.approx(j.link_len_a) == 39.0
    assert pytest.approx(j.link_len_b) == 41.0
    assert pytest.approx(j.twist_angle_alpha) == 93.0
    assert pytest.approx(j.offset_distance_s) == 20.0


def test_m456_lagmul_curvature_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Curvature Spinor Spatial Linkage Joint Aliases Test
/CURVATURE_SPINOR_SPATIAL_LINKAGE/3
101, 102, 103, 4.4e6, 12, 3.1e-5
29.0, 31.0, 83.0, 15.0
/LAGMUL/CURVATURE_SPINOR_SPATIAL_LINKAGE/4
101, 102, 103, 4.4e6, 12, 3.1e-5
29.0, 31.0, 83.0, 15.0
/CURVATURE_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM/5
101, 102, 103, 4.4e6, 12, 3.1e-5
29.0, 31.0, 83.0, 15.0
/CURVATURE_SPINOR_SPATIAL_SYMMETRIC_MECHANISM/6
101, 102, 103, 4.4e6, 12, 3.1e-5
29.0, 31.0, 83.0, 15.0
/CURVATURE_SPINOR_SPATIAL_6R_MECHANISM/7
101, 102, 103, 4.4e6, 12, 3.1e-5
29.0, 31.0, 83.0, 15.0
/CURVATURE_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM/8
101, 102, 103, 4.4e6, 12, 3.1e-5
29.0, 31.0, 83.0, 15.0
/LAGMUL/CURVATURE_TWISTOR_SPATIAL_LINKAGE_JOINT/9
101, 102, 103, 4.4e6, 12, 3.1e-5
29.0, 31.0, 83.0, 15.0
/CURVATURE_TWISTOR_SPATIAL_LINKAGE_JOINT/10
101, 102, 103, 4.4e6, 12, 3.1e-5
29.0, 31.0, 83.0, 15.0
/LAGMUL/CURVATURE_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/11
101, 102, 103, 4.4e6, 12, 3.1e-5
29.0, 31.0, 83.0, 15.0
/CURVATURE_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/12
101, 102, 103, 4.4e6, 12, 3.1e-5
29.0, 31.0, 83.0, 15.0
/LAGMUL/CURVATURE_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/13
101, 102, 103, 4.4e6, 12, 3.1e-5
29.0, 31.0, 83.0, 15.0
/CURVATURE_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/14
101, 102, 103, 4.4e6, 12, 3.1e-5
29.0, 31.0, 83.0, 15.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(3, 15):
        assert jid in model.lagmul_curvature_spinor_spatial_linkage_joints
        j = model.lagmul_curvature_spinor_spatial_linkage_joints[jid]
        assert j.node1 == 101
        assert j.node2 == 102
        assert j.node3 == 103
        assert pytest.approx(j.stiff) == 4.4e6


def test_m456_sensor_spring_coupled_pop_rate_fixed(tmp_path: Path):
    c1 = f"{1210:>10d}{9.96e6:>20.4f}{0.0090:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Coupled Pop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_COUPLED_POP_RATE/1
Spring Coupled Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_coupled_pop_rates
    sensor = model.sensor_spring_coupled_pop_rates[1]
    assert sensor.spring_id == 1210
    assert pytest.approx(sensor.jcoup_pop_max) == 9.96e6
    assert pytest.approx(sensor.j_pop_coup_max) == 9.96e6
    assert pytest.approx(sensor.j_pop_coupled_max) == 9.96e6
    assert pytest.approx(sensor.j_coup_pop_max) == 9.96e6
    assert pytest.approx(sensor.j_pop_max) == 9.96e6
    assert pytest.approx(sensor.j_pop_biaxial_max) == 9.96e6
    assert pytest.approx(sensor.jcoup_shot_max) == 9.96e6
    assert pytest.approx(sensor.jcoup_drop_max) == 9.96e6
    assert pytest.approx(sensor.jcoup_lock_max) == 9.96e6
    assert pytest.approx(sensor.jcoup_snp_max) == 9.96e6
    assert pytest.approx(sensor.jcoup_snap_max) == 9.96e6
    assert pytest.approx(sensor.jcoup_rate_max) == 9.96e6
    assert pytest.approx(sensor.jcoup_roc_rate_max) == 9.96e6
    assert pytest.approx(sensor.jcoup_drop_rate_max) == 9.96e6
    assert pytest.approx(sensor.jcoup_crk_rate_max) == 9.96e6
    assert pytest.approx(sensor.jcoup_crackle_max) == 9.96e6
    assert pytest.approx(sensor.jcoup_crk_max) == 9.96e6
    assert pytest.approx(sensor.jcoup_pop_rate_max) == 9.96e6
    assert pytest.approx(sensor.t_delay) == 0.0090
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_COUPLED_POP_RATE"


def test_m456_sensor_spring_coupled_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Coupled Pop Rate Free Format Test
/SENSOR/SPRING_COUPLED_POP_RATE/2
50, 6.3e5, 0.008
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_coupled_pop_rates
    sensor = model.sensor_spring_coupled_pop_rates[2]
    assert sensor.spring_id == 50
    assert pytest.approx(sensor.jcoup_pop_max) == 6.3e5
    assert pytest.approx(sensor.t_delay) == 0.008
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_COUPLED_POP_RATE"


def test_m456_sensor_spring_coupled_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Coupled Pop Rate Aliases Test
/SENSOR/SPRING_COUPLE_POP_RATE/3
50, 6.3e5, 0.008
/SENSOR/SPRING_COUPLED_POP/4
50, 6.3e5, 0.008
/SENSOR/SPRING_COUPLE_POP/5
50, 6.3e5, 0.008
/SENSOR/SPRING_BIAXIAL_POP_RATE/6
50, 6.3e5, 0.008
/SENSOR/SPRING_BIAXIAL_POP/7
50, 6.3e5, 0.008
/SENSOR/SPRING_POP_RATE_COUPLED/8
50, 6.3e5, 0.008
/SENSOR/SPRING_POP_RATE_COUPLE/9
50, 6.3e5, 0.008
/SENSOR/SPRING_POP_RATE_BIAXIAL/10
50, 6.3e5, 0.008
/SENSOR/COUPLED_POP_RATE_SPRING/11
50, 6.3e5, 0.008
/SENSOR/COUPLE_POP_RATE_SPRING/12
50, 6.3e5, 0.008
/SENSOR/BIAXIAL_POP_RATE_SPRING/13
50, 6.3e5, 0.008
/SENSOR/SPRING_POP_COUPLED/14
50, 6.3e5, 0.008
/SENSOR/SPRING_POP_COUPLE/15
50, 6.3e5, 0.008
/SENSOR/SPRING_POP_BIAXIAL/16
50, 6.3e5, 0.008
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(3, 17):
        assert sid in model.sensor_spring_coupled_pop_rates
        s = model.sensor_spring_coupled_pop_rates[sid]
        assert s.spring_id == 50
        assert pytest.approx(s.jcoup_pop_max) == 6.3e5


def test_m456_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_TRANSVERSE_FIBER_TENSION_RUPTURE_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPARAMAGNONPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/CURVATURE_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_COUPLED_POP_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4
