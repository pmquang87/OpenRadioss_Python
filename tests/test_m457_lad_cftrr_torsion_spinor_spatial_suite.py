"""Tests for Milestone M457: LadCoupledFiberTensionRuptureRate Failure Model, EngElectrothermoflexomagnetochiraldyonicplasmonicpolaritonicResonanceEnergy, LagmulTorsionSpinorSpatialLinkageJoint, and SensorSpringTorsionalPopRate."""

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


def test_m457_fail_lad_coupled_fiber_tension_rupture_rate_fixed(tmp_path: Path):
    c1 = f"{1130.0:>20.4f}{3350.0:>20.4f}{930.0:>20.4f}{23.0:>20.4f}{0.540:>20.4f}"
    c2 = f"{2:>10d}{2:>10d}"
    c3 = f"{830:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Fiber Tension Rupture Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLED_FIBER_TENSION_RUPTURE_RATE/2850
Ladeveze Coupled Fiber Tension Rupture Law
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2850 in model.fail_ladcoupledfibertensionrupturerates
    f = model.fail_ladcoupledfibertensionrupturerates[2850]
    assert pytest.approx(f.sigma_cftrr0) == 1130.0
    assert pytest.approx(f.sigma_cftrrc) == 3350.0
    assert pytest.approx(f.gamma_cftrr) == 930.0
    assert pytest.approx(f.p_cftrr) == 23.0
    assert pytest.approx(f.d_cftrr_max) == 0.540
    assert f.ifail_sh == 2
    assert f.ifail_so == 2
    assert f.fail_id == 830
    assert pytest.approx(f.sigma_cftr0) == 1130.0
    assert pytest.approx(f.sigma_cftrc) == 3350.0
    assert pytest.approx(f.gamma_cftr) == 930.0
    assert pytest.approx(f.p_cftr) == 23.0
    assert pytest.approx(f.d_cftr_max) == 0.540
    assert pytest.approx(f.sigma_cft0) == 1130.0
    assert pytest.approx(f.sigma_cftc) == 3350.0
    assert pytest.approx(f.gamma_cft) == 930.0
    assert pytest.approx(f.p_cft) == 23.0
    assert pytest.approx(f.d_cft_max) == 0.540
    assert pytest.approx(f.sigma_cfr0) == 1130.0
    assert pytest.approx(f.sigma_cfrc) == 3350.0
    assert pytest.approx(f.gamma_cfr) == 930.0
    assert pytest.approx(f.p_cfr) == 23.0
    assert pytest.approx(f.d_cfr_max) == 0.540


def test_m457_fail_lad_coupled_fiber_tension_rupture_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Fiber Tension Rupture Rate Free Format Test
/FAIL/LAD_COUPLED_FIBER_TENSION_RUPTURE_RATE/2851
1230.0, 3650.0, 1030.0, 25.0, 0.640
1, 1
930
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2851 in model.fail_ladcoupledfibertensionrupturerates
    f = model.fail_ladcoupledfibertensionrupturerates[2851]
    assert pytest.approx(f.sigma_cftrr0) == 1230.0
    assert pytest.approx(f.sigma_cftrrc) == 3650.0
    assert pytest.approx(f.gamma_cftrr) == 1030.0
    assert pytest.approx(f.p_cftrr) == 25.0
    assert pytest.approx(f.d_cftrr_max) == 0.640
    assert f.ifail_sh == 1
    assert f.ifail_so == 1
    assert f.fail_id == 930


def test_m457_fail_lad_coupled_fiber_tension_rupture_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Fiber Tension Rupture Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_FIBER_TENSION_RUPTURE_RATE/2852
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LAD_COUPLED_FIBER_TENSION_RUPTURE/2853
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LADEVEZE_COUPLED_FIBER_TENSION_RUPTURE/2854
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LAD_COUPLED_FIBER_TENSION_RATE/2855
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LADEVEZE_COUPLED_FIBER_TENSION_RATE/2856
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LAD_COUPLED_FIBER_RUPTURE_RATE/2857
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LADEVEZE_COUPLED_FIBER_RUPTURE_RATE/2858
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LAD_COUPLED_FIBER_TENSILE_RUPTURE_RATE/2859
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LADEVEZE_COUPLED_FIBER_TENSILE_RUPTURE_RATE/2860
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LAD_COUPLE_FIBER_TENSION_RUPTURE_RATE/2861
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LADEVEZE_COUPLE_FIBER_TENSION_RUPTURE_RATE/2862
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LAD_COUPLE_FIBER_TENSION_RUPTURE/2863
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LADEVEZE_COUPLE_FIBER_TENSION_RUPTURE/2864
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LAD_COUPLE_FIBER_TENSION_RATE/2865
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LADEVEZE_COUPLE_FIBER_TENSION_RATE/2866
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LAD_COUPLE_FIBER_RUPTURE_RATE/2867
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LADEVEZE_COUPLE_FIBER_RUPTURE_RATE/2868
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LAD_COUPLE_FIBER_TENSILE_RUPTURE_RATE/2869
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LADEVEZE_COUPLE_FIBER_TENSILE_RUPTURE_RATE/2870
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LAD_CFTRR/2871
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LAD_CFT_RR/2872
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LAD_CFTRR_MODEL/2873
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LAD_CFTRR_LAW/2874
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_FIBER_TENSION_RUPTURE/2875
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LAD_COUPLED_FIBER_TENSION_DAMAGE_RATE/2876
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LADEVEZE_COUPLED_FIBER_TENSION_DAMAGE_RATE/2877
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LAD_COUPLE_FIBER_TENSION_DAMAGE_RATE/2878
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/FAIL/LADEVEZE_COUPLE_FIBER_TENSION_DAMAGE_RATE/2879
1130.0, 3350.0, 930.0, 23.0, 0.540
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for mid in range(2852, 2880):
        assert mid in model.fail_ladcoupledfibertensionrupturerates
        f = model.fail_ladcoupledfibertensionrupturerates[mid]
        assert pytest.approx(f.sigma_cftrr0) == 1130.0
        assert pytest.approx(f.sigma_cftrrc) == 3350.0
        assert pytest.approx(f.gamma_cftrr) == 930.0
        assert pytest.approx(f.p_cftrr) == 23.0
        assert pytest.approx(f.d_cftrr_max) == 0.540


def test_m457_eng_electrothermoflexomagnetochiraldyonicplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00517:>20.6e}{91:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetochiraldyonicplasmonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALDYONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Engine Electrothermoflexomagnetochiraldyonicplasmonicpolaritonic Resonance Energy Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiraldyonicplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetochiraldyonicplasmonicpolaritonic_resonance_energies[1]
    assert pytest.approx(r.dt_etfcdyonicplp) == 0.00517
    assert r.sens_id == 91


def test_m457_eng_electrothermoflexomagnetochiraldyonicplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetochiraldyonicplasmonicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALDYONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
0.00617, 101
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiraldyonicplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetochiraldyonicplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(r.dt_etfcdyonicplp) == 0.00617
    assert r.sens_id == 101


def test_m457_eng_electrothermoflexomagnetochiraldyonicplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetochiraldyonicplasmonicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_DYONIC_PLASMON_POLARITON_RES_WORK/3
0.00517, 91
/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALDYONICPLASMONICPOLARITONICRESONANCE/4
0.00517, 91
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALDYONICPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/5
0.00517, 91
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALDYONICPLASMONICPOLARITONIC_RESONANCE/6
0.00517, 91
/ENG/ELECTROTHERMOFLEXOMAGNETODYONICCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY/7
0.00517, 91
/ENG/ELECTRO_THERM_FLEXO_MAG_DYONIC_CHIRAL_PLASMON_POLARITON_RES_WORK/8
0.00517, 91
/ENG/EELECTROTHERMOFLEXOMAGNETODYONICCHIRALPLASMONICPOLARITONICRESONANCE/9
0.00517, 91
/ENG/ELECTROTHERMOFLEXOMAGNETODYONICCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/10
0.00517, 91
/ENG/ET_ELECTROTHERMOFLEXOMAGNETODYONICCHIRALPLASMONICPOLARITONIC_RESONANCE/11
0.00517, 91
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for rid in range(3, 12):
        assert rid in model.eng_electrothermoflexomagnetochiraldyonicplasmonicpolaritonic_resonance_energies
        r = model.eng_electrothermoflexomagnetochiraldyonicplasmonicpolaritonic_resonance_energies[rid]
        assert pytest.approx(r.dt_etfcdyonicplp) == 0.00517
        assert r.sens_id == 91


def test_m457_lagmul_torsion_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{4.5e6:>20.4f}{12:>10d}{3.2e-5:>20.6e}"
    c2 = f"{30.0:>20.4f}{32.0:>20.4f}{84.0:>20.4f}{16.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Torsion Spinor Spatial Linkage Joint Fixed Format Test
2022 0
/TORSION_SPINOR_SPATIAL_LINKAGE_JOINT/1
Torsion Spinor Spatial Mechanism Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_torsion_spinor_spatial_linkage_joints
    j = model.lagmul_torsion_spinor_spatial_linkage_joints[1]
    assert j.node1 == 101
    assert j.node2 == 102
    assert j.node3 == 103
    assert pytest.approx(j.stiff) == 4.5e6
    assert j.skew_id == 12
    assert pytest.approx(j.tol) == 3.2e-5
    assert pytest.approx(j.link_len_a) == 30.0
    assert pytest.approx(j.link_len_b) == 32.0
    assert pytest.approx(j.twist_angle_alpha) == 84.0
    assert pytest.approx(j.offset_distance_s) == 16.0
    assert pytest.approx(j.offset_distance_r) == 16.0
    assert pytest.approx(j.offset_distance_v) == 16.0
    assert pytest.approx(j.offset_distance_h) == 16.0
    assert pytest.approx(j.offset_distance_u) == 16.0
    assert pytest.approx(j.offset_distance_f) == 16.0


def test_m457_lagmul_torsion_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Torsion Spinor Spatial Linkage Joint Free Format Test
/LAGMUL/TORSION_SPINOR_SPATIAL_LINKAGE_JOINT/2
Torsion Spinor Spatial Mechanism Joint Free
201, 202, 203, 5.5e6, 13, 4.2e-5
40.0, 42.0, 94.0, 21.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_torsion_spinor_spatial_linkage_joints
    j = model.lagmul_torsion_spinor_spatial_linkage_joints[2]
    assert j.node1 == 201
    assert j.node2 == 202
    assert j.node3 == 203
    assert pytest.approx(j.stiff) == 5.5e6
    assert j.skew_id == 13
    assert pytest.approx(j.tol) == 4.2e-5
    assert pytest.approx(j.link_len_a) == 40.0
    assert pytest.approx(j.link_len_b) == 42.0
    assert pytest.approx(j.twist_angle_alpha) == 94.0
    assert pytest.approx(j.offset_distance_s) == 21.0


def test_m457_lagmul_torsion_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Torsion Spinor Spatial Linkage Joint Aliases Test
/TORSION_SPINOR_SPATIAL_LINKAGE/3
101, 102, 103, 4.5e6, 12, 3.2e-5
30.0, 32.0, 84.0, 16.0
/LAGMUL/TORSION_SPINOR_SPATIAL_LINKAGE/4
101, 102, 103, 4.5e6, 12, 3.2e-5
30.0, 32.0, 84.0, 16.0
/TORSION_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM/5
101, 102, 103, 4.5e6, 12, 3.2e-5
30.0, 32.0, 84.0, 16.0
/TORSION_SPINOR_SPATIAL_SYMMETRIC_MECHANISM/6
101, 102, 103, 4.5e6, 12, 3.2e-5
30.0, 32.0, 84.0, 16.0
/TORSION_SPINOR_SPATIAL_6R_MECHANISM/7
101, 102, 103, 4.5e6, 12, 3.2e-5
30.0, 32.0, 84.0, 16.0
/TORSION_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM/8
101, 102, 103, 4.5e6, 12, 3.2e-5
30.0, 32.0, 84.0, 16.0
/LAGMUL/TORSION_TWISTOR_SPATIAL_LINKAGE_JOINT/9
101, 102, 103, 4.5e6, 12, 3.2e-5
30.0, 32.0, 84.0, 16.0
/TORSION_TWISTOR_SPATIAL_LINKAGE_JOINT/10
101, 102, 103, 4.5e6, 12, 3.2e-5
30.0, 32.0, 84.0, 16.0
/LAGMUL/TORSION_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/11
101, 102, 103, 4.5e6, 12, 3.2e-5
30.0, 32.0, 84.0, 16.0
/TORSION_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/12
101, 102, 103, 4.5e6, 12, 3.2e-5
30.0, 32.0, 84.0, 16.0
/LAGMUL/TORSION_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/13
101, 102, 103, 4.5e6, 12, 3.2e-5
30.0, 32.0, 84.0, 16.0
/TORSION_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/14
101, 102, 103, 4.5e6, 12, 3.2e-5
30.0, 32.0, 84.0, 16.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(3, 15):
        assert jid in model.lagmul_torsion_spinor_spatial_linkage_joints
        j = model.lagmul_torsion_spinor_spatial_linkage_joints[jid]
        assert j.node1 == 101
        assert j.node2 == 102
        assert j.node3 == 103
        assert pytest.approx(j.stiff) == 4.5e6


def test_m457_sensor_spring_torsional_pop_rate_fixed(tmp_path: Path):
    c1 = f"{1211:>10d}{9.97e6:>20.4f}{0.0091:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Pop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_POP_RATE/1
Spring Torsional Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_pop_rates
    sensor = model.sensor_spring_torsional_pop_rates[1]
    assert sensor.spring_id == 1211
    assert pytest.approx(sensor.jtor_pop_max) == 9.97e6
    assert pytest.approx(sensor.j_pop_tor_max) == 9.97e6
    assert pytest.approx(sensor.j_pop_torsional_max) == 9.97e6
    assert pytest.approx(sensor.j_tor_pop_max) == 9.97e6
    assert pytest.approx(sensor.j_pop_max) == 9.97e6
    assert pytest.approx(sensor.j_pop_twist_max) == 9.97e6
    assert pytest.approx(sensor.jtor_shot_max) == 9.97e6
    assert pytest.approx(sensor.jtor_drop_max) == 9.97e6
    assert pytest.approx(sensor.jtor_lock_max) == 9.97e6
    assert pytest.approx(sensor.jtor_snp_max) == 9.97e6
    assert pytest.approx(sensor.jtor_snap_max) == 9.97e6
    assert pytest.approx(sensor.jtor_rate_max) == 9.97e6
    assert pytest.approx(sensor.jtor_roc_rate_max) == 9.97e6
    assert pytest.approx(sensor.jtor_drop_rate_max) == 9.97e6
    assert pytest.approx(sensor.jtor_crk_rate_max) == 9.97e6
    assert pytest.approx(sensor.jtor_crackle_max) == 9.97e6
    assert pytest.approx(sensor.jtor_crk_max) == 9.97e6
    assert pytest.approx(sensor.jtor_pop_rate_max) == 9.97e6
    assert pytest.approx(sensor.t_delay) == 0.0091
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_POP_RATE"


def test_m457_sensor_spring_torsional_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Torsional Pop Rate Free Format Test
/SENSOR/SPRING_TORSIONAL_POP_RATE/2
51, 6.4e5, 0.009
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_torsional_pop_rates
    sensor = model.sensor_spring_torsional_pop_rates[2]
    assert sensor.spring_id == 51
    assert pytest.approx(sensor.jtor_pop_max) == 6.4e5
    assert pytest.approx(sensor.t_delay) == 0.009
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_POP_RATE"


def test_m457_sensor_spring_torsional_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Torsional Pop Rate Aliases Test
/SENSOR/SPRING_TORSION_POP_RATE/3
51, 6.4e5, 0.009
/SENSOR/SPRING_TORSIONAL_POP/4
51, 6.4e5, 0.009
/SENSOR/SPRING_TORSION_POP/5
51, 6.4e5, 0.009
/SENSOR/SPRING_TWIST_POP_RATE/6
51, 6.4e5, 0.009
/SENSOR/SPRING_TWIST_POP/7
51, 6.4e5, 0.009
/SENSOR/SPRING_POP_RATE_TORSIONAL/8
51, 6.4e5, 0.009
/SENSOR/SPRING_POP_RATE_TORSION/9
51, 6.4e5, 0.009
/SENSOR/SPRING_POP_RATE_TWIST/10
51, 6.4e5, 0.009
/SENSOR/TORSIONAL_POP_RATE_SPRING/11
51, 6.4e5, 0.009
/SENSOR/TORSION_POP_RATE_SPRING/12
51, 6.4e5, 0.009
/SENSOR/TWIST_POP_RATE_SPRING/13
51, 6.4e5, 0.009
/SENSOR/SPRING_POP_TORSIONAL/14
51, 6.4e5, 0.009
/SENSOR/SPRING_POP_TORSION/15
51, 6.4e5, 0.009
/SENSOR/SPRING_POP_TWIST/16
51, 6.4e5, 0.009
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(3, 17):
        assert sid in model.sensor_spring_torsional_pop_rates
        s = model.sensor_spring_torsional_pop_rates[sid]
        assert s.spring_id == 51
        assert pytest.approx(s.jtor_pop_max) == 6.4e5


def test_m457_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_COUPLED_FIBER_TENSION_RUPTURE_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALDYONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/TORSION_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_TORSIONAL_POP_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4
