"""Tests for Milestone M443: LadDynamicCoreMicroyieldingRate Failure Model, EngElectrothermoflexomagnetomagnonicplasmonicpolaritonicResonanceEnergy, LagmulTopologicalSpinorSpatialLinkageJoint, and SensorSpringNormalPopRate."""

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


def test_m443_fail_lad_dynamic_core_microyielding_rate_fixed(tmp_path: Path):
    c1 = f"{1020.0:>20.4f}{3060.0:>20.4f}{820.0:>20.4f}{16.70:>20.4f}{0.420:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2590:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Core Microyielding Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_CORE_MICROYIELDING_RATE/2590
Ladeveze Dynamic Core Microyielding Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2590 in model.fail_laddynamiccoremicroyieldingrates
    fdcmyr = model.fail_laddynamiccoremicroyieldingrates[2590]
    assert pytest.approx(fdcmyr.sigma_dcmyr0) == 1020.0
    assert pytest.approx(fdcmyr.sigma_dcmyrc) == 3060.0
    assert pytest.approx(fdcmyr.gamma_dcmyr) == 820.0
    assert pytest.approx(fdcmyr.p_dcmyr) == 16.70
    assert pytest.approx(fdcmyr.d_dcmyr_max) == 0.420
    assert fdcmyr.ifail_sh == 1
    assert fdcmyr.ifail_so == 2
    assert fdcmyr.fail_id == 2590
    assert pytest.approx(fdcmyr.sigma_dcmy0) == 1020.0
    assert pytest.approx(fdcmyr.sigma_dcmyc) == 3060.0
    assert pytest.approx(fdcmyr.gamma_dcmy) == 820.0
    assert pytest.approx(fdcmyr.p_dcmy) == 16.70
    assert pytest.approx(fdcmyr.d_dcmy_max) == 0.420
    assert pytest.approx(fdcmyr.sigma_dcmpr0) == 1020.0
    assert pytest.approx(fdcmyr.sigma_dcmprc) == 3060.0
    assert pytest.approx(fdcmyr.gamma_dcmpr) == 820.0
    assert pytest.approx(fdcmyr.p_dcmpr) == 16.70
    assert pytest.approx(fdcmyr.d_dcmpr_max) == 0.420
    assert pytest.approx(fdcmyr.sigma_dcmfr0) == 1020.0
    assert pytest.approx(fdcmyr.sigma_dcmfrc) == 3060.0
    assert pytest.approx(fdcmyr.gamma_dcmfr) == 820.0
    assert pytest.approx(fdcmyr.p_dcmfr) == 16.70
    assert pytest.approx(fdcmyr.d_dcmfr_max) == 0.420
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_CORE_MICROYIELDING_RATE"


def test_m443_fail_lad_dynamic_core_microyielding_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Core Microyielding Rate Free Format Test
/FAIL/LAD_DYNAMIC_CORE_MICROYIELDING_RATE/2591
1040.0, 3120.0, 840.0, 17.00, 0.410
1, 1
2591
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2591 in model.fail_laddynamiccoremicroyieldingrates
    fdcmyr = model.fail_laddynamiccoremicroyieldingrates[2591]
    assert pytest.approx(fdcmyr.sigma_dcmyr0) == 1040.0
    assert pytest.approx(fdcmyr.sigma_dcmyrc) == 3120.0
    assert pytest.approx(fdcmyr.gamma_dcmyr) == 840.0
    assert pytest.approx(fdcmyr.p_dcmyr) == 17.00
    assert pytest.approx(fdcmyr.d_dcmyr_max) == 0.410
    assert fdcmyr.fail_id == 2591


def test_m443_fail_lad_dynamic_core_microyielding_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Core Microyielding Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_CORE_MICROYIELDING_RATE/2592
970.0, 2910.0, 770.0, 15.50, 0.460
1, 1
/FAIL/LAD_DYNAMIC_CORE_MICROYIELD_RATE/2593
970.0, 2910.0, 770.0, 15.50, 0.460
1, 1
/FAIL/LADEVEZE_DYNAMIC_CORE_MICROYIELD_RATE/2594
970.0, 2910.0, 770.0, 15.50, 0.460
1, 1
/FAIL/LAD_DCMYR/2595
970.0, 2910.0, 770.0, 15.50, 0.460
1, 1
/FAIL/LAD_DCMYR_MODEL/2596
970.0, 2910.0, 770.0, 15.50, 0.460
1, 1
/FAIL/LAD_DCMYR_LAW/2597
970.0, 2910.0, 770.0, 15.50, 0.460
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_CORE_MICROYIELDING/2598
970.0, 2910.0, 770.0, 15.50, 0.460
1, 1
/FAIL/LAD_DYNAMIC_CORE_MICROPLASTICITY_RATE/2599
970.0, 2910.0, 770.0, 15.50, 0.460
1, 1
/FAIL/LADEVEZE_DYNAMIC_CORE_MICROPLASTICITY_RATE/2600
970.0, 2910.0, 770.0, 15.50, 0.460
1, 1
/FAIL/LAD_DYNAMIC_CORE_MICROPLASTIC_RATE/2601
970.0, 2910.0, 770.0, 15.50, 0.460
1, 1
/FAIL/LADEVEZE_DYNAMIC_CORE_MICROPLASTIC_RATE/2602
970.0, 2910.0, 770.0, 15.50, 0.460
1, 1
/FAIL/LAD_DYNAMIC_CORE_MICROFLOW_RATE/2603
970.0, 2910.0, 770.0, 15.50, 0.460
1, 1
/FAIL/LADEVEZE_DYNAMIC_CORE_MICROFLOW_RATE/2604
970.0, 2910.0, 770.0, 15.50, 0.460
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for mid in range(2592, 2605):
        assert mid in model.fail_laddynamiccoremicroyieldingrates
        f = model.fail_laddynamiccoremicroyieldingrates[mid]
        assert pytest.approx(f.sigma_dcmyr0) == 970.0
        assert pytest.approx(f.sigma_dcmyrc) == 2910.0


def test_m443_fail_lad_dynamic_core_microyielding_rate_properties():
    from pyradioss.model.entities import FailLadDynamicCoreMicroyieldingRate
    f = FailLadDynamicCoreMicroyieldingRate(mat_id=1)
    f.sigma_dcmy0 = 1080.0
    assert pytest.approx(f.sigma_dcmyr0) == 1080.0
    assert pytest.approx(f.sigma_dcmy0) == 1080.0

    f.sigma_dcmyc = 3240.0
    assert pytest.approx(f.sigma_dcmyrc) == 3240.0
    assert pytest.approx(f.sigma_dcmyc) == 3240.0

    f.gamma_dcmy = 860.0
    assert pytest.approx(f.gamma_dcmyr) == 860.0
    assert pytest.approx(f.gamma_dcmy) == 860.0

    f.p_dcmy = 16.80
    assert pytest.approx(f.p_dcmyr) == 16.80
    assert pytest.approx(f.p_dcmy) == 16.80

    f.d_dcmy_max = 0.400
    assert pytest.approx(f.d_dcmyr_max) == 0.400
    assert pytest.approx(f.d_dcmy_max) == 0.400

    f.sigma_dcmpr0 = 1140.0
    assert pytest.approx(f.sigma_dcmyr0) == 1140.0
    assert pytest.approx(f.sigma_dcmpr0) == 1140.0

    f.sigma_dcmprc = 3420.0
    assert pytest.approx(f.sigma_dcmyrc) == 3420.0
    assert pytest.approx(f.sigma_dcmprc) == 3420.0

    f.gamma_dcmpr = 900.0
    assert pytest.approx(f.gamma_dcmyr) == 900.0
    assert pytest.approx(f.gamma_dcmpr) == 900.0

    f.p_dcmpr = 17.80
    assert pytest.approx(f.p_dcmyr) == 17.80
    assert pytest.approx(f.p_dcmpr) == 17.80

    f.d_dcmpr_max = 0.370
    assert pytest.approx(f.d_dcmyr_max) == 0.370
    assert pytest.approx(f.d_dcmpr_max) == 0.370


def test_m443_eng_electrothermoflexomagnetomagnonicplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00165:>20.6f}{56:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Magnon Plasmon Polaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Magnon Plasmon Polaritonic Resonance Energy Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetomagnonicplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetomagnonicplasmonicpolaritonic_resonance_energies[1]
    assert pytest.approx(r.dt_etfmagplp) == 0.00165
    assert pytest.approx(r.dt_etfplp) == 0.00165
    assert r.sens_id == 56


def test_m443_eng_electrothermoflexomagnetomagnonicplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Magnon Plasmon Polaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
0.00265, 57
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetomagnonicplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetomagnonicplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(r.dt_etfmagplp) == 0.00265
    assert r.sens_id == 57


def test_m443_eng_electrothermoflexomagnetomagnonicplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Magnon Plasmon Polaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_MAGNON_PLASMON_POLARITON_RES_WORK/3
0.00365, 58
/ENG/EELECTROTHERMOFLEXOMAGNETOMAGNONICPLASMONICPOLARITONICRESONANCE/4
0.00365, 58
/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/5
0.00365, 58
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOMAGNONICPLASMONICPOLARITONIC_RESONANCE/6
0.00365, 58
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for rid in range(3, 7):
        assert rid in model.eng_electrothermoflexomagnetomagnonicplasmonicpolaritonic_resonance_energies
        r = model.eng_electrothermoflexomagnetomagnonicplasmonicpolaritonic_resonance_energies[rid]
        assert pytest.approx(r.dt_etfmagplp) == 0.00365
        assert r.sens_id == 58


def test_m443_lagmul_topological_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{3.1e6:>20.4f}{6:>10d}{1.6e-5:>20.6e}"
    c2 = f"{16.0:>20.4f}{18.0:>20.4f}{60.0:>20.4f}{5.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Topological Spinor Spatial Linkage Joint Fixed Format Test
2022 0
/TOPOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT/1
Topological Spinor Spatial Mechanism Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_topological_spinor_spatial_linkage_joints
    j = model.lagmul_topological_spinor_spatial_linkage_joints[1]
    assert j.node1 == 101
    assert j.node2 == 102
    assert j.node3 == 103
    assert pytest.approx(j.stiff) == 3.1e6
    assert j.skew_id == 6
    assert pytest.approx(j.tol) == 1.6e-5
    assert pytest.approx(j.link_len_a) == 16.0
    assert pytest.approx(j.link_len_b) == 18.0
    assert pytest.approx(j.twist_angle_alpha) == 60.0
    assert pytest.approx(j.offset_distance_s) == 5.0
    assert pytest.approx(j.offset_distance_r) == 5.0
    assert pytest.approx(j.offset_distance_v) == 5.0
    assert pytest.approx(j.offset_distance_h) == 5.0
    assert pytest.approx(j.offset_distance_u) == 5.0
    assert pytest.approx(j.offset_distance_f) == 5.0


def test_m443_lagmul_topological_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Topological Spinor Spatial Linkage Joint Free Format Test
/LAGMUL/TOPOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT/2
201, 202, 203, 4.1e6, 7, 2.6e-5
19.0, 21.0, 72.0, 6.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_topological_spinor_spatial_linkage_joints
    j = model.lagmul_topological_spinor_spatial_linkage_joints[2]
    assert j.node1 == 201
    assert j.node2 == 202
    assert j.node3 == 203
    assert pytest.approx(j.stiff) == 4.1e6
    assert j.skew_id == 7
    assert pytest.approx(j.tol) == 2.6e-5
    assert pytest.approx(j.link_len_a) == 19.0
    assert pytest.approx(j.link_len_b) == 21.0
    assert pytest.approx(j.twist_angle_alpha) == 72.0
    assert pytest.approx(j.offset_distance_s) == 6.0


def test_m443_lagmul_topological_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Topological Spinor Spatial Linkage Joint Aliases Test
/LAGMUL/TOPOLOGICAL_SPINOR_SPATIAL_LINKAGE/3
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/TOPOLOGICAL_SPINOR_SPATIAL_LINKAGE/4
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/TOPOLOGICAL_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM/5
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/TOPOLOGICAL_SPINOR_SPATIAL_SYMMETRIC_MECHANISM/6
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/TOPOLOGICAL_SPINOR_SPATIAL_6R_MECHANISM/7
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/TOPOLOGICAL_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM/8
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/TOPOLOGICAL_TWISTOR_SPATIAL_LINKAGE_JOINT/9
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/TOPOLOGICAL_TWISTOR_SPATIAL_LINKAGE_JOINT/10
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/TOPOLOGICAL_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/11
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/TOPOLOGICAL_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/12
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/TOPOLOGICAL_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/13
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/TOPOLOGICAL_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/14
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(3, 15):
        assert jid in model.lagmul_topological_spinor_spatial_linkage_joints
        j = model.lagmul_topological_spinor_spatial_linkage_joints[jid]
        assert j.node1 == 301
        assert j.node2 == 302
        assert j.node3 == 303


def test_m443_sensor_spring_normal_pop_rate_fixed(tmp_path: Path):
    c1 = f"{10:>10d}{1.9e5:>20.4f}{0.002:>20.6f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Normal Pop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_NORMAL_POP_RATE/1
Normal Pop Rate Sensor Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_normal_pop_rates
    s = model.sensor_spring_normal_pop_rates[1]
    assert s.spring_id == 10
    assert pytest.approx(s.jnorm_pop_max) == 1.9e5
    assert pytest.approx(s.t_delay) == 0.002
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_NORMAL_POP_RATE"


def test_m443_sensor_spring_normal_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Normal Pop Rate Free Format Test
/SENSOR/SPRING_NORMAL_POP_RATE/2
20, 2.9e5, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_normal_pop_rates
    s = model.sensor_spring_normal_pop_rates[2]
    assert s.spring_id == 20
    assert pytest.approx(s.jnorm_pop_max) == 2.9e5
    assert pytest.approx(s.t_delay) == 0.003


def test_m443_sensor_spring_normal_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Normal Pop Rate Aliases Test
/SENSOR/SPRING_NORM_POP_RATE/3
30, 3.5e5, 0.001
/SENSOR/SPRING_AXIAL_POP_RATE/4
30, 3.5e5, 0.001
/SENSOR/SPRING_NORMAL_POP/5
30, 3.5e5, 0.001
/SENSOR/SPRING_NORM_POP/6
30, 3.5e5, 0.001
/SENSOR/SPRING_AXIAL_POP/7
30, 3.5e5, 0.001
/SENSOR/SPRING_POP_RATE_NORMAL/8
30, 3.5e5, 0.001
/SENSOR/SPRING_POP_RATE_NORM/9
30, 3.5e5, 0.001
/SENSOR/SPRING_POP_RATE_AXIAL/10
30, 3.5e5, 0.001
/SENSOR/NORMAL_POP_RATE_SPRING/11
30, 3.5e5, 0.001
/SENSOR/SPRING_POP_NORM/12
30, 3.5e5, 0.001
/SENSOR/SPRING_POP_NORMAL/13
30, 3.5e5, 0.001
/SENSOR/SPRING_POP_AXIAL/14
30, 3.5e5, 0.001
/SENSOR/SPRING_POP_RATE_AX/15
30, 3.5e5, 0.001
/SENSOR/SPRING_AX_POP_RATE/16
30, 3.5e5, 0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(3, 17):
        assert sid in model.sensor_spring_normal_pop_rates
        s = model.sensor_spring_normal_pop_rates[sid]
        assert s.spring_id == 30
        assert pytest.approx(s.jnorm_pop_max) == 3.5e5


def test_m443_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_DYNAMIC_CORE_MICROYIELDING_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/TOPOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_NORMAL_POP_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4
