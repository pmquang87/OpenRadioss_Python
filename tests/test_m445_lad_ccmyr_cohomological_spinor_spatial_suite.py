"""Tests for Milestone M445: LadCoupledCoreMicroyieldingRate Failure Model, EngElectrothermoflexomagnetophononicmagnonicplasmonicpolaritonicResonanceEnergy, LagmulCohomologicalSpinorSpatialLinkageJoint, and SensorSpringTotalPopRate."""

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


def test_m445_fail_lad_coupled_core_microyielding_rate_fixed(tmp_path: Path):
    c1 = f"{1040.0:>20.4f}{3120.0:>20.4f}{840.0:>20.4f}{16.90:>20.4f}{0.410:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2630:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Core Microyielding Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLED_CORE_MICROYIELDING_RATE/2630
Ladeveze Coupled Core Microyielding Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2630 in model.fail_ladcoupledcoremicroyieldingrates
    fccmyr = model.fail_ladcoupledcoremicroyieldingrates[2630]
    assert pytest.approx(fccmyr.sigma_ccmyr0) == 1040.0
    assert pytest.approx(fccmyr.sigma_ccmyrc) == 3120.0
    assert pytest.approx(fccmyr.gamma_ccmyr) == 840.0
    assert pytest.approx(fccmyr.p_ccmyr) == 16.90
    assert pytest.approx(fccmyr.d_ccmyr_max) == 0.410
    assert fccmyr.ifail_sh == 1
    assert fccmyr.ifail_so == 2
    assert fccmyr.fail_id == 2630
    assert pytest.approx(fccmyr.sigma_ccmy0) == 1040.0
    assert pytest.approx(fccmyr.sigma_ccmyc) == 3120.0
    assert pytest.approx(fccmyr.gamma_ccmy) == 840.0
    assert pytest.approx(fccmyr.p_ccmy) == 16.90
    assert pytest.approx(fccmyr.d_ccmy_max) == 0.410
    assert pytest.approx(fccmyr.sigma_ccmpr0) == 1040.0
    assert pytest.approx(fccmyr.sigma_ccmprc) == 3120.0
    assert pytest.approx(fccmyr.gamma_ccmpr) == 840.0
    assert pytest.approx(fccmyr.p_ccmpr) == 16.90
    assert pytest.approx(fccmyr.d_ccmpr_max) == 0.410
    assert pytest.approx(fccmyr.sigma_ccmfr0) == 1040.0
    assert pytest.approx(fccmyr.sigma_ccmfrc) == 3120.0
    assert pytest.approx(fccmyr.gamma_ccmfr) == 840.0
    assert pytest.approx(fccmyr.p_ccmfr) == 16.90
    assert pytest.approx(fccmyr.d_ccmfr_max) == 0.410
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLED_CORE_MICROYIELDING_RATE"


def test_m445_fail_lad_coupled_core_microyielding_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Core Microyielding Rate Free Format Test
/FAIL/LAD_COUPLED_CORE_MICROYIELDING_RATE/2631
1060.0, 3180.0, 860.0, 17.20, 0.400
1, 1
2631
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2631 in model.fail_ladcoupledcoremicroyieldingrates
    fccmyr = model.fail_ladcoupledcoremicroyieldingrates[2631]
    assert pytest.approx(fccmyr.sigma_ccmyr0) == 1060.0
    assert pytest.approx(fccmyr.sigma_ccmyrc) == 3180.0
    assert pytest.approx(fccmyr.gamma_ccmyr) == 860.0
    assert pytest.approx(fccmyr.p_ccmyr) == 17.20
    assert pytest.approx(fccmyr.d_ccmyr_max) == 0.400
    assert fccmyr.fail_id == 2631


def test_m445_fail_lad_coupled_core_microyielding_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Core Microyielding Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_CORE_MICROYIELDING_RATE/2632
990.0, 2970.0, 790.0, 15.70, 0.450
1, 1
/FAIL/LAD_COUPLE_CORE_MICROYIELDING_RATE/2633
990.0, 2970.0, 790.0, 15.70, 0.450
1, 1
/FAIL/LADEVEZE_COUPLE_CORE_MICROYIELDING_RATE/2634
990.0, 2970.0, 790.0, 15.70, 0.450
1, 1
/FAIL/LAD_COUPLED_CORE_MICROYIELD_RATE/2635
990.0, 2970.0, 790.0, 15.70, 0.450
1, 1
/FAIL/LADEVEZE_COUPLED_CORE_MICROYIELD_RATE/2636
990.0, 2970.0, 790.0, 15.70, 0.450
1, 1
/FAIL/LAD_COUPLE_CORE_MICROYIELD_RATE/2637
990.0, 2970.0, 790.0, 15.70, 0.450
1, 1
/FAIL/LADEVEZE_COUPLE_CORE_MICROYIELD_RATE/2638
990.0, 2970.0, 790.0, 15.70, 0.450
1, 1
/FAIL/LAD_CCMYR/2639
990.0, 2970.0, 790.0, 15.70, 0.450
1, 1
/FAIL/LAD_CCMYR_MODEL/2640
990.0, 2970.0, 790.0, 15.70, 0.450
1, 1
/FAIL/LAD_CCMYR_LAW/2641
990.0, 2970.0, 790.0, 15.70, 0.450
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_CORE_MICROYIELDING/2642
990.0, 2970.0, 790.0, 15.70, 0.450
1, 1
/FAIL/LAD_COUPLED_CORE_MICROPLASTICITY_RATE/2643
990.0, 2970.0, 790.0, 15.70, 0.450
1, 1
/FAIL/LADEVEZE_COUPLED_CORE_MICROPLASTICITY_RATE/2644
990.0, 2970.0, 790.0, 15.70, 0.450
1, 1
/FAIL/LAD_COUPLE_CORE_MICROPLASTICITY_RATE/2645
990.0, 2970.0, 790.0, 15.70, 0.450
1, 1
/FAIL/LADEVEZE_COUPLE_CORE_MICROPLASTICITY_RATE/2646
990.0, 2970.0, 790.0, 15.70, 0.450
1, 1
/FAIL/LAD_COUPLED_CORE_MICROPLASTIC_RATE/2647
990.0, 2970.0, 790.0, 15.70, 0.450
1, 1
/FAIL/LADEVEZE_COUPLED_CORE_MICROPLASTIC_RATE/2648
990.0, 2970.0, 790.0, 15.70, 0.450
1, 1
/FAIL/LAD_COUPLE_CORE_MICROPLASTIC_RATE/2649
990.0, 2970.0, 790.0, 15.70, 0.450
1, 1
/FAIL/LADEVEZE_COUPLE_CORE_MICROPLASTIC_RATE/2650
990.0, 2970.0, 790.0, 15.70, 0.450
1, 1
/FAIL/LAD_COUPLED_CORE_MICROFLOW_RATE/2651
990.0, 2970.0, 790.0, 15.70, 0.450
1, 1
/FAIL/LADEVEZE_COUPLED_CORE_MICROFLOW_RATE/2652
990.0, 2970.0, 790.0, 15.70, 0.450
1, 1
/FAIL/LAD_COUPLE_CORE_MICROFLOW_RATE/2653
990.0, 2970.0, 790.0, 15.70, 0.450
1, 1
/FAIL/LADEVEZE_COUPLE_CORE_MICROFLOW_RATE/2654
990.0, 2970.0, 790.0, 15.70, 0.450
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for mid in range(2632, 2655):
        assert mid in model.fail_ladcoupledcoremicroyieldingrates
        f = model.fail_ladcoupledcoremicroyieldingrates[mid]
        assert pytest.approx(f.sigma_ccmyr0) == 990.0
        assert pytest.approx(f.sigma_ccmyrc) == 2970.0


def test_m445_fail_lad_coupled_core_microyielding_rate_properties():
    from pyradioss.model.entities import FailLadCoupledCoreMicroyieldingRate
    f = FailLadCoupledCoreMicroyieldingRate(mat_id=1)
    f.sigma_ccmy0 = 1100.0
    assert pytest.approx(f.sigma_ccmyr0) == 1100.0
    assert pytest.approx(f.sigma_ccmy0) == 1100.0

    f.sigma_ccmyc = 3300.0
    assert pytest.approx(f.sigma_ccmyrc) == 3300.0
    assert pytest.approx(f.sigma_ccmyc) == 3300.0

    f.gamma_ccmy = 880.0
    assert pytest.approx(f.gamma_ccmyr) == 880.0
    assert pytest.approx(f.gamma_ccmy) == 880.0

    f.p_ccmy = 17.00
    assert pytest.approx(f.p_ccmyr) == 17.00
    assert pytest.approx(f.p_ccmy) == 17.00

    f.d_ccmy_max = 0.390
    assert pytest.approx(f.d_ccmyr_max) == 0.390
    assert pytest.approx(f.d_ccmy_max) == 0.390

    f.sigma_ccmpr0 = 1160.0
    assert pytest.approx(f.sigma_ccmyr0) == 1160.0
    assert pytest.approx(f.sigma_ccmpr0) == 1160.0

    f.sigma_ccmprc = 3480.0
    assert pytest.approx(f.sigma_ccmyrc) == 3480.0
    assert pytest.approx(f.sigma_ccmprc) == 3480.0

    f.gamma_ccmpr = 920.0
    assert pytest.approx(f.gamma_ccmyr) == 920.0
    assert pytest.approx(f.gamma_ccmpr) == 920.0

    f.p_ccmpr = 18.00
    assert pytest.approx(f.p_ccmyr) == 18.00
    assert pytest.approx(f.p_ccmpr) == 18.00

    f.d_ccmpr_max = 0.360
    assert pytest.approx(f.d_ccmyr_max) == 0.360
    assert pytest.approx(f.d_ccmpr_max) == 0.360


def test_m445_eng_electrothermoflexomagnetophononicmagnonicplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00185:>20.6f}{62:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Phonon Magnon Plasmon Polaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Phonon Magnon Plasmon Polaritonic Resonance Energy Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetophononicmagnonicplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetophononicmagnonicplasmonicpolaritonic_resonance_energies[1]
    assert pytest.approx(r.dt_etfpmplp) == 0.00185
    assert pytest.approx(r.dt_etfplp) == 0.00185
    assert r.sens_id == 62


def test_m445_eng_electrothermoflexomagnetophononicmagnonicplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Phonon Magnon Plasmon Polaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
0.00285, 63
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetophononicmagnonicplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetophononicmagnonicplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(r.dt_etfpmplp) == 0.00285
    assert r.sens_id == 63


def test_m445_eng_electrothermoflexomagnetophononicmagnonicplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Phonon Magnon Plasmon Polaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_PHONON_MAGNON_PLASMON_POLARITON_RES_WORK/3
0.00385, 64
/ENG/EELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONICPLASMONICPOLARITONICRESONANCE/4
0.00385, 64
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONICPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/5
0.00385, 64
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONICPLASMONICPOLARITONIC_RESONANCE/6
0.00385, 64
/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/7
0.00385, 64
/ENG/ELECTRO_THERM_FLEXO_MAG_MAGNON_PHONON_PLASMON_POLARITON_RES_WORK/8
0.00385, 64
/ENG/EELECTROTHERMOFLEXOMAGNETOMAGNONICPHONONICPLASMONICPOLARITONICRESONANCE/9
0.00385, 64
/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICPHONONICPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/10
0.00385, 64
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOMAGNONICPHONONICPLASMONICPOLARITONIC_RESONANCE/11
0.00385, 64
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for rid in range(3, 12):
        assert rid in model.eng_electrothermoflexomagnetophononicmagnonicplasmonicpolaritonic_resonance_energies
        r = model.eng_electrothermoflexomagnetophononicmagnonicplasmonicpolaritonic_resonance_energies[rid]
        assert pytest.approx(r.dt_etfpmplp) == 0.00385
        assert r.sens_id == 64


def test_m445_lagmul_cohomological_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{3.3e6:>20.4f}{6:>10d}{1.8e-5:>20.6e}"
    c2 = f"{18.0:>20.4f}{20.0:>20.4f}{64.0:>20.4f}{6.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Cohomological Spinor Spatial Linkage Joint Fixed Format Test
2022 0
/COHOMOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT/1
Cohomological Spinor Spatial Mechanism Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_cohomological_spinor_spatial_linkage_joints
    j = model.lagmul_cohomological_spinor_spatial_linkage_joints[1]
    assert j.node1 == 101
    assert j.node2 == 102
    assert j.node3 == 103
    assert pytest.approx(j.stiff) == 3.3e6
    assert j.skew_id == 6
    assert pytest.approx(j.tol) == 1.8e-5
    assert pytest.approx(j.link_len_a) == 18.0
    assert pytest.approx(j.link_len_b) == 20.0
    assert pytest.approx(j.twist_angle_alpha) == 64.0
    assert pytest.approx(j.offset_distance_s) == 6.0
    assert pytest.approx(j.offset_distance_r) == 6.0
    assert pytest.approx(j.offset_distance_v) == 6.0
    assert pytest.approx(j.offset_distance_h) == 6.0
    assert pytest.approx(j.offset_distance_u) == 6.0
    assert pytest.approx(j.offset_distance_f) == 6.0


def test_m445_lagmul_cohomological_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Cohomological Spinor Spatial Linkage Joint Free Format Test
/LAGMUL/COHOMOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT/2
201, 202, 203, 4.3e6, 7, 2.8e-5
21.0, 23.0, 76.0, 7.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_cohomological_spinor_spatial_linkage_joints
    j = model.lagmul_cohomological_spinor_spatial_linkage_joints[2]
    assert j.node1 == 201
    assert j.node2 == 202
    assert j.node3 == 203
    assert pytest.approx(j.stiff) == 4.3e6
    assert j.skew_id == 7
    assert pytest.approx(j.tol) == 2.8e-5
    assert pytest.approx(j.link_len_a) == 21.0
    assert pytest.approx(j.link_len_b) == 23.0
    assert pytest.approx(j.twist_angle_alpha) == 76.0
    assert pytest.approx(j.offset_distance_s) == 7.0


def test_m445_lagmul_cohomological_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Cohomological Spinor Spatial Linkage Joint Aliases Test
/LAGMUL/COHOMOLOGICAL_SPINOR_SPATIAL_LINKAGE/3
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/COHOMOLOGICAL_SPINOR_SPATIAL_LINKAGE/4
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/COHOMOLOGICAL_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM/5
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/COHOMOLOGICAL_SPINOR_SPATIAL_SYMMETRIC_MECHANISM/6
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/COHOMOLOGICAL_SPINOR_SPATIAL_6R_MECHANISM/7
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/COHOMOLOGICAL_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM/8
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/COHOMOLOGICAL_TWISTOR_SPATIAL_LINKAGE_JOINT/9
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/COHOMOLOGICAL_TWISTOR_SPATIAL_LINKAGE_JOINT/10
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/COHOMOLOGICAL_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/11
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/COHOMOLOGICAL_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/12
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/COHOMOLOGICAL_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/13
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/COHOMOLOGICAL_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/14
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(3, 15):
        assert jid in model.lagmul_cohomological_spinor_spatial_linkage_joints
        j = model.lagmul_cohomological_spinor_spatial_linkage_joints[jid]
        assert j.node1 == 301
        assert j.node2 == 302
        assert j.node3 == 303


def test_m445_sensor_spring_total_pop_rate_fixed(tmp_path: Path):
    c1 = f"{10:>10d}{1.98e5:>20.4f}{0.0028:>20.6f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Pop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_POP_RATE/1
Total Pop Rate Sensor Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_pop_rates
    s = model.sensor_spring_total_pop_rates[1]
    assert s.spring_id == 10
    assert pytest.approx(s.jtot_pop_max) == 1.98e5
    assert pytest.approx(s.t_delay) == 0.0028
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_POP_RATE"


def test_m445_sensor_spring_total_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Pop Rate Free Format Test
/SENSOR/SPRING_TOTAL_POP_RATE/2
20, 2.98e5, 0.0038
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_pop_rates
    s = model.sensor_spring_total_pop_rates[2]
    assert s.spring_id == 20
    assert pytest.approx(s.jtot_pop_max) == 2.98e5
    assert pytest.approx(s.t_delay) == 0.0038


def test_m445_sensor_spring_total_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Pop Rate Aliases Test
/SENSOR/SPRING_TOT_POP_RATE/3
30, 3.7e5, 0.001
/SENSOR/SPRING_TOTAL_POP/4
30, 3.7e5, 0.001
/SENSOR/SPRING_TOT_POP/5
30, 3.7e5, 0.001
/SENSOR/SPRING_RESULTANT_POP_RATE/6
30, 3.7e5, 0.001
/SENSOR/SPRING_RES_POP_RATE/7
30, 3.7e5, 0.001
/SENSOR/SPRING_RESULTANT_POP/8
30, 3.7e5, 0.001
/SENSOR/SPRING_RES_POP/9
30, 3.7e5, 0.001
/SENSOR/SPRING_POP_RATE_TOTAL/10
30, 3.7e5, 0.001
/SENSOR/SPRING_POP_RATE_TOT/11
30, 3.7e5, 0.001
/SENSOR/SPRING_POP_RATE_RESULTANT/12
30, 3.7e5, 0.001
/SENSOR/SPRING_POP_RATE_RES/13
30, 3.7e5, 0.001
/SENSOR/TOTAL_POP_RATE_SPRING/14
30, 3.7e5, 0.001
/SENSOR/RESULTANT_POP_RATE_SPRING/15
30, 3.7e5, 0.001
/SENSOR/SPRING_POP_TOTAL/16
30, 3.7e5, 0.001
/SENSOR/SPRING_POP_RESULTANT/17
30, 3.7e5, 0.001
/SENSOR/SPRING_POP_RES/18
30, 3.7e5, 0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(3, 19):
        assert sid in model.sensor_spring_total_pop_rates
        s = model.sensor_spring_total_pop_rates[sid]
        assert s.spring_id == 30
        assert pytest.approx(s.jtot_pop_max) == 3.7e5


def test_m445_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_COUPLED_CORE_MICROYIELDING_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/COHOMOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_TOTAL_POP_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4
