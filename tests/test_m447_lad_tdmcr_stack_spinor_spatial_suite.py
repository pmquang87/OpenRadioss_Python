"""Tests for Milestone M447: LadTransverseDelaminationMicrocrackingRate Failure Model, EngElectrothermoflexomagnetochiralplasmonicpolaritonicResonanceEnergy, LagmulStackSpinorSpatialLinkageJoint, and SensorSpringBendingPopRate."""

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


def test_m447_fail_lad_transverse_delamination_microcracking_rate_fixed(tmp_path: Path):
    c1 = f"{1060.0:>20.4f}{3180.0:>20.4f}{860.0:>20.4f}{17.10:>20.4f}{0.400:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2680:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Delamination Microcracking Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_DELAMINATION_MICROCRACKING_RATE/2680
Ladeveze Transverse Delamination Microcracking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2680 in model.fail_ladtransversedelaminationmicrocrackingrates
    ftdmcr = model.fail_ladtransversedelaminationmicrocrackingrates[2680]
    assert pytest.approx(ftdmcr.sigma_tdmcr0) == 1060.0
    assert pytest.approx(ftdmcr.sigma_tdmcrc) == 3180.0
    assert pytest.approx(ftdmcr.gamma_tdmcr) == 860.0
    assert pytest.approx(ftdmcr.p_tdmcr) == 17.10
    assert pytest.approx(ftdmcr.d_tdmcr_max) == 0.400
    assert ftdmcr.ifail_sh == 1
    assert ftdmcr.ifail_so == 2
    assert ftdmcr.fail_id == 2680
    assert pytest.approx(ftdmcr.sigma_tdmc0) == 1060.0
    assert pytest.approx(ftdmcr.sigma_tdmcc) == 3180.0
    assert pytest.approx(ftdmcr.gamma_tdmc) == 860.0
    assert pytest.approx(ftdmcr.p_tdmc) == 17.10
    assert pytest.approx(ftdmcr.d_tdmc_max) == 0.400
    assert pytest.approx(ftdmcr.sigma_tdm0) == 1060.0
    assert pytest.approx(ftdmcr.sigma_tdmc) == 3180.0
    assert pytest.approx(ftdmcr.gamma_tdm) == 860.0
    assert pytest.approx(ftdmcr.p_tdm) == 17.10
    assert pytest.approx(ftdmcr.d_tdm_max) == 0.400
    assert pytest.approx(ftdmcr.sigma_timcr0) == 1060.0
    assert pytest.approx(ftdmcr.sigma_timcrc) == 3180.0
    assert pytest.approx(ftdmcr.gamma_timcr) == 860.0
    assert pytest.approx(ftdmcr.p_timcr) == 17.10
    assert pytest.approx(ftdmcr.d_timcr_max) == 0.400
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_DELAMINATION_MICROCRACKING_RATE"


def test_m447_fail_lad_transverse_delamination_microcracking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Delamination Microcracking Rate Free Format Test
/FAIL/LAD_TRANSVERSE_DELAMINATION_MICROCRACKING_RATE/2681
1080.0, 3240.0, 880.0, 17.40, 0.390
1, 1
2681
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2681 in model.fail_ladtransversedelaminationmicrocrackingrates
    ftdmcr = model.fail_ladtransversedelaminationmicrocrackingrates[2681]
    assert pytest.approx(ftdmcr.sigma_tdmcr0) == 1080.0
    assert pytest.approx(ftdmcr.sigma_tdmcrc) == 3240.0
    assert pytest.approx(ftdmcr.gamma_tdmcr) == 880.0
    assert pytest.approx(ftdmcr.p_tdmcr) == 17.40
    assert pytest.approx(ftdmcr.d_tdmcr_max) == 0.390
    assert ftdmcr.fail_id == 2681


def test_m447_fail_lad_transverse_delamination_microcracking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Delamination Microcracking Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_DELAMINATION_MICROCRACKING_RATE/2682
1010.0, 3030.0, 810.0, 15.90, 0.440
1, 1
/FAIL/LAD_TRANSVERSE_DELAMINATION_MICROCRACK_RATE/2683
1010.0, 3030.0, 810.0, 15.90, 0.440
1, 1
/FAIL/LADEVEZE_TRANSVERSE_DELAMINATION_MICROCRACK_RATE/2684
1010.0, 3030.0, 810.0, 15.90, 0.440
1, 1
/FAIL/LAD_TRANSVERSE_DELAM_MICROCRACKING_RATE/2685
1010.0, 3030.0, 810.0, 15.90, 0.440
1, 1
/FAIL/LADEVEZE_TRANSVERSE_DELAM_MICROCRACKING_RATE/2686
1010.0, 3030.0, 810.0, 15.90, 0.440
1, 1
/FAIL/LAD_TRANSVERSE_DELAM_MICROCRACK_RATE/2687
1010.0, 3030.0, 810.0, 15.90, 0.440
1, 1
/FAIL/LADEVEZE_TRANSVERSE_DELAM_MICROCRACK_RATE/2688
1010.0, 3030.0, 810.0, 15.90, 0.440
1, 1
/FAIL/LAD_TDMCR/2689
1010.0, 3030.0, 810.0, 15.90, 0.440
1, 1
/FAIL/LAD_TDMCR_MODEL/2690
1010.0, 3030.0, 810.0, 15.90, 0.440
1, 1
/FAIL/LAD_TDMCR_LAW/2691
1010.0, 3030.0, 810.0, 15.90, 0.440
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_DELAMINATION_MICROCRACKING/2692
1010.0, 3030.0, 810.0, 15.90, 0.440
1, 1
/FAIL/LAD_TRANSVERSE_INTERLAMINAR_MICROCRACKING_RATE/2693
1010.0, 3030.0, 810.0, 15.90, 0.440
1, 1
/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_MICROCRACKING_RATE/2694
1010.0, 3030.0, 810.0, 15.90, 0.440
1, 1
/FAIL/LAD_TRANSVERSE_INTERLAMINAR_MICROCRACK_RATE/2695
1010.0, 3030.0, 810.0, 15.90, 0.440
1, 1
/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_MICROCRACK_RATE/2696
1010.0, 3030.0, 810.0, 15.90, 0.440
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for mid in range(2682, 2697):
        assert mid in model.fail_ladtransversedelaminationmicrocrackingrates
        f = model.fail_ladtransversedelaminationmicrocrackingrates[mid]
        assert pytest.approx(f.sigma_tdmcr0) == 1010.0
        assert pytest.approx(f.sigma_tdmcrc) == 3030.0


def test_m447_fail_lad_transverse_delamination_microcracking_rate_properties():
    from pyradioss.model.entities import FailLadTransverseDelaminationMicrocrackingRate
    f = FailLadTransverseDelaminationMicrocrackingRate(mat_id=1)
    f.sigma_tdmc0 = 1120.0
    assert pytest.approx(f.sigma_tdmcr0) == 1120.0
    assert pytest.approx(f.sigma_tdmc0) == 1120.0

    f.sigma_tdmcc = 3360.0
    assert pytest.approx(f.sigma_tdmcrc) == 3360.0
    assert pytest.approx(f.sigma_tdmcc) == 3360.0

    f.gamma_tdmc = 900.0
    assert pytest.approx(f.gamma_tdmcr) == 900.0
    assert pytest.approx(f.gamma_tdmc) == 900.0

    f.p_tdmc = 17.20
    assert pytest.approx(f.p_tdmcr) == 17.20
    assert pytest.approx(f.p_tdmc) == 17.20

    f.d_tdmc_max = 0.380
    assert pytest.approx(f.d_tdmcr_max) == 0.380
    assert pytest.approx(f.d_tdmc_max) == 0.380

    f.sigma_timcr0 = 1180.0
    assert pytest.approx(f.sigma_tdmcr0) == 1180.0
    assert pytest.approx(f.sigma_timcr0) == 1180.0

    f.sigma_timcrc = 3540.0
    assert pytest.approx(f.sigma_tdmcrc) == 3540.0
    assert pytest.approx(f.sigma_timcrc) == 3540.0

    f.gamma_timcr = 940.0
    assert pytest.approx(f.gamma_tdmcr) == 940.0
    assert pytest.approx(f.gamma_timcr) == 940.0

    f.p_timcr = 18.20
    assert pytest.approx(f.p_tdmcr) == 18.20
    assert pytest.approx(f.p_timcr) == 18.20

    f.d_timcr_max = 0.350
    assert pytest.approx(f.d_tdmcr_max) == 0.350
    assert pytest.approx(f.d_timcr_max) == 0.350


def test_m447_eng_electrothermoflexomagnetochiralplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00205:>20.6f}{68:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Chiral Plasmon Polaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Chiral Plasmon Polaritonic Resonance Energy Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetochiralplasmonicpolaritonic_resonance_energies[1]
    assert pytest.approx(r.dt_etfcplp) == 0.00205
    assert pytest.approx(r.dt_etfplp) == 0.00205
    assert r.sens_id == 68


def test_m447_eng_electrothermoflexomagnetochiralplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Chiral Plasmon Polaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
0.00305, 69
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetochiralplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(r.dt_etfcplp) == 0.00305
    assert r.sens_id == 69


def test_m447_eng_electrothermoflexomagnetochiralplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Chiral Plasmon Polaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_PLASMON_POLARITON_RES_WORK/3
0.00405, 70
/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALPLASMONICPOLARITONICRESONANCE/4
0.00405, 70
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/5
0.00405, 70
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALPLASMONICPOLARITONIC_RESONANCE/6
0.00405, 70
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICCHIRALPOLARITONIC_RESONANCE_ENERGY/7
0.00405, 70
/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_CHIRAL_POLARITON_RES_WORK/8
0.00405, 70
/ENG/EELECTROTHERMOFLEXOMAGNETOPLASMONICCHIRALPOLARITONICRESONANCE/9
0.00405, 70
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICCHIRALPOLARITONIC_RESONANCE_DISSIPATION/10
0.00405, 70
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPLASMONICCHIRALPOLARITONIC_RESONANCE/11
0.00405, 70
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for rid in range(3, 12):
        assert rid in model.eng_electrothermoflexomagnetochiralplasmonicpolaritonic_resonance_energies
        r = model.eng_electrothermoflexomagnetochiralplasmonicpolaritonic_resonance_energies[rid]
        assert pytest.approx(r.dt_etfcplp) == 0.00405
        assert r.sens_id == 70


def test_m447_lagmul_stack_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{3.5e6:>20.4f}{6:>10d}{2.0e-5:>20.6e}"
    c2 = f"{20.0:>20.4f}{22.0:>20.4f}{68.0:>20.4f}{7.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Stack Spinor Spatial Linkage Joint Fixed Format Test
2022 0
/STACK_SPINOR_SPATIAL_LINKAGE_JOINT/1
Stack Spinor Spatial Mechanism Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_stack_spinor_spatial_linkage_joints
    j = model.lagmul_stack_spinor_spatial_linkage_joints[1]
    assert j.node1 == 101
    assert j.node2 == 102
    assert j.node3 == 103
    assert pytest.approx(j.stiff) == 3.5e6
    assert j.skew_id == 6
    assert pytest.approx(j.tol) == 2.0e-5
    assert pytest.approx(j.link_len_a) == 20.0
    assert pytest.approx(j.link_len_b) == 22.0
    assert pytest.approx(j.twist_angle_alpha) == 68.0
    assert pytest.approx(j.offset_distance_s) == 7.0
    assert pytest.approx(j.offset_distance_r) == 7.0
    assert pytest.approx(j.offset_distance_v) == 7.0
    assert pytest.approx(j.offset_distance_h) == 7.0
    assert pytest.approx(j.offset_distance_u) == 7.0
    assert pytest.approx(j.offset_distance_f) == 7.0


def test_m447_lagmul_stack_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Stack Spinor Spatial Linkage Joint Free Format Test
/LAGMUL/STACK_SPINOR_SPATIAL_LINKAGE_JOINT/2
201, 202, 203, 4.5e6, 7, 3.0e-5
23.0, 25.0, 80.0, 8.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_stack_spinor_spatial_linkage_joints
    j = model.lagmul_stack_spinor_spatial_linkage_joints[2]
    assert j.node1 == 201
    assert j.node2 == 202
    assert j.node3 == 203
    assert pytest.approx(j.stiff) == 4.5e6
    assert j.skew_id == 7
    assert pytest.approx(j.tol) == 3.0e-5
    assert pytest.approx(j.link_len_a) == 23.0
    assert pytest.approx(j.link_len_b) == 25.0
    assert pytest.approx(j.twist_angle_alpha) == 80.0
    assert pytest.approx(j.offset_distance_s) == 8.0


def test_m447_lagmul_stack_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Stack Spinor Spatial Linkage Joint Aliases Test
/LAGMUL/STACK_SPINOR_SPATIAL_LINKAGE/3
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/STACK_SPINOR_SPATIAL_LINKAGE/4
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/STACK_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM/5
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/STACK_SPINOR_SPATIAL_SYMMETRIC_MECHANISM/6
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/STACK_SPINOR_SPATIAL_6R_MECHANISM/7
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/STACK_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM/8
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/STACK_TWISTOR_SPATIAL_LINKAGE_JOINT/9
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/STACK_TWISTOR_SPATIAL_LINKAGE_JOINT/10
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/STACK_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/11
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/STACK_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/12
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/STACK_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/13
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/STACK_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/14
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(3, 15):
        assert jid in model.lagmul_stack_spinor_spatial_linkage_joints
        j = model.lagmul_stack_spinor_spatial_linkage_joints[jid]
        assert j.node1 == 301
        assert j.node2 == 302
        assert j.node3 == 303


def test_m447_sensor_spring_bending_pop_rate_fixed(tmp_path: Path):
    c1 = f"{10:>10d}{1.94e5:>20.4f}{0.0024:>20.6f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Bending Pop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_BENDING_POP_RATE/1
Bending Pop Rate Sensor Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_bending_pop_rates
    s = model.sensor_spring_bending_pop_rates[1]
    assert s.spring_id == 10
    assert pytest.approx(s.jbend_pop_max) == 1.94e5
    assert pytest.approx(s.t_delay) == 0.0024
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_BENDING_POP_RATE"


def test_m447_sensor_spring_bending_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Bending Pop Rate Free Format Test
/SENSOR/SPRING_BENDING_POP_RATE/2
20, 2.94e5, 0.0034
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_bending_pop_rates
    s = model.sensor_spring_bending_pop_rates[2]
    assert s.spring_id == 20
    assert pytest.approx(s.jbend_pop_max) == 2.94e5
    assert pytest.approx(s.t_delay) == 0.0034


def test_m447_sensor_spring_bending_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Bending Pop Rate Aliases Test
/SENSOR/SPRING_BEND_POP_RATE/3
30, 3.5e5, 0.001
/SENSOR/SPRING_BENDING_POP/4
30, 3.5e5, 0.001
/SENSOR/SPRING_BEND_POP/5
30, 3.5e5, 0.001
/SENSOR/SPRING_POP_RATE_BENDING/6
30, 3.5e5, 0.001
/SENSOR/SPRING_POP_RATE_BEND/7
30, 3.5e5, 0.001
/SENSOR/BENDING_POP_RATE_SPRING/8
30, 3.5e5, 0.001
/SENSOR/SPRING_POP_BENDING/9
30, 3.5e5, 0.001
/SENSOR/SPRING_POP_BEND/10
30, 3.5e5, 0.001
/SENSOR/SPRING_FLEX_POP_RATE/11
30, 3.5e5, 0.001
/SENSOR/SPRING_FLEX_POP/12
30, 3.5e5, 0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(3, 13):
        assert sid in model.sensor_spring_bending_pop_rates
        s = model.sensor_spring_bending_pop_rates[sid]
        assert s.spring_id == 30
        assert pytest.approx(s.jbend_pop_max) == 3.5e5


def test_m447_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_TRANSVERSE_DELAMINATION_MICROCRACKING_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/STACK_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_BENDING_POP_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4
