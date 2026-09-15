"""Tests for Milestone M446: LadDynamicDelaminationMicrocrackingRate Failure Model, EngElectrothermoflexomagnetophononicexcitonicmagnonicplasmonicpolaritonicResonanceEnergy, LagmulSheafSpinorSpatialLinkageJoint, and SensorSpringTorsionalPopRate."""

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


def test_m446_fail_lad_dynamic_delamination_microcracking_rate_fixed(tmp_path: Path):
    c1 = f"{1050.0:>20.4f}{3150.0:>20.4f}{850.0:>20.4f}{17.00:>20.4f}{0.405:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2660:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Delamination Microcracking Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_DELAMINATION_MICROCRACKING_RATE/2660
Ladeveze Dynamic Delamination Microcracking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2660 in model.fail_laddynamicdelaminationmicrocrackingrates
    fddmcr = model.fail_laddynamicdelaminationmicrocrackingrates[2660]
    assert pytest.approx(fddmcr.sigma_ddmcr0) == 1050.0
    assert pytest.approx(fddmcr.sigma_ddmcrc) == 3150.0
    assert pytest.approx(fddmcr.gamma_ddmcr) == 850.0
    assert pytest.approx(fddmcr.p_ddmcr) == 17.00
    assert pytest.approx(fddmcr.d_ddmcr_max) == 0.405
    assert fddmcr.ifail_sh == 1
    assert fddmcr.ifail_so == 2
    assert fddmcr.fail_id == 2660
    assert pytest.approx(fddmcr.sigma_ddmc0) == 1050.0
    assert pytest.approx(fddmcr.sigma_ddmcc) == 3150.0
    assert pytest.approx(fddmcr.gamma_ddmc) == 850.0
    assert pytest.approx(fddmcr.p_ddmc) == 17.00
    assert pytest.approx(fddmcr.d_ddmc_max) == 0.405
    assert pytest.approx(fddmcr.sigma_ddm0) == 1050.0
    assert pytest.approx(fddmcr.sigma_ddmc) == 3150.0
    assert pytest.approx(fddmcr.gamma_ddm) == 850.0
    assert pytest.approx(fddmcr.p_ddm) == 17.00
    assert pytest.approx(fddmcr.d_ddm_max) == 0.405
    assert pytest.approx(fddmcr.sigma_dimcr0) == 1050.0
    assert pytest.approx(fddmcr.sigma_dimcrc) == 3150.0
    assert pytest.approx(fddmcr.gamma_dimcr) == 850.0
    assert pytest.approx(fddmcr.p_dimcr) == 17.00
    assert pytest.approx(fddmcr.d_dimcr_max) == 0.405
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_DELAMINATION_MICROCRACKING_RATE"


def test_m446_fail_lad_dynamic_delamination_microcracking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Delamination Microcracking Rate Free Format Test
/FAIL/LAD_DYNAMIC_DELAMINATION_MICROCRACKING_RATE/2661
1070.0, 3210.0, 870.0, 17.30, 0.395
1, 1
2661
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2661 in model.fail_laddynamicdelaminationmicrocrackingrates
    fddmcr = model.fail_laddynamicdelaminationmicrocrackingrates[2661]
    assert pytest.approx(fddmcr.sigma_ddmcr0) == 1070.0
    assert pytest.approx(fddmcr.sigma_ddmcrc) == 3210.0
    assert pytest.approx(fddmcr.gamma_ddmcr) == 870.0
    assert pytest.approx(fddmcr.p_ddmcr) == 17.30
    assert pytest.approx(fddmcr.d_ddmcr_max) == 0.395
    assert fddmcr.fail_id == 2661


def test_m446_fail_lad_dynamic_delamination_microcracking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Delamination Microcracking Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_DELAMINATION_MICROCRACKING_RATE/2662
1000.0, 3000.0, 800.0, 15.80, 0.445
1, 1
/FAIL/LAD_DYNAMIC_DELAMINATION_MICROCRACK_RATE/2663
1000.0, 3000.0, 800.0, 15.80, 0.445
1, 1
/FAIL/LADEVEZE_DYNAMIC_DELAMINATION_MICROCRACK_RATE/2664
1000.0, 3000.0, 800.0, 15.80, 0.445
1, 1
/FAIL/LAD_DYNAMIC_DELAM_MICROCRACKING_RATE/2665
1000.0, 3000.0, 800.0, 15.80, 0.445
1, 1
/FAIL/LADEVEZE_DYNAMIC_DELAM_MICROCRACKING_RATE/2666
1000.0, 3000.0, 800.0, 15.80, 0.445
1, 1
/FAIL/LAD_DYNAMIC_DELAM_MICROCRACK_RATE/2667
1000.0, 3000.0, 800.0, 15.80, 0.445
1, 1
/FAIL/LADEVEZE_DYNAMIC_DELAM_MICROCRACK_RATE/2668
1000.0, 3000.0, 800.0, 15.80, 0.445
1, 1
/FAIL/LAD_DDMCR/2669
1000.0, 3000.0, 800.0, 15.80, 0.445
1, 1
/FAIL/LAD_DDMCR_MODEL/2670
1000.0, 3000.0, 800.0, 15.80, 0.445
1, 1
/FAIL/LAD_DDMCR_LAW/2671
1000.0, 3000.0, 800.0, 15.80, 0.445
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_DELAMINATION_MICROCRACKING/2672
1000.0, 3000.0, 800.0, 15.80, 0.445
1, 1
/FAIL/LAD_DYNAMIC_INTERLAMINAR_MICROCRACKING_RATE/2673
1000.0, 3000.0, 800.0, 15.80, 0.445
1, 1
/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_MICROCRACKING_RATE/2674
1000.0, 3000.0, 800.0, 15.80, 0.445
1, 1
/FAIL/LAD_DYNAMIC_INTERLAMINAR_MICROCRACK_RATE/2675
1000.0, 3000.0, 800.0, 15.80, 0.445
1, 1
/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_MICROCRACK_RATE/2676
1000.0, 3000.0, 800.0, 15.80, 0.445
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for mid in range(2662, 2677):
        assert mid in model.fail_laddynamicdelaminationmicrocrackingrates
        f = model.fail_laddynamicdelaminationmicrocrackingrates[mid]
        assert pytest.approx(f.sigma_ddmcr0) == 1000.0
        assert pytest.approx(f.sigma_ddmcrc) == 3000.0


def test_m446_fail_lad_dynamic_delamination_microcracking_rate_properties():
    from pyradioss.model.entities import FailLadDynamicDelaminationMicrocrackingRate
    f = FailLadDynamicDelaminationMicrocrackingRate(mat_id=1)
    f.sigma_ddmc0 = 1110.0
    assert pytest.approx(f.sigma_ddmcr0) == 1110.0
    assert pytest.approx(f.sigma_ddmc0) == 1110.0

    f.sigma_ddmcc = 3330.0
    assert pytest.approx(f.sigma_ddmcrc) == 3330.0
    assert pytest.approx(f.sigma_ddmcc) == 3330.0

    f.gamma_ddmc = 890.0
    assert pytest.approx(f.gamma_ddmcr) == 890.0
    assert pytest.approx(f.gamma_ddmc) == 890.0

    f.p_ddmc = 17.10
    assert pytest.approx(f.p_ddmcr) == 17.10
    assert pytest.approx(f.p_ddmc) == 17.10

    f.d_ddmc_max = 0.385
    assert pytest.approx(f.d_ddmcr_max) == 0.385
    assert pytest.approx(f.d_ddmc_max) == 0.385

    f.sigma_dimcr0 = 1170.0
    assert pytest.approx(f.sigma_ddmcr0) == 1170.0
    assert pytest.approx(f.sigma_dimcr0) == 1170.0

    f.sigma_dimcrc = 3510.0
    assert pytest.approx(f.sigma_ddmcrc) == 3510.0
    assert pytest.approx(f.sigma_dimcrc) == 3510.0

    f.gamma_dimcr = 930.0
    assert pytest.approx(f.gamma_ddmcr) == 930.0
    assert pytest.approx(f.gamma_dimcr) == 930.0

    f.p_dimcr = 18.10
    assert pytest.approx(f.p_ddmcr) == 18.10
    assert pytest.approx(f.p_dimcr) == 18.10

    f.d_dimcr_max = 0.355
    assert pytest.approx(f.d_ddmcr_max) == 0.355
    assert pytest.approx(f.d_dimcr_max) == 0.355


def test_m446_eng_electrothermoflexomagnetophononicexcitonicmagnonicplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00195:>20.6f}{65:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Full Hybrid Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICEXCITONICMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Full Hybrid Resonance Energy Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetophononicexcitonicmagnonicplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetophononicexcitonicmagnonicplasmonicpolaritonic_resonance_energies[1]
    assert pytest.approx(r.dt_etfpexmplp) == 0.00195
    assert pytest.approx(r.dt_etfplp) == 0.00195
    assert r.sens_id == 65


def test_m446_eng_electrothermoflexomagnetophononicexcitonicmagnonicplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Full Hybrid Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICEXCITONICMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
0.00295, 66
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetophononicexcitonicmagnonicplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetophononicexcitonicmagnonicplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(r.dt_etfpexmplp) == 0.00295
    assert r.sens_id == 66


def test_m446_eng_electrothermoflexomagnetophononicexcitonicmagnonicplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Full Hybrid Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_PHONON_EXCITON_MAGNON_PLASMON_POLARITON_RES_WORK/3
0.00395, 67
/ENG/EELECTROTHERMOFLEXOMAGNETOPHONONICEXCITONICMAGNONICPLASMONICPOLARITONICRESONANCE/4
0.00395, 67
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICEXCITONICMAGNONICPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/5
0.00395, 67
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPHONONICEXCITONICMAGNONICPLASMONICPOLARITONIC_RESONANCE/6
0.00395, 67
/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICEXCITONICPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/7
0.00395, 67
/ENG/ELECTRO_THERM_FLEXO_MAG_MAGNON_EXCITON_PHONON_PLASMON_POLARITON_RES_WORK/8
0.00395, 67
/ENG/EELECTROTHERMOFLEXOMAGNETOMAGNONICEXCITONICPHONONICPLASMONICPOLARITONICRESONANCE/9
0.00395, 67
/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICEXCITONICPHONONICPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/10
0.00395, 67
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOMAGNONICEXCITONICPHONONICPLASMONICPOLARITONIC_RESONANCE/11
0.00395, 67
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for rid in range(3, 12):
        assert rid in model.eng_electrothermoflexomagnetophononicexcitonicmagnonicplasmonicpolaritonic_resonance_energies
        r = model.eng_electrothermoflexomagnetophononicexcitonicmagnonicplasmonicpolaritonic_resonance_energies[rid]
        assert pytest.approx(r.dt_etfpexmplp) == 0.00395
        assert r.sens_id == 67


def test_m446_lagmul_sheaf_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{3.4e6:>20.4f}{6:>10d}{1.9e-5:>20.6e}"
    c2 = f"{19.0:>20.4f}{21.0:>20.4f}{66.0:>20.4f}{6.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sheaf Spinor Spatial Linkage Joint Fixed Format Test
2022 0
/SHEAF_SPINOR_SPATIAL_LINKAGE_JOINT/1
Sheaf Spinor Spatial Mechanism Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_sheaf_spinor_spatial_linkage_joints
    j = model.lagmul_sheaf_spinor_spatial_linkage_joints[1]
    assert j.node1 == 101
    assert j.node2 == 102
    assert j.node3 == 103
    assert pytest.approx(j.stiff) == 3.4e6
    assert j.skew_id == 6
    assert pytest.approx(j.tol) == 1.9e-5
    assert pytest.approx(j.link_len_a) == 19.0
    assert pytest.approx(j.link_len_b) == 21.0
    assert pytest.approx(j.twist_angle_alpha) == 66.0
    assert pytest.approx(j.offset_distance_s) == 6.5
    assert pytest.approx(j.offset_distance_r) == 6.5
    assert pytest.approx(j.offset_distance_v) == 6.5
    assert pytest.approx(j.offset_distance_h) == 6.5
    assert pytest.approx(j.offset_distance_u) == 6.5
    assert pytest.approx(j.offset_distance_f) == 6.5


def test_m446_lagmul_sheaf_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sheaf Spinor Spatial Linkage Joint Free Format Test
/LAGMUL/SHEAF_SPINOR_SPATIAL_LINKAGE_JOINT/2
201, 202, 203, 4.4e6, 7, 2.9e-5
22.0, 24.0, 78.0, 7.5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_sheaf_spinor_spatial_linkage_joints
    j = model.lagmul_sheaf_spinor_spatial_linkage_joints[2]
    assert j.node1 == 201
    assert j.node2 == 202
    assert j.node3 == 203
    assert pytest.approx(j.stiff) == 4.4e6
    assert j.skew_id == 7
    assert pytest.approx(j.tol) == 2.9e-5
    assert pytest.approx(j.link_len_a) == 22.0
    assert pytest.approx(j.link_len_b) == 24.0
    assert pytest.approx(j.twist_angle_alpha) == 78.0
    assert pytest.approx(j.offset_distance_s) == 7.5


def test_m446_lagmul_sheaf_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sheaf Spinor Spatial Linkage Joint Aliases Test
/LAGMUL/SHEAF_SPINOR_SPATIAL_LINKAGE/3
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/SHEAF_SPINOR_SPATIAL_LINKAGE/4
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/SHEAF_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM/5
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/SHEAF_SPINOR_SPATIAL_SYMMETRIC_MECHANISM/6
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/SHEAF_SPINOR_SPATIAL_6R_MECHANISM/7
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/SHEAF_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM/8
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/SHEAF_TWISTOR_SPATIAL_LINKAGE_JOINT/9
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/SHEAF_TWISTOR_SPATIAL_LINKAGE_JOINT/10
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/SHEAF_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/11
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/SHEAF_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/12
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/SHEAF_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/13
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/SHEAF_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/14
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(3, 15):
        assert jid in model.lagmul_sheaf_spinor_spatial_linkage_joints
        j = model.lagmul_sheaf_spinor_spatial_linkage_joints[jid]
        assert j.node1 == 301
        assert j.node2 == 302
        assert j.node3 == 303


def test_m446_sensor_spring_torsional_pop_rate_fixed(tmp_path: Path):
    c1 = f"{10:>10d}{1.92e5:>20.4f}{0.0022:>20.6f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Pop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_POP_RATE/1
Torsional Pop Rate Sensor Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_pop_rates
    s = model.sensor_spring_torsional_pop_rates[1]
    assert s.spring_id == 10
    assert pytest.approx(s.jtors_pop_max) == 1.92e5
    assert pytest.approx(s.t_delay) == 0.0022
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_POP_RATE"


def test_m446_sensor_spring_torsional_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Torsional Pop Rate Free Format Test
/SENSOR/SPRING_TORSIONAL_POP_RATE/2
20, 2.92e5, 0.0032
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_torsional_pop_rates
    s = model.sensor_spring_torsional_pop_rates[2]
    assert s.spring_id == 20
    assert pytest.approx(s.jtors_pop_max) == 2.92e5
    assert pytest.approx(s.t_delay) == 0.0032


def test_m446_sensor_spring_torsional_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Torsional Pop Rate Aliases Test
/SENSOR/SPRING_TORS_POP_RATE/3
30, 3.4e5, 0.001
/SENSOR/SPRING_TORSIONAL_POP/4
30, 3.4e5, 0.001
/SENSOR/SPRING_TORS_POP/5
30, 3.4e5, 0.001
/SENSOR/SPRING_POP_RATE_TORSIONAL/6
30, 3.4e5, 0.001
/SENSOR/SPRING_POP_RATE_TORS/7
30, 3.4e5, 0.001
/SENSOR/TORSIONAL_POP_RATE_SPRING/8
30, 3.4e5, 0.001
/SENSOR/SPRING_POP_TORSIONAL/9
30, 3.4e5, 0.001
/SENSOR/SPRING_POP_TORS/10
30, 3.4e5, 0.001
/SENSOR/SPRING_TWIST_POP_RATE/11
30, 3.4e5, 0.001
/SENSOR/SPRING_TWIST_POP/12
30, 3.4e5, 0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(3, 13):
        assert sid in model.sensor_spring_torsional_pop_rates
        s = model.sensor_spring_torsional_pop_rates[sid]
        assert s.spring_id == 30
        assert pytest.approx(s.jtors_pop_max) == 3.4e5


def test_m446_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_DYNAMIC_DELAMINATION_MICROCRACKING_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICEXCITONICMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/SHEAF_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_TORSIONAL_POP_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4
