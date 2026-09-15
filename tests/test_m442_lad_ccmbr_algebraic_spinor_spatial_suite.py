"""Tests for Milestone M442: LadCoupledCoreMicrobucklingRate Failure Model, EngElectrothermoflexomagnetoexcitonicplasmonicpolaritonicResonanceEnergy, LagmulAlgebraicSpinorSpatialLinkageJoint, and SensorSpringTotalAngularCrackleRate."""

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


def test_m442_fail_lad_coupled_core_microbuckling_rate_fixed(tmp_path: Path):
    c1 = f"{990.0:>20.4f}{2970.0:>20.4f}{790.0:>20.4f}{16.10:>20.4f}{0.440:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2580:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Core Microbuckling Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLED_CORE_MICROBUCKLING_RATE/2580
Ladeveze Coupled Core Microbuckling Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2580 in model.fail_ladcoupledcoremicrobucklingrates
    fccmbr = model.fail_ladcoupledcoremicrobucklingrates[2580]
    assert pytest.approx(fccmbr.sigma_ccmbr0) == 990.0
    assert pytest.approx(fccmbr.sigma_ccmbrc) == 2970.0
    assert pytest.approx(fccmbr.gamma_ccmbr) == 790.0
    assert pytest.approx(fccmbr.p_ccmbr) == 16.10
    assert pytest.approx(fccmbr.d_ccmbr_max) == 0.440
    assert fccmbr.ifail_sh == 1
    assert fccmbr.ifail_so == 2
    assert fccmbr.fail_id == 2580
    assert pytest.approx(fccmbr.sigma_ccmbk0) == 990.0
    assert pytest.approx(fccmbr.sigma_ccmbkc) == 2970.0
    assert pytest.approx(fccmbr.gamma_ccmbk) == 790.0
    assert pytest.approx(fccmbr.p_ccmbk) == 16.10
    assert pytest.approx(fccmbr.d_ccmbk_max) == 0.440
    assert pytest.approx(fccmbr.sigma_ccmk0) == 990.0
    assert pytest.approx(fccmbr.sigma_ccmkc) == 2970.0
    assert pytest.approx(fccmbr.gamma_ccmk) == 790.0
    assert pytest.approx(fccmbr.p_ccmk) == 16.10
    assert pytest.approx(fccmbr.d_ccmk_max) == 0.440
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLED_CORE_MICROBUCKLING_RATE"


def test_m442_fail_lad_coupled_core_microbuckling_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Core Microbuckling Rate Free Format Test
/FAIL/LAD_COUPLED_CORE_MICROBUCKLING_RATE/2581
1010.0, 3030.0, 810.0, 16.40, 0.430
1, 1
2581
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2581 in model.fail_ladcoupledcoremicrobucklingrates
    fccmbr = model.fail_ladcoupledcoremicrobucklingrates[2581]
    assert pytest.approx(fccmbr.sigma_ccmbr0) == 1010.0
    assert pytest.approx(fccmbr.sigma_ccmbrc) == 3030.0
    assert pytest.approx(fccmbr.gamma_ccmbr) == 810.0
    assert pytest.approx(fccmbr.p_ccmbr) == 16.40
    assert pytest.approx(fccmbr.d_ccmbr_max) == 0.430
    assert fccmbr.fail_id == 2581


def test_m442_fail_lad_coupled_core_microbuckling_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Core Microbuckling Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_CORE_MICROBUCKLING_RATE/2582
960.0, 2880.0, 760.0, 15.30, 0.470
1, 1
/FAIL/LAD_COUPLE_CORE_MICROBUCKLING_RATE/2583
960.0, 2880.0, 760.0, 15.30, 0.470
1, 1
/FAIL/LADEVEZE_COUPLE_CORE_MICROBUCKLING_RATE/2584
960.0, 2880.0, 760.0, 15.30, 0.470
1, 1
/FAIL/LAD_COUPLED_CORE_MICROBUCKLE_RATE/2585
960.0, 2880.0, 760.0, 15.30, 0.470
1, 1
/FAIL/LAD_COUPLE_CORE_MICROBUCKLE_RATE/2586
960.0, 2880.0, 760.0, 15.30, 0.470
1, 1
/FAIL/LAD_CCMBR/2587
960.0, 2880.0, 760.0, 15.30, 0.470
1, 1
/FAIL/LAD_CCMBR_MODEL/2588
960.0, 2880.0, 760.0, 15.30, 0.470
1, 1
/FAIL/LAD_CCMBR_LAW/2589
960.0, 2880.0, 760.0, 15.30, 0.470
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_CORE_MICROBUCKLING/2590
960.0, 2880.0, 760.0, 15.30, 0.470
1, 1
/FAIL/LAD_COUPLED_CORE_MICROKINKING_RATE/2591
960.0, 2880.0, 760.0, 15.30, 0.470
1, 1
/FAIL/LADEVEZE_COUPLED_CORE_MICROKINKING_RATE/2592
960.0, 2880.0, 760.0, 15.30, 0.470
1, 1
/FAIL/LAD_COUPLE_CORE_MICROKINKING_RATE/2593
960.0, 2880.0, 760.0, 15.30, 0.470
1, 1
/FAIL/LADEVEZE_COUPLE_CORE_MICROKINKING_RATE/2594
960.0, 2880.0, 760.0, 15.30, 0.470
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for mid in range(2582, 2595):
        assert mid in model.fail_ladcoupledcoremicrobucklingrates
        f = model.fail_ladcoupledcoremicrobucklingrates[mid]
        assert pytest.approx(f.sigma_ccmbr0) == 960.0
        assert pytest.approx(f.sigma_ccmbrc) == 2880.0


def test_m442_fail_lad_coupled_core_microbuckling_rate_properties():
    from pyradioss.model.entities import FailLadCoupledCoreMicrobucklingRate
    f = FailLadCoupledCoreMicrobucklingRate(mat_id=1)
    f.sigma_ccmbk0 = 1060.0
    assert pytest.approx(f.sigma_ccmbr0) == 1060.0
    assert pytest.approx(f.sigma_ccmbk0) == 1060.0

    f.sigma_ccmbkc = 3180.0
    assert pytest.approx(f.sigma_ccmbrc) == 3180.0
    assert pytest.approx(f.sigma_ccmbkc) == 3180.0

    f.gamma_ccmbk = 840.0
    assert pytest.approx(f.gamma_ccmbr) == 840.0
    assert pytest.approx(f.gamma_ccmbk) == 840.0

    f.p_ccmbk = 16.50
    assert pytest.approx(f.p_ccmbr) == 16.50
    assert pytest.approx(f.p_ccmbk) == 16.50

    f.d_ccmbk_max = 0.410
    assert pytest.approx(f.d_ccmbr_max) == 0.410
    assert pytest.approx(f.d_ccmbk_max) == 0.410

    f.sigma_ccmk0 = 1120.0
    assert pytest.approx(f.sigma_ccmbr0) == 1120.0
    assert pytest.approx(f.sigma_ccmk0) == 1120.0

    f.sigma_ccmkc = 3360.0
    assert pytest.approx(f.sigma_ccmbrc) == 3360.0
    assert pytest.approx(f.sigma_ccmkc) == 3360.0

    f.gamma_ccmk = 880.0
    assert pytest.approx(f.gamma_ccmbr) == 880.0
    assert pytest.approx(f.gamma_ccmk) == 880.0

    f.p_ccmk = 17.50
    assert pytest.approx(f.p_ccmbr) == 17.50
    assert pytest.approx(f.p_ccmk) == 17.50

    f.d_ccmk_max = 0.380
    assert pytest.approx(f.d_ccmbr_max) == 0.380
    assert pytest.approx(f.d_ccmk_max) == 0.380


def test_m442_eng_electrothermoflexomagnetoexcitonicplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00155:>20.6f}{52:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Exciton Plasmon Polaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Exciton Plasmon Polaritonic Resonance Energy Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetoexcitonicplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetoexcitonicplasmonicpolaritonic_resonance_energies[1]
    assert pytest.approx(r.dt_etfexplp) == 0.00155
    assert pytest.approx(r.dt_etfplp) == 0.00155
    assert r.sens_id == 52


def test_m442_eng_electrothermoflexomagnetoexcitonicplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Exciton Plasmon Polaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
0.00255, 53
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetoexcitonicplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetoexcitonicplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(r.dt_etfexplp) == 0.00255
    assert r.sens_id == 53


def test_m442_eng_electrothermoflexomagnetoexcitonicplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Exciton Plasmon Polaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_PLASMON_POLARITON_RES_WORK/3
0.00355, 54
/ENG/EELECTROTHERMOFLEXOMAGNETOEXCITONICPLASMONICPOLARITONICRESONANCE/4
0.00355, 54
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/5
0.00355, 54
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOEXCITONICPLASMONICPOLARITONIC_RESONANCE/6
0.00355, 54
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY/7
0.00355, 54
/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_EXCITON_POLARITON_RES_WORK/8
0.00355, 54
/ENG/EELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPOLARITONICRESONANCE/9
0.00355, 54
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPOLARITONIC_RESONANCE_DISSIPATION/10
0.00355, 54
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPOLARITONIC_RESONANCE/11
0.00355, 54
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for rid in range(3, 12):
        assert rid in model.eng_electrothermoflexomagnetoexcitonicplasmonicpolaritonic_resonance_energies
        r = model.eng_electrothermoflexomagnetoexcitonicplasmonicpolaritonic_resonance_energies[rid]
        assert pytest.approx(r.dt_etfexplp) == 0.00355
        assert r.sens_id == 54


def test_m442_lagmul_algebraic_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{2.9e6:>20.4f}{6:>10d}{1.4e-5:>20.6e}"
    c2 = f"{15.0:>20.4f}{17.0:>20.4f}{58.0:>20.4f}{4.8:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Algebraic Spinor Spatial Linkage Joint Fixed Format Test
2022 0
/ALGEBRAIC_SPINOR_SPATIAL_LINKAGE_JOINT/1
Algebraic Spinor Spatial Mechanism Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_algebraic_spinor_spatial_linkage_joints
    j = model.lagmul_algebraic_spinor_spatial_linkage_joints[1]
    assert j.node1 == 101
    assert j.node2 == 102
    assert j.node3 == 103
    assert pytest.approx(j.stiff) == 2.9e6
    assert j.skew_id == 6
    assert pytest.approx(j.tol) == 1.4e-5
    assert pytest.approx(j.link_len_a) == 15.0
    assert pytest.approx(j.link_len_b) == 17.0
    assert pytest.approx(j.twist_angle_alpha) == 58.0
    assert pytest.approx(j.offset_distance_s) == 4.8
    assert pytest.approx(j.offset_distance_r) == 4.8
    assert pytest.approx(j.offset_distance_v) == 4.8
    assert pytest.approx(j.offset_distance_h) == 4.8
    assert pytest.approx(j.offset_distance_u) == 4.8
    assert pytest.approx(j.offset_distance_f) == 4.8


def test_m442_lagmul_algebraic_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Algebraic Spinor Spatial Linkage Joint Free Format Test
/LAGMUL/ALGEBRAIC_SPINOR_SPATIAL_LINKAGE_JOINT/2
201, 202, 203, 3.9e6, 7, 2.4e-5
18.0, 20.0, 70.0, 5.8
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_algebraic_spinor_spatial_linkage_joints
    j = model.lagmul_algebraic_spinor_spatial_linkage_joints[2]
    assert j.node1 == 201
    assert j.node2 == 202
    assert j.node3 == 203
    assert pytest.approx(j.stiff) == 3.9e6
    assert j.skew_id == 7
    assert pytest.approx(j.tol) == 2.4e-5
    assert pytest.approx(j.link_len_a) == 18.0
    assert pytest.approx(j.link_len_b) == 20.0
    assert pytest.approx(j.twist_angle_alpha) == 70.0
    assert pytest.approx(j.offset_distance_s) == 5.8


def test_m442_lagmul_algebraic_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Algebraic Spinor Spatial Linkage Joint Aliases Test
/LAGMUL/ALGEBRAIC_SPINOR_SPATIAL_LINKAGE/3
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/ALGEBRAIC_SPINOR_SPATIAL_LINKAGE/4
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/ALGEBRAIC_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM/5
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/ALGEBRAIC_SPINOR_SPATIAL_SYMMETRIC_MECHANISM/6
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/ALGEBRAIC_SPINOR_SPATIAL_6R_MECHANISM/7
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/ALGEBRAIC_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM/8
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/ALGEBRAIC_TWISTOR_SPATIAL_LINKAGE_JOINT/9
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/ALGEBRAIC_TWISTOR_SPATIAL_LINKAGE_JOINT/10
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/ALGEBRAIC_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/11
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/ALGEBRAIC_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/12
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/ALGEBRAIC_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/13
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/ALGEBRAIC_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/14
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(3, 15):
        assert jid in model.lagmul_algebraic_spinor_spatial_linkage_joints
        j = model.lagmul_algebraic_spinor_spatial_linkage_joints[jid]
        assert j.node1 == 301
        assert j.node2 == 302
        assert j.node3 == 303


def test_m442_sensor_spring_total_angular_crackle_rate_fixed(tmp_path: Path):
    c1 = f"{10:>10d}{1.8e5:>20.4f}{0.002:>20.6f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Angular Crackle Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_CRACKLE_RATE/1
Total Angular Crackle Rate Sensor Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_angular_crackle_rates
    s = model.sensor_spring_total_angular_crackle_rates[1]
    assert s.spring_id == 10
    assert pytest.approx(s.jtot_ang_crk_max) == 1.8e5
    assert pytest.approx(s.t_delay) == 0.002
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_CRACKLE_RATE"


def test_m442_sensor_spring_total_angular_crackle_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Angular Crackle Rate Free Format Test
/SENSOR/SPRING_TOTAL_ANGULAR_CRACKLE_RATE/2
20, 2.8e5, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_angular_crackle_rates
    s = model.sensor_spring_total_angular_crackle_rates[2]
    assert s.spring_id == 20
    assert pytest.approx(s.jtot_ang_crk_max) == 2.8e5
    assert pytest.approx(s.t_delay) == 0.003


def test_m442_sensor_spring_total_angular_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Angular Crackle Rate Aliases Test
/SENSOR/SPRING_TOT_ANG_CRACKLE_RATE/3
30, 3.4e5, 0.001
/SENSOR/SPRING_TOTAL_ANGULAR_CRK_RATE/4
30, 3.4e5, 0.001
/SENSOR/SPRING_TOT_ANG_CRK_RATE/5
30, 3.4e5, 0.001
/SENSOR/SPRING_CRACKLE_RATE_TOT_ANG/6
30, 3.4e5, 0.001
/SENSOR/SPRING_CRACKLE_RATE_TOTAL_ANGULAR/7
30, 3.4e5, 0.001
/SENSOR/TOTAL_ANGULAR_CRACKLE_RATE_SPRING/8
30, 3.4e5, 0.001
/SENSOR/SPRING_CRK_RATE_TOT_ANG/9
30, 3.4e5, 0.001
/SENSOR/SPRING_TOTAL_ANGULAR_CRACKLE/10
30, 3.4e5, 0.001
/SENSOR/SPRING_CRACKLE_TOT_ANG/11
30, 3.4e5, 0.001
/SENSOR/SPRING_RESULTANT_ANGULAR_CRACKLE_RATE/12
30, 3.4e5, 0.001
/SENSOR/SPRING_RES_ANG_CRACKLE_RATE/13
30, 3.4e5, 0.001
/SENSOR/SPRING_RESULTANT_ANGULAR_CRK_RATE/14
30, 3.4e5, 0.001
/SENSOR/SPRING_RES_ANG_CRK_RATE/15
30, 3.4e5, 0.001
/SENSOR/RESULTANT_ANGULAR_CRACKLE_RATE_SPRING/16
30, 3.4e5, 0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(3, 17):
        assert sid in model.sensor_spring_total_angular_crackle_rates
        s = model.sensor_spring_total_angular_crackle_rates[sid]
        assert s.spring_id == 30
        assert pytest.approx(s.jtot_ang_crk_max) == 3.4e5


def test_m442_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_COUPLED_CORE_MICROBUCKLING_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/ALGEBRAIC_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_TOTAL_ANGULAR_CRACKLE_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4
