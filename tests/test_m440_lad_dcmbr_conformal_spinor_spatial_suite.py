"""Tests for Milestone M440: LadDynamicCoreMicrobucklingRate Failure Model, EngElectrothermoflexomagnetoplasmonicpolaritonicResonanceEnergy, LagmulConformalSpinorSpatialLinkageJoint, and SensorSpringTorsionalCrackleRate."""

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


def test_m440_fail_lad_dynamic_core_microbuckling_rate_fixed(tmp_path: Path):
    c1 = f"{950.0:>20.4f}{2850.0:>20.4f}{750.0:>20.4f}{15.20:>20.4f}{0.480:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2550:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Core Microbuckling Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_CORE_MICROBUCKLING_RATE/2550
Ladeveze Dynamic Core Microbuckling Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2550 in model.fail_laddynamiccoremicrobucklingrates
    fdcmbr = model.fail_laddynamiccoremicrobucklingrates[2550]
    assert pytest.approx(fdcmbr.sigma_dcmbr0) == 950.0
    assert pytest.approx(fdcmbr.sigma_dcmbrc) == 2850.0
    assert pytest.approx(fdcmbr.gamma_dcmbr) == 750.0
    assert pytest.approx(fdcmbr.p_dcmbr) == 15.20
    assert pytest.approx(fdcmbr.d_dcmbr_max) == 0.480
    assert fdcmbr.ifail_sh == 1
    assert fdcmbr.ifail_so == 2
    assert fdcmbr.fail_id == 2550
    assert pytest.approx(fdcmbr.sigma_dcmbk0) == 950.0
    assert pytest.approx(fdcmbr.sigma_dcmbkc) == 2850.0
    assert pytest.approx(fdcmbr.gamma_dcmbk) == 750.0
    assert pytest.approx(fdcmbr.p_dcmbk) == 15.20
    assert pytest.approx(fdcmbr.d_dcmbk_max) == 0.480
    assert pytest.approx(fdcmbr.sigma_dcmk0) == 950.0
    assert pytest.approx(fdcmbr.sigma_dcmkc) == 2850.0
    assert pytest.approx(fdcmbr.gamma_dcmk) == 750.0
    assert pytest.approx(fdcmbr.p_dcmk) == 15.20
    assert pytest.approx(fdcmbr.d_dcmk_max) == 0.480
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_CORE_MICROBUCKLING_RATE"


def test_m440_fail_lad_dynamic_core_microbuckling_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Core Microbuckling Rate Free Format Test
/FAIL/LAD_DYNAMIC_CORE_MICROBUCKLING_RATE/2551
960.0, 2880.0, 760.0, 15.40, 0.470
1, 1
2551
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2551 in model.fail_laddynamiccoremicrobucklingrates
    fdcmbr = model.fail_laddynamiccoremicrobucklingrates[2551]
    assert pytest.approx(fdcmbr.sigma_dcmbr0) == 960.0
    assert pytest.approx(fdcmbr.sigma_dcmbrc) == 2880.0
    assert pytest.approx(fdcmbr.gamma_dcmbr) == 760.0
    assert pytest.approx(fdcmbr.p_dcmbr) == 15.40
    assert pytest.approx(fdcmbr.d_dcmbr_max) == 0.470
    assert fdcmbr.fail_id == 2551


def test_m440_fail_lad_dynamic_core_microbuckling_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Core Microbuckling Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_CORE_MICROBUCKLING_RATE/2552
940.0, 2820.0, 740.0, 15.00, 0.490
1, 1
/FAIL/LAD_DYNAMIC_CORE_MICROBUCKLE_RATE/2553
940.0, 2820.0, 740.0, 15.00, 0.490
1, 1
/FAIL/LADEVEZE_DYNAMIC_CORE_MICROBUCKLE_RATE/2554
940.0, 2820.0, 740.0, 15.00, 0.490
1, 1
/FAIL/LAD_DCMBR/2555
940.0, 2820.0, 740.0, 15.00, 0.490
1, 1
/FAIL/LAD_DCMBR_MODEL/2556
940.0, 2820.0, 740.0, 15.00, 0.490
1, 1
/FAIL/LAD_DCMBR_LAW/2557
940.0, 2820.0, 740.0, 15.00, 0.490
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_CORE_MICROBUCKLING/2558
940.0, 2820.0, 740.0, 15.00, 0.490
1, 1
/FAIL/LAD_DYNAMIC_CORE_MICROKINKING_RATE/2559
940.0, 2820.0, 740.0, 15.00, 0.490
1, 1
/FAIL/LADEVEZE_DYNAMIC_CORE_MICROKINKING_RATE/2560
940.0, 2820.0, 740.0, 15.00, 0.490
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for mid in range(2552, 2561):
        assert mid in model.fail_laddynamiccoremicrobucklingrates
        f = model.fail_laddynamiccoremicrobucklingrates[mid]
        assert pytest.approx(f.sigma_dcmbr0) == 940.0
        assert pytest.approx(f.sigma_dcmbrc) == 2820.0


def test_m440_fail_lad_dynamic_core_microbuckling_rate_properties():
    from pyradioss.model.entities import FailLadDynamicCoreMicrobucklingRate
    f = FailLadDynamicCoreMicrobucklingRate(mat_id=1)
    f.sigma_dcmbk0 = 1020.0
    assert pytest.approx(f.sigma_dcmbr0) == 1020.0
    assert pytest.approx(f.sigma_dcmbk0) == 1020.0

    f.sigma_dcmbkc = 3060.0
    assert pytest.approx(f.sigma_dcmbrc) == 3060.0
    assert pytest.approx(f.sigma_dcmbkc) == 3060.0

    f.gamma_dcmbk = 800.0
    assert pytest.approx(f.gamma_dcmbr) == 800.0
    assert pytest.approx(f.gamma_dcmbk) == 800.0

    f.p_dcmbk = 16.00
    assert pytest.approx(f.p_dcmbr) == 16.00
    assert pytest.approx(f.p_dcmbk) == 16.00

    f.d_dcmbk_max = 0.440
    assert pytest.approx(f.d_dcmbr_max) == 0.440
    assert pytest.approx(f.d_dcmbk_max) == 0.440

    f.sigma_dcmk0 = 1080.0
    assert pytest.approx(f.sigma_dcmbr0) == 1080.0
    assert pytest.approx(f.sigma_dcmk0) == 1080.0

    f.sigma_dcmkc = 3240.0
    assert pytest.approx(f.sigma_dcmbrc) == 3240.0
    assert pytest.approx(f.sigma_dcmkc) == 3240.0

    f.gamma_dcmk = 850.0
    assert pytest.approx(f.gamma_dcmbr) == 850.0
    assert pytest.approx(f.gamma_dcmk) == 850.0

    f.p_dcmk = 17.00
    assert pytest.approx(f.p_dcmbr) == 17.00
    assert pytest.approx(f.p_dcmk) == 17.00

    f.d_dcmk_max = 0.400
    assert pytest.approx(f.d_dcmbr_max) == 0.400
    assert pytest.approx(f.d_dcmk_max) == 0.400


def test_m440_eng_electrothermoflexomagnetoplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00135:>20.6f}{45:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Plasmonic Polaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Plasmonic Polaritonic Resonance Energy Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetoplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetoplasmonicpolaritonic_resonance_energies[1]
    assert pytest.approx(r.dt_etfplp) == 0.00135
    assert pytest.approx(r.dt_etfmgmnp) == 0.00135
    assert r.sens_id == 45


def test_m440_eng_electrothermoflexomagnetoplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Plasmonic Polaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
0.00235, 46
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetoplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetoplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(r.dt_etfplp) == 0.00235
    assert r.sens_id == 46


def test_m440_eng_electrothermoflexomagnetoplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Plasmonic Polaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_POLARITON_RES_WORK/3
0.00335, 47
/ENG/EELECTROTHERMOFLEXOMAGNETOPLASMONICPOLARITONICRESONANCE/4
0.00335, 47
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/5
0.00335, 47
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPLASMONICPOLARITONIC_RESONANCE/6
0.00335, 47
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for rid in range(3, 7):
        assert rid in model.eng_electrothermoflexomagnetoplasmonicpolaritonic_resonance_energies
        r = model.eng_electrothermoflexomagnetoplasmonicpolaritonic_resonance_energies[rid]
        assert pytest.approx(r.dt_etfplp) == 0.00335
        assert r.sens_id == 47


def test_m440_lagmul_conformal_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{2.6e6:>20.4f}{5:>10d}{1.0e-5:>20.6e}"
    c2 = f"{13.0:>20.4f}{15.0:>20.4f}{50.0:>20.4f}{4.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Conformal Spinor Spatial Linkage Joint Fixed Format Test
2022 0
/CONFORMAL_SPINOR_SPATIAL_LINKAGE_JOINT/1
Conformal Spinor Spatial Mechanism Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_conformal_spinor_spatial_linkage_joints
    j = model.lagmul_conformal_spinor_spatial_linkage_joints[1]
    assert j.node1 == 101
    assert j.node2 == 102
    assert j.node3 == 103
    assert pytest.approx(j.stiff) == 2.6e6
    assert j.skew_id == 5
    assert pytest.approx(j.tol) == 1.0e-5
    assert pytest.approx(j.link_len_a) == 13.0
    assert pytest.approx(j.link_len_b) == 15.0
    assert pytest.approx(j.twist_angle_alpha) == 50.0
    assert pytest.approx(j.offset_distance_s) == 4.0
    assert pytest.approx(j.offset_distance_r) == 4.0
    assert pytest.approx(j.offset_distance_v) == 4.0
    assert pytest.approx(j.offset_distance_h) == 4.0
    assert pytest.approx(j.offset_distance_u) == 4.0
    assert pytest.approx(j.offset_distance_f) == 4.0


def test_m440_lagmul_conformal_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Conformal Spinor Spatial Linkage Joint Free Format Test
/LAGMUL/CONFORMAL_SPINOR_SPATIAL_LINKAGE_JOINT/2
201, 202, 203, 3.6e6, 6, 2.0e-5
16.0, 18.0, 65.0, 5.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_conformal_spinor_spatial_linkage_joints
    j = model.lagmul_conformal_spinor_spatial_linkage_joints[2]
    assert j.node1 == 201
    assert j.node2 == 202
    assert j.node3 == 203
    assert pytest.approx(j.stiff) == 3.6e6
    assert j.skew_id == 6
    assert pytest.approx(j.tol) == 2.0e-5
    assert pytest.approx(j.link_len_a) == 16.0
    assert pytest.approx(j.link_len_b) == 18.0
    assert pytest.approx(j.twist_angle_alpha) == 65.0
    assert pytest.approx(j.offset_distance_s) == 5.0


def test_m440_lagmul_conformal_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Conformal Spinor Spatial Linkage Joint Aliases Test
/LAGMUL/CONFORMAL_SPINOR_SPATIAL_LINKAGE/3
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/CONFORMAL_SPINOR_SPATIAL_LINKAGE/4
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/CONFORMAL_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM/5
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/CONFORMAL_SPINOR_SPATIAL_SYMMETRIC_MECHANISM/6
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/CONFORMAL_SPINOR_SPATIAL_6R_MECHANISM/7
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/CONFORMAL_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM/8
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/CONFORMAL_TWISTOR_SPATIAL_LINKAGE_JOINT/9
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/CONFORMAL_TWISTOR_SPATIAL_LINKAGE_JOINT/10
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/CONFORMAL_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/11
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/CONFORMAL_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/12
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/CONFORMAL_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/13
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/CONFORMAL_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/14
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(3, 15):
        assert jid in model.lagmul_conformal_spinor_spatial_linkage_joints
        j = model.lagmul_conformal_spinor_spatial_linkage_joints[jid]
        assert j.node1 == 301
        assert j.node2 == 302
        assert j.node3 == 303


def test_m440_sensor_spring_torsional_crackle_rate_fixed(tmp_path: Path):
    c1 = f"{10:>10d}{1.5e5:>20.4f}{0.002:>20.6f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Crackle Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE/1
Torsional Crackle Rate Sensor Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_crackle_rates
    s = model.sensor_spring_torsional_crackle_rates[1]
    assert s.spring_id == 10
    assert pytest.approx(s.jtors_crk_max) == 1.5e5
    assert pytest.approx(s.t_delay) == 0.002
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_CRACKLE_RATE"


def test_m440_sensor_spring_torsional_crackle_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Torsional Crackle Rate Free Format Test
/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE/2
20, 2.5e5, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_torsional_crackle_rates
    s = model.sensor_spring_torsional_crackle_rates[2]
    assert s.spring_id == 20
    assert pytest.approx(s.jtors_crk_max) == 2.5e5
    assert pytest.approx(s.t_delay) == 0.003


def test_m440_sensor_spring_torsional_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Torsional Crackle Rate Aliases Test
/SENSOR/SPRING_TORS_CRACKLE_RATE/3
30, 3.0e5, 0.001
/SENSOR/SPRING_TORSIONAL_CRK_RATE/4
30, 3.0e5, 0.001
/SENSOR/SPRING_TORS_CRK_RATE/5
30, 3.0e5, 0.001
/SENSOR/SPRING_CRACKLE_RATE_TORS/6
30, 3.0e5, 0.001
/SENSOR/SPRING_CRACKLE_RATE_TORSIONAL/7
30, 3.0e5, 0.001
/SENSOR/TORSIONAL_CRACKLE_RATE_SPRING/8
30, 3.0e5, 0.001
/SENSOR/SPRING_CRK_RATE_TORS/9
30, 3.0e5, 0.001
/SENSOR/SPRING_TORSIONAL_CRACKLE/10
30, 3.0e5, 0.001
/SENSOR/SPRING_CRACKLE_TORS/11
30, 3.0e5, 0.001
/SENSOR/SPRING_TWIST_CRACKLE_RATE/12
30, 3.0e5, 0.001
/SENSOR/SPRING_TWIST_CRK_RATE/13
30, 3.0e5, 0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(3, 14):
        assert sid in model.sensor_spring_torsional_crackle_rates
        s = model.sensor_spring_torsional_crackle_rates[sid]
        assert s.spring_id == 30
        assert pytest.approx(s.jtors_crk_max) == 3.0e5


def test_m440_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_DYNAMIC_CORE_MICROBUCKLING_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/CONFORMAL_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_TORSIONAL_CRACKLE_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4
