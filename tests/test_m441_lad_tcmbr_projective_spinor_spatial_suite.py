"""Tests for Milestone M441: LadTransverseCoreMicrobucklingRate Failure Model, EngElectrothermoflexomagnetophononicplasmonicpolaritonicResonanceEnergy, LagmulProjectiveSpinorSpatialLinkageJoint, and SensorSpringBendingCrackleRate."""

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


def test_m441_fail_lad_transverse_core_microbuckling_rate_fixed(tmp_path: Path):
    c1 = f"{970.0:>20.4f}{2910.0:>20.4f}{770.0:>20.4f}{15.60:>20.4f}{0.460:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2570:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Core Microbuckling Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_CORE_MICROBUCKLING_RATE/2570
Ladeveze Transverse Core Microbuckling Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2570 in model.fail_ladtransversecoremicrobucklingrates
    ftcmbr = model.fail_ladtransversecoremicrobucklingrates[2570]
    assert pytest.approx(ftcmbr.sigma_tcmbr0) == 970.0
    assert pytest.approx(ftcmbr.sigma_tcmbrc) == 2910.0
    assert pytest.approx(ftcmbr.gamma_tcmbr) == 770.0
    assert pytest.approx(ftcmbr.p_tcmbr) == 15.60
    assert pytest.approx(ftcmbr.d_tcmbr_max) == 0.460
    assert ftcmbr.ifail_sh == 1
    assert ftcmbr.ifail_so == 2
    assert ftcmbr.fail_id == 2570
    assert pytest.approx(ftcmbr.sigma_tcmbk0) == 970.0
    assert pytest.approx(ftcmbr.sigma_tcmbkc) == 2910.0
    assert pytest.approx(ftcmbr.gamma_tcmbk) == 770.0
    assert pytest.approx(ftcmbr.p_tcmbk) == 15.60
    assert pytest.approx(ftcmbr.d_tcmbk_max) == 0.460
    assert pytest.approx(ftcmbr.sigma_tcmk0) == 970.0
    assert pytest.approx(ftcmbr.sigma_tcmkc) == 2910.0
    assert pytest.approx(ftcmbr.gamma_tcmk) == 770.0
    assert pytest.approx(ftcmbr.p_tcmk) == 15.60
    assert pytest.approx(ftcmbr.d_tcmk_max) == 0.460
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_CORE_MICROBUCKLING_RATE"


def test_m441_fail_lad_transverse_core_microbuckling_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Core Microbuckling Rate Free Format Test
/FAIL/LAD_TRANSVERSE_CORE_MICROBUCKLING_RATE/2571
980.0, 2940.0, 780.0, 15.80, 0.450
1, 1
2571
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2571 in model.fail_ladtransversecoremicrobucklingrates
    ftcmbr = model.fail_ladtransversecoremicrobucklingrates[2571]
    assert pytest.approx(ftcmbr.sigma_tcmbr0) == 980.0
    assert pytest.approx(ftcmbr.sigma_tcmbrc) == 2940.0
    assert pytest.approx(ftcmbr.gamma_tcmbr) == 780.0
    assert pytest.approx(ftcmbr.p_tcmbr) == 15.80
    assert pytest.approx(ftcmbr.d_tcmbr_max) == 0.450
    assert ftcmbr.fail_id == 2571


def test_m441_fail_lad_transverse_core_microbuckling_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Core Microbuckling Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_CORE_MICROBUCKLING_RATE/2572
950.0, 2850.0, 750.0, 15.10, 0.480
1, 1
/FAIL/LAD_TRANSVERSE_CORE_MICROBUCKLE_RATE/2573
950.0, 2850.0, 750.0, 15.10, 0.480
1, 1
/FAIL/LADEVEZE_TRANSVERSE_CORE_MICROBUCKLE_RATE/2574
950.0, 2850.0, 750.0, 15.10, 0.480
1, 1
/FAIL/LAD_TCMBR/2575
950.0, 2850.0, 750.0, 15.10, 0.480
1, 1
/FAIL/LAD_TCMBR_MODEL/2576
950.0, 2850.0, 750.0, 15.10, 0.480
1, 1
/FAIL/LAD_TCMBR_LAW/2577
950.0, 2850.0, 750.0, 15.10, 0.480
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_CORE_MICROBUCKLING/2578
950.0, 2850.0, 750.0, 15.10, 0.480
1, 1
/FAIL/LAD_TRANSVERSE_CORE_MICROKINKING_RATE/2579
950.0, 2850.0, 750.0, 15.10, 0.480
1, 1
/FAIL/LADEVEZE_TRANSVERSE_CORE_MICROKINKING_RATE/2580
950.0, 2850.0, 750.0, 15.10, 0.480
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for mid in range(2572, 2581):
        assert mid in model.fail_ladtransversecoremicrobucklingrates
        f = model.fail_ladtransversecoremicrobucklingrates[mid]
        assert pytest.approx(f.sigma_tcmbr0) == 950.0
        assert pytest.approx(f.sigma_tcmbrc) == 2850.0


def test_m441_fail_lad_transverse_core_microbuckling_rate_properties():
    from pyradioss.model.entities import FailLadTransverseCoreMicrobucklingRate
    f = FailLadTransverseCoreMicrobucklingRate(mat_id=1)
    f.sigma_tcmbk0 = 1040.0
    assert pytest.approx(f.sigma_tcmbr0) == 1040.0
    assert pytest.approx(f.sigma_tcmbk0) == 1040.0

    f.sigma_tcmbkc = 3120.0
    assert pytest.approx(f.sigma_tcmbrc) == 3120.0
    assert pytest.approx(f.sigma_tcmbkc) == 3120.0

    f.gamma_tcmbk = 820.0
    assert pytest.approx(f.gamma_tcmbr) == 820.0
    assert pytest.approx(f.gamma_tcmbk) == 820.0

    f.p_tcmbk = 16.20
    assert pytest.approx(f.p_tcmbr) == 16.20
    assert pytest.approx(f.p_tcmbk) == 16.20

    f.d_tcmbk_max = 0.430
    assert pytest.approx(f.d_tcmbr_max) == 0.430
    assert pytest.approx(f.d_tcmbk_max) == 0.430

    f.sigma_tcmk0 = 1100.0
    assert pytest.approx(f.sigma_tcmbr0) == 1100.0
    assert pytest.approx(f.sigma_tcmk0) == 1100.0

    f.sigma_tcmkc = 3300.0
    assert pytest.approx(f.sigma_tcmbrc) == 3300.0
    assert pytest.approx(f.sigma_tcmkc) == 3300.0

    f.gamma_tcmk = 860.0
    assert pytest.approx(f.gamma_tcmbr) == 860.0
    assert pytest.approx(f.gamma_tcmk) == 860.0

    f.p_tcmk = 17.20
    assert pytest.approx(f.p_tcmbr) == 17.20
    assert pytest.approx(f.p_tcmk) == 17.20

    f.d_tcmk_max = 0.390
    assert pytest.approx(f.d_tcmbr_max) == 0.390
    assert pytest.approx(f.d_tcmk_max) == 0.390


def test_m441_eng_electrothermoflexomagnetophononicplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00145:>20.6f}{48:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Phononic Plasmonic Polaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Phononic Plasmonic Polaritonic Resonance Energy Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetophononicplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetophononicplasmonicpolaritonic_resonance_energies[1]
    assert pytest.approx(r.dt_etfphplp) == 0.00145
    assert pytest.approx(r.dt_etfplp) == 0.00145
    assert r.sens_id == 48


def test_m441_eng_electrothermoflexomagnetophononicplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Phononic Plasmonic Polaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
0.00245, 49
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetophononicplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetophononicplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(r.dt_etfphplp) == 0.00245
    assert r.sens_id == 49


def test_m441_eng_electrothermoflexomagnetophononicplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Phononic Plasmonic Polaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_PHONON_PLASMON_POLARITON_RES_WORK/3
0.00345, 50
/ENG/EELECTROTHERMOFLEXOMAGNETOPHONONICPLASMONICPOLARITONICRESONANCE/4
0.00345, 50
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/5
0.00345, 50
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE/6
0.00345, 50
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for rid in range(3, 7):
        assert rid in model.eng_electrothermoflexomagnetophononicplasmonicpolaritonic_resonance_energies
        r = model.eng_electrothermoflexomagnetophononicplasmonicpolaritonic_resonance_energies[rid]
        assert pytest.approx(r.dt_etfphplp) == 0.00345
        assert r.sens_id == 50


def test_m441_lagmul_projective_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{2.8e6:>20.4f}{6:>10d}{1.2e-5:>20.6e}"
    c2 = f"{14.0:>20.4f}{16.0:>20.4f}{55.0:>20.4f}{4.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Projective Spinor Spatial Linkage Joint Fixed Format Test
2022 0
/PROJECTIVE_SPINOR_SPATIAL_LINKAGE_JOINT/1
Projective Spinor Spatial Mechanism Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_projective_spinor_spatial_linkage_joints
    j = model.lagmul_projective_spinor_spatial_linkage_joints[1]
    assert j.node1 == 101
    assert j.node2 == 102
    assert j.node3 == 103
    assert pytest.approx(j.stiff) == 2.8e6
    assert j.skew_id == 6
    assert pytest.approx(j.tol) == 1.2e-5
    assert pytest.approx(j.link_len_a) == 14.0
    assert pytest.approx(j.link_len_b) == 16.0
    assert pytest.approx(j.twist_angle_alpha) == 55.0
    assert pytest.approx(j.offset_distance_s) == 4.5
    assert pytest.approx(j.offset_distance_r) == 4.5
    assert pytest.approx(j.offset_distance_v) == 4.5
    assert pytest.approx(j.offset_distance_h) == 4.5
    assert pytest.approx(j.offset_distance_u) == 4.5
    assert pytest.approx(j.offset_distance_f) == 4.5


def test_m441_lagmul_projective_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Projective Spinor Spatial Linkage Joint Free Format Test
/LAGMUL/PROJECTIVE_SPINOR_SPATIAL_LINKAGE_JOINT/2
201, 202, 203, 3.8e6, 7, 2.2e-5
17.0, 19.0, 68.0, 5.5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_projective_spinor_spatial_linkage_joints
    j = model.lagmul_projective_spinor_spatial_linkage_joints[2]
    assert j.node1 == 201
    assert j.node2 == 202
    assert j.node3 == 203
    assert pytest.approx(j.stiff) == 3.8e6
    assert j.skew_id == 7
    assert pytest.approx(j.tol) == 2.2e-5
    assert pytest.approx(j.link_len_a) == 17.0
    assert pytest.approx(j.link_len_b) == 19.0
    assert pytest.approx(j.twist_angle_alpha) == 68.0
    assert pytest.approx(j.offset_distance_s) == 5.5


def test_m441_lagmul_projective_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Projective Spinor Spatial Linkage Joint Aliases Test
/LAGMUL/PROJECTIVE_SPINOR_SPATIAL_LINKAGE/3
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/PROJECTIVE_SPINOR_SPATIAL_LINKAGE/4
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/PROJECTIVE_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM/5
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/PROJECTIVE_SPINOR_SPATIAL_SYMMETRIC_MECHANISM/6
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/PROJECTIVE_SPINOR_SPATIAL_6R_MECHANISM/7
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/PROJECTIVE_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM/8
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/PROJECTIVE_TWISTOR_SPATIAL_LINKAGE_JOINT/9
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/PROJECTIVE_TWISTOR_SPATIAL_LINKAGE_JOINT/10
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/PROJECTIVE_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/11
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/PROJECTIVE_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/12
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/PROJECTIVE_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/13
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/PROJECTIVE_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/14
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(3, 15):
        assert jid in model.lagmul_projective_spinor_spatial_linkage_joints
        j = model.lagmul_projective_spinor_spatial_linkage_joints[jid]
        assert j.node1 == 301
        assert j.node2 == 302
        assert j.node3 == 303


def test_m441_sensor_spring_bending_crackle_rate_fixed(tmp_path: Path):
    c1 = f"{10:>10d}{1.6e5:>20.4f}{0.002:>20.6f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Bending Crackle Rate Fixed Format Test
2022 0
/SENSOR/SPRING_BENDING_CRACKLE_RATE/1
Bending Crackle Rate Sensor Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_bending_crackle_rates
    s = model.sensor_spring_bending_crackle_rates[1]
    assert s.spring_id == 10
    assert pytest.approx(s.jbend_crk_max) == 1.6e5
    assert pytest.approx(s.t_delay) == 0.002
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_BENDING_CRACKLE_RATE"


def test_m441_sensor_spring_bending_crackle_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Bending Crackle Rate Free Format Test
/SENSOR/SPRING_BENDING_CRACKLE_RATE/2
20, 2.6e5, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_bending_crackle_rates
    s = model.sensor_spring_bending_crackle_rates[2]
    assert s.spring_id == 20
    assert pytest.approx(s.jbend_crk_max) == 2.6e5
    assert pytest.approx(s.t_delay) == 0.003


def test_m441_sensor_spring_bending_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Bending Crackle Rate Aliases Test
/SENSOR/SPRING_BEND_CRACKLE_RATE/3
30, 3.2e5, 0.001
/SENSOR/SPRING_BENDING_CRK_RATE/4
30, 3.2e5, 0.001
/SENSOR/SPRING_BEND_CRK_RATE/5
30, 3.2e5, 0.001
/SENSOR/SPRING_CRACKLE_RATE_BEND/6
30, 3.2e5, 0.001
/SENSOR/SPRING_CRACKLE_RATE_BENDING/7
30, 3.2e5, 0.001
/SENSOR/BENDING_CRACKLE_RATE_SPRING/8
30, 3.2e5, 0.001
/SENSOR/SPRING_CRK_RATE_BEND/9
30, 3.2e5, 0.001
/SENSOR/SPRING_BENDING_CRACKLE/10
30, 3.2e5, 0.001
/SENSOR/SPRING_CRACKLE_BEND/11
30, 3.2e5, 0.001
/SENSOR/SPRING_FLEX_CRACKLE_RATE/12
30, 3.2e5, 0.001
/SENSOR/SPRING_FLEX_CRK_RATE/13
30, 3.2e5, 0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(3, 14):
        assert sid in model.sensor_spring_bending_crackle_rates
        s = model.sensor_spring_bending_crackle_rates[sid]
        assert s.spring_id == 30
        assert pytest.approx(s.jbend_crk_max) == 3.2e5


def test_m441_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_TRANSVERSE_CORE_MICROBUCKLING_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/PROJECTIVE_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_BENDING_CRACKLE_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4
